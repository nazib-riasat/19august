"""S3 — budgeted submodular evidence packing.

**[EVIDENCE]** the objective and the algorithm are arXiv 2607.00725's
(*provisional venue, declared*). Its Eq. 1, p. 5, verbatim:

    F(S) = w_rel·Rel(S) + w_qry·QueryCov(S) + w_cov·Repr(S) + w_div·Div(S)

subject to ``cost(S) ≤ B`` and a snippet cap, with each term *"monotone and
submodular, normalized to [0, 1]"* and the paper's five constants —
``w_rel = 1.0, w_qry = 0.5, w_cov = 0.4, w_div = 0.3, α = 0.3``. The paper's own
descriptions of the terms, quoted:

* ``Rel`` — *"(modular) is the same per-snippet lexical relevance the focused
  heuristic uses"*;
* ``QueryCov`` — *"a set-cover over distinct query content terms"*;
* ``Repr`` — *"a saturated facility-location term, Σ_i min(Σ_{j∈S} sim(i,j),
  α·deg_i), that rewards covering candidate mass but saturates so it cannot be
  gamed by near-duplicates"*;
* ``Div`` — *"a concave-over-documents term, Σ_d √(relevance mass of S in d),
  spreading selection across sources"*.

Algorithm, §4.2: *"cost-scaled (per-token) greedy — at each step add the
feasible snippet with the largest marginal-gain-per-token ratio — followed by
the Lin–Bilmes singleton fallback: if the single best feasible snippet outscores
the greedy set, return it instead."*

**[ANALYSIS] — the label that matters most here** (decision 7). The *algorithm*
is the paper's; the *weights as applied* are not "the paper's objective". Those
five constants were fitted to token-budgeted HotpotQA snippets, and all four
features are re-defined below over lattice atoms with ``rel`` an admitted
stand-in (G1). Calling the result fidelity would be the overreach this plan's
own §0 corrects, in the same document.

**No approximation-ratio claim, and none is available to lose.** §4.2 declines
it itself: *"stronger constant-factor guarantees for the knapsack case require
additional partial enumeration (Sviridenko, 2004; Nemhauser et al., 1978), which
we do not perform — we use the algorithm for its empirical behaviour under a
token budget, not for a guarantee."* GRAFT has one *additional* reason to claim
none: Gate 3 scores ``U`` on the **H-filtered** set while any approximation
result concerns ``F`` on the unfiltered one — two different objectives over two
different objects.

**Cost-scaling has no work to do here** (the ruled cell): the lattice has no
token cost, the constraint is ``max_atoms``, and every atom costs 1, so
marginal-gain-per-cost collapses to marginal gain. The consequence is what the
write-up needs — **S1 and S3 differ by objective, not by search strategy.**
*(One imprecision worth recording: since a greedy move here adds a whole closure
(§3.2), 73% of moves consume more than one atom of the budget. Divide by the
atoms actually consumed instead and S3's validity **collapses** — 12/20 → 3/20
instances with any valid set on the main suite — so the unit divisor is both the
ruled cell and the stronger baseline, which is what `CLAUDE.md` §8 asks for.)*

**The Lin–Bilmes singleton fallback is measured inert at this cap — not
"provably" inert, and an earlier draft of this docstring claimed the proof.**
The claimed argument was that the chain's first pick *is* the argmax singleton,
so monotonicity gives ``F(greedy) ≥ F(best singleton)``. That premise is **false
for the code path used**: ``_chain`` starts from a *forced opener* (decision 3's
portfolio rule), and measured on the main suite the opener coincides with the
argmax feasible singleton on only **20 of 160** chains. What is true is a
measurement: the fallback fires **0/160 at ``max_atoms = 8``**, and fires
**60/160 at a budget of 3** and 11/160 at 2. So inertness is a property of the
frozen cap, which `CLAUDE.md` §6 explicitly prices for change — asserting it as a
theorem was §5's "asserting math without checking it" in a docstring.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from graft.core import resolve
from graft.core.obligations import SLOT_COMPONENTS, slot_status
from graft.core.utility import _similarity_matrix
from graft.ledger import Ledger
from graft.schemas import AtomPool, Obligations
from graft.setgen.search.base import (
    SearchResult,
    admissible_atoms,
    close_under_refs,
    dedup_sets,
    h_filter,
)

__all__ = ["SubmodularGreedy", "SubmodularObjective", "PAPER_WEIGHTS", "SATURATION"]

#: arXiv 2607.00725 §4.1, unmodified.  **Five** constants — an earlier draft of
#: the Phase-4 plan and the architecture both listed four and silently defaulted
#: the saturation.
PAPER_WEIGHTS: dict[str, float] = {"rel": 1.0, "qry": 0.5, "cov": 0.4, "div": 0.3}
SATURATION = 0.3


class SubmodularObjective:
    """``F(S)`` over one instance's pool, with each term normalised to [0, 1].

    Every term is precomputed against the pool once, so the greedy chain is
    arithmetic over cached arrays rather than repeated graph traversal.
    """

    def __init__(
        self,
        pool: AtomPool,
        q: Obligations,
        G: Any,
        relevance: Mapping[str, float],
        *,
        ground: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
        alpha: float = SATURATION,
    ) -> None:
        self.pool = pool
        self.ids = tuple(ground if ground is not None else pool.ids())
        self.index = {aid: i for i, aid in enumerate(self.ids)}
        self.weights = dict(weights or PAPER_WEIGHTS)
        self.alpha = float(alpha)

        self.rel = np.array([float(relevance.get(a, 0.0)) for a in self.ids])
        self._rel_total = float(self.rel.sum())

        # -- Repr: the paper's saturated facility location --------------------
        # sim is the project's own clamped cosine — the same matrix ``U``'s
        # ``redundancy`` uses, and cached per pool (Phase-2 G7).  Sharing it is
        # deliberate: S3's coverage term and the reward's redundancy term are the
        # same objective family, which is what makes the two comparable at all.
        sim_ids, sim = _similarity_matrix(pool)
        order = [sim_ids.index(a) for a in self.ids]
        self.sim = np.asarray(sim)[np.ix_(order, order)]
        self.cap = self.alpha * self.sim.sum(axis=1)  # α·deg_i
        self._repr_total = float(np.minimum(self.sim.sum(axis=1), self.cap).sum())

        # -- QueryCov: set-cover over the query's content requirements --------
        # **[ANALYSIS]** the lattice has no "query content terms"; the faithful
        # analogue of what the query *requires* is the obligation slot set
        # (Phase-1 gap G4's four active slots), and a coverage function over it
        # is monotone submodular exactly as the paper's is.
        self.slots = tuple(q.active_slots())
        covers: list[frozenset[str]] = []
        for aid in self.ids:
            status = slot_status([aid], pool, q, G)
            covers.append(
                frozenset(s for s in self.slots if status[SLOT_COMPONENTS[s]] < 1.0)
            )
        self.covers = tuple(covers)

        # -- Div: concave over sources ----------------------------------------
        # **[ANALYSIS]** "documents" become the atom's resolved ``Source`` node,
        # which is GRAFT's own notion of where evidence came from; atoms with no
        # resolvable source share one bucket rather than each becoming their own,
        # which would hand unsourced atoms a free diversity bonus.
        groups: dict[str, list[int]] = {}
        for i, aid in enumerate(self.ids):
            node = resolve.source_node(pool[aid], G)
            key = node.node_id if node is not None else "__unresolved__"
            groups.setdefault(key, []).append(i)
        self.groups = tuple(tuple(v) for v in groups.values())
        self._div_total = float(
            sum(math.sqrt(self.rel[list(g)].sum()) for g in self.groups)
        )

    # -- the four terms, each in [0, 1] ------------------------------------

    def rel_term(self, sel: Sequence[int]) -> float:
        if self._rel_total <= 0.0:
            return 0.0
        return float(self.rel[list(sel)].sum() / self._rel_total)

    def qry_term(self, sel: Sequence[int]) -> float:
        if not sel:
            # ``F(∅) = 0`` is required of every term: the paper states each is
            # *"monotone and submodular, normalized to [0, 1]"*, and a coverage
            # function that starts at 1 is neither normalised nor a coverage
            # function.  The no-active-slots convention below must not leak into
            # the empty set — it did, and ``F(∅)`` came out at ``w_qry = 0.5``.
            # (Found 13 Aug 2026 by replacing a self-referential test with one
            # that computes the expected values on paper.)
            return 0.0
        if not self.slots:
            # A question with no active slots leaves nothing for this term to
            # discriminate, so it is constant on non-empty sets — the same
            # convention ``U``'s ``coverage`` uses.  Constant-above-∅ is still
            # monotone and submodular: the first atom gains 1, every later one 0.
            return 1.0
        covered: set[str] = set()
        for i in sel:
            covered |= self.covers[i]
        return len(covered) / len(self.slots)

    def repr_term(self, sel: Sequence[int]) -> float:
        if self._repr_total <= 0.0:
            return 0.0
        mass = self.sim[:, list(sel)].sum(axis=1) if sel else np.zeros(len(self.ids))
        return float(np.minimum(mass, self.cap).sum() / self._repr_total)

    def div_term(self, sel: Sequence[int]) -> float:
        if self._div_total <= 0.0:
            return 0.0
        chosen = set(sel)
        total = 0.0
        for g in self.groups:
            mass = self.rel[[i for i in g if i in chosen]].sum() if chosen else 0.0
            total += math.sqrt(mass)
        return float(total / self._div_total)

    def F(self, atoms: Iterable[str]) -> float:
        sel = [self.index[a] for a in atoms if a in self.index]
        w = self.weights
        return (
            w["rel"] * self.rel_term(sel)
            + w["qry"] * self.qry_term(sel)
            + w["cov"] * self.repr_term(sel)
            + w["div"] * self.div_term(sel)
        )


class SubmodularGreedy:
    """S3.  Deterministic; reported once (decision 4).

    **Bypasses the ``ADD`` masks** and is ``H``-filtered afterwards — which is
    precisely why structural closure is a checker sub-check and not only a mask
    rule (fix F10, Phase-1 check 8). It therefore gets no free incremental
    validity and pays **1 terminal check per distinct candidate** (G6), which is
    a real cost difference between method families and belongs in the table.
    """

    name = "s3_submodular"
    deterministic = True

    def __init__(
        self,
        relevance: Mapping[str, float],
        k: int | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        alpha: float = SATURATION,
    ) -> None:
        self.relevance = dict(relevance)
        self.k = k
        self.weights = dict(weights or PAPER_WEIGHTS)
        self.alpha = float(alpha)

    def _chain(
        self, obj: SubmodularObjective, budget: int, opener: str | None
    ) -> tuple[tuple[str, ...], bool]:
        """Cost-scaled greedy over the paper's feasible set, plus its fallback.

        Returns ``(selected atoms, fallback_fired)``.

        **Cost is 1 per atom** — the ruled cell (architecture S3, P4.4): the
        lattice has no token cost, so marginal-gain-per-cost collapses to
        marginal gain. The ratio is written out anyway so the code *is* the
        paper's algorithm and the degeneracy is visibly a property of the costs.

        **Feasibility is the paper's own word**: §4.2 adds *"the feasible
        snippet"* subject to *"cost(S) ≤ B and a snippet cap"*. Here the cap is
        ``max_atoms`` and what actually gets returned is the **closed**
        completion of the selection, so a candidate is feasible when its
        completion still fits. Testing feasibility on the completed set is what
        stops every candidate breaching the cap and being rejected by ``H`` on
        size — the straddle failure decision 8 corrects for S4's ``C_e``.
        """
        sel: tuple[str, ...] = ()
        if opener is not None:
            sel = close_under_refs([opener], obj.pool)
            if len(sel) > budget:
                sel = ()
        current = obj.F(sel)
        while True:
            best, best_ratio, best_closed = None, 0.0, None
            for aid in obj.ids:
                if aid in sel:
                    continue
                closed = close_under_refs(tuple(sel) + (aid,), obj.pool)
                if len(closed) > budget:
                    continue  # infeasible: the completion breaks the cap
                cost = 1.0  # unit atom cost; the paper's per-token divisor
                ratio = (obj.F(closed) - current) / cost
                if ratio > best_ratio:
                    best, best_ratio, best_closed = aid, ratio, closed
            if best is None:
                break
            sel = best_closed
            current = obj.F(sel)

        # Lin-Bilmes singleton fallback: strict "outscores" (section 4.2).  Kept
        # faithfully and **reported**.  It is measured inert at max_atoms = 8 and
        # fires at smaller budgets (60/160 chains at 3), because the chain starts
        # from a forced opener rather than from the argmax singleton -- see the
        # module docstring.  Implemented rather than argued, which is what let
        # the false "provably inert" claim be caught.
        singles = [
            (c, obj.F(c))
            for c in (close_under_refs([a], obj.pool) for a in obj.ids)
            if len(c) <= budget
        ]
        if singles:
            best_single, best_single_f = max(singles, key=lambda t: (t[1], t[0]))
            if best_single_f > current:
                return best_single, True
        return tuple(sel), False

    def run(
        self,
        env: Any,
        obligations: Obligations,
        scorer: Callable[[Iterable[str]], float],
        ledger: Ledger | None = None,
    ) -> SearchResult:
        inst = env.instance
        cfg = inst.cfg
        k = int(self.k if self.k is not None else cfg.K)
        ground = admissible_atoms(env, obligations)
        obj = SubmodularObjective(
            inst.pool,
            obligations,
            inst.graph,
            self.relevance,
            ground=ground,
            weights=self.weights,
            alpha=self.alpha,
        )

        # Decision 3's portfolio rule, the same one S1 uses: forced-distinct
        # openers, the k highest-``rel`` atoms.  No ``ADD`` mask is consulted —
        # S3 is a direct builder and pays for it at the H-filter — so every
        # admissible atom is an eligible opener.
        openers = sorted(obj.ids, key=lambda a: (-float(obj.rel[obj.index[a]]), a))[:k]
        candidates: list[tuple[str, ...]] = []
        fallbacks = 0
        for opener in openers:
            sel, fired = self._chain(obj, cfg.max_atoms, opener)
            fallbacks += int(fired)
            candidates.append(tuple(sorted(sel)))

        breaches = sum(1 for c in candidates if len(c) > cfg.max_atoms)
        valid, spent, exhausted, rejections, portfolio = h_filter(
            candidates, obligations, inst.graph, inst.pool, cfg, ledger
        )
        scored = sorted(
            ((scorer(p.atoms), p) for p in valid), key=lambda t: -t[0]
        )
        return SearchResult(
            self.name,
            [p for _, p in scored],
            [s for s, _ in scored],
            attempted=len(candidates),
            distinct_attempted=len(dedup_sets(candidates)),
            terminal_checks=spent,
            portfolio=portfolio,
            budget_exhausted=exhausted,
            extra={
                "singleton_fallbacks": fallbacks,
                "objective_weights": dict(self.weights),
                "saturation_alpha": self.alpha,
                "ground_atoms": len(ground),
                "pool_atoms": len(inst.pool.ids()),
                "max_atoms_breaches": breaches,
                "h_rejections": rejections,
            },
        )
