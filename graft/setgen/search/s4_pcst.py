"""S4 — Prize-Collecting Steiner Tree over the pool's reference graph.

**[EVIDENCE]** the formulation is G-Retriever's (NeurIPS 2024), §5.3, verbatim:

* **Eq. 6** — *"the top k nodes/edges are assigned descending prize values from
  ``k`` down to 1, with the rest assigned zero"*, i.e.
  ``prize(n) = k − i`` for the top-``i`` node (0-indexed), 0 otherwise;
* **Eq. 7** — ``S* = argmax over connected S of Σ_{n∈V_S} prize(n) +
  Σ_{e∈E_S} prize(e) − cost(S)``;
* **Eq. 8** — ``cost(S) = |E_S| × C_e``, with ``C_e`` *"a predefined cost per
  edge, which is adjustable to control the subgraph size"*.

**Mapping A, ruled** (decision 2, G2): atoms are PCST nodes and ``refs`` are
PCST edges. Two consequences the write-up must carry rather than imply:

1. **The PCST "edges" are reference links, not evidence**, so G-Retriever's
   edge prizes have no counterpart and are **dropped** — with them goes its
   virtual-node construction, which exists only to admit edge prizes. S4 is
   therefore *PCST applied to this environment*, **not** G-Retriever's
   formulation transplanted, and fix F10's "connected output maps to a closed
   set with no conversion logic" is true only under the mapping this phase does
   **not** adopt.
2. **Solved per connected component and unioned — a declared departure from
   G-Retriever's single tree**, forced by measurement: the gold proof spans more
   than one component on **8 of the 20** main instances, so one tree could not
   reach 40% of the golds. `CLAUDE.md` §8 wants this baseline dangerous.

**Then closure completion, then ``H``** (fix F10's amendment): a connected
subtree containing an edge atom and one endpoint is connected but *not* closed,
so completion is required and its rate is reported (exit criterion 12).

**The solver is exact, and that is a departure from ruled decision 8 —
overturned on measurement.** Decision 8 pinned ``pcst_fast`` on the ground that
*"the platform risk an earlier draft raised is not real"* because a prebuilt
``cp311-win_amd64`` wheel exists. The wheel exists and **is wrong**: measured
against a brute-force reference on 60 random graphs, ``pcst_fast`` 1.0.10's
Windows wheel mismatched the optimal objective on **59**, returning arrays of
the correct *length* whose every element equals the first — a buffer-lifetime
bug in the binding, silent and not a crash. G8 named the alternative in advance
(*"implement the PCST objective directly and verify it against ``pcst_fast`` on
a handful of instances offline"*), and that verification is what found this.

Exactness is affordable rather than heroic, because **uniform edge costs collapse
the problem — once the edge prizes are gone**. That second condition is
load-bearing and an earlier draft of this paragraph omitted it: under Eq. 7 as
the paper writes it, an edge with ``prize(e) > C_e`` is worth including even when
it closes a cycle, so the optimum is not a tree and no ``|S| − 1`` argument
applies — which is exactly why the paper needs its virtual-node construction.
Mapping A drops edge prizes (above), and only then does a spanning tree of a
connected ``S`` cost ``C_e·(|S| − 1)`` and Eq. 7 become

    value(S) = Σ_{n∈S} prize(n) − C_e·(|S| − 1)

over connected vertex sets — a maximum-weight connected subgraph problem, solved
exactly here by enumerating each component's connected subsets. Components on
this environment hold a handful of atoms, so the enumeration is trivial; the
bound is asserted rather than assumed (:data:`MAX_COMPONENT_ATOMS`) so a future
pool cannot make it quietly intractable.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from graft.ledger import Ledger
from graft.schemas import AtomPool, Obligations
from graft.setgen.search.base import (
    SearchResult,
    admissible_atoms,
    close_under_refs,
    dedup_sets,
    h_filter,
)

__all__ = [
    "PCSTSearch",
    "EDGE_COST_GRID",
    "MAX_COMPONENT_ATOMS",
    "calibrate_edge_cost",
    "reference_components",
    "solve_pcst_forest",
]

#: Decision 8's predeclared grid, **anchored on G-Retriever's own two values**
#: (``C_e = 1.0`` for SceneGraphs, ``0.5`` for WebQSP, Appendix B.1) rather than
#: invented around them.
EDGE_COST_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

#: The exact solve enumerates a component's connected subsets, so its cost is
#: ``2^|component|``.  Asserted rather than hoped for: measured components on
#: this environment hold at most a handful of atoms, and a pool that broke that
#: should fail loudly instead of running for a week.
MAX_COMPONENT_ATOMS = 20


def reference_components(
    atoms: Sequence[str], pool: AtomPool
) -> list[tuple[str, ...]]:
    """Connected components of the **reference graph** over ``atoms``.

    Mapping A: a ``refs`` link is an undirected PCST edge. Only links whose both
    ends are inside ``atoms`` count — an atom referencing something outside the
    ground set is isolated here, which is correct, since that reference can never
    be satisfied within it.
    """
    index = {a: i for i, a in enumerate(atoms)}
    parent = list(range(len(atoms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in atoms:
        for ref in pool[a].refs:
            if ref in index:
                ra, rb = find(index[a]), find(index[ref])
                if ra != rb:
                    parent[ra] = rb

    groups: dict[int, list[str]] = {}
    for a in atoms:
        groups.setdefault(find(index[a]), []).append(a)
    return [tuple(sorted(g)) for g in groups.values()]


def _reference_edges(component: Sequence[str], pool: AtomPool) -> list[tuple[int, int]]:
    index = {a: i for i, a in enumerate(component)}
    edges: set[tuple[int, int]] = set()
    for a in component:
        for ref in pool[a].refs:
            if ref in index:
                u, v = index[a], index[ref]
                edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def _best_connected_subset(
    component: Sequence[str], pool: AtomPool, prize: Mapping[str, float], edge_cost: float
) -> tuple[tuple[str, ...], float]:
    """Exact ``argmax_S connected  Σ prize(S) − C_e·(|S| − 1)``.

    Ties break toward the **smaller** set, then lexicographically by atom id, so
    the method is deterministic as decision 4 declares it.
    """
    n = len(component)
    if n > MAX_COMPONENT_ATOMS:
        raise ValueError(
            f"component of {n} atoms exceeds the exact solver's declared bound "
            f"{MAX_COMPONENT_ATOMS}; widen it deliberately rather than letting "
            "the enumeration silently become intractable"
        )
    edges = _reference_edges(component, pool)
    adjacency = [0] * n
    for u, v in edges:
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u

    best_set: tuple[str, ...] = ()
    best_value = 0.0  # the empty subgraph, worth nothing and costing nothing
    for size in range(1, n + 1):
        for combo in itertools.combinations(range(n), size):
            bits = 0
            for i in combo:
                bits |= 1 << i
            # connectivity: flood from the lowest member inside `bits`
            seen = 1 << combo[0]
            frontier = [combo[0]]
            while frontier:
                node = frontier.pop()
                nxt = adjacency[node] & bits & ~seen
                while nxt:
                    low = nxt & -nxt
                    j = low.bit_length() - 1
                    seen |= low
                    frontier.append(j)
                    nxt ^= low
            if seen != bits:
                continue
            value = sum(prize[component[i]] for i in combo) - edge_cost * (size - 1)
            candidate = tuple(component[i] for i in combo)
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12
                and best_set
                and (len(candidate), candidate) < (len(best_set), best_set)
            ):
                best_set, best_value = candidate, value
    return best_set, best_value


def solve_pcst_forest(
    atoms: Sequence[str],
    pool: AtomPool,
    prize: Mapping[str, float],
    edge_cost: float,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Solve each component exactly and union the results (decision 2's forest).

    A component whose best value is not positive contributes nothing: PCST's
    objective values the empty subgraph at 0, so adding a zero-prize atom to
    reach it would be paying set size for no prize.
    """
    components = reference_components(atoms, pool)
    chosen: list[str] = []
    used = 0
    for component in components:
        best, value = _best_connected_subset(component, pool, prize, edge_cost)
        if best and value > 0.0:
            chosen.extend(best)
            used += 1
    return tuple(sorted(chosen)), {
        "components": len(components),
        "components_used": used,
        "largest_component": max((len(c) for c in components), default=0),
    }


def _prizes_top_k(
    atoms: Sequence[str], relevance: Mapping[str, float], k: int
) -> dict[str, float]:
    """G-Retriever Eq. 6: the top ``k`` atoms get ``k … 1``, the rest 0.

    Rank is by relevance descending, ties by atom id so the assignment is
    deterministic.
    """
    ranked = sorted(atoms, key=lambda a: (-float(relevance.get(a, 0.0)), a))
    prize = {a: 0.0 for a in atoms}
    for i, a in enumerate(ranked[:k]):
        prize[a] = float(k - i)
    return prize


def calibrate_edge_cost(
    envs: Sequence[Any],
    relevance_for: Callable[[Any], Mapping[str, float]],
    *,
    grid: Sequence[float] = EDGE_COST_GRID,
    k: int | None = None,
) -> dict[str, Any]:
    """Decision 8's procedure, executable — **selection on the breach rate**.

    Grid on the **tuning** suite only, the same separation Phase-2 decision 22
    enforces for β. All quantities are *post*-completion, because completion
    only ever **adds** atoms: targeting the cap before it puts roughly half the
    outputs over it afterwards, where ``H`` rejects them on size.

    **The criterion is minimum post-completion breach rate, ties to the larger
    median** — not "median closest to ``max_atoms`` without exceeding it", which
    is what decision 8 said until it was measured. A median that *equals* the cap
    satisfies "does not exceed" while putting **half** the outputs over it:
    measured on the tuning suite at ``C_e = 0.5`` under obligation relevance, the
    median is exactly 8.0 = ``max_atoms`` and the breach rate is **0.500**. That
    is the straddle the plan's own amendment warned about, reached through the
    letter of its replacement. The breach rate measures the thing the criterion
    was always *for*, and admits no straddle.

    Ties break toward the **larger** median deliberately: a bigger subgraph is
    more evidence and therefore a stronger baseline, and `CLAUDE.md` §8 wants S4
    dangerous.

    Purely structural throughout: it reads no ``U``, no best-of-K and no Gate-3
    number, so it cannot be tuned toward a result.
    """
    max_atoms = envs[0].instance.cfg.max_atoms
    rows: list[dict[str, Any]] = []
    for edge_cost in grid:
        sizes: list[int] = []
        for env in envs:
            inst = env.instance
            rel = relevance_for(env)
            ground = admissible_atoms(env, inst.obligations)
            cap = int(k if k is not None else inst.cfg.K)
            for top_k in range(1, cap + 1):
                prize = _prizes_top_k(ground, rel, top_k)
                raw, _ = solve_pcst_forest(ground, inst.pool, prize, edge_cost)
                sizes.append(len(close_under_refs(raw, inst.pool)))
        arr = np.asarray(sizes, dtype=float)
        rows.append(
            {
                "edge_cost": edge_cost,
                "median_size": float(np.median(arr)) if arr.size else float("nan"),
                "mean_size": float(arr.mean()) if arr.size else float("nan"),
                "max_size": int(arr.max()) if arr.size else 0,
                "breach_rate": float((arr > max_atoms).mean()) if arr.size else 0.0,
                "n": len(sizes),
            }
        )

    pick = min(rows, key=lambda r: (r["breach_rate"], -r["median_size"]))
    return {
        "chosen": pick["edge_cost"],
        "achieved_median": pick["median_size"],
        "achieved_breach_rate": pick["breach_rate"],
        "max_atoms": max_atoms,
        "grid": rows,
        "no_breach_free_value": pick["breach_rate"] > 0.0,
    }


class PCSTSearch:
    """S4.  Deterministic; reported once (decision 4).

    **Bypasses the ``ADD`` masks**, so it pays **1 terminal check per distinct
    candidate** (G6) — the same family cost as S3.
    """

    name = "s4_pcst"
    deterministic = True

    def __init__(
        self,
        relevance: Mapping[str, float],
        edge_cost: float,
        k: int | None = None,
    ) -> None:
        self.relevance = dict(relevance)
        self.edge_cost = float(edge_cost)
        self.k = k

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

        # Decision 3: S4's portfolio is **G-Retriever's own top-k knob** — the
        # paper's control over subgraph extent — swept over k = 1..K.
        candidates: list[tuple[str, ...]] = []
        completed_atoms = 0
        raw_atoms = 0
        meta: dict[str, Any] = {}
        for top_k in range(1, k + 1):
            prize = _prizes_top_k(ground, self.relevance, top_k)
            raw, meta = solve_pcst_forest(ground, inst.pool, prize, self.edge_cost)
            closed = close_under_refs(raw, inst.pool)
            raw_atoms += len(raw)
            completed_atoms += len(closed) - len(raw)
            candidates.append(closed)

        breaches = sum(1 for c in candidates if len(c) > cfg.max_atoms)
        needed_completion = sum(
            1
            for c, top_k in zip(candidates, range(1, k + 1))
            if len(c) > len(solve_pcst_forest(
                ground, inst.pool, _prizes_top_k(ground, self.relevance, top_k),
                self.edge_cost,
            )[0])
        )
        valid, spent, exhausted, rejections, portfolio = h_filter(
            candidates, obligations, inst.graph, inst.pool, cfg, ledger
        )
        scored = sorted(((scorer(p.atoms), p) for p in valid), key=lambda t: -t[0])
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
                "edge_cost": self.edge_cost,
                "completion_rate": needed_completion / max(1, len(candidates)),
                "atoms_added_by_completion": completed_atoms,
                "max_atoms_breaches": breaches,
                "ground_atoms": len(ground),
                "h_rejections": rejections,
                **meta,
            },
        )
