"""P4.1 — the protocol every search method implements, and the shared filter.

**Spend is a returned value, not a side effect to be read off a global** (G3,
G6), so a method that under-spends is visible in its own row rather than in an
aggregate nobody reads.

**The budget is enforced, not observed** (decision 6, `CLAUDE.md` §6):
``would_exceed()`` *before* spending, so a method stops cleanly instead of
overrunning and being noticed afterwards.

**And the spend differs by family, which is the finding, not an accounting
detail** (G6). ``masks.stop_allowed`` **is** ``H``, so a terminal reached
through the masks is ``H``-valid by construction and has nothing left to
validate: S1, S2 and S5 spend **0** terminal checks. S3 and S4 bypass the masks
and build sets directly, so they pay **1 per distinct candidate** — dedup by
``canon_set_hash`` first, which is free. Charging every method a flat ``K`` was
accounting fiction that erased exactly this difference.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from graft.config import Config
from graft.core.checker import H
from graft.core.incremental import IncrementalChecker
from graft.ids import canon_set_hash
from graft.ledger import Ledger
from graft.schemas import AtomPool, Obligations, ProofSet

__all__ = [
    "SearchModule",
    "SearchResult",
    "admissible_atoms",
    "build_checker",
    "close_under_refs",
    "dedup_sets",
    "h_filter",
]


class SearchResult:
    """One method's output on one instance.

    ``sets`` are the **valid** candidates it returns, best first by ``scorer``.
    ``attempted`` is how many candidates it generated before filtering — decision
    3 requires the distinct-valid count to be *reported*, never assumed to be
    ``K``, because greedy genuinely returns ~2.45 on this environment and a table
    saying 8 would be the label the ruling is about.
    """

    __slots__ = (
        "method",
        "sets",
        "scores",
        "portfolio",
        "attempted",
        "distinct_attempted",
        "terminal_checks",
        "budget_exhausted",
        "extra",
    )

    def __init__(
        self,
        method: str,
        sets: Sequence[ProofSet],
        scores: Sequence[float],
        *,
        attempted: int,
        distinct_attempted: int,
        terminal_checks: int,
        portfolio: Sequence[ProofSet] | None = None,
        budget_exhausted: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if len(sets) != len(scores):
            raise ValueError(f"{len(sets)} sets against {len(scores)} scores")
        self.method = method
        self.sets = tuple(sets)
        self.scores = tuple(float(s) for s in scores)
        # **The returned portfolio WITH multiplicity** — decision 5 pins the
        # diversity estimator "over the C(K,2) unordered pairs of the returned
        # sets, **duplicate pairs included**", and that convention is
        # unreachable from ``sets``, which is deduplicated for ranking. Keeping
        # both is what lets the one live Gate-3 condition be computed as ruled:
        # measured, the deduplicated estimator passes a sampler that has lost
        # 27% of its modes where the pinned one fails it.
        self.portfolio = tuple(portfolio) if portfolio is not None else tuple(sets)
        self.attempted = int(attempted)
        self.distinct_attempted = int(distinct_attempted)
        self.terminal_checks = int(terminal_checks)
        self.budget_exhausted = bool(budget_exhausted)
        self.extra = dict(extra or {})

    # -- the reported quantities -------------------------------------------

    @property
    def n_valid(self) -> int:
        return len(self.sets)

    @property
    def distinct_valid(self) -> int:
        """Distinct valid sets returned — exit criterion 12c's number.

        Reported per method and never assumed: every method *attempts* ``K = 8``
        and S1/S3 measure ~2.45 because greedy funnels to one optimum whatever
        the opener (G3, G9).
        """
        return len({canon_set_hash(s.atoms) for s in self.sets})

    @property
    def best_utility(self) -> float:
        """Best-of-K under the scorer.  ``nan`` when nothing valid was returned.

        Reported **beside the ``p*`` ceiling** (decision 10) and never against a
        rival as a Gate-3 verdict, in either direction (decision 5).
        """
        return self.scores[0] if self.scores else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_valid": self.n_valid,
            "portfolio_size": len(self.portfolio),
            "distinct_valid": self.distinct_valid,
            "best_utility": self.best_utility,
            "scores": list(self.scores),
            "sizes": [len(s.atoms) for s in self.sets],
            "attempted": self.attempted,
            "distinct_attempted": self.distinct_attempted,
            "terminal_checks": self.terminal_checks,
            "budget_exhausted": self.budget_exhausted,
            **self.extra,
        }


@runtime_checkable
class SearchModule(Protocol):
    """``run(env, obligations, scorer, ledger) -> SearchResult``.

    ``runtime_checkable`` so the registry's membership is testable rather than
    promised, the same reason Phase 2 made ``ActionPolicy`` checkable.
    """

    name: str
    deterministic: bool

    def run(
        self,
        env: Any,
        obligations: Obligations,
        scorer: Callable[[Iterable[str]], float],
        ledger: Ledger | None,
    ) -> SearchResult: ...


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def build_checker(
    env: Any,
    q: Obligations,
    atoms: Iterable[str] = (),
    ledger: Ledger | None = None,
) -> IncrementalChecker:
    """A fresh :class:`IncrementalChecker` holding ``atoms``.

    The mask-driven methods (S1, S2) need many independent partial states;
    replaying a set into a new checker is the cheapest correct way to get one,
    and ``add`` order does not affect the final verdict because closure is a
    *set-level* sub-check. The masks mid-construction **are** order-dependent,
    which is correct and is why the caller adds in the order it chose.

    ``ledger=None`` by default and by design: construction spends
    ``incremental_ops``, which are uncapped and reported rather than budgeted
    (Phase-0 gap G1), while the *search* budget being metered is
    ``terminal_checks``. Passing a ledger here would meter construction against
    a counter that the Robust-Scheduling framing defines over the expensive
    evaluator only.
    """
    inst = env.instance
    state = IncrementalChecker(inst.pool, q, inst.graph, inst.cfg, ledger=ledger)
    for atom_id in atoms:
        state.add(atom_id)
    return state


def admissible_atoms(env: Any, q: Obligations) -> tuple[str, ...]:
    """The pool's atoms that no per-atom sub-check rejects.

    **The ground set the direct builders get, and why it is not a favour.**
    A per-atom violation — a retired edge, a quarantined assertion, an
    out-of-scope provenance chain, a backwards supersession — is a *permanent*
    property of the atom and the snapshot, so it poisons every set containing
    it. The ``ADD`` masks prune exactly these for S1/S2/S5 at every state
    (Phase-1 §2.5), and Phase-2 exit criterion 17 requires the planted
    invalidated edge and quarantined assertion to be **present in the snapshot
    and asserted *unreachable*** — *"negative cases for sub-checks 4 and 7, not
    selectable evidence"*.

    Handing S3 and S4 the same view is therefore the fair comparison, not a
    concession: the alternative spends a direct builder's whole portfolio on
    atoms no method could ever legally use, which is how a baseline gets
    accidentally weakened (`CLAUDE.md` §8, G1). It is also what Phase-4 decision
    2 already rules for S4's graph, and the reasoning does not change for S3.

    **It costs zero terminal checks**: per-atom admissibility is construction-time
    validity, which Phase-0 gap G1 makes free and G6 keeps off the budget axis.
    """
    state = build_checker(env, q)
    return tuple(a for a in env.instance.pool.ids() if state.atom_is_admissible(a))


def close_under_refs(atoms: Iterable[str], pool: AtomPool) -> tuple[str, ...]:
    """The smallest closed superset of ``atoms`` — fix F10's completion step.

    A direct builder can select an edge without its endpoints; closure is a
    *feasibility* rule of this environment rather than part of anyone's
    objective, so the repair is mechanical: pull in every referenced atom
    transitively until the set is closed.

    Decision 2 rules this for S4. It applies unchanged to S3, which bypasses the
    masks for the same reason and would otherwise be the only method structurally
    unable to return anything — an asymmetry that would decide S3's strength by
    omission rather than by measurement.

    Completion only ever **adds** atoms, which is why decision 8's ``C_e``
    criterion targets the post-completion size and why the ``max_atoms`` breach
    rate is a reported number (exit criterion 12).
    """
    out = set(atoms)
    frontier = list(out)
    while frontier:
        atom_id = frontier.pop()
        if atom_id not in pool:
            continue
        for ref in pool[atom_id].refs:
            if ref not in out:
                out.add(ref)
                frontier.append(ref)
    return tuple(sorted(out))


def dedup_sets(candidates: Iterable[Iterable[str]]) -> list[tuple[str, ...]]:
    """Distinct atom sets, order-insensitively, in first-seen order.

    Dedup happens **before** validation (decision 3): ``canon_set_hash`` is free
    and a duplicate candidate is the same set, so validating it twice would
    charge the budget for no information. It is also what decides whether a
    direct builder's budget-curve point sits at 8 or at its true distinct count.
    """
    seen: set[str] = set()
    out: list[tuple[str, ...]] = []
    for cand in candidates:
        atoms = tuple(sorted(set(cand)))
        key = canon_set_hash(frozenset(atoms))
        if key in seen:
            continue
        seen.add(key)
        out.append(atoms)
    return out


def h_filter(
    candidates: Iterable[Iterable[str]],
    q: Obligations,
    graph: Any,
    pool: AtomPool,
    cfg: Config,
    ledger: Ledger | None,
) -> tuple[list[ProofSet], int, bool, dict[str, int], list[ProofSet]]:
    """Validate distinct candidates against ``H``, under the enforced budget.

    Returns ``(valid sets, terminal checks spent, budget exhausted, rejections,
    portfolio)`` — ``valid`` deduplicated for ranking, ``portfolio`` carrying the
    multiplicity decision 5's diversity estimator is defined over, and
    ``rejections`` counting failing sub-checks by name.

    **Validation still happens once per distinct candidate** (decision 3): the
    dedup is what keeps the budget honest, and re-expanding afterwards costs
    nothing because the verdict is cached per set.

    **The rejection categories are a first-class result, not a debug aid.** A
    direct builder optimises an objective with no notion of this environment's
    set-level constraints — the submodular objective has no analogue of "one
    binding per slot" (sub-check 9) — so what it gets rejected *for* is a
    measurement of the gap between a published packer's objective and formal
    proof validity. Measured on the tuning suite, S3's surviving failures are
    exactly that: duplicate ``answer`` bindings the ``ADD`` masks would have
    prevented and a direct build cannot see.

    **This is the direct-builder path only.** S1, S2 and S5 construct through
    the masks, where ``stop_allowed`` *is* ``H``, so they are valid by
    construction and must not come through here — routing them through it would
    charge them a check each and erase the family difference G6 exists to
    expose.

    ``would_exceed`` is consulted *before* each spend, so the method stops
    cleanly at the cap rather than overrunning and being noticed afterwards.
    """
    originals = [tuple(sorted(set(c))) for c in candidates]
    verdicts: dict[tuple[str, ...], ProofSet | None] = {}
    rejections: dict[str, int] = {}
    spent = 0
    exhausted = False
    for atoms in dedup_sets(originals):
        if ledger is not None and ledger.would_exceed("terminal_checks", 1):
            exhausted = True
            break
        result = H(atoms, q, graph, pool, cfg, ledger=ledger)
        spent += 1
        if result.ok:
            verdicts[atoms] = ProofSet(
                atoms=frozenset(atoms), bindings=pool.derive_bindings(atoms)
            )
        else:
            verdicts[atoms] = None
            for violation in result.violations:
                rejections[violation.check] = rejections.get(violation.check, 0) + 1

    # Two lists, deliberately: ``valid`` is deduplicated for ranking, while the
    # **portfolio keeps multiplicity** — a direct builder that produced the same
    # set from three openers returned a three-fold-collapsed portfolio, and
    # decision 5's pinned diversity estimator has to be able to see that.
    valid = [p for p in verdicts.values() if p is not None]
    portfolio = [verdicts[a] for a in originals if verdicts.get(a) is not None]
    return valid, spent, exhausted, rejections, portfolio
