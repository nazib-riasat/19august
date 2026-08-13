"""S1 — greedy on the scorer, through the masks.

Spec (architecture Phase 4, S1): *iteratively ``ADD`` the argmax ``scorer``
gain; ``STOP`` when allowed and gain < 0.*

**Why greedy is the arm to watch, not the formality.** Measured before any
method was built (G9), greedy on exact ``U`` attains the **global optimum on
30/30 instances** across all three suites, while a flawless reward-proportional
sampler reaches only 1.8865 of greedy's 1.9245 at ``K = 8``. Under fix F13's
perfect scorer, optimising the scorer directly is simply correct — which is why
decision 5 moves Gate 3's verdict to Phase 9 and forbids reading any best-of-K
comparison here as a result.

**Its portfolio, and the honest label** (decision 3, G3). Eight runs, each
forced to a different first atom — the eight highest-``rel`` legal openers. Three
mechanisms were tried for getting eight *distinct* sets out of greedy and all
three failed for one reason: G9's funnelling sends greedy to the same optimum
whatever the opener, so forced-distinct openers measure **2.45** distinct sets
(range 2–3, 20/20), barely above the ε-restarts' 2.10 they replaced. So this
module *attempts* ``K`` and **reports what it actually returned**. Engineering
eight forks out of greedy would be engineering the baseline.

**Spend: 0 terminal checks** (G6, decision 6). ``masks.stop_allowed`` *is* ``H``,
so every set this module returns is ``H``-valid by construction and there is
nothing left to validate.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

from graft.core.masks import legal_add_ids, stop_allowed
from graft.ledger import Ledger
from graft.schemas import Obligations, ProofSet
from graft.setgen.search.base import SearchResult, build_checker

__all__ = ["GreedySearch", "greedy_from"]


def greedy_from(
    env: Any,
    q: Obligations,
    scorer: Callable[[Iterable[str]], float],
    opener: str | None = None,
) -> tuple[str, ...] | None:
    """One greedy run, optionally forced to start at ``opener``.

    Returns the selected atoms, or ``None`` when the run dead-ended (no legal
    ``ADD`` and ``STOP`` masked) and therefore produced no candidate at all.

    **Tie-break: the lexicographically smallest atom id among the maximal
    gains.** Frozen here rather than left to dictionary order because the
    method is declared deterministic (decision 4) and reported once — a
    tie-break that varied per process would make "deterministic" false. It is
    also the rule under which G9's 30/30 optimality was measured, so changing it
    would invalidate that measurement rather than merely reorder a table.

    **Stopping:** at a state where ``STOP`` is allowed, stop as soon as no legal
    add has a **strictly positive** gain. A plateau (gain exactly 0) is a larger
    set with no better score, which ``U``'s ``size`` term already prices.
    """
    state = build_checker(env, q)
    if opener is not None:
        if opener not in set(legal_add_ids(state)):
            return None
        state.add(opener)

    while True:
        legal = legal_add_ids(state)
        can_stop = stop_allowed(state)
        if not legal:
            return state.selected() if can_stop else None
        current = scorer(state.selected())
        best_atom, best_gain = None, -float("inf")
        for atom_id in legal:  # legal_add_ids is already in canonical id order
            gain = scorer(state.selected() + (atom_id,)) - current
            if gain > best_gain:
                best_atom, best_gain = atom_id, gain
        if can_stop and best_gain <= 0.0:
            return state.selected()
        state.add(best_atom)


class GreedySearch:
    """S1.  Deterministic; reported once (decision 4)."""

    name = "s1_greedy"
    deterministic = True

    def __init__(self, relevance: Mapping[str, float], k: int | None = None) -> None:
        self.relevance = dict(relevance)
        self.k = k

    def _openers(self, env: Any, q: Obligations, k: int) -> Sequence[str]:
        """The ``k`` highest-``rel`` legal first atoms, ties broken by id.

        Only atoms legal at the root qualify: a forced opener that the mask
        rejects would silently produce no run, so the count would drift below
        ``k`` for a reason that has nothing to do with search.
        """
        root = build_checker(env, q)
        legal = legal_add_ids(root)
        return sorted(legal, key=lambda a: (-self.relevance.get(a, 0.0), a))[:k]

    def run(
        self,
        env: Any,
        obligations: Obligations,
        scorer: Callable[[Iterable[str]], float],
        ledger: Ledger | None = None,
    ) -> SearchResult:
        k = int(self.k if self.k is not None else env.instance.cfg.K)
        openers = self._openers(env, obligations, k)
        raw: list[tuple[str, ...]] = []
        dead_ends = 0
        for opener in openers:
            found = greedy_from(env, obligations, scorer, opener)
            if found is None:
                dead_ends += 1
                continue
            raw.append(found)

        # Valid by construction: every run returns only at a state where
        # ``stop_allowed`` -- which *is* ``H`` -- held.  No filter, no spend.
        pool = env.instance.pool
        seen: dict[tuple[str, ...], ProofSet] = {}
        for atoms in raw:
            if atoms not in seen:
                seen[atoms] = ProofSet(
                    atoms=frozenset(atoms), bindings=pool.derive_bindings(atoms)
                )
        scored = sorted(
            ((scorer(a), a) for a in seen), key=lambda t: (-t[0], t[1])
        )
        return SearchResult(
            self.name,
            [seen[a] for _, a in scored],
            [s for s, _ in scored],
            attempted=len(openers),
            distinct_attempted=len(seen),
            terminal_checks=0,
            # **With multiplicity** (decision 5): eight openers that funnel to
            # one optimum are a one-fold portfolio wearing an eight-fold label,
            # and the pinned diversity estimator must be able to see that.
            portfolio=[seen[atoms] for atoms in raw],
            extra={"forced_openers": len(openers), "dead_ended_runs": dead_ends},
        )
