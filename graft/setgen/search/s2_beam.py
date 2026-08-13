"""S2 — beam search over partial sets, through the masks.

Spec (architecture Phase 4, S2): *width ``b = K`` over partial sets, score =
``scorer``; dedup by canonical set hash.*

**Its eight candidates are the best terminals encountered *anywhere* in the
search, not the final beam layer** (decision 3, G3). Terminals appear at many
layers of this DAG — a state that may ``STOP`` can also have children — so the
final layer and "the best terminals seen" are different objects, and only the
second is what a portfolio wants. An earlier revision left "eight survivors"
undefined; this is the definition.

Measured consequence, reported rather than assumed: beam-8 returns **8.00
distinct sets on 20/20** main-suite instances (there are 353–552 terminals to
draw from), which is why decision 5 could not use a distinct-**count** rule as
Gate 3's gate — beam saturates the cap while the ``p*`` ceiling is 7.78, so
"S5 returns strictly more" has ``P(pass) = 0``.

**Spend: 0 terminal checks** — mask-driven, so ``stop_allowed`` *is* ``H`` and
every returned set is valid by construction (G6).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from graft.core.masks import legal_add_ids, stop_allowed
from graft.ids import canon_set_hash
from graft.ledger import Ledger
from graft.schemas import Obligations, ProofSet
from graft.setgen.search.base import SearchResult, build_checker

__all__ = ["BeamSearch"]


class BeamSearch:
    """S2.  Deterministic; reported once (decision 4)."""

    name = "s2_beam"
    deterministic = True

    def __init__(self, width: int | None = None, k: int | None = None) -> None:
        self.width = width
        self.k = k

    def run(
        self,
        env: Any,
        obligations: Obligations,
        scorer: Callable[[Iterable[str]], float],
        ledger: Ledger | None = None,
    ) -> SearchResult:
        cfg = env.instance.cfg
        k = int(self.k if self.k is not None else cfg.K)
        width = int(self.width if self.width is not None else k)

        # Frontier of partial sets, and every terminal seen at any layer.
        frontier: list[tuple[str, ...]] = [()]
        terminals: dict[str, tuple[float, tuple[str, ...]]] = {}
        expansions = 0

        for _ in range(cfg.max_atoms + 1):
            children: dict[str, tuple[str, ...]] = {}
            for atoms in frontier:
                state = build_checker(env, obligations, atoms)
                if stop_allowed(state):
                    key = canon_set_hash(frozenset(atoms))
                    if key not in terminals:
                        terminals[key] = (scorer(atoms), atoms)
                for atom_id in legal_add_ids(state):
                    child = tuple(sorted(atoms + (atom_id,)))
                    # Dedup by canonical set hash (Phase 0's ``ids.py``): two
                    # insertion orders reaching the same atoms are one state, so
                    # expanding both would spend width on a duplicate.
                    children.setdefault(canon_set_hash(frozenset(child)), child)
                expansions += 1
            if not children:
                break
            ranked = sorted(
                children.values(), key=lambda a: (-scorer(a), a)
            )
            frontier = ranked[:width]

        best = sorted(terminals.values(), key=lambda t: (-t[0], t[1]))[:k]
        pool = env.instance.pool
        sets = [
            ProofSet(atoms=frozenset(a), bindings=pool.derive_bindings(a))
            for _, a in best
        ]
        return SearchResult(
            self.name,
            sets,
            [s for s, _ in best],
            attempted=len(terminals),
            distinct_attempted=len(terminals),
            terminal_checks=0,
            extra={"beam_width": width, "state_expansions": expansions},
        )
