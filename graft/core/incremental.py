"""Incremental validity, so that construction spends no ``terminal_checks``.

Phase-0 gap G1 in one line: if every construction step evaluated ``H``, one
portfolio would cost up to ``K * max_atoms`` = 128 checks against a budget of 32,
and the Stage-D primary metric would be unmeasurable as specified.

**What is memoised, and what is not.** The per-atom sub-checks — type, retired,
support, scope, and the supersession half of temporal — do all the graph
traversal and depend on one atom alone, so they are computed once per atom and
reused at every state that atom appears in, across ``add`` and ``undo`` alike.
The set-level sub-checks are pure bookkeeping over at most ``max_atoms`` entries
and are recomputed; at 16 atoms that is cheaper than maintaining and having to
prove correct an incremental version of each.

That split is the honest reading of "amortised O(1) per ADD": the graph work is
done once, and what remains is a fixed-size scan with no I/O.

**Agreement with batch ``H`` is structural.** Both paths call
``checker.assemble`` over the same two dictionaries, so they cannot disagree
about ordering or content. Exit criterion 3 confirms it over 10^4 random
(set, order) pairs rather than being the only thing standing behind it.

``undo`` exists because beam search (Phase 4, S2) and local search backtrack, and
rebuilding from the root would reintroduce exactly the cost this class removes.
"""

from __future__ import annotations

import numpy as np

from graft.config import Config
from graft.core import checker, obligations
from graft.graphstore import GraphSnapshot
from graft.ledger import Ledger
from graft.schemas import AtomPool, CheckResult, Obligations, ProofSet, Violation

__all__ = ["IncrementalChecker"]


class IncrementalChecker:
    """Validity of a partial set, maintained across ``ADD`` and ``undo``."""

    def __init__(
        self,
        pool: AtomPool,
        q: Obligations,
        G: GraphSnapshot,
        cfg: Config,
        ledger: Ledger | None = None,
    ) -> None:
        self.pool = pool
        self.q = q
        self.G = G
        self.cfg = cfg
        self._ledger = ledger
        self._order: list[str] = []
        self._selected: set[str] = set()
        self._memo: dict[str, dict[str, tuple[Violation, ...]]] = {}

    # -- construction ------------------------------------------------------

    def add(self, atom_id: str) -> None:
        """Select ``atom_id``.

        Spends one ``incremental_op``, never a ``terminal_check``: construction
        maintains validity incrementally and is free, and charging it would make
        constructive methods artificially expensive against one-shot methods like
        PCST.
        """
        if atom_id in self._selected:
            raise ValueError(f"atom {atom_id} is already selected")
        if self._ledger is not None:
            self._ledger.count("incremental_ops")
        self.per_atom(atom_id)
        self._order.append(atom_id)
        self._selected.add(atom_id)

    def per_atom(self, atom_id: str) -> dict[str, tuple[Violation, ...]]:
        """This atom's per-atom verdict, computed once and reused.

        Public because the ``ADD`` mask needs the same answer for atoms that are
        *not* selected, and recomputing it there would do the graph traversal
        twice per step.
        """
        verdict = self._memo.get(atom_id)
        if verdict is None:
            verdict = checker.per_atom_violations(atom_id, self.pool, self.q, self.G)
            self._memo[atom_id] = verdict
        return verdict

    def atom_is_admissible(self, atom_id: str) -> bool:
        """False when this atom can never appear in a formally valid set.

        A per-atom violation is a permanent property of the atom and the
        snapshot, so it poisons every descendant state that contains it.
        """
        return not any(self.per_atom(atom_id).values())

    def undo(self) -> str:
        """Unselect the most recently added atom and return it.

        The memo is deliberately *not* cleared: a backtracking search will visit
        the same atom again, and its per-atom verdict cannot have changed.
        """
        if not self._order:
            raise ValueError("nothing to undo; no atoms are selected")
        atom_id = self._order.pop()
        self._selected.discard(atom_id)
        return atom_id

    # -- state -------------------------------------------------------------

    def selected(self) -> tuple[str, ...]:
        """Sorted, because the set is the state — insertion order is not part of it."""
        return tuple(sorted(self._selected))

    def insertion_order(self) -> tuple[str, ...]:
        """The order atoms were added, which only ``undo`` cares about."""
        return tuple(self._order)

    def state(self) -> ProofSet:
        """The canonical :class:`ProofSet`, with bindings derived (gap G3)."""
        ids = self.selected()
        return ProofSet(atoms=frozenset(ids), bindings=self.pool.derive_bindings(ids))

    def __len__(self) -> int:
        return len(self._selected)

    def __contains__(self, atom_id: object) -> bool:
        return atom_id in self._selected

    # -- validity ----------------------------------------------------------

    def result(self, first_failure_only: bool = False) -> CheckResult:
        """The same :class:`CheckResult` batch ``H`` would return for this set."""
        ids = self.selected()
        set_level = checker.set_level_violations(ids, self.pool, self.q, self.G, self.cfg)
        return checker.assemble(ids, self._memo, set_level, first_failure_only)

    def ok(self) -> bool:
        """Whether ``STOP`` is currently allowed.

        Short-circuits on the first failing sub-check, since the answer is all
        the caller wants.
        """
        return self.result(first_failure_only=True).ok

    def deficit(self) -> np.ndarray:
        """``d(s) ∈ [0, 1]^6`` for the current partial set."""
        return obligations.deficit(self.selected(), self.pool, self.q, self.G)

    # -- diagnostics -------------------------------------------------------

    @property
    def memo_size(self) -> int:
        """Atoms whose per-atom verdict has been computed.

        Reported so that "the graph work is done once" is observable rather than
        asserted: over a rollout with backtracking this should stay at the number
        of *distinct* atoms visited, not the number of adds.
        """
        return len(self._memo)
