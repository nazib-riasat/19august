"""Action masks: which ``ADD``s are legal, and when ``STOP`` is allowed.

**Only ``STOP`` is masked when ``H = 0``; ``ADD`` stays available.** Traversing a
formally invalid partial set is fine — the policy simply cannot stop there. This
is why v1.2 §4.3 withdrew the `H`-monotonicity proof obligation, and re-deriving
that obligation is a recurring temptation worth naming: the binding constraints
are the ``ADD`` masks and ``max_atoms``, not ``H``.

What ``ADD`` *is* masked on is different in kind: an atom that can never appear in
any formally valid set. Selecting one poisons every descendant state, so
excluding it prunes branches that could only ever reach ``FAIL``.

There is **no ``ABSTAIN`` action** (v1.2 §3.4/§4.2). Abstention is decoupled from
the flow and handled by the answerability gate before Stage D runs, plus the
``FAIL`` terminal after it.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from graft.core.incremental import IncrementalChecker

__all__ = ["Terminal", "legal_adds", "legal_add_ids", "stop_allowed", "is_dead_end", "terminal_of"]


class Terminal(Enum):
    """The two ways a trajectory can end.

    ``FAIL`` is a genuine terminal and a **member of the target's support**
    (architecture fix F3), carrying ``r_fail``.  It is not "an invalid proof was
    returned" — it is "no proof found within budget", which is exactly the
    inference-time abstain fallback, so the write path and the read path agree on
    what it means.
    """

    VALID = "valid"
    FAIL = "fail"


def legal_adds(state: IncrementalChecker) -> np.ndarray:
    """Boolean mask over ``state.pool.ids()`` — the canonical index order.

    Sorted ids, because this vector becomes tensor positions in Phase 3 and set
    iteration order is randomised per process.

    Excluded:

    * atoms already selected;
    * every atom once ``|X| = max_atoms``, since any add would break sub-check 6;
    * atoms with a **per-atom** violation — type, retired, support, scope, or a
      backwards supersession.  Each of those is a permanent property of the atom
      and the snapshot, so such an atom cannot appear in any valid set;
    * atoms whose ``refs`` are not all selected — the closure rule (fix F10),
      one membership test per reference.

    Note the third item includes the supersession half of the temporal check,
    which the build plan's parenthetical "(1, 4, 5, 7)" omitted.  It belongs:
    the verdict is per-atom and permanent, so leaving it out would let the policy
    spend budget building sets that can never stop.

    Content-duplicate atoms (sub-check 2) are deliberately **not** masked. Atom
    ids are content-derived from exactly the fields ``content_key`` compares, so
    two atoms with one content key would share an id and the pool would have
    rejected them. The check remains as a guard against externally supplied
    pools; it is not a live pruning concern.
    """
    ids = state.pool.ids()
    mask = np.zeros(len(ids), dtype=bool)
    if len(state) >= state.cfg.max_atoms:
        return mask

    selected = set(state.selected())
    for i, atom_id in enumerate(ids):
        if atom_id in selected:
            continue
        atom = state.pool[atom_id]
        if any(ref not in selected for ref in atom.refs):
            continue
        if not state.atom_is_admissible(atom_id):
            continue
        mask[i] = True
    return mask


def legal_add_ids(state: IncrementalChecker) -> tuple[str, ...]:
    """The same answer as :func:`legal_adds`, as atom ids."""
    ids = state.pool.ids()
    return tuple(aid for aid, allowed in zip(ids, legal_adds(state)) if allowed)


def stop_allowed(state: IncrementalChecker) -> bool:
    """``STOP`` is allowed exactly when the current set is formally valid.

    Evaluated incrementally, so construction costs no ``terminal_checks``
    (Phase-0 gap G1).  Always ``False`` at the root, because the empty set is not
    a legal terminal (Phase-1 gap G1).
    """
    return state.ok()


def is_dead_end(state: IncrementalChecker) -> bool:
    """No legal ``ADD`` and ``STOP`` masked.

    **This has several causes, not one.**  Budget exhaustion at
    ``|X| = max_atoms`` with ``H = 0``; an empty pool at ``|X| = 0``; a pool
    exhausted below ``max_atoms``; and every remaining atom failing a per-atom
    check or waiting on unselected refs.  All of them transition to ``FAIL``.

    What distinguishes a healthy environment from over-tight masks is *where*
    they occur, which is why Phase 2 reports the ``|X|`` distribution of dead
    ends rather than a single rate.
    """
    if stop_allowed(state):
        return False
    return not legal_adds(state).any()


def terminal_of(state: IncrementalChecker) -> Terminal | None:
    """``VALID`` if the set may stop, ``FAIL`` if it is stuck, ``None`` mid-construction.

    A state that may stop is reported as ``VALID`` even when adds remain — the
    policy chooses whether to take them.
    """
    if stop_allowed(state):
        return Terminal.VALID
    if not legal_adds(state).any():
        return Terminal.FAIL
    return None
