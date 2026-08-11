"""The closed sub-lattice, enumerated exhaustively and **once**.

**The state graph is policy-independent, and that is what makes Gate 2
affordable** (G2).  States, legal actions and stop flags depend only on the
instance and the masks; only ``P_F`` changes between policies.  So the graph is
enumerated once per instance, stored as integer-indexed ``(parent, action,
child)`` arrays, and every subsequent evaluation is one numpy pass over that edge
list.  The difference is not marginal: 7 learners x 3 seeds x 50 checkpoints x 20
instances is 35 hours of pure-Python dict passes against roughly half an hour of
numpy over a precomputed graph.

**States are ``uint64`` bitmasks over ``pool.ids()``, not hashes and not
``frozenset``s** (G3, decision 4).  With ``pool_cap <= 64`` a selected set *is* a
``uint64`` whose bit *i* marks ``pool.ids()[i]`` — an exact identity, no hashing,
no collisions to reason about, and 8 bytes against roughly 700 for a frozenset of
short strings.  ``canon_set_hash`` is retained as a **fingerprint** for logging
and cross-machine comparison, never as identity: an exact evaluator must not be
able to merge two distinct states because two 64-bit truncations happened to
agree, however unlikely.  The ``pool_cap <= 64`` assumption is asserted at
construction so a future widening fails loudly rather than truncating silently.

**Enumeration walks exactly the space the policy will.**  Successors come from
:func:`graft.core.masks.legal_adds` and validity from
:class:`~graft.core.incremental.IncrementalChecker`, so the enumerated graph
cannot drift from the environment a learner actually samples.  ``H`` is reached
with ``ledger=None``: enumerating a lattice would exhaust any per-query budget,
and this is an offline audit rather than a query (Phase-1 gap G9).

**Two fingerprints, because one cannot do both jobs** (decision 21).
:func:`environment_fingerprint` covers what was built *and enumerated* and
excludes β; :func:`graft.synth.exact.target_fingerprint` covers the β-dependent
layer.  Binding only the generator's *inputs* would let two machines whose masks
or checker differ enumerate different graphs and still agree on the digest —
which is exactly the disagreement a fingerprint exists to detect.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from graft.canonical import canonical_bytes, digest_of
from graft.config import Config
from graft.core.incremental import IncrementalChecker
from graft.core.masks import legal_adds, stop_allowed
from graft.ids import canon_set_hash
from graft.schemas import AtomPool, ProofSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.synth.lattice import LatticeInstance

__all__ = [
    "BandViolation",
    "StateGraph",
    "reachable_states",
    "valid_terminals",
    "environment_fingerprint",
    "state_fingerprints",
    "MAX_POOL_BITS",
]

#: The bitmask representation is valid while ``pool_cap`` fits in a ``uint64``.
MAX_POOL_BITS = 64


class BandViolation(RuntimeError):
    """A measurement fell outside a declared band (G1, G5, G10).

    Carries the measurement, so a rejection is diagnosable rather than a retry
    counter.  Lives here because the earliest and cheapest rejection is
    :func:`reachable_states` aborting the moment the state or edge count passes
    its cap; :mod:`graft.synth.lattice` re-exports it.
    """

    def __init__(self, band: str, measured: Any, expected: str) -> None:
        super().__init__(f"{band}: measured {measured!r}, expected {expected}")
        self.band = band
        self.measured = measured
        self.expected = expected


def _array_digest(arr: np.ndarray, dtype: str) -> dict[str, Any]:
    """Content digest of one array, byte order pinned.

    ``tobytes()`` on a native-order array would hash differently on a big-endian
    host.  Casting to an explicit little-endian dtype first costs one copy and
    removes the whole question.
    """
    fixed = np.ascontiguousarray(arr, dtype=np.dtype(dtype))
    return {
        "dtype": dtype,
        "shape": list(fixed.shape),
        "sha256": hashlib.sha256(fixed.tobytes()).hexdigest(),
    }


# --------------------------------------------------------------------------
# StateGraph
# --------------------------------------------------------------------------


class StateGraph:
    """Every reachable closed state, its legal transitions, and its flags.

    Layered by ``|S|``: state ``0`` is the root and indices are sorted by set
    size, which is the order the forward DP requires.  Edges are stored sorted by
    parent index (and therefore by parent layer), so both the DP and the flow
    recurrence are contiguous slices rather than lookups.
    """

    __slots__ = (
        "atom_ids",
        "max_atoms",
        "mask",
        "size",
        "edge_parent",
        "edge_action",
        "edge_child",
        "stop_allowed",
        "dead_end",
        "_state_layer",
        "_edge_layer",
        "_edge_start",
        "_index",
        "terminal_ix",
        "dead_ix",
        "indegree",
    )

    def __init__(
        self,
        atom_ids: Sequence[str],
        max_atoms: int,
        mask: np.ndarray,
        size: np.ndarray,
        edge_parent: np.ndarray,
        edge_action: np.ndarray,
        edge_child: np.ndarray,
        stop_flags: np.ndarray,
        dead_flags: np.ndarray,
    ) -> None:
        if len(atom_ids) > MAX_POOL_BITS:
            raise ValueError(
                f"pool holds {len(atom_ids)} atoms; the uint64 state representation "
                f"addresses at most {MAX_POOL_BITS}. Widen the representation "
                "deliberately rather than letting a mask truncate (G3)."
            )
        self.atom_ids = tuple(atom_ids)
        self.max_atoms = int(max_atoms)
        self.mask = mask
        self.size = size
        self.edge_parent = edge_parent
        self.edge_action = edge_action
        self.edge_child = edge_child
        self.stop_allowed = stop_flags
        self.dead_end = dead_flags

        n = int(mask.shape[0])
        self._index = {int(m): i for i, m in enumerate(mask.tolist())}
        # Layer offsets: states are sorted by |S| by construction (BFS over a
        # layered DAG), so one searchsorted gives every boundary.
        self._state_layer = np.searchsorted(size, np.arange(self.max_atoms + 2), side="left")
        self._edge_layer = np.searchsorted(
            size[edge_parent] if edge_parent.size else np.zeros(0, dtype=np.int64),
            np.arange(self.max_atoms + 2),
            side="left",
        )
        counts = np.bincount(edge_parent, minlength=n) if edge_parent.size else np.zeros(n, np.int64)
        self._edge_start = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.terminal_ix = np.flatnonzero(stop_flags).astype(np.int64)
        self.dead_ix = np.flatnonzero(dead_flags).astype(np.int64)
        self.indegree = (
            np.bincount(edge_child, minlength=n).astype(np.int64)
            if edge_child.size
            else np.zeros(n, np.int64)
        )

    # -- shape -------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return int(self.mask.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_parent.shape[0])

    @property
    def n_atoms(self) -> int:
        return len(self.atom_ids)

    @property
    def out_degree(self) -> np.ndarray:
        """Legal ``ADD``s per state, from the CSR offsets."""
        return (self._edge_start[1:] - self._edge_start[:-1]).astype(np.int64)

    @property
    def n_terminals(self) -> int:
        return int(self.terminal_ix.shape[0])

    @property
    def n_dead_ends(self) -> int:
        return int(self.dead_ix.shape[0])

    # -- lookup ------------------------------------------------------------

    def index_of(self, mask: int) -> int:
        return self._index[int(mask)]

    def mask_for(self, atom_ids: Iterable[str]) -> int:
        """The ``uint64`` bitmask of an atom set, as a Python int."""
        position = {aid: i for i, aid in enumerate(self.atom_ids)}
        m = 0
        for aid in atom_ids:
            m |= 1 << position[aid]
        return m

    def state_of(self, atom_ids: Iterable[str]) -> int | None:
        """State index of an atom set, or ``None`` when it is not reachable."""
        return self._index.get(self.mask_for(atom_ids))

    def atoms_of(self, state: int) -> tuple[str, ...]:
        m = int(self.mask[state])
        return tuple(self.atom_ids[i] for i in range(self.n_atoms) if (m >> i) & 1)

    def proofset_of(self, state: int, pool: AtomPool) -> ProofSet:
        atoms = self.atoms_of(state)
        return ProofSet(atoms=frozenset(atoms), bindings=pool.derive_bindings(atoms))

    def children_of(self, state: int) -> tuple[np.ndarray, np.ndarray]:
        """``(actions, child_state_ix)`` for one state.

        Exit criterion 18: every legal ``ADD`` maps to the index of the child it
        produces, so a **Look-Ahead** featuriser — the published remedy for the
        GNN-expressivity limit v1.2 §6.4 names as a risk — stays implementable
        behind the frozen protocol.  Asserting that a capability is not foreclosed
        needs a test, not a sentence.
        """
        lo, hi = int(self._edge_start[state]), int(self._edge_start[state + 1])
        return self.edge_action[lo:hi], self.edge_child[lo:hi]

    def state_slice(self, size: int) -> tuple[int, int]:
        """Half-open index range of the states with ``|S| == size``."""
        return int(self._state_layer[size]), int(self._state_layer[size + 1])

    def edge_slice(self, parent_size: int) -> tuple[int, int]:
        """Half-open index range of the edges whose parent has ``|S| == parent_size``."""
        return int(self._edge_layer[parent_size]), int(self._edge_layer[parent_size + 1])

    # -- identity ----------------------------------------------------------

    def fingerprint(self) -> str:
        """Digest of the enumerated graph itself — masks, edges and flags.

        Included in :func:`environment_fingerprint` because binding only the
        generator's inputs would let two machines whose masks or checker differ
        enumerate different graphs and still agree.
        """
        return digest_of(
            {
                "atom_ids": list(self.atom_ids),
                "max_atoms": self.max_atoms,
                "mask": _array_digest(self.mask, "<u8"),
                "size": _array_digest(self.size, "<i2"),
                "edge_parent": _array_digest(self.edge_parent, "<i8"),
                "edge_action": _array_digest(self.edge_action, "<i8"),
                "edge_child": _array_digest(self.edge_child, "<i8"),
                "stop_allowed": _array_digest(self.stop_allowed, "|b1"),
                "dead_end": _array_digest(self.dead_end, "|b1"),
            }
        )

    def counts(self) -> dict[str, int]:
        return {
            "states": self.n_states,
            "edges": self.n_edges,
            "terminals": self.n_terminals,
            "dead_ends": self.n_dead_ends,
            "atoms": self.n_atoms,
        }


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------


class _Walker:
    """One :class:`IncrementalChecker`, moved from state to state.

    The per-atom verdicts do all the graph traversal and are memoised on the
    checker, so reusing a single instance across the whole enumeration pays for
    them once for the pool rather than once per state.  Only the public
    ``add``/``undo`` surface is used, so the memo's invariants stay the
    checker's business.
    """

    __slots__ = ("chk",)

    def __init__(self, instance: "LatticeInstance", cfg: Config) -> None:
        self.chk = IncrementalChecker(
            instance.pool, instance.obligations, instance.graph, cfg, ledger=None
        )

    def goto(self, atom_ids: Iterable[str]) -> IncrementalChecker:
        while len(self.chk):
            self.chk.undo()
        for aid in atom_ids:
            self.chk.add(aid)
        return self.chk


def reachable_states(
    instance: "LatticeInstance",
    cfg: Config | None = None,
    *,
    max_states: int | None = None,
    max_edges: int | None = None,
) -> StateGraph:
    """Breadth-first over set sizes, from the empty root.

    ``max_states`` / ``max_edges`` default to the instance's spec and abort the
    enumeration the moment they are exceeded (G1): an oversized instance is
    rejected in the time it takes to pass the bound, not after a full sweep.
    """
    cfg = cfg if cfg is not None else instance.cfg
    spec = instance.spec
    cap_states = spec.max_states if max_states is None else max_states
    cap_edges = spec.max_edges if max_edges is None else max_edges

    pool = instance.pool
    atom_ids = pool.ids()
    n_atoms = len(atom_ids)
    if n_atoms > MAX_POOL_BITS:
        raise ValueError(
            f"pool holds {n_atoms} atoms; the uint64 state representation addresses "
            f"at most {MAX_POOL_BITS} (G3)"
        )

    walker = _Walker(instance, cfg)
    masks: list[int] = [0]
    sizes: list[int] = [0]
    index: dict[int, int] = {0: 0}
    stop_flags: list[bool] = []
    dead_flags: list[bool] = []
    e_parent: list[int] = []
    e_action: list[int] = []
    e_child: list[int] = []

    cursor = 0
    while cursor < len(masks):
        m = masks[cursor]
        state = walker.goto(atom_ids[i] for i in range(n_atoms) if (m >> i) & 1)
        allowed = legal_adds(state)
        can_stop = stop_allowed(state)
        stop_flags.append(can_stop)
        legal = np.flatnonzero(allowed)
        dead_flags.append(bool(not can_stop and legal.size == 0))
        for j in legal.tolist():
            child = m | (1 << j)
            ci = index.get(child)
            if ci is None:
                ci = len(masks)
                index[child] = ci
                masks.append(child)
                sizes.append(sizes[cursor] + 1)
                if len(masks) > cap_states:
                    raise BandViolation("states", f">{cap_states}", f"<= {cap_states} (G1)")
            e_parent.append(cursor)
            e_action.append(j)
            e_child.append(ci)
        if len(e_parent) > cap_edges:
            raise BandViolation("edges", f">{cap_edges}", f"<= {cap_edges} (G1)")
        cursor += 1

    return StateGraph(
        atom_ids=atom_ids,
        max_atoms=cfg.max_atoms,
        mask=np.asarray(masks, dtype=np.uint64),
        size=np.asarray(sizes, dtype=np.int16),
        edge_parent=np.asarray(e_parent, dtype=np.int64),
        edge_action=np.asarray(e_action, dtype=np.int64),
        edge_child=np.asarray(e_child, dtype=np.int64),
        stop_flags=np.asarray(stop_flags, dtype=bool),
        dead_flags=np.asarray(dead_flags, dtype=bool),
    )


def valid_terminals(
    instance: "LatticeInstance",
    cfg: Config | None = None,
    *,
    graph: StateGraph | None = None,
) -> tuple[ProofSet, ...]:
    """Every formally valid terminal, in state-index order.

    ``graph`` is accepted so a caller that already enumerated does not pay for it
    twice — the enumeration is the expensive half of Phase 2.
    """
    cfg = cfg if cfg is not None else instance.cfg
    g = graph if graph is not None else reachable_states(instance, cfg)
    return tuple(g.proofset_of(int(t), instance.pool) for t in g.terminal_ix)


def state_fingerprints(graph: StateGraph) -> tuple[str, ...]:
    """``canon_set_hash`` per state — a **fingerprint**, never the identity (G3).

    Used for cross-machine comparison and for the collision audit, which expects
    zero.  A collision here does not corrupt the evaluator (identity is the
    exact bitmask); it makes the fingerprint unusable for comparison, which is a
    different and milder failure worth knowing about.
    """
    return tuple(canon_set_hash(graph.atoms_of(i)) for i in range(graph.n_states))


def environment_fingerprint(instance: "LatticeInstance", graph: StateGraph) -> str:
    """Digest of everything built and enumerated, **independent of β**.

    Excluding β is deliberate: ``config_hash`` moves when the Phase-3 sweep
    freezes β, so a fingerprint containing it would change *after* the suites
    were frozen and "frozen" would mean nothing.  The β-dependent layer is
    :func:`graft.synth.exact.target_fingerprint`.
    """
    payload = instance.identity_payload()
    payload["graph"] = graph.fingerprint()
    # Route through canonical_bytes so a NaN anywhere in a feature vector raises
    # here rather than being written into a digest as a bare token.
    return digest_of(_as_json(payload))


def _as_json(payload: dict[str, Any]) -> dict[str, Any]:
    canonical_bytes(payload)  # raises on NaN/Infinity before anything is hashed
    return payload
