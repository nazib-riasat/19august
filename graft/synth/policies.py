"""The ``ActionPolicy`` protocol (G6) and three reference implementations.

**Batch-first, and that is not a style choice** (decision 8).  At 1e5 states per
instance a per-state call would put 1e5 Python round-trips into a network on the
critical path of *every* Gate-2 checkpoint — the difference between an
evaluation measured in seconds and one measured in hours (G2).  The protocol
therefore takes an array of **state indices** rather than materialised sets, so
the hot path holds no Python objects at all.  An implementation may chunk
internally: "batched" means "not one call per state", not "exactly one call per
instance".

**The learners do not implement this protocol.  An adapter does.**  Architecture
fix F6 freezes the learner-facing interface as
``policy(state_repr, action_reprs) -> logits`` precisely so a learner never sees
where its features came from.  A Phase-3 learner implementing ``ActionPolicy``
directly would take a ``StateGraph`` and synthetic atom ids — the exact coupling
F6 exists to prevent, arriving through the back door of an evaluation
interface::

    StateGraph ──► SyntheticFeaturizer ──► policy(state_repr, action_reprs)
                        (adapter)                 (the learner, F6)
               └────────── implements ActionPolicy ──────────┘

**``graph`` is passed alongside ``state_ix`` for a reason.**  **[EVIDENCE]**
*When Do GFlowNets Learn the Right Distribution?* (ICLR 2025) proves that the
representation limits of GNNs delineate which distributions a GFlowNet can
approximate, and its remedy is **Look-Ahead GFlowNets**: feed children-state
embeddings into the forward policy.  The children of any state are in the edge
arrays, reachable through :meth:`StateGraph.children_of`, so an LA-style
featuriser stays implementable behind the frozen protocol.  An adapter that
featurised only the current state would silently foreclose the published fix for
a risk v1.2 §6.4 already names — which is why exit criterion 18 tests it.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from graft.synth.enumerate import StateGraph
from graft.synth.exact import Target

__all__ = [
    "ActionPolicy",
    "UniformPolicy",
    "ForcedContinuationPolicy",
    "FlowOraclePolicy",
    "uniform_backward",
]


@runtime_checkable
class ActionPolicy(Protocol):
    """What the exact evaluator consumes.

    ``runtime_checkable`` so Phase 3 can assert the layering it is required to
    keep: an adapter implements this, a **learner never does** (fix F6).  The
    check is structural — it sees the method, not its types — which is all a
    "does this object claim to be a policy" guard needs.
    """

    def action_log_probs(
        self, state_ix: np.ndarray, graph: StateGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(log P_F(ADD a_j | s_i) as [n_states, n_atoms], log P_F(STOP | s_i) as [n_states])``.

        ``state_ix`` indexes into ``graph``; the states themselves are ``uint64``
        bitmasks reachable as ``graph.mask[state_ix]``.  Passing indices rather
        than materialised sets keeps the hot path free of Python objects.

        ``-inf`` on illegal actions; per row the finite entries sum to 1 in
        probability.  **Never called on a dead-end state** — one has no legal
        ``ADD`` and a masked ``STOP``, so no distribution over its actions
        exists, and returning all ``-inf`` would put NaNs into the DP.
        """


NEG_INF = -math.inf


def _check_no_dead_ends(state_ix: np.ndarray, graph: StateGraph) -> None:
    if np.any(graph.dead_end[state_ix]):
        bad = int(np.flatnonzero(graph.dead_end[state_ix])[0])
        raise ValueError(
            f"state {int(state_ix[bad])} is a dead end: it has no legal ADD and a "
            "masked STOP, so no distribution over its actions exists. The evaluator "
            "routes dead-end mass straight to FAIL and must not query them."
        )


def _row_index(state_ix: np.ndarray, graph: StateGraph) -> np.ndarray:
    rows = np.full(graph.n_states, -1, dtype=np.int64)
    rows[state_ix] = np.arange(state_ix.shape[0], dtype=np.int64)
    return rows


def _scatter(
    graph: StateGraph, rows: np.ndarray, n_query: int, edge_logp: np.ndarray
) -> np.ndarray:
    out = np.full((n_query, graph.n_atoms), NEG_INF, dtype=np.float64)
    if graph.n_edges:
        r = rows[graph.edge_parent]
        keep = r >= 0
        out[r[keep], graph.edge_action[keep]] = edge_logp[keep]
    return out


class UniformPolicy:
    """Uniform over every legal action, ``STOP`` included when it is allowed.

    The reference against which the DP is checked by Monte Carlo (exit criteria
    2 and 4), and the policy the **visitation-weighted** zero-``Δd`` fraction is
    measured under (G5) — that measurement asks what a *sampled* transition looks
    like, so it must be taken under a policy that actually samples ``STOP``.
    """

    def action_log_probs(
        self, state_ix: np.ndarray, graph: StateGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        state_ix = np.asarray(state_ix, dtype=np.int64)
        _check_no_dead_ends(state_ix, graph)
        rows = _row_index(state_ix, graph)
        total = graph.out_degree + graph.stop_allowed.astype(np.int64)
        with np.errstate(divide="ignore"):
            log_total = -np.log(total.astype(np.float64))
        log_add = _scatter(graph, rows, state_ix.shape[0], log_total[graph.edge_parent])
        log_stop = np.where(graph.stop_allowed[state_ix], log_total[state_ix], NEG_INF)
        return log_add, log_stop


class ForcedContinuationPolicy:
    """Uniform over legal ``ADD``s; ``STOP`` only when no ``ADD`` remains.

    The ``p_stop = 0`` regime Phase 1 used to find that its own dead ends all sat
    at ``max_atoms``.  It exists because the **dead-end absorption audit** must
    not be measured under :class:`UniformPolicy`, which stops early and would
    report a clean profile by construction (decision 16).
    """

    def action_log_probs(
        self, state_ix: np.ndarray, graph: StateGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        state_ix = np.asarray(state_ix, dtype=np.int64)
        _check_no_dead_ends(state_ix, graph)
        rows = _row_index(state_ix, graph)
        deg = graph.out_degree
        with np.errstate(divide="ignore"):
            log_deg = -np.log(deg.astype(np.float64))
        log_add = _scatter(graph, rows, state_ix.shape[0], log_deg[graph.edge_parent])
        must_stop = (deg[state_ix] == 0) & graph.stop_allowed[state_ix]
        log_stop = np.where(must_stop, 0.0, NEG_INF)
        return log_add, log_stop


def uniform_backward(graph: StateGraph) -> np.ndarray:
    """``P_B(s | s')`` per edge — uniform over the **removable** atoms of ``s'``.

    Not "uniform over selected": under the closure rule an atom referenced by
    another selected atom cannot be removed, so uniform-over-selected would put
    mass on parents that do not exist (Phase-3 handoff item 3).  The enumerated
    in-edges of ``s'`` *are* its removable atoms — a parent exists in the graph
    exactly when removing that atom leaves a reachable closed state — so reading
    them off the edge list is both exact and free.
    """
    return 1.0 / graph.indegree[graph.edge_child].astype(np.float64)


class FlowOraclePolicy:
    """The flow decomposition against uniform ``P_B`` (decision 9).

    "Softmax over terminal reward" is not a specification: actions lead to
    **partial states**, not to terminals, and a terminal is reachable through
    many insertion orders, so scoring an action by the reward of the terminals
    below it double-counts every terminal reachable through more than one
    child.  The correct construction, computed in decreasing ``|S|``::

        P_B(s | s')   = uniform over the removable atoms of s'
        r_dead        = r_fail / |dead-end states|
        F(s)          = R(s)·1[s is a valid stop]
                      + r_dead·1[s is a dead end]
                      + Σ_{s' ∈ Ch(s)} F(s') · P_B(s | s')
        P_F(s → s')   = F(s') · P_B(s | s') / F(s)
        P_F(STOP | s) = R(s)·1[valid] / F(s)

    **The middle term is the part that is easy to get wrong.**  ``FAIL`` is a
    *single absorbing terminal* reached from *many* dead-end states, so the
    construction has to say how ``r_fail`` is divided among them: uniformly, each
    dead end receiving ``r_dead``.  Total flow into ``FAIL`` is then exactly
    ``r_fail``, ``Z = Σ_valid R(X) + r_fail`` as fix F3 requires, and the oracle
    reaches ``p_θ(FAIL) = p*(FAIL)`` — hence TV = 0 rather than a residual.

    Without that term a dead end is neither a valid stop nor a parent, so
    ``F(s) = 0``, and the oracle routes *no* mass to ``FAIL`` at all.  Worse, it
    would still pass a ``TV < 1e-9`` check, because the resulting error is only
    ``p*(FAIL) ≈ 2.5e-12``.  A silently broken construction that goes green is
    the failure mode this phase exists to prevent, which is why exit criterion 3
    asserts ``p_θ(FAIL)`` in **relative** terms as well.

    **A test instrument, not a baseline.**  It consumes the exact enumeration and
    is therefore unavailable at any scale where the learners matter; it must not
    appear in a results table as though it were a method.
    """

    __slots__ = ("graph", "flow", "reward", "r_dead", "p_backward")

    def __init__(self, target: Target, graph: StateGraph | None = None) -> None:
        g = graph if graph is not None else target.graph
        if g.n_dead_ends == 0:
            raise ValueError(
                "this lattice has no reachable dead end, so `r_fail` has nowhere to "
                "enter the flow and the oracle cannot reach p_θ(FAIL) = p*(FAIL). "
                "A lattice without one leaves STOP-masking untested (G4)."
            )
        self.graph = g
        self.r_dead = float(target.cfg.r_fail) / g.n_dead_ends

        reward = np.zeros(g.n_states, dtype=np.float64)
        reward[g.terminal_ix] = target.r
        self.reward = reward

        flow = reward.copy()
        flow[g.dead_ix] += self.r_dead
        # Computed once: it is read by the recurrence below and again by every
        # `action_log_probs` call, which is on the Gate-2 critical path.
        self.p_backward = uniform_backward(g)
        p_b = self.p_backward
        # Decreasing |S|: a state's children must be final before it is written.
        for depth in range(g.max_atoms - 1, -1, -1):
            lo, hi = g.edge_slice(depth)
            if hi <= lo:
                continue
            par = g.edge_parent[lo:hi]
            ch = g.edge_child[lo:hi]
            contrib = flow[ch] * p_b[lo:hi]
            flow += np.bincount(par, weights=contrib, minlength=g.n_states)
        self.flow = flow

    @property
    def partition(self) -> float:
        """``F(root)``, which must equal ``Σ_valid R(X) + r_fail``."""
        return float(self.flow[0])

    def action_log_probs(
        self, state_ix: np.ndarray, graph: StateGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        state_ix = np.asarray(state_ix, dtype=np.int64)
        if graph is not self.graph:
            # Identity, not a fingerprint comparison: this is the hot path, and
            # hashing every array on each call would cost more than the DP.
            raise ValueError(
                "FlowOraclePolicy was built against a different StateGraph object; "
                "its flows are indexed by that graph's state numbering"
            )
        g = self.graph
        _check_no_dead_ends(state_ix, g)
        rows = _row_index(state_ix, g)
        p_b = self.p_backward
        with np.errstate(divide="ignore", invalid="ignore"):
            log_edge = (
                np.log(self.flow[g.edge_child]) + np.log(p_b) - np.log(self.flow[g.edge_parent])
            )
            log_add = _scatter(g, rows, state_ix.shape[0], log_edge)
            log_stop = np.log(self.reward[state_ix]) - np.log(self.flow[state_ix])
        log_stop = np.where(g.stop_allowed[state_ix], log_stop, NEG_INF)
        # `log(0) - log(0)` is NaN rather than -inf.  It cannot arise here — every
        # reachable non-dead state has positive flow, since it reaches a valid
        # stop or a dead end — but a NaN reaching the DP would poison a whole
        # distribution silently, so it is mapped to "illegal" rather than trusted.
        return np.where(np.isnan(log_add), NEG_INF, log_add), log_stop
