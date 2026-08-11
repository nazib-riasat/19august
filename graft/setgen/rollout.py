"""Trajectory sampling over the enumerated state graph.

**This is the correctness hinge of Phase 3** (P3.3). Every downstream number —
exact TV, LED's decomposition error, GAFlowNet's augmented loss — is computed
over trajectories produced here. If the sampler and Phase 2's exact DP disagree,
nothing above them means anything, and the disagreement will not show up in any
loss curve: training would proceed, the curves would look plausible, and the
Gate-2 table would be fiction. Exit criterion 3 is the test that catches it.

**One forward pass, then pure numpy.** The state graph is small and
policy-independent (Phase-2 G2), so the whole action distribution is materialised
once per sampling call and the walk is inverse-CDF sampling over it. That is not
only faster than querying per step — it is the *same* array the DP consumes, so
the two cannot drift.

**Sampling carries no gradient.** Rollouts choose *which* trajectories to learn
from; the losses recompute log-probabilities on those indices with grad attached.
Keeping the two apart is what lets the same trajectory batch be replayed by a
different objective without re-sampling, which Algorithm 1 of LED-GFN needs for
its buffer.

**A trajectory ends at a valid ``STOP`` or at a dead end**, and both are counted
against the budget (decision 3). Counting only successful rollouts would let a
learner that dead-ends often buy extra gradient steps for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graft.synth.enumerate import StateGraph

# Re-exported, never reimplemented (decision 18, P3.3 surface). Phase 2's version
# reads the enumerated in-edges of a state, which *are* its removable atoms — an
# atom referenced by another selected atom has no parent to be removed to. A
# second implementation here would be a second chance to get "uniform over
# selected" wrong, which is the error architecture §Phase 3.1 was amended to fix.
from graft.synth.policies import uniform_backward

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.features import SyntheticFeaturizer

__all__ = [
    "Trajectories",
    "action_table",
    "empirical_distribution",
    "sample_trajectories",
    "uniform_backward",
]


class Trajectories:
    """A padded batch of rollouts.

    Padded rather than ragged because ``max_atoms`` is 8: the waste is bounded
    and every consumer gets plain rectangular arrays. ``-1`` marks padding.

    ``terminal`` is the state index a trajectory stopped at, or ``-1`` when it
    dead-ended — the same ``FAIL``-last convention Phase 2's ``Target`` uses, so
    an empirical distribution built from these lines up with ``p_star`` without
    a translation step.
    """

    __slots__ = ("states", "actions", "lengths", "terminal", "is_fail", "n_atoms")

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        lengths: np.ndarray,
        terminal: np.ndarray,
        is_fail: np.ndarray,
        n_atoms: int,
    ) -> None:
        self.states = states
        self.actions = actions
        self.lengths = lengths
        self.terminal = terminal
        self.is_fail = is_fail
        self.n_atoms = int(n_atoms)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_transitions(self) -> int:
        return int(self.lengths.sum())

    def transitions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten to ``(row, parent_state, action, child_state)``.

        ``row`` says which trajectory each transition came from, which is what
        LED's per-trajectory sum (its Eq. 3) needs in order to group potentials
        back into the trajectory they decompose.
        """
        rows, parents, acts, children = [], [], [], []
        for r in range(len(self)):
            k = int(self.lengths[r])
            if k:
                rows.append(np.full(k, r, dtype=np.int64))
                parents.append(self.states[r, :k])
                acts.append(self.actions[r, :k])
                children.append(self.states[r, 1 : k + 1])
        if not rows:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, empty
        return (
            np.concatenate(rows),
            np.concatenate(parents),
            np.concatenate(acts),
            np.concatenate(children),
        )


def action_table(graph: StateGraph) -> np.ndarray:
    """``[n_states, n_atoms]`` mapping an action to the child it produces, ``-1``
    where the action is illegal.

    Dense because ``n_states × n_atoms`` is small here and a dense lookup keeps
    the walk vectorised; the CSR form Phase 2 stores is better for its layered
    passes and worse for random access during sampling.
    """
    table = np.full((graph.n_states, graph.n_atoms), -1, dtype=np.int64)
    if graph.n_edges:
        table[graph.edge_parent, graph.edge_action] = graph.edge_child
    return table


def sample_trajectories(
    featurizer: "SyntheticFeaturizer",
    graph: StateGraph,
    n: int,
    rng: np.random.Generator,
    epsilon: float = 0.0,
) -> Trajectories:
    """Walk the MDP ``n`` times from the empty root.

    At ``epsilon = 0`` this samples from exactly the distribution
    ``policy_distribution`` integrates, which is what exit criterion 3 checks.

    ``epsilon`` mixes in a uniform-over-legal-actions component — the ε-uniform
    exploration the architecture specifies, frozen and identical across arms
    (G8, decision 10). **The loss still uses the policy's own log-probabilities**,
    not the behaviour policy's: trajectory-balance-family objectives are
    off-policy-capable, so exploring with a mixture and scoring with ``P_F`` is
    correct rather than a bias to be corrected.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    n_atoms, n_states = graph.n_atoms, graph.n_states
    max_len = graph.max_atoms

    query = np.flatnonzero(~graph.dead_end).astype(np.int64)
    log_add, log_stop = featurizer.action_log_probs(query, graph)

    probs = np.zeros((n_states, n_atoms + 1), dtype=np.float64)
    probs[query, :n_atoms] = np.exp(log_add)
    probs[query, n_atoms] = np.exp(log_stop)
    # Renormalise defensively: log_softmax already sums to 1, but a policy that
    # is *nearly* degenerate can leave the last bits short, and inverse-CDF
    # sampling would then silently bias toward the final action.
    total = probs.sum(axis=1, keepdims=True)
    np.divide(probs, total, out=probs, where=total > 0)

    if epsilon > 0.0:
        legal = np.zeros_like(probs, dtype=bool)
        if graph.n_edges:
            legal[graph.edge_parent, graph.edge_action] = True
        legal[:, n_atoms] = graph.stop_allowed
        counts = legal.sum(axis=1, keepdims=True)
        uniform = np.divide(legal, counts, out=np.zeros_like(probs), where=counts > 0)
        probs = (1.0 - epsilon) * probs + epsilon * uniform

    cumulative = np.cumsum(probs, axis=1)
    children = action_table(graph)

    states = np.full((n, max_len + 1), -1, dtype=np.int64)
    actions = np.full((n, max_len), -1, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    terminal = np.full(n, -1, dtype=np.int64)
    is_fail = np.zeros(n, dtype=bool)

    current = np.zeros(n, dtype=np.int64)
    states[:, 0] = 0
    live = np.arange(n, dtype=np.int64)

    for step in range(max_len + 1):
        if live.size == 0:
            break
        here = current[live]

        dead = graph.dead_end[here]
        if dead.any():
            finished = live[dead]
            terminal[finished] = -1
            is_fail[finished] = True
            live = live[~dead]
            if live.size == 0:
                break
            here = current[live]

        u = rng.random(live.size)
        picks = (cumulative[here] < u[:, None]).sum(axis=1)
        # Guard the closed-interval edge case: u == 1.0 would index past the end.
        np.minimum(picks, n_atoms, out=picks)

        stopped = picks == n_atoms
        if stopped.any():
            done = live[stopped]
            terminal[done] = current[done]

        moving = ~stopped
        if moving.any():
            movers = live[moving]
            chosen = picks[moving]
            nxt = children[current[movers], chosen]
            if (nxt < 0).any():
                raise RuntimeError(
                    "sampled an action with no child edge; the policy assigned "
                    "positive probability to an illegal action, which masking "
                    "before the softmax is supposed to make impossible"
                )
            actions[movers, step] = chosen
            states[movers, step + 1] = nxt
            current[movers] = nxt
            lengths[movers] = step + 1
        live = live[moving]

    if live.size:
        raise RuntimeError(
            f"{live.size} trajectories did not terminate within max_atoms steps; "
            "every trajectory adds one atom per step and is capped, so this means "
            "the state graph is not layered as the DP assumes"
        )
    return Trajectories(states, actions, lengths, terminal, is_fail, n_atoms)


def empirical_distribution(graph: StateGraph, trajectories: Trajectories) -> np.ndarray:
    """Sampled terminals in ``Target.p_star``'s layout, ``FAIL`` last.

    The layout match is the point: it is what lets exit criterion 3 compare this
    against ``policy_distribution`` without a translation step that could itself
    be wrong.
    """
    position = {int(t): i for i, t in enumerate(graph.terminal_ix.tolist())}
    counts = np.zeros(graph.n_terminals + 1, dtype=np.float64)
    for value in trajectories.terminal.tolist():
        counts[-1 if value < 0 else position[value]] += 1
    total = counts.sum()
    return counts / total if total else counts
