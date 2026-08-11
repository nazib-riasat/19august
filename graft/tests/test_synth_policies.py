"""The frozen policy protocol and its three reference implementations.

Discharges the parts of exit criteria 3 and 18 that are about the *interface*
rather than the evaluator: batch-first calls, the dead-end contract, and the
backward policy being uniform over **removable** atoms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from graft.synth.enumerate import reachable_states
from graft.synth.exact import policy_distribution
from graft.synth.policies import (
    ActionPolicy,
    FlowOraclePolicy,
    ForcedContinuationPolicy,
    UniformPolicy,
    uniform_backward,
)


def _query(graph):
    return np.flatnonzero(~graph.dead_end).astype(np.int64)


@pytest.mark.parametrize("factory", [UniformPolicy, ForcedContinuationPolicy])
def test_rows_are_normalised_over_the_legal_actions(bench_graph, factory):
    """The protocol's contract: ``-inf`` on illegal actions, and per row the
    finite entries sum to 1 in probability."""
    query = _query(bench_graph)
    log_add, log_stop = factory().action_log_probs(query, bench_graph)
    assert log_add.shape == (query.shape[0], bench_graph.n_atoms)
    assert log_stop.shape == (query.shape[0],)

    mass = np.exp(log_add).sum(axis=1) + np.exp(log_stop)
    assert np.allclose(mass, 1.0, atol=1e-12)

    # Illegal actions really are illegal: the only finite entries are the ones
    # the state graph records as edges.
    legal = np.zeros_like(log_add, dtype=bool)
    rows = np.full(bench_graph.n_states, -1, dtype=np.int64)
    rows[query] = np.arange(query.shape[0])
    legal[rows[bench_graph.edge_parent], bench_graph.edge_action] = True
    assert np.all(np.isfinite(log_add) == legal)


@pytest.mark.parametrize("factory", [UniformPolicy, ForcedContinuationPolicy])
def test_a_dead_end_is_never_a_legal_query(bench_graph, factory):
    """G6.  A dead end has no legal ``ADD`` and a masked ``STOP``, so *no
    probability distribution over its actions exists* — asking for one is a
    category error, and returning all ``-inf`` would silently produce NaNs in the
    DP."""
    with pytest.raises(ValueError, match="dead end"):
        factory().action_log_probs(bench_graph.dead_ix[:1], bench_graph)


def test_the_uniform_policy_includes_stop_whenever_it_is_allowed(tiny_graph):
    query = _query(tiny_graph)
    log_add, log_stop = UniformPolicy().action_log_probs(query, tiny_graph)
    for row, state in enumerate(query.tolist()):
        actions, _ = tiny_graph.children_of(state)
        total = actions.shape[0] + int(tiny_graph.stop_allowed[state])
        if tiny_graph.stop_allowed[state]:
            assert log_stop[row] == pytest.approx(-math.log(total))
        else:
            assert log_stop[row] == -math.inf


def test_forced_continuation_stops_only_when_nothing_can_be_added(tiny_graph):
    """Decision 16.  This is the ``p_stop = 0`` regime the dead-end absorption
    audit is measured in; ``UniformPolicy`` stops early and would report a clean
    profile by construction."""
    query = _query(tiny_graph)
    log_add, log_stop = ForcedContinuationPolicy().action_log_probs(query, tiny_graph)
    degree = tiny_graph.out_degree[query]
    assert np.all(log_stop[degree > 0] == -math.inf)
    assert np.all(log_stop[degree == 0] == 0.0)


def test_forced_continuation_reaches_dead_ends_that_uniform_barely_touches(bench_graph):
    forced = policy_distribution(ForcedContinuationPolicy(), bench_graph)
    uniform = policy_distribution(UniformPolicy(), bench_graph)
    assert forced[-1] > uniform[-1]


def test_uniform_backward_is_over_removable_atoms_not_selected_ones(bench_graph):
    """Phase-3 handoff item 3.  Under the closure rule an atom referenced by
    another selected atom cannot be removed, so "uniform over selected" would put
    mass on parents that do not exist.  The enumerated in-edges of a state *are*
    its removable atoms."""
    p_b = uniform_backward(bench_graph)
    incoming = np.bincount(
        bench_graph.edge_child, weights=p_b, minlength=bench_graph.n_states
    )
    reachable = bench_graph.indegree > 0
    assert np.allclose(incoming[reachable], 1.0, atol=1e-12)

    # Every parent recorded is a state that differs by exactly one atom, and that
    # atom is not referenced by anything else selected.
    rng = np.random.default_rng(3)
    for edge in rng.choice(bench_graph.n_edges, size=40, replace=False).tolist():
        parent = int(bench_graph.edge_parent[edge])
        child = int(bench_graph.edge_child[edge])
        removed = bench_graph.atom_ids[int(bench_graph.edge_action[edge])]
        kept = set(bench_graph.atoms_of(parent))
        assert set(bench_graph.atoms_of(child)) == kept | {removed}
        assert removed not in kept


def test_the_oracle_is_bound_to_the_graph_it_was_built_from(bench, bench_graph, bench_target):
    """Its flows are indexed by that graph's state numbering, so handing it
    another graph would silently score the wrong states."""
    oracle = FlowOraclePolicy(bench_target, bench_graph)
    other = reachable_states(bench, bench.cfg)
    with pytest.raises(ValueError, match="different StateGraph"):
        oracle.action_log_probs(np.array([0], dtype=np.int64), other)


def test_the_oracle_is_a_valid_policy_before_it_is_a_correct_one(bench_graph, bench_target):
    """Every row normalised — the flow decomposition has to be a policy at all
    before "it samples in proportion to R" means anything."""
    oracle = FlowOraclePolicy(bench_target, bench_graph)
    query = _query(bench_graph)
    log_add, log_stop = oracle.action_log_probs(query, bench_graph)
    mass = np.exp(log_add).sum(axis=1) + np.exp(log_stop)
    assert np.allclose(mass, 1.0, atol=1e-12)
    assert not np.any(np.isnan(log_add))


def test_all_three_reference_policies_satisfy_the_protocol(bench_target, bench_graph):
    for policy in (
        UniformPolicy(),
        ForcedContinuationPolicy(),
        FlowOraclePolicy(bench_target, bench_graph),
    ):
        assert isinstance(policy, ActionPolicy)
        assert callable(policy.action_log_probs)


def test_the_evaluator_calls_the_policy_in_batches_not_per_state(bench_graph):
    """Decision 8.  At 1e5 states a per-state call would put 1e5 Python
    round-trips into a network on the critical path of every Gate-2 checkpoint."""
    calls: list[int] = []

    class Counting(UniformPolicy):
        def action_log_probs(self, state_ix, graph):
            calls.append(int(np.asarray(state_ix).shape[0]))
            return super().action_log_probs(state_ix, graph)

    policy_distribution(Counting(), bench_graph)
    assert len(calls) == 1
    assert calls[0] == int((~bench_graph.dead_end).sum())
    assert calls[0] > 100
