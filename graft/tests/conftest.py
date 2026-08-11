"""Shared Phase-2 fixtures and the Monte-Carlo sampler the DP is checked against.

Session-scoped because generating the benchmark suite enumerates twenty lattices
and every module below wants the same one.  Nothing here mutates an instance, so
sharing is safe; anything that would has to build its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from graft.synth.enumerate import StateGraph, reachable_states
from graft.synth.exact import target_distribution
from graft.synth.lattice import benchmark_suite, probe_suite, tiny_instance, tuning_suite
from graft.synth.policies import ActionPolicy

#: Full-scale Monte-Carlo settings (decision 18).  "Within 5 sigma" without an
#: ``N`` is not a criterion, so both are frozen here and used by every MC test.
MC_SAMPLES = 200_000
MC_SEED = 20260810


@pytest.fixture(scope="session")
def tiny():
    return tiny_instance()


@pytest.fixture(scope="session")
def tiny_graph(tiny):
    return reachable_states(tiny, tiny.cfg)


@pytest.fixture(scope="session")
def tiny_target(tiny, tiny_graph):
    return target_distribution(tiny, tiny.cfg, graph=tiny_graph)


@pytest.fixture(scope="session")
def main_suite():
    return benchmark_suite()


@pytest.fixture(scope="session")
def probe():
    return probe_suite()


@pytest.fixture(scope="session")
def tuning():
    return tuning_suite()


@pytest.fixture(scope="session")
def bench(main_suite):
    """One benchmark instance — the largest, so budgets are measured at the top
    of the band rather than at a flattering point in it."""
    return max(main_suite, key=lambda i: i.meta["bands"]["counts"]["states"])


@pytest.fixture(scope="session")
def bench_graph(bench):
    return reachable_states(bench, bench.cfg)


@pytest.fixture(scope="session")
def bench_target(bench, bench_graph):
    return target_distribution(bench, bench.cfg, graph=bench_graph)


def sample_terminals(
    graph: StateGraph, policy: ActionPolicy, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Walk the actual MDP ``n_samples`` times; return the terminal state index,
    or ``-1`` for ``FAIL``.

    This is the *independent* half of exit criteria 2 and 4: it uses only
    ``children_of`` and the policy's own probabilities, never the DP.  Walkers are
    grouped by state at each depth so the work is one vectorised draw per
    (depth, state) rather than one per sample.
    """
    query = np.flatnonzero(~graph.dead_end).astype(np.int64)
    row_of = np.full(graph.n_states, -1, dtype=np.int64)
    row_of[query] = np.arange(query.shape[0], dtype=np.int64)
    log_add, log_stop = policy.action_log_probs(query, graph)
    p_add, p_stop = np.exp(log_add), np.exp(log_stop)

    cur = np.zeros(n_samples, dtype=np.int64)
    out = np.full(n_samples, -1, dtype=np.int64)
    done = np.zeros(n_samples, dtype=bool)

    for _ in range(graph.max_atoms + 2):
        live = np.flatnonzero(~done)
        if live.size == 0:
            break
        states = cur[live]
        for s in np.unique(states).tolist():
            idx = live[states == s]
            if graph.dead_end[s]:
                out[idx] = -1
                done[idx] = True
                continue
            actions, children = graph.children_of(s)
            r = int(row_of[s])
            probs = np.append(p_add[r, actions], p_stop[r])
            probs = probs / probs.sum()
            picks = rng.choice(probs.shape[0], size=idx.shape[0], p=probs)
            stopped = picks == probs.shape[0] - 1
            out[idx[stopped]] = s
            done[idx[stopped]] = True
            cur[idx[~stopped]] = children[picks[~stopped]]
    assert done.all(), "a walker failed to terminate; every trajectory is capped at max_atoms"
    return out


def empirical_distribution(
    graph: StateGraph, terminals: np.ndarray
) -> np.ndarray:
    """Sampled terminals -> the same layout as ``Target.p_star`` (FAIL last)."""
    position = {int(t): i for i, t in enumerate(graph.terminal_ix.tolist())}
    counts = np.zeros(graph.n_terminals + 1, dtype=np.float64)
    for value in terminals.tolist():
        counts[-1 if value < 0 else position[value]] += 1
    return counts / counts.sum()
