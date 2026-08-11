"""The exact evaluator.

Discharges Phase-2 exit criteria 1 (the hand-computed table), 2 and 4 (the DP
against Monte Carlo, split by instrument per G8), 3 (the flow oracle, including
its ``FAIL`` branch), 5 (mass partitions), 6 (intra-layer permutation
invariance), 7 (the KL guard and FCS against exhaustive enumeration), 8
(``at_beta`` and ``validate_bands``) and 11 (budgets).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from graft.config import ConfigError
from graft.core.checker import H
from graft.core.reward import reward
from graft.synth.enumerate import BandViolation, StateGraph, reachable_states
from graft.synth.exact import (
    FCS_SEED,
    FCS_SUBSETS,
    fcs,
    fcs_exact,
    forward_mass,
    js,
    kl,
    partition_residual,
    policy_distribution,
    target_distribution,
    tv,
)
from graft.synth.policies import FlowOraclePolicy, ForcedContinuationPolicy, UniformPolicy
from graft.tests.conftest import MC_SAMPLES, MC_SEED, empirical_distribution, sample_terminals

# --------------------------------------------------------------------------
# criterion 1 — the hand-computed table
# --------------------------------------------------------------------------

#: ``tiny_instance()``'s target, written out.  Six atoms, ``max_atoms = 3``,
#: seventeen closed states, fifteen valid terminals and one reachable dead end
#: (``{nX, nV, bBAD}``: the binding rests on a claim the graph says was valid
#: only during ``[200, 300)``, so sub-check 3 fires, and at ``|X| = max_atoms``
#: there is no legal ``ADD`` either).
#:
#: Literals, because a table recomputed by the code it checks is not a check.
#: The independent recomputation below goes through batch ``H`` and
#: ``core.reward.reward`` instead — a different code path to the same numbers.
TINY_TABLE: tuple[tuple[tuple[str, ...], float, float], ...] = (
    (("bOK", "nC", "nV"), 1.309334992004, 0.445440307342),
    (("nC",), 0.550000000000, 0.021364342194),
    (("nC", "nE"), 0.536987482503, 0.020280771282),
    (("nC", "nE", "nV"), 0.989052667357, 0.123709236986),
    (("nC", "nE", "nX"), 0.519223482155, 0.018889707713),
    (("nC", "nV"), 0.971130962018, 0.115151324402),
    (("nC", "nV", "nX"), 0.917816105173, 0.093036110279),
    (("nC", "nX"), 0.498187628114, 0.017365297047),
    (("nE",), 0.154166666667, 0.004385878493),
    (("nE", "nV"), 0.663059985752, 0.033581093753),
    (("nE", "nV", "nX"), 0.648217077115, 0.031645361701),
    (("nE", "nX"), 0.198531644126, 0.005237529178),
    (("nV",), 0.675000000000, 0.035223845410),
    (("nV", "nX"), 0.626889303449, 0.029057612392),
    (("nX",), 0.216666666667, 0.005631579460),
)

TINY_P_FAIL = 2.3672365914209438e-09
TINY_Z = 422.433483676317


def _named(instance, graph, state: int) -> tuple[str, ...]:
    names = {v: k for k, v in instance.meta["atoms"].items()}
    return tuple(sorted(names[a] for a in graph.atoms_of(state)))


def test_the_tiny_target_matches_the_written_table(tiny, tiny_graph, tiny_target):
    """Exit criterion 1."""
    assert tiny_graph.n_terminals == len(TINY_TABLE)
    got = {
        _named(tiny, tiny_graph, int(s)): (float(u), float(p))
        for s, u, p in zip(
            tiny_graph.terminal_ix.tolist(), tiny_target.u, tiny_target.p_star[:-1]
        )
    }
    assert set(got) == {atoms for atoms, _, _ in TINY_TABLE}
    for atoms, u, p in TINY_TABLE:
        assert got[atoms][0] == pytest.approx(u, abs=1e-12)
        assert got[atoms][1] == pytest.approx(p, abs=1e-12)
    assert tiny_target.target_p_fail == pytest.approx(TINY_P_FAIL, rel=1e-12)
    assert tiny_target.z == pytest.approx(TINY_Z, rel=1e-12)


def test_the_target_sums_to_one_including_fail(tiny_target):
    """Criterion 1.  ``FAIL`` is a **member of the target's support** (fix F3),
    so the sum is over ``{valid terminals} ∪ {FAIL}`` and nothing is left out."""
    assert float(tiny_target.p_star.sum()) == pytest.approx(1.0, abs=1e-15)
    assert tiny_target.p_star[-1] == tiny_target.target_p_fail


def test_the_target_is_reproduced_by_an_independent_path(tiny, tiny_graph, tiny_target):
    """Batch ``H`` plus ``core.reward.reward`` over brute-forced subsets, rather
    than the enumerator plus the cached ``U``."""
    from itertools import combinations

    ids = tiny.pool.ids()
    rewards: dict[tuple[str, ...], float] = {}
    for size in range(len(ids) + 1):
        for combo in combinations(ids, size):
            if not H(combo, tiny.obligations, tiny.graph, tiny.pool, tiny.cfg, ledger=None):
                continue
            rewards[tuple(sorted(combo))] = reward(
                combo, tiny.obligations, tiny.graph, tiny.pool, tiny.gold, tiny.cfg
            )
    z = sum(rewards.values()) + tiny.cfg.r_fail
    assert z == pytest.approx(tiny_target.z, rel=1e-12)
    for state, p in zip(tiny_graph.terminal_ix.tolist(), tiny_target.p_star[:-1]):
        atoms = tuple(sorted(tiny_graph.atoms_of(state)))
        assert rewards[atoms] / z == pytest.approx(float(p), rel=1e-12)


def test_the_dead_end_is_the_one_the_fixture_promises(tiny, tiny_graph):
    """Criterion 3 needs ``tiny_instance()`` to contain a reachable dead end, or
    the oracle's ``FAIL`` branch is never executed by any test."""
    assert tiny_graph.n_dead_ends == 1
    assert _named(tiny, tiny_graph, int(tiny_graph.dead_ix[0])) == ("bBAD", "nV", "nX")
    assert tiny_graph.size[tiny_graph.dead_ix[0]] == tiny.cfg.max_atoms


# --------------------------------------------------------------------------
# criteria 2 and 4 — the DP against Monte Carlo, split by instrument (G8)
# --------------------------------------------------------------------------


def test_the_dp_matches_monte_carlo_on_the_small_instance(tiny_graph, tiny_target):
    """Exit criterion 2, decision 11.

    For an ``N``-sample empirical over ``k`` outcomes ``E[TV] ≈ sqrt(k/(2πN))``,
    so at ``k = 16`` and ``N = 200,000`` the sampling floor is about 0.004 and
    ``TV < 0.02`` is a real assertion.  On a 5,000-terminal lattice the same
    assertion would need ~8 million rollouts and would be measuring its own
    noise, which is why criterion 4 asserts something else.
    """
    policy = UniformPolicy()
    exact = policy_distribution(policy, tiny_graph)
    rng = np.random.default_rng(MC_SEED)
    empirical = empirical_distribution(
        tiny_graph, sample_terminals(tiny_graph, policy, MC_SAMPLES, rng)
    )
    assert tv(exact, empirical) < 0.02

    # Plus a per-terminal z-test at 5 sigma.
    se = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / MC_SAMPLES)
    z = np.abs(empirical - exact) / se
    worst = int(np.argmax(z))
    assert z[worst] < 5.0, f"outcome {worst}: exact {exact[worst]}, empirical {empirical[worst]}"


def test_the_dp_matches_monte_carlo_on_the_top_terminals_at_full_scale(bench_graph):
    """Exit criterion 4, decision 18.

    **A full-support TV assertion is not made.**  At 5,000 terminals the sampling
    floor is 0.063, so such a test would measure its own noise.  The top-20
    highest-mass terminals are where per-terminal ``p`` is large enough for a
    z-test to bite.
    """
    policy = UniformPolicy()
    exact = policy_distribution(policy, bench_graph)
    rng = np.random.default_rng(MC_SEED)
    empirical = empirical_distribution(
        bench_graph, sample_terminals(bench_graph, policy, MC_SAMPLES, rng)
    )
    top = np.argsort(-exact)[:20]
    se = np.sqrt(np.maximum(exact[top] * (1 - exact[top]), 1e-12) / MC_SAMPLES)
    z = np.abs(empirical[top] - exact[top]) / se
    assert float(z.max()) < 5.0


# --------------------------------------------------------------------------
# criterion 3 — the flow oracle
# --------------------------------------------------------------------------


def test_the_flow_oracle_recovers_the_target_exactly(tiny_graph, tiny_target):
    """Exit criterion 3.  ``TV < 1e-9`` — literally zero to floating point, not
    ``p*(FAIL)``.  An evaluator that cannot recover a distribution it was handed
    is broken in a way a uniform policy would never reveal."""
    oracle = FlowOraclePolicy(tiny_target, tiny_graph)
    p = policy_distribution(oracle, tiny_graph)
    assert tv(p, tiny_target.p_star) < 1e-9
    assert js(p, tiny_target.p_star) < 1e-9
    assert oracle.partition == pytest.approx(tiny_target.z, rel=1e-12)


def test_the_flow_oracle_routes_the_right_mass_to_fail(tiny_graph, tiny_target):
    """Exit criterion 3, second half — and the reason it is asserted separately.

    ``p*(FAIL) ≈ 2.4e-09`` here and ``≈ 2.5e-12`` on a full lattice, so an oracle
    that never routes mass to ``FAIL`` would sail through the aggregate TV check
    above while being wrong.  The assertion is on the **directly accumulated**
    sum because the complement cannot resolve that value: its relative error is
    ~2e-4 at float64, even with ``fsum``.
    """
    oracle = FlowOraclePolicy(tiny_target, tiny_graph)
    p = policy_distribution(oracle, tiny_graph)
    direct = float(p[-1])
    expected = tiny_target.cfg.r_fail / tiny_target.z
    assert abs(direct - expected) / expected < 1e-9


def test_the_complement_cannot_resolve_p_fail_on_a_full_lattice(bench_graph, bench_target):
    """Why decision 15 makes the **direct** dead-end sum authoritative.

    The complement subtracts two quantities that agree to ~12 digits, so its
    absolute error is bounded below by float64 representation near 1 (~2e-16)
    regardless of how carefully the sum is done — and ``p*(FAIL)`` here is
    ``O(1e-12)``, so that floor is a *relative* error of order 1e-4.  The direct
    sum is not subject to it, and the criterion-3 assertion is on the direct sum
    for exactly this reason.

    Measured on the full lattice rather than on ``tiny_instance()``: tiny's
    ``p*(FAIL)`` is ``O(1e-9)``, three orders larger, so the complement still
    looks accurate there and the test would prove nothing.
    """
    oracle = FlowOraclePolicy(bench_target, bench_graph)
    p = policy_distribution(oracle, bench_graph)
    expected = bench_target.cfg.r_fail / bench_target.z
    assert expected < 1e-11

    direct_error = abs(float(p[-1]) - expected) / expected
    complement_error = abs((1.0 - float(p[:-1].sum())) - expected) / expected
    assert direct_error < 1e-9
    assert complement_error > 1e-6
    assert complement_error > 1_000 * max(direct_error, 1e-16)


def test_a_lattice_without_a_dead_end_refuses_to_build_an_oracle(tiny, tiny_graph, tiny_target):
    """Without ``r_dead·1[dead end]`` the recurrence gives a dead end ``F(s) = 0``
    and the oracle routes no mass to ``FAIL`` at all — while still passing a
    ``TV < 1e-9`` check.  A silently broken construction that goes green is the
    failure mode this whole phase exists to prevent."""
    doctored = StateGraph(
        atom_ids=tiny_graph.atom_ids,
        max_atoms=tiny_graph.max_atoms,
        mask=tiny_graph.mask,
        size=tiny_graph.size,
        edge_parent=tiny_graph.edge_parent,
        edge_action=tiny_graph.edge_action,
        edge_child=tiny_graph.edge_child,
        stop_flags=tiny_graph.stop_allowed,
        dead_flags=np.zeros_like(tiny_graph.dead_end),
    )
    with pytest.raises(ValueError, match="no reachable dead end"):
        FlowOraclePolicy(tiny_target, doctored)


def test_the_oracle_recovers_the_target_on_a_full_lattice(bench_graph, bench_target):
    oracle = FlowOraclePolicy(bench_target, bench_graph)
    p = policy_distribution(oracle, bench_graph)
    assert tv(p, bench_target.p_star) < 1e-9
    expected = bench_target.cfg.r_fail / bench_target.z
    assert abs(float(p[-1]) - expected) / expected < 1e-9


# --------------------------------------------------------------------------
# criterion 5 — mass partitions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("policy_factory", [UniformPolicy, ForcedContinuationPolicy])
def test_mass_partitions_on_every_benchmark_instance(main_suite, policy_factory):
    """Exit criterion 5.

    **The tolerance is set by the worst-case bound, not by the observed
    residual.**  Scatter-add over up to 2e6 edges has a worst case of
    ``n·eps ≈ 4.4e-10``; a measured one-pass residual is ~2e-16 because the
    errors random-walk rather than conspire, but a criterion calibrated on the
    lucky case is a flaky test waiting to happen.
    """
    for instance in main_suite:
        graph = reachable_states(instance, instance.cfg)
        p = policy_distribution(policy_factory(), graph)
        assert partition_residual(p) <= 1e-9
        assert np.all(p >= 0.0)


# --------------------------------------------------------------------------
# criterion 6 — intra-layer permutation invariance
# --------------------------------------------------------------------------


def _permute_within_layers(graph: StateGraph, rng: np.random.Generator) -> tuple[StateGraph, np.ndarray]:
    """Relabel states inside each ``|S|`` layer.

    The DP requires **ascending layer order**; only intra-layer order is free, so
    asserting invariance to arbitrary visitation order would be asserting
    something false.
    """
    new_of_old = np.empty(graph.n_states, dtype=np.int64)
    for s in range(graph.max_atoms + 1):
        lo, hi = graph.state_slice(s)
        if hi <= lo:
            continue
        new_of_old[lo:hi] = lo + rng.permutation(hi - lo)
    old_of_new = np.argsort(new_of_old)

    parent = new_of_old[graph.edge_parent]
    child = new_of_old[graph.edge_child]
    order = np.argsort(parent, kind="stable")
    permuted = StateGraph(
        atom_ids=graph.atom_ids,
        max_atoms=graph.max_atoms,
        mask=graph.mask[old_of_new],
        size=graph.size[old_of_new],
        edge_parent=parent[order],
        edge_action=graph.edge_action[order],
        edge_child=child[order],
        stop_flags=graph.stop_allowed[old_of_new],
        dead_flags=graph.dead_end[old_of_new],
    )
    return permuted, new_of_old


def test_the_dp_is_invariant_to_intra_layer_state_order(bench_graph):
    """Exit criterion 6.  ``<= 1e-12`` absolute per terminal, **not** bit-for-bit:
    float addition is not associative, so reordering a scatter-add changes the
    last bits and an exact-equality assertion would fail on a correct
    implementation."""
    rng = np.random.default_rng(5)
    permuted, _ = _permute_within_layers(bench_graph, rng)
    base = policy_distribution(UniformPolicy(), bench_graph)
    other = policy_distribution(UniformPolicy(), permuted)

    by_mask = {int(m): float(p) for m, p in zip(bench_graph.mask[bench_graph.terminal_ix], base[:-1])}
    for m, p in zip(permuted.mask[permuted.terminal_ix], other[:-1]):
        assert by_mask[int(m)] == pytest.approx(float(p), abs=1e-12)
    assert float(base[-1]) == pytest.approx(float(other[-1]), abs=1e-12)


# --------------------------------------------------------------------------
# criterion 7 — divergences and FCS
# --------------------------------------------------------------------------


def test_kl_is_reported_only_when_finite(tiny_graph, tiny_target):
    """G8: ``KL(p* ‖ p_θ) = ∞`` whenever ``p_θ(X) = 0`` for a valid ``X``, which a
    deterministic policy produces.  The guard is exercised, not assumed."""

    class Deterministic:
        """Always takes the lowest-indexed legal action; stops only when forced."""

        def action_log_probs(self, state_ix, graph):
            forced = ForcedContinuationPolicy().action_log_probs(state_ix, graph)
            log_add, log_stop = forced
            out = np.full_like(log_add, -math.inf)
            legal = np.isfinite(log_add)
            first = np.argmax(legal, axis=1)
            rows = np.flatnonzero(legal.any(axis=1))
            out[rows, first[rows]] = 0.0
            return out, log_stop

    p = policy_distribution(Deterministic(), tiny_graph)
    assert math.isinf(kl(tiny_target.p_star, p))
    assert 0.0 <= js(tiny_target.p_star, p) <= math.log(2) + 1e-12
    assert 0.0 <= tv(tiny_target.p_star, p) <= 1.0

    oracle = FlowOraclePolicy(tiny_target, tiny_graph)
    assert math.isfinite(kl(tiny_target.p_star, policy_distribution(oracle, tiny_graph)))


#: The FCS fixture, built so a broken sampler cannot pass: ``p_theta`` and ``R``
#: are deliberately **non-uniform and anti-correlated**, so ``FCS > 0`` by a wide
#: margin.  A fixture where ``p_theta ∝ R`` gives ``FCS = 0`` for every sampler
#: and would pass while broken.
FCS_P_THETA = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
FCS_R = np.array([0.05, 0.10, 0.15, 0.20, 0.22, 0.28])
#: Exact FCS at ``m = 3``, by enumerating all C(6,3) = 20 subsets with their
#: ``P_S`` weights.  A literal, so the sampler is checked against something that
#: did not come from the sampler — and independently re-derived in exact rational
#: arithmetic below, because a literal copied out of the code it checks is not a
#: check either.
FCS_EXACT_M3 = 0.4728548372381662


class _FakeTarget:
    """Just the ``p_star`` surface ``fcs`` reads, for the enumerated fixture."""

    def __init__(self, p_star: np.ndarray) -> None:
        self.p_star = p_star


def test_the_fcs_literal_is_right():
    """The literal, re-derived in exact rational arithmetic.

    ``Fraction`` carries no rounding at all, so this is a genuinely independent
    path to ``FCS_EXACT_M3``: if :func:`fcs_exact` and this disagree beyond one
    ulp, one of them is wrong and neither can be used to check the sampler.
    """
    from fractions import Fraction
    from itertools import combinations
    from math import comb

    p = [Fraction(int(round(x * 100)), 100) for x in FCS_P_THETA.tolist()]
    r = [Fraction(int(round(x * 100)), 100) for x in FCS_R.tolist()]
    total = sum(r)
    star = [x / total for x in r]
    assert sum(p) == 1 and sum(star) == 1

    n, m = len(p), 3
    norm = Fraction(comb(n - 1, m - 1))
    acc = Fraction(0)
    weight_check = Fraction(0)
    for subset in combinations(range(n), m):
        weight = sum(p[i] for i in subset) / norm
        weight_check += weight
        a = sum(p[i] for i in subset)
        b = sum(star[i] for i in subset)
        acc += weight * sum(abs(p[i] / a - star[i] / b) for i in subset) / 2
    assert weight_check == 1, "P_S must normalise to 1 over all m-subsets"
    assert float(acc) == pytest.approx(FCS_EXACT_M3, abs=1e-15)


def test_fcs_matches_exhaustive_enumeration_of_every_subset():
    """Exit criterion 7.

    **Testing at ``m = #terminals`` verifies nothing about the sampler**: there
    is exactly one subset of that size, ``P_S`` is degenerate, and FCS reduces to
    TV for *any* sampler — including one that ignores ``P_S`` or inverts the
    renormalisation.  So the check is at ``m = 3`` over 20 enumerated subsets.
    """
    p_star = FCS_R / FCS_R.sum()
    assert fcs_exact(FCS_P_THETA, p_star, 3) == pytest.approx(FCS_EXACT_M3, abs=1e-12)
    assert FCS_EXACT_M3 > 0.1, "an anti-correlated fixture must give FCS well above 0"

    estimate, se = fcs(
        FCS_P_THETA,
        _FakeTarget(p_star),
        m=3,
        n_subsets=FCS_SUBSETS,
        rng=np.random.default_rng(FCS_SEED),
    )
    assert se > 0.0
    assert abs(estimate - FCS_EXACT_M3) <= 4 * se, (
        f"estimate {estimate} vs exact {FCS_EXACT_M3}, se {se} — the pass condition "
        "is 'within 4 standard errors of its own sampling distribution', not "
        "'converges to it', which is not a test"
    )


def test_fcs_reduces_to_tv_when_m_is_every_outcome():
    """Corollary 1 makes ``m`` an interpolation: at ``m = #terminals`` FCS *is*
    TV.  Checked as a property of the definition — never as a test of the
    sampler, which is what an earlier draft proposed."""
    p_star = FCS_R / FCS_R.sum()
    assert fcs_exact(FCS_P_THETA, p_star, FCS_P_THETA.shape[0]) == pytest.approx(
        tv(FCS_P_THETA, p_star), abs=1e-12
    )


def test_fcs_at_m_eight_is_a_different_statistic_from_tv(bench_graph, bench_target):
    """Decision 20.  At ``m = 8`` on a several-hundred-terminal lattice FCS is
    genuinely not TV, and this is the only environment where the gap can be
    measured rather than assumed."""
    p = policy_distribution(UniformPolicy(), bench_graph)
    value, se = fcs(p, bench_target)
    assert 0.0 <= value <= 1.0
    assert se > 0.0

    # The gap is real but not large, so the frozen 2,000-draw estimator cannot
    # resolve it at 4 SE.  Establishing "different statistic" therefore uses a
    # longer run — the *reported* number stays the frozen one.
    sharp, sharp_se = fcs(p, bench_target, n_subsets=20_000)
    assert sharp_se < se
    assert abs(sharp - tv(p, bench_target.p_star)) > 4 * sharp_se


def test_fcs_includes_fail(tiny_graph, tiny_target):
    """``FAIL`` is a terminal of this MDP and a member of ``p*``'s support, so
    excluding it would measure a different distribution than TV does."""
    p = policy_distribution(UniformPolicy(), tiny_graph)
    assert p.shape[0] == tiny_target.p_star.shape[0] == tiny_graph.n_terminals + 1
    with pytest.raises(ValueError, match="entries"):
        fcs(p[:-1], tiny_target)


def test_the_oracle_scores_zero_fcs(tiny_graph, tiny_target):
    oracle = FlowOraclePolicy(tiny_target, tiny_graph)
    value, _ = fcs(policy_distribution(oracle, tiny_graph), tiny_target, m=4, n_subsets=500)
    assert value < 1e-9


# --------------------------------------------------------------------------
# criterion 8 — at_beta and validate_bands
# --------------------------------------------------------------------------


def test_at_beta_refuses_a_beta_the_config_loader_would(tiny_target):
    """Exit criterion 8.  Phase 0's ``r_fail_margin`` assertion exists so a β
    sweep cannot quietly promote ``FAIL`` into a competitive terminal; a sweep
    moving β through ``at_beta`` would otherwise walk straight past the loader."""
    with pytest.raises(ConfigError, match="r_fail"):
        tiny_target.at_beta(25.0)


def test_at_beta_reuses_the_cached_utilities(tiny_target):
    """G7.  Each additional β in the Phase-3 sweep must be an ``exp`` over a
    vector, not a re-derivation of ``U`` for every terminal."""
    other = tiny_target.at_beta(2.0)
    assert other.u is tiny_target.u
    assert other.beta == 2.0
    assert other.z != tiny_target.z
    assert float(other.p_star.sum()) == pytest.approx(1.0, abs=1e-15)


def test_at_beta_does_not_touch_the_mass_bands(main_suite):
    """Decision 10.  ``at_beta`` is what the β sweep runs on the *tuning* suite,
    so a single call that always checked the main-suite band would abort the
    sweep on a condition that suite is exempt from."""
    instance = main_suite[0]
    target = target_distribution(instance, instance.cfg)
    assert target.at_beta(1.0).mass_profile()["mode_mass"]["neither"] > 0.5


def test_validate_bands_raises_on_the_main_suite_at_a_beta_that_breaks_it(main_suite):
    """Exit criterion 8, second half.  ``p*`` depends on β, so skipping the band
    check entirely would let the sweep leave the frozen main suite violating its
    own acceptance condition."""
    instance = main_suite[0]
    target = target_distribution(instance, instance.cfg)
    target.validate_bands("main")  # at the frozen beta, it holds
    with pytest.raises(BandViolation) as exc:
        target.at_beta(1.0).validate_bands("main")
    assert exc.value.band == "neither_mass"


@pytest.mark.parametrize("scope", ["tuning", "probe"])
def test_validate_bands_only_reports_for_the_other_suites(main_suite, scope):
    instance = main_suite[0]
    profile = target_distribution(instance, instance.cfg).at_beta(1.0).validate_bands(scope)
    assert profile["scope"] == scope
    assert profile["mode_mass"]["neither"] > 0.5


@pytest.mark.parametrize("scope", ["mian", "MAIN", "", "anything", "Main "])
def test_an_unrecognised_scope_is_refused_rather_than_waved_through(main_suite, scope):
    """A post-build review found this fail-open.

    The gate is only as good as the string it is called with: every other
    decision on this path fails closed — ``Assertion.eligibility`` defaults to
    quarantined, ``H`` rejects an unresolvable provenance chain — so a typo that
    skipped the hard band silently was the one fail-open step in the chain.
    """
    target = target_distribution(main_suite[0], main_suite[0].cfg).at_beta(1.0)
    with pytest.raises(ValueError, match="scope must be one of"):
        target.validate_bands(scope)


def test_the_scope_list_matches_the_suites_that_exist():
    """``exact`` cannot import ``lattice`` (that would be a cycle), so the two
    lists are tied together here instead of by an import."""
    from graft.synth.exact import SCOPES
    from graft.synth.lattice import SUITE_SIZES

    assert set(SCOPES) == set(SUITE_SIZES)


# --------------------------------------------------------------------------
# criterion 11 — budgets
# --------------------------------------------------------------------------


def test_one_exact_evaluation_stays_inside_its_ceiling(bench, bench_graph, bench_target):
    """Exit criterion 11, decision 17.

    Gate 2 is 7 learners x 3 seeds x **C = 50 checkpoints** x 20 instances =
    21,000 evaluations, budgeted at <= 1 h in total, which fixes the
    per-evaluation ceiling at <= 0.15 s.  The 0.15 s covers the numpy DP given
    precomputed log-probabilities and nothing else: the policy's batched forward
    pass is a property of the learner, and folding it in would make Phase 2's
    budget depend on Phase 3's architecture.

    FCS is budgeted separately at <= 0.05 s per instance, bringing the
    per-evaluation total to <= 0.20 s and the Gate-2 total to <= 1.2 h.
    """
    policy = UniformPolicy()
    policy_distribution(policy, bench_graph)  # warm numpy
    start = time.perf_counter()
    for _ in range(10):
        p = policy_distribution(policy, bench_graph)
    per_eval = (time.perf_counter() - start) / 10
    assert per_eval <= 0.15, f"{per_eval:.4f}s per exact evaluation"

    start = time.perf_counter()
    fcs(p, bench_target)
    assert (time.perf_counter() - start) <= 0.05


def test_enumeration_and_target_construction_stay_inside_sixty_seconds(bench):
    start = time.perf_counter()
    graph = reachable_states(bench, bench.cfg)
    target_distribution(bench, bench.cfg, graph=graph)
    assert (time.perf_counter() - start) <= 60.0


def test_the_divergence_report_puts_p_fail_beside_tv_and_never_inside_it(
    bench_graph, bench_target
):
    """G4 and criterion 7, as one call so the reporting rules cannot drift apart.

    ``FAIL`` is in **both** distributions, so a policy assigning it exactly
    ``p*(FAIL)`` reaches TV = 0 — the convergence target is 0 and stays 0.  The
    one true statement is narrower: a policy that *cannot* reach ``FAIL`` carries
    ``TV >= p*(FAIL)``.
    """
    from graft.synth.exact import divergence_report

    oracle = FlowOraclePolicy(bench_target, bench_graph)
    report = divergence_report(policy_distribution(oracle, bench_graph), bench_target)
    assert report["tv"] < 1e-9, "the oracle reaches 0, not p*(FAIL)"
    assert report["target_p_fail"] > 0.0
    assert report["kl_finite"]
    assert report["fcs_se"] > 0.0
    assert report["partition_residual"] <= 1e-9

    # The narrow, conditional statement: a policy that never dead-ends carries at
    # least p*(FAIL) of TV.  Reported as a diagnostic, never subtracted.
    p = policy_distribution(UniformPolicy(), bench_graph)
    p_no_fail = p.copy()
    p_no_fail[-1] = 0.0
    p_no_fail /= p_no_fail.sum()
    assert tv(p_no_fail, bench_target.p_star) >= bench_target.target_p_fail


def test_forward_mass_is_the_one_pass_everything_reuses(bench_graph):
    """``policy_distribution``, the absorption audit and the visitation-weighted
    ``Δd`` density all read the same forward pass rather than each running a DP."""
    f, edge_prob, stop = forward_mass(UniformPolicy(), bench_graph)
    assert f[0] == 1.0
    assert edge_prob.shape[0] == bench_graph.n_edges
    assert np.all(stop[bench_graph.dead_ix] == 0.0)
    direct = policy_distribution(UniformPolicy(), bench_graph)
    assert float(direct[-1]) == pytest.approx(float(f[bench_graph.dead_ix].sum()), rel=1e-15)
