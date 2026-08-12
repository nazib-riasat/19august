"""The Gate-2 harness: the paired test, capacity matching, and the decision rule.

**The decision rule is tested on synthetic inputs on purpose.** Exit criterion 26
requires a written rule that exists *before* the numbers do; the way to check
such a rule is to feed it outcomes whose verdict is known by construction, not to
run the matrix and see whether the answer looks reasonable. These tests are the
executable form of fix F12's demand that the rule be falsifiable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from graft.setgen.gate2 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONSISTENCY_FORCED_SEED,
    CONSISTENCY_P95_BAND,
    NON_INFERIORITY_MARGIN,
    Gate2Report,
    _verdict,
    audit_block,
    capacity_matched_arm,
    consistency_report,
    paired_bootstrap,
    run_matrix,
)
from graft.setgen.learners import build_arm
from graft.setgen.trainer import SEEDS, Environment, Trainer, TrainSpec
from graft.synth.lattice import tiny_instance


@pytest.fixture(scope="module")
def env():
    return Environment(tiny_instance())


# -- the paired test --------------------------------------------------------


def test_the_bootstrap_is_frozen_before_any_run():
    """Decision 20. A resample count or seed chosen after seeing results is a
    decision rule selected on the data, which Dror et al. (ACL 2018) forbids and
    which this project cites that paper to forbid elsewhere."""
    assert (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED) == (10_000, 20260814)


def test_a_clear_win_is_called_a_win_and_a_clear_loss_is_not():
    """The one-sided 95% upper bound on `TV_a - TV_b` below 0."""
    rng = np.random.default_rng(1)
    better = rng.normal(0.10, 0.01, size=(3, 20))
    worse = better + 0.05
    assert paired_bootstrap(better, worse, n_resamples=2000)["wins"]
    assert not paired_bootstrap(worse, better, n_resamples=2000)["wins"]


def test_a_tie_is_not_a_win():
    """The failure mode fix F12 exists to retire: a rule that says yes to noise."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.10, 0.02, size=(3, 20))
    b = rng.normal(0.10, 0.02, size=(3, 20))
    assert not paired_bootstrap(a, b, n_resamples=2000)["wins"]


def test_the_bootstrap_is_hierarchical_and_that_widens_the_interval():
    """Twenty instances per (arm, seed) share the seed, the weights and the
    generator, so they are not twenty independent draws. Resampling them flat
    understates the interval — the clustering error the LoCoMo caveat guards
    against elsewhere in this project.

    Constructed so the *seed* is the dominant source of variation: a flat
    bootstrap over 60 pairs would then be far too confident.
    """
    rng = np.random.default_rng(3)
    seed_effect = np.array([-0.05, 0.0, 0.05])[:, None]
    diff = seed_effect + rng.normal(0.0, 0.001, size=(3, 20))
    a, b = diff, np.zeros_like(diff)

    hierarchical = paired_bootstrap(a, b, n_resamples=4000)

    flat = np.empty(4000)
    flatten = diff.ravel()
    r = np.random.default_rng(BOOTSTRAP_SEED)
    for i in range(4000):
        flat[i] = flatten[r.integers(flatten.size, size=flatten.size)].mean()
    assert hierarchical["upper_95"] > float(np.percentile(flat, 95)), (
        "the clustered interval must be wider than the flat one; if it is not, "
        "the outer resample is not doing anything"
    )


def test_unpaired_inputs_are_refused():
    """Pairing removes instance difficulty and seed luck. It is only valid
    because every arm trains on the same suite under the same seeds, so a shape
    mismatch means that assumption has already broken."""
    with pytest.raises(ValueError, match="unpaired"):
        paired_bootstrap(np.zeros((3, 20)), np.zeros((3, 19)))


# -- capacity matching ------------------------------------------------------


def test_l6_is_matched_on_live_capacity_not_on_the_nominal_count(env):
    """G4, decision 11. The `Δd` block is present and zeroed under L6, so its
    weight columns read an identically-zero input and can never receive a
    gradient. Counted nominally the two arms agreed to 0.00% while L6 held 1.46%
    *less trainable* capacity — the wrong side of the directional clause, in the
    direction that flatters the proposed method."""
    spec = TrainSpec(n_trajectories=32)
    arm, note = capacity_matched_arm("l6_led", "l7_checker_led", [env], spec)

    # the dead block is what the old nominal match was hiding: L6 carries the
    # Δd block (2 consumers x 6 dims) on top of the LogZ pad every arm has
    assert note["control_dead_parameters"] == 20 * note["control_hidden"]
    assert note["target_live_capacity"] < note["target_capacity"], "L7 has the pad"
    assert note["control_hidden"] > spec.hidden, "L6 must widen to compensate"
    assert note["control_never_smaller"]
    assert note["narrowest_admissible_width"]
    assert 0.0 < note["relative_excess"] < note["width_step"] / note["target_live_capacity"]
    assert Trainer(arm, [env], spec).live_capacity == note["control_live_capacity"]


def test_gaflownet_is_widened_until_it_is_not_smaller_than_l7(env):
    """GAFlowNet carries no `PotentialHead` and is ~37% smaller at equal width.
    A control with fewer parameters turns any win into "L7 had more capacity",
    which is the objection capacity matching exists to remove."""
    spec = TrainSpec(n_trajectories=32)
    arm, note = capacity_matched_arm("gaflownet", "l7_checker_led", [env], spec)

    assert note["control_hidden"] > spec.hidden
    assert note["control_never_smaller"]
    assert note["narrowest_admissible_width"]
    assert Trainer(arm, [env], spec).capacity == note["control_capacity"]
    assert Trainer(arm, [env], spec).live_capacity == note["control_live_capacity"]


def test_the_capacity_gap_is_half_a_width_step_at_every_scale(env):
    """Why decision 11's 1% is unachievable rather than merely missed.

    The quantity that drives the widening is the **asymmetric** dead block — the
    `Δd` columns L6 carries and L7 does not, `12·hidden` across the two
    `action_repr` consumers. (The `LogZHead` pad is dead in both arms and cancels
    in the match.) One unit of width is worth ~`2.4·hidden`, so the gap is ~half
    a step at *any* width, and the smallest width that closes it always
    overshoots by roughly half a step. A property of the architecture, so no
    choice of base width rescues the 1%."""
    from graft.setgen.features import SyntheticFeaturizer

    state_dim, action_dim = SyntheticFeaturizer.dims(env.instance, env.graph)
    l6, l7 = build_arm("l6_led"), build_arm("l7_checker_led")
    for hidden in (64, 128, 256):
        step = (
            Trainer.capacity_of(l6, state_dim, action_dim, hidden + 1, 2)
            - Trainer.capacity_of(l6, state_dim, action_dim, hidden, 2)
        )
        gap = Trainer.dead_capacity_of(l6, hidden) - Trainer.dead_capacity_of(l7, hidden)
        assert gap == 12 * hidden, "the asymmetry is the Δd block alone"
        assert 0.45 <= gap / step <= 0.60


def test_matching_rounds_up_never_down():
    """Decision 11 has two clauses and only one is symmetric: "within 1%" is a
    tolerance, "never smaller" is a direction."""
    from graft.setgen.policy import match_capacity

    width = match_capacity(lambda h: 10 * h, target=1005, base=100, tol=0.01)
    assert 10 * width >= 1005
    with pytest.raises(ValueError, match="granularity"):
        match_capacity(lambda h: 1000 * h, target=1_500, base=1, tol=0.01)


# -- consistency ------------------------------------------------------------


def test_consistency_uses_one_frozen_set_shared_by_the_led_arms(env):
    """Decision 14. Per-arm sampling produces three p95 values measured on three
    populations, which are not comparable — and the incomparability is invisible
    in the numbers themselves."""
    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    reports = [
        consistency_report(Trainer(build_arm(name), [env], spec))
        for name in ("l6_led", "l7_checker_led", "l7b_aux")
    ]
    counts = {(r["per_instance"][0]["n_valid"], r["per_instance"][0]["n_fail"])
              for r in reports}
    assert len(counts) == 1, "the three arms saw different trajectory sets"
    assert reports[0]["band"] == CONSISTENCY_P95_BAND


def test_fail_is_covered_and_audited_on_its_own_line(env):
    """G10. `ℰ(FAIL) = -log r_fail = 13.8` sits far outside the valid range, so
    pooling would let a handful of trajectories own a p95 over thousands. The
    ForcedContinuationPolicy draw exists to *guarantee* coverage rather than hope
    for it — UniformPolicy stops early and can miss the dead ends entirely."""
    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    report = consistency_report(Trainer(build_arm("l6_led"), [env], spec))
    row = report["per_instance"][0]

    assert row["fail_covered"], "no FAIL trajectory in the frozen set"
    assert report["fail_covered_everywhere"]
    assert row["n_fail"] > 0 and row["n_valid"] > 0
    assert not math.isnan(row["fail_p95"])
    assert row["fail_p95"] != row["p95"], "FAIL was pooled into the valid line"
    assert CONSISTENCY_FORCED_SEED == 20260816


def test_an_arm_without_a_potential_cannot_be_asked_for_consistency(env):
    spec = TrainSpec(n_trajectories=32, checkpoints=1)
    with pytest.raises(ValueError, match="no potential"):
        consistency_report(Trainer(build_arm("l4_tb"), [env], spec))


# -- the decision rule ------------------------------------------------------


def _report(**tv):
    report = Gate2Report()
    report.spec = {
        "admissible": tv.get("admissible", True),
        "inadmissible_reason": None,
    }
    report.comparisons = {
        "tv_threshold": 0.10,
        "l7_vs_l6_led": {"wins": tv.get("beats_l6", True)},
        "l7_vs_gaflownet": {"wins": tv.get("beats_gafn", True)},
        "gaflownet": {"final_c_t": tv.get("c_t", [0.0, 0.0, 0.0])},
        # Criterion 15: L4 and L5 on the **main** suite, not the tuning one.
        "l4_tb": {"final_tv_mean": tv.get("tv_l4", 0.02)},
        "l5_subtb": {"final_tv_mean": tv.get("tv_l5", 0.03)},
    }
    covered = tv.get("fail_covered", True)
    report.consistency = [
        {"arm": "l6_led", "p95_worst": tv.get("p95_l6", 0.02),
         "fail_covered_everywhere": covered},
        {"arm": "l7_checker_led", "p95_worst": tv.get("p95_l7", 0.02),
         "fail_covered_everywhere": covered},
    ]
    return report


def test_contribution_3_needs_both_controls():
    """Exit criterion 26. Beating L6 alone is not sufficient: plan §4.5.4 makes
    GAFlowNet a required control precisely because "an intermediate signal helps"
    is already published."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    assert _verdict(_report(), arms)["contribution_3_supported"]
    assert not _verdict(_report(beats_gafn=False), arms)["contribution_3_supported"]
    assert not _verdict(_report(beats_l6=False), arms)["contribution_3_supported"]


def test_a_tv_win_bought_by_relaxing_the_regulariser_is_refused():
    """Decision 15 and plan §4.5.4's hard constraint. If **L7's** consistency
    leaves the band, the TV win was bought by weakening LED's regulariser — that
    is a result about C3, so it reads `not_supported`, not `inconclusive`."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    out = _verdict(_report(p95_l7=0.20), arms)
    assert out["contribution_3_supported"] is False
    assert out["outcome"] == "not_supported"
    assert not out["consistency_band_passed"]


def test_a_failed_l6_band_is_the_instrument_not_a_verdict_on_c3():
    """Item 8. `band_ok` pooled both arms, so vanilla LED-GFN failing to
    decompose its own energy at this `N` — a statement about the instrument and
    the budget — came out as `contribution_3_supported = False`. The component
    booleans were recoverable from the artefact; the headline field was not."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    out = _verdict(_report(p95_l6=0.20), arms)
    assert out["contribution_3_supported"] is None
    assert out["outcome"] == "inconclusive"
    assert out["instrument"]["l6_consistency_band"] is False
    # ...and the comparisons are still reported, because they are still true
    assert out["beats_capacity_matched_l6"] is True


def test_criterion_15_gates_the_verdict_on_the_main_suite():
    """Criterion 15 was in the plan and in no code path. Decision 6's threshold
    is checked on 5 **tuning** instances during calibration; the matrix runs 20
    **main** instances through one conditional `logZ` head, and passing the first
    does not imply the second. A matrix where the machinery failed on the scored
    suite still emitted a scientific boolean."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    assert _verdict(_report(), arms)["instrument"]["machinery_criterion_15"]

    out = _verdict(_report(tv_l5=0.42), arms)
    assert out["contribution_3_supported"] is None
    assert out["outcome"] == "inconclusive"
    assert not out["instrument"]["machinery_criterion_15"]
    assert out["machinery_final_tv"]["l5_subtb"] == 0.42


def test_third_decimal_noise_does_not_decide_a_contribution():
    """Decision 15 again: a **non-inferiority margin**, not an ordering. 0.030
    versus 0.031 is noise and both pass the band."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    assert _verdict(_report(p95_l6=0.030, p95_l7=0.031), arms)["contribution_3_supported"]
    assert not _verdict(
        _report(p95_l6=0.030, p95_l7=0.030 + 2 * NON_INFERIORITY_MARGIN), arms
    )["non_inferiority_margin_passed"]


def test_a_gaflownet_that_never_reached_zero_invalidates_its_own_comparison():
    """Criterion 14. With `c_t > 0` its target is `R + r` rather than `R`, so its
    exact TV is measured against a distribution it never aimed at — and "L7 beats
    GAFlowNet" would be an artefact of the augmentation, not a finding."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    out = _verdict(_report(c_t=[0.0, 0.01, 0.0]), arms)
    assert not out["gaflownet_intrinsic_reached_zero"]
    assert not out["contribution_3_supported"]


def test_an_uncovered_fail_line_blocks_the_verdict():
    """Criterion 16 and decision 14 require at least one FAIL trajectory per
    instance. `consistency_report` measured it and `_verdict` never read it, so a
    frozen set that happened to contain no dead end would have passed the band
    silently — and `ℰ(FAIL) = -log r_fail = 13.8` is exactly the far tail the
    separate line exists to audit."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    out = _verdict(_report(fail_covered=False), arms)
    assert not out["fail_coverage_complete"]
    assert not out["contribution_3_supported"]


def test_an_inadmissible_run_returns_no_verdict_at_all():
    """Exit criterion 10: the seed set is frozen and "the config refuses to
    shorten" it. A wiring run on a partial roster used to emit a
    `contribution_3_supported` boolean indistinguishable from a real one, from a
    bootstrap whose outer resample had a single cluster. It now returns null."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    out = _verdict(_report(admissible=False), arms)
    assert out["contribution_3_supported"] is None
    assert out["admissible"] is False
    # ...and the component findings are still reported, because they are still true
    assert out["beats_capacity_matched_l6"] is True


@pytest.fixture(scope="module")
def frozen_suites():
    """The real frozen suites, generated but not enumerated — `_admissibility`
    reads instance identity, so this costs ~4 s rather than a full enumeration."""
    from graft.synth.lattice import benchmark_suite, probe_suite

    return benchmark_suite(), probe_suite()


def _stub(instance, beta=4.0):
    """`_admissibility` reads `.instance` and `.target.beta` and nothing else."""
    from types import SimpleNamespace

    return SimpleNamespace(instance=instance, target=SimpleNamespace(beta=beta))


def test_admissibility_checks_every_clause_that_defines_the_experiment(frozen_suites):
    """A verdict needs more than the right arms and seeds. The first version of
    this gate checked only those two, and a run on **one tiny instance at an
    uncalibrated N and β with no probe suite** passed it and produced a boolean.
    Every clause here is something the plan already required and nothing
    enforced: decision 8's frozen suite, decision 4 and Phase-2 decision 22's
    calibrated `N` and β, and criterion 23's held-out read."""
    from graft.setgen.gate2 import _admissibility
    from graft.setgen.learners import FLOW_FAMILY, SUPERVISED_FAMILY

    main, probe = frozen_suites
    full = list(FLOW_FAMILY) + list(SUPERVISED_FAMILY)
    envs = [_stub(i) for i in main]
    probes = [_stub(i) for i in probe]
    spec = TrainSpec(n_trajectories=4096)
    adopted = {"verdict": "adopted",
               "adopted": {"N": 4096, "beta": 4.0, "ceiling_s": 3600.0}}

    def check(**over):
        kwargs = {"arms": full, "seeds": SEEDS, "envs": envs, "spec": spec,
                  "probe_envs": probes, "calibration": adopted}
        kwargs.update(over)
        return _admissibility(**kwargs)

    assert check()["admissible"], check()["inadmissible_reason"]

    # roster and seeds
    assert "l7_checker_led" in check(arms=full[:3])["inadmissible_reason"]
    assert check(seeds=SEEDS[:2])["inadmissible_reason"].startswith("seeds")

    # decision 8: the frozen 20, checked by identity rather than by count
    assert "main suite" in check(envs=envs[:19])["inadmissible_reason"]
    assert "main suite" in check(envs=[_stub(i) for i in probe] * 4)["inadmissible_reason"]

    # criterion 23: the held-out read has to have happened
    assert "probe" in check(probe_envs=None)["inadmissible_reason"]
    assert "probe suite" in check(probe_envs=envs)["inadmissible_reason"]

    # decision 4 and Phase-2 decision 22: N and β come from step 6
    assert "no calibration" in check(calibration=None)["inadmissible_reason"]
    assert "wiring check" in check(
        calibration={**adopted, "quick": True}
    )["inadmissible_reason"]
    assert "inconclusive" in check(
        calibration={"verdict": "inconclusive"}
    )["inadmissible_reason"]
    assert "not the adopted" in check(
        spec=TrainSpec(n_trajectories=64)
    )["inadmissible_reason"]
    assert "beta" in check(
        envs=[_stub(i, beta=8.0) for i in main]
    )["inadmissible_reason"]

    # decision 5: the ceilings are a script argument, so an "adopted" record can
    # be produced at any budget at all -- `--rungs 0.5` passes every other clause
    assert "decision 5's rungs" in check(
        calibration={"verdict": "adopted",
                     "adopted": {"N": 4096, "beta": 4.0, "ceiling_s": 0.5}}
    )["inadmissible_reason"]


def test_the_rule_and_the_partial_discharge_are_stated_in_the_artefact():
    """Criterion 28. Claiming Gate-2 item 3 closed would not be honest: FM, DB,
    FL-DB and FL-SubTB remain deferred, with FL's row discharged by decision 24's
    measurement rather than by training."""
    out = _verdict(_report(), ["l6_led", "l7_checker_led", "gaflownet"])
    assert "consolidates on Contribution 1" in out["rule"]
    assert "deferred" in out["partial_discharge"]


# -- the matrix end to end --------------------------------------------------


def test_the_matrix_runs_and_carries_its_audits(env):
    """Exit criterion 22 and Phase-2 handoff item 7. A Gate-2 win is a win *under
    a declared signal density*; without the density printed beside it, the number
    does not carry its own caveat."""
    spec = TrainSpec(n_trajectories=128, checkpoints=2)
    report = run_matrix(
        [env], spec, arms=("l4_tb", "l6_led", "l7_checker_led", "gaflownet"),
        seeds=SEEDS[:2],
    )

    audits = report.audits
    assert audits["equivalent_action_collisions"] == 0
    assert audits["unconstructible_valid_terminals"] == 0
    assert all(np.isfinite(audits["zero_delta_d_structural"]))
    assert all(np.isfinite(audits["zero_delta_d_visitation"]))
    assert audits["target_p_fail"] and audits["neither_mass"]
    assert audits["environment_fingerprints"]
    # the whole target-mass profile, not one component of it
    profile = audits["target_mass_profile"][0]
    assert {"mode_mass", "mode_counts", "zero_sufficiency_mass", "u_min", "u_max"} <= (
        profile.keys()
    )

    assert report.tv_matrix("l4_tb").shape == (2, 1)
    assert report.comparisons["l7_vs_l6_led"]["n_seeds"] == 2
    # p*(FAIL) beside TV, never subtracted from it -- and at every seed, because
    # FCS carries a standard error and one seed of three supplies no spread
    assert set(report.divergences["l4_tb"]) == set(SEEDS[:2])
    row = report.divergences["l4_tb"][SEEDS[0]][0]
    assert {"tv", "js", "fcs", "fcs_se", "target_p_fail", "policy_p_fail"} <= row.keys()

    # a two-arm, two-seed run is a wiring check and may not return a verdict
    assert report.verdict["contribution_3_supported"] is None
    assert report.verdict["admissible"] is False
    assert "not the frozen" in report.verdict["inadmissible_reason"]


def test_best_of_k_is_the_metric_that_fits_the_supervised_arms(env):
    """Criterion 17's secondary and decision 12's routing. Exact TV is
    descriptive for L1-L3 — two of them imitate a single gold set and the third
    maximises return, so none is trying to match a distribution. Best-of-K
    utility and gold exact match are questions all nine arms answer, and until
    now L1-L3 had no metric of their own at all."""
    from graft.setgen.gate2 import best_of_k

    spec = TrainSpec(n_trajectories=4096, checkpoints=1)
    trainer = Trainer(build_arm("l2_imitation"), [env], spec)
    trainer.train()
    row = best_of_k(trainer)

    assert row["k"] == env.instance.cfg.K
    assert row["k"] <= env.instance.cfg.checker_budget, "fix F5: one constant"
    assert row["per_instance"][0]["terminal_checks"] == row["k"]
    assert 0.0 <= row["valid_rate"] <= 1.0
    assert env.target.u.min() <= row["best_utility_mean"] <= env.target.u.max()
    # The arm imitates one gold set, so gold exact match is the question it is
    # actually answering — and it discriminates: an untrained policy reaches the
    # gold terminal essentially never, this one reaches it most draws.
    untrained = best_of_k(Trainer(build_arm("l2_imitation"), [env], spec))
    assert row["gold_exact_match_mean"] > untrained["gold_exact_match_mean"]
    assert row["gold_exact_match_mean"] > 0.0


def test_best_of_k_reads_every_arm_under_one_frozen_stream(env):
    """Decision 14's reasoning, one metric over: three arms read under three
    random streams are three numbers measured on three populations."""
    from graft.setgen.gate2 import BEST_OF_K_SEED, best_of_k

    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    trainer = Trainer(build_arm("l4_tb"), [env], spec)
    assert BEST_OF_K_SEED == 20260817
    assert best_of_k(trainer) == best_of_k(trainer), "the draw is not reproducible"


def test_the_probe_suite_is_read_once_and_never_trained_on(env):
    """Criterion 23 and decision 9. Gate 2 measures fitting; the probe suite is
    what lets the write-up say anything about a sparser `Δd` density (G7). It is
    a different set of instances, so it needs its own featurizers against the
    trained policy — and it is only ever read."""
    from graft.synth.lattice import probe_suite

    probe = [Environment(inst) for inst in list(probe_suite())[:1]]
    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    report = run_matrix([env], spec, arms=("l4_tb",), seeds=SEEDS[:1],
                        probe_envs=probe)

    held_out = report.probe["l4_tb"][SEEDS[0]]
    assert len(held_out) == 1 and 0.0 <= held_out[0] <= 1.0
    assert report.comparisons["l4_tb"]["probe_tv"] == [pytest.approx(held_out[0])]
    # the probe instances are not the training instances
    trained = report.logs["l4_tb"][SEEDS[0]].fingerprints
    assert probe[0].fingerprints()["environment"] != trained[0]["environment"]


def test_every_run_records_the_environment_it_was_computed_against(env):
    """Exit criterion 25, Phase-2 decision 21. No result computed against one
    environment may be silently compared with another."""
    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    report = run_matrix([env], spec, arms=("l4_tb",), seeds=SEEDS[:1])
    prints = report.logs["l4_tb"][SEEDS[0]].fingerprints
    assert len(prints) == 1
    assert set(prints[0]) == {"environment", "target"}
    assert all(len(v) >= 16 for v in prints[0].values())


def test_censored_is_reported_as_censored_not_as_the_budget(env):
    """Decision 21. A secondary that reported the budget for an arm that never
    reached the threshold would silently favour whichever arm ran longest."""
    spec = TrainSpec(n_trajectories=64, checkpoints=2)
    report = run_matrix([env], spec, arms=("l4_tb",), seeds=SEEDS[:1],
                        tv_threshold=0.0)
    assert report.comparisons["l4_tb"]["trajectories_to_threshold"] == [None]


def test_the_supervised_arms_are_flagged_descriptive(env):
    """Decision 12's routing, carried into the artefact rather than left to the
    write-up to remember."""
    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    report = run_matrix([env], spec, arms=("l4_tb", "l1_supervised"), seeds=SEEDS[:1])
    assert report.comparisons["l1_supervised"]["descriptive_tv"] is True
    assert report.comparisons["l4_tb"]["descriptive_tv"] is False


def test_the_report_round_trips_to_json(tmp_path, env):
    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    report = run_matrix([env], spec, arms=("l4_tb",), seeds=SEEDS[:1])
    import json

    path = report.save(tmp_path / "gate2.json")
    loaded = json.loads(path.read_text("utf-8"))
    assert loaded["spec"]["epsilon"] == 0.05
    assert loaded["runs"]["l4_tb"]["13"]["capacity"] > 0
