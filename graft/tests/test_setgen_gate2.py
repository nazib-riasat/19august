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


def test_l6_matches_l7_exactly_because_the_delta_d_block_is_zeroed_not_absent(env):
    """G4, decision 11. The `Δd` block exists in both featurisations and is
    zeroed for L6, so the two arms have identical parameter shapes and the match
    is 0.00% rather than "within 1%"."""
    spec = TrainSpec(n_trajectories=32)
    _, note = capacity_matched_arm("l6_led", "l7_checker_led", [env], spec)
    assert note["relative_excess"] == 0.0
    assert note["control_never_smaller"]
    assert note["control_hidden"] == spec.hidden


def test_gaflownet_is_widened_until_it_is_not_smaller_than_l7(env):
    """GAFlowNet carries no `PotentialHead` and is ~37% smaller at equal width.
    A control with fewer parameters turns any win into "L7 had more capacity",
    which is the objection capacity matching exists to remove."""
    spec = TrainSpec(n_trajectories=32)
    arm, note = capacity_matched_arm("gaflownet", "l7_checker_led", [env], spec)

    assert note["control_hidden"] > spec.hidden
    assert note["control_never_smaller"]
    assert 0.0 <= note["relative_excess"] <= 0.01
    assert Trainer(arm, [env], spec).capacity == note["control_capacity"]


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
    report.comparisons = {
        "l7_vs_l6_led": {"wins": tv.get("beats_l6", True)},
        "l7_vs_gaflownet": {"wins": tv.get("beats_gafn", True)},
        "gaflownet": {"final_c_t": tv.get("c_t", [0.0, 0.0, 0.0])},
    }
    report.consistency = [
        {"arm": "l6_led", "p95_worst": tv.get("p95_l6", 0.02)},
        {"arm": "l7_checker_led", "p95_worst": tv.get("p95_l7", 0.02)},
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
    """Decision 15 and plan §4.5.4's hard constraint. If L7's consistency leaves
    the band, the TV win was bought by weakening LED's regulariser."""
    arms = ["l6_led", "l7_checker_led", "gaflownet"]
    out = _verdict(_report(p95_l7=0.20), arms)
    assert not out["contribution_3_supported"]
    assert not out["consistency_band_passed"]


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

    assert report.tv_matrix("l4_tb").shape == (2, 1)
    assert report.comparisons["l7_vs_l6_led"]["n_seeds"] == 2
    assert "contribution_3_supported" in report.verdict
    # p*(FAIL) beside TV, never subtracted from it
    row = report.divergences["l4_tb"][0]
    assert {"tv", "js", "fcs", "fcs_se", "target_p_fail", "policy_p_fail"} <= row.keys()


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
