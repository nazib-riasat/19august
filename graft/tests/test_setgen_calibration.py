"""The calibration gate's arithmetic (Phase-3 step 6, `scripts/phase3_calibrate.py`).

``budget_for`` produces ``N`` — fix F12's primary budget, identical across every
arm by criterion 11 — and it had no test. Two defects lived in it as a result:
its docstring promised rounding to a batch multiple that the code never did, and
it billed the ceiling for training time only while the real run also pays for
``checkpoints + 1`` exact-TV evaluations. The second is the one that bites: the
throughput pilot evaluates twice and a rung evaluates ~51 times, so the rate came
out too high, ``N`` too large, and every rung overran the ceiling it was derived
from with nothing measuring it.

The script is not a package, so it is loaded by path — the same way it is run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "phase3_calibrate.py"
_SPEC = importlib.util.spec_from_file_location("phase3_calibrate", _PATH)
calib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calib)


def test_the_budget_is_a_whole_number_of_batches():
    """`N` is spent in batches, so a `N` that is not a multiple of the batch ends
    on a short one. Criterion 11 makes `N` the quantity held identical across
    every arm, which is a reason to have it be exactly spendable."""
    n = calib.budget_for(3600.0, 1000.0, batch=32, eval_cost=0.0, checkpoints=50)
    assert n % 32 == 0
    assert n == 3_600_000 - 3_600_000 % 32
    # never returns zero, however tight the ceiling
    assert calib.budget_for(1.0, 1.0, batch=32, eval_cost=0.0, checkpoints=0) == 32


def test_the_ceiling_pays_for_the_evaluations_before_it_buys_trajectories():
    """The trainer evaluates once at trajectory 0 — the random-init baseline —
    and then at each of the C checkpoints, so a rung owes `C + 1` evaluations
    before any training time. Ignoring them is what made the pilot's rate, which
    amortises 2 evaluations, over-predict a run that pays for 51."""
    free = calib.budget_for(1000.0, 100.0, batch=10, eval_cost=0.0, checkpoints=50)
    charged = calib.budget_for(1000.0, 100.0, batch=10, eval_cost=1.0, checkpoints=50)
    assert free == 100_000
    # 51 evaluations at 1 s leaves 949 s of training
    assert charged == 94_900
    assert charged < free


def test_a_ceiling_that_cannot_pay_for_its_evaluations_is_refused():
    """Silently returning a tiny `N` would make the rung look like a budget for
    something. It is not, and the ladder should say so rather than train on it."""
    with pytest.raises(ValueError, match="no training time"):
        calib.budget_for(10.0, 100.0, batch=32, eval_cost=1.0, checkpoints=50)


def test_an_over_ceiling_rung_is_never_adopted(monkeypatch):
    """Decision 5's ceiling is what makes `N` affordable. A rung whose runs cost
    more than it says the throughput estimate that produced `N` was wrong, so the
    budget the rung reports is not the budget it spent — adopting it would write
    an unfunded `N` into §6. Escalating is worse, because the next rung's ceiling
    is larger, so the ladder stops.

    An earlier version recorded `ceiling_respected` and then adopted on the
    sanity result alone, which made the field decorative."""
    monkeypatch.setattr(calib, "beta_eligibility",
                        lambda *a, **k: {"eligible": [4.0], "per_instance": []})
    monkeypatch.setattr(calib, "measure_throughput",
                        lambda *a, **k: (1000.0, 0.0, {"l6_led": 1000.0}))
    monkeypatch.setattr(calib, "tuning_suite", lambda *a, **k: ())
    monkeypatch.setattr(calib, "Environment", lambda inst: inst)
    monkeypatch.setattr(calib, "retarget", lambda env, beta: None)
    # a sweep and a sanity check that pass, but whose slowest run blows the ceiling
    monkeypatch.setattr(calib, "beta_sweep", lambda *a, **k: {
        "winner": 4.0, "per_beta": {}, "max_run_s": 10_000.0})
    monkeypatch.setattr(calib, "sanity_check", lambda *a, **k: {
        "passed": True, "arms": {}, "threshold": 0.10, "max_run_s": 1.0})

    record = calib.calibrate([3600.0, 7200.0])
    assert record["verdict"] == "over_ceiling"
    assert record["adopted"] is None, "an over-ceiling rung was adopted"
    assert len(record["rungs"]) == 1, "the ladder escalated past an overrun"
    assert not record["rungs"][0]["ceiling_respected"]
    assert "throughput estimate" in record["note"]


def test_the_ceiling_is_measured_per_run_not_averaged(monkeypatch):
    """One slow seed can hide inside a mean that clears the bar. The ceiling is a
    per-run budget, so the rung is judged on its slowest single run."""
    monkeypatch.setattr(calib, "beta_eligibility",
                        lambda *a, **k: {"eligible": [4.0], "per_instance": []})
    monkeypatch.setattr(calib, "measure_throughput",
                        lambda *a, **k: (1000.0, 0.0, {"l6_led": 1000.0}))
    monkeypatch.setattr(calib, "tuning_suite", lambda *a, **k: ())
    monkeypatch.setattr(calib, "Environment", lambda inst: inst)
    monkeypatch.setattr(calib, "retarget", lambda env, beta: None)
    # mean over the two would be 1800 s and clear a 3600 s ceiling; the max does not
    monkeypatch.setattr(calib, "beta_sweep", lambda *a, **k: {
        "winner": 4.0, "per_beta": {}, "max_run_s": 3_600.1})
    monkeypatch.setattr(calib, "sanity_check", lambda *a, **k: {
        "passed": True, "arms": {}, "threshold": 0.10, "max_run_s": 1.0})

    record = calib.calibrate([3600.0])
    assert record["rungs"][0]["max_s_per_run"] == 3_600.1
    assert record["verdict"] == "over_ceiling"


def test_the_predeclared_grid_and_the_sanity_threshold_are_unchanged():
    """Phase-2 decision 19's grid does not move — what moves is which of its
    members are eligible (PHASE2_DECISIONS §4.2). Decision 6's threshold is
    signed before calibration, because a threshold written after the run it
    judges is not a threshold."""
    assert calib.BETA_CANDIDATES == (1.0, 2.0, 4.0, 8.0)
    assert calib.SANITY_TV == 0.10
    assert calib.DEFAULT_RUNGS == (3600.0, 7200.0, 14400.0)
