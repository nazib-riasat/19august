"""Terminal reward.

Discharges Phase-1 exit criteria 11 (``R = 1[H]*exp(beta*U)``, ``R(invalid) = 0``
exactly, ``log_reward(invalid) = -inf``), 12 (the ``exp(beta*0) = 1`` regression)
and 13 (``R(FAIL) = r_fail``).
"""

from __future__ import annotations

import math
import random

import pytest

from graft.config import load_config
from graft.core.checker import H
from graft.core.reward import (
    fail_reward,
    log_fail_reward,
    log_reward,
    log_reward_from_parts,
    reward,
    reward_from_parts,
)
from graft.core.utility import U
from graft.ledger import Ledger
from graft.tests.fixtures import build_instance, instances, random_subsets

CFG = load_config()


@pytest.fixture
def inst():
    return build_instance(random.Random(13))


# -- criterion 12: the regression that motivates the whole form -------------


@pytest.mark.parametrize("beta", [0.5, 1.0, 4.0, 10.0])
def test_an_invalid_set_never_scores_one(beta):
    """v1.2 §1.1.  An earlier draft set ``U = 0`` on hard failure, which yields
    ``exp(0) = 1`` — a positive and, in a low-utility regime, competitive reward
    for a set that failed a hard check.  The indicator has to be multiplicative.

    ``U = 0`` is exactly where the error hid, so it is tested explicitly.
    """
    cfg = CFG.with_overrides(beta=beta)
    for utility in (-0.35, 0.0, 0.5, 2.25):
        assert reward_from_parts(False, utility, cfg) == 0.0
    assert reward_from_parts(False, 0.0, cfg) != 1.0


def test_the_indicator_is_multiplicative_not_an_additive_zero():
    valid = reward_from_parts(True, 0.0, CFG)
    invalid = reward_from_parts(False, 0.0, CFG)
    assert valid == 1.0      # a *valid* set with zero utility legitimately scores 1
    assert invalid == 0.0    # an invalid one scores nothing, whatever its utility


# -- criterion 11 -----------------------------------------------------------


def test_reward_matches_the_formula(inst):
    utility = U(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    expected = math.exp(CFG.beta * utility)
    got = reward(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    assert got == pytest.approx(expected)


def test_invalid_sets_score_zero_and_minus_infinity(inst):
    assert not H([], inst.obligations, inst.graph, inst.pool, CFG).ok
    assert reward([], inst.obligations, inst.graph, inst.pool, inst.gold, CFG) == 0.0
    assert log_reward([], inst.obligations, inst.graph, inst.pool, inst.gold, CFG) == -math.inf


def test_log_reward_is_beta_times_u_on_valid_sets(inst):
    utility = U(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    got = log_reward(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    assert got == pytest.approx(CFG.beta * utility)


def test_log_reward_and_reward_agree_wherever_reward_is_positive():
    rng = random.Random(6)
    for instance in instances(seed=81, count=4):
        for subset in random_subsets(rng, instance.pool, 60, CFG.max_atoms):
            r = reward(subset, instance.obligations, instance.graph, instance.pool, instance.gold, CFG)
            lr = log_reward(
                subset, instance.obligations, instance.graph, instance.pool, instance.gold, CFG
            )
            if r > 0.0:
                assert math.log(r) == pytest.approx(lr)
            else:
                assert lr == -math.inf


def test_reward_is_never_negative():
    """GFlowNet theory requires a nonnegative terminal reward."""
    rng = random.Random(8)
    for instance in instances(seed=91, count=4):
        for subset in random_subsets(rng, instance.pool, 100, CFG.max_atoms):
            assert (
                reward(subset, instance.obligations, instance.graph, instance.pool, instance.gold, CFG)
                >= 0.0
            )


def test_reward_stays_inside_the_range_beta_and_the_weights_imply():
    lo = math.exp(CFG.beta * CFG.u_weights.u_min)
    hi = math.exp(CFG.beta * CFG.u_weights.u_max)
    rng = random.Random(10)
    for instance in instances(seed=101, count=4):
        for subset in random_subsets(rng, instance.pool, 100, CFG.max_atoms):
            r = reward(subset, instance.obligations, instance.graph, instance.pool, instance.gold, CFG)
            assert r == 0.0 or lo <= r <= hi


# -- criterion 13 -----------------------------------------------------------


def test_fail_carries_r_fail():
    assert fail_reward(CFG) == CFG.r_fail
    assert log_fail_reward(CFG) == pytest.approx(math.log(CFG.r_fail))


def test_log_of_fail_is_finite_so_balance_losses_never_see_log_zero():
    """``FAIL`` is a member of the target's support, so it must carry a usable
    log-reward; the config refuses ``r_fail <= 0`` to keep it finite."""
    assert math.isfinite(log_fail_reward(CFG))


def test_fail_is_negligible_against_the_worst_valid_terminal():
    """``p*(FAIL) <= r_fail / exp(beta*U_min)``, bounded by ``r_fail_margin``."""
    assert fail_reward(CFG) < CFG.r_fail_margin * CFG.r_valid_min


def test_fail_is_far_below_every_reachable_valid_reward():
    rng = random.Random(12)
    instance = build_instance(rng)
    valid_rewards = [
        r
        for subset in random_subsets(rng, instance.pool, 300, CFG.max_atoms)
        if (
            r := reward(
                subset, instance.obligations, instance.graph, instance.pool, instance.gold, CFG
            )
        )
        > 0.0
    ]
    assert valid_rewards
    assert fail_reward(CFG) < min(valid_rewards) * CFG.r_fail_margin


# -- metering ---------------------------------------------------------------


def test_reward_spends_one_terminal_check_when_it_validates(inst):
    ledger = Ledger.from_config(CFG)
    with ledger.query_scope("q"):
        reward(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG, ledger=ledger)
        assert ledger.snapshot()["query"]["terminal_checks"] == 1


def test_a_precomputed_check_is_reused_rather_than_paid_for_twice(inst):
    """The portfolio path filters by ``H`` then ranks; paying twice would halve
    the effective budget."""
    ledger = Ledger.from_config(CFG)
    with ledger.query_scope("q"):
        result = H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG, ledger=ledger)
        reward(
            inst.gold,
            inst.obligations,
            inst.graph,
            inst.pool,
            inst.gold,
            CFG,
            ledger=ledger,
            check=result,
        )
        assert ledger.snapshot()["query"]["terminal_checks"] == 1


def test_beta_scales_the_log_reward_linearly(inst):
    """What the Phase-3 sweep will move, and why U must stay bounded."""
    utility = U(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    for beta in (1.0, 2.0, 8.0):
        cfg = CFG.with_overrides(beta=beta)
        assert log_reward_from_parts(True, utility, cfg) == pytest.approx(beta * utility)
