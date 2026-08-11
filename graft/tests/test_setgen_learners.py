"""The nine arms, the shared trainer, and the objectives' arithmetic.

Discharges the parts of exit criteria 4, 8, 9, 10, 13, 14 and 16 that do not
need a full Gate-2 run, plus the identities each objective is written against.

**Why the arithmetic is tested against a hand computation rather than against
itself.** Every number Gate 2 produces flows through ``Batch.g``. A refactor that
broke it would leave TB, SubTB, LED-DB and augmented TB all wrong in the same
direction, every loss curve would still look plausible, and no comparison would
notice — the same failure mode Phase 2 recorded for the sampler-versus-DP
agreement, one level up.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from graft.setgen.learners import ARMS, FLOW_FAMILY, SUPERVISED_FAMILY, build_arm
from graft.setgen.learners.gaflownet import intrinsic_reward
from graft.setgen.learners.l3_grpo import group_advantage
from graft.setgen.learners.l4_tb import tb_residual
from graft.setgen.learners.l6_led import consistency_error, decomposition_loss
from graft.setgen.rollout import sample_trajectories
from graft.setgen.trainer import SEEDS, Environment, Trainer, TrainSpec
from graft.synth.lattice import tiny_instance

torch.manual_seed(0)


@pytest.fixture(scope="module")
def env():
    return Environment(tiny_instance())


def _trainer(env, name, **overrides):
    spec = TrainSpec(n_trajectories=overrides.pop("n", 256), checkpoints=2, **overrides)
    return Trainer(build_arm(name), [env], spec)


def _batch(trainer, env, n=48, seed=3):
    traj = sample_trajectories(
        trainer.featurizers[0], env.graph, n, np.random.default_rng(seed), 0.0
    )
    batch = trainer.build_batch(env, trainer.featurizers[0], traj)
    if trainer.potential is not None:
        batch.potential = trainer.compute_potential(batch)
    return traj, batch


# -- the batch's coordinates -----------------------------------------------


def test_the_terminating_transition_is_a_first_class_step(env):
    """A valid rollout ends by choosing STOP; a dead-ended one ends because
    nothing is legal. Both need a slot, or `P_F(STOP | x)` never appears in a
    balance condition and FAIL trajectories are one transition short."""
    trainer = _trainer(env, "l4_tb")
    traj, batch = _batch(trainer, env)

    expected = torch.as_tensor(traj.lengths + 1, dtype=torch.long)
    assert torch.equal(batch.n_trans, expected)
    assert torch.equal(batch.valid.sum(dim=1), expected)


def test_forced_termination_of_a_dead_end_carries_log_pf_zero(env):
    """A dead end has no legal ADD and a masked STOP, so it terminates with
    probability 1. Querying it for logits would give an all -inf row and NaN."""
    from graft.synth.policies import ForcedContinuationPolicy

    trainer = _trainer(env, "l4_tb")
    traj = sample_trajectories(
        ForcedContinuationPolicy(), env.graph, 400, np.random.default_rng(11)
    )
    if not traj.is_fail.any():
        pytest.skip("no FAIL trajectory in this draw")
    batch = trainer.build_batch(env, trainer.featurizers[0], traj)

    rows = np.flatnonzero(traj.is_fail)
    last = torch.as_tensor(traj.lengths[rows], dtype=torch.long)
    terminating = batch.log_pf[torch.as_tensor(rows), last]
    assert torch.all(terminating == 0.0)
    assert torch.isfinite(batch.log_pf).all()


def test_g_reproduces_trajectory_balance_by_hand(env):
    """`g[0] - g[L]` must equal `log Z + Σ log P_F - log R - Σ log P_B`.

    Computed here from the padded arrays directly, without touching `g`, because
    an error inside `g` would otherwise be invisible: every objective in the
    phase is expressed in those coordinates, so they would all be wrong together.
    """
    trainer = _trainer(env, "l4_tb")
    _, batch = _batch(trainer, env)

    mask = batch.valid.to(batch.log_pf.dtype)
    by_hand = (
        batch.log_z
        + (batch.log_pf * mask).sum(dim=1)
        - batch.log_reward
        - (batch.log_pb * mask).sum(dim=1)
    )
    assert torch.allclose(tb_residual(batch), by_hand, atol=1e-5)


def test_the_led_boundary_is_zero_and_the_balance_boundary_is_log_r(env):
    """LED carries the reward in `Σ φ`, so its terminal flow is pinned to 0.
    Telescoping its Eq. 4 over a trajectory forces exactly that; putting `log R`
    there instead would count the reward twice."""
    trainer = _trainer(env, "l6_led")
    _, batch = _batch(trainer, env)

    g_balance = batch.g(batch.log_reward)
    g_led = batch.g(torch.zeros_like(batch.log_reward))
    at = batch.n_trans.view(-1, 1)
    assert torch.allclose(g_balance.gather(1, at).squeeze(1) - g_led.gather(1, at).squeeze(1),
                          batch.log_reward, atol=1e-5)
    assert torch.allclose(g_balance[:, 0], g_led[:, 0])   # both start at log Z


def test_backward_probabilities_come_from_phase_2s_in_degrees(env):
    """Decision 18. An atom another selected atom references has no parent to be
    removed to, so the enumerated in-degree *is* the removable count."""
    trainer = _trainer(env, "l4_tb")
    traj, batch = _batch(trainer, env)

    rows, _, _, children = traj.transitions()
    expected = np.zeros(len(traj))
    np.add.at(expected, rows, -np.log(env.graph.indegree[children]))
    got = (batch.log_pb * batch.valid.to(batch.log_pb.dtype)).sum(dim=1).numpy()
    assert np.allclose(got, expected, atol=1e-5)


# -- the objectives ---------------------------------------------------------


def test_subtb_collapses_to_db_at_small_lambda_and_to_tb_at_the_far_end(env):
    """SubTB(λ) frames DB and TB as "opposite ends of a bias-variance tradeoff"
    (ICML 2023). Both ends have a known closed form here, which is the cheapest
    check that the weighting indexes sub-trajectory **length** rather than
    position — an error a loss curve would never reveal."""
    from graft.setgen.learners.l5_subtb import subtb_loss

    trainer = _trainer(env, "l5_subtb")
    _, batch = _batch(trainer, env)
    g = batch.g(batch.log_reward).detach()
    mask = batch.valid.to(g.dtype)

    # λ → 0: only the adjacent (i, i+1) pairs survive, which is DB
    db = ((g[:, :-1] - g[:, 1:]) ** 2 * mask).sum(dim=1) / mask.sum(dim=1)
    trainer.spec = trainer.spec.replace(subtb_lambda=1e-8)
    assert torch.allclose(subtb_loss(batch, trainer, 0), db.mean(), atol=1e-4)

    # the (0, L) pair is TB, and it is the longest pair in the sum
    tb = (g[:, 0] - g.gather(1, batch.n_trans.view(-1, 1)).squeeze(1)) ** 2
    assert torch.allclose(tb_residual(batch).detach() ** 2, tb, atol=1e-5)


def test_the_led_decomposition_loss_is_unbiased_under_dropout(env):
    """LED Eq. 5 uses dropout as its variance regulariser. Kept transitions are
    rescaled by `1/(1-p)`; without that the potential learns an energy biased low
    by exactly `p`, and terminal consistency would fail for a reason that has
    nothing to do with the environment."""
    trainer = _trainer(env, "l6_led")
    _, batch = _batch(trainer, env, n=64)
    phi = trainer.compute_potential(batch).detach()

    rng = np.random.default_rng(4)
    kept = batch.valid.to(phi.dtype)
    full = (phi * kept).sum(dim=1)

    rescaled, naive = [], []
    for _ in range(3000):
        keep = torch.as_tensor(rng.random(tuple(phi.shape)) >= 0.10, dtype=phi.dtype)
        rescaled.append((phi * kept * keep / 0.90).sum(dim=1))
        naive.append((phi * kept * keep).sum(dim=1))

    scale = full.abs().max().clamp(min=1.0)
    assert torch.allclose(torch.stack(rescaled).mean(dim=0), full, atol=0.05 * scale)
    # ...and the whole point of the rescaling: without it the estimate is short
    # by exactly p, so the potential would learn an energy 10% too small and
    # terminal consistency would fail for a reason unrelated to the environment.
    assert torch.allclose(
        torch.stack(naive).mean(dim=0), 0.90 * full, atol=0.05 * scale
    )
    assert float(decomposition_loss(phi, batch, 0.0, rng)) == pytest.approx(
        float(((full - batch.energy()) ** 2).mean()), rel=1e-5
    )


def test_consistency_is_per_trajectory_and_fail_is_not_pooled(env):
    """G10, decision 13. LED decomposes a *trajectory*; one terminal is reached
    by many paths, so a per-terminal reading averages over the object being
    measured. And `ℰ(FAIL) = -log r_fail = 13.8` sits outside the valid range, so
    pooling it lets a handful of trajectories own a p95 over thousands."""
    trainer = _trainer(env, "l6_led")
    _, batch = _batch(trainer, env, n=64)
    error, is_fail = consistency_error(trainer.compute_potential(batch), batch)

    assert error.shape == (batch.n,)              # per trajectory, not per terminal
    assert is_fail.shape == (batch.n,)
    assert env.log_r_range > 0.1                  # the guard does not fire here
    summed = (trainer.compute_potential(batch) * batch.valid).sum(dim=1).detach()
    assert math.isclose(
        float(error[0]),
        abs(float(summed[0]) + float(batch.log_reward[0])) / env.log_r_range,
        rel_tol=1e-4,
    )


def test_the_range_guard_excludes_rather_than_amplifies(env):
    """Decision 13's `range < 0.1` exclusion. Dividing by a near-zero range turns
    a tolerance into noise amplification; an instance whose valid terminals all
    carry the same reward simply has no scale to normalise by."""
    trainer = _trainer(env, "l6_led")
    _, batch = _batch(trainer, env, n=16)
    batch.log_r_range = 0.0
    error, _ = consistency_error(trainer.compute_potential(batch), batch)
    assert torch.isnan(error).all()


def test_the_intrinsic_reward_pays_only_for_progress(env):
    """Decision 19. `r = c_t·‖max(Δd, 0)‖₁` — a transition that *increases* a
    deficit has not made progress, and rewarding `|Δd|` would pay for churn."""
    trainer = _trainer(env, "gaflownet")
    _, batch = _batch(trainer, env)

    r = intrinsic_reward(batch, 0.1)
    assert torch.all(r >= 0.0)
    assert torch.all(r[~batch.valid] == 0.0)
    # STOP changes no set, so the terminating slot earns nothing
    at = (batch.n_trans - 1).view(-1, 1)
    assert torch.allclose(
        batch.delta.gather(1, at.unsqueeze(2).expand(-1, -1, batch.delta.shape[2])),
        torch.zeros(batch.n, 1, batch.delta.shape[2]),
        atol=1e-6,
    )


def test_grpo_standardises_within_the_group_and_survives_a_constant(env):
    """A group in which every trajectory earns the same return has no relative
    information; the advantage must be 0, not NaN."""
    assert torch.allclose(
        group_advantage(torch.full((8,), 3.0)), torch.zeros(8), atol=1e-6
    )
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = group_advantage(values)
    assert abs(float(adv.mean())) < 1e-5
    assert float(adv.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


# -- the comparison is fair -------------------------------------------------


def test_l6_and_l7_differ_by_delta_d_and_nothing_else(env):
    """Fix F11 and decision 19a. Identical loss function, identical parameter
    shapes, one boolean apart — which is what makes their capacity match exact
    rather than "within 1%"."""
    l6, l7 = build_arm("l6_led"), build_arm("l7_checker_led")
    assert l6.delta_d is False and l7.delta_d is True
    assert (l6.needs_flow, l6.needs_potential) == (l7.needs_flow, l7.needs_potential)

    t6, t7 = _trainer(env, "l6_led"), _trainer(env, "l7_checker_led")
    assert t6.capacity == t7.capacity
    assert t6.featurizers[0].delta_d is False
    assert t7.featurizers[0].delta_d is True


def test_gaflownet_never_sees_delta_d_in_its_policy(env):
    """G11. R2 had it policy-visible, which would have made the required control
    *GAFlowNet plus L7's mechanism* — isolating nothing, invisibly."""
    trainer = _trainer(env, "gaflownet")
    assert trainer.arm.delta_d is False
    assert trainer.featurizers[0].delta_d is False
    assert trainer.featurizers[0]._delta_table is None
    # ...and yet Δd reaches the loss
    _, batch = _batch(trainer, env)
    assert float(batch.delta.abs().sum()) > 0.0


def test_delta_d_routing_matches_decision_19a():
    """The routing table, read as a table: policy-visible for L7 and L7b only."""
    visible = {name for name, cfg in ARMS.items() if cfg.get("delta_d")}
    assert visible == {"l7_checker_led", "l7b_aux"}


def test_every_arm_shares_one_training_protocol(env):
    """Exit criterion 9. A per-arm ε, lr or batch size turns a comparison of
    objectives into a comparison of tuning effort."""
    spec = TrainSpec(n_trajectories=64, checkpoints=1)
    protocols = {
        tuple(sorted(Trainer(build_arm(name), [env], spec).spec.shared_protocol().items()))
        for name in ARMS
    }
    assert len(protocols) == 1


def test_the_seed_set_is_the_frozen_one():
    """Exit criterion 10, `CLAUDE.md` §6. **[EVIDENCE]** the ACL 2018 protocol
    requires multiple seeds; the set is frozen so no arm gets a different one."""
    assert SEEDS == (13, 42, 7)


def test_training_meters_nothing(env):
    """G9, decision 16. `checker_budget = 32` is a per-query *inference* budget;
    a trainer that opened a query scope would exhaust it in the first epoch and
    would be metering the wrong axis."""
    trainer = _trainer(env, "l4_tb")
    assert trainer.ledger is None
    trainer.ledger = object()
    with pytest.raises(RuntimeError, match="ledger=None"):
        trainer.train()


def test_the_intrinsic_coefficient_reaches_exactly_zero(env):
    """Exit criterion 14, decision 19b. Without the decay GAFlowNet's target is
    `R + r` rather than `R`, and its exact TV is measured against a distribution
    it never aimed at — making "L7 beats GAFlowNet" an artefact."""
    trainer = _trainer(env, "gaflownet", n=1024)
    assert trainer.intrinsic_coefficient(0) == pytest.approx(0.1)
    assert trainer.intrinsic_coefficient(math.ceil(0.8 * 1024)) == 0.0
    assert trainer.intrinsic_coefficient(1024) == 0.0
    # exactly zero for the final 20%, not merely small
    assert all(trainer.intrinsic_coefficient(t) == 0.0 for t in range(820, 1025, 5))

    log = trainer.train()
    assert log.final_c_t == 0.0
    assert log.c_t[0] > 0.0, "the schedule must actually start above zero"

    # ...and no other arm is given one
    assert _trainer(env, "l4_tb").intrinsic_coefficient(0) == 0.0


def test_every_arm_trains_and_moves(env):
    """The machinery works before anything is claimed (build step 5). Loss must
    be finite for all nine arms, and gradients must reach the policy."""
    for name in ARMS:
        trainer = _trainer(env, name, n=128)
        before = [p.detach().clone() for p in trainer.policy.parameters()]
        log = trainer.train()
        assert math.isfinite(log.final_loss), name
        assert math.isfinite(log.final_tv), name
        moved = any(
            not torch.equal(a, b)
            for a, b in zip(before, trainer.policy.parameters())
        )
        assert moved, f"{name} produced no gradient into the policy"


def test_the_two_families_partition_the_roster():
    """Decision 12's metric routing. One table implying three methods failed at a
    task two of them never attempted is the error this split prevents."""
    assert set(FLOW_FAMILY) | set(SUPERVISED_FAMILY) == set(ARMS)
    assert not set(FLOW_FAMILY) & set(SUPERVISED_FAMILY)
    assert len(ARMS) == 9, "decision 1 ruled Option B: nine evaluated arms"


def test_the_supervised_arms_learn_the_gold_set(env):
    """L1 and L2 have a target the flow arms do not, and it must be reachable:
    the gold set has to be constructible under the closure rule and has to be
    able to STOP, or the supervision is for a set `H` rejects."""
    trainer = _trainer(env, "l1_supervised")
    feat = trainer.featurizers[0]
    canonical, _ = feat.gold_path(None)
    sampled, actions = feat.gold_path(np.random.default_rng(2))

    assert set(env.graph.atoms_of(int(canonical[-1]))) == set(env.instance.gold.atoms)
    assert int(canonical[-1]) == int(sampled[-1]), "same set, different order"
    assert bool(env.graph.stop_allowed[int(canonical[-1])])
    assert len(actions) == len(env.instance.gold.atoms)
