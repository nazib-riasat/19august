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


def test_the_led_decomposition_loss_is_equation_5(env):
    """LED-GFN Eq. 5, evaluated independently:

        ℓ_LS(τ) = E_z [ ( ℰ(x)/T − Σ_t z_t·φ_t / C )² ],   C = Σ_t z_t

    The retired implementation used `(Σ z·φ/(1-p) − ℰ)²` — the *expected* keep
    rate instead of the realised count `C`, and no `1/T` on the energy. At
    `p = 0` the two still differ by `1/T²`, so a variable-length environment
    weights its trajectories differently under the two formulas; the previous
    test asserted the retired form and could not see it."""
    trainer = _trainer(env, "l6_led", dtype=torch.float64)
    _, batch = _batch(trainer, env, n=64)
    phi = trainer.compute_potential(batch).detach()
    valid = batch.valid.to(phi.dtype)
    length = valid.sum(dim=1)

    # λ = 1: no dropout, so z = valid and C = T. Eq. 5 collapses to (ℰ − Σφ)²/T².
    full = (phi * valid).sum(dim=1)
    assert float(decomposition_loss(phi, batch, 0.0, np.random.default_rng(4))) == (
        pytest.approx(float((((batch.energy() - full) / length) ** 2).mean()), rel=1e-9)
    )
    # ...and that is *not* the retired form, because T varies here.
    assert length.min() < length.max(), "a fixed-length batch cannot see the 1/T"
    assert float(decomposition_loss(phi, batch, 0.0, np.random.default_rng(4))) != (
        pytest.approx(float(((full - batch.energy()) ** 2).mean()), rel=1e-3)
    )

    # λ < 1: normalised by the realised C, so a draw that keeps k transitions is
    # compared against the same per-step scale as one that keeps all of them.
    rng = np.random.default_rng(4)
    keep = torch.as_tensor(rng.random(tuple(phi.shape)) >= 0.10, dtype=phi.dtype)
    z = valid * keep
    expected = (
        batch.energy() / length - (phi * z).sum(dim=1) / z.sum(dim=1).clamp(min=1.0)
    )
    scored = z.sum(dim=1) > 0
    assert float(decomposition_loss(phi, batch, 0.10, np.random.default_rng(4))) == (
        pytest.approx(float((expected[scored] ** 2).mean()), rel=1e-9)
    )


def test_redistribution_makes_the_led_boundary_an_identity(env):
    """LED-GFN Appendix B.1. The paper's reported LED-GFN redistributes the
    decomposition error `ℰ(x) − Σφ` uniformly over the trajectory's transitions;
    the correction-term variant it plots against is named LED-GFN*. A form with
    neither is a third thing the paper never runs, which is what decision 23b
    used to specify.

    After redistribution `Σφ̃ = ℰ(x)` exactly, so the telescoping identity LED-DB
    is derived from holds by construction rather than approximately."""
    from graft.setgen.learners.l6_led import redistribute

    trainer = _trainer(env, "l6_led", dtype=torch.float64)
    _, batch = _batch(trainer, env, n=64)
    phi = trainer.compute_potential(batch).detach()
    valid = batch.valid.to(phi.dtype)

    adjusted = redistribute(phi, batch)
    assert torch.allclose((adjusted * valid).sum(dim=1), batch.energy(), atol=1e-10)
    # the raw potential does not satisfy that at initialisation, so the test bites
    assert not torch.allclose((phi * valid).sum(dim=1), batch.energy(), atol=1e-6)
    # padding is untouched, and the adjustment is one constant per trajectory
    delta = (adjusted - phi)[valid.bool()]
    assert torch.allclose((adjusted - phi)[~valid.bool()], torch.zeros_like(delta[:1]))


def test_consistency_is_measured_on_the_raw_potential(env):
    """Decision 13 measures how well Eq. 5 trained the potential. Computed on the
    redistributed `φ̃` it would be identically 0 for every arm on every
    trajectory — a band that always passes and measures nothing, which would
    quietly void decision 15's hard constraint from plan §4.5.4."""
    from graft.setgen.learners.l6_led import redistribute

    trainer = _trainer(env, "l6_led", dtype=torch.float64)
    _, batch = _batch(trainer, env, n=32)
    phi = trainer.compute_potential(batch)

    raw, _ = consistency_error(phi, batch)
    adjusted, _ = consistency_error(redistribute(phi, batch), batch)
    assert torch.allclose(adjusted, torch.zeros_like(adjusted), atol=1e-10)
    assert float(raw.max()) > 1e-6


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


def test_augmented_tb_is_gaflownet_equation_4_not_its_own_algebra(env):
    """GAFlowNet (ICLR 2023) Eq. 4 puts the intrinsic term **beside** `P_B`:

        Z Π P_F = R(x) Π ( P_B(s_t|s_{t+1}) + r(s_t→s_{t+1})/F(s_{t+1}) )

    An earlier build dropped the `P_B` divisor and subtracted `log(1 + r/F)`,
    which implements an intrinsic reward of `P_B·r` rather than `r` — attenuated
    by the removable-atom count, and by more of it as the set grows. Every test
    it had checked `intrinsic_reward` or the implementation against itself, so
    the arm looked right at every point the suite examined.

    Evaluated here in float64 directly from Eq. 4, in the probability domain, so
    the check shares no algebra with the loss it is checking."""
    from graft.setgen.learners.gaflownet import augmented_tb_loss

    trainer = _trainer(env, "gaflownet", dtype=torch.float64)
    _, batch = _batch(trainer, env)
    c_t = trainer.intrinsic_coefficient(0)
    assert c_t == pytest.approx(trainer.spec.gafn_c0)

    mask = batch.valid.to(batch.log_pf.dtype)
    bracket = torch.log(
        torch.exp(batch.log_pb)
        + intrinsic_reward(batch, c_t) / torch.exp(batch.flow_raw[:, 1:])
    )
    residual = (
        batch.log_z
        + (batch.log_pf * mask).sum(dim=1)
        - batch.log_reward
        - (bracket * mask).sum(dim=1)
    )
    assert torch.allclose(
        augmented_tb_loss(batch, trainer, 0), (residual**2).mean(), rtol=1e-9
    )

    # ...and the retired form is genuinely different, so this test can fail.
    wrong = residual + (
        (bracket - torch.log1p(intrinsic_reward(batch, c_t)
                               / torch.exp(batch.flow_raw[:, 1:]))) * mask
    ).sum(dim=1)
    assert not torch.allclose(wrong, residual, rtol=1e-6)


def test_grpo_standardises_within_the_group_and_survives_a_constant(env):
    """A group in which every trajectory earns the same return has no relative
    information; the advantage must be 0, not NaN."""
    assert torch.allclose(
        group_advantage(torch.full((8,), 3.0), 8), torch.zeros(8), atol=1e-6
    )
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = group_advantage(values, 4)
    assert abs(float(adv.mean())) < 1e-5
    assert float(adv.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


def test_logz_trains_at_its_own_learning_rate(env):
    """**[EVIDENCE]** Trajectory Balance (NeurIPS 2022) §3: "we found it helpful
    to set a higher learning rate for Z than for the parameters of P_F and P_B."
    SubTB (ICML 2023) Appendix C, verbatim: "for Z, use a learning rate of 10x
    the learning rate for forward logits."

    An earlier build put the policy, `LogZHead` and the flow head in one Adam at
    `lr`, which is neither paper's protocol and was declared nowhere. **The
    mechanism ships; the multiplier is decision 30's `[recommended]` value and
    the default stays 1.0 until it is signed** — at 10x the head breaks decision
    25's tolerance on `tiny_instance()`, and moving a normative tolerance to
    accommodate a change made here would be the wrong direction. One multiplier
    for every arm either way, so decision 23's "no per-arm search" holds."""
    trainer = _trainer(env, "l5_subtb", logz_lr_mult=10.0)
    groups = trainer.optimiser.param_groups
    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(trainer.spec.lr * 10.0)
    assert groups[0]["lr"] == pytest.approx(trainer.spec.lr)

    # the fast group is exactly logZ, and every stepped parameter is still clipped
    logz_ids = {id(p) for p in trainer.logz.parameters()}
    assert {id(p) for p in groups[1]["params"]} == logz_ids
    assert logz_ids <= {id(p) for p in trainer.main_params}
    assert not (logz_ids & {id(p) for p in groups[0]["params"]})

    # the shipped default changes nothing until decision 30 is ruled
    assert TrainSpec(n_trajectories=32).logz_lr_mult == 1.0
    assert trainer.spec.shared_protocol()["logz_lr_mult"] == 10.0
    with pytest.raises(ValueError, match="logz_lr_mult"):
        TrainSpec(n_trajectories=32, logz_lr_mult=0.0)


def test_grpo_uses_the_architectures_group_of_eight_not_the_batch(env):
    """Architecture §3.2 freezes `G = 8` for L3; decision 23 freezes the batch at
    32 for every arm. Standardising over the batch made `G = 32` silently — four
    groups pooled into one baseline. It is a *stronger* L3 than the plan
    specifies rather than a weaker one, which is exactly why nothing looked
    wrong."""
    trainer = _trainer(env, "l3_grpo")
    assert trainer.spec.grpo_group == 8
    assert trainer.spec.batch_size == 32
    assert trainer.spec.shared_protocol()["grpo_group"] == 8

    # two groups with different levels: standardising per group removes the
    # level, standardising over the batch would not
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0])
    per_group = group_advantage(values, 4)
    assert torch.allclose(per_group[:4], per_group[4:], atol=1e-5)
    whole = group_advantage(values, 8)
    assert not torch.allclose(whole[:4], whole[4:], atol=1e-2)

    # a trailing partial group of one carries no relative information
    assert float(group_advantage(torch.tensor([1.0, 2.0, 9.0]), 2)[2]) == 0.0
    with pytest.raises(ValueError, match="grpo_group"):
        TrainSpec(n_trajectories=32, grpo_group=1)


# -- the comparison is fair -------------------------------------------------


def test_l6_and_l7_differ_by_delta_d_and_nothing_else(env):
    """Fix F11 and decision 19a: identical loss, identical shapes, one boolean
    apart. **At equal width that is a shape match and not a capacity match** —
    L6's weights on the zeroed block never train, so decision 11 widens L6 and
    matches live parameters instead (see `test_setgen_gate2.py`)."""
    l6, l7 = build_arm("l6_led"), build_arm("l7_checker_led")
    assert l6.delta_d is False and l7.delta_d is True
    assert (l6.needs_flow, l6.needs_potential) == (l7.needs_flow, l7.needs_potential)

    t6, t7 = _trainer(env, "l6_led"), _trainer(env, "l7_checker_led")
    assert t6.capacity == t7.capacity, "same width, same shapes"
    assert t6.live_capacity < t7.live_capacity, "...and that is exactly the problem"
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
