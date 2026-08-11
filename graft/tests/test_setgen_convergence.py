"""Build step 5's "done when", and exit criterion 4 — the machinery converges.

**These are slower than the rest of the suite and that is not incidental.** They
are the only tests that assert a *learned* quantity, and the whole reason build
order puts steps 2–5 before any comparison is that a broken sampler or a broken
adapter produces plausible loss curves and meaningless numbers. The check has to
train something.

They run on ``tiny_instance()``, where ``log Z`` has a known closed form from
``Target`` — the one place in the project where the conditional partition
function can be checked against an answer rather than against itself (G6).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graft.setgen.learners import build_arm
from graft.setgen.trainer import Environment, TrainSpec, Trainer
from graft.synth.lattice import tiny_instance

#: Decision 6's threshold, signed **before** calibration.  Reproduced here rather
#: than imported so a change to one has to be a change to both — §6 is the single
#: source, and a test that silently tracked it would never notice it moving.
SANITY_TV = 0.10

#: Decision 25 — ``logZ_θ`` within 1% relative.  **[ANALYSIS]** engineering; it is
#: a machinery check, not a Gate-2 metric.
LOGZ_TOLERANCE = 0.01

#: Enough for the machinery to converge on a 17-state lattice, and small enough
#: to belong in a test suite.  Measured: L4 reaches TV ~0.028 and ``logZ`` within
#: 0.4% here; at 200,000 it reaches 0.0015 and 0.008%.
BUDGET = 50_000


@pytest.fixture(scope="module")
def env():
    return Environment(tiny_instance())


def _train(env, name, seed=13, n=BUDGET):
    trainer = Trainer(
        build_arm(name), [env], TrainSpec(n_trajectories=n, checkpoints=5, seed=seed)
    )
    return trainer, trainer.train()


@pytest.mark.parametrize("name", ["l4_tb", "l5_subtb"])
def test_the_machinery_works_before_anything_is_claimed(env, name):
    """Build step 5's "done when": L4 and L5 train end to end on
    `tiny_instance()` and reach decision 6's TV threshold.

    Distinct from exit criterion 15, which asks the same of the **main** suite at
    the adopted `(N, β)`, and from criterion 12, which is the tuning-suite
    calibration check that selects them.
    """
    _, log = _train(env, name)
    assert log.final_tv <= SANITY_TV, f"{name} reached only TV={log.final_tv:.4f}"
    assert log.tv_mean[0] > log.final_tv, "TV did not move at all"
    assert np.isfinite(log.losses).all()


def test_logz_converges_to_the_known_partition_function(env):
    """Exit criterion 4, decision 25.

    `log F(s_0) = log Z` is an identity, so the head is not learning a free
    parameter — it is learning a quantity `Target` already knows exactly. If it
    converged somewhere else, trajectory balance would still be satisfiable by
    rescaling `P_F`, and every TV would be wrong in a way no loss curve shows.
    """
    trainer, log = _train(env, "l4_tb")
    with torch.no_grad():
        got = float(trainer.logz(trainer.featurizers[0].instance_repr()))
    true = float(np.log(env.target.z))
    assert abs(got - true) / abs(true) <= LOGZ_TOLERANCE, f"{got:.4f} vs {true:.4f}"
    assert log.final_tv <= SANITY_TV


def test_the_flow_arms_all_converge(env):
    """The LED arms and GAFlowNet reach the same threshold, so a Gate-2 result is
    a comparison between converged learners rather than between one that trained
    and one that did not.

    **Measured while writing this test, and it matters for the calibration
    gate:** at 20,000 trajectories **L7 is the slowest of the six** (0.167,
    against GAFlowNet's 0.097 and L6's 0.109), and it is still last of the LED
    group at 50,000. Decision 4 sets `N` from L4 and L5 alone, deliberately, so
    the budget is not selected on the proposed method's results — which means an
    `N` satisfying decision 6 could land where L7 has not converged, and Gate 2
    would return a negative verdict on Contribution 3 that is about budget rather
    than mechanism.

    One seed, one 17-state lattice: enough to watch the risk, not enough to act
    on it. `PHASE3_DECISIONS.md` §2.3 carries the options and the reason none of
    them may be applied after step 6 has run.
    """
    for name in ("l6_led", "l7_checker_led", "gaflownet"):
        _, log = _train(env, name)
        assert log.final_tv <= SANITY_TV, f"{name} reached only {log.final_tv:.4f}"


def test_gaflownets_intrinsic_reward_is_gone_by_the_end(env):
    """Exit criterion 14 on a real run, not only on the schedule: with `c_t > 0`
    at the end, the arm's target is `R + r` and its TV is measured against a
    distribution it never aimed at."""
    _, log = _train(env, "gaflownet", n=20_000)
    assert log.final_c_t == 0.0
    assert max(log.c_t) > 0.0


def test_the_supervised_arms_concentrate_on_the_gold_set(env):
    """L1 and L2 are not samplers, so exact TV is descriptive for them (decision
    12). What *is* diagnostic is that they put their mass where they were taught
    to: the gold terminal's probability must rise well above uniform."""
    from graft.synth.exact import policy_distribution

    position = {
        int(t): i for i, t in enumerate(env.graph.terminal_ix.tolist())
    }
    gold = position[env.graph.state_of(env.instance.gold.atoms)]

    for name in ("l1_supervised", "l2_imitation"):
        trainer, _ = _train(env, name, n=20_000)
        with torch.no_grad():
            p = policy_distribution(trainer.featurizers[0], env.graph)
        assert p[gold] > 0.5, f"{name} put only {p[gold]:.3f} on the gold set"
