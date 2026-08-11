"""L6 — LED-GFN, in its **LED-DB** form (decision 23b).

**[EVIDENCE]** *Learning Energy Decomposition for Partial Inference in
GFlowNets*. Two objectives, trained together, and it is worth being precise
about which is which because R4 got it wrong once already:

**1. The decomposition** (its Eq. 3 and Eq. 5). A learned potential
``φ_θ(s→s′)`` is required to sum along a trajectory to that trajectory's terminal
energy::

    Σ_{t} φ_θ(s_t → s_{t+1})  =  ℰ(x)  =  −log R(x)

trained by one squared loss with **dropout** as the variance regulariser. There
are **no** "two regulariser coefficients" — that was an assumed parameterisation
the paper does not use. Appendix C's values (decision 23a): potential learning
rate 0.001, dropout 0.10, ``N = 8`` decomposition iterations per round,
``B2 = B1``.

**2. The GFlowNet** (its Eq. 4), DB-style over ``log F̃(s)``, with the
decomposed potential supplying per-transition credit::

    ( log F̃(s) + log P_F(s′|s) − log F̃(s′) − log P_B(s|s′) + φ_θ(s→s′) )²

**Where the reward went.** Telescoping that residual over a whole trajectory
gives ``log F̃(s_0) + Σ log P_F − log F̃(x) − Σ log P_B + ℰ(x) = 0``, and matching
it against trajectory balance forces ``log F̃(x) = 0`` at **every** terminal —
valid stop or dead end alike. So LED's boundary is ``0``, not ``log R(x)``: the
reward is carried entirely by ``Σ φ``. That is the single line where LED differs
structurally from L4 and L5, and it is why ``Batch.g`` takes its terminal
boundary as an argument.

**The terminating transition is decomposed too.** ``STOP`` is an action in this
MDP, so ``P_F(STOP | x)`` appears in the balance conditions and needs its own
``φ`` slot, or the identity above does not close. A dead end terminates with
probability 1 and gets the same slot with ``log P_F = 0``.

**Terminal consistency is measurable here and is a hard constraint** (G10,
decision 13). ``ℰ(x)`` is known exactly for every terminal of the lattice, so
``|Σ φ − ℰ|`` is not a proxy — it is the quantity itself. Plan §4.5.4 requires
that L7's TV win not be bought by relaxing this regulariser, which is why
``consistency_error`` lives here rather than in L7's file: the control and the
proposed method are held to one implementation of one definition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["decomposition_loss", "led_db_loss", "consistency_error"]


def decomposition_loss(
    potential: torch.Tensor,
    batch: "Batch",
    dropout: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """LED's Eq. 5 — the potential's own loss, with dropout as the regulariser.

    Kept transitions are rescaled by ``1/(1 − p)`` so the masked sum is an
    unbiased estimate of the full one; without the rescaling, dropout would bias
    ``Σ φ`` low by exactly ``p`` and the potential would learn an energy that is
    systematically too small.

    ``rng`` is the trainer's, not a fresh one: the run's seed has to reach every
    stochastic component or two "identical" runs differ by their dropout masks.
    """
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    mask = batch.valid.to(potential.dtype)
    if dropout > 0.0:
        keep = torch.as_tensor(
            rng.random(tuple(potential.shape)) >= dropout,
            dtype=potential.dtype,
            device=potential.device,
        )
        mask = mask * keep / (1.0 - dropout)
    predicted = (potential * mask).sum(dim=1)
    return ((predicted - batch.energy()) ** 2).mean()


def led_db_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    """LED's Eq. 4, per transition, with ``log F̃`` pinned to ``0`` at terminals."""
    if batch.potential is None:  # pragma: no cover - defensive
        raise ValueError("LED-DB needs a potential; the arm was built without one")

    terminal = torch.zeros_like(batch.log_reward)
    g = batch.g(terminal)                                # [n, L+1]
    residual = g[:, :-1] - g[:, 1:] + batch.potential    # [n, L]

    mask = batch.valid.to(residual.dtype)
    # Mean over **real** transitions, not over the padded rectangle: padding
    # varies with the batch's longest rollout, so a rectangle mean would make the
    # loss depend on which trajectories happened to be sampled together.
    return (residual**2 * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def consistency_error(
    potential: torch.Tensor, batch: "Batch"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(normalised |Σφ − ℰ| per trajectory, is_fail)`` — decision 13's metric.

    Normalised by the **per-instance valid-terminal ``log R`` range**, which is
    the correction R3 forced: the per-terminal ``|log R|`` that preceded it
    permitted 0.0018 of absolute error at one terminal of an instance and 0.3927
    at another — a 216× swing inside a single environment, from a rule that reads
    as a single tolerance.

    The ``FAIL`` flag comes back with the errors rather than being folded in,
    because ``ℰ(FAIL) = −log r_fail = 13.8`` sits far outside the valid range and
    pooling it would let one trajectory dominate a p95 over thousands.
    """
    predicted = (potential * batch.valid.to(potential.dtype)).sum(dim=1)
    error = (predicted - batch.energy()).abs()
    if batch.log_r_range < 0.1:
        # Decision 13's exclusion guard: dividing by a near-zero range turns a
        # tolerance into noise amplification, and an instance whose valid
        # terminals all carry the same reward has no scale to normalise by.
        return torch.full_like(error, float("nan")), batch.is_fail
    return error / batch.log_r_range, batch.is_fail
