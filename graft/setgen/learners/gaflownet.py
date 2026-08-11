"""GAFlowNet — augmented trajectory balance. **The second required control.**

**[EVIDENCE]** *Generative Augmented Flow Networks* (ICLR 2023). Intermediate
rewards ``r(s→s′)`` are added to the flow network as *augmented* flow, and the
trajectory balance condition becomes

.. code-block:: text

    Z_θ Π_t P_F(s_{t+1}|s_t)
        = R(x) Π_t P_B(s_t|s_{t+1}) Π_t ( 1 + r(s_t→s_{t+1}) / F(s_{t+1}) )

so the loss is TB's residual minus ``Σ_t log(1 + r_t / F(s_{t+1}))``. The
``r/F(s′)`` term is why this arm needs a ``StateFlowHead`` even though its base
objective is TB and TB does not (decision 27, G12).

**Why it is a required control and not an optional extra.** Plan §4.5.4 makes it
one, and the reason is sharp: "an intermediate signal improves credit
assignment" is *already published*. Beating L6 while losing to GAFlowNet would
mean Contribution 3's novelty claim collapses to a result GAFlowNet has. Exit
criterion 26 therefore requires L7 to beat **both**.

**The intrinsic reward, decision 19.** ``r(s→s′) = c_t · ‖max(Δd, 0)‖₁`` — the
positive part of the deficit *reduction* this transition achieves, so the control
receives the same signal L7 receives, through the channel GAFlowNet's own paper
uses.

**``Δd`` reaches the loss and never the policy** (G11, decision 19a). This arm is
built with ``delta_d=False``, so its featurisation is L6's exactly. R2 had it
policy-visible, which would have made it *GAFlowNet plus L7's mechanism* — a
control that isolates nothing, and the failure would have been invisible in every
number the run produced.

**``c_t → 0``, and the claim that rests on it** (decision 19b). Theorem 1's
unbiasedness needs both ``c_t = 0`` and ``L(θ) = 0``. The schedule
``c_t = c₀·max(0, 1 − t/0.8N)`` delivers the first exactly, for the final 20% of
training; no finite budget delivers the second. So the final augmented-TB loss is
reported and the write-up calls this a **finite-budget empirical control**, never
a provably unbiased sampler. Without the decay its target is ``R + r`` rather
than ``R``, and its exact TV would be measured against a distribution it never
aimed at — making "L7 beats GAFlowNet" an artefact. Criterion 14 checks it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["augmented_tb_loss", "intrinsic_reward"]

_LOG_ZERO = -1e30


def intrinsic_reward(batch: "Batch", c_t: float) -> torch.Tensor:
    """``r(s→s′) = c_t · ‖max(Δd, 0)‖₁`` per transition, ``[n, L]``.

    Only the **positive** part: a transition that increases a deficit has not
    made progress, and rewarding ``|Δd|`` would pay for churn. The terminating
    transition changes no set, so its ``Δd`` is zero by construction and it earns
    nothing — the same exclusion Phase-2 G5 applies when measuring ``Δd`` density.
    """
    positive = batch.delta.clamp(min=0.0).sum(dim=2)
    return c_t * positive * batch.valid.to(positive.dtype)


def augmented_tb_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    from graft.setgen.learners.l4_tb import tb_residual

    c_t = trainer.intrinsic_coefficient(spent)
    residual = tb_residual(batch)
    if c_t > 0.0:
        r = intrinsic_reward(batch, c_t)
        # log(1 + r/F(s′)) as softplus(log r − log F(s′)): F is held in log space
        # and can be large or tiny, so forming the ratio directly would overflow
        # at one end and underflow to a silent zero at the other.
        log_r = torch.where(
            r > 0.0, torch.log(r.clamp(min=1e-30)), torch.full_like(r, _LOG_ZERO)
        )
        log_flow_child = batch.flow_raw[:, 1:]
        term = torch.nn.functional.softplus(log_r - log_flow_child)
        residual = residual - (term * batch.valid.to(term.dtype)).sum(dim=1)
    return (residual**2).mean()
