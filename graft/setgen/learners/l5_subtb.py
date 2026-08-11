"""L5 — SubTB(λ).

**[EVIDENCE]** *Learning GFlowNets from Partial Episodes for Improved Convergence
and Stability* (ICML 2023). The objective applies the balance condition to every
**sub-trajectory** rather than only to the whole one, geometrically weighted by
length:

.. code-block:: text

    L_λ(τ) = Σ_{i<j} λ^{j−i} ℓ(i, j)  /  Σ_{i<j} λ^{j−i}

    ℓ(i, j) = ( log F(s_i) + Σ_{i≤k<j} log P_F  −  log F(s_j) − Σ log P_B )²

which the paper frames as interpolating between DB (``λ → 0``, every adjacent
pair) and TB (``λ → 1``, the whole trajectory) — the two ends of a bias-variance
tradeoff. In this module's coordinates ``ℓ(i, j) = (g[i] − g[j])²``.

**Boundaries.** ``log F(s_0) = log Z_θ`` and ``log F(s_L) = log R(x)``, both
supplied by the batch; interior flows come from ``StateFlowHead``, which is why
L5 is the first arm that needs one (decision 27).

**Why this arm is a load-bearing baseline and not a formality.** Plan §7 risk 10
names sparse-reward instability as a live risk here — hard-validity gating
concentrates reward on few terminals and near-zero flows destabilise log-space
losses — and cites SubTB as one of the two published responses to exactly that.
If L5 does not beat L4 on this environment, the reward landscape is milder than
the risk register assumes, and that is worth knowing before Gate 2 is read.

``λ`` is **not in §6**. See ``TrainSpec.subtb_lambda``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["subtb_loss"]


def subtb_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    lam = trainer.spec.subtb_lambda
    g = batch.g(batch.log_reward)                       # [n, L+1]

    residual = (g.unsqueeze(2) - g.unsqueeze(1)) ** 2    # [n, L+1, L+1]
    mask = batch.pair_mask()
    pos = batch.positions
    length = (pos.view(1, 1, -1) - pos.view(1, -1, 1)).clamp(min=0)
    weight = torch.where(
        mask,
        torch.pow(
            torch.as_tensor(lam, dtype=g.dtype, device=g.device), length.to(g.dtype)
        ),
        torch.zeros((), dtype=g.dtype, device=g.device),
    )

    # Normalised per trajectory, not per batch: a long rollout has O(L²)
    # sub-trajectories and a short one has few, so a batch-level normaliser would
    # weight trajectories by their length — which is a preference over set sizes
    # smuggled in as an implementation detail, and ``size`` is already a term in
    # ``U`` with a frozen weight.
    total = (weight * residual).sum(dim=(1, 2))
    norm = weight.sum(dim=(1, 2)).clamp(min=1e-12)
    return (total / norm).mean()
