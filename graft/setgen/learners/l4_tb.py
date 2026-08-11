"""L4 — Trajectory Balance.

**[EVIDENCE]** *Trajectory Balance: Improved Credit Assignment in GFlowNets*
(NeurIPS 2022). The constraint is

.. code-block:: text

    Z_θ · Π_t P_F(s_{t+1} | s_t)  =  R(x) · Π_t P_B(s_t | s_{t+1})

and the loss is the squared log-residual. In this module's ``g`` coordinates
that residual is exactly ``g[0] − g[L]``, since ``g[0] = log Z_θ`` and
``g[L] = log R(x) − Σ log P_F + Σ log P_B``.

**GAFlowNet's unaugmented counterpart is this arm** (decision 19), which is why
the two share a coordinate system: their difference must be the augmentation and
nothing incidental.

``CLAUDE.md`` §4.1 rejected Flow Matching for this environment on the grounds
that TB and its descendants propagate credit better over long sequences, and
that 16 atoms with 16! orderings is FM's worst case. L4 is the concrete form of
that decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["tb_loss", "tb_residual"]


def tb_residual(batch: "Batch") -> torch.Tensor:
    """``log Z_θ + Σ log P_F − log R(x) − Σ log P_B`` per trajectory, ``[n]``.

    ``FAIL`` trajectories are included with ``log R = log r_fail``. That is the
    whole point of architecture fix F3: ``FAIL`` is a **member of the target's
    support**, so a policy that never reaches it is wrong in the same way as one
    that reaches it too often, and the loss has to be able to say so.
    """
    g = batch.g(batch.log_reward)
    return g[:, 0] - g.gather(1, batch.n_trans.view(-1, 1)).squeeze(1)


def tb_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    return (tb_residual(batch) ** 2).mean()
