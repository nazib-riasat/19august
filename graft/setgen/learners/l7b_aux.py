"""L7b — L7 plus a **forward-looking** auxiliary head (ablation).

**The head fix F11 retired, and why this one is different.** An earlier draft
predicted ``d(s)`` from ``h_s``. ``d(s)`` is already an encoder input, so that
head was a copy task: it can be solved perfectly without learning anything about
credit assignment, which is the entire claim of Contribution 3. Fix F11 demoted
it to an ablation *and required a different target*.

Decision 26 fixes that target: ``d(X(τ))`` — the deficit vector of the terminal
the rollout **actually reached**, supervised by the realised trajectory. From a
partial state ``s`` that is genuinely not in the input; it is a property of where
this policy tends to end up from here, which is the forward-looking quantity fix
F11 asked for.

.. code-block:: text

    L(θ) = L_LED-DB(θ)  +  λ_aux · mean_s ‖ d̂(s) − d(X(τ)) ‖²

MSE over the six components, ``λ_aux = 0.1`` **[recommended]**, head discarded at
evaluation — it shapes training and never enters the sampling distribution, so
L7b's TV is measured on the same object as every other arm's.

**Reported as an ablation, never as a substitute for L7** (exit criterion 20).
If L7b beats L7, the finding is "an auxiliary forward-looking target helps",
which is a different claim from Contribution 3 and must be written as one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from graft.setgen.learners.l6_led import led_db_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["checker_led_aux_loss", "auxiliary_loss"]


def auxiliary_loss(batch: "Batch") -> torch.Tensor:
    """``mean over visited states of ‖ d̂(s) − d(X(τ)) ‖²`` (decision 26).

    The target is broadcast along the trajectory: every visited state predicts
    the *same* terminal deficit, which is what makes it a statement about where
    the rollout went rather than about where it currently is.
    """
    if batch.aux_pred is None:  # pragma: no cover - defensive
        raise ValueError("L7b needs a deficit head; the arm was built without one")
    target = batch.terminal_deficit.unsqueeze(1).expand_as(batch.aux_pred)
    error = (batch.aux_pred - target) ** 2
    mask = batch.aux_mask.unsqueeze(2).to(error.dtype)
    return (error * mask).sum() / (mask.sum() * error.shape[2]).clamp(min=1.0)


def checker_led_aux_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    return led_db_loss(batch, trainer, spent) + trainer.spec.lambda_aux * auxiliary_loss(
        batch
    )
