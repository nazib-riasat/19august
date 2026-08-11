"""L3 — GRPO (group-relative policy optimisation).

The policy-gradient control. Plan §5.1 requires it under the same matched-policy
condition as every other arm, and records the caveat: *Search-R1 is arXiv-only
and trains an LLM search agent — implementation inspiration, not venue-backed
evidence for this task.* So this arm is **[ANALYSIS]** as adapted here, and the
write-up should say so rather than borrow an LLM result's authority.

.. code-block:: text

    A_i = ( G_i − mean(G) ) / ( std(G) + ε )        within the sampled group
    L   = − mean_i  A_i · Σ_t log P_F(a_t | s_t)

**Two declared departures, both forced by this reward's shape.**

*No KL to a reference policy.* GRPO's KL term regularises toward a pretrained
reference. There is no pretrained policy here — every arm starts from random
initialisation — so there is nothing to regularise toward, and inventing one
would give L3 a component no other arm has.

*The group return is ``log R``, not ``R``.* On this environment ``R`` spans
``r_fail = 1e-6`` to ``exp(β·U_max) ≈ 2.6e3`` — nine orders of magnitude. A
group-normalised advantage over raw ``R`` is a one-hot on the group maximum for
any realistic group size: the standard deviation is set by the single largest
member, so every other trajectory receives an advantage of approximately
``−1/G``, and the method degenerates into "imitate the best sample" regardless of
how good the rest were. ``log R`` is the same monotone ordering on a scale the
normalisation can resolve. **This is a departure, it is recorded here, and it
should appear in the write-up** — it makes L3 a stronger baseline than the naive
implementation, not a weaker one, which is the direction a departure in a
baseline should run.

Exact TV is descriptive for this arm too (decision 12): GRPO maximises expected
return and has no reward-proportionality objective, so its TV measures its
distance from a target it was never asked to match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["grpo_loss", "group_advantage"]

_EPS = 1e-6


def group_advantage(returns: torch.Tensor) -> torch.Tensor:
    """Standardise within the sampled group. Constant returns give zero, not NaN."""
    if returns.numel() < 2:
        return torch.zeros_like(returns)
    centred = returns - returns.mean()
    return centred / (returns.std(unbiased=False) + _EPS)


def grpo_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    advantage = group_advantage(batch.log_reward).detach()
    log_pf = (batch.log_pf * batch.valid.to(batch.log_pf.dtype)).sum(dim=1)
    return -(advantage * log_pf).mean()
