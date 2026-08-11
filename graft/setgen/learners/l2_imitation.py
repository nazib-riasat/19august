"""L2 — canonical set imitation.

Plan §5.1: *"Tests whether distribution training adds anything over imitating one
gold set."* That is the whole question this arm exists to answer, and it is a
fair one — if imitating a single canonical proof matches the flow family on the
metrics that matter downstream, the flow machinery is not earning its cost.

**The difference from L1 is one argument and it is the point.** L1 samples a
fresh closure-legal ordering per example and so supervises the gold *set*; L2
walks the canonical order — ascending pool position — every time, and so imitates
one *sequence*. Same loss, same gold terminal, different notion of what is being
imitated.

**Same metric routing as L1** (decision 12): exact TV is descriptive here, in a
separate table, because a policy trained to reproduce one set has no mechanism
for placing mass on the others and its TV says so without saying anything about
the objective's quality at the task it was given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from graft.setgen.learners.l1_supervised import supervised_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Trainer

__all__ = ["imitation_loss"]


def imitation_loss(
    batch: tuple[torch.Tensor, torch.Tensor], trainer: "Trainer", spent: int
) -> torch.Tensor:
    """Identical arithmetic to L1; the canonical ordering is set by the trainer."""
    return supervised_loss(batch, trainer, spent)
