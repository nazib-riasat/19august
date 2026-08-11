"""L7 — the proposed method: checker-conditioned LED.

**This file is deliberately three lines of arithmetic.** Fix F11 defines L7 as
"L6 plus ``Δd`` as input features to ``φ_θ`` and the policy, **and nothing
else**", and the honest way to implement that is to reuse L6's loss verbatim and
let the entire difference live in one boolean on the featurizer
(``delta_d=True``, decision 19a). Any additional term written here would be a
second difference, and Gate 2 would no longer isolate the mechanism it is
designed to test.

**[HYPOTHESIS]** — and labelled so everywhere it appears. Contribution 3 is a
hypothesis this project tests, not a result any paper establishes. ``CLAUDE.md``
§1 marks C3 "**High — explicitly a hypothesis**", and §8 places Gate 2 as the
cheapest place in the project to learn that it is unsupported. A negative result
here is a designed outcome, not a failure of the phase.

**What ``Δd`` is, and why it is not already in the input.** ``d(s)`` is the
checker's six-component deficit vector at ``s`` — anchor, value, time, source,
binding, closure. It is already a state feature for every arm. ``Δd = d(s) −
d(s′)`` is a property of the *transition*: how much of the outstanding proof
obligation this particular ADD discharges. That is the per-action resolution the
policy and the potential see under L7 and not under L6.

**The comparison is exact, not approximate.** L6 and L7 have identical parameter
shapes — the ``Δd`` block of ``action_repr`` exists in both and is zeroed for L6
— so the capacity match required by decision 11 is 0.00%, not "within 1%". Exit
criterion 5 asserts the gating in both directions: zeroing the block must leave
an L6 forward pass numerically unchanged and must change an L7 one. One
direction alone proves nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from graft.setgen.learners.l6_led import led_db_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["checker_led_loss"]


def checker_led_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    """LED-DB, unchanged. The mechanism is in the features, not in the loss."""
    return led_db_loss(batch, trainer, spent)
