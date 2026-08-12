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

**The ``Δd`` block is present in both featurisations and zeroed for L6**, so the
two arms have identical parameter *shapes* and exit criterion 5 can assert the
gating in both directions: zeroing the block must leave an L6 forward pass
numerically unchanged and must change an L7 one. One direction alone proves
nothing — without the first ``Δd`` may be leaking to the control, without the
second L7 may be ignoring it.

**Identical shapes are not a capacity match, and this docstring used to claim
they were** ("the capacity match required by decision 11 is 0.00%, not within
1%"). L6's weights on the zeroed block never receive a gradient, so at equal
width the *control* carried 1.46% less trainable capacity than the arm it is
controlling for — the direction that flatters this file. Decision 11 matches
live parameters and widens L6; the claim retired with it.
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
