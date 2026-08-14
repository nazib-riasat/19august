"""P6.5 — D1–D4 and temperature calibration (decisions 6, 8, 9).

**Four decoders, not eight heads.**  The eight-head design is dead
(`CLAUDE.md` §4.1) and the reason is worth repeating where the replacement
lives: duplicate, conflict and supersession all depend on the *same* claim pair
and the *same* time interval, so predicting them independently discards their
mutual exclusivity and produces contradictory outputs.  D2 is one four-way
softmax, and **D2 grouped-vs-split is the one ablation that matters**, kept in
the harness rather than argued about here.

**D1's action space is open-world.**  **[EVIDENCE]** Learn to Not Link
(Findings ACL 2023) partitions unlinkable mentions into Missing-Entity vs
Non-Entity; a single NIL head conflates *a real new entity* (must create a node)
with *a non-entity phrase* (must create nothing), and in a growing memory graph
the first is the common case.  ConEL-2 / CREL (CIKM 2022) supplies the
personal-entity convention — "my car" creates.  **`DEFER` is [ANALYSIS]**, no
published precedent, which is exactly why the plan ablates it on/off.

**Scoring candidates, not classifying into a fixed vocabulary.**  D1 emits one
logit per *candidate* plus three for the non-linking actions, so the head is
independent of how many entities exist — a fixed output layer would have to be
resized every time the graph grew, and could not represent "link to this
specific entity" at all.

**Calibration is fitted on dev only** (decision 8).  **[EVIDENCE]** Guo et al.
(ICML 2017): temperature scaling is the cheap, reliable fix for modern-network
overconfidence.  It matters here beyond reporting, because decision 9 gates D3/D4
commits on a *confidence floor* — an uncalibrated floor is a number with no
meaning, so the calibration is a prerequisite for the commit rule rather than a
nicety.

**Class weights, never resampling** (decision 6, contract item 6): resampling
would make the quoted class balance unreproducible, and D2's rare classes are the
contribution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from graft.graphbuild.items import D1_ACTIONS, D2_LABELS
from graft.graphbuild.pins import TRAINING

__all__ = [
    "D1Decoder",
    "D2Decoder",
    "TypedDecoder",
    "TemperatureScaler",
    "brier_score",
    "expected_calibration_error",
    "class_weights",
]

#: The three actions that are not "link to candidate *i*".  Order is fixed and
#: exported, because a positional literal elsewhere would be invisible if it moved.
NON_LINK_ACTIONS: tuple[str, ...] = ("CREATE_NEW_ENTITY", "NON_ENTITY", "DEFER")


class D1Decoder(nn.Module):
    """Mention → ``LINK_EXISTING(id)`` / ``CREATE_NEW_ENTITY`` / ``NON_ENTITY`` / ``DEFER``.

    Scores each candidate against the mention with a bilinear-ish MLP and
    concatenates three action logits, so the output width is ``len(candidates) +
    3`` and varies per item.  **The empty-candidate case is not special-cased**:
    it yields a width-3 output, which is precisely the legal cold start (G4), and
    training on such items is what stops the head learning that linking is always
    possible.
    """

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or TRAINING["hidden"])
        self.pair = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.ReLU(), nn.Dropout(TRAINING["dropout"]), nn.Linear(hidden, 1)
        )
        self.actions = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, len(NON_LINK_ACTIONS))
        )

    def forward(self, mention: "torch.Tensor", candidates: "torch.Tensor") -> "torch.Tensor":
        """``(dim,)`` and ``(n, dim)`` → ``(n + 3,)`` logits."""
        action_logits = self.actions(mention)
        if candidates.numel() == 0:
            return action_logits
        expanded = mention.unsqueeze(0).expand(candidates.shape[0], -1)
        # The elementwise product is the interaction term: without it the head
        # can only add the two representations, and "does this mention match
        # this entity" is a *similarity* question.
        joint = torch.cat([expanded, candidates, expanded * candidates], dim=-1)
        return torch.cat([self.pair(joint).squeeze(-1), action_logits], dim=0)

    @staticmethod
    def decode(logits: "torch.Tensor", candidate_ids: Sequence[str]) -> tuple[str, str | None, float]:
        """``(action, entity_id | None, confidence)`` from one item's logits."""
        probs = torch.softmax(logits, dim=-1)
        best = int(torch.argmax(probs).item())
        confidence = float(probs[best].item())
        if best < len(candidate_ids):
            return "LINK_EXISTING", candidate_ids[best], confidence
        return NON_LINK_ACTIONS[best - len(candidate_ids)], None, confidence


class D2Decoder(nn.Module):
    """Claim pair → one of four mutually exclusive relations.

    One softmax over :data:`graft.graphbuild.items.D2_LABELS`, which is the
    grouping decision made structural: a model that cannot emit
    "DUPLICATE and CONFLICT" is a model that cannot contradict itself.
    """

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or TRAINING["hidden"])
        self.net = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.ReLU(),
            nn.Dropout(TRAINING["dropout"]),
            nn.Linear(hidden, len(D2_LABELS)),
        )

    def forward(self, a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        """Symmetric by construction in its |a-b| and a*b terms, asymmetric in the
        concatenation — SUPERSEDES has a direction and DUPLICATE does not, so the
        representation has to carry both."""
        return self.net(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))


class TypedDecoder(nn.Module):
    """D3 and D4: a typed head over a pair (or a span) representation.

    One class for both because they differ only in their label vocabulary — which
    is what "the same frozen decoder interface" means in G6, and what lets the
    external loaders train through the same trainer that trains D1/D2.
    """

    def __init__(self, dim: int, n_labels: int, hidden: int | None = None, pair: bool = True) -> None:
        super().__init__()
        hidden = int(hidden or TRAINING["hidden"])
        width = (3 * dim) if pair else dim
        self.pair = pair
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(TRAINING["dropout"]),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, a: "torch.Tensor", b: "torch.Tensor" | None = None) -> "torch.Tensor":
        if not self.pair or b is None:
            return self.net(a)
        return self.net(torch.cat([a, b, a * b], dim=-1))


# --------------------------------------------------------------------------
# calibration (decision 8)
# --------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """One scalar per head, fitted on **dev only**.

    Fitting on train would calibrate against the distribution the model already
    memorised and report a confidence that does not survive contact with new
    data; fitting on test would be the same leak one step worse.  The restriction
    is the method's, not this project's (Guo et al., ICML 2017), and it is
    enforced here by the signature: :meth:`fit` takes exactly one split.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp().item())

    def forward(self, logits: "torch.Tensor") -> "torch.Tensor":
        return logits / self.log_temperature.exp().clamp(min=1e-3)

    def fit(self, logits: "torch.Tensor", targets: "torch.Tensor", steps: int = 200) -> float:
        """Minimise dev NLL over the single temperature.  Returns the fitted value."""
        optimiser = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=steps)
        criterion = nn.CrossEntropyLoss()

        def closure() -> "torch.Tensor":
            optimiser.zero_grad()
            loss = criterion(self.forward(logits), targets)
            loss.backward()
            return loss

        optimiser.step(closure)
        return self.temperature


def brier_score(probs: "torch.Tensor", targets: "torch.Tensor") -> float:
    """Multi-class Brier score — the squared error of the whole probability vector.

    Reported alongside ECE because they fail differently: ECE is blind to a model
    that is confidently wrong in a *balanced* way across bins, and Brier is not.
    """
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(probs.shape[0]), targets] = 1.0
    return float(((probs - onehot) ** 2).sum(dim=1).mean().item())


def expected_calibration_error(
    probs: "torch.Tensor", targets: "torch.Tensor", bins: int = 10
) -> float:
    """ECE over equal-width confidence bins.

    Equal-width rather than equal-mass, which is the convention Guo et al. use;
    stated because the two give different numbers and a reader comparing against
    the paper needs to know which is here.
    """
    # Detached: ECE is a *report*, never a loss, and computing it on a graph-
    # attached tensor both warns and invites someone to backprop through a
    # calibration metric by accident.
    probs = probs.detach()
    confidence, prediction = probs.max(dim=1)
    correct = (prediction == targets).float()
    total = probs.shape[0]
    if total == 0:
        return float("nan")
    ece = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        n = int(mask.sum().item())
        if n == 0:
            continue
        ece += (n / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def class_weights(labels: Sequence[int], n_classes: int) -> "torch.Tensor":
    """Inverse-frequency weights — **never** resampling (decision 6, item 6).

    Resampling changes the empirical class balance, and D2's over-sampled rare
    classes are already a declared property of the annotation pool: resampling on
    top would make the quoted balance unreproducible and would let a rare-class
    macro-F1 be read as if it came from the natural distribution.

    A class with no examples gets weight 1.0 rather than infinity: it contributes
    nothing to the loss either way, and an infinity would poison the sum.
    """
    counts = torch.zeros(n_classes)
    for label in labels:
        counts[int(label)] += 1
    weights = torch.where(counts > 0, counts.sum() / (n_classes * counts.clamp(min=1)), torch.ones(1))
    return weights
