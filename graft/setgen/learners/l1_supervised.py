"""L1 — supervised stepwise construction.

**[EVIDENCE]** *Graph-S3* (ACL 2026) makes stepwise supervised graph retrieval a
strong baseline: +8.1% accuracy and +9.7% F1 over seven baselines in its own
setting. ``CLAUDE.md`` §5 records the correction that matters here — Graph-S3
trains an **LLM** retriever with supervised stepwise supervision, with no
GFlowNet and no energy, so it validates nothing about the flow family. It is a
strong *baseline*, and this arm is the flow-free form of it on this environment.

**The task it actually attempts, and the one it does not.** L1 learns to
reconstruct **one** proof — the gold set — by teacher-forced next-action
prediction over closure-legal orderings. It is not trying to sample proportional
to reward, and it has no way to express a distribution over the thousands of
other valid terminals. Exact TV is therefore **descriptive** for this arm and is
reported in a separate table (G2, decision 12); one table implying three methods
failed at a task two of them never attempted is exactly the error that split
prevents.

**Orderings are sampled, not fixed.** That is what distinguishes L1 from L2: L1
supervises the *set* (a fresh legal order per example, so the policy is not
taught an arbitrary sequence the generator happened to emit), L2 imitates one
canonical sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Trainer

__all__ = ["supervised_loss"]


def supervised_loss(
    batch: tuple[torch.Tensor, torch.Tensor], trainer: "Trainer", spent: int
) -> torch.Tensor:
    """Negative log-likelihood of the gold action at each visited state.

    ``logits`` are already masked log-probabilities, so this is a gather rather
    than a ``cross_entropy`` — running a second softmax over values that have
    already been normalised would quietly change the objective.
    """
    logits, targets = batch
    return -logits.gather(1, targets.view(-1, 1)).mean()
