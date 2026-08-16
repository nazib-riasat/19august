"""P8.6 — the frozen callable Phase 10 consumes (G7, exit criterion 1).

**The architecture's exit criterion for Phase 8 is "gate integrated into the
read-path orchestrator", and that cannot be met here** — the orchestrator is
Phase 10's fix F7 and does not exist. G7's declared split is what this module
implements: Phase 8 **freezes and tests the callable**, Phase 10 **owns the
integration**, and the integration criterion is transferred to Phase 10's list
*by name* (exit criterion 14) so it cannot silently vanish between two plans.

**Pure by construction.** No model loading, no file reads, no clock, no global
state: the model and the threshold are arguments. That is what makes the gate's
decision reproducible from a record — Phase 10 can replay a `GateDecision` from
the artefact and get the same bit — and it is what keeps the orchestrator, when
it lands, from acquiring an I/O dependency through this call.

**One bit and one probability, and the bit is the *only* thing Stage D sees.**
Plan §4.2's corrected design runs Stage D only when the gate says yes; the
probability travels for calibration and for the risk–coverage instrument, not so
that a downstream component can re-threshold it. Re-thresholding elsewhere would
put the operating point in two places, which is how a threshold stops being one.

**Nothing here imports torch.** The model is passed in and called through a
``predict``-shaped callable, so this module — and therefore the interface Phase
10 depends on — stays importable on a bare interpreter.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from graft.gate.features import block_mask, build_features
from graft.gate.pins import ARMS, FEATURE_BLOCKS
from graft.schemas import GateDecision, Obligations

__all__ = ["ARM_BLOCKS", "arm_mask", "decide"]

#: Which feature blocks each arm may see (decision 4).
#:
#: ``pool_only`` is the **reported** gate if ``with_question`` wins on dev but
#: loses the zero-shot LoCoMo transfer — that disagreement *is* the leakage
#: measurement (G2). Encoding the arms as block sets rather than as two models is
#: what makes the ablation a column mask over one featurisation, so the two arms
#: cannot differ by anything except the columns.
ARM_BLOCKS: dict[str, tuple[str, ...]] = {
    "pool_only": tuple(b for b in FEATURE_BLOCKS if b != "question_embedding"),
    "with_question": tuple(FEATURE_BLOCKS),
}


def arm_mask(names: Sequence[str], arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; decision 4's arms are {ARMS}")
    return block_mask(names, ARM_BLOCKS[arm])


def decide(
    predict_fn: Callable[[np.ndarray], Sequence[float]],
    threshold: float,
    *,
    arm: str = "pool_only",
    obligations: Obligations | None = None,
    pool: Any = None,
    atom_scores: Mapping[str, float] | None = None,
    assembly_report: Mapping[str, Any] | None = None,
    snapshot: Any = None,
    saturation: Mapping[str, Any] | None = None,
    question_vector: Sequence[float] | None = None,
    embed_dim: int | None = None,
) -> GateDecision:
    """Stage-C outputs → one ``GateDecision``.

    ``predict_fn`` maps an ``(n, d)`` matrix to ``n`` probabilities — normally
    ``functools.partial(model.predict, mask=...)``, but any callable will do,
    which is what keeps this module free of the ML stack.

    **The comparison is ``>=``**, so a probability exactly at the threshold
    *answers*. Stated because the boundary case is a real one when a model
    saturates, and an undeclared ``>`` would make an operating point chosen on
    the dev curve behave differently at inference than it did when it was chosen.

    The ``pool_only`` arm is served by masking the question-embedding columns
    rather than by omitting them, so the feature vector is the same width in both
    arms and a `GateDecision` records which arm produced it.
    """
    vector, names, _flags = build_features(
        obligations=obligations,
        pool=pool,
        atom_scores=atom_scores,
        assembly_report=assembly_report,
        snapshot=snapshot,
        saturation=saturation,
        question_vector=question_vector,
        embed_dim=embed_dim,
    )
    mask = arm_mask(names, arm)
    probability = float(np.asarray(predict_fn(vector[mask].reshape(1, -1))).reshape(-1)[0])
    return GateDecision(
        p_answerable=probability,
        answerable=probability >= float(threshold),
        threshold=float(threshold),
        feature_names=tuple(n for n, keep in zip(names, mask) if keep),
        arm=arm,
    )
