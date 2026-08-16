"""P8.0 — everything Phase 8 freezes, importable without an ML library.

Same shape and same reason as ``graft.ingest.pins``, ``graft.graphbuild.pins``
and ``graft.retrieve.pins``: this module is §6's signed decision table made
executable, and it carries the **stage-G fingerprint** (exit criterion 12) that
``scripts/verify_handoff.py`` prints — so it must stay importable on a bare
interpreter, and the one module below it that needs torch imports it lazily.

**Values that belong elsewhere are absent here on purpose.** The seeds are
project-wide (`CLAUDE.md` §6) and reach this phase through
``graphbuild.pins.TRAINING``; ``pool_cap`` is the config tree's. Giving a frozen
value two homes is the failure `CLAUDE.md` §5 catalogues, and a run's identity is
now the quintuple ``(config_hash, ingestion_fingerprint, stage_b_fingerprint,
stage_c_fingerprint, stage_g_fingerprint)``.

"Stage G" rather than "Stage E": the plan's stage letters name the *pipeline*
stages (A ingestion, B graph, C retrieval, D evidence selection, E reader), and
the gate sits beside D rather than after it. Calling its fingerprint "stage-E"
would collide with the reader's the moment Phase 10 lands.
"""

from __future__ import annotations

from typing import Any

from graft.canonical import digest_of
from graft.graphbuild.pins import EMBEDDER, TRAINING

__all__ = [
    "EMBEDDER",
    "ARMS",
    "MODELS",
    "FEATURE_BLOCKS",
    "THRESHOLD_RULE",
    "PREVALENCES",
    "TRAINING_GATE",
    "CLASS_HANDLING",
    "ABSTAIN_CAUSES",
    "FALLBACK_COUNTER",
    "PRIMARY_METRIC",
    "SAFETY_SECONDARY",
    "frozen_values",
    "stage_g_fingerprint",
]

# --------------------------------------------------------------------------
# decision 4 — the ablation arms (G2)
# --------------------------------------------------------------------------

#: The two feature arms, run under **identical** budgets and seeds.
#:
#: ``pool_only`` is the *reported* gate if ``with_question`` wins on dev but
#: loses the zero-shot LoCoMo transfer — that disagreement **is** the leakage
#: measurement (G2), not a disappointment. Stated here rather than decided later,
#: because a rule chosen after seeing which arm won is not a rule.
#:
#: On MuSiQue pairs the question embedding is *provably inert*: the two twins
#: carry byte-identical question strings (verified over the real dev split,
#: `GRAFT_PHASE8_BUILD.md` §6), so any separation ``with_question`` achieves
#: there comes from the pool side regardless. The arm exists for the **natural**
#: eval sets, where "unanswerable-sounding" wording could carry the label.
ARMS: tuple[str, ...] = ("pool_only", "with_question")

#: Decision 2. LR is the baseline arm, the MLP the capacity arm — the
#: architecture's own words are "deliberately small; the gate's value is the
#: *decision protocol*, not capacity". Both are reported; the dev-selected one is
#: headlined, which is what stops the choice being made after seeing the test.
#:
#: ``max_params`` is a guard rather than an aspiration: at ~400 input features a
#: 2-layer MLP at width 64 is ~26k parameters, so the cap is three orders below
#: Stage C's scorer and two below Stage B's encoders. A gate that grew past it
#: would have stopped being the small decision protocol the architecture
#: specifies, and the comparison against LR would become a capacity confound —
#: Phase 3's lesson, one stage later.
MODELS: dict[str, Any] = {
    "lr": {"kind": "logistic_regression", "hidden": 0},
    "mlp": {"kind": "mlp_2layer", "hidden": 64, "dropout": 0.1},
    "max_params": 200_000,
}

# --------------------------------------------------------------------------
# decision 3 — the feature blocks (G2's table, verbatim and ordered)
# --------------------------------------------------------------------------

#: Block names **in vector order**. The order is frozen because an ablation is a
#: column mask over this list, not a re-featurisation: if the order moved, a
#: mask computed against one run would silently select different columns in the
#: next.
#:
#: Every block ships a **presence flag** (G8). Without them the gate can learn
#: "entity/temporal features absent ⇒ this is MuSiQue ⇒ 50% unanswerable" — a
#: *dataset* classifier wearing an answerability classifier's name. The flags
#: make "absent" an explicit, learnable-about input rather than a silent zero.
FEATURE_BLOCKS: tuple[str, ...] = (
    "slot_coverage",
    "channel_scores",
    "pool_shape",
    "saturation",
    "question_embedding",
)

#: Blocks the MuSiQue adapter cannot supply (G8). Declared here so the
#: adaptation's losses are a constant the artefact reports, not a fact a reader
#: has to reconstruct from the code.
MUSIQUE_ABSENT_BLOCKS: tuple[str, ...] = ("slot_coverage",)

# --------------------------------------------------------------------------
# decisions 5 and 8 — threshold, prevalence, class handling (G4, G6)
# --------------------------------------------------------------------------

#: **[EVIDENCE]** Geifman & El-Yaniv, *Selective Classification for Deep Neural
#: Networks* (NeurIPS 2017): a confidence threshold traded against risk on a
#: risk–coverage curve, chosen on dev. The threshold is picked on the
#: **natural-prevalence** curve, never the training one.
THRESHOLD_RULE = "dev_risk_coverage_at_natural_prevalence"

#: **The desired risk the threshold is chosen against — pinned 15 Aug 2026.**
#:
#: **[EVIDENCE]** Geifman & El-Yaniv (NeurIPS 2017) require a *declared* target
#: risk and then maximise coverage subject to it.  A target that is not declared
#: is not a selective classifier; it is a number chosen after seeing the curve.
#:
#: **[ANALYSIS] the value, and why it must sit below the base rate.** At natural
#: prevalence the unanswerable share is ``PREVALENCES["natural"]`` = 0.06, so a
#: gate that answers *everything* already has risk 0.06.  Any target at or above
#: the base rate is therefore **satisfied by the trivial gate** and selects a
#: threshold that abstains on nothing — which is exactly what an unpinned 0.10
#: default did on the first full run, on every arm and seed (weighted coverage
#: 1.0, weighted risk 0.06).  0.03 is half the base rate: it demands the gate
#: remove half the residual risk, which is a real requirement rather than a
#: satisfied-by-default one.  The number is engineering judgement; the
#: *constraint that it be below the base rate* is not, and
#: :func:`graft.gate.riskcov.choose_threshold` refuses to report a target that
#: violates it as anything other than vacuous.
TARGET_RISK = 0.03

#: G4's two prevalences. Training pairs are 1:1 **by construction** — that is
#: what a contrast pair *is* — while LongMemEval's natural unanswerable rate is
#: 30/500 = 0.06. A threshold picked at 0.5 and deployed at 0.06 inflates false
#: abstention, so the same dev predictions are reweighted to both and **both
#: curves ship**. Reweighting rather than resampling: no data is discarded and no
#: extra data is needed.
PREVALENCES: dict[str, float] = {"constructed": 0.5, "natural": 0.06}

#: Decision 8. Class weights, **not** resampling — the Gate-0 item-6 rule, and
#: the weights are reported. Evaluation keeps natural prevalence; no
#: natural-frequency claim may be made from the constructed training balance.
CLASS_HANDLING = "class_weights_not_resampling"

#: Decision 7's budget, inheriting the project-wide seeds and the
#: ``graphbuild.pins.TRAINING`` shape so the gate cannot quietly get more epochs
#: than Stage B's arms did. Overridden only by tests, and reported when it is.
TRAINING_GATE: dict[str, Any] = {
    "epochs": int(TRAINING["epochs"]),
    "lr": float(TRAINING["lr"]),
    "weight_decay": float(TRAINING["weight_decay"]),
    "batch_size": int(TRAINING["batch_size"]),
    "early_stop_patience": int(TRAINING["early_stop_patience"]),
    "early_stop_metric": "dev_loss",
    "seeds": tuple(TRAINING["seeds"]),
}

# --------------------------------------------------------------------------
# decisions 6 and 10 — metrics and the abstain vocabulary (G5)
# --------------------------------------------------------------------------

#: Decision 6. One primary, fixed in advance (Gate-0 item 10's discipline).
PRIMARY_METRIC = "aurc"

#: The named safety secondary. Without it a gate scores well by abstaining
#: freely: AURC rewards ranking, and a model that abstains on everything has no
#: risk at zero coverage. This is the number that makes over-abstention visible.
SAFETY_SECONDARY = "false_abstention_rate_on_answerable"

#: G5's two-way vocabulary. Plan §4.2 has **two** distinct abstain routes — the
#: gate says no (step 2), and budget exhaustion falls back (step 3) — and
#: flattening them repeats the quarantine-cause mistake `PHASE5_DECISIONS.md` §1
#: catalogues: two different reasons, one inflated rate. Reserved here so §6.4's
#: fallback trigger rate is reportable from the first Stage-D run rather than
#: retrofitted after it.
ABSTAIN_CAUSES: tuple[str, ...] = ("gate", "fallback")

#: The counter Stage D increments on budget exhaustion. **Zero until Phase 9
#: wires it** (exit criterion 8) — it exists now so that the distinction is
#: structural rather than something a later phase has to remember.
FALLBACK_COUNTER = "abstain_fallback"


def frozen_values() -> dict[str, Any]:
    """Everything that must agree across machines for two gate numbers to compare.

    **Binds the feature *names and order*, not just the block names** — corrected
    15 Aug 2026.  As first written this returned only ``FEATURE_BLOCKS`` (five
    strings), so the pre- and post-amendment feature sets — normalised
    ``{channel}_max``/``_mean`` versus raw ``_raw_max``/``_raw_top3``/``_raw_mean``
    — produced the **same** fingerprint while training different models.  Two
    runs that cannot be told apart by their identity is the one failure a
    fingerprint exists to prevent, and this file had it.

    ``TOP_K`` and ``TARGET_RISK`` are bound for the same reason: both change the
    numbers a run reports, so both are part of what the run *is*.  ``pool_cap``
    stays absent — it is the config tree's and reaches identity through
    ``config_hash``.
    """
    from graft.gate.features import BLOCK_FEATURES, TOP_K

    return {
        "embedder": EMBEDDER,
        "arms": list(ARMS),
        "models": MODELS,
        "feature_blocks": list(FEATURE_BLOCKS),
        # the actual contract: every feature name, in vector order, per block
        "feature_names": {k: list(v) for k, v in sorted(BLOCK_FEATURES.items())},
        "top_k": int(TOP_K),
        "musique_absent_blocks": list(MUSIQUE_ABSENT_BLOCKS),
        "threshold_rule": THRESHOLD_RULE,
        "target_risk": TARGET_RISK,
        "prevalences": PREVALENCES,
        "class_handling": CLASS_HANDLING,
        "training": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TRAINING_GATE.items()},
        "primary_metric": PRIMARY_METRIC,
        "safety_secondary": SAFETY_SECONDARY,
        "abstain_causes": list(ABSTAIN_CAUSES),
    }


def stage_g_fingerprint(length: int | None = None) -> str:
    """Configuration identity for the gate, printed by ``verify_handoff.py``.

    Binds the config, not the output — the same G11 distinction Phases 5, 6 and 7
    drew. Two machines will not produce bit-identical weights from a 2-layer MLP;
    they must produce them from an identical setup, and they must threshold with
    identical arithmetic, or two abstention rates are not comparable.
    """
    return digest_of(frozen_values(), length)
