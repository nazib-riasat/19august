"""Phase 8 — the decoupled answerability gate.  Half of Contribution 2.

A small trained classifier that answers one question — *does a sufficient proof
exist in the current snapshot for this question?* — run **before** Stage D and
evaluated as **selective prediction**.

**Why a separate classifier at all.** Plan §4.2 withdrew `ABSTAIN` as a flow
action because the math is fatal: under reward-proportional sampling
``P(ABSTAIN|q) = R_abstain / (R_abstain + Σ_X R(X))``, so abstention probability
falls as the *number of alternative valid proofs* rises — even at identical
answerability — and no tuning fixes a denominator that moves per question.
(**[EVIDENCE]** the sampling law is GFlowNet Foundations, JMLR 2023; the
unsoundness argument is the plan's own.) The decoupled gate is simpler and
removes a hyperparameter.

**Module map, in build order**

=====================  ========================================================
``pins``               P8.0 — what Phase 8 freezes, plus the stage-G fingerprint
``features``           P8.1 — G2's feature contract; the claim itself
``labels``             P8.2 — the **only** gold-reading gate module (G3)
``adapt_musique``      P8.3 — the declared MuSiQue adaptation and its losses (G8)
``model``              P8.4 — LR and 2-layer MLP; the only module importing torch
``riskcov``            P8.5 — risk–coverage, AURC, dual prevalence (G4)
``decide``             P8.6 — the frozen callable Phase 10 consumes (G7)
=====================  ========================================================

**Two boundaries this package holds, both asserted structurally rather than
promised:**

* **gold** reaches ``labels.py`` and nothing else — a gate that could read
  ``has_answer`` would *be* the label;
* **torch** reaches ``model.py`` and nothing else — so the feature contract, the
  label recipe, the adaptation and the whole selective-prediction instrument stay
  checkable on a bare interpreter.

**Stage A is built; Stage B waits on scope-c ingestion** (G9). The MuSiQue track
is trainable today; the conversational deletion-pair track and every decisive
evaluation number are exit criterion 13's deferred-by-name items. No number
produced on the pilot graph is a gate quality result, and the artefact says so in
its own body.
"""
