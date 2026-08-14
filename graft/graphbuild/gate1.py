"""P6.9 — Gate 1's harness and its **predeclared** decision rule (G10, decision 11).

**The rule is written before any comparison exists, and that is the whole point.**
Fix F12's finding, in the architecture's own words: *a decision rule chosen after
seeing results is not a decision rule*.  Phase 3 declared Gate 2's rule before
training; this is the same discipline one gate earlier, and the artefact carries
the rule so a later reader can check it was not adjusted.

**Primary metric — end-to-end mention resolution.**  A prediction counts as
correct only if the four-way action is right *and*, for `LINK_EXISTING`, the
entity id is right (plan §6.4).  The reason is stated because it is easy to
weaken by accident: a model that correctly chooses `LINK_EXISTING` and attaches
to the *wrong* entity would score as correct under a plain action metric, and a
wrong merge is the single most damaging Stage-B error — it corrupts every proof
built on it afterwards (plan §8 risk #1).

**Test — McNemar, paired, α = 0.05 two-sided**, proposed against each baseline.
Paired because the arms see identical items, and McNemar is the test the power
analysis in `data/phase2_5/power.json` was computed for (D1 `n_test` ≈ 627 at
δ = 0.05, ψ = 0.2).  **[EVIDENCE]** test selection per Dror et al. (ACL 2018);
the α and the pairing are this project's own predeclaration, labelled as such
rather than attributed to the paper.

**Three D1 numbers, never one** (plan §6.4): four-way macro-F1 with
`CREATE_NEW_ENTITY` and `NON_ENTITY` broken out separately (they have opposite
failure costs), linking accuracy@1 conditional on `LINK_EXISTING`, and the
end-to-end score.  Candidate recall@k (G4) and proposer recall (G5) print in the
same table, because both bound what any decoder can achieve.

**Smoke discipline (G1, decision 14).**  Gate 1 cannot run until its four entry
conditions hold; two are human-blocked.  So a run on bootstrap labels is stamped
``smoke: true``, and the stamp is not decoration: :func:`run_gate1` **refuses to
compute any proposed-vs-baseline comparison** in smoke mode.  Nothing to quote
can exist, rather than existing and being labelled.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "GATE1_RULE",
    "mcnemar",
    "end_to_end_correct",
    "d1_report",
    "macro_f1",
    "run_gate1",
]

#: The rule, as data, so it lands in the artefact verbatim rather than as prose
#: someone paraphrases later.
GATE1_RULE: Mapping[str, Any] = {
    "primary": "end-to-end mention-resolution score (action correct AND, for "
    "LINK_EXISTING, entity id correct)",
    "test": "McNemar exact/paired, alpha=0.05 two-sided, proposed vs each baseline",
    "arms": [
        "string/embedding-similarity linker (no learning)",
        "E1 GraphMixer-style MLP",
        "E2 HGT",
        "LLM-prompted linking",
        "proposed (E3 + D1..D4)",
    ],
    "seeds": [13, 42, 7],
    "budget": "identical optimizer/epoch budget and early-stop rule across all "
    "arms; parameter counts reported (decision 10)",
    "secondaries": [
        "D1 four-way macro-F1 with CREATE_NEW_ENTITY and NON_ENTITY broken out",
        "D1 linking accuracy@1 conditional on LINK_EXISTING",
        "candidate recall@k (G4) and proposer recall (G5), printed in the same table",
        "D2 four-way macro-F1; D3 F1_c (DialogRE) / micro-F1 (Re-DocRED); D4 TORQUE EM+F1",
        "calibration: Brier and ECE after temperature scaling fitted on dev only",
        "ceiling 2 (does the committed graph contain a sufficient proof?), reported "
        "beside ceiling 1 — Stage B receives ~45% of what the extractor proposed, so "
        "a graph failure would otherwise be misread as a construction failure",
    ],
    "stop_or_redesign": "if the learned constructor does not improve component "
    "accuracy, OR if oracle use of the graph cannot support the target questions, "
    "Contribution 1 is in trouble and the answer is consolidation, not a weaker "
    "evaluation (plan Gate 1, verbatim)",
    "predeclared": "2026-08-14, before any arm was trained",
}


# --------------------------------------------------------------------------
# the primary metric
# --------------------------------------------------------------------------


def end_to_end_correct(prediction: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    """Plan §6.4's three-number report, condition three.

    Action must match; and when the action is `LINK_EXISTING`, the entity id must
    match too.  Written as one function so no caller can implement the weaker
    action-only version by omission.
    """
    if prediction.get("action") != gold.get("action"):
        return False
    if gold.get("action") == "LINK_EXISTING":
        return prediction.get("entity_id") == gold.get("entity_id")
    return True


def macro_f1(predictions: Sequence[str], gold: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    """Per-class F1 plus the macro average, every declared label present.

    A label with no gold and no predictions yields ``nan`` rather than 1.0 — the
    ``slot_level_scores`` convention: an unexercised class must not flatter the
    average, and it must be visibly absent rather than silently perfect.
    """
    out: dict[str, float] = {}
    scored: list[float] = []
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, gold) if p != label and g == label)
        if tp + fp + fn == 0:
            out[f"{label}.f1"] = math.nan
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[f"{label}.f1"] = f1
        scored.append(f1)
    out["macro_f1"] = (sum(scored) / len(scored)) if scored else math.nan
    return out


def d1_report(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    labels: Sequence[str] = ("LINK_EXISTING", "CREATE_NEW_ENTITY", "NON_ENTITY", "DEFER"),
) -> dict[str, Any]:
    """D1's three numbers, together, because reporting one of them is the defect.

    ``linking_accuracy_at_1`` is conditional on the *model* choosing to link, which
    is the question it answers: given that it decided to link, did it link
    correctly?  Conditioning on gold instead would measure a different thing and
    would hide exactly the over-linking failure this number exists to expose.

    Lengths must match *(guarded 14 Aug 2026: the ``zip`` silently truncated a
    short prediction list, so an arm that answered half the items could score
    over the half it chose — an inflated macro-F1 with nothing to notice)*.
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"paired scoring needs one prediction per gold item, got "
            f"{len(predictions)} predictions for {len(gold)} items; a silent "
            "zip-truncation here would let an arm score only the items it "
            "answered"
        )
    actions = [str(p.get("action")) for p in predictions]
    gold_actions = [str(g.get("action")) for g in gold]

    linked = [(p, g) for p, g in zip(predictions, gold) if p.get("action") == "LINK_EXISTING"]
    correct_links = sum(1 for p, g in linked if p.get("entity_id") == g.get("entity_id"))

    end_to_end = sum(1 for p, g in zip(predictions, gold) if end_to_end_correct(p, g))
    n = len(gold)
    return {
        "n": n,
        "end_to_end_score": (end_to_end / n) if n else math.nan,
        "action_macro_f1": macro_f1(actions, gold_actions, labels),
        "linking_accuracy_at_1": (correct_links / len(linked)) if linked else math.nan,
        "model_link_rate": (len(linked) / n) if n else math.nan,
        "reading": (
            "the primary is end_to_end_score. A model that chooses LINK_EXISTING "
            "and attaches to the wrong entity scores correct on action alone, and "
            "a wrong merge corrupts every proof built on it afterwards"
        ),
    }


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar on paired per-item correctness.

    The **exact binomial** form rather than the χ² approximation: the discordant
    count at Gate-1 volumes can easily be small, and χ² is unreliable below ~25
    discordant pairs — using it there would be a p-value that looks like evidence
    and is not.

    ``b01`` is "a wrong, b right"; ``b10`` is "a right, b wrong".  With no
    discordant pairs the p-value is 1.0 and is reported as such — the arms made
    identical decisions, which is a result, not a missing number.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(f"paired test needs equal lengths, got {len(a_correct)} and {len(b_correct)}")
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and (not b))
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "discordant": 0, "p_value": 1.0, "significant": False}

    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2 * tail)
    return {
        "b01": b01,
        "b10": b10,
        "discordant": n,
        "p_value": p,
        "significant": p < 0.05,
        "direction": "b better" if b01 > b10 else ("a better" if b10 > b01 else "tied"),
        "note": "exact binomial; the chi-square approximation is unreliable below "
        "~25 discordant pairs and Gate-1 volumes can sit there",
    }


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------


def run_gate1(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    gold: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every arm, and compare them **only when this is not a smoke run**.

    ``smoke=True`` is not a label on a comparison — it *suppresses* the
    comparison.  G1's discipline is that bootstrap-label runs prove the machinery
    works and are never quoted; a run that computed the McNemar table and stamped
    it "smoke" would be quotable by anyone who read past the stamp, which is the
    bootstrap-labels mistake one phase later.

    The per-arm reports *are* computed in smoke mode, because "the arm produced a
    D1 report of the right shape" is exactly the plumbing the smoke run exists to
    verify.
    """
    reports = {name: d1_report(predictions, gold) for name, predictions in arms.items()}

    artefact: dict[str, Any] = {
        "phase": 6,
        "gate": 1,
        "smoke": bool(smoke),
        "rule": dict(GATE1_RULE),
        "n_items": len(gold),
        "arms": sorted(arms),
        "per_arm": reports,
        **(dict(extras) if extras else {}),
    }

    if smoke:
        artefact["comparisons"] = None
        artefact["smoke_notice"] = (
            "SMOKE RUN — bootstrap labels. No proposed-vs-baseline comparison was "
            "computed, deliberately: Gate 1's entry conditions (a signed Gate-0 "
            "contract and human D1/D2 labels) are not met, and a comparison that "
            "existed under a stamp would be quotable by anyone who read past it. "
            "No number in this artefact may be quoted as a result."
        )
        return artefact

    if "proposed" not in arms:
        raise KeyError("a decisive Gate-1 run must include the 'proposed' arm")
    if not gold:
        raise ValueError(
            "a decisive Gate-1 run needs a non-empty test split; with n = 0 every "
            "McNemar test reports p = 1.0 and 'not significant', which reads as a "
            "measured null and is the absence of a measurement"
        )
    # **What the rule promised and what actually ran, side by side.**  The
    # predeclared rule names five arms and six secondaries; a run that produces
    # four arms and no secondaries is legitimate — the LLM baseline needs a
    # signed dollar budget, D2–D4 need their own labels — but the artefact must
    # say so, or a reader comparing the rule block to the comparisons block
    # would have to infer the omissions.  (Found by the 14 Aug review.)
    declared = {
        "similarity": "string/embedding-similarity linker (no learning)",
        "E1": "E1 GraphMixer-style MLP",
        "E2": "E2 HGT",
        "llm_prompted": "LLM-prompted linking",
        "proposed": "proposed (E3 + D1..D4)",
    }
    artefact["arms_declared"] = sorted(declared)
    artefact["arms_omitted"] = {
        name: label for name, label in sorted(declared.items()) if name not in arms
    }
    artefact["omission_reading"] = (
        "an omitted arm is a comparison not made, not a comparison lost: the "
        "primary metric is still scored on every arm that ran, and the rule "
        "block above is reproduced unedited so the difference is visible"
    )
    correct = {
        name: [end_to_end_correct(p, g) for p, g in zip(predictions, gold)]
        for name, predictions in arms.items()
    }
    artefact["comparisons"] = {
        name: mcnemar(correct[name], correct["proposed"])
        for name in sorted(arms)
        if name != "proposed"
    }
    return artefact
