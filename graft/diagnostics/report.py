"""Phase 11 Stage C — the comparison assembler, and the four things it refuses.

`GRAFT_PHASE11_BUILD.md` G1, G4, G7, criteria 7-13.  This is the module that
turns GRAFT's read-path results into a row that can sit beside a published one,
and it is written as a set of refusals rather than a set of formatters because
every one of those refusals corresponds to a way this comparison could be read
as something it is not.

**The four refusals.**

1. **No comparison without the non-comparability block** (G6).  The block names
   the backbone, embedder, LoCoMo exposure and question-subset differences.  In
   prose those get lost between draft and viva; as a required argument they
   cannot.
2. **``f1_over_answered`` may not enter a reference-comparable column** (G4).
   Phase-10 decision 7 excludes abstentions from means, which is right for
   GRAFT's own diagnostics and wrong here: the reference table's F1 is over
   *every* question, because none of those systems abstain.  A system abstaining
   on 83% of questions and correct on the rest would post a spectacular
   ``f1_over_answered`` and have answered almost nothing.
3. **Adversarial questions are never scored with F1.**  They are unanswerable;
   the question is whether the system abstained, and an F1 against a gold
   non-answer measures nothing.  `DATASET_DECISION.md` §1.2 makes them the
   primary abstention testbed, so they get abstention accuracy and their own
   section.
4. **Cost is reported per query in the ledger's units, with ingestion excluded**
   (G7).  A budget is what the packer was allowed; the ledger is what was spent.

**What this module does not do.**  It runs no baseline and re-runs nothing.  Its
reference side is `graft.baselines.reference`, which is quoted data, so Gate 4
item 4 is unmet and :func:`build_report` says so in the artefact header.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from graft.baselines.categories import (
    ADVERSARIAL,
    ANSWERABLE_CATEGORIES,
    adversarial_subset,
    categorise,
    four_way_split,
    verify_against_corpus,
)
from graft.baselines.reference import (
    GATE4_STATUS,
    PINNED_METRIC,
    comparable_rows,
    reference_block,
)
from graft.reader.parse import bleu1, token_f1

__all__ = [
    "ReportError",
    "ANSWERED_OUTCOMES",
    "score_split",
    "adversarial_report",
    "graft_row",
    "comparison_table",
    "build_report",
]


class ReportError(RuntimeError):
    """A comparison was assembled in a way that would misrepresent it."""


#: Outcomes that count as the system having answered.  ``contested`` is an
#: answer that disagreed with its runner-up, not an abstention.
ANSWERED_OUTCOMES = ("answer", "contested")


def _score_one(row: Mapping[str, Any]) -> tuple[float, float, bool]:
    """``(f1, bleu1, answered)`` for one question.

    An abstention scores 0.0 on both.  That is not a judgement that abstaining is
    wrong -- it is what makes the number comparable to a table whose systems never
    abstain, and the *un*-penalised view is reported beside it under its own name.
    """
    answered = row.get("outcome") in ANSWERED_OUTCOMES
    if not answered:
        return 0.0, 0.0, False
    predicted = str(row.get("answer_text") or "")
    gold = str(row.get("gold") or "")
    return token_f1(predicted, gold), bleu1(predicted, gold), True


def _mean(xs: Sequence[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def score_split(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-category and overall metrics, in both views, with coverage.

    ``over_all`` scores an abstention 0 and is the **only** view a reference
    comparison may use.  ``over_answered`` excludes abstentions per Phase-10
    decision 7 and travels with ``coverage`` so it cannot be read as a system-level
    accuracy.  Both are computed here rather than in two places, because the
    entire risk in G4 is one being mistaken for the other.
    """
    rows = list(rows)
    split = four_way_split(rows)

    per_category: dict[str, Any] = {}
    for name in ANSWERABLE_CATEGORIES:
        scored = [_score_one(r) for r in split[name]]
        answered = [s for s in scored if s[2]]
        per_category[name] = {
            "n": len(scored),
            "answered": len(answered),
            "coverage": (len(answered) / len(scored)) if scored else None,
            "f1_over_all": _mean([s[0] for s in scored]),
            "bleu1_over_all": _mean([s[1] for s in scored]),
            "f1_over_answered": _mean([s[0] for s in answered]),
            "bleu1_over_answered": _mean([s[1] for s in answered]),
        }

    answerable = [r for r in rows if categorise(r.get("category")) != ADVERSARIAL]
    scored = [_score_one(r) for r in answerable]
    answered = [s for s in scored if s[2]]
    overall = {
        "n": len(scored),
        "answered": len(answered),
        "coverage": (len(answered) / len(scored)) if scored else None,
        "f1_over_all": _mean([s[0] for s in scored]),
        "bleu1_over_all": _mean([s[1] for s in scored]),
        "f1_over_answered": _mean([s[0] for s in answered]),
        "bleu1_over_answered": _mean([s[1] for s in answered]),
    }

    return {
        "per_category": per_category,
        "overall": overall,
        "metric": PINNED_METRIC,
        "views": (
            "over_all scores an abstention 0 and is the reference-comparable view; "
            "over_answered excludes abstentions (Phase-10 decision 7) and is a "
            "GRAFT-internal diagnostic that must travel with coverage"
        ),
        # The adversarial set is absent by construction: `four_way_split` has no
        # branch that admits it, and `answerable` filters it explicitly.
        "adversarial_excluded": True,
    }


def adversarial_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The abstention testbed, scored as abstention rather than as answering.

    Refusal 3.  These questions are unanswerable, so an F1 against a gold
    non-answer measures nothing; what matters is whether the system declined.
    `DATASET_DECISION.md` §1.2 makes this GRAFT's *primary* abstention evidence
    and notes that Mem-T §A.1, Mem0 and A-Mem all discard the category -- so this
    section has no reference row to sit beside, which is exactly why it is the
    strongest thing in the report.
    """
    adv = adversarial_subset(rows)
    abstained = [r for r in adv if r.get("outcome") not in ANSWERED_OUTCOMES]
    by_cause: dict[str, int] = {}
    for r in abstained:
        cause = str(r.get("abstain_cause") or "unknown")
        by_cause[cause] = by_cause.get(cause, 0) + 1

    return {
        "n": len(adv),
        "abstained": len(abstained),
        # The metric. A correct abstention on an unanswerable question is the
        # right answer, so this is accuracy, not a refusal rate to minimise.
        "abstention_accuracy": (len(abstained) / len(adv)) if adv else None,
        # Split, never summed -- `PHASE5_DECISIONS.md` §1's standing precedent.
        "abstain_by_cause": dict(sorted(by_cause.items())),
        "scored_with_f1": False,
        "why": (
            "unanswerable questions are scored by whether the system declined, not "
            "by F1 against a gold non-answer; no published row reports this "
            "category (Mem-T §A.1, Mem0, A-Mem all discard it)"
        ),
    }


def graft_row(
    scores: Mapping[str, Any],
    *,
    backbone: str,
    embedder: str,
    budget_tokens: int,
) -> dict[str, Any]:
    """GRAFT's own row, in the reference table's shape so the table is uniform.

    Carries **only** the ``over_all`` view, which is refusal 2 enforced by
    construction rather than by a check: the comparable row has no field that
    could hold ``f1_over_answered``.
    """
    per = scores["per_category"]
    return {
        "method": "GRAFT",
        "backbone": backbone,
        "f1": {name: per[name]["f1_over_all"] for name in ANSWERABLE_CATEGORIES},
        "bleu1": {name: per[name]["bleu1_over_all"] for name in ANSWERABLE_CATEGORIES},
        "overall_f1": scores["overall"]["f1_over_all"],
        "overall_bleu1": scores["overall"]["bleu1_over_all"],
        "source": "measured in this run",
        "arxiv": "n/a",
        "table": "n/a",
        "metric": PINNED_METRIC,
        "locomo_exposure": "zero_shot",
        "embedder": embedder,
        "question_subset": "answerable_only_adversarial_reported_separately",
        "notes": (
            f"abstentions scored 0; coverage {scores['overall']['coverage']}; "
            f"serialization budget {budget_tokens} tokens"
        ),
        "view": "over_all",
    }


def comparison_table(
    row: Mapping[str, Any],
    *,
    non_comparability: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """GRAFT's row beside the comparable reference rows, or an exception.

    Refusals 1 and 2 both live here.  ``non_comparability=None`` raises rather
    than defaulting, because a default would be the caveat silently going missing
    exactly when someone assembles the table in a hurry.
    """
    if not non_comparability:
        raise ReportError(
            "refusing to build a comparison table without the non-comparability "
            "block: the backbone difference alone moves the reference table's own "
            "GAM row by 23.31 F1, so a table without it invites a reading the data "
            "does not support (GRAFT_PHASE11_BUILD.md G6)"
        )
    if row.get("view") != "over_all":
        raise ReportError(
            f"GRAFT's row carries view={row.get('view')!r}; only 'over_all' may be "
            "compared to a published table, whose F1 is over every question "
            "because none of its systems abstain (G4)"
        )
    if row.get("metric") != PINNED_METRIC:
        raise ReportError(
            f"GRAFT's row is on metric {row.get('metric')!r}, not the pinned "
            f"{PINNED_METRIC!r}; comparing across metric conventions is G8"
        )

    reference = comparable_rows()
    return {
        "gate4": GATE4_STATUS,
        "baselines_rerun": False,
        "metric": PINNED_METRIC,
        "graft": dict(row),
        "reference": [dict(r) for r in reference],
        "non_comparability": dict(non_comparability),
        "reading": (
            "GRAFT is measured; every reference number is quoted from "
            "arXiv:2601.23014v2 Table 2 on a different backbone. Differences in "
            "backbone, embedder, LoCoMo exposure and question subset are declared "
            "and were not controlled. No claim of superiority is licensed by this "
            "table."
        ),
    }


def cost_table(
    cost: Mapping[str, Any],
    *,
    ladder: Sequence[int],
    budget_tokens: int,
    reference_tokens_per_query: int = 9000,
    reference_calls_per_query: str = "<=6 plus construction",
) -> dict[str, Any]:
    """The axis that survives the missing baselines (G7).

    A few points of F1 do not survive a backbone change of the size the reference
    paper documents.  An order-of-magnitude cost ratio does, and call count
    survives it better than token count because it does not depend on a
    tokenizer.

    ``reference_tokens_per_query`` is Mem-T §4.3's ~9k at its chosen 6 retrieval
    steps (rising to ~21k at 10).  Quoted, like every other reference number.
    """
    calls = cost.get("llm_calls_per_query") or {}
    tokens = cost.get("llm_tokens_total_per_query") or {}
    graft_calls = calls.get("mean")
    graft_tokens = tokens.get("mean")

    return {
        "graft": {
            "llm_calls_per_query": graft_calls,
            "llm_tokens_per_query": graft_tokens,
            "wall_clock_ms_per_query": (cost.get("wall_clock_ms_per_query") or {}).get("mean"),
            "serialization_budget_tokens": budget_tokens,
        },
        "reference": {
            "llm_calls_per_query": reference_calls_per_query,
            "llm_tokens_per_query": reference_tokens_per_query,
            "source": "arXiv:2601.23014v2 §4.3 (~9k at 6 steps, ~21k at 10)",
        },
        "token_ratio": (
            round(reference_tokens_per_query / graft_tokens, 2)
            if graft_tokens else None
        ),
        "budget_ladder": list(ladder),
        "excludes": (
            "ingestion, which is an offline per-turn cost (PHASE5_DECISIONS.md §2) "
            "and is reported on its own axis; and the dense channel's corpus "
            "encode, which is index construction amortised over every query"
        ),
    }


def build_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    cost: Mapping[str, Any],
    backbone: str,
    embedder: str,
    budget_tokens: int,
    ladder: Sequence[int],
    ceilings: Mapping[str, Any] | None = None,
    honesty_stamp: Mapping[str, Any] | str | None = None,
    ingestion_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole Phase-11 artefact body.

    Criterion 12's header comes first and is not optional: a reader who stops
    after the first key must already know this is not Gate 4.
    """
    rows = list(rows)
    if honesty_stamp is None:
        raise ReportError(
            "refusing to build a report without an honesty stamp: criterion 13, "
            "and the run consumes an untrained Stage-D policy and a "
            "MuSiQue-trained gate threshold unless something says otherwise"
        )

    scores = score_split(rows)
    row = graft_row(scores, backbone=backbone, embedder=embedder, budget_tokens=budget_tokens)
    block = reference_block()

    return {
        # Criterion 12, first key deliberately.
        "what_this_is_not": GATE4_STATUS,
        "phase": 11,
        "stage": "C (the assembler)",
        "metric": PINNED_METRIC,
        "honesty_stamp": honesty_stamp,
        "corpus_check": verify_against_corpus(rows),
        "scores": scores,
        "adversarial": adversarial_report(rows),
        "comparison": comparison_table(row, non_comparability=block["non_comparability"]),
        "cost": cost_table(cost, ladder=ladder, budget_tokens=budget_tokens),
        "ingestion_cost": dict(ingestion_cost) if ingestion_cost else None,
        "ceilings": dict(ceilings) if ceilings else None,
        "reference_sha": block["reference_sha"],
        "claims": {
            "primary": [
                "cost per query, in LLM calls and total tokens, at three budgets",
                "the five-ceiling decomposition, which needs no baseline",
                "abstention accuracy on the 446 adversarial questions, which no "
                "published row reports",
            ],
            "secondary_with_caveats": [
                "F1 and BLEU-1 by category against the untrained reference row",
            ],
            "not_claimable": (
                "that GRAFT beats any system in the reference table. The backbone "
                "differs, the LoCoMo exposure differs, the question subset differs, "
                "and no baseline was re-run."
            ),
        },
    }
