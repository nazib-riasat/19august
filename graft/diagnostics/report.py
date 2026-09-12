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
    "BOOTSTRAP",
    "score_split",
    "adversarial_report",
    "graft_row",
    "comparison_table",
    "bootstrap_ci",
    "score_intervals",
    "citation_metrics",
    "efficiency_block",
    "reproducibility_block",
    "report_metrics",
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


#: Percentile-bootstrap settings, pinned so two runs of the same rows give the
#: same interval.  10,000 resamples is the conventional floor for a percentile
#: interval reported to two decimals; the seed is the project's first
#: (`CLAUDE.md` §6) so nothing new enters the frozen-values surface.
BOOTSTRAP: dict[str, Any] = {"resamples": 10_000, "seed": 13, "alpha": 0.05}


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Percentile bootstrap over a per-question metric.

    **[ANALYSIS]** -- standard practice, not a claim from any paper here.  The
    resampling unit is the **question**, which is the unit of independence: rows
    from one conversation share a graph, so a conversation-level bootstrap would
    be the more conservative choice and is noted as the road not taken rather
    than silently skipped.

    Returns the point estimate, the interval, and the settings that produced it,
    because an interval without its resample count and seed is not reproducible.
    """
    import numpy as np

    n_boot = int(BOOTSTRAP["resamples"] if resamples is None else resamples)
    rng_seed = int(BOOTSTRAP["seed"] if seed is None else seed)
    a = float(BOOTSTRAP["alpha"] if alpha is None else alpha)

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0,
                "resamples": n_boot, "seed": rng_seed, "alpha": a}

    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_boot, dtype=np.float64)
    # Chunked: a (10_000 x 1_540) index matrix is ~123 MB in one allocation, and
    # this runs on an 8 GB laptop beside a loaded reader.
    step = max(1, min(n_boot, 2_000_000 // max(arr.size, 1)))
    for start in range(0, n_boot, step):
        stop = min(start + step, n_boot)
        idx = rng.integers(0, arr.size, size=(stop - start, arr.size))
        means[start:stop] = arr[idx].mean(axis=1)

    lo, hi = np.percentile(means, [100 * a / 2, 100 * (1 - a / 2)])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(arr.size),
        "resamples": n_boot,
        "seed": rng_seed,
        "alpha": a,
        "method": "percentile bootstrap, resampling questions with replacement",
    }


def score_intervals(
    rows: Iterable[Mapping[str, Any]],
    *,
    resamples: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """B1 -- bootstrap CIs on F1 and BLEU-1, overall and per category.

    Computed on the ``over_all`` view only, because that is the only view a
    reference comparison may use (refusal 2 above) and an interval on a view
    nobody may quote is decoration.
    """
    rows = [r for r in rows if not r.get("adversarial")]
    split = four_way_split(rows)

    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        scored = [_score_one(r) for r in subset]
        return {
            "f1": bootstrap_ci([f for f, _b, _a in scored], resamples=resamples, seed=seed),
            "bleu1": bootstrap_ci([b for _f, b, _a in scored], resamples=resamples, seed=seed),
        }

    per_category = {
        name: block(subset) for name, subset in split.items()
        if name in ANSWERABLE_CATEGORIES or name in split
    }
    return {
        "overall": block(rows),
        "per_category": per_category,
        "view": "over_all -- an abstention scores 0, matching the reference convention",
        "definition": (
            "95% percentile bootstrap over questions, 10,000 resamples, seed 13. "
            "Report as `39.43 [37.9, 41.0]`."
        ),
        # The single most important sentence in this block.
        "no_comparative_test_is_possible": (
            "These intervals quantify GRAFT's OWN sampling variance and nothing "
            "else. No significance test against any reference row is computable: "
            "the published systems' per-item outputs are not available, only "
            "their table means, so there is no paired sample and no variance "
            "estimate for them. An overlapping or non-overlapping interval "
            "against a quoted mean is NOT a test and must not be reported as one."
        ),
    }


def citation_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """B2 -- verifiability, in ALCE's frame.  No Table-2 system reports these.

    **[EVIDENCE]** ALCE (Gao et al., EMNLP 2023) established the separation this
    measures: an answer can be correct and uncited, or fluent and cited to
    nothing, and its finding -- that even the best models fail to fully support
    their claims about half the time on ELI5 -- is this project's motivating
    number (`CLAUDE.md` §9).  What is measured here is the *resolution* side of
    that, not entailment: whether a cited id names a claim that was shown and
    whether that claim carries a real source span.  An NLI-backed precision
    would be the entailment side and is declined here rather than approximated.

    Three rates, over answered questions only -- an abstention emits no citation
    and would otherwise depress every one of them for the wrong reason.
    """
    answered = [r for r in rows if r.get("outcome") in ANSWERED_OUTCOMES]
    if not answered:
        return {"available": False, "reason": "no answered questions"}

    have_detail = [r for r in answered if r.get("citation_ids") is not None]
    if not have_detail:
        return {
            "available": False,
            "reason": (
                "rows carry no `citation_ids`; runs before 21 Aug 2026 stored "
                "only a citation COUNT, which cannot distinguish a resolved "
                "citation from a hallucinated one"
            ),
            "answered": len(answered),
        }

    emitted = 0
    unresolved = 0
    with_any = 0
    fully_grounded = 0
    cited_rows = 0
    for r in have_detail:
        ids = list(r.get("citation_ids") or [])
        unres = set(r.get("citations_unresolved") or [])
        spanned = set(r.get("citations_with_spans") or [])
        emitted += len(ids)
        unresolved += len(unres)
        if ids:
            cited_rows += 1
            with_any += 1
            if all(cid in spanned for cid in ids):
                fully_grounded += 1

    n = len(have_detail)
    return {
        "available": True,
        "answered_with_citation_detail": n,
        "citation_rate": with_any / n if n else None,
        "citation_resolution_rate": (
            (emitted - unresolved) / emitted if emitted else None
        ),
        "span_grounded_answer_rate": (
            fully_grounded / cited_rows if cited_rows else None
        ),
        "citations_emitted": emitted,
        "citations_unresolved": unresolved,
        "mean_citations_per_answered": emitted / n if n else None,
        "definitions": {
            "citation_rate": (
                "share of answered questions emitting at least one [c#]. A "
                "system that answers without citing is unfalsifiable at the "
                "evidence level, which is the property ALCE (EMNLP 2023) named."
            ),
            "citation_resolution_rate": (
                "resolvable citations / emitted citations, pooled over answers. "
                "A hallucinated [c9] naming no shown claim counts AGAINST this "
                "-- resolved with strict=False in the read path precisely so it "
                "is recorded as a reader-ceiling finding rather than crashing "
                "the run."
            ),
            "span_grounded_answer_rate": (
                "share of CITED answers whose every citation resolves to at "
                "least one real source span. Denominator is cited answers, not "
                "all answers: an uncited answer satisfies 'every citation "
                "resolves' vacuously, and counting it would reward not citing."
            ),
        },
        "not_measured_here": (
            "NLI-backed citation precision -- whether the cited span ENTAILS the "
            "answer. That needs the Phase-5 verifier over (span, answer) pairs "
            "and is a separate GPU pass; declined rather than approximated, "
            "because a resolution rate reported as precision would overstate "
            "verifiability by exactly the entailment gap VeriCite measures."
        ),
        "comparability": (
            "no row in the reference table reports any of these, so this is a "
            "GRAFT-only axis and is claimed as such -- not as a win"
        ),
    }


def efficiency_block(
    cost: Mapping[str, Any],
    *,
    wall_clock_s: float | None = None,
    questions: int | None = None,
    hardware: str = "single RTX 5050 Laptop GPU, 8 GB, bf16",
) -> dict[str, Any]:
    """B5 -- the cost axis, hardened with throughput, hardware and Pareto pairs.

    **The refusal that matters**: only Mem-T publishes a token count.  Every
    other row's tokens are marked ``not_published`` rather than estimated from
    its architecture, because an invented denominator would make GRAFT's cost
    claim -- the one axis it actually wins -- rest on numbers nobody measured.
    """
    graft = dict(cost.get("graft") or {})
    throughput = (
        round(questions / wall_clock_s * 3600, 1)
        if wall_clock_s and questions else None
    )

    pareto = []
    for row in comparable_rows():
        method = str(row["method"])
        published = 9000 if method.startswith("Mem-T") else None
        pareto.append({
            "method": method,
            "overall_f1": row["overall_f1"],
            "llm_tokens_per_query": published,
            "tokens_source": (
                "arXiv:2601.23014v2 §4.3 (~9k at 6 steps, ~21k at 10)"
                if published else "not_published"
            ),
        })
    pareto.append({
        "method": "GRAFT",
        "overall_f1": None,  # filled by the caller from its own scores
        "llm_tokens_per_query": graft.get("llm_tokens_per_query"),
        "tokens_source": "measured in this run, reader tokenizer",
    })

    return {
        "per_query": graft,
        "throughput_questions_per_hour": throughput,
        "hardware": hardware,
        "pareto_pairs": pareto,
        "pareto_reading": (
            "Only Mem-T publishes a per-query token count; GAM states a relative "
            "cost claim with no absolute figure. Every other row is "
            "`not_published` and is NOT estimated -- a Pareto frontier drawn "
            "through invented denominators would put GRAFT's one real advantage "
            "on fabricated ground."
        ),
        "excludes": cost.get("excludes"),
    }


def reproducibility_block(
    *,
    corpus_sha: str | None,
    prompt_sha: str | None,
    stage_e_fingerprint: str | None = None,
    config_hash: str | None = None,
    seeds: Sequence[int] = (13, 42, 7),
    decoding: Mapping[str, Any] | None = None,
    determinism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """B6 -- everything needed to say whether two runs are the same experiment."""
    return {
        "corpus_sha256": corpus_sha,
        "prompt_sha": prompt_sha,
        "stage_e_fingerprint": stage_e_fingerprint,
        "config_hash": config_hash,
        "seeds": list(seeds),
        "decoding": dict(decoding or {}),
        "determinism": dict(determinism or {}),
        "reading": (
            "A run is the same experiment as another only if corpus SHA, prompt "
            "SHA and stage-E fingerprint all match. Runs 1-2, run 3, run 4 and "
            "run 5 are four different instruments by this rule and no number "
            "crosses between them."
        ),
        "what_is_not_promised": (
            "byte-identical generations ACROSS machines. Greedy decoding is "
            "deterministic given identical inputs, dtype, kernels and batch "
            "composition -- four conditions, of which this project has already "
            "been surprised by one (dtype). Per-machine determinism is the "
            "measured claim; cross-machine is not made."
        ),
    }


def report_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    cost: Mapping[str, Any],
    wall_clock_s: float | None = None,
    corpus_sha: str | None = None,
    prompt_sha: str | None = None,
    stage_e: str | None = None,
    config_hash: str | None = None,
    decoding: Mapping[str, Any] | None = None,
    determinism: Mapping[str, Any] | None = None,
    hardware: str = "single RTX 5050 Laptop GPU, 8 GB, bf16",
    resamples: int | None = None,
) -> dict[str, Any]:
    """PART B assembled: intervals, citations, efficiency, reproducibility.

    One entry point so the artefact cannot carry three of the four.
    """
    rows = list(rows)
    answerable = [r for r in rows if not r.get("adversarial")]
    intervals = score_intervals(rows, resamples=resamples)

    eff = efficiency_block(
        cost, wall_clock_s=wall_clock_s, questions=len(rows), hardware=hardware
    )
    # The GRAFT row's own F1, so the Pareto pair is complete rather than null.
    overall_f1 = intervals["overall"]["f1"]["mean"]
    for entry in eff["pareto_pairs"]:
        if entry["method"] == "GRAFT":
            entry["overall_f1"] = (
                round(overall_f1 * 100, 2) if overall_f1 is not None else None
            )
            entry["scale_note"] = (
                "reference rows are on a 0-100 scale; GRAFT's is scaled to match"
            )

    return {
        "intervals": intervals,
        "citations": citation_metrics(rows),
        "efficiency": eff,
        "reproducibility": reproducibility_block(
            corpus_sha=corpus_sha,
            prompt_sha=prompt_sha,
            stage_e_fingerprint=stage_e,
            config_hash=config_hash,
            decoding=decoding,
            determinism=determinism,
        ),
        "questions_total": len(rows),
        "questions_answerable": len(answerable),
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
            # **Updated 21 Aug 2026 (PART B).** The primaries are the axes that
            # need no baseline and that no reference row reports -- which is what
            # makes them claimable while Gate 4 is open. F1/BLEU-1 stay secondary
            # and now travel with their intervals, because a point estimate
            # against a quoted mean invites a comparison the data cannot support.
            "primary": [
                "cost per query, in LLM calls and total tokens, with throughput "
                "and the hardware it was measured on",
                "citation rate, citation resolution rate and span-grounded "
                "answer rate -- verifiability in ALCE's frame, which no row in "
                "the reference table reports",
                "the selective-QA risk-coverage curve and AURC at the "
                "MuSiQue-chosen operating point (Geifman & El-Yaniv, NeurIPS "
                "2017), plus adversarial abstention at that point",
                "the five-ceiling decomposition, which needs no baseline",
            ],
            "secondary_with_caveats": [
                "F1 and BLEU-1 by category against the untrained reference row, "
                "reported with 95% bootstrap intervals that quantify GRAFT's own "
                "variance ONLY -- no significance test against any published row "
                "is computable, because their per-item outputs are not available",
            ],
            "not_claimable": (
                "that GRAFT beats any system in the reference table. The backbone "
                "differs, the LoCoMo exposure differs, the question subset differs, "
                "and no baseline was re-run."
            ),
        },
    }
