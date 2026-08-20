"""Phase 11 Stage B — the metrics, the categories and the reference table.

Numbering follows `GRAFT_PHASE11_BUILD.md` §5.  Nothing here runs a baseline:
the three adapters are deferred by name (§7 of that plan), and what is guarded
is that the *quoted* numbers cannot be read as something they are not.
"""

from __future__ import annotations

import pytest

from graft.baselines.categories import (
    ADVERSARIAL,
    ANSWERABLE_CATEGORIES,
    EXPECTED_ADVERSARIAL,
    EXPECTED_TOTAL,
    CategoryError,
    adversarial_subset,
    categorise,
    four_way_split,
    verify_against_corpus,
)
from graft.baselines.reference import (
    COMPARABLE_ROW,
    MEM_T_TABLE2,
    PINNED_METRIC,
    ReferenceError,
    comparable_rows,
    reference_block,
)
from graft.reader.parse import bleu1, token_f1


# --------------------------------------------------------------------------
# criterion 5 — the metrics reproduce hand-worked fixtures
# --------------------------------------------------------------------------


def test_bleu1_is_exact_on_an_exact_match():
    assert bleu1("London", "London") == pytest.approx(1.0)


def test_bleu1_penalises_a_verbose_answer_by_precision():
    """Three predicted tokens, one of them gold: clipped precision 1/3, and no
    brevity penalty because the candidate is longer than the reference."""
    assert bleu1("born in London", "London") == pytest.approx(1 / 3)


def test_bleu1_penalises_a_short_answer_by_brevity():
    """One predicted token against three gold: precision 1.0, and the brevity
    penalty exp(1 - 3/1) is what stops a one-word guess scoring perfectly.
    Without it the degenerate minimal answer -- exactly what a minimality-focused
    system might emit -- would look ideal."""
    import math

    assert bleu1("London", "born in London") == pytest.approx(math.exp(-2.0))


def test_bleu1_and_token_f1_share_one_normalisation():
    """`PHASE10_DECISIONS.md` §1.1: SQuAD normalisation stripped the punctuation
    out of `[c1]` and left the bare token `c1` counting as an answer word -- a
    defect that could only ever mark a *correct* answer wrong. A second
    normalisation path in a second metric would re-open it."""
    assert bleu1("the London", "London") == pytest.approx(1.0)
    assert token_f1("the London", "London") == pytest.approx(1.0)


def test_both_metrics_are_zero_on_a_disjoint_answer():
    assert bleu1("Paris", "London") == 0.0
    assert token_f1("Paris", "London") == 0.0


# --------------------------------------------------------------------------
# criterion 6 — adversarial is structurally excluded from the four-way split
# --------------------------------------------------------------------------


def _q(code: int, qid: str) -> dict:
    return {"category": code, "question_id": qid}


def test_the_four_way_split_has_exactly_the_reference_tables_rows():
    split = four_way_split([_q(4, "a"), _q(1, "b"), _q(2, "c"), _q(3, "d")])
    assert set(split) == set(ANSWERABLE_CATEGORIES)


def test_adversarial_cannot_enter_the_four_way_split():
    """G4's structural half. Not "the caller excludes them" -- there is no branch
    that would include them, so a caller cannot forget."""
    split = four_way_split([_q(5, "adv"), _q(4, "ok")])
    assert ADVERSARIAL not in split
    assert sum(len(v) for v in split.values()) == 1


def test_the_adversarial_subset_is_reached_by_its_own_function():
    """A separate entry point rather than a fifth key, so a report iterating the
    split to build a table against a published one cannot pick it up."""
    got = adversarial_subset([_q(5, "adv"), _q(4, "ok")])
    assert [q["question_id"] for q in got] == ["adv"]


def test_an_unmapped_category_code_is_refused_not_bucketed():
    """A silent "other" bucket would put unknown questions inside a total then
    reported as LoCoMo's."""
    with pytest.raises(CategoryError, match="unknown LoCoMo category"):
        categorise(99)


def test_the_mapping_verifies_itself_against_an_independent_count():
    """The pin is unverified against the dataset file, so it is checked against a
    count recorded somewhere else: `DATASET_DECISION.md` §1.2's 446. Agreement
    between two independent sources is evidence; disagreement is a defect."""
    corpus = [_q(5, f"a{i}") for i in range(EXPECTED_ADVERSARIAL)]
    corpus += [_q(4, f"s{i}") for i in range(EXPECTED_TOTAL - EXPECTED_ADVERSARIAL)]
    report = verify_against_corpus(corpus)
    assert report["full_corpus"] is True
    assert report["counts"][ADVERSARIAL] == EXPECTED_ADVERSARIAL
    assert "conclusive" in report["adversarial_check"]


def test_a_wrong_mapping_fails_loudly_on_the_full_corpus():
    """If the integer codes point at the wrong buckets, the adversarial count
    will not be 446 and the run must stop rather than mislabel every question."""
    corpus = [_q(5, f"a{i}") for i in range(100)]
    corpus += [_q(4, f"s{i}") for i in range(EXPECTED_TOTAL - 100)]
    with pytest.raises(CategoryError, match="failed its own check"):
        verify_against_corpus(corpus)


def test_a_subset_run_reports_inconclusive_rather_than_failing():
    """Failing on a subset would make the check unusable exactly while the corpus
    is being scaled up -- which is the whole of this project's three-day plan."""
    report = verify_against_corpus([_q(5, "a"), _q(4, "s")])
    assert report["full_corpus"] is False
    assert "inconclusive" in report["adversarial_check"]


# --------------------------------------------------------------------------
# criteria 9-11 — the reference table and its refusals
# --------------------------------------------------------------------------


def test_the_comparable_row_is_the_untrained_one():
    """G5, pinned before GRAFT has a number. Mem-T trains on LoCoMo (§4.1,
    1:1:8); GRAFT never does, and the paper prices the difference at +9.27 F1."""
    assert COMPARABLE_ROW == "Mem-T (w/o training)"
    row = reference_block()["comparable_row"]
    assert row["overall_f1"] == 49.38
    assert row["overall_bleu1"] == 44.11
    assert row["locomo_exposure"] == "zero_shot"


def test_the_pin_names_exactly_one_row():
    assert sum(1 for r in MEM_T_TABLE2 if r["method"] == COMPARABLE_ROW) == 1


def test_in_domain_rows_are_refused_against_a_zero_shot_system():
    """An in-domain row beside a zero-shot one measures benchmark exposure, not
    the systems."""
    rows = comparable_rows()
    assert all(r["locomo_exposure"] == "zero_shot" for r in rows)
    assert "Mem-T (MoT-GRPO)" not in {r["method"] for r in rows}


def test_a_metric_mismatch_is_refused():
    """G8: LLM-judge (HyperMem 92.73, Mem0 72.90) and F1 are different axes, and
    nothing in a results table announces that."""
    with pytest.raises(ReferenceError, match="refusing to emit"):
        comparable_rows(MEM_T_TABLE2, exposure="no_such_exposure")


def test_every_reference_number_carries_its_provenance():
    """Criterion 11. A number without its table is a number that cannot be
    checked, and `CLAUDE.md` §10 is the standing rule about verifying before
    relying."""
    for row in reference_block()["all_rows"]:
        assert row["arxiv"] == "arXiv:2601.23014v2"
        assert row["table"] == "Table 2"
        assert row["metric"] == PINNED_METRIC
        assert row["backbone"]


def test_the_block_states_gate_4_is_unmet():
    """Criterion 12. A comparison that reads as controlled and is not is
    `CLAUDE.md` §5's overreach pattern, in the results section."""
    block = reference_block()["non_comparability"]
    assert block["baselines_rerun"] is False
    assert "NOT MET" in block["gate4"]
    assert "Gate 4 item 4" in block["gate4"] or "item 4" in block["gate4"]


def test_the_backbone_difference_carries_the_papers_own_evidence():
    """The 23.31 F1 GAM loses on a backbone change is what makes "declared, not
    matched" defensible rather than an excuse."""
    diffs = reference_block()["non_comparability"]["declared_differences"]
    assert "23.31" in diffs["reader_backbone"]["why_it_matters"]
    assert "Qwen2.5-3B" in diffs["reader_backbone"]["graft"]


def test_the_training_difference_is_priced_from_the_papers_own_numbers():
    diffs = reference_block()["non_comparability"]["declared_differences"]
    assert "9.27" in diffs["locomo_exposure"]["measured_effect"]


def test_the_pinned_row_is_shad_and_the_sha_tracks_the_pin_not_the_table():
    """Criterion 9. The digest covers the pin and the metric, so adding a
    context row leaves it alone while editing the row GRAFT is judged against
    moves it -- which makes a changed baseline visible in an artefact diff rather
    than silent."""
    from graft.baselines.reference import reference_sha

    sha = reference_sha()
    assert isinstance(sha, str) and len(sha) >= 8
    assert reference_block()["reference_sha"] == sha
    # Stable across calls: a pin that moved on its own would be no pin.
    assert reference_sha() == sha


# --------------------------------------------------------------------------
# Stage C — the assembler's four refusals (criteria 7, 8, 10, 12, 13)
# --------------------------------------------------------------------------


def _row_for(code: int, outcome: str, answer: str, gold: str, qid: str = "q") -> dict:
    return {
        "question_id": qid,
        "category": code,
        "outcome": outcome,
        "answer_text": answer,
        "gold": gold,
    }


def _cost() -> dict:
    return {
        "llm_calls_per_query": {"mean": 2.0, "median": 2.0, "max": 2.0},
        "llm_tokens_total_per_query": {"mean": 526.8, "median": 526.5, "max": 709.0},
        "wall_clock_ms_per_query": {"mean": 1054.2, "median": 976.5, "max": 1673.0},
    }


def test_an_abstention_scores_zero_in_the_comparable_view_and_is_absent_from_the_other():
    """Criterion 7 and G4 -- the easiest number in the phase to get wrong. The
    reference table's F1 is over every question because none of its systems
    abstain; excluding abstentions would let a system that answers 1 of 5 post a
    perfect score."""
    from graft.diagnostics.report import score_split

    rows = [
        _row_for(4, "answer", "London", "London", "a"),
        _row_for(4, "abstain", "", "Paris", "b"),
    ]
    got = score_split(rows)["per_category"]["single_hop"]
    assert got["f1_over_all"] == pytest.approx(0.5)
    assert got["f1_over_answered"] == pytest.approx(1.0)
    assert got["coverage"] == pytest.approx(0.5)


def test_the_comparable_row_cannot_carry_the_answered_only_view():
    """Refusal 2, enforced by construction: `graft_row` has no field that could
    hold `f1_over_answered`, and `comparison_table` checks the view tag as well."""
    from graft.diagnostics.report import ReportError, comparison_table, graft_row, score_split

    scores = score_split([_row_for(4, "answer", "London", "London")])
    row = graft_row(scores, backbone="Qwen2.5-3B", embedder="bge-small-en-v1.5", budget_tokens=512)
    assert row["view"] == "over_all"
    bad = dict(row, view="over_answered")
    with pytest.raises(ReportError, match="only 'over_all'"):
        comparison_table(bad, non_comparability={"declared": True})


def test_a_comparison_without_the_non_comparability_block_is_refused():
    """Refusal 1. In prose the caveats get lost between draft and viva; as a
    required argument they cannot."""
    from graft.diagnostics.report import ReportError, comparison_table, graft_row, score_split

    scores = score_split([_row_for(4, "answer", "London", "London")])
    row = graft_row(scores, backbone="Qwen2.5-3B", embedder="bge-small-en-v1.5", budget_tokens=512)
    with pytest.raises(ReportError, match="non-comparability"):
        comparison_table(row, non_comparability=None)


def test_adversarial_questions_are_scored_by_abstention_not_f1():
    """Refusal 3. They are unanswerable, so an F1 against a gold non-answer
    measures nothing; and no published row reports the category at all."""
    from graft.diagnostics.report import adversarial_report

    rows = [
        _row_for(5, "abstain", "", "", "a"),
        _row_for(5, "answer", "London", "", "b"),
    ]
    got = adversarial_report(rows)
    assert got["n"] == 2
    assert got["abstention_accuracy"] == pytest.approx(0.5)
    assert got["scored_with_f1"] is False


def test_adversarial_never_enters_the_scored_split():
    from graft.diagnostics.report import score_split

    got = score_split([_row_for(5, "abstain", "", ""), _row_for(4, "answer", "London", "London")])
    assert got["overall"]["n"] == 1
    assert got["adversarial_excluded"] is True


def test_the_cost_table_reports_the_ratio_and_names_what_it_excludes():
    """G7: the axis that survives the missing baselines. Ingestion is excluded and
    said to be excluded, because folding an offline per-turn cost into a per-query
    figure produces a number comparable to nothing."""
    from graft.diagnostics.report import cost_table

    got = cost_table(_cost(), ladder=(160, 512, 1024), budget_tokens=512)
    assert got["token_ratio"] == pytest.approx(9000 / 526.8, rel=1e-3)
    assert got["graft"]["llm_calls_per_query"] == 2.0
    assert "ingestion" in got["excludes"]
    assert got["budget_ladder"] == [160, 512, 1024]


def test_the_report_refuses_without_an_honesty_stamp():
    """Criterion 13. The run consumes an untrained Stage-D policy and a
    MuSiQue-trained gate threshold; nothing may read as a result by default."""
    from graft.diagnostics.report import ReportError, build_report

    with pytest.raises(ReportError, match="honesty stamp"):
        build_report(
            [_row_for(4, "answer", "London", "London")],
            cost=_cost(), backbone="Qwen2.5-3B", embedder="bge-small-en-v1.5",
            budget_tokens=512, ladder=(160, 512, 1024), honesty_stamp=None,
        )


def test_the_reports_first_key_says_it_is_not_gate_4():
    """Criterion 12. A reader who stops after the first key must already know."""
    from graft.diagnostics.report import build_report

    report = build_report(
        [_row_for(4, "answer", "London", "London")],
        cost=_cost(), backbone="Qwen2.5-3B", embedder="bge-small-en-v1.5",
        budget_tokens=512, ladder=(160, 512, 1024),
        honesty_stamp="WIRING TEST",
    )
    assert list(report)[0] == "what_this_is_not"
    assert "NOT MET" in report["what_this_is_not"]
    assert report["comparison"]["baselines_rerun"] is False
    assert "not_claimable" in report["claims"]
