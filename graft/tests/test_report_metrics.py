"""PART B — the paper-grade metric block, and the four things each metric refuses.

`graft/diagnostics/report.py`'s run-5 additions: bootstrap intervals (B1),
citation metrics in ALCE's frame (B2), the hardened cost axis (B5) and the
reproducibility block (B6), plus the runner's seeded stratified ceiling sample
(B4) and the Phase-8 gate loader (B3).

Every test here is written against a **failure the metric could plausibly
produce quietly**, not against its happy path: an interval that moves between
identical runs, a span-grounding rate inflated by answers that cite nothing, a
Pareto table with invented token counts, a ceiling sample that is one category
from one conversation, a gate restored without the mask that says which columns
it reads.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from graft.diagnostics.report import (
    BOOTSTRAP,
    bootstrap_ci,
    citation_metrics,
    efficiency_block,
    reproducibility_block,
    score_intervals,
    score_split,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "locomo_eval", REPO / "scripts" / "locomo_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["locomo_eval"] = module
    spec.loader.exec_module(module)
    return module


def _row(qid, category, gold, answer, *, adversarial=False, outcome="answer", **extra):
    row = {
        "question_id": qid,
        "conv_id": qid.split("/")[0],
        "category": category,
        "adversarial": adversarial,
        "gold": gold,
        "answer_text": answer,
        "outcome": outcome,
        "abstain_cause": None if outcome == "answer" else "fallback",
    }
    row.update(extra)
    return row


# -- B1 ----------------------------------------------------------------------


def test_the_bootstrap_is_reproducible_and_brackets_its_own_mean():
    """An interval that moves between identical runs is not an interval.

    The seed and resample count are pinned and *reported inside the result*,
    because a CI whose settings are not recorded cannot be reproduced by a
    reader who has only the artefact.
    """
    values = [0.0, 0.25, 0.5, 0.75, 1.0] * 40

    a = bootstrap_ci(values)
    b = bootstrap_ci(values)
    assert a == b, "same values, same seed, same interval"

    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["resamples"] == BOOTSTRAP["resamples"] == 10_000
    assert a["seed"] == BOOTSTRAP["seed"] == 13
    assert a["n"] == len(values)

    # A different seed must move the interval, or the seed is decorative.
    other = bootstrap_ci(values, seed=999)
    assert (other["ci_low"], other["ci_high"]) != (a["ci_low"], a["ci_high"])


def test_the_bootstrap_narrows_as_the_sample_grows():
    """The property that makes an interval worth reporting at all: run 4's
    open-domain category has n=96 and a CI roughly 5x wider than single-hop's
    n=841, which is the whole reason a point estimate there is unreadable."""
    small = bootstrap_ci([0.0, 1.0] * 25, resamples=2_000)
    large = bootstrap_ci([0.0, 1.0] * 500, resamples=2_000)
    width = lambda d: d["ci_high"] - d["ci_low"]  # noqa: E731
    assert width(large) < width(small)


def test_an_empty_sample_returns_nulls_rather_than_a_zero():
    """A category with no questions must not report 0.0 with a tight interval —
    that reads as "measured and bad" instead of "not measured"."""
    empty = bootstrap_ci([])
    assert empty["mean"] is None and empty["ci_low"] is None and empty["n"] == 0


def test_the_intervals_agree_with_the_point_estimates_they_accompany():
    """If `score_intervals` and `score_split` disagreed, the artefact would carry
    a mean and an interval computed from different data — and the interval would
    silently be the one nobody checked."""
    rows = [
        _row("c1/q1", 1, "Paris", "Paris"),
        _row("c1/q2", 1, "Rome", "Milan"),
        _row("c1/q3", 2, "blue", "blue"),
        _row("c1/q4", 3, "8 May 2023", "8 May 2023"),
        _row("c1/q5", 4, "hiking", "", outcome="abstain"),
        _row("c1/q6", 5, "", "", adversarial=True, outcome="abstain"),
    ]
    split = score_split(rows)
    intervals = score_intervals(rows, resamples=500)
    assert intervals["overall"]["f1"]["mean"] == pytest.approx(
        split["overall"]["f1_over_all"]
    )
    assert intervals["overall"]["bleu1"]["mean"] == pytest.approx(
        split["overall"]["bleu1_over_all"]
    )
    assert intervals["overall"]["f1"]["n"] == split["overall"]["n"], (
        "the adversarial question belongs to neither statistic"
    )


def test_the_interval_block_forbids_a_comparative_test_in_writing():
    """The one sentence that stops a reader drawing the conclusion the data
    cannot support. Published systems' per-item outputs do not exist, so there
    is no paired sample and no variance for them — an overlapping interval
    against a quoted mean is not a test."""
    intervals = score_intervals([_row("c1/q1", 1, "Paris", "Paris")], resamples=100)
    note = intervals["no_comparative_test_is_possible"]
    assert "per-item outputs" in note
    assert "is NOT a test" in note


# -- B2 ----------------------------------------------------------------------


def test_span_grounding_is_not_inflated_by_answers_that_cite_nothing():
    """The vacuity trap. "Every citation resolves to a span" is trivially true
    of an answer with no citations, so a rate over ALL answers would reward not
    citing — the opposite of what ALCE's separation is for."""
    rows = [
        _row("c1/q1", 1, "Paris", "Paris",
             citation_ids=["c1"], citations_unresolved=[], citations_with_spans=["c1"]),
        _row("c1/q2", 1, "Rome", "Rome",
             citation_ids=[], citations_unresolved=[], citations_with_spans=[]),
    ]
    m = citation_metrics(rows)
    assert m["citation_rate"] == pytest.approx(0.5), "one of two answers cited"
    assert m["span_grounded_answer_rate"] == pytest.approx(1.0), (
        "denominator is CITED answers: the uncited one is neither grounded nor "
        "ungrounded and must not enter"
    )
    assert "vacuously" in m["definitions"]["span_grounded_answer_rate"]


def test_a_hallucinated_citation_counts_against_the_resolution_rate():
    """A citation naming no shown claim is the reader-ceiling finding the read
    path deliberately keeps (strict=False) rather than crashing on."""
    rows = [
        _row("c1/q1", 1, "Paris", "Paris",
             citation_ids=["c1", "c9"], citations_unresolved=["c9"],
             citations_with_spans=["c1"]),
    ]
    m = citation_metrics(rows)
    assert m["citations_emitted"] == 2 and m["citations_unresolved"] == 1
    assert m["citation_resolution_rate"] == pytest.approx(0.5)
    assert m["span_grounded_answer_rate"] == pytest.approx(0.0), (
        "not every citation resolved to a span, so the answer is not grounded"
    )


def test_citation_metrics_refuse_rows_that_carry_only_a_count():
    """Runs before 21 Aug 2026 stored `citations: int`. A count cannot tell a
    resolved citation from a hallucinated one, so the metric says it is
    unavailable rather than computing something else and naming it this."""
    rows = [_row("c1/q1", 1, "Paris", "Paris", citations=2)]
    m = citation_metrics(rows)
    assert m["available"] is False
    assert "only a citation COUNT" in m["reason"]


def test_abstentions_do_not_enter_the_citation_denominator():
    """An abstention emits no citation by construction; counting it would drag
    every rate down for a reason that has nothing to do with verifiability."""
    rows = [
        _row("c1/q1", 1, "Paris", "Paris",
             citation_ids=["c1"], citations_unresolved=[], citations_with_spans=["c1"]),
        _row("c1/q2", 1, "Rome", "", outcome="abstain",
             citation_ids=[], citations_unresolved=[], citations_with_spans=[]),
    ]
    m = citation_metrics(rows)
    assert m["answered_with_citation_detail"] == 1
    assert m["citation_rate"] == pytest.approx(1.0)


# -- B5 ----------------------------------------------------------------------


def test_no_reference_system_gets_an_invented_token_count():
    """GRAFT's cost advantage is the one axis it actually wins. A Pareto table
    that estimated the other systems' tokens from their architectures would put
    that advantage on numbers nobody measured."""
    block = efficiency_block(
        {"graft": {"llm_tokens_per_query": 1142.4, "llm_calls_per_query": 1.0}},
        wall_clock_s=3600.0,
        questions=1200,
    )
    published = [
        e for e in block["pareto_pairs"] if e["llm_tokens_per_query"] is not None
    ]
    assert {e["method"] for e in published} == {
        "Mem-T (w/o training)", "GRAFT"
    }, "only Mem-T publishes a token count; everything else must be null"
    for entry in block["pareto_pairs"]:
        if entry["llm_tokens_per_query"] is None:
            assert entry["tokens_source"] == "not_published"
    assert block["throughput_questions_per_hour"] == pytest.approx(1200.0)
    assert "8 GB" in block["hardware"], "the hardware is part of a latency claim"


# -- B6 ----------------------------------------------------------------------


def test_the_reproducibility_block_does_not_promise_cross_machine_bytes():
    """Greedy decoding is deterministic given identical inputs, dtype, kernels
    and batch composition — four conditions, of which this project has already
    been surprised by one. Per-machine determinism is the measured claim."""
    block = reproducibility_block(
        corpus_sha="abc", prompt_sha="def", config_hash="ghi",
    )
    assert block["seeds"] == [13, 42, 7]
    assert "byte-identical generations ACROSS machines" in block["what_is_not_promised"]
    assert "stage-E fingerprint" in block["reading"]


# -- B4 ----------------------------------------------------------------------


def _questions(n_per_stratum=6):
    out = []
    for conv in ("c1", "c2", "c3"):
        for category in (1, 2, 3, 4):
            for i in range(n_per_stratum):
                out.append({
                    "question_id": f"{conv}/cat{category}/q{i}",
                    "conv_id": conv,
                    "category": category,
                    "adversarial": False,
                })
    out.append({"question_id": "c1/adv/q0", "conv_id": "c1", "category": 5,
                "adversarial": True})
    return out


def test_the_ceiling_sample_is_seeded_stratified_and_exact(runner):
    """B4. A uniform sample of 100 is mostly single-hop questions from whichever
    conversations are largest — ceilings 1 and 2 are conversation-level and the
    categories differ ninefold in size, so the mean would describe the sample's
    composition rather than the corpus."""
    questions = _questions()

    a = runner.ceiling_sample(questions, 24, seed=13)
    b = runner.ceiling_sample(questions, 24, seed=13)
    assert a == b, "same seed, same sample"
    assert len(a) == 24, "the size is exact, not 'about N'"

    chosen = [q for q in questions if q["question_id"] in a]
    assert {q["category"] for q in chosen} == {1, 2, 3, 4}, "every category present"
    assert {q["conv_id"] for q in chosen} == {"c1", "c2", "c3"}, "every conversation"

    assert not any(q["adversarial"] for q in chosen), (
        "adversarial questions have no gold evidence; the ceiling pass skips "
        "them anyway and including them would shrink the real sample silently"
    )


def test_a_sample_larger_than_the_corpus_is_every_eligible_question(runner):
    questions = _questions()
    everything = runner.ceiling_sample(questions, 10_000, seed=13)
    assert len(everything) == sum(1 for q in questions if not q["adversarial"])


# -- B3 ----------------------------------------------------------------------


def test_the_gate_loader_refuses_a_mask_that_disagrees_with_the_weights(tmp_path):
    """A gate restored without its arm mask reads the wrong columns and still
    returns confident-looking probabilities. If the widths happen to agree, the
    numbers are wrong and nothing crashes — so the loader checks."""
    import torch

    from graft.gate.model import build_gate

    spec = importlib.util.spec_from_file_location(
        "phase8_gate", REPO / "scripts" / "phase8_gate.py"
    )
    gate_mod = importlib.util.module_from_spec(spec)
    sys.modules["phase8_gate"] = gate_mod
    spec.loader.exec_module(gate_mod)

    model = build_gate("lr", 3, seed=13)
    good = tmp_path / "good.pt"
    torch.save(
        {
            "state_dict": model.state_dict(), "arm": "pool_only", "model": "lr",
            "seed": 13, "mask": [True, True, True, False], "in_dim": 3,
            "feature_names": ["a", "b", "c", "d"], "threshold": 0.5,
        },
        good,
    )
    restored, mask, blob = gate_mod.load_gate(good)
    assert int(np.asarray(mask).sum()) == 3 and blob["arm"] == "pool_only"

    bad = tmp_path / "bad.pt"
    torch.save(
        {
            "state_dict": model.state_dict(), "arm": "pool_only", "model": "lr",
            "seed": 13, "mask": [True, True, False, False], "in_dim": 3,
            "feature_names": ["a", "b", "c", "d"], "threshold": 0.5,
        },
        bad,
    )
    with pytest.raises(ValueError, match="disagree about what the gate reads"):
        gate_mod.load_gate(bad)


def test_ceilings_sample_without_ceilings_is_refused_not_ignored(runner):
    """A flag that silently does nothing is the §1.3 defect: the run would go
    unsampled and the artefact would report `sampled: false` — a run the caller
    did not ask for, recorded as one they did."""
    args = runner.build_parser().parse_args(["--ceilings-sample", "100"])
    assert args.ceilings_sample == 100 and args.ceilings is False, (
        "the parser accepts the combination; the refusal is in main(), before "
        "any model loads"
    )
    source = (REPO / "scripts" / "locomo_eval.py").read_text(encoding="utf-8")
    assert "--ceilings-sample needs --ceilings" in source


def _posthoc():
    spec = importlib.util.spec_from_file_location(
        "locomo_gate_posthoc", REPO / "scripts" / "locomo_gate_posthoc.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["locomo_gate_posthoc"] = module
    spec.loader.exec_module(module)
    return module


def test_a_gate_arm_is_refused_when_the_rows_lack_the_block_it_reads():
    """**The check a feature-NAME comparison cannot make**, and the one that
    caught a real defect on 21 Aug 2026.

    LoCoMo eval rows carry all 435 feature names, 384 of them ``q_emb_*`` — and
    every one of those columns is exactly 0.0, because the runner records
    ``question_embedding: False``. The Phase-8 winner is the ``with_question``
    arm, which was trained to use them. Applied to those rows it reads 384 zeros
    and returns probabilities that look entirely normal. Names matching is not
    features matching.
    """
    mod = _posthoc()
    rows = [{
        "question_id": "q1",
        "gate_blocks_present": {
            "slot_coverage": True, "channel_scores": True, "pool_shape": True,
            "saturation": True, "question_embedding": False,
        },
    }]

    with pytest.raises(SystemExit, match="pool_only"):
        mod.check_blocks(rows, {"arm": "with_question",
                                "requires_blocks": ["slot_coverage", "question_embedding"]})

    # The arm whose blocks the rows DO carry passes.
    mod.check_blocks(rows, {"arm": "pool_only",
                            "requires_blocks": ["slot_coverage", "channel_scores"]})


def test_the_value_guard_catches_a_row_that_mislabels_its_own_blocks():
    """`check_blocks` trusts the row's flag. This trusts the numbers, so a row
    claiming a block it did not populate is still caught — which is what
    actually fired on the pre-`requires_blocks` checkpoint."""
    mod = _posthoc()
    blob = {"arm": "with_question"}

    live = np.ones((4, 4), dtype=np.float32)
    mask = np.array([True, True, True, True])
    mod.check_selected_columns_are_populated(live, mask, blob)  # no raise

    mostly_dead = np.zeros((4, 4), dtype=np.float32)
    mostly_dead[:, 0] = 1.0
    with pytest.raises(SystemExit, match="never populated"):
        mod.check_selected_columns_are_populated(mostly_dead, mask, blob)

    with pytest.raises(SystemExit, match="empty columns"):
        mod.check_selected_columns_are_populated(
            np.zeros((4, 4), dtype=np.float32), mask, blob
        )


def test_the_float32_sigmoid_does_not_collapse_the_gate_ranking():
    """**A correction, kept because the failure mode is the reusable part.**

    This test was first written to prove a defect that does not exist: that
    `predict`'s float32 sigmoid underflowed LoCoMo's very negative logits to
    exactly 0.0 and tied every question. Measured on the real checkpoint,
    `predict` returns 1,974 distinct values across 1,986 rows and no exact
    zeros — the smallest is 7.66e-36, inside float32's subnormal range. The
    "collapse" was a `:.4f` print format rendering 1.3e-14 as `0.0000`.

    So the assertion is inverted: the ranking is INTACT under float32, and this
    guards against anyone "fixing" a bug that was never there.
    """
    mod = _posthoc()

    class _Saturated:
        """Logits as extreme as the real ones, ordered."""

        def __call__(self, t):
            import torch

            return torch.linspace(-81.0, -32.0, t.shape[0], dtype=torch.float64)

        def __getattr__(self, name):
            return lambda *a, **k: None

    x = np.zeros((64, 4), dtype=np.float32)
    mask = np.array([True, True, True, True])
    scores, probability, diag = mod.gate_scores(_Saturated(), x, mask)

    assert diag["float32_ranking_is_intact"] is True, (
        "sigmoid(-81) is 6.6e-36, which float32 represents; it does not underflow"
    )
    assert diag["distinct_probabilities_in_float32"] == 64
    assert len(np.unique(scores)) == 64 and np.all(np.diff(scores) > 0)
    assert probability.max() < 1e-13, (
        "the probabilities are real but tiny -- which is why a 0.60 threshold "
        "is unreachable, and that IS the transfer finding"
    )


def test_auroc_averages_ranks_over_ties():
    """A real bug this test found. With `argsort` tie-breaking, a gate that
    scores every question identically reads as AUROC 0.0 or 1.0 depending only
    on input order — a system with no ranking scoring as a perfect one. That is
    the degenerate case a transfer study is most likely to hit."""
    mod = _posthoc()
    labels = np.array([1.0] * 50 + [0.0] * 50)

    assert mod.auroc(np.concatenate([np.ones(50), np.zeros(50)]), labels) == pytest.approx(1.0)
    assert mod.auroc(np.concatenate([np.zeros(50), np.ones(50)]), labels) == pytest.approx(0.0)
    assert mod.auroc(np.zeros(100), labels) == pytest.approx(0.5), (
        "no ranking must read as exactly chance"
    )
    # A half-tied case: the tied block contributes 0.5 per pair.
    half = np.concatenate([np.ones(50), np.ones(25), np.zeros(25)])
    assert 0.5 < mod.auroc(half, labels) < 1.0
