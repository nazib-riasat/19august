"""The corpus adapter, the extractor interface, the pins and the bakeoff rule.

Exit criteria 3, 5, 10, 13, 15 and the parts of the build order that do not
need a model.  The corpus tests skip when the raw file is absent — it is 265 MB,
gitignored, and re-fetched per machine — but the SHA pin itself is checked
unconditionally, because a pin that only exists on machines that already have
the file is not a pin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graft.ingest import corpus, pins
from graft.ingest.bakeoff import CandidateResult, decide
from graft.ingest.extractor import ExtractionContext, ReplayExtractor, parse_extraction
from graft.ingest.records import Quote, RawAssertion, RawExtraction
from graft.schemas import ASSERTION_KINDS, SCHEMA_VERSION, Assertion, Turn

REPO = Path(__file__).resolve().parents[2]


# -- the corpus pin ---------------------------------------------------------


def test_the_corpus_pin_matches_the_spike_exactly():
    """One value, two homes, checked rather than trusted.

    ``graft.ingest`` must not import from ``scripts/``, and the Phase-2.5 tooling
    must keep running unchanged, so the SHA genuinely lives in both places.  The
    project's answer to that is always the same: assert they agree.
    """
    source = (REPO / "scripts" / "phase2_5" / "common.py").read_text(encoding="utf-8")
    assert f'CORPUS_SHA256 = "{corpus.CORPUS_SHA256}"' in source
    assert f'CORPUS_LICENCE = "{corpus.CORPUS_LICENCE}"' in source


def test_turn_ids_follow_the_phase_2_5_convention():
    """A Phase-5 record and a spike label must point at the same turn, or the
    guidelines and the flagged items cannot be carried forward (§8)."""
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "phase2_5"))
    try:
        import common  # type: ignore
    finally:
        sys.path.pop(0)
    assert corpus.turn_id_for("q1", "s1", 3) == common.turn_id("q1", "s1", 3)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2023/05/20 (Sat) 02:21", "2023-05-20T02:21:00+00:00"),
        ("2023/05/20 (Sat) 00:00", "2023-05-20T00:00:00+00:00"),
    ],
)
def test_corpus_timestamps_read_as_utc_by_declared_convention(raw, expected):
    """The corpus carries no timezone.  Reading it as UTC is a *choice*, and a
    silent localtime read would make ingestion machine-dependent."""
    assert corpus.parse_corpus_ts(raw) == expected


def test_an_unrecognised_timestamp_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        corpus.parse_corpus_ts("last Tuesday")


# -- the corpus itself (skipped without the file) --------------------------

_CORPUS = REPO / "data" / "phase2_5" / "raw" / "longmemeval_s"
needs_corpus = pytest.mark.skipif(
    not _CORPUS.is_file(), reason="the pinned corpus is gitignored and fetched per machine"
)


@needs_corpus
def test_the_pinned_corpus_verifies_and_has_the_declared_shape():
    data = corpus.load_corpus(_CORPUS)
    assert len(data) == 500
    assert set(corpus.QUESTION_TYPES) == {inst["question_type"] for inst in data}


@needs_corpus
def test_turns_re_derive_byte_identically_from_the_pinned_corpus():
    """Build-order step 1's exit condition: the 2.5 sample's turns come back
    unchanged through the Phase-5 adapter."""
    sample = json.loads(
        (REPO / "data" / "phase2_5" / "sample.json").read_text(encoding="utf-8")
    )
    index = corpus.question_index(corpus.load_corpus(_CORPUS))

    checked = 0
    for session in sample["sessions"]:
        instance = index[session["question_id"]]
        by_id = {t.turn_id: t for t in corpus.turns_of(instance, [session["session_id"]])}
        for turn in session["turns"]:
            got = by_id[turn["turn_id"]]
            assert got.text == turn["text"]
            assert got.speaker == turn["role"]
            assert got.conv_id == session["question_id"]
            assert got.ts == corpus.parse_corpus_ts(session["session_date"])
            checked += 1
    assert checked == sample["provenance"]["n_turns"]


@needs_corpus
def test_the_question_sidecar_holds_the_gold_and_the_turn_stream_does_not():
    """``answer_session_ids`` is a gold label.  A pipeline that wrote it into the
    log would build the graph with the answer already in it, and every ceiling
    measured afterwards would be meaningless."""
    instance = corpus.load_corpus(_CORPUS)[0]
    meta = corpus.question_meta(instance)
    assert "answer" in meta and "evidence_session_ids" in meta
    for turn in corpus.turns_of(instance, instance["haystack_session_ids"][:1]):
        assert meta["answer"] not in json.dumps(turn.to_dict())


@needs_corpus
def test_a_tampered_corpus_is_refused(tmp_path):
    path = tmp_path / "corpus"
    path.write_bytes(b'[{"question_id": "x"}]')
    with pytest.raises(ValueError, match="SHA mismatch"):
        corpus.load_corpus(path)


# -- the extraction schema (G9, decision 9) --------------------------------


def test_the_assertion_kind_vocabulary_is_the_frozen_one():
    assert ASSERTION_KINDS == ("claim", "value", "event", "time")
    assert pins.ASSERTION_KINDS == ASSERTION_KINDS
    # The prompt's schema — which candidate B constrains against — must not drift
    # from the code's vocabulary, or B would decode valid JSON the parser rejects.
    from graft.ingest.prompts import EXTRACTION_JSON_SCHEMA

    schema_kinds = EXTRACTION_JSON_SCHEMA["properties"]["assertions"]["items"][
        "properties"
    ]["kind"]["enum"]
    assert tuple(schema_kinds) == ASSERTION_KINDS


def test_tier_b_is_frozen_at_this_schema_version():
    """G9's freeze, pinned to a value rather than to a promise.  A later gap that
    forces a field moves this literal *and* needs a log migration, which is
    exactly the friction the freeze is for."""
    assert SCHEMA_VERSION == "0.3.0"
    assert set(Assertion.__dataclass_fields__) == {
        "assertion_id", "kind", "text_norm", "spans", "flags", "t_created", "eligibility",
    }
    assert set(Turn.__dataclass_fields__) == {
        "turn_id", "conv_id", "session_id", "speaker", "ts", "text",
    }


# -- parsing a model reply --------------------------------------------------


def test_a_fenced_reply_parses():
    got, error = parse_extraction(
        '```json\n{"mentions": [{"text": "Priya"}], "assertions": []}\n```'
    )
    assert error == "" and got is not None
    assert got.mentions == ("Priya",)


def test_a_parse_failure_returns_the_error_for_the_repair_prompt():
    """The reprompt carries the parser's complaint back, because malformed JSON
    is the failure and the error message is the one thing the model lacks."""
    got, error = parse_extraction('{"mentions": [ }')
    assert got is None
    assert error and "Expecting" in error


def test_no_json_at_all_is_a_named_failure_not_a_crash():
    got, error = parse_extraction("I think the user moved house.")
    assert got is None and "no JSON object" in error


def test_one_malformed_assertion_does_not_discard_the_others():
    """All-or-nothing on a well-formed object would recreate the loss this phase
    exists to remove."""
    got, _ = parse_extraction(
        json.dumps(
            {
                "mentions": [],
                "assertions": [
                    {"kind": "claim", "text_norm": "ok", "quotes": [{"turn_offset": 0, "text": "a"}]},
                    {"kind": "claim", "text_norm": "", "quotes": []},
                    {"kind": "nonsense", "text_norm": "still usable",
                     "quotes": [{"turn_offset": -1, "text": "b"}]},
                ],
            }
        )
    )
    assert got is not None
    assert [a.text_norm for a in got.assertions] == ["ok", "still usable"]
    assert got.assertions[1].kind == "claim"  # out-of-vocabulary kind, recorded generally


def test_a_future_turn_offset_is_refused():
    """A quote cannot come from a turn that has not been said yet."""
    with pytest.raises(ValueError):
        Quote(1, "text")
    got, _ = parse_extraction(
        json.dumps({"assertions": [
            {"kind": "claim", "text_norm": "x", "quotes": [{"turn_offset": 2, "text": "y"}]}
        ]})
    )
    assert got is not None and got.assertions == ()


def test_the_spike_single_quote_shape_still_parses():
    """Tolerated, not blessed — it is what lets the recorded spike extractions
    drive the write-path tests without rewriting the file."""
    got, _ = parse_extraction(
        json.dumps({"assertions": [{"kind": "value", "text_norm": "x", "quote": "y"}]})
    )
    assert got is not None
    assert got.assertions[0].quotes[0].turn_offset == 0


# -- the context window -----------------------------------------------------


def test_context_offsets_number_the_window_backwards_from_zero():
    turns = [
        Turn(f"t{i}", "c", "s", "user", "2023-01-01T00:00:00+00:00", f"text {i}")
        for i in range(3)
    ]
    context = ExtractionContext(window=turns[:2])
    table = context.offsets(turns[2])
    assert table[0][0] == "t2"
    assert table[-1][0] == "t1"
    assert table[-2][0] == "t0"


def test_the_replay_extractor_returns_an_empty_extraction_for_an_unknown_turn():
    turn = Turn("t9", "c", "s", "user", "2023-01-01T00:00:00+00:00", "hello")
    got = ReplayExtractor({}).extract(turn, ExtractionContext())
    assert got.parse_ok and got.assertions == () and got.mentions == ()


# -- pins and the ingestion fingerprint (G11) ------------------------------


def test_no_extractor_is_frozen_until_the_bakeoff_says_so():
    """Fails closed.  An unrun bakeoff silently falling back to a candidate
    would put an unmeasured extractor behind every downstream artefact."""
    if pins.EXTRACTOR is None:
        with pytest.raises(RuntimeError, match="phase5_bakeoff"):
            pins.require_extractor()
    else:
        frozen = pins.require_extractor()
        assert frozen["model_id"] and frozen["revision"]
        assert frozen["candidate"] in pins.EXTRACTOR_CANDIDATES


def test_every_candidate_pins_a_revision_not_a_tag():
    """A tag can be moved; a sha cannot, and "same model id" is not the same
    claim as "same weights"."""
    for name, config in pins.EXTRACTOR_CANDIDATES.items():
        assert len(config["revision"]) == 40, name
        assert set(config["revision"]) <= set("0123456789abcdef"), name


def test_a_withdrawn_candidate_stays_declared_but_cannot_be_selected():
    """Candidate C — the architecture's 7B — was withdrawn by ruling before the
    bakeoff ran, and the two halves of that are both load-bearing.

    It stays in ``EXTRACTOR_CANDIDATES`` so the predeclared candidate set is
    still readable after the fact; deleting it would leave a table that looks
    like only two candidates were ever considered.  And it is unselectable, so no
    code path can quietly reintroduce a model the project does not use.
    """
    assert pins.EXTRACTOR_CANDIDATES["C"]["withdrawn"]
    assert "C" not in pins.RUNNABLE_CANDIDATES
    assert pins.RUNNABLE_CANDIDATES == ("A", "B")
    from graft.ingest.extractor import build_extractor

    with pytest.raises(RuntimeError, match="withdrawn"):
        build_extractor("C")


def test_the_runnable_set_is_derived_not_hand_listed():
    """A withdrawal must be one edit.  Two lists is how one of them goes stale —
    the recurring defect ``CLAUDE.md`` §5 catalogues."""
    assert pins.RUNNABLE_CANDIDATES == tuple(
        name
        for name, config in sorted(pins.EXTRACTOR_CANDIDATES.items())
        if not config.get("withdrawn")
    )


def test_the_fingerprint_binds_the_config_and_moves_with_it():
    base = pins.ingestion_fingerprint()
    other = pins.ingestion_fingerprint(pins.EXTRACTOR_CANDIDATES["C"])
    assert base != other
    assert pins.ingestion_fingerprint() == base


def test_the_fingerprint_does_not_duplicate_a_config_value():
    """``tau_nli`` and ``support_policy`` already move ``config_hash``.  Giving a
    frozen value two homes is the failure mode ``CLAUDE.md`` §5 catalogues, so a
    run's identity is the *pair* of hashes."""
    frozen = json.dumps(pins.frozen_values())
    assert "tau_nli" not in frozen
    assert "support_policy" not in frozen


def test_decoding_is_greedy():
    assert pins.DECODING["do_sample"] is False


# -- the bakeoff rule (G2, decision 2) -------------------------------------


def _result(name, *, turns=60, failures=0, grounded=100, seconds=600.0, error=None):
    row = CandidateResult(name, pins.EXTRACTOR_CANDIDATES.get(name, {}))
    row.turns = turns
    row.parse_failures = failures
    row.assertions_grounded = grounded
    row.seconds = seconds
    row.error = error
    return row


def test_stage_one_is_a_hard_filter_not_a_tiebreak():
    """A candidate over the 2% ceiling is out however fast it is."""
    fast_but_lossy = _result("A", failures=6, grounded=1000, seconds=60.0)
    slow_but_clean = _result("C", failures=0, grounded=50, seconds=600.0)
    verdict = decide([fast_but_lossy, slow_but_clean])
    assert verdict["winner"] == "C"
    assert verdict["survivors"] == ["C"]


def test_stage_two_ranks_by_rate_not_by_count():
    """The candidates differ in throughput by construction, so "most assertions"
    would reward whichever is slowest to be honest about its failures."""
    many_but_slow = _result("C", grounded=120, seconds=1200.0)
    fewer_but_fast = _result("A", grounded=100, seconds=600.0)
    assert decide([many_but_slow, fewer_but_fast])["winner"] == "A"


def test_a_close_result_is_a_tie_for_the_human_sub_audit_not_a_coin_flip():
    a = _result("A", grounded=100, seconds=600.0)
    b = _result("B", grounded=1005, seconds=6030.0)
    verdict = decide([a, b])
    assert verdict["verdict"] == "tie"
    assert sorted(verdict["tied"]) == ["A", "B"]
    assert verdict["winner"] is None


def test_every_candidate_failing_the_ceiling_is_a_finding_not_a_reason_to_raise_it():
    verdict = decide([_result("A", failures=9), _result("C", failures=5)])
    assert verdict["verdict"] == "no_survivor"
    assert "not a reason to raise the ceiling" in verdict["reading"]


def test_a_candidate_that_could_not_run_records_a_row_rather_than_vanishing():
    """A missing candidate is a result — "we could not run it here, and here is
    why" — and losing the other rows to it would be the expensive failure."""
    broken = _result("B", error="RuntimeError: no grammar backend installed")
    verdict = decide([broken, _result("A")])
    assert verdict["winner"] == "A"
    table = {row["candidate"]: row for row in verdict["table"]}
    assert table["B"]["error"].startswith("RuntimeError")
    assert table["B"]["passes_parse_filter"] is False


def test_a_candidate_that_never_ran_scores_nan_not_zero():
    """Zero is a measurement; this is its absence, and they must not sort
    together."""
    import math

    assert math.isnan(_result("B", seconds=0.0, error="boom").grounded_per_minute)


def test_the_declared_ceiling_is_two_percent():
    assert pins.PARSE_FAILURE_CEILING == 0.02
    assert pins.SPAN_PRECISION_FLOOR == 0.90


# -- records ----------------------------------------------------------------


def test_worst_rung_is_the_least_exact_span_not_an_average():
    from graft.ingest.records import DraftAssertion, Grounding, GroundedQuote

    draft = DraftAssertion(
        "a1",
        "claim",
        "x",
        [
            GroundedQuote("t0", "s0", "aa", Grounding(0, 2, "exact")),
            GroundedQuote("t1", "s1", "bb", Grounding(0, 2, "fuzzy")),
        ],
        "user",
    )
    assert draft.worst_rung == "fuzzy"
    assert draft.premise == "aa bb"


def test_an_unknown_rung_sorts_as_the_worst():
    """Fail closed: an unrecoverable provenance is not evidence of a clean one."""
    from graft.ingest.records import DraftAssertion, Grounding, GroundedQuote

    draft = DraftAssertion(
        "a1",
        "claim",
        "x",
        [GroundedQuote("t0", "s0", "aa", Grounding(0, 2, "resumed"))],
        "user",
    )
    assert draft.worst_rung == "resumed"


def test_a_parse_failure_is_visible_on_the_record():
    """The spike's 15.5% was invisible precisely because a failure and an empty
    extraction looked identical downstream."""
    empty = RawExtraction()
    lost = RawExtraction(parse_ok=False, error="Expecting ',' delimiter")
    assert empty.parse_ok and not lost.parse_ok
    assert RawAssertion("claim", "x", [Quote(0, "y")]).to_dict()["quotes"][0][
        "turn_offset"
    ] == 0


# -- the 13-14 Aug 2026 audit's regressions ----------------------------------


def test_whitespace_only_quotes_are_filtered_like_whitespace_mentions():
    """The mention filter required ``strip()``; the quote filter did not, so a
    blank quote reached the grounder and 'grounded' against any blank run."""
    got, _ = parse_extraction(
        '{"mentions": [], "assertions": [{"kind": "claim", "text_norm": "x", '
        '"quotes": [{"turn_offset": 0, "text": "   "}]}]}'
    )
    assert got is not None
    assert got.assertions == ()  # its only quote was blank, so it has none


def test_context_window_turns_are_clipped_and_the_current_turn_is_not():
    """Decision 4's declared adaptation: m = 10 over essay-length turns measured
    5,017 input tokens/turn and overflowed the card.  The clip applies to the
    rendered window only — offsets keep the full text, because grounding
    resolves quotes against the raw turn."""
    long_text = "word " * 500
    turns = [
        Turn(
            turn_id="lme_s/q1/s1/0", conv_id="q1", session_id="s1",
            speaker="assistant", ts="2023-05-20T02:21:00+00:00", text=long_text,
        )
    ]
    current = Turn(
        turn_id="lme_s/q1/s1/1", conv_id="q1", session_id="s1",
        speaker="user", ts="2023-05-20T02:21:00+00:00", text="short current turn",
    )
    context = ExtractionContext(window=turns, session_date=current.ts)

    rendered = context.rendered()
    assert "[clipped]" in rendered
    assert len(rendered) < len(long_text)

    offsets = context.offsets(current)
    assert offsets[-1][1] == long_text  # full text, for the grounder
    assert offsets[0][1] == "short current turn"

    from graft.ingest.pins import CONTEXT_CLIP_CHARS

    assert ExtractionContext.clip("x" * (CONTEXT_CLIP_CHARS + 50)).endswith("[clipped]")
    short = "y" * (CONTEXT_CLIP_CHARS - 1)
    assert ExtractionContext.clip(short) == short


def test_the_grammar_never_touches_unconstrained_completions(monkeypatch):
    """``complete`` serves the summary and the obligation parser; a grammar
    compiled from the *extraction* schema would force those replies into
    extraction JSON — under candidate B every summary would be schema junk.
    (Found while wiring the summary into the bakeoff harness.)"""
    from graft.ingest import extractor as extractor_module
    from graft.ingest.extractor import LlmExtractor
    from graft.ingest.pins import EXTRACTOR_CANDIDATES

    sentinel = object()
    monkeypatch.setattr(
        extractor_module, "GrammarLogitsProcessor", lambda grammar: sentinel
    )
    llm = LlmExtractor(EXTRACTOR_CANDIDATES["B"])
    llm._grammar = object()  # as if load() had compiled it

    constrained = llm._decoding_kwargs(constrained=True)
    assert constrained["logits_processor"] == [sentinel]

    free = llm._decoding_kwargs(constrained=False)
    assert "logits_processor" not in free


def test_the_summary_token_cap_is_enforced_at_generation():
    """Decision 3 says 512 *tokens*; a 512-word backstop alone admits ~725 on
    the pinned tokenizer.  The override is the enforcement path."""
    from graft.ingest.extractor import LlmExtractor
    from graft.ingest.pins import DECODING, EXTRACTOR_CANDIDATES, SUMMARY_MAX_TOKENS

    llm = LlmExtractor(EXTRACTOR_CANDIDATES["A"])
    kwargs = llm._decoding_kwargs(constrained=False, max_new_tokens=SUMMARY_MAX_TOKENS)
    assert kwargs["max_new_tokens"] == SUMMARY_MAX_TOKENS
    assert llm._decoding_kwargs(constrained=False)["max_new_tokens"] == DECODING["max_new_tokens"]


def test_the_fingerprint_covers_the_clip_and_the_cap():
    """Both change what the model sees or may emit, so both are configuration
    identity (G11).

    Two separate assertions, because they check different things and conflating
    them made this test fail for the wrong reason when the cap moved: the first
    is *coverage* — whatever the constant is, the fingerprint tracks it — and the
    second is the *pin*, which exists so a change is deliberate rather than
    incidental.  Raising the cap is a §6 decision (`PHASE5_DECISIONS.md` §2.1a);
    editing this literal is how it gets acknowledged.
    """
    values = pins.frozen_values()
    assert values["context_clip_chars"] == pins.CONTEXT_CLIP_CHARS
    assert values["decoding"]["max_new_tokens"] == pins.DECODING["max_new_tokens"]

    # 600 → 1024 (13 Aug, instrument correction) → 2048 (14 Aug, §2.1a: the cap
    # is a runaway guard and must not bind on real content).
    assert pins.DECODING["max_new_tokens"] == 2048
    assert pins.CONTEXT_CLIP_CHARS == 600


def test_a_sole_survivor_is_decided_by_stage_one_not_stage_two():
    """The artefact records the basis of a frozen decision, so it has to name the
    stage that actually eliminated the field.

    With one survivor the throughput comparison had nothing to compare against;
    reporting "decided by stage 2" would tell a later reader the winner was
    picked on throughput when it was picked by the parse-failure filter.
    (Found 14 Aug 2026, on the run that froze decision 2.)
    """
    lossy = _result("A", failures=14)
    clean = _result("B", failures=1)
    verdict = decide([lossy, clean])
    assert verdict["winner"] == "B"
    assert verdict["survivors"] == ["B"]
    assert verdict["decided_by"].startswith("stage 1")

    # Two survivors, and stage 2 genuinely decides.
    faster = _result("B", failures=1, grounded=200, seconds=600.0)
    slower = _result("A", failures=1, grounded=50, seconds=600.0)
    two = decide([faster, slower])
    assert two["decided_by"].startswith("stage 2")
