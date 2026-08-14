"""The Stage-A write path (P5.8) — exit criteria 1, 2, 3, 4, 8, 14, 15.

**Every test here runs without a GPU**, which is the whole reason the build order
puts the write path (step 2) before the extractor (step 3): what Phase 6 depends
on must not wait on GPU work.  A ``ReplayExtractor`` serves recorded extractions
and a ``StubVerifier`` returns fixed scores — neither makes any claim about
extraction or entailment *quality*, and every assertion below is about plumbing:
the record is stored, the verdict is written, the digest does not move.
"""

from __future__ import annotations

import json

import pytest

from graft.config import Config
from graft.eventlog import EventLog
from graft.graphstore import ReplayGraphStore
from graft.ingest.extractor import ReplayExtractor
from graft.ingest.nli import StubVerifier
from graft.ingest.pipeline import (
    IngestPipeline,
    ingested_turn_ids,
    unverified_assertions,
)
from graft.ingest.summary import RollingSummary
from graft.ledger import Ledger
from graft.schemas import Turn

CFG = Config(profile="real")

TURNS = [
    Turn(
        turn_id="lme_s/q1/s1/0",
        conv_id="q1",
        session_id="s1",
        speaker="user",
        ts="2023-05-20T02:21:00+00:00",
        text="I finally moved to Yokohama last March, and the rent is 92000 yen.",
    ),
    Turn(
        turn_id="lme_s/q1/s1/1",
        conv_id="q1",
        session_id="s1",
        speaker="assistant",
        ts="2023-05-20T02:21:00+00:00",
        text="That sounds like a good deal for the area.",
    ),
    Turn(
        turn_id="lme_s/q1/s1/2",
        conv_id="q1",
        session_id="s1",
        speaker="user",
        ts="2023-05-20T02:21:00+00:00",
        text="It is. I share it with Priya, who works at the aquarium.",
    ),
]

RECORDED = {
    "lme_s/q1/s1/0": {
        "turn_id": "lme_s/q1/s1/0",
        "mentions": [{"text": "Yokohama"}],
        "assertions": [
            {
                "kind": "event",
                "text_norm": "The user moved to Yokohama in March 2023.",
                "quotes": [{"turn_offset": 0, "text": "moved to Yokohama last March"}],
            },
            {
                "kind": "value",
                "text_norm": "The user's rent is 92000 yen.",
                "quotes": [{"turn_offset": 0, "text": "the rent is 92000 yen"}],
            },
        ],
    },
    "lme_s/q1/s1/2": {
        "turn_id": "lme_s/q1/s1/2",
        "mentions": [{"text": "Priya"}],
        "assertions": [
            {
                "kind": "claim",
                "text_norm": "Priya works at the aquarium and shares the flat.",
                # Cross-turn provenance (G9): one span here, one in turn 0.
                "quotes": [
                    {"turn_offset": 0, "text": "I share it with Priya, who works at the aquarium"},
                    {"turn_offset": -2, "text": "moved to Yokohama last March"},
                ],
            }
        ],
    },
}


def _pipeline(log, *, scores=1.0, summary=None, ledger=None):
    return IngestPipeline(
        log,
        CFG,
        ReplayExtractor(RECORDED),
        StubVerifier(scores),
        summary=summary,
        ledger=ledger,
    )


@pytest.fixture()
def log(tmp_path):
    handle = EventLog.open(tmp_path / "events.jsonl")
    yield handle
    handle.close()


# -- criterion 1: spans resolve on a fresh replay --------------------------


def test_every_stored_span_resolves_against_the_raw_turn_on_replay(log):
    """The architecture's own exit criterion, as a test.

    Not "a span exists" but "the offsets select the text the record claims" —
    the difference is the whole value of provenance, and it is the property
    rungs 2 and 3 could break.
    """
    _pipeline(log).run(TURNS)

    snapshot = ReplayGraphStore(log).at()
    assert snapshot.counts()["assertions"] == 3
    seen = 0
    for event in log.replay():
        if event.op != "assertion.add":
            continue
        assertion = snapshot.assertion(event.payload["assertion_id"])
        assert assertion is not None
        assert assertion.spans, "an assertion with no spans must not be storable"
        for span_id in assertion.spans:
            span = snapshot.span(span_id)
            assert span is not None, f"span {span_id} is referenced but not stored"
            turn = snapshot.turn(span.turn_id)
            assert turn is not None, "a span whose turn is missing has no provenance"
            assert 0 <= span.start < span.end <= len(turn.text)
            assert turn.text[span.start : span.end].strip()
            seen += 1
    assert seen >= 4


def test_cross_turn_provenance_is_stored_as_two_spans_on_two_turns(log):
    """Plan §3.1: "a claim assembled across turns records every supporting span,
    not one" — the shape Tier B always permitted and nothing had exercised."""
    _pipeline(log).run(TURNS)
    snapshot = ReplayGraphStore(log).at()

    multi = [
        a
        for a in (
            snapshot.assertion(e.payload["assertion_id"])
            for e in log.replay()
            if e.op == "assertion.add"
        )
        if a is not None and len(a.spans) > 1
    ]
    assert len(multi) == 1
    turns = {snapshot.span(s).turn_id for s in multi[0].spans}
    assert turns == {"lme_s/q1/s1/0", "lme_s/q1/s1/2"}


# -- criterion 2: idempotence ----------------------------------------------


def test_re_ingesting_changes_neither_digest_and_skips_every_turn(log):
    """G4's decision 8, measured: skip-and-count at ``turn_id``."""
    _pipeline(log).run(TURNS)
    log_digest = log.digest()
    graph_digest = ReplayGraphStore(log).at().state_digest()

    again = _pipeline(log)
    report = again.run(TURNS)

    assert report["turns_skipped"] == len(TURNS)
    assert report["turns_processed"] == 0
    assert log.digest() == log_digest
    assert ReplayGraphStore(log).at().state_digest() == graph_digest


def test_two_runs_of_the_same_work_produce_the_same_digest(tmp_path):
    """Per-machine determinism (G11).  With a replay extractor this is exact;
    the live claim is the same property with greedy decoding, and the pilot
    asserts it by running twice."""
    digests = []
    for name in ("a", "b"):
        handle = EventLog.open(tmp_path / f"{name}.jsonl")
        _pipeline(handle).run(TURNS)
        digests.append(handle.digest())
        handle.close()
    assert digests[0] == digests[1]


def test_ingested_turn_ids_is_the_idempotence_key(log):
    _pipeline(log).run(TURNS)
    assert ingested_turn_ids(log) == {t.turn_id for t in TURNS}


# -- crash resume ----------------------------------------------------------


def test_a_run_that_died_before_verifying_is_finished_by_re_running(log):
    """The failure the log-derived pass 2 exists to prevent: a re-run skips every
    turn, so a run-local pending list would leave those assertions quarantined by
    omission — a silent recall loss with nothing to notice it."""
    first = _pipeline(log)
    first.extract_slice(TURNS)  # pass 1 only: the process "dies" here
    assert unverified_assertions(log)

    second = _pipeline(log)
    report = second.run(TURNS)
    assert report["turns_skipped"] == len(TURNS)
    assert report["verdicts_written"] == 3
    assert report["resumed_drafts"] == 3
    assert unverified_assertions(log) == []


def test_a_turn_interrupted_mid_pass_is_reprocessed_not_skipped(log):
    """``turn.add`` is written last on purpose.  Written first, a crash between
    the turn and its assertions would make the re-run skip the turn and lose that
    evidence permanently."""
    pipeline = _pipeline(log)
    pipeline.extract_slice(TURNS[:1])
    ops = [op for _, op in ReplayGraphStore(log).iter_ops()]
    assert ops[-1] == "turn.add"
    assert ops.index("assertion.add") < len(ops) - 1


# -- criterion 4: provenance and verdicts by construction ------------------


def test_no_eligibility_without_a_stored_verdict(log):
    """Phase-0 §2.10 on real data: every stored assertion gets an explicit
    verdict event, including the ones the gate quarantines."""
    _pipeline(log).run(TURNS)
    stored = {e.payload["assertion_id"] for e in log.replay() if e.op == "assertion.add"}
    decided = {
        e.payload["assertion_id"]
        for e in log.replay()
        if e.op == "assertion.set_eligibility"
    }
    assert stored == decided


def test_the_verifier_writes_flags_and_the_gate_writes_the_verdict(log):
    """Two ops, because they are two components with two authorities (fix F9:
    the verifier never blocks storage; the gate decides eligibility)."""
    _pipeline(log, scores=0.95).run(TURNS)
    ops = [e.op for e in log.replay()]
    assert "assertion.set_flags" in ops
    assert ops.index("assertion.set_flags") < ops.index("assertion.set_eligibility")

    snapshot = ReplayGraphStore(log).at()
    for aid in {e.payload["assertion_id"] for e in log.replay() if e.op == "assertion.add"}:
        assertion = snapshot.assertion(aid)
        assert assertion.flags.entailed_score == pytest.approx(0.95)
        assert assertion.flags.entailed_by_span is True
        assert snapshot.is_eligible(aid)


def test_an_assertion_is_stored_quarantined_before_the_verifier_has_an_opinion(log):
    """The audit layer.  Storage happens first and unconditionally; that is what
    makes extraction quality measurable at all (fix F9)."""
    pipeline = _pipeline(log)
    pipeline.extract_slice(TURNS)
    snapshot = ReplayGraphStore(log).at()
    for aid in {e.payload["assertion_id"] for e in log.replay() if e.op == "assertion.add"}:
        assert snapshot.assertion(aid).eligibility == "quarantined"
        assert not snapshot.is_eligible(aid)


# -- criterion 8: the gate and the quarantine breakdown --------------------


def test_a_score_below_tau_nli_quarantines_but_does_not_block_storage(log):
    report = _pipeline(log, scores=0.4).run(TURNS)
    snapshot = ReplayGraphStore(log).at()

    assert snapshot.counts()["assertions"] == 3
    assert snapshot.counts()["eligible_assertions"] == 0
    assert report["support"]["quarantine_rate"] == 1.0
    assert report["support"]["gating_causes"]["nli_below_threshold"] == 3


def test_the_threshold_is_inclusive_at_tau_nli(log):
    """``tau_nli = 0.8`` must be satisfiable by a model scoring exactly 0.8, or
    the config's own ``tau_nli <= 1`` bound would be unreachable at 1.0."""
    _pipeline(log, scores=CFG.tau_nli).run(TURNS)
    snapshot = ReplayGraphStore(log).at()
    assert snapshot.counts()["eligible_assertions"] == 3


def test_the_quarantine_report_carries_the_reading_not_just_the_rate(log):
    """Architecture F9's sentence ships beside the number, so a later reader
    cannot take the rate as a threshold to tune."""
    report = _pipeline(log, scores=0.4).run(TURNS)
    assert "never grounds for quietly lowering tau_nli" in report["support"]["reading"]


def test_a_mixed_run_reports_both_sides(log):
    scores = {
        "The user moved to Yokohama in March 2023.": 0.99,
        "The user's rent is 92000 yen.": 0.10,
        "Priya works at the aquarium and shares the flat.": 0.85,
    }
    report = _pipeline(log, scores=scores).run(TURNS)
    assert report["support"]["eligible"] == 2
    assert report["support"]["quarantined"] == 1
    assert report["support"]["quarantine_rate"] == pytest.approx(1 / 3)


def test_the_nli_premise_is_the_spans_and_nothing_else(log):
    """G6's semantics, asserted rather than promised: if the turn or the
    conversation were in the premise, ``entailed_by_span`` would stop meaning
    what its name says."""
    seen: list[str] = []

    class _Recording(StubVerifier):
        def score(self, pairs):
            seen.extend(p for p, _ in pairs)
            return super().score(pairs)

    IngestPipeline(log, CFG, ReplayExtractor(RECORDED), _Recording(1.0)).run(TURNS)
    joined = " || ".join(seen)
    assert "moved to Yokohama last March" in joined
    assert "That sounds like a good deal" not in joined
    for premise in seen:
        assert "I finally moved" not in premise  # the whole turn is not the premise


# -- grounding failures are drops, not quarantines -------------------------


def test_an_ungrounded_assertion_is_dropped_and_counted_outside_the_gate(log):
    """It was never stored — ``Assertion`` refuses to exist without spans — so
    counting it as quarantined would inflate a rate that measures the gate."""
    records = {
        "lme_s/q1/s1/0": {
            "turn_id": "lme_s/q1/s1/0",
            "mentions": [],
            "assertions": [
                {
                    "kind": "claim",
                    "text_norm": "Something the turn never said.",
                    "quotes": [{"turn_offset": 0, "text": "the quarterly revenue forecast"}],
                }
            ],
        }
    }
    report = IngestPipeline(
        log, CFG, ReplayExtractor(records), StubVerifier(1.0)
    ).run(TURNS[:1])
    assert report["assertions_stored"] == 0
    assert report["assertions_dropped_ungrounded"] == 1
    assert report["support"]["dropped_before_gate_grounding_failure"] == 1
    assert report["support"]["stored"] == 0


# -- criterion 14: metering ------------------------------------------------


def test_the_ledger_separates_the_two_stages(log):
    """One ``ledger.stage()`` per pass, which is what makes extraction cost and
    verification cost separable rather than one total (F7's stage-sequential
    execution, measured)."""
    ledger = Ledger.from_config(CFG, log=None)
    _pipeline(log, ledger=ledger).run(TURNS)
    stages = ledger.snapshot()["stages"]
    assert set(stages) == {"extract", "verify"}


def test_metrics_rows_carry_one_row_per_turn_including_skips(log):
    pipeline = _pipeline(log)
    pipeline.run(TURNS)
    rows = pipeline.metrics_rows()
    assert len(rows) == len(TURNS)
    assert all("turn_id" in row and "rungs" in row for row in rows)
    assert json.dumps(rows)  # the rows must be serialisable as they stand


# -- the summary (G3) -------------------------------------------------------


def test_the_summary_is_never_written_to_the_event_log(log):
    """Derived state, not evidence.  Writing it would put model prose into a
    provenance chain that `H`'s scope sub-check has to walk to a ``conv_id``."""
    calls: list[str] = []

    def summarizer(system, user):
        calls.append(user)
        return "- the user lives in Yokohama", 20, 8

    summary = RollingSummary(summarizer, every=2)
    _pipeline(log, summary=summary).run(TURNS)

    assert calls, "the summariser should have been called at the refresh point"
    for event in log.replay():
        assert "summary" not in json.dumps(event.payload).lower()


def test_the_summary_refreshes_on_the_declared_cadence_not_per_turn(log):
    calls: list[str] = []

    def summarizer(system, user):
        calls.append(user)
        return "- a bullet", 10, 4

    summary = RollingSummary(summarizer, every=2)
    _pipeline(log, summary=summary).run(TURNS)
    # Three turns at s = 2: turns 0 and 1 share the empty summary, turn 2 forces
    # exactly one refresh.
    assert len(calls) == 1


# -- the 13-14 Aug 2026 audit's regressions ----------------------------------


def test_mentions_are_persisted_as_events_phase_6_can_read(log):
    """Before the audit a mention survived only as a bare ``span.add`` plus a
    per-turn count — indistinguishable from a quote span, while plan §0/§8
    promise D1 items derive from Stage-A mentions.  The permanent record now
    carries them."""
    from graft.ingest.pipeline import mentions_of

    _pipeline(log).run(TURNS)
    snapshot = ReplayGraphStore(log).at()

    mentions = mentions_of(log)
    assert {m["text"] for m in mentions} == {"Yokohama", "Priya"}
    for m in mentions:
        span = snapshot.span(m["span_id"])
        assert span is not None, "a mention names a span that must be stored"
        turn = snapshot.turn(m["turn_id"])
        assert turn.text[span.start : span.end] == m["text"]
        assert m["rung"] in ("exact", "normalised", "fuzzy")


def test_mention_events_do_not_duplicate_on_re_ingestion(log):
    from graft.ingest.pipeline import mentions_of

    _pipeline(log).run(TURNS)
    first = mentions_of(log)
    _pipeline(log).run(TURNS)
    assert mentions_of(log) == first


def test_mention_events_are_not_graph_state(log):
    """``GRAPH_OPS`` excludes ``mention.add`` deliberately: no ``Mention`` node
    exists until Phase 6's D1 decides.  Replay must ignore the op, not crash."""
    from graft.graphstore import GRAPH_OPS

    assert "mention.add" not in GRAPH_OPS
    _pipeline(log).run(TURNS)
    snapshot = ReplayGraphStore(log).at()  # would raise if replay touched it
    assert snapshot.counts()["assertions"] == 3


def test_truncated_generations_reach_the_report(log):
    """Generation-level, not final-attempt: a first-attempt truncation repaired
    on the retry must not disappear from the count."""

    class _Truncating(ReplayExtractor):
        def extract(self, turn, context):
            got = super().extract(turn, context)
            got.truncations = 2  # two capped generations behind one turn
            got.truncated = False  # ...whose final attempt parsed
            return got

    pipeline = IngestPipeline(log, CFG, _Truncating(RECORDED), StubVerifier(1.0))
    report = pipeline.run(TURNS)
    assert report["truncated_generations"] == 2 * len(TURNS)


def test_a_resumed_summary_rebuilds_the_chain_it_did_not_watch(tmp_path):
    """A resumed run skips ingested turns, so nothing walks the earlier refresh
    points — the old code silently defaulted the previous summary to "" and the
    resumed prompt differed from an uninterrupted run's (a G11 determinism break
    on exactly the crash path G4 promises to repair)."""
    calls: list[str] = []

    def summarizer(system, user):
        calls.append(user)
        return f"- bullet {len(calls)}", 10, 4

    turns = [
        Turn(
            turn_id=f"lme_s/q1/s1/{i}",
            conv_id="q1",
            session_id="s1",
            speaker="user",
            ts="2023-05-20T02:21:00+00:00",
            text=f"turn number {i}",
        )
        for i in range(25)
    ]

    # A cold instance asked for turn 25's context: refresh points 10 and 20 are
    # both missing and both must be rebuilt, chained.
    cold = RollingSummary(summarizer, cache_dir=tmp_path, every=10)
    text = cold.summary_for("q1", turns, 25)
    assert text == "- bullet 2"
    assert len(calls) == 2
    assert "SUMMARY OF EVERYTHING BEFORE" in calls[1]
    assert "SUMMARY OF EVERYTHING BEFORE" not in calls[0]


def test_the_summary_cache_is_written_through_not_flushed_at_exit(tmp_path):
    """The cache must survive the crash that makes the resume necessary; a
    flush-at-end cache dies with the process."""
    summary = RollingSummary(lambda s, u: ("- a bullet", 10, 4), cache_dir=tmp_path, every=2)
    turns = TURNS
    summary.summary_for("q1", turns, 2)
    # No flush() call — the refresh itself must have persisted the cache.
    reread = RollingSummary(None, cache_dir=tmp_path, every=2)
    assert reread.summary_for("q1", turns, 2) == "- a bullet"


def test_a_cross_turn_duplicate_gives_the_same_graph_whether_or_not_a_crash_intervened(tmp_path):
    """The dedup is scoped to the turn, not the run — because the stored record
    takes ``asserted_by`` and ``t_created`` from the turn that *writes* it while
    the content-derived id covers neither.  A run-scoped set kept the first
    turn's speaker and timestamp on an uninterrupted run and the last turn's on
    a resumed one, and replay is last-write-wins: the same corpus produced two
    different graphs, falsifying this module's own idempotence claim."""
    turns = [
        Turn(turn_id="lme_s/q1/s1/0", conv_id="q1", session_id="s1", speaker="user",
             ts="2023-05-20T02:21:00+00:00", text="I moved to Yokohama last March."),
        Turn(turn_id="lme_s/q1/s2/0", conv_id="q1", session_id="s2", speaker="assistant",
             ts="2023-05-21T09:00:00+00:00", text="Right, the move."),
        Turn(turn_id="lme_s/q1/s3/0", conv_id="q1", session_id="s3", speaker="user",
             ts="2023-05-27T11:00:00+00:00", text="As I said before."),
    ]
    # Turns 1 and 2 restate one claim, both quoting turn 0's span: same kind,
    # same text_norm, same span ids, therefore the same assertion_id.
    same = {"kind": "event", "text_norm": "The user moved to Yokohama in March 2023."}
    records = {
        "lme_s/q1/s2/0": {**same, "turn_id": "lme_s/q1/s2/0", "mentions": [],
                          "assertions": [{**same, "quotes": [
                              {"turn_offset": -1, "text": "moved to Yokohama last March"}]}]},
        "lme_s/q1/s3/0": {**same, "turn_id": "lme_s/q1/s3/0", "mentions": [],
                          "assertions": [{**same, "quotes": [
                              {"turn_offset": -2, "text": "moved to Yokohama last March"}]}]},
    }

    straight = EventLog.open(tmp_path / "a.jsonl")
    IngestPipeline(straight, CFG, ReplayExtractor(records), StubVerifier(1.0)).run(turns)
    expected = ReplayGraphStore(straight).at().state_digest()
    straight.close()

    crashed = EventLog.open(tmp_path / "b.jsonl")
    first = IngestPipeline(crashed, CFG, ReplayExtractor(records), StubVerifier(1.0))
    first.extract_slice(turns[:2])          # the process "dies" after turn 1
    second = IngestPipeline(crashed, CFG, ReplayExtractor(records), StubVerifier(1.0))
    second.run(turns)                        # re-run: turns 0-1 skip, turn 2 lands
    resumed = ReplayGraphStore(crashed).at().state_digest()
    crashed.close()

    assert resumed == expected


def test_a_within_turn_duplicate_is_still_suppressed_and_now_counted(log):
    """The dedup's stated intent, kept — and the drop is reported, so the three
    dispositions account for every extraction."""
    records = {
        "lme_s/q1/s1/0": {
            "turn_id": "lme_s/q1/s1/0",
            "mentions": [],
            "assertions": [
                {"kind": "value", "text_norm": "The user's rent is 92000 yen.",
                 "quotes": [{"turn_offset": 0, "text": "the rent is 92000 yen"}]},
                {"kind": "value", "text_norm": "The user's rent is 92000 yen.",
                 "quotes": [{"turn_offset": 0, "text": "the rent is 92000 yen"}]},
            ],
        }
    }
    report = IngestPipeline(
        log, CFG, ReplayExtractor(records), StubVerifier(1.0)
    ).run(TURNS[:1])

    assert report["assertions_extracted"] == 2
    assert report["assertions_stored"] == 1
    assert report["assertions_duplicate_within_turn"] == 1
    assert (
        report["assertions_stored"]
        + report["assertions_dropped_ungrounded"]
        + report["assertions_duplicate_within_turn"]
        == report["assertions_extracted"]
    )


def test_a_resumed_draft_reports_unknown_repair_provenance_not_a_clean_one(log):
    """The repair count is not in the log, so a rebuilt draft must not claim
    ``from_repair=False`` — the cross-tab would read as measured zeros."""
    first = _pipeline(log)
    first.extract_slice(TURNS)               # pass 1 only
    report = _pipeline(log).run(TURNS)       # pass 2 rebuilds every draft

    assert report["resumed_drafts"] == 3
    assert report["support"]["repair_provenance_unknown"] == 3
    assert report["support"]["cross_tab_repaired_extraction"] == {
        "eligible": 0, "quarantined": 0
    }


def test_a_torn_summary_cache_is_a_cache_miss_not_a_crash(tmp_path):
    """The class promises deleting the file changes throughput and nothing
    else; a half-written one must not kill the next run at construction — the
    crash path the write-through flush exists to serve."""
    (tmp_path / "summaries.json").write_text('{"q1#10": "- a bul', encoding="utf-8")
    summary = RollingSummary(lambda s, u: ("- rebuilt", 1, 1), cache_dir=tmp_path, every=10)
    assert summary.corrupt_cache is True
    assert summary._memory == {}

    # And the flush that follows must leave a readable file behind.
    summary._memory[("q1", 10)] = "- rebuilt"
    summary.flush()
    assert RollingSummary(None, cache_dir=tmp_path, every=10)._memory == {("q1", 10): "- rebuilt"}
