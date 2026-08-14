"""The learned obligation parser and its time resolver (P5.7, F2, G7).

Exit criteria 11 and 12.  Criterion 12 names three expressions — "last May",
"in 2023", "yesterday" — against a fixed ``question_date``, and they are the
first three tests here.

Everything in this file runs without a GPU, which is the point of resolving
relative dates in code rather than in a prompt: the arithmetic that Phase 1's
``covered_fraction`` depends on is testable on a bare interpreter.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from graft.core import obligations as core_obligations
from graft.ingest import oblparse
from graft.ingest.oblparse import (
    ObligationParser,
    obligations_from_slots,
    parse_slots,
    unbounded_rate,
)
from graft.ingest.timeexpr import parse_question_date, resolve
from graft.schemas import Interval, Obligations

QUESTION_DATE = "2023/05/30 (Tue) 23:40"


def _utc(y: int, m: int, d: int) -> float:
    return datetime(y, m, d, tzinfo=timezone.utc).timestamp()


# -- exit criterion 12 ------------------------------------------------------


def test_yesterday_widens_to_the_whole_previous_day():
    got = resolve("yesterday", QUESTION_DATE)
    assert got == Interval(_utc(2023, 5, 29), _utc(2023, 5, 30))


def test_last_may_is_the_previous_year_when_the_question_is_asked_in_may():
    """A declared convention: "last <month>" names the most recent occurrence
    **strictly before** the month the question stands in."""
    got = resolve("last May", QUESTION_DATE)
    assert got == Interval(_utc(2022, 5, 1), _utc(2022, 6, 1))


def test_in_2023_is_the_calendar_year():
    got = resolve("in 2023", QUESTION_DATE)
    assert got == Interval(_utc(2023, 1, 1), _utc(2024, 1, 1))


def test_every_resolved_interval_is_half_open_and_non_empty():
    """A zero-width half-open interval contains nothing, so a point expression
    that failed to widen would silently make ``covered_fraction`` zero."""
    for phrase in ("yesterday", "today", "last May", "in 2023", "March 2022", "last week"):
        got = resolve(phrase, QUESTION_DATE)
        assert got is not None, phrase
        assert not got.is_empty, phrase
        assert got.start is not None and got.end is not None and got.end > got.start


# -- the rest of the declared vocabulary -----------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("today", (_utc(2023, 5, 30), _utc(2023, 5, 31))),
        ("last month", (_utc(2023, 4, 1), _utc(2023, 5, 1))),
        ("this month", (_utc(2023, 5, 1), _utc(2023, 6, 1))),
        ("last year", (_utc(2022, 1, 1), _utc(2023, 1, 1))),
        ("March 2022", (_utc(2022, 3, 1), _utc(2022, 4, 1))),
        ("2022-03", (_utc(2022, 3, 1), _utc(2022, 4, 1))),
        ("2023-05-20", (_utc(2023, 5, 20), _utc(2023, 5, 21))),
        ("last week", (_utc(2023, 5, 22), _utc(2023, 5, 29))),
        ("this week", (_utc(2023, 5, 29), _utc(2023, 6, 5))),
    ],
)
def test_the_declared_vocabulary(phrase, expected):
    assert resolve(phrase, QUESTION_DATE) == Interval(*expected)


def test_a_bare_month_reads_as_the_most_recent_occurrence():
    assert resolve("April", QUESTION_DATE) == Interval(_utc(2023, 4, 1), _utc(2023, 5, 1))
    assert resolve("July", QUESTION_DATE) == Interval(_utc(2022, 7, 1), _utc(2022, 8, 1))


def test_open_ended_expressions_resolve_to_unbounded_intervals():
    """They are legal, and they are exactly what the unbounded-constraint rate
    counts: an unbounded interval scores ``temporal_correctness = 1.0``."""
    since = resolve("since 2022", QUESTION_DATE)
    assert since == Interval(_utc(2022, 1, 1), None)
    before = resolve("before 2022", QUESTION_DATE)
    assert before == Interval(None, _utc(2022, 1, 1))


def test_an_expression_outside_the_vocabulary_is_unresolved_not_guessed():
    assert resolve("around the time of the big move", QUESTION_DATE) is None
    assert resolve(None, QUESTION_DATE) is None


def test_question_date_accepts_both_shapes():
    assert parse_question_date(QUESTION_DATE) == date(2023, 5, 30)
    assert parse_question_date("2023-05-30") == date(2023, 5, 30)


# -- slot parsing -----------------------------------------------------------


def test_slots_parse_out_of_a_fenced_reply():
    reply = """Sure, here you go:
```json
{"entity_anchor": "Priya", "value_type": "date", "time_expression": "last May",
 "needs_source": false, "aggregate": true}
```"""
    slots = parse_slots(reply)
    assert slots == {
        "entity_anchor": "Priya",
        "value_type": "date",
        "time_expression": "last May",
        "needs_source": False,
        "aggregate": True,
    }


def test_an_empty_string_slot_reads_as_absent():
    """A model writing "" has declined to fill the slot.  Counting it as a
    prediction would make ``active_slots()`` claim an obligation nothing can
    satisfy, and would score as a false positive in the audit."""
    slots = parse_slots('{"entity_anchor": "  ", "value_type": null}')
    assert slots is not None
    assert slots["entity_anchor"] is None and slots["value_type"] is None


def test_an_unparseable_reply_is_none_not_an_exception():
    assert parse_slots("I'm not sure what you mean.") is None


# -- slots -> Obligations ---------------------------------------------------


def test_scope_comes_from_metadata_and_is_never_asked_of_the_model():
    """A hallucinated ``conv_id`` would not give a wrong answer — it would
    silently over-restrict the proof search, rejecting correct evidence for being
    outside a scope nobody asked for."""
    assert "scope" not in (parse_slots('{"entity_anchor": "x"}') or {})
    q, trace = obligations_from_slots(
        {"entity_anchor": "x", "time_expression": None}, QUESTION_DATE, scope=("conv7",)
    )
    assert q.scope == ("conv7",)
    assert trace["scope_source"] == "metadata"


def test_an_unresolved_phrase_is_recorded_as_such_not_dropped_silently():
    _, trace = obligations_from_slots(
        {"time_expression": "around the big move"}, QUESTION_DATE
    )
    assert trace["time_expression"] == "around the big move"
    assert trace["time_resolved"] is False


def test_the_trace_marks_an_unbounded_constraint():
    _, trace = obligations_from_slots({"time_expression": "since 2022"}, QUESTION_DATE)
    assert trace["time_unbounded"] is True


# -- the parser and its reported numbers -----------------------------------


class _Canned:
    """A completer that replays fixed replies.  No model, no claim about one."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, system, user):
        self.seen.append(user)
        return self.replies.pop(0), 10, 5


def test_the_parser_reports_the_unbounded_constraint_rate():
    """G7's measurable warning: a parser emitting unbounded intervals freely
    quietly disables ``temporal_correctness``."""
    parser = ObligationParser(
        _Canned(
            [
                '{"entity_anchor": "a", "time_expression": "since 2022"}',
                '{"entity_anchor": "b", "time_expression": "last May"}',
                '{"entity_anchor": "c", "time_expression": null}',
                '{"entity_anchor": "d", "time_expression": "before 2021"}',
            ]
        )
    )
    for _ in range(4):
        parser.parse("q?", QUESTION_DATE)
    report = parser.report()
    assert report["questions"] == 4
    assert report["unbounded_constraint_rate"] == 0.5
    assert report["llm_calls"] == 4


def test_a_slot_parse_failure_yields_the_empty_obligation_and_is_counted():
    parser = ObligationParser(_Canned(["no json here"]))
    q = parser.parse("q?", QUESTION_DATE)
    assert q.active_slots() == ()
    assert parser.report()["slot_parse_failures"] == 1


def test_unbounded_rate_over_gold_obligations():
    items = [
        Obligations(time_constraint=Interval(0.0, 10.0)),
        Obligations(time_constraint=Interval(0.0, None)),
        Obligations(),
    ]
    assert unbounded_rate(items) == pytest.approx(1 / 3)


# -- the core dispatch hook -------------------------------------------------


def test_learned_mode_refuses_until_a_parser_is_registered():
    core_obligations.set_learned_parser(None)
    with pytest.raises(NotImplementedError):
        core_obligations.parse("who did I meet in May?", mode="learned")


def test_register_routes_learned_mode_through_the_phase5_parser():
    class _Question:
        text = "when did I move?"
        question_date = QUESTION_DATE
        scope = ("conv1",)
        question_id = "q1"

    parser = ObligationParser(
        _Canned(['{"entity_anchor": "the move", "value_type": "date", '
                 '"time_expression": "in 2023"}'])
    )
    oblparse.register(parser)
    try:
        got = core_obligations.parse(_Question(), mode="learned")
        assert isinstance(got, Obligations)
        assert got.entity_anchor == "the move"
        assert got.value_type == "date"
        assert got.scope == ("conv1",)
        assert got.time_constraint == Interval(_utc(2023, 1, 1), _utc(2024, 1, 1))
    finally:
        oblparse.unregister()
    with pytest.raises(NotImplementedError):
        core_obligations.parse(_Question(), mode="learned")


def test_a_parser_returning_the_wrong_type_is_refused_by_the_core():
    core_obligations.set_learned_parser(lambda q: {"entity_anchor": "x"})
    try:
        with pytest.raises(TypeError):
            core_obligations.parse("q", mode="learned")
    finally:
        core_obligations.set_learned_parser(None)


def test_slot_level_scores_still_reads_the_parsers_output():
    """Fix F2's audit metric, on obligations the Phase-5 parser produced —
    the number that must be reported wherever coverage is reported."""
    parser = ObligationParser(
        _Canned(
            [
                '{"entity_anchor": "Priya", "value_type": "date", "needs_source": false}',
                '{"entity_anchor": "the flat", "value_type": null, "needs_source": true}',
            ]
        )
    )
    predicted = [parser.parse("q1", QUESTION_DATE), parser.parse("q2", QUESTION_DATE)]
    gold = [
        Obligations(entity_anchor="Priya", value_type="date", needs_source=False),
        Obligations(entity_anchor="the flat", value_type="price", needs_source=True),
    ]
    scores = core_obligations.slot_level_scores(predicted, gold)
    assert scores["entity_anchor.f1"] == 1.0
    assert scores["value_type.recall"] == 0.5
    assert scores["needs_source.accuracy"] == 1.0


# -- the 13-14 Aug 2026 audit's regressions ----------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["2023-02-30", "2023-13-01", "2023/13", "9999", "in 9999",
     "past 3000 years", "since 2023-02-30"],
)
def test_calendar_invalid_model_output_is_unresolved_not_an_exception(phrase):
    """The phrase is model output; a hallucinated date must resolve to ``None``
    (counted as unresolved), never raise inside a read path — the module's own
    contract, which naked ``date()`` constructors violated."""
    assert resolve(phrase, QUESTION_DATE) is None


def test_a_malformed_question_date_still_raises():
    """The other side of the same line: ``question_date`` comes from the corpus,
    not the model, and a bad one is a caller bug worth crashing on."""
    with pytest.raises(ValueError):
        parse_question_date("not a date")


def test_between_bare_months_is_one_forward_span():
    """Asked on 2023-05-30, 'between March and June' resolved each month's year
    independently — March 2023, June *2022* — and hulled them into an interval
    that started at its own end bound.  The second endpoint now reads forward
    from the first."""
    got = resolve("between March and June", QUESTION_DATE)
    assert got is not None
    assert got.start == _utc(2023, 3, 1)
    assert got.end == _utc(2023, 7, 1)

    got = resolve("between May and July", QUESTION_DATE)
    assert got is not None
    assert got.start == _utc(2023, 5, 1)
    assert got.end == _utc(2023, 8, 1)


def test_between_that_stays_incoherent_is_unresolved_not_hulled():
    """'between 2023 and 2020' cannot be repaired by a year shift; a knowingly
    wrong hull is worse than a counted unresolved."""
    assert resolve("between 2023 and 2020", QUESTION_DATE) is None


def test_the_trace_records_a_slot_parse_failure_honestly():
    """The counter was right and the per-question trace lied: ``parse_ok`` was
    computed after the fallback dict was installed, so it was constant True."""
    parser = ObligationParser(lambda system, user: ("no json here at all", 10, 5))
    parser.parse("When did I move?", QUESTION_DATE, question_id="q1")
    assert parser.parse_failures == 1
    assert parser.traces[-1]["parse_ok"] is False

    ok = ObligationParser(
        lambda system, user: ('{"entity_anchor": "the move"}', 10, 5)
    )
    ok.parse("When did I move?", QUESTION_DATE, question_id="q2")
    assert ok.parse_failures == 0
    assert ok.traces[-1]["parse_ok"] is True


def test_the_register_path_survives_a_time_bearing_question_without_a_date():
    """``core_obligations.parse(<bare string>, mode='learned')`` supplies no
    ``question_date``; with a time expression in the reply this crashed in
    ``parse_question_date('')``.  A read path degrades to 'constraint
    unresolved, counted' — it does not raise."""
    parser = ObligationParser(
        lambda system, user: ('{"entity_anchor": "move", "time_expression": "in 2023"}', 10, 5)
    )
    oblparse.register(parser)
    try:
        got = core_obligations.parse("when did I move to Tokyo?", mode="learned")
    finally:
        oblparse.unregister()
    assert isinstance(got, Obligations)
    assert got.time_constraint is None
    assert parser.traces[-1]["time_resolved"] is False
