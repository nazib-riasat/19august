"""LoCoMo loader — the evaluation corpus, on `graft.ingest.corpus`'s interface.

Runs on a synthetic fixture in the shape the release documents.  The real file is
not in the repo, which is exactly why `locomo.probe` exists and why the checks
here are about the loader's *refusals*: a corpus-wide parse error is silent in
every number it touches, and ingesting LoCoMo costs ~43 GPU hours.
"""

from __future__ import annotations

import json

import pytest

from graft.ingest.locomo import (
    ADVERSARIAL_CATEGORY,
    LoCoMoError,
    evidence_turn_ids,
    load_corpus,
    parse_locomo_ts,
    probe,
    questions_of,
    sample_index,
    session_keys,
    turn_id_for,
    turns_of,
)


def _sample(sid: str = "conv-1", *, sessions: int = 2) -> dict:
    conv: dict = {"speaker_a": "Alice", "speaker_b": "Bob"}
    for n in range(1, sessions + 1):
        conv[f"session_{n}"] = [
            {"speaker": "Alice", "text": f"turn {n}a", "dia_id": f"D{n}:1"},
            {"speaker": "Bob", "text": f"turn {n}b", "dia_id": f"D{n}:2"},
        ]
        conv[f"session_{n}_date_time"] = f"1:56 pm on {n} May, 2023"
    return {
        "sample_id": sid,
        "conversation": conv,
        "qa": [
            {"question": "where?", "answer": "London", "category": 4, "evidence": ["D1:1"]},
            {"question": "really?", "adversarial_answer": "no info",
             "category": ADVERSARIAL_CATEGORY, "evidence": []},
        ],
    }


@pytest.fixture()
def corpus_file(tmp_path):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps([_sample("conv-1"), _sample("conv-2")]), encoding="utf-8")
    return path


# -- structure ---------------------------------------------------------------


def test_a_missing_file_names_what_to_download(corpus_file):
    with pytest.raises(LoCoMoError, match="locomo10.json"):
        load_corpus(corpus_file.parent / "absent.json", verify=False)


def test_a_sha_mismatch_is_refused(corpus_file):
    """A different file is a different experiment -- `corpus.py`'s rule, kept."""
    with pytest.raises(LoCoMoError, match="different file is a different experiment"):
        load_corpus(corpus_file, expect_sha="0" * 64)


def test_a_sample_missing_a_key_says_which_keys_it_has(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"sample_id": "x", "qa": []}]), encoding="utf-8")
    with pytest.raises(LoCoMoError, match="no 'conversation'"):
        load_corpus(path, verify=False)


def test_a_sample_with_no_sessions_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps([{"sample_id": "x", "conversation": {"speaker_a": "A"}, "qa": []}]),
        encoding="utf-8",
    )
    with pytest.raises(LoCoMoError, match="no session_"):
        load_corpus(path, verify=False)


# -- the ordering bug that would produce plausible output --------------------


def test_sessions_are_ordered_numerically_not_lexically():
    """`session_10` precedes `session_2` as a string. That reordering would
    invalidate every temporal claim in the conversation while producing output
    that looks entirely normal -- the one bug in this module that is silent."""
    conv = {f"session_{n}": [] for n in (1, 2, 10, 11, 3)}
    assert session_keys(conv) == (
        "session_1", "session_2", "session_3", "session_10", "session_11",
    )


def test_turns_come_out_oldest_session_first():
    sample = _sample(sessions=3)
    got = [t.turn_id for t in turns_of(sample)]
    assert got[0] == turn_id_for("conv-1", "session_1", 0)
    assert got[-1] == turn_id_for("conv-1", "session_3", 1)


def test_a_session_restriction_takes_only_that_session():
    """The mechanism for conversation-by-conversation scope -- how the three-day
    plan harvests whatever ingestion finished."""
    sample = _sample(sessions=3)
    got = list(turns_of(sample, session_ids=["session_2"]))
    assert {t.session_id for t in got} == {"session_2"}


# -- timestamps --------------------------------------------------------------


def test_a_timestamp_is_read_as_utc_by_declaration():
    assert parse_locomo_ts("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00+00:00"
    assert parse_locomo_ts("12:05 am on 1 January, 2024") == "2024-01-01T00:05:00+00:00"
    assert parse_locomo_ts("12:30 pm on 2 Feb, 2024") == "2024-02-02T12:30:00+00:00"


def test_an_unparseable_timestamp_raises_rather_than_defaulting():
    """A wrong date is invisible in every downstream number it touches, so the
    loader refuses instead of substituting one."""
    with pytest.raises(LoCoMoError, match="unparseable LoCoMo timestamp"):
        parse_locomo_ts("2023-05-08 13:56")


def test_a_session_without_a_timestamp_is_refused():
    sample = _sample()
    del sample["conversation"]["session_1_date_time"]
    with pytest.raises(LoCoMoError, match="no 'session_1_date_time'"):
        list(turns_of(sample))


def test_every_turn_in_a_session_shares_the_session_timestamp():
    """LoCoMo timestamps sessions, not turns. Recorded rather than smoothed over:
    a synthesised per-turn offset would be invented evidence, and Phase 6's
    valid_during edges inherit whatever this decides."""
    got = list(turns_of(_sample(sessions=1)))
    assert len({t.ts for t in got}) == 1


# -- image-only turns --------------------------------------------------------


def test_an_image_only_turn_is_skipped_not_substituted():
    """LoCoMo carries a blip_caption for image turns. A caption is not something
    the speaker said, and Stage A's provenance is a span in an utterance."""
    sample = _sample(sessions=1)
    sample["conversation"]["session_1"].append(
        {"speaker": "Alice", "text": "", "blip_caption": "a dog", "dia_id": "D1:3"}
    )
    assert len(list(turns_of(sample))) == 2


# -- the gold sidecar --------------------------------------------------------


def test_an_adversarial_question_carries_no_gold_answer():
    """LoCoMo puts it under `adversarial_answer` precisely because there is no
    answer. Reading it as one would give the abstention testbed a gold string to
    be scored against, and `diagnostics.report` scores that category by
    abstention instead."""
    qs = questions_of(_sample())
    adv = [q for q in qs if q["adversarial"]]
    assert len(adv) == 1
    assert adv[0]["gold"] == ""


def test_an_answerable_question_carries_its_gold():
    qs = questions_of(_sample())
    ans = [q for q in qs if not q["adversarial"]]
    assert ans[0]["gold"] == "London"
    assert ans[0]["category"] == 4


def test_a_question_without_a_category_is_refused():
    sample = _sample()
    del sample["qa"][0]["category"]
    with pytest.raises(LoCoMoError, match="no 'category'"):
        questions_of(sample)


def test_evidence_markers_resolve_to_turn_ids():
    sample = _sample()
    assert evidence_turn_ids(sample, ["D1:1"]) == [turn_id_for("conv-1", "session_1", 0)]


def test_an_unresolvable_evidence_marker_is_dropped_not_raised():
    """LoCoMo's evidence lists are known to name image-only turns, which this
    loader skips. The probe counts them so the rate is visible."""
    assert evidence_turn_ids(_sample(), ["D9:9"]) == []


# -- the probe ---------------------------------------------------------------


def test_the_probe_reports_every_mismatch_rather_than_the_first(corpus_file):
    """Ingesting LoCoMo costs ~43 GPU hours. Discovering a structural mismatch at
    hour 30 is the expensive failure; the probe makes it free, and reports all
    findings at once so one run surfaces everything."""
    got = probe(corpus_file)
    assert got["verdict"] == "MISMATCH"  # the fixture is 2 samples, not 10
    assert got["samples"] == 2
    assert got["turns"] == 8
    assert got["questions"] == 4
    assert len(got["findings"]) >= 3  # samples, questions, adversarial all differ
    assert "fix the loader" in got["next_step"]


def test_the_probe_reports_a_sha_to_pin(corpus_file):
    got = probe(corpus_file)
    assert len(got["sha256"]) == 64
    assert got["category_counts"] == {4: 2, ADVERSARIAL_CATEGORY: 2}


def test_the_probe_counts_unresolved_evidence_markers(corpus_file):
    got = probe(corpus_file)
    assert got["evidence_markers"] == 2
    assert got["unresolved_evidence_markers"] == 0


def test_sample_index_addresses_by_id(corpus_file):
    idx = sample_index(load_corpus(corpus_file, verify=False))
    assert set(idx) == {"conv-1", "conv-2"}
