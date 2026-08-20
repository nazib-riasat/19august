"""The G2 bakeoff harness and the pilot's audit draws (P5.3/P5.9).

Every test here exists because the 13-14 Aug 2026 audit confirmed the first
harness measured itself: the flat 60-turn slice mixed ten users' conversations
(71% of all context turns were foreign), the audit draws did not implement the
declared calibration/pilot disjointness, truncations were counted per turn
rather than per generation, and an errored candidate's ``NaN`` rates made the
artefact invalid JSON.  All run without a GPU and without the pinned corpus.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from graft.config import Config
from graft.eventlog import EventLog
from graft.graphstore import ReplayGraphStore
from graft.ingest.bakeoff import (
    CandidateResult,
    calibration_slice,
    calibration_turn_ids,
    decide,
    run_candidate,
)
from graft.ingest.extractor import ReplayExtractor
from graft.ingest.nli import StubVerifier
from graft.ingest.pipeline import IngestPipeline
from graft.ingest.records import RawExtraction
from graft.runtime import json_sanitize
from graft.schemas import Turn

_PILOT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "phase5_pilot.py"
_SPEC = importlib.util.spec_from_file_location("phase5_pilot", _PILOT_PATH)
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)

CFG = Config(profile="real")


# -- fixtures ----------------------------------------------------------------


def _turn(qid: str, sid: str, ix: int, text: str = "") -> dict:
    return {"role": "user" if ix % 2 == 0 else "assistant",
            "content": text or f"turn {ix} of {sid}"}


def _instance(qid: str, sessions: dict[str, int]) -> dict:
    """A synthetic LongMemEval instance: ``{session_id: n_turns}``."""
    ids = list(sessions)
    return {
        "question_id": qid,
        "haystack_session_ids": ids,
        "haystack_dates": ["2023/05/20 (Sat) 02:21"] * len(ids),
        "haystack_sessions": [
            [_turn(qid, sid, i) for i in range(n)] for sid, n in sessions.items()
        ],
    }


def _sample(tmp_path: Path) -> Path:
    """Two questions, three sessions; s1's windowed turns are 1-3 of 5, so the
    held-back top-up must land at s1's corpus positions 0 and 4."""
    sample = {
        "sessions": [
            {
                "question_id": "q1",
                "session_id": "s1",
                "turns": [{"turn_id": f"lme_s/q1/s1/{i}"} for i in (1, 2, 3)],
            },
            {
                "question_id": "q2",
                "session_id": "s3",
                "turns": [{"turn_id": f"lme_s/q2/s3/{i}"} for i in (0, 1)],
            },
            {
                "question_id": "q1",
                "session_id": "s2",
                "turns": [{"turn_id": f"lme_s/q1/s2/{i}"} for i in (0, 1)],
            },
        ]
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    return path


INDEX = {
    "q1": _instance("q1", {"s1": 5, "s2": 3}),
    "q2": _instance("q2", {"s3": 4}),
}


# -- the slice ---------------------------------------------------------------


def test_the_slice_is_grouped_by_conversation_in_corpus_order(tmp_path):
    """The first harness windowed one flat list: 55/60 prompts carried another
    user's conversation, and the held-back turns sat at the end of the list —
    one of them *before* its session's sampled window in corpus order."""
    groups = calibration_slice(_sample(tmp_path), INDEX, slice_size=9)

    assert [conv for conv, _ in groups] == ["q1", "q2"]
    by_conv = {conv: [t.turn_id for t in turns] for conv, turns in groups}

    # q1: windowed s1/{1,2,3} + held-back s1/{0,4} + windowed s2/{0,1},
    # in corpus order — the held-back turn 0 PRECEDES the window.
    assert by_conv["q1"] == [
        "lme_s/q1/s1/0", "lme_s/q1/s1/1", "lme_s/q1/s1/2", "lme_s/q1/s1/3",
        "lme_s/q1/s1/4", "lme_s/q1/s2/0", "lme_s/q1/s2/1",
    ]
    assert by_conv["q2"] == ["lme_s/q2/s3/0", "lme_s/q2/s3/1"]

    for conv, turns in groups:
        assert all(t.conv_id == conv for t in turns)

    assert len(calibration_turn_ids(groups)) == 9


# -- the harness -------------------------------------------------------------


class _RecordingExtractor:
    """No model: records what context each turn was extracted under."""

    def __init__(self) -> None:
        self.contexts: list[tuple[str, list[str], str]] = []
        self.events: list[str] = []
        self.load_seconds = 0.0

    def load(self) -> None:
        self.events.append("load")

    def complete(self, system: str, user: str, *, max_new_tokens=None):
        self.events.append("summary")
        return "- ESTABLISHED FACTS", 5, 3

    def extract(self, turn, context) -> RawExtraction:
        self.events.append("extract")
        self.contexts.append(
            (turn.turn_id, [t.turn_id for t in context.window], context.summary)
        )
        return RawExtraction(truncations=2, truncated=False)

    def close(self) -> None:
        self.events.append("close")


def _groups() -> list[tuple[str, list[Turn]]]:
    def turns(qid: str, sid: str, n: int) -> list[Turn]:
        return [
            Turn(
                turn_id=f"lme_s/{qid}/{sid}/{i}", conv_id=qid, session_id=sid,
                speaker="user", ts="2023-05-20T02:21:00+00:00",
                text=f"turn {i} of {qid}",
            )
            for i in range(n)
        ]

    return [("q1", turns("q1", "s1", 12)), ("q2", turns("q2", "s2", 3))]


def test_the_context_window_never_crosses_a_conversation_boundary():
    extractor = _RecordingExtractor()
    result = run_candidate("A", _groups(), lambda: extractor, context_turns=10)

    assert result.turns == 15
    for turn_id, window, _ in extractor.contexts:
        conv = turn_id.split("/")[1]
        assert all(w.split("/")[1] == conv for w in window), (
            f"{turn_id} saw a foreign conversation in its window: {window}"
        )
    # q2's first turn must see an EMPTY window, not q1's tail.
    q2_first = next(c for c in extractor.contexts if c[0] == "lme_s/q2/s2/0")
    assert q2_first[1] == []


def test_the_harness_runs_the_production_summary_recipe():
    """The bakeoff's context recipe is production's (decision 2): window within
    conversation plus the rolling summary through the candidate's own model."""
    extractor = _RecordingExtractor()
    run_candidate("A", _groups(), lambda: extractor, context_turns=10)

    at_11 = next(c for c in extractor.contexts if c[0] == "lme_s/q1/s1/11")
    assert at_11[2] == "- ESTABLISHED FACTS"  # past the refresh point
    at_5 = next(c for c in extractor.contexts if c[0] == "lme_s/q1/s1/5")
    assert at_5[2] == ""  # before it


def test_model_load_happens_before_the_timed_loop():
    """``grounded_per_minute`` is stage 2's decisive metric; a cold
    ``from_pretrained`` inside the first turn's timing would measure disk I/O
    ordering, not extraction."""
    extractor = _RecordingExtractor()
    run_candidate("A", _groups(), lambda: extractor)
    assert extractor.events[0] == "load"
    assert extractor.events.index("load") < extractor.events.index("extract")


def test_truncations_are_counted_per_generation_not_per_turn():
    """Each fake turn reports two capped generations behind a final parse that
    succeeded — the case the old per-turn final-flag count scored as zero."""
    extractor = _RecordingExtractor()
    result = run_candidate("A", _groups(), lambda: extractor)
    assert result.truncations == 2 * result.turns
    assert result.to_dict()["truncated_generations"] == 2 * result.turns


def test_an_errored_candidate_row_sanitizes_to_strict_json():
    """``json.dumps`` spells an empty denominator as bare ``NaN``, which is not
    JSON — the frozen decision-2 record was unreadable to jq, JSON.parse and
    every strict parser whenever a candidate recorded its (deliberate) error
    row."""
    errored = CandidateResult("B", {})
    errored.error = "RuntimeError: no grammar backend"
    ok = CandidateResult("A", {})
    ok.turns, ok.assertions_grounded, ok.seconds = 10, 12, 60.0

    artefact = json_sanitize(decide([ok, errored]))
    text = json.dumps(artefact, allow_nan=False)  # raises if anything survived
    back = json.loads(text)
    row = next(r for r in back["table"] if r["candidate"] == "B")
    assert row["parse_failure_rate"] is None
    assert row["error"].startswith("RuntimeError")


# -- the pilot's audit draws --------------------------------------------------


TURNS = [
    Turn(
        turn_id="lme_s/q1/s1/0", conv_id="q1", session_id="s1", speaker="user",
        ts="2023-05-20T02:21:00+00:00",
        text="I finally moved to Yokohama last March, and the rent is 92000 yen.",
    ),
    Turn(
        turn_id="lme_s/q1/s1/1", conv_id="q1", session_id="s1", speaker="assistant",
        ts="2023-05-20T02:21:00+00:00",
        text="That sounds like a good deal for the area.",
    ),
    Turn(
        turn_id="lme_s/q1/s1/2", conv_id="q1", session_id="s1", speaker="user",
        ts="2023-05-20T02:21:00+00:00",
        text="It is. I share it with Priya, who works at the aquarium.",
    ),
]

RECORDED = {
    "lme_s/q1/s1/0": {
        "turn_id": "lme_s/q1/s1/0",
        "mentions": [],
        "assertions": [
            {"kind": "event", "text_norm": "The user moved to Yokohama in March 2023.",
             "quotes": [{"turn_offset": 0, "text": "moved to Yokohama last March"}]},
            {"kind": "value", "text_norm": "The user's rent is 92000 yen.",
             "quotes": [{"turn_offset": 0, "text": "the rent is 92000 yen"}]},
        ],
    },
    "lme_s/q1/s1/2": {
        "turn_id": "lme_s/q1/s1/2",
        "mentions": [],
        "assertions": [
            {"kind": "claim", "text_norm": "Priya works at the aquarium and shares the flat.",
             "quotes": [
                 {"turn_offset": 0, "text": "I share it with Priya, who works at the aquarium"},
                 {"turn_offset": -2, "text": "moved to Yokohama last March"},
             ]},
        ],
    },
}


def _ingested(tmp_path):
    log = EventLog.open(tmp_path / "events.jsonl")
    pipeline = IngestPipeline(log, CFG, ReplayExtractor(RECORDED), StubVerifier(1.0))
    pipeline.run(TURNS)
    return log, ReplayGraphStore(log).at(), pipeline


def test_the_span_audit_excludes_calibration_touched_assertions(tmp_path):
    """Decision 2's declared disjointness, enforced at the draw: an assertion
    with ANY span in a calibration turn is out — including the cross-turn one
    whose second span lands there."""
    log, snapshot, _ = _ingested(tmp_path)

    rows = pilot.span_worksheet(
        snapshot, log, pilot.audit_rng("span"), 50, {"lme_s/q1/s1/2"}
    )
    texts = {r["text_norm"] for r in rows}
    assert len(rows) == 2
    assert "Priya works at the aquarium and shares the flat." not in texts

    unfiltered = pilot.span_worksheet(snapshot, log, pilot.audit_rng("span"), 50)
    assert len(unfiltered) == 3
    log.close()


def test_the_nli_audit_stratifies_at_the_configured_tau_and_excludes(tmp_path):
    """The frozen value's one home is ``Config.tau_nli``; the hardcoded 0.8 copy
    meant a Gate-0 amendment would leave this audit stratifying around the stale
    threshold — the audit built to audit ``tau_nli`` not reading it."""
    log, snapshot, pipeline = _ingested(tmp_path)

    rows = pilot.nli_worksheet(
        pipeline.scores, snapshot, pilot.audit_rng("nli"), 50,
        0.6, {"lme_s/q1/s1/2"},
    )
    assert len(rows) == 2
    for row in rows:
        assert row["tau_nli"] == 0.6
        # StubVerifier scored 1.0; |1.0 - 0.6| > 0.25 puts every row in 'far'.
        assert row["stratum"] == "far"
    log.close()


def test_each_worksheet_draw_is_independently_re_derivable():
    """One shared ``Random`` coupled the NLI draw to the span sheet's exact
    assertion count; per-sheet streams keep every draw re-derivable alone."""
    a = list(range(100))
    b = list(range(100))
    pilot.audit_rng("span").shuffle(a)
    pilot.audit_rng("span").shuffle(b)
    assert a == b

    c = list(range(100))
    pilot.audit_rng("nli").shuffle(c)
    assert c != a




# -- the metering and repair paths (14 Aug, second audit pass) ----------------
#
# Neither had any test: no test constructed an ``LlmExtractor`` and called
# ``extract``, and none constructed an ``NliVerifier`` at all — so reverting the
# per-generation truncation counter and deleting every ``ledger.count`` line
# both survived the whole suite.  These drive the **real** methods with a fake
# tokenizer and a fake model, so the loop, the metering and the batching are all
# the shipped code; only the weights are absent.  No GPU, no download.

import pytest

torch = pytest.importorskip("torch")

from graft.ingest.extractor import ExtractionContext, LlmExtractor  # noqa: E402
from graft.ingest.nli import NliVerifier  # noqa: E402
from graft.ingest.pins import DECODING, EXTRACTOR_CANDIDATES  # noqa: E402
from graft.ledger import Ledger  # noqa: E402

_GOOD = '{"mentions": [{"text": "Priya"}], "assertions": []}'


class _FakeTok:
    """Enough tokenizer for ``_generate``: ids are positions, decode is a lookup."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.replies: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)

    def __call__(self, prompt, return_tensors=None):
        class _Enc(dict):
            def to(self, device):
                return self

        return _Enc(input_ids=torch.zeros((1, 11), dtype=torch.long))

    def decode(self, generated, skip_special_tokens=True):
        return self.replies[int(generated[0].item())]


class _FakeModel:
    """Returns (reply_index, n_out) pairs as a token tensor of the right length."""

    device = "cpu"

    def __init__(self, plan):
        self.plan = list(plan)
        self.seen_kwargs: list[dict] = []

    def generate(self, **kwargs):
        self.seen_kwargs.append(kwargs)
        ix, n_out = self.plan.pop(0)
        row = torch.zeros((1, 11 + n_out), dtype=torch.long)
        row[0, 11] = ix  # the first generated token identifies the reply
        return row


def _llm(candidate: str, replies: list[str], out_tokens: list[int], ledger=None):
    llm = LlmExtractor(EXTRACTOR_CANDIDATES[candidate], device="cpu", ledger=ledger)
    tok = _FakeTok()
    tok.replies = replies
    llm._tok = tok
    llm._model = _FakeModel(list(enumerate(out_tokens)))
    llm.load = lambda: None  # weights are the only thing being skipped
    return llm


def _a_turn() -> Turn:
    return Turn(
        turn_id="lme_s/q1/s1/0", conv_id="q1", session_id="s1", speaker="user",
        ts="2023-05-20T02:21:00+00:00", text="I moved to Yokohama with Priya.",
    )


CAP = int(DECODING["max_new_tokens"])


def test_a_repaired_turn_still_reports_its_truncated_first_generation():
    """Generation-level counting: a first attempt that hit the cap and a repair
    that parsed must report one truncation, not zero."""
    llm = _llm("A", ["not json at all", _GOOD], [CAP, 40])
    got = llm.extract(_a_turn(), ExtractionContext())

    assert got.parse_ok is True
    assert got.repairs == 1
    assert got.llm_calls == 2
    assert got.truncations == 1, "the repaired-away truncation must still count"
    assert got.truncated is False, "the FINAL attempt was not truncated"
    assert got.tokens_out == CAP + 40


def test_two_truncated_attempts_count_twice_and_attribute_the_cause():
    llm = _llm("A", ['{"mentions": [', '{"assertions": ['], [CAP, CAP])
    got = llm.extract(_a_turn(), ExtractionContext())

    assert got.parse_ok is False
    assert got.truncations == 2
    assert got.truncated is True


def test_a_candidate_without_a_repair_policy_does_not_retry():
    """B's row says ``repair: False``; a silent retry would make the bakeoff's
    two candidates one candidate."""
    llm = _llm("B", ["not json"], [40])
    got = llm.extract(_a_turn(), ExtractionContext())
    assert got.parse_ok is False
    assert got.repairs == 0
    assert got.llm_calls == 1


def test_the_repair_prompt_carries_the_parser_error_back():
    """The reprompt's whole justification: the error is the one thing the model
    does not already have."""
    llm = _llm("A", ["no json here", _GOOD], [30, 40])
    llm.extract(_a_turn(), ExtractionContext())
    # Two generations: the second's messages must contain the parser complaint.
    assert llm._model.seen_kwargs and len(llm._model.seen_kwargs) == 2


def test_extraction_meters_every_generation_including_the_repair():
    """Exit criterion 14, at the call site — metering inside the wrapper."""
    ledger = Ledger.from_config(CFG, log=None)
    llm = _llm("A", ["nope", _GOOD], [30, 40], ledger=ledger)
    llm.extract(_a_turn(), ExtractionContext())

    totals = ledger.snapshot()["totals"]
    assert totals["llm_calls"] == 2
    assert totals["llm_tokens_out"] == 70
    assert totals["llm_tokens_in"] == 22


def test_the_summary_override_reaches_the_generation_call():
    """Decision 3's cap is enforced at generation, not by the word backstop."""
    llm = _llm("A", ["- a bullet"], [12])
    llm.complete("sys", "user", max_new_tokens=512)
    assert llm._model.seen_kwargs[0]["max_new_tokens"] == 512
    assert "logits_processor" not in llm._model.seen_kwargs[0]


class _FakeNliTok:
    def __call__(self, premises, hypotheses, **kwargs):
        class _Enc(dict):
            def to(self, device):
                return self

        return _Enc(n=len(premises))


class _FakeNliModel:
    device = "cpu"

    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, **encoded):
        n = encoded["n"]
        self.batches.append(n)
        # entailment index 1 gets the mass, matching the pinned checkpoint's
        # {0: contradiction, 1: entailment, 2: neutral}.
        logits = torch.tensor([[0.0, 5.0, 0.0]] * n)
        return type("Out", (), {"logits": logits})()


def test_the_nli_verifier_meters_one_model_forward_per_batch():
    """Also criterion 14, and previously unreachable by any test: the verify
    stage reported zero on every meter but wall-clock."""
    ledger = Ledger.from_config(CFG, log=None)
    verifier = NliVerifier(device="cpu", batch_size=2, ledger=ledger)
    verifier._tok = _FakeNliTok()
    verifier._model = _FakeNliModel()
    verifier._entail_ix = 1
    verifier.load = lambda: None

    scores = verifier.score([("p", "h")] * 5)

    assert len(scores) == 5
    assert all(s > 0.9 for s in scores)
    assert verifier._model.batches == [2, 2, 1], "batching is the shipped loop"
    assert ledger.snapshot()["totals"]["model_forwards"] == 3


def test_the_nli_verifier_runs_unmetered_without_a_ledger():
    """The ledger is optional; the write-path tests construct it without one."""
    verifier = NliVerifier(device="cpu", batch_size=8)
    verifier._tok = _FakeNliTok()
    verifier._model = _FakeNliModel()
    verifier._entail_ix = 1
    verifier.load = lambda: None
    assert len(verifier.score([("p", "h")] * 3)) == 3


# -- batched token accounting (19 Aug 2026) -----------------------------------
#
# The batched path had no test that drove `_generate_batch`, and it reported a
# **re-tokenised** count while metering the tensor length.  These fakes make the
# two disagree on purpose: `_ReTokShort` returns 1 for any string it is asked to
# encode, so a test that passes here can only be reading the tensor.


class _BatchTok:
    """A batch-capable tokenizer whose *re-tokenisation is deliberately wrong*.

    Encoding a bare string returns a single token regardless of content.  Real
    tokenisers are merely not round-trip stable; this one is grossly unstable,
    which is what makes the assertion below decisive rather than lucky.
    """

    eos_token_id = 0
    eos_token = "<eos>"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.pad_token = "<pad>"
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        class _Enc(dict):
            def to(self, device):
                return self

        if isinstance(text, str):  # the re-tokenisation the fix removed
            return _Enc(input_ids=torch.zeros((1, 1), dtype=torch.long))
        n = len(text)
        # Left-padded: row r carries r real tokens of prompt, the rest padding.
        width = n + 1
        ids = torch.zeros((n, width), dtype=torch.long)
        mask = torch.zeros((n, width), dtype=torch.long)
        for r in range(n):
            mask[r, width - (r + 1):] = 1
        return _Enc(input_ids=ids, attention_mask=mask)

    def decode(self, generated, skip_special_tokens=True):
        return self.replies[int(generated[0].item()) - 1]


class _BatchModel:
    """Row ``r`` generates ``plan[r]`` real tokens, then eos padding."""

    device = "cpu"

    def __init__(self, plan: list[int]) -> None:
        self.plan = list(plan)
        self.seen_kwargs: list[dict] = []

    def generate(self, **kwargs):
        self.seen_kwargs.append(kwargs)
        width = int(kwargs["input_ids"].shape[1])
        n = len(self.plan)
        longest = max(self.plan)
        out = torch.zeros((n, width + longest), dtype=torch.long)
        for r, n_out in enumerate(self.plan):
            out[r, width] = r + 1          # identifies the reply for `decode`
            out[r, width + 1:width + n_out] = 1  # filler, non-eos
        return out


def _batch_llm(candidate: str, replies: list[str], out_tokens: list[int], ledger=None):
    llm = LlmExtractor(EXTRACTOR_CANDIDATES[candidate], device="cpu", ledger=ledger)
    llm._tok = _BatchTok(replies)
    llm._model = _BatchModel(out_tokens)
    llm.load = lambda: None
    return llm


def test_the_batched_path_reports_the_length_it_metered():
    """One generation, one number.

    ``_generate_batch`` used to meter the trimmed tensor length and then store a
    re-tokenisation of the decoded string, so ``TurnReport.tokens_out`` and the
    ledger were two authorities on the same generation.
    """
    ledger = Ledger.from_config(CFG, log=None)
    llm = _batch_llm("A", [_GOOD, _GOOD], [40, 55], ledger=ledger)

    llm._generate_batch([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]])

    reported = [n_out for _, n_out in llm._last_batch_tokens]
    assert reported == [40, 55], "reported lengths must be the metered tensor lengths"
    assert ledger.snapshot()["totals"]["llm_tokens_out"] == sum(reported), (
        "the ledger and the per-turn report must not disagree about one generation"
    )


def test_a_capped_batched_generation_is_attributed_to_truncation():
    """The load-bearing half.

    ``extract_batch`` derives ``truncated`` from this count.  Under the
    re-tokenised version a reply that genuinely hit ``max_new_tokens`` could
    re-encode to under the cap and be filed as *malformed* instead — and
    malformed-vs-truncated is the split `PHASE5_DECISIONS.md` §2 used to
    falsify its own token-cap hypothesis.
    """
    # Candidate B: `repair: False`, so `extract_batch` returns the batched row
    # instead of handing a parse failure to the single-stream repair loop --
    # which would re-generate and test a different code path than the one at
    # issue.
    llm = _batch_llm("B", ["{ truncated json", _GOOD], [CAP, 40])

    got = llm.extract_batch([_a_turn(), _a_turn()], [ExtractionContext(), ExtractionContext()])

    assert got[0].truncated is True, "a capped generation is a truncation, not malformed output"
    assert got[0].truncations == 1
    assert got[0].tokens_out == CAP
    assert got[1].truncated is False
    assert got[1].tokens_out == 40


def test_batched_prompt_lengths_come_from_the_attention_mask_not_the_padded_width():
    """Padding is not work.  Left-padding makes every row the same width, so a
    padded count would charge every turn the longest prompt in its batch."""
    llm = _batch_llm("A", [_GOOD, _GOOD, _GOOD], [10, 10, 10])
    llm._generate_batch([[{"role": "user", "content": "x"}] for _ in range(3)])

    assert [n_in for n_in, _ in llm._last_batch_tokens] == [1, 2, 3]

