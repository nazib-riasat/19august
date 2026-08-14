"""The G2 extractor bakeoff: three candidates, one rule, declared before the run.

**Why a bakeoff at all.**  The spike falsified the "just prompt it" assumption
and measured the cost: **9 of 58 turns (15.5%) produced unparseable JSON**, and
those turns yielded *nothing*.  A production Stage A cannot ship a silent 15%
evidence loss — it would poison ceiling 1 ("are all gold evidence statements
represented as grounded assertions?") before anything downstream ran, and no
number computed afterwards would show it.

**The rule, in order, fixed in advance** (decision 2):

1. parse-failure rate **< 2%** — a hard filter, not a tiebreak;
2. among survivors, highest **grounded assertions per minute**;
3. tie-break: higher span precision on a 20-assertion sub-audit.

This is instrument calibration — no learner exists anywhere near it — and the
predeclaration is the project's own protocol discipline.  (Its significance-
testing authority, Dror et al. ACL 2018, covers *test selection*; fixing a rule
in advance is **[ANALYSIS]**, the project's own rule, and is labelled as such
rather than attributed to the paper.)

**Stage 2 is a rate, not a count**, because the three candidates have different
throughputs by construction: a 7B is roughly twice a 3B's cost per token and
constrained decoding pays per step.  "Most assertions" would hand it to whichever
candidate is slowest to be honest about its failures; "assertions per minute"
asks the question the sizing memo (G8) actually needs answered.

**Stage 3 needs a human and says so.**  The sub-audit is 20 assertions read by a
person against the G1 protocol; :func:`decide` returns a *tie* verdict naming the
candidates rather than inventing a number, and the pilot script writes the
worksheet.  A tie broken by a machine-generated span judgement would be the
bootstrap-labels mistake of Phase 2.5, one phase later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from graft.ingest import corpus
from graft.ingest.extractor import ExtractionContext
from graft.ingest.grounding import ground_assertion_quotes
from graft.ingest.pins import PARSE_FAILURE_CEILING, SUMMARY_MAX_TOKENS
from graft.ingest.summary import RollingSummary, context_window
from graft.schemas import Turn

__all__ = [
    "run_candidate",
    "decide",
    "CandidateResult",
    "calibration_slice",
    "calibration_turn_ids",
    "SLICE_SIZE",
]

#: The declared slice size: the spike's 58 windowed turns + 2 held-back turns.
SLICE_SIZE = 60


def calibration_slice(
    sample_path: str | Path,
    index: Mapping[str, Mapping[str, Any]],
    *,
    slice_size: int = SLICE_SIZE,
) -> list[tuple[str, list[Turn]]]:
    """The calibration slice, **grouped by conversation, in corpus order**.

    Returns ``[(conv_id, turns), ...]`` — one group per LongMemEval question,
    each group's turns in corpus order (sessions as the haystack lists them,
    turns in session order).

    **Why groups and not a flat list** (found by the 13 Aug 2026 audit): the
    spike's sample is 21 sessions from 10 *different* questions — ten different
    simulated users.  The first harness windowed one flat 60-turn list, so 55 of
    60 prompts carried another user's conversation as context (71% of all
    context turns were foreign), the two held-back turns sat at the end of the
    list instead of at their in-session positions, and a cross-turn quote could
    "ground" into a foreign user's turn — inflating the stage-2 decision metric
    itself.  A bakeoff run under a context regime production never produces
    measures the harness, not the candidates.

    Lives here rather than in the script so the pilot can import the same
    definition and exclude these turns from its audit draws (the declared
    disjointness), instead of trusting a second copy.
    """
    sample = json.loads(Path(sample_path).read_text(encoding="utf-8"))

    # Which turn_ids belong to the slice, per question: the spike's windowed
    # turns, plus — for the first sampled session — the held-back extras, taken
    # as the first unwindowed turns of that session in corpus order.
    wanted: dict[str, set[str]] = {}
    sessions_of: dict[str, list[str]] = {}
    total = 0
    for session in sample["sessions"]:
        qid = session["question_id"]
        wanted.setdefault(qid, set())
        sessions_of.setdefault(qid, [])
        if session["session_id"] not in sessions_of[qid]:
            sessions_of[qid].append(session["session_id"])
        for turn in session["turns"]:
            if turn["turn_id"] not in wanted[qid]:
                wanted[qid].add(turn["turn_id"])
                total += 1

    first = sample["sessions"][0]
    first_qid = first["question_id"]
    for turn in corpus.turns_of(index[first_qid], [first["session_id"]]):
        if total >= slice_size:
            break
        if turn.turn_id not in wanted[first_qid]:
            wanted[first_qid].add(turn.turn_id)
            total += 1

    # Materialise each group in corpus order — the order production ingests.
    groups: list[tuple[str, list[Turn]]] = []
    seen_q: list[str] = []
    for session in sample["sessions"]:
        qid = session["question_id"]
        if qid in seen_q:
            continue
        seen_q.append(qid)
        turns = [
            t
            for t in corpus.turns_of(index[qid], sessions_of[qid])
            if t.turn_id in wanted[qid]
        ]
        groups.append((qid, turns))
    return groups


def calibration_turn_ids(groups: Sequence[tuple[str, Sequence[Turn]]]) -> set[str]:
    """The slice's turn ids — what the pilot's audit draws must exclude."""
    return {t.turn_id for _, turns in groups for t in turns}


def _cause_of(error: str | None) -> str:
    """A parser complaint → a countable category.

    Three categories, because they have three different fixes: the model emitted
    no JSON at all (a prompting or decoding problem), it emitted JSON that does
    not parse (a decoding problem), or it emitted something that is not an
    object.  Truncation is handled by the caller, which knows the token count and
    the cap — the parser only sees a string with a missing brace and cannot tell
    that apart from malformed output.
    """
    text = (error or "").lower()
    if "no json object" in text:
        return "no_json_emitted"
    if "top-level value" in text:
        return "not_an_object"
    return "malformed_json"


class CandidateResult:
    """One candidate's row of the bakeoff table.

    Every field the rule reads is here, plus the ones a reader needs to tell a
    *fast* candidate from a *lucky* one: tokens, wall clock, VRAM, and the
    grounding failures that separate "extracted a lot" from "extracted a lot that
    could be located in the turn".
    """

    __slots__ = (
        "candidate",
        "config",
        "turns",
        "parse_failures",
        "repairs",
        "mentions",
        "assertions_extracted",
        "assertions_grounded",
        "quotes_ungrounded",
        "cross_turn_quotes",
        "seconds",
        "tokens_in",
        "tokens_out",
        "peak_vram_mb",
        "load_seconds",
        "error",
        "samples",
        "failure_causes",
        "truncations",
        "summary_calls",
    )

    def __init__(self, candidate: str, config: Mapping[str, Any]) -> None:
        self.candidate = candidate
        self.config = dict(config)
        self.turns = 0
        self.parse_failures = 0
        self.repairs = 0
        self.mentions = 0
        self.assertions_extracted = 0
        self.assertions_grounded = 0
        self.quotes_ungrounded = 0
        self.cross_turn_quotes = 0
        self.seconds = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.peak_vram_mb: int | None = None
        self.load_seconds = 0.0
        self.error: str | None = None
        self.samples: list[dict[str, Any]] = []
        #: Why parses failed, not merely how often.  A truncated JSON object has
        #: no closing brace and fails for a reason that has nothing to do with
        #: the model's ability to emit JSON, so a bare rate is uninterpretable —
        #: and the two causes have opposite fixes (a bigger token budget vs a
        #: different decoding strategy).
        self.failure_causes: dict[str, int] = {}
        #: **Generation-level** since 13 Aug 2026: every capped generation
        #: counts, including a first attempt whose repair then parsed.  The old
        #: per-turn final-attempt count was candidate-asymmetric — exactly the
        #: candidates whose policy is repair hid their truncations while the
        #: truncated attempt's tokens stayed in ``tokens_out``.
        self.truncations = 0
        self.summary_calls = 0

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.turns if self.turns else float("nan")

    @property
    def grounded_per_minute(self) -> float:
        """Stage 2's metric.  ``nan`` on a candidate that never ran.

        Deliberately **not** ``0.0`` for a candidate that errored: zero is a
        measurement and this is its absence, and the two must not sort together.
        """
        if self.seconds <= 0:
            return float("nan")
        return self.assertions_grounded / (self.seconds / 60.0)

    def passes_filter(self, ceiling: float = PARSE_FAILURE_CEILING) -> bool:
        return self.error is None and self.turns > 0 and self.parse_failure_rate < ceiling

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "config": self.config,
            "turns": self.turns,
            "parse_failures": self.parse_failures,
            "parse_failure_rate": self.parse_failure_rate,
            "parse_failure_causes": dict(sorted(self.failure_causes.items())),
            "truncated_generations": self.truncations,
            "repairs": self.repairs,
            "mentions": self.mentions,
            "mentions_per_turn": self.mentions / self.turns if self.turns else float("nan"),
            "assertions_extracted": self.assertions_extracted,
            "assertions_grounded": self.assertions_grounded,
            "assertions_per_turn": (
                self.assertions_grounded / self.turns if self.turns else float("nan")
            ),
            "quotes_ungrounded": self.quotes_ungrounded,
            "cross_turn_quotes": self.cross_turn_quotes,
            "seconds": round(self.seconds, 2),
            "grounded_per_minute": self.grounded_per_minute,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_per_second": (
                self.tokens_out / self.seconds if self.seconds > 0 else float("nan")
            ),
            "peak_vram_mb": self.peak_vram_mb,
            "load_seconds": round(self.load_seconds, 2),
            "summary_calls": self.summary_calls,
            "passes_parse_filter": self.passes_filter(),
            "error": self.error,
        }


def run_candidate(
    name: str,
    groups: Sequence[tuple[str, Sequence[Turn]]],
    build: Callable[[], Any],
    *,
    context_turns: int = 10,
    progress: Callable[[str], None] | None = None,
) -> CandidateResult:
    """Run one candidate over the grouped calibration slice and measure it.

    ``build`` is a thunk rather than an extractor so that a candidate which
    cannot be constructed at all — candidate B without a grammar backend,
    candidate C without the 7B weights — records an ``error`` row instead of
    aborting the bakeoff.  A missing candidate is a *result* ("we could not run
    it here, and here is why"), and losing the other two rows to it would be the
    expensive kind of failure.

    **The same slice, the same prompt, the same context recipe for all
    candidates** (decision 2) — and since 13 Aug 2026 that recipe is the
    *production* one: the window never crosses a conversation boundary, and the
    rolling summary runs in the harness through the candidate's own model
    (identical for A and B, which share the model), its calls timed inside
    ``seconds`` exactly as the pilot's throughput counts them.  One declared
    difference from production remains and is stated in the artefact: the slice
    holds each session's *windowed* turns, not the full session stream, so the
    summary is built over that windowed stream.

    **Model load is excluded from the timed loop**: ``load()`` is forced before
    the first turn, so ``grounded_per_minute`` measures extraction, not disk I/O
    ordering — ``load_seconds`` is its own column.
    """
    from graft.ingest.pins import EXTRACTOR_CANDIDATES

    result = CandidateResult(name, EXTRACTOR_CANDIDATES.get(name, {}))
    try:
        extractor = build()
        load = getattr(extractor, "load", None)
        if callable(load):
            load()
    except Exception as exc:  # noqa: BLE001 - the error is the row
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    n_total = sum(len(turns) for _, turns in groups)
    complete = getattr(extractor, "complete", None)
    summary = (
        RollingSummary(
            lambda system, user: extractor.complete(
                system, user, max_new_tokens=SUMMARY_MAX_TOKENS
            )
        )
        if callable(complete)
        else None
    )

    try:
        for conv_id, turns in groups:
            for ix, turn in enumerate(turns):
                started = time.perf_counter()
                if summary is not None:
                    summary_text, window = summary.context_for(
                        conv_id, turns, ix, context_turns
                    )
                else:
                    summary_text, window = "", context_window(turns, ix, context_turns)
                context = ExtractionContext(
                    summary=summary_text, window=window, session_date=turn.ts
                )
                offsets = context.offsets(turn)
                extraction = extractor.extract(turn, context)
                result.seconds += time.perf_counter() - started

                result.turns += 1
                result.repairs += extraction.repairs
                result.tokens_in += extraction.tokens_in
                result.tokens_out += extraction.tokens_out
                result.truncations += extraction.truncations
                if not extraction.parse_ok:
                    result.parse_failures += 1
                    cause = (
                        "truncated_at_token_cap"
                        if extraction.truncated
                        else _cause_of(extraction.error)
                    )
                    result.failure_causes[cause] = result.failure_causes.get(cause, 0) + 1
                result.mentions += len(extraction.mentions)
                result.assertions_extracted += len(extraction.assertions)

                for raw in extraction.assertions:
                    if any(q.turn_offset != 0 for q in raw.quotes):
                        result.cross_turn_quotes += 1
                    grounded = ground_assertion_quotes(raw.quotes, turn.turn_id, offsets)
                    if grounded is None:
                        result.quotes_ungrounded += 1
                        continue
                    result.assertions_grounded += 1
                    if len(result.samples) < 40:
                        result.samples.append(
                            {
                                "turn_id": turn.turn_id,
                                "kind": raw.kind,
                                "text_norm": raw.text_norm,
                                "spans": [q.to_dict() for q in grounded],
                            }
                        )
                if progress is not None:
                    progress(
                        f"  [{name}] {result.turns}/{n_total} "
                        f"{turn.turn_id}: {len(extraction.assertions)} assertions"
                        + ("" if extraction.parse_ok else "  PARSE FAIL")
                    )
    except Exception as exc:  # noqa: BLE001 - a mid-run failure is also a row
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.load_seconds = getattr(extractor, "load_seconds", 0.0)
        if summary is not None:
            result.summary_calls = summary.calls
        close = getattr(extractor, "close", None)
        if callable(close):
            close()
        result.peak_vram_mb = getattr(extractor, "peak_vram_mb", None)
    return result


def decide(
    results: Sequence[CandidateResult], ceiling: float = PARSE_FAILURE_CEILING
) -> dict[str, Any]:
    """Apply the rule, in its declared order, and say which stage decided.

    ``verdict`` is one of ``winner`` / ``tie`` / ``no_survivor``.  A tie is
    returned rather than resolved: stage 3 is a human sub-audit, and a machine
    breaking it would be Phase 2.5's bootstrap-labels mistake one phase later.
    ``no_survivor`` is a real outcome too — it means every candidate fails the
    2% filter, which is a Gate-0 finding, not a reason to raise the ceiling.
    """
    survivors = [r for r in results if r.passes_filter(ceiling)]
    table = [r.to_dict() for r in results]

    if not survivors:
        return {
            "rule": "parse-failure < 2% -> grounded assertions/min -> 20-assertion span sub-audit",
            "verdict": "no_survivor",
            "winner": None,
            "decided_by": "stage 1 (parse-failure filter)",
            "survivors": [],
            "table": table,
            "reading": (
                "no candidate meets the declared 2% ceiling. That is a Gate-0 "
                "finding about extraction feasibility on this machine, not a "
                "reason to raise the ceiling after seeing the numbers."
            ),
        }

    best = max(survivors, key=lambda r: r.grounded_per_minute)
    # A tie band rather than exact equality: two candidates whose throughput
    # differs by under 1% have not been separated by a 60-turn slice, and
    # declaring a winner on that margin would be reading noise as a decision.
    band = 0.01 * best.grounded_per_minute
    tied = [
        r
        for r in survivors
        if abs(r.grounded_per_minute - best.grounded_per_minute) <= band
    ]
    if len(tied) > 1:
        return {
            "rule": "parse-failure < 2% -> grounded assertions/min -> 20-assertion span sub-audit",
            "verdict": "tie",
            "winner": None,
            "decided_by": "stage 2 tied within 1%; stage 3 is a human sub-audit",
            "survivors": [r.candidate for r in survivors],
            "tied": [r.candidate for r in tied],
            "table": table,
        }
    return {
        "rule": "parse-failure < 2% -> grounded assertions/min -> 20-assertion span sub-audit",
        "verdict": "winner",
        "winner": best.candidate,
        "winner_config": best.config,
        # **Which stage actually decided, not which stage ran last.**  With one
        # survivor the throughput comparison had nothing to compare against —
        # stage 1's filter is what eliminated the field — and an artefact saying
        # "decided by stage 2" would misreport the basis of a frozen decision to
        # anyone reading it later.  Stage 2 only decides when it had a choice.
        "decided_by": (
            "stage 2 (grounded assertions per minute)"
            if len(survivors) > 1
            else "stage 1 (parse-failure filter) — sole survivor; stage 2 had "
            "nothing to compare against"
        ),
        "survivors": [r.candidate for r in survivors],
        "table": table,
    }
