"""P5.8 — turns → the event log, in two stage-sequential passes (gaps G4, G9).

**Two passes over the slice, not two steps per turn** (fix F7).  The extractor
and the NLI model are never resident together, and on an 8 GB card that is
mandatory rather than stylistic.  So pass 1 extracts and grounds every turn and
stores everything; the extractor is then freed; pass 2 verifies and gates.  Each
pass is one ``ledger.stage()``, which is what makes "extraction cost" and
"verification cost" separable numbers in the pilot report rather than one total.

**The audit layer is written before the verifier has an opinion** (fix F9).
Every grounded assertion is stored with ``eligibility="quarantined"`` — the
schema's fail-closed default — and pass 2 writes two further events: the
verifier's flags, then the gate's verdict.  Two events because they are two
components with two different authorities: F9 is explicit that the verifier
never blocks storage and the *gate* decides eligibility, and a single combined
event would erase that boundary in the one place it is permanently recorded.

**Idempotence is skip-and-count at the turn** (G4, decision 8).  A ``turn_id``
already in the log short-circuits that turn's whole pipeline, and the skip count
is a reported metric.  Every id is content-derived (Phase 0), so a re-run of a
completed slice appends nothing and neither ``EventLog.digest()`` nor the graph
digest moves.

**``turn.add`` is appended last, after that turn's spans and assertions**, and
the ordering is the crash-resume decision rather than an accident.  With the
turn written first, a crash mid-turn leaves a turn whose assertions never landed
— and the re-run skips it, losing that evidence permanently with nothing to
notice.  Written last, the same crash leaves spans and assertions whose turn is
absent, which every reader already fails closed on (`H`'s scope sub-check
refuses a provenance chain it cannot resolve), and the re-run reprocesses the
turn and repairs it.  Content-derived ids make the repair idempotent at the
graph level: the replayed snapshot is identical, and only the log carries the
few duplicated lines that say a crash happened.

**Crash-resume also has to finish pass 2, so pass 2 works from the log.**  Every
assertion with no ``assertion.set_eligibility`` event is unverified, whichever
run stored it.  Otherwise a re-run would skip every turn and leave those
assertions quarantined-by-omission — again a silent recall loss.

**Reads during ingestion are pinned.**  The gate and every lookup run against
``ReplayGraphStore.at(snapshot_id)`` semantics, so a record's eligibility verdict
is reproducible from the log alone.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Sequence

from graft.config import Config
from graft.eventlog import EventLog
from graft.graphstore import ReplayGraphStore
from graft.ids import assertion_id as assertion_id_of
from graft.ids import span_id as span_id_of
from graft.ingest.extractor import ExtractionContext
from graft.ingest.grounding import ground, ground_assertion_quotes, rung_counts
from graft.ingest.nli import apply_threshold, score_drafts
from graft.ingest.pins import CONTEXT_TURNS
from graft.ingest.records import DraftAssertion, Grounding, GroundedQuote, TurnReport
from graft.ingest.summary import RollingSummary, context_window
from graft.ingest.support import QuarantineTally, eligibility
from graft.ledger import Ledger
from graft.schemas import Assertion, AssertionFlags, SourceSpan, Turn

__all__ = [
    "IngestPipeline",
    "ingested_turn_ids",
    "unverified_assertions",
    "mentions_of",
    "OP_MENTION",
]

_OP_TURN = "turn.add"
_OP_SPAN = "span.add"
_OP_ASSERTION = "assertion.add"
_OP_FLAGS = "assertion.set_flags"
_OP_ELIGIBILITY = "assertion.set_eligibility"

#: Mentions are Stage-A observations, not graph state: no entity exists until
#: Phase 6's D1 decides LINK/CREATE/NON_ENTITY/DEFER, so the graph replay
#: deliberately ignores this op (``GRAPH_OPS`` excludes it) and Phase 6 reads it
#: from the log.  Before 13 Aug 2026 a mention survived only as a bare
#: ``span.add`` plus a per-turn *count* — indistinguishable from a quote span,
#: and "unreferenced span = mention" is unsafe because a mention's span can
#: coincide with an assertion quote's.  Plan §0/§8 promise D1 items derive from
#: Stage-A mentions; this op is what makes that true from the permanent record.
OP_MENTION = "mention.add"


def ingested_turn_ids(log: EventLog, upto: int | None = None) -> set[str]:
    """Every ``turn_id`` already in the log — the idempotence key (G4)."""
    return {
        event.payload["turn_id"] for event in log.replay(upto=upto) if event.op == _OP_TURN
    }


def mentions_of(log: EventLog, upto: int | None = None) -> list[dict[str, Any]]:
    """Every stored mention, in log order — Phase 6's D1 item source.

    Duplicates (a crash-repaired turn rewrites its mentions) collapse on
    ``span_id``, keeping the first occurrence, exactly as replay collapses a
    rewritten span.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for event in log.replay(upto=upto):
        if event.op != OP_MENTION:
            continue
        sid = event.payload["span_id"]
        if sid in seen:
            continue
        seen.add(sid)
        out.append(dict(event.payload))
    return out


def unverified_assertions(log: EventLog, upto: int | None = None) -> list[str]:
    """Stored assertions with no eligibility verdict, in first-stored order.

    The crash-resume key.  Deliberately derived from the log rather than from a
    run-local list: a resumed run has no memory of what the dead one stored, and
    the log is the only thing that does.
    """
    stored: list[str] = []
    seen: set[str] = set()
    decided: set[str] = set()
    for event in log.replay(upto=upto):
        if event.op == _OP_ASSERTION:
            aid = event.payload["assertion_id"]
            if aid not in seen:
                seen.add(aid)
                stored.append(aid)
        elif event.op == _OP_ELIGIBILITY:
            decided.add(event.payload["assertion_id"])
    return [aid for aid in stored if aid not in decided]


class IngestPipeline:
    """The per-turn orchestration, and the two passes it runs in.

    ``extractor`` and ``verifier`` are duck-typed on their one method each
    (``extract`` / ``score``), which is what lets the whole write path be tested
    with a ``ReplayExtractor`` and a ``StubVerifier`` on a bare interpreter — and
    what keeps the GPU work to the bakeoff and the live pilot.
    """

    def __init__(
        self,
        log: EventLog,
        cfg: Config,
        extractor: Any,
        verifier: Any,
        *,
        summary: RollingSummary | None = None,
        ledger: Ledger | None = None,
        context_turns: int = CONTEXT_TURNS,
    ) -> None:
        self.log = log
        self.cfg = cfg
        self.extractor = extractor
        self.verifier = verifier
        self.summary = summary
        self.ledger = ledger
        self.context_turns = int(context_turns)
        self.reports: list[TurnReport] = []
        self.tally = QuarantineTally()
        self.skipped = 0
        self.rungs: dict[str, int] = {}
        self.fuzzy_spans: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.resumed_drafts = 0
        self.extract_seconds = 0.0
        self.verify_seconds = 0.0
        self._drafts: dict[str, DraftAssertion] = {}
        self._written_spans: set[str] = set()
        self._written_mentions: set[str] = set()

    # -- pass 1 ------------------------------------------------------------

    def extract_slice(self, turns: Sequence[Turn], conv_id: str | None = None) -> None:
        """Extract, ground and store every turn of one conversation, in order.

        Order matters and is the caller's: the context window and the rolling
        summary are both defined over "the previous turns", so a reordering would
        change what the model saw and therefore what it extracted.
        """
        already = ingested_turn_ids(self.log)
        conv_id = conv_id or (turns[0].conv_id if turns else "")

        for ix, turn in enumerate(turns):
            if turn.turn_id in already:
                self.skipped += 1
                self.reports.append(TurnReport(turn.turn_id, skipped=True))
                continue
            report = self._ingest_turn(conv_id, turns, ix, turn)
            self.extract_seconds += report.seconds
            self.reports.append(report)

    def _ingest_turn(
        self, conv_id: str, turns: Sequence[Turn], ix: int, turn: Turn
    ) -> TurnReport:
        started = time.perf_counter()

        if self.summary is not None:
            summary_text, window = self.summary.context_for(
                conv_id, turns, ix, self.context_turns
            )
        else:
            summary_text, window = "", context_window(turns, ix, self.context_turns)
        context = ExtractionContext(summary=summary_text, window=window, session_date=turn.ts)

        extraction = self.extractor.extract(turn, context)
        offsets = context.offsets(turn)

        report = TurnReport(
            turn.turn_id,
            parse_ok=extraction.parse_ok,
            repairs=extraction.repairs,
            truncations=extraction.truncations,
            assertions_extracted=len(extraction.assertions),
            llm_calls=extraction.llm_calls,
            tokens_in=extraction.tokens_in,
            tokens_out=extraction.tokens_out,
        )
        turn_groundings: list[Grounding] = []

        # Mentions carry no assertion, so they are not gated.  Phase 6's D1 items
        # derive from them (§8), and a mention that cannot be located in its turn
        # is not an item — so it is dropped and counted, like any other grounding
        # failure.  A grounded mention is written as its span **plus a
        # ``mention.add`` event** naming that span — see ``OP_MENTION``.
        for text in extraction.mentions:
            hit = ground(text, turn.text)
            if hit is None:
                report.dropped_mentions += 1
                continue
            self._write_span(turn.turn_id, hit)
            self._write_mention(turn.turn_id, hit, turn.text)
            self._note_grounding(turn.turn_id, hit, text, turn_groundings)
            report.mentions += 1

        seen_here: set[str] = set()
        for raw in extraction.assertions:
            grounded = ground_assertion_quotes(raw.quotes, turn.turn_id, offsets)
            if grounded is None:
                report.dropped_assertions += 1
                self.tally.drop()
                continue

            span_ids = [q.span_id for q in grounded]
            aid = assertion_id_of(raw.kind, raw.text_norm, span_ids)
            if aid in seen_here:
                # Same kind, same text, same spans: the same assertion extracted
                # twice **inside one turn**.  Content-derived ids make that
                # visible instead of duplicating evidence in the log.
                #
                # **Scoped to the turn, not to the run** (corrected 14 Aug 2026,
                # second audit pass).  A run-scoped set also suppressed the same
                # assertion derived by a *later* turn — reachable whenever the
                # quotes are cross-turn — and `_write_assertion` takes
                # ``asserted_by`` and ``t_created`` from the turn doing the
                # writing, which the content-derived id does not cover.  So an
                # uninterrupted run kept the *first* turn's speaker and
                # timestamp while a crash-resumed run (whose dedup map starts
                # empty) kept the *last* turn's, and replay is last-write-wins:
                # the same corpus produced two different graphs, falsifying this
                # module's own "the replayed snapshot is identical" claim.
                # Per-turn scope makes both orders end on the same final write.
                report.duplicate_assertions += 1
                continue
            seen_here.add(aid)

            for quote in grounded:
                self._write_span(quote.turn_id, quote.grounding)
                self._note_grounding(quote.turn_id, quote.grounding, quote.text, turn_groundings)
                report.quotes_total += 1

            draft = DraftAssertion(
                assertion_id=aid,
                kind=raw.kind,
                text_norm=raw.text_norm,
                quotes=grounded,
                asserted_by=turn.turn_id,
                from_repair=extraction.repairs > 0,
            )
            self._drafts[aid] = draft
            self._write_assertion(draft, turn)
            report.assertions_stored += 1

        # Last, deliberately — see the module docstring's crash-resume paragraph.
        self.log.append(_OP_TURN, turn.to_dict())

        report.rungs = dict(sorted(rung_counts(turn_groundings).items()))
        report.seconds = time.perf_counter() - started
        return report

    # -- writing -----------------------------------------------------------

    def _write_span(self, turn_id: str, grounding: Grounding) -> None:
        """Append a ``span.add``, once per distinct span in this run.

        Two assertions quoting the same words, or a mention inside an
        assertion's quote, produce the same content-derived ``span_id``.  Replay
        is idempotent either way — a later identical write overwrites an equal
        record — so the dedup is about the log staying legible, not about
        correctness.  It is scoped to the run rather than to the log, so a
        resumed run still rewrites the spans of the turns it reprocesses, which
        is what makes the crash-repair path work.
        """
        sid = span_id_of(turn_id, grounding.start, grounding.end)
        if sid in self._written_spans:
            return
        self._written_spans.add(sid)
        self.log.append(
            _OP_SPAN,
            SourceSpan(
                span_id=sid, turn_id=turn_id, start=grounding.start, end=grounding.end
            ).to_dict(),
        )

    def _write_mention(self, turn_id: str, grounding: Grounding, turn_text: str) -> None:
        """Append a ``mention.add`` naming the span a mention grounded to.

        ``text`` is the *turn's* substring (the offsets' selection), matching
        ``GroundedQuote``'s rule: on rungs 2 and 3 the model's surface form and
        the turn's differ, and stored text must be what the offsets select or
        provenance is a fiction.  ``rung`` travels with it because D1 items are
        training data and "how was this offset obtained" is part of the label's
        quality (decision 5's argument, applied to mentions).
        """
        sid = span_id_of(turn_id, grounding.start, grounding.end)
        if sid in self._written_mentions:
            return
        self._written_mentions.add(sid)
        self.log.append(
            OP_MENTION,
            {
                "span_id": sid,
                "turn_id": turn_id,
                "text": turn_text[grounding.start : grounding.end],
                "rung": grounding.rung,
            },
        )

    def _write_assertion(self, draft: DraftAssertion, turn: Turn) -> None:
        assertion = Assertion(
            assertion_id=draft.assertion_id,
            kind=draft.kind,
            text_norm=draft.text_norm,
            spans=draft.span_ids,
            flags=AssertionFlags(
                # The **speaker**, per G9: the assertion event is the utterance
                # being processed, and who said it is the source relation the
                # four-flag discipline names.
                asserted_by=turn.speaker,
                entailed_by_span=False,
                entailed_score=0.0,
                externally_verified=False,
                current_under_update_policy=True,
            ),
            # The turn's timestamp, never a wall clock: a clock would make two
            # runs of the same work write different bytes, which is exactly the
            # property ``EventLog.digest`` exists to check.
            t_created=turn.ts,
            eligibility="quarantined",
        )
        self.log.append(_OP_ASSERTION, assertion.to_dict())

    def _note_grounding(
        self, turn_id: str, grounding: Grounding, quoted: str, sink: list[Grounding]
    ) -> None:
        sink.append(grounding)
        self.rungs[grounding.rung] = self.rungs.get(grounding.rung, 0) + 1
        if grounding.rung == "fuzzy":
            # **All rung-3 spans are manually audited** (G5), so the worksheet
            # has to be produced by the run that made them.
            self.fuzzy_spans.append(
                {
                    "turn_id": turn_id,
                    "start": grounding.start,
                    "end": grounding.end,
                    "score": round(grounding.score, 4),
                    "snapped": grounding.snapped,
                    "model_quote": quoted,
                }
            )

    # -- pass 2 ------------------------------------------------------------

    def verify_and_gate(self) -> int:
        """Score every unverified assertion and write its flags and verdict.

        Returns how many verdicts were written.  Works from the log (see the
        module docstring), so a resumed run finishes what a dead one started.
        """
        started = time.perf_counter()
        pending = unverified_assertions(self.log)
        if not pending:
            self.verify_seconds += time.perf_counter() - started
            return 0

        snapshot = ReplayGraphStore(self.log).at()
        drafts: list[DraftAssertion] = []
        for aid in pending:
            draft = self._drafts.get(aid)
            if draft is None:
                draft = self._draft_from_snapshot(aid, snapshot)
                if draft is not None:
                    self.resumed_drafts += 1
            if draft is not None:
                drafts.append(draft)

        scores = score_drafts(self.verifier, drafts) if drafts else []
        for draft, score in zip(drafts, scores):
            entailed = apply_threshold(score, self.cfg.tau_nli)
            all_grounded = bool(draft.span_ids) and all(
                snapshot.span(sid) is not None for sid in draft.span_ids
            )
            verdict, reasons = eligibility(
                entailed_score=score, all_spans_grounded=all_grounded, cfg=self.cfg
            )
            self.log.append(
                _OP_FLAGS,
                {
                    "assertion_id": draft.assertion_id,
                    "entailed_by_span": entailed,
                    "entailed_score": float(score),
                    "verifier": getattr(self.verifier, "name", "unknown"),
                    "tau_nli": self.cfg.tau_nli,
                },
            )
            self.log.append(
                _OP_ELIGIBILITY,
                {
                    "assertion_id": draft.assertion_id,
                    "eligibility": verdict,
                    "reasons": list(reasons),
                    "support_policy": self.cfg.support_policy,
                },
            )
            self.tally.add(
                verdict, reasons, from_repair=draft.from_repair, rung=draft.worst_rung
            )
            self.scores.append(
                {
                    "assertion_id": draft.assertion_id,
                    "score": float(score),
                    "entailed_by_span": entailed,
                    "eligibility": verdict,
                    "worst_rung": draft.worst_rung,
                    "from_repair": draft.from_repair,
                }
            )
        self.verify_seconds += time.perf_counter() - started
        return len(drafts)

    def _draft_from_snapshot(self, aid: str, snapshot: Any) -> DraftAssertion | None:
        """Rebuild a draft for an assertion this process did not store.

        The resume path.  The premise is reconstructed from the *stored spans and
        turns*, which is the same premise pass 1 would have built, because a span
        is by definition ``turn.text[start:end]``.

        The **rung is not recoverable** — ``SourceSpan`` stores offsets, not how
        they were found — so a resumed draft reports ``resumed`` rather than
        claiming ``exact``.  The rung cross-tab is a pilot number and a resumed
        pilot must not silently report a cleaner one; ``resumed_drafts`` is in the
        report for the same reason.
        """
        stored = snapshot.assertion(aid)
        if stored is None:
            return None
        quotes: list[GroundedQuote] = []
        for sid in stored.spans:
            span = snapshot.span(sid)
            if span is None:
                return None
            turn = snapshot.turn(span.turn_id)
            if turn is None:
                return None
            quotes.append(
                GroundedQuote(
                    turn_id=span.turn_id,
                    span_id=sid,
                    text=turn.text[span.start : span.end],
                    grounding=Grounding(span.start, span.end, "resumed"),
                )
            )
        if not quotes:
            return None
        return DraftAssertion(
            assertion_id=aid,
            kind=stored.kind,
            text_norm=stored.text_norm,
            quotes=quotes,
            asserted_by=stored.flags.asserted_by,
            # Unknown, not clean — the repair count is not in the log, and the
            # rung above already reports ``resumed`` for the same reason.
            from_repair=None,
        )

    # -- the whole thing ---------------------------------------------------

    def run(self, turns: Sequence[Turn], conv_id: str | None = None) -> dict[str, Any]:
        """Both passes, stage-sequentially, with the extractor freed in between."""
        self.staged("extract", lambda: self.extract_slice(turns, conv_id))

        self.close(self.extractor)
        if self.summary is not None:
            self.summary.flush()

        verified = 0

        def _verify() -> None:
            nonlocal verified
            verified = self.verify_and_gate()

        self.staged("verify", _verify)
        self.close(self.verifier)
        return self.report(verified)

    def staged(self, name: str, fn: Any) -> None:
        """Run ``fn`` inside one ledger stage, or plainly when there is no ledger.

        Public because the pilot runs the two passes itself — it interleaves
        several conversations inside the extract stage — and a caller reaching for
        a private method to get stage attribution would be a caller quietly
        losing it.
        """
        if self.ledger is None:
            fn()
            return
        with self.ledger.stage(name):
            fn()

    @staticmethod
    def close(component: Any) -> None:
        """Free a stage's model if it has anything to free (fix F7)."""
        close = getattr(component, "close", None)
        if callable(close):
            close()

    # -- reporting ---------------------------------------------------------

    def report(self, verified: int = 0) -> dict[str, Any]:
        processed = [r for r in self.reports if not r.skipped]
        turns = len(processed)
        parse_failures = sum(1 for r in processed if not r.parse_ok)
        mentions = sum(r.mentions for r in processed)
        stored = sum(r.assertions_stored for r in processed)
        seconds = self.extract_seconds + self.verify_seconds
        return {
            "turns_seen": len(self.reports),
            "turns_processed": turns,
            "turns_skipped": self.skipped,
            "parse_failures": parse_failures,
            "parse_failure_rate": (parse_failures / turns) if turns else float("nan"),
            "repairs": sum(r.repairs for r in processed),
            "truncated_generations": sum(r.truncations for r in processed),
            "mentions": mentions,
            "mentions_per_turn": (mentions / turns) if turns else float("nan"),
            "mentions_dropped_ungrounded": sum(r.dropped_mentions for r in processed),
            "assertions_extracted": sum(r.assertions_extracted for r in processed),
            "assertions_stored": stored,
            "assertions_per_turn": (stored / turns) if turns else float("nan"),
            "assertions_dropped_ungrounded": sum(r.dropped_assertions for r in processed),
            # The third disposition, so the three account for every extraction:
            # extracted = stored + dropped_ungrounded + duplicates_within_turn.
            "assertions_duplicate_within_turn": sum(
                r.duplicate_assertions for r in processed
            ),
            "grounding_rungs": dict(sorted(self.rungs.items())),
            "fuzzy_spans_to_audit": len(self.fuzzy_spans),
            "boundary_snapped_spans": sum(1 for s in self.fuzzy_spans if s["snapped"]),
            "verdicts_written": verified,
            "resumed_drafts": self.resumed_drafts,
            "support": self.tally.to_dict(),
            "llm_calls": sum(r.llm_calls for r in processed),
            "llm_tokens_in": sum(r.tokens_in for r in processed),
            "llm_tokens_out": sum(r.tokens_out for r in processed),
            "llm_calls_median_per_turn": _median([r.llm_calls for r in processed]),
            "llm_tokens_out_median_per_turn": _median([r.tokens_out for r in processed]),
            "extract_seconds": self.extract_seconds,
            "verify_seconds": self.verify_seconds,
            "seconds_per_turn": (seconds / turns) if turns else float("nan"),
            "turns_per_hour": (turns / seconds * 3600.0) if seconds > 0 else float("nan"),
            "summary_calls": 0 if self.summary is None else self.summary.calls,
        }

    def metrics_rows(self) -> list[Mapping[str, Any]]:
        """One row per turn, for ``metrics.jsonl``."""
        return [r.to_dict() for r in self.reports]


def _median(values: Iterable[float]) -> float:
    items = sorted(values)
    if not items:
        return float("nan")
    mid = len(items) // 2
    if len(items) % 2:
        return float(items[mid])
    return (items[mid - 1] + items[mid]) / 2.0
