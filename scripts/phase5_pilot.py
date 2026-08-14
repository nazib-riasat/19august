"""The Phase-5 pilot (G10) — one declared object, one command, one report.

    python scripts/phase5_pilot.py --out artefacts/phase5_pilot.json
    python scripts/phase5_pilot.py --replay          # no GPU: the recorded spike extractions
    python scripts/phase5_pilot.py --questions 2     # a wiring check

**The pilot is part of the experimental record, so it is defined, not chosen**
(G10, decision 10):

> the Phase-2.5 sample's 10 questions with their **full** evidence sessions (no
> windowing — ~250 turns), ingested end to end through the frozen G2 winner; plus
> the ~50-question obligation-parser audit set (G7); plus the 50-assertion span
> audit (G1) and the ~50-pair NLI audit (G6) drawn from its output by seeded
> sample.

Three things come out: the report (every G-gap's measured number), the **sizing
memo** (G8), and the audit worksheets a human fills — span support, NLI
agreement, and the obligation slots.  The worksheets are CSVs in the same shape
as the Phase-2.5 annotate CLI's, minus the timing, because the person filling
them has already used that shape.

**Nothing here retunes anything.**  ``tau_nli`` is read from the config and
audited; a miscalibrated threshold is a reported finding and a Gate-0 amendment.
The quarantine rate is reported beside the sentence saying it is a quality
signal, not a knob.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from graft.config import load_config  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import corpus, pins  # noqa: E402
from graft.ingest.bakeoff import calibration_slice, calibration_turn_ids  # noqa: E402
from graft.ingest.extractor import ReplayExtractor, build_extractor  # noqa: E402
from graft.ingest.nli import NliVerifier, StubVerifier  # noqa: E402
from graft.ingest.oblparse import ObligationParser  # noqa: E402
from graft.ingest.pipeline import IngestPipeline  # noqa: E402
from graft.ingest.summary import RollingSummary  # noqa: E402
from graft.ledger import Ledger  # noqa: E402
from graft.runtime import json_sanitize, run_manifest  # noqa: E402

SAMPLE = REPO / "data" / "phase2_5" / "sample.json"
SPIKE_EXTRACTION = REPO / "data" / "phase2_5" / "extraction.jsonl"
AUDIT_SEED = 20260813  # the spike's seed convention, so the draw is re-derivable


def audit_rng(sheet: str) -> random.Random:
    """One independent seeded stream per worksheet.

    A single shared ``Random`` couples every later draw to the exact size of
    every earlier one — re-deriving the NLI sample would require replaying the
    span sheet's shuffle first, and any amendment to one sheet would silently
    re-deal the others.  Deriving each stream from ``(AUDIT_SEED, sheet name)``
    keeps every draw independently re-derivable.
    """
    return random.Random(f"{AUDIT_SEED}:{sheet}")


# --------------------------------------------------------------------------
# the pilot corpus
# --------------------------------------------------------------------------


def pilot_questions(limit: int | None = None) -> list[dict]:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    questions = sample["questions"]
    return questions[:limit] if limit else questions


def pilot_turns(questions, index, *, windowed: bool) -> dict[str, list]:
    """``question_id -> turns``, evidence sessions only.

    ``windowed=True`` restricts to the spike's ``has_answer ± 1`` windows, which
    is the only slice the recorded extractions cover — so ``--replay`` uses it and
    the live pilot does not.  G10's pilot is the **full** evidence sessions.
    """
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    keep: dict[str, set[str]] = {}
    for session in sample["sessions"]:
        keep.setdefault(session["question_id"], set()).update(
            t["turn_id"] for t in session["turns"]
        )

    out: dict[str, list] = {}
    for question in questions:
        qid = question["question_id"]
        instance = index[qid]
        turns = list(corpus.turns_of(instance, question["evidence_session_ids"]))
        if windowed:
            turns = [t for t in turns if t.turn_id in keep.get(qid, set())]
        out[qid] = turns
    return out


# --------------------------------------------------------------------------
# the sizing memo (G8)
# --------------------------------------------------------------------------


def sizing_memo(turns_per_hour: float, corpus_stats: dict, *, live: bool = True) -> dict:
    """Projected wall-clock for the three candidate scopes, on two targets.

    **This memo does not decide the scope.**  That is Gate-0 item 9's decision,
    taken with this in hand (decision 11).  What the memo is for is making the
    decision *possible*: the architecture's own exit criterion asks for a budget
    check for full corpora, and the honest answer on this machine is a number
    with an option table beside it, not a plan to let the ingestion run forever.

    The Kaggle column is a **scaled projection, not a measurement** — a T4 has no
    bf16 tensor cores and roughly 2× the memory of the dev card, so the factor is
    stated as an assumption and labelled.  Anyone quoting it must quote the label.
    """
    if not live:
        # A replay run's throughput is the speed of reading a JSONL file.  Costing
        # a corpus against it would produce a table of plausible, meaningless
        # hours — the exact failure mode this memo exists to prevent.
        return {
            "error": "no throughput measured: replay mode runs no model",
            "corpus": corpus_stats,
        }
    if not turns_per_hour or turns_per_hour != turns_per_hour:  # nan-safe
        return {"error": "no throughput measured; run the pilot without --replay"}

    kaggle_factor = 1.0  # [ANALYSIS] assumption: T4 fp16 ≈ this card's bf16 rate
    total_turns = corpus_stats["total_turns"]
    evidence_turns = corpus_stats["evidence_turns"]
    per_question_turns = total_turns / corpus_stats["questions"]
    evidence_per_question = evidence_turns / corpus_stats["questions"]

    def hours(n_turns: float, factor: float = 1.0) -> float:
        return n_turns / (turns_per_hour * factor)

    options = []
    options.append(
        {
            "scope": "a - the full corpus",
            "questions": corpus_stats["questions"],
            "turns": total_turns,
            "hours_dev_gpu": round(hours(total_turns), 1),
            "hours_kaggle_projected": round(hours(total_turns, kaggle_factor), 1),
        }
    )
    for d in (0, 2, 5):
        turns = evidence_turns + d * corpus_stats["questions"] * (
            per_question_turns / max(corpus_stats["sessions_per_question"], 1)
        )
        options.append(
            {
                "scope": f"b - evidence sessions + {d} sampled distractor sessions per question",
                "questions": corpus_stats["questions"],
                "turns": round(turns),
                "hours_dev_gpu": round(hours(turns), 1),
                "hours_kaggle_projected": round(hours(turns, kaggle_factor), 1),
            }
        )
    for q in (50, 100, 200):
        turns = q * evidence_per_question
        options.append(
            {
                "scope": f"c - {q}-question subset, evidence sessions only",
                "questions": q,
                "turns": round(turns),
                "hours_dev_gpu": round(hours(turns), 1),
                "hours_kaggle_projected": round(hours(turns, kaggle_factor), 1),
            }
        )

    return {
        "measured_turns_per_hour": round(turns_per_hour, 2),
        "corpus": corpus_stats,
        "options": options,
        "assumptions": [
            "throughput is the pilot's end-to-end rate, extraction plus verification",
            "[ANALYSIS] the Kaggle column scales the measured rate by "
            f"{kaggle_factor} and is a projection, not a measurement",
            "a 12-hour Kaggle session cap means any option over ~11 h needs "
            "resumable runs; ingestion is resumable by re-running (G4)",
        ],
        "decision": (
            "the corpus scope for Gates 1 and 4 is Gate-0 item 9's decision, taken "
            "with this memo — not Phase 5's (decision 11). The one commitment made "
            "here: the knowledge-update evidence sessions are in every candidate "
            "scope, because D2's supervision lives there and is the binding "
            "constraint on Contribution 1."
        ),
    }


def corpus_statistics(index, questions) -> dict:
    total_turns = 0
    total_sessions = 0
    for instance in index.values():
        total_turns += sum(len(s) for s in instance["haystack_sessions"])
        total_sessions += len(instance["haystack_session_ids"])
    evidence_turns = 0
    evidence_sessions = 0
    for instance in index.values():
        ids = set(instance.get("answer_session_ids", ()))
        for ix, sid in enumerate(instance["haystack_session_ids"]):
            if sid in ids:
                evidence_turns += len(instance["haystack_sessions"][ix])
                evidence_sessions += 1
    n = len(index)
    return {
        "questions": n,
        "total_turns": total_turns,
        "total_sessions": total_sessions,
        "sessions_per_question": round(total_sessions / n, 1),
        "evidence_turns": evidence_turns,
        "evidence_sessions": evidence_sessions,
        "pilot_questions": len(questions),
    }


# --------------------------------------------------------------------------
# the audit worksheets
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def from_calibration(assertion, snapshot, exclude_turn_ids: set) -> bool:
    """True when any of the assertion's spans lies in a calibration-slice turn.

    The G2 bakeoff selected the extractor **on** those turns, so auditing
    quality on them measures the turns the prompt and parse strategy were
    calibrated against — the declared disjointness (decision 2) is enforced
    here, at the draw, because the pilot corpus necessarily *ingests* the
    calibration turns (G10 defines it as the full evidence sessions).
    """
    for sid in assertion.spans:
        span = snapshot.span(sid)
        if span is not None and span.turn_id in exclude_turn_ids:
            return True
    return False


def span_worksheet(snapshot, log, rng, n: int, exclude_turn_ids: set = frozenset()) -> list[dict]:
    """G1's 50-assertion span-support audit, drawn by seeded sample.

    The protocol travels with the worksheet: *an assertion counts as supported
    iff its grounded span, read alone plus the turn it came from, textually
    commits to the assertion's ``text_norm``* — written into the file so a second
    auditor applies the same rule.
    """
    ids = sorted(
        {e.payload["assertion_id"] for e in log.replay() if e.op == "assertion.add"}
    )
    rng.shuffle(ids)
    rows = []
    for aid in ids:
        if len(rows) >= n:
            break
        assertion = snapshot.assertion(aid)
        if assertion is None:
            continue
        if from_calibration(assertion, snapshot, exclude_turn_ids):
            continue
        spans = []
        turns = []
        for sid in assertion.spans:
            span = snapshot.span(sid)
            turn = snapshot.turn(span.turn_id) if span else None
            if span and turn:
                spans.append(turn.text[span.start : span.end])
                turns.append(turn.text)
        rows.append(
            {
                "assertion_id": aid,
                "kind": assertion.kind,
                "text_norm": assertion.text_norm,
                "spans": " || ".join(spans),
                "turn_text": " || ".join(turns),
                "eligibility": assertion.eligibility,
                "supported": "",
                "note": "",
            }
        )
    return rows


def nli_worksheet(
    scores: list[dict],
    snapshot,
    rng,
    n: int,
    tau: float,
    exclude_turn_ids: set = frozenset(),
) -> list[dict]:
    """G6's ~50-pair NLI audit.

    Stratified around the threshold rather than uniform: a uniform draw from a
    distribution piled at one end measures agreement where the model is never in
    doubt, and the number that matters is agreement *at* ``tau_nli``.  The strata
    are recorded on each row so the reported agreement can be read per stratum
    and not only pooled.

    ``tau`` is a **parameter fed from the config**, never a literal: the frozen
    value's one home is ``Config.tau_nli`` (the pins module says so in its own
    docstring), and a hardcoded copy here would stratify the audit around a
    stale threshold after any Gate-0 amendment — the audit built to audit
    ``tau_nli`` would be the one thing not reading it.
    """
    usable = [
        s
        for s in scores
        if (a := snapshot.assertion(s["assertion_id"])) is not None
        and not from_calibration(a, snapshot, exclude_turn_ids)
    ]
    near = [s for s in usable if abs(s["score"] - tau) <= 0.25]
    far = [s for s in usable if abs(s["score"] - tau) > 0.25]
    rng.shuffle(near)
    rng.shuffle(far)
    take_near = min(len(near), max(n // 2, n - len(far)))
    chosen = near[:take_near] + far[: n - take_near]

    rows = []
    for item in chosen:
        assertion = snapshot.assertion(item["assertion_id"])
        if assertion is None:
            continue
        premise = []
        for sid in assertion.spans:
            span = snapshot.span(sid)
            turn = snapshot.turn(span.turn_id) if span else None
            if span and turn:
                premise.append(turn.text[span.start : span.end])
        rows.append(
            {
                "assertion_id": item["assertion_id"],
                "premise_spans": " ".join(premise),
                "hypothesis_text_norm": assertion.text_norm,
                "model_score": round(item["score"], 4),
                "model_entailed_at_tau": item["entailed_by_span"],
                "tau_nli": tau,
                "stratum": "near_threshold" if abs(item["score"] - tau) <= 0.25 else "far",
                "human_entailed": "",
                "note": "",
            }
        )
    return rows


def fuzzy_worksheet(fuzzy_spans: list[dict], snapshot) -> list[dict]:
    """**Every** rung-3 span (G5), not a sample: they are few, and the mis-bound
    rate is only meaningful over all of them."""
    rows = []
    for item in fuzzy_spans:
        turn = snapshot.turn(item["turn_id"])
        found = turn.text[item["start"] : item["end"]] if turn else ""
        rows.append(
            {
                **item,
                "grounded_text": found,
                "context": (turn.text[max(0, item["start"] - 60) : item["end"] + 60]
                            if turn else ""),
                "well_bounded": "",
                "note": "",
            }
        )
    return rows


def obligation_worksheet(questions: list[dict], predicted, traces) -> list[dict]:
    rows = []
    for question, obligations, trace in zip(questions, predicted, traces):
        rows.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "question_date": question["question_date"],
                "pred_entity_anchor": obligations.entity_anchor or "",
                "pred_value_type": obligations.value_type or "",
                "pred_time_expression": trace.get("time_expression") or "",
                "pred_time_resolved": trace.get("time_resolved"),
                "pred_time_unbounded": trace.get("time_unbounded"),
                "pred_needs_source": obligations.needs_source,
                "pred_aggregate": obligations.aggregate,
                "gold_entity_anchor": "",
                "gold_value_type": "",
                "gold_time_expression": "",
                "gold_needs_source": "",
                "gold_aggregate": "",
                "note": "",
            }
        )
    return rows


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artefacts/phase5_pilot.json")
    parser.add_argument("--run-dir", default="artefacts/phase5_pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="drive the write path from the recorded spike extractions; no GPU, "
        "and the report is marked as not measuring extraction quality",
    )
    parser.add_argument(
        "--no-obligations", action="store_true", help="skip the G7 parser audit set"
    )
    parser.add_argument(
        "--twice",
        action="store_true",
        help="run the whole pilot a SECOND time into a fresh log and compare "
        "digests (exit criterion 15 / G11). Roughly doubles the live run's GPU "
        "time; the same-log idempotence check (criterion 2) is reported too.",
    )
    args = parser.parse_args()

    cfg = load_config()
    run_dir = REPO / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    index = corpus.question_index(corpus.load_corpus())
    questions = pilot_questions(args.questions)
    turns_by_q = pilot_turns(questions, index, windowed=args.replay)
    n_turns = sum(len(v) for v in turns_by_q.values())
    print(
        f"pilot: {len(questions)} questions, {n_turns} turns "
        f"({'windowed replay' if args.replay else 'full evidence sessions'})",
        flush=True,
    )

    # The G2 calibration turns — the audit draws exclude any assertion touching
    # them (decision 2's declared disjointness; see from_calibration).  The same
    # definition the bakeoff runs, imported, not copied.
    excluded = calibration_turn_ids(calibration_slice(SAMPLE, index))

    ledger = Ledger.from_config(cfg, log=None)
    log = EventLog.open(log_path, fsync=cfg.fsync)

    if args.replay:
        extractor = ReplayExtractor.from_jsonl(SPIKE_EXTRACTION)
        verifier = StubVerifier(1.0)
        summary = None
        extractor_config = {"replay": str(SPIKE_EXTRACTION)}
    else:
        extractor = build_extractor(device=args.device, ledger=ledger)
        extractor_config = extractor.config
        summary = RollingSummary(
            lambda system, user: extractor.complete(
                system, user, max_new_tokens=pins.SUMMARY_MAX_TOKENS
            ),
            cache_dir=run_dir,
        )
        verifier = NliVerifier(device=args.device, ledger=ledger)

    pipeline = IngestPipeline(
        log, cfg, extractor, verifier, summary=summary, ledger=ledger
    )

    for question in questions:
        qid = question["question_id"]
        print(f"  [{qid}] {len(turns_by_q[qid])} turns", flush=True)
        pipeline.staged("extract", lambda q=qid: pipeline.extract_slice(turns_by_q[q], q))

    close = getattr(extractor, "close", None)
    if callable(close):
        close()
    if summary is not None:
        summary.flush()

    verified = 0

    def _verify() -> None:
        nonlocal verified
        verified = pipeline.verify_and_gate()

    pipeline.staged("verify", _verify)
    close = getattr(verifier, "close", None)
    if callable(close):
        close()

    report = pipeline.report(verified)
    snapshot = ReplayGraphStore(log).at()

    # -- the obligation-parser audit set (G7, decision 14) -----------------
    obligation_report = None
    obligation_rows: list[dict] = []
    if not args.replay and not args.no_obligations:
        # A seeded draw over sorted ids, stratified by question type: the six
        # types impose different slots (a temporal-reasoning question is where
        # `time_constraint` is exercised at all), so an unstratified 50 would
        # under-sample the slot the unbounded-constraint rate is about.
        by_type: dict[str, list[str]] = {}
        for qid in sorted(index):
            by_type.setdefault(index[qid]["question_type"], []).append(qid)
        picker = audit_rng("obligations")
        per_type = max(1, pins.OBLIGATION_AUDIT_N // len(by_type))
        chosen: list[str] = []
        for qtype in sorted(by_type):
            pool = list(by_type[qtype])
            picker.shuffle(pool)
            chosen.extend(pool[:per_type])
        chosen = sorted(chosen)[: pins.OBLIGATION_AUDIT_N]
        audit_questions = [corpus.question_meta(index[qid]) for qid in chosen]
        oparser = ObligationParser(lambda s, u: extractor.complete(s, u))
        extractor.load()
        predicted = [
            oparser.parse(
                q["question"], q["question_date"], scope=(q["question_id"],),
                question_id=q["question_id"],
            )
            for q in audit_questions
        ]
        extractor.close()
        obligation_report = oparser.report()
        obligation_rows = obligation_worksheet(audit_questions, predicted, oparser.traces)

    # -- worksheets --------------------------------------------------------
    # The span and NLI draws exclude calibration-slice assertions (decision 2's
    # disjointness).  The fuzzy sheet does not: it is *every* rung-3 span by
    # definition (G5), an instrument audit rather than a quality sample.
    sheets = {
        "span_support": write_csv(
            run_dir / "audit_span_support.csv",
            span_worksheet(
                snapshot, log, audit_rng("span"), pins.SPAN_AUDIT_N, excluded
            ),
            ["assertion_id", "kind", "text_norm", "spans", "turn_text", "eligibility",
             "supported", "note"],
        ),
        "nli_agreement": write_csv(
            run_dir / "audit_nli.csv",
            nli_worksheet(
                pipeline.scores, snapshot, audit_rng("nli"), 50, cfg.tau_nli, excluded
            ),
            ["assertion_id", "premise_spans", "hypothesis_text_norm", "model_score",
             "model_entailed_at_tau", "tau_nli", "stratum", "human_entailed", "note"],
        ),
        "fuzzy_spans": write_csv(
            run_dir / "audit_fuzzy_spans.csv",
            fuzzy_worksheet(pipeline.fuzzy_spans, snapshot),
            ["turn_id", "start", "end", "score", "snapped", "model_quote",
             "grounded_text", "context", "well_bounded", "note"],
        ),
    }
    if obligation_rows:
        sheets["obligation_slots"] = write_csv(
            run_dir / "audit_obligation_slots.csv",
            obligation_rows,
            list(obligation_rows[0]),
        )

    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in pipeline.metrics_rows():
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with open(run_dir / "nli_scores.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in pipeline.scores:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    log_digest = log.digest()
    graph_digest = snapshot.state_digest()

    # -- the G11 determinism check ----------------------------------------
    rerun = None
    if args.twice:
        # Criterion 2 at pilot scale: re-running over the SAME log must skip
        # every turn and move no digest.  This is idempotence, not determinism —
        # nothing is generated, so it cannot detect a nondeterministic model.
        second = IngestPipeline(log, cfg, extractor, verifier, ledger=None)
        for question in questions:
            second.extract_slice(turns_by_q[question["question_id"]],
                                 question["question_id"])
        second.verify_and_gate()
        idempotence = {
            "log_digest_unchanged": log.digest() == log_digest,
            "graph_digest_unchanged": ReplayGraphStore(log).at().state_digest() == graph_digest,
            "turns_skipped": second.skipped,
            "turns_expected": n_turns,
        }

        # Criterion 15 proper: the SAME work done twice, into a FRESH log, must
        # produce identical digests.  The old check reused the populated log, so
        # every turn skipped and the comparison could not fail — a
        # nondeterministic extraction stack would have shipped undetected (found
        # by the 13 Aug 2026 audit).  Models regenerate here, so this leg costs
        # a full second pass; the rerun is unmetered so the primary ledger stays
        # the first run's record.
        rerun_dir = run_dir / "rerun"
        rerun_dir.mkdir(parents=True, exist_ok=True)
        # A stale rerun log would make every turn skip (idempotence again, not
        # determinism), and a stale summary cache would serve the first run's
        # text instead of regenerating it.  The rerun must start cold.
        (rerun_dir / "events.jsonl").unlink(missing_ok=True)
        (rerun_dir / "summaries.json").unlink(missing_ok=True)
        fresh_log = EventLog.open(rerun_dir / "events.jsonl", fsync=cfg.fsync)
        old_ledgers = (getattr(extractor, "ledger", None), getattr(verifier, "ledger", None))
        if hasattr(extractor, "ledger"):
            extractor.ledger = None
        if hasattr(verifier, "ledger"):
            verifier.ledger = None
        fresh_summary = None
        if summary is not None:
            fresh_summary = RollingSummary(
                summary.summarizer, cache_dir=rerun_dir
            )
        fresh = IngestPipeline(
            log=fresh_log, cfg=cfg, extractor=extractor, verifier=verifier,
            summary=fresh_summary, ledger=None,
        )
        for question in questions:
            fresh.extract_slice(turns_by_q[question["question_id"]],
                                question["question_id"])
        close = getattr(extractor, "close", None)
        if callable(close):
            close()
        if fresh_summary is not None:
            fresh_summary.flush()
        fresh.verify_and_gate()
        close = getattr(verifier, "close", None)
        if callable(close):
            close()
        fresh_digest = fresh_log.digest()
        fresh_graph = ReplayGraphStore(fresh_log).at().state_digest()
        fresh_log.close()
        if hasattr(extractor, "ledger"):
            extractor.ledger = old_ledgers[0]
        if hasattr(verifier, "ledger"):
            verifier.ledger = old_ledgers[1]

        rerun = {
            "idempotence_same_log": idempotence,
            "determinism_fresh_log": {
                "log_digest_match": fresh_digest == log_digest,
                "graph_digest_match": fresh_graph == graph_digest,
                "first": {"log": log_digest, "graph": graph_digest},
                "second": {"log": fresh_digest, "graph": fresh_graph},
            },
        }

    artefact = {
        "phase": 5,
        "pilot": "GRAFT_PHASE5_BUILD.md G10, decision 10",
        "mode": "replay" if args.replay else "live",
        "manifest": run_manifest(cfg, seed=cfg.seeds[0], root=REPO),
        "ingestion_fingerprint": pins.ingestion_fingerprint(
            extractor_config if not args.replay else None
        ),
        "config": {
            "extractor": extractor_config,
            "device": args.device,
            "nli": pins.NLI,
            "tau_nli": cfg.tau_nli,
            "support_policy": cfg.support_policy,
            "context_turns": pins.CONTEXT_TURNS,
            "context_clip_chars": pins.CONTEXT_CLIP_CHARS,
            "summary_every": pins.SUMMARY_EVERY,
            "prompt_registry_sha": pins.frozen_values()["prompt_registry_sha"],
        },
        "audit_exclusion": {
            "rule": "assertions with any span in a G2 calibration-slice turn are "
            "excluded from the span and NLI audit draws (decision 2's declared "
            "disjointness); the fuzzy sheet is exhaustive by definition and is not",
            "excluded_turn_ids": len(excluded),
        },
        "corpus": {
            "source": corpus.CORPUS_SOURCE,
            "sha256": corpus.CORPUS_SHA256,
            "licence": corpus.CORPUS_LICENCE,
        },
        "questions": [q["question_id"] for q in questions],
        "report": report,
        "graph": snapshot.counts(),
        "ledger": ledger.snapshot(),
        "obligation_parser": obligation_report,
        "worksheets": sheets,
        "digests": {"log": log_digest, "graph": graph_digest},
        # A resumed pilot skips already-ingested turns, so every in-memory
        # number (per-turn metrics, NLI scores, fuzzy list, quarantine tally)
        # covers only THIS process's segment while the log and graph are
        # complete.  The marker makes a partial report say so instead of
        # quietly shipping a complete graph beside an incomplete report.
        "partial_report": report["turns_skipped"] > 0,
        "rerun": rerun,
        "sizing_memo": sizing_memo(
            report["turns_per_hour"],
            corpus_statistics(index, questions),
            live=not args.replay,
        ),
        "thresholds": {
            "parse_failure_ceiling": pins.PARSE_FAILURE_CEILING,
            "span_precision_floor": pins.SPAN_PRECISION_FLOOR,
            "span_audit_n": pins.SPAN_AUDIT_N,
        },
        "pending_human_audits": [
            "span support (G1): fill `supported` in audit_span_support.csv; "
            f"the criterion is >= {pins.SPAN_PRECISION_FLOOR:.2f}",
            "NLI agreement (G6): fill `human_entailed` in audit_nli.csv; tau_nli is "
            "audited here and NOT retuned",
            "rung-3 spans (G5): fill `well_bounded` in audit_fuzzy_spans.csv; every "
            "fuzzy span is audited, not a sample",
            "obligation slots (G7): fill the `gold_*` columns in "
            "audit_obligation_slots.csv, then score with core.obligations.slot_level_scores",
        ],
        "limitations": [
            "cross-machine byte identity is explicitly NOT promised for Stage A "
            "(G11): a bf16 forward pass is not bit-stable across GPU "
            "architectures. What must match across machines is the "
            "ingestion_fingerprint above — model id, revision, decoding and "
            "prompt SHA — not the log digest.",
            "the quarantine rate is an extraction-quality signal, never grounds "
            "for quietly lowering tau_nli (architecture fix F9).",
        ],
    }
    if args.replay:
        artefact["limitations"].insert(
            0,
            "REPLAY MODE: extraction was served from the Phase-2.5 recording and "
            "entailment from a fixed-score stub, so no number in this artefact "
            "measures extraction or entailment quality. It exercises the write "
            "path, idempotence and the report shape only. The span and NLI "
            "worksheets are empty by construction here: the replay slice IS the "
            "calibration slice, and the audit draws exclude it.",
        )
    if artefact["partial_report"]:
        artefact["limitations"].insert(
            0,
            "PARTIAL REPORT: this run resumed an existing log, so per-turn "
            "metrics, NLI scores, the fuzzy-span worksheet and the quarantine "
            "tally cover only the resumed segment. The digests and graph counts "
            "cover everything. For a complete report, re-run into a fresh "
            "--run-dir.",
        )

    log.close()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_sanitize(artefact), indent=1, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )

    print("\n=== pilot report ===")
    for key in (
        "turns_processed", "parse_failure_rate", "truncated_generations",
        "mentions_per_turn", "assertions_per_turn", "assertions_dropped_ungrounded",
        "grounding_rungs", "turns_per_hour", "llm_calls", "llm_tokens_out",
    ):
        print(f"  {key:<32} {report[key]}")
    print(f"  {'quarantine_rate':<32} {report['support']['quarantine_rate']}")
    print(f"  {'gating_causes':<32} {report['support']['gating_causes']}")
    print(f"\n  {report['support']['reading']}")
    print(f"\nworksheets  {sheets}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
