"""LoCoMo ingestion — Stage A over the evaluation corpus, conversation by conversation.

**Why this is a sibling of ``phase5_pilot.py`` rather than a flag on it.**  The
pilot is an audited instrument that produced `PHASE5_DECISIONS.md` §2's run
record; adding a second corpus to it would put a frozen measurement and a new one
in the same code path.  Everything that does the actual work -- ``IngestPipeline``,
the extractor, the verifier, the summary -- is imported, not re-implemented, so
this runner is a driver and nothing else.

**The design is "harvest whatever finished".**  `DATASET_DECISION.md` §1 prices
full LoCoMo at ~43 GPU hours on the dev card, which against a three-day deadline
is arithmetically possible and calendar-reckless.  Two properties make the choice
unnecessary:

* **Questions belong to conversations.**  Ingest 6 of 10 and you can evaluate
  every question those 6 own -- roughly 1,200 of 1,986 -- so scope is decided by
  when you stop, not in advance.
* **Resume is measured, not hoped for.**  `PHASE5_DECISIONS.md` §2 records
  ``turns_skipped = 248/248`` on a re-run against the same log, because
  ``turn.add`` is appended *last* and pass 2 works from the log.  A crash costs
  one turn.

So: run it, let it go, stop it when the GPU is needed.  ``--conversations``
bounds a run; re-running with a larger bound continues rather than restarts.

**Two stages, and the order is deliberate.**  Extraction is 99.87% of wall clock
(`artefacts/phase5_pilot.json`), so ``--extract-only`` lets extraction run
unattended for hours and the cheap verify pass (8.56 s across 248 turns) run
later against the log.

**Probe first.**  ``--probe`` reads the corpus, checks every structural
assumption in ``graft.ingest.locomo`` against the counts recorded in
`DATASET_DECISION.md`, and exits. Seconds of CPU, no GPU, no model load. Run it
before spending 43 hours.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graft.config import load_config  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import locomo  # noqa: E402
from graft.ingest import pins  # noqa: E402
from graft.ingest.extractor import build_extractor  # noqa: E402
from graft.ingest.nli import NliVerifier  # noqa: E402
from graft.ingest.pipeline import IngestPipeline  # noqa: E402
from graft.ingest.summary import RollingSummary  # noqa: E402
from graft.ledger import Ledger  # noqa: E402

#: Measured on the pilot: 135.67 turns/hour, 26.5 s/turn, extraction 99.87% of it.
MEASURED_TURNS_PER_HOUR = 135.67


def _eta(turns: int) -> str:
    hours = turns / MEASURED_TURNS_PER_HOUR
    return f"{turns} turns ~= {hours:.1f} h at the pilot's measured 135.67 turns/h"


def cmd_probe(args: argparse.Namespace) -> int:
    report = locomo.probe(args.corpus)
    print(json.dumps(report, indent=2))
    print()
    print(f"verdict: {report['verdict']}")
    print(f"next:    {report['next_step']}")
    if report["findings"]:
        print()
        print("findings:")
        for f in report["findings"]:
            print(f"  - {f}")
    print()
    print(f"cost if ingested in full: {_eta(report['turns'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    # A mismatch is a finding, not a crash: the point is to report all of them.
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Per-conversation turn counts and cumulative time. No model, no GPU.

    Exists so the stopping decision is made against arithmetic rather than
    against a progress bar at 2 a.m.
    """
    corpus = locomo.load_corpus(args.corpus)
    rows = []
    cumulative = 0
    for sample in corpus:
        turns = sum(1 for _ in locomo.turns_of(sample))
        questions = len(locomo.questions_of(sample))
        adversarial = sum(1 for q in locomo.questions_of(sample) if q["adversarial"])
        cumulative += turns
        rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "turns": turns,
                "questions": questions,
                "adversarial": adversarial,
                "cumulative_turns": cumulative,
                "cumulative_hours": round(cumulative / MEASURED_TURNS_PER_HOUR, 2),
            }
        )

    print(f"{'#':>3}  {'sample_id':<24} {'turns':>6} {'quest':>6} {'adv':>5} "
          f"{'cum turns':>10} {'cum h':>7}")
    for i, r in enumerate(rows, start=1):
        print(f"{i:>3}  {r['sample_id']:<24} {r['turns']:>6} {r['questions']:>6} "
              f"{r['adversarial']:>5} {r['cumulative_turns']:>10} {r['cumulative_hours']:>7}")
    print()
    print("Stop wherever the calendar says stop; questions belong to conversations,")
    print("so whatever finished is a complete, reportable evaluation set.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    # A verify-only pass must not clobber the extract run's artefact: on the
    # 19 Aug 2026 run both wrote the default --out, and step 4's near-empty
    # summary overwrote step 3's per-conversation record (the log and
    # progress.json kept everything, but the JSON summary was lost).
    if args.verify_only and args.out == "artefacts/locomo_ingest.json":
        args.out = "artefacts/locomo_verify.json"
    cfg = load_config()
    run_dir = REPO / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    corpus = locomo.load_corpus(args.corpus, expect_sha=args.expect_sha)
    samples = list(corpus)
    if args.conversations is not None:
        samples = samples[: args.conversations]

    planned = sum(sum(1 for _ in locomo.turns_of(s)) for s in samples)
    print(f"LoCoMo ingest: {len(samples)} of {len(corpus)} conversations, {_eta(planned)}")
    print(f"log: {log_path} (resumable -- a re-run continues, it does not restart)")
    if args.extract_only:
        print("extract-only: the verify pass is deferred; re-run with --verify-only")
    if args.batch_size > 1:
        print(
            f"batch size {args.batch_size}: amortising the weight read across turns. "
            "Verify with `verify-batch` if you have not -- batching can change "
            "numerics even though it does not move the fingerprint."
        )

    ledger = Ledger.from_config(cfg, log=None)
    log = EventLog.open(log_path, fsync=cfg.fsync)

    extractor = None
    verifier = None
    summary = None
    started = time.perf_counter()
    per_conversation: list[dict] = []

    try:
        if not args.verify_only:
            extractor = build_extractor(device=args.device, ledger=ledger)
            summary = RollingSummary(
                lambda system, user: extractor.complete(
                    system, user, max_new_tokens=pins.SUMMARY_MAX_TOKENS
                ),
                cache_dir=run_dir,
            )
            pipeline = IngestPipeline(
                log, cfg, extractor, verifier=None, summary=summary, ledger=ledger
            )
            for i, sample in enumerate(samples, start=1):
                sid = str(sample["sample_id"])
                turns = list(locomo.turns_of(sample))
                mark = time.perf_counter()
                pipeline.staged(
                    "extract",
                    lambda t=turns, s=sid: pipeline.extract_slice_batched(
                        t, s, batch_size=args.batch_size
                    ),
                )
                elapsed = time.perf_counter() - mark
                row = {
                    "sample_id": sid,
                    "turns": len(turns),
                    "seconds": round(elapsed, 1),
                    "turns_per_hour": round(len(turns) / max(elapsed, 1e-9) * 3600, 1),
                }
                per_conversation.append(row)
                print(
                    f"  [{i}/{len(samples)}] {sid}: {len(turns)} turns in "
                    f"{elapsed / 60:.1f} min ({row['turns_per_hour']:.0f} turns/h)",
                    flush=True,
                )
                # Written after every conversation, not at the end: the whole
                # point of harvest-what-finished is that a run killed mid-corpus
                # still leaves a usable record of what completed.
                _write_progress(run_dir, per_conversation, samples, corpus, args)
            close = getattr(extractor, "close", None)
            if callable(close):
                close()
            if summary is not None:
                summary.flush()

        verified = None
        if not args.extract_only:
            verifier = NliVerifier(device=args.device, ledger=ledger)
            pipeline = IngestPipeline(
                log, cfg, extractor=None, verifier=verifier, summary=None, ledger=ledger
            )
            holder = {}
            def _verify() -> None:
                holder["n"] = pipeline.verify_and_gate()
            pipeline.staged("verify", _verify)
            verified = holder.get("n")
            print(f"verify: {verified} assertions gated")
            close = getattr(verifier, "close", None)
            if callable(close):
                close()
    finally:
        for obj in (extractor, verifier):
            close = getattr(obj, "close", None)
            if callable(close):
                close()

    elapsed = time.perf_counter() - started
    snapshot = ReplayGraphStore(log).at()
    artefact = {
        "corpus": {
            "source": locomo.CORPUS_SOURCE,
            "licence": locomo.CORPUS_LICENCE,
            "sha256": locomo.corpus_sha256(args.corpus),
            "conversations_available": len(corpus),
            "conversations_ingested": len(samples),
        },
        "batch_size": args.batch_size,
        "batching_note": (
            "batch_size does not enter ingestion_fingerprint (it hashes model_id, "
            "revision, dtype, quantization, repair, constrained), so a batched run "
            "is the same experiment. It can change numerics: different matmul "
            "shapes reduce in a different order and a near-tie argmax can flip. "
            "`verify-batch` measures that rather than assuming it."
        ),
        "zero_shot_declaration": (
            "LoCoMo is evaluation only. No component trains on it: Stage D trains on "
            "2Wiki + MuSiQue-Ans, Stage B on LongMemEval, the gate on MuSiQue-Full, "
            "and the reader is frozen (DATASET_DECISION.md §1)."
        ),
        "per_conversation": per_conversation,
        "verified": verified,
        "wall_clock_s": round(elapsed, 1),
        "ledger": ledger.snapshot(),
        "turns_in_graph": len(getattr(snapshot, "turns", ()) or ()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    print(f"written: {out} ({elapsed / 3600:.2f} h)")
    return 0


def _write_progress(run_dir: Path, rows: list[dict], samples, corpus, args) -> None:
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "conversations_done": len(rows),
                "conversations_planned": len(samples),
                "conversations_available": len(corpus),
                "turns_done": sum(r["turns"] for r in rows),
                "per_conversation": rows,
                "reading": (
                    "questions belong to conversations, so the conversations listed "
                    "here are a complete evaluation set on their own"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def cmd_verify_batch(args: argparse.Namespace) -> int:
    """Extract the same turns single-stream and batched, and compare. **~10 min GPU.**

    The one claim batching makes that cannot be reasoned about: that it does not
    change what the model emits.  ``ingestion_fingerprint`` hashes ``model_id``,
    ``revision``, ``dtype``, ``quantization``, ``repair`` and ``constrained`` -- not
    batch size -- so a batched run is the same *experiment* by that definition.
    But different matmul shapes reduce in a different order, so a near-tie argmax
    can flip and one turn can decode differently in a batch than alone.

    `CLAUDE.md` §5's standing lesson is that this is exactly the kind of thing to
    check at the boundary rather than assert.  So: N turns, both ways, on the same
    model, and a per-turn diff.

    A mismatch is **not** a failure -- it is the number that lets the speed/
    byte-identity trade be made deliberately.
    """
    cfg = load_config()
    corpus = locomo.load_corpus(args.corpus, expect_sha=args.expect_sha)
    sample = corpus[0]
    turns = list(locomo.turns_of(sample))[: args.turns]
    print(f"verify-batch: {len(turns)} turns of {sample['sample_id']}, "
          f"single-stream vs batch {args.batch_size}")

    ledger = Ledger.from_config(cfg, log=None)
    extractor = build_extractor(device=args.device, ledger=ledger)
    try:
        extractor.load()
        # Contexts built exactly as the pipeline builds them, from raw turns only.
        from graft.ingest.summary import context_window
        from graft.ingest.extractor import ExtractionContext

        contexts = [
            ExtractionContext(
                summary="", window=context_window(turns, ix, pins.CONTEXT_TURNS),
                session_date=t.ts,
            )
            for ix, t in enumerate(turns)
        ]

        t0 = time.perf_counter()
        single = [extractor.extract(t, c) for t, c in zip(turns, contexts)]
        single_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        batched: list = []
        for lo in range(0, len(turns), args.batch_size):
            batched.extend(
                extractor.extract_batch(
                    turns[lo : lo + args.batch_size], contexts[lo : lo + args.batch_size]
                )
            )
        batch_s = time.perf_counter() - t0
    finally:
        close = getattr(extractor, "close", None)
        if callable(close):
            close()

    def fingerprint(e) -> str:
        """The extraction's **content**, via the record's own ``to_dict``.

        Built from ``to_dict`` rather than hand-picked attributes: the first
        version read ``a.text`` and ``RawAssertion``'s field is ``text_norm``, so
        it crashed *after* all the GPU work was done. Delegating to the record
        means a field rename cannot silently break this again.

        Cost fields are deliberately excluded. ``llm_calls``, ``tokens_in`` and
        ``tokens_out`` legitimately differ between the two paths -- a repaired turn
        costs an extra call, and batched token counts come from the attention mask
        -- so including them would report a mismatch for every turn and measure
        nothing about what the model *said*.
        """
        d = e.to_dict()
        return json.dumps(
            {
                "parse_ok": d["parse_ok"],
                "mentions": sorted(d["mentions"]),
                # Quotes included: provenance is content, and a batch that changed
                # which span an assertion cites has changed the graph.
                "assertions": sorted(
                    json.dumps(a, sort_keys=True) for a in d["assertions"]
                ),
            },
            sort_keys=True,
        )

    rows = []
    identical = 0
    for i, (a, b) in enumerate(zip(single, batched)):
        same = fingerprint(a) == fingerprint(b)
        identical += int(same)
        rows.append({
            "turn_id": turns[i].turn_id,
            "identical": same,
            "single_parse_ok": a.parse_ok,
            "batched_parse_ok": b.parse_ok,
            "single_assertions": len(a.assertions or ()),
            "batched_assertions": len(b.assertions or ()),
        })

    rate = identical / max(len(rows), 1)
    speedup = single_s / max(batch_s, 1e-9)
    report = {
        "turns": len(rows),
        "batch_size": args.batch_size,
        "single_stream_seconds": round(single_s, 1),
        "batched_seconds": round(batch_s, 1),
        "speedup": round(speedup, 2),
        "identical_extractions": identical,
        "identical_rate": round(rate, 4),
        "projected_full_corpus_hours": round(
            locomo.MEASURED_TURNS / MEASURED_TURNS_PER_HOUR / max(speedup, 1e-9), 1
        ),
        "reading": (
            "identical_rate 1.0 means batching is free on this machine and the "
            "speedup is pure gain. Below 1.0 it is a deliberate trade: the "
            "fingerprint does not move, so the run is the same experiment, but the "
            "extractions are not byte-identical to a single-stream run."
        ),
        "per_turn": rows,
    }
    print()
    print(f"single-stream : {single_s:7.1f} s")
    print(f"batched       : {batch_s:7.1f} s   ({speedup:.2f}x)")
    print(f"identical     : {identical}/{len(rows)}  ({100 * rate:.1f}%)")
    print(f"full corpus at this speedup: ~{report['projected_full_corpus_hours']} h "
          f"(vs 43.4 h single-stream)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


def main() -> int:
    # **Measured, after a first version of this comment overstated it.**  These
    # docstrings carry U+2192 and U+03B2, Windows consoles default to cp1252, and
    # `--help` died on the description before printing a word of it -- that part
    # is reproduced, on six runners.
    #
    # The guard also covers `print` of *data*, but the original justification
    # ("a curly apostrophe would kill the run") was **wrong**: U+2019 and U+2014
    # are cp1252 0x92/0x97 and encode fine.  What LoCoMo actually holds outside
    # cp1252 is 18 occurrences of 11 characters -- 8 zero-width spaces and 9
    # emoji -- across 7 turns and 1 gold answer.  And no current print path in
    # these runners emits corpus text, so this is insurance against a future
    # debug print, not a live crash averted.  `scripts/phase3_calibrate.py` set
    # the convention; extended here 19 Aug 2026.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default="data/locomo/locomo10.json",
        help="the LoCoMo JSON file (snap-research/locomo release)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="check the corpus against every recorded expectation; seconds, no GPU")
    p.add_argument("--out", default="artefacts/locomo_probe.json")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("plan", help="per-conversation turn counts and cumulative hours; no GPU")
    p.add_argument("--out", default="artefacts/locomo_plan.json")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("ingest", help="run Stage A over LoCoMo; resumable")
    p.add_argument("--out", default="artefacts/locomo_ingest.json")
    p.add_argument("--run-dir", default="artefacts/locomo")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--conversations", type=int, default=None,
        help="ingest only the first N conversations. Re-running with a larger N "
        "continues from the log rather than restarting.",
    )
    p.add_argument("--expect-sha", default=None, help="refuse a corpus whose SHA differs")
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="turns per model call. **1 is the default and is the audited "
        "single-stream path.** Higher amortises the 6.18 GB weight read across "
        "sequences: measured 19 Aug 2026, only 1.6%% of a 26.5 s turn is grammar "
        "work, so ~98%% is what a batch amortises. 8 is the recommended try on "
        "8 GB. Batching does NOT move ingestion_fingerprint, but it can change "
        "numerics -- run `verify-batch` first.",
    )
    p.add_argument(
        "--extract-only", action="store_true",
        help="skip the verify pass. Extraction is 99.87%% of wall clock, so this "
        "lets it run unattended and the cheap verify run later.",
    )
    p.add_argument(
        "--verify-only", action="store_true",
        help="run only the verify+gate pass, from the existing log. No extractor "
        "is loaded, so this needs almost no GPU.",
    )
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser(
        "verify-batch",
        help="extract the same turns single-stream and batched, and report both the "
        "speedup and whether the extractions are identical. ~10 min GPU.",
    )
    p.add_argument("--turns", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-sha", default=None)
    p.add_argument("--out", default="artefacts/locomo_verify_batch.json")
    p.set_defaults(fn=cmd_verify_batch)

    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
