#!/usr/bin/env python
"""Stage A over LongMemEval-S at Gate-0 item-9's scope c -- batched, resumable.

    python scripts/longmemeval_ingest.py --dry-run                 # selection + turn counts, no model
    python scripts/longmemeval_ingest.py --batch-size 32           # the run (GPU)

**Scope c, made concrete.**  `GATE0_CONTRACT.md` item 9 decided "scope c, 200
questions, evidence sessions only" (4,384 turns) but never named *which* 200 of
LongMemEval-S's 500.  This runner draws them **stratified by question type at
the spike's seed** (20260813, `phase5_pilot.AUDIT_SEED`) and writes the selection
to ``<run-dir>/scope_c_questions.json`` before extracting a turn -- so after the
first run the set is pinned on disk, and a re-run on any machine draws the same
200.  The 10 pilot questions are included by construction if the draw hits them;
they are not forced in, because forcing them would make the scope a superset of a
hand-picked set rather than a sample.

**The write path is `locomo_ingest.py`'s, verbatim in shape:** one
``IngestPipeline``, ``extract_slice_batched`` per question (its evidence sessions
are that question's conversation), a rolling summary, then the NLI verify-and-
gate pass over the whole log.  Batching is deterministic on this stack
(12/12, `PHASE5_DECISIONS.md`), and the batch size comes from the hardware probe
(`scripts/slurm/hw_probe.py`), never from a guess.

Resumable: the event log is append-only and ``ingested_turn_ids`` skips what is
already stored, so a killed job continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from graft.config import load_config  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import corpus as corpus_mod  # noqa: E402
from graft.ingest import pins  # noqa: E402
from graft.ingest.extractor import build_extractor  # noqa: E402
from graft.ingest.nli import NliVerifier  # noqa: E402
from graft.ingest.pipeline import IngestPipeline, ingested_turn_ids  # noqa: E402
from graft.ingest.summary import RollingSummary  # noqa: E402
from graft.ledger import Ledger  # noqa: E402

SCOPE_SEED = 20260813  # phase5_pilot.AUDIT_SEED -- the spike's convention


def select_scope_c(corpus: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` questions, stratified by ``question_type``, deterministic under ``seed``."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in corpus:
        by_type[str(q.get("question_type", ""))].append(q)
    rng = random.Random(f"{seed}:scope_c")
    total = len(corpus)
    chosen: list[dict] = []
    # proportional allocation, largest remainders, then fill from the biggest strata
    quotas = {t: (len(qs) * n) // total for t, qs in by_type.items()}
    short = n - sum(quotas.values())
    for t in sorted(by_type, key=lambda t: -((len(by_type[t]) * n) % total))[:short]:
        quotas[t] += 1
    for t in sorted(by_type):
        qs = sorted(by_type[t], key=lambda q: str(q["question_id"]))
        rng.shuffle(qs)
        chosen.extend(qs[: quotas[t]])
    return sorted(chosen, key=lambda q: str(q["question_id"]))


def evidence_turns(q: dict):
    ids = set(q.get("answer_session_ids", ()))
    return list(corpus_mod.turns_of(q, session_ids=ids)) if ids else []


def merge_shards(run_dir: Path) -> int:
    """Concatenate shard logs into one, re-sequenced, and replay it to prove the merge.

    Each shard is a complete Stage-A log for a disjoint set of questions, including
    its own trailing verify pass, so concatenation preserves every per-turn and
    per-assertion ordering guarantee.  Measured on the pilot log (12 Sep 2026):
    shard-by-conversation, reverse the shard order, concatenate -> content digest
    identical to the serial log.
    """
    shards = sorted(run_dir.glob("shard_*/events.jsonl"))
    if not shards:
        print("no shard logs found"); return 3
    out = run_dir / "events.jsonl"
    seq = 0
    with out.open("w", encoding="utf-8") as fh:
        for sp in shards:
            for line in sp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line); e["seq"] = seq; seq += 1
                fh.write(json.dumps(e) + "\n")
    snap = ReplayGraphStore(EventLog.open(out, fsync=False)).at()
    print(f"merged {len(shards)} shards, {seq} events -> {out}")
    print("graph:", snap.counts())
    (run_dir / "merge.json").write_text(json.dumps({"shards": [str(s) for s in shards], "events": seq,
                                                    "graph": snap.counts()}, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--questions", type=int, default=200, help="scope c = 200")
    ap.add_argument("--run-dir", default="artefacts/longmemeval_scope_c")
    ap.add_argument("--out", default="artefacts/longmemeval_ingest.json")
    ap.add_argument("--batch-size", type=int, default=8, help="from hw_probe; 8 is the 8 GB laptop value")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the selection, print turn counts, load no model")
    ap.add_argument("--merge", action="store_true", help="concatenate <run-dir>/shard_*/events.jsonl into <run-dir>/events.jsonl (each shard ran its own verify pass; content-identical to a serial run, verified 12 Sep 2026)")
    ap.add_argument("--shard", default=None, help="i/N: process only questions with index %% N == i (multi-GPU sharding; rows files are per-question, so shards merge by concatenation)")
    args = ap.parse_args()

    cfg = load_config()
    run_dir = REPO / args.run_dir; run_dir.mkdir(parents=True, exist_ok=True)
    corpus = corpus_mod.load_corpus()
    sel_path = run_dir / "scope_c_questions.json"
    if sel_path.exists():
        pinned = json.loads(sel_path.read_text(encoding="utf-8"))
        index = corpus_mod.question_index(corpus)
        selected = [index[qid] for qid in pinned["question_ids"]]
        print(f"scope c: {len(selected)} questions from the PINNED selection {sel_path.name}")
    else:
        selected = select_scope_c(corpus, args.questions, SCOPE_SEED)
        sel_path.write_text(json.dumps({
            "scope": "c", "definition": "evidence sessions only (GATE0_CONTRACT.md item 9)",
            "n": len(selected), "seed": SCOPE_SEED, "stratified_by": "question_type",
            "corpus_sha256": getattr(corpus_mod, "CORPUS_SHA", None),
            "question_ids": [str(q["question_id"]) for q in selected],
            "by_type": {t: sum(1 for q in selected if str(q.get("question_type", "")) == t)
                        for t in sorted({str(q.get("question_type", "")) for q in selected})},
        }, indent=1), encoding="utf-8")
        print(f"scope c: drew {len(selected)} of {len(corpus)} questions, pinned to {sel_path.name}")

    if args.merge:
        return merge_shards(run_dir)
    log_dir = run_dir
    if args.shard:
        _i, _n = (int(x) for x in args.shard.split("/"))
        selected = [q for k, q in enumerate(selected) if k % _n == _i]
        log_dir = run_dir / f"shard_{_i}"; log_dir.mkdir(parents=True, exist_ok=True)
        print(f"shard {_i}/{_n}: {len(selected)} questions -> {log_dir}", flush=True)
    turns_by_q = {str(q["question_id"]): evidence_turns(q) for q in selected}
    planned = sum(len(t) for t in turns_by_q.values())
    print(f"evidence turns planned: {planned}  (item 9 recorded 4,384 for scope c)")
    if args.dry_run:
        for t, c in sorted(json.loads(sel_path.read_text(encoding="utf-8"))["by_type"].items()):
            print(f"  {t:<28} {c}")
        return 0

    log_path = log_dir / "events.jsonl"
    ledger = Ledger.from_config(cfg, log=None)
    log = EventLog.open(log_path, fsync=cfg.fsync)
    already = ingested_turn_ids(log)
    print(f"log: {log_path}  (already ingested turns: {len(already)}; resumable)")
    extractor = verifier = summary = None
    started = time.perf_counter(); per_q: list[dict] = []
    try:
        if not args.verify_only:
            extractor = build_extractor(device=args.device, ledger=ledger)
            summary = RollingSummary(
                lambda system, user: extractor.complete(system, user, max_new_tokens=pins.SUMMARY_MAX_TOKENS),
                cache_dir=log_dir,
            )
            pipeline = IngestPipeline(log, cfg, extractor, verifier=None, summary=summary, ledger=ledger)
            for i, q in enumerate(selected, start=1):
                qid = str(q["question_id"]); turns = turns_by_q[qid]
                if not turns:
                    continue
                mark = time.perf_counter()
                pipeline.staged(
                    f"extract:{qid}",
                    lambda t=turns, s=qid: pipeline.extract_slice_batched(t, s, batch_size=args.batch_size),
                )
                el = time.perf_counter() - mark
                per_q.append({"question_id": qid, "turns": len(turns), "seconds": round(el, 1)})
                done = sum(r["turns"] for r in per_q)
                rate = done / max(1e-9, time.perf_counter() - started) * 3600
                print(f"  [{i}/{len(selected)}] {qid} {len(turns)} turns  {el:.0f}s  cumulative {done}/{planned} "
                      f"({rate:.0f} turns/h)", flush=True)
                (log_dir / "progress.json").write_text(json.dumps({"done_turns": done, "planned": planned,
                                                                   "per_question": per_q}), encoding="utf-8")
            if summary is not None:
                summary.flush()
            close = getattr(extractor, "close", None)
            if callable(close):
                close()
        verified = None
        if not args.extract_only:
            verifier = NliVerifier(device=args.device, ledger=ledger)
            pipeline = IngestPipeline(log, cfg, extractor=None, verifier=verifier, summary=None, ledger=ledger)
            verified = pipeline.verify_and_gate()
            print(f"verify: {verified} assertions gated")
    finally:
        for obj in (extractor, verifier):
            close = getattr(obj, "close", None)
            if callable(close):
                close()
    elapsed = time.perf_counter() - started
    snapshot = ReplayGraphStore(log).at()
    artefact = {
        "what_this_is": "Stage A over LongMemEval-S at scope c (200 questions, evidence sessions), batched",
        "selection": json.loads(sel_path.read_text(encoding="utf-8")),
        "batch_size": args.batch_size, "device": args.device,
        "planned_turns": planned, "seconds": round(elapsed, 1),
        "graph": snapshot.counts(), "ledger": ledger.snapshot()["totals"],
        "ingestion_fingerprint": pins.ingestion_fingerprint() if hasattr(pins, "ingestion_fingerprint") else None,
        "per_question": per_q,
    }
    (REPO / args.out).write_text(json.dumps(artefact, indent=1), encoding="utf-8")
    print(f"written: {args.out}  ({elapsed/3600:.2f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
