#!/usr/bin/env python
"""Full-context baseline on LoCoMo -- the comparator `CLAUDE.md` §9 calls non-negotiable.

    python scripts/locomo_fullcontext_baseline.py --conversations 1     # chunk by conversation
    python scripts/locomo_fullcontext_baseline.py                       # resumes per question

**What it is.**  The reader is handed the **entire conversation** -- every turn,
chronological, dated, speaker-tagged, numbered ``[c1]..[cN]`` so prompt rule 5
is satisfiable -- and the same question.  No retrieval, no graph, no budget: the
budget *is* the conversation (11k-22k reader tokens on LoCoMo, inside
Qwen2.5-3B's 32k window).  Same frozen reader, same ``PROMPT_TEMPLATE``, same
decoding, same ``clean_answer`` hygiene, same metric code as GRAFT run 5 and the
RAG baseline -- imported from the runner, not copied.

**Why it must exist.**  `CLAUDE.md` §9: Mem0's own table has full-context at
72.90 LLM-judge, above every memory system it tested, and omitting it "will read
as evasion". This is the row that says what the whole conversation buys a 3B
reader, and therefore what the ~15x token saving of GRAFT (1.4k) and RAG (1.25k)
actually costs in accuracy.  `GRAFT_PHASE11_BUILD.md` §7 records it as the
deferred baseline that completes Gate 4 item 4.

**Two honest limits, stated.**  (1) The reader is 3B; Mem0's 72.90 used a much
larger backbone, so this row is comparable to *GRAFT's* runs, not to Mem0's
table.  (2) Lost in the Middle (TACL 2024) predicts a long context hurts a small
reader on evidence in the middle; this run measures that rather than assuming it.

Resumable per question; chunk with ``--conversations`` and cool-down between.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from graft.config import load_config  # noqa: E402
from graft.diagnostics.report import build_report, report_metrics  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import locomo  # noqa: E402
from graft.ledger import Ledger  # noqa: E402
from graft.reader import pins as rpins  # noqa: E402
from graft.reader.orchestrator import ReadPathStamp, cost_report  # noqa: E402
from graft.reader.parse import parse_answer  # noqa: E402
from graft.reader.serialize import format_date  # noqa: E402


def _runner():
    spec = importlib.util.spec_from_file_location("locomo_eval", REPO / "scripts" / "locomo_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["locomo_eval"] = module
    spec.loader.exec_module(module)
    return module


def full_context(turns, count_tokens=None, max_tokens: int | None = None) -> tuple[str, dict[str, str], int]:
    """Every turn, chronological, numbered.  Returns ``(text, claim_map, turns_dropped)``.

    **Window limit, when one is set.**  A 22k-token prompt does not fit an 8 GB
    card with this reader's attention path (measured 12 Sep 2026: SDPA fell back
    to the dense kernel and asked for 58 GiB).  With ``max_tokens`` the
    conversation is truncated **from the beginning** -- the most recent turns are
    kept -- which is the standard no-retrieval deployment when a conversation
    exceeds the window.  It is stamped ``full_context_truncated`` and reported
    with ``turns_dropped``, because a truncated context is a different baseline
    from a full one and the artefact must say which it measured.
    """
    ordered = sorted(turns, key=lambda t: (t.ts, t.turn_id))
    dropped = 0
    while True:
        lines = [f"[c{i + 1}] ({format_date(str(t.ts))}) {t.speaker}: {t.text}".strip() for i, t in enumerate(ordered)]
        text = "\n".join(lines)
        if max_tokens is None or count_tokens is None or count_tokens(text) <= max_tokens or len(ordered) <= 1:
            return text, {f"c{i + 1}": t.turn_id for i, t in enumerate(ordered)}, dropped
        # drop the oldest ~5% per step: a per-turn loop re-tokenises 20k tokens hundreds of times
        step = max(1, len(ordered) // 20)
        ordered = ordered[step:]; dropped += step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="data/locomo/locomo10.json")
    ap.add_argument("--run-dir", default="artefacts/locomo")
    ap.add_argument("--out", default="results/locomo_fullcontext_baseline.json")
    ap.add_argument("--rows", default="results/locomo_fullcontext_baseline_rows.jsonl")
    ap.add_argument("--conversations", type=int, default=None)
    ap.add_argument("--conv-offset", type=int, default=0, help="skip the first N conversations (chunking)")
    ap.add_argument("--questions", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--shard", default=None, help="i/N: only questions with index %% N == i (multi-GPU sharding)")
    ap.add_argument("--max-context-tokens", type=int, default=None,
                    help="window limit in reader tokens; None = the whole conversation. Truncates from the "
                         "beginning (most recent turns kept) and stamps the run as truncated.")
    args = ap.parse_args()

    runner = _runner()
    config = load_config()
    samples = list(locomo.load_corpus(args.corpus))
    snapshot = ReplayGraphStore(EventLog.open(REPO / args.run_dir / "events.jsonl", fsync=False)).at()
    ingested = {t.conv_id for t in snapshot._turns.values()}
    samples = [s for s in samples if str(s["sample_id"]) in ingested]
    samples = samples[args.conv_offset:]
    if args.conversations:
        samples = samples[: args.conversations]
    questions = [q for s in samples for q in locomo.questions_of(s)]
    if args.questions:
        questions = questions[: args.questions]
    if args.shard:
        _i, _n = (int(x) for x in args.shard.split("/"))
        questions = [q for k, q in enumerate(questions) if k % _n == _i]
        print(f"shard {_i}/{_n}: {len(questions)} questions", flush=True)
    print(f"full-context baseline: {len(questions)} questions over {len(samples)} conversations", flush=True)

    rows_path = REPO / args.rows; rows_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if rows_path.exists() and not args.fresh:
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); done[r["question_id"]] = r
    elif args.fresh and rows_path.exists():
        rows_path.unlink()

    reader = None
    if args.smoke:
        read_fn = lambda evidence, question: "London [c1]"  # noqa: E731
        count_tokens, backbone = None, "stub"
    else:
        from graft.reader.read import Reader
        reader = Reader(device=args.device); reader.__enter__()
        read_fn, count_tokens, backbone = reader.generate, reader.count_tokens, rpins.READER["model_id"]
    counter = count_tokens or (lambda t: len(t.split()))

    stamp = ReadPathStamp(
        policy_trained=False, head_trained=False, gate_source="none", scorer_source="none",
        token_counter="approx_tokens" if count_tokens is None else "reader_tokenizer",
        ordering="chronological_full_conversation",
        selection="full_context_baseline" if args.max_context_tokens is None else f"full_context_truncated_recent_{args.max_context_tokens}",
        notes=("BASELINE, not GRAFT: the entire conversation in the prompt; no retrieval, no graph, no budget",
               "turns numbered [c#] so prompt rule 5 is satisfiable; F1/BLEU strip them",
               "3B reader: comparable to GRAFT's runs, not to Mem0's 72.90 (larger backbone)"),
    )

    # One evidence block per conversation, built once.
    blocks: dict[str, tuple[str, dict[str, str], int]] = {}
    def block_for(conv_id: str):
        if conv_id not in blocks:
            turns = [t for t in snapshot._turns.values() if t.conv_id == conv_id]
            text, cmap, dropped = full_context(turns, counter, args.max_context_tokens)
            blocks[conv_id] = (text, cmap, counter(text), dropped)
        return blocks[conv_id]

    started = time.perf_counter(); results: list[dict] = []
    fh = rows_path.open("a", encoding="utf-8")
    try:
        for index, q in enumerate(questions):
            qid = q["question_id"]
            if qid in done:
                results.append(done[qid]); continue
            conv_id = q["conv_id"]
            evidence, claim_map, ev_tokens, dropped = block_for(conv_id)
            ledger = Ledger.from_config(config)
            with ledger.query_scope(qid):
                if reader is not None:
                    reader.ledger = ledger
                with ledger.stage("stage_e"):
                    parsed = parse_answer(read_fn(evidence, q["question"]))
                    text = parsed.answer_text
                    if parsed.abstained:
                        outcome, cause, cites = "abstain", "fallback", ()
                    else:
                        clean, abstained, junk = runner.clean_answer(text or "")
                        if abstained:
                            outcome, cause, cites = "abstain", "fallback", ()
                        else:
                            outcome, cause, text = "answer", None, clean
                            cites = tuple(c for c in parsed.citations if c in claim_map)
                snap = ledger.snapshot()
            row = {"question_id": qid, "conv_id": conv_id, "category": q["category"], "adversarial": q["adversarial"],
                   "gold": q["gold"], "outcome": outcome, "abstain_cause": cause, "answer_text": text,
                   "citations": len(cites), "citation_ids": list(cites), "evidence_tokens": ev_tokens,
                   "turns_in_context": len(claim_map), "turns_dropped": dropped,
                   "ledger_snapshot": snap, "system": "full_context" if args.max_context_tokens is None else "full_context_truncated"}
            results.append(row); fh.write(json.dumps(row) + "\n"); fh.flush()
            if index % 25 == 0:
                rate = (index + 1) / max(1e-9, time.perf_counter() - started) * 3600
                print(f"  [{index + 1}/{len(questions)}] {qid} {outcome} ({rate:.0f} q/h, ctx {ev_tokens} tok)", flush=True)
    finally:
        fh.close()
        if reader is not None:
            reader.__exit__(None, None, None)
    elapsed = time.perf_counter() - started

    class _R:
        def __init__(self, r):
            self.record = type("rec", (), {"ledger_snapshot": r["ledger_snapshot"], "outcome": r["outcome"],
                                           "abstain_cause": r["abstain_cause"]})()
    cost = cost_report([_R(r) for r in results])
    body = build_report(results, cost=cost, ceilings=None, backbone=backbone, embedder="n/a (no retrieval)",
                        budget_tokens=max((r["evidence_tokens"] for r in results), default=0),
                        ladder=rpins.BUDGET_LADDER, honesty_stamp=stamp.to_dict(), ingestion_cost=None)
    body["report_metrics"] = report_metrics(
        results, cost=cost, wall_clock_s=elapsed, corpus_sha=locomo.corpus_sha256(args.corpus),
        prompt_sha=rpins.PROMPT_SHA, stage_e=rpins.stage_e_fingerprint(), config_hash=runner.config_hash(config),
        decoding=dict(rpins.DECODING),
        determinism={"per_machine": "greedy decoding, one question per call", "cross_machine": "not promised"})
    body["what_this_is"] = ("FULL-CONTEXT BASELINE on LoCoMo (Gate-4 item 4). Entire conversation in the prompt; "
                            "same reader, prompt, decoding, hygiene and metric code as GRAFT run 5; no retrieval, no graph.")
    body["run"] = {"conversations": len(samples), "questions": len(questions), "wall_clock_s": round(elapsed, 1),
                   "questions_per_hour": round(len(results) / max(elapsed, 1e-9) * 3600, 1), "smoke": bool(args.smoke),
                   "corpus_sha256": locomo.corpus_sha256(args.corpus),
                   "context_tokens": {c: v[2] for c, v in blocks.items()},
                   "turns_dropped": {c: v[3] for c, v in blocks.items()},
                   "max_context_tokens": args.max_context_tokens}
    out = REPO / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    g = body["comparison"]["graft"]
    tk = cost["llm_tokens_total_per_query"]["mean"] if cost["llm_tokens_total_per_query"] else 0
    print(f"\nfull-context  F1={g['overall_f1']*100:.2f}  B1={g['overall_bleu1']*100:.2f}  tokens/query={tk:.0f}")
    print(f"written: {out} ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
