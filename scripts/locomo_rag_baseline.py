#!/usr/bin/env python
"""Matched-budget RAG baseline on LoCoMo -- the Gate-4 item-4 adapter.

    python scripts/locomo_rag_baseline.py --conversations 3            # chunk 1 of ~3
    python scripts/locomo_rag_baseline.py                               # resumes per question

**What "matched" means here, stated so nobody has to infer it.**  Same frozen
reader, same ``PROMPT_TEMPLATE`` (the SHA is stamped), same decoding, same
**total evidence budget** as GRAFT run 5 (1,280 reader tokens), same retriever
over raw turns (``top_raw_turns`` + ``expand_windows`` with run 5's k / radius /
cap), same chronological ordering, same answer hygiene, same metric code
(``build_report`` / ``report_metrics``), same reference table.  The one thing
that differs is the *system*: no graph, no claims tier, no `H`, no Stage D.  The
whole 1,280-token budget goes to raw dialogue, where GRAFT spent 256 of it on
checker-validated claims and the rest on the same raw turns.

**One design decision, recorded.**  RAG passages carry ``[c1]``..``[ck]`` ids.
GRAFT's raw tier is deliberately *uncitable* -- its ids belong to checker-
validated claims.  A RAG baseline has no claims, so an uncitable block would
leave the prompt's rule 5 ("cite the evidence you used") with nothing to cite
and would score the baseline against a prompt it cannot follow.  Numbering the
passages is what every RAG system does; it changes nothing about F1/BLEU, which
strip citations before scoring.

**Why this exists.**  `GRAFT_PHASE11_BUILD.md` §7: Gate 4 item 4 requires
re-running system baselines rather than quoting them, and matched-budget RAG is
"the single addition that would most repair G1" -- ~2 h, because Stage C already
computes the retrieval and only the reader pass is new.  Mem-T's own RAG row
(41.59 F1) shows RAG is not a strawman on this benchmark.

**Resumable per question** (rows file), so it can run in chunks with
``--conversations N`` and cool-down between -- `CLAUDE.md` §7's hardware row is
an 8 GB laptop card.  Every GPU run needs explicit permission; this one had it.
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
from graft.graphbuild.embed import Embedder  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import locomo  # noqa: E402
from graft.ledger import Ledger  # noqa: E402
from graft.reader import pins as rpins  # noqa: E402
from graft.reader.orchestrator import ReadPathStamp, cost_report  # noqa: E402
from graft.reader.parse import parse_answer  # noqa: E402
from graft.reader.serialize import format_date  # noqa: E402


def _runner():
    """The GRAFT runner as a module -- imported, not duplicated.

    ``ChannelCache``, ``top_raw_turns``, ``expand_windows``, ``clean_answer`` and
    ``config_hash`` are reused verbatim so the baseline cannot drift from the
    system it is compared with.  A baseline with its own retrieval code is a
    different experiment wearing the same name.
    """
    spec = importlib.util.spec_from_file_location("locomo_eval", REPO / "scripts" / "locomo_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["locomo_eval"] = module
    spec.loader.exec_module(module)
    return module


def render_passages(turns, count_tokens, budget: int) -> tuple[str, list, dict[str, str]]:
    """Numbered raw passages in chronological order, inside ``budget`` tokens.

    Returns ``(text, kept_turns, claim_map)``.  Dropping is from the *least
    relevant* end (the input arrives ranked; we keep a prefix of that ranking and
    then re-sort chronologically), which is run 5's rule for the raw tier.
    """
    if not turns:
        return "", [], {}
    for keep in range(len(turns), 0, -1):
        kept = sorted(turns[:keep], key=lambda t: (t.ts, t.turn_id))
        lines = [
            f"[c{i + 1}] ({format_date(str(t.ts))}) {t.speaker}: {t.text}".strip()
            for i, t in enumerate(kept)
        ]
        text = "\n".join(lines)
        if count_tokens(text) <= budget:
            return text, kept, {f"c{i + 1}": t.turn_id for i, t in enumerate(kept)}
    return "", [], {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="data/locomo/locomo10.json")
    ap.add_argument("--run-dir", default="artefacts/locomo", help="Stage-A log dir (turns only are read)")
    ap.add_argument("--out", default="results/locomo_rag_baseline.json")
    ap.add_argument("--rows", default="results/locomo_rag_baseline_rows.jsonl")
    ap.add_argument("--conversations", type=int, default=None, help="cap conversations (chunking)")
    ap.add_argument("--questions", type=int, default=None)
    ap.add_argument("--evidence-budget", type=int, default=1280, help="run 5's total")
    ap.add_argument("--seeds-k", type=int, default=5, help="run 5's raw_tier.seeds_k")
    ap.add_argument("--window", type=int, default=2, help="run 5's raw_tier.window_radius")
    ap.add_argument("--cap", type=int, default=25, help="run 5's raw_tier.cap_texts")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="stub reader, no GPU")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--shard", default=None, help="i/N: only questions with index %% N == i (multi-GPU sharding)")
    args = ap.parse_args()

    runner = _runner()
    config = load_config()
    corpus = locomo.load_corpus(args.corpus)
    samples = list(corpus)
    if args.conversations:
        samples = samples[: args.conversations]

    snapshot = ReplayGraphStore(
        EventLog.open(REPO / args.run_dir / "events.jsonl", fsync=False)
    ).at()
    ingested = {t.conv_id for t in snapshot._turns.values()}
    samples = [s for s in samples if str(s["sample_id"]) in ingested]

    questions: list[dict[str, Any]] = []
    for sample in samples:
        for q in locomo.questions_of(sample):
            questions.append(q)
    if args.questions:
        questions = questions[: args.questions]
    if args.shard:
        _i, _n = (int(x) for x in args.shard.split("/"))
        questions = [q for k, q in enumerate(questions) if k % _n == _i]
        print(f"shard {_i}/{_n}: {len(questions)} questions", flush=True)
    print(f"RAG baseline: {len(questions)} questions over {len(samples)} conversations, "
          f"evidence budget {args.evidence_budget}")

    rows_path = REPO / args.rows
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if rows_path.exists() and not args.fresh:
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["question_id"]] = r
    elif args.fresh and rows_path.exists():
        rows_path.unlink()

    reader = None
    if args.smoke:
        def read_fn(evidence: str, question: str) -> str:
            return "London [c1]"
        count_tokens = None
        backbone = "stub"
    else:
        from graft.reader.read import Reader

        reader = Reader(device=args.device)
        reader.__enter__()
        read_fn = reader.generate
        count_tokens = reader.count_tokens
        backbone = rpins.READER["model_id"]

    embedder = Embedder()
    cache = runner.ChannelCache(snapshot, embedder, config)
    counter = count_tokens or (lambda t: len(t.split()))

    stamp = ReadPathStamp(
        policy_trained=False, head_trained=False,
        gate_source="none", scorer_source="none",
        token_counter="approx_tokens" if count_tokens is None else "reader_tokenizer",
        ordering="chronological_raw_passages",
        selection="matched_budget_rag_baseline",
        notes=(
            "BASELINE, not GRAFT: no graph, no claims tier, no H, no Stage D",
            f"matched to run 5: total evidence budget {args.evidence_budget} reader tokens, "
            f"seeds_k={args.seeds_k}, window_radius={args.window}, cap={args.cap}",
            "passages are numbered [c#] so prompt rule 5 is satisfiable; F1/BLEU strip them",
        ),
    )

    started = time.perf_counter()
    results: list[dict] = []
    fh = rows_path.open("a", encoding="utf-8")
    try:
        for index, q in enumerate(questions):
            qid = q["question_id"]
            if qid in done:
                results.append(done[qid])
                continue
            conv_id = q["conv_id"]
            ledger = Ledger.from_config(config)
            with ledger.query_scope(qid):
                with ledger.stage("stage_c"):
                    seeds = runner.top_raw_turns(cache, conv_id, q["question"], embedder, k=int(args.seeds_k))
                    turns, _rank = runner.expand_windows(cache, conv_id, seeds, radius=int(args.window), cap=int(args.cap))
                    evidence, kept, claim_map = render_passages(turns, counter, int(args.evidence_budget))
                if reader is not None:
                    reader.ledger = ledger
                with ledger.stage("stage_e"):
                    if not evidence:
                        outcome, cause, text, cites = "abstain", "fallback", None, ()
                    else:
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

            row = {
                "question_id": qid, "conv_id": conv_id, "category": q["category"],
                "adversarial": q["adversarial"], "gold": q["gold"],
                "outcome": outcome, "abstain_cause": cause, "answer_text": text,
                "citations": len(cites), "citation_ids": list(cites),
                "passages": len(kept), "evidence_tokens": counter(evidence) if evidence else 0,
                "ledger_snapshot": snap,
                "system": "matched_budget_rag",
            }
            results.append(row)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if index % 25 == 0:
                rate = (index + 1) / max(1e-9, time.perf_counter() - started) * 3600
                print(f"  [{index + 1}/{len(questions)}] {qid} {outcome} ({rate:.0f} q/h)", flush=True)
    finally:
        fh.close()
        if reader is not None:
            reader.__exit__(None, None, None)
    elapsed = time.perf_counter() - started

    class _R:
        def __init__(self, r):
            self.record = type("rec", (), {
                "ledger_snapshot": r["ledger_snapshot"], "outcome": r["outcome"],
                "abstain_cause": r["abstain_cause"],
            })()

    cost = cost_report([_R(r) for r in results])
    body = build_report(
        results, cost=cost, ceilings=None, backbone=backbone,
        embedder=getattr(embedder, "name", "bge-small-en-v1.5"),
        budget_tokens=int(args.evidence_budget), ladder=rpins.BUDGET_LADDER,
        honesty_stamp=stamp.to_dict(), ingestion_cost=None,
    )
    body["report_metrics"] = report_metrics(
        results, cost=cost, wall_clock_s=elapsed,
        corpus_sha=locomo.corpus_sha256(args.corpus), prompt_sha=rpins.PROMPT_SHA,
        stage_e=rpins.stage_e_fingerprint(), config_hash=runner.config_hash(config),
        decoding=dict(rpins.DECODING),
        determinism={"per_machine": "greedy decoding, one question per call", "cross_machine": "not promised"},
    )
    body["what_this_is"] = (
        "MATCHED-BUDGET RAG BASELINE on LoCoMo (Gate-4 item 4 adapter). Same reader, prompt, "
        "decoding, evidence budget, retriever, ordering, hygiene and metric code as GRAFT run 5 "
        "(results/locomo_eval3.json); no graph, no claims tier, no H, no Stage D."
    )
    body["matched_to"] = {"graft_run": "results/locomo_eval3.json", "evidence_budget_tokens": int(args.evidence_budget),
                          "seeds_k": int(args.seeds_k), "window_radius": int(args.window), "cap_texts": int(args.cap)}
    body["run"] = {"conversations": len(samples), "questions": len(questions), "wall_clock_s": round(elapsed, 1),
                   "questions_per_hour": round(len(questions) / max(elapsed, 1e-9) * 3600, 1), "smoke": bool(args.smoke),
                   "corpus_sha256": locomo.corpus_sha256(args.corpus)}
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    g = body["comparison"]["graft"]
    print(f"\nRAG baseline  F1={g['overall_f1']*100:.2f}  B1={g['overall_bleu1']*100:.2f}  "
          f"tokens/query={cost['llm_tokens_total_per_query']['mean'] if cost['llm_tokens_total_per_query'] else 0:.0f}")
    print(f"written: {out} ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
