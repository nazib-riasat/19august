#!/usr/bin/env python
"""GRAFT end to end on 2WikiMultiHopQA dev -- the second dataset row (item F).

    python scripts/wiki2_eval.py --questions 500        # stratified over question type
    python scripts/wiki2_eval.py --smoke --questions 8  # stub reader, no GPU

**What runs.**  The same read path as the LoCoMo runner, per question: 2Wiki row
-> ``wiki2.build_one`` (paragraph claims, ``about_entity`` edges from titles, the
five-channel fused scores) -> ``RealEnvironment`` -> ``answer()`` with the frozen
reader, the trained utility head as scorer, and Stage D by
``training_free_relevance`` (the stamp says so).  Scoring is SQuAD token-F1 and
exact match through ``normalise_answer`` -- the metric 2Wiki itself inherits.
No raw-turn tier: 2Wiki has no dialogue, and its claims *are* the paragraphs.

**What this is not.**  Not a controlled comparison with any published 2Wiki row:
those systems are trained on 2Wiki train; GRAFT's head was distilled on 200
2Wiki + 200 MuSiQue train examples and everything else is training-free.  It is
a **within-system** row, and the five-ceiling protocol is the instrument for
reading it.  ``is_wiring_test`` stays True for the same reasons as on LoCoMo.

**Dev, stratified by type, capped.**  ``--questions`` is drawn by
``stratified_sample`` over ``row["type"]`` (comparison / compositional / bridge
/ inference) with the pinned seed, so the subset is reproducible and not a head
slice.  The cap is recorded.  Resumable per question.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from graft.config import load_config  # noqa: E402
from graft.graphbuild.embed import Embedder, StubEmbedder  # noqa: E402
from graft.ledger import Ledger  # noqa: E402
from graft.reader import pins as rpins  # noqa: E402
from graft.reader.orchestrator import ReadPathStamp, answer, cost_report  # noqa: E402
from graft.reader.parse import normalise_answer, token_f1  # noqa: E402
from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer  # noqa: E402
from graft.setgen.corpora import stratified_sample, wiki2  # noqa: E402
from graft.setgen.distill import HeadScorer, build_head  # noqa: E402
from graft.setgen.pins import SUBSET  # noqa: E402
from graft.setgen.policy import Policy  # noqa: E402
from graft.setgen.realenv import RealEnvironment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--questions", type=int, default=500)
    ap.add_argument("--head", default="artefacts/utility_head.pt")
    ap.add_argument("--out", default="results/wiki2_eval.json")
    ap.add_argument("--rows", default="results/wiki2_eval_rows.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    config = load_config()
    from graft.graphbuild.loaders import load_split
    # Phase 9 keeps the Stage-D corpora under data/phase9/raw (its own SHA-pinned
    # download), not the Phase-6 decoder-corpus root the loader defaults to.
    rows_all = load_split("2wiki", args.split, root=REPO / "data" / "phase9" / "raw")
    selected = stratified_sample(list(rows_all), lambda r: str(r.get("type", "")), args.questions, int(SUBSET["seed"]))
    print(f"2Wiki {args.split}: {len(selected)} of {len(rows_all)} rows, stratified by type", flush=True)

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
        count_tokens = None; backbone = "stub"; embedder = StubEmbedder(384)
    else:
        from graft.reader.read import Reader
        reader = Reader(device=args.device); reader.__enter__()
        read_fn, count_tokens, backbone = reader.generate, reader.count_tokens, rpins.READER["model_id"]
        embedder = Embedder()

    head_state, head_rho = None, None
    if args.head and not args.smoke:
        blob = torch.load(REPO / args.head, map_location="cpu", weights_only=False)
        if int(blob.get("atom_width", ATOM_WIDTH)) != int(ATOM_WIDTH):
            raise SystemExit(f"head atom_width {blob.get('atom_width')} != build {ATOM_WIDTH}")
        head_state, head_rho = blob["state_dict"], blob.get("dev_spearman")

    stamp = ReadPathStamp(
        policy_trained=False, head_trained=bool(head_state),
        gate_source="none", scorer_source="distilled_head",
        token_counter="approx_tokens" if count_tokens is None else "reader_tokenizer",
        ordering="u_shaped_inference_computable", selection="training_free_relevance",
        notes=("2Wiki dev, stratified by type; claims tier only (paragraph claims), no raw tier",
               f"utility head dev rho vs exact U = {head_rho}",
               "within-system row; not a controlled comparison with published 2Wiki systems"),
    )

    started = time.perf_counter(); results: list[dict] = []
    fh = rows_path.open("a", encoding="utf-8")
    try:
        for index, row in enumerate(selected):
            qid = str(row["_id"])
            if qid in done:
                results.append(done[qid]); continue
            example = wiki2.build_one(row, embedder, config=config)
            ledger = Ledger.from_config(config)
            with ledger.query_scope(qid):
                env = RealEnvironment(example, config, range_samples=0)
                featurizer = RealFeaturizer(example, Policy(*RealFeaturizer.dims(), hidden=16), config, delta_d=False)
                head = build_head(ATOM_WIDTH, seed=config.seeds[0])
                if head_state is not None:
                    head.load_state_dict(head_state); head.eval()
                scorer = HeadScorer(head, featurizer)
                if reader is not None:
                    reader.ledger = ledger
                result = answer(
                    str(row["question"]), env=env, featurizer=featurizer, scorer=scorer,
                    read_fn=read_fn, gate_decision=None, obligations=example.obligations,
                    atom_scores=example.atom_scores, rng=np.random.default_rng(config.seeds[0] + index),
                    ledger=ledger, config=config, stamp=stamp, count_tokens=count_tokens,
                    query_id=qid, contested_check=False,
                )
            rec = result.record
            gold = str(row.get("answer", ""))
            text = rec.answer_text or ""
            f1 = token_f1(text, gold) if rec.outcome == "answer" else 0.0
            em = float(normalise_answer(text) == normalise_answer(gold)) if rec.outcome == "answer" else 0.0
            r = {"question_id": qid, "type": row.get("type", ""), "question": row["question"], "gold": gold,
                 "outcome": rec.outcome, "abstain_cause": rec.abstain_cause, "answer_text": rec.answer_text,
                 "f1": f1, "em": em, "pool_size": len(example.pool.ids()), "gold_atoms": len(example.gold_atom_ids),
                 "gold_complete": example.gold_complete, "ledger_snapshot": dict(rec.ledger_snapshot)}
            results.append(r); fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
            if index % 25 == 0:
                rate = (index + 1) / max(1e-9, time.perf_counter() - started) * 3600
                print(f"  [{index + 1}/{len(selected)}] {qid[:12]} {rec.outcome} ({rate:.0f} q/h)", flush=True)
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
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    def block(rs):
        n = len(rs); ans = [r for r in rs if r["outcome"] == "answer"]
        return {"n": n, "f1_over_all": float(np.mean([r["f1"] for r in rs])) if rs else None,
                "em_over_all": float(np.mean([r["em"] for r in rs])) if rs else None,
                "f1_answered": float(np.mean([r["f1"] for r in ans])) if ans else None,
                "coverage": len(ans) / n if n else None,
                "gold_complete_rate": float(np.mean([r["gold_complete"] for r in rs])) if rs else None}
    out = {
        "what_this_is": "GRAFT end to end on 2WikiMultiHopQA dev (stratified subset). Within-system row; "
                        "NOT a controlled comparison with published 2Wiki systems (they train on 2Wiki train).",
        "dataset": {"name": "2WikiMultiHopQA", "split": args.split, "rows_total": len(rows_all),
                    "rows_evaluated": len(results), "stratified_by": "type", "seed": int(SUBSET["seed"])},
        "metric": "SQuAD token-F1 and exact match via normalise_answer; abstentions score 0 in *_over_all",
        "overall": block(results), "by_type": {k: block(v) for k, v in sorted(by_type.items())},
        "abstentions": {"n": sum(1 for r in results if r["outcome"] != "answer"),
                        "by_cause": dict(__import__("collections").Counter(r["abstain_cause"] for r in results if r["outcome"] != "answer"))},
        "cost": cost, "honesty_stamp": stamp.to_dict(),
        "run": {"backbone": backbone, "wall_clock_s": round(elapsed, 1), "smoke": bool(args.smoke),
                "questions_per_hour": round(len(results) / max(elapsed, 1e-9) * 3600, 1),
                "prompt_sha": rpins.PROMPT_SHA, "stage_e": rpins.stage_e_fingerprint(), "head": args.head},
    }
    (REPO / args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    o = out["overall"]
    print(f"\n2Wiki {args.split}: F1(all)={o['f1_over_all']*100:.2f}  EM={o['em_over_all']*100:.2f}  "
          f"coverage={o['coverage']:.3f}  F1(answered)={ (o['f1_answered'] or 0)*100:.2f}")
    print(f"written: {args.out} ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
