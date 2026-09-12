#!/usr/bin/env python
"""D3/D4 pretrain on their external corpora -- item 6 of the remaining-work table.

    python scripts/phase6_pretrain_d3d4.py                 # all three corpora, seed 13
    python scripts/phase6_pretrain_d3d4.py --only dialogre

**What this trains.**  ``TypedDecoder`` heads (decoders.py) -- D3 on DialogRE
(multi-label, BCE) and Re-DocRED (single-label, CE), D4 on TORQUE framed as its
own task: (question+passage, event span) -> is-an-answer.  Inputs are frozen
pinned-embedder vectors (bge-small-en-v1.5, the Stage-B embedder), so the head is
the only thing learned and it trains in seconds once the vectors exist.  Encoding
the pairs is the GPU cost, and it is the *only* GPU cost.

**Budget discipline.**  ``pins.TRAINING`` is read for epochs / lr / patience /
hidden, never an argument default -- G6's "same frozen decoder interface" and
the reason `train_d2` refuses to take a budget from its caller.  Early stopping
restores the argmin-dev state.  Temperature is fitted on **dev** (`calibrate`,
decision 8) because the D3/D4 commit floor (`pins.COMMIT_FLOOR`) is a confidence
threshold and an uncalibrated threshold is a number with no meaning.

**Volumes are capped and the caps are recorded in the artefact.**  Re-DocRED is
~35k labelled pairs and TORQUE fans out to every event per question; on an 8 GB
laptop card the full sets are hours of embedding for a head that saturates long
before.  ``--max-docs`` / ``--max-passages`` default to a size that keeps the whole
run under ~30 min.  A cap is a declared choice, not a silent truncation, so the
artefact states it beside every number.

**Native label sets, as the loaders keep them.**  DialogRE stays multi-label
(collapsing it changes the task); Re-DocRED keeps its 96 relations; TORQUE keeps
its answer-span framing.  Metrics are the datasets' own kind (micro-F1 / F1),
reported on dev.  Nothing here is comparable to a published row -- these are
pretrained heads for Gate 1's construction pipeline, and the artefact says so.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from graft.graphbuild import loaders  # noqa: E402
from graft.graphbuild.decoders import TypedDecoder, class_weights  # noqa: E402
from graft.graphbuild.embed import Embedder  # noqa: E402
from graft.graphbuild.pins import COMMIT_FLOOR, DATASETS, EMBEDDER, TRAINING  # noqa: E402
from graft.graphbuild.train import calibrate  # noqa: E402

CTX = 480  # chars of context folded into each side; bge-small's window is 512 tokens


def _pairs_relation(items):
    """(a_text, b_text) for a relation item: each side carries its mention, type and the context."""
    a = [f"{it['head']} ({it['head_type']}) in: {it['text'][:CTX]}" for it in items]
    b = [f"{it['tail']} ({it['tail_type']}) in: {it['text'][:CTX]}" for it in items]
    return a, b


def _torque_pairs(items):
    """Flatten TORQUE (passage, question, events, answers) -> per-event binary rows."""
    a, b, y, ids = [], [], [], []
    for it in items:
        if it.get("is_default_question"):
            continue  # the default "what events have already happened" is not a temporal relation query
        answers = set(it.get("answer_spans") or ())
        events = list(dict.fromkeys(it.get("events") or ()))
        if not events:
            continue
        q = f"{it['question']} || {it['text'][:CTX]}"
        for ev in events:
            a.append(q); b.append(f"{ev} in: {it['text'][:CTX]}")
            y.append(1 if ev in answers else 0); ids.append(it["item_id"])
    return a, b, y, ids


def _embed(emb, texts, batch=128):
    out = []
    for i in range(0, len(texts), batch):
        out.append(np.asarray(emb.embed(texts[i:i + batch]), dtype=np.float32))
    return torch.from_numpy(np.concatenate(out)) if out else torch.zeros((0, EMBEDDER["dim"]))


def _fit(head, A, B, Y, multi, dev, seed, budget):
    """The shared loop: Adam, early stop on dev loss, restore best."""
    torch.manual_seed(seed); random.seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=float(budget["lr"]), weight_decay=float(budget["weight_decay"]))
    # Class weights from TRAIN only, as `train_d2` does -- without them TORQUE's
    # 77% NOT_ANSWER majority collapsed the head to a constant (measured, run 1).
    if multi:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        w = class_weights(Y.tolist(), int(Y.max()) + 1).to(Y.device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    n = A.shape[0]; bs = int(budget["batch_size"])
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(int(budget["epochs"])):
        head.train(); perm = torch.randperm(n); total = 0.0
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            loss = loss_fn(head(A[ix], B[ix]), Y[ix])
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach()) * len(ix)
        head.eval()
        with torch.no_grad():
            dl = float(loss_fn(head(dev[0], dev[1]), dev[2]))
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "dev_loss": dl})
        if dl < best - 1e-6:
            best, bad = dl, 0; best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= int(budget["early_stop_patience"]):
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return best, history


def _micro_f1(pred, gold):
    tp = int(((pred == 1) & (gold == 1)).sum()); fp = int(((pred == 1) & (gold == 0)).sum()); fn = int(((pred == 0) & (gold == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def run_relation(name, emb, seed, max_docs, budget, device):
    tr = loaders.load_split(name, "train"); dv = loaders.load_split(name, "dev")
    items_tr = loaders.dialogre_items(tr) if name == "dialogre" else loaders.redocred_items(tr, max_docs=max_docs)
    items_dv = loaders.dialogre_items(dv) if name == "dialogre" else loaders.redocred_items(dv, max_docs=max(1, max_docs // 5) if max_docs else None)
    vocab = sorted({l for it in items_tr for l in it["labels"]} | {l for it in items_dv for l in it["labels"]})
    ix = {l: i for i, l in enumerate(vocab)}
    multi = name == "dialogre"

    def Y(items):
        if multi:
            y = torch.zeros((len(items), len(vocab)))
            for r, it in enumerate(items):
                for l in it["labels"]: y[r, ix[l]] = 1.0
            return y
        return torch.tensor([ix[it["labels"][0]] for it in items], dtype=torch.long)

    t0 = time.perf_counter()
    a, b = _pairs_relation(items_tr); A, B = _embed(emb, a).to(device), _embed(emb, b).to(device)
    a, b = _pairs_relation(items_dv); DA, DB = _embed(emb, a).to(device), _embed(emb, b).to(device)
    embed_s = time.perf_counter() - t0
    Ytr, Ydv = Y(items_tr).to(device), Y(items_dv).to(device)

    head = TypedDecoder(EMBEDDER["dim"], len(vocab), pair=True).to(device)
    best, hist = _fit(head, A, B, Ytr, multi, (DA, DB, Ydv), seed, budget)
    head.eval()
    with torch.no_grad():
        logits = head(DA, DB)
    if multi:
        pred = (torch.sigmoid(logits) >= 0.5).long(); metric = _micro_f1(pred.cpu(), Ydv.long().cpu()); cal = None
    else:
        pred = logits.argmax(-1)
        # Single-label: accuracy is the honest headline. Macro-F1 over the
        # relation vocabulary beside it, since accuracy alone hides rare classes.
        pc, gc = pred.cpu(), Ydv.cpu(); f1s = []
        for c in range(len(vocab)):
            if (gc == c).any() or (pc == c).any():
                f1s.append(_micro_f1((pc == c).long(), (gc == c).long())["f1"])
        metric = {"accuracy": float((pred == Ydv).float().mean()), "macro_f1": float(np.mean(f1s)) if f1s else 0.0}
        cal = calibrate(logits.cpu(), Ydv.cpu())
    return head, {"dataset": name, "decoder": "D3", "native_metric": DATASETS[name]["metric"], "labels": len(vocab),
                  "train_items": len(items_tr), "dev_items": len(items_dv), "max_docs": max_docs,
                  "multi_label": multi, "best_dev_loss": best, "epochs_run": len(hist), "dev": metric,
                  "calibration": cal, "embed_seconds": round(embed_s, 1), "history": hist}, vocab


def run_torque(emb, seed, max_passages, budget, device):
    tr = loaders.torque_items(loaders.load_split("torque", "train"), max_passages=max_passages)
    dv = loaders.torque_items(loaders.load_split("torque", "dev"), max_passages=max(1, max_passages // 5) if max_passages else None)
    a, b, y, _ = _torque_pairs(tr); da, db, dy, _ = _torque_pairs(dv)
    t0 = time.perf_counter()
    A, B = _embed(emb, a).to(device), _embed(emb, b).to(device); DA, DB = _embed(emb, da).to(device), _embed(emb, db).to(device)
    embed_s = time.perf_counter() - t0
    Ytr, Ydv = torch.tensor(y, dtype=torch.long, device=device), torch.tensor(dy, dtype=torch.long, device=device)
    head = TypedDecoder(EMBEDDER["dim"], 2, pair=True).to(device)
    best, hist = _fit(head, A, B, Ytr, False, (DA, DB, Ydv), seed, budget)
    head.eval()
    with torch.no_grad():
        logits = head(DA, DB)
    pred = logits.argmax(-1)
    return head, {"dataset": "torque", "decoder": "D4", "native_metric": DATASETS["torque"]["metric"],
                  "framing": "(question+passage, event) -> is_answer; default questions excluded",
                  "train_pairs": len(y), "dev_pairs": len(dy), "positive_rate_train": float(np.mean(y)) if y else 0.0,
                  "max_passages": max_passages, "best_dev_loss": best, "epochs_run": len(hist),
                  "dev": _micro_f1(pred.cpu(), Ydv.cpu()), "calibration": calibrate(logits.cpu(), Ydv.cpu()),
                  "embed_seconds": round(embed_s, 1), "history": hist}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=("dialogre", "redocred", "torque"), default=None)
    ap.add_argument("--seed", type=int, default=int(TRAINING["seeds"][0]))
    ap.add_argument("--max-docs", type=int, default=1500, help="Re-DocRED train docs (dev = /5)")
    ap.add_argument("--max-passages", type=int, default=800, help="TORQUE train passages (dev = /5)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="artefacts/phase6/d3d4_pretrain.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    emb = Embedder(device=args.device, cache_dir=REPO / "artefacts" / "phase6" / "embed_cache")
    budget = dict(TRAINING)
    out_dir = REPO / "artefacts" / "phase6"; out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "what_this_is": ("D3/D4 heads pretrained on their external corpora over frozen bge-small vectors. "
                         "Volumes are CAPPED (see each block); metrics are the datasets' own, on dev; "
                         "not comparable to any published row. Consumed by Gate 1's construction pipeline."),
        "budget": {k: v for k, v in budget.items() if k != "seeds"}, "seed": args.seed,
        "embedder": f"{EMBEDDER.get('model_id', 'bge-small-en-v1.5')}", "device": str(device),
        "commit_floor": dict(COMMIT_FLOOR), "runs": {},
    }
    started = time.perf_counter()
    todo = [args.only] if args.only else ["dialogre", "redocred", "torque"]
    for name in todo:
        t0 = time.perf_counter()
        print(f"=== {name} ===", flush=True)
        if name == "torque":
            head, rep = run_torque(emb, args.seed, args.max_passages, budget, device); vocab = ["NOT_ANSWER", "ANSWER"]
        else:
            head, rep, vocab = run_relation(name, emb, args.seed, args.max_docs, budget, device)
        rep["seconds"] = round(time.perf_counter() - t0, 1)
        ckpt = out_dir / f"{rep['decoder'].lower()}_{name}.pt"
        torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "labels": vocab,
                    "dim": EMBEDDER["dim"], "pair": True, "seed": args.seed, "dataset": name,
                    "decoder": rep["decoder"], "dev": rep["dev"], "calibration": rep["calibration"],
                    "caps": {"max_docs": args.max_docs, "max_passages": args.max_passages}}, ckpt)
        rep["checkpoint"] = str(ckpt.relative_to(REPO))
        report["runs"][name] = rep
        d = rep["dev"]; print(f"  {name}: dev f1={d.get('f1', 0):.3f}  epochs={rep['epochs_run']}  "
                              f"{rep['seconds']}s  -> {ckpt.name}", flush=True)
    report["total_seconds"] = round(time.perf_counter() - started, 1)
    (REPO / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written: {args.out} ({report['total_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
