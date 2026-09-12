#!/usr/bin/env python
"""Offline retrieval coverage for the raw-dialogue tier -- no reader, no GPU spend.

**Why this exists.**  Run 4 changes what the reader is shown (`--raw-turns` 3->6,
a +-1 same-session window, a rebalanced budget).  Choosing that configuration by
running the reader and keeping whichever variant scored best would be tuning on
the test set: LoCoMo has no dev split, and every run costs ~50 GPU-minutes.

So the variant is selected here instead, on a quantity the reader never touches:
**does the raw tier put at least one of the question's own gold evidence turns in
front of the reader?**  Gold turn ids come from `locomo.evidence_turn_ids`, the
same markers ceiling 1 is defined over.  A variant that cannot retrieve the
evidence cannot be rescued by any prompt, and one that can may still fail for
reader reasons -- which is exactly the separation the five-ceiling protocol
exists to keep.

The grid is measured **before** run 4 and recorded in `PHASE11_DECISIONS.md`
§1.10 with the chosen row named, so the choice is on the record ahead of the
number it produces.  That is the whole difference between a prediction and a
post-hoc story.

Adversarial questions are excluded: they have no evidence by construction, so a
coverage denominator including them would measure the corpus, not the tier.

    python scripts/raw_tier_grid.py --out artefacts/raw_tier_grid.json

Cost: one embed pass over each conversation's turns plus one per question, on
CPU.  Every variant is derived from the *same* ranking, so the grid is one pass
regardless of how many rows it has.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from graft.config import load_config  # noqa: E402
from graft.graphbuild.embed import Embedder  # noqa: E402
from graft.ingest import locomo  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402

#: ``(seeds_k, window_radius, cap)``.  Run 3 is the first row and is in the grid
#: as the incumbent, not as a formality -- a change that does not beat what it
#: replaces on the offline measure has no business costing GPU time.
GRID: tuple[tuple[int, int, int], ...] = (
    (3, 0, 15),   # run 3, as shipped
    (6, 0, 15),   # more seeds, no window: isolates "is k the binding constraint?"
    (3, 1, 15),   # window only: isolates "is the missing half a neighbour?"
    (6, 1, 15),   # run 4's proposal
    (6, 1, 12),   # the same, with the cap actually binding
    (8, 1, 15),   # is k=6 already saturating?
    (6, 2, 15),   # a wider window, which the cap starts to fight
    # -- run-5 candidates, added 21 Aug 2026 -------------------------------
    # Run 4's grid found radius 2 WORSE than radius 1 (0.7100 vs 0.7276) -- but
    # at cap 15, which drops a wider window whole to fit, so the row measured
    # the cap and not the radius. These raise the cap so the radius is the only
    # thing varying, and trade seeds for depth at a comparable text count.
    (5, 2, 25),   # run 5's proposal: fewer seeds, deeper windows
    (6, 2, 25),   # the same depth at run-4's k, to isolate the k reduction
    (6, 1, 25),   # run-4's shape with the cap unbound, as the control
    (4, 2, 25),   # is k=5 already saturating on the deeper window?
)


def _runner():
    """The eval runner as a module -- the same `top_raw_turns` / `expand_windows`
    run 4 will call, imported rather than reimplemented so the grid cannot
    measure a different selector than the one that ships."""
    spec = importlib.util.spec_from_file_location(
        "locomo_eval", REPO / "scripts" / "locomo_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["locomo_eval"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/locomo/locomo10.json")
    ap.add_argument("--run-dir", default="artefacts/locomo_stageb")
    ap.add_argument("--out", default="artefacts/raw_tier_grid.json")
    ap.add_argument("--questions", type=int, default=None, help="head slice, for a smoke")
    args = ap.parse_args()

    runner = _runner()
    config = load_config()

    log_path = REPO / args.run_dir / "events.jsonl"
    if not log_path.is_file():
        raise SystemExit(f"no ingest log at {log_path}")
    snapshot = ReplayGraphStore(EventLog.open(log_path, fsync=False)).at()
    embedder = Embedder()
    cache = runner.ChannelCache(snapshot, embedder, config)

    corpus = locomo.load_corpus(args.corpus)
    max_k = max(k for k, _, _ in GRID)

    hits = {v: 0 for v in GRID}
    texts = {v: 0 for v in GRID}
    scored = 0
    no_gold = 0
    started = time.perf_counter()

    for sample in corpus:
        conv_id = str(sample["sample_id"])
        questions = locomo.questions_of(sample)
        if args.questions is not None:
            questions = questions[: args.questions]
        for q in questions:
            if q["adversarial"]:
                continue
            gold = set(locomo.evidence_turn_ids(sample, q["evidence"]))
            if not gold:
                # Evidence markers that resolve to nothing -- image-only turns the
                # loader skips. Counted, never imputed.
                no_gold += 1
                continue
            scored += 1
            seeds = runner.top_raw_turns(
                cache, conv_id, q["question"], embedder, k=max_k
            )
            for variant in GRID:
                k, radius, cap = variant
                if radius <= 0:
                    chosen = [t.turn_id for t in seeds[:k]][:cap]
                else:
                    turns, _ = runner.expand_windows(
                        cache, conv_id, seeds[:k], radius=radius, cap=cap
                    )
                    chosen = [t.turn_id for t in turns]
                texts[variant] += len(chosen)
                if gold & set(chosen):
                    hits[variant] += 1
        print(
            f"  {conv_id}: {scored} scored, "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )

    rows = []
    base = None
    for variant in GRID:
        k, radius, cap = variant
        coverage = hits[variant] / scored if scored else 0.0
        if base is None:
            base = coverage
        rows.append({
            "seeds_k": k,
            "window_radius": radius,
            "cap_texts": cap,
            "gold_turn_coverage": coverage,
            "delta_vs_run3": coverage - base,
            "mean_texts_shown": texts[variant] / scored if scored else 0.0,
        })

    report = {
        "what_this_measures": (
            "the fraction of answerable LoCoMo questions whose raw-turn tier "
            "contains at least one of the question's own gold evidence turns. "
            "No reader is run: this is a retrieval property, and a variant that "
            "misses the evidence cannot be rescued by any prompt."
        ),
        "what_this_does_not_measure": (
            "answer quality. Coverage is necessary, not sufficient -- ceiling 5 "
            "(the frozen 3B reader) is the binding constraint measured in run 2, "
            "and nothing here moves it."
        ),
        "selection_rule": (
            "declared BEFORE run 4: take the highest gold_turn_coverage, and "
            "among rows within 0.005 of it take the one showing fewest texts. "
            "Written down first so the row is chosen by the rule, not the rule "
            "by the row."
        ),
        "excludes_adversarial": True,
        "questions_scored": scored,
        "questions_without_resolvable_gold": no_gold,
        "corpus_sha256": locomo.corpus_sha256(args.corpus),
        "embedder": getattr(embedder, "name", "bge-small-en-v1.5"),
        "run_dir": args.run_dir,
        "wall_clock_s": round(time.perf_counter() - started, 1),
        "grid": rows,
    }

    best = max(rows, key=lambda r: r["gold_turn_coverage"])
    near = [
        r for r in rows
        if best["gold_turn_coverage"] - r["gold_turn_coverage"] <= 0.005
    ]
    report["selected"] = min(near, key=lambda r: r["mean_texts_shown"])

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    for r in rows:
        print(
            f"  k={r['seeds_k']} radius={r['window_radius']} cap={r['cap_texts']}: "
            f"coverage {r['gold_turn_coverage']:.4f} "
            f"({r['delta_vs_run3']:+.4f})  texts {r['mean_texts_shown']:.2f}"
        )
    print(f"selected: {report['selected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
