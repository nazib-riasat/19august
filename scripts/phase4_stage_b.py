#!/usr/bin/env python
"""Phase 4 Stage B -- S5 (the trained Phase-3 samplers) on the lattice main suite.

    python scripts/phase4_stage_b.py              # all 9 arms x 3 seeds, 20 instances, CPU

**What this fills.**  `GRAFT_PHASE4_BUILD.md` G7 / exit criterion 15: Stage A's
table (S1-S4 + ceiling, `artefacts/phase4_stage_a.json`) is labelled Stage A and
"is not called Gate 3"; Stage B fills S5's row from Phase-3 checkpoints. The 27
checkpoints have sat in `artefacts/checkpoints/` since 15 Aug 2026 with no
driver, and S5's arm-selection rule was never ruled.

**The unruled selection is dissolved rather than ruled: every arm runs.**  Nine
arms x three seeds x twenty instances is minutes of CPU, so choosing one arm to
represent "the learned sampler" would trade completeness for a decision nobody
has to make. Each arm gets its own row, per seed and pooled; L7
(checker-conditioned LED, the C3 arm) and GAFlowNet (the best flow arm on TV at
Gate 2) are the two a reader will look at first, and both are there.

**What it is not** (decision 5, ruled 12 Aug 2026): **a diagnostic, never a
Gate-3 verdict.** G9 measured, before any method was built, that greedy on exact
`U` is globally optimal on 30/30 instances and a *flawless* sampler reaches only
1.8865 of greedy's 1.9245 at K = 8 -- so on this lattice the comparison answers
by arithmetic. `GRAFT_PHASE4_BUILD.md` §783 pre-declares "S5 loses best-of-K to
greedy" as *expected, and not a finding*. Gate 3's decision lives in Phase 9,
under a noisy scorer. This artefact stamps all of that on itself.

Same scorer (`exact_scorer`, fix F13), same ledger discipline (`terminal_checks`
metered per query and cross-checked against the method's own report), same
diversity statistics as Stage A, so the S5 rows sit beside S1-S4 in one table.
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

from graft.ledger import Ledger  # noqa: E402
from graft.setgen.search.gate3 import (  # noqa: E402
    best_of_k_ceiling, exact_scorer, excess_diversity, jaccard_diversity, random_portfolio_diversity,
)
from graft.setgen.search.s5_portfolio import PortfolioSearch  # noqa: E402
from graft.setgen.trainer import Environment  # noqa: E402
from graft.synth.lattice import benchmark_suite  # noqa: E402

ARMS = ("l1_supervised", "l2_imitation", "l3_grpo", "l4_tb", "l5_subtb", "l6_led",
        "l7_checker_led", "l7b_aux", "gaflownet")
SEEDS = (13, 42, 7)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints", default="artefacts/checkpoints")
    ap.add_argument("--stage-a", default="artefacts/phase4_stage_a.json")
    ap.add_argument("--out", default="artefacts/phase4_stage_b.json")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    args = ap.parse_args()

    envs = [Environment(i) for i in benchmark_suite()]
    stage_a = json.loads((REPO / args.stage_a).read_text(encoding="utf-8"))
    k = int(envs[0].instance.cfg.K)
    ceiling = [best_of_k_ceiling(e.target, k) for e in envs]
    global_max = [float(np.max(e.target.u)) for e in envs]
    print(f"Stage B: {len(envs)} instances, K={k}, arms={len(args.arms)} x seeds={len(SEEDS)}", flush=True)

    rows: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    started = time.perf_counter()
    for arm in args.arms:
        for seed in SEEDS:
            ckpt = REPO / args.checkpoints / f"{arm}.seed{seed}.pt"
            if not ckpt.exists():
                print(f"  missing {ckpt.name}", flush=True); continue
            method = PortfolioSearch(ckpt)
            for ix, env in enumerate(envs):
                inst = env.instance; cfg = inst.cfg
                ledger = Ledger.from_config(cfg)
                with ledger.query_scope(f"s5:{arm}:{seed}:{ix}"):
                    result = method.run(env, inst.obligations, exact_scorer(env), ledger)
                    spent = ledger.snapshot()["query"]["terminal_checks"]
                if spent != result.terminal_checks:
                    raise RuntimeError(f"{arm} seed {seed} inst {ix}: reported {result.terminal_checks} "
                                       f"terminal checks, ledger metered {spent}")
                row = result.to_dict()
                pool_n = len(inst.pool.ids())
                row.update({
                    "instance": ix, "arm": arm, "seed": seed,
                    "diversity": jaccard_diversity(result.portfolio),
                    "diversity_random_baseline": random_portfolio_diversity(result.portfolio, pool_n),
                    "excess_diversity": excess_diversity(result.portfolio, pool_n),
                    "mean_size": float(np.mean([len(s.atoms) for s in result.sets])) if result.sets else float("nan"),
                    "best_utility": float(max(result.scores)) if result.scores else float("nan"),
                    "distinct_valid": len({frozenset(s.atoms) for s in result.sets}),
                    "ceiling": ceiling[ix], "global_max": global_max[ix],
                })
                rows[arm][seed].append(row)
            print(f"  {arm} seed {seed}: mean best-U {np.nanmean([r['best_utility'] for r in rows[arm][seed]]):.4f}"
                  f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    def pooled(arm):
        per_seed = {}
        for seed, rs in rows[arm].items():
            bu = np.array([r["best_utility"] for r in rs], dtype=float)
            per_seed[seed] = {
                "best_utility_mean": float(np.nanmean(bu)),
                "best_utility_mean_failures_as_zero": float(np.nan_to_num(bu, nan=0.0).mean()),
                "reached_global_max": int(sum(1 for r in rs if not np.isnan(r["best_utility"]) and abs(r["best_utility"] - r["global_max"]) < 1e-9)),
                "fallback_instances": int(sum(1 for r in rs if np.isnan(r["best_utility"]))),
                "distinct_valid_mean": float(np.mean([r["distinct_valid"] for r in rs])),
                "excess_diversity_mean": float(np.nanmean([r["excess_diversity"] for r in rs])),
                "terminal_checks_mean": float(np.mean([r["terminal_checks"] for r in rs])),
            }
        seeds = list(per_seed.values())
        return {
            "per_seed": per_seed,
            "best_utility_mean_over_seeds": float(np.mean([s["best_utility_mean"] for s in seeds])) if seeds else None,
            "best_utility_std_over_seeds": float(np.std([s["best_utility_mean"] for s in seeds])) if seeds else None,
            "distinct_valid_mean_over_seeds": float(np.mean([s["distinct_valid_mean"] for s in seeds])) if seeds else None,
            "excess_diversity_mean_over_seeds": float(np.mean([s["excess_diversity_mean"] for s in seeds])) if seeds else None,
        }

    sa = stage_a["stage_a"]["obligation"]
    greedy = sa["methods"]["s1_greedy"]["best_utility_mean"]
    out = {
        "what_this_is": ("Phase 4 STAGE B: S5 = trained Phase-3 samplers on the lattice main suite, exact-U scorer, "
                         "every arm x every seed. DIAGNOSTIC ONLY (decision 5): with a perfect scorer greedy is "
                         "globally optimal by arithmetic (G9), so S5 losing best-of-K to greedy is expected and is "
                         "not a finding. Gate 3's decision is Phase 9's."),
        "gate": "diagnostic only (decision 5): no best-of-K comparison is a Gate-3 verdict",
        "arm_selection": ("dissolved, not ruled: all nine arms reported. L7 (checker_led) is the C3 arm; "
                          "gaflownet was the best flow arm on TV at Gate 2 (inconclusive)."),
        "instances": len(envs), "K": k, "seeds": list(SEEDS), "scorer": "exact U (fix F13)",
        "reference": {"ceiling_mean": float(np.mean(ceiling)), "global_max_mean": float(np.mean(global_max)),
                      "s1_greedy_best_utility_mean_stage_a": greedy},
        "s5": {arm: pooled(arm) for arm in args.arms if rows.get(arm)},
        "rows": {arm: {str(s): rs for s, rs in seeds.items()} for arm, seeds in rows.items()},
        "seconds": round(time.perf_counter() - started, 1),
    }
    (REPO / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{'arm':<16}{'best-U (3 seeds)':>18}{'std':>8}{'distinct':>10}{'exc.div':>9}")
    print(f"{'ceiling':<16}{out['reference']['ceiling_mean']:>18.4f}")
    print(f"{'s1_greedy (A)':<16}{greedy:>18.4f}")
    for arm, p in out["s5"].items():
        print(f"{arm:<16}{p['best_utility_mean_over_seeds']:>18.4f}{p['best_utility_std_over_seeds']:>8.4f}"
              f"{p['distinct_valid_mean_over_seeds']:>10.2f}{p['excess_diversity_mean_over_seeds']:>9.3f}")
    print(f"written: {args.out} ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
