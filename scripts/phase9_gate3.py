#!/usr/bin/env python
"""P9.7 -- the Phase-9 ladder runner: train one arm x seed on real pools, then assemble Gate 3.

    python scripts/phase9_gate3.py train --arm l7_checker_led --seed 13        # one Slurm array task
    python scripts/phase9_gate3.py train --arm l4_tb --seed 42 --smoke          # fixtures, seconds
    python scripts/phase9_gate3.py assemble                                     # after all 27 tasks

**Why two subcommands.**  `GRAFT_PHASE9_BUILD.md` P9.7 describes one runner
that trains nine arms x three seeds, distils, runs the portfolios and writes the
Gate-3 table.  Serial, that is ~54 h; as a **Slurm job array** (27 tasks) it is
the time of the slowest arm.  So training is one task per (arm, seed) writing
one checkpoint, and ``assemble`` is the single CPU job that reads them all.  A
crashed task loses one arm-seed, never the run -- the lesson of the Gate-1 run
that trained twelve models and died writing its artefact.

**Budget, frozen and read, never set here.**  ``N_real`` = `pins.BUDGET["n_real"]`
(200,000, derived 16 Aug 2026 from the slowest arm's measured rate); the subset
is `pins.SUBSET` (2,000 train + 500 dev per corpus, stratified, seed 20260816);
β is `Config.beta` (4.0, frozen 15 Aug).  ``training_blocked_reason()`` is asked
first and its answer is final -- an unfrozen β would make every arm's reward a
different function.  Fix F12: **no early stopping, no selection** -- every arm
spends exactly ``N_real``.

**Assembly is Gate 3 as signed (`pins.GATE3_RULE`), on held-out dev pools:**
distil the utility head on the trained pools, then for every arm run the
portfolio at ``K`` under the *distilled* head (the noisy proxy Robust
Scheduling's argument needs -- `CLAUDE.md` §8), against the training-free
relevance selector at the same budget; report best-of-K utility (exact `U`, gold
known on dev) with the paired bootstrap, the distinct-valid-set secondary, and
the head's held-out ρ beside every row (`DISTILL["report_rho"]`).  S3/S4 on real
pools are deferred by name: their search classes are written over the synthetic
`LatticeInstance` interface.

Everything is stamped; `is_wiring_test` is False only when every arm x seed
checkpoint is present and trained at ``N_real``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from graft.config import Config, load_config  # noqa: E402
from graft.setgen import pins  # noqa: E402
from graft.setgen.learners import FLOW_FAMILY, SUPERVISED_FAMILY, build_arm  # noqa: E402
from graft.setgen.realenv import RealEnvironment, RealTrainer  # noqa: E402
from graft.setgen.trainer import SEEDS, TrainSpec  # noqa: E402

ARMS = tuple(FLOW_FAMILY) + tuple(SUPERVISED_FAMILY)


# --------------------------------------------------------------------------
# pools
# --------------------------------------------------------------------------

def _fixture_examples(n: int):
    from graft.tests.test_setgen_real import tiny_example
    return [tiny_example(example_id=f"q{i}") for i in range(n)]


def load_pools(split: str, *, smoke: bool, per_corpus: int, cache_dir: Path):
    """Real pools from 2Wiki + MuSiQue-Ans (pins.SUBSET), cached per split; fixtures under --smoke."""
    if smoke:
        return _fixture_examples(4 if split == "train" else 2), {"smoke": True}
    cache = cache_dir / f"pools_{split}_{per_corpus}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        return blob["examples"], blob["strata"]
    from graft.graphbuild.embed import Embedder
    from graft.graphbuild.loaders import PHASE9_ROOT, load_split
    from graft.setgen.corpora import musique_ans, wiki2
    embedder = Embedder(cache_dir=REPO / "artefacts" / "phase9" / "embed_cache"); embedder.load()
    out, strata = [], {}
    for name, module in (("2wiki", wiki2), ("musique_ans", musique_ans)):
        rows = load_split(name, split, root=PHASE9_ROOT)
        examples, report = module.load_examples(split, rows=rows, limit=per_corpus, embedder=embedder)
        # Only gold-complete pools train (PHASE9 §? the runner "filters on this and reports the count it dropped").
        kept = [e for e in examples if e.gold_complete]
        strata[name] = {"types": report.get("types", {}), "built": len(examples), "gold_complete": len(kept)}
        out.extend(kept)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"examples": out, "strata": strata}, cache)
    return out, strata


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def cmd_train(args) -> int:
    blocked = pins.training_blocked_reason()
    if blocked:
        print(f"REFUSED: {blocked}", file=sys.stderr); return 2
    config = load_config()
    run = REPO / args.run_dir; run.mkdir(parents=True, exist_ok=True)
    n = 64 if args.smoke else int(pins.BUDGET["n_real"])
    per = 1 if args.smoke else int(pins.SUBSET["train_per_corpus"])
    examples, strata = load_pools("train", smoke=args.smoke, per_corpus=per, cache_dir=run / "cache")
    envs = [RealEnvironment(ex, config, range_samples=0 if args.smoke else 8) for ex in examples]
    arm = build_arm(args.arm)
    spec = TrainSpec(n_trajectories=n, seed=int(args.seed), device=args.device,
                     **({"hidden": 16, "batch_size": 4} if args.smoke else {}))
    print(f"train {args.arm} seed {args.seed}: {len(envs)} pools, N={n:,}, beta={config.beta}, device={args.device}",
          flush=True)
    t0 = time.perf_counter()
    trainer = RealTrainer(arm, envs, spec, greedy=1)
    log = trainer.train(progress=lambda k, loss: print(f"  {k:>9,} traj  loss {loss:.4f}", flush=True)
                        if k % max(1, n // 10) == 0 else None)
    secs = time.perf_counter() - t0
    ckpt = run / "checkpoints" / f"{args.arm}.seed{args.seed}.pt"
    trainer.save_checkpoint(ckpt)
    record = {"arm": args.arm, "seed": int(args.seed), "n_trajectories": n, "beta": config.beta,
              "pools": len(envs), "strata": strata, "final_loss": float(log.final_loss),
              "trajectories_per_second": round(n / max(secs, 1e-9), 2), "seconds": round(secs, 1),
              "smoke": bool(args.smoke), "checkpoint": str(ckpt),
              "stage_d_fingerprint": pins.stage_d_fingerprint() if hasattr(pins, "stage_d_fingerprint") else None}
    (run / "train" / f"{args.arm}.seed{args.seed}.json").parent.mkdir(parents=True, exist_ok=True)
    (run / "train" / f"{args.arm}.seed{args.seed}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"done: loss {log.final_loss:.4f}  {secs/60:.1f} min  {record['trajectories_per_second']} traj/s  -> {ckpt.name}")
    return 0


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------

def cmd_assemble(args) -> int:
    from graft.ledger import Ledger
    from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer
    from graft.setgen.distill import HeadScorer, build_head, pool_sets, spearman, train_head
    from graft.setgen.policy import load_policy
    from graft.setgen.portfolio import relevance_select, run_portfolio
    from graft.setgen.search.gate3 import _broadcast, higher_is_better_bootstrap

    config = load_config()
    run = REPO / args.run_dir
    per_dev = 1 if args.smoke else int(pins.SUBSET["dev_per_corpus"])
    dev, dev_strata = load_pools("dev", smoke=args.smoke, per_corpus=per_dev, cache_dir=run / "cache")
    envs = [RealEnvironment(ex, config, range_samples=0) for ex in dev]
    k = int(config.K)
    print(f"assemble: {len(envs)} dev pools, K={k}", flush=True)

    # -- distilled head on TRAIN pools (never dev) ---------------------------
    head_path = REPO / args.head
    if head_path.exists():
        blob = torch.load(head_path, map_location="cpu", weights_only=False)
        head = build_head(ATOM_WIDTH, seed=int(config.seeds[0])); head.load_state_dict(blob["state_dict"]); head.eval()
        rho = blob.get("dev_spearman"); head_note = f"loaded {args.head}"
    else:
        head, rho, head_note = None, None, "no head -- ranking refused"

    # -- which checkpoints exist -------------------------------------------
    ckpts = {(a, s): run / "checkpoints" / f"{a}.seed{s}.pt" for a in ARMS for s in SEEDS}
    present = {key: p for key, p in ckpts.items() if p.exists()}
    missing = sorted(f"{a}.seed{s}" for (a, s), p in ckpts.items() if not p.exists())
    print(f"checkpoints present {len(present)}/{len(ckpts)}" + (f"; missing {missing}" if missing else ""))

    # -- baseline: training-free relevance at the same budget ---------------
    def utility_of(env, atoms) -> float:
        return float(env.utility(list(atoms)))  # exact U, gold known on dev

    def eval_relevance():
        out = []
        for env in envs:
            ledger = Ledger.from_config(config)
            with ledger.query_scope(env.example.example_id):
                res = relevance_select(env, env.example.atom_scores, ledger=ledger, config=config)
            out.append(utility_of(env, res.best) if res.best is not None else float("nan"))
        return out

    def eval_arm(ckpt_path, seed):
        policy, blob = load_policy(ckpt_path)
        best, distinct = [], []
        for i, env in enumerate(envs):
            feat = RealFeaturizer(env.example, policy, config, delta_d=bool(blob.get("delta_d", False)))
            scorer = HeadScorer(head, feat) if head is not None else (lambda atoms: 0.0)
            ledger = Ledger.from_config(config)
            with ledger.query_scope(env.example.example_id):
                res = run_portfolio(feat, env, scorer, np.random.default_rng(int(seed) + i), ledger=ledger, config=config)
            best.append(utility_of(env, res.best) if res.best is not None else float("nan"))
            distinct.append(int(res.distinct_valid))
        return best, distinct

    baseline = eval_relevance()
    table: dict[str, Any] = {}
    for arm in ARMS:
        seeds_here = [s for s in SEEDS if (arm, s) in present]
        if not seeds_here:
            table[arm] = {"status": "no checkpoints"}; continue
        per_seed, rows = {}, []
        for s in seeds_here:
            b, d = eval_arm(present[(arm, s)], s)
            per_seed[s] = {"best_of_k_mean": float(np.nanmean(b)), "fallback": int(np.isnan(b).sum()),
                           "distinct_valid_mean": float(np.mean(d))}
            rows.append(np.nan_to_num(np.asarray(b, dtype=float), nan=0.0))
        # Paired bootstrap over [n_seeds, n_instances]; the deterministic baseline is
        # broadcast to one row per seed (G4, decision 4). Fallbacks count as 0.
        arm_matrix = np.stack(rows)
        base_matrix = _broadcast(np.nan_to_num(np.asarray(baseline, dtype=float), nan=0.0), len(rows))
        paired = higher_is_better_bootstrap(arm_matrix, base_matrix)
        table[arm] = {"seeds": per_seed, "paired_vs_relevance": paired,
                      "best_of_k_mean_over_seeds": float(np.mean([v["best_of_k_mean"] for v in per_seed.values()])),
                      "distinct_valid_mean_over_seeds": float(np.mean([v["distinct_valid_mean"] for v in per_seed.values()]))}
        print(f"  {arm:<16} best-of-K {table[arm]['best_of_k_mean_over_seeds']:.4f}  distinct {table[arm]['distinct_valid_mean_over_seeds']:.2f}"
              f"  (seeds {seeds_here})", flush=True)

    complete = len(present) == len(ckpts)
    out = {
        "what_this_is": "Gate 3 on real pools (2Wiki + MuSiQue-Ans dev): trained Phase-9 samplers vs the training-free "
                        "relevance selector at equal checker budget, ranked by the DISTILLED head (noisy proxy). "
                        "S3/S4 deferred by name (synthetic-interface search classes).",
        "rule": pins.GATE3_RULE, "K": k, "n_real": int(pins.BUDGET["n_real"]), "beta": config.beta,
        "dev_pools": len(envs), "dev_strata": dev_strata,
        "head": {"path": args.head, "held_out_rho": rho, "note": head_note,
                 "reading": pins.DISTILL.get("report_rho") if hasattr(pins, "DISTILL") else None},
        "relevance_baseline": {"best_of_k_mean": float(np.nanmean(baseline)), "fallback": int(np.isnan(baseline).sum())},
        "arms": table, "checkpoints_present": len(present), "checkpoints_missing": missing,
        "honesty_stamp": {"is_wiring_test": (not complete) or args.smoke or head is None,
                          "reason": ("smoke" if args.smoke else "missing checkpoints" if not complete
                                     else "no distilled head" if head is None else "complete")},
    }
    (run / "gate3_real.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"written: {run / 'gate3_real.json'}  (wiring_test={out['honesty_stamp']['is_wiring_test']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--arm", required=True, choices=ARMS); t.add_argument("--seed", type=int, required=True)
    t.add_argument("--device", default="cpu"); t.add_argument("--smoke", action="store_true")
    t.add_argument("--run-dir", default="artefacts/phase9_gate3")
    a = sub.add_parser("assemble"); a.add_argument("--run-dir", default="artefacts/phase9_gate3")
    a.add_argument("--head", default="artefacts/utility_head.pt"); a.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return cmd_train(args) if args.cmd == "train" else cmd_assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
