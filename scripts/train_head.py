"""Train the distilled utility head — Stage D's scorer. **CPU only, minutes.**

**Why this is the training that matters for a LoCoMo number, and the whole
Phase-9 ladder is not.**

`PHASE10_DECISIONS.md` §1.4 measured that ``sufficiency(X, empty) = 1.0``, so
``U`` is **vacuous at inference** and the orchestrator refuses to rank without a
distilled head.  That head — not the GFlowNet policy — is what stands between
Stage D ranking candidates and Stage D returning them in tie-break order.  It is
a two-layer MLP under a 200,000-parameter cap fitting ``Û`` to exact ``U``:
supervised regression, minutes of CPU.

`GRAFT_PHASE9_BUILD.md` §6 decision 11's ladder (``N_real`` = 200,000
trajectories, derived from the slowest arm at 29.67 traj/s) trains the *policy*,
and is ~2 h **per arm per seed**.  That ladder answers Gate 3.  It is not needed
to get a defensible end-to-end number, and on a three-day deadline the head is
the entire return on the first hour.

**What the head being trained changes in the artefact.** Untrained, ``HeadScorer``
wraps a randomly-initialised MLP, so best-of-K ranks by noise and
``read_path_stamp`` has to say ``policy_trained=False`` with the ranking
unusable.  Trained, the run reports a held-out Spearman ρ against exact ``U`` —
and `graft/setgen/pins.py` DISTILL's ``report_rho`` makes that ρ mandatory beside
any Gate-3 row, because "the learned sampler won" and "the scorer was nearly
perfect" are different findings.

**Training corpora are 2Wiki and MuSiQue-Ans, and LoCoMo is structurally
refused.** `DATASET_DECISION.md` §1 makes LoCoMo evaluation-only in every
component.  A head fitted on LoCoMo pools would silently void the zero-shot
declaration that the entire comparison rests on, so :func:`_refuse_locomo` checks
rather than trusts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from graft.config import load_config  # noqa: E402
from graft.setgen import pins as spins  # noqa: E402
from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer  # noqa: E402
from graft.setgen.distill import (  # noqa: E402
    build_head,
    pool_sets,
    spearman,
    train_head,
)
from graft.setgen.policy import Policy  # noqa: E402
from graft.setgen.realenv import RealEnvironment, sample_real  # noqa: E402

PHASE9_ROOT = REPO / "data" / "phase9" / "raw"


def _refuse_locomo(examples) -> None:
    """A structural guard, not a comment.

    `DATASET_DECISION.md` §1: LoCoMo is evaluation only, in every component. A
    head fitted on LoCoMo pools would void the zero-shot declaration the whole
    reference-table comparison rests on, and it would do so invisibly -- the
    weights look identical either way. Checked here because this is the only
    place a corpus reaches the head.
    """
    bad = [
        e.example_id
        for e in examples
        if str((e.meta or {}).get("corpus", "")).lower() == "locomo"
        or str(e.example_id).startswith("locomo/")
    ]
    if bad:
        raise SystemExit(
            f"refusing to train on {len(bad)} LoCoMo example(s), e.g. {bad[:3]}. "
            "LoCoMo is evaluation only in every component (DATASET_DECISION.md §1); "
            "training on it voids the zero-shot declaration silently."
        )


def collect(examples, config, *, rollouts: int, epsilon: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sampled valid terminals → ``(pooled_features, exact_U)``.

    The target is exact ``U`` **at visited terminals**, which is DISTILL's own
    ``target`` string. Sampling rather than enumerating because the real state
    space is not enumerable -- the reason Phase 9 exists as a separate phase.
    """
    xs: list[torch.Tensor] = []
    ys: list[float] = []
    per_example: list[dict[str, Any]] = []

    for i, ex in enumerate(examples):
        env = RealEnvironment(ex, config, range_samples=4)
        torch.manual_seed(seed + i)
        feat = RealFeaturizer(
            ex, Policy(*RealFeaturizer.dims(), hidden=16), config, delta_d=False
        )
        traj = sample_real(feat, env, rollouts, np.random.default_rng(seed + i), epsilon=epsilon)
        sets = [t for r, t in enumerate(traj.terminals()) if not traj.is_fail[r]]
        # Deduplicated per example: the same terminal sampled twice is one
        # training row, and leaving duplicates in weights popular sets by how
        # often the *sampler* found them rather than by their utility.
        seen: set[frozenset[str]] = set()
        unique = []
        for s in sets:
            key = frozenset(s)
            if key not in seen:
                seen.add(key)
                unique.append(list(s))
        if not unique:
            per_example.append({"example_id": ex.example_id, "sets": 0, "note": "no valid terminal sampled"})
            continue
        x = pool_sets(feat, unique)
        y = [float(env.utility(s)) for s in unique]
        xs.append(x)
        ys.extend(y)
        per_example.append(
            {
                "example_id": ex.example_id,
                "sets": len(unique),
                "u_mean": round(float(np.mean(y)), 4),
                "u_spread": round(float(np.max(y) - np.min(y)), 4),
            }
        )

    if not xs:
        raise SystemExit(
            "no valid terminals sampled on any example; the head has nothing to fit"
        )
    return (
        torch.cat(xs, dim=0),
        torch.as_tensor(np.asarray(ys, dtype=np.float32), dtype=torch.float32),
        {"per_example": per_example, "rows": len(ys)},
    )


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
    parser.add_argument("--examples", type=int, default=200, help="training examples per corpus, stratified")
    parser.add_argument("--rollouts", type=int, default=40, help="sampled trajectories per example")
    parser.add_argument("--epsilon", type=float, default=0.3, help="exploration for the sampler")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None, help="defaults to the config's first seed")
    parser.add_argument("--out", default="artefacts/utility_head.pt")
    parser.add_argument("--report", default="artefacts/utility_head.json")
    parser.add_argument(
        "--real-embedder", action="store_true",
        help="use the pinned embedder rather than the stub. Slower; needed for a "
        "head whose features match what locomo_eval.py will feed it.",
    )
    parser.add_argument("--split", default="train", help="corpus split (train|dev)")
    parser.add_argument(
        "--embedder-device", default="cpu",
        help="device for the pinned embedder. **Defaults to cpu** so this can run "
        "in parallel with a GPU ingestion or verify pass -- the head is a CPU "
        "regression and the embedder is the only part that would contend for VRAM. "
        "Same model and revision either way, so the head stays compatible with "
        "what locomo_eval.py feeds it.",
    )
    args = parser.parse_args()

    config = load_config()
    seed = int(args.seed if args.seed is not None else config.seeds[0])

    if args.real_embedder:
        from graft.graphbuild.embed import Embedder

        embedder: Any = Embedder(
            device=args.embedder_device,
            cache_dir=REPO / "artefacts" / "head" / "embed_cache",
        )
        embedder.load()
        print(f"embedder on {embedder.device} (requested {args.embedder_device})")
    else:
        from graft.graphbuild.embed import StubEmbedder

        embedder = StubEmbedder(dim=384)

    from graft.graphbuild.loaders import load_split
    from graft.setgen.corpora import musique_ans, wiki2

    print(f"loading examples (split={args.split}, {args.examples} per corpus, stratified)")
    examples = []
    strata: dict[str, Any] = {}
    for name, module in (("2wiki", wiki2), ("musique_ans", musique_ans)):
        rows = load_split(name, args.split, root=PHASE9_ROOT)
        got, report = module.load_examples(
            args.split, rows=rows, limit=args.examples, embedder=embedder
        )
        strata[name] = report.get("types", {})
        examples.extend(got)
        print(f"  {name}: {len(got)} examples, types {strata[name]}")

    _refuse_locomo(examples)

    started = time.perf_counter()
    print(f"sampling {args.rollouts} rollouts x {len(examples)} examples on CPU")
    x, y, collected = collect(
        examples, config, rollouts=args.rollouts, epsilon=args.epsilon, seed=seed
    )
    sampled_s = time.perf_counter() - started
    print(f"  {collected['rows']} (set, U) rows in {sampled_s / 60:.1f} min")

    # Split by row, shuffled once under the run's seed: the head is a global
    # regression, so a per-example split would hold out whole pools and measure
    # transfer rather than fit -- a different question from the one rho answers.
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    x, y = x[order], y[order]
    cut = int(0.8 * x.shape[0])
    if cut == 0 or cut == x.shape[0]:
        raise SystemExit(f"{x.shape[0]} rows is too few to split; raise --examples or --rollouts")

    head = build_head(ATOM_WIDTH, seed=seed)
    n_params = sum(p.numel() for p in head.parameters())
    cap = int(spins.DISTILL["max_params"])
    if n_params > cap:
        raise SystemExit(f"head has {n_params} params against the {cap} cap")

    print(f"training: {cut} train / {x.shape[0] - cut} dev rows, {n_params} params (cap {cap})")
    history = train_head(
        head, (x[:cut], y[:cut]), (x[cut:], y[cut:]), seed=seed, epochs=args.epochs
    )

    head.eval()
    with torch.no_grad():
        predicted = head(x[cut:]).cpu().numpy()
    rho = spearman(predicted, y[cut:].cpu().numpy())

    torch.save(
        {
            "state_dict": head.state_dict(),
            "atom_width": int(ATOM_WIDTH),
            "seed": seed,
            "dev_spearman": float(rho),
            "embedder": "stub" if not args.real_embedder else getattr(embedder, "name", "pinned"),
        },
        REPO / args.out,
    )

    report = {
        "what_this_is": (
            "the distilled utility head -- Stage D's inference scorer. NOT the "
            "Phase-9 policy ladder, which is ~2 h per arm per seed and answers "
            "Gate 3 (GRAFT_PHASE9_BUILD.md §6 decision 11)."
        ),
        "training_corpora": ["2wiki", "musique_ans"],
        "locomo_used": False,
        "zero_shot_declaration_intact": (
            "LoCoMo appears in no training component; _refuse_locomo asserts it"
        ),
        "split": args.split,
        "strata": strata,
        "embedder": "stub" if not args.real_embedder else getattr(embedder, "name", "pinned"),
        "embedder_device": None if not args.real_embedder else getattr(embedder, "device", None),
        "examples": len(examples),
        "rollouts_per_example": args.rollouts,
        "rows": int(x.shape[0]),
        "train_rows": cut,
        "dev_rows": int(x.shape[0] - cut),
        "params": n_params,
        "param_cap": cap,
        "dev_spearman": float(rho),
        "rho_reading": (
            "DISTILL['report_rho'] makes this mandatory beside any Gate-3 row: "
            "'the learned sampler won' and 'the scorer was nearly perfect' are "
            "different findings, and rho is what separates them."
        ),
        "best_epoch": history.get("best", {}),
        "sampling_minutes": round(sampled_s / 60, 2),
        "total_minutes": round((time.perf_counter() - started) / 60, 2),
        "per_example": collected["per_example"][:50],
        "checkpoint": str(args.out),
    }
    Path(REPO / args.report).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"dev Spearman rho vs exact U: {rho:.4f}")
    print(f"saved: {args.out}")
    print(f"report: {args.report}")
    if args.real_embedder is False:
        print()
        print("NOTE: trained on the STUB embedder. For a head whose features match")
        print("what locomo_eval.py feeds it, re-run with --real-embedder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
