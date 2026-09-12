"""Phase 8's runner: the MuSiQue track, both arms, and the dev risk–coverage curve.

    python scripts/phase8_gate.py --smoke              # stub embedder, 64 pairs
    python scripts/phase8_gate.py --pairs 2000         # the pinned bge-small
    python scripts/phase8_gate.py                      # the full dev/train split

**Stage A only.** The conversational deletion-pair track and every decisive
evaluation number are exit criterion 13's deferred-by-name items and wait on
scope-c ingestion (Gate-0 item 9: decided, not run). This runner trains on
MuSiQue-Full contrast pairs and produces a **dev** curve; nothing it prints is a
gate quality result, and the artefact says so in its own body.

**G9's honesty stamp carries over from Phase 7 whole**, plus one bound specific
to this phase: MuSiQue **substitutes** distractors for the removed supporting
paragraphs, so both twins carry ~20 paragraphs and the whole ``pool_shape`` block
is near-identical across the label (measured 397/400 on dev). On this corpus the
gate can only learn from ``channel_scores`` — which makes it a *harder* training
set than the conversational one will be, not an easier one, but also means a
feature-importance reading here does not transfer.

**Both arms under identical budgets and seeds** (exit criterion 9): they differ
by a **column mask** over one featurisation, never by a re-featurisation, so any
difference between them is the arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from graft.config import load_config  # noqa: E402
from graft.gate import pins as gpins  # noqa: E402
from graft.gate.adapt_musique import ADAPTATION_LOSSES, UNANSWERABILITY_CONSTRUCTION, adapt_pairs  # noqa: E402
from graft.gate.decide import ARM_BLOCKS, arm_mask  # noqa: E402
from graft.gate.riskcov import contrast_pair_accuracy, evaluate  # noqa: E402
from graft.graphbuild.loaders import MUSIQUE_ROOT, load_split, musique_pairs  # noqa: E402
from graft.canonical import digest_of  # noqa: E402
from graft.runtime import deterministic_view, json_sanitize, run_manifest  # noqa: E402

OUT = REPO / "artefacts" / "phase8_gate.json"

HONESTY_STAMP: dict[str, str] = {
    "stage": (
        "Stage A only. The conversational deletion-pair track and every decisive "
        "evaluation number are deferred by name to Stage B, post-scope-c "
        "(GRAFT_PHASE8_BUILD.md exit criterion 13)."
    ),
    "corpus": (
        "trained and evaluated on MuSiQue-Full contrast pairs — a declared "
        "adaptation, never native supervision. LoCoMo adversarial (446) and "
        "LongMemEval's 30 are the evaluation sets and are NOT touched here."
    ),
    "pool_shape": (
        "MuSiQue substitutes distractors for removed supporting paragraphs, so "
        "pool_shape is near-identical across the label (397/400 dev pairs). The "
        "gate learns from channel_scores here; feature importance does not "
        "transfer to the conversational track."
    ),
    "numbers": (
        "a dev curve on a training-interface corpus. Not a gate quality result, "
        "and not comparable to any published abstention number."
    ),
}


def _split_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.stack([r["vector"] for r in rows]).astype(np.float64)
    y = np.asarray([r["label"] for r in rows], dtype=np.float64)
    return x, y, [r["question_id"] for r in rows]


def run(*, smoke: bool, pairs: int | None, out_path: Path, seeds: tuple[int, ...]) -> int:
    from graft.gate.model import build_gate, predict, train_gate

    config = load_config()
    started = time.perf_counter()

    if smoke:
        from graft.graphbuild.embed import StubEmbedder

        embedder: Any = StubEmbedder(dim=int(gpins.EMBEDDER["dim"]))
        limit = pairs or 64
    else:
        from graft.graphbuild.embed import Embedder

        embedder = Embedder(cache_dir=REPO / "artefacts" / "phase8" / "embed_cache")
        embedder.load()
        limit = pairs

    train_rows_raw = load_split("musique_full", "train", root=MUSIQUE_ROOT)
    dev_rows_raw = load_split("musique_full", "dev", root=MUSIQUE_ROOT)
    train_pairs, train_pair_report = musique_pairs(train_rows_raw)
    dev_pairs, dev_pair_report = musique_pairs(dev_rows_raw)

    dev_limit = None if limit is None else max(8, limit // 4)
    train_feats, train_report = adapt_pairs(
        train_pairs, embedder, pool_cap=config.pool_cap, with_question=True, limit=limit
    )
    dev_feats, dev_report = adapt_pairs(
        dev_pairs, embedder, pool_cap=config.pool_cap, with_question=True, limit=dev_limit
    )
    names = train_feats[0]["names"]
    x_tr, y_tr, _ = _split_rows(train_feats)
    x_dev, y_dev, dev_ids = _split_rows(dev_feats)

    arms: dict[str, Any] = {}
    # **B3(i): the models, not only their numbers.** Until 21 Aug 2026 this
    # script trained 3 seeds x 2 models x every arm and persisted *none* of
    # them, so applying the gate to LoCoMo's stored features meant retraining
    # from scratch -- the checkpoint gap PHASE11_DECISIONS.md §1.8 named.
    trained: dict[tuple, Any] = {}
    for arm in gpins.ARMS:
        mask = arm_mask(names, arm)
        for model_name in ("lr", "mlp"):
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                model = build_gate(model_name, int(mask.sum()), seed=seed)
                history = train_gate(
                    model, (x_tr, y_tr), (x_dev, y_dev), seed=seed, mask=mask
                )
                scores = predict(model, x_dev, mask)
                trained[(arm, model_name, seed)] = model
                metrics = evaluate(scores, y_dev, seed=seed, group_ids=dev_ids)
                pair_acc = contrast_pair_accuracy(scores, y_dev, dev_ids)
                per_seed.append(
                    {
                        "seed": seed,
                        "training": {k: v for k, v in history.items() if k != "history"},
                        "aurc_primary": metrics["aurc_primary"],
                        "contrast_pair_accuracy": pair_acc,
                        "aurc": metrics["aurc"],
                        "threshold": metrics["threshold"],
                        "at_threshold": metrics["at_threshold"],
                        "brier": metrics["brier"],
                        "ece": metrics["ece"],
                        "aurc_interval": metrics.get("aurc_interval"),
                        # **Exit criterion 11's per-question listing.**  The full
                        # curves were 14 MB of recomputable derivative while the
                        # auditable primitive -- which question, which label, what
                        # probability -- was absent entirely, so no paired
                        # re-analysis was reconstructible from the artefact.  The
                        # curves are dropped in favour of the rows they are
                        # computed from; `riskcov.evaluate` regenerates them
                        # exactly from these three columns.
                        "per_question": [
                            {"question_id": q, "label": float(l), "p_answerable": float(p)}
                            for q, l, p in zip(dev_ids, y_dev.tolist(), scores.tolist())
                        ],
                    }
                )
            aurcs = [s["aurc_primary"] for s in per_seed]
            accs = [s["contrast_pair_accuracy"]["accuracy"] for s in per_seed]
            accs = [a for a in accs if a is not None]
            arms[f"{arm}/{model_name}"] = {
                "arm": arm,
                "model": model_name,
                "blocks": list(ARM_BLOCKS[arm]),
                "features_used": int(mask.sum()),
                "seeds": per_seed,
                "aurc_mean": float(np.mean(aurcs)),
                "aurc_std": float(np.std(aurcs)),
                "pair_accuracy_mean": float(np.mean(accs)) if accs else None,
                "pair_accuracy_std": float(np.std(accs)) if accs else None,
            }

    elapsed = time.perf_counter() - started
    artefact = {
        "phase": 8,
        "stage": "A",
        "smoke": bool(smoke),
        "honesty_stamp": HONESTY_STAMP,
        "primary_metric": gpins.PRIMARY_METRIC,
        "safety_secondary": gpins.SAFETY_SECONDARY,
        "stage_g_fingerprint": gpins.stage_g_fingerprint(),
        "stage_g_frozen": gpins.frozen_values(),
        "corpus": {
            "train_pairs_available": train_pair_report["pairs"],
            "dev_pairs_available": dev_pair_report["pairs"],
            "train_used": train_report,
            "dev_used": dev_report,
            "adaptation_losses": dict(ADAPTATION_LOSSES),
            "unanswerability_construction": dict(UNANSWERABILITY_CONSTRUCTION),
        },
        "arms": arms,
        "leakage_reading": (
            "AURC is a GLOBAL ranking over the dev set, so a gap between the arms "
            "is NOT by itself evidence of leakage: the question embedding can "
            "improve cross-question calibration without ever separating a twin. "
            "The leakage-immune test is `contrast_pair_accuracy` — within a pair "
            "the question is byte-identical, so it can only be won on pool-side "
            "features. Verified 15 Aug 2026: the embedding block is identical "
            "across the label for 40/40 sampled dev pairs, and the pool_only mask "
            "selects no q_emb column. The decisive leakage measurement remains the "
            "zero-shot LoCoMo transfer (G2), which is Stage B."
        ),
        "deferred_by_name": {
            "conversational_track": "exit criterion 13 — needs scope-c ingestion",
            "locomo_adversarial": "primary evaluation, Stage B",
            "longmemeval_30": "in-domain check, interval-reported, Stage B",
            "orchestrator_integration": "exit criterion 14 — transferred to Phase 10 (G7)",
        },
        "fallback_counter": {gpins.FALLBACK_COUNTER: 0, "note": "zero until Phase 9 wires it (G5)"},
        "wall_clock_s": round(elapsed, 2),
        "manifest": run_manifest(config, seeds[0]),
    }
    # A **hash**, not the object.  Phase 7 used `digest_of(deterministic_view(...))`
    # and this stored the whole deterministic view instead -- which is not a digest,
    # cannot be compared at a glance, and was most of the artefact's size.  Exit
    # criterion 12 asks for equality across two runs; a hash is what makes that a
    # one-line check.
    artefact["determinism_digest"] = digest_of(
        deterministic_view({k: artefact[k] for k in ("arms", "stage_g_fingerprint", "corpus")})
    )

    # -- B3(i): persist the winning arm ------------------------------------
    # Selection is by the DECLARED primary (`pins.PRIMARY_METRIC` = AURC, lower
    # is better), not by pair accuracy and not by a look at both -- picking on
    # whichever of two metrics flatters the result is how a primary stops being
    # one. Ties break on the safety secondary, then on the arm name so the
    # choice is deterministic.
    checkpoint_report = None
    if arms and trained:
        import torch

        def _save(key: str, path: Path) -> dict[str, Any]:
            """One arm to disk. The MEDIAN seed by the declared primary, so the
            saved model is representative rather than the luckiest of three -- a
            best-seed checkpoint makes every downstream number an outlier."""
            entry = arms[key]
            by_primary = sorted(entry["seeds"], key=lambda r: r["aurc_primary"])
            middle = by_primary[len(by_primary) // 2]
            chosen_seed = int(middle["seed"])
            mask = arm_mask(names, entry["arm"])
            torch.save(
                {
                    "state_dict": trained[
                        (entry["arm"], entry["model"], chosen_seed)
                    ].state_dict(),
                    "arm": entry["arm"],
                    "model": entry["model"],
                    "seed": chosen_seed,
                    "mask": mask.tolist(),
                    "feature_names": list(names),
                    "in_dim": int(mask.sum()),
                    "threshold": middle["threshold"],
                    # **The blocks this arm actually READS.** A consumer can
                    # match these against a row's `gate_blocks_present` and
                    # refuse; without it, an arm trained on the question
                    # embedding applied to rows that carry 384 zeros there
                    # returns confident probabilities from empty columns, and
                    # nothing in the feature NAMES would show it.
                    "requires_blocks": list(ARM_BLOCKS[entry["arm"]]),
                    "stage_g_fingerprint": gpins.stage_g_fingerprint(),
                    "trained_on": "musique_full contrast pairs (train split)",
                    "selected_by": gpins.PRIMARY_METRIC,
                    "smoke": bool(smoke),
                },
                path,
            )
            return {
                "path": path.name,
                "arm": key,
                "seed": chosen_seed,
                "aurc_mean": entry["aurc_mean"],
                "features_used": int(mask.sum()),
                "requires_blocks": list(ARM_BLOCKS[entry["arm"]]),
            }

        # **Every arm is persisted, not only the winner** (21 Aug 2026). The
        # winner is chosen on MuSiQue dev, where every feature block is
        # populated; a downstream corpus may not carry them all. LoCoMo eval
        # rows carry `question_embedding: False` and 384 zero columns, so the
        # `with_question` winner is exactly the arm that CANNOT be applied
        # there -- and saving only the winner made the better-suited
        # `pool_only` arm unavailable without a 32-minute retrain.
        saved = {key: _save(key, out_path.with_name(f"phase8_gate_{key.replace('/', '_')}.pt"))
                 for key in sorted(arms)}

        winner_key = min(
            arms,
            key=lambda k: (
                arms[k]["aurc_mean"],
                -(arms[k]["pair_accuracy_mean"] or 0.0),
                k,
            ),
        )
        # The winner also lands under the plain name, as the default.
        default = _save(winner_key, out_path.with_name("phase8_gate.pt"))
        checkpoint_report = {
            **default,
            "seed_rule": "median of the three by the primary metric",
            "selected_by": gpins.PRIMARY_METRIC,
            "all_arms": saved,
            "reading": (
                "the threshold saved here was chosen on MuSiQue DEV. Applying it "
                "to another corpus is a transfer claim, and re-choosing it on "
                "that corpus's own curve is the leak the threshold rule forbids."
            ),
            "choosing_an_arm_downstream": (
                "the winner is chosen where every block is populated. A corpus "
                "whose rows lack a block this arm reads needs a different arm -- "
                "`requires_blocks` is on every checkpoint so the consumer can "
                "refuse rather than score zeros. LoCoMo eval rows carry no "
                "question embedding, so `pool_only/*` is the applicable arm there."
            ),
        }
    artefact["checkpoint"] = checkpoint_report

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(json_sanitize(artefact), indent=1, sort_keys=True), encoding="utf-8"
    )
    # `relative_to` raises on a relative --out, *after* the artefact is written —
    # a crash that loses nothing but reports failure on a successful run.
    shown = out_path.resolve()
    print(f"wrote {shown.relative_to(REPO) if shown.is_relative_to(REPO) else shown}  ({elapsed:.1f}s)")
    print(f"  train rows {len(train_feats)}  dev rows {len(dev_feats)}  features {len(names)}")
    print(f"  {'arm/model':22} {'AURC(nat)':>10} {'std':>7} {'pair-acc':>9} {'std':>7} {'feats':>6}")
    for key in sorted(arms):
        a = arms[key]
        pa = a["pair_accuracy_mean"]
        ps = a["pair_accuracy_std"]
        print(
            f"  {key:22} {a['aurc_mean']:>10.4f} {a['aurc_std']:>7.4f} "
            f"{(f'{pa:.4f}' if pa is not None else 'n/a'):>9} "
            f"{(f'{ps:.4f}' if ps is not None else 'n/a'):>7} {a['features_used']:>6}")
    print("\n  NOTE: Stage A. A dev curve on the training-interface corpus — not a")
    print("  gate quality result. pair-acc is the leakage-immune number (chance 0.5):")
    print("  within a pair the question is byte-identical, so it is won on pool features.")
    return 0


def load_gate(path: Path):
    """B3(i) -- ``(model, mask, blob)`` from a checkpoint written by :func:`run`.

    The mask travels **with** the weights and is applied by the caller, because
    a gate restored without its arm mask would silently score the wrong columns:
    the state dict's ``in_dim`` is the masked width, so a full feature matrix
    would either crash or -- worse, if the widths happened to agree -- produce
    numbers from the wrong features.
    """
    import numpy as np
    import torch

    from graft.gate.model import build_gate

    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    mask = np.asarray(blob["mask"], dtype=bool)
    if int(mask.sum()) != int(blob["in_dim"]):
        raise ValueError(
            f"checkpoint mask selects {int(mask.sum())} features but the model "
            f"was built for {blob['in_dim']}; the two disagree about what the "
            "gate reads"
        )
    model = build_gate(blob["model"], int(blob["in_dim"]), seed=int(blob["seed"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, mask, blob


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="stub embedder, 64 pairs, no GPU")
    parser.add_argument("--pairs", type=int, default=None, help="cap training pairs")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return run(
        smoke=args.smoke,
        pairs=args.pairs,
        out_path=args.out,
        seeds=tuple(int(s) for s in gpins.TRAINING_GATE["seeds"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
