#!/usr/bin/env python
"""B3(iii) — apply the MuSiQue-trained gate to a finished LoCoMo run. CPU only.

**The whole point is that this costs no GPU.** Every eval row already carries
``gate_features`` and ``gate_feature_names``, recorded per question precisely so
a threshold could be applied later without touching the reader again (Phase-8
decision 2). What was missing until 21 Aug 2026 was a *persisted* gate: the
Phase-8 script trained three seeds of every arm and saved none of them, so
"apply the gate" meant "retrain it first" — the checkpoint gap
`PHASE11_DECISIONS.md` §1.8 named. `scripts/phase8_gate.py` now writes
``artefacts/phase8_gate.pt`` (plus one file per arm) and this reads it.

**The leak rule, which is the reason this script is written the way it is.**
The operating threshold is chosen on **MuSiQue dev** and transferred. It is
never re-chosen on LoCoMo's own risk–coverage curve, because a threshold picked
on the evaluation set is not a selective classifier — it is a fit to the test
data wearing one's clothes. This script therefore reports two different things
and never confuses them:

* the **transferred** operating point — the checkpoint's own threshold, applied
  as-is. This is the honest number.
* the **full curve and AURC** — a ranking property that needs no threshold, and
  which is safe to report because it commits to no operating point.

An oracle threshold chosen on LoCoMo is computed too, and labelled
``leaked_do_not_report`` in the artefact, so the gap between transfer and oracle
is visible as a *diagnostic* rather than quotable as a result.

Selective prediction framing: **[EVIDENCE]** Geifman & El-Yaniv, "Selective
Classification for Deep Neural Networks", NeurIPS 2017 — risk at a coverage
level, not accuracy.

    python scripts/locomo_gate_posthoc.py --rows results/locomo_eval_rows2.jsonl

Labels: a LoCoMo question is *answerable* (label 1) unless it is adversarial.
That is the same convention `EVAL_PREVALENCES["locomo"] = 0.2246` counts, and it
is the corpus's own annotation rather than a derived judgement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from graft.gate import pins as gpins  # noqa: E402
from graft.gate.riskcov import (  # noqa: E402
    aurc,
    choose_threshold,
    reweight,
    risk_coverage,
    selective_metrics,
)

DEFAULT_CKPT = REPO / "artefacts" / "phase8_gate.pt"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def feature_matrix(rows: list[dict], expected_names: list[str]) -> np.ndarray:
    """Rows → the gate's input matrix, refusing on any feature-name disagreement.

    **Checked, not assumed.** The gate was trained on MuSiQue features and is
    being applied to LoCoMo ones; if the two orders differ by even one column
    the model reads the wrong numbers and still returns confident-looking
    probabilities. `PHASE8_DECISIONS.md` §3.3 is the standing example of a
    feature block quietly meaning something else than its consumer thought.
    """
    names = rows[0].get("gate_feature_names")
    if not names:
        raise SystemExit(
            "rows carry no `gate_feature_names`; this run predates Phase-8 "
            "decision 2 and cannot be gated post hoc"
        )
    if list(names) != list(expected_names):
        missing = set(expected_names) - set(names)
        extra = set(names) - set(expected_names)
        raise SystemExit(
            "feature names disagree between the checkpoint and the rows.\n"
            f"  only in checkpoint: {sorted(missing)}\n"
            f"  only in rows:       {sorted(extra)}\n"
            "Refusing: a gate applied to mis-ordered features returns "
            "probabilities that mean nothing."
        )
    for r in rows:
        if list(r.get("gate_feature_names") or []) != list(names):
            raise SystemExit(f"row {r['question_id']} has a different feature order")
    return np.asarray([r["gate_features"] for r in rows], dtype=np.float32)


def gate_scores(model, x, mask) -> tuple:
    """``(ranking_score, probability, diagnostics)`` — ranked on **logits**.

    **A correction is recorded here, because the first version of this docstring
    claimed a defect that does not exist** (21 Aug 2026). It asserted that
    `graft.gate.model.predict`'s float32 sigmoid underflowed to exactly 0.0 on
    LoCoMo's very negative logits, tying every question and making the
    risk–coverage curve vacuous. **Measured: it does not.** `predict` returns
    **1,974 distinct values across 1,986 rows and no exact zeros** — the
    smallest is 7.66e-36, comfortably inside float32's subnormal range. The
    "collapse" was a `:.4f` print format rendering 1.3e-14 as `0.0000`, in a
    diagnostic written to inspect the very thing it then misreported.

    This is `PHASE11_DECISIONS.md` §1.1's pattern exactly — reading a display
    artefact as a finding — and it is kept rather than quietly deleted because
    the failure mode is the reusable part. **The AURC computed before this was
    'found' was correct.**

    Ranking on logits is retained anyway, for two honest reasons that do not
    require a bug: every selective-prediction quantity here is rank-based, so
    the monotone logit is lossless and float64 gives headroom if a future
    off-distribution corpus pushes logits further; and computing the
    probability explicitly in float64 is what makes ``probability_max`` — and
    therefore the *unreachable threshold* — visible in the artefact at all.
    `graft.gate.model.predict` is correct and is unchanged.
    """
    import torch

    with torch.no_grad():
        logits = model(torch.as_tensor(x[:, mask])).numpy().astype(np.float64)
    probability = 1.0 / (1.0 + np.exp(-logits))
    distinct_f32 = int(len(np.unique(np.float32(probability))))
    return logits, probability, {
        "ranked_on": "logits",
        "why": (
            "rank-based metrics are invariant to the monotone sigmoid, so the "
            "logit is lossless and keeps headroom off-distribution; the float64 "
            "probability is computed alongside because the threshold comparison "
            "needs a probability and `probability_max` is what shows an "
            "unreachable operating point"
        ),
        "logit_min": float(logits.min()),
        "logit_max": float(logits.max()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "distinct_logits": int(len(np.unique(logits))),
        "distinct_probabilities_in_float32": distinct_f32,
        # Measured, not assumed. A float32 sigmoid was ONCE claimed here to
        # collapse this ranking; it does not, and the flag reports the fact
        # rather than the claim.
        "float32_ranking_is_intact": bool(distinct_f32 > 1),
    }


def auroc(scores, labels) -> float:
    """Mann-Whitney AUROC — the share of (answerable, adversarial) pairs the
    gate orders correctly. 0.5 is chance.

    Reported because it is **threshold-free and leak-free**: unlike anything at
    an operating point it cannot be flattered by a threshold chosen anywhere,
    so it is the honest headline for a transfer claim.
    """
    z = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    # **Average ranks over ties**, not `argsort` positions. With arbitrary tie
    # ordering a gate that scores every question identically reads as AUROC 0.0
    # or 1.0 depending only on input order -- a system with no ranking scoring
    # as a perfect one. Ties must be 0.5, and that is exactly the degenerate
    # case a transfer study is most likely to hit.
    order = np.argsort(z, kind="mergesort")
    ranks = np.empty(len(z), dtype=np.float64)
    ranks[order] = np.arange(1, len(z) + 1)
    ordered = z[order]
    start = 0
    for i in range(1, len(ordered) + 1):
        if i == len(ordered) or ordered[i] != ordered[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def check_blocks(rows: list[dict], blob: dict) -> None:
    """Refuse an arm that reads a feature block these rows do not carry.

    **The check the feature-name comparison cannot make.** LoCoMo eval rows
    carry all 435 feature *names* including 384 ``q_emb_*`` columns — and every
    one of those columns is exactly 0.0, because the runner records
    ``question_embedding: False``. A ``with_question`` arm applied to them reads
    384 zeros it was trained to use and returns probabilities that look entirely
    normal. Names matching is not features matching.
    """
    required = set(blob.get("requires_blocks") or ())
    if not required:
        return  # a checkpoint from before this field existed; column check still runs
    present = {
        name for name, ok in (rows[0].get("gate_blocks_present") or {}).items() if ok
    }
    missing = sorted(required - present - {"present"})
    if missing:
        raise SystemExit(
            f"the checkpoint's arm {blob.get('arm')!r} reads {missing}, which "
            f"these rows do not carry (gate_blocks_present says so).\n"
            "Refusing: the columns exist by name and are empty by value, so "
            "the gate would score zeros and report them confidently.\n"
            "Use an arm whose blocks the rows have — for LoCoMo eval rows that "
            "is `pool_only`:\n"
            "  --checkpoint artefacts/phase8_gate_pool_only_mlp.pt"
        )


def check_selected_columns_are_populated(x, mask, blob) -> None:
    """A second, value-level guard: refuse if a masked-in block is all zeros.

    `check_blocks` trusts the row's own `gate_blocks_present` flag. This trusts
    nothing and looks at the numbers, so a row that mislabels its own blocks is
    still caught.
    """
    selected = x[:, mask]
    if selected.size and not selected.any():
        raise SystemExit(
            "every feature this gate reads is 0.0 across all rows — the arm and "
            "the rows do not match. Refusing to report a curve computed from "
            "empty columns."
        )
    dead = int((~selected.any(axis=0)).sum())
    if dead > selected.shape[1] // 2:
        raise SystemExit(
            f"{dead} of {selected.shape[1]} features this gate reads are 0.0 in "
            "every row. That is not a sparse feature set, it is a block the "
            "rows never populated. Pick the arm that matches these rows."
        )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True, help="a locomo_eval rows JSONL")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=REPO / "artefacts" / "locomo_gate_posthoc.json")
    ap.add_argument(
        "--target-risk", type=float, default=None,
        help=f"defaults to the pinned {gpins.TARGET_RISK}",
    )
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"no gate checkpoint at {args.checkpoint}.\n"
            "Train and persist one first:\n"
            "  python scripts/phase8_gate.py\n"
            "(seeded, minutes of CPU; it now writes phase8_gate.pt beside its "
            "artefact)"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase8_gate", REPO / "scripts" / "phase8_gate.py"
    )
    gate_mod = importlib.util.module_from_spec(spec)
    sys.modules["phase8_gate"] = gate_mod
    spec.loader.exec_module(gate_mod)

    model, mask, blob = gate_mod.load_gate(args.checkpoint)
    rows = load_rows(args.rows)
    check_blocks(rows, blob)
    x = feature_matrix(rows, list(blob["feature_names"]))
    check_selected_columns_are_populated(x, mask, blob)
    scores, probability, score_diag = gate_scores(model, x, mask)

    # Answerable = 1. The corpus's own annotation, not a derived judgement.
    labels = np.asarray([0.0 if r.get("adversarial") else 1.0 for r in rows])
    prevalence = float(gpins.EVAL_PREVALENCES["locomo"])
    target = float(gpins.TARGET_RISK if args.target_risk is None else args.target_risk)

    observed_prevalence = float((labels < 0.5).mean())
    weights = reweight(labels, prevalence)
    curve = risk_coverage(scores, labels, weights)

    # -- the transferred operating point (the honest number) ------------------
    transferred = float(blob["threshold"]["threshold"]) if isinstance(
        blob.get("threshold"), dict
    ) else float(blob.get("threshold") or 0.5)
    # `selective_metrics` takes no weights: the operating point is measured on
    # the rows as they are, and prevalence reweighting applies to the curve and
    # AURC. Reported side by side rather than silently mixed.
    at_transferred = selective_metrics(probability, labels, transferred)
    reachable = bool(probability.max() >= transferred)

    # -- the oracle, computed and labelled as unquotable ---------------------
    oracle = choose_threshold(
        scores, labels, target_risk=target, prevalence=prevalence
    )

    def abstention_at(threshold: float, on=None) -> dict:
        """Adversarial abstention accuracy at one operating point.

        The safety secondary: AURC rewards ranking, and a gate that abstains on
        everything has no risk at zero coverage. This is the number that makes
        over-abstention visible.
        """
        answered = (probability if on is None else on) >= threshold
        adversarial = labels < 0.5
        declined = adversarial & ~answered
        answerable_declined = (~adversarial) & ~answered
        return {
            "threshold": float(threshold),
            "adversarial_n": int(adversarial.sum()),
            "adversarial_abstained": int(declined.sum()),
            "adversarial_abstention_accuracy": (
                float(declined.sum() / adversarial.sum()) if adversarial.sum() else None
            ),
            "answerable_wrongly_abstained": int(answerable_declined.sum()),
            "false_abstention_rate": (
                float(answerable_declined.sum() / (~adversarial).sum())
                if (~adversarial).sum() else None
            ),
            "raw_coverage": float(answered.mean()),
        }

    report = {
        "what_this_is": (
            "the Phase-8 answerability gate, trained on MuSiQue contrast pairs, "
            "applied post hoc to a finished LoCoMo run's stored gate features. "
            "CPU only: no reader ran and no question was re-answered."
        ),
        "framing": (
            "selective prediction — risk at a coverage level. [EVIDENCE] "
            "Geifman & El-Yaniv, NeurIPS 2017."
        ),
        "leak_rule": (
            "the operating threshold is the checkpoint's, chosen on MuSiQue dev "
            "and transferred unchanged. It is NEVER re-chosen on LoCoMo's own "
            "curve. `oracle_threshold_leaked_do_not_report` is computed so the "
            "transfer gap is visible as a diagnostic, and is named so it cannot "
            "be quoted by accident."
        ),
        "rows": str(args.rows.name),
        "checkpoint": {
            "path": str(args.checkpoint.name),
            "arm": blob.get("arm"),
            "model": blob.get("model"),
            "seed": blob.get("seed"),
            "features_used": int(mask.sum()),
            "trained_on": blob.get("trained_on"),
            "stage_g_fingerprint": blob.get("stage_g_fingerprint"),
            "was_smoke": bool(blob.get("smoke")),
        },
        "questions": len(rows),
        "prevalence": {
            "observed_unanswerable": observed_prevalence,
            "reweighted_to": prevalence,
            "source": "EVAL_PREVALENCES['locomo'] = 446/1986",
            "note": (
                "reweighting, not resampling: the same predictions are reused so "
                "the curves differ only by the prevalence assumption"
            ),
        },
        "separation": {
            "auroc": auroc(scores, labels),
            "chance": 0.5,
            "reading": (
                "threshold-free and leak-free: the share of (answerable, "
                "adversarial) pairs the gate orders correctly. This is the "
                "honest headline for a transfer claim, because no choice of "
                "operating point can flatter it."
            ),
        },
        "scoring": score_diag,
        "aurc": {
            "value": float(aurc(curve)),
            "reweighted": True,
            "reading": (
                "lower is better; needs no threshold, so it commits to no "
                "operating point and carries no leak. Not comparable with any "
                "published AURC — the risk definition is this project's."
            ),
        },
        "risk_coverage_curve": {
            "coverage": [float(v) for v in curve["coverage"]],
            "risk": [float(v) for v in curve["risk"]],
            "thresholds": [float(v) for v in curve.get("thresholds", [])],
        },
        "operating_point_transferred": {
            **at_transferred,
            "threshold_source": "MuSiQue dev, via the checkpoint",
            "abstention": abstention_at(transferred),
        },
        "coverage_at_target_risk": {
            "target_risk": target,
            "selective_risk_at_transferred": at_transferred.get("selective_risk"),
            "coverage_at_transferred": at_transferred.get("coverage"),
            "transferred_threshold_is_reachable": reachable,
            "met_by_transferred_threshold": bool(
                reachable
                and at_transferred.get("selective_risk") is not None
                and at_transferred["selective_risk"] <= target
            ),
            "reading": (
                "whether the TRANSFERRED threshold holds the risk budget on "
                "LoCoMo. If it does not, that is the transfer result — not a "
                "reason to move the threshold."
            ),
        },
        "oracle_threshold_leaked_do_not_report": {
            **oracle,
            "abstention": abstention_at(float(oracle.get("threshold", 0.5))),
            "why_it_is_here": (
                "the gap between this and the transferred point IS the "
                "Wikipedia-to-conversation transfer cost, which is a declared "
                "open claim (CLAUDE.md §7). Reporting the oracle as the gate's "
                "performance would hide exactly that cost."
            ),
        },
        "definitions": {
            "risk": "error rate among ANSWERED questions at a coverage level",
            "coverage": "share of questions answered, reweighted to prevalence",
            "aurc": "area under the risk-coverage curve; lower is better",
            "adversarial_abstention_accuracy": (
                "share of the 446 adversarial questions the gate declines — the "
                "safety secondary, because AURC alone rewards a gate that "
                "abstains on everything"
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"  questions {len(rows)}  features {int(mask.sum())}  arm {blob.get('arm')}")
    print(f"  AURC (reweighted to {prevalence}): {report['aurc']['value']:.4f}")
    print(f"  AUROC (threshold-free separation): {report['separation']['auroc']:.4f}"
          "   [0.5 = chance]")
    ab = report["operating_point_transferred"]["abstention"]
    if not reachable:
        print(f"  transferred threshold {transferred:.4f} is UNREACHABLE: the "
              f"highest probability on this corpus is {probability.max():.3e}.")
        print("  Coverage is 0 by construction -- that is the transfer result,")
        print("  not a reason to move the threshold.")
    print(f"  transferred threshold {transferred:.4f}")
    print(f"    adversarial abstention {ab['adversarial_abstention_accuracy']}")
    print(f"    false abstention       {ab['false_abstention_rate']}")
    print("  NOTE: the oracle threshold in this artefact is a diagnostic and is")
    print("  named `leaked_do_not_report`. Quote the transferred point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
