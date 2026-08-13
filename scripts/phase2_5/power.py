"""Phase 2.5 gap G1 — the go/no-go's denominator, derived rather than guessed.

Two numbers per decoder, with every assumption printed beside its consequence:

* ``n_test`` — from a paired power argument on the **primary** Stage-B metric
  (v1.2 §6.4: D1's end-to-end score; D2's macro-F1 via its rarest class).
  McNemar-style paired-proportion test: n ≈ (z_a·√ψ + z_b·√(ψ − δ²))² / δ²,
  where ψ is the disagreement rate between the two compared systems and δ the
  detectable accuracy difference.  **[ANALYSIS]** — the standard two-sided
  α = 0.05 / power 0.8 frame; ψ and δ are assumptions, so n_test is reported
  as a RANGE over them, per the plan ("a stated range beats a false point
  estimate").
* ``n_train`` — stated as an assumption with its basis (comparable published
  supervision scales), never derived: the plan says so explicitly.

Usage:  python scripts/phase2_5/power.py
Writes: data/phase2_5/power.json (and prints the table)
"""
from __future__ import annotations

import json
import math

from common import DATA

Z_ALPHA = 1.96  # two-sided 0.05
Z_BETA = 0.84   # power 0.80


def mcnemar_n(psi: float, delta: float) -> int:
    """Paired-proportion sample size (Connor's approximation)."""
    n = (Z_ALPHA * math.sqrt(psi) + Z_BETA * math.sqrt(psi - delta**2)) ** 2 / delta**2
    return math.ceil(n)


def main() -> None:
    # δ: the smallest end-to-end D1 accuracy difference that would change the
    # Gate-1 encoder decision.  ψ: how often two compared encoders disagree on
    # one item — unknowable before Gate 1, so both are ranged.
    deltas = (0.05, 0.08)
    psis = (0.10, 0.20, 0.30)
    d1_grid = {
        f"delta={d:.2f},psi={p:.2f}": mcnemar_n(p, d) for d in deltas for p in psis
    }

    # D2's binding constraint is its rarest class: macro-F1 over four classes
    # needs every class measurable.  With CONFLICT/SUPERSEDES at ~10% each of a
    # *deliberately over-sampled* test set, >= 30 rare-class items per class
    # (a conventional floor for a stable per-class F1) needs >= 300 pairs; at
    # 5% natural incidence it needs >= 600.
    d2_rows = {
        "rare_share=0.10, floor=30": math.ceil(30 / 0.10),
        "rare_share=0.05, floor=30": math.ceil(30 / 0.05),
    }

    out = {
        "d1": {
            "primary_metric": "end-to-end mention-resolution score (action AND id, v1.2 §6.4)",
            "test": "McNemar paired proportions, alpha=0.05 two-sided, power=0.8",
            "n_test_grid": d1_grid,
            "n_test_range": [min(d1_grid.values()), max(d1_grid.values())],
            "n_train_assumption": {
                "value": 1500,
                "basis": "[ANALYSIS] ConEL-2-scale conversational EL fine-tunes "
                         "candidate-ranking heads at 1-2k labelled mentions; the "
                         "encoder is pretrained and only heads are learned",
            },
        },
        "d2": {
            "primary_metric": "four-way macro-F1 (v1.2 §6.4)",
            "test": "rare-class floor: >= 30 items in the rarest class",
            "n_test_grid": d2_rows,
            "n_test_range": [min(d2_rows.values()), max(d2_rows.values())],
            "n_train_assumption": {
                "value": 1200,
                "basis": "[ANALYSIS] DialogRE supplies ~300-3,000 examples per "
                         "frequent relation; four mutually exclusive classes at "
                         "~300 each is the same order",
            },
        },
        "note": "n_test is per decoder and per split; the go/no-go compares "
                "(items/hour x hours/week x weeks) against n_test + n_train.",
    }
    (DATA / "power.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
