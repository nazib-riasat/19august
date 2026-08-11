"""Phase-3 step 6 — the calibration gate. **Nothing past step 6 runs until this
has been written into `GRAFT_PHASE3_BUILD.md` §6.**

The order is decision 4's, and each step is there because a review round found
the previous order wrong:

.. code-block:: text

    (a) beta eligibility        which candidates are admissible at all
                                (Phase-2 decision 19 — eligibility precedes
                                selection, so an eligible winner cannot then
                                fail re-validation)
    per rung i of the ladder:
    (b) N_i from ceiling c_i    a throughput measurement, on the tuning suite
    (c) beta sweep at N_i       L5, 3 seeds, argmin mean exact TV over the
                                ELIGIBLE candidates (Phase-2 decision 22)
    (d) sanity check            L4 and L5 mean exact TV <= decision 6's 0.10
    pass => adopt; fail => next rung; fail at the last => GATE 2 INCONCLUSIVE

**Why (b) precedes (c).** Phase-2 decision 22 fixes β as the candidate
minimising mean exact TV *at the same fixed trajectory budget the Gate-2 primary
comparison uses* — the sweep needs ``N``. Freezing β first leaves that budget
undefined at the moment it is required. There is no circularity: ``N`` is a
wall-clock quantity and rollout cost does not depend on β, so ``N`` is fixed at
the current default β and the sweep then runs at that ``N``.

**Why the whole gate runs on the tuning suite.** A sanity check that can raise
``N`` would otherwise select the primary budget using main-suite results — the
instances every arm is scored on.

**Failure is `inconclusive`, never a negative verdict on C3** (criterion 12). A
null result that cannot distinguish "no effect" from "no budget" is the worst
outcome available, and this script is what stops one being reported as the
other.

Usage::

    python scripts/phase3_calibrate.py --out artefacts/phase3_calibration.json
    python scripts/phase3_calibrate.py --rungs 0.02 0.04 0.08 --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graft.setgen.learners import build_arm                       # noqa: E402
from graft.setgen.trainer import SEEDS, Environment, Trainer, TrainSpec  # noqa: E402
from graft.synth.enumerate import BandViolation                   # noqa: E402
from graft.synth.lattice import benchmark_suite, tuning_suite     # noqa: E402

#: Phase-2 decision 19, predeclared and unchanged.  The *grid* does not move;
#: what moves is which of its members are eligible.
BETA_CANDIDATES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

#: Decision 6 — signed **before** calibration, because a threshold written after
#: the run it judges is not a threshold.  **[ANALYSIS]** engineering, not paper.
SANITY_TV = 0.10

#: Decision 5's ladder: c0 = 1 h, rungs 1 -> 2 -> 4, at most two doublings.
DEFAULT_RUNGS: tuple[float, ...] = (3600.0, 7200.0, 14400.0)

#: Throughput is measured, not guessed: rollout cost depends on the instance
#: sizes and on the machine, and both change.
PILOT_TRAJECTORIES = 2_000


# --------------------------------------------------------------------------
# (a) eligibility
# --------------------------------------------------------------------------


def beta_eligibility(candidates: Sequence[float] = BETA_CANDIDATES) -> dict[str, Any]:
    """Which candidates the **main** suite's bands admit (Phase-2 decision 19).

    Reads no learner and no training run: it is a property of ``p*``, which is a
    function of the environment and β alone. Decision 10 already required this
    measurement on this suite; what decision 19 changed is *when* it happens.
    """
    from graft.synth.exact import target_distribution

    rows = []
    for instance in benchmark_suite():
        target = target_distribution(instance, instance.cfg)
        for beta in candidates:
            try:
                shifted = target.at_beta(beta)
                profile = shifted.validate_bands("main")
                rows.append({"beta": beta, "ok": True,
                             "neither": profile["mode_mass"]["neither"]})
            except (BandViolation, ValueError) as exc:
                rows.append({"beta": beta, "ok": False, "reason": str(exc)[:200]})

    eligible = sorted(
        {r["beta"] for r in rows if r["ok"]}
        - {r["beta"] for r in rows if not r["ok"]}
    )
    return {
        "candidates": list(candidates),
        "eligible": eligible,
        "per_instance": rows,
        "note": (
            "Eligibility precedes selection. If no candidate is eligible the "
            "environment cannot be built at any declared beta, which is a "
            "regeneration under GRAFT_PHASE2_BUILD.md 6b -- not a shrug."
        ),
    }


# --------------------------------------------------------------------------
# (b) N from the ceiling
# --------------------------------------------------------------------------


def measure_throughput(envs: Sequence[Environment], spec: TrainSpec) -> float:
    """Trajectories per second for **L5**, the costliest calibration arm.

    L5 rather than L4 because ``N`` must be affordable for the arm that pays
    most for it among the two that exist at this point in the build order;
    sizing on L4 would set a budget L5 cannot spend inside the ceiling.
    """
    pilot = spec.replace(n_trajectories=PILOT_TRAJECTORIES, checkpoints=1)
    trainer = Trainer(build_arm("l5_subtb"), list(envs), pilot)
    started = time.perf_counter()
    trainer.train()
    return PILOT_TRAJECTORIES / max(time.perf_counter() - started, 1e-9)


def budget_for(ceiling_s: float, rate: float) -> int:
    """``N`` at a wall-clock ceiling, rounded down to a multiple of the batch.

    **Throughput only.** Exit criterion 12 requires the write-up to say ``N`` was
    set by a bounded adaptive rule reading L4/L5 — this half of it reads no TV at
    all; the adaptivity is the ladder, which escalates on decision 6's threshold
    and is bounded at two doublings.
    """
    return max(1, int(ceiling_s * rate))


# --------------------------------------------------------------------------
# (c) the beta sweep
# --------------------------------------------------------------------------


def retarget(env: Environment, beta: float) -> None:
    """Move one environment to a new β **without re-enumerating it**.

    Phase-2 G7 again: ``p*`` depends on β only through ``R = exp(β·U)`` and ``U``
    is independent of β, so a sweep is an ``exp`` over a cached vector rather
    than thousands of recomputed utilities. ``at_beta`` re-runs the
    ``r_fail_margin`` assertion on the way through, which is Phase 0's protection
    against a sweep quietly promoting ``FAIL`` into a competitive terminal — the
    open item ``CLAUDE.md`` §7 flags for exactly this moment.
    """
    env.target = env.target.at_beta(beta)
    env.log_reward[env.graph.terminal_ix] = beta * env.target.u
    values = env.log_reward[env.graph.terminal_ix]
    env.log_r_range = float(values.max() - values.min()) if values.size else 0.0


def beta_sweep(
    envs: Sequence[Environment], eligible: Sequence[float], spec: TrainSpec
) -> dict[str, Any]:
    """Phase-2 decision 22, executable: L5, 3 seeds, argmin mean exact TV.

    **On L5, never on L7.** Selecting β on the proposed method would tune the
    reward in favour of Contribution 3 — the reward would then be part of what
    Gate 2 tests, rather than the fixed thing it tests against.

    Ties go to the **smaller** β: a lower temperature is the flatter target and
    the less committal choice, and breaking a tie toward the sharper one would be
    a silent preference for whichever candidate concentrates mass.
    """
    results: dict[str, Any] = {}
    for beta in eligible:
        for env in envs:
            retarget(env, beta)
        per_seed = [
            Trainer(build_arm("l5_subtb"), list(envs), spec.replace(seed=seed))
            .train()
            .final_tv
            for seed in SEEDS
        ]
        results[str(beta)] = {
            "mean_tv": float(np.mean(per_seed)),
            "per_seed_tv": per_seed,
        }

    winner = min(eligible, key=lambda b: (results[str(b)]["mean_tv"], b))
    return {"per_beta": results, "winner": float(winner)}


# --------------------------------------------------------------------------
# (d) the sanity check
# --------------------------------------------------------------------------


def sanity_check(envs: Sequence[Environment], spec: TrainSpec) -> dict[str, Any]:
    """Decision 6: mean exact TV <= 0.10 on the tuning suite, for **both** L4 and L5."""
    rows: dict[str, Any] = {}
    for name in ("l4_tb", "l5_subtb"):
        per_seed = [
            Trainer(build_arm(name), list(envs), spec.replace(seed=seed)).train().final_tv
            for seed in SEEDS
        ]
        rows[name] = {"mean_tv": float(np.mean(per_seed)), "per_seed_tv": per_seed}
    passed = all(row["mean_tv"] <= SANITY_TV for row in rows.values())
    return {"threshold": SANITY_TV, "arms": rows, "passed": passed}


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


def calibrate(rungs: Sequence[float], quick: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {"rungs": [], "adopted": None, "verdict": "inconclusive"}

    record["eligibility"] = beta_eligibility()
    eligible = record["eligibility"]["eligible"]
    if not eligible:
        record["verdict"] = "regenerate"
        return record

    instances = list(tuning_suite())
    if quick:
        instances = instances[:2]
    envs = [Environment(inst) for inst in instances]

    base = TrainSpec(n_trajectories=PILOT_TRAJECTORIES)
    rate = measure_throughput(envs, base)
    record["throughput_traj_per_s"] = rate

    for i, ceiling in enumerate(rungs):
        n = budget_for(ceiling, rate)
        spec = base.replace(n_trajectories=n)
        rung: dict[str, Any] = {"rung": i, "ceiling_s": ceiling, "N": n}

        rung["beta_sweep"] = beta_sweep(envs, eligible, spec)
        beta = rung["beta_sweep"]["winner"]
        for env in envs:
            retarget(env, beta)
        rung["beta"] = beta
        rung["sanity"] = sanity_check(envs, spec)
        record["rungs"].append(rung)          # every rung tried is recorded

        if rung["sanity"]["passed"]:
            record["adopted"] = {"N": n, "beta": beta, "rung": i, "ceiling_s": ceiling}
            record["verdict"] = "adopted"
            break

    if record["verdict"] != "adopted":
        record["note"] = (
            "Decision 6's threshold failed at the final rung. Gate 2 is recorded "
            "INCONCLUSIVE, never as a negative result for Contribution 3 "
            "(criterion 12). A null result that cannot distinguish 'no effect' "
            "from 'no budget' is the worst outcome available."
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artefacts/phase3_calibration.json"))
    parser.add_argument("--rungs", type=float, nargs="*", default=list(DEFAULT_RUNGS),
                        help="wall-clock ceilings in seconds (decision 5: 3600 7200 14400)")
    parser.add_argument("--quick", action="store_true",
                        help="two tuning instances, for wiring checks only -- NOT a calibration")
    args = parser.parse_args()

    record = calibrate(args.rungs, quick=args.quick)
    record["quick"] = args.quick
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), "utf-8")

    print(f"eligible beta: {record['eligibility']['eligible']}")
    print(f"throughput:    {record.get('throughput_traj_per_s', math.nan):.0f} traj/s")
    for rung in record["rungs"]:
        print(
            f"rung {rung['rung']}  N={rung['N']:>9}  beta={rung['beta']}  "
            f"sanity={'PASS' if rung['sanity']['passed'] else 'fail'}"
        )
    print(f"verdict: {record['verdict']}  -> {args.out}")
    if args.quick:
        print("\nQUICK MODE: a wiring check, not a calibration. Do not write these "
              "values into section 6.")
    return 0 if record["verdict"] == "adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
