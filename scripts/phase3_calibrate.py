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

from graft.setgen.learners import (                               # noqa: E402
    FLOW_FAMILY,
    SUPERVISED_FAMILY,
    build_arm,
)
from graft.setgen.trainer import (                                # noqa: E402
    DECISION5_RUNGS,
    SEEDS,
    Environment,
    Trainer,
    TrainSpec,
)
from graft.synth.enumerate import BandViolation                   # noqa: E402
from graft.synth.lattice import benchmark_suite, tuning_suite     # noqa: E402

#: Phase-2 decision 19, predeclared and unchanged.  The *grid* does not move;
#: what moves is which of its members are eligible.
BETA_CANDIDATES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

#: Decision 6 — signed **before** calibration, because a threshold written after
#: the run it judges is not a threshold.  **[ANALYSIS]** engineering, not paper.
SANITY_TV = 0.10

#: Decision 5's ladder, imported rather than restated: `gate2` refuses an
#: adopted ceiling that is not one of these, so the two must agree.
DEFAULT_RUNGS = DECISION5_RUNGS

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


def measure_throughput(
    envs: Sequence[Environment], spec: TrainSpec
) -> tuple[float, float, dict[str, float]]:
    """``(slowest rate, seconds per evaluation, per-arm rates)`` over **every arm**.

    **``N`` is spent by all nine arms, so it has to be affordable for the
    slowest, not the cheapest.** An earlier version piloted L5 alone. Measured on
    the tuning suite, L5 runs at ~2,700 trajectories/s and the LED arms at
    ~900 — a **2.9× gap**, because decision 23a's ``N = 8`` decomposition
    iterations mean L6/L7/L7b take eight potential steps per policy step. An
    ``N`` sized on L5 at a 1 h ceiling therefore buys L7 a **2.9 h** run against
    the ceiling that produced it, and decision 5's GPU-day table understates the
    matrix by the same factor. Nothing could see it: the rung's own guard reads
    ``beta_sweep`` and ``sanity_check``, which run L4 and L5 exclusively, so the
    ladder passed its ceiling check while the matrix overran.

    **This does not breach decision 4's "reads L4/L5 only".** That rule exists so
    the primary budget is not selected on the proposed method's *results* —
    on how well L7 scores. A rate in trajectories per second is a property of the
    machine and the architecture, not of learning: it is available before a
    single gradient step means anything, and it cannot move in L7's favour. β
    selection and the decision-6 sanity check still read L4 and L5 alone, which
    is where the contamination risk actually lives.

    Capacity-matched arms are piloted, not default-width ones: the matrix runs a
    widened L6 and GAFlowNet, and their width is what their rate depends on.

    **The two costs are separated because the real run pays them in different
    proportions.** A rung run evaluates exact TV at ``checkpoints + 1`` points;
    the pilot evaluates at 2. Timing the pilot as a whole and dividing by its
    trajectory count therefore folds ~2 evaluations into a per-trajectory rate
    and then bills the rung for ~51 — the rate comes out too high, ``N`` too
    large, and every rung silently overruns the ceiling it was derived from.
    """
    from graft.setgen.gate2 import capacity_matched_arm

    pilot = spec.replace(n_trajectories=PILOT_TRAJECTORIES, checkpoints=1)
    rates: dict[str, float] = {}
    eval_cost = 0.0

    for name in list(FLOW_FAMILY) + list(SUPERVISED_FAMILY):
        arm = build_arm(name)
        if name in ("l6_led", "gaflownet"):
            arm, _ = capacity_matched_arm(name, "l7_checker_led", envs, pilot)
        trainer = Trainer(arm, list(envs), pilot)

        if not eval_cost:
            started = time.perf_counter()
            trainer.exact_tv()
            eval_cost = time.perf_counter() - started

        started = time.perf_counter()
        trainer.train()
        # The pilot's own two evaluations are not training time.
        training = max(time.perf_counter() - started - 2.0 * eval_cost, 1e-9)
        rates[name] = PILOT_TRAJECTORIES / training

    return min(rates.values()), eval_cost, rates


def budget_for(
    ceiling_s: float, rate: float, *, batch: int, eval_cost: float, checkpoints: int
) -> int:
    """``N`` at a wall-clock ceiling, **rounded down to a multiple of the batch**.

    .. code-block:: text

        N = floor_to_batch( (ceiling − (checkpoints + 2)·eval_cost) · rate )

    The docstring here used to promise the batch rounding and the code did not
    do it, and the evaluation term was missing entirely. Both matter for the same
    reason: ``N`` is the primary budget of fix F12, identical across every arm
    (criterion 11), so it should be a number the trainer can spend exactly rather
    than one it approaches with a short final batch.

    ``checkpoints + 2`` because the trainer evaluates once at trajectory 0 — the
    random-initialisation baseline — then at each of the ``C`` checkpoints, and
    once more at the end whenever the last checkpoint did not land exactly on
    ``N`` (decision 2's own arithmetic: up to 52 evaluations at ``C = 50``; this
    line billed 51 until 13 Aug 2026 — one ``eval_cost`` of slack, sub-second,
    but the budget should match what the code does).

    **Throughput only.** Exit criterion 12 requires the write-up to say ``N`` was
    set by a bounded adaptive rule reading L4/L5 — this half of it reads no TV at
    all; the adaptivity is the ladder, which escalates on decision 6's threshold
    and is bounded at two doublings.
    """
    usable = ceiling_s - (checkpoints + 2) * eval_cost
    if usable <= 0.0:
        raise ValueError(
            f"a ceiling of {ceiling_s:.0f}s cannot pay for {checkpoints + 1} "
            f"evaluations at {eval_cost:.3f}s each; there is no training time "
            "left, so the rung is not a budget for anything"
        )
    n = int(usable * rate)
    return max(batch, n - n % batch)


# --------------------------------------------------------------------------
# (c) the beta sweep
# --------------------------------------------------------------------------


def retarget(env: Environment, beta: float) -> None:
    """Thin alias for :meth:`Environment.at_beta`, kept for this script's prose.

    The implementation moved into the package: it is the only correct way to move
    an environment's β, and leaving it here meant the Gate-2 runner had to import
    from ``scripts/`` or reimplement three coupled updates.
    """
    env.at_beta(beta)


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
    slowest = 0.0
    for beta in eligible:
        for env in envs:
            retarget(env, beta)
        logs = [
            Trainer(build_arm("l5_subtb"), list(envs), spec.replace(seed=seed)).train()
            for seed in SEEDS
        ]
        per_seed = [log.final_tv for log in logs]
        # Per run, not averaged: the ceiling is a *per-run* budget (decision 5),
        # and one slow seed can hide inside a mean that clears it.
        slowest = max([slowest] + [log.wall_clock for log in logs])
        results[str(beta)] = {
            "mean_tv": float(np.mean(per_seed)),
            "per_seed_tv": per_seed,
            "max_run_s": max(log.wall_clock for log in logs),
        }

    winner = min(eligible, key=lambda b: (results[str(b)]["mean_tv"], b))
    return {"per_beta": results, "winner": float(winner), "max_run_s": slowest}


# --------------------------------------------------------------------------
# (d) the sanity check
# --------------------------------------------------------------------------


def sanity_check(envs: Sequence[Environment], spec: TrainSpec) -> dict[str, Any]:
    """Decision 6: mean exact TV <= 0.10 on the tuning suite, for **both** L4 and L5."""
    rows: dict[str, Any] = {}
    slowest = 0.0
    for name in ("l4_tb", "l5_subtb"):
        logs = [
            Trainer(build_arm(name), list(envs), spec.replace(seed=seed)).train()
            for seed in SEEDS
        ]
        per_seed = [log.final_tv for log in logs]
        slowest = max([slowest] + [log.wall_clock for log in logs])
        rows[name] = {
            "mean_tv": float(np.mean(per_seed)),
            "per_seed_tv": per_seed,
            "max_run_s": max(log.wall_clock for log in logs),
        }
    passed = all(row["mean_tv"] <= SANITY_TV for row in rows.values())
    return {
        "threshold": SANITY_TV, "arms": rows, "passed": passed, "max_run_s": slowest
    }


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


def calibrate(
    rungs: Sequence[float], quick: bool = False, device: str = "cpu"
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rungs": [], "adopted": None, "verdict": "inconclusive", "device": device,
    }

    record["eligibility"] = beta_eligibility()
    eligible = record["eligibility"]["eligible"]
    if not eligible:
        record["verdict"] = "regenerate"
        return record

    instances = list(tuning_suite())
    if quick:
        instances = instances[:2]
    envs = [Environment(inst) for inst in instances]

    # `--quick` shrinks the checkpoint count along with everything else. The
    # ceiling has to pay for `checkpoints + 1` evaluations before it buys a
    # single trajectory, so a three-second wiring run against the real C = 50
    # has no training time left and `budget_for` rejects it — correctly, but it
    # would make the wiring check unrunnable. A quick run is explicitly not a
    # calibration and its output may not be written into §6.
    base = TrainSpec(
        n_trajectories=PILOT_TRAJECTORIES, checkpoints=2 if quick else 50,
        device=device,
    )
    rate, eval_cost, per_arm = measure_throughput(envs, base)
    print(f"throughput pilot: slowest {min(per_arm, key=per_arm.__getitem__)} at "
          f"{rate:.0f} traj/s, eval {eval_cost:.3f}s", flush=True)
    record["throughput_traj_per_s"] = rate
    record["throughput_per_arm"] = per_arm
    record["slowest_arm"] = min(per_arm, key=per_arm.__getitem__)
    record["evaluation_cost_s"] = eval_cost

    for i, ceiling in enumerate(rungs):
        n = budget_for(
            ceiling, rate, batch=base.batch_size, eval_cost=eval_cost,
            checkpoints=base.checkpoints,
        )
        spec = base.replace(n_trajectories=n)
        rung: dict[str, Any] = {"rung": i, "ceiling_s": ceiling, "N": n}

        print(f"  rung {i}: ceiling {ceiling:.0f}s -> N={n:,}; sweeping beta over "
              f"{eligible} on L5 x {len(SEEDS)} seeds...", flush=True)
        rung["beta_sweep"] = beta_sweep(envs, eligible, spec)
        beta = rung["beta_sweep"]["winner"]
        for env in envs:
            retarget(env, beta)
        rung["beta"] = beta
        print(f"  rung {i}: beta={beta}; decision-6 sanity on L4/L5...", flush=True)
        rung["sanity"] = sanity_check(envs, spec)

        # **The slowest single run, not the mean of them.** Decision 5's ceiling
        # is a per-run budget, so averaging over the sweep's runs lets one slow
        # seed clear a bar it individually broke.
        #
        # The ceiling is still deliberately **not** passed to the trainer as
        # `wall_clock_ceiling`: a truncated run spends fewer than `N`
        # trajectories and criterion 11 requires `N` identical across every arm,
        # so a hard stop would silently break the one thing the budget exists to
        # hold fixed. The run finishes, and an overrun ends the ladder instead.
        #
        # **Measured and predicted, because the measured half is blind to six of
        # the nine arms.** `beta_sweep` and `sanity_check` run L4 and L5 only —
        # by design, that is decision 4's rule — so on their own they would
        # certify a ceiling the LED arms then break by ~3x. The predicted term
        # closes that with the per-arm rates the pilot already measured, at no
        # extra training: `N / rate_slowest + (C+1)·eval`.
        measured = max(
            rung["beta_sweep"]["max_run_s"], rung["sanity"]["max_run_s"]
        )
        predicted = n / rate + (base.checkpoints + 1) * eval_cost
        rung["measured_s_per_run"] = measured
        rung["predicted_slowest_arm_s"] = predicted
        rung["predicted_for"] = record["slowest_arm"]
        rung["max_s_per_run"] = max(measured, predicted)
        rung["ceiling_respected"] = rung["max_s_per_run"] <= ceiling
        print(f"  rung {i}: sanity "
              f"{'PASS' if rung['sanity']['passed'] else 'fail'}, "
              f"slowest run {rung['max_s_per_run']:.0f}s vs ceiling {ceiling:.0f}s",
              flush=True)
        record["rungs"].append(rung)          # every rung tried is recorded

        if not rung["ceiling_respected"]:
            # Never adopt a rung that cost more than the ceiling it was derived
            # from: `N` came from a throughput estimate, and an overrun says that
            # estimate was wrong, so the budget this rung reports is not the
            # budget it spent. Escalating is worse -- the next rung's ceiling is
            # larger -- so the ladder stops here.
            record["verdict"] = "over_ceiling"
            record["note"] = (
                f"Rung {i} took {rung['max_s_per_run']:.0f}s for its slowest run "
                f"against a ceiling of {ceiling:.0f}s, so the throughput estimate "
                "that produced N was wrong and N is not the budget this rung "
                "spent. Not adopted, and the ladder stops rather than escalating "
                "into a larger ceiling. Re-run: the pilot's rate and evaluation "
                "cost are recorded above and are what need correcting."
            )
            break

        if rung["sanity"]["passed"]:
            record["adopted"] = {"N": n, "beta": beta, "rung": i, "ceiling_s": ceiling}
            record["verdict"] = "adopted"
            break

    if record["verdict"] == "inconclusive":
        record["note"] = (
            "Decision 6's threshold failed at the final rung. Gate 2 is recorded "
            "INCONCLUSIVE, never as a negative result for Contribution 3 "
            "(criterion 12). A null result that cannot distinguish 'no effect' "
            "from 'no budget' is the worst outcome available."
        )
    return record


def main() -> int:
    # These docstrings carry beta, section signs and arrows, and Windows
    # consoles default to cp1252 -- `--help` died on the description before
    # printing a word of it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artefacts/phase3_calibration.json"))
    parser.add_argument("--rungs", type=float, nargs="*", default=list(DEFAULT_RUNGS),
                        help="wall-clock ceilings in seconds (decision 5: 3600 7200 14400)")
    parser.add_argument("--quick", action="store_true",
                        help="two tuning instances, for wiring checks only -- NOT a calibration")
    parser.add_argument("--device", default="cpu",
                        help="torch device. Phase 3 is a CPU workload by default: the "
                             "networks are ~52k-parameter MLPs and the sampler is pure "
                             "numpy, so every batch would round-trip the bus. Decision "
                             "5's ceiling is therefore wall-clock ON THE MACHINE THAT "
                             "RAN IT, and the flag exists so that machine is a choice "
                             "rather than a default nobody noticed.")
    args = parser.parse_args()

    record = calibrate(args.rungs, quick=args.quick, device=args.device)
    record["quick"] = args.quick
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), "utf-8")

    print(f"eligible beta: {record['eligibility']['eligible']}")
    print(
        f"throughput:    {record.get('throughput_traj_per_s', math.nan):.0f} traj/s "
        f"(slowest arm: {record.get('slowest_arm')})"
    )
    for name, r in sorted((record.get("throughput_per_arm") or {}).items(),
                          key=lambda kv: kv[1]):
        print(f"    {name:<18}{r:8.0f} traj/s")
    for rung in record["rungs"]:
        print(
            f"rung {rung['rung']}  N={rung['N']:>9}  beta={rung['beta']}  "
            f"sanity={'PASS' if rung['sanity']['passed'] else 'fail'}  "
            f"{rung['max_s_per_run']:.0f}s slowest run"
            f"{'' if rung['ceiling_respected'] else '  CEILING OVERRUN — NOT ADOPTED'}"
        )
    print(f"verdict: {record['verdict']}  -> {args.out}")
    if args.quick:
        print("\nQUICK MODE: a wiring check, not a calibration. Do not write these "
              "values into section 6.")
    return 0 if record["verdict"] == "adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
