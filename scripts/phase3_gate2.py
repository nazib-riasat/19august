"""Phase-3 step 10 — the Gate-2 run. **Nothing here may run before step 6.**

``run_matrix`` existed and nothing called it but tests, so the last mile of the
phase — calibration record → retarget → suites → matrix → report — was never
composed. That is not a cosmetic gap: the composition is where the ordering
constraints live, and each one was a thing an operator had to remember rather
than a thing the code did.

.. code-block:: text

    read artefacts/phase3_calibration.json    N and beta, adopted at step 6
    build the frozen main + probe suites      decision 8, criterion 23
    retarget every environment to beta        Environment.at_beta
    run_matrix(..., calibration=record)       admissibility checks all of it
    write the report + per-arm checkpoints    section 8 requirement 1

**The admissibility gate is the point, not this script.** ``run_matrix`` refuses
to emit a verdict unless the roster, the seeds, the suites and ``(N, β)`` all
match what the plan fixed, so a hand-assembled run cannot quietly become Gate 2.
This script's job is to assemble the one run that *is* admissible, and to fail
loudly rather than helpfully when the calibration record is missing.

Usage::

    python scripts/phase3_gate2.py                     # after step 6
    python scripts/phase3_gate2.py --dry-run           # what would run, no training
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graft.setgen.gate2 import format_table, run_matrix          # noqa: E402
from graft.setgen.learners import FLOW_FAMILY, SUPERVISED_FAMILY  # noqa: E402
from graft.setgen.trainer import SEEDS, Environment, TrainSpec    # noqa: E402
from graft.synth.lattice import benchmark_suite, probe_suite      # noqa: E402


def load_calibration(path: Path) -> dict:
    """Step 6's artefact, or a refusal that says what to do about it."""
    if not path.is_file():
        raise SystemExit(
            f"no calibration record at {path}.\n"
            "Step 6 has not run, so N and beta are not frozen and no L6/L7/"
            "GAFlowNet number may be quoted (GRAFT_PHASE3_BUILD.md section 6).\n"
            "    python scripts/phase3_calibrate.py --out " + str(path)
        )
    record = json.loads(path.read_text("utf-8"))
    if record.get("quick"):
        raise SystemExit(
            f"{path} is a --quick wiring check. Its own output says its values "
            "may not be written into section 6, and they may not size a matrix "
            "either."
        )
    if record.get("verdict") != "adopted" or not record.get("adopted"):
        raise SystemExit(
            f"{path} records verdict {record.get('verdict')!r}, so no N or beta "
            "was adopted. Gate 2 does not run on an unadopted budget; see "
            "criterion 12 for why 'inconclusive' is not 'proceed anyway'."
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
    parser.add_argument("--calibration", type=Path,
                        default=Path("artefacts/phase3_calibration.json"))
    parser.add_argument("--out", type=Path, default=Path("artefacts/gate2_report.json"))
    parser.add_argument("--checkpoints", type=Path, default=Path("artefacts/checkpoints"),
                        help="where per-(arm, seed) model weights are written")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run and stop before training")
    args = parser.parse_args()

    record = load_calibration(args.calibration)
    adopted = record["adopted"]
    n, beta = int(adopted["N"]), float(adopted["beta"])

    envs = [Environment(i) for i in benchmark_suite()]
    probes = [Environment(i) for i in probe_suite()]
    # Both suites move to the adopted beta. The probe is scored at the same beta
    # as the main suite or its TV is not comparable with anything.
    for env in list(envs) + list(probes):
        env.at_beta(beta)

    spec = TrainSpec(n_trajectories=n)
    arms = list(FLOW_FAMILY) + list(SUPERVISED_FAMILY)

    print(f"calibration  {args.calibration}  (rung {adopted.get('rung')})")
    print(f"N            {n:,}")
    print(f"beta         {beta}")
    print(f"arms         {len(arms)}  x  seeds {list(SEEDS)}  = {len(arms) * len(SEEDS)} runs")
    print(f"instances    {len(envs)} main, {len(probes)} probe (held out)")
    if args.dry_run:
        print("\n--dry-run: nothing trained.")
        return 0

    started = time.perf_counter()
    report = run_matrix(
        envs, spec, arms=arms, seeds=SEEDS, probe_envs=probes,
        calibration=record, checkpoint_dir=args.checkpoints,
    )
    report.save(args.out)

    print(f"\n{format_table(report)}\n")
    verdict = report.verdict
    print(f"outcome: {verdict['outcome']}")
    print(f"contribution_3_supported: {verdict['contribution_3_supported']}")
    if verdict["outcome"] == "inadmissible":
        print(f"  reason: {verdict['inadmissible_reason']}")
    elif verdict["outcome"] == "inconclusive":
        failed = [k for k, v in verdict["instrument"].items() if not v]
        print(f"  the instrument did not work: {failed}")
        print("  this is NOT a negative result for Contribution 3 (criterion 12).")
    print(f"\n{time.perf_counter() - started:.0f}s  ->  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
