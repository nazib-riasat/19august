"""Phase 9, build step 4: throughput per arm, capacity match, and ``N_real``.

    python scripts/phase9_measure.py --smoke                 # fixtures, no data
    python scripts/phase9_measure.py --examples 24           # real pools, stub embedder
    python scripts/phase9_measure.py --examples 24 --real-embedder

**This is the one measurement that must happen before any scored run**, and its
output is a *derived* value rather than a chosen one.  ``pins.BUDGET`` freezes the
ceiling (2 h per run) and the ladder (50k → 100k → 200k); ``N_real`` is the
largest rung the **slowest arm** completes inside the ceiling at its measured
rate.

**Why the slowest arm and not the mean, in one measured sentence.**  Phase 3
sized ``N`` on L5 at ~2,700 traj/s and spent it on the LED arms at ~900 traj/s,
buying L7 2.9 h against a 1 h ceiling — and the rung's own guard could not see
it, because that guard read only ``beta_sweep`` and ``sanity_check``, which run
L4 and L5 exclusively.  The same ~3× gap exists here for the same reason: the
LED arms run eight decomposition iterations per policy step, and L7/L7b
additionally evaluate ``Δd`` per legal action.

**A rate is not a learner result.**  It is a property of the machine and the
architecture, available before a gradient step means anything and unable to move
in the proposed method's favour, so measuring it before ``N`` is frozen is
`GRAFT_PHASE2_BUILD.md` §6b-clean.  That is the Phase-3 decision-4 argument,
reused deliberately and recorded in this phase's §6.

**Nothing here is a Gate-3 number.**  Losses are printed to show the arms train
without NaN at the reward floor; they are two-epoch smoke values on a handful of
examples and the artefact says so in its own body.
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
import torch  # noqa: E402

from graft.canonical import digest_of  # noqa: E402
from graft.config import load_config  # noqa: E402
from graft.runtime import deterministic_view, json_sanitize, run_manifest  # noqa: E402
from graft.setgen import pins  # noqa: E402
from graft.setgen.atomfeat import RealFeaturizer  # noqa: E402
from graft.setgen.gate2 import CAPACITY_SANITY_CEILING  # noqa: E402
from graft.setgen.learners import ARMS  # noqa: E402
from graft.setgen.policy import match_capacity  # noqa: E402
from graft.setgen.realenv import RealEnvironment, RealTrainer  # noqa: E402
from graft.setgen.trainer import Arm, Trainer, TrainSpec  # noqa: E402

OUT = REPO / "artefacts" / "phase9_measure.json"

HONESTY_STAMP: dict[str, str] = {
    "purpose": (
        "build step 4 only: per-arm throughput, capacity match at real dims, and "
        "the N_real derivation. NOT a Gate-3 result and not comparable to one."
    ),
    "losses": (
        "two-epoch smoke values on a handful of examples, printed to show every "
        "arm trains without a NaN at the reward floor. They carry no ranking."
    ),
    "beta": (
        "beta is a placeholder until Phase-3 step 6 freezes it. Throughput does "
        "not depend on it — a rate is a property of the machine and the "
        "architecture — so this measurement is valid now and a SCORED run is not."
    ),
    "stage": "Stage A (Wikipedia track). The conversational track waits on scope-c.",
}


def build_examples(
    count: int, *, smoke: bool, real_embedder: bool
) -> tuple[list[Any], dict[str, Any]]:
    """Real pools from the two Tier-1 corpora, or hand-built fixtures under ``--smoke``.

    Split evenly between corpora so the measured rate reflects the mix the
    training subset will actually have — MuSiQue's pools run ~60 atoms against
    2Wiki's ~30, and a rate measured on one alone would misprice the other.
    """
    if smoke:
        from graft.schemas import Obligations
        from graft.setgen.proofs import SourceDoc, build_example

        rng = np.random.default_rng(13)
        out = []
        for q in range(count):
            docs = [
                SourceDoc(f"p{i}", f"Q{q} document {i} concerning London and topic {i}.",
                          ("London",), is_gold=(i < 2))
                for i in range(10)
            ]
            out.append(
                build_example(
                    f"smoke{q}", docs,
                    Obligations(entity_anchor="London", scope=(f"smoke{q}",)),
                    {f"p{i}": 1.0 - 0.05 * i for i in range(10)},
                    embed=lambda t: np.stack([rng.normal(size=384) for _ in t]),
                )
            )
        return out, {"smoke": {"fixture": len(out)}}

    from graft.graphbuild.loaders import PHASE9_ROOT, load_split
    from graft.setgen.corpora import musique_ans, wiki2

    if real_embedder:
        from graft.graphbuild.embed import Embedder

        embedder: Any = Embedder(cache_dir=REPO / "artefacts" / "phase9" / "embed_cache")
        embedder.load()
    else:
        from graft.graphbuild.embed import StubEmbedder

        embedder = StubEmbedder(dim=384)

    # **`limit`, not a head slice** (16 Aug 2026 audit). This used to pass
    # `load_split(...)[:half]`, which pre-sliced the rows *before* the adapter
    # could stratify them -- defeating `corpora.stratified_sample`, which exists
    # for exactly this and which both adapters already call. Measured on the dev
    # splits: the first 12 MuSiQue-Ans rows are **100% 2-hop** against a true mix
    # of 51.8% / 31.4% / 16.7%, so the sample was not a sample. A11 recorded the
    # same failure for `train`; `dev` is sorted the same way.
    #
    # **The direction of the error was measured, and it was the opposite of the
    # obvious guess.** The intuition is that an all-2-hop slice is cheap and
    # would flatter the rate. Measured (5 reps, l7b_aux): head-slice 28.69 traj/s
    # against stratified 33.04 -- the wrong workload was *slower*, so the bias ran
    # in the safe direction and `N_real` was never at risk from it. Recorded
    # because a plausible mechanism that measurement contradicts is exactly what
    # `CLAUDE.md` §5 says to write down rather than quietly drop.
    half = max(1, count // 2)
    out: list[Any] = []
    strata: dict[str, Any] = {}
    for name, module in (("2wiki", wiki2), ("musique_ans", musique_ans)):
        rows = load_split(name, "dev", root=PHASE9_ROOT)
        examples, report = module.load_examples(
            "dev", rows=rows, limit=half, embedder=embedder
        )
        # Criterion 10a asks for the subset's stratification counts to be
        # *reported*, not merely produced; the report was being discarded.
        strata[name] = report.get("types", {})
        out.extend(examples)
    return out, strata


def derive_n_real(rate: float, ladder: tuple[int, ...], ceiling_s: float) -> dict[str, Any]:
    """The largest rung the slowest arm finishes inside the ceiling.

    Every rung is recorded, not only the adopted one — Phase-3 criterion 11's
    rule, because a ladder that reports only its answer cannot be audited for
    whether it had headroom or was pinned at its floor.

    A rate too slow for even the first rung does **not** silently adopt a smaller
    ``N``: the ladder's floor is a declared value, and going below it would be
    choosing a budget by wall clock after the fact. It returns ``None`` and names
    what would have to change.
    """
    rungs = []
    adopted = None
    for rung in ladder:
        seconds = rung / rate if rate > 0 else float("inf")
        fits = seconds <= ceiling_s
        rungs.append(
            {"n": rung, "projected_s": round(seconds, 1),
             "projected_h": round(seconds / 3600.0, 2), "fits": fits}
        )
        if fits:
            adopted = rung
    return {
        "rungs": rungs,
        "n_real": adopted,
        "note": (
            "largest rung inside the ceiling at the slowest arm's measured rate"
            if adopted is not None
            else "no rung fits: raise the ceiling deliberately or reduce per-trajectory "
                 "cost; adopting a sub-ladder N would be choosing a budget after the fact"
        ),
    }


def run(*, examples: int, smoke: bool, real_embedder: bool, trajectories: int,
        out_path: Path, reps: int = 3) -> int:
    config = load_config()
    started = time.perf_counter()

    pool, strata = build_examples(examples, smoke=smoke, real_embedder=real_embedder)
    envs = [RealEnvironment(ex, config, range_samples=8) for ex in pool]
    sizes = sorted(len(ex.pool) for ex in pool)
    state_dim, action_dim = RealFeaturizer.dims()

    # -- capacity match at the REAL dims (Phase-3 decision 11's form) -------
    #
    # Recomputed rather than inherited: atom width moved from the lattice's ~16
    # dims to 534, so a hidden size matched on synthetic dims matches nothing
    # here, and an unmatched comparison measures capacity instead of learning.
    #
    # **Live capacity, not nominal.** An arm with ``delta_d = False`` carries
    # ``N_DEFICIT x hidden`` weights per ``action_repr`` consumer that no gradient
    # ever reaches; matching the nominal count is how L6 came to sit 1.46% *below*
    # L7 while reporting a 0.00% match — the wrong side of decision 11's
    # directional clause, in the direction that flatters the proposed method.
    #
    # The target is **L7**, the proposed method: controls are widened up to it, so
    # a win is a win against a strictly larger control and "L7 had more capacity"
    # is unavailable as an objection.
    def arm_of(name: str, hidden: int = 64) -> Arm:
        kw = dict(ARMS[name])
        return Arm(name, kw.pop("loss"), hidden=hidden, **kw)

    def live_of(name: str, hidden: int) -> int:
        arm = arm_of(name, hidden)
        return Trainer.capacity_of(arm, state_dim, action_dim, hidden, 2) - \
            Trainer.dead_capacity_of(arm, hidden)

    # ``tol`` is gate2's **pathology ceiling**, not decision 11's criterion, and
    # passing it explicitly is the fix for the defect this line used to be
    # (16 Aug 2026 audit). Taking `match_capacity`'s old 0.01 default -- the
    # tolerance §6.4 retired as unachievable by width -- made the raise fire on
    # the *correct* width, and the handler below then recorded hidden 64. For
    # `l6_led` that is 220,164 live against L7's 220,932: a control **smaller
    # than the proposed method**, which is the one thing decision 11's
    # directional clause exists to forbid, in the direction that flatters L7.
    target_live = live_of("l7_checker_led", 64)
    widths: dict[str, dict[str, Any]] = {}
    for name in sorted(ARMS):
        try:
            width = match_capacity(
                lambda h, n=name: live_of(n, h), target_live, tol=CAPACITY_SANITY_CEILING
            )
        except ValueError as exc:  # reach failure, reported not hidden
            widths[name] = {"hidden": 64, "live": live_of(name, 64), "error": str(exc)}
            continue
        achieved = live_of(name, width)
        widths[name] = {
            "hidden": width,
            "live": achieved,
            "excess_pct": round(100.0 * (achieved - target_live) / target_live, 3),
            # Decision 11's two clauses, verified per arm rather than assumed --
            # gate2.capacity_matched_arm's pattern, which this caller had not
            # inherited. `narrowest` is vacuously true at the base width.
            "control_never_smaller": bool(achieved >= target_live),
            "narrowest_admissible": bool(width == 64 or live_of(name, width - 1) < target_live),
        }

    # A directional violation is not a row in a report; it invalidates the
    # comparison the whole phase exists to make. Refuse rather than record.
    violations = [n for n, w in widths.items() if not w.get("control_never_smaller", False)]
    if violations:
        raise SystemExit(
            f"capacity match failed decision 11's directional clause for {violations}: "
            f"live capacity below l7_checker_led's {target_live}. A control smaller "
            "than the proposed method makes any L7 win unreadable."
        )

    rows: dict[str, Any] = {}
    for name in sorted(ARMS):
        spec_kw = {k: v for k, v in ARMS[name].items() if k != "loss"}
        hidden = int(widths[name]["hidden"])
        arm = arm_of(name, hidden)
        spec = TrainSpec(seed=13, n_trajectories=trajectories, batch_size=4,
                         hidden=hidden, epsilon=0.05)
        # **Timed `reps` times, and `N_real` derives from the SLOWEST observed
        # rate** (16 Aug 2026 audit). A single sample of a wall clock is noise:
        # three runs of this script produced 34.68 / 35.58 / 36.16 traj/s for the
        # same arm, a 4% spread, and a rate that drifts is a rate that makes
        # `pins.BUDGET` disagree with the artefact it cites. Taking the minimum
        # is the same conservatism the "slowest arm" rule already applies one
        # level up: a budget costed at the slowest observed rate cannot be made
        # optimistic by a re-run.
        rates: list[float] = []
        for rep in range(max(1, int(reps))):
            torch.manual_seed(13)
            trainer = RealTrainer(arm, envs, spec, greedy=1)
            t0 = time.perf_counter()
            log = trainer.train()
            elapsed = time.perf_counter() - t0
            rates.append(trajectories / elapsed if elapsed > 0 else float("inf"))
        rate = min(rates)
        rows[name] = {
            "arm": name,
            "delta_d": bool(spec_kw.get("delta_d", False)),
            "trains_potential": bool(spec_kw.get("trains_potential", False)),
            "capacity": int(trainer.capacity),
            "live_capacity": int(trainer.live_capacity),
            "hidden": int(trainer.hidden),
            "capacity_match": widths[name],
            "trajectories": trajectories,
            "wall_clock_s": round(elapsed, 3),
            "traj_per_s": round(rate, 3),
            "traj_per_s_reps": [round(x, 3) for x in rates],
            "traj_per_s_max": round(max(rates), 3),
            "final_loss": float(log.final_loss),
            "loss_finite": bool(np.isfinite(log.final_loss)),
        }

    slowest = min(rows, key=lambda k: rows[k]["traj_per_s"])
    ladder = tuple(int(x) for x in pins.BUDGET["ladder"])
    derivation = derive_n_real(
        rows[slowest]["traj_per_s"], ladder, float(pins.BUDGET["ceiling_s"])
    )

    artefact = {
        "phase": 9,
        "step": 4,
        "smoke": bool(smoke),
        "real_embedder": bool(real_embedder),
        "honesty_stamp": HONESTY_STAMP,
        "stage_d_fingerprint": pins.stage_d_fingerprint(),
        "training_blocked": pins.training_blocked_reason(),
        "examples": {
            "count": len(pool),
            # criterion 10a: the subset's stratification counts, reported
            "strata": strata,
            "pool_size": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]},
            "gold_complete": sum(1 for ex in pool if ex.gold_complete),
        },
        "dims": {"state_dim": state_dim, "action_dim": action_dim},
        "capacity_match": {
            "target_arm": "l7_checker_led",
            "target_live": target_live,
            "rule": "controls widened to the narrowest width whose LIVE capacity is not below L7's",
            "widths": widths,
        },
        "arms": rows,
        "slowest_arm": slowest,
        "slowest_rate": rows[slowest]["traj_per_s"],
        "fastest_arm": max(rows, key=lambda k: rows[k]["traj_per_s"]),
        "spread": round(
            max(r["traj_per_s"] for r in rows.values())
            / max(1e-9, min(r["traj_per_s"] for r in rows.values())), 2
        ),
        "budget": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pins.BUDGET.items()},
        "derivation": derivation,
        "wall_clock_s": round(time.perf_counter() - started, 2),
        "manifest": run_manifest(config, 13),
    }
    artefact["determinism_digest"] = digest_of(
        deterministic_view({k: artefact[k] for k in ("dims", "examples", "stage_d_fingerprint")})
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_sanitize(artefact), indent=1, sort_keys=True),
                        encoding="utf-8")

    # `.resolve()` on BOTH halves: the guard resolved the path and the call did
    # not, so a *relative* --out passed the check and then raised — after the
    # artefact was already on disk, i.e. a successful run reporting failure.
    # The identical defect was found in Phase 7's runner (16 Aug 2026).
    resolved = out_path.resolve()
    print(f"wrote {resolved.relative_to(REPO) if resolved.is_relative_to(REPO) else resolved}")
    print(f"  examples {len(pool)}  pool {sizes[0]}-{sizes[-1]} atoms  "
          f"dims {state_dim}/{action_dim}")
    # ASCII in console output: the Windows console is cp1252 and a Greek phi
    # crashes the run *after* the artefact is written -- a successful run
    # reporting failure. The artefact itself is UTF-8 and keeps the real names.
    print(f"  {'arm':16} {'traj/s':>9} {'live cap':>9} {'loss':>12} {'dd':>4} {'phi':>4}")
    for name in sorted(rows, key=lambda k: rows[k]["traj_per_s"]):
        r = rows[name]
        print(f"  {name:16} {r['traj_per_s']:>9.2f} {r['capacity']:>9} {r['final_loss']:>12.4f}"
              f" {'y' if r['delta_d'] else '.':>4} {'y' if r['trains_potential'] else '.':>3}")
    print(f"\n  slowest: {slowest} at {rows[slowest]['traj_per_s']:.2f} traj/s "
          f"(spread {artefact['spread']}x)")
    for rung in derivation["rungs"]:
        mark = "ADOPT" if rung["fits"] else "over "
        print(f"    {mark} N={rung['n']:>7}  {rung['projected_h']:>6.2f} h "
              f"(ceiling {pins.BUDGET['ceiling_s']/3600:.1f} h)")
    print(f"  N_real = {derivation['n_real']}")
    if not all(r["loss_finite"] for r in rows.values()):
        print("  WARNING: a non-finite loss appeared; the reward floor is suspect")
    print("\n  NOTE: build step 4. Throughput and capacity only — not a Gate-3 result.")
    return 0


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
    parser.add_argument("--smoke", action="store_true", help="hand-built fixtures, no corpora")
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--trajectories", type=int, default=64,
                        help="per arm; the rate is what matters, not the count")
    parser.add_argument("--real-embedder", action="store_true",
                        help="the pinned bge-small rather than the stub")
    parser.add_argument("--reps", type=int, default=3,
                        help="timing repeats per arm; N_real derives from the slowest")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return run(examples=args.examples, smoke=args.smoke, real_embedder=args.real_embedder,
               trajectories=args.trajectories, out_path=args.out, reps=args.reps)


if __name__ == "__main__":
    raise SystemExit(main())
