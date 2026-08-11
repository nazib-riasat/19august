"""The Gate-2 harness: the matrix, the paired test, and the report.

**Gate 2's question, in one sentence** (``CLAUDE.md`` §8): does
checker-conditioned LED beat **capacity-matched** LED-GFN on exact TV at a fixed
training budget, three seeds, paired test? And decision 26 adds the second half
plan §4.5.4 requires: it must beat **capacity-matched GAFlowNet** too, because
"an intermediate signal helps credit assignment" is already published, and
beating L6 alone would establish only that.

**The decision rule is written before the numbers exist.** Fix F12 exists
because an unfalsifiable rule was the previous draft's defect:

    Contribution 3 is supported **iff** the one-sided 95% upper bound on
    ``TV_L7 − TV_control`` is below 0 against **both** L6 and GAFlowNet, under
    the hierarchical paired bootstrap of decision 20.

If either comparison fails, C3 is not supported and the thesis consolidates on
Contribution 1 (v1.2 §9 fallback). That is a designed outcome of this phase, not
a failure of it.

**Why the bootstrap is hierarchical.** Twenty instances per (arm, seed) are not
twenty independent samples of method quality — they share the seed, the trained
weights and the generator. Resampling 60 (seed, instance) pairs as if they were
independent understates the interval, which is the same clustering error the
LoCoMo caveat guards against elsewhere in this project. Seeds are the outer unit
because that is what the significance protocol replicates. **[EVIDENCE]** *The
Hitchhiker's Guide to Testing Statistical Significance in Natural Language
Processing* (ACL 2018).

**Every table carries its audits** (exit criterion 22, Phase-2 handoff item 7):
``FAIL`` rate, equivalent-action collision rate (expected 0),
unconstructible-valid-terminal rate (expected 0), ``p*(FAIL)`` **beside** TV and
never subtracted from it, the structural and visitation-weighted ``Δd``
densities, the target-mass profile, and the ``neither``-mass at the run's β. A
Gate-2 win is a win *under a declared signal density*; without the density
printed next to it, the number does not carry its own caveat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from graft.setgen.learners import FLOW_FAMILY, SUPERVISED_FAMILY, build_arm
from graft.setgen.learners.l6_led import consistency_error
from graft.setgen.policy import match_capacity
from graft.setgen.rollout import sample_trajectories
from graft.setgen.trainer import SEEDS, Environment, TrainLog, Trainer, TrainSpec
from graft.synth.audits import run_audits
from graft.synth.exact import divergence_report, policy_distribution
from graft.synth.policies import ForcedContinuationPolicy, UniformPolicy

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONSISTENCY_UNIFORM_SEED",
    "CONSISTENCY_FORCED_SEED",
    "CONSISTENCY_UNIFORM_N",
    "CONSISTENCY_FORCED_N",
    "CONSISTENCY_P95_BAND",
    "NON_INFERIORITY_MARGIN",
    "paired_bootstrap",
    "capacity_matched_arm",
    "run_matrix",
    "consistency_report",
    "audit_block",
    "Gate2Report",
]

#: Decision 20, frozen before any run.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260814

#: Decision 14 — one **common, frozen** trajectory set per instance, shared by
#: L6, L7 and L7b.  The sharing is what makes the three p95 values comparable;
#: per-arm sampling would produce three numbers measured on three populations.
CONSISTENCY_UNIFORM_SEED = 20260815
CONSISTENCY_FORCED_SEED = 20260816
CONSISTENCY_UNIFORM_N = 2_000
CONSISTENCY_FORCED_N = 500

#: Decision 13's band and decision 15's margin.  **[ANALYSIS]** both engineering.
CONSISTENCY_P95_BAND = 0.05
NON_INFERIORITY_MARGIN = 0.01


# --------------------------------------------------------------------------
# the paired test
# --------------------------------------------------------------------------


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Decision 20, executable: is ``mean(a − b)`` below 0 with 95% confidence?

    ``a`` and ``b`` are ``[n_seeds, n_instances]`` of final exact TV, **paired**:
    row ``i`` of each is the same seed, column ``j`` the same instance. Pairing
    is what removes instance difficulty and seed luck from the comparison, and it
    is only valid because every arm trains on the same suite under the same seed
    set (exit criteria 9 and 10).

    Two-stage resampling: seeds with replacement (the outer cluster), then for
    each drawn seed its own instances with replacement. Resampling instances once
    and reusing that draw across seeds would treat the inner units as shared,
    which they are not.

    ``a`` wins iff ``upper < 0`` — a **one-sided** 95% bound, because the
    question fix F12 asks is directional.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"unpaired inputs: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected [n_seeds, n_instances], got {a.ndim} dimensions")

    diff = a - b
    n_seeds, n_instances = diff.shape
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        picks = rng.integers(n_seeds, size=n_seeds)
        inner = rng.integers(n_instances, size=(n_seeds, n_instances))
        stats[r] = diff[picks[:, None], inner].mean()

    upper = float(np.percentile(stats, 95))
    return {
        "mean_difference": float(diff.mean()),
        "per_seed_difference": diff.mean(axis=1).tolist(),
        "upper_95": upper,
        "wins": bool(upper < 0.0),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n_seeds": int(n_seeds),
        "n_instances": int(n_instances),
    }


# --------------------------------------------------------------------------
# capacity matching
# --------------------------------------------------------------------------


def capacity_matched_arm(
    name: str, target_arm: str, envs: Sequence[Environment], spec: TrainSpec
) -> tuple[Any, dict[str, Any]]:
    """``name`` widened until it is capacity-matched to ``target_arm`` (G4).

    L6 needs no widening: it and L7 have identical parameter shapes, because the
    ``Δd`` block of ``action_repr`` is *present and zeroed* under L6 rather than
    absent. GAFlowNet carries no ``PotentialHead`` and is ~37% smaller at equal
    width, so its trunk grows instead — the only knob that closes the gap without
    adding parameters no gradient ever reaches, which would be a match on paper
    and dead capacity in fact.
    """
    from graft.setgen.features import SyntheticFeaturizer

    state_dim, action_dim = SyntheticFeaturizer.dims(envs[0].instance, envs[0].graph)
    proposed = build_arm(target_arm)
    target = Trainer.capacity_of(proposed, state_dim, action_dim, spec.hidden, spec.depth)

    control = build_arm(name)
    base = Trainer.capacity_of(control, state_dim, action_dim, spec.hidden, spec.depth)
    if base >= target:
        hidden = spec.hidden
    else:
        hidden = match_capacity(
            lambda h: Trainer.capacity_of(control, state_dim, action_dim, h, spec.depth),
            target,
            base=spec.hidden,
        )
    achieved = Trainer.capacity_of(control, state_dim, action_dim, hidden, spec.depth)
    return build_arm(name, hidden=hidden), {
        "control": name,
        "proposed": target_arm,
        "target_capacity": target,
        "control_capacity": achieved,
        "control_hidden": hidden,
        "relative_excess": achieved / target - 1.0,
        # Decision 11's directional clause: the control is never smaller.
        "control_never_smaller": achieved >= target,
    }


# --------------------------------------------------------------------------
# audits
# --------------------------------------------------------------------------


def audit_block(envs: Sequence[Environment]) -> dict[str, Any]:
    """Exit criterion 22, aggregated over the suite.

    Everything here comes from ``graft.synth.audits`` at zero marginal cost —
    Phase 2 already computes it — which is why criterion 27 restores the
    collision and unconstructible rates rather than treating them as optional.
    """
    rows = [run_audits(env.instance, env.graph, env.target) for env in envs]
    collisions = sum(r["collisions"]["equivalent_action_collisions"] for r in rows)
    unconstructible = sum(r["closure"]["unconstructible_valid_terminals"] for r in rows)
    return {
        "instances": len(rows),
        "equivalent_action_collisions": collisions,
        "unconstructible_valid_terminals": unconstructible,
        "reachable_dead_ends": [r["fail"]["reachable_dead_ends"] for r in rows],
        "zero_delta_d_structural": [
            r["delta_d"]["zero_delta_d_structural"] for r in rows
        ],
        "zero_delta_d_visitation": [
            r["delta_d"]["zero_delta_d_visitation"] for r in rows
        ],
        "target_p_fail": [r["target_mass"]["target_p_fail"] for r in rows],
        "neither_mass": [r["target_mass"]["mode_mass"]["neither"] for r in rows],
        "beta": [r["target_mass"]["beta"] for r in rows],
        "n_terminals": [r["target_mass"]["n_terminals"] for r in rows],
        "environment_fingerprints": [r["environment_fingerprint"] for r in rows],
    }


def divergences(trainer: Trainer) -> list[dict[str, float]]:
    """The full Phase-2 divergence report per instance, at the current θ.

    TV alone would be a thinner claim than Phase 2 can support: ``divergence_report``
    already binds JS, KL-when-finite, FCS with its standard error, and
    ``p*(FAIL)`` **beside** TV rather than subtracted from it.
    """
    with torch.no_grad():
        return [
            divergence_report(policy_distribution(feat, env.graph), env.target)
            for env, feat in zip(trainer.envs, trainer.featurizers)
        ]


# --------------------------------------------------------------------------
# terminal consistency (G10, decisions 13 and 14)
# --------------------------------------------------------------------------


def consistency_report(trainer: Trainer) -> dict[str, Any]:
    """Per-trajectory ``|Σφ − ℰ|`` on the frozen common set (criterion 16).

    **Per trajectory, not per terminal.** LED decomposes a trajectory; one
    terminal is reached by many paths, and a per-terminal reading would average
    over exactly the object being measured.

    ``FAIL`` is audited on its own line and never pooled: ``ℰ(FAIL) = −log r_fail
    = 13.8`` sits far outside the valid range, so a handful of ``FAIL``
    trajectories would dominate a p95 over thousands. At least one is asserted
    per instance — the ``ForcedContinuationPolicy`` draw exists to guarantee it
    rather than hope for it, since ``UniformPolicy`` stops early and can miss the
    dead ends entirely.
    """
    if trainer.potential is None:
        raise ValueError(f"{trainer.arm.name} has no potential to check")

    per_instance: list[dict[str, Any]] = []
    for env, feat in zip(trainer.envs, trainer.featurizers):
        errors: list[np.ndarray] = []
        fails: list[np.ndarray] = []
        for policy, n, seed in (
            (UniformPolicy(), CONSISTENCY_UNIFORM_N, CONSISTENCY_UNIFORM_SEED),
            (ForcedContinuationPolicy(), CONSISTENCY_FORCED_N, CONSISTENCY_FORCED_SEED),
        ):
            traj = sample_trajectories(policy, env.graph, n, np.random.default_rng(seed))
            with torch.no_grad():
                batch = trainer.build_batch(env, feat, traj)
                error, is_fail = consistency_error(
                    trainer.compute_potential(batch), batch
                )
            errors.append(error.cpu().numpy())
            fails.append(is_fail.cpu().numpy())

        error = np.concatenate(errors)
        is_fail = np.concatenate(fails)
        valid, failed = error[~is_fail], error[is_fail]
        per_instance.append(
            {
                "log_r_range": env.log_r_range,
                "excluded_by_range_guard": bool(env.log_r_range < 0.1),
                "n_valid": int(valid.size),
                "n_fail": int(failed.size),
                "fail_covered": bool(failed.size > 0),
                "mean": float(np.mean(valid)) if valid.size else float("nan"),
                "p95": float(np.percentile(valid, 95)) if valid.size else float("nan"),
                "max": float(np.max(valid)) if valid.size else float("nan"),
                "fail_mean": float(np.mean(failed)) if failed.size else float("nan"),
                "fail_p95": (
                    float(np.percentile(failed, 95)) if failed.size else float("nan")
                ),
            }
        )

    scored = [r for r in per_instance if not r["excluded_by_range_guard"]]
    p95 = [r["p95"] for r in scored]
    return {
        "arm": trainer.arm.name,
        "seed": trainer.spec.seed,
        "per_instance": per_instance,
        "p95_worst": float(np.max(p95)) if p95 else float("nan"),
        "p95_mean": float(np.mean(p95)) if p95 else float("nan"),
        "band": CONSISTENCY_P95_BAND,
        "passes_band": bool(p95) and bool(np.max(p95) <= CONSISTENCY_P95_BAND),
        "fail_covered_everywhere": all(r["fail_covered"] for r in per_instance),
        "excluded_instances": sum(
            1 for r in per_instance if r["excluded_by_range_guard"]
        ),
    }


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


class Gate2Report:
    """Everything the write-up needs, and the decision rule already applied."""

    __slots__ = ("logs", "audits", "capacity", "consistency", "comparisons",
                 "divergences", "spec", "verdict")

    def __init__(self) -> None:
        self.logs: dict[str, dict[int, TrainLog]] = {}
        self.audits: dict[str, Any] = {}
        self.capacity: list[dict[str, Any]] = []
        self.consistency: list[dict[str, Any]] = []
        self.comparisons: dict[str, Any] = {}
        self.divergences: dict[str, list[dict[str, float]]] = {}
        self.spec: dict[str, Any] = {}
        self.verdict: dict[str, Any] = {}

    def tv_matrix(self, arm: str) -> np.ndarray:
        """``[n_seeds, n_instances]`` of **final** exact TV, seeds in frozen order."""
        rows = self.logs[arm]
        return np.asarray(
            [rows[s].tv_per_env[-1] for s in SEEDS if s in rows], dtype=np.float64
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "verdict": self.verdict,
            "comparisons": self.comparisons,
            "capacity": self.capacity,
            "consistency": self.consistency,
            "audits": self.audits,
            "divergences": self.divergences,
            "flow_family": list(FLOW_FAMILY),
            "supervised_family": list(SUPERVISED_FAMILY),
            "runs": {
                arm: {str(seed): log.to_dict() for seed, log in rows.items()}
                for arm, rows in self.logs.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), "utf-8")
        return path


def run_matrix(
    envs: Sequence[Environment],
    spec: TrainSpec,
    *,
    arms: Iterable[str] = FLOW_FAMILY + SUPERVISED_FAMILY,
    seeds: Sequence[int] = SEEDS,
    tv_threshold: float = 0.10,
    progress: Any = None,
) -> Gate2Report:
    """Train every arm at every seed and apply decision 26's rule.

    ``l6_led`` and ``gaflownet`` are widened to L7's capacity before they train
    (G4). Nothing else is adjusted per arm — decision 23's protocol is the same
    for all of them, and ``TrainSpec.shared_protocol`` is asserted identical in
    every log so a divergence is visible in the artefact rather than only in the
    code that produced it.
    """
    arms = list(arms)
    report = Gate2Report()
    report.spec = {
        "n_trajectories": spec.n_trajectories,
        "seeds": list(seeds),
        "arms": arms,
        "instances": len(envs),
        "tv_threshold": tv_threshold,
        **spec.shared_protocol(),
    }
    report.audits = audit_block(envs)

    matched: dict[str, Any] = {}
    for control in ("l6_led", "gaflownet"):
        if control in arms and "l7_checker_led" in arms:
            arm, note = capacity_matched_arm(control, "l7_checker_led", envs, spec)
            matched[control] = arm
            report.capacity.append(note)

    for name in arms:
        report.logs[name] = {}
        for seed in seeds:
            arm = matched.get(name) or build_arm(name)
            trainer = Trainer(arm, envs, spec.replace(seed=seed))
            log = trainer.train(progress=progress)
            report.logs[name][seed] = log
            if seed == seeds[0]:
                report.divergences[name] = divergences(trainer)
            if trainer.potential is not None:
                report.consistency.append(consistency_report(trainer))
            del trainer

    protocols = {json.dumps(log.protocol, sort_keys=True)
                 for rows in report.logs.values() for log in rows.values()}
    if len(protocols) > 1:
        raise RuntimeError(
            "arms did not share one training protocol (exit criterion 9): "
            f"{len(protocols)} distinct protocol blocks in one matrix"
        )

    report.comparisons = _compare(report, arms, tv_threshold)
    report.verdict = _verdict(report, arms)
    return report


def _compare(report: Gate2Report, arms: Sequence[str], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {"tv_threshold": threshold}
    for name in arms:
        rows = report.logs[name]
        matrix = report.tv_matrix(name)
        out[name] = {
            "final_tv_mean": float(matrix.mean()),
            "final_tv_per_seed": matrix.mean(axis=1).tolist(),
            "final_tv_std_over_instances": matrix.std(axis=1).tolist(),
            "capacity": [rows[s].capacity for s in SEEDS if s in rows],
            # Decision 21: censored is None, never the budget.
            "trajectories_to_threshold": [
                rows[s].trajectories_to(threshold) for s in SEEDS if s in rows
            ],
            "final_c_t": [rows[s].final_c_t for s in SEEDS if s in rows],
            "final_loss": [rows[s].final_loss for s in SEEDS if s in rows],
            "descriptive_tv": name in SUPERVISED_FAMILY,
        }
    if "l7_checker_led" in arms:
        proposed = report.tv_matrix("l7_checker_led")
        for control in ("l6_led", "gaflownet"):
            if control in arms:
                out[f"l7_vs_{control}"] = paired_bootstrap(
                    proposed, report.tv_matrix(control)
                )
    return out


def _verdict(report: Gate2Report, arms: Sequence[str]) -> dict[str, Any]:
    """Decision 26 and decision 15, applied. Written before the numbers existed."""
    beats_l6 = report.comparisons.get("l7_vs_l6_led", {}).get("wins")
    beats_gafn = report.comparisons.get("l7_vs_gaflownet", {}).get("wins")

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in report.consistency:
        by_arm.setdefault(row["arm"], []).append(row)

    def worst(arm: str) -> float:
        rows = by_arm.get(arm, [])
        values = [r["p95_worst"] for r in rows if np.isfinite(r["p95_worst"])]
        return float(np.max(values)) if values else float("nan")

    p95_l6, p95_l7 = worst("l6_led"), worst("l7_checker_led")
    band_ok = bool(
        np.isfinite(p95_l6)
        and np.isfinite(p95_l7)
        and p95_l6 <= CONSISTENCY_P95_BAND
        and p95_l7 <= CONSISTENCY_P95_BAND
    )
    # Decision 15: a **non-inferiority margin**, not an ordering.  0.030 versus
    # 0.031 is noise and must not decide a contribution.
    margin_ok = bool(np.isfinite(p95_l7 - p95_l6) and (p95_l7 - p95_l6) <= NON_INFERIORITY_MARGIN)

    # Criterion 14: GAFlowNet's control is only valid if c_t reached exactly 0.
    gafn_ct = report.comparisons.get("gaflownet", {}).get("final_c_t", [])
    gafn_ok = bool(gafn_ct) and all(v == 0.0 for v in gafn_ct)

    supported = bool(beats_l6 and beats_gafn and band_ok and margin_ok and gafn_ok)
    return {
        "contribution_3_supported": supported,
        "beats_capacity_matched_l6": beats_l6,
        "beats_capacity_matched_gaflownet": beats_gafn,
        "consistency_band_passed": band_ok,
        "non_inferiority_margin_passed": margin_ok,
        "p95_l6": p95_l6,
        "p95_l7": p95_l7,
        "gaflownet_intrinsic_reached_zero": gafn_ok,
        "rule": (
            "Contribution 3 is supported iff L7 beats capacity-matched L6 AND "
            "capacity-matched GAFlowNet on the one-sided 95% upper bound of "
            "TV_L7 - TV_control (decision 20), with both arms inside decision "
            "13's consistency band and within decision 15's non-inferiority "
            "margin. If either comparison fails, C3 is not supported and the "
            "thesis consolidates on Contribution 1 (v1.2 §9)."
        ),
        "partial_discharge": (
            "Gate-2 item 3 is discharged in part only: Tier 1 plus GAFlowNet "
            "ran; FM, DB, FL-DB and FL-SubTB remain deferred, with FL's row "
            "discharged by the measurement of decision 24 rather than by "
            "training (criterion 28)."
        ),
    }


def format_table(report: Gate2Report) -> str:
    """The two tables of decision 12, as text. TV is descriptive for L1-L3."""
    lines = ["arm                 capacity   final TV   traj-to-threshold"]
    for family, title in (
        (FLOW_FAMILY, "flow family — exact TV is the primary metric"),
        (SUPERVISED_FAMILY, "L1-L3 — exact TV is DESCRIPTIVE (decision 12)"),
    ):
        lines.append(f"-- {title}")
        for name in family:
            row = report.comparisons.get(name)
            if row is None:
                continue
            reached = row["trajectories_to_threshold"]
            shown = ", ".join("censored" if v is None else str(v) for v in reached)
            lines.append(
                f"{name:<20}{row['capacity'][0]:>8}   "
                f"{row['final_tv_mean']:.4f}   {shown}"
            )
    return "\n".join(lines)
