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
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from graft.canonical import digest_of
from graft.setgen.learners import FLOW_FAMILY, SUPERVISED_FAMILY, build_arm
from graft.setgen.learners.l6_led import consistency_error
from graft.setgen.policy import match_capacity
from graft.setgen.rollout import sample_trajectories
from graft.setgen.trainer import (
    DECISION5_RUNGS,
    SEEDS,
    Environment,
    TrainLog,
    Trainer,
    TrainSpec,
)
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
    "CAPACITY_SANITY_CEILING",
    "BEST_OF_K_SEED",
    "paired_bootstrap",
    "capacity_matched_arm",
    "best_of_k",
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

#: One frozen stream for every arm's best-of-K draw, for the same reason
#: decision 14 freezes the consistency set: three arms read under three streams
#: are three numbers measured on three populations.
BEST_OF_K_SEED = 20260817

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


#: A **sanity ceiling**, not decision 11's tolerance.  The guarantee that matters
#: is *minimality* — the control is the narrowest width that is not smaller —
#: which ``capacity_matched_arm`` verifies directly and which bounds the excess
#: by one width step whatever that step happens to be.  This exists only so a
#: future architecture with enormous width steps fails loudly instead of handing
#: the control 50% more capacity in silence.  It is deliberately far above the
#: measured 1.4–2.2% so that it never becomes a number anyone tunes.
CAPACITY_SANITY_CEILING = 0.05


def capacity_matched_arm(
    name: str, target_arm: str, envs: Sequence[Environment], spec: TrainSpec
) -> tuple[Any, dict[str, Any]]:
    """``name`` widened until its **live** capacity is not below ``target_arm``'s.

    **Live, not nominal, and that is the correction.** ``Trainer.dead_capacity_of``
    explains why an arm with ``delta_d = False`` carries ``N_DEFICIT × hidden``
    weights per ``action_repr`` consumer that no gradient ever reaches. Matching
    the nominal count made L6 and L7 agree to 0.00% while L6 held **1.46% less
    trainable capacity than L7** — the wrong side of the directional clause, in
    the direction that flatters the proposed method.

    **Why §6's "within 1%" is unachievable rather than merely missed.** The dead
    block is ``12·hidden`` weights and one unit of width is worth ~``2.4·hidden``
    parameters at depth 2, so the block is almost exactly **half a width step** —
    measured 0.50, 0.52 and 0.53 at hidden 64, 128 and 256, a property of the
    architecture rather than of this width. The smallest width that closes it
    therefore overshoots by roughly half a step, ~1.4–2.2%, at *every* scale, and
    no choice of base width rescues the 1%. Widening is also the only knob
    available: the alternative of *removing* L7's extra parameters would remove
    the mechanism under test, since the ``Δd`` weights and the checker
    conditioning are the same 768 parameters.

    **So the guarantee is minimality, not a percentage.** Two clauses replace it,
    and neither is a tuned number:

    *never smaller* — kept hard, because it carries the whole argument: if L7
    wins it wins against a **strictly larger** control, and "L7 had more
    capacity" is unavailable as an objection.

    *narrowest such width* — so the excess is the smallest the architecture
    admits, bounded by one width step by construction, verified here rather than
    assumed, and reported per arm so the write-up carries the actual figure
    instead of a tolerance it was compared against.

    ``CAPACITY_SANITY_CEILING`` is a guard against a pathological future
    architecture, not the criterion.
    """
    from graft.setgen.features import SyntheticFeaturizer

    state_dim, action_dim = SyntheticFeaturizer.dims(envs[0].instance, envs[0].graph)

    def live(arm: Any, h: int) -> int:
        return Trainer.live_capacity_of(arm, state_dim, action_dim, h, spec.depth)

    proposed = build_arm(target_arm)
    target = live(proposed, spec.hidden)

    control = build_arm(name)
    if live(control, spec.hidden) >= target:
        hidden = spec.hidden
    else:
        hidden = match_capacity(
            lambda h: live(control, h),
            target,
            base=spec.hidden,
            tol=CAPACITY_SANITY_CEILING,
        )

    achieved_live = live(control, hidden)
    # Minimality, checked rather than assumed: one width narrower must be below
    # the target, or the search overshot and the excess is not the smallest the
    # architecture admits.
    minimal = hidden == spec.hidden or live(control, hidden - 1) < target
    achieved = Trainer.capacity_of(control, state_dim, action_dim, hidden, spec.depth)
    return build_arm(name, hidden=hidden), {
        "control": name,
        "proposed": target_arm,
        # Nominal counts, kept so the artefact shows both readings side by side.
        "target_capacity": Trainer.capacity_of(
            proposed, state_dim, action_dim, spec.hidden, spec.depth
        ),
        "control_capacity": achieved,
        "control_dead_parameters": Trainer.dead_capacity_of(control, hidden),
        # ...and the pair decision 11 is actually about.
        "target_live_capacity": target,
        "control_live_capacity": achieved_live,
        "control_hidden": hidden,
        "relative_excess": achieved_live / target - 1.0,
        "width_step": live(control, hidden + 1) - achieved_live,
        # Decision 11's directional clause: the control is never smaller.
        "control_never_smaller": achieved_live >= target,
        # ...and the clause that replaces its unachievable 1%.
        "narrowest_admissible_width": minimal,
        "sanity_ceiling": CAPACITY_SANITY_CEILING,
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
        # Phase-2 handoff item 7 asks for the target-mass **profile**, not one
        # component of it.  `neither_mass` above is the band-bearing number and
        # stays flattened for the table; this is every mode's mass and count,
        # the zero-sufficiency mass, the effective support and the U range —
        # without which a reader cannot tell a bimodal environment from a
        # unimodal one at the β the run used (PHASE2_DECISIONS §4.3).
        "target_mass_profile": [r["target_mass"] for r in rows],
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
# best-of-K (criterion 17's secondary; decision 12's metric for L1-L3)
# --------------------------------------------------------------------------


@torch.no_grad()
def best_of_k(trainer: Trainer, *, k: int | None = None) -> dict[str, Any]:
    """Sample ``K`` sets, keep the valid ones, report the best utility.

    **Plan §6.4's Stage-D primary, appearing here as criterion 17's secondary.**
    "Sample K sets, keep those passing ``H``, report the utility of the best
    one." On the lattice ``H`` is settled by enumeration and ``U`` is exact, so
    a dead-ended rollout is the only invalid outcome and no checker call is
    needed to recognise it.

    **And it is the metric that fits L1, L2 and L3** (decision 12, G2). Exact TV
    is descriptive for those three — L1 and L2 imitate a single gold set and L3
    maximises return, so none of them is trying to match a distribution, and a
    table ranking them by TV would be reporting that two of them failed a task
    they never attempted. Best-of-K utility and gold exact match are questions
    all nine arms are actually answering.

    **The portfolio is fix F5's: 1 greedy + 7 sampled**, not eight stochastic
    draws. F5 fixes ``K = 8`` *and* its composition, and the greedy candidate is
    the one a reward-maximiser would return — leaving it out measures a different
    portfolio from the one Phase 9 ships and the one Phase 4's S5 compares
    against. An earlier build sampled all eight and did not say so.

    ``K`` and ``checker_budget`` are one constant used everywhere (fix F5). Only
    ``K`` terminals are scored, so ``K = 8`` spends 8 of the 32 checks; the
    budget binds at Phase 4, which is the inference path, not here (Phase 3 runs
    ``ledger=None`` by decision 16).

    The sampled draws are at ``ε = 0`` — the trained policy, not the behaviour
    policy — from one frozen seed, so every arm is read under the same stream and
    the differences between them are differences of policy. (The stream is the
    same *call*, not the same trajectories as a non-greedy call would give; see
    ``sample_trajectories``.)
    """
    rng = np.random.default_rng(BEST_OF_K_SEED)
    rows: list[dict[str, Any]] = []
    k_eff = 0
    for env, feat in zip(trainer.envs, trainer.featurizers):
        cfg = env.instance.cfg
        k_eff = int(k if k is not None else cfg.K)
        if k_eff > cfg.checker_budget:
            raise ValueError(
                f"K={k_eff} exceeds checker_budget={cfg.checker_budget}; fix F5 "
                "makes them one constant and K may never outrun the budget"
            )
        traj = sample_trajectories(feat, env.graph, k_eff, rng, 0.0, greedy=1)
        position = {int(s): i for i, s in enumerate(env.graph.terminal_ix.tolist())}
        utilities = [
            float(env.target.u[position[int(t)]])
            for t, failed in zip(traj.terminal.tolist(), traj.is_fail.tolist())
            if not failed
        ]
        gold_state = int(feat.gold_path(None)[0][-1])
        rows.append({
            "best_utility": max(utilities) if utilities else float("nan"),
            "mean_utility": float(np.mean(utilities)) if utilities else float("nan"),
            "valid_of_k": len(utilities),
            "terminal_checks": k_eff,
            # Fix F5's composition, recorded so a reader can see it was honoured.
            "greedy_candidates": 1,
            "sampled_candidates": k_eff - 1,
            "gold_exact_match": float(
                np.mean([int(t) == gold_state for t in traj.terminal.tolist()])
            ),
        })

    best = [r["best_utility"] for r in rows if not math.isnan(r["best_utility"])]
    return {
        "arm": trainer.arm.name,
        "seed": trainer.spec.seed,
        "k": k_eff,
        "per_instance": rows,
        # NaN when an instance produced no valid set at all -- reported as
        # instances_scored rather than quietly dropped from the denominator.
        "instances_scored": len(best),
        "instances": len(rows),
        "best_utility_mean": float(np.mean(best)) if best else float("nan"),
        "gold_exact_match_mean": float(
            np.mean([r["gold_exact_match"] for r in rows])
        ) if rows else float("nan"),
        "valid_rate": float(
            np.mean([r["valid_of_k"] / max(r["terminal_checks"], 1) for r in rows])
        ) if rows else float("nan"),
    }


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


class Gate2Report:
    """Everything the write-up needs, and the decision rule already applied."""

    __slots__ = ("logs", "audits", "probe_audits", "capacity", "consistency",
                 "comparisons", "divergences", "best_of_k", "probe", "spec",
                 "verdict")

    def __init__(self) -> None:
        self.logs: dict[str, dict[int, TrainLog]] = {}
        self.audits: dict[str, Any] = {}
        self.probe_audits: dict[str, Any] = {}
        self.capacity: list[dict[str, Any]] = []
        self.consistency: list[dict[str, Any]] = []
        self.comparisons: dict[str, Any] = {}
        self.divergences: dict[str, dict[int, list[dict[str, float]]]] = {}
        self.best_of_k: list[dict[str, Any]] = []
        self.probe: dict[str, dict[int, list[float]]] = {}
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
            "probe_audits": self.probe_audits,
            "best_of_k": self.best_of_k,
            "probe": {
                arm: {str(seed): rows for seed, rows in by_seed.items()}
                for arm, by_seed in self.probe.items()
            },
            "divergences": {
                arm: {str(seed): rows for seed, rows in by_seed.items()}
                for arm, by_seed in self.divergences.items()
            },
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


def _suite_identity(instances: Iterable[Any]) -> list[str]:
    """Per-instance digests of everything the generator produced.

    ``identity_payload`` is instance-only, so the reference suite is *generated*
    and never enumerated — ~4 s against the hours a matrix run takes, and it is
    the only way to establish that these 20 instances are the frozen 20 rather
    than 20 instances.
    """
    return [digest_of(inst.identity_payload()) for inst in instances]


@lru_cache(maxsize=1)
def _frozen_identities() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(main, probe)`` identities of the frozen suites, computed once.

    Both are deterministic functions of generator seeds frozen at Gate 0
    (``MAIN_SEED``, ``PROBE_SEED``), so caching cannot go stale within a process
    and the ~4 s of generation is paid once rather than once per clause.
    """
    from graft.synth.lattice import benchmark_suite, probe_suite

    return (
        tuple(_suite_identity(benchmark_suite())),
        tuple(_suite_identity(probe_suite())),
    )


def _admissibility(
    arms: Sequence[str],
    seeds: Sequence[int],
    envs: Sequence[Environment],
    spec: TrainSpec,
    probe_envs: Sequence[Environment] | None,
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Whether this run may produce a scientific verdict at all.

    **The hole this closes.** ``run_matrix`` takes everything that defines the
    experiment as a parameter, and every one of them had a default that produced
    a ``contribution_3_supported`` boolean indistinguishable in the artefact from
    a real one. The first version of this function checked the roster and the
    seeds; a run on **one tiny instance at an uncalibrated ``N`` and ``β`` with no
    probe suite** still passed it. Each clause below is a thing the plan already
    requires and nothing enforced:

    ``roster`` and ``seeds``
        decision 1 and criterion 10.

    ``suite``
        decision 8 scores on **the frozen 20-instance main suite**, and criterion
        25 says no result computed against one environment may be compared with
        another. Checked by instance identity, not by count.

    ``N`` and ``β``
        decision 4 and Phase-2 decision 22 produce both from the calibration gate
        (step 6), and §6 says "nothing proceeds until this is written into §6".
        A matrix run that invented its own budget is the exact thing that
        ordering exists to prevent, so the adopted record must be handed in and
        must match. ``--quick`` records are refused by name: they are wiring
        checks whose own output says it may not enter §6.

    ``probe``
        criterion 23 reads it once at the end, and §6b's risk row makes the
        result conditional on the ``Δd`` density it was measured under — "both or
        neither".

    Refusing by raising would make every wiring test assemble a full Gate-2 run
    it does not need, so the run proceeds and the **verdict** is withheld
    instead: ``contribution_3_supported`` is ``None``, with every failed clause
    named. A reduced run stays useful for what it is for and cannot be mistaken
    for Gate 2.
    """
    reasons: list[str] = []

    missing = sorted((set(FLOW_FAMILY) | set(SUPERVISED_FAMILY)) - set(arms))
    if missing:
        reasons.append(f"arms missing from the decision-1 roster: {missing}")
    if tuple(seeds) != SEEDS:
        reasons.append(f"seeds {list(seeds)} are not the frozen {list(SEEDS)}")

    # Only pay for suite generation once the cheap clauses have passed; a wiring
    # run fails above and never reaches it.
    if not reasons:
        main_ids, probe_ids = _frozen_identities()
        if tuple(_suite_identity(e.instance for e in envs)) != main_ids:
            reasons.append(
                f"the {len(envs)} scored instances are not the frozen 20-instance "
                "main suite (decision 8, criterion 25)"
            )
        if probe_envs is None:
            reasons.append("no probe suite: criterion 23's held-out read did not happen")
        elif tuple(_suite_identity(e.instance for e in probe_envs)) != probe_ids:
            reasons.append("the held-out instances are not the frozen probe suite")

        adopted = (calibration or {}).get("adopted") or {}
        if not calibration:
            reasons.append(
                "no calibration record: N and beta must come from step 6 "
                "(decision 4, Phase-2 decision 22), not from a default"
            )
        elif calibration.get("quick"):
            reasons.append("the calibration record is a --quick wiring check")
        elif calibration.get("verdict") != "adopted" or not adopted:
            reasons.append(
                f"the calibration verdict is {calibration.get('verdict')!r}, so no "
                "N or beta was adopted"
            )
        else:
            # The ceilings are a script argument, so an "adopted" record can be
            # produced at any budget at all. Without this clause every other one
            # passes a `--rungs 0.5` calibration and Gate 2 runs at a budget
            # nobody declared.
            if float(adopted.get("ceiling_s", -1.0)) not in DECISION5_RUNGS:
                reasons.append(
                    f"the adopted ceiling {adopted.get('ceiling_s')}s is not one of "
                    f"decision 5's rungs {list(DECISION5_RUNGS)}"
                )
            if spec.n_trajectories != adopted.get("N"):
                reasons.append(
                    f"N={spec.n_trajectories} is not the adopted {adopted.get('N')}"
                )
            betas = {float(e.target.beta) for e in envs}
            if betas != {float(adopted.get("beta"))}:
                reasons.append(
                    f"environments are at beta={sorted(betas)}, not the adopted "
                    f"{adopted.get('beta')}"
                )

    return {
        "admissible": not reasons,
        "inadmissible_reason": "; ".join(reasons) or None,
    }


def run_matrix(
    envs: Sequence[Environment],
    spec: TrainSpec,
    *,
    arms: Iterable[str] = FLOW_FAMILY + SUPERVISED_FAMILY,
    seeds: Sequence[int] = SEEDS,
    tv_threshold: float = 0.10,
    probe_envs: Sequence[Environment] | None = None,
    calibration: Mapping[str, Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    progress: Any = None,
) -> Gate2Report:
    """Train every arm at every seed and apply decision 26's rule.

    ``l6_led`` and ``gaflownet`` are widened to L7's **live** capacity before
    they train (G4). Nothing else is adjusted per arm — decision 23's protocol is
    the same for all of them, and ``TrainSpec.shared_protocol`` is asserted
    identical in every log so a divergence is visible in the artefact rather than
    only in the code that produced it.

    ``probe_envs`` is decision 9's held-out suite, read **once, at the end of
    each run, and never used to select anything** (criterion 23). Training and
    scoring happen on ``envs``; the write-up says "fitting", and the probe line
    is what lets it say anything about a sparser ``Δd`` density (G7, §6b).

    ``calibration`` is step 6's artefact (``scripts/phase3_calibrate.py``). It is
    what lets ``_admissibility`` check that ``N`` and β are the adopted ones
    rather than whatever the caller happened to construct — the ordering §6
    states as "nothing proceeds until this is written into §6".

    ``checkpoint_dir`` persists each trained model. Phase-3 §8 requirement 1
    makes Phase 4's S5 a consumer of these, and without it a Gate-2 run discards
    every set of weights it produces — the trainers are deleted as the loop goes.
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
        **_admissibility(arms, seeds, envs, spec, probe_envs, calibration),
    }
    report.audits = audit_block(envs)
    # Criterion 23 and §6b's density row: the probe result is only interpretable
    # beside the density it was measured under, and "both or neither" means the
    # held-out suite carries the same audit block as the scored one — its Δd
    # densities, target-mass profile, β and fingerprints, not just its TV.
    if probe_envs:
        report.probe_audits = audit_block(probe_envs)

    matched: dict[str, Any] = {}
    for control in ("l6_led", "gaflownet"):
        if control in arms and "l7_checker_led" in arms:
            arm, note = capacity_matched_arm(control, "l7_checker_led", envs, spec)
            matched[control] = arm
            report.capacity.append(note)

    for name in arms:
        report.logs[name] = {}
        report.divergences[name] = {}
        report.probe[name] = {}
        for seed in seeds:
            arm = matched.get(name) or build_arm(name)
            trainer = Trainer(arm, envs, spec.replace(seed=seed))
            log = trainer.train(progress=progress)
            report.logs[name][seed] = log
            report.best_of_k.append(best_of_k(trainer))
            if probe_envs:
                report.probe[name][seed] = trainer.evaluate_on(probe_envs)
            if checkpoint_dir is not None:
                trainer.save_checkpoint(Path(checkpoint_dir) / f"{name}.seed{seed}.pt")
            # Every seed, not just the first.  FCS is a correctness proxy with a
            # standard error (plan §6.4); measured on one seed of three it comes
            # with no spread, and the seed protocol exists precisely to supply
            # one.  Cost is one divergence_report per (arm, seed) -- ~0.05 s per
            # instance, ~26 s across the whole matrix.
            report.divergences[name][seed] = divergences(trainer)
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
        portfolio = [r for r in report.best_of_k if r["arm"] == name]
        probe = report.probe.get(name, {})
        out[name] = {
            # Criterion 17's secondary, and the only metric that fits L1-L3.
            "best_of_k_utility": [r["best_utility_mean"] for r in portfolio],
            "gold_exact_match": [r["gold_exact_match_mean"] for r in portfolio],
            "valid_rate": [r["valid_rate"] for r in portfolio],
            # Criterion 23: held out, read once, never used to select anything.
            "probe_tv": [float(np.mean(probe[s])) for s in SEEDS if s in probe],
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
    admissible = bool(report.spec.get("admissible"))
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
    # **The two halves of decision 13's band mean different things**, and pooling
    # them into one boolean made a published baseline's failure read as evidence
    # against the proposed method. If *L7* leaves the band, its TV win was bought
    # by relaxing LED's regulariser and plan §4.5.4's hard constraint is broken —
    # that is C3's result. If *L6* leaves it, vanilla LED-GFN did not decompose
    # its own energy at this `N`, which is a statement about the instrument.
    l6_band_ok = bool(np.isfinite(p95_l6) and p95_l6 <= CONSISTENCY_P95_BAND)
    l7_band_ok = bool(np.isfinite(p95_l7) and p95_l7 <= CONSISTENCY_P95_BAND)
    band_ok = l6_band_ok and l7_band_ok
    # Decision 15: a **non-inferiority margin**, not an ordering.  0.030 versus
    # 0.031 is noise and must not decide a contribution.
    margin_ok = bool(np.isfinite(p95_l7 - p95_l6) and (p95_l7 - p95_l6) <= NON_INFERIORITY_MARGIN)

    # Criterion 14: GAFlowNet's control is only valid if c_t reached exactly 0.
    gafn_ct = report.comparisons.get("gaflownet", {}).get("final_c_t", [])
    gafn_ok = bool(gafn_ct) and all(v == 0.0 for v in gafn_ct)

    # Criterion 16 and decision 14: at least one FAIL trajectory per instance.
    # It was measured and then never read, so a consistency band computed over a
    # frozen set that happened to contain no dead end would have passed silently
    # — and `ℰ(FAIL) = -log r_fail` is the far tail the separate line exists for.
    fail_ok = bool(report.consistency) and all(
        row["fail_covered_everywhere"] for row in report.consistency
    )

    # **Exit criterion 15**, which was written into the plan and implemented
    # nowhere: at the adopted (N, β), L4 and L5 must reach decision 6's TV level
    # **on the main suite**. Decision 6's own check runs on 5 tuning instances;
    # the matrix runs 20 main instances through one conditional `logZ` head, and
    # passing the first does not imply the second. Without it a matrix in which
    # the machinery failed on the scored suite still emitted a verdict.
    threshold = report.comparisons.get("tv_threshold", 0.10)
    machinery = {
        arm: report.comparisons.get(arm, {}).get("final_tv_mean")
        for arm in ("l4_tb", "l5_subtb")
    }
    machinery_ok = bool(machinery) and all(
        v is not None and np.isfinite(v) and v <= threshold for v in machinery.values()
    )

    # **Three outcomes, not two.** `inconclusive` is not a polite word for `False`:
    # criterion 12 exists because a null result that cannot distinguish "no
    # effect" from "no budget" is the worst outcome available, and the same logic
    # covers an instrument that did not work at this `N`. A failed L6 band, a
    # GAFlowNet whose `c_t` never reached 0, an uncovered FAIL line, or L4/L5
    # missing criterion 15 are all statements about the harness. Only the
    # comparisons and L7's own consistency speak to C3.
    instrument = {
        "machinery_criterion_15": machinery_ok,
        "l6_consistency_band": l6_band_ok,
        "gaflownet_intrinsic_reached_zero": gafn_ok,
        "fail_coverage_complete": fail_ok,
    }
    instrument_ok = all(instrument.values())

    if not admissible:
        supported, outcome = None, "inadmissible"
    elif not instrument_ok:
        supported, outcome = None, "inconclusive"
    else:
        supported = bool(beats_l6 and beats_gafn and l7_band_ok and margin_ok)
        outcome = "supported" if supported else "not_supported"

    return {
        "contribution_3_supported": supported,
        "outcome": outcome,
        "instrument_ok": instrument_ok,
        "instrument": instrument,
        "machinery_final_tv": machinery,
        "admissible": admissible,
        "inadmissible_reason": report.spec.get("inadmissible_reason"),
        "beats_capacity_matched_l6": beats_l6,
        "beats_capacity_matched_gaflownet": beats_gafn,
        "consistency_band_passed": band_ok,
        "non_inferiority_margin_passed": margin_ok,
        "fail_coverage_complete": fail_ok,
        "p95_l6": p95_l6,
        "p95_l7": p95_l7,
        "gaflownet_intrinsic_reached_zero": gafn_ok,
        "rule": (
            "Contribution 3 is supported iff L7 beats capacity-matched L6 AND "
            "capacity-matched GAFlowNet on the one-sided 95% upper bound of "
            "TV_L7 - TV_control (decision 20), with L7 inside decision 13's "
            "consistency band and within decision 15's non-inferiority margin. "
            "If either comparison fails, C3 is not supported and the thesis "
            "consolidates on Contribution 1 (v1.2 §9). Three outcomes, not two: "
            "a run that is not admissible returns null; a run whose INSTRUMENT "
            "did not work -- L4/L5 missing criterion 15 on the main suite, "
            "vanilla LED-GFN outside its own consistency band, GAFlowNet's c_t "
            "not reaching 0, or an uncovered FAIL line -- returns null and "
            "'inconclusive', never False. Those are statements about the "
            "harness, and criterion 12's reasoning is that a null which cannot "
            "distinguish 'no effect' from 'no budget' is the worst outcome "
            "available."
        ),
        "bootstrap_resolution": (
            "The hierarchical bootstrap resamples 3 outer clusters (the frozen "
            "seed set), so 1 in 9 resamples draws a single seed three times and "
            "the one-sided 95% upper bound sits close to the worst seed's mean. "
            "10,000 resamples is the inner resolution, not the outer: with three "
            "seeds the test is close to 'wins on all three'. That is the "
            "conservative direction and it is what the ACL 2018 protocol's "
            "three-seed requirement buys; it is stated here so no reader infers "
            "more resolution than three clusters carry."
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
    lines = ["arm                    live   final TV   best-of-K   gold@K   traj-to-thr"]
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
            live = report.logs[name][SEEDS[0]].live_capacity if report.logs.get(name) else 0
            lines.append(
                f"{name:<20}{live:>7}   {row['final_tv_mean']:.4f}   "
                f"{float(np.mean(row['best_of_k_utility'])):>9.4f}   "
                f"{float(np.mean(row['gold_exact_match'])):>6.3f}   {shown}"
            )
    return "\n".join(lines)
