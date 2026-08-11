"""The one trainer every arm shares, and the tensor batch it hands them.

**One trainer, nine arms, and the sharing is the point.** Plan §5.1 requires the
graph encoder, candidate pool, action space, terminal utility, checker, training
examples, reward budget and seeds to be matched across every row of the learning
comparison — otherwise the table measures engineering effort rather than
objectives. The cheapest way to guarantee that is for there to be exactly one
place where any of it can be set.

**What is frozen here, and why each would otherwise confound Gate 2**

============================  =================================================
ε and its schedule (G8)       an ε tuned per arm turns a comparison of
                              objectives into a comparison of exploration
optimiser, lr, batch, clip    same, one level down (decision 23)
capacity (G4, decision 11)    Contribution 3 adds parameters; a win that
                              vanishes under matching is not a win
seeds ``{13, 42, 7}``         the significance protocol replicates over these
trajectory budget ``N``       fix F12's primary metric is TV *at fixed budget*
============================  =================================================

**Training meters nothing** (G9, decision 16). ``checker_budget = 32`` is a
*per-query inference* budget; a training rollout on the lattice spends **zero**
``terminal_checks`` because Phase 2 precomputed validity and cached ``U``. A
trainer that opened a ``query_scope`` would exhaust the budget in the first epoch
and would be metering the wrong axis entirely.

**A ``Batch`` is tensors and nothing else.** Exit criterion 6 forbids any module
under ``learners/`` from importing a ``StateGraph``, a ``LatticeInstance`` or an
atom id. Passing an ``Environment`` to a loss would satisfy the letter of that
and defeat its purpose, since the loss could reach the graph through it. So the
trainer flattens each rollout batch into padded tensors indexed by *position in
the trajectory*, and the objectives see only those.

**The one identity every objective in this module is written against.** With

.. code-block:: text

    A[i] = Σ_{k<i} log P_F(k)      B[i] = Σ_{k<i} log P_B(k)
    g[i] = log F(s_i) − A[i] + B[i]

the balance residual between any two positions is ``g[i] − g[j]`` — no partial
sums at the call site, and no chance of TB and SubTB disagreeing about what a
sub-trajectory is. TB is ``(g[0] − g[L])²``; SubTB(λ) is the λ-weighted sum over
all ``i < j``; DB is the ``j = i+1`` case. Positions run ``0 … L`` where ``L`` is
the number of transitions **including the terminating one**, ``log F(s_0) = log
Z_θ``, and the boundary at ``L`` is set by the objective: ``log R(x)`` for the
balance family, ``0`` for LED, whose reward is carried by ``Σ φ`` instead.

**[ANALYSIS] The shared protocol is uniformly suboptimal, and that is what makes
the comparison readable.** A per-arm sweep sounds fairer and is not: with nine
arms, three seeds and a wall-clock ceiling, a sweep large enough to be fair would
consume the budget the comparison needs, and a sweep small enough to afford would
favour whichever arm suited the grid. The write-up must say these are not tuned
results and no arm's number is its best achievable.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch

from graft.core import obligations as ob
from graft.setgen.features import SyntheticFeaturizer
from graft.setgen.policy import (
    DeficitHead,
    LogZHead,
    Policy,
    PotentialHead,
    StateFlowHead,
    capacity,
)
from graft.setgen.rollout import Trajectories, sample_trajectories
from graft.synth.enumerate import StateGraph, reachable_states
from graft.synth.exact import Target, policy_distribution, target_distribution, tv

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.synth.lattice import LatticeInstance

__all__ = [
    "TrainSpec",
    "Environment",
    "Batch",
    "Arm",
    "Trainer",
    "TrainLog",
    "SEEDS",
]

#: The frozen seed set (``CLAUDE.md`` §6, exit criterion 10).  **[EVIDENCE]** the
#: ACL 2018 significance protocol requires multiple seeds; the *set* is frozen so
#: no arm is ever replicated over a different one.
SEEDS: tuple[int, ...] = (13, 42, 7)

N_DEFICIT = len(ob.DEFICIT_COMPONENTS)


class TrainSpec:
    """Everything frozen across arms (§6).  Constructed once, shared by all."""

    __slots__ = (
        "n_trajectories", "batch_size", "checkpoints", "lr", "epsilon",
        "grad_clip", "hidden", "depth", "seed", "device", "dtype",
        "subtb_lambda", "led_lr", "led_dropout", "led_iters", "gafn_c0",
        "gafn_tail", "lambda_aux", "wall_clock_ceiling",
    )

    def __init__(
        self,
        *,
        n_trajectories: int,
        batch_size: int = 32,       # decision 23
        checkpoints: int = 50,      # C = 50, frozen by Phase-2 criterion 11
        lr: float = 3e-4,           # decision 23
        epsilon: float = 0.05,      # decision 10 — constant, no decay
        grad_clip: float = 1.0,     # decision 23
        hidden: int = 64,
        depth: int = 2,
        seed: int = SEEDS[0],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        # --- SubTB(λ) ----------------------------------------------------
        # **Not in §6, and this build found that rather than filled it.**
        # Decision 23 freezes the optimiser, lr, batch and clipping; λ is a
        # per-objective hyperparameter of exactly that kind and is listed
        # nowhere in the plan.  0.9 is the value SubTB (ICML 2023) carries
        # through its own experiments, so it is at least the published default
        # rather than a number invented here — but applying it to *this*
        # environment is **[ANALYSIS]**, and it belongs in §6 before Gate 2 is
        # read.  Recorded in PHASE3_DECISIONS.md as an open §6 item.
        subtb_lambda: float = 0.9,
        # --- LED-GFN, decision 23a, from Appendix C -----------------------
        led_lr: float = 1e-3,
        led_dropout: float = 0.10,
        led_iters: int = 8,
        # --- GAFlowNet, decisions 19 and 19b ------------------------------
        gafn_c0: float = 0.1,
        gafn_tail: float = 0.8,
        # --- L7b, decision 26 ---------------------------------------------
        lambda_aux: float = 0.1,
        wall_clock_ceiling: float | None = None,
    ) -> None:
        if n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not 0.0 < subtb_lambda <= 1.0:
            raise ValueError(f"subtb_lambda must be in (0, 1], got {subtb_lambda}")
        if not 0.0 < gafn_tail <= 1.0:
            raise ValueError(f"gafn_tail must be in (0, 1], got {gafn_tail}")
        self.n_trajectories = int(n_trajectories)
        self.batch_size = int(batch_size)
        self.checkpoints = int(checkpoints)
        self.lr = float(lr)
        self.epsilon = float(epsilon)
        self.grad_clip = float(grad_clip)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self.subtb_lambda = float(subtb_lambda)
        self.led_lr = float(led_lr)
        self.led_dropout = float(led_dropout)
        self.led_iters = int(led_iters)
        self.gafn_c0 = float(gafn_c0)
        self.gafn_tail = float(gafn_tail)
        self.lambda_aux = float(lambda_aux)
        self.wall_clock_ceiling = wall_clock_ceiling

    def replace(self, **overrides: Any) -> "TrainSpec":
        kwargs = {name: getattr(self, name) for name in TrainSpec.__slots__}
        kwargs.update(overrides)
        return TrainSpec(**kwargs)

    def shared_protocol(self) -> dict[str, Any]:
        """The values exit criterion 9 asserts identical across every arm.

        ``hidden`` is deliberately **not** here: capacity matching moves it per
        arm on purpose (G4), and asserting it identical would make decision 11
        unimplementable.  What must match is everything that shapes *training*.
        """
        return {
            "n_trajectories": self.n_trajectories,
            "batch_size": self.batch_size,
            "checkpoints": self.checkpoints,
            "lr": self.lr,
            "epsilon": self.epsilon,
            "grad_clip": self.grad_clip,
            "depth": self.depth,
            "subtb_lambda": self.subtb_lambda,
            "led_lr": self.led_lr,
            "led_dropout": self.led_dropout,
            "led_iters": self.led_iters,
            "gafn_c0": self.gafn_c0,
            "gafn_tail": self.gafn_tail,
            "lambda_aux": self.lambda_aux,
        }


class Environment:
    """One instance, enumerated once, with everything policy-independent cached.

    Phase-2 G2 again: the state graph, the target and the per-state deficits do
    not depend on θ, so they are built once and reused by every arm and every
    checkpoint.  Rebuilding them per arm would dominate the run and would also
    risk two arms being scored against subtly different targets.
    """

    __slots__ = ("instance", "graph", "target", "log_reward", "log_r_range")

    def __init__(self, instance: "LatticeInstance", graph: StateGraph | None = None) -> None:
        self.instance = instance
        self.graph = graph if graph is not None else reachable_states(instance, instance.cfg)
        self.target: Target = target_distribution(instance, instance.cfg, graph=self.graph)

        # log R per *state*: β·U at a valid stop, −inf everywhere else so a
        # non-terminal read is a visible NaN rather than a plausible number.
        # Materialised because every objective needs it, and recomputing U
        # inside the training loop would be the most expensive thing in it.
        log_r = np.full(self.graph.n_states, -math.inf, dtype=np.float64)
        log_r[self.graph.terminal_ix] = instance.cfg.beta * self.target.u
        self.log_reward = log_r

        # Decision 13's normaliser: the **per-instance valid-terminal log R
        # range**.  Measured on the frozen main suite it sits at 6.93–7.79,
        # stable to ~12%; the per-terminal |log R| that R2 proposed swung the
        # effective tolerance 216× inside a single instance.
        values = self.log_reward[self.graph.terminal_ix]
        self.log_r_range = float(values.max() - values.min()) if values.size else 0.0

    def fingerprints(self) -> dict[str, str]:
        """Exit criterion 25: which environment a number was computed against."""
        from graft.synth.enumerate import environment_fingerprint

        return {
            "environment": environment_fingerprint(self.instance, self.graph),
            "target": self.target.fingerprint(),
        }


class Batch:
    """One rollout batch, flattened to padded tensors indexed by position.

    Every array is ``[n, …]`` over trajectories.  ``L`` is the longest
    trajectory's transition count **including its terminating transition**, so
    flow positions run ``0 … L`` and transition positions ``0 … L−1``.

    ``valid[r, k]`` marks a real transition.  Padding is *not* left to be
    harmless by luck — ``log_pf`` and ``log_pb`` are zero there so the cumulative
    sums stay correct past the end, and every loss masks explicitly on top of
    that.  Relying on either alone has already produced one silent bug in this
    codebase.
    """

    __slots__ = (
        "n", "steps", "log_z", "flow_raw", "log_pf", "log_pb",
        "log_reward", "n_trans", "valid", "is_fail", "state_repr", "action_repr",
        "delta", "terminal_deficit", "aux_pred", "aux_mask", "potential",
        "log_r_range", "_cum_pf", "_cum_pb", "_pos", "_is_terminal_pos",
    )

    def __init__(
        self,
        *,
        log_z: torch.Tensor,
        flow_raw: torch.Tensor,
        log_pf: torch.Tensor,
        log_pb: torch.Tensor,
        log_reward: torch.Tensor,
        n_trans: torch.Tensor,
        valid: torch.Tensor,
        is_fail: torch.Tensor,
        state_repr: torch.Tensor,
        action_repr: torch.Tensor,
        delta: torch.Tensor,
        terminal_deficit: torch.Tensor,
        log_r_range: float,
        aux_pred: torch.Tensor | None = None,
        aux_mask: torch.Tensor | None = None,
        potential: torch.Tensor | None = None,
    ) -> None:
        self.n = int(log_reward.shape[0])
        self.steps = int(log_pf.shape[1])
        self.log_z = log_z
        self.flow_raw = flow_raw
        self.log_pf = log_pf
        self.log_pb = log_pb
        self.log_reward = log_reward
        self.n_trans = n_trans
        self.valid = valid
        self.is_fail = is_fail
        self.state_repr = state_repr
        self.action_repr = action_repr
        self.delta = delta
        self.terminal_deficit = terminal_deficit
        self.log_r_range = float(log_r_range)
        self.aux_pred = aux_pred
        self.aux_mask = aux_mask
        self.potential = potential

        zero = torch.zeros((self.n, 1), dtype=log_pf.dtype, device=log_pf.device)
        self._cum_pf = torch.cat([zero, torch.cumsum(log_pf, dim=1)], dim=1)
        self._cum_pb = torch.cat([zero, torch.cumsum(log_pb, dim=1)], dim=1)
        self._pos = torch.arange(self.steps + 1, device=log_pf.device)
        self._is_terminal_pos = self._pos.unsqueeze(0) == n_trans.unsqueeze(1)

    def g(self, terminal: torch.Tensor) -> torch.Tensor:
        """``g[i] = log F(s_i) − A[i] + B[i]``, with ``log F`` at ``L`` supplied.

        The boundary is an argument because it is the whole difference between
        the balance family and LED: TB, SubTB and augmented TB put ``log R(x)``
        there; LED puts ``0`` and carries the reward in ``Σ φ`` instead.  Writing
        it once, with the choice exposed, is what keeps the two conventions from
        drifting apart in separate files.
        """
        flow = torch.where(self._is_terminal_pos, terminal.unsqueeze(1), self.flow_raw)
        return flow - self._cum_pf + self._cum_pb

    @property
    def positions(self) -> torch.Tensor:
        """``[L+1]`` — the flow positions, for length-dependent weights."""
        return self._pos

    def pair_mask(self) -> torch.Tensor:
        """``[n, L+1, L+1]`` — true where ``i < j <= n_trans``, for SubTB."""
        i = self._pos.view(1, -1, 1)
        j = self._pos.view(1, 1, -1)
        return (i < j) & (j <= self.n_trans.view(-1, 1, 1))

    def energy(self) -> torch.Tensor:
        """``ℰ(x) = −log R(x)`` — what LED's potentials must sum to."""
        return -self.log_reward


class Arm:
    """One learner: a loss over a :class:`Batch`, and the switches around it.

    ``delta_d`` is decision 19a's routing, and it is the *only* thing separating
    L6 from L7.  It is a constructor flag rather than something the loss reads,
    because a loss that could turn it on could turn it on for the control.
    """

    __slots__ = (
        "name", "loss", "delta_d", "needs_flow", "needs_potential", "needs_aux",
        "trains_potential", "supervised", "hidden",
    )

    def __init__(
        self,
        name: str,
        loss: Callable[..., torch.Tensor],
        *,
        delta_d: bool = False,
        needs_flow: bool = False,
        needs_potential: bool = False,
        needs_aux: bool = False,
        trains_potential: bool = False,
        supervised: bool = False,
        hidden: int | None = None,
    ) -> None:
        self.name = name
        self.loss = loss
        self.delta_d = bool(delta_d)
        self.needs_flow = bool(needs_flow)
        self.needs_potential = bool(needs_potential)
        self.needs_aux = bool(needs_aux)
        self.trains_potential = bool(trains_potential)
        self.supervised = bool(supervised)
        self.hidden = hidden

    def at_hidden(self, hidden: int) -> "Arm":
        """The same arm at a different width — how capacity matching is applied."""
        return Arm(
            self.name,
            self.loss,
            delta_d=self.delta_d,
            needs_flow=self.needs_flow,
            needs_potential=self.needs_potential,
            needs_aux=self.needs_aux,
            trains_potential=self.trains_potential,
            supervised=self.supervised,
            hidden=hidden,
        )


class TrainLog:
    """Checkpointed exact TV per environment, plus what the run was."""

    __slots__ = (
        "arm", "seed", "trajectories", "tv_mean", "tv_per_env", "losses",
        "capacity", "hidden", "c_t", "final_loss", "wall_clock", "protocol",
        "fingerprints",
    )

    def __init__(self, arm: str, seed: int) -> None:
        self.arm = arm
        self.seed = seed
        self.trajectories: list[int] = []
        self.tv_mean: list[float] = []
        self.tv_per_env: list[list[float]] = []
        self.losses: list[float] = []
        self.c_t: list[float] = []
        self.capacity = 0
        self.hidden = 0
        self.final_loss = float("nan")
        self.wall_clock = 0.0
        self.protocol: dict[str, Any] = {}
        self.fingerprints: list[dict[str, str]] = []

    @property
    def final_tv(self) -> float:
        return self.tv_mean[-1] if self.tv_mean else float("nan")

    @property
    def final_c_t(self) -> float:
        """Criterion 14 reads this and requires **exactly** 0 for GAFlowNet."""
        return self.c_t[-1] if self.c_t else 0.0

    def trajectories_to(self, threshold: float) -> int | None:
        """Decision 21's secondary.  ``None`` is **censored**, never the budget."""
        for spent, value in zip(self.trajectories, self.tv_mean):
            if value <= threshold:
                return spent
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "capacity": self.capacity,
            "hidden": self.hidden,
            "trajectories": list(self.trajectories),
            "tv_mean": list(self.tv_mean),
            "tv_per_env": [list(row) for row in self.tv_per_env],
            "final_tv": self.final_tv,
            "final_loss": self.final_loss,
            "final_c_t": self.final_c_t,
            "wall_clock_s": self.wall_clock,
            "protocol": dict(self.protocol),
            "fingerprints": list(self.fingerprints),
        }


class Trainer:
    """Trains one arm over one suite, checkpointing exact TV.

    ``logZ_θ`` is conditioned on the instance (G6), so a single model spans the
    whole suite and TV is computed **per instance and averaged unweighted**
    (decision 8) — weighting by terminal count would let the largest lattice
    dominate a number meant to describe the method.
    """

    def __init__(self, arm: Arm, envs: Sequence[Environment], spec: TrainSpec) -> None:
        if not envs:
            raise ValueError("no environments to train on")
        self.arm = arm
        self.envs = list(envs)
        self.spec = spec
        self.hidden = int(arm.hidden if arm.hidden is not None else spec.hidden)

        torch.manual_seed(spec.seed)
        self.rng = np.random.default_rng(spec.seed)

        state_dim, action_dim = SyntheticFeaturizer.dims(envs[0].instance, envs[0].graph)
        for env in self.envs[1:]:
            if SyntheticFeaturizer.dims(env.instance, env.graph) != (state_dim, action_dim):
                raise ValueError(
                    "instances disagree on feature width; one model cannot span them"
                )
        self.state_dim, self.action_dim = state_dim, action_dim

        self.policy = Policy(state_dim, action_dim, self.hidden, spec.depth).to(spec.dtype)
        self.logz = LogZHead(state_dim, self.hidden, spec.depth).to(spec.dtype)
        self.flow = (
            StateFlowHead(self.hidden, spec.depth).to(spec.dtype) if arm.needs_flow else None
        )
        self.potential = (
            PotentialHead(state_dim, action_dim, self.hidden, spec.depth).to(spec.dtype)
            if arm.needs_potential
            else None
        )
        self.aux = (
            DeficitHead(self.hidden, N_DEFICIT, spec.depth).to(spec.dtype)
            if arm.needs_aux
            else None
        )

        self.featurizers = [
            SyntheticFeaturizer(
                env.instance, env.graph, self.policy, env.instance.cfg,
                delta_d=arm.delta_d, device=spec.device, dtype=spec.dtype,
            )
            for env in self.envs
        ]

        # G9 / decision 16: no ledger, no query scope, no terminal_checks.  The
        # attribute exists so the assertion at ``train`` has something to read.
        self.ledger = None

        self.modules = [
            m for m in (self.policy, self.logz, self.flow, self.potential, self.aux)
            if m is not None
        ]
        self.capacity = capacity(*self.modules)

        # The potential trains under its own optimiser at LED's lr (decision
        # 23a); everything else under the shared protocol (decision 23).
        main = [m for m in (self.policy, self.logz, self.flow, self.aux) if m is not None]
        self.main_params = [p for m in main for p in m.parameters()]
        self.optimiser = torch.optim.Adam(self.main_params, lr=spec.lr)
        self.potential_params = (
            list(self.potential.parameters()) if self.potential is not None else []
        )
        self.potential_optimiser = (
            torch.optim.Adam(self.potential_params, lr=spec.led_lr)
            if arm.trains_potential
            else None
        )

    # -- capacity ----------------------------------------------------------

    @staticmethod
    def capacity_of(
        arm: Arm, state_dim: int, action_dim: int, hidden: int, depth: int
    ) -> int:
        """What ``arm`` would cost at ``hidden``, without training anything.

        The counterfactual capacity matching needs (G4): ``match_capacity``
        searches over widths, and building a full trainer per candidate width
        would build a featurizer per environment each time.
        """
        mods: list[torch.nn.Module] = [
            Policy(state_dim, action_dim, hidden, depth),
            LogZHead(state_dim, hidden, depth),
        ]
        if arm.needs_flow:
            mods.append(StateFlowHead(hidden, depth))
        if arm.needs_potential:
            mods.append(PotentialHead(state_dim, action_dim, hidden, depth))
        if arm.needs_aux:
            mods.append(DeficitHead(hidden, N_DEFICIT, depth))
        return capacity(*mods)

    # -- evaluation --------------------------------------------------------

    @torch.no_grad()
    def exact_tv(self) -> list[float]:
        """Exact TV per environment, via Phase 2's evaluator, unchanged."""
        return [
            tv(policy_distribution(feat, env.graph), env.target.p_star)
            for env, feat in zip(self.envs, self.featurizers)
        ]

    # -- training ----------------------------------------------------------

    def train(self, *, progress: Callable[[int, float], None] | None = None) -> TrainLog:
        if self.ledger is not None:  # pragma: no cover - defensive
            raise RuntimeError(
                "the trainer must run with ledger=None (G9, decision 16): "
                "checker_budget is a per-query inference budget and training "
                "rollouts would exhaust it in the first epoch"
            )

        log = TrainLog(self.arm.name, self.spec.seed)
        log.capacity = self.capacity
        log.hidden = self.hidden
        log.protocol = self.spec.shared_protocol()
        log.fingerprints = [env.fingerprints() for env in self.envs]

        every = max(1, self.spec.n_trajectories // self.spec.checkpoints)
        spent = 0
        started = time.perf_counter()
        self._checkpoint(log, spent)

        while spent < self.spec.n_trajectories:
            index = int(self.rng.integers(len(self.envs)))
            env, feat = self.envs[index], self.featurizers[index]
            size = min(self.spec.batch_size, self.spec.n_trajectories - spent)

            value = self._step(env, feat, size, spent)

            before = spent
            spent += size          # FAIL rollouts count too (decision 3)
            log.losses.append(value)
            log.c_t.append(self.intrinsic_coefficient(spent))
            if spent // every != before // every:
                self._checkpoint(log, spent)
                if progress is not None:
                    progress(spent, log.tv_mean[-1])
            if (
                self.spec.wall_clock_ceiling is not None
                and time.perf_counter() - started > self.spec.wall_clock_ceiling
            ):
                break

        if not log.trajectories or log.trajectories[-1] != spent:
            self._checkpoint(log, spent)
        log.final_loss = log.losses[-1] if log.losses else float("nan")
        log.wall_clock = time.perf_counter() - started
        return log

    def _step(
        self, env: Environment, feat: SyntheticFeaturizer, size: int, spent: int
    ) -> float:
        if self.arm.supervised:
            # The logits are computed here, not in the loss: a learner that could
            # call the featurizer could reach the graph through it (criterion 6).
            states, targets = self.gold_batch(env, feat, size)
            loss = self.arm.loss((feat.logits(states), targets), self, spent)
        else:
            traj = sample_trajectories(feat, env.graph, size, self.rng, self.spec.epsilon)
            batch = self.build_batch(env, feat, traj)
            if self.potential_optimiser is not None:
                self._train_potential(batch)
                # Detached: LED's Eq. 4 treats φ as the *decomposed energy*, and
                # letting the GFlowNet loss backpropagate into it would let the
                # policy reshape its own reward.
                batch.potential = self.compute_potential(batch).detach()
            elif self.potential is not None:
                batch.potential = self.compute_potential(batch)
            loss = self.arm.loss(batch, self, spent)

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_params, self.spec.grad_clip)
        self.optimiser.step()
        return float(loss.detach())

    def _train_potential(self, batch: Batch) -> None:
        """LED's inner loop: ``N = 8`` decomposition iterations (decision 23a).

        **[EVIDENCE]** LED-GFN Eq. 5 trains the potential with a squared loss on
        ``Σ φ − ℰ(x)`` and uses **dropout** as the variance regulariser — one
        loss, not the "two regulariser coefficients" R4 assumed.  Fresh masks
        every iteration, which is what makes the dropout a regulariser rather
        than a fixed sub-sampling of the trajectory.
        """
        from graft.setgen.learners.l6_led import decomposition_loss

        for _ in range(self.spec.led_iters):
            loss = decomposition_loss(
                self.compute_potential(batch), batch, self.spec.led_dropout, self.rng
            )
            self.potential_optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.potential_params, self.spec.grad_clip)
            self.potential_optimiser.step()

    def compute_potential(self, batch: Batch) -> torch.Tensor:
        """``φ_θ`` per transition, ``[n, L]``."""
        return self.potential(batch.state_repr, batch.action_repr)

    def intrinsic_coefficient(self, spent: int) -> float:
        """``c_t`` (decisions 19 and 19b) — linear decay with a **zero tail**.

        ``c_t = c₀·max(0, 1 − t/(0.8N))``, so the last 20% of training runs at
        exactly zero.  GAFlowNet's Theorem 1 needs ``c_t → 0`` *and* a vanishing
        loss; the tail is what lets a run approach the first without asserting
        the second, and criterion 14 checks the final checkpoint reads 0.
        """
        if not self.arm.name.startswith("gaflownet"):
            return 0.0
        horizon = self.spec.gafn_tail * self.spec.n_trajectories
        if horizon <= 0:
            return 0.0
        return float(self.spec.gafn_c0 * max(0.0, 1.0 - spent / horizon))

    def _checkpoint(self, log: TrainLog, spent: int) -> None:
        per_env = self.exact_tv()
        log.trajectories.append(spent)
        log.tv_per_env.append(per_env)
        log.tv_mean.append(float(np.mean(per_env)))

    # -- batch construction ------------------------------------------------

    def build_batch(
        self, env: Environment, feat: SyntheticFeaturizer, traj: Trajectories
    ) -> Batch:
        """Flatten rollouts into the padded tensors every objective reads.

        **The terminating transition is a first-class step**, not something
        bolted on at the end.  A valid rollout terminates by choosing ``STOP``; a
        dead-ended one terminates because nothing is legal, with probability 1.
        Both get a slot, so ``P_F(STOP | x)`` participates in the balance
        conditions and a ``FAIL`` trajectory is not silently one transition
        shorter than the states it visited.
        """
        graph = env.graph
        n = len(traj)
        n_atoms = graph.n_atoms
        lengths = traj.lengths.astype(np.int64)
        n_trans = lengths + 1
        steps = int(n_trans.max())
        device, dtype = self.spec.device, self.spec.dtype

        def T(a: np.ndarray, dt: torch.dtype = torch.long) -> torch.Tensor:
            return torch.as_tensor(a, dtype=dt, device=device)

        pad = np.full((n, 1), -1, dtype=np.int64)
        states = np.concatenate([traj.states, pad], axis=1)
        acts = np.concatenate([traj.actions, pad], axis=1)
        rows = np.arange(n, dtype=np.int64)
        last = states[rows, lengths]                       # the terminal state
        idx = np.arange(steps, dtype=np.int64)[None, :]
        k = lengths[:, None]

        # Parent of each transition; padding repeats the terminal state so every
        # feature lookup is finite and nothing NaNs behind a mask.
        parent = np.where(idx <= k, states[:, :steps], last[:, None])
        child = np.where(idx + 1 <= k, states[:, 1 : steps + 1], last[:, None])
        action = np.where(idx < k, acts[:, :steps], n_atoms)
        valid_np = idx <= k
        is_fail = traj.is_fail

        # Flow positions 0 … steps.  The state at position i is states[i] up to
        # the terminal; the boundary at n_trans is supplied by the objective.
        fpos = np.arange(steps + 1, dtype=np.int64)[None, :]
        flow_state = np.where(fpos <= k, states[:, : steps + 1], last[:, None])

        parent_t = T(parent).reshape(-1)
        action_t = T(action).reshape(-1)

        # Only LED's potential consumes these, and ``action_reprs`` materialises
        # an ``[n·L, n_atoms+1, action_dim]`` block to gather one row from.  Arms
        # without a potential head pay nothing for it.
        state_repr = action_repr = None
        if self.potential is not None:
            state_repr = feat.state_repr(parent_t).view(n, steps, -1)
            action_repr = feat.action_reprs(parent_t)[
                torch.arange(parent_t.shape[0], device=parent_t.device), action_t
            ].view(n, steps, -1)

        # -- log P_F --------------------------------------------------------
        # Dead ends are never asked for logits: they have no legal ADD and a
        # masked STOP, so log_softmax over their row is all −inf and any read
        # would be NaN.  Their termination is forced, hence log P_F = 0.
        need = valid_np.copy()
        need[rows, lengths] = ~is_fail
        log_pf = torch.zeros((n, steps), dtype=dtype, device=device)
        want = np.unique(parent[need])
        if want.size:
            logits = feat.logits(T(want))
            lookup = np.full(graph.n_states, -1, dtype=np.int64)
            lookup[want] = np.arange(want.shape[0], dtype=np.int64)
            sel_rows, sel_cols = np.nonzero(need)
            picked = logits[
                T(lookup[parent[sel_rows, sel_cols]]), T(action[sel_rows, sel_cols])
            ]
            log_pf = log_pf.index_put((T(sel_rows), T(sel_cols)), picked)

        # -- log P_B (constant in θ, decision 18) ---------------------------
        # Uniform over **removable** atoms, read off the enumerated in-degrees:
        # an atom that another selected atom references has no parent to be
        # removed to, so the in-degree is exactly the removable count under the
        # closure rule.  Never reimplemented here.
        log_pb_np = np.zeros((n, steps), dtype=np.float64)
        adds = valid_np & (idx < k)
        if adds.any():
            log_pb_np[adds] = -np.log(graph.indegree[child[adds]])
        log_pb = torch.as_tensor(log_pb_np, dtype=dtype, device=device)

        # -- flows ----------------------------------------------------------
        log_z = self.logz(feat.instance_repr()).squeeze(0)
        h_flow = None
        if self.flow is not None or self.aux is not None:
            h_flow = self.policy.encode(feat.state_repr(T(flow_state).reshape(-1)))
        if self.flow is not None:
            flow_raw = self.flow(h_flow).view(n, steps + 1)
        else:
            flow_raw = torch.zeros((n, steps + 1), dtype=dtype, device=device)
        # F(s_0) = Z is an identity, not something to be learned twice.
        flow_raw = torch.cat([log_z.view(1, 1).expand(n, 1), flow_raw[:, 1:]], dim=1)

        # -- rewards, deficits ----------------------------------------------
        log_reward = torch.as_tensor(
            np.where(
                is_fail,
                math.log(env.instance.cfg.r_fail),
                env.log_reward[np.where(is_fail, 0, traj.terminal)],
            ),
            dtype=dtype,
            device=device,
        )
        delta = (
            feat.deficits(parent_t) - feat.deficits(T(child).reshape(-1))
        ).view(n, steps, N_DEFICIT)
        terminal_deficit = feat.deficits(T(last))

        aux_pred = aux_mask = None
        if self.aux is not None:
            aux_pred = self.aux(h_flow).view(n, steps + 1, N_DEFICIT)
            aux_mask = T(fpos <= k, torch.bool)

        return Batch(
            log_z=log_z,
            flow_raw=flow_raw,
            log_pf=log_pf,
            log_pb=log_pb,
            log_reward=log_reward,
            n_trans=T(n_trans),
            valid=T(valid_np, torch.bool),
            is_fail=T(is_fail, torch.bool),
            state_repr=state_repr,
            action_repr=action_repr,
            delta=delta,
            terminal_deficit=terminal_deficit,
            log_r_range=env.log_r_range,
            aux_pred=aux_pred,
            aux_mask=aux_mask,
        )

    # -- supervision (L1, L2) ----------------------------------------------

    def gold_batch(
        self, env: Environment, feat: SyntheticFeaturizer, size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(state_ix, target_action)`` along gold construction paths.

        L1 samples a fresh closure-legal order per example — stepwise supervision
        of the *set*.  L2 walks the one canonical order and repeats it — imitation
        of a single sequence, which is the distinction plan §5.1 draws between
        "supervised stepwise" and "canonical set imitation".

        ``STOP`` at the gold terminal is a supervised action like any other: a
        policy taught to build the gold set and never taught to stop would walk
        off it at the last step.
        """
        canonical = self.arm.name == "l2_imitation"
        states: list[int] = []
        targets: list[int] = []
        for _ in range(size):
            path, actions = feat.gold_path(None if canonical else self.rng)
            states.extend(int(s) for s in path[:-1])
            targets.extend(int(a) for a in actions)
            states.append(int(path[-1]))
            targets.append(env.graph.n_atoms)   # STOP
        return (
            torch.as_tensor(states, dtype=torch.long, device=self.spec.device),
            torch.as_tensor(targets, dtype=torch.long, device=self.spec.device),
        )
