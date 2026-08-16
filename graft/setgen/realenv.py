"""P9.4 — the incremental environment and sampler (G1).

**Phase 2's enumerated state graph does not survive contact with real pools, and
Phase 3's losses must not notice.**  Plan §3.4 gives the arithmetic: ~50
candidates at sets ≤ 16 is on the order of **10¹³ subsets**, so "enumeration is
impossible; incremental construction costs ~16 × 50 = 800 action scores".  A pool
at ``pool_cap = 64`` is past every enumerable bound.

So this module rebuilds the *environment layer* — and only that layer — over
Phase 1's :class:`~graft.core.incremental.IncrementalChecker` and
``graft.core.masks``, which were built for exactly this and have been sitting
unused by the training path since Phase 1.  What it emits is
:class:`~graft.setgen.trainer.Batch`, byte-for-byte the object Phase 3's losses
already consume, so **every file in** ``graft/setgen/learners/`` **stays
untouched** (exit criterion 1).  That is fix F6's payoff and the reason Phase 3
was built against an adapter rather than against a state graph.

**What is genuinely lost, and stays lost.**  Exact TV at checkpoints.  It exists
only where the state space can be enumerated, and no approximation of it is
computed here — an estimated TV reported beside Gate 2's exact one would invite
exactly the comparison that is invalid.  Checkpoint monitoring is training loss
plus a small dev best-of-K probe, recorded as *monitoring, never selection*:
selection at a fixed budget is fix F12's discipline, and early-stopping one arm
would break the identical-budget comparison Gate 3 rests on.

**``log_r_range`` is estimated, and the estimate is frozen before training.**
Decision 13's normaliser is the per-instance *valid-terminal log R range*, which
Phase 3 read off an enumeration.  Here it comes from a seeded uniform-policy
pre-pass, so it is (a) a fixed property of the example rather than something
drifting as the policy improves, and (b) **identical across every arm**, since
the seed is derived from the example id and not from the run.  It normalises
``consistency_error`` — decision 13's *diagnostic*, not any training loss — and
that function already refuses a degenerate range by returning NaN rather than
amplifying noise.  An estimate is acceptable there; it would not be in a loss,
and it is not used in one.

**Sampling carries no gradient**, exactly as Phase 3: rollouts choose *which*
trajectories to learn from, and the batch builder recomputes log-probabilities on
those states with grad attached.  Keeping them apart is what lets one trajectory
batch be replayed by a different objective without re-sampling.

**No ledger during training** (the Phase-3 rule): metering is the *inference*
path's, and Phase 9 is the first phase where both exist in one package.
``portfolio.py`` meters; nothing here does.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from graft.config import Config
from graft.core import obligations as ob
from graft.core.incremental import IncrementalChecker
from graft.core.masks import legal_adds, stop_allowed
from graft.core.reward import log_reward_from_parts
from graft.core.utility import U
from graft.setgen.trainer import Batch, Trainer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.atomfeat import RealFeaturizer
    from graft.setgen.proofs import ProofExample

__all__ = [
    "RealEnvironment",
    "RealTrajectories",
    "RealTrainer",
    "sample_real",
    "build_real_batch",
]

N_DEFICIT = len(ob.DEFICIT_COMPONENTS)


class RealEnvironment:
    """One :class:`~graft.setgen.proofs.ProofExample`, walkable step by step.

    Holds no enumeration.  Every question about a state is answered by
    constructing that state in an :class:`IncrementalChecker` and asking Phase 1
    — which is what makes the state space's size irrelevant rather than merely
    large.
    """

    __slots__ = ("example", "cfg", "atom_ids", "index", "n_atoms", "log_r_range", "_range_report")

    def __init__(
        self,
        example: "ProofExample",
        cfg: Config | None = None,
        *,
        range_samples: int = 64,
    ) -> None:
        self.example = example
        self.cfg = cfg if cfg is not None else Config()
        self.atom_ids = tuple(example.pool.ids())
        self.index = {aid: i for i, aid in enumerate(self.atom_ids)}
        self.n_atoms = len(self.atom_ids)
        self.log_r_range, self._range_report = self._estimate_range(range_samples)

    # -- construction ------------------------------------------------------

    def checker(self, atoms: Sequence[str] = ()) -> IncrementalChecker:
        """A checker positioned at ``atoms``.

        ``ledger=None`` deliberately: see the module docstring.  Training does
        not meter; ``portfolio.py`` does.
        """
        state = IncrementalChecker(
            self.example.pool, self.example.obligations, self.example.snapshot, self.cfg
        )
        for atom_id in atoms:
            state.add(atom_id)
        return state

    # -- reward ------------------------------------------------------------

    def utility(self, atoms: Sequence[str]) -> float:
        """Exact train-time ``U``, with fix F1's sufficiency against gold.

        **[EVIDENCE]** Graph-S3 (ACL 2026) validates dense supervision from
        offline golden subgraphs: its Table 3 ablation loses 11.8 accuracy and
        17.1 F1 when stepwise rewards are replaced by outcome-only ones
        (**macro-averaged over the table's five dataset columns by this project;
        the pair appears nowhere in the paper**).  (The +8.1/+9.7
        headline is against seven baselines — a different comparison, and
        `CLAUDE.md` §5 records the erratum that once conflated them.)

        This is the *train-time* half of fix F1.  At inference there is no gold
        and ranking runs on ``distill.py``'s head instead; the gap between the two
        is what Gate 3 is measured under, which is why the head's fidelity ships
        beside every row.
        """
        ex = self.example
        return U(
            atoms,
            ex.obligations,
            ex.snapshot,
            ex.pool,
            ex.gold_atom_ids,
            self.cfg,
        )

    def log_reward(self, atoms: Sequence[str], valid: bool) -> float:
        """``log R = β·U`` at a valid stop, ``log r_fail`` at ``FAIL``.

        ``β`` comes from the config, which is a **placeholder until Phase-3 step
        6 freezes it** — ``pins.training_blocked_reason()`` is what refuses a
        scored run in the meantime.  Nothing here re-decides it.
        """
        if not valid:
            return math.log(self.cfg.r_fail)
        return log_reward_from_parts(True, self.utility(atoms), self.cfg)

    def _estimate_range(self, samples: int) -> tuple[float, dict[str, Any]]:
        """Decision 13's normaliser, from a seeded uniform-policy pre-pass.

        Seeded from the **example id**, not from the run's seed: two arms must
        normalise their consistency error by the same number, or decision 13's
        band would compare arms against different scales and the C3 guarantee it
        protects would be unfalsifiable.
        """
        # `hash()` on a str is salted per process (PYTHONHASHSEED), so seeding
        # from it would give a different range on every launch — and this number
        # normalises a *reported* diagnostic. A content digest is stable across
        # processes and machines, which is what "frozen before training" has to
        # mean to be worth anything.
        from graft.canonical import digest_of

        seed = int(digest_of(["log_r_range", self.example.example_id])[:8], 16)
        rng = np.random.default_rng(seed)
        values: list[float] = []
        for _ in range(int(samples)):
            state = self.checker()
            for _ in range(self.cfg.max_atoms):
                if stop_allowed(state) and rng.random() < 0.25:
                    break
                mask = legal_adds(state)
                choices = np.flatnonzero(mask)
                if choices.size == 0:
                    break
                state.add(self.atom_ids[int(rng.choice(choices))])
            if stop_allowed(state):
                values.append(self.log_reward(state.selected(), True))
        if not values:
            return 0.0, {"samples": int(samples), "valid": 0, "range": 0.0}
        span = float(max(values) - min(values))
        return span, {
            "samples": int(samples),
            "valid": len(values),
            "range": span,
            "min": float(min(values)),
            "max": float(max(values)),
            "estimated": True,
            "note": (
                "sampled, not enumerated: the state space is not enumerable (G1). "
                "Normalises consistency_error (decision 13's diagnostic) only, "
                "never a training loss."
            ),
        }

    def report(self) -> dict[str, Any]:
        return {
            "example_id": self.example.example_id,
            "n_atoms": self.n_atoms,
            "max_atoms": self.cfg.max_atoms,
            "log_r_range": self._range_report,
        }

    def fingerprints(self) -> dict[str, str]:
        """Which environment a number was computed against.

        The synthetic environment hashes an *enumeration*; there is none here, so
        this hashes what actually determines the example: its pool's atom ids, its
        obligations and its gold.  Two runs that agree on this walked the same
        MDP, which is the property the fingerprint exists to certify.
        """
        from graft.canonical import digest_of

        return {
            "environment": digest_of(
                {
                    "example_id": self.example.example_id,
                    "atoms": list(self.atom_ids),
                    "obligations": self.example.obligations.to_dict(),
                    "gold": sorted(self.example.gold_atom_ids),
                }
            ),
            "target": "not enumerable — no exact p* exists on real pools (G1)",
        }


class RealTrajectories:
    """A padded batch of rollouts over atom-id states.

    ``states[r][k]`` is the *selected set* at step ``k`` of trajectory ``r``, as
    a sorted tuple — the canonical form, because the set is the state and
    insertion order is not part of it (plan §3.4).  ``actions`` are pool
    positions, ``-1`` for padding.

    ``is_fail`` marks a trajectory that reached a dead end: no legal ``ADD`` and
    ``STOP`` masked.  Both terminations count against the budget (Phase-3
    decision 3) — counting only successful rollouts would let a learner that
    dead-ends often buy extra gradient steps for free.
    """

    __slots__ = ("states", "actions", "lengths", "is_fail", "n_atoms")

    def __init__(
        self,
        states: list[list[tuple[str, ...]]],
        actions: np.ndarray,
        lengths: np.ndarray,
        is_fail: np.ndarray,
        n_atoms: int,
    ) -> None:
        self.states = states
        self.actions = actions
        self.lengths = lengths
        self.is_fail = is_fail
        self.n_atoms = int(n_atoms)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def n_transitions(self) -> int:
        return int(self.lengths.sum())

    def terminals(self) -> list[tuple[str, ...]]:
        """The set each trajectory ended at."""
        return [self.states[r][int(self.lengths[r])] for r in range(len(self))]


def legal_deltas(
    env: RealEnvironment,
    atoms: Sequence[str],
    legal: np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[n_atoms, 6]`` — ``Δd`` for **every legal ADD** at ``atoms``, zero elsewhere.

    **One implementation, called by both the sampler and the batch builder, and
    that is the whole point of it existing.**  They had two, and the two
    disagreed in both directions at once (found by adversarial audit, 16 Aug
    2026): the batch builder computed ``child − parent`` where the sampler
    computed ``parent − child``, and it filled only the *taken* action's column
    where the sampler filled every legal one.

    Both halves mattered and both ran against Contribution 3:

    *sign* — ``ob.delta_deficit`` is ``d(s) − d(s′)``, so a **positive** component
    means the transition *discharged* an obligation.  Inverted, L7's policy was
    trained to read discharge as accrual.

    *density* — the synthetic featurizer's ``_delta_table`` is dense over all
    edges precisely so ``Δd`` is a **comparison across candidate actions**.
    Filled only at the taken action, it degenerates into a label attached to the
    choice already made, and "prefer the ADD that discharges more obligation" —
    the entire C3 mechanism — becomes unlearnable.  ``features.py`` lines 123-137
    document this exact failure being found and fixed once already on the
    synthetic side, where it produced *"a false negative for Contribution 3"*.

    The cost is real and is why L7/L7b are the slowest arms: one ``add``/``undo``
    plus a deficit evaluation per legal action per state.  Decision 11 sizes the
    budget on the slowest arm for exactly this reason.
    """
    out = torch.zeros((env.n_atoms, N_DEFICIT), dtype=dtype)
    state = env.checker(atoms)
    here = state.deficit()
    for j in np.flatnonzero(np.asarray(legal)[: env.n_atoms]):
        state.add(env.atom_ids[int(j)])
        out[int(j)] = torch.as_tensor(
            ob.delta_deficit(here, state.deficit()), dtype=dtype
        )
        state.undo()
    return out


def _removable(atoms: Sequence[str], pool: Any) -> int:
    """How many selected atoms could be removed and leave a closed set.

    ``P_B`` is uniform over the **removable** atoms, not over the selected ones:
    under the closure rule an atom referenced by another selected atom cannot be
    removed, so uniform-over-selected would put mass on parents that do not
    exist.  Phase 3 read this off the enumerated in-degrees and never
    reimplemented it; there is no enumeration here, so it is computed directly
    from the same definition — an atom is removable exactly when no *other*
    selected atom references it.
    """
    selected = set(atoms)
    referenced: set[str] = set()
    for aid in selected:
        for ref in pool[aid].refs:
            if ref in selected:
                referenced.add(ref)
    return sum(1 for aid in selected if aid not in referenced)


@torch.no_grad()
def sample_real(
    featurizer: "RealFeaturizer",
    env: RealEnvironment,
    n: int,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    greedy: int = 0,
) -> RealTrajectories:
    """Walk the MDP ``n`` times from the empty set, one step at a time.

    ``epsilon`` mixes in a uniform-over-legal-actions component — the frozen
    ε-uniform exploration, identical across arms (Phase-3 decision 10).  **The
    loss still uses the policy's own log-probabilities**, not the behaviour
    policy's: trajectory-balance-family objectives are off-policy-capable, so
    exploring with a mixture and scoring with ``P_F`` is correct.  That
    justification does **not** cover L3 — GRPO is an on-policy policy gradient
    and gets a genuine (small, arm-identical) bias, recorded as a declared L3
    departure in `PHASE3_DECISIONS.md` §1.4.

    ``greedy`` makes the first ``greedy`` trajectories take the highest-
    probability legal action at every step.  Fix F4 defines the portfolio as
    ``K = 8`` = 1 greedy + 7 sampled, and best-of-K is measured over that.

    Trajectories are stepped **together**, one MDP step per iteration, so the
    policy sees a batch rather than one state at a time.  Live trajectories
    shrink as they terminate; the checkers are per-trajectory because each holds
    its own selected set.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if not 0 <= greedy <= n:
        raise ValueError(f"greedy must be in [0, {n}], got {greedy}")

    n_atoms = env.n_atoms
    max_len = env.cfg.max_atoms

    checkers = [env.checker() for _ in range(n)]
    states: list[list[tuple[str, ...]]] = [[c.selected()] for c in checkers]
    actions = np.full((n, max_len), -1, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    is_fail = np.zeros(n, dtype=bool)
    live = list(range(n))

    for step in range(max_len + 1):
        if not live:
            break

        # -- masks first: a dead end terminates before the policy is asked ---
        rows: list[int] = []
        legal_np = np.zeros((len(live), n_atoms + 1), dtype=bool)
        for slot, r in enumerate(live):
            adds = legal_adds(checkers[r])
            can_stop = stop_allowed(checkers[r])
            if not adds.any() and not can_stop:
                is_fail[r] = True
                continue
            legal_np[slot, :n_atoms] = adds
            legal_np[slot, n_atoms] = can_stop
            rows.append(slot)
        # Trajectories dropped here have *terminated* as FAIL, so `live` must
        # shrink to `keep` before the emptiness check -- not after it. Leaving
        # the dead-ended rows in `live` and breaking made the loop exit with a
        # non-empty `live`, which the guard below then reported as "the ADD mask
        # is not shrinking": a correct dead end misdiagnosed as a masking bug,
        # and it fired on exactly the all-FAIL case the abstain fallback exists
        # for. Found by the fallback test, 16 Aug 2026.
        keep = [live[s] for s in rows]
        live = keep
        if not keep:
            break
        legal_np = legal_np[rows]

        # A dead end has no distribution over its actions -- log_softmax over an
        # all-masked row is all NaN -- so those rows are dropped above rather
        # than queried and repaired afterwards.
        sel = torch.zeros((len(keep), n_atoms), dtype=featurizer.dtype)
        deficits = torch.zeros((len(keep), N_DEFICIT), dtype=featurizer.dtype)
        sizes = torch.zeros(len(keep), dtype=featurizer.dtype)
        delta = (
            torch.zeros((len(keep), n_atoms, N_DEFICIT), dtype=featurizer.dtype)
            if featurizer.delta_d
            else None
        )
        for slot, r in enumerate(keep):
            state = checkers[r]
            here = state.selected()
            for aid in here:
                sel[slot, env.index[aid]] = 1.0
            d_here = state.deficit()
            deficits[slot] = torch.as_tensor(d_here, dtype=featurizer.dtype)
            sizes[slot] = float(len(here))
            if delta is not None:
                delta[slot] = legal_deltas(env, here, legal_np[slot], featurizer.dtype)

        legal = torch.as_tensor(legal_np)
        logits = featurizer.logits(sel, deficits, sizes, legal, delta)
        probs = torch.exp(logits).double().numpy()
        total = probs.sum(axis=1, keepdims=True)
        np.divide(probs, total, out=probs, where=total > 0)

        if epsilon > 0.0:
            counts = legal_np.sum(axis=1, keepdims=True)
            uniform = np.divide(
                legal_np.astype(np.float64), counts,
                out=np.zeros_like(probs), where=counts > 0,
            )
            probs = (1.0 - epsilon) * probs + epsilon * uniform

        cumulative = np.cumsum(probs, axis=1)
        u = rng.random(len(keep))
        picks = (cumulative < u[:, None]).sum(axis=1)
        np.minimum(picks, n_atoms, out=picks)
        if greedy:
            take_max = np.array([r < greedy for r in keep], dtype=bool)
            if take_max.any():
                picks[take_max] = probs[take_max].argmax(axis=1)

        still: list[int] = []
        for slot, r in enumerate(keep):
            action = int(picks[slot])
            if action == n_atoms:            # STOP
                continue
            if not legal_np[slot, action]:   # pragma: no cover - masking guarantees this
                raise RuntimeError(
                    "sampled an illegal action; masking before the softmax is "
                    "supposed to make that impossible"
                )
            checkers[r].add(env.atom_ids[action])
            actions[r, step] = action
            states[r].append(checkers[r].selected())
            lengths[r] = step + 1
            still.append(r)
        live = still

    if live:
        raise RuntimeError(
            f"{len(live)} trajectories did not terminate within max_atoms steps; "
            "every step adds one atom and the set is capped, so this means the "
            "ADD mask is not shrinking as atoms are selected"
        )
    return RealTrajectories(states, actions, lengths, is_fail, n_atoms)


def build_real_batch(
    featurizer: "RealFeaturizer",
    env: RealEnvironment,
    traj: RealTrajectories,
    *,
    logz: torch.nn.Module,
    flow: torch.nn.Module | None = None,
    aux: torch.nn.Module | None = None,
    needs_potential: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Batch:
    """Real trajectories → the :class:`Batch` Phase 3's losses already consume.

    **The terminating transition is a first-class step.**  Flow positions run
    ``0 … L`` and transition positions ``0 … L−1``; the trajectory's own ``STOP``
    carries its ``log P_F(STOP | x)``, ``log P_B = 0`` and its own ``φ`` slot.
    `PHASE3_DECISIONS.md` §1.2 recorded the terminal convention as **measured**
    on ``tiny_instance()`` — a state that is both a valid stop and has children
    gives ``F(s) = 181.49`` against ``R(s) = 14.88``, so ``R`` is the flow on the
    terminating edge, not the terminal state's flow.  Four objectives depend on
    it, and a loss assuming ``F(X) = R(X)`` would be wrong by the continuation
    flow and would misreport as a decomposition failure.

    Padding is **not** left to be harmless by luck: ``log_pf`` and ``log_pb`` are
    zero past the end so the cumulative sums stay correct, and every loss masks
    on top of that.  Relying on either alone has already produced one silent bug
    in this codebase.
    """
    n = len(traj)
    n_atoms = env.n_atoms
    lengths = traj.lengths
    steps = max(1, int(lengths.max()) + 1)  # + 1 for the terminating transition

    def T(a: np.ndarray, dt: torch.dtype = torch.long) -> torch.Tensor:
        return torch.as_tensor(a, dtype=dt, device=device)

    rows = np.arange(n, dtype=np.int64)
    k = lengths[:, None]
    idx = np.arange(steps, dtype=np.int64)[None, :]
    valid_np = idx <= k
    n_trans = lengths + 1

    # -- gather the states each position needs -----------------------------
    # Padding repeats the terminal set, so every feature lookup is finite and
    # nothing NaNs behind a mask.
    terminal_sets = traj.terminals()
    parent_sets: list[tuple[str, ...]] = []
    child_sets: list[tuple[str, ...]] = []
    action_ix = np.full((n, steps), n_atoms, dtype=np.int64)
    for r in range(n):
        length = int(lengths[r])
        for s in range(steps):
            if s < length:
                parent_sets.append(traj.states[r][s])
                child_sets.append(traj.states[r][s + 1])
                action_ix[r, s] = int(traj.actions[r, s])
            elif s == length:
                parent_sets.append(traj.states[r][length])   # the terminating step
                child_sets.append(traj.states[r][length])
            else:
                parent_sets.append(terminal_sets[r])
                child_sets.append(terminal_sets[r])

    flow_sets: list[tuple[str, ...]] = []
    for r in range(n):
        length = int(lengths[r])
        for s in range(steps + 1):
            flow_sets.append(traj.states[r][min(s, length)])

    def encode(sets: Sequence[tuple[str, ...]]) -> tuple[torch.Tensor, ...]:
        m = len(sets)
        sel = torch.zeros((m, n_atoms), dtype=dtype, device=device)
        deficits = torch.zeros((m, N_DEFICIT), dtype=dtype, device=device)
        sizes = torch.zeros(m, dtype=dtype, device=device)
        for i, atoms in enumerate(sets):
            for aid in atoms:
                sel[i, env.index[aid]] = 1.0
            sizes[i] = float(len(atoms))
            deficits[i] = torch.as_tensor(
                ob.deficit(atoms, env.example.pool, env.example.obligations, env.example.snapshot),
                dtype=dtype,
            )
        return sel, deficits, sizes

    p_sel, p_def, p_size = encode(parent_sets)
    c_sel, c_def, c_size = encode(child_sets)

    # -- log P_F -----------------------------------------------------------
    # A dead-ended trajectory's final position has no distribution over its
    # actions, so it is never asked: its termination is forced and log P_F = 0.
    need = valid_np.copy()
    need[rows, lengths] = ~traj.is_fail
    legal_np = np.zeros((n * steps, n_atoms + 1), dtype=bool)
    for i, atoms in enumerate(parent_sets):
        state = env.checker(atoms)
        legal_np[i, :n_atoms] = legal_adds(state)
        legal_np[i, n_atoms] = stop_allowed(state)
    # Rows that are masked out entirely would log_softmax to NaN. They are
    # exactly the rows `need` already excludes, but the mask is repaired anyway
    # so a NaN cannot reach the loss even if `need` were wrong -- the padding
    # lesson of this codebase, applied to its own repeat.
    empty = ~legal_np.any(axis=1)
    legal_np[empty, n_atoms] = True

    # **Dense over legal actions, through the shared helper** — see
    # :func:`legal_deltas` for what the two-implementation version got wrong in
    # both directions at once. The policy must see the same action features here
    # (with grad) that it saw while sampling (without), or log P_F is not the
    # log-probability the trajectory was drawn under and L7/L7b are silently
    # off-policy against themselves.
    delta_full = None
    if featurizer.delta_d:
        delta_full = torch.zeros((n * steps, n_atoms, N_DEFICIT), dtype=dtype, device=device)
        for i, atoms in enumerate(parent_sets):
            delta_full[i] = legal_deltas(env, atoms, legal_np[i], dtype)

    logits = featurizer.logits(
        p_sel, p_def, p_size, torch.as_tensor(legal_np, device=device), delta_full
    )
    picked = logits.gather(1, T(action_ix.reshape(-1)).view(-1, 1)).view(n, steps)
    # `torch.where`, not `picked * need`. A masked action gathers `-inf`, and
    # `-inf * 0` is **nan**, not 0 -- so the multiplicative form would turn a
    # single mask slip into a nan that propagates through `_cum_pf` into every
    # objective. Probed clean here (the mask repair above covers the dead-end
    # rows), but this codebase's own Batch docstring records that "relying on
    # either alone has already produced one silent bug", so the arithmetic is
    # made incapable of it rather than shown to be safe today.
    log_pf = torch.where(T(need, torch.bool), picked, torch.zeros_like(picked))

    # -- log P_B (constant in θ) -------------------------------------------
    log_pb_np = np.zeros((n, steps), dtype=np.float64)
    for r in range(n):
        for s in range(int(lengths[r])):
            count = _removable(traj.states[r][s + 1], env.example.pool)
            if count > 0:
                log_pb_np[r, s] = -math.log(count)
    log_pb = torch.as_tensor(log_pb_np, dtype=dtype, device=device)

    # -- flows -------------------------------------------------------------
    log_z = logz(featurizer.instance_repr()).squeeze(0)
    f_sel, f_def, f_size = encode(flow_sets)
    h_flow = None
    if flow is not None or aux is not None:
        h_flow = featurizer.policy.encode(featurizer.state_repr(f_sel, f_def, f_size))
    if flow is not None:
        flow_raw = flow(h_flow).view(n, steps + 1)
    else:
        flow_raw = torch.zeros((n, steps + 1), dtype=dtype, device=device)
    # F(s_0) = Z is an identity, not something to be learned twice.
    flow_raw = torch.cat([log_z.view(1, 1).expand(n, 1), flow_raw[:, 1:]], dim=1)

    # -- rewards, deficits -------------------------------------------------
    log_reward = torch.as_tensor(
        [
            env.log_reward(terminal_sets[r], not bool(traj.is_fail[r]))
            for r in range(n)
        ],
        dtype=dtype,
        device=device,
    )
    delta = (p_def - c_def).view(n, steps, N_DEFICIT)
    t_sel, terminal_deficit, t_size = encode(terminal_sets)

    state_repr = action_repr = None
    if needs_potential:
        state_repr = featurizer.state_repr(p_sel, p_def, p_size).view(n, steps, -1)
        action_repr = featurizer.action_reprs(n * steps, delta_full)[
            torch.arange(n * steps, device=device), T(action_ix.reshape(-1))
        ].view(n, steps, -1)

    aux_pred = aux_mask = None
    if aux is not None:
        aux_pred = aux(h_flow).view(n, steps + 1, N_DEFICIT)
        aux_mask = T(np.arange(steps + 1)[None, :] <= k, torch.bool)

    return Batch(
        log_z=log_z,
        flow_raw=flow_raw,
        log_pf=log_pf,
        log_pb=log_pb,
        log_reward=log_reward,
        n_trans=T(n_trans),
        valid=T(valid_np, torch.bool),
        is_fail=T(traj.is_fail, torch.bool),
        state_repr=state_repr,
        action_repr=action_repr,
        delta=delta,
        terminal_deficit=terminal_deficit,
        log_r_range=env.log_r_range,
        aux_pred=aux_pred,
        aux_mask=aux_mask,
    )


class RealTrainer(Trainer):
    """Phase 3's trainer, hosted on real pools.

    **A subclass rather than a second trainer, and that is a deliberate
    anti-duplication choice.**  ``Trainer.__init__`` builds the heads, the
    optimisers, the parameter groups and the capacity accounting — and capacity
    matching is a *ruled* Phase-3 decision (11) that Gate 2 depends on.  A
    parallel implementation here would be two versions of it drifting apart in
    separate files, which is the failure `CLAUDE.md` §5 catalogues under "a fix
    landing in three places out of four".  Only the two genuinely
    environment-dependent seams are overridden.

    **Exact TV is not computed and not approximated.**  It exists only where the
    state space can be enumerated (G1).  ``tv_mean`` records ``nan`` at every
    checkpoint, which is the honest signal — an estimate reported in the column
    Gate 2's exact numbers occupy would invite exactly the comparison that is
    invalid.  Checkpoints here are *monitoring*: training loss, and the runner's
    dev best-of-K probe.  **Never selection** — no arm early-stops, because
    selection at a fixed budget is fix F12's discipline and stopping one arm
    early would break the identical-budget comparison Gate 3 rests on.
    """

    def __init__(self, arm, envs, spec, *, greedy: int = 0) -> None:
        self.greedy = int(greedy)
        super().__init__(arm, envs, spec)

    # -- the two seams -----------------------------------------------------

    def _dims(self) -> tuple[int, int]:
        from graft.setgen.atomfeat import RealFeaturizer

        return RealFeaturizer.dims()

    def _build_featurizers(self) -> list["RealFeaturizer"]:
        from graft.setgen.atomfeat import RealFeaturizer

        return [
            RealFeaturizer(
                env.example, self.policy, env.cfg,
                delta_d=self.arm.delta_d,
                device=self.spec.device, dtype=self.spec.dtype,
            )
            for env in self.envs
        ]

    # -- the training step -------------------------------------------------

    def _step(self, env, feat, size: int, spent: int) -> float:
        if self.arm.supervised:
            logits, targets = self.real_gold_batch(env, feat, size)
            loss = self.arm.loss((logits, targets), self, spent)
        else:
            traj = sample_real(
                feat, env, size, self.rng, self.spec.epsilon, greedy=self.greedy
            )
            batch = build_real_batch(
                feat, env, traj,
                logz=self.logz, flow=self.flow, aux=self.aux,
                needs_potential=self.arm.needs_potential,
                dtype=self.spec.dtype, device=self.spec.device,
            )
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

    def real_gold_batch(
        self, env: RealEnvironment, feat: "RealFeaturizer", size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(logits, target_action)`` along gold construction paths.

        L1 samples a fresh closure-legal order per example — stepwise supervision
        of the *set*; L2 walks the one canonical order and repeats it — imitation
        of a single sequence.  That is the distinction plan §5.1 draws, and it is
        the only difference between the two arms.

        ``STOP`` at the gold terminal is a supervised action like any other: a
        policy taught to build the gold set and never taught to stop would walk
        off it at the last step.

        The logits are computed **here rather than inside the loss**, exactly as
        the synthetic path does it: a learner that could call the featurizer
        could reach the environment through it.
        """
        canonical = self.arm.name == "l2_imitation"
        sets: list[tuple[str, ...]] = []
        targets: list[int] = []
        for _ in range(size):
            path, actions = feat.gold_path(None if canonical else self.rng)
            sets.extend(path[:-1])
            targets.extend(actions)
            sets.append(path[-1])
            targets.append(env.n_atoms)      # STOP

        m, n_atoms = len(sets), env.n_atoms
        dtype, device = self.spec.dtype, self.spec.device
        sel = torch.zeros((m, n_atoms), dtype=dtype, device=device)
        deficits = torch.zeros((m, N_DEFICIT), dtype=dtype, device=device)
        sizes = torch.zeros(m, dtype=dtype, device=device)
        legal_np = np.zeros((m, n_atoms + 1), dtype=bool)
        delta = (
            torch.zeros((m, n_atoms, N_DEFICIT), dtype=dtype, device=device)
            if feat.delta_d
            else None
        )
        for i, atoms in enumerate(sets):
            for aid in atoms:
                sel[i, env.index[aid]] = 1.0
            sizes[i] = float(len(atoms))
            state = env.checker(atoms)
            deficits[i] = torch.as_tensor(state.deficit(), dtype=dtype)
            legal_np[i, :n_atoms] = legal_adds(state)
            legal_np[i, n_atoms] = stop_allowed(state)
        if delta is not None:
            # **Filled, not merely allocated.** The first version allocated this
            # tensor and passed it to the policy without ever writing to it, so
            # any arm that was both `supervised` and `delta_d` would have trained
            # on an identically-zero Δd block while believing it was
            # checker-conditioned — silently reducing to L1/L2 with dead
            # capacity, and reading out as "checker-conditioning adds nothing
            # under supervision". No such arm exists in the ruled roster, so the
            # branch was dead; it becomes live the moment anyone adds a
            # supervised C3 ablation, which is exactly when nobody would look.
            for i, atoms in enumerate(sets):
                delta[i] = legal_deltas(env, atoms, legal_np[i], dtype)
        # The gold terminal's own STOP must be reachable, or the target would be
        # a masked action and the loss would be −inf. `gold_path` already refuses
        # a gold set that cannot stop; this is the assertion of that, in tensors.
        legal_np[~legal_np.any(axis=1), n_atoms] = True

        logits = feat.logits(
            sel, deficits, sizes, torch.as_tensor(legal_np, device=device), delta
        )
        return logits, torch.as_tensor(targets, dtype=torch.long, device=device)

    # -- what does not exist here ------------------------------------------

    def _checkpoint(self, log, spent: int) -> None:
        """Monitoring only: ``nan`` where exact TV would be (G1)."""
        per_env = [float("nan")] * len(self.envs)
        log.trajectories.append(spent)
        log.tv_per_env.append(per_env)
        log.tv_mean.append(float("nan"))

    def exact_tv(self) -> list[float]:
        raise NotImplementedError(
            "exact TV needs an enumerated state space and real pools do not have "
            "one (~10^13 subsets, plan §3.4). Gate 2 — the distributional claim — "
            "stays on the synthetic environment; Phase 9's decision is Gate 3, "
            "which is a best-of-K comparison and needs no p*."
        )

    def evaluate_on(self, envs) -> list[float]:
        raise NotImplementedError(
            "held-out exact TV is not computable on real pools; the held-out read "
            "here is the dev best-of-K probe, which the runner owns"
        )
