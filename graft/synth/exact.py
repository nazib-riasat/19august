"""``p*``, ``p_θ``, and the divergences — the only exact distribution comparison
in the project.

**[EVIDENCE]** Exact evaluation on an enumerable space is the standard instrument
in the GFlowNet training literature: Shen et al. (ICML 2023) evaluate on
enumerable spaces, and *When Do GFlowNets Learn the Right Distribution?*
(ICLR 2025, Spotlight) builds correctness metrics precisely because sampled
proxies mislead.  Everywhere else in this project a distribution claim is a
sampled estimate.

**One forward DP, not one per terminal** (G2).  The forward reachability mass
``f(S)`` is a function of the *state*, and states are shared across terminals, so
a single pass over the enumerated edge list yields every terminal's probability
at once::

    f(∅) = 1
    for S in states, in increasing |S|:
        for a in legal_adds(S):  f(S ∪ {a}) += f(S) · P_F(a | S)
    p_θ(X)    = f(X) · P_F(STOP | X)        for every valid terminal X
    p_θ(FAIL) = Σ_{dead-end states d} f(d)

**``p_θ(FAIL)`` is accumulated directly; the complement is a consistency check —
not the other way round** (decision 15).  Architecture fix F3 writes it as
``1 − Σ_valid p_θ(X)``, which is right in general and is the only option outside
an enumerable environment.  Here the state graph already labels dead ends, so the
direct sum is available and far better conditioned: the complement subtracts two
quantities that agree to ~12 digits, so its absolute error is bounded below by
float64 representation near 1 regardless of how carefully the sum is done —
about ``2e-4`` *relative* at ``p*(FAIL) ≈ 2.5e-12``, and no better with
``math.fsum``, because the loss is in ``1 − x`` rather than in ``Σ``.

**``U`` is cached per terminal, never ``R``** (G7).  ``p*`` depends on β only
through ``R = exp(β·U)`` and ``U`` is independent of β, so each additional β in
the Phase-3 sweep is an ``exp`` over a vector.  Caching ``R`` instead would
re-derive ``U`` for thousands of terminals at every β.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from graft.canonical import digest_of
from graft.config import Config, validate as validate_config
from graft.core.utility import U
from graft.synth.enumerate import StateGraph, environment_fingerprint, reachable_states

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.synth.lattice import LatticeInstance
    from graft.synth.policies import ActionPolicy

__all__ = [
    "Target",
    "target_distribution",
    "policy_distribution",
    "forward_mass",
    "partition_residual",
    "tv",
    "js",
    "kl",
    "divergence_report",
    "fcs",
    "fcs_exact",
    "target_fingerprint",
    "MODES",
    "SCOPES",
    "FCS_M",
    "FCS_SUBSETS",
    "FCS_SEED",
    "TARGET_QUANTUM",
]

#: Mode buckets, by **completion** of the generator's complete minimal proof
#: templates (decision 13).  Not by unique-atom membership — a terminal can hold
#: every atom unique to chain A while lacking the shared anchor, and ``H`` would
#: not object — and not by clustering.
MODES: tuple[str, ...] = ("A", "B", "mixed", "neither")

#: The suites :meth:`Target.validate_bands` knows about.  An unrecognised scope
#: is an error rather than a pass: the band is a **hard gate** and every other
#: decision on this path fails closed — ``Assertion.eligibility`` defaults to
#: quarantined, ``H`` rejects an unresolvable provenance chain — so a typo that
#: silently skipped the gate would be the one fail-open step in the chain.
#: Kept in step with ``lattice.SUITE_SIZES`` by a test rather than by an import,
#: because ``lattice`` imports this module transitively.
SCOPES: tuple[str, ...] = ("main", "probe", "tuning")

#: FCS estimator, frozen here (decision 20).  ``m`` is the paper's ``β``, renamed
#: because β is already the reward temperature in this project, and **fixed**
#: rather than tied to the training batch size so the metric stays comparable
#: across learners that batch differently — a declared deviation from the paper's
#: usage.
FCS_M = 8
FCS_SUBSETS = 2_000
FCS_SEED = 20260812

#: Quantum for ``p*`` values inside ``target_fingerprint``.  There is no
#: quantisation that both avoids false alarms and catches everything: the DP's
#: cross-machine float noise is ~1e-12 absolute, so a finer quantum would flag
#: identical implementations and a coarser one hides real differences.  At 1e-12
#: over 5,000 terminals the worst hidden discrepancy is ~1.25e-9 in TV —
#: comparable to the evaluator's own partition and oracle tolerances, and
#: therefore the resolution limit of the whole numeric layer.
TARGET_QUANTUM = 1e-12


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------


class Target:
    """``p*`` over ``{valid terminals} ∪ {FAIL}``, plus what re-deriving it needs.

    **Layout.**  ``p_star`` has length ``n_terminals + 1``; entry ``i`` is the
    terminal at ``graph.terminal_ix[i]`` and the **last entry is ``FAIL``**.
    :func:`policy_distribution` returns the same layout, so every divergence
    below compares like with like — and ``FAIL`` participates in all of them,
    because it is a member of the target's support (architecture fix F3).
    """

    __slots__ = (
        "instance",
        "graph",
        "cfg",
        "beta",
        "terminal_ix",
        "u",
        "sizes",
        "mode_labels",
        "zero_sufficiency",
        "templates",
        "r",
        "z",
        "p_star",
        "target_p_fail",
    )

    def __init__(
        self,
        instance: "LatticeInstance",
        graph: StateGraph,
        cfg: Config,
        u: np.ndarray,
        sizes: np.ndarray,
        mode_labels: Sequence[str],
        zero_sufficiency: np.ndarray,
    ) -> None:
        self.instance = instance
        self.graph = graph
        self.cfg = cfg
        self.beta = float(cfg.beta)
        self.terminal_ix = graph.terminal_ix
        self.u = u
        self.sizes = sizes
        self.mode_labels = tuple(mode_labels)
        self.zero_sufficiency = zero_sufficiency
        self.templates = (instance.template_a, instance.template_b)

        self.r = np.exp(self.beta * u)
        self.z = float(self.r.sum() + cfg.r_fail)
        p = np.empty(u.shape[0] + 1, dtype=np.float64)
        p[:-1] = self.r / self.z
        p[-1] = cfg.r_fail / self.z
        self.p_star = p
        self.target_p_fail = float(p[-1])

    # -- shape -------------------------------------------------------------

    @property
    def n_terminals(self) -> int:
        return int(self.u.shape[0])

    def __len__(self) -> int:
        return int(self.p_star.shape[0])

    # -- β ------------------------------------------------------------------

    def at_beta(self, beta: float) -> "Target":
        """The target at a new β, from the cached ``U`` (G7).

        **Re-runs the ``r_fail_margin`` check, and that is the point.**  Phase 0
        added a load-time assertion that ``r_fail < r_fail_margin · exp(β·U_min)``
        precisely so a β sweep cannot quietly promote ``FAIL`` into a competitive
        terminal.  A sweep that moved β through here would otherwise walk straight
        past the loader and around that protection.

        It does **not** touch the target-mass bands: those are a main-suite
        condition, and a single call that always checked them would abort the β
        sweep — which runs on the *tuning* suite — on a condition that suite is
        exempt from (decision 10).  Use :meth:`validate_bands` for those.
        """
        cfg = self.cfg.with_overrides(beta=float(beta))
        validate_config(cfg)
        return Target(
            self.instance,
            self.graph,
            cfg,
            self.u,
            self.sizes,
            self.mode_labels,
            self.zero_sufficiency,
        )

    # -- mass profile -------------------------------------------------------

    def mass_profile(self) -> dict[str, Any]:
        """Where ``p*`` actually sits (G10).  Cheap: ``p*`` is already in hand."""
        p = self.p_star[:-1]
        labels = np.asarray(self.mode_labels)
        by_mode = {m: float(p[labels == m].sum()) for m in MODES}
        counts = {m: int((labels == m).sum()) for m in MODES}
        order = np.argsort(-p)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = float(-np.sum(np.where(self.p_star > 0, self.p_star * np.log(self.p_star), 0.0)))
        return {
            "beta": self.beta,
            "z": self.z,
            "target_p_fail": self.target_p_fail,
            "n_terminals": self.n_terminals,
            "mode_mass": by_mode,
            "mode_counts": counts,
            "zero_sufficiency_mass": float(p[self.zero_sufficiency].sum()),
            "effective_support": float(math.exp(entropy)),
            "top10_mass": float(p[order[:10]].sum()),
            "u_min": float(self.u.min()) if self.n_terminals else float("nan"),
            "u_max": float(self.u.max()) if self.n_terminals else float("nan"),
        }

    def validate_bands(self, scope: str) -> dict[str, Any]:
        """Check the β-dependent target-mass bands, **main suite only**.

        ``p*`` depends on β through ``R = exp(β·U)``, so the hard `neither`-mass
        band of G10 is a function of β: a suite accepted at β = 4 could violate
        its own acceptance condition the moment the Phase-3 sweep lands on a
        different value, with nothing failing.  This is the call that notices.

        The probe suite is distractor-heavy by design and the tuning suite is
        where the sweep runs, so both report their profile and neither is gated
        (decision 14).
        """
        from graft.synth.enumerate import BandViolation

        if scope not in SCOPES:
            raise ValueError(
                f"scope must be one of {SCOPES}, got {scope!r}. This gate is only "
                "as good as the string it is called with, and an unrecognised one "
                "would skip the band silently."
            )
        profile = self.mass_profile()
        profile["scope"] = scope
        if scope == "main":
            neither = profile["mode_mass"]["neither"]
            band = self.instance.spec.max_neither_mass
            if neither > band:
                raise BandViolation(
                    "neither_mass",
                    round(neither, 4),
                    f"<= {band} of p* on terminals completing no designed proof, "
                    f"at beta={self.beta} (G10). The distractor tail is what exact "
                    "TV would then be measuring.",
                )
        return profile

    # -- identity -----------------------------------------------------------

    def fingerprint(self) -> str:
        return target_fingerprint(self)


def _mode_of(atoms: frozenset[str], template_a: frozenset[str], template_b: frozenset[str]) -> str:
    a, b = template_a <= atoms, template_b <= atoms
    if a and b:
        return "mixed"
    if a:
        return "A"
    if b:
        return "B"
    return "neither"


def target_distribution(
    instance: "LatticeInstance",
    cfg: Config | None = None,
    *,
    graph: StateGraph | None = None,
) -> Target:
    """Enumerate, score every valid terminal once, and build ``p*``."""
    cfg = cfg if cfg is not None else instance.cfg
    g = graph if graph is not None else reachable_states(instance, cfg)

    q, G, pool, gold = instance.obligations, instance.graph, instance.pool, instance.gold
    ta, tb = instance.template_a, instance.template_b
    gold_atoms = set(gold.atoms)

    n = g.n_terminals
    u = np.empty(n, dtype=np.float64)
    sizes = np.empty(n, dtype=np.int16)
    zero_suff = np.zeros(n, dtype=bool)
    labels: list[str] = []
    for i, t in enumerate(g.terminal_ix.tolist()):
        atoms = frozenset(g.atoms_of(t))
        u[i] = U(atoms, q, G, pool, gold, cfg)
        sizes[i] = len(atoms)
        zero_suff[i] = not (gold_atoms & atoms)
        labels.append(_mode_of(atoms, ta, tb))
    return Target(instance, g, cfg, u, sizes, labels, zero_suff)


# --------------------------------------------------------------------------
# the forward DP
# --------------------------------------------------------------------------


def forward_mass(
    policy: "ActionPolicy", graph: StateGraph
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(f per state, P_F(a | s) per edge, P_F(STOP | s) per state)``.

    The one forward pass every downstream number reuses: ``p_θ`` (below),
    dead-end absorption mass, and the visitation-weighted ``Δd`` density all read
    it rather than each running their own DP.

    **Dead-end states are never queried.**  A dead end has no legal ``ADD`` and a
    masked ``STOP``, so no probability distribution over its actions exists —
    asking for one is a category error, and returning all ``-inf`` would put NaNs
    into the DP.  They get ``P_F(STOP) = 0`` here and their mass is routed
    straight to ``FAIL``.
    """
    n_states = graph.n_states
    query = np.flatnonzero(~graph.dead_end).astype(np.int64)
    row_of = np.full(n_states, -1, dtype=np.int64)
    row_of[query] = np.arange(query.shape[0], dtype=np.int64)

    log_add, log_stop = policy.action_log_probs(query, graph)
    p_add = np.exp(np.asarray(log_add, dtype=np.float64))
    stop = np.zeros(n_states, dtype=np.float64)
    stop[query] = np.exp(np.asarray(log_stop, dtype=np.float64))

    edge_prob = (
        p_add[row_of[graph.edge_parent], graph.edge_action]
        if graph.n_edges
        else np.zeros(0, dtype=np.float64)
    )

    f = np.zeros(n_states, dtype=np.float64)
    f[0] = 1.0
    # Ascending layer order is *required* — every edge goes from |S| to |S|+1, so
    # a parent's mass must be final before its children are written.  Order
    # *within* a layer is free, and the DP must not depend on it (criterion 6).
    for depth in range(graph.max_atoms):
        lo, hi = graph.edge_slice(depth)
        if hi <= lo:
            continue
        contrib = f[graph.edge_parent[lo:hi]] * edge_prob[lo:hi]
        f += np.bincount(graph.edge_child[lo:hi], weights=contrib, minlength=n_states)
    return f, edge_prob, stop


def policy_distribution(policy: "ActionPolicy", graph: StateGraph) -> np.ndarray:
    """``p_θ`` over ``{valid terminals} ∪ {FAIL}``, in ``Target``'s layout."""
    f, _, stop = forward_mass(policy, graph)
    out = np.empty(graph.n_terminals + 1, dtype=np.float64)
    if graph.n_terminals:
        out[:-1] = f[graph.terminal_ix] * stop[graph.terminal_ix]
    out[-1] = float(f[graph.dead_ix].sum()) if graph.n_dead_ends else 0.0
    return out


def partition_residual(p: np.ndarray) -> float:
    """``|1 − Σ_valid p_θ − p_θ(FAIL)|``.

    The complement of ``p_θ(FAIL)`` as a **check**, never as the reported value
    (decision 15).  Every trajectory adds one atom per step and is capped at
    ``max_atoms``, so every trajectory terminates — at a valid ``STOP`` or at a
    dead end — and the two outcomes partition the mass.
    """
    return float(abs(1.0 - float(p.sum())))


# --------------------------------------------------------------------------
# divergences
# --------------------------------------------------------------------------


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation, in [0, 1].  The Gate-2 primary metric.

    ``FAIL`` is in **both** distributions, so a policy that assigns it exactly
    ``p*(FAIL)`` achieves TV = 0.  ``p*(FAIL)`` is reported **beside** TV and is
    never subtracted from it: the convergence target is 0 and stays 0 (G4).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p - q).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in **nats**, bounded by ``ln 2``.

    Reported always, because it is bounded and therefore always finite — unlike
    KL, which is not.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return float(0.5 * _kl_raw(p, m) + 0.5 * _kl_raw(q, m))


def _kl_raw(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    if np.any(q[mask] <= 0):
        return math.inf
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """``KL(p ‖ q)`` in nats, or ``inf``.

    **The guard is reported, not assumed** (G8): ``KL(p* ‖ p_θ) = ∞`` whenever
    ``p_θ(X) = 0`` for a valid ``X``, which a deterministic policy produces.
    Under fix F10 every valid terminal is constructible, so a softmax policy
    keeps it finite — but that is a property of the policy, not of the metric.
    """
    return _kl_raw(np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64))


# --------------------------------------------------------------------------
# FCS
# --------------------------------------------------------------------------


def _restricted_tv(p: np.ndarray, q: np.ndarray, idx: np.ndarray) -> float:
    ps, qs = p[idx], q[idx]
    sp, sq = ps.sum(), qs.sum()
    if sp <= 0.0 or sq <= 0.0:
        # A subset carrying no mass under one of the two distributions has no
        # restriction to compare.  Counting it as maximal disagreement would be a
        # choice; counting it as 0 would hide one.  It cannot arise under P_S
        # when both distributions are positive, and P_S itself gives such a
        # subset weight 0 on the p_T side.
        return 1.0 if (sp > 0.0) != (sq > 0.0) else 0.0
    return float(0.5 * np.abs(ps / sp - qs / sq).sum())


def fcs(
    p_theta: np.ndarray,
    target: Target,
    *,
    m: int = FCS_M,
    n_subsets: int = FCS_SUBSETS,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Flow Consistency in Subgraphs, and its standard error (decision 20).

    **[EVIDENCE]** Definition 1 of *When Do GFlowNets Learn the Right
    Distribution?* (ICLR 2025, Spotlight): for a positive distribution ``P_S``
    over ``m``-sized subsets of the terminal set,
    ``FCS(p_T, R) := E_{S ~ P_S}[ TV( p_T|S , R|S ) ]``.  The paper relates it to
    TV (Thm. 5, Cor. 1) and, in a section titled *Inadequacy of commonly used
    evaluation protocols*, finds it "often the only metric accurately
    reflecting" convergence.

    **It needs the reward, not just the policy** — ``R`` is an argument, and a
    signature without it cannot compute the metric it names.  ``R`` enters
    through ``target.p_star``, which is proportional to it by construction, so
    the restriction-and-renormalise step is identical either way.

    **Sampling ``P_S`` exactly.**  Corollary 1's ``P_S(S; m) ∝ Σ_{x∈S} p_T(x)``
    normalises to ``C(n−1, m−1)`` because ``Σ_x p_T(x) = 1``.  So drawing one
    distinguished element ``x ~ p_T`` and then ``m−1`` others uniformly from the
    remaining ``n−1`` yields exactly ``P_S`` — no rejection, no weights.

    ``FAIL`` participates: it is a terminal of this MDP and a member of ``p*``'s
    support, so excluding it would measure a different distribution than TV does.

    At ``m = 2`` FCS is a ratio-matching-style metric and at ``m = #terminals``
    it *is* TV, so ``m = 8`` on a 200-5,000-terminal lattice is a genuinely
    different statistic — and this is the one environment where its gap to the
    true TV can be measured rather than assumed.
    """
    p = np.asarray(p_theta, dtype=np.float64)
    star = target.p_star
    n = p.shape[0]
    if star.shape[0] != n:
        raise ValueError(f"p_theta has {n} entries, target has {star.shape[0]}")
    if not 2 <= m <= n:
        raise ValueError(f"need 2 <= m <= {n} outcomes, got m={m}")
    rng = rng if rng is not None else np.random.default_rng(FCS_SEED)

    total = p.sum()
    if total <= 0:
        raise ValueError("p_theta carries no mass; P_S is undefined")
    weights = p / total

    draws = np.empty(n_subsets, dtype=np.float64)
    heads = rng.choice(n, size=n_subsets, p=weights)
    # Chunked so the key matrix stays bounded: at 5,000 terminals a single
    # (n_subsets x n) draw would be 80 MB for a metric budgeted at 0.05 s.
    chunk = max(1, min(n_subsets, 4_000_000 // max(n, 1)))
    for lo in range(0, n_subsets, chunk):
        hi = min(lo + chunk, n_subsets)
        head = heads[lo:hi]
        # The m-1 smallest of n-1 i.i.d. uniforms is a uniformly random
        # (m-1)-subset of the remaining outcomes, which is exactly what
        # Corollary 1's P_S needs after the head is drawn from p_T.
        keys = rng.random((hi - lo, n - 1))
        others = np.argpartition(keys, m - 2, axis=1)[:, : m - 1]
        others = others + (others >= head[:, None])
        idx = np.concatenate((head[:, None], others), axis=1)
        ps, qs = p[idx], star[idx]
        sp, sq = ps.sum(axis=1), qs.sum(axis=1)
        good = (sp > 0.0) & (sq > 0.0)
        block = np.where((sp > 0.0) != (sq > 0.0), 1.0, 0.0)
        if np.any(good):
            diff = np.abs(
                ps[good] / sp[good, None] - qs[good] / sq[good, None]
            ).sum(axis=1)
            block[good] = 0.5 * diff
        draws[lo:hi] = block

    mean = float(draws.mean())
    se = float(draws.std(ddof=1) / math.sqrt(n_subsets)) if n_subsets > 1 else 0.0
    return mean, se


def divergence_report(
    p_theta: np.ndarray,
    target: Target,
    *,
    m: int = FCS_M,
    n_subsets: int = FCS_SUBSETS,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Every distribution number for one policy, in one place.

    Exists so the reporting rules cannot drift apart from each other:

    * **``p*(FAIL)`` sits beside TV and is never subtracted from it** (G4).
      ``FAIL`` is in *both* distributions — that is why fix F3 put it in the
      target's support — so a policy assigning it exactly ``p*(FAIL)`` reaches
      TV = 0.  The convergence target is 0 and stays 0.  The one true statement
      is narrower: *a policy that cannot reach ``FAIL`` carries* ``TV >=
      p*(FAIL)``, which is a useful diagnostic and not a floor on the metric.
    * **FCS is reported alongside exact TV, with its standard error** (v1.2 §6.4,
      Gate 2 item 3).
    * **KL appears only when finite**, with the guard reported rather than
      assumed (G8).
    """
    value, se = fcs(p_theta, target, m=m, n_subsets=n_subsets, rng=rng)
    divergence = kl(target.p_star, p_theta)
    return {
        "tv": tv(p_theta, target.p_star),
        "js": js(p_theta, target.p_star),
        "kl": divergence,
        "kl_finite": math.isfinite(divergence),
        "fcs": value,
        "fcs_se": se,
        "fcs_m": float(m),
        # Reported beside TV, never subtracted from it.
        "target_p_fail": target.target_p_fail,
        "policy_p_fail": float(p_theta[-1]),
        "partition_residual": partition_residual(p_theta),
    }


def fcs_exact(p_theta: np.ndarray, p_star: np.ndarray, m: int) -> float:
    """FCS by exhaustive enumeration of all ``C(n, m)`` subsets.

    The reference the sampler is checked against (exit criterion 7).  Only
    tractable for a handful of outcomes, which is exactly why the sampler exists.
    """
    p = np.asarray(p_theta, dtype=np.float64)
    star = np.asarray(p_star, dtype=np.float64)
    n = p.shape[0]
    norm = math.comb(n - 1, m - 1)
    acc = 0.0
    for combo in itertools.combinations(range(n), m):
        idx = np.asarray(combo, dtype=np.int64)
        weight = float(p[idx].sum()) / norm
        acc += weight * _restricted_tv(p, star, idx)
    return acc


# --------------------------------------------------------------------------
# the β-dependent fingerprint
# --------------------------------------------------------------------------


def target_fingerprint(target: Target, env_fingerprint: str | None = None) -> str:
    """Digest of the β-dependent layer, **and of the computed target itself**.

    Hashing only the inputs would repeat, one layer up, the mistake
    :func:`~graft.synth.enumerate.environment_fingerprint` fixes: two
    implementations could compute *different* targets from identical inputs and
    still agree.  So the terminal ids and their ``p*`` values go in, quantised to
    :data:`TARGET_QUANTUM`.

    ``Z`` goes in at 12 significant digits rather than at an absolute quantum:
    it is a sum over thousands of rewards and is ``O(1e5)``, so an absolute
    1e-12 comparison on it would flag two correct implementations.
    """
    env = (
        env_fingerprint
        if env_fingerprint is not None
        else environment_fingerprint(target.instance, target.graph)
    )
    g = target.graph
    terminals = [
        [",".join(sorted(g.atoms_of(int(t)))), int(round(float(p) / TARGET_QUANTUM))]
        for t, p in zip(g.terminal_ix.tolist(), target.p_star[:-1].tolist())
    ]
    terminals.sort()
    return digest_of(
        {
            "environment": env,
            "beta": target.beta,
            "u_weights": target.cfg.u_weights.to_dict(),
            "r_fail": target.cfg.r_fail,
            "terminals": terminals,
            "p_fail": int(round(target.target_p_fail / TARGET_QUANTUM)),
            "z": float(f"{target.z:.12e}"),
        }
    )
