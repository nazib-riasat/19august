"""P4.7 — the Gate-3 harness: the matrix, the ceiling row, the audits.

**Phase 4 publishes a diagnostic, not a Gate-3 verdict** (decision 5, ruled
before any method ran, from properties of ``p*``, ``U`` and greedy — no learner
result inspected, so `GRAFT_PHASE2_BUILD.md` §6b's second procedure is
satisfied). Three candidate criteria were *measured* before being written down
and none survived as a gate:

============================  ==========================================
best-of-K vs any rival        greedy is globally optimal on 30/30; a
                              flawless sampler is 0.038 short. Arithmetic.
distinct-set **count**        capped at ``K = 8``; beam saturates it on
                              20/20 while the ``p*`` ceiling is 7.78, so a
                              "strictly more" rule has ``P(pass) = 0``.
distinct-set **diversity**    live, but near-definitional: a GFlowNet is
                              *defined* as sampling ∝ reward, so conditional
                              on Gate 2 passing this is determined by Gate 2.
============================  ==========================================

So the gate's **decision moves to Phase 9**, and exactly one necessary condition
stays testable here: **S5's portfolio diversity must exceed the training-free
arms'**. A mode-collapsed or under-trained sampler fails it, and that failure
would mean the flow method is not producing a portfolio at all — its
Robust-Scheduling justification gone before the noisy-scorer test.

**Three things this harness must do that "reuse ``paired_bootstrap`` verbatim"
does not** (G4, G5):

1. **Negate before calling.** ``gate2.paired_bootstrap`` computes ``wins =
   upper < 0`` for exact TV, where *lower is better*. Every Phase-4 metric is
   *higher* better, so passing them unnegated would report the winner as the
   loser.
2. **Broadcast deterministic arms** to the frozen seed count, so shapes match.
   Their seed variance really *is* zero; broadcasting is what lets the bootstrap
   see zero rather than raise.
3. **Compute the ``p*`` ceiling row** — ``E[best-of-K | p*]`` in closed form —
   printed beside every method's best-of-K, so a loss splits into
   S5-to-ceiling (**learning**) and ceiling-to-greedy (**an artefact of fix
   F13's perfect scorer**).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from graft.core.utility import U
from graft.ledger import Ledger
from graft.schemas import ProofSet
from graft.setgen.gate2 import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, audit_block, paired_bootstrap
from graft.setgen.search.base import SearchResult
from graft.setgen.search.relevance import RELEVANCE_VARIANTS, relevance_vector

__all__ = [
    "Gate3Report",
    "best_of_k_ceiling",
    "budget_curve",
    "exact_scorer",
    "excess_diversity",
    "jaccard_diversity",
    "higher_is_better_bootstrap",
    "random_portfolio_diversity",
    "run_stage_a",
]

#: Frozen stream for the size-control baseline below, for the same reason every
#: other estimator in this project is seeded: an unpinned control is not a
#: control.
DIVERSITY_CONTROL_SEED = 20260819
DIVERSITY_CONTROL_DRAWS = 2_000


def exact_scorer(env: Any) -> Callable[[Iterable[str]], float]:
    """Fix F13's scorer: **exact ``U``**, gold known, every term deterministic.

    The caveat travels with it (exit criterion 9): the comparison is fair across
    methods and **optimistic relative to deployment**, where Phase 9's distilled
    head is noisy.
    """
    inst = env.instance

    def scorer(atoms: Iterable[str]) -> float:
        return U(atoms, inst.obligations, inst.graph, inst.pool, inst.gold, inst.cfg)

    return scorer


def best_of_k_ceiling(target: Any, k: int) -> float:
    """``E[best-of-K | p*]`` — what a *flawless* sampler attains (G9, decision 10).

    Closed form over terminals sorted by ``U`` with ``F`` the cumulative ``p*``:

        ``E = Σ_i u_i · (F_i^K − F_{i−1}^K)``

    Costs nothing — ``Target`` already holds both vectors — and without it Gate 3
    reports a predetermined number as though it were a result.

    ``FAIL`` carries ``p*(FAIL) ≈ 2.5e−12`` and no utility, so it is excluded
    from the order statistic and its mass left in ``F``'s tail; at that magnitude
    it moves nothing, and pretending it has a utility would be worse than
    ignoring it.
    """
    u = np.asarray(target.u, dtype=np.float64)
    p = np.asarray(target.p_star[:-1], dtype=np.float64)
    order = np.argsort(u, kind="stable")
    u, p = u[order], p[order]
    cumulative = np.cumsum(p)
    lower = np.concatenate(([0.0], cumulative[:-1]))
    return float(np.sum(u * (cumulative**k - lower**k)))


def jaccard_diversity(sets: Sequence[ProofSet]) -> float:
    """Mean pairwise Jaccard **distance** over the C(n,2) unordered pairs.

    **The convention is pinned** (decision 5's amended cell) because this is the
    one live gate and a floating estimator would make it unreadable: duplicate
    pairs are **included**, so a collapsed portfolio scores low — which is the
    point. Fewer than two sets scores 0: a portfolio of one has no diversity,
    and ``nan`` would silently drop the arm from an aggregate.
    """
    if len(sets) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i].atoms, sets[j].atoms
            union = len(a | b)
            total += 1.0 - (len(a & b) / union if union else 1.0)
            pairs += 1
    return total / pairs


@lru_cache(maxsize=4096)
def _expected_pair_distance(size_a: int, size_b: int, pool_size: int) -> float:
    """``E[Jaccard distance]`` for two uniformly random sets of these sizes.

    Canonicalised on the size pair: the quantity is symmetric, and seeding on
    the ordered pair made ``E(2,3) ≠ E(3,2)`` by ~3e-3 of MC noise — so the
    published baseline depended on portfolio *order*, not only its size
    multiset (post-fix audit, 13 Aug 2026).
    """
    size_a, size_b = sorted((size_a, size_b))
    if pool_size <= 0 or size_a <= 0 or size_b <= 0:
        return 0.0
    rng = np.random.default_rng(DIVERSITY_CONTROL_SEED + 1000 * size_a + size_b)
    universe = np.arange(pool_size)
    total = 0.0
    for _ in range(DIVERSITY_CONTROL_DRAWS):
        a = set(rng.choice(universe, size=min(size_a, pool_size), replace=False).tolist())
        b = set(rng.choice(universe, size=min(size_b, pool_size), replace=False).tolist())
        union = len(a | b)
        total += 1.0 - (len(a & b) / union if union else 1.0)
    return total / DIVERSITY_CONTROL_DRAWS


def random_portfolio_diversity(sets: Sequence[ProofSet], pool_size: int) -> float:
    """What :func:`jaccard_diversity` would score for **random** sets of these
    same sizes drawn from a pool this large.

    The size control, and it is needed rather than fastidious: under this
    function's own seeded estimator on a 24-atom pool, a portfolio of 8
    uniformly random sets scores **0.945** at set size 2 and **0.792** at size
    8 — mean pairwise Jaccard distance falls monotonically as sets grow,
    because two small sets can be disjoint while two large ones drawn from the
    same pool must overlap.
    """
    if len(sets) < 2:
        return 0.0
    sizes = [len(s.atoms) for s in sets]
    total, pairs = 0.0, 0
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            total += _expected_pair_distance(sizes[i], sizes[j], pool_size)
            pairs += 1
    return total / pairs


def excess_diversity(sets: Sequence[ProofSet], pool_size: int) -> float:
    """Observed diversity **minus** the random-portfolio baseline at the same sizes.

    **Reported beside the ruled metric, never instead of it.** Decision 5 makes
    raw mean pairwise Jaccard distance the one live Gate-3 condition, and that
    ruling stands; this is the diagnostic that says how much of any gap is
    mechanical. On the main suite the difference is not academic — S4 under
    informed relevance returns size-3.78 sets and scores **0.483** under the
    pinned duplicates-included convention (0.551 under the pre-fix deduplicated
    estimator), above ``p*``'s own **0.4506** either way — which would make the
    last surviving criterion a *fourth* predetermined rule (G5 already retired
    three). Under the control the comparison is between methods rather than
    between set sizes.
    """
    return jaccard_diversity(sets) - random_portfolio_diversity(sets, pool_size)


def higher_is_better_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """``paired_bootstrap`` with the sign corrected for a higher-better metric.

    ``gate2.paired_bootstrap``'s ``wins = upper < 0`` is written for exact TV.
    Negating both inputs turns "``a`` is smaller" into "``a`` is larger" without
    reimplementing the two-stage resampling — and the negation is **recorded in
    the return value**, so a reader cannot mistake which direction was tested.
    """
    out = paired_bootstrap(
        -np.asarray(a, dtype=np.float64),
        -np.asarray(b, dtype=np.float64),
        n_resamples=n_resamples,
        seed=seed,
    )
    out["negated_for_higher_is_better"] = True
    out["mean_difference"] = -out["mean_difference"]
    out["per_seed_difference"] = [-d for d in out["per_seed_difference"]]
    # ``upper_95`` stays on the negated scale, because that is the scale ``wins``
    # is defined on and renaming a bound is worse than leaving it. What was
    # genuinely misleading was publishing it beside a **re-negated** mean, so the
    # point estimate sat on the far side of its own "upper" bound. The bound in
    # the reader's frame is one-sided and *lower*, so it is published under that
    # name and the negated one is kept for the ``wins`` audit trail.
    out["lower_95_on_a_minus_b"] = -out["upper_95"]
    out["upper_95_on_negated_scale"] = out.pop("upper_95")
    return out


def _broadcast(values: Sequence[float], n_seeds: int) -> np.ndarray:
    """One row per seed for a deterministic arm (G4, decision 4).

    Its seed variance genuinely *is* zero, so broadcasting states the truth and
    lets the bootstrap see it; reporting one row while the test expects three
    was the contradiction decision 4 resolves.
    """
    return np.tile(np.asarray(values, dtype=np.float64), (n_seeds, 1))


class Gate3Report:
    """Stage A's table. ``S5`` stays empty until Phase 3's matrix exists (G7)."""

    def __init__(self, variant: str, edge_cost: float) -> None:
        self.variant = variant
        self.edge_cost = edge_cost
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.ceiling: list[float] = []
        self.global_max: list[float] = []
        self.audits: dict[str, Any] = {}
        #: Criterion 14 names **both** digests. ``audit_block`` carries only the
        #: environment one, and the ceiling and global-max rows are ``p*``-derived,
        #: so a ``u_weights`` move would otherwise pass unseen.
        self.fingerprints: list[dict[str, str]] = []
        self.stage = "A"

    def add(self, method: str, row: dict[str, Any]) -> None:
        self.rows.setdefault(method, []).append(row)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "stage": self.stage,
            "relevance_variant": self.variant,
            "edge_cost": self.edge_cost,
            "instances": len(self.ceiling),
            "ceiling_mean": float(np.mean(self.ceiling)) if self.ceiling else float("nan"),
            "global_max_mean": float(np.mean(self.global_max)) if self.global_max else float("nan"),
            "methods": {},
            "audits": self.audits,
            "fingerprints": self.fingerprints,
            "caveats": {
                "scorer": "exact U (fix F13): fair across methods, optimistic "
                          "relative to deployment, where Phase 9's head is noisy",
                "gate": "diagnostic only (decision 5): no best-of-K comparison "
                        "against a rival may be reported as a Gate-3 pass or failure",
                "checker_budget": "checker_budget = 32 is INERT here — the most any "
                                  "method can spend is K = 8, so the budget curve's "
                                  "levels are {1, 2, 4, 8} (G3)",
                "stage": "Stage A is not Gate 3: the table has no S5 row (G7)",
            },
        }
        for method, rows in self.rows.items():
            best = np.array([r["best_utility"] for r in rows], dtype=float)
            n_scored = int(np.sum(~np.isnan(best)))

            def _rate(key: str) -> float | None:
                # ``None``, not 0.0, when no row carries the key: publishing
                # 0.0 for a method that never reports completion reads as
                # "never needed completion", which is a different claim.
                vals = [r.get(key) for r in rows]
                vals = [v for v in vals if isinstance(v, (int, float))]
                return float(np.mean(vals)) if vals else None

            rejections: dict[str, int] = {}
            for r in rows:
                for check, n in (r.get("h_rejections") or {}).items():
                    rejections[check] = rejections.get(check, 0) + int(n)
            out["methods"][method] = {
                # **`nanmean` over the instances the method survived.** The
                # population is published beside it rather than implied, because
                # a conditional mean over 12 of 20 is not the same number as a
                # mean over 20 and the reader has to be able to tell.
                "best_utility_mean": float(np.nanmean(best)) if n_scored else float("nan"),
                "best_utility_scored_on": n_scored,
                "best_utility_mean_failures_as_zero": float(
                    np.nan_to_num(best, nan=0.0).mean()
                ),
                "instances_with_no_valid_set": int(np.sum(np.isnan(best))),
                "mean_utility_of_returned": float(
                    np.nanmean([r["mean_utility"] for r in rows])
                ),
                "portfolio_size_mean": float(np.mean([r["portfolio_size"] for r in rows])),
                "h_rejections_total": rejections,
                "completion_rate_mean": _rate("completion_rate"),
                "max_atoms_breaches_total": int(
                    sum(r.get("max_atoms_breaches", 0) or 0 for r in rows)
                ),
                "distinct_valid_mean": float(np.mean([r["distinct_valid"] for r in rows])),
                "diversity_mean": float(np.mean([r["diversity"] for r in rows])),
                "diversity_random_baseline_mean": float(
                    np.mean([r["diversity_random_baseline"] for r in rows])
                ),
                "excess_diversity_mean": float(
                    np.mean([r["excess_diversity"] for r in rows])
                ),
                "terminal_checks_mean": float(np.mean([r["terminal_checks"] for r in rows])),
                "terminal_checks_total": int(np.sum([r["terminal_checks"] for r in rows])),
                "size_mean": float(np.nanmean([r["mean_size"] for r in rows])),
                "gap_to_ceiling_mean": float(
                    np.nanmean(best - np.asarray(self.ceiling, dtype=float))
                ),
            }
        return out


def budget_curve(
    envs: Sequence[Any],
    *,
    variant: str = "obligation",
    edge_cost: float = 2.0,
    levels: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Best-of-K per method across the budget levels — criterion 10's first half.

    G3: because ``stop_allowed`` *is* ``H``, the mask-driven arms spend 0 checks
    and the direct builders 1 per distinct candidate, so ``checker_budget = 32``
    binds on nobody here and **the curve's levels are ``{1, 2, 4, 8}``** — the
    portfolio sizes themselves. G3 calls the resulting shape *"itself the
    finding"*: informative for S3/S4 (whose spend rises with K) and flat-to-
    saturating for the mask-driven arms.

    The ceiling row rides along at every level, because a best-of-K at ``k`` is
    only readable against ``E[best-of-k | p*]`` (decision 10).
    """
    curve: dict[str, Any] = {
        "variant": variant,
        "edge_cost": float(edge_cost),
        "levels": [int(k) for k in levels],
        "ceiling": [],
        "methods": {},
    }
    for k in levels:
        report = run_stage_a(envs, variant=variant, edge_cost=edge_cost, k=int(k))
        summary = report.summary()
        curve["ceiling"].append(summary["ceiling_mean"])
        for method, row in summary["methods"].items():
            curve["methods"].setdefault(
                method, {"best_of_k": [], "best_of_k_zero": [], "scored_on": [], "checks": []}
            )
            # Two readings, both published (§1.4 F8's rule): the conditional
            # mean can DIP as k grows — a larger portfolio lets a direct builder
            # survive on harder instances, and the newly-scored ones drag the
            # nanmean down — while the failures-as-zero mean is monotone
            # whenever the candidate sets nest (S1/S3's openers and S4's top-k
            # sweep are prefixes of the next level's). A dip in the first
            # column is a population effect, not a search regression.
            curve["methods"][method]["best_of_k"].append(row["best_utility_mean"])
            curve["methods"][method]["best_of_k_zero"].append(
                row["best_utility_mean_failures_as_zero"]
            )
            curve["methods"][method]["scored_on"].append(row["best_utility_scored_on"])
            curve["methods"][method]["checks"].append(row["terminal_checks_mean"])
    return curve


def run_stage_a(
    envs: Sequence[Any],
    *,
    variant: str = "obligation",
    edge_cost: float = 2.0,
    k: int | None = None,
) -> Gate3Report:
    """S1–S4 across a suite, metered, with the ceiling row (build step 6).

    **Stage A only** (G7, exit criterion 15): a table without S5 is labelled
    Stage A and is *not* called Gate 3. Stage B fills S5's row from a Phase-3
    checkpoint and applies decision 5's rule.

    The ledger is opened **per (method, instance)** — one query scope each,
    since ``checker_budget`` is a *per-query* cap (fix F5) and a shared scope
    would let one method's spend starve another's.
    """
    if variant not in RELEVANCE_VARIANTS:
        raise ValueError(f"unknown relevance variant {variant!r}")

    from graft.setgen.search.s1_greedy import GreedySearch
    from graft.setgen.search.s2_beam import BeamSearch
    from graft.setgen.search.s3_submodular import SubmodularGreedy
    from graft.setgen.search.s4_pcst import PCSTSearch

    report = Gate3Report(variant, edge_cost)
    for env in envs:
        inst = env.instance
        cfg = inst.cfg
        k_eff = int(k if k is not None else cfg.K)
        scorer = exact_scorer(env)
        rel = relevance_vector(env, variant)
        report.ceiling.append(best_of_k_ceiling(env.target, k_eff))
        report.global_max.append(float(np.max(env.target.u)))

        methods = [
            GreedySearch(rel, k_eff),
            BeamSearch(k=k_eff),
            SubmodularGreedy(rel, k_eff),
            PCSTSearch(rel, edge_cost, k_eff),
        ]
        for method in methods:
            ledger = Ledger.from_config(cfg)
            with ledger.query_scope(f"{method.name}:{inst.meta.get('seed', '')}"):
                result = method.run(env, inst.obligations, scorer, ledger)
                spent = ledger.snapshot()["query"]["terminal_checks"]
            if spent != result.terminal_checks:
                raise RuntimeError(
                    f"{method.name} reported {result.terminal_checks} terminal "
                    f"checks but the ledger metered {spent}; the reported spend "
                    "is the number the budget curve is drawn from"
                )
            row = result.to_dict()
            # **Over the portfolio, not the deduplicated ranking** — decision 5
            # pins "the C(K,2) unordered pairs of the returned sets, duplicate
            # pairs included", and ``sets`` is deduplicated for ranking. Measured,
            # the deduplicated estimator passes a sampler that has lost 27% of
            # its modes where the pinned one fails it, and this is the one live
            # Gate-3 condition, so the estimator may not float.
            row["diversity"] = jaccard_diversity(result.portfolio)
            # The size control, reported beside the ruled metric rather than
            # instead of it, on the same population.
            row["diversity_random_baseline"] = random_portfolio_diversity(
                result.portfolio, len(inst.pool.ids())
            )
            # Through the exported helper, so the definition cannot drift from
            # the one external callers get (post-fix audit NOTE, 13 Aug 2026).
            row["excess_diversity"] = excess_diversity(
                result.portfolio, len(inst.pool.ids())
            )
            row["mean_size"] = (
                float(np.mean([len(s.atoms) for s in result.sets]))
                if result.sets
                else float("nan")
            )
            # Criteria 11c/12/12d: the per-method structural numbers the plan
            # requires reported.  They reach ``to_dict`` through ``extra`` and
            # were being dropped by the summary, so the rows carried them and
            # the artefact did not.
            row["mean_utility"] = (
                float(np.mean(result.scores)) if result.scores else float("nan")
            )
            report.add(method.name, row)

    report.audits = audit_block(envs)
    report.fingerprints = [env.fingerprints() for env in envs]
    return report
