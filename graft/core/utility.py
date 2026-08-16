"""Executable ``U`` — six terms, every one normalised to [0, 1].

v1.2 §4.1 calls publishing ``U`` as an executable function a Gate-0 blocker: a
word description cannot be trained against, reproduced, or held fixed across
baselines. This module is that function.

```
U = w_suff*sufficiency + w_cov*coverage + w_src*source_quality
  + w_temp*temporal_correctness - w_red*redundancy - w_size*size
```

Every term lands in [0, 1] (Phase-0 gap G4), so ``U`` is bounded by
``[-(w_red + w_size), w_suff + w_cov + w_src + w_temp]`` and ``beta`` scales a
bounded quantity. Without that the Phase-3 ``beta`` sweep is uninterpretable.

``U`` is **frozen across every row of the learning comparison** (v1.2 §5.1). If
different learners see different rewards, the comparison measures reward
engineering rather than learning.

``U`` must also genuinely *discriminate*: if every formally valid set scored the
same, the target would be uniform over valid sets and the objective would apply
no pressure toward stronger or smaller proofs. That is why exit criterion 9
requires each term to take at least two distinct values on the fixtures — a term
that has quietly become constant fails the build rather than a Gate-2 table.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping

import numpy as np

from graft.config import Config, UWeights
from graft.core import obligations, resolve
from graft.graphstore import GraphSnapshot
from graft.schemas import AtomPool, Obligations, ProofSet

__all__ = [
    "U",
    "u_terms",
    "sufficiency",
    "coverage",
    "source_quality",
    "temporal_correctness",
    "redundancy",
    "size",
    "U_TERMS",
    "GoldRequired",
]

U_TERMS: tuple[str, ...] = (
    "sufficiency",
    "coverage",
    "source_quality",
    "temporal_correctness",
    "redundancy",
    "size",
)


class GoldRequired(ValueError):
    """Raised when ``U`` is called without gold.

    Train-time ``U`` is deterministic against gold (architecture fix F1);
    inference ranks with the distilled utility head (Phase 9) and never calls
    this function.  Failing loudly is what stops a silent ``sufficiency = 0``
    from looking like a legitimately weak proof.
    """


def _ids(X: ProofSet | Iterable[str]) -> tuple[str, ...]:
    atoms = X.atoms if isinstance(X, ProofSet) else X
    return tuple(sorted(atoms))


# --------------------------------------------------------------------------
# the six terms
# --------------------------------------------------------------------------


def sufficiency(X: ProofSet | Iterable[str], gold: ProofSet | Iterable[str]) -> float:
    """Fraction of the gold proof's atoms this set covers.

    **[EVIDENCE]** Graph-S3 (ACL 2026) validates exactly this pattern of dense
    supervision from offline golden subgraphs — its dense-vs-sparse ablation
    (Table 3) gains +11.8 accuracy / +17.1 F1 macro (macro-averaged from Table 3 by this project; the pair is not printed in the paper) over sparse final-answer
    reward, in its own setting.  (Its separate headline, +8.1 accuracy / +9.7 F1,
    is measured against seven *baselines*, not against sparse reward.)

    A strict superset of gold still scores 1.0 and pays for the excess through
    ``size`` and ``redundancy``.  That is the intended minimality pressure: being
    right is worth more than being small, but not for free.
    """
    gold_ids = set(_ids(gold))
    if not gold_ids:
        # A question with no gold proof has nothing to be insufficient about.
        # Phase 2 must not emit these; the convention exists so the term is
        # total rather than to make them acceptable.
        return 1.0
    return len(gold_ids & set(_ids(X))) / len(gold_ids)


def coverage(
    X: ProofSet | Iterable[str], pool: AtomPool, q: Obligations, G: GraphSnapshot
) -> float:
    """Fraction of the question's active obligation slots that are addressed.

    Shares ``slot_status`` with ``d(s)`` so the reward and the policy feature
    cannot drift apart (Phase-1 gap G4).  See
    :func:`graft.core.obligations.coverage` for why a slot counts as addressed
    when its deficit is below 1 rather than by one minus the graded deficit.
    """
    return obligations.coverage(obligations.slot_status(_ids(X), pool, q, G), q)


def source_quality(
    X: ProofSet | Iterable[str], pool: AtomPool, G: GraphSnapshot, cfg: Config
) -> float:
    """Mean reliability of the selected atoms' sources, from ``cfg.source_tiers``.

    Metadata-derived, never learned (v1.2 §4.4 routing table).  An atom whose
    ``Source`` cannot be resolved scores ``default_tier``: a missing edge makes
    evidence weak, and disqualifying it is `H`'s job.
    """
    ids = [aid for aid in _ids(X) if aid in pool]
    if not ids:
        return 0.0
    return float(np.mean([resolve.source_tier(pool[aid], G, cfg) for aid in ids]))


def temporal_correctness(
    X: ProofSet | Iterable[str], pool: AtomPool, q: Obligations, G: GraphSnapshot
) -> float:
    """How much of the requested interval the selected evidence actually covers.

    ``H`` rejects only hard contradictions — evidence *disjoint* from the
    constraint — so every set reaching here is non-contradictory.  This grades
    precision instead: vague evidence scores low, tight and complete evidence
    scores 1.0.  Splitting the two is Phase-1 gap G5; without it, every set
    surviving ``H`` would score 1.0 and a fifth of ``U``'s positive weight would
    do nothing.

    1.0 when there is no constraint, and 1.0 when the constraint is unbounded —
    see :func:`graft.core.obligations.covered_fraction` for why, and for the
    diagnostic that keeps the concession visible.
    """
    if q.time_constraint is None:
        return 1.0
    intervals = [
        iv for aid in _ids(X) if aid in pool for iv in resolve.atom_intervals(pool[aid], G)
    ]
    return obligations.covered_fraction(q.time_constraint, intervals)


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=64)
def _similarity_matrix(pool: AtomPool) -> tuple[tuple[str, ...], np.ndarray]:
    """Pairwise ``max(0, cosine)`` over the pool's feature vectors.

    Clamped because raw cosines over arbitrary features are negative roughly half
    the time, which would make ``F({x})`` negative, break monotonicity from
    ``F(∅) = 0``, and drive the redundancy ratio outside [0, 1].

    **Cached per pool (Phase-2 gap G7).**  ``redundancy`` is called once per
    candidate set, and Phase 2 enumerates thousands of terminals against a single
    pool — rebuilding a 32x32 matrix each time is a 30-million-flop tax per
    instance for an answer that cannot have changed.  ``AtomPool`` defines no
    ``__eq__``, so it hashes by identity and the key is exactly "this pool
    object"; a pool built from the same atoms a second time is a cache miss,
    which is the conservative direction.

    The cache holds a strong reference to every pool it has seen, so ``maxsize``
    is bounded rather than ``None``: a long sweep that builds many pools must not
    keep all of them alive.  The returned array is marked read-only, because it
    is now shared between callers and a mutation would silently change every
    later ``redundancy``.
    """
    ids = pool.ids()
    if not ids:
        return ids, _frozen(np.zeros((0, 0), dtype=np.float64))
    width = max((pool[aid].feat.shape[0] for aid in ids), default=0)
    if width == 0:
        return ids, _frozen(np.zeros((len(ids), len(ids)), dtype=np.float64))
    matrix = np.zeros((len(ids), width), dtype=np.float64)
    for i, aid in enumerate(ids):
        feat = pool[aid].feat
        matrix[i, : feat.shape[0]] = feat
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    return ids, _frozen(np.clip(unit @ unit.T, 0.0, 1.0))


def redundancy(X: ProofSet | Iterable[str], pool: AtomPool) -> float:
    """Overlap between the selected atoms, as lost facility-location coverage.

    ``F(S) = (1/|V|) * sum_v max_{x in S} sim(v, x)`` over the **pool** as ground
    set, and ``red(X) = (sum_x F({x}) - F(X)) / sum_x F({x})``.

    0 for a perfectly complementary set, approaching 1 as atoms become
    interchangeable.  This is the v1.2 replacement for the degenerate
    ``max(0, S(X\\e) - S(X) + eps)`` form, which is near-constant whenever the
    proof score is monotone in evidence.  It is also the same objective family as
    the Phase-4 submodular baseline (S3), which makes the two directly
    comparable.  **[EVIDENCE]** Lin & Bilmes (ACL 2011) for the applied form;
    Nemhauser–Wolsey–Fisher (1978) for the guarantee.

    Ground set is the pool, not ``X``: with ``V = X`` every ``v`` has
    ``max_x sim(v, x) = sim(v, v) = 1``, so ``F(X) = 1`` identically and the term
    is meaningless.
    """
    ids = [aid for aid in _ids(X) if aid in pool]
    if len(ids) < 2:
        return 0.0
    pool_ids, sim = _similarity_matrix(pool)
    if sim.size == 0:
        return 0.0
    index = {aid: i for i, aid in enumerate(pool_ids)}
    columns = [index[aid] for aid in ids if aid in index]
    if len(columns) < 2:
        return 0.0

    singles = sim[:, columns].mean(axis=0)          # F({x}) for each selected x
    joint = float(sim[:, columns].max(axis=1).mean())  # F(X)
    total = float(singles.sum())
    if total <= 0.0:
        # All-zero or mutually orthogonal features.  Reported by u_terms'
        # diagnostic rather than silently worth a quarter of the reward.
        return 0.0
    return float(min(1.0, max(0.0, (total - joint) / total)))


def size(X: ProofSet | Iterable[str], cfg: Config) -> float:
    """``|X| / max_atoms``.

    One normalised atom count with one weight (Phase-0 gap G3).  Node/edge
    asymmetry is unmotivated and would add a free parameter to a reward that must
    stay frozen across seven learners.
    """
    return min(1.0, len(_ids(X)) / cfg.max_atoms)


# --------------------------------------------------------------------------
# U
# --------------------------------------------------------------------------


def u_terms(
    X: ProofSet | Iterable[str],
    q: Obligations,
    G: GraphSnapshot,
    pool: AtomPool,
    gold: ProofSet | Iterable[str] | None,
    cfg: Config,
) -> dict[str, float]:
    """Every term of ``U`` separately, plus diagnostics.

    Diagnostics are part of the return value rather than a side channel because
    both of them mark a term that has stopped discriminating, and a silently
    inert reward term is the failure mode gaps G5 and G7 exist to catch.
    """
    if gold is None:
        raise GoldRequired(
            "U requires gold. Train-time sufficiency is measured against the gold "
            "proof (fix F1); at inference the distilled utility head ranks instead "
            "and U is not called. Passing None would score a correct proof as "
            "sufficiency 0 and look like a legitimately weak one."
        )
    ids = _ids(X)
    terms = {
        "sufficiency": sufficiency(ids, gold),
        "coverage": coverage(ids, pool, q, G),
        "source_quality": source_quality(ids, pool, G, cfg),
        "temporal_correctness": temporal_correctness(ids, pool, q, G),
        "redundancy": redundancy(ids, pool),
        "size": size(ids, cfg),
    }
    terms["_featureless_atoms"] = float(
        sum(1 for aid in ids if aid in pool and pool[aid].feat.size == 0)
    )
    terms["_temporal_unbounded"] = float(
        q.time_constraint is not None
        and (q.time_constraint.start is None or q.time_constraint.end is None)
    )
    return terms


def U(
    X: ProofSet | Iterable[str],
    q: Obligations,
    G: GraphSnapshot,
    pool: AtomPool,
    gold: ProofSet | Iterable[str] | None,
    cfg: Config,
    w: UWeights | None = None,
) -> float:
    """The scalar utility of a formally valid set.

    Order-invariant by construction: every term reads the set, never a sequence.
    """
    weights = w if w is not None else cfg.u_weights
    t = u_terms(X, q, G, pool, gold, cfg)
    return (
        weights.suff * t["sufficiency"]
        + weights.cov * t["coverage"]
        + weights.src * t["source_quality"]
        + weights.temp * t["temporal_correctness"]
        - weights.red * t["redundancy"]
        - weights.size * t["size"]
    )
