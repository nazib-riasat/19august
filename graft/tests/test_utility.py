"""Executable ``U``.

Discharges Phase-1 exit criteria 7 (every term in [0, 1] over 10^4 generated
inputs), 8 (order-invariance), 9 (non-degeneracy — every term takes at least two
distinct values) and 10 (redundancy is 0 for a singleton and rises with
duplication).

Criterion 9 is the one that earns its keep: it is what catches a term that has
quietly stopped discriminating, which is the failure gaps G5 and G7 exist to
prevent and which would otherwise surface as an inexplicable Gate-2 table.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from graft.config import load_config
from graft.core.utility import (
    U,
    U_TERMS,
    GoldRequired,
    coverage,
    redundancy,
    size,
    source_quality,
    sufficiency,
    temporal_correctness,
    u_terms,
)
from graft.schemas import AtomPool, CandidateAtom, Interval, Obligations, ProofSet
from graft.graphstore import DictGraphSnapshot
from graft.tests.fixtures import build_instance, closed_subsets, instances, random_subsets

CFG = load_config()


@pytest.fixture
def inst():
    return build_instance(random.Random(13))


def _sample_terms(count_per_instance=650):
    """Terms over a broad spread of sets, reused by criteria 7 and 9."""
    rng = random.Random(20260808)
    rows = []
    for instance in instances(seed=71, count=8, scoped=False):
        subsets = list(random_subsets(rng, instance.pool, count_per_instance, CFG.max_atoms))
        subsets += list(closed_subsets(rng, instance.pool, count_per_instance, CFG.max_atoms))
        for subset in subsets:
            rows.append(
                u_terms(
                    subset,
                    instance.obligations,
                    instance.graph,
                    instance.pool,
                    instance.gold,
                    CFG,
                )
            )
    return rows


TERM_ROWS = _sample_terms()


# -- criterion 7 ------------------------------------------------------------


def test_every_term_is_in_range_over_ten_thousand_inputs():
    assert len(TERM_ROWS) >= 10_000
    for row in TERM_ROWS:
        for name in U_TERMS:
            assert 0.0 <= row[name] <= 1.0, f"{name} = {row[name]}"


def test_u_is_bounded_by_the_declared_range():
    w = CFG.u_weights
    for row in TERM_ROWS[:2000]:
        value = (
            w.suff * row["sufficiency"]
            + w.cov * row["coverage"]
            + w.src * row["source_quality"]
            + w.temp * row["temporal_correctness"]
            - w.red * row["redundancy"]
            - w.size * row["size"]
        )
        assert w.u_min <= value <= w.u_max


# -- criterion 9 ------------------------------------------------------------


def test_no_term_is_degenerate():
    """Each of the six must take at least two distinct values.

    A constant term is weight doing nothing.  ``temporal_correctness`` would be
    constant if ``H`` and ``U`` shared a predicate (G5); ``source_quality`` would
    be constant if every atom resolved to one tier (G7).
    """
    for name in U_TERMS:
        distinct = {round(row[name], 9) for row in TERM_ROWS}
        assert len(distinct) >= 2, f"{name} is constant at {distinct}"


def test_the_graded_terms_are_actually_graded():
    """Two distinct values is not enough for the two terms that are *supposed* to
    be continuous.

    A ``temporal_correctness`` taking only {0, 1} would pass the check above
    while behaving as a presence flag — which is exactly the collapse G5 split it
    out to avoid, and it would be invisible unless the fixtures contain intervals
    that *partially* cover the constraint.
    """
    for name in ("temporal_correctness", "redundancy"):
        interior = {
            round(row[name], 9) for row in TERM_ROWS if 0.0 < row[name] < 1.0
        }
        assert len(interior) >= 2, f"{name} only ever takes endpoint values"


def test_the_diagnostics_are_clean_on_well_formed_fixtures():
    assert all(row["_featureless_atoms"] == 0.0 for row in TERM_ROWS)
    assert all(row["_temporal_unbounded"] == 0.0 for row in TERM_ROWS)


# -- criterion 8 ------------------------------------------------------------


def test_u_is_order_invariant(inst):
    forward = U(sorted(inst.gold.atoms), inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    backward = U(
        sorted(inst.gold.atoms, reverse=True),
        inst.obligations,
        inst.graph,
        inst.pool,
        inst.gold,
        CFG,
    )
    as_set = U(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    assert forward == backward == as_set


# -- criterion 10 -----------------------------------------------------------


def _pool_of(vectors):
    atoms = [
        CandidateAtom(atom_id=f"a{i}", kind="node", target=f"n{i}", feat=np.array(v, np.float32))
        for i, v in enumerate(vectors)
    ]
    return AtomPool(atoms)


def test_redundancy_is_zero_for_a_singleton():
    pool = _pool_of([[1, 0, 0], [0, 1, 0]])
    assert redundancy(["a0"], pool) == 0.0


def test_redundancy_is_zero_for_an_orthogonal_pair():
    pool = _pool_of([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert redundancy(["a0", "a1"], pool) == pytest.approx(0.0, abs=1e-12)


def test_redundancy_rises_when_an_atom_is_replaced_by_a_near_duplicate():
    pool = _pool_of([[1, 0, 0], [0, 1, 0], [0.99, 0.01, 0.0]])
    complementary = redundancy(["a0", "a1"], pool)
    duplicated = redundancy(["a0", "a2"], pool)
    assert duplicated > complementary


def test_redundancy_approaches_one_for_identical_evidence():
    pool = _pool_of([[1, 0], [1, 0], [1, 0], [1, 0]])
    assert redundancy(["a0", "a1", "a2", "a3"], pool) == pytest.approx(0.75, abs=1e-9)


def test_the_ground_set_is_the_pool_not_the_selection():
    """With ``V = X`` every ``v`` has ``max_x sim(v, x) = 1``, so ``F(X) = 1``
    identically and the term is meaningless."""
    small = _pool_of([[1, 0], [0, 1]])
    large = _pool_of([[1, 0], [0, 1], [1, 1], [2, 1], [0.5, 0.9]])
    assert redundancy(["a0", "a1"], small) != redundancy(["a0", "a1"], large)


def test_negative_cosines_are_clamped_not_propagated():
    """Raw cosines are negative about half the time, which would make ``F({x})``
    negative, break monotonicity from ``F(0) = 0``, and push the ratio outside
    [0, 1]."""
    pool = _pool_of([[1, 0], [-1, 0], [0, 1], [0, -1]])
    for subset in (["a0", "a1"], ["a1", "a3"], ["a0", "a1", "a2", "a3"]):
        assert 0.0 <= redundancy(subset, pool) <= 1.0


def test_all_zero_features_give_zero_rather_than_dividing_by_zero():
    pool = _pool_of([[0, 0], [0, 0]])
    assert redundancy(["a0", "a1"], pool) == 0.0


def test_featureless_atoms_are_reported_not_silently_zeroed(inst):
    """A misconfigured featurizer must be visible, not silently worth 0.25."""
    bare = AtomPool(
        [
            CandidateAtom(atom_id="x", kind="node", target="n1"),
            CandidateAtom(atom_id="y", kind="node", target="n2"),
        ]
    )
    graph = DictGraphSnapshot()
    terms = u_terms(["x", "y"], Obligations(), graph, bare, ProofSet(frozenset({"x"})), CFG)
    assert terms["redundancy"] == 0.0
    assert terms["_featureless_atoms"] == 2.0


# -- individual terms -------------------------------------------------------


def test_sufficiency_is_the_fraction_of_gold_covered():
    gold = ProofSet(frozenset({"a", "b", "c", "d"}))
    assert sufficiency([], gold) == 0.0
    assert sufficiency(["a", "b"], gold) == 0.5
    assert sufficiency(["a", "b", "c", "d"], gold) == 1.0


def test_a_superset_of_gold_is_fully_sufficient_and_pays_elsewhere():
    """Being right is worth more than being small, but not for free."""
    gold = ProofSet(frozenset({"a"}))
    assert sufficiency(["a", "b", "c"], gold) == 1.0
    assert size(["a", "b", "c"], CFG) > size(["a"], CFG)


def test_source_quality_spans_the_configured_tiers(inst):
    scores = {
        round(source_quality([a.atom_id], inst.pool, inst.graph, CFG), 6) for a in inst.pool
    }
    assert len(scores) >= 2
    assert scores <= set(CFG.source_tiers.values())


def test_an_unresolvable_source_scores_the_default_tier_not_zero():
    """A missing edge makes evidence weak; disqualifying it is H's job."""
    graph = DictGraphSnapshot()
    pool = AtomPool([CandidateAtom(atom_id="x", kind="node", target="nowhere")])
    assert source_quality(["x"], pool, graph, CFG) == CFG.source_tiers[CFG.default_tier]


def test_source_quality_of_an_empty_set_is_zero():
    assert source_quality([], AtomPool([]), DictGraphSnapshot(), CFG) == 0.0


def test_temporal_correctness_grades_precision_not_mere_consistency(inst):
    """G5's split: H rejects contradictions, U grades how much of the window the
    evidence actually pins down."""
    q = inst.obligations
    full = temporal_correctness(inst.gold.atoms, inst.pool, q, inst.graph)
    without_interval = temporal_correctness(
        [a for a in inst.gold.atoms if not inst.pool[a].label.startswith("valid_during")],
        inst.pool,
        q,
        inst.graph,
    )
    assert full == 1.0
    assert without_interval <= full


def test_temporal_correctness_is_one_without_a_constraint(inst):
    unconstrained = Obligations(entity_anchor=inst.obligations.entity_anchor)
    assert temporal_correctness(inst.gold.atoms, inst.pool, unconstrained, inst.graph) == 1.0


def test_size_is_normalised_by_max_atoms():
    assert size([], CFG) == 0.0
    assert size(range(CFG.max_atoms), CFG) == 1.0
    assert size(range(CFG.max_atoms * 2), CFG) == 1.0  # clamped, never above 1


def test_coverage_matches_the_obligation_module(inst):
    from graft.core import obligations as ob

    ids = sorted(inst.gold.atoms)
    assert coverage(ids, inst.pool, inst.obligations, inst.graph) == ob.coverage(
        ob.slot_status(ids, inst.pool, inst.obligations, inst.graph), inst.obligations
    )


# -- gold is mandatory ------------------------------------------------------


def test_u_without_gold_raises(inst):
    """Fix F1: train-time U is measured against gold; inference uses the
    distilled head and never calls this."""
    with pytest.raises(GoldRequired, match="U requires gold"):
        U(inst.gold, inst.obligations, inst.graph, inst.pool, None, CFG)


def test_gold_scores_higher_than_a_random_subset(inst):
    """U must genuinely discriminate, or the target is uniform over valid sets."""
    rng = random.Random(2)
    best = U(inst.gold, inst.obligations, inst.graph, inst.pool, inst.gold, CFG)
    for subset in random_subsets(rng, inst.pool, 50, CFG.max_atoms):
        if set(subset) == set(inst.gold.atoms):
            continue
        assert U(subset, inst.obligations, inst.graph, inst.pool, inst.gold, CFG) <= best
