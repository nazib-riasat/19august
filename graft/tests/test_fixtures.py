"""The fixture generator itself.

Everything in the Phase-1 suite is measured against these instances, so the
properties they are *assumed* to have need asserting somewhere. A fixture that
quietly stopped producing a quarantined assertion would turn sub-check 7's
negative case into a test that passes by vacuity.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from graft.config import load_config
from graft.core import resolve
from graft.core.checker import H
from graft.schemas import PAYLOAD_ASSERTION_ID, PAYLOAD_TIER, AtomPool
from graft.tests.fixtures import build_instance, closed_subsets, instances, random_subsets

CFG = load_config()


@pytest.fixture
def inst():
    return build_instance(random.Random(13), scoped=True)


def test_the_pool_validates(inst):
    """Phase 0's generators cannot do this: their refs are random tokens."""
    inst.pool.validate()
    assert len(inst.pool) > 0


def test_every_target_resolves_in_the_backing_snapshot(inst):
    """Phase-1 gap G2's second half: without this, checks 3/4/5/7 would only ever
    be exercised in-pool and would first meet a real snapshot in Phase 9."""
    for atom in inst.pool:
        assert resolve.target_resolves(atom, inst.graph), atom


def test_gold_is_closed_under_refs(inst):
    """Otherwise it is not constructible and every sufficiency test is nonsense."""
    referenced = {r for a in inst.gold.atoms for r in inst.pool[a].refs}
    assert referenced <= set(inst.gold.atoms)


def test_gold_is_formally_valid(inst):
    assert H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG).ok


def test_gold_fits_the_atom_budget(inst):
    assert 1 <= len(inst.gold) <= CFG.max_atoms


def test_there_is_a_quarantined_assertion(inst):
    """Sub-check 7's negative case depends on it."""
    quarantined = [
        aid
        for node in (inst.graph.node(a.target) for a in inst.pool if a.kind == "node")
        if node is not None and (aid := node.payload.get(PAYLOAD_ASSERTION_ID))
        and not inst.graph.is_eligible(aid)
    ]
    assert quarantined


def test_there_is_an_invalidated_edge(inst):
    """Sub-check 4's negative case depends on it."""
    counts = inst.graph.counts()
    assert counts["live_edges"] < counts["edges"]


def test_there_is_an_out_of_scope_atom(inst):
    """Sub-check 5's negative case depends on it."""
    outside = [
        a.atom_id
        for a in inst.pool
        if any(c != inst.conv_id for c in resolve.conv_ids(a, inst.graph, inst.pool)[0])
    ]
    assert outside


def test_there_is_a_temporally_disjoint_claim(inst):
    """Sub-check 3's negative case depends on it."""
    disjoint = [
        a.atom_id
        for a in inst.pool
        if (ivs := resolve.validity_intervals(a, inst.graph))
        and not any(iv.overlaps(inst.obligations.time_constraint) for iv in ivs)
    ]
    assert disjoint


def test_intervals_partially_overlap_the_constraint(inst):
    """Exact-or-disjoint only would collapse ``temporal_correctness`` to a
    presence flag while still passing a two-distinct-values check."""
    from graft.core.obligations import covered_fraction

    fractions = {
        round(covered_fraction(inst.obligations.time_constraint, ivs), 6)
        for a in inst.pool
        if (ivs := resolve.validity_intervals(a, inst.graph))
    }
    assert any(0.0 < f < 1.0 for f in fractions), fractions


def test_source_tiers_are_mixed(inst):
    """``source_quality`` is constant otherwise (Phase-1 gap G7)."""
    tiers = {
        node.payload.get(PAYLOAD_TIER)
        for node in (inst.graph.node(a.target) for a in inst.pool if a.kind == "node")
        if node is not None and node.ntype == "Source"
    }
    assert len(tiers) >= 2
    assert tiers <= set(CFG.source_tiers)


def test_features_are_non_trivial(inst):
    """``redundancy`` needs both overlapping and complementary atoms."""
    feats = [a.feat for a in inst.pool if a.feat.size]
    assert len(feats) == len(inst.pool)
    assert len({tuple(np.round(f, 4)) for f in feats}) > 1


def test_pools_stay_within_their_cap():
    for instance in instances(seed=13, count=12):
        assert len(instance.pool) <= instance.pool.cap


def test_instances_are_deterministic():
    """Two teammates must generate the same fixtures from the same seed."""
    left = build_instance(random.Random(99))
    right = build_instance(random.Random(99))
    assert left.pool.ids() == right.pool.ids()
    assert left.graph.state_digest() == right.graph.state_digest()
    assert left.gold == right.gold


def test_closed_subsets_are_actually_closed():
    rng = random.Random(2)
    instance = build_instance(rng)
    for subset in closed_subsets(rng, instance.pool, 200, CFG.max_atoms):
        for aid in subset:
            assert all(r in subset for r in instance.pool[aid].refs)


def test_random_subsets_are_mostly_not_closed():
    """The checker suite needs invalid sets; a generator that only made valid
    ones would exercise nothing."""
    rng = random.Random(3)
    instance = build_instance(rng)
    subsets = list(random_subsets(rng, instance.pool, 200, CFG.max_atoms))
    unclosed = sum(
        1
        for s in subsets
        if any(r not in s for aid in s for r in instance.pool[aid].refs)
    )
    assert unclosed > len(subsets) // 4


def test_an_unbounded_constraint_can_be_requested():
    instance = build_instance(random.Random(4), bounded_constraint=False)
    assert instance.obligations.time_constraint.end is None
