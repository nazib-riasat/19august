"""Schema round-trips and the invariants the data model is supposed to enforce.

Discharges Phase-0 exit criterion 1 (round-trip, property-based).
"""

from __future__ import annotations

import dataclasses
import json
import random

import numpy as np
import pytest

from graft import schemas
from graft.canonical import canonical_json
from graft.schemas import (
    Assertion,
    AssertionFlags,
    CandidateAtom,
    Edge,
    Interval,
    Node,
    Obligations,
    ProofSet,
    SourceSpan,
)
from graft.tests.generators import GENERATORS

ROUNDS = 200


@pytest.mark.parametrize("cls", list(GENERATORS), ids=lambda c: c.__name__)
def test_round_trip_is_identical(cls):
    """Criterion 1: every Tier-A and Tier-B schema survives to_dict/from_dict."""
    rng = random.Random(20260808)
    generate = GENERATORS[cls]
    for _ in range(ROUNDS):
        original = generate(rng)
        as_dict = original.to_dict()
        restored = cls.from_dict(json.loads(json.dumps(as_dict)))
        assert restored.to_dict() == as_dict
        assert canonical_json(restored.to_dict()) == canonical_json(as_dict)


def test_every_schema_class_has_a_generator():
    """A new schema class must join the round-trip test, not skip it."""
    exported = {
        getattr(schemas, name)
        for name in schemas.__all__
        if dataclasses.is_dataclass(getattr(schemas, name, None))
    }
    # AssertionFlags is exercised through Assertion, which embeds it.
    missing = exported - set(GENERATORS) - {AssertionFlags}
    assert not missing, f"schema classes without a generator: {sorted(c.__name__ for c in missing)}"


def test_candidate_atom_feature_round_trips_exactly():
    atom = CandidateAtom(
        atom_id="a", kind="node", feat=np.array([0.1, -2.5, 1e-7], dtype=np.float32)
    )
    restored = CandidateAtom.from_dict(json.loads(json.dumps(atom.to_dict())))
    assert restored.feat.dtype == np.float32
    assert np.array_equal(restored.feat, atom.feat)


def test_candidate_atom_identity_is_the_atom_id():
    """Equality and hashing must not touch the numpy field.

    A dataclass-generated __eq__ over an ndarray returns an array, which makes
    the type unusable in the sets and dict keys the pipeline is built on.
    """
    left = CandidateAtom(atom_id="x", kind="node", feat=np.zeros(4, dtype=np.float32))
    right = CandidateAtom(atom_id="x", kind="node", feat=np.ones(4, dtype=np.float32))
    assert left == right
    assert len({left, right}) == 1
    assert {left: "ok"}[right] == "ok"


def test_candidate_atom_features_are_immutable():
    atom = CandidateAtom(atom_id="x", kind="node", feat=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        atom.feat[0] = 1.0


def test_node_atoms_may_not_have_refs():
    """Nodes reference nothing.  This is what makes nodes-first construction
    always valid, and therefore what proves the unconstructible rate to be 0."""
    with pytest.raises(ValueError, match="nodes reference nothing"):
        CandidateAtom(atom_id="x", kind="node", refs=("y",))


def test_proof_set_serialises_sorted():
    """Set iteration order is randomised per interpreter run; the log must not be."""
    atoms = frozenset({"zz", "aa", "mm"})
    assert ProofSet(atoms=atoms).to_dict()["atoms"] == ["aa", "mm", "zz"]


def test_proof_set_equality_ignores_insertion_order():
    a = ProofSet(atoms=frozenset({"a", "b"}), bindings={"x": "1", "y": "2"})
    b = ProofSet(atoms=frozenset({"b", "a"}), bindings={"y": "2", "x": "1"})
    assert a == b and hash(a) == hash(b)


def test_proof_set_bindings_are_copied():
    bindings = {"slot": "atom"}
    ps = ProofSet(atoms=frozenset(), bindings=bindings)
    bindings["slot"] = "tampered"
    assert ps.bindings["slot"] == "atom"


# -- Interval: the half-open convention frozen in Phase 0 -------------------


def test_interval_is_half_open():
    iv = Interval(start=10.0, end=20.0)
    assert iv.contains(10.0)
    assert iv.contains(19.999)
    assert not iv.contains(20.0)


def test_interval_unbounded_sides():
    assert Interval().contains(-1e9)
    assert Interval(start=5.0).contains(1e9)
    assert not Interval(end=5.0).contains(5.0)


def test_zero_width_interval_is_empty_not_an_instant():
    iv = Interval(start=7.0, end=7.0)
    assert iv.is_empty
    assert not iv.contains(7.0)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((0.0, 10.0), (5.0, 15.0), True),
        ((0.0, 10.0), (10.0, 20.0), False),  # touching, half-open: no overlap
        ((0.0, 10.0), (11.0, 20.0), False),
        ((None, None), (3.0, 4.0), True),
        ((None, 5.0), (5.0, None), False),
    ],
)
def test_interval_overlaps(a, b, expected):
    assert Interval(*a).overlaps(Interval(*b)) is expected


def test_interval_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="after end"):
        Interval(start=10.0, end=1.0)


# -- Obligations -----------------------------------------------------------


def test_active_slots_excludes_the_aggregate_flag():
    """`aggregate` selects a route; it does not name something evidence supplies."""
    ob = Obligations(entity_anchor="e", needs_source=True, aggregate=True)
    assert ob.active_slots() == ("entity_anchor", "needs_source")


def test_active_slots_empty_when_nothing_is_required():
    assert Obligations().active_slots() == ()


# -- provenance and support ------------------------------------------------


def test_edge_requires_provenance():
    """A schema that permits an unsourced edge permits an unsourced proof."""
    with pytest.raises(ValueError, match="no provenance"):
        Edge(
            edge_id="e",
            etype="same_as",
            src="a",
            dst="b",
            t_created="2026-08-08T00:00:00+00:00",
            provenance=(),
        )


def test_assertion_requires_spans():
    with pytest.raises(ValueError, match="no spans"):
        Assertion(
            assertion_id="a",
            kind="claim",
            text_norm="x",
            spans=(),
            flags=AssertionFlags(asserted_by="user"),
            t_created="2026-08-08T00:00:00+00:00",
        )


def test_type_vocabularies_are_enforced():
    with pytest.raises(ValueError):
        Node(node_id="n", ntype="NotAType")
    with pytest.raises(ValueError):
        CandidateAtom(atom_id="a", kind="not-a-kind")


def test_edge_is_invalidated_never_deleted():
    edge = Edge(
        edge_id="e",
        etype="has_value",
        src="a",
        dst="b",
        t_created="2026-08-08T00:00:00+00:00",
        provenance=("s1",),
    )
    assert edge.is_live
    retired = Edge.from_dict({**edge.to_dict(), "t_invalid": "2026-09-01T00:00:00+00:00"})
    assert not retired.is_live
    assert retired.edge_id == edge.edge_id  # still present, still addressable


def test_non_finite_values_are_rejected_before_they_reach_the_log():
    with pytest.raises(ValueError):
        Interval(start=float("nan"))
    with pytest.raises(ValueError):
        CandidateAtom(atom_id="a", kind="node", feat=np.array([float("inf")], dtype=np.float32))
