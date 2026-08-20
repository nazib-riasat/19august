"""Schema round-trips and the invariants the data model is supposed to enforce.

Discharges Phase-0 exit criterion 1 (round-trip, property-based).
"""

from __future__ import annotations

import dataclasses
import json
import random
from types import MappingProxyType

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
    OutputRecord,
    ProofSet,
    SourceSpan,
    Violation,
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


# -- frozen means frozen ----------------------------------------------------


def test_proof_set_identity_is_the_atom_set_alone():
    """Bindings are derived from atoms and the pool, so a difference can only
    mean one side was built wrongly.  Including them in the key contradicted
    'two orders reaching the same atoms are the same state'."""
    derived = ProofSet(atoms=frozenset({"a"}), bindings={"answer": "a"})
    forgot_to_derive = ProofSet(atoms=frozenset({"a"}))
    assert derived == forgot_to_derive
    assert hash(derived) == hash(forgot_to_derive)


def test_proof_set_bindings_cannot_be_mutated():
    """A ``ProofSet`` is a dict key in Phase 2's DP and Phase 4's dedup; a
    mutable field on a hashable type makes an entry silently unreachable."""
    ps = ProofSet(atoms=frozenset({"a"}), bindings={"answer": "a"})
    cache = {ps: "value"}
    with pytest.raises(TypeError):
        ps.bindings["answer"] = "tampered"
    assert cache[ps] == "value"


def test_node_payload_cannot_be_mutated():
    node = Node(node_id="n", ntype="Entity", payload={"name": "Alice"})
    with pytest.raises(TypeError):
        node.payload["name"] = "Bob"


def test_output_record_ledger_snapshot_cannot_be_mutated():
    record = OutputRecord(outcome="abstain", ledger_snapshot={"terminal_checks": 3})
    with pytest.raises(TypeError):
        record.ledger_snapshot["terminal_checks"] = 99


@pytest.mark.parametrize(
    "make",
    [
        lambda: ProofSet(atoms=frozenset({"a"}), bindings={"answer": "a"}).bindings,
        lambda: Node(node_id="n", ntype="Entity", payload={"k": 1}).payload,
        lambda: OutputRecord(outcome="abstain", ledger_snapshot={"k": 1}).ledger_snapshot,
    ],
    ids=["ProofSet.bindings", "Node.payload", "OutputRecord.ledger_snapshot"],
)
def test_frozen_dataclasses_have_no_mutable_mappings(make):
    """``frozen=True`` blocks rebinding a field, not mutating what it points at.
    Every mapping inside a frozen dataclass has to be read-only or the promise
    is not kept."""
    mapping = make()
    assert isinstance(mapping, MappingProxyType)


# -- supersession implies invalidation --------------------------------------


def test_a_superseded_edge_must_also_be_invalidated():
    """``is_live`` is the one place invalidation semantics live; if supersession
    implied invalidation while ``is_live`` read only ``t_invalid``, a superseded
    fact would keep answering questions."""
    with pytest.raises(ValueError, match="no t_invalid"):
        Edge(
            edge_id="e",
            etype="has_value",
            src="a",
            dst="b",
            t_created="2026-08-08T00:00:00+00:00",
            provenance=("s1",),
            superseded_by="e2",
        )


def test_invalidation_without_supersession_is_still_legal():
    """The converse is not required: an edge may be retired with nothing
    replacing it."""
    edge = Edge(
        edge_id="e",
        etype="has_value",
        src="a",
        dst="b",
        t_created="2026-08-08T00:00:00+00:00",
        provenance=("s1",),
        t_invalid="2026-09-01T00:00:00+00:00",
    )
    assert not edge.is_live and edge.superseded_by is None


# -- the support gate fails closed ------------------------------------------


def test_an_assertion_is_quarantined_unless_told_otherwise():
    """Every other decision on this path fails closed; a default of 'eligible'
    was the one fail-open step, and it sat exactly where an omitted support-gate
    result would land."""
    assertion = Assertion(
        assertion_id="a",
        kind="claim",
        text_norm="x",
        spans=("s1",),
        flags=AssertionFlags(asserted_by="user"),  # entailed_by_span defaults False
        t_created="2026-08-08T00:00:00+00:00",
    )
    assert assertion.flags.entailed_by_span is False
    assert assertion.eligibility == "quarantined"


def test_a_record_written_before_the_field_existed_also_fails_closed():
    data = {
        "assertion_id": "a",
        "kind": "claim",
        "text_norm": "x",
        "spans": ["s1"],
        "flags": AssertionFlags(asserted_by="user").to_dict(),
        "t_created": "2026-08-08T00:00:00+00:00",
    }
    assert Assertion.from_dict(data).eligibility == "quarantined"


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


# -- a bare string is not a sequence of strings -----------------------------

TS = "2026-08-08T00:00:00+00:00"

#: (id, constructor call, from_dict call) for every field coerced with
#: ``tuple(...)`` or ``frozenset(...)``.  Both iterate a ``str`` character by
#: character instead of raising, so ``scope="conv-26"`` used to store seven
#: one-character scopes and `H`'s scope check then compared against ``'c'``,
#: ``'o'``, ``'n'`` ... and silently never matched.  These fields all carry
#: fail-closed guards; a per-character explosion defeats every one of them by
#: looking like an ordinary lookup miss.
BARE_STRING_SITES = [
    (
        "Obligations.scope",
        lambda: Obligations(scope="conv-26"),
        lambda: Obligations.from_dict({"scope": "conv-26"}),
    ),
    (
        "ProofSet.atoms",
        lambda: ProofSet(atoms="abc"),
        lambda: ProofSet.from_dict({"atoms": "abc"}),
    ),
    (
        "Violation.atoms",
        lambda: Violation(check="closure", message="m", atoms="a1"),
        lambda: Violation.from_dict({"check": "closure", "message": "m", "atoms": "a1"}),
    ),
    (
        "Assertion.spans",
        lambda: Assertion(
            assertion_id="a",
            kind="claim",
            text_norm="x",
            spans="s1",
            flags=AssertionFlags(asserted_by="user"),
            t_created=TS,
        ),
        lambda: Assertion.from_dict(
            {
                "assertion_id": "a",
                "kind": "claim",
                "text_norm": "x",
                "spans": "s1",
                "flags": AssertionFlags(asserted_by="user").to_dict(),
                "t_created": TS,
            }
        ),
    ),
    (
        "Edge.provenance",
        lambda: Edge(
            edge_id="e", etype="same_as", src="a", dst="b", t_created=TS, provenance="s1"
        ),
        lambda: Edge.from_dict(
            {
                "edge_id": "e",
                "etype": "same_as",
                "src": "a",
                "dst": "b",
                "t_created": TS,
                "provenance": "s1",
            }
        ),
    ),
    (
        "OutputRecord.citations",
        lambda: OutputRecord(outcome="answer", citations="c1"),
        lambda: OutputRecord.from_dict({"outcome": "answer", "citations": "c1"}),
    ),
]


@pytest.mark.parametrize(
    "field_name, construct",
    [(name, ctor) for name, ctor, _ in BARE_STRING_SITES],
    ids=[name for name, _, _ in BARE_STRING_SITES],
)
def test_bare_string_is_refused_by_the_constructor(field_name, construct):
    with pytest.raises(TypeError, match="explode into characters"):
        construct()


@pytest.mark.parametrize(
    "field_name, revive",
    [(name, from_dict) for name, _, from_dict in BARE_STRING_SITES],
    ids=[name for name, _, _ in BARE_STRING_SITES],
)
def test_bare_string_is_refused_by_from_dict(field_name, revive):
    """``from_dict`` coerces before ``__post_init__`` runs, so a guard on the
    constructor alone would never see the string.  Both doors need the check."""
    with pytest.raises(TypeError, match="explode into characters"):
        revive()


def test_the_error_names_the_field_it_came_from():
    """A ``TypeError`` from six call sites is only actionable if it says which."""
    with pytest.raises(TypeError, match=r"Obligations\.scope"):
        Obligations(scope="conv-26")
    with pytest.raises(TypeError, match=r"Edge\.provenance"):
        Edge(edge_id="e", etype="same_as", src="a", dst="b", t_created=TS, provenance="s")


def test_real_sequences_are_untouched():
    """The guard rejects ``str`` only; every other iterable still coerces."""
    assert Obligations(scope=["c1", "c2"]).scope == ("c1", "c2")
    assert Obligations(scope=("c1",)).scope == ("c1",)
    assert Violation(check="closure", message="m", atoms=iter(["a1"])).atoms == ("a1",)
    assert ProofSet(atoms=["a1", "a2"]).atoms == frozenset({"a1", "a2"})


def test_proof_set_keeps_the_frozenset_fast_path():
    """``frozenset(fs) is fs`` in CPython, and ``ProofSet`` is a dict key in
    Phase 2's DP and Phase 4's beam dedup.  Guarding via a ``tuple`` round-trip
    would have re-hashed every atom on every construction; the check returns the
    value unchanged so the identity path survives."""
    atoms = frozenset({"a1", "a2"})
    assert ProofSet(atoms=atoms).atoms is atoms
