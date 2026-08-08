"""Deterministic identity.

Discharges Phase-0 exit criterion 4 (canon_set_hash invariant across all 720
permutations of a 6-atom set) and the build-step-4 check that ids do not collide
across 10^6 generated atoms.
"""

from __future__ import annotations

import itertools

import pytest

from graft import ids


def test_canon_set_hash_is_invariant_over_all_permutations():
    """Criterion 4: exhaustive over 6! = 720 orderings."""
    atoms = ["a1", "b2", "c3", "d4", "e5", "f6"]
    expected = ids.canon_set_hash(atoms)
    seen = {ids.canon_set_hash(list(p)) for p in itertools.permutations(atoms)}
    assert seen == {expected}
    assert len(list(itertools.permutations(atoms))) == 720


def test_canon_set_hash_distinguishes_different_sets():
    assert ids.canon_set_hash(["a", "b"]) != ids.canon_set_hash(["a", "c"])
    assert ids.canon_set_hash(["a", "b"]) != ids.canon_set_hash(["a"])


def test_canon_set_hash_defines_the_empty_set():
    """The empty set is the root state of every construction trajectory."""
    value = ids.canon_set_hash([])
    assert isinstance(value, str) and len(value) == ids.ID_LENGTH
    assert value != ids.canon_set_hash(["a"])


def test_canon_set_hash_accepts_any_iterable_form():
    assert ids.canon_set_hash(frozenset({"b", "a"})) == ids.canon_set_hash(["a", "b"])


def test_no_id_collisions_across_a_million_atoms():
    """Build step 4: 10^6 generated atoms, no collisions."""
    seen = set()
    for i in range(1_000_000):
        seen.add(ids.atom_id("edge", (f"n{i}", f"n{i + 1}"), "has_value"))
    assert len(seen) == 1_000_000


def test_atom_id_reference_order_is_significant():
    """An edge is directed; (src, dst) is not (dst, src)."""
    assert ids.atom_id("edge", ("a", "b")) != ids.atom_id("edge", ("b", "a"))


def test_atom_label_disambiguates_parallel_atoms():
    """Without the label, two differently-typed edges over one pair collide."""
    same_refs = ("a", "b")
    assert ids.atom_id("edge", same_refs, "has_value") != ids.atom_id(
        "edge", same_refs, "valid_during"
    )
    # And the two-argument form from the build plan still works.
    assert ids.atom_id("edge", same_refs) == ids.atom_id("edge", same_refs, "")


def test_ids_are_stable_across_calls():
    assert ids.span_id("t1", 3, 9) == ids.span_id("t1", 3, 9)
    assert ids.node_id("Entity", "alice") == ids.node_id("Entity", "alice")
    assert ids.edge_id("same_as", "a", "b") == ids.edge_id("same_as", "a", "b")


def test_assertion_id_ignores_span_order():
    left = ids.assertion_id("claim", "x", ["s2", "s1"])
    right = ids.assertion_id("claim", "x", ["s1", "s2"])
    assert left == right


def test_separator_cannot_be_forged():
    """Field boundaries must not be confusable by moving text between fields."""
    assert ids.node_id("Entity", "ab") != ids.node_id("Entitya", "b")


@pytest.mark.parametrize(
    "value",
    [
        ids.span_id("t", 0, 1),
        ids.node_id("Entity", "x"),
        ids.edge_id("same_as", "a", "b"),
        ids.atom_id("node", ()),
        ids.canon_set_hash(["a"]),
    ],
)
def test_all_ids_share_one_shape(value):
    assert len(value) == ids.ID_LENGTH
    assert all(c in "0123456789abcdef" for c in value)
