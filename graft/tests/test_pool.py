"""``AtomPool`` — resolution, invariants, and derived bindings.

The invariants are the point. Both rejected conditions produce atoms that can
never be legally added, and both are invisible downstream: the mask simply never
offers them, so a pool where part of the budget is unreachable looks like a pool
with worse recall.
"""

from __future__ import annotations

import numpy as np
import pytest

from graft import ids
from graft.schemas import AtomPool, CandidateAtom


def node_atom(name: str, target: str | None = None) -> CandidateAtom:
    return CandidateAtom(
        atom_id=name,
        kind="node",
        target=target if target is not None else f"N_{name}",
        feat=np.ones(3, dtype=np.float32),
    )


def edge_atom(name: str, refs: tuple[str, ...], label: str = "same_as") -> CandidateAtom:
    return CandidateAtom(
        atom_id=name,
        kind="edge",
        refs=refs,
        label=label,
        target=f"E_{name}",
        feat=np.ones(3, dtype=np.float32),
    )


def binding_atom(name: str, refs: tuple[str, ...], slot: str = "answer") -> CandidateAtom:
    return CandidateAtom(atom_id=name, kind="binding", refs=refs, label=slot)


# -- resolution -------------------------------------------------------------


def test_pool_resolves_atoms_by_id():
    pool = AtomPool([node_atom("a"), node_atom("b")])
    assert pool["a"].kind == "node"
    assert pool.get("b").target == "N_b"
    assert pool.get("missing") is None
    assert "a" in pool and "missing" not in pool
    assert len(pool) == 2


def test_ids_are_sorted_because_they_index_mask_vectors():
    pool = AtomPool([node_atom("z"), node_atom("a"), node_atom("m")])
    assert pool.ids() == ("a", "m", "z")
    assert [atom.atom_id for atom in pool] == ["a", "m", "z"]


def test_reverse_index_answers_which_atoms_a_ref_unlocks():
    pool = AtomPool(
        [node_atom("a"), node_atom("b"), edge_atom("e1", ("a", "b")), edge_atom("e2", ("a",))]
    )
    assert pool.referencing("a") == ("e1", "e2")
    assert pool.referencing("b") == ("e1",)
    assert pool.referencing("e1") == ()


# -- invariants -------------------------------------------------------------


def test_out_of_pool_reference_is_rejected():
    """Such an atom can never satisfy closure, so it is unreachable dead weight."""
    with pytest.raises(ValueError, match="permanently unaddable"):
        AtomPool([node_atom("a"), edge_atom("e1", ("a", "ghost"))])


def test_reference_cycle_is_rejected():
    """Each atom in a cycle waits on the others, so none can ever be added."""
    left = CandidateAtom(atom_id="x", kind="edge", refs=("y",), target="E_x")
    right = CandidateAtom(atom_id="y", kind="edge", refs=("x",), target="E_y")
    with pytest.raises(ValueError, match="reference cycle"):
        AtomPool([left, right])


def test_self_reference_is_a_cycle():
    atom = CandidateAtom(atom_id="x", kind="edge", refs=("x",), target="E_x")
    with pytest.raises(ValueError, match="reference cycle"):
        AtomPool([atom])


def test_long_cycle_is_found():
    chain = [
        CandidateAtom(atom_id=f"n{i}", kind="edge", refs=(f"n{(i + 1) % 6}",), target=f"E{i}")
        for i in range(6)
    ]
    with pytest.raises(ValueError, match="reference cycle"):
        AtomPool(chain)


def test_a_deep_acyclic_chain_is_accepted():
    """Cycle detection is iterative, so depth must not blow the stack."""
    atoms = [node_atom("n0")]
    for i in range(1, 2000):
        atoms.append(edge_atom(f"n{i}", (f"n{i - 1}",)))
    pool = AtomPool(atoms)
    assert len(pool) == 2000


def test_duplicate_atom_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate atom id"):
        AtomPool([node_atom("a"), node_atom("a")])


def test_pool_cap_is_enforced():
    with pytest.raises(ValueError, match="over pool_cap"):
        AtomPool([node_atom(f"n{i}") for i in range(5)], cap=4)
    assert len(AtomPool([node_atom(f"n{i}") for i in range(4)], cap=4)) == 4


def test_an_empty_pool_is_legal():
    """A query that retrieved nothing is a dead end, not a malformed pool — the
    mask and FAIL handle it, and refusing it here would hide the case."""
    pool = AtomPool([], cap=8)
    assert len(pool) == 0 and pool.ids() == ()


# -- bindings ---------------------------------------------------------------


def test_derive_bindings_reads_slot_from_the_label():
    pool = AtomPool([node_atom("a"), binding_atom("b1", ("a",), "answer")])
    assert pool.derive_bindings(["a", "b1"]) == {"answer": "b1"}


def test_unselected_bindings_do_not_appear():
    pool = AtomPool([node_atom("a"), binding_atom("b1", ("a",))])
    assert pool.derive_bindings(["a"]) == {}


def test_binding_slots_reports_every_claimant_so_check9_can_see_them():
    pool = AtomPool(
        [node_atom("a"), binding_atom("b1", ("a",), "answer"), binding_atom("b2", ("a",), "answer")]
    )
    assert pool.binding_slots(["a", "b1", "b2"]) == {"answer": ("b1", "b2")}
    # derive_bindings still returns something deterministic on a set H will reject.
    assert pool.derive_bindings(["a", "b1", "b2"]) == {"answer": "b1"}


def test_derive_bindings_is_order_invariant():
    pool = AtomPool(
        [node_atom("a"), binding_atom("b1", ("a",), "answer"), binding_atom("b2", ("a",), "subject")]
    )
    assert pool.derive_bindings(["b2", "b1", "a"]) == pool.derive_bindings(["a", "b1", "b2"])


def test_unknown_ids_are_ignored_by_binding_derivation():
    pool = AtomPool([node_atom("a")])
    assert pool.derive_bindings(["a", "not-in-pool"]) == {}


# -- atom-level invariants the pool relies on -------------------------------


def test_binding_atoms_may_not_denote_a_graph_object():
    with pytest.raises(ValueError, match="does not denote a graph object"):
        CandidateAtom(atom_id="b", kind="binding", label="answer", target="E_1")


def test_binding_atoms_must_name_a_slot():
    with pytest.raises(ValueError, match="has no label"):
        CandidateAtom(atom_id="b", kind="binding")


def test_content_key_separates_atoms_denoting_different_objects():
    """Sub-check 2 compares these; target must be part of the key."""
    left = CandidateAtom(atom_id="x", kind="edge", refs=("a",), target="E_1")
    right = CandidateAtom(atom_id="y", kind="edge", refs=("a",), target="E_2")
    assert left.content_key() != right.content_key()


def test_atom_id_is_keyed_on_target():
    assert ids.atom_id("edge", ("a",), "E_1") != ids.atom_id("edge", ("a",), "E_2")
    assert ids.atom_id("edge", ("a",), "E_1", "same_as") != ids.atom_id(
        "edge", ("a",), "E_1", "has_value"
    )
    # The two-argument form from the build plan still works.
    assert ids.atom_id("node", ()) == ids.atom_id("node", (), "", "")
