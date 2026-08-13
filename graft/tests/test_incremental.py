"""Incremental validity.

Discharges Phase-1 exit criterion 3: for 10^4 random ``(set, insertion order)``
pairs, ``IncrementalChecker`` agrees with batch ``H`` on both the verdict and the
full violation list.

Agreement is structural — both paths assemble through ``checker.assemble`` over
the same two dictionaries — so this suite confirms the property rather than
being the only thing holding it up. Which is the point: a test that *carried* the
guarantee would be one refactor away from silently stopping.
"""

from __future__ import annotations

import random

import pytest

from graft.config import load_config
from graft.core.checker import H
from graft.core.incremental import IncrementalChecker
from graft.ledger import BudgetExceeded, Ledger
from graft.tests.fixtures import build_instance, closed_subsets, instances, random_subsets

CFG = load_config()


def _run(instance, ids, rng):
    order = list(ids)
    rng.shuffle(order)
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    for atom_id in order:
        inc.add(atom_id)
    return inc


def test_incremental_agrees_with_batch_on_10k_pairs():
    """Criterion 3.

    Run with ``ledger=None``: ``terminal_checks`` is capped at 32 and
    ``Ledger.count`` raises, so a metered run of this test would die at call 33
    (Phase-1 gap G9).
    """
    rng = random.Random(20260808)
    pairs = 0
    for instance in instances(seed=13, count=8, scoped=True):
        per_instance = 10_000 // 8
        subsets = list(random_subsets(rng, instance.pool, per_instance // 2, CFG.max_atoms))
        subsets += list(closed_subsets(rng, instance.pool, per_instance // 2, CFG.max_atoms))
        for subset in subsets:
            inc = _run(instance, subset, rng)
            batch = H(subset, instance.obligations, instance.graph, instance.pool, CFG)
            incremental = inc.result()
            assert incremental.ok == batch.ok
            assert incremental.violations == batch.violations
            pairs += 1
    assert pairs >= 10_000


def test_agreement_holds_under_undo():
    """Backtracking must land on exactly the state it started from."""
    rng = random.Random(5)
    instance = build_instance(rng, scoped=True)
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    trail = list(instance.pool.ids())[:10]

    for atom_id in trail:
        inc.add(atom_id)
    deep = inc.result()

    for _ in range(4):
        inc.undo()
    shallow_ids = inc.selected()
    assert inc.result() == H(shallow_ids, instance.obligations, instance.graph, instance.pool, CFG)

    for atom_id in reversed(trail[-4:]):
        inc.add(atom_id)
    assert inc.result() == deep


def test_state_is_a_set_not_a_sequence():
    rng = random.Random(11)
    instance = build_instance(rng)
    chosen = list(instance.gold.atoms)
    a = _run(instance, chosen, random.Random(1))
    b = _run(instance, chosen, random.Random(2))
    assert a.selected() == b.selected()
    assert a.state() == b.state()
    assert a.result() == b.result()


def test_bindings_are_derived_not_carried():
    """Phase-1 gap G3: nothing but the selected binding atoms can set them."""
    instance = build_instance(random.Random(3))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    for atom_id in sorted(instance.gold.atoms):
        inc.add(atom_id)
    assert inc.state().bindings == instance.pool.derive_bindings(instance.gold.atoms)
    assert inc.state().bindings != {}


# -- metering ---------------------------------------------------------------


def test_construction_spends_no_terminal_checks():
    """Second half of criterion 14, and the reason Phase-0 gap G1 exists: at one
    terminal check per step a single portfolio would cost 8 x 16 = 128 against a
    budget of 32."""
    instance = build_instance(random.Random(13))
    ledger = Ledger.from_config(CFG)
    with ledger.query_scope("q"):
        inc = IncrementalChecker(
            instance.pool, instance.obligations, instance.graph, CFG, ledger=ledger
        )
        for atom_id in sorted(instance.gold.atoms):
            inc.add(atom_id)
            inc.ok()
        counts = ledger.snapshot()["query"]
    assert counts["terminal_checks"] == 0
    assert counts["incremental_ops"] == len(instance.gold.atoms)


def test_a_sixteen_atom_construction_meters_sixteen_incremental_ops():
    """Criterion 14's letter: "a full **16-atom** incremental construction
    increments [terminal_checks] by 0 and incremental_ops by 16."  The gold-set
    variant above runs at 8 atoms; this one builds to the default profile's
    ``max_atoms = 16`` — nodes first, so closure never blocks — and asserts the
    letter, not just the shape.  (Added 13 Aug 2026 — the audit found the
    criterion tested at half its stated size.)"""
    rng = random.Random(42)
    # A bigger universe than the default fixture so 16 node atoms exist; the
    # fixture pool then exceeds the real-data pool_cap, which is irrelevant to
    # what this test meters, so the cap is lifted.
    instance = build_instance(rng, n_entities=8, n_claims=10, pool_cap=None)
    assert CFG.max_atoms == 16
    node_atoms = sorted(
        a.atom_id for a in instance.pool if a.kind == "node"
    )[: CFG.max_atoms]
    assert len(node_atoms) == 16
    ledger = Ledger.from_config(CFG)
    with ledger.query_scope("q16"):
        inc = IncrementalChecker(
            instance.pool, instance.obligations, instance.graph, CFG, ledger=ledger
        )
        for atom_id in node_atoms:
            inc.add(atom_id)
        counts = ledger.snapshot()["query"]
    assert counts["terminal_checks"] == 0
    assert counts["incremental_ops"] == 16


def test_a_full_construction_would_blow_the_budget_if_it_metered():
    """The regression that motivates the whole class: batch H at every step."""
    instance = build_instance(random.Random(13))
    ledger = Ledger.from_config(CFG)
    with pytest.raises(BudgetExceeded):
        with ledger.query_scope("q"):
            for _ in range(CFG.checker_budget + 1):
                H(instance.gold, instance.obligations, instance.graph, instance.pool, CFG, ledger=ledger)


def test_memo_counts_distinct_atoms_not_adds():
    """'The graph work is done once' made observable rather than asserted."""
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    first = sorted(instance.gold.atoms)[0]
    for _ in range(5):
        inc.add(first)
        inc.undo()
    assert inc.memo_size == 1


# -- misuse -----------------------------------------------------------------


def test_double_add_is_refused():
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    atom_id = instance.pool.ids()[0]
    inc.add(atom_id)
    with pytest.raises(ValueError, match="already selected"):
        inc.add(atom_id)


def test_undo_on_an_empty_state_is_refused():
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    with pytest.raises(ValueError, match="nothing to undo"):
        inc.undo()


def test_deficit_is_available_at_every_step():
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    seen = [inc.deficit()]
    for atom_id in sorted(instance.gold.atoms):
        inc.add(atom_id)
        seen.append(inc.deficit())
    assert all(d.shape == (6,) for d in seen)
    assert all(0.0 <= float(x) <= 1.0 for d in seen for x in d)
    assert seen[0].sum() > seen[-1].sum()  # obligations get discharged
