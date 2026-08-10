"""Action masks and terminal semantics.

Discharges Phase-1 exit criteria 2 (no formally invalid set is a reachable
terminal other than ``FAIL``), 4 (masks never permit an atom whose refs are
absent), 5 (every dead end transitions to ``FAIL``) and 6 (the empty set is not a
legal terminal).
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from graft.config import load_config
from graft.core.checker import H
from graft.core.incremental import IncrementalChecker
from graft.core.masks import (
    Terminal,
    is_dead_end,
    legal_add_ids,
    legal_adds,
    stop_allowed,
    terminal_of,
)
from graft.schemas import AtomPool, CandidateAtom, Node, Obligations
from graft.graphstore import DictGraphSnapshot
from graft.tests.fixtures import build_instance, instances

CFG = load_config()


def rollout(instance, rng, p_stop=0.35, cfg=CFG):
    """One random trajectory to a terminal.  Returns ``(Terminal, checker)``."""
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, cfg)
    while True:
        can_stop = stop_allowed(inc)
        allowed = legal_add_ids(inc)
        if can_stop and (not allowed or rng.random() < p_stop):
            return Terminal.VALID, inc
        if not allowed:
            # No legal ADD and STOP masked: a dead end, by any of its causes.
            return Terminal.FAIL, inc
        inc.add(rng.choice(allowed))


# -- criterion 6 ------------------------------------------------------------


def test_the_root_may_not_stop():
    """The empty set is not a legal terminal (Phase-1 gap G1)."""
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    assert stop_allowed(inc) is False
    assert terminal_of(inc) is None  # adds remain, so not a dead end either


def test_no_terminal_is_ever_empty():
    rng = random.Random(4)
    for instance in instances(seed=21, count=6):
        for _ in range(40):
            outcome, inc = rollout(instance, rng)
            if outcome is Terminal.VALID:
                assert len(inc) >= 1


# -- criterion 4 ------------------------------------------------------------


def test_masks_never_offer_an_atom_with_absent_refs():
    rng = random.Random(9)
    for instance in instances(seed=31, count=6):
        inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
        for _ in range(12):
            allowed = legal_add_ids(inc)
            selected = set(inc.selected())
            for atom_id in allowed:
                assert all(ref in selected for ref in instance.pool[atom_id].refs)
            if not allowed:
                break
            inc.add(rng.choice(allowed))


def test_only_node_atoms_are_legal_at_the_root():
    """Nodes reference nothing, which is what makes nodes-first always work."""
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    for atom_id in legal_add_ids(inc):
        assert instance.pool[atom_id].kind == "node"


def test_mask_is_indexed_by_sorted_pool_ids():
    """This vector becomes tensor positions in Phase 3; set order is randomised."""
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    mask = legal_adds(inc)
    assert mask.shape == (len(instance.pool),)
    assert mask.dtype == np.bool_
    ids = instance.pool.ids()
    assert list(legal_add_ids(inc)) == [i for i, m in zip(ids, mask) if m]


def test_selected_atoms_are_never_offered_again():
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    first = legal_add_ids(inc)[0]
    inc.add(first)
    assert first not in legal_add_ids(inc)


def test_inadmissible_atoms_are_never_offered():
    """A per-atom violation is permanent, so such an atom poisons every
    descendant state and pruning it is sound."""
    instance = build_instance(random.Random(13), scoped=True)
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    inadmissible = [a.atom_id for a in instance.pool if not inc.atom_is_admissible(a.atom_id)]
    assert inadmissible, "fixture should contain quarantined and out-of-scope atoms"
    for _ in range(20):
        allowed = set(legal_add_ids(inc))
        assert not (allowed & set(inadmissible))
        if not allowed:
            break
        inc.add(sorted(allowed)[0])


def test_the_size_limit_closes_the_mask():
    instance = build_instance(random.Random(13))
    small = CFG.with_overrides(max_atoms=3)
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, small)
    for _ in range(3):
        inc.add(legal_add_ids(inc)[0])
    assert len(inc) == 3
    assert not legal_adds(inc).any()


# -- criteria 2 and 5 -------------------------------------------------------


@pytest.mark.parametrize("p_stop", [0.35, 0.0])
def test_every_reachable_terminal_is_valid_or_fail(p_stop):
    """Criteria 2 and 5 together, over random trajectories.

    ``p_stop = 0`` never stops voluntarily, so trajectories run to the size
    limit and the ``FAIL`` branch is exercised in bulk rather than by luck.
    """
    rng = random.Random(17)
    outcomes = Counter()
    dead_end_sizes = Counter()
    for instance in instances(seed=41, count=8, scoped=True):
        for _ in range(60):
            outcome, inc = rollout(instance, rng, p_stop=p_stop)
            outcomes[outcome] += 1
            if outcome is Terminal.VALID:
                # The reachable-terminal guarantee: what STOP permits, H accepts.
                assert H(
                    inc.selected(), instance.obligations, instance.graph, instance.pool, CFG
                ).ok
            else:
                assert is_dead_end(inc)
                dead_end_sizes[len(inc)] += 1

    assert set(outcomes) <= {Terminal.VALID, Terminal.FAIL}
    assert outcomes[Terminal.VALID] > 0
    print(f"\np_stop={p_stop}: {dict(outcomes)}  dead-end |X|: {dict(sorted(dead_end_sizes.items()))}")

    if p_stop == 0.0:
        assert dead_end_sizes, "the FAIL branch was never exercised"
        # The healthy signature.  Every dead end sits at the size limit, meaning
        # the *budget* ran out; a mass at small |X| would mean the ADD masks are
        # too tight and would be a finding, not noise.  Phase 2 reports this
        # distribution for the lattice.
        assert set(dead_end_sizes) == {CFG.max_atoms}


def test_a_lone_node_atom_is_formally_valid():
    """It looks like a bug and is not.

    ``H`` is *formal* validity, never sufficiency (v1.2 §4.1). A single Entity
    node violates no schema, id, interval, support or scope constraint, so it may
    stop. What makes it a useless proof is ``U`` — near-zero sufficiency and
    coverage — which is exactly the division of labour the two are for. A checker
    that rejected it would be smuggling semantic judgment into a formal predicate.
    """
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    inc.add(legal_add_ids(inc)[0])
    assert stop_allowed(inc)
    assert H(inc.selected(), instance.obligations, instance.graph, instance.pool, CFG).ok


def _dangling_edge_atom(instance):
    """An edge atom whose endpoints are not selected — closure fails."""
    return next(a for a in instance.pool if a.kind == "edge" and a.refs)


def test_a_dead_end_is_reported_as_fail_not_as_a_valid_stop():
    """Budget exhaustion with ``H = 0``: the classic dead end."""
    instance = build_instance(random.Random(13))
    small = CFG.with_overrides(max_atoms=2)
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, small)
    dangling = _dangling_edge_atom(instance)
    inc.add(dangling.atom_id)          # closure violated: endpoints absent
    inc.add(dangling.refs[0])          # still one endpoint short, and now |X| = 2
    assert not stop_allowed(inc)
    assert not legal_adds(inc).any()   # size limit reached
    assert is_dead_end(inc)
    assert terminal_of(inc) is Terminal.FAIL


def test_an_empty_pool_is_a_dead_end_at_size_zero():
    """One of the several causes the plan's earlier 'only budget exhaustion'
    wording missed — a query that retrieved nothing."""
    instance_graph = DictGraphSnapshot()
    empty = AtomPool([], cap=8)
    inc = IncrementalChecker(empty, Obligations(), instance_graph, CFG)
    assert len(inc) == 0
    assert not stop_allowed(inc)
    assert is_dead_end(inc)
    assert terminal_of(inc) is Terminal.FAIL


def test_a_pool_of_only_inadmissible_atoms_is_a_dead_end():
    """Another cause: every remaining atom fails a per-atom check."""
    graph = DictGraphSnapshot(nodes=[Node(node_id="c1", ntype="Claim", payload={})])
    pool = AtomPool([CandidateAtom(atom_id="a", kind="node", target="c1")])
    inc = IncrementalChecker(pool, Obligations(), graph, CFG)
    assert not legal_adds(inc).any()
    assert is_dead_end(inc)


def test_a_valid_state_is_not_a_dead_end_even_with_adds_remaining():
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    for atom_id in sorted(instance.gold.atoms):
        inc.add(atom_id)
    assert stop_allowed(inc)
    assert not is_dead_end(inc)
    assert terminal_of(inc) is Terminal.VALID


def test_add_stays_available_when_stop_is_masked():
    """Only STOP is masked when H = 0; traversing an invalid partial set is fine.

    This is why v1.2 §4.3 withdrew the H-monotonicity proof obligation, and the
    property is worth pinning because re-deriving that obligation is a recurring
    temptation.
    """
    instance = build_instance(random.Random(13))
    inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
    inc.add(_dangling_edge_atom(instance).atom_id)
    assert not stop_allowed(inc)   # closure fails: the endpoints are not selected
    assert legal_adds(inc).any()   # but ADD is still open, including the endpoints


def test_gold_is_reachable_by_a_nodes_first_order():
    """Constructibility: any closed terminal can be built nodes-first, which is
    what proves the unconstructible-valid-terminal rate to be 0."""
    for instance in instances(seed=51, count=6):
        inc = IncrementalChecker(instance.pool, instance.obligations, instance.graph, CFG)
        order = sorted(
            instance.gold.atoms,
            key=lambda a: {"node": 0, "edge": 1, "binding": 2}[instance.pool[a].kind],
        )
        for atom_id in order:
            assert atom_id in legal_add_ids(inc), f"{atom_id} was not offered"
            inc.add(atom_id)
        assert stop_allowed(inc)
