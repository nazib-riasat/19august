"""The closed sub-lattice and its two fingerprints.

Discharges Phase-2 exit criteria 6 (intra-layer permutation invariance is the
DP's business but the layering is this module's), 13 (collisions), 18 (children
reachable from the policy interface), 19-20 (fingerprints bind what they claim,
and are deterministic).
"""

from __future__ import annotations

import numpy as np
import pytest

from graft.core.checker import H
from graft.core.masks import legal_add_ids, stop_allowed
from graft.core.incremental import IncrementalChecker
from graft.schemas import CandidateAtom
from graft.synth.enumerate import (
    MAX_POOL_BITS,
    BandViolation,
    StateGraph,
    environment_fingerprint,
    reachable_states,
    state_fingerprints,
    valid_terminals,
)
from graft.synth.exact import target_distribution, target_fingerprint
from graft.synth.lattice import tiny_instance


# -- shape ------------------------------------------------------------------


def test_the_root_is_state_zero_and_cannot_stop(tiny_graph):
    """Phase-1 gap G1: the empty set is not a legal terminal, so the root is
    never a valid stop.  If it were, ``exp(beta*0) = 1`` would put a competitive
    reward on a proof that cites nothing."""
    assert int(tiny_graph.mask[0]) == 0
    assert not tiny_graph.stop_allowed[0]
    assert not tiny_graph.dead_end[0]


def test_states_are_layered_by_set_size(bench_graph):
    """The DP requires ascending ``|S|``; BFS over a layered DAG delivers it."""
    sizes = bench_graph.size
    assert np.all(np.diff(sizes) >= 0)
    for s in range(bench_graph.max_atoms + 1):
        lo, hi = bench_graph.state_slice(s)
        assert np.all(sizes[lo:hi] == s)


def test_every_edge_adds_exactly_one_atom(bench_graph):
    parent = bench_graph.mask[bench_graph.edge_parent].astype(object)
    child = bench_graph.mask[bench_graph.edge_child].astype(object)
    bit = np.left_shift(1, bench_graph.edge_action.astype(object))
    assert np.all(child == (parent | bit))
    assert np.all((parent & bit) == 0)


def test_edges_are_sorted_by_parent_so_layer_slices_are_contiguous(bench_graph):
    assert np.all(np.diff(bench_graph.edge_parent) >= 0)
    seen = 0
    for s in range(bench_graph.max_atoms + 1):
        lo, hi = bench_graph.edge_slice(s)
        assert lo == seen
        seen = hi
        assert np.all(bench_graph.size[bench_graph.edge_parent[lo:hi]] == s)
    assert seen == bench_graph.n_edges


def test_the_enumeration_walks_the_same_space_the_policy_would(tiny, tiny_graph):
    """Successors come from ``legal_adds`` and validity from the incremental
    checker, so the enumerated graph cannot drift from the environment a learner
    samples.  Re-derived here through the public Phase-1 surface."""
    chk = IncrementalChecker(tiny.pool, tiny.obligations, tiny.graph, tiny.cfg)
    for state in range(tiny_graph.n_states):
        while len(chk):
            chk.undo()
        for aid in tiny_graph.atoms_of(state):
            chk.add(aid)
        expected = set(legal_add_ids(chk))
        actions, _ = tiny_graph.children_of(state)
        assert {tiny_graph.atom_ids[a] for a in actions.tolist()} == expected
        assert bool(tiny_graph.stop_allowed[state]) == stop_allowed(chk)


def test_batch_H_agrees_with_the_stop_flag_on_every_state(tiny, tiny_graph):
    for state in range(tiny_graph.n_states):
        atoms = tiny_graph.atoms_of(state)
        result = H(atoms, tiny.obligations, tiny.graph, tiny.pool, tiny.cfg, ledger=None)
        assert bool(result) == bool(tiny_graph.stop_allowed[state])


def test_valid_terminals_are_the_stop_allowed_states(tiny, tiny_graph):
    terminals = valid_terminals(tiny, tiny.cfg, graph=tiny_graph)
    assert len(terminals) == tiny_graph.n_terminals
    for proof, state in zip(terminals, tiny_graph.terminal_ix.tolist()):
        assert set(proof.atoms) == set(tiny_graph.atoms_of(state))
        # Bindings are derived, never chosen (Phase-1 gap G3).
        assert dict(proof.bindings) == tiny.pool.derive_bindings(proof.atoms)


# -- criterion 18: children are reachable from the policy interface ---------


def test_every_legal_add_maps_to_the_index_of_the_child_it_produces(bench_graph):
    """Exit criterion 18.  A Look-Ahead featuriser — the published remedy for the
    GNN-expressivity limit v1.2 §6.4 names as a risk — needs children-state
    embeddings, so the frozen protocol must not foreclose reaching them.
    Asserting that a capability is not foreclosed requires a test, not a sentence
    saying no test would catch its loss."""
    rng = np.random.default_rng(11)
    for state in rng.choice(bench_graph.n_states, size=50, replace=False).tolist():
        actions, children = bench_graph.children_of(state)
        assert actions.shape == children.shape
        for a, c in zip(actions.tolist(), children.tolist()):
            assert int(bench_graph.mask[c]) == int(bench_graph.mask[state]) | (1 << a)
            assert bench_graph.size[c] == bench_graph.size[state] + 1


# -- criterion 13: collisions ----------------------------------------------


def test_distinct_actions_never_produce_the_same_child(bench_graph):
    """G3, as a theorem discharged rather than a measurement taken: the children
    are the **sets** ``S ∪ {a}`` and ``S ∪ {b}`` with ``a, b ∉ S``, so they
    differ.  Over exact child-set equality, never over hashes."""
    for state in range(bench_graph.n_states):
        _, children = bench_graph.children_of(state)
        masks = bench_graph.mask[children]
        assert np.unique(masks).shape[0] == masks.shape[0]


def test_no_two_states_share_a_canon_set_hash(bench_graph):
    """The fingerprint is a 64-bit truncation of SHA-256, so it is injective only
    with overwhelming probability.  State identity does not depend on it — that
    is the exact bitmask — but cross-machine comparison does."""
    prints = state_fingerprints(bench_graph)
    assert len(set(prints)) == len(prints)


# -- the uint64 assumption --------------------------------------------------


def test_a_pool_wider_than_uint64_is_refused_loudly():
    """Decision 4.  The bitmask is valid while ``pool_cap <= 64``; beyond that it
    needs a wider representation, and silently truncating would give an *exact*
    evaluator that merges two distinct states."""
    with pytest.raises(ValueError, match="uint64"):
        StateGraph(
            atom_ids=tuple(f"a{i}" for i in range(MAX_POOL_BITS + 1)),
            max_atoms=8,
            mask=np.zeros(1, dtype=np.uint64),
            size=np.zeros(1, dtype=np.int16),
            edge_parent=np.zeros(0, dtype=np.int64),
            edge_action=np.zeros(0, dtype=np.int64),
            edge_child=np.zeros(0, dtype=np.int64),
            stop_flags=np.zeros(1, dtype=bool),
            dead_flags=np.zeros(1, dtype=bool),
        )


# -- G1: the enumeration aborts early on its upper bounds -------------------


def test_enumeration_aborts_the_moment_the_state_cap_is_passed(tiny):
    """G1.  An oversized instance must be rejected in the time it takes to exceed
    the bound, not after a full sweep."""
    with pytest.raises(BandViolation) as exc:
        reachable_states(tiny, tiny.cfg, max_states=5)
    assert exc.value.band == "states"


def test_enumeration_aborts_on_the_edge_cap(tiny):
    with pytest.raises(BandViolation) as exc:
        reachable_states(tiny, tiny.cfg, max_edges=3)
    assert exc.value.band == "edges"


# -- criteria 19-20: the fingerprints ---------------------------------------


def test_the_environment_fingerprint_is_deterministic(tiny, tiny_graph):
    again = reachable_states(tiny_instance(), tiny.cfg)
    assert environment_fingerprint(tiny, tiny_graph) == environment_fingerprint(
        tiny_instance(), again
    )


def test_the_suites_are_deterministic(main_suite, probe, tuning):
    """Criterion 20: same seed -> identical ``environment_fingerprint``.

    "All three suites are deterministic" — until 13 Aug 2026 only the main
    suite was rebuilt in-repo, with probe/tuning determinism covered
    operationally by ``verify_handoff.py`` alone.  All three now rebuild.
    """
    from graft.synth.lattice import benchmark_suite, probe_suite, tuning_suite

    for suite, rebuild in (
        (main_suite, benchmark_suite),
        (probe, probe_suite),
        (tuning, tuning_suite),
    ):
        rebuilt = rebuild()
        assert len(rebuilt) == len(suite)
        for a, b in zip(suite, rebuilt):
            ga = reachable_states(a, a.cfg)
            gb = reachable_states(b, b.cfg)
            assert environment_fingerprint(a, ga) == environment_fingerprint(b, gb)


def _mutated(instance, **atom_changes):
    """A copy of ``instance`` with one atom rebuilt."""
    import copy

    from graft.schemas import AtomPool
    from graft.synth.lattice import LatticeInstance

    target_id = sorted(instance.pool.ids())[0]
    atoms = []
    for atom in instance.pool:
        if atom.atom_id == target_id:
            fields = {
                "atom_id": atom.atom_id,
                "kind": atom.kind,
                "refs": atom.refs,
                "feat": atom.feat,
                "label": atom.label,
                "target": atom.target,
            }
            fields.update(atom_changes)
            atoms.append(CandidateAtom(**fields))
        else:
            atoms.append(atom)
    return LatticeInstance(
        pool=AtomPool(atoms, cap=instance.pool.cap),
        obligations=instance.obligations,
        graph=instance.graph,
        gold=instance.gold,
        template_a=instance.template_a,
        template_b=instance.template_b,
        spec=instance.spec,
        meta=copy.deepcopy(instance.meta),
    )


def test_a_changed_feature_value_moves_the_environment_digest(tiny, tiny_graph):
    """Criterion 19.  ``feat`` is in the digest because it changes
    ``redundancy`` and therefore every reward on the lattice."""
    base = environment_fingerprint(tiny, tiny_graph)
    feat = np.array(tiny.pool[sorted(tiny.pool.ids())[0]].feat, copy=True)
    feat[0] += 0.5
    other = _mutated(tiny, feat=feat)
    assert environment_fingerprint(other, reachable_states(other, tiny.cfg)) != base


def test_a_changed_source_tier_moves_the_environment_digest(tiny, tiny_graph):
    from graft.graphstore import DictGraphSnapshot
    from graft.schemas import PAYLOAD_TIER, Node
    from graft.synth.lattice import LatticeInstance

    base = environment_fingerprint(tiny, tiny_graph)
    snapshot = DictGraphSnapshot(
        snapshot_id=tiny.graph.snapshot_id,
        nodes=[
            Node(node_id=n, ntype="Source", payload={PAYLOAD_TIER: "reported"})
            if tiny.graph.node(n).ntype == "Source"
            else tiny.graph.node(n)
            for n in [node.node_id for node in _nodes_of(tiny.graph)]
        ],
        edges=_edges_of(tiny.graph),
        assertions=_assertions_of(tiny.graph),
        turns=_turns_of(tiny.graph),
        spans=_spans_of(tiny.graph),
    )
    other = LatticeInstance(
        pool=tiny.pool,
        obligations=tiny.obligations,
        graph=snapshot,
        gold=tiny.gold,
        template_a=tiny.template_a,
        template_b=tiny.template_b,
        spec=tiny.spec,
        meta=dict(tiny.meta),
    )
    assert environment_fingerprint(other, reachable_states(other, tiny.cfg)) != base


def test_a_changed_interval_bound_moves_the_environment_digest(tiny, tiny_graph):
    from graft.graphstore import DictGraphSnapshot
    from graft.schemas import Node
    from graft.synth.lattice import LatticeInstance

    base = environment_fingerprint(tiny, tiny_graph)
    nodes = []
    for node in _nodes_of(tiny.graph):
        if node.ntype == "TimeInterval" and node.payload.get("start") == 0.0:
            nodes.append(
                Node(node_id=node.node_id, ntype=node.ntype, payload={"start": 0.0, "end": 80.0})
            )
        else:
            nodes.append(node)
    snapshot = DictGraphSnapshot(
        snapshot_id=tiny.graph.snapshot_id,
        nodes=nodes,
        edges=_edges_of(tiny.graph),
        assertions=_assertions_of(tiny.graph),
        turns=_turns_of(tiny.graph),
        spans=_spans_of(tiny.graph),
    )
    other = LatticeInstance(
        pool=tiny.pool,
        obligations=tiny.obligations,
        graph=snapshot,
        gold=tiny.gold,
        template_a=tiny.template_a,
        template_b=tiny.template_b,
        spec=tiny.spec,
        meta=dict(tiny.meta),
    )
    assert environment_fingerprint(other, reachable_states(other, tiny.cfg)) != base


def test_a_changed_graph_edge_moves_the_environment_digest(tiny, tiny_graph):
    """The part an earlier draft missed.  Binding only the generator's *inputs*
    means two machines whose masks or checker differ would enumerate different
    graphs and still agree on the fingerprint — which is exactly the disagreement
    it exists to detect."""
    base = environment_fingerprint(tiny, tiny_graph)
    doctored = StateGraph(
        atom_ids=tiny_graph.atom_ids,
        max_atoms=tiny_graph.max_atoms,
        mask=tiny_graph.mask,
        size=tiny_graph.size,
        edge_parent=tiny_graph.edge_parent,
        edge_action=tiny_graph.edge_action,
        edge_child=np.concatenate((tiny_graph.edge_child[:-1], [0])),
        stop_flags=tiny_graph.stop_allowed,
        dead_flags=tiny_graph.dead_end,
    )
    assert environment_fingerprint(tiny, doctored) != base


def test_the_target_fingerprint_binds_the_target_and_not_the_environment(tiny, tiny_graph):
    """Criterion 19.  An earlier draft mutated only environment components, so it
    never demonstrated that the target fingerprint binds the target at all.

    Perturbing one terminal's ``U`` by 1e-9 moves ``p*`` well above the 1e-12
    quantum; the environment digest must not move, because ``U`` is derived."""
    target = target_distribution(tiny, tiny.cfg, graph=tiny_graph)
    env_before = environment_fingerprint(tiny, tiny_graph)
    before = target_fingerprint(target)

    perturbed = target_distribution(tiny, tiny.cfg, graph=tiny_graph)
    perturbed.u[0] += 1e-9
    perturbed = type(perturbed)(
        tiny, tiny_graph, tiny.cfg, perturbed.u, perturbed.sizes,
        perturbed.mode_labels, perturbed.zero_sufficiency,
    )
    assert target_fingerprint(perturbed) != before
    assert environment_fingerprint(tiny, tiny_graph) == env_before


def _same_instance_at(instance, **cfg_overrides):
    """The same environment under a different ``Config``.

    The point is that ``spec.cfg`` itself moves — ``Target.at_beta`` builds a new
    ``Config`` on the *target* and never touches the spec, so a test written
    through it cannot see a leak in the environment payload.
    """
    from graft.synth.lattice import LatticeInstance

    return LatticeInstance(
        pool=instance.pool,
        obligations=instance.obligations,
        graph=instance.graph,
        gold=instance.gold,
        template_a=instance.template_a,
        template_b=instance.template_b,
        spec=instance.spec.replace(cfg=instance.cfg.with_overrides(**cfg_overrides)),
        meta=dict(instance.meta),
    )


@pytest.mark.parametrize(
    "override", [{"beta": 8.0}, {"r_fail": 1e-8}, {"checker_budget": 64}]
)
def test_the_environment_fingerprint_ignores_the_reward_and_budget_layers(
    main_suite, override
):
    """Decision 21, and the defect a post-build review found.

    ``config_hash`` moves when the Phase-3 sweep freezes β. A fingerprint
    carrying β would therefore change *after* the suites were frozen at Gate 0,
    so "frozen" would mean nothing — and, worse, a genuine structural divergence
    between two machines could be waved away as "β differs".

    The earlier version of this test compared ``environment_fingerprint(tiny,
    tiny_graph)`` with itself and could not fail. This one rebuilds the spec's
    ``Config``, which is where the leak was.
    """
    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    twin = _same_instance_at(instance, **override)
    twin_graph = reachable_states(twin, twin.cfg)

    assert twin_graph.fingerprint() == graph.fingerprint(), (
        "the enumerated graph must be identical, or this tests two things at once"
    )
    assert environment_fingerprint(twin, twin_graph) == environment_fingerprint(
        instance, graph
    )


def test_the_target_fingerprint_still_moves_with_beta(main_suite):
    """The other half: β must move the *target* digest, or nothing binds it."""
    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    base = target_distribution(instance, instance.cfg, graph=graph)
    assert target_fingerprint(base.at_beta(8.0)) != target_fingerprint(base)

    twin = _same_instance_at(instance, beta=8.0)
    twin_graph = reachable_states(twin, twin.cfg)
    assert target_fingerprint(
        target_distribution(twin, twin.cfg, graph=twin_graph)
    ) != target_fingerprint(base)


def test_the_environment_fingerprint_still_binds_the_caps(main_suite):
    """Narrowing the payload must not empty it.

    ``pool_cap`` is chosen because raising it changes neither the pool (20-26
    atoms) nor the enumeration, so the digest can only move because the field is
    genuinely bound.
    """
    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    twin = _same_instance_at(instance, pool_cap=48)
    twin_graph = reachable_states(twin, twin.cfg)
    assert twin_graph.fingerprint() == graph.fingerprint()
    assert environment_fingerprint(twin, twin_graph) != environment_fingerprint(
        instance, graph
    )


# -- small helpers ----------------------------------------------------------


def _nodes_of(snapshot):
    return [snapshot.node(nid) for nid in sorted(snapshot._nodes)]


def _edges_of(snapshot):
    return [snapshot.edge(eid) for eid in sorted(snapshot._edges)]


def _assertions_of(snapshot):
    return [snapshot.assertion(aid) for aid in sorted(snapshot._assertions)]


def _turns_of(snapshot):
    return [snapshot.turn(tid) for tid in sorted(snapshot._turns)]


def _spans_of(snapshot):
    return [snapshot.span(sid) for sid in sorted(snapshot._spans)]
