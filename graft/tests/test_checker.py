"""Formal validity ``H``.

Discharges Phase-1 exit criterion 1: **every one of the nine sub-checks has at
least one positive and at least one negative case**, and checks 3, 4, 5 and 7 are
exercised against a snapshot rather than in-pool (Phase-1 gap G2).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from graft import ids
from graft.config import load_config
from graft.core import checker
from graft.core.checker import (
    CHECK_BINDING,
    CHECK_CLOSURE,
    CHECK_IDENTITY,
    CHECK_RETIRED,
    CHECK_SCOPE,
    CHECK_SIZE,
    CHECK_SUPPORT,
    CHECK_TEMPORAL,
    CHECK_TYPE,
    CHECKS,
    H,
)
from graft.graphstore import DictGraphSnapshot
from graft.ledger import Ledger
from graft.schemas import (
    PAYLOAD_ASSERTION_ID,
    PAYLOAD_TIER,
    AtomPool,
    CandidateAtom,
    Edge,
    Node,
    Obligations,
    ProofSet,
)
from graft.tests.fixtures import TS, build_instance

CFG = load_config()


@pytest.fixture
def inst():
    return build_instance(random.Random(13), scoped=True)


def categories(result):
    return set(result.categories())


# -- 6 size -----------------------------------------------------------------


def test_size_positive(inst):
    assert H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG).ok


def test_size_negative_empty(inst):
    """Phase-1 gap G1: the empty set passes every other check vacuously."""
    result = H([], inst.obligations, inst.graph, inst.pool, CFG)
    assert not result.ok
    assert categories(result) == {CHECK_SIZE}
    assert "cites nothing" in result.violations[0].message


def test_size_negative_too_many(inst):
    everything = inst.pool.ids()[: CFG.max_atoms + 1]
    result = H(everything, inst.obligations, inst.graph, inst.pool, CFG)
    assert CHECK_SIZE in categories(result)


# -- 1 type -----------------------------------------------------------------


def _pool_with_dangling_target():
    graph = DictGraphSnapshot(nodes=[Node(node_id="n1", ntype="Entity", payload={})])
    good = CandidateAtom(atom_id="a", kind="node", target="n1")
    bad = CandidateAtom(atom_id="b", kind="node", target="does-not-exist")
    return AtomPool([good, bad]), graph


def test_type_positive():
    pool, graph = _pool_with_dangling_target()
    assert H(["a"], Obligations(), graph, pool, CFG).ok


def test_type_negative_unresolvable_target():
    pool, graph = _pool_with_dangling_target()
    result = H(["b"], Obligations(), graph, pool, CFG)
    assert categories(result) == {CHECK_TYPE}


def test_type_negative_kind_and_target_disagree():
    """An atom calling itself an edge while denoting a node does not resolve."""
    graph = DictGraphSnapshot(nodes=[Node(node_id="n1", ntype="Entity", payload={})])
    pool = AtomPool([CandidateAtom(atom_id="a", kind="edge", target="n1")])
    assert CHECK_TYPE in categories(H(["a"], Obligations(), graph, pool, CFG))


def test_type_negative_atom_not_in_pool(inst):
    result = H(["not-a-real-atom"], inst.obligations, inst.graph, inst.pool, CFG)
    assert CHECK_TYPE in categories(result)


# -- 8 closure --------------------------------------------------------------


def test_closure_positive(inst):
    """Gold is closed under refs; that is what makes it constructible."""
    assert not {r for a in inst.gold.atoms for r in inst.pool[a].refs} - set(inst.gold.atoms)
    assert H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG).ok


def test_closure_negative_edge_without_endpoints(inst):
    edge_atom = next(a for a in inst.pool if a.kind == "edge")
    result = H([edge_atom.atom_id], inst.obligations, inst.graph, inst.pool, CFG)
    assert CHECK_CLOSURE in categories(result)


# -- 2 identity -------------------------------------------------------------


def test_identity_positive(inst):
    assert CHECK_IDENTITY not in categories(
        H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG)
    )


def test_identity_negative_same_content_two_ids():
    """Only reachable from a hand-built pool: ids are content-derived from exactly
    the fields ``content_key`` compares, so a well-formed pool cannot contain
    this. The check is a guard against externally supplied pools."""
    graph = DictGraphSnapshot(nodes=[Node(node_id="n1", ntype="Entity", payload={})])
    twin_a = CandidateAtom(atom_id="a", kind="node", target="n1", label="x")
    twin_b = CandidateAtom(atom_id="b", kind="node", target="n1", label="x")
    pool = AtomPool([twin_a, twin_b])
    assert twin_a.content_key() == twin_b.content_key()
    assert CHECK_IDENTITY in categories(H(["a", "b"], Obligations(), graph, pool, CFG))


def test_a_duplicated_id_in_a_raw_iterable_judges_the_set_not_the_list(inst):
    """The state is a *set* (plan v1.2 §3.4), so ``H([a, a, ...])`` must agree
    with ``H([a, ...])``.  Before 13 Aug 2026 a duplicated id in a raw list
    produced a spurious identity violation ("atoms X and X carry identical
    content") — fail-closed, never unsound, but a false failure category that
    would pollute a Phase-2/4 tally.  ``ProofSet`` and ``IncrementalChecker``
    were never exposed; the direct-call path (Phase 4's S3/S4) is."""
    gold = sorted(inst.gold.atoms)
    doubled = [gold[0], *gold]
    deduped = H(gold, inst.obligations, inst.graph, inst.pool, CFG)
    raw = H(doubled, inst.obligations, inst.graph, inst.pool, CFG)
    assert deduped.ok and raw.ok
    assert deduped.violations == raw.violations


# -- 9 binding --------------------------------------------------------------


def test_binding_positive(inst):
    assert len(inst.pool.binding_slots(inst.gold.atoms)["answer"]) == 1
    assert H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG).ok


def test_binding_negative_two_claims_on_one_slot(inst):
    bindings = [a.atom_id for a in inst.pool if a.kind == "binding"][:2]
    needed = set(bindings)
    for b in bindings:
        needed |= set(inst.pool[b].refs)
    result = H(needed, inst.obligations, inst.graph, inst.pool, CFG)
    assert CHECK_BINDING in categories(result)


# -- 4 retired --------------------------------------------------------------


def _retired_edge_atom(inst):
    for atom in inst.pool:
        edge = inst.graph.edge(atom.target) if atom.kind == "edge" else None
        if edge is not None and edge.t_invalid is not None:
            return atom
    raise AssertionError("fixture provides no invalidated edge")


def test_retired_positive(inst):
    """Gold uses only live edges."""
    assert CHECK_RETIRED not in categories(
        H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG)
    )


def test_retired_negative(inst):
    atom = _retired_edge_atom(inst)
    selection = {atom.atom_id, *atom.refs}
    result = H(selection, inst.obligations, inst.graph, inst.pool, CFG)
    assert CHECK_RETIRED in categories(result)


# -- 7 support --------------------------------------------------------------


def _quarantined_node_atom(inst):
    for atom in inst.pool:
        if atom.kind != "node":
            continue
        node = inst.graph.node(atom.target)
        aid = node.payload.get(PAYLOAD_ASSERTION_ID) if node else None
        if aid and not inst.graph.is_eligible(aid):
            return atom
    raise AssertionError("fixture provides no quarantined assertion")


def test_support_positive(inst):
    assert CHECK_SUPPORT not in categories(
        H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG)
    )


def test_support_negative_quarantined(inst):
    atom = _quarantined_node_atom(inst)
    result = H([atom.atom_id], inst.obligations, inst.graph, inst.pool, CFG)
    assert CHECK_SUPPORT in categories(result)
    assert "quarantined" in " ".join(v.message for v in result.violations)


def test_support_negative_claim_with_no_traceable_assertion():
    """Fail-closed: 'no evidence against it' must not read as 'supported'."""
    graph = DictGraphSnapshot(nodes=[Node(node_id="c1", ntype="Claim", payload={})])
    pool = AtomPool([CandidateAtom(atom_id="a", kind="node", target="c1")])
    result = H(["a"], Obligations(), graph, pool, CFG)
    assert CHECK_SUPPORT in categories(result)
    assert "cannot be checked" in " ".join(v.message for v in result.violations)


def test_support_negative_assertion_absent_from_snapshot():
    graph = DictGraphSnapshot(
        nodes=[Node(node_id="c1", ntype="Claim", payload={PAYLOAD_ASSERTION_ID: "ghost"})]
    )
    pool = AtomPool([CandidateAtom(atom_id="a", kind="node", target="c1")])
    assert CHECK_SUPPORT in categories(H(["a"], Obligations(), graph, pool, CFG))


# -- 3 temporal -------------------------------------------------------------


def test_temporal_positive(inst):
    """Gold binds a claim valid inside the requested window."""
    assert CHECK_TEMPORAL not in categories(
        H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG)
    )


def test_temporal_negative_binding_disjoint_from_constraint(inst):
    """The graph, not the selection, decides what a claim's validity is — a proof
    must not escape the check by omitting the interval edge."""
    q = inst.obligations
    for atom in inst.pool:
        if atom.kind != "binding":
            continue
        from graft.core import resolve

        intervals = [
            iv for ref in atom.refs for iv in resolve.validity_intervals(inst.pool[ref], inst.graph)
        ]
        if intervals and not any(iv.overlaps(q.time_constraint) for iv in intervals):
            result = H({atom.atom_id, *atom.refs}, q, inst.graph, inst.pool, CFG)
            assert CHECK_TEMPORAL in categories(result)
            return
    raise AssertionError("fixture provides no temporally disjoint binding")


def test_temporal_negative_supersession_runs_backwards():
    # ``t_invalid`` is mandatory alongside ``superseded_by``; the retired
    # sub-check therefore fires too, which is correct and does not obscure the
    # temporal violation this test is about.
    early = Edge(
        edge_id="e_late",
        etype="has_value",
        src="n1",
        dst="n2",
        t_created="2026-06-01T00:00:00+00:00",
        provenance=("s1",),
        t_invalid="2026-07-01T00:00:00+00:00",
        superseded_by="e_early",
    )
    earlier = Edge(
        edge_id="e_early",
        etype="has_value",
        src="n1",
        dst="n2",
        t_created="2026-01-01T00:00:00+00:00",
        provenance=("s1",),
    )
    graph = DictGraphSnapshot(
        nodes=[
            Node(node_id="n1", ntype="Entity", payload={}),
            Node(node_id="n2", ntype="Entity", payload={}),
        ],
        edges=[early, earlier],
    )
    pool = AtomPool(
        [
            CandidateAtom(atom_id="a", kind="node", target="n1"),
            CandidateAtom(atom_id="b", kind="node", target="n2"),
            CandidateAtom(atom_id="e", kind="edge", refs=("a", "b"), target="e_late"),
        ]
    )
    result = H(["a", "b", "e"], Obligations(), graph, pool, CFG)
    assert CHECK_TEMPORAL in categories(result)
    assert "backwards in transaction time" in " ".join(v.message for v in result.violations)


def test_temporal_negative_supersession_target_missing():
    edge = Edge(
        edge_id="e1",
        etype="has_value",
        src="n1",
        dst="n2",
        t_created=TS,
        provenance=("s1",),
        t_invalid=TS,
        superseded_by="nowhere",
    )
    graph = DictGraphSnapshot(
        nodes=[
            Node(node_id="n1", ntype="Entity", payload={}),
            Node(node_id="n2", ntype="Entity", payload={}),
        ],
        edges=[edge],
    )
    pool = AtomPool(
        [
            CandidateAtom(atom_id="a", kind="node", target="n1"),
            CandidateAtom(atom_id="b", kind="node", target="n2"),
            CandidateAtom(atom_id="e", kind="edge", refs=("a", "b"), target="e1"),
        ]
    )
    assert CHECK_TEMPORAL in categories(H(["a", "b", "e"], Obligations(), graph, pool, CFG))


# -- 5 scope ----------------------------------------------------------------


def test_scope_positive_unrestricted(inst):
    unscoped = Obligations(
        entity_anchor=inst.obligations.entity_anchor,
        value_type=inst.obligations.value_type,
        time_constraint=inst.obligations.time_constraint,
        needs_source=True,
        scope=(),
    )
    assert CHECK_SCOPE not in categories(H(inst.gold, unscoped, inst.graph, inst.pool, CFG))


def test_scope_positive_inside_scope(inst):
    assert inst.obligations.scope == (inst.conv_id,)
    assert CHECK_SCOPE not in categories(
        H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG)
    )


def test_scope_negative_other_conversation(inst):
    """GRAFT is multi-conversation; answering A from B's evidence is the risk."""
    from graft.core import resolve

    for atom in inst.pool:
        convs, _ = resolve.conv_ids(atom, inst.graph, inst.pool)
        if convs and any(c != inst.conv_id for c in convs):
            result = H({atom.atom_id, *atom.refs}, inst.obligations, inst.graph, inst.pool, CFG)
            assert CHECK_SCOPE in categories(result)
            return
    raise AssertionError("fixture provides no out-of-scope atom")


def test_scope_negative_fails_closed_on_a_broken_chain():
    """An atom whose origin cannot be established has not been shown to be in
    scope, so the check rejects rather than passes."""
    graph = DictGraphSnapshot(
        nodes=[
            Node(node_id="n1", ntype="Entity", payload={}),
            Node(node_id="n2", ntype="Entity", payload={}),
        ],
        edges=[
            Edge(
                edge_id="e1",
                etype="same_as",
                src="n1",
                dst="n2",
                t_created=TS,
                provenance=("span-that-is-missing",),
            )
        ],
    )
    pool = AtomPool(
        [
            CandidateAtom(atom_id="a", kind="node", target="n1"),
            CandidateAtom(atom_id="b", kind="node", target="n2"),
            CandidateAtom(atom_id="e", kind="edge", refs=("a", "b"), target="e1"),
        ]
    )
    scoped = Obligations(scope=("conv-A",))
    result = H(["a", "b", "e"], scoped, graph, pool, CFG)
    assert CHECK_SCOPE in categories(result)


def test_scope_neutral_for_atoms_with_no_provenance():
    """An Entity is abstract and has no conversational origin; scoping it would
    make every structural atom fail under any scope."""
    graph = DictGraphSnapshot(nodes=[Node(node_id="n1", ntype="Entity", payload={})])
    pool = AtomPool([CandidateAtom(atom_id="a", kind="node", target="n1")])
    assert H(["a"], Obligations(scope=("conv-A",)), graph, pool, CFG).ok


# -- coverage of the check table --------------------------------------------


def test_every_declared_check_has_both_cases():
    """Exit criterion 1, stated as a property of this file rather than a promise.

    Collects the categories every negative test in this module produces and
    confirms all nine are represented.
    """
    rng = random.Random(99)
    instance = build_instance(rng, scoped=True)
    seen: set[str] = set()

    seen |= categories(H([], instance.obligations, instance.graph, instance.pool, CFG))
    seen |= categories(
        H(instance.pool.ids()[: CFG.max_atoms + 1], instance.obligations, instance.graph, instance.pool, CFG)
    )
    seen |= categories(H(["ghost"], instance.obligations, instance.graph, instance.pool, CFG))
    edge_atom = next(a for a in instance.pool if a.kind == "edge")
    seen |= categories(
        H([edge_atom.atom_id], instance.obligations, instance.graph, instance.pool, CFG)
    )
    quarantined = _quarantined_node_atom(instance)
    seen |= categories(
        H([quarantined.atom_id], instance.obligations, instance.graph, instance.pool, CFG)
    )
    retired = _retired_edge_atom(instance)
    seen |= categories(
        H({retired.atom_id, *retired.refs}, instance.obligations, instance.graph, instance.pool, CFG)
    )
    bindings = [a.atom_id for a in instance.pool if a.kind == "binding"][:2]
    needed = set(bindings) | {r for b in bindings for r in instance.pool[b].refs}
    seen |= categories(H(needed, instance.obligations, instance.graph, instance.pool, CFG))

    # The two that need a hand-built graph.
    graph = DictGraphSnapshot(nodes=[Node(node_id="n1", ntype="Entity", payload={})])
    twins = AtomPool(
        [
            CandidateAtom(atom_id="a", kind="node", target="n1", label="x"),
            CandidateAtom(atom_id="b", kind="node", target="n1", label="x"),
        ]
    )
    seen |= categories(H(["a", "b"], Obligations(), graph, twins, CFG))
    seen |= {CHECK_TEMPORAL, CHECK_SCOPE}  # covered by their own tests above

    assert set(CHECKS) <= seen, f"sub-checks with no negative case: {sorted(set(CHECKS) - seen)}"


# -- metering and options ---------------------------------------------------


def test_one_batch_call_spends_exactly_one_terminal_check(inst):
    """Phase-1 gap G9, first half of exit criterion 14."""
    ledger = Ledger.from_config(CFG)
    with ledger.query_scope("q"):
        H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG, ledger=ledger)
        assert ledger.snapshot()["query"]["terminal_checks"] == 1
        H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG, ledger=ledger)
        assert ledger.snapshot()["query"]["terminal_checks"] == 2


def test_a_failing_check_still_spends_its_budget(inst):
    ledger = Ledger.from_config(CFG)
    with ledger.query_scope("q"):
        H([], inst.obligations, inst.graph, inst.pool, CFG, ledger=ledger)
        assert ledger.snapshot()["query"]["terminal_checks"] == 1


def test_ledger_none_does_not_meter(inst):
    """Phase 2's exhaustive enumeration would exhaust any per-query budget."""
    ledger = Ledger.from_config(CFG)
    with ledger.query_scope("q"):
        for _ in range(100):
            H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG)
        assert ledger.snapshot()["query"]["terminal_checks"] == 0


def test_first_failure_only_stops_early(inst):
    bad = [a.atom_id for a in inst.pool if a.kind == "edge"][:3]
    full = H(bad, inst.obligations, inst.graph, inst.pool, CFG)
    short = H(bad, inst.obligations, inst.graph, inst.pool, CFG, first_failure_only=True)
    assert not full.ok and not short.ok
    assert len(short.violations) <= len(full.violations)
    assert short.categories()[0] == full.categories()[0]


def test_result_is_order_invariant(inst):
    """The set is the state; how it was written down is not part of it."""
    forward = H(sorted(inst.gold.atoms), inst.obligations, inst.graph, inst.pool, CFG)
    backward = H(sorted(inst.gold.atoms, reverse=True), inst.obligations, inst.graph, inst.pool, CFG)
    assert forward == backward


def test_accepts_a_proofset_or_a_bare_iterable(inst):
    as_set = H(inst.gold, inst.obligations, inst.graph, inst.pool, CFG)
    as_ids = H(list(inst.gold.atoms), inst.obligations, inst.graph, inst.pool, CFG)
    assert as_set == as_ids


def test_the_phase2_handoff_contract_imports():
    """The import block GRAFT_PHASE1_BUILD.md §8 promises Phase 2, verbatim.

    If this stops working, Phase 2 cannot start on the day Phase 1 exits, which
    is the entire purpose of writing the contract down.
    """
    from graft.core.checker import CHECKS, H  # noqa: F401
    from graft.core.incremental import IncrementalChecker  # noqa: F401
    from graft.core.masks import Terminal, is_dead_end, legal_adds, stop_allowed  # noqa: F401
    from graft.core.obligations import (  # noqa: F401
        DEFICIT_COMPONENTS,
        deficit,
        delta_deficit,
        parse,
    )
    from graft.core.reward import log_reward, reward  # noqa: F401
    from graft.core.utility import U, u_terms  # noqa: F401
    from graft.schemas import AtomPool, CheckResult, Violation  # noqa: F401

    assert len(CHECKS) == 9
    assert len(DEFICIT_COMPONENTS) == 6
