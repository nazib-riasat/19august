"""The commit validator and the endpoint table (P6.0, gaps G3 and G8).

Exit criteria 1, 4 and part of 5.  Everything here runs on a bare interpreter —
step 0 lands before any torch import precisely because the commit pipeline is
what Phase 7 reads and what the corruption audit certifies, and none of it needs
a learner to exist.
"""

from __future__ import annotations

import pytest

from graft.graphbuild.validate import (
    CHECK_DUPLICATE,
    CHECK_ELIGIBILITY,
    CHECK_ENDPOINT,
    CHECK_ID,
    CHECK_INTERVAL,
    CHECK_PROVENANCE,
    CHECK_SUPERSESSION,
    CHECKS,
    Commit,
    tally,
    validate,
)
from graft.graphstore import DictGraphSnapshot
from graft.schemas import (
    EDGE_TYPES,
    ENDPOINT_TABLE,
    NODE_TYPES,
    SELF_LOOP_FORBIDDEN,
    Assertion,
    AssertionFlags,
    Edge,
    Node,
    SourceSpan,
    Turn,
)

TS = "2023-01-01T00:00:00+00:00"
LATER = "2023-06-01T00:00:00+00:00"


@pytest.fixture()
def snap() -> DictGraphSnapshot:
    """A minimal but *real* snapshot: a turn, a span, one eligible and one
    quarantined assertion, and an entity to point at."""
    s = DictGraphSnapshot()
    s.add_turn(Turn("t1", "c1", "s1", "user", TS, "I moved to Yokohama last March"))
    s.add_span(SourceSpan("sp1", "t1", 2, 20))
    s.add_assertion(
        Assertion("a_ok", "event", "moved", ("sp1",), AssertionFlags(asserted_by="user"), TS, "eligible")
    )
    s.add_assertion(
        Assertion("a_bad", "claim", "q", ("sp1",), AssertionFlags(asserted_by="user"), TS, "quarantined")
    )
    s.add_node(Node("ent", "Entity", {"name": "Yokohama"}))
    return s


def _claim(node_id: str = "cl", assertion_id: str = "a_ok") -> Node:
    return Node(node_id, "Claim", {"assertion_id": assertion_id})


# -- the endpoint table (G3) ------------------------------------------------


def test_the_endpoint_table_covers_every_edge_type_and_names_real_node_types():
    """A table with a gap is worse than no table: the commit validator would pass
    the uncovered edge type silently."""
    assert set(ENDPOINT_TABLE) == set(EDGE_TYPES)
    for etype, (src, dst) in ENDPOINT_TABLE.items():
        assert src and dst, etype
        assert set(src) <= set(NODE_TYPES), etype
        assert set(dst) <= set(NODE_TYPES), etype


def test_certificate_is_absent_and_that_is_the_recorded_decision():
    """Plan §3.2 lists eleven node types; the vocabulary has ten.

    ``Certificate`` is dropped deliberately (G3, decision 2) because nothing
    writes or reads one.  The test pins the *count* so that adding it back is a
    deliberate edit against a failing assertion rather than a quiet widening of a
    frozen vocabulary.
    """
    assert len(NODE_TYPES) == 10
    assert "Certificate" not in NODE_TYPES


def test_the_endpoint_table_is_read_only():
    """It feeds commit decisions, so a caller mutating it would change what the
    graph accepts for every subsequent write in the process."""
    with pytest.raises(TypeError):
        ENDPOINT_TABLE["same_as"] = (("Claim",), ("Claim",))  # type: ignore[index]


# -- check 1: endpoint typing ----------------------------------------------


def test_a_valid_commit_passes_every_check(snap):
    commit = Commit(
        nodes=[_claim()],
        edges=[Edge("e1", "about_entity", "cl", "ent", TS, ("sp1",))],
    )
    assert validate(commit, snap).ok


def test_an_edge_may_reference_a_node_the_same_commit_creates(snap):
    """A commit is validated as a unit.  Resolving endpoints against the snapshot
    alone would reject every first-time write."""
    commit = Commit(
        nodes=[_claim(), Node("iv", "TimeInterval", {"start": 0.0, "end": 10.0})],
        edges=[Edge("e1", "valid_during", "cl", "iv", TS, ("sp1",))],
    )
    assert validate(commit, snap).ok


def test_a_wrongly_typed_endpoint_is_refused(snap):
    commit = Commit(
        nodes=[_claim()],
        edges=[Edge("e1", "valid_during", "cl", "ent", TS, ("sp1",))],
    )
    result = validate(commit, snap)
    assert not result.ok
    assert CHECK_ENDPOINT in result.categories()


def test_an_edge_to_a_nonexistent_node_is_a_dangling_reference(snap):
    commit = Commit(edges=[Edge("e1", "about_entity", "ghost", "ent", TS, ("sp1",))])
    result = validate(commit, snap)
    assert not result.ok
    assert CHECK_ENDPOINT in result.categories()


# -- check 2: ids -----------------------------------------------------------


def test_rewriting_an_identical_record_is_idempotent_replay_not_a_violation(snap):
    """The Phase-5 write path depends on it, and refusing it would make crash
    recovery impossible."""
    commit = Commit(nodes=[Node("ent", "Entity", {"name": "Yokohama"})])
    assert validate(commit, snap).ok


def test_the_same_id_with_different_content_is_refused(snap):
    """A content-derived id carrying two contents means the derivation lost a
    distinguishing field — the failure this check exists to surface."""
    commit = Commit(nodes=[Node("ent", "Entity", {"name": "Yokohama City"})])
    result = validate(commit, snap)
    assert not result.ok
    assert CHECK_ID in result.categories()


# -- check 3: provenance ----------------------------------------------------


def test_an_unresolvable_provenance_span_is_refused(snap):
    """``Edge`` guarantees non-empty; this guarantees non-fictional."""
    commit = Commit(
        nodes=[_claim()], edges=[Edge("e1", "about_entity", "cl", "ent", TS, ("nope",))]
    )
    result = validate(commit, snap)
    assert not result.ok
    assert CHECK_PROVENANCE in result.categories()


def test_the_schema_still_refuses_an_edge_with_no_provenance_at_all():
    """Enforced one level lower, and the validator does not duplicate it — a
    schema that permits an unsourced edge permits an unsourced proof."""
    with pytest.raises(ValueError, match="provenance"):
        Edge("e", "about_entity", "a", "b", TS, ())


# -- check 4: eligibility, fix F9's boundary -------------------------------


def test_a_quarantined_assertion_cannot_enter_the_active_graph(snap):
    """Exit criterion 4, and the reason the check is here at all: this is the
    moment after which retrieval could reach the record."""
    commit = Commit(nodes=[_claim("cl2", "a_bad")])
    result = validate(commit, snap)
    assert not result.ok
    assert CHECK_ELIGIBILITY in result.categories()


def test_an_assertion_backed_node_without_an_assertion_id_is_refused(snap):
    """It cannot be support-gated at all, so it fails closed rather than passing
    for lack of anything to check."""
    commit = Commit(nodes=[Node("cl3", "Claim", {})])
    result = validate(commit, snap)
    assert not result.ok
    assert CHECK_ELIGIBILITY in result.categories()


def test_an_unknown_assertion_id_is_ineligible(snap):
    """``is_eligible`` returns False for an id the snapshot has never seen — a
    proof may not lean on something that does not exist."""
    commit = Commit(nodes=[_claim("cl4", "a_never_stored")])
    assert CHECK_ELIGIBILITY in validate(commit, snap).categories()


def test_a_node_type_that_carries_no_assertion_is_not_gated(snap):
    """Entities, intervals and sources are not assertion-backed, so the gate does
    not apply and must not invent a requirement."""
    commit = Commit(nodes=[Node("iv", "TimeInterval", {"start": 0.0, "end": 1.0})])
    assert validate(commit, snap).ok


# -- check 5: intervals -----------------------------------------------------


def test_an_edge_invalid_before_it_was_created_is_refused(snap):
    commit = Commit(
        nodes=[_claim()],
        edges=[
            Edge("e1", "about_entity", "cl", "ent", LATER, ("sp1",), t_invalid=TS,
                 superseded_by=None)
        ],
    )
    assert CHECK_INTERVAL in validate(commit, snap).categories()


def test_invalidating_an_edge_the_snapshot_does_not_have_is_refused(snap):
    commit = Commit(invalidations=[("ghost", LATER, None)])
    assert CHECK_INTERVAL in validate(commit, snap).categories()


def test_an_empty_valid_during_interval_is_refused(snap):
    """Phase-1 gap G5's reason: no instant satisfies it, so it is not an
    answerable temporal claim."""
    commit = Commit(
        nodes=[_claim(), Node("iv0", "TimeInterval", {"start": 5.0, "end": 5.0})],
        edges=[Edge("e1", "valid_during", "cl", "iv0", TS, ("sp1",))],
    )
    assert CHECK_INTERVAL in validate(commit, snap).categories()


# -- check 6: self-loops and duplicates ------------------------------------


@pytest.mark.parametrize("etype", SELF_LOOP_FORBIDDEN)
def test_a_self_loop_on_a_reflexive_meaningless_type_is_refused(snap, etype):
    node = Node("cl", "Claim", {"assertion_id": "a_ok"}) if etype != "same_as" else None
    nodes = [node] if node else []
    endpoint = "cl" if etype != "same_as" else "ent"
    commit = Commit(
        nodes=nodes, edges=[Edge("e1", etype, endpoint, endpoint, TS, ("sp1",))]
    )
    assert CHECK_DUPLICATE in validate(commit, snap).categories()


def test_a_duplicate_live_edge_is_refused(snap):
    snap.add_node(Node("cl", "Claim", {"assertion_id": "a_ok"}))
    snap.add_edge(Edge("e0", "about_entity", "cl", "ent", TS, ("sp1",)))
    commit = Commit(edges=[Edge("e1", "about_entity", "cl", "ent", LATER, ("sp1",))])
    assert CHECK_DUPLICATE in validate(commit, snap).categories()


def test_re_asserting_a_relation_whose_edge_was_invalidated_is_allowed(snap):
    """Refusing it would make a superseded fact unrecoverable — the opposite of
    what non-destructive versioning is for."""
    snap.add_node(Node("cl", "Claim", {"assertion_id": "a_ok"}))
    snap.add_edge(Edge("e0", "about_entity", "cl", "ent", TS, ("sp1",)))
    snap.invalidate_edge("e0", LATER)
    commit = Commit(edges=[Edge("e1", "about_entity", "cl", "ent", LATER, ("sp1",))])
    assert validate(commit, snap).ok


def test_superseding_in_the_same_commit_that_invalidates_is_allowed(snap):
    """The normal supersession shape: invalidate the old edge and add the new one
    atomically."""
    snap.add_node(Node("cl", "Claim", {"assertion_id": "a_ok"}))
    snap.add_edge(Edge("e0", "about_entity", "cl", "ent", TS, ("sp1",)))
    commit = Commit(
        edges=[Edge("e1", "about_entity", "cl", "ent", LATER, ("sp1",))],
        invalidations=[("e0", LATER, "e1")],
    )
    assert validate(commit, snap).ok


# -- check 7: supersession --------------------------------------------------


def test_an_invalidation_naming_a_successor_that_does_not_exist_is_refused(snap):
    snap.add_node(Node("cl", "Claim", {"assertion_id": "a_ok"}))
    snap.add_edge(Edge("e0", "about_entity", "cl", "ent", TS, ("sp1",)))
    commit = Commit(invalidations=[("e0", LATER, "e_never")])
    assert CHECK_SUPERSESSION in validate(commit, snap).categories()


def test_the_schema_still_refuses_supersession_without_invalidation():
    """One field authoritative: ``is_live`` reads ``t_invalid``, so a superseded
    edge that still read as live would keep answering questions."""
    with pytest.raises(ValueError, match="t_invalid"):
        Edge("e", "about_entity", "a", "b", TS, ("sp",), superseded_by="e2")


# -- reporting --------------------------------------------------------------


def test_every_check_runs_even_after_one_fails(snap):
    """The rejection categories are a reported number, so short-circuiting would
    make the tally depend on check order."""
    commit = Commit(
        nodes=[_claim("cl5", "a_bad")],
        edges=[Edge("e1", "valid_during", "cl5", "ent", TS, ("ghost",))],
    )
    result = validate(commit, snap)
    assert {CHECK_ELIGIBILITY, CHECK_ENDPOINT, CHECK_PROVENANCE} <= set(result.categories())


def test_the_tally_reports_every_category_including_zeros(snap):
    """'No endpoint violations' and 'endpoint typing was never checked' must not
    look the same in a report."""
    counts = tally([validate(Commit(nodes=[_claim()]), snap)])
    assert set(counts) == set(CHECKS)
    assert all(v == 0 for v in counts.values())


def test_a_failing_result_always_says_why(snap):
    """Phase-1's rule: an unexplained rejection cannot be counted by category or
    debugged."""
    result = validate(Commit(nodes=[_claim("x", "a_bad")]), snap)
    assert not result.ok
    assert result.violations
    assert all(v.message for v in result.violations)


def test_an_empty_commit_is_vacuously_valid(snap):
    """A decoder that proposes nothing has violated nothing.  Stated as a test
    because the alternative — treating "no writes" as a failure — would make the
    DEFER path (which deliberately writes nothing) look like corruption."""
    empty = Commit()
    assert not empty
    assert validate(empty, snap).ok


def test_the_validator_imports_no_ml_library():
    """It decides what may enter the active graph, so it sits on the same side of
    the line as ``H``: if a learned score could reach it, "formally checkable"
    would stop meaning anything."""
    import ast
    from pathlib import Path

    import graft.graphbuild.validate as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    forbidden = {"torch", "torch_geometric", "transformers", "sentence_transformers"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (names & forbidden), f"validate.py imports {names & forbidden}"
