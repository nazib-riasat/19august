"""The commit pipeline, the DEFER queue and the corruption audit (P6.1/3/6).

Exit criteria 2, 3, 4, 5, 7, 8 and 14.  All of it on a bare interpreter with stub
decoders — step 3 lands before any torch import because the commit pipeline is
what Phase 7 reads and what the corruption audit certifies, and none of that
needs a learner to exist.
"""

from __future__ import annotations

import pytest

from graft.eventlog import EventLog
from graft.graphbuild.candidates import candidates_for, proposer_recall, recall_at_k
from graft.graphbuild.commit import (
    Committer,
    DeferQueue,
    claim_node,
    corruption_audit,
    entity_node,
)
from graft.graphbuild.items import d1_items, d2_items, mention_records, yields
from graft.graphbuild.validate import Commit
from graft.graphstore import ReplayGraphStore
from graft.schemas import (
    Assertion,
    AssertionFlags,
    Edge,
    Node,
    SourceSpan,
    Turn,
)

T1 = "2023-01-01T00:00:00+00:00"
T2 = "2023-06-01T00:00:00+00:00"


@pytest.fixture()
def log(tmp_path):
    """A miniature Stage-A log in the real write path's shape: one turn, one
    mention (span + ``mention.add``), two eligible claims and one quarantined."""
    handle = EventLog.open(tmp_path / "events.jsonl")
    handle.append("span.add", SourceSpan("sp_m", "t1", 8, 23).to_dict())
    handle.append(
        "mention.add",
        {"span_id": "sp_m", "turn_id": "t1", "text": "Fitbit Charge 3", "rung": "exact"},
    )
    handle.append("span.add", SourceSpan("sp1", "t1", 0, 23).to_dict())
    for aid, text, elig in (
        ("a1", "The user uses a Fitbit Charge 3 for 6 months.", "eligible"),
        ("a2", "The user uses a Fitbit Charge 3 for 9 months.", "eligible"),
        ("a3", "Something the span does not carry.", "quarantined"),
    ):
        handle.append(
            "assertion.add",
            Assertion(
                aid, "claim", text, ("sp1",), AssertionFlags(asserted_by="user"), T1
            ).to_dict(),
        )
        handle.append(
            "assertion.set_eligibility", {"assertion_id": aid, "eligibility": elig}
        )
    handle.append(
        "turn.add",
        Turn("t1", "c1", "s1", "user", T1, "I use a Fitbit Charge 3 every day").to_dict(),
    )
    yield handle
    handle.close()


# -- P6.1 items -------------------------------------------------------------


def test_mentions_come_from_mention_add_events_alone(log):
    """Exit criterion 7, verbatim — and the two cases the retired set-difference
    heuristic got wrong (PHASE6_DECISIONS §7): a mention whose span coincides
    with a quote span must survive, and a crash-repaired duplicate must not
    become two items."""
    records = mention_records(log)
    assert [m.span_id for m in records] == ["sp_m"]
    assert records[0].text == "Fitbit Charge 3"
    assert records[0].rung == "exact"


def test_a_mention_whose_span_coincides_with_a_quote_span_survives(tmp_path):
    """Under set difference this mention vanished: the quote cites the same
    content-derived span id, so 'uncited span' was false.  The mention.add event
    is what makes it recoverable — the reason Phase 5 added the op."""
    handle = EventLog.open(tmp_path / "e.jsonl")
    handle.append("span.add", SourceSpan("sp_x", "t1", 8, 23).to_dict())
    handle.append(
        "mention.add",
        {"span_id": "sp_x", "turn_id": "t1", "text": "Fitbit Charge 3", "rung": "exact"},
    )
    handle.append(
        "assertion.add",
        Assertion(
            "a9", "claim", "The user uses a Fitbit.", ("sp_x",),
            AssertionFlags(asserted_by="user"), T1,
        ).to_dict(),
    )
    handle.append(
        "turn.add",
        Turn("t1", "c1", "s1", "user", T1, "I use a Fitbit Charge 3 every day").to_dict(),
    )
    records = mention_records(handle)
    assert [m.span_id for m in records] == ["sp_x"], (
        "a mention cited by an assertion quote is still a mention"
    )
    handle.close()


def test_a_crash_repaired_duplicate_mention_is_one_item_not_two(tmp_path):
    """The Phase-5 crash-repair path legitimately rewrites a turn's mention
    events; the dedup on span_id is what keeps one mention one item."""
    handle = EventLog.open(tmp_path / "e.jsonl")
    for _ in range(2):  # the same mention written twice across a crash-resume
        handle.append("span.add", SourceSpan("sp_x", "t1", 8, 23).to_dict())
        handle.append(
            "mention.add",
            {"span_id": "sp_x", "turn_id": "t1", "text": "Fitbit Charge 3", "rung": "exact"},
        )
    handle.append(
        "turn.add",
        Turn("t1", "c1", "s1", "user", T1, "I use a Fitbit Charge 3 every day").to_dict(),
    )
    assert len(mention_records(handle)) == 1
    assert yields(handle)["mentions"] == 1
    handle.close()


def test_d1_items_derive_from_the_log_with_no_re_extraction(log):
    """Exit criterion 7.  Re-running the extractor to rebuild items would make a
    label collected in week 1 stop referring to anything in week 3."""
    items = d1_items(log)
    assert len(items) == 1
    item = items[0]
    assert item["mention"] == "Fitbit Charge 3"
    assert item["candidates"] == []  # legal cold start
    assert set(item["actions"]) == {
        "LINK_EXISTING(<entity_id>)", "CREATE_NEW_ENTITY", "NON_ENTITY", "DEFER"
    }


def test_every_item_carries_the_snapshot_sequence(log):
    """G12's leak guard, established at derivation time so no downstream
    featurizer has to remember to ask for the right snapshot."""
    assert all("snapshot_seq" in i for i in d1_items(log))


def test_d2_items_are_empty_without_a_proposer_rather_than_falling_back(log):
    """Negatives come from the proposer, never from random pairing
    (`GATE0_CONTRACT.md` item 6): a random negative is trivially INDEPENDENT and
    would inflate macro-F1 on the class nobody cares about."""
    assert d2_items(log, pairs_for=None) == []


def test_yields_report_the_measured_rates_and_the_multi_span_absence(log):
    report = yields(log)
    assert report["turns"] == 1
    assert report["mentions"] == 1
    assert report["assertions_eligible"] == 2
    assert report["multi_span_assertions"] == 0
    assert "Do not size D1/D2 items assuming multi-span input" in report["multi_span_note"]


# -- P6.3 candidates --------------------------------------------------------


def test_an_exact_normalised_match_is_never_displaced_by_a_similarity_score():
    """Letting a cosine ranking bury an exact match would make the generator's
    recall depend on the embedder's mood."""
    from graft.graphstore import DictGraphSnapshot

    snap = DictGraphSnapshot()
    snap.add_node(Node("e1", "Entity", {"name": "Sankeien  GARDEN"}))

    class M:
        text = "sankeien garden"

    got = candidates_for(M(), snap, 10)
    assert [c["entity_id"] for c in got] == ["e1"]
    assert got[0]["how"] == "normalised_exact"


def test_candidate_recall_scores_only_the_items_gold_says_are_links():
    """An item whose gold action is CREATE_NEW_ENTITY has no correct candidate to
    recall; counting it would measure the class balance instead of the
    generator."""
    items = [
        {"item_id": "d1_0", "candidates": [{"entity_id": "e1"}]},
        {"item_id": "d1_1", "candidates": []},
    ]
    got = recall_at_k(items, {"d1_0": "e1"})
    assert got["linkable_items"] == 1
    assert got["candidate_recall_at_k"] == 1.0


def test_recall_with_no_gold_links_is_nan_not_a_flattering_one():
    import math

    got = recall_at_k([{"item_id": "d1_0", "candidates": []}], {})
    assert math.isnan(got["candidate_recall_at_k"])


def test_proposer_recall_reports_the_pairs_it_never_proposed():
    """A conflict never proposed is a conflict never classified, and no
    downstream macro-F1 can see the loss."""
    items = [
        {"claim_a": {"assertion_id": "a1"}, "claim_b": {"assertion_id": "a2"}},
    ]
    got = proposer_recall(items, [("a1", "a2"), ("a1", "a3")])
    assert got["gold_pairs"] == 2 and got["recalled"] == 1
    assert got["proposer_recall"] == 0.5


# -- P6.6 the commit pipeline ----------------------------------------------


def test_create_link_and_supersede_is_the_knowledge_update_shape(log):
    committer = Committer(log)
    assert committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"]).accepted
    entity = entity_node("Fitbit Charge 3", "c1").node_id
    assert committer.link_existing("a2", "claim", entity, T2, ["sp1"]).accepted
    assert committer.supersede(claim_node("a1").node_id, claim_node("a2").node_id, T2, ["sp1"]).accepted
    assert committer.report()["violation_rate"] == 0.0


def test_a_quarantined_assertion_cannot_commit_through_any_decoder_path(log):
    """Exit criterion 4, at the pipeline level rather than the validator's."""
    committer = Committer(log)
    result = committer.create_entity("a3", "claim", "Ghost", "c1", T1, ["sp1"])
    assert not result.accepted
    assert "eligibility" in result.result.categories()
    assert committer.rejections["eligibility"] == 1


def test_supersession_invalidates_and_never_deletes(log):
    """Exit criterion 2, and the whole reason for the Zep precedent."""
    committer = Committer(log)
    committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"])
    entity = entity_node("Fitbit Charge 3", "c1").node_id
    committer.link_existing("a2", "claim", entity, T2, ["sp1"])
    old, new = claim_node("a1").node_id, claim_node("a2").node_id

    before_seq = log.snapshot_id()
    committer.supersede(old, new, T2, ["sp1"])

    after = ReplayGraphStore(log).at()
    old_edges = [e for e in after.edges_of(old, "about_entity") if e.src == old]
    assert old_edges and not after.is_live(old_edges[0].edge_id)

    # ...and the pre-supersession snapshot still serves the old fact live.
    before = ReplayGraphStore(log).at(before_seq)
    assert before.is_live(old_edges[0].edge_id)


def test_the_committed_graph_replays_to_an_identical_digest(log):
    """Exit criterion 5: rebuilding from the log twice gives identical state, and
    no new op was invented — everything replays through the existing GRAPH_OPS."""
    committer = Committer(log)
    committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"])
    first = ReplayGraphStore(log).at().state_digest()
    second = ReplayGraphStore(log).at().state_digest()
    assert first == second

    from graft.graphstore import GRAPH_OPS
    from graft.ingest.pipeline import OP_MENTION

    # ``mention.add`` is a Phase-5 op that replay deliberately ignores
    # (GRAPH_OPS excludes it by design); everything else must be a graph op.
    ops = {op for _, op in ReplayGraphStore(log).iter_ops()}
    assert ops <= set(GRAPH_OPS) | {OP_MENTION}


def test_a_conflict_leaves_both_claims_live(log):
    """A conflict is not a supersession: neither claim wins.  Collapsing the two
    would silently retire a fact the model only said was contested."""
    committer = Committer(log)
    committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"])
    entity = entity_node("Fitbit Charge 3", "c1").node_id
    committer.link_existing("a2", "claim", entity, T2, ["sp1"])
    old, new = claim_node("a1").node_id, claim_node("a2").node_id

    assert committer.contradict(new, old, T2, ["sp1"]).accepted
    after = ReplayGraphStore(log).at()
    for claim in (old, new):
        edges = [e for e in after.edges_of(claim, "about_entity") if e.src == claim]
        assert all(after.is_live(e.edge_id) for e in edges)


def test_entity_ids_are_scoped_to_the_conversation(log):
    """Two users who both mention "my car" have two different cars.  A globally
    keyed entity would merge them — the most damaging Stage-B error, introduced
    by the id scheme rather than by a decoder."""
    assert entity_node("my car", "convA").node_id != entity_node("my car", "convB").node_id
    assert entity_node("My  Car", "convA").node_id == entity_node("my car", "convA").node_id


# -- the DEFER queue --------------------------------------------------------


def test_defer_writes_nothing_to_the_graph():
    """A DEFER that quietly created a placeholder entity would be
    CREATE_NEW_ENTITY under another name, and the on/off ablation would measure
    nothing."""
    queue = DeferQueue()
    queue.park("fitbit", {"item_id": "d1_0"})
    assert len(queue) == 1
    assert queue.anchors() == ("fitbit",)
    assert queue.revisit("fitbit") == [{"item_id": "d1_0"}]
    assert len(queue) == 0


# -- G9 the corruption audit ------------------------------------------------


def test_the_corruption_audit_runs_green_on_a_clean_update_stream(log):
    """Exit criterion 3, on synthetic decoder outputs — the harness property,
    before any learned decoder exists."""
    committer = Committer(log)
    proposals: list[Commit] = []
    results = []

    entity = entity_node("Fitbit Charge 3", "c1")
    claim_a, claim_b = claim_node("a1"), claim_node("a2")
    first = Commit(
        nodes=[entity, claim_a],
        edges=[
            Edge(
                edge_id="edge_a",
                etype="about_entity",
                src=claim_a.node_id,
                dst=entity.node_id,
                t_created=T1,
                provenance=("sp1",),
            )
        ],
        label="create",
    )
    proposals.append(first)
    results.append(committer.submit(first))

    second = Commit(
        nodes=[claim_b],
        edges=[
            Edge(
                edge_id="edge_b",
                etype="about_entity",
                src=claim_b.node_id,
                dst=entity.node_id,
                t_created=T2,
                provenance=("sp1",),
            ),
            Edge(
                edge_id="edge_sup",
                etype="supersedes",
                src=claim_b.node_id,
                dst=claim_a.node_id,
                t_created=T2,
                provenance=("sp1",),
            ),
        ],
        invalidations=[("edge_a", T2, "edge_sup")],
        label="supersede",
    )
    proposals.append(second)
    results.append(committer.submit(second))

    assert all(r.accepted for r in results)
    audit = corruption_audit(log, results, proposals)
    assert audit["green"], audit["corrupted_steps"]
    assert audit["steps_audited"] == 2
    assert audit["failures"] == {
        "a_supersession_effective": 0,
        "b_history_intact": 0,
        "c_no_collateral": 0,
    }


def test_the_audit_catches_a_collateral_flip(log):
    """Property (c) is the one that catches real damage: a commit that changed a
    live edge it never declared."""
    committer = Committer(log)
    entity = entity_node("Fitbit Charge 3", "c1")
    claim_a = claim_node("a1")
    edge = Edge("edge_a", "about_entity", claim_a.node_id, entity.node_id, T1, ("sp1",))
    proposal = Commit(nodes=[entity, claim_a], edges=[edge], label="create")
    result = committer.submit(proposal)
    assert result.accepted

    # The audit is handed a *declaration that omits the edge* — exactly what an
    # undeclared collateral change looks like from the outside.
    lying = Commit(nodes=[entity, claim_a], label="create (undeclared edge)")
    audit = corruption_audit(log, [result], [lying])
    assert not audit["green"]
    assert audit["failures"]["c_no_collateral"] == 1


def test_the_audit_counts_rather_than_raises(log):
    """At Gate 1 this becomes a measured error rate of the decoders, and an
    exception would stop the measurement at the first bad step."""
    committer = Committer(log)
    entity = entity_node("E", "c1")
    proposal = Commit(nodes=[entity], label="x")
    result = committer.submit(proposal)
    audit = corruption_audit(log, [result], [Commit(label="empty")])
    assert isinstance(audit["failures"], dict)
    assert audit["steps_audited"] == 1


def test_the_audit_names_its_three_properties_in_the_artefact(log):
    """The architecture required the audit and no document defined it; the
    definition ships with the numbers so it cannot drift."""
    audit = corruption_audit(log, [], [])
    assert set(audit["properties"]) == {"a", "b", "c"}
    assert "measured error rate of the" in audit["reading"]


# -- the 14 Aug 2026 audit's regressions --------------------------------------


def test_a_value_update_is_supersedable(log):
    """The canonical knowledge update the corpus contains — "my weight is 70kg"
    -> "68kg" — is a **Value** pair, and under the original Claim-only endpoint
    rows it was refused on endpoint typing: 13.2% of the live pilot's eligible
    assertions (value/event kinds) could never be superseded or contradicted,
    which is C1's own update case structurally unrepresentable."""
    committer = Committer(log)
    committer.create_entity("a1", "value", "weight", "c1", T1, ["sp1"])
    entity = entity_node("weight", "c1").node_id
    committer.link_existing("a2", "value", entity, T2, ["sp1"])
    old, new = claim_node("a1", "value").node_id, claim_node("a2", "value").node_id

    outcome = committer.supersede(old, new, T2, ["sp1"])
    assert outcome.accepted, [v.message for v in outcome.result.violations]

    conflict = committer.contradict(new, old, T2, ["sp1"])
    assert conflict.accepted


def test_supersession_retires_every_currency_edge_not_only_about_entity(log):
    """The audit measured the narrow version leaving a superseded claim's
    valid_during edge live — so the retired fact kept answering temporal
    queries.  History edges (supersedes/contradicts) stay live by design."""
    committer = Committer(log)
    committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"])
    entity = entity_node("Fitbit Charge 3", "c1").node_id
    old = claim_node("a1").node_id
    vd = committer.valid_during(old, 100.0, 200.0, T1, ["sp1"])
    assert vd is not None and vd.accepted

    committer.link_existing("a2", "claim", entity, T2, ["sp1"])
    new = claim_node("a2").node_id
    outcome = committer.supersede(old, new, T2, ["sp1"])
    assert outcome.accepted

    after = ReplayGraphStore(log).at()
    live_currency = [
        e for e in after.edges_of(old)
        if e.etype not in ("supersedes", "contradicts") and after.is_live(e.edge_id)
    ]
    assert live_currency == [], (
        f"superseded claim still current via {[e.etype for e in live_currency]}"
    )
    supersedes = [e for e in after.edges_of(old, "supersedes")]
    assert supersedes and after.is_live(supersedes[0].edge_id)


def test_one_assertion_cannot_become_two_live_nodes(log):
    """The kind-to-ntype mapping applied twice is still one assertion: committing
    the same assertion as a Claim and again as a Value derived two node ids and
    passed every per-node check (audit finding).  Refused now."""
    committer = Committer(log)
    committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"])
    entity = entity_node("Fitbit Charge 3", "c1").node_id

    second = committer.link_existing("a1", "value", entity, T2, ["sp1"])
    assert not second.accepted
    assert any("two live nodes" in v.message for v in second.result.violations)


def test_alias_growth_is_the_one_permitted_payload_evolution(log):
    """Surface forms accumulate through linking decisions (PHASE6_DECISIONS
    1.5's promise); any other payload change under the same id is still an id
    collision."""
    committer = Committer(log)
    committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"])
    entity = entity_node("Fitbit Charge 3", "c1").node_id

    grown = committer.add_aliases(entity, ["the Charge 3", "my Fitbit"])
    assert grown.accepted
    node = ReplayGraphStore(log).at().node(entity)
    assert set(node.payload["aliases"]) >= {"the Charge 3", "my Fitbit"}

    from graft.graphbuild.validate import Commit

    mutated = Node(
        node_id=entity, ntype="Entity",
        payload={**dict(node.payload), "name": "different name"},
    )
    refused = committer.submit(Commit(nodes=[mutated]))
    assert not refused.accepted


def test_the_raw_surface_form_is_preserved_as_an_alias():
    """The canonical name is normalized; losing the raw form would make the
    payload lie about what was said."""
    node = entity_node("Fitbit Charge 3", "c1")
    assert node.payload["name"] == "fitbit charge 3"
    assert "Fitbit Charge 3" in node.payload["aliases"]
    assert node.payload["conv_id"] == "c1"


def test_d3_and_d4_commits_are_gated_by_the_calibrated_floor(log):
    """Decision 9, previously declared and unused: below the floor is a decline
    (counted, not a violation); at or above, the edge commits."""
    committer = Committer(log)
    committer.create_entity("a1", "claim", "Fitbit Charge 3", "c1", T1, ["sp1"])
    old = claim_node("a1").node_id

    declined = committer.valid_during(old, 100.0, 200.0, T1, ["sp1"], confidence=0.2)
    assert declined is None
    assert committer.below_floor == 1
    assert committer.report()["below_confidence_floor"] == 1

    accepted = committer.valid_during(old, 100.0, 200.0, T1, ["sp1"], confidence=0.9)
    assert accepted is not None and accepted.accepted
    assert committer.report()["violation_rate"] == 0.0


def test_a_torn_commit_converges_by_resubmission(tmp_path):
    """Atomicity is by resubmission, not by write: adds land before
    invalidations, so the torn state is the recoverable one (both facts live and
    visible), and re-submitting the same commit reaches the intended graph."""
    handle = EventLog.open(tmp_path / "e.jsonl")
    handle.append("span.add", SourceSpan("sp1", "t1", 0, 5).to_dict())
    handle.append(
        "assertion.add",
        Assertion("a1", "claim", "x", ("sp1",), AssertionFlags(asserted_by="u"), T1,
                  eligibility="eligible").to_dict(),
    )
    handle.append(
        "assertion.add",
        Assertion("a2", "claim", "y", ("sp1",), AssertionFlags(asserted_by="u"), T1,
                  eligibility="eligible").to_dict(),
    )
    handle.append("turn.add", Turn("t1", "c1", "s1", "user", T1, "hello").to_dict())

    committer = Committer(handle)
    committer.create_entity("a1", "claim", "Widget", "c1", T1, ["sp1"])
    entity = entity_node("Widget", "c1").node_id
    committer.link_existing("a2", "claim", entity, T2, ["sp1"])
    old, new = claim_node("a1").node_id, claim_node("a2").node_id

    from graft.graphbuild.commit import _edge

    snapshot = ReplayGraphStore(handle).at()
    edge = _edge("supersedes", new, old, T2, ["sp1"])
    invalidations = [
        (e.edge_id, T2, edge.edge_id)
        for e in snapshot.edges_of(old)
        if e.etype not in ("supersedes", "contradicts") and snapshot.is_live(e.edge_id)
    ]
    handle.append("edge.add", edge.to_dict())  # ...and the process dies here

    torn = ReplayGraphStore(handle).at()
    assert torn.is_live(edge.edge_id)
    assert all(torn.is_live(eid) for eid, _, _ in invalidations), (
        "the torn state must be the recoverable one: old fact still visible"
    )

    from graft.graphbuild.validate import Commit

    resubmit = Committer(handle).submit(
        Commit(edges=[edge], invalidations=invalidations, label="retry")
    )
    assert resubmit.accepted, [v.message for v in resubmit.result.violations]
    final = ReplayGraphStore(handle).at()
    assert final.is_live(edge.edge_id)
    assert all(not final.is_live(eid) for eid, _, _ in invalidations)
    handle.close()


# -- the stand-in constructor (G1/G12, rebuilt 14 Aug) -------------------------


def _two_turn_log(tmp_path, second_conv: str = "c1"):
    """Two turns, each with one mention and one eligible claim; the second
    mention's text matches the first's entity."""
    handle = EventLog.open(tmp_path / "e.jsonl")
    for ix, (turn_id, conv, text_norm, aid) in enumerate(
        (
            ("t1", "c1", "The user has a Fitbit Charge 3.", "a1"),
            ("t2", second_conv, "The Fitbit Charge 3 tracks sleep.", "a2"),
        )
    ):
        sp = f"sp{ix}"
        handle.append("span.add", SourceSpan(sp, turn_id, 8, 23).to_dict())
        handle.append(
            "mention.add",
            {"span_id": sp, "turn_id": turn_id, "text": "Fitbit Charge 3", "rung": "exact"},
        )
        handle.append(
            "assertion.add",
            Assertion(
                aid, "claim", text_norm, (sp,),
                AssertionFlags(asserted_by="user"), T1 if ix == 0 else T2,
            ).to_dict(),
        )
        handle.append(
            "assertion.set_eligibility", {"assertion_id": aid, "eligibility": "eligible"}
        )
        handle.append(
            "turn.add",
            Turn(turn_id, conv, f"s{ix}", "user",
                 T1 if ix == 0 else T2,
                 "I use a Fitbit Charge 3 every day").to_dict(),
        )
    return handle


def test_candidates_cannot_see_the_future(tmp_path):
    """Exit criterion 14's trap, on the real constructor: the first mention's
    candidate list must be empty (its entity does not exist yet), and the second
    mention — same surface, later turn — must see the first's entity."""
    from graft.graphbuild.standin import construct

    handle = _two_turn_log(tmp_path)
    out = construct(handle)
    items = out["d1_items"]
    assert len(items) == 2
    assert items[0]["candidates"] == [], "a mention must not see entities created later"
    assert items[1]["candidates"], "a later mention must see the earlier entity"
    assert items[1]["candidates"][0]["how"] == "normalised_exact"
    # And the stage_b pin reproduces the exact graph each list was drawn from.
    first_snap = ReplayGraphStore(handle).at(items[0]["stage_b_seq"])
    assert first_snap.counts()["nodes"] == 0
    handle.close()


def test_candidates_are_scoped_to_the_mention_s_conversation(tmp_path):
    """The wrong-merge guard at retrieval: an identical surface form in another
    conversation must not surface the first conversation's entity."""
    from graft.graphbuild.standin import construct

    handle = _two_turn_log(tmp_path, second_conv="c2")
    out = construct(handle)
    items = out["d1_items"]
    assert items[1]["candidates"] == [], (
        "another user's 'Fitbit Charge 3' is another user's entity"
    )
    # Two conversations, two entities.
    assert out["graph"]["nodes"] >= 4  # 2 entities + 2 claim nodes
    handle.close()


def test_the_standin_builds_edges_and_d2_items(tmp_path):
    """The first smoke run committed bare entities: zero edges, zero D2 items,
    and a corruption audit green over nothing.  The constructor now links every
    eligible assertion, so anchors exist and pairs are proposable."""
    from graft.graphbuild.standin import construct

    handle = _two_turn_log(tmp_path)
    out = construct(handle)
    assert out["graph"]["edges"] >= 2, "about_entity edges must exist"
    assert len(out["d2_items"]) == 1, "same-anchor claims must pair"
    item = out["d2_items"][0]
    assert item["claim_a"]["session_date"] <= item["claim_b"]["session_date"]
    assert out["corruption_audit"]["green"]
    assert out["commit"]["refused"] == 0
    handle.close()


def test_a_new_surface_form_grows_the_alias_set(tmp_path):
    """Linking a variant surface to a known entity accumulates it as an alias —
    the shape Phase 7's entity-match channel indexes."""
    from graft.graphbuild.standin import construct

    handle = EventLog.open(tmp_path / "e.jsonl")
    for ix, mention_text in enumerate(("Fitbit Charge 3", "FITBIT  charge 3")):
        turn_id = f"t{ix}"
        sp = f"sp{ix}"
        handle.append("span.add", SourceSpan(sp, turn_id, 0, len(mention_text)).to_dict())
        handle.append(
            "mention.add",
            {"span_id": sp, "turn_id": turn_id, "text": mention_text, "rung": "exact"},
        )
        handle.append(
            "turn.add",
            Turn(turn_id, "c1", f"s{ix}", "user", T1, mention_text + " every day").to_dict(),
        )
    out = construct(handle)
    assert out["graph"]["nodes"] == 1, "one entity for both surface forms"
    snap = ReplayGraphStore(handle).at()
    nodes = getattr(snap, "_nodes")
    entity = next(n for n in nodes.values() if n.ntype == "Entity")
    assert "Fitbit Charge 3" in entity.payload["aliases"]
    handle.close()


def test_d1_report_refuses_mismatched_lengths():
    """A silent zip-truncation would let an arm score only the items it
    answered."""
    from graft.graphbuild.gate1 import d1_report

    gold = [{"action": "NON_ENTITY"}] * 4
    with pytest.raises(ValueError, match="one prediction per gold item"):
        d1_report([{"action": "NON_ENTITY"}] * 2, gold)
