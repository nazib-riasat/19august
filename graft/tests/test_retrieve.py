"""Phase 7 — Stage C's exit criteria, as tests.

The numbering follows `GRAFT_PHASE7_BUILD.md` §5.  Criteria 12 (the ML boundary)
and part of 7 (the gold quarantine) live in ``test_structure.py``, where the rest
of the import-graph rules already are.

Everything here runs on a bare interpreter with ``StubEmbedder``: the five
channels are training-free by design (G6), and a test suite that needed a GPU to
check the temporal filter's fail-open semantics would stop being run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from graft.graphbuild.embed import StubEmbedder
from graft.graphstore import DictGraphSnapshot
from graft.retrieve import pins as cpins
from graft.retrieve.bm25 import BM25Channel
from graft.retrieve.dense import DenseChannel
from graft.retrieve.entity import entity_channel, match_entities
from graft.retrieve.expand import expand_channel
from graft.retrieve.fuse import assemble, fuse, minmax
from graft.retrieve.pool import (
    build_pool,
    eligible_nodes,
    node_atom_id,
    node_text,
    validate_edge_refs,
)
from graft.retrieve.recall import (
    HONESTY_STAMP,
    ceilings,
    channel_table,
    gold_nodes,
    recall_of,
    saturation,
    tier_a_gold,
    tier_b_gold,
)
from graft.retrieve.temporal import intervals_of, temporal_filter
from graft.schemas import (
    Assertion,
    AssertionFlags,
    CandidateAtom,
    Edge,
    Interval,
    Node,
    Obligations,
    SourceSpan,
    Turn,
)

DAY = 86400.0


@pytest.fixture
def snap() -> DictGraphSnapshot:
    """Two conversations, an eligible/quarantined pair, and a real interval.

    Deliberately not minimal.  The interval is populated (the pilot graph has
    none, so a fixture without one would leave the temporal *drop* path untested
    on every machine), the quarantined assertion gives criterion 2 something to
    be negative about, and ``c2`` exists so the conversation scope has a wrong
    answer available to return.
    """
    flags = AssertionFlags(asserted_by="user", entailed_by_span=True, entailed_score=0.99)
    turns = [
        Turn("t1", "c1", "s1", "user", "2023-05-01T10:00:00Z", "my weight is 70kg"),
        Turn("t2", "c1", "s2", "user", "2023-06-01T10:00:00Z", "my weight is 68kg"),
        Turn("t3", "c1", "s2", "user", "2023-06-01T10:05:00Z", "the yoga mat is thick"),
        Turn("t4", "c2", "s1", "user", "2023-05-01T10:00:00Z", "another user entirely"),
    ]
    spans = [
        SourceSpan("sp1", "t1", 0, 17),
        SourceSpan("sp2", "t2", 0, 17),
        SourceSpan("sp3", "t3", 0, 21),
        SourceSpan("sp4", "t4", 0, 21),
    ]
    assertions = [
        Assertion("a1", "value", "my weight is 70kg", ("sp1",), flags, "2023-05-01T10:00:00Z", "eligible"),
        Assertion("a2", "value", "my weight is 68kg", ("sp2",), flags, "2023-06-01T10:00:00Z", "eligible"),
        Assertion("a3", "claim", "the yoga mat is thick", ("sp3",), flags, "2023-06-01T10:05:00Z", "quarantined"),
        Assertion("a4", "claim", "another user entirely", ("sp4",), flags, "2023-05-01T10:00:00Z", "eligible"),
    ]
    nodes = [
        Node("n_v1", "Value", {"assertion_id": "a1", "value_type": "mass"}),
        Node("n_v2", "Value", {"assertion_id": "a2", "value_type": "mass"}),
        Node("n_q", "Claim", {"assertion_id": "a3"}),
        Node("n_other", "Claim", {"assertion_id": "a4"}),
        Node("n_e1", "Entity", {"name": "my weight", "aliases": ["body weight"], "conv_id": "c1"}),
        Node("n_e2", "Entity", {"name": "my weight", "aliases": [], "conv_id": "c2"}),
        Node("n_ti", "TimeInterval", {"start": 0.0, "end": 10 * DAY}),
    ]
    edges = [
        Edge("e1", "about_entity", "n_v1", "n_e1", "2023-05-01T10:00:00Z", ("sp1",)),
        Edge("e2", "about_entity", "n_v2", "n_e1", "2023-06-01T10:00:00Z", ("sp2",)),
        Edge("e3", "valid_during", "n_v1", "n_ti", "2023-05-01T10:00:00Z", ("sp1",)),
        Edge("e4", "supersedes", "n_v2", "n_v1", "2023-06-01T10:00:00Z", ("sp2",)),
        Edge("e5", "about_entity", "n_other", "n_e2", "2023-05-01T10:00:00Z", ("sp4",)),
        Edge("e6", "about_entity", "n_q", "n_e1", "2023-06-01T10:05:00Z", ("sp3",),
             t_invalid="2023-07-01T10:00:00Z"),
    ]
    return DictGraphSnapshot(1, nodes, edges, assertions, turns, spans)


# -- criteria 1, 2: the pool is sound --------------------------------------


def test_every_pool_is_closed_under_refs(snap):
    """Criterion 1.  Phase 1 proved the unconstructible-terminal rate to 0 on the
    strength of this; a Stage-C pool that broke it would revive the failure
    silently, since nothing downstream re-checks."""
    pool, _, _ = build_pool(snap, {"n_v1": 0.9, "n_v2": 0.7})
    for atom in pool:
        for ref in atom.refs:
            assert ref in pool, f"{atom.atom_id} references {ref}, absent from the pool"


def test_edge_atom_refs_are_its_target_edges_endpoints(snap):
    """The check ``CandidateAtom``'s docstring asks Phase 7 for **by name**.

    Without it an edge atom can satisfy closure — refs resolve, refs are selected
    — while denoting endpoints that are not its edge's own, and `H` would certify
    a proof whose structure does not match the graph it claims to read.
    """
    pool, _, _ = build_pool(snap, {"n_v1": 0.9, "n_v2": 0.7})
    validate_edge_refs(pool, snap)  # the real pool passes

    # And it is a real check: transpose one edge atom's refs and it must refuse.
    atoms = list(pool)
    edge_atom = next(a for a in atoms if a.kind == "edge")
    broken = [
        CandidateAtom(
            atom_id=a.atom_id,
            kind=a.kind,
            refs=tuple(reversed(a.refs)) if a is edge_atom else a.refs,
            target=a.target,
            label=a.label,
        )
        for a in atoms
    ]
    from graft.schemas import AtomPool

    with pytest.raises(ValueError, match="closure would hold while meaning"):
        validate_edge_refs(AtomPool(broken), snap)


def test_the_cap_counts_the_closed_set(snap):
    """Criterion 1.  ``pool_cap`` bounds what Stage D receives, which is the closed
    set — not the hit count.  A cap counted on hits would let 64 claims drag in a
    hundred structural atoms and blow the state-space bound the cap exists for."""
    pool, _, report = build_pool(snap, {"n_v1": 0.9, "n_v2": 0.7}, cap=3)
    assert len(pool) <= 3
    assert report["pool_size"] == len(pool)
    assert report["cap_skipped"] >= 1


def test_a_candidate_that_does_not_fit_is_skipped_not_terminal(snap):
    """Skip-and-continue, not break: a later, cheaper candidate may still fit.

    ``n_v1`` costs 5 atoms (itself, an entity, an interval and two edges);
    ``n_v2`` costs 3.  At cap 3 the higher-scoring ``n_v1`` cannot be admitted and
    the lower-scoring ``n_v2`` must be.
    """
    pool, _, report = build_pool(snap, {"n_v1": 0.9, "n_v2": 0.7}, cap=3)
    targets = {a.target for a in pool}
    assert "n_v2" in targets and "n_v1" not in targets
    assert report["hits_admitted"] == 1 and report["cap_skipped"] == 1


def test_only_eligible_assertions_enter_a_pool(snap):
    """Criterion 2, negative-tested.  Fix F9's boundary holds at retrieval too: an
    invented claim the support gate quarantined must not become retrievable
    evidence just because a channel scored it."""
    assert "n_q" not in eligible_nodes(snap, "c1")
    pool, _, _ = build_pool(snap, {"n_v1": 1.0, "n_q": 5.0})
    assert node_atom_id("n_q") not in pool
    assert all(a.target != "n_q" for a in pool)


def test_dead_edges_never_enter_a_pool(snap):
    """An invalidated edge is a retired fact.  Walking it back into the pool would
    undo the property Contribution 1 claims."""
    pool, _, _ = build_pool(snap, {"n_v1": 1.0, "n_v2": 1.0})
    assert all(a.target != "e6" for a in pool)


def test_the_conversation_scope_is_the_wrong_merge_guard(snap):
    """``c2`` holds an entity with the *same* name.  Phase 6 had to add this filter
    retroactively (plan §8 risk #1); Stage C applies it from the start."""
    assert eligible_nodes(snap, "c1") == ("n_v1", "n_v2")
    assert eligible_nodes(snap, "c2") == ("n_other",)


# -- criterion 3: determinism ----------------------------------------------


def test_two_runs_produce_identical_pools(snap):
    """Criterion 3.  Set iteration order is randomised per process, so anything
    unsorted here would differ between launches and the artefact would stop being
    comparable across machines for a reason unrelated to retrieval."""
    channels = {"bm25": {"n_v1": 2.0, "n_v2": 1.0}, "entity": {"n_v1": 1.0}}
    first = assemble(snap, channels)
    second = assemble(snap, channels)
    assert first[0].ids() == second[0].ids()
    assert first[1] == second[1]
    assert first[2] == second[2]


# -- criteria 8, 9, 10, 11: the channels behave as declared ----------------


def test_the_temporal_filter_never_runs_without_a_constraint(snap):
    """Criterion 8, direction one (G4)."""
    scored = {"n_v1": 1.0, "n_v2": 0.5}
    kept, report = temporal_filter(snap, scored, None)
    assert kept == scored
    assert report["applied"] is False


def test_the_temporal_filter_passes_atoms_without_intervals(snap):
    """Criterion 8, direction two.  **No ``valid_during`` edges exist on real data
    yet**, so a filter that dropped interval-less atoms would empty every pool and
    report 0.0 recall for a reason that is not retrieval."""
    assert intervals_of(snap, "n_v2") == ()
    kept, report = temporal_filter(snap, {"n_v2": 1.0}, Interval(start=100 * DAY, end=200 * DAY))
    assert kept == {"n_v2": 1.0}
    assert report["applied"] is True and report["dropped"] == 0


def test_the_temporal_filter_drops_a_provable_contradiction(snap):
    """The filter must still be able to *do* something, or fail-open is just off.

    ``n_v1`` is valid during [0, 10d); a question about [100d, 200d) provably
    excludes it.
    """
    assert len(intervals_of(snap, "n_v1")) == 1
    kept, report = temporal_filter(
        snap, {"n_v1": 1.0, "n_v2": 1.0}, Interval(start=100 * DAY, end=200 * DAY)
    )
    assert "n_v1" not in kept and "n_v2" in kept
    assert report["dropped"] == 1 and report["with_interval"] == 1


def test_one_overlapping_interval_is_enough(snap):
    """Drop only when *none* overlaps.  A claim valid across two periods must not
    be discarded for the period the question did not ask about."""
    kept, _ = temporal_filter(snap, {"n_v1": 1.0}, Interval(start=5 * DAY, end=200 * DAY))
    assert kept == {"n_v1": 1.0}


def test_the_entity_channel_is_empty_without_an_anchor(snap):
    """Criterion 9, decision 7.  A guessed anchor is a hallucinated scope one field
    over, so there is no keyword fallback — the channel goes quiet."""
    assert entity_channel(snap, Obligations(), "c1") == {}
    assert match_entities(snap, None, "c1") == ()
    assert match_entities(snap, "", "c1") == ()


def test_the_entity_channel_matches_aliases_through_the_shared_normalisation(snap):
    """Criterion 9.  Aliases are what D1 exists to populate, and the normalisation
    is the grounder's — one function, or Stage B and Stage C disagree about
    whether two surface forms are the same string."""
    assert match_entities(snap, "my weight", "c1") == ("n_e1",)
    assert match_entities(snap, "  MY   Weight ", "c1") == ("n_e1",)
    assert match_entities(snap, "body weight", "c1") == ("n_e1",)
    # and it stays inside the conversation
    assert match_entities(snap, "my weight", "c2") == ("n_e2",)
    hits = entity_channel(snap, Obligations(entity_anchor="my weight"), "c1")
    assert set(hits) == {"n_v1", "n_v2"}
    assert set(hits.values()) == {1.0}


def test_expansion_respects_the_hop_bound(snap):
    """Criterion 10.  Depth is bounded, and the bound is a declared constant."""
    hits, report = expand_channel(snap, ("n_e1",), max_hops=1)
    assert report["max_hops"] == 1
    assert set(hits) == {"n_v1", "n_v2"}
    assert set(hits.values()) == {1.0}


def test_expansion_reports_when_the_fan_out_cap_binds(snap):
    """Criterion 10.  A hop bound does not bound width; the fan-out cap does, and
    the binding count is in the artefact so a cap doing real work to the recall
    number is visible rather than inferred."""
    _, loose = expand_channel(snap, ("n_e1",), fan_out=32)
    assert loose["fan_out_binds"] == 0
    _, tight = expand_channel(snap, ("n_e1",), fan_out=1)
    assert tight["fan_out_binds"] >= 1


def test_expansion_keeps_the_most_recent_edge_when_the_cap_binds(snap):
    """Recency as the tie-break is [ANALYSIS] and is declared; this is what it
    means concretely.  ``n_e1``'s two live edges are e1 (May) and e2 (June); at
    ``fan_out=1`` the June one survives, so ``n_v2`` is reached and ``n_v1`` is
    not."""
    hits, _ = expand_channel(snap, ("n_e1",), max_hops=1, fan_out=1)
    assert set(hits) == {"n_v2"}


def test_expansion_walks_only_live_edges(snap):
    """The dead ``e6`` would otherwise reach the quarantined ``n_q``."""
    hits, _ = expand_channel(snap, ("n_e1",))
    assert "n_q" not in hits


def test_the_declared_fusion_arithmetic(snap):
    """Criterion 11.  min–max per channel, weights 1.0, max-union, id ties."""
    assert minmax({}) == {}
    assert minmax({"a": 7.0}) == {"a": 1.0}
    assert minmax({"a": 1.0, "b": 1.0}) == {"a": 1.0, "b": 1.0}
    assert minmax({"a": 2.0, "b": 4.0, "c": 6.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}

    fused, report = fuse({"bm25": {"x": 10.0, "y": 20.0}, "entity": {"x": 1.0}})
    # bm25 normalises to x=0.0, y=1.0; entity is flat so x=1.0. max wins for x.
    assert fused == {"x": 1.0, "y": 1.0}
    assert report["union_size"] == 2


def test_a_flat_channel_is_not_deleted_by_the_normalisation():
    """The degenerate ``max == min`` case maps to 1.0, and it has to.

    The entity channel scores every hit 1.0 by design, so it is *permanently*
    degenerate; mapping that to 0.0 would silently delete the one channel that
    knows about atoms no text query can reach.
    """
    fused, _ = fuse({"entity": {"a": 1.0, "b": 1.0}})
    assert fused == {"a": 1.0, "b": 1.0}


def test_the_channel_report_separates_wins_from_unique():
    """``wins`` alone is misleading and the instrument says so in code.

    Ties go to the alphabetically first channel, so a channel reaching exactly the
    same atoms at the same normalised score reads ``wins: 0``.  ``unique`` — atoms
    no other channel returned — is the count that answers "what would we lose by
    dropping this channel", which is what the G7 table is read for.
    """
    # Two channels reaching *exactly* the same atoms at the same score: the later
    # one reads wins 0 despite having found everything the first did.
    _, tied = fuse({"aaa": {"x": 1.0, "y": 1.0}, "zzz": {"x": 1.0, "y": 1.0}})
    assert tied["channels"]["aaa"]["wins"] == 2
    assert tied["channels"]["zzz"]["wins"] == 0
    assert tied["channels"]["zzz"]["unique"] == 0

    # ``unique`` is what separates "found nothing new" from "lost the tie-break".
    _, mixed = fuse({"aaa": {"x": 1.0, "y": 1.0}, "zzz": {"x": 1.0, "q": 1.0}})
    assert mixed["channels"]["zzz"]["unique"] == 1
    assert mixed["channels"]["aaa"]["unique"] == 1


# -- the text channels ------------------------------------------------------


def test_both_text_channels_index_the_same_string(snap):
    """If BM25 and the dense encoder saw different text, a per-channel recall
    difference would partly be an artefact of what each was shown."""
    bm = BM25Channel(snap, "c1")
    dn = DenseChannel(snap, StubEmbedder(dim=32), "c1")
    assert bm.node_ids == dn.node_ids
    assert node_text(snap, "n_v1") == "my weight is 70kg"
    assert node_text(snap, "n_e1") == ""  # an entity is support, never text-retrieved


def test_bm25_returns_only_lexical_matches(snap):
    """A zero BM25 score means "no shared term", not "weakly relevant" — it is the
    artefact of ``retrieve`` returning k rows regardless.  Dropping them keeps the
    channel's own minimum off documents it never matched."""
    bm = BM25Channel(snap, "c1")
    assert bm.query("weight") != {}
    assert bm.query("xylophone submarine") == {}
    assert bm.query("") == {}


def test_the_dense_channel_is_scoped_and_ordered(snap):
    dn = DenseChannel(snap, StubEmbedder(dim=32), "c1")
    hits = dn.query("what is my weight")
    assert set(hits) <= {"n_v1", "n_v2"}
    assert dn.query("") == {}


def test_channels_never_surface_a_quarantined_assertion(snap):
    """Fix F9 at every channel, not only at assembly."""
    bm = BM25Channel(snap, "c1")
    dn = DenseChannel(snap, StubEmbedder(dim=32), "c1")
    assert "n_q" not in bm.query("yoga mat thick")
    assert "n_q" not in dn.query("yoga mat thick")
    assert "n_q" not in entity_channel(snap, Obligations(entity_anchor="my weight"), "c1")


# -- criteria 4, 5, 6: recall is measured, honestly ------------------------


def test_tier_b_refuses_by_name_until_gate_0_signs():
    """Criterion 4.  A fallback to Tier A would put two different meanings behind
    one number in two different documents — the exact failure decision 8 names."""
    with pytest.raises(NotImplementedError, match="GATE0_CONTRACT"):
        tier_b_gold()


def test_tier_a_gold_is_the_closed_superset(snap):
    """Criterion 4.  Gold is built through the same ``build_pool`` the channels
    feed, so the two cannot disagree about what is pool-representable."""
    closed, reachable = tier_a_gold(snap, ["t1"], "c1")
    assert reachable == {node_atom_id("n_v1")}
    assert reachable <= closed
    # the closure brought the entity and interval in as support
    assert len(closed) > len(reachable)


def test_gold_takes_a_claim_whose_any_span_lands_in_a_gold_turn(snap):
    assert gold_nodes(snap, ["t1"], "c1") == ("n_v1",)
    assert gold_nodes(snap, ["t1", "t2"], "c1") == ("n_v1", "n_v2")
    assert gold_nodes(snap, [], "c1") == ()


def test_gold_never_includes_a_quarantined_assertion(snap):
    """``t3`` is a ``has_answer`` turn in this call, and its assertion is
    quarantined.  Gold that included it would make ceiling 3 unreachable by
    construction and blame retrieval for a Stage-A drop."""
    assert gold_nodes(snap, ["t3"], "c1") == ()


def test_empty_gold_scores_none_rather_than_one(snap):
    """A question with nothing to recall must not contribute a perfect score.

    On a 10-question pilot, scoring the unmeasurable as 1.0 is the difference
    between a number and an artefact.
    """
    assert recall_of([], [])["recall"] is None
    assert recall_of(["a"], [])["recall"] is None
    assert recall_of(["a"], ["a", "b"])["recall"] == 0.5


def test_the_channel_table_scores_channels_against_reachable_gold(snap):
    """Criterion 5.  No channel emits an edge atom — they arrive by closure — so
    scoring a channel against the closed set would charge it for atoms it
    structurally cannot return."""
    _, reachable = tier_a_gold(snap, ["t1", "t2"], "c1")
    table = channel_table(
        {"entity": {"n_v1": 1.0, "n_v2": 1.0}, "bm25": {"n_v1": 3.0}},
        reachable,
        latency_ms={"entity": 1.5},
    )
    assert table["entity"]["recall"] == 1.0
    assert table["bm25"]["recall"] == 0.5
    assert table["entity"]["latency_ms"] == 1.5
    assert table["bm25"]["latency_ms"] is None


def test_ceilings_are_computed_from_the_snapshot(snap):
    """Criterion 6.  Ceiling 3 is never reported alone: ceiling 1 already took 55%
    on the pilot, and a retrieval failure would otherwise be misread into a stage
    that did not cause it."""
    result = ceilings(snap, "c1")
    assert result["available"] is True
    assert result["assertions_total"] == 4
    assert result["assertions_eligible"] == 3
    assert result["ceiling_1_extraction"] == pytest.approx(0.75)
    assert 0.0 <= result["ceiling_2_graph"] <= 1.0


def test_the_saturation_flag_says_when_recall_meant_nothing(snap):
    """**The finding this fixture exists to keep.**

    Measured on the pilot graph 15 Aug 2026: every conversation holds 8–23
    eligible candidates against ``pool_cap = 64``, so the pool is the whole
    conversation and Tier-A recall is 1.000 by arithmetic.  Phase 4's G9 one stage
    later, and the reason ``DATASET_DECISION.md`` §5 argues for distractor
    sessions.
    """
    assert saturation(snap, "c1", cap=64)["exercised"] is False
    assert saturation(snap, "c1", cap=1)["exercised"] is True


def test_the_honesty_stamp_names_all_three_g9_bounds():
    assert set(HONESTY_STAMP) == {"graph", "questions", "ceilings", "tier"}
    assert "stand-in" in HONESTY_STAMP["graph"]
    assert "Tier B refuses" in HONESTY_STAMP["tier"]


# -- criteria 13, 15: the frozen surface ------------------------------------


def test_the_scorer_is_declared_but_not_built():
    """Criterion 13, in the state Gate 0 leaves it.  The interface is fixed — the
    other modules are written against it — and the model is not built, because it
    trains and the contract is unsigned."""
    assert cpins.SCORER["built"] is False
    assert cpins.SCORER["max_params"] == 8_000_000
    assert cpins.SCORER["training_signal"] == "distant_answer_session_ids"
    assert cpins.SCORER["optional_in_fusion"] is True
    assert not (Path(__file__).parent.parent / "retrieve" / "scorer.py").exists()


def test_the_five_channel_stack_runs_without_the_scorer(snap):
    """G6's requirement, as a test: the scorer's absence is a legal configuration,
    not a crash."""
    channels = {
        "bm25": BM25Channel(snap, "c1").query("weight"),
        "dense": DenseChannel(snap, StubEmbedder(dim=32), "c1").query("weight"),
        "entity": entity_channel(snap, Obligations(entity_anchor="my weight"), "c1"),
        "expand": expand_channel(snap, match_entities(snap, "my weight", "c1"))[0],
    }
    pool, scores, report = assemble(snap, channels)
    assert len(pool) > 0
    assert set(scores) == set(pool.ids())
    assert set(report) == {"fusion", "temporal", "pool", "union_pre_cap", "after_temporal"}


def test_the_stage_c_fingerprint_is_stable_and_covers_the_arithmetic():
    """Criterion 15.  Two machines fusing with different arithmetic produce recall
    numbers that are not comparable, and nothing else in the run would say so."""
    first = cpins.stage_c_fingerprint()
    assert first == cpins.stage_c_fingerprint()
    frozen = cpins.frozen_values()
    assert frozen["fusion"] == "max"
    assert frozen["channel_weights"] == {name: 1.0 for name in cpins.CHANNELS}
    assert frozen["embedder"]["model_id"] == "BAAI/bge-small-en-v1.5"
    # pool_cap belongs to the config tree and must NOT be duplicated here
    assert "pool_cap" not in frozen


def test_the_shared_embedder_pin_is_not_restated():
    """Decision 2 as an identity check rather than a promise: Stage C's pin *is*
    Stage B's object, so they cannot drift apart."""
    from graft.graphbuild.pins import EMBEDDER as STAGE_B_EMBEDDER

    assert cpins.EMBEDDER is STAGE_B_EMBEDDER


# -- criterion 7: the gold quarantine, as source ---------------------------

GOLD_FIELDS = ("has_answer", "answer_session_ids", "evidence_session_ids")


@pytest.mark.parametrize(
    "path",
    sorted(
        p
        for p in (Path(__file__).parent.parent / "retrieve").rglob("*.py")
        if p.name != "recall.py"
    ),
    ids=lambda p: p.name,
)
def test_no_channel_module_mentions_a_gold_field(path):
    """Criterion 7.  ``recall.py`` is the sidecar boundary and is exempt by name;
    every other module in the package must be unable to see a gold field.

    A channel that could read ``has_answer`` would retrieve the answer because it
    was told which turn holds it, and ceiling 3 would be measuring the instrument.
    Asserted over the source rather than promised in a docstring.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    offending = sorted(n for n in names if n in GOLD_FIELDS)
    assert not offending, f"{path.name} references gold field(s) {offending}"
