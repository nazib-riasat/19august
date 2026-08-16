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
import math
from pathlib import Path

import pytest

from graft.config.schema import Config
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
    ASSERTION_BACKED_NTYPES,
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

#: Turn ids follow ``graft.ingest.corpus.turn_id_for`` — ``lme_s/{question}/{session}/{ix}``
#: — rather than toy strings.  The distant-label path derives turn ids from a
#: corpus instance through that same function, so a fixture with invented ids
#: would make the label test unwritable and, worse, would let it pass vacuously
#: (every atom scoring 0 because nothing joined).
T_W70 = "lme_s/c1/s1/0"   # session s1
T_W68 = "lme_s/c1/s2/0"   # session s2
T_YOGA = "lme_s/c1/s2/1"  # session s2, quarantined assertion
T_ISO = "lme_s/c1/s3/0"   # session s3, an assertion with no committed relations
T_OTHER = "lme_s/c2/s1/0"  # a different conversation


@pytest.fixture
def snap() -> DictGraphSnapshot:
    """Two conversations, an eligible/quarantined pair, and a real interval.

    Deliberately not minimal.  The interval is populated (the pilot graph has
    none, so a fixture without one would leave the temporal *drop* path untested
    on every machine), the quarantined assertion gives criterion 2 something to
    be negative about, ``c2`` exists so the conversation scope has a wrong answer
    available to return, and ``n_iso`` is an eligible node with **no edges at
    all** — a legal pool shape (nodes reference nothing) that the scorer's
    forward pass has to handle without a special case at the call site.
    """
    flags = AssertionFlags(asserted_by="user", entailed_by_span=True, entailed_score=0.99)
    turns = [
        Turn(T_W70, "c1", "s1", "user", "2023-05-01T10:00:00Z", "my weight is 70kg"),
        Turn(T_W68, "c1", "s2", "user", "2023-06-01T10:00:00Z", "my weight is 68kg"),
        Turn(T_YOGA, "c1", "s2", "user", "2023-06-01T10:05:00Z", "the yoga mat is thick"),
        Turn(T_ISO, "c1", "s3", "user", "2023-06-02T10:00:00Z", "I prefer morning runs"),
        Turn(T_OTHER, "c2", "s1", "user", "2023-05-01T10:00:00Z", "another user entirely"),
    ]
    spans = [
        SourceSpan("sp1", T_W70, 0, 17),
        SourceSpan("sp2", T_W68, 0, 17),
        SourceSpan("sp3", T_YOGA, 0, 21),
        SourceSpan("sp5", T_ISO, 0, 21),
        SourceSpan("sp4", T_OTHER, 0, 21),
    ]
    assertions = [
        Assertion("a1", "value", "my weight is 70kg", ("sp1",), flags, "2023-05-01T10:00:00Z", "eligible"),
        Assertion("a2", "value", "my weight is 68kg", ("sp2",), flags, "2023-06-01T10:00:00Z", "eligible"),
        Assertion("a3", "claim", "the yoga mat is thick", ("sp3",), flags, "2023-06-01T10:05:00Z", "quarantined"),
        Assertion("a5", "claim", "I prefer morning runs", ("sp5",), flags, "2023-06-02T10:00:00Z", "eligible"),
        Assertion("a4", "claim", "another user entirely", ("sp4",), flags, "2023-05-01T10:00:00Z", "eligible"),
    ]
    nodes = [
        Node("n_v1", "Value", {"assertion_id": "a1", "value_type": "mass"}),
        Node("n_v2", "Value", {"assertion_id": "a2", "value_type": "mass"}),
        Node("n_q", "Claim", {"assertion_id": "a3"}),
        Node("n_iso", "Claim", {"assertion_id": "a5"}),
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
    assert eligible_nodes(snap, "c1") == ("n_iso", "n_v1", "n_v2")
    assert eligible_nodes(snap, "c2") == ("n_other",)


def test_assembly_refuses_another_conversations_node(snap):
    """The scope boundary at assembly (15 Aug 2026 audit).  The walking channels
    follow edges, and an edge's far endpoint can belong to another conversation —
    another user's claim entering this question's pool is plan §8's risk #1.
    Like fix F9's eligibility boundary, scope is enforced where pools are built,
    not only in the callers."""
    from graft.retrieve.pool import build_pool

    pool, _, report = build_pool(snap, {"n_v1": 1.0, "n_other": 5.0}, conv_id="c1")
    assert all(a.target != "n_other" for a in pool)
    assert report["hits_refused_ineligible"] == 1


def test_uncapped_pool_admits_everything_and_says_so(snap):
    """``uncapped_pool`` replaced three guessed finite caps (15 Aug 2026 audit).
    Its contract: every eligible offered node is admitted, ``cap_skipped`` is 0
    by monotonicity, and the result equals ``build_pool`` at any
    sufficiently-large cap — one mapping code path, not two."""
    from graft.retrieve.pool import build_pool, uncapped_pool

    offered = {"n_v1": 0.9, "n_v2": 0.7, "n_iso": 0.1}
    pool, scores, report = uncapped_pool(snap, offered, conv_id="c1")
    assert report["cap_skipped"] == 0
    assert report["uncapped"] is True
    assert report["hits_admitted"] == 3
    assert len(pool) == 9  # 5 node atoms + 4 live edges among them
    huge, _, _ = build_pool(snap, offered, cap=999, conv_id="c1")
    assert pool.ids() == huge.ids()
    assert set(scores) == set(pool.ids())


def test_required_sets_split_gold_by_kind(snap):
    """Required-node and required-edge Recall@k are the two Stage-C metrics the
    plan names separately (§3.3, §6.4); required-edge recall had no
    implementation at all before the 15 Aug 2026 audit.  Atom ids are opaque
    hashes, so the kind split must be made where the pool object exists."""
    from graft.retrieve.recall import required_sets

    sets = required_sets(snap, [T_W70, T_W68], "c1")
    assert sets["reachable"] == {node_atom_id("n_v1"), node_atom_id("n_v2")}
    assert sets["node_atoms"] | sets["edge_atoms"] == sets["closed"]
    assert sets["node_atoms"] & sets["edge_atoms"] == frozenset()
    assert len(sets["node_atoms"]) == 4  # v1, v2, entity, interval
    assert len(sets["edge_atoms"]) == 4  # e1, e2, e3, e4
    empty = required_sets(snap, [], "c1")
    assert all(v == frozenset() for v in empty.values())


# -- criterion 3: determinism ----------------------------------------------


def test_two_runs_produce_identical_pools(snap):
    """Criterion 3 (as amended 15 Aug 2026).  Set iteration order is randomised
    per process, so anything unsorted here would differ between launches and the
    artefact would stop being comparable across machines for a reason unrelated
    to retrieval.  Reports are compared through ``deterministic_view`` because
    they now carry wall-clock ``timings_ms`` (criterion 14), which two runs can
    never agree on — the determinism claim is over everything else."""
    from graft.runtime import deterministic_view

    channels = {"bm25": {"n_v1": 2.0, "n_v2": 1.0}, "entity": {"n_v1": 1.0}}
    first = assemble(snap, channels)
    second = assemble(snap, channels)
    assert first[0].ids() == second[0].ids()
    assert first[1] == second[1]
    assert deterministic_view(first[2]) == deterministic_view(second[2])


def test_deterministic_view_strips_exactly_the_volatile_keys():
    """The determinism digest's exclusion list, asserted rather than promised.

    If a volatile key survived, two identical runs would digest differently and
    criterion 3 would fail on a wall clock; if a *non*-volatile key were
    stripped, a real nondeterminism could hide behind the exclusion.
    """
    from graft.runtime import VOLATILE_KEYS, deterministic_view

    nested = {
        "recall": 1.0,
        "latency": {"p50": 3.2},
        "inner": [{"timings_ms": {"fuse": 1.0}, "pool_size": 9, "latency_ms": 4.2}],
        "environment": {"utc_started": "now", "hostname": "x"},
    }
    view = deterministic_view(nested)
    assert view == {"recall": 1.0, "inner": [{"pool_size": 9}]}
    assert set(VOLATILE_KEYS) == {"latency", "latency_ms", "timings_ms", "environment"}


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


def test_a_channel_weight_actually_scales_its_scores():
    """The weights were applied but only ever exercised at 1.0 (15 Aug 2026
    audit) — a knob no test had turned is a knob that can silently stop
    existing."""
    from graft.retrieve.fuse import weighted_scores

    weighted = weighted_scores(
        {"bm25": {"x": 2.0, "y": 4.0}}, weights={"bm25": 0.5}
    )
    assert weighted["bm25"] == {"x": 0.0, "y": 0.5}
    fused, _ = fuse(
        {"bm25": {"x": 2.0, "y": 4.0}, "entity": {"x": 1.0}},
        weights={"bm25": 0.5, "entity": 1.0},
    )
    assert fused == {"x": 1.0, "y": 0.5}  # entity's 1.0 beats bm25's halved 0.0


def test_assemble_reports_per_channel_scores_per_atom(snap):
    """**The Phase-8/9 handoff contract** (architecture §9.1; 15 Aug 2026 audit
    blocker).  Phase 8's features are "max/mean channel scores" and Phase 9's
    ``AtomFeaturizer`` carries "retrieval channel scores" per atom — before
    ``channel_scores`` existed, fusion computed the per-channel values and
    immediately collapsed them to one scalar, so neither contract was
    implementable from the handoff."""
    channels = {"bm25": {"n_v1": 2.0, "n_v2": 4.0}, "entity": {"n_v1": 1.0}}
    pool, atom_scores, report = assemble(snap, channels)
    per_atom = report["channel_scores"]
    assert set(per_atom) == set(pool.ids())
    v1 = per_atom[node_atom_id("n_v1")]
    v2 = per_atom[node_atom_id("n_v2")]
    assert v1 == {"bm25": 0.0, "entity": 1.0}  # min-max floor + flat channel
    assert v2 == {"bm25": 1.0}  # absence of a channel key means it scored 0.0
    assert atom_scores[node_atom_id("n_v1")] == 1.0  # fused max over the row
    # structural and edge atoms: no channel can return them -> empty maps
    structural = [a for a in pool if a.kind != "node" or a.target not in ("n_v1", "n_v2")]
    assert structural and all(per_atom[a.atom_id] == {} for a in structural)


def test_assemble_applies_the_temporal_filter_before_closure(snap):
    """The fuse → filter → close order, end to end (15 Aug 2026 audit gap: the
    filter was only ever tested standalone).  ``n_v1`` is valid during [0, 10d);
    a question about [100d, 200d) provably excludes it, and the pool must come
    out closed *without* it — filtering an already-closed pool would instead
    strand the edge atoms that reference the removed node."""
    channels = {"bm25": {"n_v1": 2.0, "n_v2": 1.0}}
    pool, _, report = assemble(
        snap, channels, constraint=Interval(start=100 * DAY, end=200 * DAY)
    )
    assert report["temporal"]["applied"] is True
    assert report["temporal"]["dropped"] == 1
    assert all(a.target != "n_v1" for a in pool)
    for atom in pool:  # still closed by construction
        for ref in atom.refs:
            assert ref in pool


def test_a_scorer_hit_enters_the_pool_the_five_channels_never_returned(snap):
    """**The sixth channel is a union member, not a re-ranker** (15 Aug 2026
    audit blocker).  The first wiring scored the already-capped five-channel
    pool, so the scorer could never surface an atom the cheap channels missed.
    Under the corrected wiring its scores fuse like any channel's: here the
    five-channel side never returned ``n_iso`` at all, and the scorer's score
    alone admits it."""
    five = {"bm25": {"n_v1": 2.0, "n_v2": 1.0}}
    without, _, _ = assemble(snap, five, cap=3)
    assert all(a.target != "n_iso" for a in without)

    six = dict(five, **{cpins.SCORER_CHANNEL: {"n_iso": 0.7}})
    with_scorer, _, _ = assemble(snap, six, cap=3)
    assert node_atom_id("n_iso") in with_scorer


def test_support_atoms_count_by_membership_not_score(snap):
    """Regression for PHASE7_DECISIONS §4.2, which shipped without a test
    (15 Aug 2026 audit).  min–max puts a channel's lowest genuine hit at exactly
    0.0, so counting support atoms by ``score == 0.0`` reported real hits as
    support — most often on pools where one channel dominates, the case the G7
    table is read to detect."""
    pool, _, report = build_pool(snap, {"n_v1": 0.0, "n_v2": 1.0})
    # both offered nodes are hits despite the 0.0 score; support = the rest
    hit_atoms = {node_atom_id("n_v1"), node_atom_id("n_v2")}
    assert report["support_atoms"] == len(pool) - len(hit_atoms)


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
    # Scoped to c1's eligible nodes and to nothing else: never the quarantined
    # ``n_q``, never the other conversation's ``n_other``.
    assert set(hits) <= {"n_v1", "n_v2", "n_iso"}
    assert dn.query("") == {}


def test_channels_never_surface_a_quarantined_assertion(snap):
    """Fix F9 at every channel, not only at assembly."""
    bm = BM25Channel(snap, "c1")
    dn = DenseChannel(snap, StubEmbedder(dim=32), "c1")
    assert "n_q" not in bm.query("yoga mat thick")
    assert "n_q" not in dn.query("yoga mat thick")
    assert "n_q" not in entity_channel(snap, Obligations(entity_anchor="my weight"), "c1")


def test_the_walking_channels_filter_eligibility_even_over_live_edges():
    """**The non-vacuous version of the test above** (15 Aug 2026 audit).  The
    fixture's only edge to the quarantined ``n_q`` is dead, so liveness alone
    excluded it there and the entity/expand assertions were tautologies.  Here
    the edge is **live** — a graph state Stage B can produce, since edge
    commitment and assertion quarantine are separate decisions — and the walking
    channels must still refuse to emit the quarantined node as a hit."""
    flags = AssertionFlags(asserted_by="user", entailed_by_span=True, entailed_score=0.99)
    live_snap = DictGraphSnapshot(
        1,
        [
            Node("n_ok", "Claim", {"assertion_id": "b1"}),
            Node("n_bad", "Claim", {"assertion_id": "b2"}),
            Node("n_ent", "Entity", {"name": "my weight", "aliases": [], "conv_id": "c1"}),
        ],
        [
            Edge("le1", "about_entity", "n_ok", "n_ent", "2023-05-01T10:00:00Z", ("sq1",)),
            Edge("le2", "about_entity", "n_bad", "n_ent", "2023-05-01T10:01:00Z", ("sq2",)),  # LIVE
        ],
        [
            Assertion("b1", "claim", "fine claim", ("sq1",), flags, "2023-05-01T10:00:00Z", "eligible"),
            Assertion("b2", "claim", "invented claim", ("sq2",), flags, "2023-05-01T10:01:00Z", "quarantined"),
        ],
        [
            Turn("lme_s/c1/s1/0", "c1", "s1", "user", "2023-05-01T10:00:00Z", "fine claim and more"),
        ],
        [
            SourceSpan("sq1", "lme_s/c1/s1/0", 0, 10),
            SourceSpan("sq2", "lme_s/c1/s1/0", 0, 10),
        ],
    )
    hits = entity_channel(live_snap, Obligations(entity_anchor="my weight"), "c1")
    assert "n_bad" not in hits and "n_ok" in hits
    walked, _ = expand_channel(live_snap, ("n_ent",), conv_id="c1")
    assert "n_bad" not in walked and "n_ok" in walked


# -- criteria 4, 5, 6: recall is measured, honestly ------------------------


def test_tier_b_is_a_subset_of_tier_a_and_still_satisfies_H(snap):
    """Criterion 4, in the state the 15 Aug Gate-0 signature leaves it.

    The two defining properties of item 3.2's minimisation: the result is
    contained in the Tier-A closed set it was minimised from, and it is still a
    valid proof.  A minimisation that broke either would be silently producing a
    gold set that no correct method could match.
    """
    from graft.core.checker import H

    obligations = Obligations(entity_anchor="my weight")
    closed, _ = tier_a_gold(snap, [T_W70], "c1")
    minimal, reachable, report = tier_b_gold(snap, [T_W70], obligations, "c1")

    # Assert, never skip (15 Aug 2026 audit): this is criterion 4's central
    # positive test, and a regression that invalidated the fixture's Tier-A
    # superset would have silently converted every assertion below into a skip.
    assert report["status"] == "ok", f"Tier-A superset invalid on the fixture: {report}"
    assert minimal <= closed
    assert reachable <= minimal
    assert len(minimal) <= len(closed)

    pool, _, _ = build_pool(snap, {"n_v1": 1.0}, cap=999)
    assert H(minimal, obligations, snap, pool, Config(), ledger=None).ok


def test_tier_b_returns_a_wider_tuple_than_tier_a(snap):
    """The type is the guard now that the refusal is gone: Tier A gives 2, Tier B
    gives 3, so a caller cannot silently swap one meaning for the other."""
    assert len(tier_a_gold(snap, [T_W70], "c1")) == 2
    assert len(tier_b_gold(snap, [T_W70], Obligations(entity_anchor="my weight"), "c1")) == 3


def test_tier_b_refuses_to_invent_a_proof_when_the_superset_is_invalid(snap):
    """`H` failing on the ``has_answer``-derived set means no valid proof exists
    inside it.  Returning something anyway would make the recall denominator "the
    answer we could not find", which flatters every method scored against it.

    Forced through **scope**, which is a real Stage-B failure rather than a
    contrived one: obligations scoped to a conversation the gold atoms do not
    belong to.  (``max_atoms`` used to force this and deliberately no longer can —
    see the size-exemption test below.)
    """
    wrong_scope = Obligations(entity_anchor="my weight", scope=("c2",))
    minimal, reachable, report = tier_b_gold(snap, [T_W70], wrong_scope, "c1")
    assert minimal == frozenset() and reachable == frozenset()
    assert report["status"] == "tier_a_superset_invalid"
    assert "scope" in report["violations"]


def test_max_atoms_does_not_bind_a_gold_set(snap):
    """Item 3.2 as amended 15 Aug 2026 — the size exemption.

    ``max_atoms`` bounds what a *candidate* set may select (Stage D's action
    budget).  A gold set is not a candidate set, and Tier A is capped at
    ``pool_cap`` = 64 precisely so gold can exceed what one pool holds.  Before
    the exemption, **every question with more than 16 gold atoms had no Tier-B
    gold at all** — thinning coverage exactly on the questions carrying the most
    evidence.  The pilot's largest was 9 so it never fired there; scope c reaches
    it.
    """
    minimal, _reachable, report = tier_b_gold(
        snap, [T_W70, T_W68], Obligations(entity_anchor="my weight"), "c1",
        config=Config(max_atoms=1),
    )
    assert report["status"] == "ok"
    assert len(minimal) > 1  # would be impossible if max_atoms bound the gold set


def test_tier_b_on_empty_gold_is_empty_not_an_error(snap):
    minimal, reachable, report = tier_b_gold(snap, [], Obligations(), "c1")
    assert minimal == frozenset() and reachable == frozenset()
    assert report["status"] == "empty_tier_a"


def test_minimisation_runs_to_a_fixpoint_not_one_sweep(snap):
    """Removing an edge atom can make its endpoint removable in a *later* pass —
    sub-check 8 requires refs to be selected — so a single sweep would stop early
    and report a set that is not actually irreducible."""
    from graft.retrieve.recall import minimise

    obligations = Obligations(entity_anchor="my weight")
    closed, _ = tier_a_gold(snap, [T_W70, T_W68], "c1")
    nodes = gold_nodes(snap, [T_W70, T_W68], "c1")
    pool, _, _ = build_pool(snap, {n: 1.0 for n in nodes}, cap=999)
    minimal, report = minimise(closed, pool, obligations, snap, Config())

    assert report["passes"] >= 1
    assert minimal <= closed
    assert report["kept"] == len(minimal)
    assert report["minimality"].startswith("local")
    # Irreducible **over structural atoms** — the scope the amendment restricts
    # removal to. Evidence atoms are removable in `H`'s eyes and are kept anyway,
    # which is the whole point, so they are not part of the irreducibility claim.
    from graft.core.checker import H

    for atom_id in sorted(minimal):
        atom = pool[atom_id]
        structural = not (
            atom.kind == "node" and snap.ntype(atom.target) in ASSERTION_BACKED_NTYPES
        )
        if structural:
            assert not H(minimal - {atom_id}, obligations, snap, pool, Config(), ledger=None).ok


def test_minimisation_keeps_evidence_and_drops_scaffolding(snap):
    """**The defect this ordering exists to prevent, as a regression test.**

    A bare ``Entity`` or ``TimeInterval`` node atom alone satisfies `H` — only
    ``size`` rejects the empty set — so an id-ordered minimisation returns a gold
    "proof" that asserts nothing. Measured on this fixture before the fix: the
    survivor was ``TimeInterval:n_ti``.
    """
    from graft.retrieve.recall import minimise

    obligations = Obligations(entity_anchor="my weight")
    closed, _ = tier_a_gold(snap, [T_W70, T_W68], "c1")
    nodes = gold_nodes(snap, [T_W70, T_W68], "c1")
    pool, _, _ = build_pool(snap, {n: 1.0 for n in nodes}, cap=999)
    minimal, report = minimise(closed, pool, obligations, snap, Config())

    assert report["degenerate"] is False
    assert report["evidence_atoms"] >= 1
    survivors = {(pool[a].kind, pool[a].label) for a in minimal}
    assert all(label not in ("Entity", "TimeInterval") for _kind, label in survivors)


def test_minimisation_never_drops_evidence(snap):
    """Item 3.2 as amended 15 Aug 2026 — removal is restricted to structural atoms.

    **This is the regression test for the degeneracy the amendment fixed.**
    Unrestricted `H`-minimisation collapsed all 9 pilot questions with valid gold
    to a single atom and lost evidence on 5 of them, because `H` is formal
    validity only and cannot say "this question needs *both* claims".
    """
    from graft.retrieve.recall import minimise

    obligations = Obligations(entity_anchor="my weight")
    closed, _ = tier_a_gold(snap, [T_W70, T_W68], "c1")
    nodes = gold_nodes(snap, [T_W70, T_W68], "c1")
    pool, _, _ = build_pool(snap, {n: 1.0 for n in nodes}, cap=999)
    minimal, report = minimise(closed, pool, obligations, snap, Config())

    assert report["evidence_offered"] == 2
    assert report["evidence_atoms"] == 2
    assert report["evidence_dropped"] == 0
    assert report["under_constrained"] is False
    assert report["degenerate"] is False
    # and it is still a real minimisation: scaffolding was removed
    assert report["removed"] > 0
    assert len(minimal) < len(closed)


def test_minimisation_is_deterministic(snap):
    """Greedy removal reaches *a* locally minimal set, not *the* minimum one, so
    the order is what makes recall reproducible.  Sorted-id order is the choice;
    this is the test that it is actually applied."""
    from graft.retrieve.recall import minimise

    obligations = Obligations(entity_anchor="my weight")
    closed, _ = tier_a_gold(snap, [T_W70, T_W68], "c1")
    nodes = gold_nodes(snap, [T_W70, T_W68], "c1")
    pool, _, _ = build_pool(snap, {n: 1.0 for n in nodes}, cap=999)
    first, _ = minimise(closed, pool, obligations, snap, Config())
    second, _ = minimise(closed, pool, obligations, snap, Config())
    assert first == second


def test_tier_a_gold_is_the_closed_superset(snap):
    """Criterion 4.  Gold is built through the same ``build_pool`` the channels
    feed, so the two cannot disagree about what is pool-representable."""
    closed, reachable = tier_a_gold(snap, [T_W70], "c1")
    assert reachable == {node_atom_id("n_v1")}
    assert reachable <= closed
    # the closure brought the entity and interval in as support
    assert len(closed) > len(reachable)


def test_gold_takes_a_claim_whose_any_span_lands_in_a_gold_turn(snap):
    assert gold_nodes(snap, [T_W70], "c1") == ("n_v1",)
    assert gold_nodes(snap, [T_W70, T_W68], "c1") == ("n_v1", "n_v2")
    assert gold_nodes(snap, [], "c1") == ()


def test_has_answer_turns_reads_the_turn_level_gold():
    """The function that defines Tier-A gold had no test at all (15 Aug 2026
    audit) — its sibling ``evidence_turns`` was tested while the evaluation-side
    derivation was not."""
    from graft.retrieve.recall import has_answer_turns

    instance = {
        "question_id": "c1",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [{"has_answer": True}, {}],
            [{}, {"has_answer": True}],
        ],
    }
    assert has_answer_turns(instance) == frozenset({"lme_s/c1/s1/0", "lme_s/c1/s2/1"})
    assert has_answer_turns({"question_id": "c1", "haystack_session_ids": [], "haystack_sessions": []}) == frozenset()


def test_gold_never_includes_a_quarantined_assertion(snap):
    """``t3`` is a ``has_answer`` turn in this call, and its assertion is
    quarantined.  Gold that included it would make ceiling 3 unreachable by
    construction and blame retrieval for a Stage-A drop."""
    assert gold_nodes(snap, [T_YOGA], "c1") == ()


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
    _, reachable = tier_a_gold(snap, [T_W70, T_W68], "c1")
    table = channel_table(
        {"entity": {"n_v1": 1.0, "n_v2": 1.0}, "bm25": {"n_v1": 3.0}},
        reachable,
        latency_ms={"entity": 1.5},
    )
    assert table["entity"]["recall"] == 1.0
    assert table["bm25"]["recall"] == 0.5
    assert table["entity"]["latency_ms"] == 1.5
    assert table["bm25"]["latency_ms"] is None
    # per channel the miss *list* is a count -- the full id list would otherwise
    # be serialised once per channel per question at scope c
    assert "missed" not in table["bm25"]
    assert table["bm25"]["missed_n"] == 1
    assert table["entity"]["missed_n"] == 0


def test_ceilings_are_computed_from_the_snapshot(snap):
    """Criterion 6.  Ceiling 3 is never reported alone: ceiling 1 already took 55%
    on the pilot, and a retrieval failure would otherwise be misread into a stage
    that did not cause it."""
    result = ceilings(snap, "c1")
    assert result["available"] is True
    assert result["assertions_total"] == 5
    assert result["assertions_eligible"] == 4
    assert result["ceiling_1_extraction"] == pytest.approx(0.8)
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


def test_saturation_is_measured_in_closed_atom_units(snap):
    """**The units regression** (15 Aug 2026 audit).  G8 applies ``pool_cap`` to
    the closed atom set; the first ``saturation()`` compared eligible *node*
    counts against it.  On this fixture c1 has 3 candidate nodes closing to 9
    atoms — at ``cap = 8`` the cap genuinely binds (9 > 8) while the node count
    (3 <= 8) would have read "unexercised", which is wrong in exactly the regime
    the flag exists for.
    """
    row = saturation(snap, "c1", cap=8)
    assert row["candidates_in_scope"] == 3
    assert row["closed_atoms_in_scope"] == 9
    assert row["exercised"] is True  # the node-count comparison said False here


def test_the_honesty_stamp_names_the_g9_bounds_and_the_tier_semantics():
    """Three G9 bounds plus the tier key.  The tier wording is pinned *forward*:
    an earlier version of this test pinned the stale 'Tier B refuses' text after
    the Gate-0 signature had made it false, so the test was preserving a
    falsehood in every artefact (15 Aug 2026 audit)."""
    assert set(HONESTY_STAMP) == {"graph", "questions", "ceilings", "tier"}
    assert "stand-in" in HONESTY_STAMP["graph"]
    assert "refuses" not in HONESTY_STAMP["tier"]
    assert "amended 15 Aug 2026" in HONESTY_STAMP["tier"]
    assert "conservative" in HONESTY_STAMP["tier"]


# -- criteria 13, 15: the frozen surface ------------------------------------


def test_the_scorer_is_declared_and_now_built():
    """Criterion 13's declared half.  Built 15 Aug 2026, once Gate 0 signed."""
    assert cpins.SCORER["built"] is True
    assert cpins.SCORER["max_params"] == 8_000_000
    assert cpins.SCORER["passes"] == 1
    # Depth is configuration identity: a hard-coded constructor default sat
    # outside the fingerprint, so two differently-deep scorers shared one
    # Stage-C identity (15 Aug 2026 audit).
    assert cpins.SCORER["layers"] == 2
    assert cpins.SCORER["query_conditioned"] is True
    assert cpins.SCORER["training_signal"] == "distant_answer_session_ids"
    assert cpins.SCORER["optional_in_fusion"] is True
    assert (Path(__file__).parent.parent / "retrieve" / "scorer.py").exists()


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
    assert set(report) == {
        "fusion",
        "temporal",
        "pool",
        "union_pre_cap",
        "after_temporal",
        "channel_scores",
        "raw_channel_scores",
        "timings_ms",
    }
    assert set(report["timings_ms"]) == {"fuse", "temporal", "pool"}


def test_the_stage_c_fingerprint_is_stable_and_covers_the_arithmetic():
    """Criterion 15.  Two machines fusing with different arithmetic produce recall
    numbers that are not comparable, and nothing else in the run would say so."""
    first = cpins.stage_c_fingerprint()
    assert first == cpins.stage_c_fingerprint()
    frozen = cpins.frozen_values()
    assert frozen["fusion"] == "max"
    # every training-free channel, *and* the learned one: the scorer's weight is
    # listed explicitly so the one channel that learns is not the only one whose
    # weight sits outside configuration identity.
    assert frozen["channel_weights"] == {
        **{name: 1.0 for name in cpins.CHANNELS},
        cpins.SCORER_CHANNEL: 1.0,
    }
    assert cpins.SCORER_CHANNEL not in cpins.CHANNELS
    assert frozen["embedder"]["model_id"] == "BAAI/bge-small-en-v1.5"
    # pool_cap belongs to the config tree and must NOT be duplicated here
    assert "pool_cap" not in frozen


def test_the_shared_embedder_pin_is_not_restated():
    """Decision 2 as an identity check rather than a promise: Stage C's pin *is*
    Stage B's object, so they cannot drift apart."""
    from graft.graphbuild.pins import EMBEDDER as STAGE_B_EMBEDDER

    assert cpins.EMBEDDER is STAGE_B_EMBEDDER


# -- P7.8: the scorer ------------------------------------------------------

torch = pytest.importorskip("torch", reason="the scorer is the only ML piece of Stage C")


def _prepared(snap, pool, question="what is my weight", labels=None):
    """One training example in the shape ``train_scorer`` consumes."""
    from graft.retrieve.scorer import atom_features, pool_adjacency

    embedder = StubEmbedder(dim=384)
    features, ids = atom_features(pool, snap, embedder)
    edge_index, edge_rel = pool_adjacency(pool, ids)
    return {
        "features": features,
        "edge_index": edge_index,
        "edge_rel": edge_rel,
        "query": embedder.embed([question])[0],
        "labels": labels if labels is not None else [1.0] + [0.0] * (len(ids) - 1),
    }


def test_the_scorer_is_within_the_eight_million_cap(snap):
    """Criterion 13.  8M is the GFM-RAG scale point (NeurIPS 2025) and decision
    9's number; the claim of comparable scale rests on it, so it is asserted."""
    from graft.graphbuild.encoders import parameter_count
    from graft.retrieve.scorer import build_scorer

    scorer = build_scorer(seed=13)
    assert parameter_count(scorer) <= cpins.SCORER["max_params"]
    with pytest.raises(ValueError, match="over decision 9's"):
        build_scorer(seed=13, hidden=4096, layers=8)


def test_the_scorer_honours_the_fix_f6_interface(snap):
    """``score(question, pool) -> per-atom scores`` — **atom** ids, the shape
    Stage D's featurizer consumes.  Fusion consumes node ids, and the conversion
    is :func:`channel_scores`' job, tested separately below.  (An earlier
    version of this test "integrated" the scorer into fusion by passing an
    empty mapping, which asserted nothing — 15 Aug 2026 audit.)"""
    from graft.retrieve.scorer import build_scorer, score_pool

    pool, _, _ = build_pool(snap, {"n_v1": 1.0, "n_v2": 0.5})
    scores = score_pool(build_scorer(seed=13), pool, snap, StubEmbedder(dim=384), "my weight")
    assert set(scores) == set(pool.ids())
    assert all(isinstance(v, float) for v in scores.values())


def test_the_scorer_channel_scores_the_whole_scope_in_node_space(snap):
    """**The sixth channel's contract** (15 Aug 2026 audit blocker).  It emits
    node ids over the *whole eligible scope* — including nodes no other channel
    returned — never atom ids, never structural atoms, never another
    conversation's nodes.  The first wiring scored the already-capped
    five-channel pool, so the GNN could not surface anything the cheap
    channels' cap had dropped."""
    from graft.retrieve.pool import eligible_nodes
    from graft.retrieve.scorer import build_scorer, channel_scores

    hits = channel_scores(build_scorer(seed=13), snap, StubEmbedder(dim=384), "my weight", "c1")
    assert set(hits) == set(eligible_nodes(snap, "c1"))  # the whole scope, node ids
    assert "n_other" not in hits  # c2's node
    assert "n_e1" not in hits and "n_ti" not in hits  # structural, never hits
    # and this shape genuinely fuses: a scorer-only stack assembles a real pool
    pool, _, _ = assemble(snap, {cpins.SCORER_CHANNEL: hits})
    assert len(pool) > 0
    assert channel_scores(build_scorer(seed=13), snap, StubEmbedder(dim=384), "q", "c9") == {}


def test_the_scorer_is_deterministic_given_a_seed(snap):
    """The ``build_arm`` pattern: the seed must reach *initialisation*.

    Seeding only inside the training loop leaves every seed sharing one random
    initialisation, so three seeds would estimate variance over dropout and batch
    order alone. Phase 6 shipped exactly that defect (caught 14 Aug 2026).
    """
    from graft.retrieve.scorer import build_scorer, score_pool

    pool, _, _ = build_pool(snap, {"n_v1": 1.0})
    a = score_pool(build_scorer(seed=13), pool, snap, StubEmbedder(dim=384), "q")
    b = score_pool(build_scorer(seed=13), pool, snap, StubEmbedder(dim=384), "q")
    c = score_pool(build_scorer(seed=42), pool, snap, StubEmbedder(dim=384), "q")
    assert a == b
    assert a != c


def test_the_scorer_is_query_conditioned(snap):
    """Two questions over one pool must not produce identical scores.

    The property that makes it *query-conditioned* rather than query-augmented,
    and the reason the query gates messages instead of only entering the head.
    """
    from graft.retrieve.scorer import build_scorer, score_pool

    pool, _, _ = build_pool(snap, {"n_v1": 1.0, "n_v2": 0.5})
    scorer, embedder = build_scorer(seed=13), StubEmbedder(dim=384)
    assert score_pool(scorer, pool, snap, embedder, "my weight") != score_pool(
        scorer, pool, snap, embedder, "yoga mat thickness"
    )


def test_the_scorer_handles_a_pool_with_no_edges(snap):
    """A pool of bare node atoms is legal — nodes reference nothing — so the
    forward pass must not need a special case at the call site."""
    from graft.retrieve.scorer import build_scorer, pool_adjacency, score_pool

    # ``n_iso`` is eligible and has no committed relations, so its closure is
    # itself: one node atom, no companions, no edge atoms.
    pool, _, _ = build_pool(snap, {"n_iso": 1.0})
    assert len(pool) == 1
    edge_index, edge_rel = pool_adjacency(pool, pool.ids())
    assert edge_index.shape == (2, 0) and edge_rel.shape == (0,)
    assert len(score_pool(build_scorer(seed=13), pool, snap, StubEmbedder(dim=384), "q")) == 1


def test_training_restores_the_best_state_rather_than_merely_stopping(snap):
    """Guard two, inherited from P6.11 — and now actually asserted: the returned
    model's recomputed dev loss must equal the reported best.  The first version
    checked only that training ran, so deleting the ``load_state_dict`` line
    would not have failed it (15 Aug 2026 audit)."""
    from graft.retrieve.scorer import _example_loss, build_scorer, train_scorer

    pool, _, _ = build_pool(snap, {"n_v1": 1.0, "n_v2": 0.5})
    example = _prepared(snap, pool)
    scorer = build_scorer(seed=13)
    result = train_scorer(
        scorer, {"train": [example], "dev": [example]}, seed=13, budget=_tiny_budget()
    )
    assert result["epochs_run"] >= 1
    assert math.isfinite(result["best_dev_loss"])
    assert result["parameters"] <= cpins.SCORER["max_params"]
    scorer.eval()
    with torch.no_grad():
        restored_loss, n = _example_loss(scorer, example)
    assert n > 0
    assert abs(float(restored_loss.item()) - result["best_dev_loss"]) < 1e-6


def test_training_refuses_when_no_dev_example_is_scorable(snap):
    """Guard three.  A scorer that never saw a scorable dev item is a *random*
    scorer, and returning it as 'early stopped' would put noise into the fusion
    as a sixth channel while every report read as a completed run."""
    from graft.retrieve.scorer import build_scorer, train_scorer

    pool, _, _ = build_pool(snap, {"n_v1": 1.0})
    with pytest.raises(ValueError, match="no scorable dev example"):
        train_scorer(
            build_scorer(seed=13),
            {"train": [_prepared(snap, pool)], "dev": []},
            seed=13,
            budget=_tiny_budget(),
        )


def test_the_distant_labels_come_from_recall_not_the_scorer(snap):
    """Item 2's signal, derived at the sanctioned gold boundary.

    ``evidence_turns`` is session-level (training) and ``has_answer_turns`` is
    turn-level (evaluation) — item 2's whole distinction — and the label vector
    is in ``pool.ids()`` order so it lines up with the feature matrix row for row.
    """
    from graft.retrieve.recall import distant_labels, evidence_turns

    instance = {
        "question_id": "c1",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [[{"has_answer": True}], [{}, {}]],
    }
    assert evidence_turns(instance) == frozenset({"lme_s/c1/s1/0"})
    pool, _, _ = build_pool(snap, {"n_v1": 1.0, "n_v2": 0.5})
    labels = distant_labels(snap, pool, instance, "c1")
    assert len(labels) == len(pool)
    assert set(labels) <= {0.0, 1.0}
    # n_v1's span is in turn t1 of session s1 -> relevant; structural atoms are not
    by_id = dict(zip(pool.ids(), labels))
    assert by_id[node_atom_id("n_v1")] == 1.0
    assert by_id[node_atom_id("n_e1")] == 0.0


def _tiny_budget() -> dict:
    """Two epochs, so the guard tests do not train a real model.

    ``train_scorer`` reads ``pins.TRAINING`` and never an argument default, so a
    caller cannot quietly give the scorer more epochs than Stage B's arms got;
    this is the escape hatch that exists for tests and is reported when used.
    """
    from graft.graphbuild.pins import TRAINING

    return {**TRAINING, "epochs": 2, "early_stop_patience": 1}


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
