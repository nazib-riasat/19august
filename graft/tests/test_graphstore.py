"""The GraphSnapshot protocol and its two implementations.

Discharges Phase-0 exit criterion 2 (replay reconstructs an identical graph from
a 1,000-event log) and the build-step-6 check that a hand-built dummy satisfies
the protocol Phase 1 will be written against.
"""

from __future__ import annotations

import random

from graft.eventlog import EventLog
from graft.graphstore import (
    DictGraphSnapshot,
    ReplayGraphStore,
    implements_graph_snapshot,
)
from graft.schemas import Assertion, AssertionFlags, Edge, Node

TS = "2026-08-08T00:00:00+00:00"


def _node(i: int) -> Node:
    return Node(node_id=f"n{i}", ntype="Entity", payload={"i": i})


def _edge(i: int, src: str, dst: str, etype: str = "same_as") -> Edge:
    return Edge(
        edge_id=f"e{i}",
        etype=etype,
        src=src,
        dst=dst,
        t_created=TS,
        provenance=(f"s{i}",),
    )


def _assertion(i: int, eligibility: str = "eligible") -> Assertion:
    return Assertion(
        assertion_id=f"a{i}",
        kind="claim",
        text_norm=f"claim {i}",
        spans=(f"s{i}",),
        flags=AssertionFlags(asserted_by="user", entailed_by_span=True, entailed_score=0.9),
        t_created=TS,
        eligibility=eligibility,
    )


# -- the protocol -----------------------------------------------------------


def test_a_hand_built_dummy_satisfies_the_protocol():
    """Build step 6: this is all Phase 1 needs in order to start."""
    snapshot = DictGraphSnapshot(
        snapshot_id=3,
        nodes=[_node(0), _node(1), _node(2)],
        edges=[_edge(0, "n0", "n1")],
    )
    assert implements_graph_snapshot(snapshot)
    assert snapshot.snapshot_id == 3
    assert snapshot.ntype("n0") == "Entity"
    assert snapshot.node("n1").payload["i"] == 1
    assert snapshot.edge("e0").etype == "same_as"
    assert snapshot.is_live("e0")


def test_missing_ids_return_none_rather_than_raising():
    snapshot = DictGraphSnapshot()
    assert snapshot.node("nope") is None
    assert snapshot.edge("nope") is None
    assert snapshot.ntype("nope") is None
    assert snapshot.is_live("nope") is False


def test_edges_of_covers_both_endpoints_and_filters_by_type():
    snapshot = DictGraphSnapshot(
        nodes=[_node(0), _node(1), _node(2)],
        edges=[
            _edge(0, "n0", "n1", "same_as"),
            _edge(1, "n1", "n2", "has_value"),
        ],
    )
    assert {e.edge_id for e in snapshot.edges_of("n1")} == {"e0", "e1"}
    assert [e.edge_id for e in snapshot.edges_of("n1", "has_value")] == ["e1"]
    assert snapshot.edges_of("n0", "has_value") == ()


def test_invalidation_keeps_the_edge_addressable():
    """Nothing is deleted, so a wrong supersession is recoverable."""
    snapshot = DictGraphSnapshot(nodes=[_node(0), _node(1)], edges=[_edge(0, "n0", "n1")])
    snapshot.invalidate_edge("e0", TS, superseded_by="e9")
    assert snapshot.is_live("e0") is False
    assert snapshot.edge("e0") is not None
    assert snapshot.edge("e0").superseded_by == "e9"
    assert snapshot.counts()["edges"] == 1
    assert snapshot.counts()["live_edges"] == 0


def test_eligibility_is_a_stored_flag_read():
    """H's support sub-check reads this; it never computes entailment."""
    snapshot = DictGraphSnapshot(
        assertions=[_assertion(0, "eligible"), _assertion(1, "quarantined")]
    )
    assert snapshot.is_eligible("a0") is True
    assert snapshot.is_eligible("a1") is False


def test_unknown_assertions_are_ineligible():
    """A proof may not lean on something the snapshot has never seen."""
    assert DictGraphSnapshot().is_eligible("never-stored") is False


def test_set_eligibility_preserves_everything_else():
    snapshot = DictGraphSnapshot(assertions=[_assertion(0)])
    snapshot.set_eligibility("a0", "quarantined")
    assert snapshot.is_eligible("a0") is False


# -- replay -----------------------------------------------------------------


def _write_synthetic_log(path, n_events=1000, seed=13):
    rng = random.Random(seed)
    log = EventLog.open(path)
    n_nodes = 120
    for i in range(n_nodes):
        log.append("node.add", _node(i).to_dict())
    for i in range(30):
        log.append("assertion.add", _assertion(i).to_dict())
    edges_written = 0
    while len(log) < n_events:
        remaining = n_events - len(log)
        if edges_written > 20 and rng.random() < 0.15:
            victim = rng.randrange(edges_written)
            log.append(
                "edge.invalidate",
                {"edge_id": f"e{victim}", "t_invalid": TS, "superseded_by": None},
            )
        elif remaining > 1 and rng.random() < 0.05:
            log.append(
                "assertion.set_eligibility",
                {"assertion_id": f"a{rng.randrange(30)}", "eligibility": "quarantined"},
            )
        else:
            src, dst = rng.randrange(n_nodes), rng.randrange(n_nodes)
            log.append("edge.add", _edge(edges_written, f"n{src}", f"n{dst}").to_dict())
            edges_written += 1
    return log


def test_replay_reconstructs_an_identical_graph(tmp_path):
    """Criterion 2, over a 1,000-event synthetic log."""
    log = _write_synthetic_log(tmp_path / "e.jsonl", n_events=1000)
    assert len(log) == 1000

    store = ReplayGraphStore(log)
    first = store.at()
    second = store.at()
    assert first.state_digest() == second.state_digest()
    assert first.counts() == second.counts()
    assert first.counts()["nodes"] == 120
    assert first.counts()["live_edges"] < first.counts()["edges"]
    assert first.counts()["eligible_assertions"] < 30
    log.close()


def test_replay_is_identical_after_a_close_and_reopen(tmp_path):
    """The handoff case: the same folder opened on another machine must rebuild
    the same graph."""
    path = tmp_path / "e.jsonl"
    log = _write_synthetic_log(path, n_events=1000)
    digest = ReplayGraphStore(log).at().state_digest()
    log_digest = log.digest()
    log.close()

    reopened = EventLog.open(path)
    assert ReplayGraphStore(reopened).at().state_digest() == digest
    assert reopened.digest() == log_digest
    reopened.close()


def test_snapshot_pinning_is_a_sequence_number(tmp_path):
    path = tmp_path / "e.jsonl"
    with EventLog.open(path) as log:
        log.append("node.add", _node(0).to_dict())
        log.append("node.add", _node(1).to_dict())
        pinned = log.snapshot_id()
        log.append("node.add", _node(2).to_dict())

        store = ReplayGraphStore(log)
        earlier = store.at(pinned)
        assert earlier.snapshot_id == 2
        assert earlier.node("n2") is None
        assert store.at().node("n2") is not None


def test_an_edge_is_live_in_a_snapshot_taken_before_its_invalidation(tmp_path):
    """Transaction time, not valid time: the snapshot answers 'had this been
    invalidated by then', and nothing else."""
    with EventLog.open(tmp_path / "e.jsonl") as log:
        log.append("node.add", _node(0).to_dict())
        log.append("node.add", _node(1).to_dict())
        log.append("edge.add", _edge(0, "n0", "n1").to_dict())
        before = log.snapshot_id()
        log.append(
            "edge.invalidate",
            {"edge_id": "e0", "t_invalid": TS, "superseded_by": None},
        )
        store = ReplayGraphStore(log)
        assert store.at(before).is_live("e0") is True
        assert store.at().is_live("e0") is False


def test_non_graph_ops_are_ignored_by_replay(tmp_path):
    """The ledger shares the stream; the graph store must not choke on it."""
    with EventLog.open(tmp_path / "e.jsonl") as log:
        log.append("node.add", _node(0).to_dict())
        log.append("ledger.checkpoint", {"totals": {}, "stages": {}, "queries_seen": 0})
        log.append("run.marker", {"note": "anything"})
        snapshot = ReplayGraphStore(log).at()
        assert snapshot.counts()["nodes"] == 1
        assert snapshot.snapshot_id == 3
