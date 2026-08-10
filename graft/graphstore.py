"""The ``GraphSnapshot`` protocol and its two implementations.

``GraphSnapshot`` is a ``Protocol``, not a class (Phase-0 gap G2).  Phase 1's
checker takes a graph snapshot, but the real event-log-backed store is Phase 6.
If the snapshot were a concrete class, Phase 1 would block on Phase 6 and the
whole "build Phases 1-4 with zero real data" strategy would collapse.  Phases
1-4 depend on the protocol alone and never import a concrete store.

Keep the protocol minimal.  Every method added here is a method the Phase-2
synthetic lattice must also implement, so convenience helpers are a tax on the
lattice, not a free win.

**Transaction time vs valid time.**  The snapshot pins *transaction* time: it is
the state of the world as the log knew it at a given sequence number, and
``is_live`` answers "had this edge been invalidated by then".  *Valid* time —
whether a fact holds during the interval a question asks about — is carried by
``Obligations.time_constraint`` and checked against ``valid_during`` edges by
Phase 1.  The two are separate axes and neither substitutes for the other.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Protocol

from graft.schemas import Assertion, Edge, Node, SourceSpan, Turn

__all__ = [
    "GraphSnapshot",
    "DictGraphSnapshot",
    "ReplayGraphStore",
    "GRAPH_OPS",
    "implements_graph_snapshot",
]

#: Event-log operations that mutate graph state.  Anything else in the log is
#: ignored by replay, which lets the ledger and other subsystems share the
#: stream without the graph store knowing about them.
GRAPH_OPS = (
    "node.add",
    "edge.add",
    "edge.invalidate",
    "assertion.add",
    "assertion.set_eligibility",
    "turn.add",
    "span.add",
)

_PROTOCOL_MEMBERS = (
    "snapshot_id",
    "node",
    "edge",
    "edges_of",
    "is_live",
    "ntype",
    "is_eligible",
    "assertion",
    "span",
    "turn",
)


class GraphSnapshot(Protocol):
    """The entire dependency surface Phase 1 has on graph state."""

    @property
    def snapshot_id(self) -> int:
        """Event-log sequence number this snapshot was taken at."""

    def node(self, node_id: str) -> Node | None: ...

    def edge(self, edge_id: str) -> Edge | None: ...

    def edges_of(self, node_id: str, etype: str | None = None) -> tuple[Edge, ...]:
        """Every edge incident to ``node_id``, optionally filtered by type."""

    def is_live(self, edge_id: str) -> bool:
        """True when the edge has not been invalidated as of this snapshot.

        This is the one place invalidation semantics live, so that no downstream
        component reimplements ``t_invalid`` / ``superseded_by`` handling.
        """

    def ntype(self, node_id: str) -> str | None: ...

    def is_eligible(self, assertion_id: str) -> bool:
        """True when the assertion passed the Phase-5 support gate.

        Required by ``H``'s support-eligibility sub-check (architecture fix F9).
        Reading an already-stored flag is deterministic and belongs in the
        checker; *computing* entailment is learned and does not.  Unknown
        assertion ids are ineligible — a proof may not lean on something the
        snapshot has never seen.
        """

    def assertion(self, assertion_id: str) -> Assertion | None:
        """The stored assertion, for its ``spans``.

        Added in Phase 1 beyond the two members the build plan listed.  Scope
        (sub-check 5) has to walk an atom's provenance to a ``conv_id``, and for
        a node atom the only route to spans is the backing assertion.  The
        alternative — duplicating ``Assertion.spans`` into ``Node.payload`` at
        commit time — would give provenance two sources of truth, which is worse
        than one more read method.
        """

    def span(self, span_id: str) -> SourceSpan | None: ...

    def turn(self, turn_id: str) -> Turn | None:
        """Needed for ``conv_id``: the unit conversational scope is defined over."""


def implements_graph_snapshot(obj: object) -> bool:
    """Structural conformance check.

    Written out rather than using ``@runtime_checkable`` because a protocol with
    a non-method member (``snapshot_id``) cannot be used with ``isinstance`` on
    Python 3.11.
    """
    return all(hasattr(obj, name) for name in _PROTOCOL_MEMBERS)


class DictGraphSnapshot:
    """In-memory snapshot over plain dicts.

    Used by the Phase-2 lattice directly and produced by ``ReplayGraphStore``.
    """

    def __init__(
        self,
        snapshot_id: int = 0,
        nodes: Iterable[Node] = (),
        edges: Iterable[Edge] = (),
        assertions: Iterable[Assertion] = (),
        turns: Iterable[Turn] = (),
        spans: Iterable[SourceSpan] = (),
    ) -> None:
        self._snapshot_id = int(snapshot_id)
        self._nodes: dict[str, Node] = {n.node_id: n for n in nodes}
        self._edges: dict[str, Edge] = {e.edge_id: e for e in edges}
        self._assertions: dict[str, Assertion] = {a.assertion_id: a for a in assertions}
        self._turns: dict[str, Turn] = {t.turn_id: t for t in turns}
        self._spans: dict[str, SourceSpan] = {s.span_id: s for s in spans}
        self._incident: dict[str, list[str]] = {}
        for edge in self._edges.values():
            self._index_edge(edge)

    def _index_edge(self, edge: Edge) -> None:
        for endpoint in (edge.src, edge.dst):
            bucket = self._incident.setdefault(endpoint, [])
            if edge.edge_id not in bucket:
                bucket.append(edge.edge_id)

    # -- protocol ----------------------------------------------------------

    @property
    def snapshot_id(self) -> int:
        return self._snapshot_id

    def node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def edge(self, edge_id: str) -> Edge | None:
        return self._edges.get(edge_id)

    def edges_of(self, node_id: str, etype: str | None = None) -> tuple[Edge, ...]:
        found = (self._edges[eid] for eid in self._incident.get(node_id, ()))
        if etype is not None:
            found = (e for e in found if e.etype == etype)
        return tuple(found)

    def is_live(self, edge_id: str) -> bool:
        edge = self._edges.get(edge_id)
        return edge is not None and edge.t_invalid is None

    def ntype(self, node_id: str) -> str | None:
        node = self._nodes.get(node_id)
        return None if node is None else node.ntype

    def is_eligible(self, assertion_id: str) -> bool:
        assertion = self._assertions.get(assertion_id)
        return assertion is not None and assertion.eligibility == "eligible"

    def assertion(self, assertion_id: str) -> Assertion | None:
        return self._assertions.get(assertion_id)

    def span(self, span_id: str) -> SourceSpan | None:
        return self._spans.get(span_id)

    def turn(self, turn_id: str) -> Turn | None:
        return self._turns.get(turn_id)

    # -- construction ------------------------------------------------------

    def add_node(self, node: Node) -> None:
        self._nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        self._edges[edge.edge_id] = edge
        self._index_edge(edge)

    def add_assertion(self, assertion: Assertion) -> None:
        self._assertions[assertion.assertion_id] = assertion

    def add_turn(self, turn: Turn) -> None:
        self._turns[turn.turn_id] = turn

    def add_span(self, span: SourceSpan) -> None:
        self._spans[span.span_id] = span

    def invalidate_edge(
        self, edge_id: str, t_invalid: str, superseded_by: str | None = None
    ) -> None:
        """Mark an edge invalid.  It is never removed — Zep's precedent, and it
        is what makes a wrong supersession recoverable."""
        edge = self._edges[edge_id]
        self._edges[edge_id] = Edge(
            edge_id=edge.edge_id,
            etype=edge.etype,
            src=edge.src,
            dst=edge.dst,
            t_created=edge.t_created,
            provenance=edge.provenance,
            t_invalid=t_invalid,
            superseded_by=superseded_by,
        )

    def set_eligibility(self, assertion_id: str, eligibility: str) -> None:
        current = self._assertions[assertion_id]
        self._assertions[assertion_id] = Assertion(
            assertion_id=current.assertion_id,
            kind=current.kind,
            text_norm=current.text_norm,
            spans=current.spans,
            flags=current.flags,
            t_created=current.t_created,
            eligibility=eligibility,
        )

    # -- diagnostics -------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "live_edges": sum(1 for e in self._edges.values() if e.t_invalid is None),
            "assertions": len(self._assertions),
            "eligible_assertions": sum(
                1 for a in self._assertions.values() if a.eligibility == "eligible"
            ),
            "turns": len(self._turns),
            "spans": len(self._spans),
        }

    def state_digest(self) -> str:
        """Content hash of the whole snapshot, for replay-equality tests."""
        from graft.canonical import digest_of

        return digest_of(
            {
                "nodes": [self._nodes[k].to_dict() for k in sorted(self._nodes)],
                "edges": [self._edges[k].to_dict() for k in sorted(self._edges)],
                "assertions": [
                    self._assertions[k].to_dict() for k in sorted(self._assertions)
                ],
                "turns": [self._turns[k].to_dict() for k in sorted(self._turns)],
                "spans": [self._spans[k].to_dict() for k in sorted(self._spans)],
            }
        )


class ReplayGraphStore:
    """Builds a snapshot by replaying the event log.

    Replay is O(n) in log length for every snapshot.  That is fine at Phase 0 —
    there is no real data yet — and a checkpoint/cache belongs in Phase 6 when
    there is a measured reason for it.  Adding one now is exactly the premature
    optimisation the boring-stack rule forbids.

    TODO(Phase 6): periodic materialised checkpoints keyed by seq, so that
    ``at()`` replays only the tail.
    """

    def __init__(self, log: Any) -> None:
        self._log = log

    def at(self, snapshot_id: int | None = None) -> DictGraphSnapshot:
        """Reconstruct graph state as of ``snapshot_id`` (default: the whole log)."""
        snapshot = DictGraphSnapshot(snapshot_id=0)
        last_seq = -1
        for event in self._log.replay(upto=snapshot_id):
            last_seq = event.seq
            self._apply(snapshot, event.op, event.payload)
        snapshot._snapshot_id = last_seq + 1
        return snapshot

    @staticmethod
    def _apply(snapshot: DictGraphSnapshot, op: str, payload: Mapping[str, Any]) -> None:
        if op == "node.add":
            snapshot.add_node(Node.from_dict(payload))
        elif op == "edge.add":
            snapshot.add_edge(Edge.from_dict(payload))
        elif op == "edge.invalidate":
            snapshot.invalidate_edge(
                payload["edge_id"], payload["t_invalid"], payload.get("superseded_by")
            )
        elif op == "assertion.add":
            snapshot.add_assertion(Assertion.from_dict(payload))
        elif op == "assertion.set_eligibility":
            snapshot.set_eligibility(payload["assertion_id"], payload["eligibility"])
        elif op == "turn.add":
            snapshot.add_turn(Turn.from_dict(payload))
        elif op == "span.add":
            snapshot.add_span(SourceSpan.from_dict(payload))
        # Anything else in the log (ledger checkpoints, run markers) is not
        # graph state and is deliberately ignored.

    def iter_ops(self, upto: int | None = None) -> Iterator[tuple[int, str]]:
        for event in self._log.replay(upto=upto):
            yield event.seq, event.op
