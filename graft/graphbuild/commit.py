"""P6.6 — propose → validate → commit, and the corruption audit (G8, G9).

**The pipeline never trusts its decoders, and the sentence matters.**  Every
proposal goes through ``validate.validate`` before a single event is appended,
and a refusal is *counted by category*, not logged and swallowed.  What that buys
is bounded: the seven checks catch malformed writes, and they cannot catch a
semantically wrong ``same_as`` — plan §3.2's own warning, repeated wherever this
pipeline is described.

**Nothing is ever deleted.**  Supersession is ``edge.invalidate`` plus a new
edge; the old edge stays in the log and stays live at every earlier snapshot.
That is Zep's bi-temporal edge-invalidation model (vendor-authored preprint,
flagged as everywhere it appears), and it is what makes a *wrong* supersession
recoverable — which matters because a wrong merge is the single most damaging
Stage-B error (plan §8 risk #1).

**`DEFER` writes nothing at all.**  A deferred mention parks in a revisit queue
keyed by its entity anchor; it produces no node, no edge, no event.  A `DEFER`
that quietly created a placeholder entity would be `CREATE_NEW_ENTITY` wearing a
different name, and the ablation the plan asks for (DEFER on/off) would measure
nothing.

**The write path is the Phase-5 op vocabulary, unchanged.**  ``node.add``,
``edge.add`` and ``edge.invalidate`` already replay; exit criterion 5 requires no
new op be invented silently, and none is.

**The corruption audit (G9) has three properties**, declared because the
architecture required the audit and no document defined it:

* **(a) supersession is effective** — the superseded edge is not live, the
  superseding one is;
* **(b) history is intact** — ``at(seq)`` *before* the update still returns the
  old fact live;
* **(c) no collateral flips** — the set of live edges that changed across the
  commit is *exactly* the committed edge plus its declared invalidations.

(c) is the one that catches real damage.  It runs green as a harness property on
synthetic decoder outputs before any learned decoder exists; at Gate 1 it becomes
a measured error rate **of the decoders**, not of the pipeline.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from graft.eventlog import EventLog
from graft.graphstore import ReplayGraphStore
from graft.graphbuild.validate import CHECKS, Commit, validate
from graft.ids import edge_id as edge_id_of
from graft.ids import node_id as node_id_of
from graft.schemas import (
    PAYLOAD_ALIASES,
    PAYLOAD_ASSERTION_ID,
    PAYLOAD_CONV_ID,
    PAYLOAD_NAME,
    CheckResult,
    Edge,
    Node,
)

__all__ = [
    "CommitResult",
    "Committer",
    "DeferQueue",
    "entity_node",
    "claim_node",
    "corruption_audit",
]


class DeferQueue:
    """Mentions parked by a `DEFER` decision, keyed by normalized anchor.

    A queue rather than a graph write, because `DEFER` means *not yet decided*
    and the graph has no way to represent that without inventing a placeholder
    entity — which would be `CREATE_NEW_ENTITY` under another name and would make
    the on/off ablation meaningless.

    It is deliberately in-memory and run-scoped: a deferred mention is a
    *decision to revisit within this construction pass*, not a durable record.
    Persisting it would put an undecided item in the permanent log, where every
    later reader would have to know it was not evidence.
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: dict[str, list[Any]] = {}

    def park(self, anchor: str, mention: Any) -> None:
        self._items.setdefault(anchor, []).append(mention)

    def revisit(self, anchor: str) -> list[Any]:
        """Take everything parked under ``anchor``, clearing it."""
        return self._items.pop(anchor, [])

    def pending(self) -> int:
        return sum(len(v) for v in self._items.values())

    def anchors(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __len__(self) -> int:
        return self.pending()


class CommitResult:
    """What one commit attempt did, or refused to do, and why."""

    __slots__ = ("accepted", "result", "seqs", "label")

    def __init__(
        self, accepted: bool, result: CheckResult, seqs: Sequence[int] = (), label: str = ""
    ) -> None:
        self.accepted = bool(accepted)
        self.result = result
        self.seqs = tuple(seqs)
        self.label = label

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "accepted": self.accepted,
            "seqs": list(self.seqs),
            "violations": [v.to_dict() for v in self.result.violations],
        }


# --------------------------------------------------------------------------
# node builders — content-derived ids, one home
# --------------------------------------------------------------------------


def entity_node(name: str, conv_id: str, aliases: Iterable[str] = ()) -> Node:
    """An ``Entity`` node, its id derived from ``(conv_id, normalized name)``.

    **Scoped to the conversation on purpose.**  Two users who both mention "my
    car" have two different cars, and a globally-keyed entity would merge them —
    a wrong merge, the most damaging Stage-B error, introduced by the id scheme
    rather than by a decoder.  Cross-conversation identity, if it is ever wanted,
    is what the ``same_as`` edge is for, and that is a *decision* a decoder makes
    and the log records.

    **``name`` is stored normalized, and the raw surface form is an alias.**
    Found by the validator on the first smoke run over the pilot log, which is
    what it is for: storing the *raw* name made the payload a function of which
    surface form happened to be seen first, while the id was a function of the
    normalized one — so "Fitbit Charge 3" and "fitbit charge 3" derived the same
    id with two different payloads and the commit was refused for id collision.
    The alternative — putting the raw form in the id — is worse: it makes the two
    surface forms two entities, which is the duplicate-entity bug normalization
    exists to prevent.

    So the payload is a pure function of the id's inputs, which makes a repeated
    creation idempotent, and **alias sets accumulate through D1's linking
    decisions** rather than at creation (the shape Phase 7 is promised in §8).
    """
    from graft.graphbuild.candidates import normalise_name

    canonical = normalise_name(name)
    key = f"{conv_id}\x1f{canonical}"
    # The raw surface form is preserved as an alias when it differs from the
    # canonical form — losing it would make the payload lie about what was said
    # — and the conversation id is IN the payload, not only folded into the id:
    # the candidate generator scopes its scan by conversation (the wrong-merge
    # guard applied at retrieval as well as at creation), and a scope living
    # only inside a hash is a scope nothing can filter on (found by the
    # 14 Aug 2026 audit).
    alias_set = {str(a) for a in aliases}
    if name != canonical:
        alias_set.add(name)
    return Node(
        node_id=node_id_of("Entity", key),
        ntype="Entity",
        payload={
            PAYLOAD_NAME: canonical,
            PAYLOAD_ALIASES: sorted(alias_set),
            PAYLOAD_CONV_ID: conv_id,
        },
    )


def claim_node(assertion_id: str, kind: str = "claim") -> Node:
    """A ``Claim``/``Value``/``Event`` node backed by an eligible assertion.

    The Phase-5 assertion kinds map onto node types: ``claim``/``value``/``event``
    to their namesakes, and ``time`` to ``Claim`` — a temporal assertion is a
    claim *about* time, and ``TimeInterval`` is the node an interval lives in, not
    the node an assertion becomes.  Stated because the mapping is not the
    identity and a reader will assume it is.
    """
    ntype = {"claim": "Claim", "value": "Value", "event": "Event", "time": "Claim"}.get(
        kind, "Claim"
    )
    return Node(
        node_id=node_id_of(ntype, assertion_id),
        ntype=ntype,
        payload={PAYLOAD_ASSERTION_ID: assertion_id},
    )


def _edge(etype: str, src: str, dst: str, t_created: str, provenance: Sequence[str]) -> Edge:
    return Edge(
        edge_id=edge_id_of(etype, src, dst),
        etype=etype,
        src=src,
        dst=dst,
        t_created=t_created,
        provenance=tuple(provenance),
    )


# --------------------------------------------------------------------------
# the committer
# --------------------------------------------------------------------------


class Committer:
    """Validates a proposal and, if it passes, writes it to the event log.

    Deliberately thin and decoder-agnostic: it takes a :class:`Commit` and knows
    nothing about how it was proposed.  That is what lets the corruption audit run
    on *synthetic* decoder outputs (G9) before any learned decoder exists, and it
    is what stops the pipeline growing a special case per decoder.
    """

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.store = ReplayGraphStore(log)
        self.accepted = 0
        self.refused = 0
        #: D3/D4 proposals declined on the calibrated-confidence floor
        #: (decision 9).  Counted apart from ``refused``: a floor decline is the
        #: commit *rule* working, not a malformed write, and folding it into the
        #: violation rate would make exit criterion 1 unreadable.
        self.below_floor = 0
        self.rejections: dict[str, int] = {name: 0 for name in CHECKS}
        self.history: list[CommitResult] = []
        #: The proposals, parallel to ``history`` — what each commit was *asked*
        #: to do, which is what the corruption audit's property (c) checks the
        #: log against.  Captured here so convenience methods (``link_existing``,
        #: ``supersede``…) are audited exactly like raw ``submit`` calls.
        self.proposals: list[Commit] = []

    def snapshot(self, upto: int | None = None) -> Any:
        return self.store.at(upto)

    def submit(self, commit: Commit) -> CommitResult:
        """Validate against the *current* snapshot, then write or refuse.

        The snapshot is taken fresh each time rather than cached: a commit is
        validated against the graph it will actually land in, and a stale
        snapshot is how a duplicate-edge check passes for an edge that was
        committed one call earlier.
        """
        snapshot = self.store.at()
        result = validate(commit, snapshot)
        if not result.ok:
            self.refused += 1
            for violation in result.violations:
                self.rejections[violation.check] = self.rejections.get(violation.check, 0) + 1
            outcome = CommitResult(False, result, (), commit.label)
            self.history.append(outcome)
            self.proposals.append(commit)
            return outcome

        seqs: list[int] = []
        for node in commit.nodes:
            seqs.append(self.log.append("node.add", node.to_dict()))
        for edge in commit.edges:
            seqs.append(self.log.append("edge.add", edge.to_dict()))
        for edge_id, t_invalid, superseded_by in commit.invalidations:
            payload: dict[str, Any] = {"edge_id": edge_id, "t_invalid": t_invalid}
            if superseded_by is not None:
                payload["superseded_by"] = superseded_by
            seqs.append(self.log.append("edge.invalidate", payload))

        self.accepted += 1
        outcome = CommitResult(True, result, seqs, commit.label)
        self.history.append(outcome)
        self.proposals.append(commit)
        return outcome

    # -- the shapes the decoders produce ----------------------------------

    def link_existing(
        self, assertion_id: str, kind: str, entity_id: str, t: str, spans: Sequence[str]
    ) -> CommitResult:
        """D1's `LINK_EXISTING`: a claim node and its ``about_entity`` edge."""
        node = claim_node(assertion_id, kind)
        return self.submit(
            Commit(
                nodes=[node],
                edges=[_edge("about_entity", node.node_id, entity_id, t, spans)],
                label=f"link:{assertion_id}->{entity_id}",
            )
        )

    def create_entity(
        self,
        assertion_id: str,
        kind: str,
        name: str,
        conv_id: str,
        t: str,
        spans: Sequence[str],
    ) -> CommitResult:
        """D1's `CREATE_NEW_ENTITY`: entity, claim, and the link between them."""
        entity = entity_node(name, conv_id)
        node = claim_node(assertion_id, kind)
        return self.submit(
            Commit(
                nodes=[entity, node],
                edges=[_edge("about_entity", node.node_id, entity.node_id, t, spans)],
                label=f"create:{name}",
            )
        )

    def supersede(
        self,
        old_claim_id: str,
        new_claim_id: str,
        t: str,
        spans: Sequence[str],
    ) -> CommitResult:
        """D2's `SUPERSEDES`: a new ``supersedes`` edge, and **every live edge
        that carries the old claim as a current fact** invalidated and linked to
        the successor.

        "Every", not "its ``about_entity`` edges" *(widened 14 Aug 2026 — the
        audit measured the narrow version leaving a superseded claim's
        ``valid_during`` and ``supported_by`` edges live, so the retired fact
        kept answering temporal queries and `H` kept accepting it as current)*.
        Two relation families are deliberately **exempt** because they record
        history rather than currency: ``supersedes`` and ``contradicts`` edges
        stay live — a superseded claim was still contradicted by what
        contradicted it, and retiring the record of a disagreement is exactly
        the erasure non-destructive versioning exists to prevent.

        The invalidation is what makes supersession *effective* (G9 property a);
        the fact that the old edges remain in the log is what makes it
        *recoverable* (property b).  Both halves are needed and neither is
        optional.
        """
        snapshot = self.store.at()
        edge = _edge("supersedes", new_claim_id, old_claim_id, t, spans)
        history = ("supersedes", "contradicts")
        invalidations = [
            (e.edge_id, t, edge.edge_id)
            for e in snapshot.edges_of(old_claim_id)
            if e.etype not in history and snapshot.is_live(e.edge_id)
        ]
        return self.submit(
            Commit(
                edges=[edge],
                invalidations=invalidations,
                label=f"supersede:{old_claim_id}->{new_claim_id}",
            )
        )

    def contradict(
        self, claim_a: str, claim_b: str, t: str, spans: Sequence[str]
    ) -> CommitResult:
        """D2's `CONFLICT`: a ``contradicts`` edge and nothing invalidated.

        A conflict is *not* a supersession: neither claim wins, both stay live,
        and the graph records that they disagree.  Collapsing the two would make
        the four-way decision three-way and would silently retire a fact the
        model only said was contested.
        """
        return self.submit(
            Commit(
                edges=[_edge("contradicts", claim_a, claim_b, t, spans)],
                label=f"contradict:{claim_a}~{claim_b}",
            )
        )

    def add_aliases(self, entity_id: str, surfaces: Sequence[str], label: str = "") -> CommitResult:
        """Grow an entity's alias set — the accumulation §1.5 promised.

        D1's linking decisions are where new surface forms for a known entity
        arrive ("the Charge 3" for an entity created as "my Fitbit"), and this is
        the one payload evolution the validator's id check permits: a rewrite of
        the same entity that differs **only** by a superset of aliases (see
        ``validate._is_alias_growth``).  Anything else under the same id is still
        an id collision and is still refused.
        """
        current = self.store.at().node(entity_id)
        if current is None:
            raise KeyError(f"no entity {entity_id} to grow aliases on")
        canonical = str(current.payload.get(PAYLOAD_NAME, ""))
        grown = set(current.payload.get(PAYLOAD_ALIASES, ()) or ())
        grown.update(s for s in surfaces if s and s != canonical)
        updated = Node(
            node_id=current.node_id,
            ntype=current.ntype,
            payload={**dict(current.payload), PAYLOAD_ALIASES: sorted(grown)},
        )
        return self.submit(
            Commit(nodes=[updated], label=label or f"alias:{entity_id[:12]}")
        )

    def relation(
        self,
        etype: str,
        src: str,
        dst: str,
        t: str,
        spans: Sequence[str],
        confidence: float | None = None,
    ) -> CommitResult | None:
        """D3's commit path: one typed relation edge, **gated by the calibrated
        confidence floor** (decision 9, ``pins.COMMIT_FLOOR['d3_relation']``).

        Returns ``None`` when the confidence is below the floor — a decline, not
        a violation, counted in ``below_floor`` and kept out of the violation
        rate, because the floor working as designed must not read as the
        pipeline misbehaving.  ``confidence=None`` bypasses the floor and exists
        for gold-label replays only, where there is no model confidence to gate
        on.
        """
        from graft.graphbuild.pins import COMMIT_FLOOR

        if confidence is not None and confidence < COMMIT_FLOOR["d3_relation"]:
            self.below_floor += 1
            return None
        return self.submit(
            Commit(
                edges=[_edge(etype, src, dst, t, spans)],
                label=f"relation:{etype}:{src[:8]}->{dst[:8]}",
            )
        )

    def valid_during(
        self,
        claim_id: str,
        start: float | None,
        end: float | None,
        t: str,
        spans: Sequence[str],
        confidence: float | None = None,
    ) -> CommitResult | None:
        """D4's commit path: a ``TimeInterval`` node and the ``valid_during``
        edge onto it, gated by ``pins.COMMIT_FLOOR['d4_temporal']`` (decision 9).

        The interval node's id is content-derived from its bounds, so two claims
        valid during the same interval share one node — which is what makes
        interval-equality queries a node identity check instead of a float
        comparison scattered across readers.
        """
        from graft.graphbuild.pins import COMMIT_FLOOR

        if confidence is not None and confidence < COMMIT_FLOOR["d4_temporal"]:
            self.below_floor += 1
            return None
        interval = Node(
            node_id=node_id_of("TimeInterval", f"{start}\x1f{end}"),
            ntype="TimeInterval",
            payload={"start": start, "end": end},
        )
        return self.submit(
            Commit(
                nodes=[interval],
                edges=[_edge("valid_during", claim_id, interval.node_id, t, spans)],
                label=f"valid_during:{claim_id[:12]}",
            )
        )

    def report(self) -> dict[str, Any]:
        total = self.accepted + self.refused
        return {
            "commits_attempted": total,
            "accepted": self.accepted,
            "refused": self.refused,
            "violation_rate": (self.refused / total) if total else 0.0,
            "below_confidence_floor": self.below_floor,
            "rejections_by_check": dict(sorted(self.rejections.items())),
            "reading": (
                "the validator bounds a specific, enumerable class of damage "
                "(plan §3.2); it cannot reject a semantically wrong same_as and "
                "does not make neural predictions safe. below_confidence_floor "
                "counts decision 9's floor declining a D3/D4 edge — the rule "
                "working, not a malformed write"
            ),
        }


# --------------------------------------------------------------------------
# G9 — the corruption-after-sequential-updates audit
# --------------------------------------------------------------------------


def _live_edges(snapshot: Any) -> dict[str, Edge]:
    edges = getattr(snapshot, "_edges", None)
    if not isinstance(edges, dict):
        return {}
    return {eid: e for eid, e in edges.items() if e.t_invalid is None}


def corruption_audit(
    log: EventLog, commits: Sequence[CommitResult], expected: Sequence[Commit]
) -> dict[str, Any]:
    """G9's three properties, checked after every commit in a stream.

    ``commits`` and ``expected`` are parallel: what the committer did, and what
    it was asked to do.  Property (c) needs both — "the live-edge set changed by
    exactly the committed edge plus its declared invalidations" is only checkable
    against the declaration.

    Returns per-property failure counts and the offending steps.  Failures are
    *counted*, not raised: at Gate 1 this becomes a measured error rate of the
    decoders, and an exception would stop the measurement at the first bad step.
    """
    store = ReplayGraphStore(log)
    steps: list[dict[str, Any]] = []
    failures = {"a_supersession_effective": 0, "b_history_intact": 0, "c_no_collateral": 0}

    for step, (outcome, proposal) in enumerate(zip(commits, expected)):
        if not outcome.accepted or not outcome.seqs:
            continue
        before_seq = min(outcome.seqs)
        after_seq = max(outcome.seqs) + 1
        before = store.at(before_seq)
        after = store.at(after_seq)

        live_before = _live_edges(before)
        live_after = _live_edges(after)

        problems: list[str] = []

        # (a) supersession is effective.
        for edge_id, _, superseded_by in proposal.invalidations:
            if edge_id in live_after:
                problems.append(f"a: superseded edge {edge_id} is still live")
                failures["a_supersession_effective"] += 1
            if superseded_by is not None and superseded_by not in live_after:
                problems.append(f"a: successor {superseded_by} is not live")
                failures["a_supersession_effective"] += 1

        # (b) history is intact — the pre-commit snapshot still serves the old
        # fact.  This is the bi-temporal property, and it is the reason nothing
        # is deleted.
        for edge_id, _, _ in proposal.invalidations:
            if edge_id not in live_before:
                problems.append(f"b: {edge_id} was not live before its own invalidation")
                failures["b_history_intact"] += 1

        # (c) no collateral flips.
        declared_added = {e.edge_id for e in proposal.edges if e.t_invalid is None}
        declared_removed = {eid for eid, _, _ in proposal.invalidations}
        actual_added = set(live_after) - set(live_before)
        actual_removed = set(live_before) - set(live_after)
        stray_added = actual_added - declared_added
        stray_removed = actual_removed - declared_removed
        if stray_added or stray_removed:
            problems.append(
                f"c: undeclared live-edge changes — added {sorted(stray_added)}, "
                f"removed {sorted(stray_removed)}"
            )
            failures["c_no_collateral"] += len(stray_added) + len(stray_removed)

        if problems:
            steps.append({"step": step, "label": proposal.label, "problems": problems})

    return {
        "steps_audited": sum(1 for c in commits if c.accepted and c.seqs),
        "failures": failures,
        "corrupted_steps": steps,
        "green": not any(failures.values()),
        "properties": {
            "a": "a superseded claim's edge is not live, and the superseding one is",
            "b": "at(seq) before the update still returns the old fact live",
            "c": "live edges changed across the commit = the committed edge plus "
            "its declared invalidations, and nothing else",
        },
        "reading": (
            "green here is a property of the *pipeline*; at Gate 1, run on real "
            "decoder outputs, property (c) becomes a measured error rate of the "
            "decoders (G9)"
        ),
    }
