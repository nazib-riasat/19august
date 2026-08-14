"""P6.0 — the commit validator: the seven formally checkable constraints (G8).

Architecture §6.3 specifies propose → schema-validate → commit, rejecting *only
formally checkable violations*.  That phrase needed an enumerated list or the
validator would be a mood rather than a contract, and this module is the list.

**What it cannot do, said first.**  Plan §3.2 is explicit and the wording matters
for the write-up: the validator **bounds a specific, enumerable class of damage**.
It cannot reject a semantically wrong ``same_as`` — a merge of two different
people with similar names is type-correct, provenance-bearing, id-unique and
completely wrong — and describing this as "making neural predictions safe" would
be exactly the overreach `CLAUDE.md` §5 catalogues.  What it guarantees is that
*no committed edge is malformed in any of the seven ways below*, and that
guarantee is worth having precisely because it is narrow enough to be true.

**The seven checks** (G8, decision 3):

1. **endpoint typing** against :data:`graft.schemas.ENDPOINT_TABLE`;
2. **id determinism and uniqueness** — a commit that would overwrite a
   *different* payload under an existing id is a violation, while rewriting an
   identical one is idempotent replay and is fine;
3. **provenance non-empty and resolvable** — ``Edge`` already refuses the empty
   case, so what is added here is that every span id actually resolves in the
   snapshot;
4. **eligibility** — a node of an :data:`ASSERTION_BACKED_NTYPES` type may enter
   the active graph only if its backing assertion ``is_eligible``.  This is fix
   F9's boundary, re-enforced at commit so a quarantined assertion cannot become
   retrievable evidence through a decoder's enthusiasm;
5. **interval arithmetic** — ``t_invalid`` strictly after ``t_created``;
   ``superseded_by`` implies ``t_invalid`` (already in ``Edge``); a
   ``valid_during`` target must be a well-formed, non-empty half-open interval;
6. **no self-loops** on :data:`graft.schemas.SELF_LOOP_FORBIDDEN`, and **no
   duplicate live edge** of the same type between the same endpoints;
7. **non-destructive supersession** — a supersession must invalidate the edge it
   replaces rather than removing it (Zep's edge-invalidation precedent —
   vendor-authored preprint, flagged as everywhere it appears).

**Violations carry reasons, never a bare bool.**  The Phase-1 precedent: a
failing ``CheckResult`` must say why, because an unexplained rejection cannot be
counted by category or debugged — and the *by-category* rejection tally is what
exit criterion 1 reports.

**No ML import, ever.**  This module decides what may enter the active graph, so
it sits on the same side of the line as ``H``: if a learned score could reach it,
"formally checkable" would stop meaning anything.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from graft.graphstore import GraphSnapshot
from graft.schemas import (
    ASSERTION_BACKED_NTYPES,
    EDGE_TYPES,
    ENDPOINT_TABLE,
    NODE_TYPES,
    PAYLOAD_ALIASES,
    PAYLOAD_ASSERTION_ID,
    SELF_LOOP_FORBIDDEN,
    CheckResult,
    Edge,
    Node,
    Violation,
)

__all__ = [
    "CHECKS",
    "CHECK_ENDPOINT",
    "CHECK_ID",
    "CHECK_PROVENANCE",
    "CHECK_ELIGIBILITY",
    "CHECK_INTERVAL",
    "CHECK_DUPLICATE",
    "CHECK_SUPERSESSION",
    "Commit",
    "validate",
]

CHECK_ENDPOINT = "endpoint_typing"
CHECK_ID = "id_uniqueness"
CHECK_PROVENANCE = "provenance"
CHECK_ELIGIBILITY = "eligibility"
CHECK_INTERVAL = "interval"
CHECK_DUPLICATE = "duplicate_edge"
CHECK_SUPERSESSION = "supersession"

#: In application order.  Exported so a rejection tally can be built over named
#: categories rather than by matching message strings — the same reason
#: ``core.checker.CHECKS`` exists.
CHECKS: tuple[str, ...] = (
    CHECK_ENDPOINT,
    CHECK_ID,
    CHECK_PROVENANCE,
    CHECK_ELIGIBILITY,
    CHECK_INTERVAL,
    CHECK_DUPLICATE,
    CHECK_SUPERSESSION,
)


class Commit:
    """One proposed write unit: nodes, edges, and edges to invalidate.

    Not a dataclass (Phase-0 criterion 12) and not a bare tuple, because a commit
    is validated *as a unit*: an edge may legally reference a node that does not
    exist in the snapshot yet **provided this same commit creates it**, and a
    validator that saw only one at a time would reject every real write.

    **Atomic by resubmission, not by write** *(worded precisely after the
    14 Aug 2026 audit; an earlier draft said "atomic write")*.  The committer
    appends N events with no transaction, so a crash mid-commit can leave a
    prefix — deliberately ordered adds-then-invalidations, so the torn state is
    the *recoverable* one (old and new both live and visible) rather than the
    silent one (a fact invalidated with its successor missing).  Re-submitting
    the same commit converges: identical rewrites are idempotent under the id
    check, a live edge equal to itself is not a duplicate, and re-invalidation
    with the same timestamp is a no-op.  The regression test walks a torn
    commit to convergence.

    ``invalidations`` carries ``(edge_id, t_invalid, superseded_by | None)``.
    Supersession is expressed here rather than by mutating an ``Edge`` because
    nothing is ever deleted or rewritten in place — the log gets an
    ``edge.invalidate`` op and the old edge stays readable at every earlier
    snapshot.
    """

    __slots__ = ("nodes", "edges", "invalidations", "label")

    def __init__(
        self,
        nodes: Iterable[Node] = (),
        edges: Iterable[Edge] = (),
        invalidations: Iterable[tuple[str, str, str | None]] = (),
        label: str = "",
    ) -> None:
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.invalidations = tuple(
            (eid, ts, sup if sup else None) for eid, ts, sup in invalidations
        )
        self.label = label

    def __bool__(self) -> bool:
        return bool(self.nodes or self.edges or self.invalidations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "invalidations": [list(i) for i in self.invalidations],
        }


def _ntype_of(node_id: str, commit: Commit, snapshot: GraphSnapshot) -> str | None:
    """The node's type, looking in the commit *before* the snapshot.

    Order matters: an edge and its endpoints usually arrive in the same commit,
    and resolving against the snapshot alone would make every first-time write
    fail endpoint typing.
    """
    for node in commit.nodes:
        if node.node_id == node_id:
            return node.ntype
    return snapshot.ntype(node_id)


def _check_endpoints(commit: Commit, snapshot: GraphSnapshot) -> list[Violation]:
    out: list[Violation] = []
    for edge in commit.edges:
        if edge.etype not in ENDPOINT_TABLE:
            out.append(
                Violation(
                    CHECK_ENDPOINT,
                    f"edge {edge.edge_id} has type {edge.etype!r}, which is not in "
                    f"the endpoint table ({sorted(EDGE_TYPES)})",
                    (edge.edge_id,),
                )
            )
            continue
        allowed_src, allowed_dst = ENDPOINT_TABLE[edge.etype]
        for endpoint, node_id, allowed in (
            ("src", edge.src, allowed_src),
            ("dst", edge.dst, allowed_dst),
        ):
            ntype = _ntype_of(node_id, commit, snapshot)
            if ntype is None:
                out.append(
                    Violation(
                        CHECK_ENDPOINT,
                        f"edge {edge.edge_id} ({edge.etype}) has {endpoint} "
                        f"{node_id}, which neither this commit nor the snapshot "
                        "defines; an edge to a node that does not exist is a "
                        "dangling reference, not a fact",
                        (edge.edge_id,),
                    )
                )
            elif ntype not in allowed:
                out.append(
                    Violation(
                        CHECK_ENDPOINT,
                        f"edge {edge.edge_id}: {edge.etype} may not take a "
                        f"{ntype} as {endpoint} (allowed: {list(allowed)})",
                        (edge.edge_id,),
                    )
                )
    return out


def _is_alias_growth(existing: Node, new: Node) -> bool:
    """The one payload evolution the id check permits: an ``Entity`` rewrite
    identical in every field except a **superset** of aliases.

    D1's linking decisions accumulate surface forms onto known entities
    (`PHASE6_DECISIONS.md` §1.5's promised shape, and what Phase 7's
    entity-match channel indexes), and alias growth cannot change identity —
    the id derives from the canonical name, which must be unchanged here.
    Anything else under the same id is still a collision.  Replay is
    last-write-wins and resubmission is monotone, so the final state is the
    union regardless of crash timing.
    """
    if existing.ntype != "Entity" or new.ntype != "Entity":
        return False
    old_p, new_p = dict(existing.payload), dict(new.payload)
    old_aliases = set(old_p.pop(PAYLOAD_ALIASES, ()) or ())
    new_aliases = set(new_p.pop(PAYLOAD_ALIASES, ()) or ())
    return old_p == new_p and new_aliases >= old_aliases


def _check_ids(commit: Commit, snapshot: GraphSnapshot) -> list[Violation]:
    """Ids are content-derived, so a *collision with different content* is the bug.

    Rewriting an identical record is idempotent replay and is explicitly allowed
    — the Phase-5 write path depends on it, and refusing it would make crash
    recovery impossible.  What is refused is the same id carrying *different*
    content, which means either an id derivation lost a distinguishing field or
    two genuinely different objects hashed together.  One deliberate exception:
    :func:`_is_alias_growth`.

    **One assertion, one live node** *(added 14 Aug 2026 — the audit measured
    one assertion entering the active graph as two live nodes with two live
    ``about_entity`` edges and zero violations, via the kind→ntype mapping: the
    same assertion committed once as a ``Claim`` and once as a ``Value`` derives
    two node ids, and every per-node check passes)*.  An assertion is one fact;
    two nodes for it are the same double-counting a duplicate edge is refused
    for, one level up.
    """
    out: list[Violation] = []
    seen_nodes: dict[str, Node] = {}
    for node in commit.nodes:
        if node.node_id in seen_nodes and seen_nodes[node.node_id].to_dict() != node.to_dict():
            out.append(
                Violation(
                    CHECK_ID,
                    f"node {node.node_id} appears twice in one commit with "
                    "different payloads",
                    (node.node_id,),
                )
            )
        seen_nodes[node.node_id] = node
        existing = snapshot.node(node.node_id)
        if (
            existing is not None
            and existing.to_dict() != node.to_dict()
            and not _is_alias_growth(existing, node)
        ):
            out.append(
                Violation(
                    CHECK_ID,
                    f"node {node.node_id} already exists with a different payload; "
                    "a content-derived id carrying two contents means the "
                    "derivation lost a distinguishing field",
                    (node.node_id,),
                )
            )

    # One assertion, one node: a second assertion-backed node for an assertion
    # that already backs a *different* node is refused, whichever ntype the
    # kind mapping produced.
    backing: dict[str, str] = {}
    all_nodes = getattr(snapshot, "_nodes", None)
    if isinstance(all_nodes, dict):
        for other in all_nodes.values():
            if other.ntype in ASSERTION_BACKED_NTYPES:
                aid = other.payload.get(PAYLOAD_ASSERTION_ID)
                if aid:
                    backing[str(aid)] = other.node_id
    for node in commit.nodes:
        if node.ntype not in ASSERTION_BACKED_NTYPES:
            continue
        aid = node.payload.get(PAYLOAD_ASSERTION_ID)
        if not aid:
            continue  # the eligibility check reports the missing id
        holder = backing.get(str(aid))
        if holder is not None and holder != node.node_id:
            out.append(
                Violation(
                    CHECK_ID,
                    f"assertion {aid} already backs node {holder}; committing "
                    f"{node.node_id} ({node.ntype}) would give one fact two live "
                    "nodes — the kind→ntype mapping applied twice is still one "
                    "assertion",
                    (node.node_id,),
                )
            )
        backing[str(aid)] = node.node_id

    seen_edges: dict[str, Edge] = {}
    for edge in commit.edges:
        if edge.edge_id in seen_edges and seen_edges[edge.edge_id].to_dict() != edge.to_dict():
            out.append(
                Violation(
                    CHECK_ID,
                    f"edge {edge.edge_id} appears twice in one commit with "
                    "different content",
                    (edge.edge_id,),
                )
            )
        seen_edges[edge.edge_id] = edge
        existing_edge = snapshot.edge(edge.edge_id)
        if existing_edge is not None and existing_edge.to_dict() != edge.to_dict():
            out.append(
                Violation(
                    CHECK_ID,
                    f"edge {edge.edge_id} already exists with different content",
                    (edge.edge_id,),
                )
            )
    return out


def _check_provenance(commit: Commit, snapshot: GraphSnapshot) -> list[Violation]:
    """Every provenance span must resolve.  ``Edge`` guarantees non-empty; this
    guarantees non-fictional."""
    out: list[Violation] = []
    for edge in commit.edges:
        for span_id in edge.provenance:
            if snapshot.span(span_id) is None:
                out.append(
                    Violation(
                        CHECK_PROVENANCE,
                        f"edge {edge.edge_id} cites span {span_id}, which the "
                        "snapshot does not contain; unresolvable provenance is "
                        "indistinguishable from none",
                        (edge.edge_id,),
                    )
                )
    return out


def _check_eligibility(commit: Commit, snapshot: GraphSnapshot) -> list[Violation]:
    """Fix F9's boundary, at the one place a decoder could cross it.

    Phase 5 decides eligibility and Phase 1's `H` reads it; this is the third
    reader, and it exists because the commit pipeline is where an assertion
    becomes *graph* — the moment after which retrieval can reach it.  Re-checking
    a flag two other components already respect is deliberate redundancy at a
    boundary, not distrust of them.
    """
    out: list[Violation] = []
    for node in commit.nodes:
        if node.ntype not in ASSERTION_BACKED_NTYPES:
            continue
        assertion_id = node.payload.get(PAYLOAD_ASSERTION_ID)
        if not assertion_id:
            out.append(
                Violation(
                    CHECK_ELIGIBILITY,
                    f"node {node.node_id} is a {node.ntype} with no "
                    f"{PAYLOAD_ASSERTION_ID}; an assertion-backed node with no "
                    "assertion cannot be support-gated at all",
                    (node.node_id,),
                )
            )
            continue
        if not snapshot.is_eligible(str(assertion_id)):
            out.append(
                Violation(
                    CHECK_ELIGIBILITY,
                    f"node {node.node_id} is backed by assertion {assertion_id}, "
                    "which is not eligible; a quarantined assertion may not enter "
                    "the active graph (architecture fix F9)",
                    (node.node_id,),
                )
            )
    return out


def _check_intervals(commit: Commit, snapshot: GraphSnapshot) -> list[Violation]:
    out: list[Violation] = []
    for edge in commit.edges:
        if edge.t_invalid is not None and edge.t_invalid <= edge.t_created:
            out.append(
                Violation(
                    CHECK_INTERVAL,
                    f"edge {edge.edge_id} is invalid at {edge.t_invalid}, which is "
                    f"not after its creation {edge.t_created}",
                    (edge.edge_id,),
                )
            )

    for edge_id, t_invalid, _ in commit.invalidations:
        existing = snapshot.edge(edge_id)
        if existing is None:
            out.append(
                Violation(
                    CHECK_INTERVAL,
                    f"cannot invalidate {edge_id}: the snapshot has no such edge",
                    (edge_id,),
                )
            )
        elif t_invalid <= existing.t_created:
            out.append(
                Violation(
                    CHECK_INTERVAL,
                    f"invalidating {edge_id} at {t_invalid} is not after its "
                    f"creation {existing.t_created}; a fact cannot stop holding "
                    "before it started",
                    (edge_id,),
                )
            )

    # A ``valid_during`` target must be a well-formed, non-empty half-open
    # interval.  Empty is refused for the Phase-1 G5 reason: no instant satisfies
    # it, so it is not an answerable temporal claim.
    for edge in commit.edges:
        if edge.etype != "valid_during":
            continue
        node = _node_in(edge.dst, commit, snapshot)
        if node is None:
            continue  # endpoint typing already reported the dangling reference
        start, end = node.payload.get("start"), node.payload.get("end")
        if start is not None and end is not None and float(end) <= float(start):
            out.append(
                Violation(
                    CHECK_INTERVAL,
                    f"valid_during edge {edge.edge_id} targets TimeInterval "
                    f"{edge.dst} with [{start}, {end}), which is empty or "
                    "reversed; a half-open interval that contains no instant "
                    "cannot be satisfied",
                    (edge.edge_id,),
                )
            )
    return out


def _node_in(node_id: str, commit: Commit, snapshot: GraphSnapshot) -> Node | None:
    for node in commit.nodes:
        if node.node_id == node_id:
            return node
    return snapshot.node(node_id)


def _check_duplicates(commit: Commit, snapshot: GraphSnapshot) -> list[Violation]:
    """Self-loops on the three reflexive-meaningless types, and duplicate live edges.

    "Duplicate" is scoped to *live* edges deliberately: re-asserting a relation
    whose previous edge was invalidated is a legitimate re-instatement, and
    refusing it would make a superseded fact unrecoverable — the opposite of what
    non-destructive versioning is for.
    """
    out: list[Violation] = []
    for edge in commit.edges:
        if edge.etype in SELF_LOOP_FORBIDDEN and edge.src == edge.dst:
            out.append(
                Violation(
                    CHECK_DUPLICATE,
                    f"edge {edge.edge_id} is a {edge.etype} self-loop on "
                    f"{edge.src}; nothing is same_as, contradicts or supersedes "
                    "itself",
                    (edge.edge_id,),
                )
            )

    live: dict[tuple[str, str, str], str] = {}
    for existing_id in _live_edge_ids(snapshot):
        existing = snapshot.edge(existing_id)
        if existing is not None:
            live[(existing.etype, existing.src, existing.dst)] = existing_id

    invalidated = {eid for eid, _, _ in commit.invalidations}
    for edge in commit.edges:
        if edge.t_invalid is not None:
            continue
        key = (edge.etype, edge.src, edge.dst)
        clash = live.get(key)
        if clash is not None and clash != edge.edge_id and clash not in invalidated:
            out.append(
                Violation(
                    CHECK_DUPLICATE,
                    f"edge {edge.edge_id} duplicates live edge {clash} "
                    f"({edge.etype}: {edge.src} -> {edge.dst}); the same relation "
                    "asserted twice is one relation counted twice",
                    (edge.edge_id,),
                )
            )
        live[key] = edge.edge_id
    return out


def _live_edge_ids(snapshot: GraphSnapshot) -> tuple[str, ...]:
    """Every live edge id.

    ``GraphSnapshot`` is deliberately minimal (Phase-0 gap G2: every method added
    is a method the Phase-2 lattice must also implement), so there is no
    ``all_edges``.  ``DictGraphSnapshot`` has the private index; the protocol
    does not, and falling back to an empty tuple keeps the validator usable
    against any conforming snapshot rather than blocking on a protocol change.
    """
    edges = getattr(snapshot, "_edges", None)
    if isinstance(edges, dict):
        return tuple(eid for eid, e in edges.items() if e.t_invalid is None)
    return ()


def _check_supersession(commit: Commit, snapshot: GraphSnapshot) -> list[Violation]:
    """Supersession invalidates and links; it never removes.

    The commit-level statement of the property ``Edge`` enforces per record: an
    invalidation naming a superseding edge must actually have that edge, and a
    ``supersedes`` edge must come with the invalidation it implies — otherwise
    the graph would carry two live contradictory claims and ``is_live`` would
    stop being the single authority on what still holds.
    """
    out: list[Violation] = []
    committed = {e.edge_id for e in commit.edges} | {
        eid for eid in _all_edge_ids(snapshot)
    }
    for edge_id, _, superseded_by in commit.invalidations:
        if superseded_by is not None and superseded_by not in committed:
            out.append(
                Violation(
                    CHECK_SUPERSESSION,
                    f"invalidation of {edge_id} names successor {superseded_by}, "
                    "which neither this commit nor the snapshot contains",
                    (edge_id,),
                )
            )
    return out


def _all_edge_ids(snapshot: GraphSnapshot) -> tuple[str, ...]:
    edges = getattr(snapshot, "_edges", None)
    if isinstance(edges, dict):
        return tuple(edges)
    return ()


_VALIDATORS = {
    CHECK_ENDPOINT: _check_endpoints,
    CHECK_ID: _check_ids,
    CHECK_PROVENANCE: _check_provenance,
    CHECK_ELIGIBILITY: _check_eligibility,
    CHECK_INTERVAL: _check_intervals,
    CHECK_DUPLICATE: _check_duplicates,
    CHECK_SUPERSESSION: _check_supersession,
}


def validate(commit: Commit, snapshot: GraphSnapshot) -> CheckResult:
    """Run all seven checks and compose one :class:`CheckResult`.

    Every check runs even after one fails, because the rejection *categories* are
    a reported number (exit criterion 1) and short-circuiting would make the
    tally depend on check order.
    """
    violations: list[Violation] = []
    for name in CHECKS:
        violations.extend(_VALIDATORS[name](commit, snapshot))
    if not violations:
        return CheckResult(ok=True)
    return CheckResult(ok=False, violations=tuple(violations))


def tally(results: Sequence[CheckResult]) -> Mapping[str, int]:
    """Rejections by category, every category present even at zero.

    Zeros are in the table for the same reason the grounding ladder reports
    them: "no endpoint violations" and "endpoint typing was never checked" must
    not look the same in a report.
    """
    counts = {name: 0 for name in CHECKS}
    for result in results:
        for violation in result.violations:
            counts[violation.check] = counts.get(violation.check, 0) + 1
    return counts
