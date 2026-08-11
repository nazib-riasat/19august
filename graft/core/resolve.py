"""Atom → graph resolution, in one place.

Not a module the Phase-1 build plan names. It exists because the checker, the
deficit vector and the utility all have to walk the same three paths — atom to
graph object, atom to backing assertion, atom to provenance — and three copies
of that walk would drift.

**Everything here reads; nothing decides.** These helpers return facts and, where
a chain is broken, say so. Deciding what a broken chain *means* is the caller's
job: `H` treats an unresolvable provenance under a non-empty scope as a
violation, while `U` treats an unresolvable source as the default tier. Keeping
the read and the judgment apart is what stops a lookup helper from quietly
becoming policy.

The two reference systems stay separate throughout (Phase-1 gap G2):
``CandidateAtom.refs`` is pool-internal and never touches the graph;
``CandidateAtom.target`` is the graph object the atom denotes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graft.graphstore import GraphSnapshot
from graft.schemas import (
    ASSERTION_BACKED_NTYPES,
    PAYLOAD_ALIASES,
    PAYLOAD_ASSERTION_ID,
    PAYLOAD_NAME,
    PAYLOAD_TIER,
    PAYLOAD_VALUE_TYPE,
    AtomPool,
    CandidateAtom,
    Edge,
    Interval,
    Node,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.config import Config

__all__ = [
    "target_node",
    "target_edge",
    "target_resolves",
    "endpoint_nodes",
    "assertion_dependencies",
    "provenance_spans",
    "conv_ids",
    "atom_intervals",
    "validity_intervals",
    "source_node",
    "source_tier",
    "matches_anchor",
    "supplies_value_type",
]


# --------------------------------------------------------------------------
# target
# --------------------------------------------------------------------------


def target_node(atom: CandidateAtom, G: GraphSnapshot) -> Node | None:
    """The node this atom denotes, if it denotes one."""
    return G.node(atom.target) if atom.kind == "node" and atom.target else None


def target_edge(atom: CandidateAtom, G: GraphSnapshot) -> Edge | None:
    """The edge this atom denotes, if it denotes one."""
    return G.edge(atom.target) if atom.kind == "edge" and atom.target else None


def target_resolves(atom: CandidateAtom, G: GraphSnapshot) -> bool:
    """False when the atom claims a graph object the snapshot does not hold.

    A binding denotes nothing and always resolves; a node or edge atom with an
    empty target does not, because every one of them is supposed to denote
    something.
    """
    if atom.kind == "binding":
        return True
    if not atom.target:
        return False
    return target_node(atom, G) is not None or target_edge(atom, G) is not None


def endpoint_nodes(atom: CandidateAtom, G: GraphSnapshot) -> tuple[Node, ...]:
    """The nodes an atom touches: itself for a node atom, both ends for an edge."""
    node = target_node(atom, G)
    if node is not None:
        return (node,)
    edge = target_edge(atom, G)
    if edge is None:
        return ()
    return tuple(n for n in (G.node(edge.src), G.node(edge.dst)) if n is not None)


# --------------------------------------------------------------------------
# assertions and provenance
# --------------------------------------------------------------------------


def assertion_dependencies(
    atom: CandidateAtom, G: GraphSnapshot
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(assertion_ids, problems)`` for the assertions this atom leans on.

    A node atom depends on its own backing assertion; an edge atom on those of
    both endpoints; a binding on nothing directly, because its referents are
    themselves selected atoms and are checked in their own right.

    ``problems`` is non-empty when a node of an assertion-backed type
    (``Claim``/``Value``/``Event``) carries no ``assertion_id``, or names one the
    snapshot does not hold. **That is reported rather than skipped**: check 7's
    whole purpose is that unsupported claims cannot reach a proof, so a claim
    with no traceable assertion has to fail rather than pass for lack of
    evidence against it.
    """
    found: list[str] = []
    problems: list[str] = []
    for node in endpoint_nodes(atom, G):
        if node.ntype not in ASSERTION_BACKED_NTYPES:
            continue
        aid = node.payload.get(PAYLOAD_ASSERTION_ID)
        if not aid:
            problems.append(
                f"node {node.node_id} is a {node.ntype} but carries no "
                f"{PAYLOAD_ASSERTION_ID}, so its support cannot be checked"
            )
            continue
        if G.assertion(aid) is None:
            problems.append(
                f"node {node.node_id} names assertion {aid}, which is not in this snapshot"
            )
            continue
        found.append(aid)
    return tuple(dict.fromkeys(found)), tuple(problems)


def provenance_spans(
    atom: CandidateAtom, G: GraphSnapshot, pool: AtomPool | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(span_ids, problems)`` for everything that sourced this atom.

    An edge carries its provenance directly (the schema guarantees it is
    non-empty); a node reaches spans through its backing assertion.

    A binding has no provenance of its own — it denotes nothing in the graph.
    Given ``pool`` it inherits the provenance of its referents, which is what
    makes "does this binding rest on a source?" answerable at all, and therefore
    what ``d_source`` is measuring.
    """
    edge = target_edge(atom, G)
    if edge is not None:
        return tuple(edge.provenance), ()
    if atom.kind == "binding":
        if pool is None:
            return (), ()
        spans: list[str] = []
        problems: list[str] = []
        for ref in atom.refs:
            referent = pool.get(ref)
            if referent is None:
                problems.append(f"binding {atom.atom_id} references {ref}, absent from the pool")
                continue
            ref_spans, ref_problems = provenance_spans(referent, G, pool)
            spans.extend(ref_spans)
            problems.extend(ref_problems)
        return tuple(dict.fromkeys(spans)), tuple(problems)
    aids, problems = assertion_dependencies(atom, G)
    spans = []
    for aid in aids:
        assertion = G.assertion(aid)
        if assertion is not None:
            spans.extend(assertion.spans)
    return tuple(dict.fromkeys(spans)), problems


def conv_ids(
    atom: CandidateAtom, G: GraphSnapshot, pool: AtomPool | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(conv_ids, problems)`` — the conversations this atom's evidence came from.

    ``problems`` records a span or turn the snapshot cannot resolve. Under a
    non-empty scope the caller must treat that as a violation, not a pass: an
    atom whose origin cannot be established has not been shown to be in scope.
    """
    spans, problems = provenance_spans(atom, G, pool)
    convs: list[str] = []
    issues = list(problems)
    for sid in spans:
        span = G.span(sid)
        if span is None:
            issues.append(f"span {sid} is not in this snapshot, so its conversation is unknown")
            continue
        turn = G.turn(span.turn_id)
        if turn is None:
            issues.append(
                f"span {sid} points at turn {span.turn_id}, which is not in this snapshot"
            )
            continue
        convs.append(turn.conv_id)
    return tuple(dict.fromkeys(convs)), tuple(issues)


# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------


def _interval_of_node(node: Node | None) -> Interval | None:
    if node is None or node.ntype != "TimeInterval":
        return None
    try:
        return Interval(start=node.payload.get("start"), end=node.payload.get("end"))
    except ValueError:
        # A malformed stored interval is a graph defect, not a proof defect; the
        # atom simply contributes no temporal evidence and check 1 reports it.
        return None


def atom_intervals(atom: CandidateAtom, G: GraphSnapshot) -> tuple[Interval, ...]:
    """Validity intervals this atom **contributes as evidence**.

    Either the atom denotes a ``TimeInterval`` node directly, or it denotes a
    ``valid_during`` edge pointing at one.

    Selection-based on purpose: this answers "what temporal evidence does the
    proof actually present", which is what ``U``'s ``temporal_correctness`` and
    ``d_time`` grade.  For "what does the graph say this claim's validity *is*",
    which is what a hard contradiction is measured against, see
    :func:`validity_intervals`.
    """
    own = _interval_of_node(target_node(atom, G))
    if own is not None:
        return (own,)
    edge = target_edge(atom, G)
    if edge is not None and edge.etype == "valid_during":
        via = _interval_of_node(G.node(edge.dst))
        if via is not None:
            return (via,)
    return ()


def validity_intervals(atom: CandidateAtom, G: GraphSnapshot) -> tuple[Interval, ...]:
    """Validity the **graph** asserts for what this atom denotes.

    Follows live ``valid_during`` edges out of the atom's node, rather than
    reading only what the proof selected.

    That distinction matters for ``H``'s temporal sub-check.  A proof that binds a
    claim as its answer, where the graph says that claim was only valid outside
    the window the question asks about, is contradictory whether or not the proof
    bothered to select the interval edge.  Making the check selection-dependent
    would let a proof escape it by omitting evidence — the opposite of what a
    formal check is for.
    """
    found: list[Interval] = list(atom_intervals(atom, G))
    node = target_node(atom, G)
    if node is not None:
        for edge in G.edges_of(node.node_id, "valid_during"):
            if edge.src != node.node_id or not G.is_live(edge.edge_id):
                continue
            interval = _interval_of_node(G.node(edge.dst))
            if interval is not None:
                found.append(interval)
    # Deduplicate while keeping order stable.
    return tuple(dict.fromkeys(found))


# --------------------------------------------------------------------------
# source quality
# --------------------------------------------------------------------------


def source_node(atom: CandidateAtom, G: GraphSnapshot) -> Node | None:
    """The ``Source`` node this atom's evidence rests on, or ``None``.

    Split out from :func:`source_tier` so that "did the source resolve?" and
    "what is it worth?" are separately answerable. They are different questions
    and conflating them hides one inside the other: ``source_tier`` returns
    ``default_tier`` for an atom with no source at all, and ``unknown`` is
    *itself* a tier with that same score — so a caller counting distinct scores
    cannot tell a resolved ``unknown`` from an unresolved anything.

    Either the atom touches a ``Source`` node directly, or it reaches one through
    a live ``asserted_by`` edge out of one of its endpoints.
    """
    for node in endpoint_nodes(atom, G):
        if node.ntype == "Source":
            return node
    for node in endpoint_nodes(atom, G):
        for edge in G.edges_of(node.node_id, "asserted_by"):
            if edge.src != node.node_id or not G.is_live(edge.edge_id):
                continue
            source = G.node(edge.dst)
            if source is not None and source.ntype == "Source":
                return source
    return None


def source_tier(atom: CandidateAtom, G: GraphSnapshot, cfg: "Config") -> float:
    """Reliability score in [0, 1] for the atom's source.

    Resolved through a live ``asserted_by`` edge to a ``Source`` node. An
    unresolvable source scores ``default_tier`` rather than 0 — a missing edge
    makes evidence *weak*, and disqualifying it is `H`'s job, not `U`'s.
    """
    default = cfg.source_tiers[cfg.default_tier]
    node = source_node(atom, G)
    if node is None:
        return default
    return cfg.source_tiers.get(node.payload.get(PAYLOAD_TIER, ""), default)


# --------------------------------------------------------------------------
# obligation slots
# --------------------------------------------------------------------------


def matches_anchor(atom: CandidateAtom, G: GraphSnapshot, anchor: str) -> bool:
    """True when this atom selects the ``Entity`` the question is anchored on.

    Matched on the canonical name or any alias. Only a node atom counts: an edge
    touching the entity is *about* it, but selecting the edge without the entity
    would leave the proof with no node to anchor a binding to — and the closure
    rule means selecting the edge already required selecting the node.
    """
    node = target_node(atom, G)
    if node is None or node.ntype != "Entity":
        return False
    if node.payload.get(PAYLOAD_NAME) == anchor:
        return True
    return anchor in tuple(node.payload.get(PAYLOAD_ALIASES, ()))


def supplies_value_type(atom: CandidateAtom, G: GraphSnapshot, value_type: str) -> bool:
    """True when this atom selects a ``Value`` node of the requested type."""
    node = target_node(atom, G)
    return (
        node is not None
        and node.ntype == "Value"
        and node.payload.get(PAYLOAD_VALUE_TYPE) == value_type
    )
