"""P7.3 — the entity-match channel (G3, decision 7).

**The anchor is ``Obligations.entity_anchor`` and nothing else.**  No keyword
fallback, no "if the parser found nothing, guess from the question text": a
guessed anchor is a hallucinated scope one field over, which is the failure
``scope``'s metadata rule exists to prevent.  **No anchor → empty channel**, and
the four other channels carry the question.  That is a declared behaviour, not a
degradation to apologise for — Phase 5 measured the obligation parser's slot
rates and they are part of Stage C's error budget (fix F2: the parser's measured
quality is reported wherever coverage is).

**Matching is normalised-exact over names *and* aliases.**  One shared
normalisation — ``graphbuild.candidates.normalise_name``, itself the grounder's —
because a second one would make Stage B and Stage C disagree about whether "my
Fitbit" and "the Charge 3" are the same string, and the alias set is exactly what
D1 exists to populate.  No substring or fuzzy matching: this channel is the
*precise* one, the dense channel is the fuzzy one, and blurring them would leave
the G7 table unable to attribute a recall change to either.

**Conversation-scoped, like the candidate generator.**  ``conv_id`` is the
wrong-merge guard, and Phase 6 had to add it retroactively after a global scan
re-opened the hole (`graphbuild.candidates._entity_nodes`, corrected 14 Aug 2026:
D1 could link one user's "my car" to another's, plan §8 risk #1).  Applying it
here from the start is cheaper than the same correction a third time.

**Scores are 1.0, uniformly, and that is the decision.**  This channel answers a
*membership* question — is this assertion about the anchored entity — and it has
no basis to rank within its own hits.  Inventing a spread (edge count, recency)
would put an unjustified ranking into the fused ``max``.  Under G5's declared
min–max the degenerate ``max == min`` case maps to 1.0, so the channel's hits all
arrive at full strength, which is the right answer for a recall stage.
"""

from __future__ import annotations

from typing import Any

from graft.graphbuild.candidates import normalise_name
from graft.retrieve.pool import eligible_nodes
from graft.schemas import (
    ASSERTION_BACKED_NTYPES,
    PAYLOAD_ALIASES,
    PAYLOAD_CONV_ID,
    PAYLOAD_NAME,
    Obligations,
)

__all__ = ["match_entities", "entity_channel"]


def _surface_forms(node: Any) -> list[str]:
    """Canonical name plus aliases — the shape ``graphbuild.candidates`` hands over."""
    forms = [str(node.payload.get(PAYLOAD_NAME, ""))]
    forms.extend(str(a) for a in node.payload.get(PAYLOAD_ALIASES, ()) or ())
    return [f for f in forms if f]


def match_entities(snapshot: Any, anchor: str | None, conv_id: str | None = None) -> tuple[str, ...]:
    """``Entity`` node ids whose name or any alias normalises to ``anchor``.

    Public because the expansion walk (P7.5) seeds from exactly these, and a
    second implementation of "which entities does this anchor name" would be a
    second place for the normalisation to drift.
    """
    if not anchor:
        return ()
    target = normalise_name(str(anchor))
    if not target:
        return ()
    nodes = getattr(snapshot, "_nodes", None)
    if not isinstance(nodes, dict):
        return ()
    out: list[str] = []
    for node in nodes.values():
        if node.ntype != "Entity":
            continue
        if conv_id is not None and node.payload.get(PAYLOAD_CONV_ID) != conv_id:
            continue
        if any(normalise_name(f) == target for f in _surface_forms(node)):
            out.append(node.node_id)
    return tuple(sorted(out))


def entity_channel(
    snapshot: Any,
    obligations: Obligations,
    conv_id: str | None = None,
    *,
    seeds: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Anchor → matched entities → their live edges' assertion-backed endpoints.

    Returns ``node_id -> score`` over assertion-backed nodes, the shape every
    channel in this package emits (see the package docstring for why the protocol
    is keyed by node id rather than atom id).

    Every incident **live** edge is followed, not just ``about_entity``: an
    ``Entity -> Value`` ``has_value`` edge points the other way, and taking
    whichever endpoint is not the entity picks up both without enumerating the
    edge vocabulary here — which would be a fifth place to update when
    :data:`graft.schemas.ENDPOINT_TABLE` changes.

    ``seeds`` lets a caller that already matched the anchor (the runner matches
    once and hands the same seeds to the expansion walk) skip the second
    ``match_entities`` scan — before this, matching ran twice per question and
    its cost landed in *both* channels' latency rows (15 Aug 2026 audit).
    """
    if seeds is None:
        seeds = match_entities(snapshot, obligations.entity_anchor, conv_id)
    if not seeds:
        return {}
    # **Eligibility and scope, filtered here as well as at assembly** (15 Aug
    # 2026 audit).  This channel follows *edges*, and a live edge can reach a
    # quarantined assertion's node or another conversation's — ``build_pool``
    # would refuse both, but only by inflating ``hits_refused_ineligible``, the
    # count that exists to flag quarantine leakage, and the refused hits would
    # still pollute this channel's G7 recall row and its ``unique`` count.  The
    # text channels already filter through ``eligible_nodes``; this one now does
    # the same, so all four emit over one candidate space.
    allowed = set(eligible_nodes(snapshot, conv_id))
    hits: dict[str, float] = {}
    for entity_id in seeds:
        for edge in snapshot.edges_of(entity_id):
            if not snapshot.is_live(edge.edge_id):
                continue
            other = edge.dst if edge.src == entity_id else edge.src
            if other == entity_id:
                continue
            if snapshot.ntype(other) in ASSERTION_BACKED_NTYPES and other in allowed:
                hits[other] = 1.0
    return dict(sorted(hits.items()))
