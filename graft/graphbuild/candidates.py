"""P6.3 — D1's candidate generator (G4) and D2's pair proposer (G5, fix F8).

**Both are ceilings, and they live in the same module as the mechanisms that
cause them.**  A linker cannot link to a candidate that was never proposed, and a
conflict never proposed is a conflict never classified — so candidate recall@k
bounds every D1 number and proposer recall bounds every D2 number, exactly the
ceiling discipline of plan §6.3 one level down.  Both are computed here, against
gold when gold is supplied, so that no table can report the decoder's score
without the bound it sits under.

**One `k`, one precedent.**  ``k = 10`` for candidates and ``s = 10`` for pairs
are the *same* constant: fix F8 takes top-*s* = 10 from Mem0's
retrieval-before-update pattern (**[EVIDENCE-adjacent]** — Mem0 retrieves 10
similar memories before its update decision, ECAI 2025), and giving the
candidate generator its own separate knob would be a second tunable with no
second precedent.

**One name normalization, reused.**  ``graft.ingest.grounding.normalise`` is the
project's only casefold-and-collapse, and this module calls it rather than
writing a second — "Sankeien  Garden" and "sankeien garden" becoming different
entities is precisely how a duplicate-entity bug enters, and it enters through
the second implementation.

**Cold start is legal, not an error.**  An empty candidate list is the correct
output for the first mention of a conversation, whose only available actions are
`CREATE_NEW_ENTITY`, `NON_ENTITY` and `DEFER`.  The D1 head must be *trained*
with empty-candidate examples present or it learns that linking is always
possible (G4).

**The embedder is injected, never imported.**  Passing an ``embed`` callable
keeps this module importable on a bare interpreter, which is what lets the
recall plumbing be tested without a GPU — and it is the same fix-F6 shape the
rest of the project uses at component boundaries.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from graft.ingest.grounding import normalise
from graft.schemas import PAYLOAD_ALIASES, PAYLOAD_CONV_ID, PAYLOAD_NAME, Assertion, Node

__all__ = [
    "DEFAULT_K",
    "DEFAULT_S",
    "normalise_name",
    "candidates_for",
    "pairs_for",
    "recall_at_k",
    "proposer_recall",
    "list_reply_rate",
]

#: Fix F8's top-*s* = 10, used for both jobs.  See the module docstring.
DEFAULT_K = 10
DEFAULT_S = 10

#: Embedding callable: ``embed(texts) -> (n, d) float array``.
Embedder = Callable[[Sequence[str]], "np.ndarray"]


def normalise_name(text: str) -> str:
    """The single normalization, borrowed from the grounder.

    ``normalise`` returns ``(text, offset_map)`` because the grounder needs to
    map matches back into the raw turn; here only the text is wanted, and
    unpacking it in one place beats every caller remembering the tuple.
    """
    return normalise(text)[0]


# --------------------------------------------------------------------------
# D1 candidates (G4)
# --------------------------------------------------------------------------


def _entity_nodes(snapshot: Any, conv_id: str | None = None) -> list[Node]:
    """Every ``Entity`` node in the snapshot, scoped to one conversation.

    **The conversation filter is the wrong-merge guard, applied at retrieval as
    well as at creation** *(corrected 14 Aug 2026: the scan was global, which
    re-opened the hole ``entity_node``'s conv-scoped id closes — D1 could link
    one user's "my car" to another user's, and the wrong merge is the most
    damaging Stage-B error, plan §8 risk #1).*  ``conv_id=None`` keeps the
    global scan for callers that genuinely have no conversation (tests over toy
    snapshots); every production caller passes the mention's own ``conv_id``.

    ``GraphSnapshot`` is deliberately minimal (Phase-0 gap G2), so there is no
    ``nodes_of_type``; the concrete store's index is used when present and an
    empty list otherwise, which keeps this usable against any conforming
    snapshot rather than forcing a protocol change on the Phase-2 lattice.
    """
    nodes = getattr(snapshot, "_nodes", None)
    if not isinstance(nodes, dict):
        return []
    out = [n for n in nodes.values() if n.ntype == "Entity"]
    if conv_id is not None:
        out = [n for n in out if n.payload.get(PAYLOAD_CONV_ID) == conv_id]
    return out


def _surface_forms(node: Node) -> list[str]:
    """The entity's canonical name plus its aliases.

    Aliases matter for recall in exactly the case D1 exists for: the same entity
    referred to as "my Fitbit" and "the Charge 3" across sessions.
    """
    forms = [str(node.payload.get(PAYLOAD_NAME, ""))]
    forms.extend(str(a) for a in node.payload.get(PAYLOAD_ALIASES, ()) or ())
    return [f for f in forms if f]


def candidates_for(
    mention: Any,
    snapshot: Any,
    k: int = DEFAULT_K,
    embed: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Top-``k`` existing entities for one mention: normalized-exact, then dense.

    **Exact matches come first and are never displaced by a similarity score.**
    A mention whose normalized form *is* an entity's normalized name or alias is
    the strongest evidence available, and letting a cosine ranking bury it would
    make the generator's recall depend on the embedder's mood.  Dense neighbours
    fill the remaining slots.

    Returns dicts rather than nodes because these go straight into an annotation
    item that a human reads (P6.1) — the entity id alone is not something anyone
    can adjudicate.
    """
    text = getattr(mention, "text", mention)
    conv_id = getattr(mention, "conv_id", None)
    target = normalise_name(str(text))
    entities = sorted(_entity_nodes(snapshot, conv_id), key=lambda n: n.node_id)
    if not entities or not target:
        return []

    exact: list[dict[str, Any]] = []
    rest: list[Node] = []
    for node in entities:
        forms = {normalise_name(f) for f in _surface_forms(node)}
        if target in forms:
            exact.append(
                {
                    "entity_id": node.node_id,
                    "name": node.payload.get(PAYLOAD_NAME, ""),
                    "score": 1.0,
                    "how": "normalised_exact",
                }
            )
        else:
            rest.append(node)

    out = exact[:k]
    if len(out) >= k or embed is None or not rest:
        return out

    names = [_surface_forms(n)[0] for n in rest]
    vectors = embed([str(text)] + names)
    query, matrix = vectors[0], vectors[1:]
    scores = matrix @ query  # both sides are L2-normalised by the embedder pin
    order = np.argsort(-scores)
    for ix in order[: k - len(out)]:
        node = rest[int(ix)]
        out.append(
            {
                "entity_id": node.node_id,
                "name": node.payload.get(PAYLOAD_NAME, ""),
                "score": round(float(scores[int(ix)]), 4),
                "how": "embedding",
            }
        )
    return out


def recall_at_k(
    items: Iterable[Mapping[str, Any]], gold: Mapping[str, str]
) -> dict[str, Any]:
    """Candidate recall against gold links — D1's own ceiling (G4).

    Scored **only over items gold says are links**: an item whose gold action is
    `CREATE_NEW_ENTITY` has no correct candidate to recall, and counting it would
    turn a measure of the generator into a measure of the class balance.

    ``nan`` when gold contains no links at all, rather than a flattering 1.0 —
    the same convention ``slot_level_scores`` uses for an unexercised slot.
    """
    total = hits = 0
    for item in items:
        gold_entity = gold.get(str(item.get("item_id")))
        if not gold_entity:
            continue
        total += 1
        if any(c.get("entity_id") == gold_entity for c in item.get("candidates", ())):
            hits += 1
    return {
        "linkable_items": total,
        "recalled": hits,
        "candidate_recall_at_k": (hits / total) if total else float("nan"),
        "reading": (
            "a linker cannot link to a candidate that was never proposed, so this "
            "bounds every D1 number and is reported beside them (G4)"
        ),
    }


# --------------------------------------------------------------------------
# D2 pair proposer (G5, fix F8)
# --------------------------------------------------------------------------


def _anchor_of(assertion_id: str, snapshot: Any) -> str | None:
    """The entity this claim is about, via its committed ``about_entity`` edge.

    ``None`` means the claim has no linked entity *yet* — its D1 decision has not
    been made, or was `DEFER`.  G5 is explicit that such a claim's pairing parks
    too: pairing on a missing anchor would compare claims that share nothing but
    a corpus.
    """
    for node_id, node in (getattr(snapshot, "_nodes", {}) or {}).items():
        if node.ntype not in ("Claim", "Value", "Event"):
            continue
        if node.payload.get("assertion_id") != assertion_id:
            continue
        for edge in snapshot.edges_of(node_id, "about_entity"):
            if edge.src == node_id and snapshot.is_live(edge.edge_id):
                return edge.dst
    return None


def pairs_for(
    assertion: Assertion,
    snapshot: Any,
    s: int = DEFAULT_S,
    embed: Embedder | None = None,
) -> list[Assertion]:
    """The top-``s`` similar existing claims **sharing an entity anchor** (fix F8).

    Anchor-sharing is the load-bearing filter, and not only for cost: it is what
    suppresses the flood the spike measured — assistant list-replies produce
    piles of `INDEPENDENT` advice pairs (`PHASE2_5_DECISIONS.md`, guidelines v1)
    — at the *proposer*, which is where G5 rules it must be handled rather than
    patched downstream.

    Without an anchor the result is empty, deliberately (see :func:`_anchor_of`).
    """
    anchor = _anchor_of(assertion.assertion_id, snapshot)
    if anchor is None:
        return []

    others: list[Assertion] = []
    for other_id, other in (getattr(snapshot, "_assertions", {}) or {}).items():
        if other_id == assertion.assertion_id or other.eligibility != "eligible":
            continue
        if _anchor_of(other_id, snapshot) == anchor:
            others.append(other)
    if len(others) <= s or embed is None:
        return others[:s]

    vectors = embed([assertion.text_norm] + [o.text_norm for o in others])
    scores = vectors[1:] @ vectors[0]
    order = np.argsort(-scores)[:s]
    return [others[int(i)] for i in order]


def proposer_recall(
    items: Iterable[Mapping[str, Any]], gold_pairs: Iterable[tuple[str, str]]
) -> dict[str, Any]:
    """Proposer recall on gold pairs — D2's ceiling (G5).

    Reported wherever D2 is reported, because a conflict never proposed is a
    conflict never classified and no downstream macro-F1 can see the loss.
    """
    proposed = {
        tuple(sorted((i["claim_a"]["assertion_id"], i["claim_b"]["assertion_id"])))
        for i in items
    }
    gold = {tuple(sorted(p)) for p in gold_pairs}
    hits = len(gold & proposed)
    return {
        "gold_pairs": len(gold),
        "recalled": hits,
        "proposer_recall": (hits / len(gold)) if gold else float("nan"),
        "reading": (
            "a conflict never proposed is a conflict never classified; no D2 "
            "macro-F1 can see this loss, so it is printed beside every D2 number "
            "(G5)"
        ),
    }


def list_reply_rate(items: Iterable[Mapping[str, Any]], speakers: Mapping[str, str]) -> dict[str, Any]:
    """How much of the pair pool comes from assistant turns (G5, exit criterion 8).

    The spike's measured failure mode, quantified rather than assumed: assistant
    list-replies flood the pool with advice pairs that are trivially
    `INDEPENDENT`.  ``speakers`` maps ``turn_id -> speaker``, which the caller has
    from the log.
    """
    items = list(items)
    if not items:
        return {"pairs": 0, "assistant_side_rate": float("nan")}
    assistant = sum(
        1
        for i in items
        if speakers.get(i["claim_a"]["turn_id"]) == "assistant"
        or speakers.get(i["claim_b"]["turn_id"]) == "assistant"
    )
    return {
        "pairs": len(items),
        "pairs_with_an_assistant_side": assistant,
        "assistant_side_rate": assistant / len(items),
        "reading": (
            "the spike found assistant list-replies flood the pair pool with "
            "INDEPENDENT advice pairs; anchor-sharing suppresses most of it and "
            "this is the residual, measured rather than patched blind (G5)"
        ),
    }
