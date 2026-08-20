"""The deterministic stand-in constructor for smoke runs (G1, decision 14).

**A stand-in, not a model — and the distinction is what makes the smoke run
mean anything.**  The corruption audit and the violation tally are properties of
the *pipeline*; run them under a learned decoder and their result depends on
training, which is precisely the confound the smoke run exists to avoid.  So
construction decisions here are the simplest deterministic rules that exercise
every write path:

* every mention resolves by **normalized exact match** against its own
  conversation's entities — link (growing the alias set when the surface form is
  new) when one exists, create otherwise.  No embedding, no scores: exactness is
  the one decision rule with no knob;
* every **eligible** assertion of a turn links to that turn's first mention's
  entity.  Crude on purpose: the point is that ``about_entity`` edges exist, so
  the pair proposer has anchors and D2 items exist, not that the links are good;
* an eligible assertion in a turn with **no mention** is committed as a
  standalone assertion-backed node, with no ``about_entity`` edge (see the
  coverage note below);
* pairs are proposed **at link time against the constructor's own snapshot**,
  which is the G5 ordering ("the anchor for a claim is its D1 output") and the
  G12 leak guard in one move — a proposal can only see what construction has
  already committed.

*(Rebuilt 14 Aug 2026 after the audit: the first smoke driver derived items
before construction — so every candidate list was empty and D2 was zero by
sequence arithmetic, not by finding — and committed bare entity nodes only, so
the graph had zero edges and "exercises every component" was false.  This module
is the fix: items are built during construction, at the position their
candidates were actually drawn from, and the committed graph carries the edges
Phase 7 is promised.)*

**Coverage: every eligible assertion becomes a node** *(decided 19 Aug 2026,
before any end-to-end evaluation number existed — a declared construction
policy, not a response to a result).*

The first version iterated turns drawn from ``mention_records``, so a turn with
no mention was never visited and its assertions were silently unreachable, and
a second guard dropped a turn's assertions outright when no entity was resolved.
Measured on the LoCoMo Stage-A log: of **2,268** eligible assertions, only
**866** sat in a turn carrying a mention and **850** became nodes — the other
**1,402** were never considered, because 1,413 of 5,882 turns carry a mention at
all.  Retrieval cannot return what construction never committed, so roughly
two-thirds of the support-gated evidence was unreachable by arithmetic rather
than by any retrieval or reader property.

Two changes, together: turn order now comes from ``turn.add`` (the write path's
own order, so G12's "candidates see only the past" is unchanged), and an
assertion with no available entity anchor is committed standalone.  An edge-less
assertion-backed node is a **legal pool shape** — ``CandidateAtom`` forbids
``refs`` on node atoms, which is exactly what makes nodes-first construction
always valid — so such a node is retrievable, closes trivially, and is
``H``-selectable.  What it is not is *anchored*: ``pairs_for`` requires a shared
entity anchor and returns nothing without one, so no D2 item is fabricated from
an anchor that does not exist, and the D2 supervision this stand-in produces is
unchanged in kind.

The cost is honest and worth stating: these nodes carry no relational structure,
so the entity channel and the 2-hop expansion cannot reach them and only the
lexical and dense channels can.  That is strictly better than unreachable, and
it is **not** a claim that the stand-in links well — it does not.  Linking is
what Gate 1's learned decoders are for.

**Why candidates here cannot leak the future** (G12, exit criterion 14): the
loop processes turns in Stage-A order and computes a mention's candidates
*before* committing that mention's decision, against ``committer.snapshot()`` —
a snapshot that contains only earlier commits by construction.  Each item
records that position as ``stage_b_seq``, so ``at(stage_b_seq)`` reproduces the
exact graph the candidates were drawn from, checkably.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from graft.eventlog import EventLog
from graft.graphbuild.candidates import candidates_for, normalise_name, pairs_for
from graft.graphbuild.commit import (
    Committer,
    claim_node,
    corruption_audit,
    entity_node,
)
from graft.graphbuild.items import (
    MentionRecord,
    assertions_by_turn,
    build_d1_item,
    build_d2_item,
    mention_records,
)
from graft.graphbuild.validate import Commit
from graft.graphstore import ReplayGraphStore

__all__ = ["construct"]


def construct(
    log: EventLog,
    *,
    embed: Callable[[Sequence[str]], Any] | None = None,
    k: int = 10,
    s: int = 10,
    question_types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the stand-in construction over a Stage-A log, in turn order.

    Returns the smoke run's whole measurable surface: D1/D2 items (with
    construction-time candidates and both sequence pins), the committer's
    report, and the G9 corruption audit over every commit made.
    """
    final = ReplayGraphStore(log).at()
    committer = Committer(log)
    by_turn = assertions_by_turn(log)

    mentions_of_turn: dict[str, list[MentionRecord]] = {}
    for record in mention_records(log):
        mentions_of_turn.setdefault(record.turn_id, []).append(record)

    # **Every turn in Stage-A order, not only the turns that carry a mention**
    # (widened 19 Aug 2026 — see the module docstring's coverage note).  The
    # order comes from `turn.add` so it is the write path's own, and a turn with
    # neither a mention nor an eligible assertion costs one dict lookup.
    turn_order: list[str] = [
        event.payload["turn_id"] for event in log.replay() if event.op == "turn.add"
    ]

    d1: list[dict[str, Any]] = []
    d2: list[dict[str, Any]] = []
    pair_seen: set[tuple[str, str]] = set()

    for turn_id in turn_order:
        mentions = mentions_of_turn.get(turn_id, ())
        assertions = by_turn.get(turn_id, ())
        if not mentions and not assertions:
            continue
        turn = final.turn(turn_id)
        if turn is None:
            continue
        turn_entity: str | None = None

        for record in mentions:
            snapshot = committer.snapshot()
            stage_b_seq = snapshot.snapshot_id
            candidates = candidates_for(record, snapshot, k, embed=embed)
            d1.append(
                build_d1_item(
                    len(d1), record, candidates, turn.text,
                    stage_b_seq=stage_b_seq, turn_ts=turn.ts,
                )
            )

            # The stand-in decision: exact normalized match links, else create.
            exact = [c for c in candidates if c.get("how") == "normalised_exact"]
            if exact:
                entity_id = str(exact[0]["entity_id"])
                node = snapshot.node(entity_id)
                known = {node.payload.get("name", "")} | set(
                    node.payload.get("aliases", ()) or ()
                )
                if record.text not in known:
                    committer.add_aliases(entity_id, [record.text])
            else:
                entity = entity_node(record.text, record.conv_id)
                committer.submit(
                    Commit(nodes=[entity], label=f"entity:{record.text[:30]}")
                )
                entity_id = entity.node_id
            if turn_entity is None:
                turn_entity = entity_id

        # Link the turn's eligible assertions to its first mention's entity, then
        # propose pairs at link time — anchors exist only for already-linked
        # claims, so the proposal sees exactly the constructor's past (G5, G12).
        for aid in assertions:
            assertion = final.assertion(aid)
            if assertion is None or assertion.eligibility != "eligible":
                continue
            if turn_entity is not None:
                outcome = committer.link_existing(
                    aid, assertion.kind, turn_entity, turn.ts, list(assertion.spans)
                )
            else:
                # **Standalone, rather than dropped** (19 Aug 2026).  An
                # assertion the support gate passed is evidence; whether the
                # extractor also produced a *mention* in the same turn is a
                # property of the extractor, not of the claim's admissibility.
                # An edge-less assertion-backed node is a legal pool shape —
                # nodes reference nothing, which is the invariant that makes
                # nodes-first construction always valid — so it is retrievable
                # and `H`-selectable without an anchor.  It carries no
                # `about_entity` edge, so `pairs_for` returns nothing for it and
                # no D2 item is invented out of an anchor that does not exist.
                outcome = committer.submit(
                    Commit(
                        nodes=[claim_node(aid, assertion.kind)],
                        label=f"standalone:{aid}",
                    )
                )
            if not outcome.accepted:
                continue
            snapshot = committer.snapshot()
            for other in pairs_for(assertion, snapshot, s, embed=embed):
                key = tuple(sorted((aid, other.assertion_id)))
                if key in pair_seen or key[0] == key[1]:
                    continue
                pair_seen.add(key)
                a_side = _claim_side(key[0], snapshot)
                b_side = _claim_side(key[1], snapshot)
                if a_side is None or b_side is None:
                    continue
                d2.append(
                    build_d2_item(
                        len(d2),
                        a_side,
                        b_side,
                        question_type=(question_types or {}).get(
                            a_side["question_id"], ""
                        ),
                        stage_b_seq=snapshot.snapshot_id,
                        turn_ts=turn.ts,
                    )
                )

    # The committer records every proposal beside its outcome — convenience
    # methods included — so the audit covers the whole construction, not a
    # hand-captured subset.
    audit = corruption_audit(log, committer.history, committer.proposals)
    return {
        "d1_items": d1,
        "d2_items": d2,
        "commit": committer.report(),
        "corruption_audit": audit,
        "graph": committer.snapshot().counts(),
    }


def _claim_side(assertion_id: str, snapshot: Any) -> dict[str, Any] | None:
    assertion = snapshot.assertion(assertion_id)
    if assertion is None or not assertion.spans:
        return None
    span = snapshot.span(assertion.spans[0])
    turn = snapshot.turn(span.turn_id) if span is not None else None
    if turn is None:
        return None
    return {
        "assertion_id": assertion_id,
        "text": assertion.text_norm,
        "turn_id": turn.turn_id,
        "session_id": turn.session_id,
        "question_id": turn.conv_id,
        "session_date": turn.ts,
    }
