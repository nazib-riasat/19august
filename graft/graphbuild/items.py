"""P6.1 — D1 and D2 annotation items, derived from the permanent record.

**From the log, never by re-extraction.** Exit criterion 7 says D1 items derive
from stored mention spans alone, and the reason is not tidiness: re-running the
extractor to rebuild items would produce *different* items every time the model,
the prompt or the token budget moved, so a hand label collected in week 1 would
silently stop referring to anything in week 3. The event log is what makes a
label durable, and content-derived ids are what make it re-findable.

**The Phase-2.5 item shapes are kept, deliberately.** The spike's guidelines
(`data/phase2_5/GUIDELINES_D{1,2}_v0.md`), its 21 flagged examples and its
`annotate.py` CLI all key on these fields, and the annotator has already used
that tool. Changing the shape here would strand all three
(`PHASE5_DECISIONS.md` §6). The one addition is `snapshot_seq` — see below.

**Two sequence numbers per item, because Stage A and Stage B are two streams**
(corrected 14 Aug 2026 — the audit's two blockers were one number serving both).
``snapshot_seq`` is the item's **Stage-A** position: the pin for
ingestion-stream features. ``stage_b_seq`` is the **construction log's**
position at the moment candidates/pairs were computed, supplied by the
constructor (``graphbuild.standin``): ``at(stage_b_seq)`` on the construction
log reproduces exactly the graph the candidate list was drawn from — G12's
leak guard made checkable per item. A candidate list computed against
``at(snapshot_seq)`` sees no entities at all (Stage-B events append after the
whole Stage-A prefix); one computed against the *final* graph sees entities
created later. Only the constructor's own live snapshot is right, which is why
candidate generation lives with construction order, not here.

**What the pilot actually measured, and what it means for sizing** (G1, and
`PHASE5_DECISIONS.md` §2.2a): 248 turns yielded **187 mentions** (0.754/turn) and
**222 assertions of which 151 are eligible** (0.895 and 0.609/turn). The spike's
0.59 mentions/turn and the bakeoff slice's 1.13 are both superseded for planning
— the pilot's is the production recipe's rate. **No multi-span assertion exists
in any real run** (0 of 222 live, 0 of 78 replay), so nothing here may assume
one; the shape is supported and untested by the corpus, and that is recorded
rather than designed around.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from graft.eventlog import EventLog
from graft.graphstore import ReplayGraphStore
from graft.schemas import Assertion, Turn

__all__ = [
    "D1_ACTIONS",
    "D2_LABELS",
    "MentionRecord",
    "mention_records",
    "assertions_by_turn",
    "build_d1_item",
    "build_d2_item",
    "adjudication_items",
    "d1_items",
    "d2_items",
    "write_items",
    "yields",
]

#: The four-way open-world action space (plan §3.2's correction).
#: **[EVIDENCE]** Learn to Not Link (Findings ACL 2023) partitions unlinkable
#: mentions into Missing-Entity vs Non-Entity; a single NIL head conflates a real
#: new entity (must create a node) with a non-entity phrase (must create
#: nothing), and in a growing memory graph the first is the common case.
D1_ACTIONS: tuple[str, ...] = (
    "LINK_EXISTING(<entity_id>)",
    "CREATE_NEW_ENTITY",
    "NON_ENTITY",
    "DEFER",
)

#: D2's four-way mutually exclusive pair decision (plan §3.2's decoder grouping).
#: Grouped rather than split into independent heads because duplicate, conflict
#: and supersession all depend on the *same* claim pair and the *same* interval —
#: predicting them independently discards their mutual exclusivity and produces
#: contradictory outputs (`CLAUDE.md` §4.1).
D2_LABELS: tuple[str, ...] = ("INDEPENDENT", "DUPLICATE", "CONFLICT", "SUPERSEDES")


class MentionRecord:
    """One stored mention, with everything an item or a feature needs.

    Built from the log's **``mention.add`` events** — the op Phase 5 added
    precisely so that mentions would not have to be reconstructed by inference.
    *(Corrected 14 Aug 2026: this module originally recovered mentions by set
    difference — "a span no assertion cites is a mention" — which is the exact
    heuristic `PHASE5_DECISIONS.md` §7.1 retired: a mention whose grounded span
    coincides with an assertion quote's span collapses to one content-derived
    span id and vanishes from the set difference, and a crash-repaired turn's
    rewritten span appears twice.  Measured: both cases reproduce through the
    real write path.  The two derivations agreed on the pilot log by luck —
    187 = 187 — which is why nothing failed until the audit constructed the
    divergence.)*
    """

    __slots__ = ("span_id", "turn_id", "conv_id", "session_id", "start", "end", "text", "rung", "seq")

    def __init__(
        self,
        span_id: str,
        turn: Turn,
        start: int,
        end: int,
        rung: str,
        seq: int,
    ) -> None:
        self.span_id = span_id
        self.turn_id = turn.turn_id
        self.conv_id = turn.conv_id
        self.session_id = turn.session_id
        self.start = int(start)
        self.end = int(end)
        self.text = turn.text[self.start : self.end]
        self.rung = str(rung)
        self.seq = int(seq)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "turn_id": self.turn_id,
            "conv_id": self.conv_id,
            "session_id": self.session_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "rung": self.rung,
            "snapshot_seq": self.seq,
        }


def mention_records(log: EventLog) -> list[MentionRecord]:
    """Every stored mention, in log order, from ``mention.add`` events alone.

    Deduplicated on ``span_id`` (first occurrence wins), mirroring
    ``graft.ingest.pipeline.mentions_of`` — a crash-repaired turn legitimately
    rewrites its mention events, and without the dedup one mention becomes two
    annotation items and ``yields()`` double-counts.

    ``seq`` is the mention event's sequence number — a **Stage-A** position.
    It locates the mention in the ingestion stream (which turn's context it
    belongs to) and is the right pin for Stage-A-side features.  It is *not* a
    handle on construction state: Stage-B commits append after the whole
    Stage-A prefix, so ``at(seq)`` can never contain an entity.  Candidates are
    therefore computed **by the constructor, against its own live snapshot**
    (see ``graphbuild.standin`` and G12) — the audit's two blockers were this
    module trying to serve both purposes with one number.
    """
    snapshot = ReplayGraphStore(log).at()

    out: list[MentionRecord] = []
    seen: set[str] = set()
    for event in log.replay():
        if event.op != "mention.add":
            continue
        span_id = event.payload["span_id"]
        if span_id in seen:
            continue
        seen.add(span_id)
        span = snapshot.span(span_id)
        turn = snapshot.turn(event.payload["turn_id"]) if span is not None else None
        if span is None or turn is None:
            # A mention whose span or turn is absent is a torn write (Phase-5
            # writes ``turn.add`` last, deliberately).  Skipped and not counted:
            # an item whose text cannot be shown is not an annotatable item.
            continue
        out.append(
            MentionRecord(
                span_id, turn, span.start, span.end, event.payload.get("rung", ""), event.seq
            )
        )
    return out


def assertions_by_turn(log: EventLog) -> dict[str, list[str]]:
    """``turn_id -> [assertion_id]``, attributing each assertion to the turn
    whose processing stored it.

    The attribution uses the write path's own ordering guarantee: Phase 5
    appends a turn's spans and assertions **before** its ``turn.add`` (the
    crash-resume decision, `PHASE5_DECISIONS.md` §1.2a), so every
    ``assertion.add`` belongs to the *next* ``turn.add`` in the log.  The stored
    ``Assertion`` itself carries no turn field — ``asserted_by`` is the speaker
    — which is why this is derived from log order rather than read off a record.
    """
    out: dict[str, list[str]] = {}
    pending: list[str] = []
    for event in log.replay():
        if event.op == "assertion.add":
            pending.append(event.payload["assertion_id"])
        elif event.op == "turn.add":
            if pending:
                out.setdefault(event.payload["turn_id"], []).extend(pending)
                pending = []
    return out


def build_d1_item(
    ix: int,
    mention: MentionRecord,
    candidates: Sequence[Mapping[str, Any]],
    turn_text: str,
    stage_b_seq: int | None = None,
    turn_ts: str = "",
) -> dict[str, Any]:
    """One D1 item in the Phase-2.5 shape.

    Two sequence numbers, because they answer two different questions
    (the audit's blocker was one number trying to answer both):

    * ``snapshot_seq`` — the mention's **Stage-A** position: which point of the
      ingestion stream this mention belongs to.  The pin for Stage-A features.
    * ``stage_b_seq`` — the **construction log's** position at the moment the
      candidate list was computed, supplied by the constructor.  ``at(stage_b_seq)``
      on the construction log reproduces exactly the graph the candidates were
      drawn from — G12's leak guard, now checkable per item.
    """
    return {
        "item_id": f"d1_{ix:04d}",
        "actions": list(D1_ACTIONS),
        "candidates": list(candidates),
        "mention": mention.text,
        "start": mention.start,
        "end": mention.end,
        "turn_id": mention.turn_id,
        "turn_text": turn_text,
        "question_id": mention.conv_id,
        "span_id": mention.span_id,
        "snapshot_seq": mention.seq,
        "stage_b_seq": stage_b_seq,
        # The decision's own "now" — the anchor for the relative temporal
        # encoding.  Without it the encoder times every node against the latest
        # edge in the whole construction log, i.e. against other conversations'
        # calendars (found by the 14 Aug review).
        "turn_ts": turn_ts,
    }


def d1_items(log: EventLog, candidates_for: Any = None) -> list[dict[str, Any]]:
    """D1 annotation items derived from the log, in the Phase-2.5 shape.

    ``candidates_for`` is a **single-argument closure** ``(MentionRecord) ->
    candidates`` — the caller owns the snapshot it draws candidates from,
    because only the caller knows construction order.  *(Corrected 14 Aug 2026:
    this function previously computed candidates itself against
    ``at(mention.seq)`` — a Stage-A sequence that precedes every Stage-B commit,
    so the candidate list was empty for every mention on every corpus.  The
    audit's first blocker.)*  ``graphbuild.standin.construct`` is the caller
    that supplies construction-time candidates and the ``stage_b_seq`` pin;
    calling this with ``None`` yields candidate-less items, which is legal only
    for flows that fill candidates later.
    """
    snapshot = ReplayGraphStore(log).at()  # once — not once per mention
    items: list[dict[str, Any]] = []
    for ix, mention in enumerate(mention_records(log)):
        candidates: list[Mapping[str, Any]] = []
        if candidates_for is not None:
            candidates = list(candidates_for(mention))
        turn = snapshot.turn(mention.turn_id)
        items.append(
            build_d1_item(ix, mention, candidates, turn.text if turn is not None else "")
        )
    return items


def _eligible_assertions(log: EventLog) -> list[tuple[Assertion, int]]:
    """``(assertion, seq_at_which_it_became_eligible)``, in log order.

    The sequence is the *eligibility* event's, not the ``assertion.add``'s: an
    assertion is not part of the active graph until the gate says so, and a
    feature computed at the earlier sequence would be reading a graph state in
    which this very assertion was not yet admissible.
    """
    store = ReplayGraphStore(log)
    final = store.at()
    decided: dict[str, int] = {}
    for event in log.replay():
        if event.op == "assertion.set_eligibility":
            decided[event.payload["assertion_id"]] = event.seq
    out: list[tuple[Assertion, int]] = []
    for assertion_id, seq in sorted(decided.items(), key=lambda kv: kv[1]):
        assertion = final.assertion(assertion_id)
        if assertion is not None and assertion.eligibility == "eligible":
            out.append((assertion, seq))
    return out


def build_d2_item(
    ix: int,
    a_side: Mapping[str, Any],
    b_side: Mapping[str, Any],
    question_type: str = "",
    stage_a_seq: int | None = None,
    stage_b_seq: int | None = None,
    turn_ts: str = "",
) -> dict[str, Any]:
    """One D2 item, with **``claim_a`` strictly the earlier side**.

    Ordered by ``(session_date, turn_id, assertion_id)`` — *(corrected 14 Aug
    2026: the sides were previously assigned by sorted assertion-id, i.e. by
    hash, which broke the D2 guidelines' invariant that ``claim_b`` is the later
    session.  The CONFLICT-vs-SUPERSEDES rule the annotator applies depends on
    which side is later, so a hash ordering would have made roughly half the
    collected labels mean the opposite of what the guideline says.)*
    """
    earlier, later = sorted(
        (dict(a_side), dict(b_side)),
        key=lambda side: (side["session_date"], side["turn_id"], side["assertion_id"]),
    )
    return {
        "item_id": f"d2_{ix:04d}",
        "labels": list(D2_LABELS),
        "claim_a": earlier,
        "claim_b": later,
        "cross_session": earlier["session_id"] != later["session_id"],
        "question_id": earlier["question_id"],
        "question_type": question_type,
        "token_overlap": _overlap(earlier["text"], later["text"]),
        "snapshot_seq": stage_a_seq,
        "stage_b_seq": stage_b_seq,
        # The pair decision's own "now" — the linking turn (G5 proposes at link
        # time), so the temporal encoding is anchored where the decision stood,
        # not at the newest edge some other conversation contributed.
        "turn_ts": turn_ts,
    }


def d2_items(
    log: EventLog,
    pairs_for: Any = None,
    question_types: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """D2 pair items, in the Phase-2.5 shape.

    **Negatives come from the proposer, never from random pairing**
    (`GATE0_CONTRACT.md` item 6): a random negative is trivially `INDEPENDENT`
    and would inflate macro-F1 on the class nobody cares about. So when
    ``pairs_for`` is ``None`` this returns **no items at all** rather than
    falling back to all-pairs — an empty batch is an honest "the proposer was not
    supplied", and a fabricated one would poison the class balance the Gate-0
    contract fixes.

    ``pairs_for`` is a single-argument closure ``(Assertion) -> [Assertion]``;
    the caller owns the snapshot it proposes against.  *(Corrected 14 Aug 2026:
    this function previously proposed against ``at(eligibility_seq)`` — a
    Stage-A sequence that precedes every ``about_entity`` edge, so the
    anchor-sharing proposer returned zero pairs on every corpus, reported as the
    honest-empty case.  The audit's second blocker.  The constructor is the
    caller that proposes at link time — ``graphbuild.standin.construct``.)*
    """
    if pairs_for is None:
        return []

    final = ReplayGraphStore(log).at()
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for assertion, seq in _eligible_assertions(log):
        for other in pairs_for(assertion):
            key = tuple(sorted((assertion.assertion_id, other.assertion_id)))
            if key in seen or key[0] == key[1]:
                continue
            seen.add(key)
            a_side = _claim_side(key[0], final)
            b_side = _claim_side(key[1], final)
            if a_side is None or b_side is None:
                continue
            items.append(
                build_d2_item(
                    len(items),
                    a_side,
                    b_side,
                    question_type=(question_types or {}).get(a_side["question_id"], ""),
                    stage_a_seq=seq,
                )
            )
    return items


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


def _overlap(a: str, b: str) -> float:
    """Jaccard over lowercased word types — the spike's own field, kept so the
    annotator's sort order and intuitions carry over."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return round(len(wa & wb) / len(wa | wb), 3)


def adjudication_items(
    items: Sequence[Mapping[str, Any]],
    pass_a: Mapping[str, str],
    pass_b: Mapping[str, str],
    collapse: Any = None,
) -> list[dict[str, Any]]:
    """The third annotation batch: items two passes labelled differently.

    `GATE0_CONTRACT.md` item 7 makes adjudication **self-adjudication of the
    disagreeing items after the κ pass** — single annotator, so what κ measures
    is self-consistency and the write-up must say "self-agreement", never "IAA".
    This emits exactly those items, each carrying both labels and a blank
    `adjudicated` column, in the same shape the annotate CLI already reads.

    ``collapse`` folds a label before comparison — for D1 the κ convention
    collapses `LINK_EXISTING(<id>)` to its action (`scripts/phase2_5/annotate.py`),
    because a four-way agreement statistic over an open id space is not the
    quantity item 7 asks for.  The **uncollapsed** labels are still carried on
    the row: an id-only disagreement is a real disagreement to resolve, it is
    simply not what κ counts.
    """
    fold = collapse or (lambda label: label)
    by_id = {str(i["item_id"]): i for i in items}
    out: list[dict[str, Any]] = []
    for item_id in sorted(set(pass_a) & set(pass_b)):
        a, b = pass_a[item_id], pass_b[item_id]
        if fold(a) == fold(b) and a == b:
            continue
        item = by_id.get(item_id, {})
        out.append(
            {
                "item_id": item_id,
                "pass_1": a,
                "pass_2": b,
                "action_level_disagreement": fold(a) != fold(b),
                "mention": item.get("mention", ""),
                "turn_text": item.get("turn_text", ""),
                "claim_a": (item.get("claim_a") or {}).get("text", ""),
                "claim_b": (item.get("claim_b") or {}).get("text", ""),
                "adjudicated": "",
                "reason": "",
            }
        )
    return out


def write_items(path: str | Path, items: Iterable[Mapping[str, Any]]) -> int:
    """JSONL, sorted keys, LF endings — the Phase-2.5 storage convention."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def yields(log: EventLog) -> dict[str, Any]:
    """The measured per-turn yields that size an annotation batch.

    Reported rather than assumed because the number has moved three times — the
    spike's 0.59 mentions/turn, the bakeoff slice's 1.13, the live pilot's 0.754
    — and only the last is the production recipe's. An annotation batch sized on
    a superseded figure is a batch that runs out or overruns.
    """
    snapshot = ReplayGraphStore(log).at()
    counts = snapshot.counts()
    turns = counts["turns"]
    mentions = len(mention_records(log))
    stored = counts["assertions"]
    eligible = counts["eligible_assertions"]
    multi_span = sum(
        1
        for event in log.replay()
        if event.op == "assertion.add" and len(event.payload.get("spans", ())) > 1
    )
    return {
        "turns": turns,
        "mentions": mentions,
        "mentions_per_turn": (mentions / turns) if turns else float("nan"),
        "assertions_stored": stored,
        "assertions_eligible": eligible,
        "eligible_per_turn": (eligible / turns) if turns else float("nan"),
        "multi_span_assertions": multi_span,
        "multi_span_note": (
            "0 in every real run so far (0/222 live, 0/78 replay): the schema and "
            "write path support cross-turn provenance, the corpus has not produced "
            "it. Do not size D1/D2 items assuming multi-span input."
        ),
    }
