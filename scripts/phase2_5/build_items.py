"""Phase 2.5 — derive the D1 and D2 annotation items from the extraction.

Item definitions are gap G2's, verbatim:

* **D1** — one *mention* → LINK_EXISTING(id) / CREATE_NEW_ENTITY / NON_ENTITY /
  DEFER, **plus the entity id when linking**. The candidate list is what makes
  it slow, so each item carries one: the distinct entities registered so far in
  the same conversation (name-normalised mention texts), which is the
  spike-grade stand-in for D1's top-k retrieval.
* **D2** — one *claim pair* → INDEPENDENT / DUPLICATE / CONFLICT / SUPERSEDES,
  with both claims' session dates shown, because supersession needs the
  temporal relation and not just the text (G2).

Pairing follows the fix-F8 pattern in miniature: a pair is proposed only when
the two assertions share an anchor token, ranked by token overlap, and pairs
that cross sessions of one conversation are preferred — the knowledge-update
sample exists precisely so CONFLICT/SUPERSEDES candidates occur.

Usage:  python scripts/phase2_5/build_items.py
Reads:  data/phase2_5/extraction.jsonl
Writes: data/phase2_5/d1_items.jsonl, data/phase2_5/d2_items.jsonl
"""
from __future__ import annotations

import itertools
import re

from common import DATA, read_jsonl, write_jsonl

D1_TARGET = 100
D2_TARGET = 50

_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "was", "were", "is", "are", "be", "been", "have", "has", "had",
    "i", "my", "me", "you", "your", "it", "its", "that", "this", "they",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9']+", text.lower())
        if len(t) >= 4 and t not in _STOP
    }


def build_d1(rows: list[dict]) -> list[dict]:
    items: list[dict] = []
    # One entity registry per conversation (= per question), chronological.
    registries: dict[str, dict[str, str]] = {}
    for row in rows:
        reg = registries.setdefault(row["question_id"], {})
        for m in row["mentions"]:
            candidates = [
                {"entity_id": eid, "name": name} for name, eid in sorted(reg.items())
            ]
            items.append(
                {
                    "item_id": f"d1_{len(items):04d}",
                    "turn_id": row["turn_id"],
                    "question_id": row["question_id"],
                    "turn_text": row["text"],
                    "mention": m["text"],
                    "start": m["start"],
                    "end": m["end"],
                    "candidates": candidates,
                    "actions": [
                        "LINK_EXISTING(<entity_id>)",
                        "CREATE_NEW_ENTITY",
                        "NON_ENTITY",
                        "DEFER",
                    ],
                }
            )
            key = _norm(m["text"])
            if key and key not in reg:
                reg[key] = f"e_{row['question_id'][:8]}_{len(reg):03d}"
    return items[:D1_TARGET]


def build_d2(rows: list[dict]) -> list[dict]:
    claims = []
    for row in rows:
        for a in row["assertions"]:
            claims.append(
                {
                    "assertion_id": a["assertion_id"],
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "session_id": row["session_id"],
                    "session_date": row["session_date"],
                    "turn_id": row["turn_id"],
                    "text": a["text_norm"] or a["quote"],
                    "tokens": _tokens(a["text_norm"] or a["quote"]),
                }
            )
    scored = []
    by_q: dict[str, list[dict]] = {}
    for c in claims:
        by_q.setdefault(c["question_id"], []).append(c)
    for q_claims in by_q.values():
        for a, b in itertools.combinations(q_claims, 2):
            if a["assertion_id"] == b["assertion_id"]:
                continue
            shared = a["tokens"] & b["tokens"]
            if not shared:
                continue
            overlap = len(shared) / max(1, len(a["tokens"] | b["tokens"]))
            cross_session = a["session_id"] != b["session_id"]
            is_ku = a["question_type"] == "knowledge-update"
            # Cross-session pairs in knowledge-update conversations are where
            # CONFLICT/SUPERSEDES live; rank them first, then by overlap.
            scored.append((is_ku and cross_session, cross_session, overlap, a, b))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    items, used = [], set()
    for is_ku_x, cross, overlap, a, b in scored:
        pair_key = frozenset((a["assertion_id"], b["assertion_id"]))
        if pair_key in used:
            continue
        used.add(pair_key)
        items.append(
            {
                "item_id": f"d2_{len(items):04d}",
                "question_id": a["question_id"],
                "question_type": a["question_type"],
                "cross_session": cross,
                "token_overlap": round(overlap, 3),
                "claim_a": {k: a[k] for k in
                            ("assertion_id", "text", "turn_id", "session_id", "session_date")},
                "claim_b": {k: b[k] for k in
                            ("assertion_id", "text", "turn_id", "session_id", "session_date")},
                "labels": ["INDEPENDENT", "DUPLICATE", "CONFLICT", "SUPERSEDES"],
            }
        )
        if len(items) >= D2_TARGET:
            break
    return items


def main() -> None:
    rows = read_jsonl(DATA / "extraction.jsonl")
    d1 = build_d1(rows)
    d2 = build_d2(rows)
    n1 = write_jsonl(DATA / "d1_items.jsonl", d1)
    n2 = write_jsonl(DATA / "d2_items.jsonl", d2)
    ku = sum(1 for i in d2 if i["question_type"] == "knowledge-update")
    cross = sum(1 for i in d2 if i["cross_session"])
    print(f"D1 items: {n1} (target {D1_TARGET})")
    print(f"D2 items: {n2} (target {D2_TARGET}); knowledge-update {ku}, cross-session {cross}")


if __name__ == "__main__":
    main()
