"""Phase 2.5 step 1 — sample ~50 turns from LongMemEval-S, reproducibly.

Gap G3: the sample **over-samples the knowledge-updates subset deliberately** —
a uniform sample of conversational turns yields almost no CONFLICT or
SUPERSEDES pairs, and timing 50 INDEPENDENT pairs answers nothing. Composition
(decision, recorded in PHASE2_5_DECISIONS.md): 6 knowledge-update questions,
2 multi-session, 1 temporal-reasoning, 1 single-session-user; from each, its
**evidence sessions** (`answer_session_ids`), which are the sessions that
actually carry the facts — and, for knowledge-update, the update pair.

Over-sampling is right for a *timing* measurement and would be wrong for a
*training* set (G3); the class mix here says nothing about natural class
balance and the write-up must not use it as if it did.

Usage:  python scripts/phase2_5/sample_turns.py
Output: data/phase2_5/sample.json  (+ provenance block inside it)
"""
from __future__ import annotations

import json
import random

from common import CORPUS_LICENCE, CORPUS_SHA256, DATA, SAMPLE_SEED, load_corpus, turn_id

COMPOSITION = (
    ("knowledge-update", 6),
    ("multi-session", 2),
    ("temporal-reasoning", 1),
    ("single-session-user", 1),
)
MAX_SESSIONS_PER_QUESTION = 3
MAX_TURNS_PER_SESSION = 6
TARGET_TURNS = (45, 70)  # acceptance band around the plan's "~50"


def _session_window(turns: list[dict]) -> list[int]:
    """Indices of the turns kept from one evidence session.

    Full evidence sessions run ~25 turns, so ten questions' worth is ~250 —
    five times the plan's "~50 turns". The information-bearing turns are the
    ones LongMemEval marks ``has_answer`` (plus one preceding turn each, so
    pronouns and ellipsis still resolve); sessions with no mark keep their
    first four turns. Windowing is recorded per turn via ``turn_ix``, so
    provenance back into the pinned corpus stays exact.
    """
    marked = [i for i, t in enumerate(turns) if t.get("has_answer")]
    if marked:
        keep: set[int] = set()
        for i in marked:
            keep.update(j for j in (i - 1, i, i + 1) if 0 <= j < len(turns))
    else:
        keep = set(range(min(4, len(turns))))
    return sorted(keep)[:MAX_TURNS_PER_SESSION]


def main() -> None:
    corpus = load_corpus()
    rng = random.Random(SAMPLE_SEED)

    by_type: dict[str, list[dict]] = {}
    for inst in corpus:
        by_type.setdefault(inst["question_type"], []).append(inst)
    for insts in by_type.values():
        insts.sort(key=lambda x: x["question_id"])  # order independent of file order

    sessions_out: list[dict] = []
    questions_out: list[dict] = []
    n_turns = 0
    for qtype, count in COMPOSITION:
        chosen = rng.sample(by_type[qtype], count)
        for inst in chosen:
            sid_to_ix = {sid: i for i, sid in enumerate(inst["haystack_session_ids"])}
            evidence_ids = [s for s in inst["answer_session_ids"] if s in sid_to_ix]
            evidence_ids = evidence_ids[:MAX_SESSIONS_PER_QUESTION]
            questions_out.append(
                {
                    "question_id": inst["question_id"],
                    "question_type": qtype,
                    "question": inst["question"],
                    "answer": inst["answer"],
                    "question_date": inst["question_date"],
                    "evidence_session_ids": evidence_ids,
                }
            )
            for sid in evidence_ids:
                ix = sid_to_ix[sid]
                turns = inst["haystack_sessions"][ix]
                window = _session_window(turns)
                sessions_out.append(
                    {
                        "question_id": inst["question_id"],
                        "question_type": qtype,
                        "session_id": sid,
                        "session_date": inst["haystack_dates"][ix],
                        "n_turns_full_session": len(turns),
                        "turns": [
                            {
                                "turn_id": turn_id(inst["question_id"], sid, t_ix),
                                "turn_ix": t_ix,
                                "role": turns[t_ix]["role"],
                                "text": turns[t_ix]["content"],
                                "has_answer": bool(turns[t_ix].get("has_answer", False)),
                            }
                            for t_ix in window
                        ],
                    }
                )
                n_turns += len(window)

    lo, hi = TARGET_TURNS
    if not lo <= n_turns <= hi:
        # The band is part of the sample's identity: a sample far off "~50
        # turns" times a different amount of reading and the measurement
        # describes a different workload. Adjust MAX_SESSIONS_PER_QUESTION or
        # the composition — never silently accept.
        raise SystemExit(f"sampled {n_turns} turns, outside the declared band {TARGET_TURNS}")

    out = {
        "provenance": {
            "corpus": "HF dataset xiaowu0162/longmemeval, file longmemeval_s",
            "corpus_sha256": CORPUS_SHA256,
            "corpus_licence": CORPUS_LICENCE,
            "sample_seed": SAMPLE_SEED,
            "composition": list(COMPOSITION),
            "max_sessions_per_question": MAX_SESSIONS_PER_QUESTION,
            "n_questions": len(questions_out),
            "n_sessions": len(sessions_out),
            "n_turns": n_turns,
        },
        "questions": questions_out,
        "sessions": sessions_out,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "sample.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {path}: {len(questions_out)} questions, {len(sessions_out)} sessions, {n_turns} turns")


if __name__ == "__main__":
    main()
