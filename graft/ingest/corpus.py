"""P5.1 — LongMemEval-S → a deterministic ``Turn`` stream.

**The corpus is pinned by SHA and verified at every load.**  The pin already
exists in ``scripts/phase2_5/common.py``, and it lives in two places on purpose:
the spike's tooling must keep running unchanged, and ``graft.ingest`` must not
import from ``scripts/``.  A test asserts the two constants are equal, which is
the project's usual answer to a value that genuinely needs two homes — check it,
do not trust it.

**Three id conventions, all inherited rather than invented** (P5.1):

``conv_id``
    the LongMemEval ``question_id``.  A haystack is one simulated user's
    history, and ``conv_id`` is the unit `H`'s scope sub-check is defined over
    (Phase-1 gap G6), so the mapping has to be the one a scope constraint would
    name.
``session_id``
    the corpus's own session id.
``turn_id``
    ``lme_s/{question_id}/{session_id}/{turn_ix}`` — the Phase-2.5 convention,
    kept so a Phase-5 record and a spike label point at the same turn.  Content
    hashing would have been the ``graft.ids`` house style, but the unit of
    provenance here is *a position in a pinned file*, which is already
    deterministic and stays legible in a label file.

**Timestamps carry a declared convention, not a discovered one.**  The corpus
writes ``2023/05/20 (Sat) 02:21`` with no timezone.  ``Turn.ts`` is therefore
that instant *read as UTC* — stated here because a silent localtime read would
make ingestion machine-dependent, and because Phase-6's ``valid_during`` edges
will inherit whatever this decides.

Per-question metadata (type, answer, evidence session ids) is returned as a
**sidecar**, never written to the event log.  ``answer_session_ids`` is a gold
label; putting it in the evidence stream would leak the answer into the graph
the system is supposed to build without it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from graft.schemas import Turn

__all__ = [
    "CORPUS_SHA256",
    "CORPUS_LICENCE",
    "CORPUS_SOURCE",
    "QUESTION_TYPES",
    "load_corpus",
    "question_index",
    "turn_id_for",
    "parse_corpus_ts",
    "turns_of",
    "question_meta",
    "evidence_turns",
]

#: Pinned at download time (13 Aug 2026), identical to the spike's pin.
CORPUS_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
CORPUS_LICENCE = "mit"
CORPUS_SOURCE = "HF dataset xiaowu0162/longmemeval, file longmemeval_s"

#: The six question types, in the corpus's own vocabulary.  Listed so that a
#: scope decision (Gate-0 item 9) can name types rather than counts.
QUESTION_TYPES: tuple[str, ...] = (
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
)

_DEFAULT_PATH = Path("data") / "phase2_5" / "raw" / "longmemeval_s"


def load_corpus(path: str | Path | None = None, *, verify: bool = True) -> list[dict[str, Any]]:
    """Read and SHA-verify the pinned corpus file.

    ``verify=False`` exists for tests over a synthetic fixture and for nothing
    else; a run that reads a different file than the one pinned is a different
    experiment, and the loader says so rather than proceeding.
    """
    path = Path(path) if path is not None else _DEFAULT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"corpus not found at {path}. Fetch it from {CORPUS_SOURCE} "
            "(the raw file is gitignored and re-fetched per machine)."
        )
    blob = path.read_bytes()
    if verify:
        sha = hashlib.sha256(blob).hexdigest()
        if sha != CORPUS_SHA256:
            raise ValueError(
                f"corpus SHA mismatch at {path}: expected {CORPUS_SHA256}, got {sha}. "
                "The pin describes a different file; re-fetch, or re-pin deliberately."
            )
    return json.loads(blob.decode("utf-8"))


def question_index(corpus: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """``question_id -> instance``, for the callers that address by id."""
    return {inst["question_id"]: inst for inst in corpus}


def turn_id_for(question_id: str, session_id: str, turn_ix: int) -> str:
    """The Phase-2.5 turn-id convention, unchanged."""
    return f"lme_s/{question_id}/{session_id}/{turn_ix}"


def parse_corpus_ts(raw: str) -> str:
    """``'2023/05/20 (Sat) 02:21'`` → ``'2023-05-20T02:21:00+00:00'``.

    The weekday in parentheses is redundant with the date and is discarded
    rather than validated: it is display text in the corpus, and treating a
    mismatch as an error would make the loader fail on a cosmetic defect in data
    this project does not own.
    """
    head = raw.split("(")[0].strip()
    tail = raw.split(")")[-1].strip() if ")" in raw else ""
    stamp = f"{head} {tail}".strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            naive = datetime.strptime(stamp, fmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=timezone.utc).isoformat()
    raise ValueError(f"unrecognised corpus timestamp {raw!r}")


def turns_of(
    instance: Mapping[str, Any],
    session_ids: Iterable[str] | None = None,
) -> Iterator[Turn]:
    """Every turn of one question's haystack, in corpus order.

    ``session_ids`` restricts to named sessions — how the pilot takes evidence
    sessions only, and how a Gate-0 item-9 scope of "evidence + *d* distractors"
    would be expressed.  Order is the corpus's: sessions as listed, turns within
    a session as listed, because a rolling summary and an *m*-turn window are
    both defined over "the previous turns" and a reordering would silently
    change the extraction context.
    """
    wanted = None if session_ids is None else set(session_ids)
    question_id = instance["question_id"]
    for ix, session_id in enumerate(instance["haystack_session_ids"]):
        if wanted is not None and session_id not in wanted:
            continue
        ts = parse_corpus_ts(instance["haystack_dates"][ix])
        for turn_ix, turn in enumerate(instance["haystack_sessions"][ix]):
            yield Turn(
                turn_id=turn_id_for(question_id, session_id, turn_ix),
                conv_id=question_id,
                session_id=session_id,
                speaker=str(turn["role"]),
                ts=ts,
                text=str(turn["content"]),
            )


def evidence_turns(instance: Mapping[str, Any]) -> Iterator[Turn]:
    """The turns of the sessions LongMemEval marks as carrying the answer.

    The pilot's corpus (G10) and the one commitment G8 makes about scope — the
    knowledge-update evidence sessions are in *every* candidate scope, because
    D2's supervision lives there and it is the binding constraint on C1.
    """
    return turns_of(instance, instance.get("answer_session_ids", ()))


def question_meta(instance: Mapping[str, Any]) -> dict[str, Any]:
    """The sidecar: everything about the question that is **not** evidence.

    Kept out of the event log deliberately.  ``answer`` and
    ``answer_session_ids`` are gold labels; a pipeline that wrote them into the
    log would be building the graph with the answer already in it, and every
    ceiling measured afterwards would be meaningless.
    """
    return {
        "question_id": instance["question_id"],
        "question_type": instance["question_type"],
        "question": instance["question"],
        "question_date": instance["question_date"],
        "answer": instance["answer"],
        "evidence_session_ids": list(instance.get("answer_session_ids", ())),
        "n_haystack_sessions": len(instance["haystack_session_ids"]),
        "n_haystack_turns": sum(len(s) for s in instance["haystack_sessions"]),
    }
