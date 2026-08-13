"""Phase 2.5 shared plumbing: paths, JSONL io, ids, provenance.

The spike lives entirely under ``scripts/phase2_5/`` and ``data/phase2_5/`` —
exit criterion 10 of `GRAFT_PHASE2_5_BUILD.md` forbids `graft/` gaining an
encoder, a decoder or a training loop, and gap G6 forbids new schemas in
`graft/schemas.py`. Labels are written in Phase-0 id conventions so Phase 6 can
trace every label back to the turn and offsets it came from.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / "data" / "phase2_5" / "raw" / "longmemeval_s"
DATA = REPO / "data" / "phase2_5"
LABELS = DATA / "labels"

# The corpus pin of GRAFT_PHASE2_5_BUILD.md G3, measured at download time
# (13 Aug 2026).  fetch provenance: HF dataset `xiaowu0162/longmemeval`,
# file `longmemeval_s`, licence `mit` per the HF dataset card.
CORPUS_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
CORPUS_LICENCE = "mit"

SAMPLE_SEED = 20260813  # date convention, same family as the Phase-2 suite seeds


def turn_id(question_id: str, session_id: str, turn_ix: int) -> str:
    """Deterministic, content-addressable turn identity for the spike.

    Not `graft.ids` — those hash *content*; the spike's unit of provenance is
    (question, session, index) into a pinned corpus file, which is already
    deterministic and stays human-legible in a label file.
    """
    return f"lme_s/{question_id}/{session_id}/{turn_ix}"


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_corpus() -> list[dict[str, Any]]:
    if not RAW.is_file():
        sys.exit(
            f"corpus not found at {RAW} — run the fetch step first "
            "(HF dataset xiaowu0162/longmemeval, file longmemeval_s)"
        )
    blob = RAW.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != CORPUS_SHA256:
        sys.exit(
            f"corpus SHA mismatch: expected {CORPUS_SHA256}, got {sha} — "
            "the pin in common.py describes a different file; re-fetch or re-pin deliberately"
        )
    return json.loads(blob.decode("utf-8"))
