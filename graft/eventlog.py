"""Append-only JSONL event log — the single source of truth for graph state.

Line format::

    {"op": ..., "payload": {...}, "schema_version": ..., "seq": n, "ts": "..."}

``seq`` is monotonic from 0 and a ``snapshot_id`` **is** a ``seq``, so pinning a
read to a snapshot costs nothing and reads are reproducible.

Two portability decisions, both because this project moves between Windows
laptops and Kaggle and hands artefacts around as folders:

**All I/O is binary.**  Opening in text mode on Windows translates ``\\n`` into
``\\r\\n`` on write, so a log written on a laptop would not be byte-identical to
the same log written on Kaggle, and the digest below would diverge for identical
content.  Binary mode removes the translation entirely.

**``digest()`` excludes ``ts``.**  Wall-clock time is genuinely different
between two runs of the same work, so byte-identity of the file is the wrong
claim.  What must match is the ``(seq, op, payload)`` stream, and the digest
hashes exactly that — which also gives a teammate a one-line way to check that
their run produced the same log as yours.

**Crash-resume.**  On open the file is scanned forward; the first line that
fails to parse or breaks ``seq`` monotonicity is treated as a torn write and the
file is truncated there.  No hash chaining — over-engineering for a
single-writer local file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from graft.canonical import canonical_bytes, digest_of
from graft.schemas import SCHEMA_VERSION

__all__ = ["Event", "EventLog", "LogCorruption"]


class LogCorruption(RuntimeError):
    """Raised when the log is damaged in a way truncation cannot repair."""


@dataclass(frozen=True)
class Event:
    """One replayed log line."""

    seq: int
    ts: str
    schema_version: str
    op: str
    payload: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class EventLog:
    """Append-only JSONL log with crash-safe open."""

    def __init__(self, path: str | Path, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = bool(fsync)
        self._next_seq = 0
        self._repaired_bytes = 0
        self._handle = None
        self._open()

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, fsync: bool = False) -> "EventLog":
        return cls(path, fsync=fsync)

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._scan_and_repair()
        self._handle = self.path.open("ab")

    def _scan_and_repair(self) -> None:
        """Validate the existing file, truncating at the first damaged line."""
        raw = self.path.read_bytes()
        good_upto = 0
        expected_seq = 0
        cursor = 0

        while cursor < len(raw):
            newline = raw.find(b"\n", cursor)
            if newline == -1:
                # Trailing bytes with no terminator: a torn final write.
                break
            line = raw[cursor:newline]
            cursor = newline + 1
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
                seq = int(record["seq"])
                record["op"]
                record["payload"]
            except Exception:
                break
            if seq != expected_seq:
                break
            expected_seq += 1
            good_upto = cursor

        self._next_seq = expected_seq
        if good_upto != len(raw):
            self._repaired_bytes = len(raw) - good_upto
            os.truncate(self.path, good_upto)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def append(self, op: str, payload: Mapping[str, Any]) -> int:
        """Append one event and return its sequence number."""
        if self._handle is None:
            raise LogCorruption(f"log {self.path} is closed")
        if not op:
            raise ValueError("op must be a non-empty string")

        seq = self._next_seq
        record = {
            "seq": seq,
            "ts": _utc_now(),
            "schema_version": SCHEMA_VERSION,
            "op": op,
            "payload": dict(payload),
        }
        self._handle.write(canonical_bytes(record) + b"\n")
        self._handle.flush()
        if self.fsync:
            os.fsync(self._handle.fileno())
        self._next_seq += 1
        return seq

    # -- reading -----------------------------------------------------------

    def replay(self, upto: int | None = None) -> Iterator[Event]:
        """Yield events with ``seq < upto`` (or all events when ``upto`` is None)."""
        if self._handle is not None:
            self._handle.flush()
        if not self.path.exists():
            return

        with self.path.open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line.decode("utf-8"))
                seq = int(record["seq"])
                if upto is not None and seq >= upto:
                    return
                yield Event(
                    seq=seq,
                    ts=record["ts"],
                    schema_version=record["schema_version"],
                    op=record["op"],
                    payload=record["payload"],
                )

    def snapshot_id(self) -> int:
        """The sequence number a read pinned *now* would see (one past the end)."""
        return self._next_seq

    def digest(self, upto: int | None = None) -> str:
        """Content hash of the ``(seq, op, payload)`` stream, excluding timestamps.

        Two runs of the same work produce the same digest on any machine.  This
        is the checkable form of "reruns produce identical logs".
        """
        return digest_of(
            [[e.seq, e.op, e.payload] for e in self.replay(upto=upto)]
        )

    # -- diagnostics -------------------------------------------------------

    @property
    def repaired_bytes(self) -> int:
        """Bytes discarded as a torn write when this log was opened."""
        return self._repaired_bytes

    def __len__(self) -> int:
        return self._next_seq

    def __repr__(self) -> str:
        return f"EventLog(path={str(self.path)!r}, events={self._next_seq}, fsync={self.fsync})"
