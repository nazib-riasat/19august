"""The append-only event log.

Discharges Phase-0 exit criterion 3 (torn-write recovery) and supplies the log
half of criterion 2 (replay reconstructs an identical graph).
"""

from __future__ import annotations

import json

import pytest

from graft.eventlog import EventLog
from graft.schemas import SCHEMA_VERSION


def _payload(i: int) -> dict:
    return {"node_id": f"n{i}", "ntype": "Entity", "payload": {"i": i}}


def test_sequence_numbers_are_monotonic_from_zero(tmp_path):
    with EventLog.open(tmp_path / "e.jsonl") as log:
        assert [log.append("node.add", _payload(i)) for i in range(5)] == [0, 1, 2, 3, 4]
        assert log.snapshot_id() == 5
        assert len(log) == 5


def test_replay_returns_what_was_appended(tmp_path):
    with EventLog.open(tmp_path / "e.jsonl") as log:
        for i in range(10):
            log.append("node.add", _payload(i))
        events = list(log.replay())
    assert [e.seq for e in events] == list(range(10))
    assert all(e.schema_version == SCHEMA_VERSION for e in events)
    assert events[3].payload == _payload(3)


def test_replay_upto_is_exclusive_and_is_the_snapshot_semantics(tmp_path):
    with EventLog.open(tmp_path / "e.jsonl") as log:
        for i in range(10):
            log.append("node.add", _payload(i))
        assert [e.seq for e in log.replay(upto=4)] == [0, 1, 2, 3]
        assert list(log.replay(upto=0)) == []


def test_reopening_continues_the_sequence(tmp_path):
    path = tmp_path / "e.jsonl"
    with EventLog.open(path) as log:
        log.append("node.add", _payload(0))
        log.append("node.add", _payload(1))
    with EventLog.open(path) as log:
        assert log.snapshot_id() == 2
        assert log.append("node.add", _payload(2)) == 2


def test_lines_are_written_with_lf_on_every_platform(tmp_path):
    """Text mode on Windows would translate \\n to \\r\\n, so the same logical log
    written on a laptop and on Kaggle would not agree byte for byte."""
    path = tmp_path / "e.jsonl"
    with EventLog.open(path) as log:
        log.append("node.add", _payload(0))
        log.append("node.add", _payload(1))
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 2
    assert raw.endswith(b"\n")


def test_torn_write_is_truncated_and_the_log_resumes(tmp_path):
    """Criterion 3: truncate mid-line, reopen, confirm the last valid seq is
    recovered and the partial line is discarded."""
    path = tmp_path / "e.jsonl"
    with EventLog.open(path) as log:
        for i in range(5):
            log.append("node.add", _payload(i))

    raw = path.read_bytes()
    cut = raw.rfind(b"\n", 0, len(raw) - 1) + 1  # start of the last full line
    path.write_bytes(raw[: cut + 12])  # keep a fragment of line 4

    with EventLog.open(path) as log:
        assert log.repaired_bytes == 12
        assert log.snapshot_id() == 4
        assert [e.seq for e in log.replay()] == [0, 1, 2, 3]
        assert log.append("node.add", _payload(99)) == 4


def test_a_corrupt_middle_line_truncates_from_that_point(tmp_path):
    path = tmp_path / "e.jsonl"
    with EventLog.open(path) as log:
        for i in range(5):
            log.append("node.add", _payload(i))

    lines = path.read_bytes().split(b"\n")[:-1]
    lines[2] = b'{"seq":2,"op":"node.add"'  # damaged, unparseable
    path.write_bytes(b"\n".join(lines) + b"\n")

    with EventLog.open(path) as log:
        assert log.snapshot_id() == 2
        assert [e.seq for e in log.replay()] == [0, 1]


def test_a_sequence_gap_is_treated_as_damage(tmp_path):
    path = tmp_path / "e.jsonl"
    with EventLog.open(path) as log:
        for i in range(4):
            log.append("node.add", _payload(i))

    lines = path.read_bytes().split(b"\n")[:-1]
    record = json.loads(lines[2].decode())
    record["seq"] = 99
    lines[2] = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"\n".join(lines) + b"\n")

    with EventLog.open(path) as log:
        assert log.snapshot_id() == 2


def test_digest_ignores_timestamps_but_tracks_content(tmp_path):
    """Two runs of the same work differ in wall clock, so file byte-identity is
    the wrong claim; the (seq, op, payload) stream is the right one."""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    for path in (first, second):
        with EventLog.open(path) as log:
            for i in range(20):
                log.append("node.add", _payload(i))

    with EventLog.open(first) as a, EventLog.open(second) as b:
        assert a.digest() == b.digest()
        assert a.digest(upto=10) != a.digest()
        b.append("node.add", _payload(999))
        assert a.digest() != b.digest()


def test_digest_of_an_empty_log_is_defined(tmp_path):
    with EventLog.open(tmp_path / "e.jsonl") as log:
        assert len(log.digest()) == 64


def test_empty_op_is_rejected(tmp_path):
    with EventLog.open(tmp_path / "e.jsonl") as log:
        with pytest.raises(ValueError, match="non-empty"):
            log.append("", {})


def test_nan_never_reaches_the_permanent_log(tmp_path):
    with EventLog.open(tmp_path / "e.jsonl") as log:
        with pytest.raises(ValueError):
            log.append("node.add", {"score": float("nan")})
        assert log.snapshot_id() == 0


def test_fsync_mode_writes_the_same_bytes(tmp_path):
    plain, synced = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with EventLog.open(plain, fsync=False) as a, EventLog.open(synced, fsync=True) as b:
        for i in range(3):
            a.append("node.add", _payload(i))
            b.append("node.add", _payload(i))
        assert a.digest() == b.digest()
