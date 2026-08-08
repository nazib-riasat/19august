"""Canonical serialisation — the single place an object becomes bytes.

Everything that has to be identical across machines routes through here: the
config hash, every event-log line, the canonical set hash, and the log digest.
The project is developed on several laptops and on Kaggle, and artefacts are
handed between people as folders, so "identical" has to survive a change of
operating system, locale and filesystem encoding.

Three settings do that work:

``sort_keys=True``
    dict insertion order never reaches the bytes.
``ensure_ascii=True``
    output is pure ASCII whatever the host encoding is.  Windows defaults to
    cp1252 and Kaggle to UTF-8; neither can alter an ASCII byte string.
``allow_nan=False``
    ``json`` would otherwise emit bare ``NaN`` / ``Infinity``, which is not
    valid JSON and is unreadable outside Python.  A NaN arriving here is an
    upstream bug and should surface as one rather than be written to a
    permanent append-only log.

Floats are emitted by ``repr``, which round-trips IEEE-754 doubles exactly, so
a value survives write -> read -> write unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "canonical_json",
    "canonical_bytes",
    "sha256_hex",
    "digest_of",
]


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to the one string form this project considers canonical."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON as ASCII bytes, ready to hash or append to the log."""
    return canonical_json(obj).encode("ascii")


def sha256_hex(payload: bytes | str) -> str:
    """Full 64-character SHA-256 hex digest of ``payload``."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def digest_of(obj: Any, length: int | None = None) -> str:
    """SHA-256 of ``obj``'s canonical form, optionally truncated to ``length``."""
    full = sha256_hex(canonical_bytes(obj))
    return full if length is None else full[:length]
