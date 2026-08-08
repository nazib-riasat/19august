"""Canonical serialisation — the property everything else's portability rests on."""

from __future__ import annotations

import json

import pytest

from graft.canonical import canonical_bytes, canonical_json, digest_of, sha256_hex


def test_key_order_does_not_reach_the_bytes():
    left = {"b": 1, "a": 2, "c": {"z": 0, "y": 1}}
    right = {"c": {"y": 1, "z": 0}, "a": 2, "b": 1}
    assert canonical_json(left) == canonical_json(right)


def test_output_is_pure_ascii():
    """Windows defaults to cp1252 and Kaggle to UTF-8; ASCII survives both."""
    payload = canonical_bytes({"text": "café — naïve 日本語"})
    assert payload.decode("ascii")
    assert json.loads(payload.decode("ascii"))["text"] == "café — naïve 日本語"


def test_nan_and_infinity_are_refused():
    """`json` would emit bare NaN/Infinity, which nothing outside Python reads —
    and this goes into a permanent append-only log."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json({"x": bad})


def test_floats_round_trip_exactly():
    values = [0.1, 1e-9, -2.5, 1.7976931348623157e308, 3.141592653589793]
    restored = json.loads(canonical_json({"v": values}))["v"]
    assert restored == values


def test_digest_is_stable_and_truncatable():
    obj = {"a": [1, 2, 3], "b": "x"}
    full = digest_of(obj)
    assert len(full) == 64
    assert digest_of(obj, 8) == full[:8]
    assert digest_of({"b": "x", "a": [1, 2, 3]}) == full


def test_digest_changes_with_content():
    assert digest_of({"a": 1}) != digest_of({"a": 2})


def test_sha256_accepts_str_and_bytes():
    assert sha256_hex("abc") == sha256_hex(b"abc")
