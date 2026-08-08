"""Deterministic, content-derived identity.

Every id in this system is a truncated SHA-256 of the content that defines the
thing.  No counters and no UUIDs: both make a rerun produce a different log for
the same inputs, which would destroy the ability to compare two machines' runs
at all.

``canon_set_hash`` is load-bearing in three separate places and therefore lives
here rather than in the phase that first needs it:

* **policy state identity** — two insertion orders that select the same atoms
  are the same state (research plan v1.2 §3.4);
* **beam-search dedup** (Phase 4, S2);
* **the equivalent-action collision audit** (Phase 2).  That audit exists
  because uncorrected equivalent actions bias sampling — Symmetry-Aware
  GFlowNets (ICML 2025) measured L1 ~= 0.12 uncorrected vs ~= 0.01 corrected.
  The audit is meaningless without a canonical hash, so if each phase invented
  its own the audit would silently measure nothing.

**Reference order is significant.**  ``atom_id("edge", ("a", "b"))`` and
``atom_id("edge", ("b", "a"))`` are different atoms, because an edge is
directed.  A caller whose references are genuinely unordered must sort them
before calling, so that the id is stable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

__all__ = [
    "ID_LENGTH",
    "span_id",
    "node_id",
    "edge_id",
    "assertion_id",
    "atom_id",
    "canon_set_hash",
]

# 16 hex characters = 64 bits.  At the scale this project reaches (Phase-0 exit
# criterion tests 10^6 atoms) the birthday collision probability is ~2.7e-8.
ID_LENGTH = 16

# Unit separator.  Chosen because it cannot occur in an id, a type name or a
# normalised text field, so no combination of parts can be ambiguous.
_SEP = "\x1f"


def _hash_id(*parts: str) -> str:
    payload = _SEP.join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:ID_LENGTH]


def span_id(turn_id: str, start: int, end: int) -> str:
    """Identity of a character span inside a turn."""
    return _hash_id("span", turn_id, str(start), str(end))


def node_id(ntype: str, key: str) -> str:
    """Identity of a graph node.  ``key`` is the type-specific natural key."""
    return _hash_id("node", ntype, key)


def edge_id(etype: str, src: str, dst: str, label: str = "") -> str:
    """Identity of a directed graph edge.

    ``label`` disambiguates parallel edges of the same type between the same
    endpoints (for example two ``valid_during`` edges covering different
    intervals).  Without it such edges would collide.
    """
    return _hash_id("edge", etype, src, dst, label)


def assertion_id(kind: str, text_norm: str, spans: Iterable[str]) -> str:
    """Identity of an extracted assertion.

    Spans are sorted, because the same assertion supported by the same set of
    spans is the same assertion regardless of the order extraction found them.
    """
    return _hash_id("assertion", kind, text_norm, ",".join(sorted(spans)))


def atom_id(kind: str, refs: Iterable[str], label: str = "") -> str:
    """Identity of a candidate atom.

    ``label`` is an addition to the signature sketched in the Phase-0 build plan
    (``atom_id(kind, refs)``).  Without it two atoms of the same kind over the
    same references collide — two differently-typed edges between one pair of
    endpoints being the common case — and Phase 0's own exit criterion requires
    no collisions across 10^6 generated atoms.  Callers with nothing to
    disambiguate pass nothing and get the two-argument behaviour.
    """
    return _hash_id("atom", kind, ",".join(refs), label)


def canon_set_hash(atoms: Iterable[str]) -> str:
    """Order-invariant identity of a set of atom ids.

    Sorting is what makes this canonical, and it is also what makes it stable
    across processes: Python randomises string hashing per interpreter run, so
    iterating a ``frozenset[str]`` yields a different order on every launch.
    Serialising or hashing a set without sorting it first is a portability bug
    that only shows up when two machines compare results.

    The empty set is a well-defined value — it is the root state of every
    construction trajectory.
    """
    return _hash_id("set", ",".join(sorted(atoms)))
