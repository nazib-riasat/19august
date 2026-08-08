"""Every dataclass in the system.  Nothing else may define one that crosses a
module boundary.

Two tiers, one home (Phase-0 gap G5):

**Tier A — frozen in Phase 0.**  ``Interval``, ``Obligations``,
``CandidateAtom``, ``ProofSet``.  Phases 1-4 consume these immediately and run
entirely on synthetic data, so changing one later means re-running experiments.

**Tier B — declared in Phase 0, frozen at its consumer phase.**  ``Turn``,
``SourceSpan``, ``Assertion``, ``Node``, ``Edge``, ``OutputRecord``.  These are
written now as the architecture specifies but are explicitly amendable until
Phase 5, 6 and 10 respectively.  Each carries that note in its docstring so that
nobody designs around a guess.  The event log records ``schema_version`` per
line, so an amendment is a migration rather than a catastrophe.

Serialisation is hand-written rather than delegated to a validation library.
The event log persists to disk permanently and append-only, so the on-disk
format has to be explicit and version-migratable; ten lines per class makes
every format change visible in a diff, which is the right trade for a permanent
log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "Interval",
    "Obligations",
    "CandidateAtom",
    "ProofSet",
    "Turn",
    "SourceSpan",
    "AssertionFlags",
    "Assertion",
    "Node",
    "Edge",
    "OutputRecord",
    "NODE_TYPES",
    "EDGE_TYPES",
    "ATOM_KINDS",
    "ELIGIBILITY",
    "OUTCOMES",
]

SCHEMA_VERSION = "0.1.0"

NODE_TYPES = (
    "Entity",
    "Claim",
    "Value",
    "Event",
    "TimeInterval",
    "Source",
    "Mention",
    "Turn",
    "SourceSpan",
    "Conflict",
)

EDGE_TYPES = (
    "mentioned_in",
    "asserted_by",
    "about_entity",
    "has_value",
    "valid_during",
    "supported_by",
    "same_as",
    "contradicts",
    "supersedes",
    "derived_from",
    "retired_by",
)

ATOM_KINDS = ("node", "edge", "binding")
ASSERTION_KINDS = ("claim", "value", "event", "time")
ELIGIBILITY = ("eligible", "quarantined")
OUTCOMES = ("answer", "abstain", "contested")

# Feature vectors are stored at single precision.  float32 round-trips exactly
# through JSON (widening to Python float is lossless, and repr round-trips the
# double), it halves the log size, and it matches what the embedder emits, so
# no machine ever writes a subtly different byte string for the same vector.
FEAT_DTYPE = np.float32


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _encode_feat(arr: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": [float(x) for x in arr.ravel().tolist()],
    }


def _decode_feat(data: Mapping[str, Any]) -> np.ndarray:
    arr = np.asarray(data["data"], dtype=np.dtype(data["dtype"]))
    return arr.reshape(tuple(data["shape"]))


def _check_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


# --------------------------------------------------------------------------
# Tier A
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A half-open time interval ``[start, end)`` in epoch seconds (UTC).

    ``None`` on either side means unbounded.  ``start == end`` is the empty
    interval and contains nothing.

    **The convention is half-open and is frozen here**, because Phase 1's
    interval arithmetic, Phase 2's temporal toys and Phase 6's ``valid_during``
    edges all depend on it and none of them may re-derive it.  A consequence
    worth stating once: a point-in-time expression has to be widened to a window
    at its natural granularity by Stage A (a date becomes ``[day, next_day)``),
    because a zero-width half-open interval is empty rather than instantaneous.

    Epoch seconds rather than ISO strings because this type exists to be
    *compared and intersected*.  ``Turn.ts`` stays an ISO-8601 string, since it
    is an immutable human-auditable record; conversion happens in Stage A.
    """

    start: float | None = None
    end: float | None = None

    def __post_init__(self) -> None:
        if self.start is not None:
            object.__setattr__(self, "start", _check_finite(self.start, "start"))
        if self.end is not None:
            object.__setattr__(self, "end", _check_finite(self.end, "end"))
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"interval start {self.start} is after end {self.end}")

    @property
    def is_unbounded(self) -> bool:
        return self.start is None and self.end is None

    @property
    def is_empty(self) -> bool:
        return self.start is not None and self.end is not None and self.start == self.end

    def contains(self, t: float) -> bool:
        """Half-open membership: ``start <= t < end``."""
        if self.start is not None and t < self.start:
            return False
        if self.end is not None and t >= self.end:
            return False
        return True

    def overlaps(self, other: "Interval") -> bool:
        """True when the two intervals share at least one instant."""
        lo_a = -math.inf if self.start is None else self.start
        hi_a = math.inf if self.end is None else self.end
        lo_b = -math.inf if other.start is None else other.start
        hi_b = math.inf if other.end is None else other.end
        return max(lo_a, lo_b) < min(hi_a, hi_b)

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Interval":
        return cls(start=data.get("start"), end=data.get("end"))


@dataclass(frozen=True)
class Obligations:
    """What a question formally requires, as typed slots.

    Produced by the obligation parser (Phase 1, architecture fix F2) — exactly
    on the synthetic environment, extractor-derived on real questions.
    """

    entity_anchor: str | None = None
    value_type: str | None = None
    time_constraint: Interval | None = None
    needs_source: bool = False
    aggregate: bool = False

    def active_slots(self) -> tuple[str, ...]:
        """The slots this question actually imposes.

        ``U``'s ``coverage`` term is "fraction of obligation slots addressed",
        which is only well-defined once *which* slots count is fixed.  Fixed
        here so that Phase 1 does not have to invent it: the four requirement
        slots count, and ``aggregate`` does not — it selects a route rather
        than naming something the evidence must supply.

        A question with no active slots yields an empty tuple; by convention
        Phase 1 scores its coverage as 1.0, since there is nothing to cover.
        """
        slots: list[str] = []
        if self.entity_anchor is not None:
            slots.append("entity_anchor")
        if self.value_type is not None:
            slots.append("value_type")
        if self.time_constraint is not None:
            slots.append("time_constraint")
        if self.needs_source:
            slots.append("needs_source")
        return tuple(slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_anchor": self.entity_anchor,
            "value_type": self.value_type,
            "time_constraint": (
                None if self.time_constraint is None else self.time_constraint.to_dict()
            ),
            "needs_source": self.needs_source,
            "aggregate": self.aggregate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Obligations":
        tc = data.get("time_constraint")
        return cls(
            entity_anchor=data.get("entity_anchor"),
            value_type=data.get("value_type"),
            time_constraint=None if tc is None else Interval.from_dict(tc),
            needs_source=bool(data.get("needs_source", False)),
            aggregate=bool(data.get("aggregate", False)),
        )


@dataclass(frozen=True, eq=False)
class CandidateAtom:
    """One selectable unit of evidence.

    ``refs`` is the closure mechanism (architecture fix F10): an atom may be
    added only once every atom it references is already selected.  Edges
    reference their endpoints, bindings reference their referents, and nodes
    reference nothing — the last is enforced here rather than left to
    convention, because it is what makes "nodes first, then edges" a construction
    order that always works, which in turn is what proves the
    unconstructible-valid-terminal rate to be 0.

    Identity is ``atom_id`` and nothing else.  Equality and hashing are defined
    explicitly for two reasons: ``atom_id`` is already content-derived, so two
    atoms sharing one *are* the same atom; and a dataclass-generated ``__eq__``
    over a numpy field returns an array rather than a bool, which would make
    this type unusable in the sets and dict keys the whole pipeline is built on.
    """

    atom_id: str
    kind: str
    refs: tuple[str, ...] = ()
    feat: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=FEAT_DTYPE))
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ATOM_KINDS:
            raise ValueError(f"kind must be one of {ATOM_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "refs", tuple(self.refs))
        if self.kind == "node" and self.refs:
            raise ValueError(
                f"node atom {self.atom_id} has refs {self.refs}; nodes reference "
                "nothing, which is what makes nodes-first construction always valid"
            )
        # Copied, not viewed: the array is frozen below, and freezing a caller's
        # array in place would be a surprising side effect of construction.
        arr = np.array(self.feat, dtype=FEAT_DTYPE, copy=True)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"atom {self.atom_id} has a non-finite feature value")
        arr.flags.writeable = False
        object.__setattr__(self, "feat", arr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateAtom):
            return NotImplemented
        return self.atom_id == other.atom_id

    def __hash__(self) -> int:
        return hash(self.atom_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "kind": self.kind,
            "refs": list(self.refs),
            "feat": _encode_feat(self.feat),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateAtom":
        return cls(
            atom_id=data["atom_id"],
            kind=data["kind"],
            refs=tuple(data.get("refs", ())),
            feat=_decode_feat(data["feat"]),
            label=data.get("label", ""),
        )


@dataclass(frozen=True, eq=False)
class ProofSet:
    """A completed (or partial) evidence set.

    ``atoms`` is a ``frozenset`` because the object *is* a set, not a sequence
    (research plan v1.2 §3.4).  Any list-typed proof set is a bug: it would
    reintroduce the action-order dependence that the canonical state exists to
    remove.

    ``bindings`` maps an answer slot to the atom that fills it.  It is copied on
    construction and must not be mutated afterwards.
    """

    atoms: frozenset[str] = frozenset()
    bindings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", frozenset(self.atoms))
        object.__setattr__(self, "bindings", dict(self.bindings))

    def __len__(self) -> int:
        return len(self.atoms)

    def __contains__(self, atom_id: object) -> bool:
        return atom_id in self.atoms

    def _key(self) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        return (tuple(sorted(self.atoms)), tuple(sorted(self.bindings.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProofSet):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def to_dict(self) -> dict[str, Any]:
        # Sorted, always.  Python randomises string hashing per interpreter run,
        # so an unsorted frozenset serialises in a different order on every
        # launch and two machines would write different bytes for the same set.
        return {
            "atoms": sorted(self.atoms),
            "bindings": dict(sorted(self.bindings.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProofSet":
        return cls(
            atoms=frozenset(data.get("atoms", ())),
            bindings=dict(data.get("bindings", {})),
        )


# --------------------------------------------------------------------------
# Tier B — amendable until the phase named in each docstring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One conversational turn, immutable.  Amendable until Phase 5."""

    turn_id: str
    conv_id: str
    session_id: str
    speaker: str
    ts: str  # ISO-8601 UTC; see Interval for why this one is not epoch seconds
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "conv_id": self.conv_id,
            "session_id": self.session_id,
            "speaker": self.speaker,
            "ts": self.ts,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Turn":
        return cls(**{k: data[k] for k in ("turn_id", "conv_id", "session_id", "speaker", "ts", "text")})


@dataclass(frozen=True)
class SourceSpan:
    """Character offsets into a turn, immutable.  Amendable until Phase 5."""

    span_id: str
    turn_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span offsets [{self.start}, {self.end})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "turn_id": self.turn_id,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceSpan":
        return cls(
            span_id=data["span_id"],
            turn_id=data["turn_id"],
            start=data["start"],
            end=data["end"],
        )


@dataclass(frozen=True)
class AssertionFlags:
    """The four independent flags of research plan v1.2 §3.1.

    They are never collapsed into a single "true" bit.  Grounding is not truth:
    "I was born on Mars" is entailed by its span and false in the world, and the
    schema has to be able to say so.

    ``externally_verified`` names a *source relation* — corroborated outside the
    conversation — not a truth value, which is why a field whose name contains
    "verified" is permitted here while a field meaning "this is true" is not.
    The structural test in ``graft/tests/test_structure.py`` enforces that.
    """

    asserted_by: str
    entailed_by_span: bool = False
    entailed_score: float = 0.0
    externally_verified: bool = False
    current_under_update_policy: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "entailed_score", _check_finite(self.entailed_score, "entailed_score"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "asserted_by": self.asserted_by,
            "entailed_by_span": self.entailed_by_span,
            "entailed_score": self.entailed_score,
            "externally_verified": self.externally_verified,
            "current_under_update_policy": self.current_under_update_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssertionFlags":
        return cls(
            asserted_by=data["asserted_by"],
            entailed_by_span=bool(data.get("entailed_by_span", False)),
            entailed_score=float(data.get("entailed_score", 0.0)),
            externally_verified=bool(data.get("externally_verified", False)),
            current_under_update_policy=bool(data.get("current_under_update_policy", True)),
        )


@dataclass(frozen=True)
class Assertion:
    """An extracted assertion with its provenance.  Amendable until Phase 5.

    ``eligibility`` is set by the Phase-5 support gate (architecture fix F9).
    Quarantined assertions stay in the event log — that is what makes extraction
    quality measurable at all — and never reach graph construction, so an
    invented claim cannot become retrievable evidence.

    Provenance may be multi-span and cross-turn: a claim assembled across turns
    records every supporting span, not one.
    """

    assertion_id: str
    kind: str
    text_norm: str
    spans: tuple[str, ...]
    flags: AssertionFlags
    t_created: str
    eligibility: str = "eligible"

    def __post_init__(self) -> None:
        if self.kind not in ASSERTION_KINDS:
            raise ValueError(f"kind must be one of {ASSERTION_KINDS}, got {self.kind!r}")
        if self.eligibility not in ELIGIBILITY:
            raise ValueError(f"eligibility must be one of {ELIGIBILITY}, got {self.eligibility!r}")
        object.__setattr__(self, "spans", tuple(self.spans))
        if not self.spans:
            raise ValueError(
                f"assertion {self.assertion_id} has no spans; an assertion without "
                "provenance cannot be grounded and must not be storable"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "kind": self.kind,
            "text_norm": self.text_norm,
            "spans": list(self.spans),
            "flags": self.flags.to_dict(),
            "t_created": self.t_created,
            "eligibility": self.eligibility,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Assertion":
        return cls(
            assertion_id=data["assertion_id"],
            kind=data["kind"],
            text_norm=data["text_norm"],
            spans=tuple(data["spans"]),
            flags=AssertionFlags.from_dict(data["flags"]),
            t_created=data["t_created"],
            eligibility=data.get("eligibility", "eligible"),
        )


@dataclass(frozen=True)
class Node:
    """A typed graph node.  Amendable until Phase 6."""

    node_id: str
    ntype: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ntype not in NODE_TYPES:
            raise ValueError(f"ntype must be one of {NODE_TYPES}, got {self.ntype!r}")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "ntype": self.ntype, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        return cls(node_id=data["node_id"], ntype=data["ntype"], payload=data.get("payload", {}))


@dataclass(frozen=True)
class Edge:
    """A typed, versioned graph edge.  Amendable until Phase 6.

    **Nothing is ever deleted.**  Supersession sets ``t_invalid`` and
    ``superseded_by``; the edge stays in the log, so a wrong supersession is
    recoverable.  This follows Zep's bi-temporal edge-invalidation model.

    ``provenance`` is non-empty by construction: a schema that permits an
    unsourced edge permits an unsourced proof.  If Phase 6 finds a purely
    structural edge type that genuinely has no span, that is the moment to
    amend this rule deliberately — not to pass an empty list today.
    """

    edge_id: str
    etype: str
    src: str
    dst: str
    t_created: str
    provenance: tuple[str, ...]
    t_invalid: str | None = None
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        if self.etype not in EDGE_TYPES:
            raise ValueError(f"etype must be one of {EDGE_TYPES}, got {self.etype!r}")
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if not self.provenance:
            raise ValueError(f"edge {self.edge_id} has no provenance spans")

    @property
    def is_live(self) -> bool:
        return self.t_invalid is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "etype": self.etype,
            "src": self.src,
            "dst": self.dst,
            "t_created": self.t_created,
            "provenance": list(self.provenance),
            "t_invalid": self.t_invalid,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        return cls(
            edge_id=data["edge_id"],
            etype=data["etype"],
            src=data["src"],
            dst=data["dst"],
            t_created=data["t_created"],
            provenance=tuple(data["provenance"]),
            t_invalid=data.get("t_invalid"),
            superseded_by=data.get("superseded_by"),
        )


@dataclass(frozen=True)
class OutputRecord:
    """What the read path returns.  Amendable until Phase 10.

    ``config_hash`` and ``ledger_snapshot`` are part of the record rather than
    side channels, so that "same frozen reader, same prompt, same budget for
    every compared system" is enforced by hash equality rather than by promise.
    """

    outcome: str
    citations: tuple[str, ...] = ()
    answer_text: str | None = None
    proofset: ProofSet | None = None
    ledger_snapshot: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "ledger_snapshot", dict(self.ledger_snapshot))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "citations": list(self.citations),
            "answer_text": self.answer_text,
            "proofset": None if self.proofset is None else self.proofset.to_dict(),
            "ledger_snapshot": dict(self.ledger_snapshot),
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputRecord":
        ps = data.get("proofset")
        return cls(
            outcome=data["outcome"],
            citations=tuple(data.get("citations", ())),
            answer_text=data.get("answer_text"),
            proofset=None if ps is None else ProofSet.from_dict(ps),
            ledger_snapshot=data.get("ledger_snapshot", {}),
            config_hash=data.get("config_hash", ""),
        )
