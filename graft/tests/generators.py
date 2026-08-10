"""Seeded random instance generators for the property-based round-trip tests.

Hand-rolled rather than using ``hypothesis``.  Phase 0's dependency set is
PyYAML + numpy + pytest and nothing else, which is what lets the suite run
unchanged inside a Kaggle notebook or on a machine with no network.  A seeded
generator gives reproducible counterexamples, which is the property that
actually matters here.
"""

from __future__ import annotations

import random
import string

import numpy as np

from graft import ids
from graft.schemas import (
    ASSERTION_KINDS,
    ATOM_KINDS,
    EDGE_TYPES,
    ELIGIBILITY,
    NODE_TYPES,
    OUTCOMES,
    Assertion,
    AssertionFlags,
    CandidateAtom,
    CheckResult,
    Edge,
    Interval,
    Node,
    Obligations,
    OutputRecord,
    ProofSet,
    SourceSpan,
    Turn,
    Violation,
)

_ALPHABET = string.ascii_letters + string.digits + " _-.:'"


def text(rng: random.Random, lo: int = 1, hi: int = 40) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(lo, hi)))


def token(rng: random.Random, n: int = 16) -> str:
    return "".join(rng.choice(string.hexdigits.lower()[:16]) for _ in range(n))


def iso_ts(rng: random.Random) -> str:
    return (
        f"20{rng.randint(20, 30):02d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}+00:00"
    )


def interval(rng: random.Random) -> Interval:
    mode = rng.randint(0, 3)
    if mode == 0:
        return Interval()
    if mode == 1:
        return Interval(start=float(rng.randint(0, 10_000)))
    if mode == 2:
        return Interval(end=float(rng.randint(0, 10_000)))
    start = float(rng.randint(0, 10_000))
    return Interval(start=start, end=start + rng.randint(0, 5_000))


def nonempty_interval(rng: random.Random) -> Interval:
    """An interval usable as a ``time_constraint``.

    ``interval`` may return the empty interval (``start == end``), which
    ``Obligations`` refuses because it has measure zero and no instant satisfies
    it (Phase-1 gap G5).
    """
    while True:
        iv = interval(rng)
        if not iv.is_empty:
            return iv


def obligations(rng: random.Random) -> Obligations:
    return Obligations(
        entity_anchor=text(rng) if rng.random() < 0.75 else None,
        value_type=text(rng, 3, 12) if rng.random() < 0.6 else None,
        time_constraint=nonempty_interval(rng) if rng.random() < 0.5 else None,
        needs_source=rng.random() < 0.5,
        aggregate=rng.random() < 0.3,
        scope=tuple(token(rng, 8) for _ in range(rng.randint(0, 2))),
    )


def candidate_atom(rng: random.Random) -> CandidateAtom:
    """A single well-formed atom.

    Its ``refs`` are random tokens, so a *collection* of these will not form a
    valid :class:`AtomPool` — see ``graft/tests/fixtures.py`` for coherent pools.
    This generator exists for schema round-trips, which is what it was written
    for, and that limitation is deliberate rather than an oversight.
    """
    kind = rng.choice(ATOM_KINDS)
    refs = () if kind == "node" else tuple(token(rng) for _ in range(rng.randint(1, 3)))
    dim = rng.randint(0, 8)
    feat = np.asarray(
        [rng.uniform(-5.0, 5.0) for _ in range(dim)], dtype=np.float32
    )
    # A binding names the slot it fills and denotes no graph object; a node or
    # edge atom denotes one and carries no slot.
    if kind == "binding":
        label, target = rng.choice(["answer", "subject", "object"]), ""
    else:
        label, target = text(rng, 0, 10), token(rng)
    return CandidateAtom(
        atom_id=ids.atom_id(kind, refs, target, label),
        kind=kind,
        refs=refs,
        feat=feat,
        label=label,
        target=target,
    )


def violation(rng: random.Random) -> Violation:
    return Violation(
        check=rng.choice(["size", "closure", "eligibility", "temporal", "scope"]),
        message=text(rng, 5, 60),
        atoms=tuple(token(rng) for _ in range(rng.randint(0, 3))),
    )


def check_result(rng: random.Random) -> CheckResult:
    # A CheckResult is ok iff it has no violations; the class enforces it.
    if rng.random() < 0.5:
        return CheckResult(ok=True)
    return CheckResult(
        ok=False,
        violations=tuple(violation(rng) for _ in range(rng.randint(1, 4))),
    )


def proof_set(rng: random.Random) -> ProofSet:
    atoms = {token(rng) for _ in range(rng.randint(0, 8))}
    bindings = {text(rng, 3, 8): token(rng) for _ in range(rng.randint(0, 3))}
    return ProofSet(atoms=frozenset(atoms), bindings=bindings)


def turn(rng: random.Random) -> Turn:
    return Turn(
        turn_id=token(rng),
        conv_id=token(rng, 8),
        session_id=token(rng, 8),
        speaker=rng.choice(["user", "assistant", "system"]),
        ts=iso_ts(rng),
        text=text(rng, 0, 120),
    )


def source_span(rng: random.Random) -> SourceSpan:
    turn_id = token(rng)
    start = rng.randint(0, 500)
    end = start + rng.randint(0, 200)
    return SourceSpan(span_id=ids.span_id(turn_id, start, end), turn_id=turn_id, start=start, end=end)


def assertion(rng: random.Random) -> Assertion:
    kind = rng.choice(ASSERTION_KINDS)
    text_norm = text(rng, 1, 60)
    spans = tuple(token(rng) for _ in range(rng.randint(1, 3)))
    return Assertion(
        assertion_id=ids.assertion_id(kind, text_norm, spans),
        kind=kind,
        text_norm=text_norm,
        spans=spans,
        flags=AssertionFlags(
            asserted_by=rng.choice(["user", "assistant"]),
            entailed_by_span=rng.random() < 0.7,
            entailed_score=round(rng.random(), 6),
            externally_verified=rng.random() < 0.1,
            current_under_update_policy=rng.random() < 0.9,
        ),
        t_created=iso_ts(rng),
        eligibility=rng.choice(ELIGIBILITY),
    )


def node(rng: random.Random) -> Node:
    ntype = rng.choice(NODE_TYPES)
    key = text(rng, 1, 20)
    return Node(
        node_id=ids.node_id(ntype, key),
        ntype=ntype,
        payload={"key": key, "n": rng.randint(0, 100)},
    )


def edge(rng: random.Random, src: str | None = None, dst: str | None = None) -> Edge:
    etype = rng.choice(EDGE_TYPES)
    src = src or token(rng)
    dst = dst or token(rng)
    invalid = rng.random() < 0.25
    return Edge(
        edge_id=ids.edge_id(etype, src, dst, str(rng.randint(0, 3))),
        etype=etype,
        src=src,
        dst=dst,
        t_created=iso_ts(rng),
        provenance=tuple(token(rng) for _ in range(rng.randint(1, 2))),
        t_invalid=iso_ts(rng) if invalid else None,
        superseded_by=token(rng) if invalid and rng.random() < 0.5 else None,
    )


def output_record(rng: random.Random) -> OutputRecord:
    outcome = rng.choice(OUTCOMES)
    return OutputRecord(
        outcome=outcome,
        citations=tuple(token(rng) for _ in range(rng.randint(0, 4))),
        answer_text=text(rng, 0, 80) if outcome != "abstain" else None,
        proofset=proof_set(rng) if rng.random() < 0.8 else None,
        ledger_snapshot={"terminal_checks": rng.randint(0, 32)},
        config_hash=token(rng, 64),
    )


#: Every schema class paired with its generator.  The round-trip test iterates
#: this, so a new schema class that is not added here fails the coverage check.
GENERATORS = {
    Interval: interval,
    Obligations: obligations,
    CandidateAtom: candidate_atom,
    ProofSet: proof_set,
    Violation: violation,
    CheckResult: check_result,
    Turn: turn,
    SourceSpan: source_span,
    Assertion: assertion,
    Node: node,
    Edge: edge,
    OutputRecord: output_record,
}
