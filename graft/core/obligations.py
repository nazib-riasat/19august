"""Obligation parsing, the deficit vector ``d(s)``, and interval measure.

``d(s)`` is frozen here at six components (Phase-1 gap G4). Its dimension and
semantics are part of the Gate-2 experimental condition, because L7 —
Contribution 3 — is exactly "L6 plus ``Δd`` as input features". Changing either
re-runs Gate 2.

**``d`` is a feature and an auxiliary target, never an energy.** FL-GFN's
Assumption 4.1 requires a scalar with terminal consistency, and a six-component
vector is not that object (v1.2 §1.2). **[EVIDENCE]** LED-GFN (ICLR 2024 Oral) is
the method built for the regime where intermediate energy is unavailable, which
is why ``d`` enters as conditioning rather than as ``ℰ``.

**It must be evaluable on partial sets.** That is its entire purpose: a dense
signal at every construction step, where ``U`` is only meaningful at terminals.

"Closure" here means the structural refs rule (architecture fix F10), **never**
the old aggregate-query closure certificate. The `GRAFT2_9` documents use the
word the other way and the collision is the kind that silently reintroduces a cut
idea.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from graft.core import resolve
from graft.graphstore import GraphSnapshot
from graft.schemas import AtomPool, Interval, Obligations

__all__ = [
    "DEFICIT_COMPONENTS",
    "SLOT_COMPONENTS",
    "deficit",
    "delta_deficit",
    "slot_status",
    "coverage",
    "covered_fraction",
    "parse",
    "slot_level_precision",
]

#: The six components of ``d(s)``, in order.  Exported so that no downstream
#: phase hard-codes an index — Phase 3 slices this vector into policy features
#: and a positional literal there would be invisible if the order ever moved.
DEFICIT_COMPONENTS: tuple[str, ...] = (
    "anchor",
    "value",
    "time",
    "source",
    "binding",
    "closure",
)

#: The four components that correspond to obligation slots, keyed by the slot
#: name ``Obligations.active_slots()`` uses.  Components 5 and 6 are structural,
#: not slots: ``binding`` is what makes an answer extractable and ``closure`` is
#: the refs rule.
SLOT_COMPONENTS: Mapping[str, str] = {
    "entity_anchor": "anchor",
    "value_type": "value",
    "time_constraint": "time",
    "needs_source": "source",
}

_INDEX = {name: i for i, name in enumerate(DEFICIT_COMPONENTS)}


# --------------------------------------------------------------------------
# interval measure
# --------------------------------------------------------------------------


def covered_fraction(constraint: Interval, intervals: Iterable[Interval]) -> float:
    """Fraction of ``constraint`` covered by the union of ``intervals``, in [0, 1].

    Returns 1.0 when the constraint is unbounded on either side (Phase-1 gap G5).
    The measure is infinite there, so no ratio is meaningful, and scoring it as
    satisfied at least matches ``H``, which equally cannot find a contradiction
    against an unbounded window.  The cost — the term does no work on such
    questions — is why the Phase-2 lattice must emit bounded constraints and why
    Phase 5 reports how often the parser emits unbounded ones.

    The empty constraint cannot reach here: ``Obligations`` refuses it.
    """
    if constraint.start is None or constraint.end is None:
        return 1.0
    total = constraint.end - constraint.start
    if total <= 0:  # pragma: no cover - Obligations rejects the empty interval
        return 1.0

    clipped: list[tuple[float, float]] = []
    for iv in intervals:
        lo = constraint.start if iv.start is None else max(iv.start, constraint.start)
        hi = constraint.end if iv.end is None else min(iv.end, constraint.end)
        if hi > lo:
            clipped.append((lo, hi))
    if not clipped:
        return 0.0

    clipped.sort()
    covered = 0.0
    cur_lo, cur_hi = clipped[0]
    for lo, hi in clipped[1:]:
        if lo > cur_hi:
            covered += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    covered += cur_hi - cur_lo
    return min(1.0, covered / total)


# --------------------------------------------------------------------------
# slot status and the deficit vector
# --------------------------------------------------------------------------


def slot_status(
    atoms: Iterable[str],
    pool: AtomPool,
    q: Obligations,
    G: GraphSnapshot,
) -> dict[str, float]:
    """Deficit in [0, 1] for each of the four obligation slots.

    Zero means satisfied.  A slot the question does not impose has deficit 0 —
    there is nothing outstanding — and is excluded from ``coverage`` by
    ``active_slots()``.

    This is the single implementation shared by ``d(s)`` and ``U``'s
    ``coverage``.  Computing them separately would let the reward and the policy
    feature drift apart silently.
    """
    selected = [pool[aid] for aid in atoms if aid in pool]

    anchor = 1.0
    if q.entity_anchor is None:
        anchor = 0.0
    elif any(resolve.matches_anchor(a, G, q.entity_anchor) for a in selected):
        anchor = 0.0

    value = 1.0
    if q.value_type is None:
        value = 0.0
    elif any(resolve.supplies_value_type(a, G, q.value_type) for a in selected):
        value = 0.0

    if q.time_constraint is None:
        time = 0.0
    else:
        intervals = [iv for a in selected for iv in resolve.atom_intervals(a, G)]
        time = 1.0 - covered_fraction(q.time_constraint, intervals)

    source = _source_deficit(selected, pool, q, G)

    return {"anchor": anchor, "value": value, "time": time, "source": source}


def _source_deficit(
    selected: list[Any], pool: AtomPool, q: Obligations, G: GraphSnapshot
) -> float:
    """Deficit for ``needs_source``.

    **1 when the requirement is active and no binding rests on a resolvable
    span** — not 0 (Phase-1 gap G4).  The obvious phrasing, "fraction of bindings
    with no span, 0 when there are no bindings", is the empty-set defect in
    miniature: a set that binds nothing would score ``needs_source`` as fully
    covered by selecting no binding at all.  ``d_binding`` notices the missing
    binding, but it is component 5 and ``coverage`` reads only 1–4, so the reward
    would genuinely credit the null case.
    """
    if not q.needs_source:
        return 0.0
    bindings = [a for a in selected if a.kind == "binding"]
    if not bindings:
        return 1.0
    unsourced = 0
    for binding in bindings:
        spans, _ = resolve.provenance_spans(binding, G, pool)
        resolvable = any(G.span(sid) is not None for sid in spans)
        if not resolvable:
            unsourced += 1
    return unsourced / len(bindings)


def deficit(
    atoms: Iterable[str],
    pool: AtomPool,
    q: Obligations,
    G: GraphSnapshot,
) -> np.ndarray:
    """``d(s) ∈ [0, 1]^6`` for a partial or complete set."""
    selected = tuple(atoms)
    status = slot_status(selected, pool, q, G)

    present = [aid for aid in selected if aid in pool]
    has_binding = any(pool[aid].kind == "binding" for aid in present)

    if present:
        unresolved = sum(
            1 for aid in present if any(r not in set(present) for r in pool[aid].refs)
        )
        closure = unresolved / len(present)
    else:
        closure = 0.0

    return np.array(
        [
            status["anchor"],
            status["value"],
            status["time"],
            status["source"],
            0.0 if has_binding else 1.0,
            closure,
        ],
        dtype=np.float64,
    )


def delta_deficit(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """``Δd = d(s) − d(s′)``.

    Positive components are obligations the transition *discharged*.  This is the
    quantity L7 conditions ``φ_θ`` and the policy on, and the whole of
    Contribution 3's difference from L6.
    """
    return np.asarray(before, dtype=np.float64) - np.asarray(after, dtype=np.float64)


def coverage(status: Mapping[str, float], q: Obligations) -> float:
    """Fraction of the question's active obligation slots that are addressed.

    **A slot counts as addressed when its deficit is below 1 — some evidence
    bears on it — rather than by one minus its graded deficit.**

    Taking the graded form literally would make ``coverage`` and
    ``temporal_correctness`` share a term exactly: ``temporal_correctness`` is
    ``1 − d_time`` by construction, so ``d_time`` would enter ``U`` twice and
    temporal evidence would carry 2–5× the weight of any other slot depending on
    how many are active.  Nothing in the plan intends that.

    Splitting them binary-versus-graded is the same move as Phase-1 gap G5's
    split of ``H`` from ``U``: ``coverage`` asks *is this obligation addressed at
    all*, ``temporal_correctness`` asks *how precisely*.  Two questions, two
    terms.

    A question with no active slots scores 1.0 — there is nothing to cover.
    """
    active = q.active_slots()
    if not active:
        return 1.0
    addressed = sum(1 for slot in active if status[SLOT_COMPONENTS[slot]] < 1.0)
    return addressed / len(active)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def parse(question: Any, mode: str = "exact") -> Obligations:
    """Question → typed obligation slots.

    ``mode="exact"``
        The synthetic instance carries its obligations, so this is a lookup with
        zero error.  Thin by design, not by shortcut: the parser only becomes a
        component when there is text to parse.
    ``mode="learned"``
        Phase 5's extractor fills the slots.  Raises until then, so a caller
        cannot silently get exact-mode behaviour on real questions.
    """
    if mode == "exact":
        if isinstance(question, Obligations):
            return question
        obligations = getattr(question, "obligations", None)
        if isinstance(obligations, Obligations):
            return obligations
        raise TypeError(
            "exact mode needs an Obligations, or something carrying one; got "
            f"{type(question).__name__}. Obligations are generated with a synthetic "
            "instance rather than recovered from its text."
        )
    if mode == "learned":
        raise NotImplementedError(
            "learned obligation parsing arrives with the Phase-5 extractor. Its "
            "slot-level precision must be audited on a labelled pilot before any "
            "downstream use, and reported wherever coverage is reported."
        )
    raise ValueError(f"mode must be 'exact' or 'learned', got {mode!r}")


def slot_level_precision(
    predicted: Iterable[Obligations], gold: Iterable[Obligations]
) -> dict[str, float]:
    """Per-slot exact-match precision of a parser against gold obligations.

    Defined here so the Phase-5 audit number exists the moment the extractor
    does, rather than being invented alongside the thing it audits.
    """
    predicted, gold = list(predicted), list(gold)
    if len(predicted) != len(gold):
        raise ValueError(f"got {len(predicted)} predictions for {len(gold)} gold items")
    fields = ("entity_anchor", "value_type", "time_constraint", "needs_source", "scope")
    out: dict[str, float] = {}
    for name in fields:
        considered = [
            (p, g) for p, g in zip(predicted, gold) if getattr(g, name) not in (None, False, ())
        ]
        if not considered:
            out[name] = math.nan
            continue
        hits = sum(1 for p, g in considered if getattr(p, name) == getattr(g, name))
        out[name] = hits / len(considered)
    return out
