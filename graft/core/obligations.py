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
    "set_learned_parser",
    "learned_parser",
    "slot_level_scores",
    "OPTIONAL_SLOTS",
    "BOOLEAN_SLOTS",
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


#: The learned parser, installed by ``graft.ingest.oblparse.register`` (Phase 5).
#:
#: **A slot, not an import.**  ``graft.core`` may not import anything that could
#: reach a model — that prohibition is v1.2 §4.4 expressed as a property of the
#: import graph, and it is what keeps ``H`` a predicate.  So the dependency
#: points the other way: the core owns the *routing table* of §4.4 and Phase 5
#: owns the *implementation*, and the boundary stays checkable.
_LEARNED_PARSER: Any = None


def set_learned_parser(parser: Any) -> None:
    """Install (or, with ``None``, remove) the learned obligation parser.

    Callable-or-``None`` is the whole contract; the core deliberately knows
    nothing about what is behind it.  Removing it restores the Phase-1 refusal,
    which is what a test needs in order to assert that the refusal still exists.
    """
    global _LEARNED_PARSER
    if parser is not None and not callable(parser):
        raise TypeError(f"learned parser must be callable, got {type(parser).__name__}")
    _LEARNED_PARSER = parser


def learned_parser() -> Any:
    """The installed learned parser, or ``None``."""
    return _LEARNED_PARSER


def parse(question: Any, mode: str = "exact") -> Obligations:
    """Question → typed obligation slots.

    ``mode="exact"``
        The synthetic instance carries its obligations, so this is a lookup with
        zero error.  Thin by design, not by shortcut: the parser only becomes a
        component when there is text to parse.
    ``mode="learned"``
        Phase 5's extractor fills the slots, through the parser installed by
        :func:`set_learned_parser`.  Raises when none is installed, so a caller
        cannot silently get exact-mode behaviour on real questions.  Its
        slot-level quality must be audited with :func:`slot_level_scores` before
        any downstream use, and reported wherever coverage is reported (fix F2).
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
        if _LEARNED_PARSER is None:
            raise NotImplementedError(
                "no learned obligation parser is installed. Phase 5 provides one: "
                "graft.ingest.oblparse.register(ObligationParser(...)). Its "
                "slot-level precision must be audited on a labelled pilot before "
                "any downstream use, and reported wherever coverage is reported."
            )
        result = _LEARNED_PARSER(question)
        if not isinstance(result, Obligations):
            raise TypeError(
                f"the learned parser returned {type(result).__name__}, not "
                "Obligations; the core's routing contract is typed slots in and "
                "typed slots out"
            )
        return result
    raise ValueError(f"mode must be 'exact' or 'learned', got {mode!r}")


#: Slots that may be absent.  Precision and recall are meaningful here: a
#: prediction is either made or not.
OPTIONAL_SLOTS: tuple[str, ...] = ("entity_anchor", "value_type", "time_constraint", "scope")

#: Slots that are always present as ``True``/``False``.  Precision and recall do
#: not fit — ``False`` is a prediction, not an absence — so these get accuracy.
BOOLEAN_SLOTS: tuple[str, ...] = ("needs_source", "aggregate")


def _is_absent(value: Any) -> bool:
    return value is None or value == ()


def slot_level_scores(
    predicted: Iterable[Obligations], gold: Iterable[Obligations]
) -> dict[str, float]:
    """Per-slot parser quality against gold obligations, as a flat dict.

    Keys are ``"{slot}.{metric}"`` — ``entity_anchor.precision``,
    ``needs_source.accuracy``, and so on — so the result drops straight into a
    metrics table without nested shapes.

    Defined here so the Phase-5 audit number exists the moment the extractor
    does, rather than being invented alongside the thing it audits (architecture
    fix F2 requires it to be reported wherever coverage is reported).

    **Why this replaced a function called ``slot_level_precision``.** That one
    scored only the items where *gold* had the slot active, so a parser
    hallucinating an anchor on every question gold left blank scored 1.0. It was
    a recall-shaped number wearing the name precision, and it would have gone
    into a paper — exactly the overreaching pattern ``CLAUDE.md`` §5 catalogues.
    It also silently omitted ``aggregate``.

    Two slot families, because one metric does not fit both:

    *Optional slots* get precision (of the predictions made, how many were
    right), recall (of the slots gold imposed, how many were recovered) and F1.
    A slot the parser never predicts and gold never imposes yields ``nan``
    rather than a flattering 1.0 — there is nothing to score, and saying so is
    better than inventing a number.

    *Boolean slots* get accuracy over every item, since there is no such thing as
    declining to predict them.
    """
    predicted, gold = list(predicted), list(gold)
    if len(predicted) != len(gold):
        raise ValueError(f"got {len(predicted)} predictions for {len(gold)} gold items")

    out: dict[str, float] = {}

    for name in OPTIONAL_SLOTS:
        tp = fp = fn = 0
        for p, g in zip(predicted, gold):
            pv, gv = getattr(p, name), getattr(g, name)
            p_active, g_active = not _is_absent(pv), not _is_absent(gv)
            if p_active and g_active and pv == gv:
                tp += 1
            else:
                # A wrong value counts once on each side: a prediction that was
                # not right, and a gold slot that was not recovered.
                if p_active:
                    fp += 1
                if g_active:
                    fn += 1
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        recall = tp / (tp + fn) if (tp + fn) else math.nan
        if math.isnan(precision) and math.isnan(recall):
            # Genuinely nothing to score: no prediction and no gold anywhere.
            f1 = math.nan
        elif precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            # There WAS something to score and the parser got all of it wrong.
            # NaN here would make a maximally-bad parser indistinguishable from
            # an unexercised slot — the metric ambiguity §5.5 was fixed for,
            # one row over.  The conventional value is 0.  (13 Aug 2026.)
            f1 = 0.0
        out[f"{name}.precision"] = precision
        out[f"{name}.recall"] = recall
        out[f"{name}.f1"] = f1

    for name in BOOLEAN_SLOTS:
        if not gold:
            out[f"{name}.accuracy"] = math.nan
            continue
        hits = sum(1 for p, g in zip(predicted, gold) if getattr(p, name) == getattr(g, name))
        out[f"{name}.accuracy"] = hits / len(gold)

    return out
