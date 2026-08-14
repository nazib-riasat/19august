"""P5.7 — the learned obligation parser (architecture fix F2, gap G7).

Phase 1 built the interface and the audit hook and left ``parse(mode="learned")``
raising a ``NotImplementedError``; this is the implementation, and it lands here
rather than in ``graft.core`` so that the core still imports no ML library and
plan §4.4's routing table stays a module boundary rather than a promise.  The
core keeps only a dispatch slot; :func:`register` fills it.

**Three decisions inside, each of which is a place this could go wrong quietly.**

*The model fills five slots; ``scope`` is metadata and is not asked for.*
``Obligations.scope`` is a tuple of ``conv_id``s that `H`'s sub-check 5 rejects
evidence against.  A hallucinated conversation id would not produce a wrong
answer — it would produce a **silently over-restricted proof search**, rejecting
correct evidence for being out of a scope nobody asked for.  So scope is
supplied by the caller from the question's own binding, and the audit reports it
as metadata-derived rather than as parser quality.

*Relative dates are resolved in code, not by the model* (``graft.ingest.timeexpr``).
The model emits the phrase; the widening to a half-open interval at natural
granularity is arithmetic, is a Phase-0 §2.5 requirement on Stage A, and is
testable without a GPU.

*The unbounded-constraint rate is a reported number.*  An unbounded interval
scores ``temporal_correctness = 1.0`` by the Phase-1 G5 convention, so a parser
that emits them freely quietly disables a reward term.  The Phase-1 plan warned
about it; :class:`ObligationParser` measures it.

The audit (decision 14): ~50 LongMemEval questions hand-labelled for all six
slots, scored with the existing ``slot_level_scores``, and reported **wherever
coverage is reported** — fix F2's own words.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from graft.core import obligations as core_obligations
from graft.ingest import prompts
from graft.ingest.timeexpr import resolve as resolve_time
from graft.schemas import Interval, Obligations

__all__ = [
    "ObligationParser",
    "parse_slots",
    "obligations_from_slots",
    "register",
    "unregister",
    "unbounded_rate",
]

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_slots(raw: str) -> dict[str, Any] | None:
    """A model reply → the five typed slots, or ``None`` when unparseable.

    Absent and empty are both read as *absent*: a model that writes ``""`` for
    ``entity_anchor`` has declined to fill it, and treating an empty string as a
    prediction would count it as a false positive in the audit and, worse, make
    ``active_slots()`` claim an anchor obligation nothing can satisfy.
    """
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, Mapping):
        return None

    def text(name: str) -> str | None:
        value = obj.get(name)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    def flag(name: str) -> bool:
        value = obj.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    return {
        "entity_anchor": text("entity_anchor"),
        "value_type": text("value_type"),
        "time_expression": text("time_expression"),
        "needs_source": flag("needs_source"),
        "aggregate": flag("aggregate"),
    }


def obligations_from_slots(
    slots: Mapping[str, Any],
    question_date: str,
    scope: Sequence[str] = (),
) -> tuple[Obligations, dict[str, Any]]:
    """Typed slots + the question's date and scope → ``Obligations`` and a trace.

    The trace records what the time expression *was* and what it resolved to,
    because "the parser emitted no constraint" and "the parser emitted a phrase
    outside the declared vocabulary" are different failures and only the second
    is fixable by extending :mod:`graft.ingest.timeexpr`.

    An interval that comes back empty is dropped rather than passed on:
    ``Obligations`` refuses the empty interval (Phase-1 gap G5) because no
    instant can satisfy it, and a parser artefact should not be able to raise
    inside a read path.
    """
    phrase = slots.get("time_expression")
    interval: Interval | None = None
    resolved = False
    if phrase:
        try:
            interval = resolve_time(phrase, question_date)
        except ValueError:
            # ``resolve_time`` raises only on a malformed *question_date* (the
            # phrase side fails soft by contract).  On the ``register`` dispatch
            # path a plain-string question arrives with no date at all, and a
            # read path must degrade to "constraint unresolved, counted" rather
            # than crash (found by the 13 Aug 2026 audit: any time-bearing
            # question through ``core_obligations.parse(mode="learned")`` with a
            # bare string died in ``parse_question_date``).
            interval = None
        resolved = interval is not None
    if interval is not None and interval.is_empty:
        interval, resolved = None, False

    obligations = Obligations(
        entity_anchor=slots.get("entity_anchor") or None,
        value_type=slots.get("value_type") or None,
        time_constraint=interval,
        needs_source=bool(slots.get("needs_source", False)),
        aggregate=bool(slots.get("aggregate", False)),
        scope=tuple(scope),
    )
    trace = {
        "time_expression": phrase,
        "time_resolved": resolved,
        "time_unbounded": bool(interval is not None and not (
            interval.start is not None and interval.end is not None
        )),
        "scope_source": "metadata",
    }
    return obligations, trace


class ObligationParser:
    """Question text → ``Obligations``, with the slot trace kept for the audit.

    ``completer`` is ``(system, user) -> (text, tokens_in, tokens_out)`` — the
    shape ``LlmExtractor.complete`` returns — so the **same frozen extractor**
    parses questions and extracts turns.  That is fix F2's own specification
    ("the extractor LLM fills the typed slots") and it is also what keeps one
    prompt-registry SHA covering every prompt this phase runs.
    """

    def __init__(self, completer: Callable[[str, str], tuple[str, int, int]]) -> None:
        self.completer = completer
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.parse_failures = 0
        self.traces: list[dict[str, Any]] = []

    def parse(
        self,
        question: str,
        question_date: str,
        scope: Sequence[str] = (),
        question_id: str | None = None,
    ) -> Obligations:
        user = prompts.OBLIGATION_USER.format(
            question_date=question_date, question=question
        )
        reply, n_in, n_out = self.completer(prompts.OBLIGATION_SYSTEM, user)
        self.calls += 1
        self.tokens_in += int(n_in)
        self.tokens_out += int(n_out)

        slots = parse_slots(reply)
        parse_ok = slots is not None  # captured BEFORE the fallback is installed
        if slots is None:
            # A parse failure yields the **empty obligation**, not a guess.  An
            # Obligations with no active slots scores coverage 1.0 by the Phase-1
            # convention, which is generous — so the failure is counted and
            # reported rather than absorbed, exactly as the extractor's is.
            self.parse_failures += 1
            slots = {
                "entity_anchor": None,
                "value_type": None,
                "time_expression": None,
                "needs_source": False,
                "aggregate": False,
            }
        obligations, trace = obligations_from_slots(slots, question_date, scope)
        trace.update({"question_id": question_id, "parse_ok": parse_ok})
        self.traces.append(trace)
        return obligations

    # -- the reported numbers ---------------------------------------------

    def report(self) -> dict[str, Any]:
        total = len(self.traces)
        with_phrase = sum(1 for t in self.traces if t["time_expression"])
        unresolved = sum(
            1 for t in self.traces if t["time_expression"] and not t["time_resolved"]
        )
        unbounded = sum(1 for t in self.traces if t["time_unbounded"])
        return {
            "questions": total,
            "llm_calls": self.calls,
            "llm_tokens_in": self.tokens_in,
            "llm_tokens_out": self.tokens_out,
            "slot_parse_failures": self.parse_failures,
            "time_expression_rate": (with_phrase / total) if total else float("nan"),
            "time_unresolved_rate": (unresolved / with_phrase) if with_phrase else float("nan"),
            "unbounded_constraint_rate": (unbounded / total) if total else float("nan"),
            "unbounded_reading": (
                "an unbounded interval scores temporal_correctness = 1.0 by the "
                "Phase-1 G5 convention, so this rate is how often the parser "
                "silently disables that reward term"
            ),
        }


def unbounded_rate(obligations: Iterable[Obligations]) -> float:
    """Fraction of parsed questions carrying an unbounded time constraint.

    Standalone as well as on the parser, because the audit scores *gold* and
    *predicted* obligations side by side and gold has no parser attached.
    """
    items = list(obligations)
    if not items:
        return float("nan")
    n = sum(
        1
        for q in items
        if q.time_constraint is not None
        and (q.time_constraint.start is None or q.time_constraint.end is None)
    )
    return n / len(items)


# --------------------------------------------------------------------------
# the graft.core dispatch hook
# --------------------------------------------------------------------------


def register(parser: ObligationParser) -> None:
    """Install ``parser`` as ``core.obligations.parse(mode="learned")``.

    Registration rather than an import: ``graft.core`` may not import anything
    that could reach a model (``test_the_deterministic_core_imports_nothing_it_
    should_not``), so the dependency has to point the other way.  The core owns
    the *routing*, ``graft.ingest`` owns the *implementation*, and the boundary
    is checkable rather than promised.
    """
    core_obligations.set_learned_parser(
        lambda question: parser.parse(
            question=getattr(question, "text", None) or str(question),
            question_date=getattr(question, "question_date", ""),
            scope=getattr(question, "scope", ()),
            question_id=getattr(question, "question_id", None),
        )
    )


def unregister() -> None:
    """Remove the learned parser, restoring the Phase-1 refusal."""
    core_obligations.set_learned_parser(None)
