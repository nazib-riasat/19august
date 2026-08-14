"""The in-flight shapes of Stage A, between the model and the event log.

**None of these is a dataclass, and none of them is persisted.**  Phase-0
criterion 12 makes ``schemas.py`` the single home of the data model, and G9's
Tier-B freeze says the persisted shapes are ``Turn``, ``SourceSpan`` and
``Assertion`` — full stop.  What lives here is the *transient* stage between an
LLM reply and those records: a parsed extraction, a grounded quote, a per-turn
report.  They exist because passing raw dicts between six modules is how a typo
becomes a silent empty list, and they use ``__slots__`` for the same reason
Phase 4's ``SearchResult`` does.

The one rule they all obey: **a field that would mean "this is true" does not
exist here either.**  ``grounded`` means an offset was recovered, ``entailed``
means a model scored the span above a threshold, and both are recorded next to
*how* they were obtained so that a downstream reader cannot mistake either for a
claim about the world (plan §3.1, §4.4).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

__all__ = [
    "Quote",
    "RawAssertion",
    "RawExtraction",
    "Grounding",
    "GroundedQuote",
    "DraftAssertion",
    "TurnReport",
]


class Quote:
    """A verbatim span the model claims supports an assertion, before grounding.

    ``turn_offset`` is G9's cross-turn provenance: ``0`` is the turn being
    processed, ``-1`` the one before it.  A positive offset is a model error and
    is rejected at construction rather than silently clamped — it would ground
    against a turn that had not been said yet.
    """

    __slots__ = ("turn_offset", "text")

    def __init__(self, turn_offset: int, text: str) -> None:
        self.turn_offset = int(turn_offset)
        self.text = str(text)
        if self.turn_offset > 0:
            raise ValueError(
                f"turn_offset {self.turn_offset} is in the future; quotes may only "
                "reference the current turn (0) or earlier ones"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"turn_offset": self.turn_offset, "text": self.text}


class RawAssertion:
    """One assertion as the model emitted it: a kind, a normalised sentence, quotes."""

    __slots__ = ("kind", "text_norm", "quotes")

    def __init__(self, kind: str, text_norm: str, quotes: Iterable[Quote]) -> None:
        self.kind = str(kind)
        self.text_norm = str(text_norm)
        self.quotes = tuple(quotes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text_norm": self.text_norm,
            "quotes": [q.to_dict() for q in self.quotes],
        }


class RawExtraction:
    """One turn's parsed extraction, plus how much it cost and how it went.

    ``parse_ok`` is **False on a repair-exhausted turn**, and the turn still
    flows through the pipeline carrying empty lists.  The spike's 15.5% parse
    failures were invisible precisely because a failure and an empty extraction
    look identical downstream; keeping the flag on the record is what lets the
    pilot report distinguish "this turn said nothing extractable" from "this turn
    was lost".

    Token counts are filled **inside** the extractor wrapper (exit criterion 14).
    A caller that counted them would drift, which is the same argument Phase 0
    made for ``terminal_checks`` being counted inside the checker.
    """

    __slots__ = (
        "mentions",
        "assertions",
        "parse_ok",
        "repairs",
        "llm_calls",
        "tokens_in",
        "tokens_out",
        "error",
        "truncated",
        "truncations",
        "raw",
    )

    def __init__(
        self,
        mentions: Iterable[str] = (),
        assertions: Iterable[RawAssertion] = (),
        *,
        parse_ok: bool = True,
        repairs: int = 0,
        llm_calls: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: str | None = None,
        truncated: bool = False,
        truncations: int = 0,
        raw: str = "",
    ) -> None:
        self.mentions = tuple(str(m) for m in mentions)
        self.assertions = tuple(assertions)
        self.parse_ok = bool(parse_ok)
        self.repairs = int(repairs)
        self.llm_calls = int(llm_calls)
        self.tokens_in = int(tokens_in)
        self.tokens_out = int(tokens_out)
        self.error = error
        # **Whether the *final* generation hit ``max_new_tokens``.**  A truncated
        # JSON object has no closing brace, so it fails to parse for a reason that
        # has nothing to do with the model's ability to emit JSON — and a bare
        # "parse failure rate" cannot tell the two apart.  Distinguishing them is
        # the difference between "this extractor cannot produce valid JSON" and
        # "this budget cannot hold the extractor's output", which have opposite
        # fixes.  ``truncated`` attributes the final outcome (it is what decides
        # the ``truncated_at_token_cap`` failure cause); ``truncations`` counts
        # **every** capped generation across repair attempts — without it, a
        # first-attempt truncation repaired on the retry disappears from the
        # count while its tokens stay in ``tokens_out``, and the row becomes
        # internally inconsistent (found by the 13 Aug 2026 audit).
        self.truncated = bool(truncated)
        self.truncations = int(truncations)
        self.raw = raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "mentions": list(self.mentions),
            "assertions": [a.to_dict() for a in self.assertions],
            "parse_ok": self.parse_ok,
            "repairs": self.repairs,
            "llm_calls": self.llm_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error": self.error,
            "truncated": self.truncated,
            "truncations": self.truncations,
        }


class Grounding:
    """Where a quote landed in a turn, and which rung of the ladder put it there.

    ``rung`` is reported per span (decision 5) because Phase 6 trains on these
    offsets: an exact match and a fuzzy window are not the same evidence quality,
    and a table that reports only "grounded: yes" has thrown that away.

    ``snapped`` records that word-boundary snapping moved the window — the
    specific repair G5 adds after the spike produced one mis-bounded span.
    """

    __slots__ = ("start", "end", "rung", "score", "snapped")

    def __init__(
        self, start: int, end: int, rung: str, score: float = 1.0, snapped: bool = False
    ) -> None:
        self.start = int(start)
        self.end = int(end)
        self.rung = str(rung)
        self.score = float(score)
        self.snapped = bool(snapped)
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid grounding offsets [{self.start}, {self.end})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "rung": self.rung,
            "score": self.score,
            "snapped": self.snapped,
        }


class GroundedQuote:
    """A quote resolved against a *specific* turn, with the text actually found.

    ``text`` is the substring the offsets select, **not** what the model wrote.
    They differ on rungs 2 and 3, and the stored one has to be the turn's, or the
    span offsets and the span text would disagree and provenance would be a
    fiction.
    """

    __slots__ = ("turn_id", "span_id", "text", "grounding")

    def __init__(self, turn_id: str, span_id: str, text: str, grounding: Grounding) -> None:
        self.turn_id = turn_id
        self.span_id = span_id
        self.text = text
        self.grounding = grounding

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "span_id": self.span_id,
            "text": self.text,
            **self.grounding.to_dict(),
        }


class DraftAssertion:
    """A grounded assertion on its way to the log, before the verifier has run.

    Carries the provenance attributes the quarantine breakdown is cross-tabulated
    against — whether the extraction it came from needed a repair, and the worst
    rung any of its spans needed — because both are properties of *how* the
    record was obtained and neither is knowable from the stored ``Assertion``.
    """

    __slots__ = ("assertion_id", "kind", "text_norm", "quotes", "asserted_by", "from_repair")

    def __init__(
        self,
        assertion_id: str,
        kind: str,
        text_norm: str,
        quotes: Sequence[GroundedQuote],
        asserted_by: str,
        from_repair: bool | None = False,
    ) -> None:
        self.assertion_id = assertion_id
        self.kind = kind
        self.text_norm = text_norm
        self.quotes = tuple(quotes)
        self.asserted_by = asserted_by
        #: ``None`` = **not recoverable**, which is what a draft rebuilt from
        #: the log has to say: the repair count lives nowhere in the permanent
        #: record (``Assertion`` stores no extraction provenance), so a resumed
        #: run reporting ``False`` would silently claim a clean extraction it
        #: cannot know about — and ``cross_tab_repaired_extraction`` would read
        #: as measured zeros.  The rung has the same problem and already says
        #: ``"resumed"``; this is the same disclosure for the other attribute
        #: (found by the second audit pass, 14 Aug 2026).
        self.from_repair = None if from_repair is None else bool(from_repair)

    @property
    def span_ids(self) -> tuple[str, ...]:
        return tuple(q.span_id for q in self.quotes)

    @property
    def worst_rung(self) -> str:
        """The least-exact rung any of this assertion's spans needed.

        Least-exact rather than an average: an assertion is only as well grounded
        as its weakest span, and an average over two spans would hide a fuzzy one
        behind an exact one.

        A rung outside the declared ladder — ``resumed``, which the pipeline uses
        when a crashed run's span offsets are re-read from the log without their
        provenance — sorts as the worst.  Fail-closed again: an unknown
        provenance is not evidence of a clean one.
        """
        from graft.ingest.pins import GROUNDING_RUNGS

        order = {name: i for i, name in enumerate(GROUNDING_RUNGS)}
        unknown = len(order)
        return max(
            (q.grounding.rung for q in self.quotes), key=lambda r: order.get(r, unknown)
        )

    @property
    def premise(self) -> str:
        """The NLI premise: the grounded span texts, in turn order, and nothing else.

        **Nothing else is the specification** (G6).  Entailment is *by the span*;
        if the conversation were in the premise, ``entailed_by_span`` would stop
        meaning what its name says and the support gate would be admitting claims
        the span does not carry.
        """
        return " ".join(q.text for q in self.quotes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "kind": self.kind,
            "text_norm": self.text_norm,
            "quotes": [q.to_dict() for q in self.quotes],
            "asserted_by": self.asserted_by,
            "from_repair": self.from_repair,
            "worst_rung": self.worst_rung,
        }


class TurnReport:
    """What one turn cost and produced.  One row of ``metrics.jsonl``."""

    __slots__ = (
        "turn_id",
        "skipped",
        "parse_ok",
        "repairs",
        "truncations",
        "mentions",
        "assertions_extracted",
        "assertions_stored",
        "quotes_total",
        "rungs",
        "dropped_assertions",
        "duplicate_assertions",
        "dropped_mentions",
        "llm_calls",
        "tokens_in",
        "tokens_out",
        "seconds",
    )

    def __init__(self, turn_id: str, **kwargs: Any) -> None:
        self.turn_id = turn_id
        self.skipped = False
        self.parse_ok = True
        self.repairs = 0
        self.truncations = 0
        self.mentions = 0
        self.assertions_extracted = 0
        self.assertions_stored = 0
        self.quotes_total = 0
        self.rungs: dict[str, int] = {}
        self.dropped_assertions = 0
        #: Extracted twice inside one turn with identical kind/text/spans, so
        #: written once.  Counted because otherwise ``assertions_extracted``
        #: and ``assertions_stored + assertions_dropped_ungrounded`` disagree
        #: with nothing explaining the gap — which is exactly what the 14 Aug
        #: audit found in the live pilot's report (333 vs 330).
        self.duplicate_assertions = 0
        self.dropped_mentions = 0
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}
