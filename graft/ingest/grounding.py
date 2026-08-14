"""P5.4 — the four-rung grounding ladder (decision 5, gap G5).

The model is asked for **exact quotes and never for offsets**: LLM character
offsets are unreliable, and the spike established the working pattern of
recovering them by search.  What the spike also established is the failure mode
this module exists to fix — one recovered span (`d1_0022`) came back with a
leading ``"s "`` glued to *Sankeien Garden*, a fuzzy window that landed one word
off.  Phase 6 trains on these offsets, so a mis-bound is not cosmetic.

**The ladder, frozen:**

1. exact substring match;
2. case- and whitespace-normalised exact match, offsets mapped back;
3. fuzzy window (``difflib``, acceptance ≥ 0.85) **with word-boundary snapping**;
4. failure — dropped and counted, per plan §3.1.

**Snapping is decided by measurement, not by rule.**  "Snap to the nearest word
boundary unless the quote starts or ends mid-word" cannot be applied directly,
because whether a quote starts mid-word is not knowable from the quote alone —
``"arden"`` is a legitimate mid-word quote of *Garden*.  So the ladder
*generates* the boundary-snapped candidates and keeps whichever scores highest
against the quote the model actually wrote, ties going to the unsnapped window.
That fixes the measured defect (dropping a leading partial word raises the
ratio) and cannot damage a genuinely mid-word quote (snapping would lower it).

**Every rung is reported per span** and the mis-bound rate is a *reported
number* with the same status as the quarantine rate: a signal, never a threshold
to quietly tune.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Mapping

from graft.ingest.pins import FUZZY_ACCEPT, GROUNDING_RUNGS
from graft.ingest.records import Grounding, GroundedQuote, Quote
from graft.ids import span_id

__all__ = [
    "ground",
    "ground_quote",
    "ground_assertion_quotes",
    "normalise",
    "rung_counts",
    "RUNG_EXACT",
    "RUNG_NORMALISED",
    "RUNG_FUZZY",
    "RUNG_FAILED",
]

RUNG_EXACT, RUNG_NORMALISED, RUNG_FUZZY, RUNG_FAILED = GROUNDING_RUNGS

_WORD = re.compile(r"\w")
_WS = re.compile(r"\s+")


def normalise(text: str) -> tuple[str, list[int]]:
    """Lowercased, whitespace-collapsed text plus a map back to original offsets.

    The map is what makes rung 2 usable: a normalised match is worthless if the
    stored offsets index the normalised string, because ``SourceSpan`` offsets
    are checked against the *raw* turn on replay (exit criterion 1).
    ``index[i]`` is the original offset **at which** normalised character ``i``
    starts, and the list carries one trailing entry so ``index[end]`` is always
    defined.

    **``index`` is non-decreasing but not strictly increasing**, and a caller
    computing an end offset must not assume otherwise: one original character
    can produce several normalised ones (``'İ'.lower()`` is ``'i'`` plus a
    combining dot), and every one of them maps back to the same original
    offset.  Use :func:`orig_end` for the exclusive end of a normalised match —
    ``index[j + n]`` lands on the *start* of the original character whenever
    the match ends inside an expansion, which stores a span one character short
    or of zero width, at score 1.0 (found by the second audit pass, 14 Aug
    2026, in the fix for the first one).
    """
    out: list[str] = []
    index: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_space or not out:
                continue
            out.append(" ")
            index.append(i)
            in_space = True
        else:
            # ``ch.lower()`` is not always one character — 'İ' (U+0130) lowers
            # to 'i' + a combining dot, and the pinned corpus contains 13 turns
            # with such characters.  Appending the multi-character result as one
            # list element against one index entry shifted every offset after it
            # and could push ``index[jx + len(norm_quote)]`` out of range (found
            # by the 13 Aug 2026 audit).  Each expanded character maps back to
            # the original position, so offsets stay exact.
            for low in ch.lower():
                out.append(low)
                index.append(i)
            in_space = False
    # A collapsed run of trailing whitespace leaves a dangling " " — drop it, so
    # a normalised quote never matches on a space the raw text does not end with.
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    index.append(len(text))
    return "".join(out), index


def orig_end(index: list[int], last: int) -> int:
    """Exclusive original end offset of the normalised character at ``last``.

    The first entry strictly greater than ``index[last]`` — which is one past
    the original character, however many normalised characters it produced.
    ``index`` always ends with ``len(text)``, so the scan terminates.
    """
    start = index[last]
    for k in range(last + 1, len(index)):
        if index[k] > start:
            return index[k]
    return index[-1]


def _is_word(text: str, i: int) -> bool:
    return 0 <= i < len(text) and bool(_WORD.match(text[i]))


def _word_start(text: str, i: int) -> int:
    while i > 0 and _is_word(text, i - 1):
        i -= 1
    return i


def _next_word_start(text: str, i: int) -> int:
    n = len(text)
    while i < n and _is_word(text, i):
        i += 1
    while i < n and not _is_word(text, i):
        i += 1
    return i


def _word_end(text: str, i: int) -> int:
    n = len(text)
    while i < n and _is_word(text, i):
        i += 1
    return i


def _prev_word_end(text: str, i: int) -> int:
    while i > 0 and not _is_word(text, i - 1):
        i -= 1
    while i > 0 and _is_word(text, i - 1):
        i -= 1
    return i


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _snap(text: str, start: int, end: int, quote: str) -> tuple[int, int, bool]:
    """Boundary-snap a fuzzy window, keeping the variant that best fits ``quote``.

    Start and end are snapped independently, because the measured defect moved
    only one edge.  Candidates per edge: leave it, pull out to the enclosing
    word's boundary, push in to the next word's boundary.  Only edges that sit
    *inside* a word are candidates at all — an edge already on a boundary has
    nothing to repair, and generating candidates for it would let a good window
    drift.
    """
    snapped = False

    if _is_word(text, start) and _is_word(text, start - 1):
        options = {start, _word_start(text, start), _next_word_start(text, start)}
        best = max(
            sorted(options),
            key=lambda s: (_ratio(text[s:end], quote), s == start),
        )
        snapped = snapped or best != start
        start = best

    if _is_word(text, end - 1) and _is_word(text, end):
        options = {end, _word_end(text, end), _prev_word_end(text, end)}
        options = {e for e in options if e > start}
        best = max(
            sorted(options),
            key=lambda e: (_ratio(text[start:e], quote), e == end),
        )
        snapped = snapped or best != end
        end = best

    return start, end, snapped


_FUZZY_CANDIDATES = 4


def _fuzzy_window(text: str, quote: str) -> tuple[int, float] | None:
    """Best-matching window of ``len(quote)`` characters, coarse then refined.

    A step-1 sweep is O(n·m) ``SequenceMatcher`` calls and is genuinely slow on
    the multi-thousand-character turns this corpus contains.  A coarse pass at
    ``m // 8`` followed by step-1 refinement is the compromise; the spike used
    the coarse pass alone, which is one of the two reasons its window landed
    off by a word.

    **Refinement covers the top few coarse neighbourhoods, not only the
    winner's.**  Refining around the single coarse best is unsound: a decoy
    region whose alignment happens to sit on the coarse grid can outscore the
    true window's off-grid coarse samples, and the refinement then never visits
    the true region at all (measured, 13 Aug 2026 — a one-typo quote of an
    "I paid forty euros" sentence bound to a nearby "I paid ninety euros"
    sentence at ratio 0.95).  Refining the top ``_FUZZY_CANDIDATES``
    neighbourhoods fixes the measured case at a small constant factor.  It is
    still a heuristic, not a proof — the residual is exactly what the G5
    mis-bound audit (every rung-3 span, hand-checked) exists to measure.
    """
    n, m = len(text), len(quote)
    if m == 0 or m > n:
        return None
    coarse = max(1, m // 8)
    scored = [
        (_ratio(text[start : start + m], quote), start)
        for start in range(0, n - m + 1, coarse)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_ratio, best_start = scored[0]
    seen: set[int] = set()
    for _, anchor in scored[:_FUZZY_CANDIDATES]:
        lo = max(0, anchor - coarse)
        hi = min(n - m, anchor + coarse)
        for start in range(lo, hi + 1):
            if start in seen:
                continue
            seen.add(start)
            r = _ratio(text[start : start + m], quote)
            if r > best_ratio or (r == best_ratio and start < best_start):
                best_ratio, best_start = r, start
    return best_start, best_ratio


def ground(quote: str, text: str) -> Grounding | None:
    """``quote`` → offsets into ``text``, or ``None`` when no rung succeeds.

    Returns the *offsets*, not the text: the caller stores ``text[start:end]``,
    which on rungs 2 and 3 differs from what the model wrote.  Storing the
    model's string beside the turn's offsets would make provenance a fiction —
    the span would not say what the record claims it says.
    """
    if not quote or not quote.strip():
        # A whitespace-only quote would "match" any blank run at rung 1 with
        # score 1.0, storing a blank span as provenance.  The extractor's parser
        # filters these too; this guard is the fail-closed backstop for callers
        # that do not go through it (e.g. the replay extractor).
        return None

    ix = text.find(quote)
    if ix >= 0:
        return Grounding(ix, ix + len(quote), RUNG_EXACT, 1.0)

    norm_text, index = normalise(text)
    norm_quote, _ = normalise(quote)
    if norm_quote:
        jx = norm_text.find(norm_quote)
        if jx >= 0:
            start = index[jx]
            # ``orig_end``, not ``index[jx + len(norm_quote)]``: the map is
            # non-decreasing, so an end landing inside a case expansion would
            # index the *start* of that original character.
            end = orig_end(index, jx + len(norm_quote) - 1)
            # The normalised match can end on collapsed whitespace; trim back to
            # the last non-space character so the stored span has no dangling
            # blank, which would then differ from the same span found exactly.
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                return Grounding(start, end, RUNG_NORMALISED, 1.0)

    found = _fuzzy_window(text, quote)
    if found is None:
        return None
    start, score = found
    if score < FUZZY_ACCEPT:
        return None
    start, end, snapped = _snap(text, start, start + len(quote), quote)
    # Same rule as rung 2's trailing trim, both edges: two windows tying on
    # ratio can differ only by a blank at an edge, and the stored span must not
    # carry it — a span is evidence text, not padding.
    while end > start and text[end - 1].isspace():
        end -= 1
    while start < end and text[start].isspace():
        start += 1
    if end <= start:
        return None
    return Grounding(start, end, RUNG_FUZZY, _ratio(text[start:end], quote), snapped)


def ground_quote(quote: Quote, turn_id: str, text: str) -> GroundedQuote | None:
    """One quote against **its** turn (G9), as a storable span."""
    hit = ground(quote.text, text)
    if hit is None:
        return None
    return GroundedQuote(
        turn_id=turn_id,
        span_id=span_id(turn_id, hit.start, hit.end),
        text=text[hit.start : hit.end],
        grounding=hit,
    )


def ground_assertion_quotes(
    quotes: Iterable[Quote],
    current_turn_id: str,
    texts: Mapping[int, tuple[str, str]],
) -> list[GroundedQuote] | None:
    """Every quote of one assertion, or ``None`` if any of them fails.

    ``texts`` maps a ``turn_offset`` to ``(turn_id, text)`` — the current turn at
    ``0`` and the context window at negative offsets.  **All-or-nothing is the
    rule** (G9): an assertion is grounded iff *every* quote grounds, because a
    multi-span claim whose second span was dropped is a claim with unrecorded
    provenance, and plan §3.1 requires every supporting span, not one.

    An offset outside the window is a failure rather than a fallback to the
    current turn: silently re-homing a quote would attach provenance to a turn
    that does not contain it.
    """
    out: list[GroundedQuote] = []
    for quote in quotes:
        entry = texts.get(quote.turn_offset)
        if entry is None:
            return None
        turn_id, text = entry
        hit = ground_quote(quote, turn_id, text)
        if hit is None:
            return None
        out.append(hit)
    if not out:
        return None
    # Turn order, then position: the NLI premise is "the grounded spans, in turn
    # order" (G6), and the ordering has to be defined somewhere single.
    order = {off: i for i, off in enumerate(sorted(texts))}
    by_turn = {tid: order.get(off, 0) for off, (tid, _) in texts.items()}
    out.sort(key=lambda g: (by_turn.get(g.turn_id, 0), g.grounding.start))
    del current_turn_id  # kept in the signature for call-site legibility
    return out


def rung_counts(groundings: Iterable[Grounding]) -> dict[str, int]:
    """Per-rung tally, every rung present even at zero.

    Zeros are in the table on purpose: "no fuzzy spans" and "the fuzzy count was
    never computed" must not look the same in a report.
    """
    counts = {rung: 0 for rung in GROUNDING_RUNGS}
    for g in groundings:
        counts[g.rung] = counts.get(g.rung, 0) + 1
    return counts
