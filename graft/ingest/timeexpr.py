"""Relative time expressions → half-open ``Interval``s, deterministically.

**This is arithmetic, so it is code, not a prompt** (G7).  The obligation parser
asks the extractor LLM for the time *phrase* the question uses and resolves it
here; asking a model to compute a date range instead adds an error source with
no upside, and makes exit criterion 12 untestable without a GPU.

**Widening to natural granularity is a Phase-0 requirement, not a nicety.**
``Interval`` is half-open, so a zero-width interval contains nothing and
``covered_fraction`` would score any evidence against it as uncovered.
``PHASE0_DECISIONS.md`` §2.5 puts the widening in Stage A explicitly: a date
becomes ``[day, next_day)``, a month becomes ``[first, next_first)``, a year
becomes ``[Jan 1, next Jan 1)``.

**Three conventions are declared here because the language does not settle
them.**  They are choices, and a reader of the write-up should be able to find
them:

* ``"last <month>"`` names the most recent occurrence of that month **strictly
  before** the month the question is asked in — so a question asked in May 2023
  about "last May" means May 2022, not the month it is standing in;
* ``"last week"`` is the previous **ISO calendar week**, Monday to Monday, not a
  rolling seven days — calendar weeks are what a person means by "last week";
* an open-ended expression (``"since 2023"``, ``"before May"``) resolves to a
  genuinely **unbounded** ``Interval``, which by the Phase-1 G5 convention scores
  ``temporal_correctness = 1.0``.  That is why the parser reports the
  unbounded-constraint rate: a parser emitting them freely quietly disables a
  reward term, and the number is the only way to see it.

An expression outside the declared vocabulary returns ``None`` — recorded as
unresolved and counted, never guessed at.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

from graft.schemas import Interval

__all__ = [
    "MONTHS",
    "parse_question_date",
    "resolve",
    "day_interval",
    "month_interval",
    "year_interval",
]

MONTHS: dict[str, int] = {
    name: i + 1
    for i, name in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        )
    )
}
MONTHS.update(
    {
        name[:3]: number
        for name, number in list(MONTHS.items())
    }
)
MONTHS["sept"] = 9

_UNITS = {
    "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "month": "month", "months": "month",
    "year": "year", "years": "year",
}


def _epoch(d: date) -> float:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def parse_question_date(raw: str) -> date:
    """The corpus's ``question_date`` (or a plain ISO date) as a ``date``.

    Accepts ``'2023/05/30 (Tue) 23:40'`` and ``'2023-05-30'`` alike, so a caller
    outside LongMemEval does not have to reformat.
    """
    head = raw.split("(")[0].strip().split(" ")[0].strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(head, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised question_date {raw!r}")


# -- the natural-granularity windows ----------------------------------------


def day_interval(d: date) -> Interval:
    return Interval(_epoch(d), _epoch(d + timedelta(days=1)))


def month_interval(year: int, month: int) -> Interval:
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return Interval(_epoch(start), _epoch(end))


def year_interval(year: int) -> Interval:
    return Interval(_epoch(date(year, 1, 1)), _epoch(date(year + 1, 1, 1)))


def _week_interval(anchor: date) -> Interval:
    monday = anchor - timedelta(days=anchor.weekday())
    return Interval(_epoch(monday), _epoch(monday + timedelta(days=7)))


def _shift_months(d: date, months: int) -> date:
    total = (d.year * 12 + d.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


# -- the resolver -----------------------------------------------------------

_CLEAN = re.compile(r"[^\w\s/-]+")


def _norm(phrase: str) -> str:
    return _CLEAN.sub(" ", phrase.strip().lower()).replace("_", " ").strip()


def _bounded(phrase: str, today: date) -> Interval | None:
    """The closed expressions.  ``None`` when nothing in the vocabulary matches."""
    text = _norm(phrase)
    if not text:
        return None

    # ISO / slashed absolute dates, with or without a day.
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return day_interval(date(y, mo, d))
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", text)
    if m:
        return month_interval(int(m.group(1)), int(m.group(2)))

    words = text.split()

    # today / yesterday / tomorrow
    if text in ("today", "this day"):
        return day_interval(today)
    if text == "yesterday":
        return day_interval(today - timedelta(days=1))
    if text == "tomorrow":
        return day_interval(today + timedelta(days=1))

    # this / last / next {week, month, year}
    m = re.fullmatch(r"(this|last|past|previous|next)\s+(week|month|year)", text)
    if m:
        which, unit = m.groups()
        step = 0 if which == "this" else (1 if which == "next" else -1)
        if unit == "week":
            return _week_interval(today + timedelta(days=7 * step))
        if unit == "month":
            anchor = _shift_months(today, step)
            return month_interval(anchor.year, anchor.month)
        return year_interval(today.year + step)

    # "in 2023" / "2023" / "the year 2023"
    m = re.fullmatch(r"(?:in\s+|during\s+|the\s+year\s+)?(\d{4})", text)
    if m:
        return year_interval(int(m.group(1)))

    # "<month> <year>" / "<year> <month>" / "in <month> <year>"
    tokens = [w for w in words if w not in ("in", "during", "of", "the", "on")]
    if len(tokens) == 2:
        a, b = tokens
        for month_token, year_token in ((a, b), (b, a)):
            if month_token in MONTHS and re.fullmatch(r"\d{4}", year_token):
                return month_interval(int(year_token), MONTHS[month_token])

    # "last May" / "this May" / bare "May"
    if len(tokens) == 2 and tokens[0] in ("last", "past", "previous") and tokens[1] in MONTHS:
        month = MONTHS[tokens[1]]
        year = today.year if month < today.month else today.year - 1
        return month_interval(year, month)
    if len(tokens) == 2 and tokens[0] in ("this",) and tokens[1] in MONTHS:
        month = MONTHS[tokens[1]]
        year = today.year if month <= today.month else today.year - 1
        return month_interval(year, month)
    if len(tokens) == 1 and tokens[0] in MONTHS:
        # A bare month is read the same way as "this <month>": the most recent
        # occurrence at or including the question's own month.  Declared rather
        # than left to the caller, since both readings are defensible and only
        # one can be in the code.
        month = MONTHS[tokens[0]]
        year = today.year if month <= today.month else today.year - 1
        return month_interval(year, month)

    # "last 3 months" / "past 2 weeks" — a rolling window ending at today's end.
    m = re.fullmatch(r"(?:last|past|previous)\s+(\d+)\s+(\w+)", text)
    if m and m.group(2) in _UNITS:
        n, unit = int(m.group(1)), _UNITS[m.group(2)]
        end = today + timedelta(days=1)
        if unit == "day":
            start = end - timedelta(days=n)
        elif unit == "week":
            start = end - timedelta(weeks=n)
        elif unit == "month":
            start = _shift_months(today, -n).replace(day=1)
        else:
            start = date(today.year - n, today.month, 1)
        return Interval(_epoch(start), _epoch(end))

    return None


def _shift_interval_years(interval: Interval, years: int) -> Interval | None:
    """The same calendar interval, ``years`` later.  ``None`` if out of range."""
    try:
        lo = datetime.fromtimestamp(interval.start, tz=timezone.utc).date()
        hi = datetime.fromtimestamp(interval.end, tz=timezone.utc).date()
        return Interval(
            _epoch(lo.replace(year=lo.year + years)),
            _epoch(hi.replace(year=hi.year + years)),
        )
    except ValueError:
        return None


def resolve(phrase: str | None, question_date: str | date) -> Interval | None:
    """A time phrase and the date it was asked on → a half-open ``Interval``.

    ``None`` means *not in the declared vocabulary* — an outcome the parser
    counts rather than papers over, because a silently dropped constraint and a
    question with no constraint are different things and only one of them is a
    parser defect.

    **A calendar-invalid phrase is out of vocabulary, not an exception.**  The
    phrase is model output; a hallucinated "2023-06-31" or "past 3000 years"
    must resolve to ``None`` (counted as unresolved), never raise inside a read
    path — the module's own contract, which the naked ``date()`` constructors
    violated until the 13 Aug 2026 audit reproduced the crash.  The one caller
    error that still raises is a malformed ``question_date``: that value comes
    from the corpus, not the model, and a bad one is a bug worth crashing on.
    """
    if phrase is None:
        return None
    today = question_date if isinstance(question_date, date) else parse_question_date(question_date)
    text = _norm(phrase)
    if not text:
        return None
    try:
        return _resolve_normed(text, today)
    except (ValueError, OverflowError):
        return None


def _resolve_normed(text: str, today: date) -> Interval | None:
    # Open-ended forms.  Resolved against the *inner* expression, then opened on
    # one side — so "since May 2023" starts at the beginning of that month and
    # "before 2023" ends at the beginning of that year.
    m = re.match(r"(since|after|from|later\s+than)\s+(.*)", text)
    if m:
        inner = _bounded(m.group(2), today)
        if inner is None:
            return None
        return Interval(inner.start, None)
    m = re.match(r"(before|prior\s+to|until|up\s+to|earlier\s+than)\s+(.*)", text)
    if m:
        inner = _bounded(m.group(2), today)
        if inner is None:
            return None
        return Interval(None, inner.start)

    # "between X and Y" — the window from the start of X to the end of Y.
    # **Y is read forward from X**: each endpoint of "between March and June"
    # resolves per the bare-month rule independently, and asked in May 2023 that
    # yields March 2023 and June *2022* — an interval that starts at its own end
    # bound (measured, 13 Aug 2026).  "Between" names one forward span in every
    # English reading, so when Y lands before X it is shifted forward a year
    # (twice at most); an expression still incoherent after that returns None
    # and is counted, never hulled into a window that covers neither endpoint.
    m = re.match(r"between\s+(.*?)\s+and\s+(.*)", text)
    if m:
        lo = _bounded(m.group(1), today)
        hi = _bounded(m.group(2), today)
        if lo is None or hi is None:
            return None
        for _ in range(2):
            if hi.start >= lo.start:
                break
            shifted = _shift_interval_years(hi, 1)
            if shifted is None:
                return None
            hi = shifted
        if hi.start < lo.start:
            return None
        return Interval(lo.start, max(lo.end, hi.end))

    return _bounded(text, today)
