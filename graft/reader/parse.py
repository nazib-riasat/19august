"""P10.2 — the answer parser: generation → answer, citations, resolved spans.

**[EVIDENCE]** ALCE (EMNLP 2023) established that answer correctness and
citation correctness are **separate** measurements, and supplies this project's
motivating number: on ELI5, even the best models lacked complete citation
support **50% of the time**.  `CLAUDE.md` §9 calls that *"the motivating number
for the whole project"*.

**[EVIDENCE]** VeriCite (SIGIR-AP 2025) shows what drives the citation half:
removing its NLI verification dropped citation F1 **77.73 → 68.91** while answer
correctness barely moved (41.63 → 41.59).  Verification, not generation, is what
makes a citation trustworthy — which is why this project's citation is resolved
through the *same* provenance chain ``H``'s support sub-check validates, rather
than through anything the reader says about itself.

**An unresolvable citation raises rather than being dropped.**  A reader that
cites `[c9]` when only `[c1]`–`[c6]` were shown has hallucinated a citation, and
that is a **finding about the reader** — one of the five ceilings' whole purpose
is to localise failures like it.  Silently discarding it would inflate citation
precision by removing exactly the cases that should lower it, which is the
`CLAUDE.md` §5 pattern of an error that flatters the method.  The caller decides
what to do; this module refuses to hide it.

**Abstention is parsed, not inferred.**  ``pins.INSUFFICIENT`` is the string the
prompt asks for and the string this module matches — one constant, two
consumers.  A parser hunting for hedging ("I'm not sure…") would make the
abstention rate a function of the reader's politeness rather than of its
decision, and plan §4.2 needs abstention to be a decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from graft.reader.pins import ANSWER_EQUIVALENCE, INSUFFICIENT

__all__ = [
    "ParsedAnswer",
    "parse_answer",
    "resolve_citations",
    "normalise_answer",
    "answers_equivalent",
    "token_f1",
    "CitationError",
]

#: ``[c1]``, ``[c12]`` — the format `pins.CLAIM_ID_FORMAT` writes, read back.
#: Anchored to the bracket form so prose containing the letter c is not a
#: citation, and case-insensitive because a reader that shouts ``[C1]`` has still
#: cited c1.
_CITE = re.compile(r"\[(c\d+)\]", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_ARTICLES = frozenset(ANSWER_EQUIVALENCE["articles"])


class CitationError(ValueError):
    """A citation that names no shown claim.  Raised, never swallowed."""


def normalise_answer(text: str) -> str:
    """SQuAD-style normalisation — §6 decision 9, the one rule.

    **[EVIDENCE]** the normalisation SQuAD (EMNLP 2016) introduced and which
    HotpotQA, 2WikiMultiHopQA and MuSiQue all inherit, so a local score computed
    through it is commensurable with published numbers rather than being this
    project's private dialect.

    Order matters.  **Citation markers are stripped first** — found by run R2 on
    16 Aug 2026, and it would have corrupted every answer number in the phase.
    The reader answered ``"London\\n[c1]"`` against gold ``"London"``, which is
    *correct*, and ceiling 5 scored it **False**: SQuAD normalisation removes
    punctuation, so ``[c1]`` became the bare token ``c1`` and the normalised
    answer was ``"london c1"``.

    This module's own docstring had asserted the opposite — *"``normalise_answer``
    already discards brackets when scoring"* — which is true of the brackets and
    false of what they contain.  The error was systematic and one-directional: it
    could only ever mark a correct answer wrong, and it would have done so on
    every cited answer, i.e. on every answer the prompt asks for.

    **Punctuation is DELETED, not replaced by a space** — corrected 16 Aug 2026
    after an adversarial audit checked this against the official SQuAD evaluation
    script, which does ``"".join(ch for ch in text if ch not in exclude)``.
    Substituting a space splits one token into several, and the article filter
    then eats any single ``a`` the split produced.  Measured consequences:
    ``"U.S.A."`` normalised to ``"u s"`` instead of ``"usa"``, so
    ``answers_equivalent("U.S.A.", "USA")`` was **False** where SQuAD says True;
    ``"O'Brien"`` and ``"Wal-Mart"`` split likewise.

    That is the *same one-directional class* as the citation-marker defect
    recorded above — it can only mark a correct answer **wrong** — and it reached
    every consumer: ``answers_equivalent``, ``token_f1``, ceiling 5 and the
    orchestrator's contested check.  Citations are still replaced by a *space*,
    deliberately: ``[c1]`` is a delimiter between words, not intra-word
    punctuation, and deleting it would weld the tokens on either side together.

    Then the published order: casefold, strip punctuation, drop articles,
    collapse whitespace.  Dropping articles *before* stripping punctuation would
    leave ``"the-dog"`` intact.
    """
    lowered = _PUNCT.sub("", _CITE.sub(" ", text).casefold())
    words = [w for w in lowered.split() if w not in _ARTICLES]
    return _WS.sub(" ", " ".join(words)).strip()


def answers_equivalent(
    predicted: str, gold: str, aliases: Sequence[str] = ()
) -> bool:
    """Decision 9's equivalence, used by **both** consumers.

    The contested check (G8) and the local scorer (decision 10) call this same
    function, because two equivalence rules in one pipeline is how a system comes
    to disagree with itself about whether two answers are the same.
    """
    p = normalise_answer(predicted)
    if not p:
        return False
    candidates = [gold, *aliases]
    return any(p == normalise_answer(c) for c in candidates if c)


def token_f1(predicted: str, gold: str) -> float:
    """Token-overlap F1 under the same normalisation — the reproducible floor.

    Reported *beside* the benchmark's own metric, never instead of it
    (`pins.ANSWER_SCORING`): a local number is reproducible without an API and a
    published number is comparable, and the project needs both.
    """
    from collections import Counter

    p_tokens = normalise_answer(predicted).split()
    g_tokens = normalise_answer(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class ParsedAnswer:
    """What the reader said, split into the two things ALCE scores separately."""

    answer_text: str
    citations: tuple[str, ...]
    abstained: bool
    raw: str
    unresolved: tuple[str, ...] = ()
    spans: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "abstained": self.abstained,
            "citations": len(self.citations),
            "unresolved_citations": len(self.unresolved),
            "answer_chars": len(self.answer_text),
        }


def parse_answer(generation: str) -> ParsedAnswer:
    """Split a generation into answer text and cited claim ids.

    Citations are **kept in the answer text** on the record, and stripped by
    :func:`normalise_answer` at scoring time.  Keeping them on the record is what
    makes a generation auditable — the raw string is what the reader actually
    emitted — and stripping them at the metric is what keeps ALCE's two
    measurements separable rather than entangled.

    An earlier version of this docstring said normalisation "already discards
    brackets", which was true of the brackets and false of their contents; run R2
    measured the consequence and :func:`normalise_answer` now records it.
    """
    text = (generation or "").strip()
    cites = tuple(dict.fromkeys(m.group(1).lower() for m in _CITE.finditer(text)))
    stripped = _WS.sub(" ", _CITE.sub(" ", text)).strip()
    abstained = stripped.casefold().startswith(INSUFFICIENT.casefold())
    return ParsedAnswer(
        answer_text=text,
        citations=() if abstained else cites,
        abstained=abstained,
        raw=generation or "",
    )


def resolve_citations(
    parsed: ParsedAnswer,
    claim_map: Mapping[str, str],
    snapshot: Any = None,
    pool: Any = None,
    *,
    strict: bool = True,
) -> ParsedAnswer:
    """Resolve cited claim ids to atoms, and atoms to source spans.

    ``strict`` raises :class:`CitationError` on a citation naming no shown claim.
    That is the default because a hallucinated citation is a **reader ceiling**
    finding, and the five-ceiling protocol exists to attribute failures rather
    than absorb them.  ``strict=False`` records them in ``unresolved`` instead —
    for a batch run that must not die on one bad generation, and which then has
    the count to report.
    """
    unresolved = tuple(c for c in parsed.citations if c not in claim_map)
    if unresolved and strict:
        raise CitationError(
            f"the reader cited {list(unresolved)}, which name no shown claim "
            f"(shown: {sorted(claim_map)}). A citation to nothing is a reader "
            "finding, not a parse error to discard."
        )

    spans: dict[str, tuple[str, ...]] = {}
    if snapshot is not None and pool is not None:
        from graft.core import resolve as core_resolve

        for claim_id in parsed.citations:
            atom_id = claim_map.get(claim_id)
            if atom_id is None or atom_id not in pool:
                continue
            span_ids, _problems = core_resolve.provenance_spans(pool[atom_id], snapshot)
            spans[claim_id] = tuple(span_ids)

    return ParsedAnswer(
        answer_text=parsed.answer_text,
        citations=parsed.citations,
        abstained=parsed.abstained,
        raw=parsed.raw,
        unresolved=unresolved,
        spans=dict(sorted(spans.items())),
    )
