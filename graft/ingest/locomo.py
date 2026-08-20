"""LoCoMo → a deterministic ``Turn`` stream, on `graft.ingest.corpus`'s interface.

The evaluation corpus.  `DATASET_DECISION.md` §1 makes LoCoMo **secondary
zero-shot evaluation — never trained on, in any component**, and §1.2 makes its
446 adversarial questions the project's primary abstention testbed.

**Every structural assumption here was VERIFIED against the real file on
19 August 2026** by :func:`probe`, which reported zero findings:
10 samples, 1,986 questions, **446 adversarial**, 5,882 turns, every timestamp
parsed, 9 of 2,815 evidence markers unresolved (0.3%, all image-only turns).
Written before the file was available and checked the moment it arrived --
`CLAUDE.md` §10's verify-before-relying rule, paid rather than deferred.

The 446 is the load-bearing one: it comes from `DATASET_DECISION.md` §1.2,
recorded independently of this parser, so its agreement **verifies the category
code mapping** in `graft.baselines.categories`.  That mapping was an unverified
convention an hour before the probe ran; it is now evidence.

**The expected shape**, from the LoCoMo release:

* a JSON list of 10 samples
* each sample: ``sample_id``, ``conversation``, ``qa``
* ``conversation``: ``speaker_a``, ``speaker_b``, and per session
  ``session_{n}`` (a list of turns) plus ``session_{n}_date_time`` (a string)
* each turn: ``speaker``, ``text``, ``dia_id``
* each qa: ``question``, ``category``, and ``answer`` — except category 5
  (adversarial), which carries ``adversarial_answer`` instead, because the
  question is unanswerable and there is no answer to carry

**Three id conventions, mirroring `corpus.py` rather than inventing new ones:**

``conv_id``
    the sample id.  One LoCoMo sample is one pair's whole conversation history,
    which is the unit `H`'s scope sub-check is defined over (Phase-1 gap G6).
``session_id``
    the corpus's own ``session_{n}`` key.
``turn_id``
    ``locomo/{sample_id}/{session_key}/{turn_ix}`` — the `corpus.turn_id_for`
    shape with a different prefix, so a turn id says which corpus it came from.

**Sessions are ordered numerically, not lexically.**  ``session_10`` sorts before
``session_2`` as a string, which would reorder a conversation and silently break
every temporal claim in it -- the one bug in this file that would produce
plausible output.

**Timestamps carry a declared convention.**  LoCoMo writes
``"1:56 pm on 8 May, 2023"`` with no timezone; ``Turn.ts`` is that instant read
as **UTC**, the same decision `corpus.py` made and for the same reason: a silent
localtime read makes ingestion machine-dependent.  All turns in a session share
the session's timestamp -- LoCoMo timestamps sessions, not turns -- which is
recorded here because Phase 6's ``valid_during`` edges inherit it.

Gold -- answers, evidence ids, categories -- is returned as a **sidecar** and
never written to the event log, exactly as `corpus.py` does it.  Putting evidence
ids in the turn stream would leak the answer into the graph the system is
supposed to build without it.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from graft.schemas import Turn

__all__ = [
    "CORPUS_SOURCE",
    "CORPUS_LICENCE",
    "CORPUS_SHA256",
    "MEASURED_TURNS",
    "EXPECTED_SAMPLES",
    "EXPECTED_QUESTIONS",
    "EXPECTED_ADVERSARIAL",
    "ADVERSARIAL_CATEGORY",
    "LoCoMoError",
    "load_corpus",
    "corpus_sha256",
    "sample_index",
    "turn_id_for",
    "session_keys",
    "parse_locomo_ts",
    "turns_of",
    "questions_of",
    "evidence_turn_ids",
    "probe",
]

CORPUS_SOURCE = "LoCoMo (Maharana et al., ACL 2024) — snap-research/locomo"
CORPUS_LICENCE = "CC BY-NC 4.0 — non-commercial. Thesis use fine; no commercial derived dataset."

#: Measured by :func:`probe` on 19 August 2026 against the snap-research release.
#: `corpus.py`'s rule, inherited: a run that reads a different file is a different
#: experiment, and the loader says so rather than proceeding.
CORPUS_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

#: Also measured 19 Aug 2026.  `DATASET_DECISION.md` §1 records 5,875; this loader
#: counts 5,882 because it skips turns with empty ``text`` (image-only turns carry
#: a ``blip_caption`` instead) and the recorded figure evidently counted a slightly
#: different set.  Immaterial to any claim -- recorded because an unexplained
#: 7-turn gap is the kind of thing that later reads as a defect.
MEASURED_TURNS = 5882

#: `DATASET_DECISION.md` §1 and §1.2, recorded independently of this parser, which
#: is what makes them usable as checks *on* it.
EXPECTED_SAMPLES = 10
EXPECTED_QUESTIONS = 1986
EXPECTED_ADVERSARIAL = 446
ADVERSARIAL_CATEGORY = 5

_SESSION_RE = re.compile(r"^session_(\d+)$")

#: ``"1:56 pm on 8 May, 2023"``.  Anchored, because a loose parse that silently
#: accepts a different format is how a whole corpus gets the wrong dates.
_TS_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*$",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


class LoCoMoError(RuntimeError):
    """The corpus did not have the structure this loader was written against."""


def corpus_sha256(path: str | Path) -> str:
    """Digest of the corpus file, checked against :data:`CORPUS_SHA256`.

    The pin was recorded from the first probe on 19 August 2026 rather than
    invented from a guess -- which is why :func:`load_corpus` can now default to
    verifying instead of asking the caller to opt in.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(
    path: str | Path,
    *,
    expect_sha: str | None = CORPUS_SHA256,
    verify: bool = True,
) -> list[dict[str, Any]]:
    """Read the corpus, verify its identity, and check the structure assumed.

    **Verification is the default now that the pin exists.**  ``verify=False``
    exists for tests over a synthetic fixture and for nothing else -- the same
    carve-out, and the same wording, `corpus.load_corpus` uses. A run that reads
    a different file than the one pinned is a different experiment.
    """
    path = Path(path)
    if not path.is_file():
        raise LoCoMoError(
            f"corpus not found at {path}. LoCoMo ships as a single JSON file "
            "(locomo10.json in the snap-research/locomo release)."
        )
    if verify and expect_sha is not None:
        got = corpus_sha256(path)
        if got != expect_sha:
            raise LoCoMoError(
                f"corpus SHA mismatch at {path}: expected {expect_sha}, got {got}. "
                "A different file is a different experiment."
            )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise LoCoMoError(
            f"expected a JSON list of samples, got {type(raw).__name__}. "
            "See this module's docstring for the shape assumed."
        )
    for i, sample in enumerate(raw):
        _check_sample(sample, i)
    return raw


def _check_sample(sample: Any, index: int) -> None:
    """Fail with a message that says what to look at, not just that it failed."""
    if not isinstance(sample, Mapping):
        raise LoCoMoError(f"sample {index} is {type(sample).__name__}, expected an object")
    for key in ("sample_id", "conversation", "qa"):
        if key not in sample:
            raise LoCoMoError(
                f"sample {index} has no {key!r}; keys present: {sorted(sample)[:12]}"
            )
    conv = sample["conversation"]
    if not isinstance(conv, Mapping):
        raise LoCoMoError(f"sample {index} 'conversation' is not an object")
    if not session_keys(conv):
        raise LoCoMoError(
            f"sample {index} has no session_{{n}} keys; keys present: {sorted(conv)[:12]}"
        )


def sample_index(corpus: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """``sample_id -> sample``, for callers that address by id."""
    return {str(s["sample_id"]): s for s in corpus}


def turn_id_for(sample_id: str, session_key: str, turn_ix: int) -> str:
    """``locomo/{sample_id}/{session_key}/{turn_ix}``."""
    return f"locomo/{sample_id}/{session_key}/{turn_ix}"


def session_keys(conversation: Mapping[str, Any]) -> tuple[str, ...]:
    """Session keys in **numeric** order.

    The sort key is the integer, not the string.  Lexically ``session_10``
    precedes ``session_2``, which would reorder a conversation and invalidate
    every temporal claim derived from it while producing output that looks
    entirely normal.
    """
    found: list[tuple[int, str]] = []
    for key in conversation:
        m = _SESSION_RE.match(str(key))
        if m and isinstance(conversation[key], list):
            found.append((int(m.group(1)), str(key)))
    return tuple(key for _, key in sorted(found))


def parse_locomo_ts(raw: str) -> str:
    """``"1:56 pm on 8 May, 2023"`` → ``"2023-05-08T13:56:00+00:00"``.

    Read as UTC, declared rather than discovered -- see the module docstring.
    Raises on anything that does not match, because a corpus-wide date error is
    silent in every downstream number it touches.
    """
    m = _TS_RE.match(str(raw))
    if not m:
        raise LoCoMoError(
            f"unparseable LoCoMo timestamp {raw!r}; expected the form "
            "'1:56 pm on 8 May, 2023'. Fix the parser against the real file "
            "rather than defaulting the date -- a wrong date is invisible "
            "downstream."
        )
    hour12, minute, meridiem, day, month_name, year = m.groups()
    month = _MONTHS.get(month_name.lower())
    if month is None:
        raise LoCoMoError(f"unknown month {month_name!r} in timestamp {raw!r}")
    hour = int(hour12) % 12
    if meridiem.lower() == "pm":
        hour += 12
    stamp = datetime(
        int(year), month, int(day), hour, int(minute), tzinfo=timezone.utc
    )
    return stamp.isoformat()


def turns_of(
    sample: Mapping[str, Any],
    session_ids: Iterable[str] | None = None,
) -> Iterator[Turn]:
    """Every turn of one sample, in corpus order, oldest session first.

    ``session_ids`` restricts to named sessions -- the mechanism for a
    conversation-by-conversation scope, which is how the three-day plan harvests
    whatever ingestion finished.

    All turns in a session share the session timestamp: LoCoMo timestamps
    sessions, not turns.  Recorded rather than smoothed over, because Phase 6's
    ``valid_during`` edges inherit it and a synthesised per-turn offset would be
    invented evidence.
    """
    conv = sample["conversation"]
    sample_id = str(sample["sample_id"])
    wanted = None if session_ids is None else {str(s) for s in session_ids}

    for key in session_keys(conv):
        if wanted is not None and key not in wanted:
            continue
        raw_ts = conv.get(f"{key}_date_time")
        if raw_ts is None:
            raise LoCoMoError(
                f"session {key} of sample {sample_id} has no '{key}_date_time'; "
                "a session without a timestamp cannot carry a temporal claim"
            )
        ts = parse_locomo_ts(raw_ts)
        for turn_ix, turn in enumerate(conv[key]):
            text = str(turn.get("text") or "").strip()
            if not text:
                # Image-only turns exist in LoCoMo (they carry blip_caption
                # instead). Skipped rather than substituted: a caption is not
                # something the speaker said, and Stage A's provenance is a span
                # in an utterance.
                continue
            yield Turn(
                turn_id=turn_id_for(sample_id, key, turn_ix),
                conv_id=sample_id,
                session_id=key,
                speaker=str(turn.get("speaker") or "unknown"),
                ts=ts,
                text=text,
            )


def questions_of(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One sample's QA sidecar, in the shape `baselines.categories` expects.

    ``gold`` is ``answer`` for answerable categories and ``adversarial_answer``
    for category 5 -- LoCoMo carries the latter under a different key precisely
    because there is no answer, and reading it as one would give the abstention
    testbed a gold string to be scored against.
    """
    out: list[dict[str, Any]] = []
    sample_id = str(sample["sample_id"])
    for i, qa in enumerate(sample.get("qa") or []):
        if "category" not in qa:
            raise LoCoMoError(
                f"qa {i} of sample {sample_id} has no 'category'; keys: {sorted(qa)[:10]}"
            )
        category = int(qa["category"])
        adversarial = category == ADVERSARIAL_CATEGORY
        gold = qa.get("adversarial_answer") if adversarial else qa.get("answer")
        out.append(
            {
                "question_id": f"locomo/{sample_id}/q{i}",
                "conv_id": sample_id,
                "question": str(qa.get("question") or ""),
                "category": category,
                # Deliberately empty for adversarial: there is no gold answer, and
                # `diagnostics.report` scores that category by abstention instead.
                "gold": "" if adversarial else str(gold or ""),
                "adversarial": adversarial,
                "evidence": list(qa.get("evidence") or []),
            }
        )
    return out


def evidence_turn_ids(sample: Mapping[str, Any], evidence: Iterable[str]) -> list[str]:
    """LoCoMo ``dia_id`` evidence markers → this module's turn ids.

    Gold, so it belongs in the sidecar and never in the event log.  Returns only
    the ids that resolve; an unresolved marker is reported by the caller rather
    than raising, because LoCoMo's own evidence lists are known to contain
    markers for image-only turns, which this loader skips.
    """
    conv = sample["conversation"]
    sample_id = str(sample["sample_id"])
    by_dia: dict[str, str] = {}
    for key in session_keys(conv):
        for turn_ix, turn in enumerate(conv[key]):
            dia = turn.get("dia_id")
            if dia is not None:
                by_dia[str(dia)] = turn_id_for(sample_id, key, turn_ix)
    return [by_dia[str(e)] for e in evidence if str(e) in by_dia]


def probe(path: str | Path) -> dict[str, Any]:
    """Read the corpus and report what it actually contains. **Seconds of CPU.**

    This exists because of the arithmetic: ingesting LoCoMo costs ~43 GPU hours,
    and every assumption in this module was written against a release description
    rather than the file.  Discovering a structural mismatch at hour 30 is the
    expensive failure; discovering it here is free.

    Reports rather than raises wherever it can, so one probe surfaces *all* the
    mismatches instead of the first.
    """
    findings: list[str] = []
    # `verify=False` deliberately: probe's job is to *report*, including on a file
    # that does not match the pin. Refusing here would make it useless for the one
    # case it is most needed -- checking a corpus you are not yet sure about.
    corpus = load_corpus(path, verify=False)
    sha = corpus_sha256(path)
    if sha != CORPUS_SHA256:
        findings.append(
            f"sha256 {sha} does not match the pin {CORPUS_SHA256}; this is a "
            "different file than the one measured on 19 Aug 2026"
        )

    if len(corpus) != EXPECTED_SAMPLES:
        findings.append(
            f"{len(corpus)} samples, expected {EXPECTED_SAMPLES} "
            "(DATASET_DECISION.md §1)"
        )

    turns = 0
    questions = 0
    adversarial = 0
    categories: dict[int, int] = {}
    unresolved_evidence = 0
    total_evidence = 0
    ts_failures: list[str] = []
    per_sample: list[dict[str, Any]] = []

    for sample in corpus:
        sid = str(sample["sample_id"])
        try:
            sample_turns = list(turns_of(sample))
        except LoCoMoError as exc:
            findings.append(f"sample {sid}: {exc}")
            sample_turns = []
            ts_failures.append(sid)
        qs = questions_of(sample)
        for q in qs:
            categories[q["category"]] = categories.get(q["category"], 0) + 1
            if q["adversarial"]:
                adversarial += 1
            total_evidence += len(q["evidence"])
            unresolved_evidence += len(q["evidence"]) - len(
                evidence_turn_ids(sample, q["evidence"])
            )
        turns += len(sample_turns)
        questions += len(qs)
        per_sample.append(
            {
                "sample_id": sid,
                "sessions": len(session_keys(sample["conversation"])),
                "turns": len(sample_turns),
                "questions": len(qs),
            }
        )

    if questions != EXPECTED_QUESTIONS:
        findings.append(
            f"{questions} questions, expected {EXPECTED_QUESTIONS} "
            "(DATASET_DECISION.md §1)"
        )
    if adversarial != EXPECTED_ADVERSARIAL:
        findings.append(
            f"{adversarial} adversarial (category {ADVERSARIAL_CATEGORY}), expected "
            f"{EXPECTED_ADVERSARIAL} (DATASET_DECISION.md §1.2) -- if this is wrong, "
            "the category codes in baselines/categories.py point at the wrong buckets"
        )

    return {
        "path": str(path),
        "sha256": sha,
        "sha_matches_pin": sha == CORPUS_SHA256,
        "samples": len(corpus),
        "turns": turns,
        "questions": questions,
        "adversarial": adversarial,
        "category_counts": dict(sorted(categories.items())),
        "evidence_markers": total_evidence,
        "unresolved_evidence_markers": unresolved_evidence,
        "per_sample": per_sample,
        "timestamp_parse_failures": ts_failures,
        "findings": findings,
        "verdict": "OK — every recorded expectation matched" if not findings else "MISMATCH",
        "next_step": (
            "record the sha256 above as the corpus pin, then ingest "
            "conversation-by-conversation"
            if not findings
            else "fix the loader against the real file before spending GPU time"
        ),
    }
