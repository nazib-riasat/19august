"""LoCoMo question categories, and the structural wall around the adversarial set.

`GRAFT_PHASE11_BUILD.md` G3 and G4.  Two jobs, and the second one is the reason
this is a module rather than a dict at the top of a runner.

**The four-way split has to match the reference table's rows.**  Mem-T
(arXiv 2601.23014v2) Table 2 reports single-hop, multi-hop, temporal,
open-domain and overall.  A GRAFT row that grouped differently could not be read
against it.

**The adversarial set has to be structurally unable to enter that split.**  Not
"excluded by the caller" -- unable.  `DATASET_DECISION.md` §1.2 makes LoCoMo's
446 adversarial questions the project's *primary* abstention testbed, and
Mem-T §A.1, Mem0 and A-Mem all discard them.  So the adversarial subset is the
one place GRAFT reports something nobody else does, and the one place a
comparison against Table 2 would be flatly invalid.  Those two facts pull in
opposite directions on the same data, which is exactly when a convention that
lives only in a caller's discipline fails.

**The mapping is VERIFIED, as of 19 August 2026.**  The integer codes below were
the convention in common use across LoCoMo evaluation code and unverified against
the dataset file.  ``scripts/locomo_ingest.py probe`` then measured the real
corpus and found **446 questions at code 5** -- matching `DATASET_DECISION.md`
§1.2's independently recorded count exactly, which is what turns the mapping from
a convention into evidence.  The full measured distribution:

===== =============== =======
code  name            count
===== =============== =======
1     multi_hop           282
2     temporal            321
3     open_domain          96
4     single_hop          841
5     adversarial         446
===== =============== =======

:func:`verify_against_corpus` stays, and stays load-bearing: it re-checks on every
run, so a swapped corpus file or an edited mapping fails loudly instead of
silently mislabelling every question.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "LOCOMO_CATEGORIES",
    "ANSWERABLE_CATEGORIES",
    "ADVERSARIAL",
    "ADVERSARIAL_CODE",
    "EXPECTED_ADVERSARIAL",
    "EXPECTED_TOTAL",
    "CategoryError",
    "categorise",
    "four_way_split",
    "adversarial_subset",
    "verify_against_corpus",
]


class CategoryError(RuntimeError):
    """An unknown category code, or a mapping that failed its own check."""


#: LoCoMo's integer category codes.  **Verified against the dataset file on
#: 19 August 2026** -- see the module docstring's measured distribution.
LOCOMO_CATEGORIES: Mapping[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

#: The four the reference table has rows for, in its own column order.
ANSWERABLE_CATEGORIES: tuple[str, ...] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
)

ADVERSARIAL = "adversarial"
ADVERSARIAL_CODE = 5

#: `DATASET_DECISION.md` §1.2, recorded independently of the code mapping, which
#: is what makes it usable as a check *on* that mapping rather than a restatement
#: of it.
EXPECTED_ADVERSARIAL = 446
EXPECTED_TOTAL = 1986


def categorise(code: Any) -> str:
    """Integer category code → name, refusing anything unmapped.

    Refuses rather than bucketing to ``"other"``: an unmapped code means the
    dataset is not the one this mapping was written for, and a silent bucket
    would put those questions in a total that is then reported as LoCoMo's.
    """
    try:
        key = int(code)
    except (TypeError, ValueError):
        raise CategoryError(f"category code {code!r} is not an integer") from None
    if key not in LOCOMO_CATEGORIES:
        raise CategoryError(
            f"unknown LoCoMo category code {key}; known: "
            f"{sorted(LOCOMO_CATEGORIES)}. An unmapped code means this is not "
            "the corpus the mapping was written for -- verify before relaxing."
        )
    return LOCOMO_CATEGORIES[key]


def four_way_split(questions: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """The reference table's four rows, and **only** those four.

    Adversarial questions cannot appear in the result: the return dict is keyed
    by :data:`ANSWERABLE_CATEGORIES` and nothing else is ever inserted.  That is
    the structural half of G4 -- a caller cannot forget to exclude them, because
    this function has no branch that would include them.
    """
    out: dict[str, list[Any]] = {name: [] for name in ANSWERABLE_CATEGORIES}
    for q in questions:
        name = categorise(q.get("category"))
        if name == ADVERSARIAL:
            continue
        out[name].append(q)
    return out


def adversarial_subset(questions: Iterable[Mapping[str, Any]]) -> list[Any]:
    """The abstention testbed, reached by its own function.

    Separate entry point rather than a fifth key, so that a report which means to
    quote the four-way split against a published table cannot pick this up by
    iterating.
    """
    return [q for q in questions if categorise(q.get("category")) == ADVERSARIAL]


def verify_against_corpus(questions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the unverified pin against two independently recorded counts.

    Raises :class:`CategoryError` when the adversarial bucket is not
    :data:`EXPECTED_ADVERSARIAL`.  That is the load-bearing assertion: the count
    comes from `DATASET_DECISION.md` §1.2 and the mapping comes from elsewhere,
    so agreement between them is evidence and disagreement is a real defect.

    The total is reported but **not** enforced -- a run on a subset of
    conversations legitimately has fewer questions, and failing on that would
    make the check unusable exactly when the corpus is being scaled up.  When the
    total *does* match :data:`EXPECTED_TOTAL`, the adversarial check is on the
    full corpus and its verdict is conclusive.
    """
    counts: dict[str, int] = {name: 0 for name in LOCOMO_CATEGORIES.values()}
    total = 0
    for q in questions:
        counts[categorise(q.get("category"))] += 1
        total += 1

    full_corpus = total == EXPECTED_TOTAL
    if full_corpus and counts[ADVERSARIAL] != EXPECTED_ADVERSARIAL:
        raise CategoryError(
            f"category mapping failed its own check: {counts[ADVERSARIAL]} "
            f"adversarial questions on a full corpus of {total}, expected "
            f"{EXPECTED_ADVERSARIAL} (DATASET_DECISION.md §1.2). The integer "
            "codes in LOCOMO_CATEGORIES are pointing at the wrong buckets; fix "
            "the mapping, do not relax the count."
        )

    return {
        "counts": dict(sorted(counts.items())),
        "total": total,
        "full_corpus": full_corpus,
        "adversarial_check": (
            "conclusive -- adversarial count matches the independently recorded 446"
            if full_corpus
            else f"inconclusive -- {total} questions is a subset, not the full "
            f"{EXPECTED_TOTAL}, so the count cannot verify the mapping"
        ),
    }
