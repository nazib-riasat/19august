"""P5.6 — the support gate: eligible or quarantined (architecture fix F9).

**Deliberately thin.**  The gate is the last step of the write path and the
first thing `H` reads on the read path, and its whole value is that it is one
short, auditable predicate rather than a policy engine.  Under
``support_policy = strict`` (frozen since Phase 0) an assertion is **eligible**
iff:

1. ``entailed_by_span`` holds at ``tau_nli``, and
2. every one of its spans grounded.

Condition 2 is redundant by construction — the grounder drops an assertion whose
quotes do not all resolve (G9) — and it is checked anyway, for the same reason
``Assertion.eligibility`` defaults to ``quarantined``: every other decision on
this path fails closed, and a gate that trusts an upstream invariant is the one
step that would fail open if that invariant ever moved.

**The two layers this implements** (fix F9).  The *audit layer* stores
everything, which is what makes extraction quality measurable at all; the
*active layer* takes only eligible assertions, so the open path — extractor
invents a claim → NLI marks it unsupported → stored → committed → retrieved as
evidence — is closed at the one place it can be closed cheaply.

**The quarantine rate is a quality signal, never a knob.**  Architecture F9 says
it in those words, and the report prints the sentence beside the number.  A high
rate means extraction is producing claims their spans do not carry; the response
is a better extractor or a narrower schema, not a lower ``tau_nli``.

**On the cause breakdown.**  P5.6 asks for the rate "broken out by cause
(parse-repaired, rung-3-grounded, NLI-below-threshold, grounding failure)", and
those four are not the same kind of thing — which is worth saying rather than
flattening.  Exactly one of them, *NLI-below-threshold*, is a **gating cause**
under the strict policy.  *Parse-repaired* and *rung-3-grounded* are
**provenance attributes**, reported cross-tabulated against the verdict, which is
the more informative form: "of the quarantined assertions, how many came from a
repaired parse" is a testable hypothesis about where extraction error
concentrates.  *Grounding failure* is a **drop**, not a quarantine — such an
assertion is never stored, because ``Assertion`` refuses to exist without spans —
so it is counted in its own row of the report and never inside the quarantine
denominator.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from graft.config import Config
from graft.ingest.nli import apply_threshold
from graft.schemas import ELIGIBILITY

__all__ = ["ELIGIBLE", "QUARANTINED", "eligibility", "QuarantineTally"]

ELIGIBLE, QUARANTINED = ELIGIBILITY


def eligibility(
    *,
    entailed_score: float,
    all_spans_grounded: bool,
    cfg: Config,
) -> tuple[str, tuple[str, ...]]:
    """``(verdict, reasons)`` under the configured support policy.

    ``reasons`` is empty exactly when the verdict is ``eligible``, mirroring
    ``CheckResult``'s rule that a failing verdict must say why: an unexplained
    quarantine cannot be counted by category or debugged, and the by-cause
    breakdown is an exit criterion.

    An unknown policy **raises** rather than defaulting to strict.  Every other
    decision on this path fails closed, and ``validate_bands`` in Phase 2 was
    fixed for exactly this — a validator that fails open on an unknown scope is
    worse than no validator, because it reads as a check.
    """
    if cfg.support_policy != "strict":
        raise ValueError(
            f"unknown support_policy {cfg.support_policy!r}; the only policy this "
            "project defines is 'strict', and a looser one lets unsupported "
            "assertions into the active graph"
        )
    reasons: list[str] = []
    if not all_spans_grounded:
        reasons.append("grounding_incomplete")
    if not apply_threshold(entailed_score, cfg.tau_nli):
        reasons.append("nli_below_threshold")
    return (QUARANTINED, tuple(reasons)) if reasons else (ELIGIBLE, ())


class QuarantineTally:
    """Counts the verdicts and cross-tabulates them against provenance.

    Not a dataclass (Phase-0 criterion 12) and not a bare dict, because the
    report needs both the counts *and* the denominators they are rates over, and
    a dict of counts always ends up with the denominator computed twice, slightly
    differently, in two places.
    """

    __slots__ = (
        "eligible",
        "quarantined",
        "reasons",
        "by_repair",
        "repair_unknown",
        "by_rung",
        "dropped",
    )

    def __init__(self) -> None:
        self.eligible = 0
        self.quarantined = 0
        self.reasons: dict[str, int] = {}
        self.by_repair: dict[str, int] = {"eligible": 0, "quarantined": 0}
        #: Drafts whose extraction provenance is **unrecoverable** — rebuilt
        #: from the log by a resumed run.  Counted separately rather than
        #: folded into the clean side, because a cross-tab that silently
        #: absorbs unknowns reports measured zeros it did not measure.
        self.repair_unknown = 0
        self.by_rung: dict[str, dict[str, int]] = {}
        self.dropped = 0

    def add(
        self,
        verdict: str,
        reasons: Iterable[str],
        *,
        from_repair: bool | None,
        rung: str,
    ) -> None:
        if verdict == ELIGIBLE:
            self.eligible += 1
        else:
            self.quarantined += 1
        for reason in reasons:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
        if from_repair is None:
            self.repair_unknown += 1
        elif from_repair:
            self.by_repair[verdict] = self.by_repair.get(verdict, 0) + 1
        bucket = self.by_rung.setdefault(rung, {"eligible": 0, "quarantined": 0})
        bucket[verdict] = bucket.get(verdict, 0) + 1

    def drop(self, n: int = 1) -> None:
        """An assertion that never reached the gate because a quote failed to ground.

        Kept out of the quarantine denominator on purpose: it was never stored,
        so calling it "quarantined" would inflate a rate that is supposed to
        measure the *gate*, and would double-count a loss the grounding row
        already reports.
        """
        self.dropped += int(n)

    @property
    def total(self) -> int:
        return self.eligible + self.quarantined

    @property
    def rate(self) -> float:
        """Quarantined / stored.  ``nan`` on an empty run, never a flattering 0."""
        return float("nan") if not self.total else self.quarantined / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "stored": self.total,
            "eligible": self.eligible,
            "quarantined": self.quarantined,
            "quarantine_rate": self.rate,
            "gating_causes": dict(sorted(self.reasons.items())),
            "cross_tab_repaired_extraction": dict(sorted(self.by_repair.items())),
            "repair_provenance_unknown": self.repair_unknown,
            "cross_tab_worst_rung": {k: dict(sorted(v.items())) for k, v in sorted(self.by_rung.items())},
            "dropped_before_gate_grounding_failure": self.dropped,
            "reading": (
                "a high quarantine rate is an extraction-quality signal, never "
                "grounds for quietly lowering tau_nli (architecture fix F9)"
            ),
        }


def summarise(verdicts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Tally a sequence of already-computed verdict records.  Reporting only."""
    tally = QuarantineTally()
    for row in verdicts:
        raw = row.get("from_repair", False)
        tally.add(
            row["eligibility"],
            row.get("reasons", ()),
            from_repair=None if raw is None else bool(raw),
            rung=str(row.get("worst_rung", "exact")),
        )
    return tally.to_dict()
