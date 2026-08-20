"""Published LoCoMo numbers, as data — and the guards that stop them being misread.

`GRAFT_PHASE11_BUILD.md` G1, G5, G6, G8.  **No baseline runs here.**  The three
adapters the architecture names for Phase 11 are deferred by name (§7 of that
plan); this module is the reference table they would otherwise have produced,
stored with enough provenance that a reader can see exactly what it is not.

**What this is not: Gate 4.**  Plan §7's Gate 4 item 4 is "re-run system
baselines rather than quoting published numbers".  This module quotes them.
:data:`GATE4_STATUS` says so in the artefact, because a comparison that reads as
controlled and is not is `CLAUDE.md` §5's overreach pattern committed in the
results section.

**Three things a reader of a results table cannot see, so they are enforced:**

* *Different backbone.*  The paper's own GAM row loses **23.31 F1** on nothing
  but a backbone change from gpt-4o-mini to Qwen3-4B.  Backbone sensitivity on
  this benchmark is larger than the spread between the best and worst memory
  system in the table, so a GRAFT-vs-Mem-T gap of a few points means nothing on
  its own.
* *Different LoCoMo exposure.*  Mem-T trains on LoCoMo (§4.1, a 1:1:8
  train/validation/test split following Memory-R1, ACL 2026).  GRAFT never
  does.  The paper prices the difference itself: **49.38 → 58.65 F1**, +9.27
  bought by training on the benchmark.  :data:`COMPARABLE_ROW` is therefore the
  *untrained* row, pinned here before GRAFT has a number, because choosing it
  afterwards is the `GRAFT_PHASE2_BUILD.md` §6b failure.
* *Different metric conventions in circulation.*  F1/BLEU-1 here; LLM-judge for
  HyperMem (92.73) and Mem0 (72.90).  Rows are stored with their metric and
  :func:`comparable_rows` refuses any whose metric is not the pinned one.

**Venue tiering.**  arXiv 2601.23014v2 is a preprint — **Provisional** under
`CLAUDE.md` §3.  It may frame a comparison; it may not carry a claim alone.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "PINNED_METRIC",
    "GATE4_STATUS",
    "NON_COMPARABILITY",
    "COMPARABLE_ROW",
    "MEM_T_TABLE2",
    "ROW_FIELDS",
    "ReferenceError",
    "comparable_rows",
    "reference_block",
    "reference_sha",
]


class ReferenceError(RuntimeError):
    """A reference row was used outside the conditions it is valid under."""


#: The one metric convention this phase reports in.  G8.
PINNED_METRIC = "token_f1+bleu1"

GATE4_STATUS = (
    "NOT MET. Plan §7 Gate 4 item 4 requires re-running system baselines rather "
    "than quoting published numbers; this phase quotes them, at the project "
    "owner's instruction, on a three-day deadline. Meeting it costs ~8-15 h GPU "
    "for the full-context and matched-budget-RAG adapters (GRAFT_PHASE11_BUILD.md §7)."
)


#: Every field a stored row carries.  Rows are plain dicts, not a dataclass:
#: `test_structure.py`'s cross-module dataclass rule exempts only the six ML
#: packages, and widening that exemption for a data table would trade a real
#: guard for a convenience.  They are serialised and never mutated, so a dict is
#: also the shape they end up in anyway.
ROW_FIELDS: tuple[str, ...] = (
    "method",
    "backbone",
    "f1",
    "bleu1",
    "overall_f1",
    "overall_bleu1",
    "source",
    "arxiv",
    "table",
    "metric",
    # "in_domain" when the system trained on LoCoMo, "zero_shot" when it did not.
    "locomo_exposure",
    "embedder",
    # Mem-T §A.1: adversarial questions are discarded, as in Mem0 and A-Mem.
    "question_subset",
    "notes",
)


_SRC = "Mem-T: Densifying Rewards for Long-Horizon Memory Agents (Yue et al., 2026)"
_ARXIV = "arXiv:2601.23014v2"
_TBL = "Table 2"


def _row(
    method: str,
    backbone: str,
    single: tuple[float, float],
    multi: tuple[float, float],
    temporal: tuple[float, float],
    open_domain: tuple[float, float],
    overall: tuple[float, float],
    *,
    exposure: str = "in_domain",
    notes: str = "",
) -> dict[str, Any]:
    row = {
        "method": method,
        "backbone": backbone,
        "f1": {
            "single_hop": single[0],
            "multi_hop": multi[0],
            "temporal": temporal[0],
            "open_domain": open_domain[0],
        },
        "bleu1": {
            "single_hop": single[1],
            "multi_hop": multi[1],
            "temporal": temporal[1],
            "open_domain": open_domain[1],
        },
        "overall_f1": overall[0],
        "overall_bleu1": overall[1],
        "source": _SRC,
        "arxiv": _ARXIV,
        "table": _TBL,
        "metric": PINNED_METRIC,
        "locomo_exposure": exposure,
        "embedder": "BGE-M3",
        "question_subset": "answerable_only_adversarial_discarded",
        "notes": notes,
    }
    missing = set(ROW_FIELDS) - set(row)
    if missing:
        raise ReferenceError(f"reference row {method!r} is missing {sorted(missing)}")
    return row


#: Mem-T Table 2, Qwen3-4B rows.  Training-free systems are marked ``zero_shot``
#: on LoCoMo because they do no training at all; the trained Mem-T variants are
#: ``in_domain`` because §4.1 splits LoCoMo 1:1:8 and trains on it.
MEM_T_TABLE2: tuple[dict[str, Any], ...] = (
    _row("VANILLA", "Qwen3-4B", (40.68, 31.54), (23.23, 16.76), (18.97, 13.42),
         (13.87, 10.70), (31.50, 23.94), exposure="zero_shot"),
    _row("RAG", "Qwen3-4B", (49.45, 44.94), (23.50, 17.13), (43.07, 37.35),
         (20.23, 14.94), (41.59, 36.45), exposure="zero_shot",
         notes="beats four of the eight training-free memory systems; not a strawman"),
    _row("MemGPT", "Qwen3-4B", (14.00, 11.77), (16.68, 13.99), (12.56, 10.94),
         (11.61, 9.16), (14.05, 11.84), exposure="zero_shot"),
    _row("MemoryBank", "Qwen3-4B", (26.65, 17.72), (25.52, 19.44), (9.15, 7.44),
         (16.42, 12.39), (22.34, 15.66), exposure="zero_shot"),
    _row("Mem0", "Qwen3-4B", (47.28, 40.72), (35.40, 27.36), (46.84, 39.48),
         (26.64, 21.04), (43.71, 36.78), exposure="zero_shot"),
    _row("MemoryOS", "Qwen3-4B", (48.35, 42.57), (35.24, 27.30), (40.98, 32.68),
         (22.08, 17.93), (42.83, 36.26), exposure="zero_shot"),
    _row("LightMem", "Qwen3-4B", (43.78, 38.84), (30.78, 25.80), (44.71, 40.72),
         (18.93, 14.42), (40.01, 35.27), exposure="zero_shot"),
    _row("A-Mem", "Qwen3-4B", (44.62, 38.26), (27.24, 21.07), (43.85, 35.97),
         (15.40, 12.71), (39.43, 33.04), exposure="zero_shot"),
    _row("GAM", "Qwen3-4B", (32.23, 25.54), (32.23, 28.66), (26.26, 22.52),
         (18.45, 14.47), (30.17, 24.81), exposure="zero_shot",
         notes="loses 23.31 overall F1 against its own gpt-4o-mini run (53.48) "
               "on nothing but a backbone change -- the paper's own evidence that "
               "cross-backbone comparison on this benchmark is unsafe"),
    _row("Mem-T (w/o training)", "Qwen3-4B", (53.97, 49.15), (38.44, 31.70),
         (53.99, 48.08), (26.44, 23.37), (49.38, 44.11), exposure="zero_shot",
         notes="THE COMPARABLE ROW -- see COMPARABLE_ROW"),
    _row("Mem-T (GRPO)", "Qwen3-4B", (59.43, 54.65), (38.40, 30.51),
         (60.78, 56.10), (23.46, 20.16), (53.56, 48.33)),
    _row("Mem-T (MoT-GRPO)", "Qwen3-4B", (63.75, 57.95), (45.09, 36.58),
         (65.13, 60.12), (32.97, 28.94), (58.65, 52.63),
         notes="the paper's headline; in-domain, so not comparable to a zero-shot system"),
)

#: G5, pinned before GRAFT has a number.
COMPARABLE_ROW = "Mem-T (w/o training)"

NON_COMPARABILITY: Mapping[str, Any] = {
    "gate4": GATE4_STATUS,
    "baselines_rerun": False,
    "declared_differences": {
        "reader_backbone": {
            "graft": "Qwen2.5-3B-Instruct (bf16, pinned by model_id and revision)",
            "reference": "Qwen3-4B",
            "why_it_matters": (
                "the reference table's own GAM row moves 23.31 overall F1 on a "
                "backbone change alone, which is wider than the spread between "
                "its best and worst memory system"
            ),
        },
        "embedder": {"graft": "bge-small-en-v1.5 (pinned by revision)", "reference": "BGE-M3"},
        "locomo_exposure": {
            "graft": "zero-shot; Stage D trains on 2Wiki + MuSiQue-Ans, Stage B on "
                     "LongMemEval, the gate on MuSiQue-Full, the reader on nothing",
            "reference": "in-domain for the trained rows -- LoCoMo split 1:1:8 (§4.1)",
            "measured_effect": "+9.27 overall F1, 49.38 -> 58.65, the paper's own numbers",
        },
        "question_subset": {
            "graft": "all categories; the 446 adversarial reported separately",
            "reference": "adversarial discarded (§A.1), as in Mem0 and A-Mem",
        },
    },
    "venue_tier": "Provisional (arXiv preprint) -- may frame a comparison, "
                  "may not carry a claim alone (CLAUDE.md §3)",
}


def comparable_rows(
    rows: tuple[dict[str, Any], ...] = MEM_T_TABLE2,
    *,
    exposure: str = "zero_shot",
) -> tuple[dict[str, Any], ...]:
    """Rows valid to place beside a zero-shot GRAFT number.

    Filters on metric convention *and* LoCoMo exposure.  Both are refusals rather
    than warnings: a row on a different metric is a different axis (G8), and an
    in-domain row against a zero-shot system measures the benchmark exposure, not
    the systems.
    """
    out = tuple(
        r for r in rows
        if r["metric"] == PINNED_METRIC and r["locomo_exposure"] == exposure
    )
    if not out:
        raise ReferenceError(
            f"no reference row with metric {PINNED_METRIC!r} and exposure "
            f"{exposure!r}; refusing to emit a comparison against nothing"
        )
    return out


def reference_block() -> dict[str, Any]:
    """The whole reference side of the artefact, provenance included.

    Every consumer takes this or nothing: :mod:`graft.diagnostics.report` raises
    without it, so a comparison table cannot reach disk with the caveats stripped.
    """
    pinned = [r for r in MEM_T_TABLE2 if r["method"] == COMPARABLE_ROW]
    if len(pinned) != 1:
        raise ReferenceError(
            f"COMPARABLE_ROW {COMPARABLE_ROW!r} matches {len(pinned)} rows; the "
            "pin must name exactly one"
        )
    return {
        "pinned_metric": PINNED_METRIC,
        "comparable_row": dict(pinned[0]),
        "all_rows": [dict(r) for r in MEM_T_TABLE2],
        "non_comparability": dict(NON_COMPARABILITY),
        "source": _SRC,
        "arxiv": _ARXIV,
        "table": _TBL,
        "reference_sha": reference_sha(),
    }


def reference_sha() -> str:
    """Digest of the pinned row and the metric convention.

    Criterion 9 asks for the comparable row to be *SHA'd before any GRAFT number
    is computed*, and this is that SHA.  It covers the pin and the metric rather
    than the whole table, so adding a row for context does not invalidate the
    pin, while silently editing the row GRAFT is judged against does.

    The point is not tamper-proofing -- anyone can recompute it.  It is that a
    changed digest makes a changed baseline *visible* in a diff of the artefact,
    which is what stops `GRAFT_PHASE2_BUILD.md` §6b's failure happening by
    accident rather than by intent.
    """
    from graft.canonical import digest_of

    pinned = [r for r in MEM_T_TABLE2 if r["method"] == COMPARABLE_ROW]
    return digest_of({"metric": PINNED_METRIC, "comparable_row": pinned[0]})
