"""Fail when a superseded specification survives somewhere in the live plans.

**Why this exists.** Five review rounds on the Phase-2 plan found ~40 defects.
Seven of the eleven found in the fifth round were *introduced by the fourth
round's own fixes*: the documents restate each decision in four or five places —
the gap section, the module spec, the exit criterion, the decisions table, the
next phase's handoff — so every change is a four-or-five-place edit, and landing
three of them leaves a contradiction that reads as authoritative.

The code has needed one review round for the same volume of work, because tests
execute it. Nothing executes prose. This is the nearest available substitute: a
blocklist of retired wordings that must not reappear, and a set of pairs that
must never co-occur.

It does not check that the plans are *right*. It checks that they do not
contradict themselves, which is the failure mode that actually recurred.

Add an entry every time a review retires a specification, with the round that
retired it. Run it from the repo root:

    python scripts/check_plan_consistency.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Documents that are normative. Historical records are excluded on purpose:
#: `GRAFT2_9_*` and the crosscheck describe superseded designs and *should*
#: contain retired wording.
LIVE_DOCS = (
    "GRAFT_RESEARCH_PLAN_v1.md",
    "GRAFT_EXECUTION_ARCHITECTURE_v1.md",
    "GRAFT_PHASE0_BUILD.md",
    "GRAFT_PHASE1_BUILD.md",
    "GRAFT_PHASE2_BUILD.md",
    # Added 11 Aug 2026. The R12 retired wordings below name this file and had
    # never been checked against it, because it was not on this list — a guard
    # that lists rules for a document it does not read is the same defect class
    # the rules themselves exist to catch.
    "GRAFT_PHASE3_BUILD.md",
    "PHASE0_DECISIONS.md",
    "PHASE1_DECISIONS.md",
    "PHASE2_DECISIONS.md",
    "PHASE3_DECISIONS.md",
    "CLAUDE.md",
    "README.md",
)

#: (retired string, what replaced it, round that retired it, scope).
#: `scope` names the one document the retirement applies to, or None for all
#: live documents. Scoping matters: "Nine specification gaps" is wrong in the
#: Phase-2 plan and correct in the Phase-1 plan, which really does have nine.
#: The first run of this checker flagged exactly that, which is the argument for
#: the field existing.
#:
#: Strings match the *normative* statement, not prose explaining the change —
#: "an earlier draft said X" is how a retirement gets recorded and stays legal.
RETIRED: tuple[tuple[str, str, str, str | None], ...] = (
    ("p_fail_floor", "target_p_fail — 'floor' was rejected in round 3", "R4", None),
    ("TemperedOraclePolicy", "FlowOraclePolicy, built by flow decomposition", "R3", None),
    ("Sequence[frozenset[str]]", "np.ndarray of state indices into StateGraph", "R6", None),
    ("uniform over **selected atoms**", "uniform over **removable** atoms (fix F10)", "R5", None),
    ("| 4 | State identity | the exact `frozenset`", "uint64 bitmask over pool.ids()", "R6", None),
    ("partition check at ≤ 10⁻¹² absolute", "≤ 10⁻⁹, set by the worst-case n·eps bound", "R6", None),
    ("`A*` / `B*` chains, not by membership", "complete templates P_A / P_B", "R6", None),
    ("Nine specification gaps", "Ten specification gaps", "R2", "GRAFT_PHASE2_BUILD.md"),
    ("all five run", "every audit runs", "R4", "GRAFT_PHASE2_BUILD.md"),
    ("inside both bands of G1", "inside all three bands of G1", "R4", "GRAFT_PHASE2_BUILD.md"),
    ("Four audits, all", "Five audits, all", "R5", "GRAFT_PHASE2_BUILD.md"),
    ("declared band on two quantities", "declared band on three quantities", "R5", "GRAFT_PHASE2_BUILD.md"),
    ("only the second is always true", "a dead end licenses only 'no proof found under this pool/policy/budget'", "R7", None),
    ("no proof was constructible at all", "the non-existence reading is not licensed by a dead end", "R7", None),
    ("reached only on budget exhaustion", "reached when construction can neither continue nor stop", "R7", None),
    ("reached only by budget exhaustion", "reached when construction can neither continue nor stop", "R7", None),
    ("action_log_probs(states, graph)", "action_log_probs(state_ix: np.ndarray, graph)", "R7", None),
    ("with two reference policies", "three reference policies", "R7", "GRAFT_PHASE2_BUILD.md"),
    ("(G6) and two reference", "(G6) and three reference", "R7", "GRAFT_PHASE2_BUILD.md"),
    ("passes or fails on noise", "exact TV has no estimation noise; the risk is triviality and variance", "R7", "GRAFT_PHASE2_BUILD.md"),
    ("`fcs(policy, state_graph)`** ·", "fcs(p_theta, target, m, n_subsets, rng) — FCS needs the reward", "R8", None),
    ("Phase 9 has no exact TV; if FCS is", "no document specifies a Phase-9 distribution metric; do not invent the requirement", "R8", None),
    ("loses `w_suff · β = 4` in log-reward", "loses beta*w_suff*(1 - template overlap)", "R8", None),
    ("assertion are present and reachable", "present in the snapshot and asserted UNREACHABLE (masks prune them)", "R8", None),
    ("is reached by budget exhaustion, never by an action", "reached when construction can neither continue nor stop", "R8", None),
    ("Both suites are frozen at Gate 0", "all three suites (main, probe, tuning)", "R8", "GRAFT_PHASE2_BUILD.md"),
    ("both suites", "all three suites — main, probe and tuning", "R9", "GRAFT_PHASE2_BUILD.md"),
    ("before Phase 9 has to rely on it", "no document specifies a Phase-9 distribution metric", "R9", None),
    ("both content-hashed", "all three content-hashed", "R9", "GRAFT_PHASE2_BUILD.md"),
    ("FCS at `m = #terminals` reproduces exact TV", "verify FCS against exhaustive m-subset enumeration; m=#X tests no sampler", "R9", None),
    ("`at_beta` recomputes the target-mass bands", "at_beta checks r_fail_margin only; validate_bands('main') checks the bands", "R9", None),
    ("which checks `r_fail_margin` **and** the target-mass bands", "at_beta checks r_fail_margin only (decision 10)", "R10", None),
    ("re-checks `r_fail_margin` *and* the", "at_beta checks r_fail_margin only (decision 10)", "R10", None),
    ("rounded to 10 decimal places", "quantised to 1e-12; 1e-10 hides 1.25e-7 of TV", "R10", None),
    ("still awaiting sign-off is the", "delta-d 0.6 was signed off 9 Aug 2026; no open items in section 6", "R10", None),
    ("must converge to it", "state a tolerance: within 4 SE of the enumerated literal", "R10", None),
    # R11 — the beta lifecycle. The grid `{1, 2, 4, 8}` is unchanged and stays
    # legal; what was retired is selecting a winner *before* checking whether the
    # main suite can be scored at that beta. Measured after the Phase-2 build:
    # {1, 2} fail the neither-mass band on 20/20 main-suite instances, so the old
    # ordering could only ever discover that after a full seven-learner sweep,
    # by which point Section 6b forbids amending anything.
    ("predeclared; swept on a **separate tuning suite**",
     "eligibility precedes selection — ineligible candidates leave the argmin (decision 19)", "R11", None),
    ("the frozen β re-validated on the main suite per **decision 10**",
     "eligibility is checked before the sweep, not after the winner is picked", "R11", None),
    ("exact TV across the tuning suite**, for **L5",
     "argmin over the ELIGIBLE candidates of G9 item 3 (decision 22)", "R11", None),
    ("If the main suite fails its bands at the frozen β, that is a regeneration",
     "regeneration is the branch for *no* eligible candidate, not for a failed winner", "R11", None),
    ("Amend the candidate set to `{4, 8}` before any sweep runs",
     "the grid is unchanged; the eligibility rule derives the feasible set from decision 14's band", "R11", "PHASE2_DECISIONS.md"),
    # R12 — Phase-3 plan. Four review rounds; in three of them a correction
    # landed in the gap section and not in the exit criteria, the build order or
    # the risk table. These are the values that were left behind each time.
    ("True for L7, L7b and GAFlowNet",
     "delta_d is True for L7/L7b only; GAFlowNet's Δd reaches its loss, never its policy (decision 19a)", "R12", "GRAFT_PHASE3_BUILD.md"),
    ("by wall-clock only",
     "decision 4's bounded adaptive ladder, recorded as adaptive rather than throughput-only", "R12", "GRAFT_PHASE3_BUILD.md"),
    ("/ |log R(X(τ))|",
     "per-instance valid-terminal log R range, with a zero-range guard (decision 13)", "R12", "GRAFT_PHASE3_BUILD.md"),
    ("p95 is **not worse** than",
     "both pass decision 13's band, plus a non-inferiority margin (decision 15)", "R12", "GRAFT_PHASE3_BUILD.md"),
    ("6 h per (arm, seed)",
     "c0 = 1 h with rungs 1/2/4 h; the GPU-day cost is tabulated at decision 5", "R12", "GRAFT_PHASE3_BUILD.md"),
    ("reference implementation's defaults",
     "literal LED coefficients plus a pinned commit hash (decision 23a)", "R12", "GRAFT_PHASE3_BUILD.md"),
    ("8 × 3 × C × 20",
     "nine evaluated arms: 9 x 3 x 50 x 20 = 27,000 (decision 2)", "R12", "GRAFT_PHASE3_BUILD.md"),
    ("the seven architecture learners",
     "nine evaluated arms — L1-L7, L7b and GAFlowNet (decision 1)", "R12", "GRAFT_PHASE3_BUILD.md"),
    # R13 — an external code review, checked against the papers. Three of its
    # findings were defects in *controls*, all running in the direction that
    # flatters the proposed method. These are the specifications that permitted
    # them, and every one reads as reasonable until the equation is opened.
    ("Π_t ( 1 + r(s_t→s_{t+1}) / F(s_{t+1}) )",
     "GAFlowNet Eq. 4 puts the term beside P_B: Π (P_B + r/F), so the TB correction is log(1 + r/(F·P_B))", "R13", None),
    ("the **plain** form, *not* the correction-term variant",
     "the redistribution form of Appendix B.1, which is what LED-GFN denotes in the paper's experiments; LED-GFN* stays out (decision 23b)", "R13", None),
    ("trainable params within 1%, control never smaller",
     "live trainable params, control never smaller, narrowest admissible width — the 1% is unachievable by width (decision 11)", "R13", None),
    ("their capacity match is exact rather than within 1%",
     "the nominal match was exact and the live match was 1.46% short; matching is on live capacity (decision 11)", "R13", None),
    ("the match is 0.00% rather than",
     "a 0.00% nominal match concealed 768 dead parameters; decision 11 matches live capacity", "R13", None),
    ("so FCS is the binding term",
     "the per-evaluation cost is the DP; FCS runs once per (arm, seed) at final θ (decision 2)", "R13", None),
    ("50 checkpoints × 20 instances = **27,000**",
     "51 evaluations per run — the trajectory-0 baseline plus C = 50 — so 27,540 (decision 2)", "R13", None),
)

#: Pairs that must never both appear in one document — each is a contradiction
#: that a single-string blocklist cannot catch.
INCOMPATIBLE: tuple[tuple[str, str, str], ...] = (
    (
        "`uint32` whose bit",
        "pool_cap ≤ 64",
        "a uint32 holds 32 bits, so it cannot represent a 64-atom pool",
    ),
    (
        "p_θ(FAIL)` by complement.",
        "direct dead-end sum",
        "p_θ(FAIL) must have one authoritative definition",
    ),
    (
        "at_beta` **re-validates** `r_fail_margin` (G7)",
        "and on a β at which the main suite would fail its target-mass bands",
        "at_beta's contract must be stated identically in the table and the criteria",
    ),
)


def _tables(text: str) -> list[list[str]]:
    """Markdown tables, as blocks of consecutive pipe-prefixed lines."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _check_numbering(name: str, text: str) -> list[str]:
    """Numbered lists must run 1..n with no gaps or repeats.

    Renumbering by hand after an insertion produced a duplicate or a hole in
    three separate rounds. This is cheaper than remembering.

    **Two things this deliberately does not require, both learned from turning
    the check on against the Phase-3 plan** (11 Aug 2026), where it fired twice
    and both were the checker's fault rather than the document's:

    *Decision tables are checked as a **set**, not as a sequence.* Phase-3 §6
    ends `… 23, 26, 27, 25, 24, 28` because each review round appended its own,
    and the numbers are cited from code and tests ("decision 19a", "decision
    23b"). Sorting them would break every cross-reference to buy tidiness. What
    matters — and what a hand renumber actually broke three times — is that no
    number is missing and none is used twice.

    *Only a table whose header says "Decision" is one.* The old pattern matched
    any table with a bare integer in its first column, so it read the Phase-3
    build-order table (steps 1–10, one of them bolded and therefore invisible to
    it) and the ladder-cost table (rungs 0, 1, 2) as decision tables and reported
    both as corrupt.
    """
    out: list[str] = []

    for block in _tables(text):
        if len(block) < 3 or "decision" not in block[0].lower():
            continue
        nums = [int(m) for line in block for m in re.findall(r"^\| (\d+) \|", line)]
        if len(nums) < 4:
            continue
        if len(nums) != len(set(nums)):
            duplicated = sorted({v for v in nums if nums.count(v) > 1})
            out.append(f"{name}: decisions table repeats {duplicated}")
        missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
        if missing:
            out.append(
                f"{name}: decisions table is missing {missing} "
                f"(runs to {max(nums)})"
            )

    nums = [int(m) for m in re.findall(r"^(\d+)\. ", text, re.M)]
    seen: list[int] = []
    for value in nums:
        if value == 1:
            if len(seen) > 3 and seen != list(range(1, len(seen) + 1)):
                out.append(f"{name}: numbered list runs {seen} — expected 1..{len(seen)}")
            seen = [1]
        else:
            seen.append(value)
    if len(seen) > 3 and seen != list(range(1, len(seen) + 1)):
        out.append(f"{name}: numbered list runs {seen[:6]}… — expected 1..{len(seen)}")
    return out


def _live_sources() -> list[Path]:
    """Every `.py` under `graft/` and `scripts/`.

    **Added in R6, after a retirement landed in four documents and survived in
    three docstrings.** This project keeps most of its reasoning in module
    docstrings — `chatcontext1.md` says they "are not decoration and are the
    fastest way in" — so the medium the guard could not read held as much
    normative prose as the medium it could. One of the three survivors did not
    merely restate the retired claim, it asserted the disproved version as fact.

    Scoped retirements name a `.md` file, so they skip sources; unscoped ones
    (`scope=None`) are exactly the project-wide claims that should hold
    everywhere, and those are what this catches.
    """
    me = Path(__file__).resolve()
    return sorted(
        p
        for d in ("graft", "scripts")
        for p in (REPO / d).rglob("*.py")
        # This file *is* the blocklist: every retired string appears in it by
        # construction, so scanning it reports each retirement as its own
        # violation.
        if p.resolve() != me
    )


def main() -> int:
    failures: list[str] = []

    targets = [(name, REPO / name) for name in LIVE_DOCS]
    targets += [
        (str(p.relative_to(REPO)).replace("\\", "/"), p) for p in _live_sources()
    ]

    for name, path in targets:
        if not path.is_file():
            failures.append(f"{name}: listed as live but missing")
            continue
        text = path.read_text(encoding="utf-8")
        is_doc = name in LIVE_DOCS

        for retired, replacement, round_, scope in RETIRED:
            # A scoped retirement names one document; sources only ever carry
            # the unscoped, project-wide claims.
            if scope is not None and scope != name:
                continue
            if retired in text:
                line = next(
                    (i for i, l in enumerate(text.splitlines(), 1) if retired in l), 0
                )
                failures.append(
                    f"{name}:{line}: retired in {round_} — {retired!r}\n"
                    f"    superseded by: {replacement}"
                )

        if not is_doc:
            continue

        failures.extend(_check_numbering(name, text))

        for left, right, why in INCOMPATIBLE:
            if left in text and right in text:
                failures.append(f"{name}: {left!r} and {right!r} cannot coexist — {why}")

    if failures:
        print(f"{len(failures)} plan-consistency failure(s):\n", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nA retired wording reappeared, or two authoritative statements "
            "disagree.\nFix the document; do not relax this list.",
            file=sys.stderr,
        )
        return 1

    print(f"plan consistency OK — {len(LIVE_DOCS)} live documents, "
          f"{len(_live_sources())} sources, {len(RETIRED)} retired wordings, "
          f"{len(INCOMPATIBLE)} incompatible pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
