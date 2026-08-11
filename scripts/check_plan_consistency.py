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
    "PHASE0_DECISIONS.md",
    "PHASE1_DECISIONS.md",
    "PHASE2_DECISIONS.md",
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


def _check_numbering(name: str, text: str) -> list[str]:
    """Numbered lists must run 1..n with no gaps or repeats.

    Renumbering by hand after an insertion produced a duplicate or a hole in
    three separate rounds. This is cheaper than remembering.
    """
    out: list[str] = []
    for label, pattern in (
        ("decisions table", r"^\| (\d+) \|"),
        ("numbered list", r"^(\d+)\. "),
    ):
        nums = [int(m) for m in re.findall(pattern, text, re.M)]
        seen: list[int] = []
        for value in nums:
            if value == 1:
                if len(seen) > 3 and seen != list(range(1, len(seen) + 1)):
                    out.append(f"{name}: {label} runs {seen} — expected 1..{len(seen)}")
                seen = [1]
            else:
                seen.append(value)
        if len(seen) > 3 and seen != list(range(1, len(seen) + 1)):
            out.append(f"{name}: {label} runs {seen[:6]}… — expected 1..{len(seen)}")
    return out


def main() -> int:
    failures: list[str] = []

    for name in LIVE_DOCS:
        path = REPO / name
        if not path.is_file():
            failures.append(f"{name}: listed as live but missing")
            continue
        text = path.read_text(encoding="utf-8")

        for retired, replacement, round_, scope in RETIRED:
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
          f"{len(RETIRED)} retired wordings, {len(INCOMPATIBLE)} incompatible pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
