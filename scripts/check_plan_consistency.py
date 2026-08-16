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
    "GRAFT_PHASE2_5_BUILD.md",
    # Added 11 Aug 2026. The R12 retired wordings below name this file and had
    # never been checked against it, because it was not on this list — a guard
    # that lists rules for a document it does not read is the same defect class
    # the rules themselves exist to catch.
    "GRAFT_PHASE3_BUILD.md",
    "GRAFT_PHASE4_BUILD.md",
    "GRAFT_PHASE5_BUILD.md",
    # Added 14 Aug 2026 with the Phase-6 plan.
    "GRAFT_PHASE6_BUILD.md",
    # Added 14 Aug 2026 with the Phase-7 plan.
    "GRAFT_PHASE7_BUILD.md",
    # Added 15 Aug 2026 with the Phase-8 plan — same day it was written, because
    # a guard that does not read a live document cannot retire anything in it
    # (the GRAFT_PHASE3_BUILD.md lesson above, pre-empted this time).
    "GRAFT_PHASE8_BUILD.md",
    # Added 16 Aug 2026 by the Phase-9 audit. All three were live and none was
    # read — the same gap this list already records for GRAFT_PHASE3_BUILD.md,
    # reopened by three documents landing faster than the registry. The Phase-9
    # pair matters most: PHASE9_DECISIONS.md *wins conflicts* with its own build
    # plan, so an unretired wording there outranks the corrected one everywhere.
    "GRAFT_PHASE9_BUILD.md",
    "PHASE8_DECISIONS.md",
    "PHASE9_DECISIONS.md",
    "DATASET_DECISION.md",
    "PHASE0_DECISIONS.md",
    "PHASE1_DECISIONS.md",
    "PHASE2_DECISIONS.md",
    "PHASE2_5_DECISIONS.md",
    "PHASE3_DECISIONS.md",
    "PHASE4_DECISIONS.md",
    # Added 13 Aug 2026 with the Phase-5 build.  The Gate-0 contract restates
    # decisions from the plan, the architecture and three DECISIONS files, which
    # makes it precisely the kind of document this checker exists for: a
    # four-or-five-place edit whose fifth place is the one that gets missed.
    "GATE0_CONTRACT.md",
    "PHASE5_DECISIONS.md",
    "GRAFT_PHASE6_BUILD.md",
    "PHASE6_DECISIONS.md",
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
    # R14 — the Phase-4 paper audit. Every one of these was a compression of a
    # compression: the architecture summarised a paper, a later document
    # summarised the architecture, and the condition attached to the number was
    # dropped at one of the two hops. Checked against the PDFs, not the summary.
    ("PCST's connected output maps to a closed set with no conversion logic",
     "PCST needs a declared graph mapping; under atoms-as-nodes it needs explicit closure completion (Phase-4 G2)", "R14", None),
    ("guarantee from Nemhauser–Wolsey–Fisher 1978",
     "arXiv 2607.00725 section 4.2 declines the guarantee: no partial enumeration is performed", "R14", None),
    ("node prizes = relevance scores, uniform edge costs",
     "G-Retriever prizes nodes AND edges; C_e is tuned per dataset, so it is declared once and never swept", "R14", None),
    ("(0.451 F1 vs 0.429 for a tuned heuristic on multi-hop HotpotQA)",
     "one cell of Table 6 — the only significant budget of four, at 3B; the paper concludes parity", "R14", None),
    # R15 — the Phase-4 measurement audit. Every one of these was a rule written
    # from reasoning that a measurement then contradicted: greedy is globally
    # optimal, the pool is disconnected, closure only adds atoms, and
    # stop_allowed IS H. Measured before any method was built.
    ("S5 must beat S3 and S4",
     "S5 must beat S1, S2, S3 AND S4 on distinct valid sets; best-of-K is reported against the p* ceiling, never tested (G5, G9)", "R15", None),
    ("8 ε-greedy restarts at ε = 0.05",
     "forced-distinct first atoms: eps-restarts measured 2.10 distinct from 8, all deviations non-argmax (G3)", "R15", None),
    ("median pre-completion output size is closest to",
     "median POST-completion size: completion only adds atoms, so the pre- criterion selected for H rejecting on size (P4.5)", "R15", None),
    ("every method spends 8 of its 32 checks",
     "0 for mask-driven arms (stop_allowed IS H), 1 per candidate for direct builders (G3, G6)", "R15", None),
    # R16 - Gate 3's placement. R15's entries were strings that appeared in no
    # upstream document, so the guard passed while three of them still carried
    # the retired rule. These are the sentences that were actually there.
    ("Explicit written decision: does the learned sampler beat S3/S4 on the lattice?",
     "Phase 4 is a diagnostic; the gate's decision moves to Phase 9 under a proxy scorer (G5, G9)", "R16", None),
    ("Does the learned sampler beat training-free PCST and submodular greedy at equal budget? |",
     "the same question UNDER A PROXY SCORER; the lattice answers it by arithmetic (CLAUDE.md Gate-3 row)", "R16", None),
    ("S5 returns strictly more **distinct valid proof sets** than",
     "the count is capped at K=8, beam saturates it and the p* ceiling is 7.78, so a strictly-more rule cannot pass", "R16", None),
    ("8 genuinely different sets **deterministically**",
     "forced-distinct openers measured 2.45 distinct; the count is reported, not guaranteed (decision 3)", "R16", None),
    # R17 — the independent citation re-read of 13 Aug 2026, against the PDFs in
    # the library rather than against any document that summarises them. Two
    # figures had survived every earlier round: Graph-S3's headline gains (the
    # paper's abstract reports +8.1% accuracy / +9.7% F1; the quoted pair was
    # nearly double, and one round had already corrected its *interpretation*
    # without checking the numbers), and SubTB's λ = 0.9 being called the
    # paper's own default (it is the hypergrid value from Appendix A; the
    # paper's other tasks use 0.99–1.9).
    ("+15.6% accuracy / +17.2% F1",
     "Graph-S3's abstract reports +8.1% accuracy / +9.7% F1 (corrected 13 Aug 2026)", "R17", None),
    ("+15.6% acc, +17.2% F1",
     "Graph-S3's abstract reports +8.1% accuracy / +9.7% F1 (corrected 13 Aug 2026)", "R17", None),
    ("(+15.6/+17.2)",
     "Graph-S3's abstract reports +8.1% accuracy / +9.7% F1 (corrected 13 Aug 2026)", "R17", None),
    ("+15.6 acc / +17.2 F1",
     "Graph-S3's abstract reports +8.1% accuracy / +9.7% F1 (corrected 13 Aug 2026)", "R17", None),
    ("+15.6 accuracy / +17.2 F1",
     "Graph-S3's abstract reports +8.1% accuracy / +9.7% F1 (corrected 13 Aug 2026)", "R17", None),
    ("+15.6% accuracy and +17.2% F1",
     "Graph-S3's abstract reports +8.1% accuracy / +9.7% F1 (corrected 13 Aug 2026)", "R17", None),
    ("the value SubTB (ICML 2023) carries through its own experiments",
     "0.9 is SubTB's hypergrid value (Appendix A); other tasks use 0.99–1.9 — no single paper default (decision 28)", "R17", None),
    ("SubTB (ICML 2023)'s own default",
     "0.9 is SubTB's hypergrid value only, not a paper-wide default (decision 28)", "R17", None),
    # Also R17: the SA-GFN bias direction. The 5,220/1,042 cyclohexane figure is
    # the paper's FRAGMENT-BASED result — over-production of a highly symmetric
    # fragment — while "toward low-symmetry objects" is its node-by-node
    # direction. Four documents and two docstrings paired the number with the
    # wrong direction; the non-applicability argument itself was unaffected.
    ("bias toward low-symmetry objects",
     "paradigm-dependent: fragment-based over-produces symmetric fragments (the 5,220/1,042 figure); node-by-node biases toward fewer symmetries", "R17", None),
    # R18 — the Phase-4 build, 13 Aug 2026. Three sub-decisions of the ruled §6
    # table that measurement overturned: pcst_fast's Windows wheel returns
    # corrupted arrays (59/60 wrong optima), C_e's "median without exceeding"
    # admits a median AT the cap with a 0.500 breach rate, and decision 5's one
    # live condition is size-confounded. None read a learner result.
    ("**`pcst_fast`, pinned.** The platform risk an earlier draft raised is not real",
     "the wheel exists and is WRONG (59/60 wrong optima); an exact solver is adopted (PHASE4_DECISIONS.md 1.1)", "R18", None),
    ("selected by median **post**-completion output size closest to `max_atoms`",
     "selection is on the post-completion BREACH RATE; a median AT the cap straddles it at 0.500 (PHASE4_DECISIONS.md 1.2)", "R18", None),
    ("median POST-completion output size is closest to",
     "the post-completion breach rate, ties to the larger median (PHASE4_DECISIONS.md 1.2)", "R18", None),
    # Also R18, from the post-fix audit (13 Aug 2026): two restatements the fix
    # round corrected at their source and missed in the plan's build-order and
    # exit-criteria rows.
    ("recorded as **provably inert on the lattice**",
     "measured inert at the frozen cap — the theorem's premise is false for the forced-opener path (PHASE4_DECISIONS.md 1.4 F2)", "R18", None),
    ("selected on median **post**-completion output size",
     "the post-completion breach rate, ties to the larger median (PHASE4_DECISIONS.md 1.2)", "R18", None),
    # R19 — the Phase-5 build audit, 13-14 Aug 2026. Candidate B's grammar
    # guarantees a valid JSON *prefix*, not that the object closes within
    # max_new_tokens; B's own 3-turn check produced a truncation parse failure.
    ("parse failure → 0 by construction",
     "never malformed — prefix validity, not completion within the token budget (PHASE5_DECISIONS.md 1.6)", "R19", None),

    # R20 — the stale-ruling sweep, 15 Aug 2026.  Decision 6 was RULED on
    # 12 Aug 2026 (option 1, threshold unchanged at 0.10, PHASE3_DECISIONS.md
    # 6.8b), and PHASE3_DECISIONS.md 4 item 8 recorded it correctly — but six
    # other places went on describing it as the open item blocking the
    # calibration gate, including the handoff's own run instructions.  A reader
    # following those would have believed a ruling was owed before step 6 could
    # run.  This is the "a correction lands in four or five places and three is
    # the usual score" pattern with the score at one.
    #
    # Prose explaining the change stays legal (see the RETIRED docstring), which
    # is why the correction notes left in place may quote these without emphasis.
    ("option 2 must be ruled **before** step 6 runs",
     "RULED 12 Aug 2026 — option 1, decision 6 unchanged at 0.10 (PHASE3_DECISIONS.md 6.8b)", "R20", None),
    ("option 2 — raising decision 6's threshold — is the only clean one",
     "the ruling predates the gate; nothing is left to rule before step 6 runs (6.8b)", "R20", None),
    ("Option 2 is worth a ruling before step 6 runs.",
     "RULED 12 Aug 2026 — option 1; 2.3 is the pre-ruling analysis (6.8b)", "R20", "PHASE3_DECISIONS.md"),
    ("DECISION 1 RULED; THE REST AWAIT SIGN-OFF",
     "FIVE RULED (1, 6, 11, 29, 30); THE REST AWAIT SIGN-OFF", "R20", "GRAFT_PHASE3_BUILD.md"),
    ("decision 1 is ruled, the rest are recommended",
     "five decisions are ruled: 1, 6, 11, 29, 30", "R20", "CLAUDE.md"),

    # R21 — Gate-0 item 9 (corpus scope) decided 15 Aug 2026: scope c, 200
    # questions, extending to scope b' for final numbers.  Six documents called
    # it the sole open item or asserted it "still open"; all six are corrected
    # and none of the six retired strings below may reappear as a normative
    # claim.  Ingestion at scope c has NOT run yet -- that remains true and is
    # not what these strings retire.
    ("Item 9's corpus scope is the only open item",
     "DECIDED 15 Aug 2026 -- scope c, 200 questions (GATE0_CONTRACT.md item 9)", "R21", None),
    ("the scope sub-decision is the only open item",
     "DECIDED 15 Aug 2026 -- scope c, 200 questions (GATE0_CONTRACT.md item 9)", "R21", "GATE0_CONTRACT.md"),
    ("Item 9's corpus-scope sub-decision is the one thing left before signature",
     "item 9 is decided; only the signature itself remains", "R21", "GATE0_CONTRACT.md"),
    ("it is still open, and the build", "item 9 is now decided (scope c); ingestion at that scope has not run", "R21", "GRAFT_PHASE7_BUILD.md"),
    ("item 9's, it is still open, and the build", "item 9 is now decided (scope c); ingestion at that scope has not run", "R21", "PHASE7_DECISIONS.md"),
    ("Undecided until the pilot's sizing memo is read together with item 8's",
     "DECIDED 15 Aug 2026 -- scope c, 200 questions", "R21", "GATE0_CONTRACT.md"),
    # R22 -- the Gate-0 SIGNATURE (15 Aug 2026, delegated and recorded as such)
    # and its Phase-7 consequences.  R21 retired only the item-9 wording; a
    # 15 Aug audit found the signature itself still described as outstanding in
    # SIX live documents (CLAUDE.md, chatcontext1.md x3, README.md,
    # PHASE5_DECISIONS.md, PHASE6_DECISIONS.md, GRAFT_PHASE7_BUILD.md), the
    # Tier-B by-name refusal still asserted in the honesty stamp, the artefact
    # and a test that PINNED the stale wording -- and the same audit found the
    # Phase-3 header still claiming only decision 1 was ruled, contradicting
    # its own signed section-6 heading.  None of these may reappear.
    ("all ten items decided, UNSIGNED",
     "SIGNED 15 Aug 2026 (delegated, recorded as such); Gate 1 unblocked", "R22", None),
    ("filled and decided; UNSIGNED",
     "SIGNED 15 Aug 2026 (delegated, recorded as such); Gate 1 unblocked", "R22", None),
    ("Gate 1 is blocked on it being signed",
     "Gate 0 signed 15 Aug 2026; Gate 1 is unblocked", "R22", None),
    ("Gate 1 is blocked on this document being signed",
     "signed 15 Aug 2026; Gate 1 is unblocked", "R22", None),
    ("nine of ten items drafted, unsigned",
     "all ten decided and signed 15 Aug 2026", "R22", None),
    ("but only the Gate-0 signature blocks it",
     "Gate 0 signed 15 Aug 2026; what remains for Gate 1 is the label work", "R22", None),
    ("Tier B refuses until GATE0_CONTRACT.md signs",
     "both tiers live since the 15 Aug 2026 signature (item 3.2 as amended)", "R22", None),
    ("refused until GATE0_CONTRACT.md signs",
     "both tiers live since the 15 Aug 2026 signature", "R22", None),
    ("Tier B refuses by name until Gate-0 signs",
     "Tier B live since the 15 Aug 2026 signature", "R22", None),
    ("Decision 1 is ruled; the remaining values are recommended and awaiting sign-off",
     "section 6 signed off in full 15 Aug 2026 (rulings 11-12 Aug, remainder signed as written)", "R22", "GRAFT_PHASE3_BUILD.md"),
    # R23 -- the 16 Aug 2026 Phase-9 audit.  Three blockers, and each left a
    # wording that would reintroduce it.  The capacity one is the important
    # entry: the retired 1% tolerance survived as an *executable default*, which
    # this guard structurally cannot see (it reads prose, not floats), so the
    # prose retirement below is paired with a regression test in
    # test_setgen_real.py that reads the default itself.
    ("Cor 5.1's terminal-reward scaling applies to exactly it",
     "Cor 5.1 corrects graph automorphisms under node-by-node generation; an id-defined set-state MDP has no automorphism group (PHASE9_DECISIONS.md 1.4)", "R23", None),
    ("the one Symmetry-Aware GFlowNets (ICML 2025) is actually about",
     "duplicate corpus text is a dataset diagnostic, not the phenomenon SA-GFN corrects (PHASE9_DECISIONS.md 1.4)", "R23", None),
    ("recorded in ``pins.GRANULARITY``",
     "no such symbol; the decision lives in pins.CORPORA['wiki2']['gold'] (PHASE9_DECISIONS.md 2.1)", "R23", None),
    ("h_filter after dedup by canon_set_hash",
     "the portfolio constructs through the masks and spends 0 terminal checks (PHASE9_DECISIONS.md 1.1)", "R23", None),
    ("supporting sentences over the shipped distractor context",
     "documents are paragraphs on both corpora, decided by measurement (PHASE9_DECISIONS.md 2.1)", "R23", None),
    ("content_key duplicates per pool",
     "structurally zero by id derivation; the varying quantity is equivalent_evidence_atoms (PHASE9_DECISIONS.md 1.4)", "R23", None),
    ("+8.1% acc / +9.7% F1 macro over sparse final-answer",
     "8.1/9.7 is Graph-S3's abstract against seven baselines; its Table 3 ablation is 11.8/17.1, macro-averaged by this project (PHASE9_DECISIONS.md 7.7)", "R23", None),
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



def _table_cells(line: str) -> int:
    """Pipes that actually delimit cells — an escaped ``\|`` does not."""
    count, i, n = 0, 0, len(line)
    while i < n:
        if line[i] == "\\" and i + 1 < n and line[i + 1] == "|":
            i += 2
            continue
        if line[i] == "|":
            count += 1
        i += 1
    return count - 1


def _check_tables(name: str, text: str) -> list[str]:
    """Rows whose cell count disagrees with their table's.

    **Added R16.** A literal ``|`` inside a cell — ``E[best-of-K | p*]`` — splits
    that row into an extra column, and markdown silently drops the overflow. It
    cost `GRAFT_PHASE4_BUILD.md` its headline measurement: the flawless-sampler
    row rendered three cells into a two-column table and the number 1.8865
    vanished from the rendered page while reading correctly in the source. The
    blocklist cannot catch that, because nothing is misspelled.
    """
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("|"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        widths = [_table_cells(l) for l in lines[i:j]]
        expected = max(set(widths), key=widths.count)
        for k, (line, w) in enumerate(zip(lines[i:j], widths), start=i + 1):
            if w != expected:
                out.append(
                    f"{name}:{k}: table row has {w} cells, table has {expected}"
                    " — an unescaped '|' inside a cell drops content when rendered"
                    f"\n    {line[:90]}"
                )
        i = j
    return out


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
        failures.extend(_check_tables(name, text))

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
