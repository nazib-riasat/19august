# Session handoff — GRAFT, Phases 0–5 (+2.5)

**Rewritten 11 Aug 2026; amended 13–15 Aug 2026 (last: Gate-0 item 8 closed, §3a)** after a four-way external
audit (Phases 0–3 code, Phase 2.5 + Phase 4 plans), an independent citation
re-read, the Phase-2.5 spike build, the Phase-4 Stage-A build, and the Phase-5
build with its own two-source audit (`PHASE5_DECISIONS.md` §7). Read this plus
`CLAUDE.md` and you can continue without re-deriving anything.

---

## 1. Where the project actually is

| | State |
|---|---|
| **Phase 0** — scaffold, schemas, event log, ledger, config | **Built, green.** All 13 exit criteria. |
| **Phase 1** — `H`, `U`, `R`, masks, obligations, `d(s)` | **Built, green.** All 16 exit criteria. |
| **Phase 2** — ProofLattice + exact evaluator | **Built, green.** All 21 exit criteria. |
| **Phase 2.5** — annotation feasibility spike | **Tooling built and run 13 Aug 2026** (`scripts/phase2_5/`, `data/phase2_5/`): corpus pinned, 58 turns extracted with Qwen2.5-**3B** (ruled deviation from the 7B), G7 span floor passed at 0.85, items + guidelines + machine-assisted bootstrap labels exist. **Item 8 answered 15 Aug 2026 — GO** (§3a): 16.7 h for 3,625 items, D1 κ 0.829 / D2 κ 0.813, both real inter-annotator agreement. Guidelines v1 supersede v0; the spike's own batches are archived, their candidate ids having resolved to nothing in the real graph. |
| **Phase 3** — 9 learner arms + Gate-2 harness | **Code complete, uncalibrated.** Steps 1–5 and 7–10 built and green; **step 6, the calibration gate, has not run.** A 13 Aug audit added five harness guards (tv_threshold pinned, truncation refused, realized-spend instrument clause, probe-β check, `budget_for` 52 evals) — all latent-hazard fixes, no result invalidated. |
| **Phase 4** — five Tier-1 search methods + Gate-3 harness | **Stage A built, green and RUN (13 Aug 2026);** Stage B waits on Phase 3's matrix, as G7's two-stage exit requires. `graft/setgen/search/`. **Three sub-decisions of the ruled §6 table were overturned by measurement** — `PHASE4_DECISIONS.md` §1. |
| **Phase 5** — Stage A ingestion | **Built, green, and run (14 Aug 2026)** — `graft/ingest/`, Gate-0 contract drafted 9/10, **G2 bakeoff frozen on candidate B** (1.7% parse failure vs A's 23.3%), **live pilot 248 turns end to end**. Every machine-measurable exit criterion met. A two-source audit confirmed 19 defects + 1 found in-house — two of them blockers in the bakeoff harness itself — all fixed and regression-tested (`PHASE5_DECISIONS.md` §7); the bakeoff's first run was aborted and its instrument corrected before anything froze (§1.6), and the token cap was raised once more under a written-down prediction after a `no_survivor` verdict (§2.1a/b). **Remaining: four human audit worksheets + Gate-0 item 8.** Two findings to carry forward: **a third of extracted assertions fail to ground** (108/333) and the **quarantine rate is 32%** — the NLI audit is the instrument that separates "extractor overreaches" from "0.8 is strict". |
| **Phase 6** — Stage B graph construction | **Built, audited, green (14 Aug 2026)** — `graft/graphbuild/` (validate, items, candidates, embed, features, encoders, decoders, loaders, llmlink, commit, standin, gate1). A two-source audit confirmed **21 defects (2 blockers)**, refuted 8; all fixed and regression-tested (`PHASE6_DECISIONS.md` §7). Smoke run rebuilt: 187 D1 items (177 with construction-time candidates), 120 D2 items, 130 edges, corruption audit green over real commits. **The trainer (P6.11) is now built** and the decisive path exists; a review of it found 9 more defects (2 blockers — gold labels joined by a non-durable id, and every LINK label naming the spike's entity namespace), all fixed (`PHASE6_DECISIONS.md` §8). Gate 1 waits on the **Gate-0 signature alone**: the pilot batch (15 Aug) supplies 40 D1 + 49 D2 human labels in the graph's own namespace (40/40 span-join, 9/9 LINK reachable), and a second audit fixed 8 more defects (`PHASE6_DECISIONS.md` §8.5 — incl. a cwd-dependent test set and a filename-sort gold collision). |
| **Phase 7** — Stage C retrieval | **Planned 14 Aug 2026** (`GRAFT_PHASE7_BUILD.md`, G1–G10, §6 unsigned); no code. Five training-free channels + fusion first (no GPU), the ≤8M GNN scorer last; the two-tier recall instrument (Tier A runs today, Tier B waits on the Gate-0 signature); `pool_cap = 64` inherited frozen. |
| Phases 8–11 | Not started. |

**958 tests, ~3 min, all passing. Nothing skipped, nothing xfailed.**
The 13 Aug audit round is recorded in the R17/R18 retirements of
`scripts/check_plan_consistency.py`; the Phase-5 audit added R19. Each phase's
DECISIONS file carries its own record.

**Three things to know before touching Phase 4** (all in `PHASE4_DECISIONS.md` §1,
all measured, none reading a learner result):

1. **`pcst_fast` is installed and must not be used.** Its `cp311-win_amd64`
   wheel returned the wrong optimum on **59 of 60** random graphs — every output
   array the right length with every element equal to its first. S4 uses an exact
   solver; the library survives only in the regression test that documents the bug.
2. **`C_e`'s selection moved to the breach rate.** "Median closest to `max_atoms`
   without exceeding it" admits a median *equal* to the cap: at `C_e = 0.5` the
   median is 8.0 = `max_atoms` and **half** the outputs are rejected on size.
3. **Decision 5's one live Gate-3 condition is size-confounded.** Mean pairwise
   Jaccard distance falls monotonically with set size even for *random*
   portfolios, and S4-informed already scores 0.483 against `p*`'s own 0.4506
   (pinned duplicates-included convention) — so S5 cannot beat it whatever it
   learned. The ruled metric ships; a size control ships beside it.

**Read `PHASE3_DECISIONS.md` §6 first if you are touching a learner.** An
external review on 12 Aug 2026 found three defects in *controls* — GAFlowNet's
Eq. 4, LED's Eq. 5 and its decomposition-error redistribution, and the L6/L7
capacity match — and **all three ran in the direction that flatters the proposed
method**. They are fixed. The convergence and capacity tables in §2.2 and §2.4
predate the fixes and are marked stale.

The suite got slower because Phase 3 added `test_setgen_convergence.py`, which
actually trains: it is the only place a *learned* quantity is asserted, and steps
2–5 exist precisely because a broken sampler or adapter yields plausible curves
and meaningless numbers.

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pytest -q
python scripts/check_plan_consistency.py
python scripts/verify_handoff.py --preset synthetic
```

Python **3.11** (`py -3.11`); the system default here is 3.14, which PyTorch has
no wheels for. `.venv` is never copied between machines — `PHASE0_DECISIONS.md`
§3.1.

**Git state.** `master`, no remote, linear solo history.

---

## 2. Read these, in this order

**Tier 1 — you cannot work without these (~1 hour).**

1. `CLAUDE.md` — why decisions are what they are, what was cut, cost to reverse.
   §4.1 (ideas that died and why), §5 (errors already caught), §6 (frozen values)
   and §8 (the four gates) are the load-bearing sections.
2. `GRAFT_RESEARCH_PLAN_v1.md` (v1.2) — the science. **Wins any conflict with
   every other document.** §4.1 (reward), §4.3 (masking), §4.5 (local credit),
   §5.1 (matched baselines), §6.3 (five ceilings), §7 (the gates).
3. `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (v1.1) — 12 phases in build order, and
   the F1–F13 fix table that phase plans cite constantly.
4. `GRAFT_PHASE3_BUILD.md` — **the thing being built right now.** §6 is
   normative; the changelog at the top records five revision rounds and what
   each fixed.

**Tier 2 — read before touching the corresponding code.**

5. `PHASE0_DECISIONS.md` — before changing a config default.
6. `PHASE1_DECISIONS.md` — before touching `H`, `U` or `d(s)`.
7. `PHASE2_DECISIONS.md` — before touching the generator, `Target` or
   `ActionPolicy`. §4.2 (the β eligibility finding) and §5 (three post-build
   fixes) matter most.
7b. `PHASE3_DECISIONS.md` — **before running the calibration gate or touching any
   learner.** §2.3 (L7 is the slowest arm to converge, and what that risks for
   Contribution 3) and §4 (eight open items) are the ones to read first.
8. `GRAFT_PHASE0_BUILD.md` / `GRAFT_PHASE1_BUILD.md` / `GRAFT_PHASE2_BUILD.md` —
   the gap sections (G*n*) explain why each module is shaped as it is.
9. `README.md` — setup on a new machine, Kaggle, what must match across machines.
10. `Research papers/INDEX.md` — 103 papers with verified titles and venues.
    **Also holds the dataset selection** (§7), which is the thing `CLAUDE.md` §7
    wrongly says was never written down.

**Tier 3 — historical. Do not treat as live.**

`GRAFT2_9_EPISETFLOW_PIPELINE.md`, `GRAFT2_9_EVIDENCE_REVIEW.md`,
`GRAFT2_9_EPISETFLOW_EVIDENCE_AUDIT.md`, `GRAFT_MERGED_PLAN_EVIDENCE_CROSSCHECK.md`.
These describe superseded designs and *should* contain retired wording;
`scripts/check_plan_consistency.py` excludes them for that reason.

**Code, in dependency order.** `graft/schemas.py` → `graft/config/schema.py` →
`graft/core/{checker,masks,obligations,utility,reward,incremental,resolve}.py` →
`graft/synth/{lattice,enumerate,exact,policies,audits}.py` →
`graft/setgen/{features,policy,rollout,flgfn_probe,trainer,gate2}.py` →
`graft/setgen/learners/*.py`. Every module's docstring carries its own reasoning;
they are not decoration and are the fastest way in. **Start with `trainer.py`'s
docstring** — it defines the `g[i]` coordinates all four flow objectives are
written in, and none of the learner files makes sense without it.

---

## 3. What to do next

**Four tracks, three of them independent** (updated 14 Aug 2026):

| Track | Next action | Blocks |
|---|---|---|
| **A — Phase 3** | the calibration gate below (~12 GPU-h at rung 0) | Gate 2, Phase 4 Stage B (S5's row), Phase 9 |
| **B — Phase 5 GPU** | **done** — bakeoff frozen (candidate B), live pilot run. Only a scope-scale re-ingestion remains, and that waits on item 9 | — |
| **C — human** | **Gate-0 item 8 is DONE — GO (§3a).** Remaining: **item 9's corpus scope** (a decision, not a measurement), then signature; plus the four Phase-5 audit worksheets in `artefacts/phase5_pilot/` | **signing `GATE0_CONTRACT.md`**, and therefore Gate 1 |
| **D — Phase 6** | **code complete + twice audited.** Next: nothing in code — Gate 1 needs the Gate-0 signature, D2 labels, and D1 link labels re-annotated against the current 187-item batch | a decisive Gate 1 |

A and D contend for the same 8 GB GPU and cannot run simultaneously; C is human
time and contends with nothing. Track D's build is mostly CPU work until its
step 4.

### 3a. The annotation pass — DONE, Gate-0 item 8 is GO (15 Aug 2026)

**Item 8 is measured and the stop condition does not fire.** Four annotators:
Sabbir and Nazib on D1, Meherin and Sakib on D2.

| | Required | Measured rate | Hours (slower annotator) |
|---|---|---|---|
| **D1** | 625 test + 1,500 train = 2,125 | 228-735 items/h | **9.3 h** |
| **D2** | 300 test + 1,200 train = 1,500 | 203-352 items/h | **7.4 h** |
| | | | **16.7 h; 8.4 h each split two ways** |

Costed at the *slower* annotator on each decoder, not the mean. 18.1 h at the
pessimistic end of the power grid. **No scope reduction is required on
feasibility grounds.**

**Agreement — item 7, and it took three attempts.**

| Attempt | D1 | D2 | What it established |
|---|---|---|---|
| v0 guidelines | κ 0.262 | κ 0.179 | **failed.** Five of ten D1 disagreements shared one cause: a rule v0 never stated |
| v1 guidelines | **κ 0.829** | κ 0.517 | D1 passes; D2's batch was contaminated (16/20 pairs seen by one annotator only) |
| v1, fresh pair | — | **κ 0.813** | D2 settled, class-varied batch, neither annotator had seen any of it |

Both finals are **real inter-annotator agreement between two people**, so
`GATE0_CONTRACT.md` item 7's old caveat ("say self-agreement, never IAA") no
longer binds.

**A fourth D2 measurement exists on disk and must NOT be quoted.** The
`*_clean` batch gave raw agreement 0.850 with **κ exactly 0**: one annotator
used a single label on all 20 items, so chance agreement equalled observed and
κ collapsed. That is the kappa paradox — zero class variance, not disagreement.
It is why the decisive D2 run used the class-varied batch instead.

**Guidelines v1** (`GUIDELINES_D{1,2}_v1.md`) supersede v0 and carry five rules,
each followed by the disagreement that produced it. The load-bearing one, which
v0 never stated: *the candidate list is the graph's memory, not the annotator's*
— if the entity is in the list, link, regardless of whether you saw it created.

**Gold sets:** `labels/d{1,2}_labels_adjudicated_*.jsonl`,each row carrying its own
provenance (`both_agreed` / `adjudicated`). **Caveat recorded in every
adjudicated row and both guideline docs: the adjudications were
assistant-derived and human-accepted, not resolved item-by-item by both
annotators.** Weaker than item 7 specifies.

**The first attempt is archived, not deleted** —
`data/phase2_5/archive_pass1_v0/` and `labels/superseded/`, each with a README.
Its labels are **not** carried into gold: v1 changes the answer to whole classes
of item (rule 1 alone flips 10 of the original 40 D1 labels), and mixing two
labelling standards behind one gold column would be worse than having fewer
labels.

**Two tooling flags on `scripts/phase2_5/annotate.py`:** `--items <tag>` selects
a named batch so two batches' labels cannot be mixed; `--second-annotator
<name>` records a different person's pass under their own name, reports
inter-annotator rather than self-agreement, and drops the 2-day gap check
(independence comes from being two people). A markdown form
(`D2_ANNOTATION_FORM.md`) exists for annotators without the CLI — it produces
labels but **no timing**.

**Two findings that are not about the annotators**, both for Phase 6:

1. **The pair proposer surfaced zero `CONFLICT` across 49 items**, 34 of them
   knowledge-update by design. D2's rare classes are the binding constraint on
   C1, so a pool that does not surface them is a Gate-1 problem arriving early.
2. **A question is stored as an assertion** — *"Can you provide more information
   on social identity theory?"* appears as claim text in a D2 pair. A question
   asserts nothing; this is a Stage-A extraction defect polluting the pair pool.

**Item 9's corpus scope is now the only thing between here and signing the
Gate-0 contract**, and it is a decision, not a measurement: full corpus 1,820
GPU-h · evidence sessions only 80.8 h · 100 questions 16.2 h · 200 questions
32.3 h. One constraint already fixed: the knowledge-update evidence sessions are
in every candidate scope.

---

**Step 6, the calibration gate. It is a hard stop and it is the only thing
standing between here and a Gate-2 result.**

```bash
python scripts/phase3_calibrate.py --out artefacts/phase3_calibration.json
```

The script implements decision 4's ladder in decision 4's order: β eligibility →
`N` from the wall-clock ceiling by an L5 throughput pilot on the **tuning** suite
→ β sweep at that `N` → freeze β → the L4/L5 sanity check at decision 6's 0.10.
Pass ⇒ adopt; fail ⇒ next rung; fail at the last ⇒ **Gate 2 inconclusive**, never
a negative verdict on C3.

**Rung 0 should adopt — measured forecast, 12 Aug 2026.** L5 on the tuning
suite, seed 13, at the default β: TV **0.386** at 100k trajectories, **0.191** at
400k, **0.0990** at 1.2M. So L5 crosses decision 6's 0.10 at ~1.2M, and rung 0's
ceiling buys **N ≈ 4.0M** — 3.3x headroom. The ladder should not escalate, which
puts calibration at ~4.5 h rather than the ~31 h a walk to rung 2 would cost.
A scheduling forecast only: one seed, one arm, and decision 6 needs L4 **and** L5
averaged over three. It reads L5 on the tuning suite, which is exactly the read
the ladder itself performs, and decision 6 was already ruled before it was taken.

Budget: **12 GPU-hours** at rung 0 (`c₀ = 1 h`), rising to 84 cumulative if the
ladder goes to its top rung. `--quick --rungs 3` is a three-second wiring check
and its output must never be written into §6.

**Then, and only then:** write `N` and the frozen β into `GRAFT_PHASE3_BUILD.md`
§6, and run the matrix — `graft.setgen.gate2.run_matrix(envs, spec)`. After any
L6/L7/GAFlowNet run the §6b amendment procedures are contaminated by learner
results, which is why nothing may be adjusted afterwards.

**Read `PHASE3_DECISIONS.md` §2.3 before running the gate.** Measured on
`tiny_instance()`, L7 is the *slowest* of the six flow arms to converge. If the
adopted `N` lands where it has not converged, criterion 26 records C3 unsupported
for a reason that is about budget, not mechanism. §2.3 lists three options and
notes that option 2 — raising decision 6's threshold — is the only clean one and
must be ruled **before** the gate runs, not after.

### Built in Phase 3

| Step | Module | What it is |
|---|---|---|
| 1 | `requirements-ml.txt`, narrowed structural test | the ML boundary: `graft.core` and `graft.synth` still import no ML library; `graft.setgen` may |
| 2 | `graft/setgen/flgfn_probe.py` | the FL-GFN discharge — numpy only, no learner needed |
| 2 | `graft/setgen/features.py` | `SyntheticFeaturizer`, the fix-F6 boundary; the only module that reads an atom id |
| 2 | `graft/setgen/policy.py` | `Policy`, `LogZHead`, `StateFlowHead`, `PotentialHead`, `DeficitHead`, `capacity`, `match_capacity` |
| 3 | `graft/setgen/rollout.py` | `sample_trajectories`; re-exports `uniform_backward` rather than reimplementing it |
| 4 | `graft/setgen/trainer.py` | `TrainSpec` / `Environment` / `Batch` / `Arm` / `Trainer`. Owns ε, `N`, the checkpoint schedule, capacity, the seeds, `ledger=None`, **and the `g[i]` coordinates every flow objective is written in** |
| 5, 7–9 | `graft/setgen/learners/` | all nine arms. L6 and L7 share `led_db_loss` verbatim; the entire difference is `delta_d=True` on L7's featurizer |
| 10 | `graft/setgen/gate2.py` | `run_matrix`, `paired_bootstrap`, `consistency_report`, `audit_block`, and decision 26's verdict applied |
| 6 | `scripts/phase3_calibrate.py` | **the gate — wired, not run** |

### Measured results already in hand

**The FL-GFN discharge** (decision 24). Best-fitting member of the deficit-potential
family over all 8,638 main-suite terminals, every coefficient and a per-instance
constant free: RMS **0.4666**, R² 0.910, all 8,638 terminals off tolerance, at
every eligible β. Named special cases: `deficits_only` RMS 0.4717, `uniform_omega`
RMS 1.0298.

**This disproves that potential family. It does not show FL-GFN is inapplicable** —
plan §4.5.2 retired that stronger reading once already, and the two-part claim
string ships with the numbers so they cannot be separated later.

---

## 4. The environment, and one discrepancy

`torch 2.11.0+cu128`, verified working: `sm_120` in `torch.cuda.get_arch_list()`
and a real matmul executes. **A cu124 build would install, import, report
`cuda.is_available() == True`, and then fail at the first kernel launch** — this
GPU is Blackwell. `requirements-ml.txt` records that trap and the check that
catches it.

**`CLAUDE.md` line 7 says "1× RTX 5090 (32 GB)". The actual machine is an
RTX 5050 Laptop GPU with 8 GB.** Phases 0–4 use no GPU models, so nothing built
so far is affected. But fix F7 (VRAM collision — 7B extractor + 3B reader +
embedder, stage-sequential) was reasoned against 32 GB and is far tighter at 8.
**Not corrected**, because "Setup assumed" may describe a target machine rather
than this laptop. Someone has to say which.

---

## 5. Open items, honestly

| Item | Blocks | Where |
|---|---|---|
| **The terminal convention was never written down** | nothing now; it should still be recorded | plan §4.1 says "pick one and write it down"; nobody did. The code *has* picked one — measured on `tiny_instance()`, a state that is both a valid stop and has children gives `F(s) = 181.49` vs `R(s) = 14.88`, so `R` is the flow on the terminating `STOP` edge, **not** the terminal state flow. Phase 3 now depends on it in four objectives and honours it: the terminating transition is a first-class step with its own `log P_F(STOP \| x)`, `log P_B = 0` and `φ` slot, so no loss assumes `F(X) = R(X)`. **Three lines to record; still not done.** |
| **Phase-3 §6 not fully signed off** | quoting any Gate-2 number | ε, lr, batch, the 1 h ceiling, the 0.10 sanity threshold, `c₀`, the 0.05 consistency band, `λ_aux` are all `[recommended]`; **the build adopted them as written and says so** (`PHASE3_DECISIONS.md` §4 item 2). `N` and β come from step 6. `23a` is filled from LED-GFN Appendix C. |
| **SubTB's λ is in no decision table** | L5, and therefore β and `N` | set to 0.9 in `TrainSpec.subtb_lambda` — SubTB (ICML 2023)'s *hypergrid* value (its other tasks use 0.99–1.9, so it is not a single paper default), and **[ANALYSIS]** as applied to this environment. Added to §6 as decision 28 by the build; **not signed.** L5 is the arm that selects β *and* sizes `N`, so it is the worst arm to have an unlisted hyperparameter. |
| **L7 is the slowest flow arm to converge** | the calibration gate | measured on `tiny_instance()`, one seed: 0.167 at 20k against GAFlowNet's 0.097. If `N` lands where L7 has not converged, criterion 26 records C3 unsupported for a budget reason. `PHASE3_DECISIONS.md` §2.3 has the three options; option 2 must be ruled **before** step 6 runs. |
| **Gate-0 re-sign-off** for the β eligibility amendment | formally, Phase 3 | `GRAFT_PHASE2_BUILD.md` §6b, decision-rule procedure, step 2. Has not happened. |
| **The 0.001 margin** | possibly a regeneration | one main-suite instance clears the `neither` band at β = 4 by 0.001. Both exits cost something; deliberately undecided. `PHASE2_DECISIONS.md` §4.2. |
| **Gate-0 data contract** (v1.2 §7 items 1–10) | Phases 5–9 | **all ten items filled; UNSIGNED.** Item 8 measured 15 Aug 2026 — **GO**, 16.7 h, D1 κ 0.829 / D2 κ 0.813 (§3a). **Item 9's corpus scope is the only open item**, and it is a decision, not a measurement. D1 and D2 still have no off-the-shelf supervision. |
| **The D2 pair pool surfaced no CONFLICT at all** | D2's rare classes, and therefore C1 | 0 of 49 items, with 34 drawn from knowledge-update questions. A proposer property, not a labelling one (§3a) — Phase 6's pair proposer, not Phase 2.5's annotation. |
| **A question is stored as an assertion** | D2's pair pool, and ceiling 1 | "Can you provide more information on…" appears as claim text in a D2 pair. A question asserts nothing and should not enter a memory graph — a Stage-A extraction defect (§3a). |
| **Decision 11's tolerance moved; decision 29 is new** | signing off §6 | R13 found the 1% capacity tolerance unachievable by width (the dead block is ~half a width step at every scale), so the criterion is now minimality plus "control never smaller". Decision 29 fixes GRPO's `G = 8`. Both were changed **before** any L6/L7/GAFlowNet result was inspected, which is what §6b's second procedure is about — but neither is signed. |
| **Three late-found baselines** | plan §5.3 | HyperMem, Chain-of-Memory, *How Memory Management Impacts LLM Agents* — in the library, not in the plan. |

---

## 6. How this project stays correct

These are the habits that produced the parts that hold up. They are not optional
decoration; skipping them is how the defects below happened.

**Label every claim.** `[EVIDENCE]` (named paper, venue stated) / `[HYPOTHESIS]`
(this project tests it) / `[ANALYSIS]` (judgment made here). Never blur them.

**Verify a citation before relying on it.** Cite the verified published title,
never the short name. Several numbers in this project's history survived multiple
review rounds and turned out not to be in the paper.

**Do not fabricate numbers.** This has been caught three times: an FCS literal
invented past the digits actually printed, a "70% restates INDEX" statistic with
nothing behind it, and a `torch==2.5.1` pin that would have failed on this GPU.
Every literal in a test should be independently re-derivable, and several now are
(the FCS reference is recomputed in exact rational arithmetic in its own test).

**A correction lands in four or five places, and three is the usual score.**
Decisions are restated in the gap section, the module spec, the exit criterion,
the normative table and the next phase's handoff. Phase 3 hit this in three
consecutive review rounds. Two mitigations are in place: **restatements point at
a decision number instead of repeating its value**, and
`scripts/check_plan_consistency.py` runs inside the test suite with 49 retired
wordings that fail the build if they reappear.

**The consistency script only catches textual recurrence.** It does not catch
semantic contradictions, and several rounds of real defects passed it cleanly.

**Say what is not known.** "No evidence found" is a finding. So is "this band is
a guess calibrated on Phase-1 fixtures, not on the lattice".

---

## 7. Defects found by review that should not be re-introduced

The reasoning matters more than the string; every one is now a registered
retirement in `scripts/check_plan_consistency.py`.

| Defect | Correct position |
|---|---|
| `exp(β·0) = 1` — an invalid set scoring 1 | the validity indicator is **multiplicative** |
| `p*(FAIL)` called a TV floor | `FAIL` is in **both** distributions, so TV = 0 is reachable |
| "A dead end proves no proof exists" | it licenses only *no proof found under this pool, policy, attempt count, budget* |
| Per-terminal DP for `p_θ` | one forward pass over the policy-independent state graph |
| `environment_fingerprint` carrying β | it must be β-independent, or freezing β after Gate 0 changes the identity of frozen suites |
| `validate_bands` failing open on an unknown scope | it raises; every other decision on that path fails closed |
| Counting source tiers over the whole pool | count tier **keys** among atoms whose `Source` resolves |
| LED consistency measured per terminal | LED decomposes per **trajectory**; one terminal has many |
| GAFlowNet with `Δd` in its policy | `Δd` reaches its **loss** only, or it becomes L7 with extra steps |
| `c_N = 0` proving unbiasedness | Theorem 1 also assumes `L = 0`; call it a finite-budget empirical control |
| Scattering per-edge `Δd` by a state→row map | duplicate states in one batch silently lose their features — a **false negative** for Contribution 3 |
| Asserting 1e-12 agreement for a float32 network | float32 carries ~1e-7; test the adapter in float64 and report the dtype separately |
