# GRAFT — Phase 3 Build Plan: the set-construction policy and the Tier-1 learners (`graft/setgen/`)

**Nine evaluated arms, the frozen policy interface, and the Gate-2 harness.**

Date: 11 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 3, fixes F6/F11/F12) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §4.5, §5.1, Gate 2) · `GRAFT_PHASE2_BUILD.md` §8 · `PHASE2_DECISIONS.md` §7
Effort: ~3 weeks of development **[ANALYSIS]** — an estimate, not a measurement — plus **1.6 to 8.0 GPU-days** of runs depending on which rung of decision 4's ladder is adopted (costed in §6). The largest phase so far.
Status: **§6 partially signed off (11 Aug 2026).** Decision 1 is ruled; the remaining values are recommended and awaiting sign-off. Revision **R6** — see the changelog.
Build: **steps 1–5 and 7–10 built and green as of 12 Aug 2026** (590 tests). **Step 6, the calibration gate, has not run**, so `N` and β are not frozen and no L6/L7/GAFlowNet number may be quoted. The build adopted every `[recommended]` value as written and recorded that it did; see `PHASE3_DECISIONS.md`, which also carries the departures, the FL-GFN measurement, and one **[ANALYSIS]** value this table does not list (SubTB's λ — decision 28 below).

Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps found while making this phase concrete are numbered **G1–G12** and are referenced from code as "Phase-3 gap G*n*", matching the Phase-0/1/2 convention.

### Changelog

**R6 (12 Aug 2026)** — an external code review, checked against the papers rather than against the code. **Three of its findings were defects in controls, and all three ran in the direction that flatters the proposed method** — which is the pattern `CLAUDE.md` §5 records for prose, appearing in code:

1. **GAFlowNet did not implement Eq. 4.** It subtracted `Σ log(1 + r/F)`; the paper's bracket is `P_B + r/F`, so the correction to TB's residual is `Σ log(1 + r/(F·P_B))`. The missing divisor implements an intrinsic reward of `P_B·r` — attenuated by the removable-atom count, and by more of it as the set grows. The required control of plan §4.5.4 had not been run (decision 19).
2. **L6 was not the LED-GFN of the paper's experiments.** Appendix B.1 names the correction-term variant `LED-GFN*` and states the reported results instead **uniformly redistribute the decomposition error** over the trajectory. R5's decision 23b read the alternatives as "plain versus correction term" and chose plain — a third variant the paper never runs, standing in for the baseline plan §5.1 marks mandatory (decision 23b).
3. **The decomposition loss was not Eq. 5.** It used the expected keep rate `1/(1−p)` against an unnormalised energy; Eq. 5 divides by the realised `C = Σz` and by `T`. The `1/T` matters here and not in the paper's own set benchmark, because that benchmark is fixed-length and this environment is not (decision 23a).
4. **The L6/L7 capacity match was nominal, not live.** L6's zeroed `Δd` block leaves `12·hidden` weights that no gradient ever reaches, so a 0.00% nominal match was **1.46% less trainable capacity for the control**. Decision 11's 1% turns out to be unachievable by width — the dead block is ~half a width step at every scale — so the criterion is now minimality plus the directional clause (decision 11).
5. **Gate 2 could return a verdict from a run that was not Gate 2.** A reduced roster or a shortened seed set produced a `contribution_3_supported` boolean from a bootstrap with one outer cluster. It now returns null with the reason recorded (criterion 10). FAIL coverage was measured and never read (criterion 16); best-of-K and the held-out probe read were specified and never built (criteria 17, 23); no model was ever persisted, so Phase 4's S5 had nothing to consume (§8).
6. **GRPO's group was the batch.** Architecture §3.2 freezes `G = 8`; standardising over the batch of 32 made it `G = 32` — a *stronger* baseline than specified, which is why nothing looked wrong (decision 29).

One finding was **not** upheld: that LED's potential must train from a replay buffer. Algorithm 1 does, but Appendix C states molecule generation uses the round's own samples and set generation inherits its settings — and set generation is the closest published task to this one. The build's behaviour was already the paper's; only the citation was missing.

**A second pass on the R6 fixes found three more**, all of the same shape — a value recorded and then not acted on:

7. **An over-ceiling rung could still be adopted.** `ceiling_respected` was computed and the adoption branch read only the sanity result, so the field was decorative; and it was an *average* over the rung's runs, which lets one slow seed hide inside a passing mean. Now the slowest single run decides, an overrun ends the ladder with verdict `over_ceiling`, and nothing is adopted (decision 5).
8. **Best-of-K sampled all eight candidates.** Fix F5 fixes `K = 8` **and** its composition — 1 greedy + 7 sampled. The metric existed but was not the architecture's metric (criterion 17). Greedy rollout added to `sample_trajectories` rather than to a second walk.
9. **Admissibility checked arms and seeds only.** A run on one tiny instance at an uncalibrated `N` and β with no probe suite still produced a scientific boolean. It now checks the frozen suite by instance identity, the probe's presence, and that `N` and β came from an adopted calibration record — and the probe carries its own audit block, without which criterion 23's "under its declared `Δd` density" is unsatisfiable (criteria 10, 23).

**A third pass found five more, two of them blocking.** The two blocking ones are the same defect in different clothes: *a rule stated in the plan that no code path evaluated*.

10. **`N` was calibrated on the cheapest arm and spent by the most expensive.** Measured, L5 runs at ~2,700 trajectories/s and the LED arms at ~900 — decision 23a's eight decomposition iterations per policy step — so an `N` sized to fit L5 in the 1 h ceiling bought L7 **2.9 h**, and decision 5's GPU-day table understated the matrix by the same factor. The ladder could not see it, because its own guard reads `beta_sweep` and `sanity_check` and both run L4 and L5 exclusively. `N` now comes from the slowest arm's measured rate, with a predicted worst-arm time in the ceiling check (decision 4).
11. **Criterion 15 was implemented nowhere.** "L4 and L5 reach exact TV below decision 6's level **on the main suite**" appeared in no code path, so a matrix in which the machinery failed on the scored suite still emitted a verdict. Decision 6's own check runs on five *tuning* instances; the matrix runs twenty *main* instances through one conditional `logZ` head, and passing the first does not imply the second (criterion 15).
12. **`logZ_θ` trained at the shared learning rate**, contrary to both papers this environment's objectives come from (decision 30 — mechanism shipped, value awaiting sign-off).
13. **The retired capacity claim survived in three code docstrings**, one of them asserting the disproved version as fact. The guard read twelve `.md` files and no source, so a correction that landed in four documents missed three modules — in the medium where this project keeps most of its reasoning. `check_plan_consistency.py` now scans `graft/` and `scripts/` too.
14. **There was no Gate-2 runner.** `run_matrix` was called only by tests, and `retarget` — the only correct way to move an `Environment` to the adopted β — was stranded in `scripts/phase3_calibrate.py`. It is now `Environment.at_beta`, and `scripts/phase3_gate2.py` composes the run.

**R5 (11 Aug 2026)** — one root blocker: the two claim-critical learners were named but not specified. Reading LED-GFN and GAFlowNet closed it, and **two of the gaps were structural, not numerical**:

1. **Both methods need a learned state flow, and no module held one.** LED's Eq. 4 is DB-style over `log F̃(s)`; GAFlowNet's augmented TB carries `r(s→s′)/F(s′)`. `StateFlowHead` is now in P3.2, shared by L5/L6/L7/L7b/GAFlowNet and **counted in capacity matching** — without it every matched pair was wrong by the size of a head (G12, decision 27).
2. **LED has no "two regulariser coefficients".** R4's decision 23a assumed a parameterisation the paper does not use: LED has **one** squared loss with **dropout** as the variance regulariser (Eq. 5). 23a now carries the paper's Appendix-C values — potential lr 0.001, dropout 0.10, `N = 8`, `B2 = B1` — read off the paper rather than deferred to a repo lookup.
3. **LED-DB over LED-subTB**, plain rather than the correction variant, identically for L6/L7/L7b (decision 23b). The name "LED\\*" does not appear in the paper text and is not used.
4. **`c_N = 0` does not establish unbiasedness.** Theorem 1 also assumes `L_GAFlowNet(θ) = 0`, which no finite budget delivers. Added a predeclared **zero-intrinsic tail** and the reported final loss, and **downgraded the claim** to a finite-budget empirical control (decision 19b).
5. **The FL-GFN fit omitted the energy offset.** Energy is defined up to an additive constant per instance, so the fit now frees a per-instance `C_i` — otherwise the family is charged for a constant that changes no distribution. The probe also runs at **every eligible β**, since it precedes the freeze (decision 24).
6. **Normative exits corrected**: C3 requires beating L6 **and** GAFlowNet (criterion 26); collision and unconstructible rates restored (criterion 27); Gate-2 item 3 recorded as a **deliberate partial discharge** (criterion 28); L7b's target, loss and weight frozen (decision 26).

**R4 (11 Aug 2026)** — six blockers from review of R3. **Five of the six were R3's corrections failing to propagate or failing to check their own arithmetic**, which is the defect class this project has now hit three rounds running:

1. **R3 was not propagated into §4, §5 or §6b.** Five restatements still carried retired values — GAFlowNet's `delta_d`, `N`'s wall-clock-only rule, the `|log R(X)|` denominator, the strict p95 ordering, and the old calibration procedure. Since §6 declares conflicting restatements to be bugs, these were bugs. **Fixed structurally**: those criteria now *point at decision numbers* instead of repeating values, which is §6's own prescribed remedy.
2. **The ladder did not re-run the β sweep.** Phase-2 decision 22 selects β *at the budget the comparison uses*, so a new `Nᵢ` invalidates the previous β. Each rung is now `Nᵢ` → sweep → freeze `βᵢ` → sanity (G3).
3. **The calibration suite was unnamed.** It is the **tuning** suite, explicitly — a sanity check on the scored main suite that can raise `N` would select the primary budget using main-suite results (G3).
4. **The ceiling broke the schedule.** R3's 6 h with two doublings permits 24 h/run = 27 GPU-days for the matrix alone, exceeding the phase estimate before development. Now `c₀ = 1 h`, rungs 1→2→4 h, worst case **8.0 GPU-days**, with the arithmetic tabulated at decision 5 where the ceiling is chosen.
5. **The FL-GFN probe could not compute its promised quantity** — `ω_j`, `λ_v`, `λ_e` are undefined everywhere. It now reports the **irreducible residual of the best-fitting member** of the potential family, which is a stronger claim than any single instantiation and removes the "you chose the wrong weights" objection (P3.55, decision 24).
6. **"Reference-implementation defaults" is not a frozen value.** Decision 23a now requires literal coefficients plus a pinned commit hash.

**R3 (11 Aug 2026)** — eight blockers plus two inconsistencies, all found by review of R2. Three were *R2's own corrections being wrong*, which is the pattern `CLAUDE.md` §5 and the Phase-2 handoff both warn about:

1. **`N` was still selected using TV.** R2 moved the check after the freeze and called that a separation; but a failed check raised the ceiling and produced a new `N`, so TV influenced it through a slower path. Replaced by a **predeclared, bounded adaptive rule** that says so (G3).
2. **The calibration order was wrong.** Phase-2 decision 22 runs the β sweep *at* `N`; R2 froze β first, leaving that budget undefined when it was needed. Order is now eligibility → `N` → sweep → freeze β → sanity (§4 step 6).
3. **The consistency normaliser was unusable.** Measured on the frozen suite: `min |log R| = 0.0363`, `max = 7.8547`, so a 5% rule permitted 0.0018 absolute error at one terminal and 0.3927 at another *of the same instance*. Now the **per-instance `log R` range** (6.93–7.79, stable to ~12%) with a zero-range guard (G10).
4. **The consistency sample had no sampler, seed or shared set**, so three arms would have been scored on three different path populations. Now one frozen common set per instance, with `FAIL` coverage guaranteed rather than hoped for (decision 14).
5. **"L7 p95 ≤ L6 p95" was noise-sensitive** — 0.030 vs 0.031 would fail while both pass comfortably. Replaced by *both pass the band* plus a **non-inferiority margin** (decision 15).
6. **GAFlowNet was receiving `Δd` in its policy**, making it GAFlowNet *plus* L7's mechanism rather than the published control. `Δd` now reaches the loss only; base objective stated as **augmented TB** (G11).
7. **The FL-GFN discharge had nowhere to happen.** New module **P3.55**, build step 2, decision 24 (G1).
8. **No optimisation protocol existed**, so optimiser, lr, batch, clipping and LED coefficients could be tuned per arm. One frozen shared protocol (decision 23).

Minor, same round: step 5 pointed at "small exact TV" with no threshold (now decision 6); §0 still excluded every Tier-2 learner despite decision 1; and the §6 note claimed decision 6 was produced during calibration when its threshold must be signed before it.

**R2 (11 Aug 2026)** — nine blockers, all found by review of R1:

1. **Decision 1 ruled: Option B.** Its FL-GFN wording was overreaching and is corrected — the measurement disproves *the proposed deficit potential*, not FL-GFN (G1).
2. **GAFlowNet was named but never specified.** New **G11**: variant, the `Δd` → intrinsic-reward map, coefficient and **decay** — the last is a correctness requirement, not a hyperparameter.
3. **The build order was impossible** — it ran an L5 pilot before L5 existed. The calibration gate moved to step 6 (§4).
4. **The `N` rule contradicted itself** — G3 forbade inspecting TV-versus-`N` while §6b required TV to be "moving". Resolved as throughput-only plus a separate threshold — *which R3 found was still not a separation*.
5. **Terminal consistency was measured per terminal**; LED decomposes per **trajectory** (G10).
6. **The evaluation count omitted L7b.** Nine evaluated arms: `9 × 3 × 50 × 20 = 27,000`, so the per-evaluation ceiling is **0.133 s** (decision 2).
7. **Six "frozen" values were blank.** Now carried with recommended values in §6, including an executable definition of "L7 beats L6" (decisions 4, 11, 18–22).
8. **The adapter did not expose action-specific `Δd`**, which is the entire L6/L7 difference (G5).
9. **§7 contradicted decision 1** — GAFlowNet is Tier 2 in the research plan. The exception is now explicit (§7).

---

## 0. What Phase 3 is for, and what it is not

Phase 2 built the instrument. Phase 3 brings something to measure.

**This is Gate 2**, and Gate 2 is where the project is designed to learn cheaply that Contribution 3 does not hold. The research plan's fallback is explicit: *"If Gate 2 or Gate 3 fails, consolidate around Contribution 1 plus the five-ceiling analysis... That is still a complete, publishable systems-and-analysis paper. Structure the work so this fallback survives."* Every decision below is made so that a **negative** result is as publishable as a positive one.

**[EVIDENCE]** The framing — improvement at a *fixed training budget* rather than at convergence — is the one *Better Training of GFlowNets with Local Credit and Incomplete Trajectories* (ICML 2023) and *Learning Energy Decompositions for Partial Inference in GFlowNets* (ICLR 2024 Oral) both use, because both make claims about **credit assignment**, which is a claim about how fast usable signal arrives, not about the asymptote.

| Downstream | Blocked on Phase 3 by |
|---|---|
| Phase 4 (5 search algos) | S5 is "sample K from the trained sampler"; without a trained sampler there is no S5 row, so Gate 3 cannot run |
| Phase 9 (Stage D real) | the featurizer swap is validated by re-running these learners against Stage-B embeddings; without a known-good synthetic result there is no control |
| The thesis | Contribution 3 stands or falls here |

**Not in Phase 3:** any search algorithm (Phase 4), any real data, the answerability gate (Phase 8 — see G2 on why Gate-2 item 5 is not a Phase-3 deliverable), the distilled utility head (Phase 9), Tier-2 and Tier-3 learners **except GAFlowNet**, which decision 1 promotes solely as Contribution 3's required control (§7).

---

## 1. Twelve specification gaps Phase 3 must close

### G1 — The Tier-1 roster contradicts the research plan [ANALYSIS]

**This is the largest gap in the phase and the implementation cannot resolve it.**

The architecture's Phase 3 defines seven Tier-1 learners: L1 supervised stepwise, L2 canonical set imitation, L3 GRPO, L4 TB, L5 SubTB, L6 LED-GFN, L7 proposed.

The research plan says three things that do not fit inside that roster:

1. **Gate 2 item 3** names Tier 1 as *"TB · SubTB · LED-GFN · proposed"* — **four**, all flow methods — and puts *"FM · DB · FL-DB · FL-SubTB · GAFlowNet"* in Tier 2.
2. **§4.5.4** states a **required control**: *"Contribution 3 must beat capacity-matched vanilla LED-GFN **and capacity-matched GAFlowNet**."* GAFlowNet appears nowhere in the architecture's Phase 3.
3. **§5.1** marks **FL-DB / FL-SubTB** as **Mandatory**, with the reason: *"Beating TB/SubTB alone proves nothing — FL-GFN already does, on set generation."*

`CLAUDE.md` §2 states the precedence: the research plan *"Wins any conflict with other docs."* So the architecture's seven-learner roster is **incomplete as written**, and building exactly seven would leave a declared required control unrun.

**Why GAFlowNet specifically is not optional.** **[EVIDENCE]** *Generative Augmented Flow Networks* (ICLR 2023 Spotlight) incorporates intermediate intrinsic rewards — edge- and state-based augmented flows — for sparse-reward exploration, with an asymptotically unbiased treatment. That is **the published way to use an intermediate signal in a GFlowNet**, and Contribution 3's whole proposal is to use an intermediate signal (`Δd`). A reviewer's first question is "did you compare against the obvious published alternative?", and without GAFlowNet the answer is no. Plan §5.1 says this outright: *"It is the direct published alternative for 'use the obligation signal as an intermediate reward,' and Contribution 3 must beat it."*

**What the FL-GFN row becomes, stated carefully.** Plan §4.5.2 concludes GRAFT sits in the "terminal-only global properties" row, *"which makes FL-DB a baseline to run and report, not a method to build on"*; §4.5.5 permits it being *"reported as inapplicable with the reason."* The lattice makes part of that reason a measurement: Phase 2 supplies exact `R(X)` for every terminal, so the **proposed deficit potential's** terminal identity `Φ(X) = β·U(X)` can be *evaluated* across all valid terminals rather than argued.

**[ANALYSIS] What that measurement does and does not establish.** It disproves **the particular potential of plan §4.5.2** — the deficit-based `Φ(s) = −Σ_j ω_j d_j(s) − λ_v|V_s| − λ_e|E_s|` — by showing the terminal identity fails on this environment. It does **not** show that FL-GFN is inapplicable, and calling it "measurably inapplicable" would be exactly the overreach plan §4.5.2 already retired once, where a `[CORRECTION]` records that v1.0's flat "not applicable" was *"too strong"*. The reportable claim is two-part and both parts are needed:

> (i) *measured* — the proposed deficit potential violates terminal consistency on the lattice, by the following margin; and (ii) *argued* — no justified, informative scalar extension of `sufficiency` to partial states is currently available, since proof sufficiency is a global non-additive property of the whole set. Therefore FL-GFN's local-credit advantage cannot be realised here **with any potential we can currently justify**, which is a statement about availability, not about the method.

Reintroducing the retired wording would be a recurrence of a defect this project has already recorded, which is the specific failure `CLAUDE.md` §5 exists to prevent.

**Options, with costs.**

| Option | Roster | Cost |
|---|---|---|
| **A** | 7 as in the architecture | A required control goes unrun; Contribution 3's claim is weak at review, and the plan is violated |
| **B** — **RULED, 11 Aug 2026** | 7 + **GAFlowNet** as an 8th trained arm; the FL-GFN row discharged by the two-part report above, not by training | One extra learner. Discharges §4.5.4's control and §5.1's mandatory row honestly |
| **C** | 7 + GAFlowNet + FL-DB/FL-SubTB trained | Two extra learners. `CLAUDE.md` §10 names scope creep here as the project risk |

**[ANALYSIS]** B, because §4.5.4's GAFlowNet control is load-bearing for the *claim* while §5.1's FL-DB row is load-bearing for the *literature position* — and the second is discharged by a measurement plus a stated limitation, at a fraction of the cost of training.

**Consequence for every count downstream.** `CLAUDE.md` §6 and Phase-2 exit criterion 11 derive budgets from "**seven** learners". The evaluated roster is **nine arms** — L1–L7, **L7b** (reported at every checkpoint, so it is evaluated) and GAFlowNet — giving

```
9 arms × 3 seeds × C = 50 checkpoints × 20 instances = 27,000 evaluations
```

At Phase-2's ≤ 1 h total that is **≤ 0.133 s per evaluation**, down from 0.15 s. Phase 2 measured ~0.0002 s for the DP and ~0.049 s for FCS, so the binding term is FCS, not the DP (decision 2).

### G2 — "Exact TV" is not comparable across learners that are not samplers [ANALYSIS]

L4, L5, L6, L7 and GAFlowNet are trained to sample in proportion to `R`. **L1, L2 and L3 are not.**

* L1 (supervised stepwise) and L2 (canonical set imitation) are trained toward **one gold set**. Their induced terminal distribution is mode-seeking by design.
* L3 (GRPO) is **reward-maximising**, not reward-matching.

Putting all of them in one exact-TV column implies they were all trying to do the same thing and one did it better. They were not. A mode-seeking learner that perfectly imitates gold would score a *large* TV against `p*` while being exactly what it was asked to be.

The research plan already forbids the merge in general terms: §5 opens *"Three separate comparisons answering three different questions. Never merged into one table."*

**Decision.** Exact TV is the **primary** metric for the distribution-matching family only. L1/L2/L3 are reported in the same experiment but in a **separate table**, on the secondary metrics that are meaningful for them — best-of-K valid-set utility at `checker_budget`, and gold-set exact-match rate. Their TV is reported as a **descriptive** number with a one-line note that they do not target `p*`.

**Why report their TV at all.** Because plan §5.1's stated purpose for L2 is *"Tests whether distribution training adds anything over imitating one gold set"* — and that question is only answerable if the imitator's distance from the target is on the page. It is context, not a ranking.

**Gate-2 item 5 is not a Phase-3 deliverable.** The plan's Gate 2 has five items; item 5 is *"Train and calibrate the answerability gate; report the risk–coverage curve"*, and the gate is Phase 8. Phase 3 discharges items 3 and 4; items 1 and 2 were discharged by Phases 0–2. **Gate 2 as a gate spans more phases than Phase 3 does**, and the Phase-3 exit criteria below say so rather than implying Phase 3 closes it alone.

### G3 — Fix F12's "fixed training budget" has no unit and no value [ANALYSIS]

F12 replaced an unfalsifiable criterion with *"exact TV at a fixed number of sampled training trajectories, three seeds, paired bootstrap"*. **The number is not declared anywhere**, and neither is what counts as one trajectory. Until both exist, F12 is still unfalsifiable — it has only moved the ambiguity.

Three things need fixing:

1. **What a trajectory is.** Decision: one root-to-terminal rollout, counted whether it ends at a valid `STOP` **or** at `FAIL`. Counting only successful rollouts would let a learner that dead-ends often buy extra gradient steps for free.
2. **The value `N`.** Undeclared.
3. **Where it is spent.** One model is trained across the 20 main-suite instances (G6), so `N` is a **per-run total across the suite**, not per instance. Stating it per instance would make runs on different suite sizes incomparable.

**How `N` is chosen without contaminating the comparison.** The property to protect is decision 22's: the budget must be fixed on a **non-proposed** learner, on a **non-scored** suite, before any L6/L7/GAFlowNet run. R2 tried to obtain that with a throughput-only rule; the next paragraph explains why that rule was not the one the procedure actually followed.

> `N` is fixed by a throughput pilot on the **tuning** suite using **L5 (SubTB)** only, as the largest `N` whose run completes inside a declared wall-clock ceiling. It is recorded in `§6` before L6, L7 or GAFlowNet trains once.

**[ANALYSIS] What must *not* happen:** choosing `N` by inspecting TV-versus-`N` curves. That selects the budget at which the proposed method looks best — decision 22's error, one level down. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing* (ACL 2018) is this project's significance-testing authority; fixing the decision rule in advance is the project's own **[ANALYSIS]** discipline, motivated by it rather than stated in it.

**A contradiction R1 contained, and the honest resolution.** R1 said "throughput only, never inspect TV" here and, in its risk table, "the pilot must show L4/L5 TV *moving* at `N` before `N` is frozen". R2 moved the TV check *after* the freeze and called that a separation. **It is not.** If failing the check raises the ceiling and produces a new `N`, then TV influenced `N` — through a slower path, but it influenced it. Pretending otherwise would be a purity claim the procedure does not support.

**Decision — a predeclared adaptive rule, honestly labelled.** Rather than assert a throughput-only rule the loop violates:

> **`N` is set by a predeclared adaptive calibration that reads L4/L5 convergence and nothing else. Every rung runs entirely on the *tuning* suite.**
> For `i = 0, 1, 2`:
> 1. **`Nᵢ`** = the largest budget whose L5 run fits ceiling `cᵢ` (decision 5; `c₀`, then two doublings).
> 2. **β sweep at `Nᵢ`** over the eligible candidates, L5, 3 seeds — Phase-2 decision 22 requires the sweep to run *at the budget the primary comparison uses*, so a new `Nᵢ` invalidates the previous β and the sweep is repeated. **Freeze `βᵢ`.**
> 3. **Sanity check** (decision 6) for L4 and L5 at `(Nᵢ, βᵢ)`.
> 4. Pass ⇒ stop, adopt `(Nᵢ, βᵢ)`. Fail ⇒ next rung. Fail at `i = 2` ⇒ **Gate 2 is recorded inconclusive** and the phase stops there.
>
> Every `(cᵢ, Nᵢ, βᵢ)` tried and its outcome is recorded, not only the one adopted.

**Why the whole rung repeats rather than just `N`.** Phase-2 decision 22 fixes β as the candidate minimising mean exact TV *"at the same fixed trajectory budget the Gate-2 primary comparison uses"*. A β chosen at `N₀` is not a β chosen at `N₁`; carrying it forward would silently break the property that decision 22 exists to guarantee.

**The whole ladder runs on the tuning suite, and that is load-bearing.** If the sanity check read the **main** suite and could raise `N`, then the primary budget — shared by every arm and by the Gate-2 comparison — would have been selected using results from the instances the learners are scored on, which is the violation of v1.2 §4.1 that the main/tuning split exists to prevent. Only L4 and L5 run during calibration, and only on tuning.

**[ANALYSIS] What this preserves and what it concedes.**

*Preserved — the property that actually matters for Contribution 3:* **no result from L6, L7, L7b or GAFlowNet ever informs `N`.** The escalation reads only L4 and L5, which are neither the proposed method nor its controls. That is the same structure decision 22 uses for β, and it is what stops the budget being chosen where the proposed method looks best.

*Conceded — plainly, in the write-up:* `N` was **not** chosen by throughput alone. It was chosen by a bounded adaptive rule fixed in advance that reads baseline convergence. The bound (two doublings) and the pre-registration are what keep it a rule rather than a search; calling it "throughput-only" would be false.

**Why not simply stop as inconclusive on the first failure.** That is the stricter option and it remains available, but a single throughput estimate on an untuned L5 is a weak basis for abandoning the gate, and an unbounded retry loop is the thing that must not exist. Two predeclared doublings is the compromise, and its cost is stated above rather than hidden.

**[ANALYSIS]** The distinction being protected is the one Phase 2 drew for a null result: *"a null result that cannot distinguish 'the hypothesis is false' from 'the instrument could not resolve it' is the worst outcome available, because it looks like an answer."* A budget too small for *any* method to converge cannot separate "L7 does not help" from "nothing had room to work".

`C = 50` checkpoints is frozen (Phase-2 exit criterion 11), so checkpoints fall every `N/50` trajectories.

### G4 — Capacity matching is asserted, not operationalised [ANALYSIS]

Plan §4.5.4: *"Adding checker features also adds parameters; a win that disappears under parameter matching is not a win."* Gate 2: *"Capacity matching is mandatory here."* The architecture says only *"widen L6's hiddens to equal L7's parameter count."*

Three things are undefined: **what is counted**, **to what tolerance**, and **in which direction**.

**Decision.**

* **Counted:** all trainable parameters, including the `logZ_θ` head and any potential network `φ_θ`, excluding frozen buffers. Reported per arm.
* **Tolerance:** within **1%**, with the achieved counts printed in the Gate-2 report. A match asserted but never printed is not a match.
* **Direction:** where an exact match is unreachable, the **control must not be smaller** — L6 ≥ L7 and GAFlowNet ≥ L7 in trainable parameters. **[ANALYSIS]** The conservative direction: if the control is larger and L7 still wins, capacity is not the explanation.

This is a build-time assertion, not a reporting convention — the trainer refuses to start a matched pair outside tolerance.

### G5 — The F6 adapter is the entire coupling risk, and has no spec [ANALYSIS]

Fix F6 exists so that *"the learners never know"* whether features came from an MLP over synthetic atoms or from the Stage-B graph encoder. Phase 2 froze the evaluator side (`ActionPolicy.action_log_probs(state_ix, graph)`, decision 8) and stated the layering. **Phase 3 writes the one adapter that joins them, and if it leaks, Phase 9 is a rewrite.**

```
StateGraph ──► SyntheticFeaturizer ──► policy(state_repr, action_reprs) → logits
                    (adapter)                  (the learner, F6 interface)
           └────────── implements ActionPolicy ──────────┘
```

**Decision — the adapter's contract:**

* it takes **state indices**, reads `graph.mask[state_ix]`, and is the **only** place atom ids appear;
* `state_repr` carries what architecture §3.1 names: pooled atom features over the selected set, the **`d(s)` obligation vector**, and `budget_left`. `d(s)` enters as a **feature, never an energy** (v1.2 §4.5.4);
* it **chunks internally**; "batched" means "not one call per state" (decision 8), not one call per instance;
* **no learner module may import `StateGraph`, `LatticeInstance`, or any atom id.** Enforced as an import test in the same style as `test_the_deterministic_core_imports_nothing_it_should_not`, not as a convention.

**`Δd` is a property of a transition, so it lives in `action_reprs` — and this is the whole L6/L7 difference.** R1 put `d(s)` in `state_repr` and stopped there, which is not enough: `Δd = d(s) − d(s′)` is indexed by *(state, action)*, and fix F11 defines L7 as *"`Δd` as input features to `φ_θ` and the policy"*. Placing it anywhere else either loses the per-action resolution or leaks it to every arm.

> `action_reprs(state_ix) -> Tensor` of shape `[n_states, n_atoms + 1, feat]`, whose per-action block carries `Δd` for the transition that action induces. The `+ 1` slot is `STOP`, whose `Δd` is zero by construction — `STOP` does not change the selected set — which is the same exclusion Phase-2 G5 applies when measuring `Δd` density.

**The gating is explicit and asserted, not conventional.** The featurizer takes a `delta_d: bool` flag, set per decision 19a: **True for L7 and L7b only**; **False for L1–L6 *and for GAFlowNet***, whose `Δd` reaches its loss and never its policy (G11). A test asserts that an L6 forward pass is numerically unchanged when the `Δd` block is replaced with zeros, and that an L7 pass is *not*. Without that test, "L7 is L6 plus `Δd` and nothing else" is a claim about source code rather than about the model, and a leak would make Gate 2 measure nothing — the two arms would differ by zero.

**[ANALYSIS]** Computing `Δd` for every legal action at every state is the one genuinely new cost this adapter adds. It is a lookup, not a recomputation: Phase 2 already evaluates `d(s)` per state for its audits, and the child of every legal action is in the edge arrays, so `Δd` per edge is a difference of two rows that are already materialised.

**One capability that must not be foreclosed.** **[EVIDENCE]** *When Do GFlowNets Learn the Right Distribution?* (ICLR 2025 Spotlight) proves that the representational limits of GNNs delineate which distributions a GFlowNet can approximate, and its remedy is **Look-Ahead GFlowNets**: feed children-state embeddings into the forward policy. Phase 2's exit criterion 18 tested that children remain reachable through the protocol. Phase 3 must keep that true through the adapter — an LA-style featuriser is a supported **Tier-2** variant, and the adapter's shape must not rule it out.

### G6 — One model per (learner, seed), conditioned on the instance — derivable but never stated [ANALYSIS]

Architecture §3.1 specifies a *"conditional partition function `logZ_θ(q)` head over the pooled instance embedding — conditional GFlowNet"*, **[EVIDENCE]** per *GFlowNet Foundations* (JMLR 2023). Nothing states whether that means one model across the suite or one model per instance, and the difference is a factor of twenty in run count.

It **is** derivable: Phase 2's exit criterion 11 computes the Gate-2 evaluation count as `7 learners × 3 seeds × C = 50 checkpoints × 20 instances = 21,000`. Instances multiply *evaluations*, not *runs*. So:

> **One model per (learner, seed), trained across all 20 main-suite instances, with `logZ_θ` conditioned on the instance. Exact TV is computed per instance and aggregated.**

Recorded here because a reader should not have to reverse-engineer it from a budget arithmetic, and because Phase 9 inherits the same shape.

**Aggregation must also be declared**, or "mean exact TV" is ambiguous. Decision: the **unweighted mean over the 20 instances**, with the per-instance spread reported. Weighting by terminal count would let the largest lattice dominate a number that is supposed to describe the method.

### G7 — Gate 2 measures fitting, not generalisation, and the write-up must say so [ANALYSIS]

If a learner trains on the main suite and is scored on the main suite, exact TV measures **how well the objective fits a known target on the instances it saw**. That is the correct question for a distribution-correctness gate, and it is standard in the GFlowNet training literature. It is **not** a generalisation result, and a reader will assume otherwise unless told.

There is a live reason to be careful: **[EVIDENCE]** *Generalization and Distributed Learning of GFlowNets* (ICLR 2025) gives data-dependent bounds that degrade with state-space size — the same paper `CLAUDE.md` §4.2 cites to justify bounded pools.

**Decision.** Train and score on the **main** suite; report the **probe** suite as a held-out environment at the end of the run, unfitted. The write-up states plainly that Gate 2 is a fitting result on the main suite plus a held-out reading on the probe suite — never "generalises".

The probe suite is the right held-out set for a second reason already declared: it is generated **without** the `Δd` band, so it answers "does the Gate-2 conclusion survive where L7's signal is sparse?" (Phase-2 G9). Its `Δd` density is measured and recorded per instance, so the claim can be stated *under a declared signal density* rather than unconditionally.

### G8 — Exploration and replay are free parameters that would confound the comparison [ANALYSIS]

The architecture says *"on-policy sampling with ε-uniform exploration; optional prioritized replay flag... default off for main runs, on only as the labeled ablation, so objective effects stay separable"*, **[EVIDENCE]** citing *Towards Understanding and Improving GFlowNet Training* (ICML 2023) for the replay design.

**ε is never given a value.** Plan §5.1's matched list — *"graph encoder and size, candidate pool, action space, terminal utility and checker, training examples, approximate reward/checker call budget, frozen SLM, random seeds"* — does not name it either, but the same logic applies with full force: an ε tuned per learner turns a comparison of objectives into a comparison of exploration schedules.

**Decision.** ε is a single frozen constant, **identical for every arm**, declared in `§6` and asserted at trainer start. Any schedule (decay) is part of that constant and is frozen with it. Replay stays **off** for every main run; the replay ablation is a separately labelled row and never enters the Gate-2 primary.

L3 (GRPO) is the exception that proves the rule: its `G = 8` group size is intrinsic to the objective, not an exploration knob, and is declared separately.

### G9 — Training on the lattice needs no checker calls, and metering it as if it did would be wrong [ANALYSIS]

Phase 2 precomputed everything training needs: `StateGraph.stop_allowed` gives validity for free, and `Target` caches `U` per terminal. **So a training rollout on the lattice spends zero `terminal_checks`.**

This matters in two directions.

*It must not be metered.* `checker_budget = 32` is a **per-query inference** budget (fix F5, `CLAUDE.md` §6). Charging training rollouts against it would exhaust it in one epoch and would be measuring the wrong thing entirely. The trainer runs with `ledger=None`, in the same way and for the same reason Phase 2's enumeration does (Phase-1 gap G9).

*It must be stated as a limitation.* §5.1 requires an *"approximate reward/checker call budget"* matched across rows. On the lattice that budget is zero for every row, so the requirement is trivially met **and carries no information**. The real test of it is Phase 9, on real data, where `U` is not cached and gold is absent. The Gate-2 write-up must say so; otherwise "matched checker budget" reads as a controlled variable when it was a constant.

### G10 — LED's terminal consistency is exactly measurable here, and both L6 and L7 must be held to it [ANALYSIS]

**[EVIDENCE]** *Learning Energy Decompositions for Partial Inference in GFlowNets* (ICLR 2024 Oral) decomposes the terminal energy into learnable transition potentials, regularised to (a) preserve the ground-truth terminal energy and (b) minimise trajectory variance, and shows training with the learned potential can preserve the optimal policy.

Plan §4.5.4 makes (a) a **hard constraint** on Contribution 3: *"If checker-derived features are allowed to modify the learned potential, that terminal-consistency regularizer must be retained unchanged. Dropping or weakening it... silently discards the property that makes LED-GFN correct. This must be verified on the enumerable environment, not assumed."*

The architecture's Phase-3 exit criterion 2 asks for it — *"L6's learned potentials satisfy terminal consistency within tolerance"* — **with no tolerance given**.

**The unit is the trajectory, not the terminal.** **[EVIDENCE]** LED-GFN decomposes *"terminal state energy into a sum of learnable [potentials] over the trajectory"*, and states the condition for each complete trajectory `τ = (s₀, s₁, …, s_T)`. Under the closure rule a terminal is reachable by many insertion orders, so **one terminal has many trajectories** and the condition must hold on each of them. R1 indexed the error by terminal, which silently averages over paths and would hide a potential that is consistent on average and wrong on every individual trajectory.

**"Exact" was also an overclaim.** `R(X)` is exact; the *pass* is sampled, so what is measured is an exact-`R` error on a sampled set of trajectories. R1 said the lattice makes this "exact rather than approximate", which is true only of the reward term.

**The denominator must be per instance, not per trajectory.** R2 divided by `|log R(X(τ))|`, which is unusable. Measured on the frozen main suite: `min |log R| = 0.0363` and `max = 7.8547`, so a 5% rule permits **0.0018** absolute error at one terminal and **0.3927** at another *of the same instance* — a 216× swing in effective tolerance, driven entirely by how close that terminal's utility happens to sit to zero. A statistic whose strictness depends on which terminal a trajectory landed on measures the sampler, not the decomposition.

**Decision.** Per sampled trajectory `τ` ending at terminal `X(τ)` on instance `i`:

```
range_i = max_{valid X of i} log R(X)  −  min_{valid X of i} log R(X)
err(τ)  = | Σ_{(s→s′) ∈ τ} φ_θ(s→s′)  +  log R(X(τ)) |  /  range_i
```

Measured: `range_i` spans **6.93–7.79** across the main suite — stable to about 12%, against the 216× swing of the per-terminal form. **Zero-range guard:** if `range_i < 0.1` the instance carries no reward spread for a decomposition to explain, so it is **excluded from the statistic and reported as excluded**, never silently divided by. The guard does not fire on the current suite; it exists so a future regeneration cannot produce a division by ~0 unnoticed.

Report **mean and p95** at the **final checkpoint** for **L6, L7 and L7b**. Requirement: **p95 ≤ 0.05** of the instance range.

**[ANALYSIS] The 0.05 is an engineering decision and is not paper-backed.** LED-GFN states the regulariser, not a numerical tolerance for it. At the measured ranges it permits about 0.35–0.39 in absolute log-reward, which is tight enough that a materially broken decomposition fails and loose enough to survive optimisation noise. If it proves miscalibrated, moving it is a decision-rule amendment under §6b made **before** L7 trains.

**The trajectory set is frozen and shared across arms.** The consistency condition is a property of `φ_θ`, not of the policy that produced the path, so all three arms are evaluated on **one common set per instance** — otherwise each arm visits different paths and its p95 is not comparable with the others'. Sampler, seed and counts are `§6` values (decision 14), not per-run choices.

**`FAIL` trajectories are audited separately, and their presence is guaranteed rather than hoped for.** `FAIL` carries `r_fail = 1e-6`, so `log R = −13.82` against a good proof near `+7.85`; pooling them would let the `FAIL` tail dominate a statistic meant to describe proof construction. Because `UniformPolicy` stops early and barely reaches dead ends — Phase 2 measured exactly this when choosing `ForcedContinuationPolicy` for its absorption audit — the `FAIL` sample is drawn under `ForcedContinuationPolicy` instead, and **the build asserts at least one `FAIL` trajectory per instance**. Zero is a defect to surface, not an empty line to average over.

**[ANALYSIS] The failure this catches** is specific and silent: L7 improving TV *because* its checker features quietly relax terminal consistency. That would not be Contribution 3 — it would be a different method with LED's guarantee removed, and nothing except this measurement distinguishes the two.

### G12 — The two claim-critical learners were named, not specified [ANALYSIS]

R4 specified GAFlowNet's *intrinsic reward* and LED's *tolerance* while leaving both **methods** underdetermined. Reading the papers closes it, and two of the gaps are structural rather than numerical.

**Both methods need a learned state flow, and no module held one.**

**[EVIDENCE]** LED-GFN's objective (its Eq. 4) is DB-style over a transition and reads

```
L_LED(s, s′) = ( log F̃(s) + log P_F(s′|s) + φ_θ(s→s′) − log F̃(s′) − log P_B(s|s′) )²
```

and Algorithm 1 initialises *"forward and backward policy `P_F`, `P_B`, state flow `F̃`, and model `φ_θ`"*. **[EVIDENCE]** GAFlowNet's augmented trajectory balance (its Eq. 4) reads

```
Z · Π_t P_F(s_{t+1}|s_t)  =  R(x) · Π_t [ P_B(s_t|s_{t+1}) + r(s_t→s_{t+1}) / F(s_{t+1}) ]
```

with `Z` the *augmented* total flow — so `F(s_{t+1})` appears in the correction term and cannot be dropped. Plain TB needs no state flow; **augmented TB does**.

**Decision.** `StateFlowHead` is added to P3.2 and shared by **L5 (SubTB), L6, L7, L7b and GAFlowNet**. It is checkpointed with the policy, and its parameters count toward capacity matching (decision 11) — omitting it would have made every matched pair wrong by the size of a head.

**LED's configuration is in the paper, not only in the repo.** R4's decision 23a assumed "two regulariser coefficients", which LED-GFN does not have: it has **one** squared loss with **dropout** as the variance regulariser (its Eq. 5),

```
L_LS(τ) = E_{z∼Bern(α)} [ ( (1/T)·E(s_T) − (1/C)·Σ_t z_t·φ_θ(s_t→s_{t+1}) )² ],   C = Σ_t z_t
```

and Appendix C reports the settings: potential network *"identical to that of GFlowNet policy"*, **potential learning rate 0.001**, **dropout probability 10%** *"for tasks with a trajectory length less than 10"* (ours is at most 8 transitions), **`N = 8`** decomposition iterations per round, and mini-batch `B2 = B1`. Those are decision 23a's values now, and they are read off the paper rather than resolved at a keyboard. *(Provenance nuance, recorded 13 Aug 2026: `N = 8` and `B2 = B1` are stated in Appendix C's bag-generation paragraph and inherited by RNA; molecule generation — whose decomposition settings set generation inherits — states only the no-buffer choice. Adopting them here is the nearest-stated-value reading, not a verbatim lookup for the set task.)*

**The correction variant is described but not adopted.** LED-GFN notes one *"can also introduce an approximation error `E(x) − φ_θ(τ)` as an additional correction term to preserve the optimal policy … even when the potential function `φ_θ` is inaccurate"*. **[ANALYSIS]** L6, L7 and L7b all use the **plain** form (Eq. 4) — identically. Enabling it for one arm would confound the comparison; enabling it for all three is scope this phase does not need, because criterion 16 measures decomposition quality directly instead of assuming it. Recorded so the choice is visible. *(The name "LED\*" for that variant does not appear in the paper text; it is not used here.)*

**[ANALYSIS] LED-DB, not LED-subTB.** The paper reports both. The architecture already specifies *"a DB-style objective"* for L6, and Eq. 4 is the DB form, so this follows the governing document rather than choosing freshly. L5 remains SubTB and is not L6's control — L7 is.

### G11 — GAFlowNet is a required control that R1 named without specifying [ANALYSIS]

Decision 1 promotes GAFlowNet to a trained arm. R1 stopped there, which is not a specification: four choices are free, and **one of them is a correctness condition rather than a hyperparameter.**

**[EVIDENCE]** *Generative Augmented Flow Networks* (ICLR 2023 Spotlight) offers **edge-based**, **state-based** and **joint** augmented flows, and its Theorem 1 reads: *"Suppose that ∀θ, L_GAFlowNet(θ) = 0, and ∀x, R(x) + r(x) > 0. When edge-based intrinsic [rewards go] to 0, then P(x) is an unbiased sample distribution."*

**The condition bites directly on Gate 2.** A permanent, non-decaying intrinsic reward derived from `Δd` leaves GAFlowNet targeting `R + r`, not `R`. Its exact TV would then be measured against a target it is not aiming at — which is precisely G2's error appearing in a new place, and it would make the C3-versus-GAFlowNet comparison unfair *in GAFlowNet's disfavour*, i.e. it would manufacture a win for the proposed method.

**`Δd` reaches the loss, and must *not* reach the policy.** This is the subtlest requirement in the arm. The featurizer flag of G5 controls what the **policy** sees; the intrinsic reward is a term in the **loss**. If GAFlowNet is built with `delta_d = True` it becomes *GAFlowNet plus a checker-conditioned policy* — which is L7's mechanism bolted onto the control, so the comparison would no longer isolate anything. Decision:

> **GAFlowNet's policy is featurised exactly as L6's (`delta_d = False`).** `Δd` enters **only** through the intrinsic reward term. A test asserts that its policy forward pass is unchanged when the `Δd` block is zeroed — the same assertion criterion 5 makes for L6, and for the same reason.

**Base objective, stated.** **[EVIDENCE]** The paper's edge-based augmentation is applied to trajectory balance — *"substituting the augmented trajectory balance loss"* — so the arm is **augmented TB**, making L4 its unaugmented counterpart. Naming this matters: an augmented-SubTB arm compared against a TB baseline would confound the augmentation with the objective.

**Decision — five things frozen before the arm is built:**

1. **Variant:** **edge-based**. It is the variant Theorem 1's vanishing condition is stated over, and `Δd` is natively a transition quantity (G5), so an edge-based reward is the faithful mapping rather than a pooled approximation.
2. **`Δd` → intrinsic reward:** `r(s→s′) = c_t · ‖max(Δd, 0)‖₁`, i.e. the **positive part** summed over the six components. Non-negative as Theorem 1 requires (`R + r > 0`), and positive-part because a *discharged* obligation is what the signal is about — a component going the wrong way is not evidence of progress and must not be paid for.
3. **Coefficient and decay:** `c_t = c₀ · (1 − t/N)`, linear to **exactly zero** at the trajectory budget `N`. Linear rather than exponential so that it *reaches* zero rather than approaching it — the theorem's condition is about the limit, and a schedule that never arrives leaves the bias in place at every checkpoint that is actually scored.
4. **`c₀`:** a `§6` value, identical across seeds, **not tuned against TV**. Tuning it on TV would select the intrinsic-reward strength at which the control performs worst.

**Reported, not assumed:** `c_t` at every checkpoint, so a reader can see the bias term vanishing, and the final-checkpoint value must be 0.

**`c_N = 0` is necessary and not sufficient, and R4 overclaimed by implying otherwise.** Theorem 1 assumes **two** things: the intrinsic rewards vanish *and* `L_GAFlowNet(θ) = 0`. A finite-budget run does not drive its loss to zero, so the trained arm is **not** a provably unbiased sampler of `p*`. Two changes, both needed:

* **A predeclared zero-intrinsic tail.** `c_t = c₀·max(0, 1 − t/(0.8N))` — decaying to exactly 0 at `0.8N` and held at 0 for the final 20% of the budget, so the last checkpoints train on the unaugmented objective. The final augmented-TB loss is reported.
* **The claim is downgraded in the write-up.** GAFlowNet is described as a **finite-budget empirical control**, not as an unbiased sampler. **[ANALYSIS]** The tail plus the reported loss let a reader judge how near the theorem's conditions the run got; asserting they were met would be the overreach `CLAUDE.md` §5 catalogues.

---

## 2. Scope

**In.** The `SyntheticFeaturizer` adapter and the frozen `policy(state_repr, action_reprs) → logits` interface, with `Δd` carried per action and gated per arm (G5); the trajectory sampler over `StateGraph`; the shared trainer with frozen exploration and budget metering; the **nine evaluated arms** of decision 1 — L1–L7, L7b and GAFlowNet — with capacity-matched controls; the Gate-2 harness, its hierarchical paired bootstrap and its report; the ML dependency boundary.

**Out.** Any search algorithm (Phase 4). Any real data. The answerability gate (Phase 8). The distilled utility head (Phase 9). Tier-2 and Tier-3 learners — Look-Ahead featurisation, OP-GFN, COFlowNet, FM, DB, FL-DB/FL-SubTB — which stay stubs, **with the single named exception of GAFlowNet** (§7). **No change to `H`, `U`, `R`, the masks, `d(s)`, the lattice generator or the exact evaluator.** Phase 3 consumes Phase 2's instrument; it does not adjust it. If a Gate-2 result seems to require adjusting the instrument, that is `GRAFT_PHASE2_BUILD.md` §6b's amendment procedure and its contamination rule, not an implementation choice.

---

## 3. Modules

### P3.0 The Phase-0 structural amendment (do this first)

`graft/tests/test_structure.py::test_phase0_imports_no_ml_library` imports every module under `graft/` and asserts no ML library is in `sys.modules`. **Phase 3 needs torch, so that test fails the moment the first learner is written.**

The amendment is not "delete the test" — the test is what keeps Phases 0–2 runnable on a bare interpreter and inside a Kaggle notebook before any heavy install, which `README.md` promises. Instead:

* `requirements-ml.txt` is added (torch pinned), separate from `requirements.txt`, as `README.md` already anticipates;
* the structural test is **narrowed, not weakened**: `graft.core`, `graft.synth`, `graft.config`, `graft.schemas` and the Phase-0 scaffold must still import no ML library; `graft.setgen` may. The narrowed rule is asserted per-package rather than globally, so a torch import leaking back into `graft.core` still fails the build;
* Phase 2's exit criterion 21 (`graft.synth` imports no ML library) is **unchanged and must keep passing** — it is what keeps the exact evaluator usable without a GPU stack.

**[ANALYSIS]** Recorded as a module rather than a footnote because it is a deliberate relaxation of a Phase-0 exit criterion, and the honest version narrows the guard rather than removing it.

### P3.1 `graft/setgen/features.py` — the adapter (G5)

**Surface.** `SyntheticFeaturizer(instance, graph, cfg)` implementing `ActionPolicy` · `state_repr(state_ix) -> Tensor` · `action_reprs(state_ix) -> Tensor`.

The only module in `setgen/` that may touch `StateGraph` or atom ids. Everything above it sees tensors.

### P3.2 `graft/setgen/policy.py` — the frozen learner-facing interface (fix F6)

**Surface.** `Policy(nn.Module)` with `action_logits(state_repr, action_reprs) -> Tensor` (ADD-per-atom ∪ STOP, masked **before** softmax) · `LogZHead` conditioned on the pooled instance embedding (G6) · **`StateFlowHead` returning `log F(s)`**, required by L5, L6, L7, L7b and GAFlowNet (G12) · `PotentialHead` for `φ_θ(s→s′)`, architecture identical to the policy per LED Appendix C · `capacity(module) -> int` for G4, **counting every head**.

Masking before the softmax, not after, so that illegal actions receive exactly zero probability rather than a small one — the same property Phase 2's evaluator asserts of its reference policies.

### P3.3 `graft/setgen/rollout.py` — trajectories over the state graph

**Surface.** `sample_trajectories(policy, graph, n, rng) -> Trajectories` (states, actions, terminal, `is_fail`) · `uniform_backward` re-exported from Phase 2 (decision: **do not reimplement it**; Phase 2's version reads the enumerated in-edges, which *are* the removable atoms).

**The correctness hinge of the phase.** The sampler and Phase 2's exact DP must agree: sampled terminal frequencies from a fixed policy must converge to `policy_distribution(policy, graph)`. If they disagree, every downstream TV is meaningless, and the disagreement will not be visible in any loss curve.

### P3.4 `graft/setgen/learners/` — one file per arm

`l1_supervised.py` · `l2_imitation.py` · `l3_grpo.py` · `l4_tb.py` · `l5_subtb.py` · `l6_led.py` · `l7_checker_led.py` · `l7b_aux.py` · `gaflownet.py` (decision 1). Each exposes `loss(batch, policy, ...) -> Tensor` and nothing else; none imports `StateGraph`.

**L7b, fully specified** (fix F11 demoted it to an ablation but never said what it optimises): the auxiliary head predicts, for each visited state `s` on a trajectory `τ`, the **deficit vector of the terminal actually reached**, `d(X(τ)) ∈ [0,1]^6` — forward-looking, and genuinely not in the input, unlike `d(s)`. Loss is mean-squared error over the six components, added to L7's loss with weight `λ_aux` (decision 26). The head is discarded at evaluation: it shapes training and never enters the sampling distribution.

L7 is **L6 plus `Δd` as input features to `φ_θ` and the policy, and nothing else** (fix F11). L7b is an ablation arm with a **forward-looking** target — the terminal deficit reachable from `s`, supervised by the realised rollout — never a head predicting `d(s)`, which is already an encoder input and therefore a copy task.

### P3.5 `graft/setgen/trainer.py` — the shared trainer

One trainer, every arm. Owns the frozen ε (G8), the trajectory budget `N` and its checkpoint schedule (G3), the capacity assertion (G4), seeding over `{13, 42, 7}`, and checkpoint persistence.

**It also owns the optimisation protocol, which R2 left entirely unspecified.** Optimiser, learning rate, batch size, gradient clipping and LED's regulariser coefficients are all free parameters, and per-arm tuning of any of them turns a comparison of objectives into a comparison of tuning effort — the same failure §5.1's matched list guards against for the reward. **One protocol, frozen in `§6` (decision 23), identical for every arm, with no per-arm search.**

**[ANALYSIS] Why a shared protocol rather than per-arm tuning is the right call even though it disadvantages someone.** A per-arm sweep is the fairer-sounding option and the wrong one here: with nine arms, three seeds and a fixed wall-clock ceiling, any sweep large enough to be fair would consume the budget the comparison itself needs, and a sweep small enough to afford would favour whichever arm happened to suit the grid. A single frozen protocol is *uniformly* suboptimal, which is the property that makes the comparison interpretable. It must be stated as a limitation in the write-up: **these are not tuned results, and no arm's number should be read as its best achievable.**

**Gotcha.** The trainer runs with `ledger=None` (G9). A trainer that opens a `query_scope` and meters `terminal_checks` will exhaust `checker_budget` in the first epoch and will be measuring the inference budget with training rollouts.

### P3.55 `graft/setgen/flgfn_probe.py` — the FL-GFN discharge (decision 1)

**Decision 1 promises a measurement and R2 provided nowhere to produce it.** This module is that place, and it needs no learner, no policy and no training — only Phase 2's `Target`, which already holds exact `U` per valid terminal.

**Surface.** `deficit_potential(instance, graph, cfg) -> np.ndarray` · `terminal_identity_residual(instance, graph, target) -> dict`.

**The free coefficients are not defined anywhere, so the probe fits them rather than guessing.** Plan §4.5.2's candidate is `Φ(s) = −Σ_j ω_j d_j(s) − λ_v|V_s| − λ_e|E_s|`, and neither this plan nor any governing document fixes `ω_j`, `λ_v` or `λ_e`. Picking values would make the result an artefact of that pick — a reader could always answer "you chose the wrong weights".

**The fit must also absorb the energy offset, or it measures the wrong thing.** An energy is defined only up to an additive constant *per instance*: `R = exp(−ℰ)` is normalised by `Z` on each instance, so a uniform shift of `Φ` on one instance changes no distribution. Fitting without that freedom would charge the family for a constant that carries no information, and would report a failure that is an artefact of the parameterisation.

**Decision — report the irreducible residual of the whole family, with a per-instance constant.** Solve

```
min over (ω, λ_v, λ_e, {C_i})   Σ_{i}  Σ_{valid X of i}  ( Φ(X) − β·U(X) − C_i )²
```

by least squares over **every valid terminal of the main suite** — exhaustive, since the terminal set is enumerated and small. Report the residual at the optimum: mean, p95, max, `R²`, and the count of terminals off by more than float tolerance. Two **named special cases** are reported alongside: `uniform_omega` (every obligation weighted equally, size terms off) and `deficits_only` (size terms pinned off, deficits free). **Not** `ω = u_weights`, which R5 named and which is not definable: the deficit components (anchor, value, time, source, binding, closure) and the utility weights (suff, cov, src, temp, red, size) are disjoint vocabularies of the same length — `sufficiency` and `redundancy` have no deficit component, `binding` and `closure` have no utility term. Reporting it would have meant inventing a correspondence and then reporting the number that invention produced.

**[ANALYSIS] Why the best fit is the right thing to report.** If even the *best-fitting* member of the family — with every coefficient and every per-instance offset free — cannot satisfy terminal identity, the failure is a property of the family, not of a weight choice. That is materially stronger than any single instantiation could support, and it is what plan §4.5.2's argument actually needs. `Φ` is linear in all of `(ω, λ_v, λ_e, C_i)`, so this stays a closed-form least-squares solve.

**β ordering.** `Φ(X) = β·U(X)` depends on β, and step 2 runs before β is frozen. The probe is therefore run **at every eligible β** at step 2 — cheap, since it is one linear solve over an enumerated terminal set — and the **reported** figure is the one at the β adopted at step 6. Reporting the spread across eligible β is itself informative: if the residual is large at every candidate, the conclusion does not depend on the β that happened to win.

**What the output licenses**, and the wording is fixed here so it cannot drift: it disproves *this potential*, not FL-GFN (G1). The paired limitation statement — no justified informative scalar extension of `sufficiency` to partial states is currently available — is **argued, not measured**, and is reported as such.

**Runs early** (build step 2), because it depends on nothing this phase builds and its result is one of the few things Phase 3 can deliver whatever happens at Gate 2.

### P3.6 `graft/setgen/gate2.py` — the harness

**Surface.** `run_matrix(spec) -> Gate2Report` · `paired_bootstrap(a, b, seeds) -> dict` · `report(...)`.

Consumes Phase 2 unchanged: `target_distribution`, `policy_distribution`, `divergence_report` (which already binds TV, JS, KL-when-finite, FCS with its standard error, and `p*(FAIL)` **beside** TV rather than subtracted from it).

**Every Gate-2 table carries**, per Phase 2's handoff item 7: the environment's `Δd` density (structural **and** visitation-weighted), the target-mass profile, and — added by the Phase-2 β amendment — the `neither`-mass at the β the run used.

### P3.7 `graft/tests/`

See §5.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 1 | P3.0 ML boundary | `graft.core` and `graft.synth` still import no ML library; `graft.setgen` may; Phase-2 criterion 21 still green |
| 2 | P3.55 FL-GFN discharge **and** P3.1 + P3.2 adapter, policy and heads | the residual of decision 24 is measured **at every eligible β** and recorded;  a policy emitting constant logits reproduces `UniformPolicy`'s distribution to ≤ 1e-12 per terminal; the `Δd` gating test passes (G5) |
| 3 | P3.3 rollout sampler | sampled terminal frequencies match `policy_distribution` within the Phase-2 G8 sampling floor, on `tiny_instance()` |
| 4 | P3.5 trainer skeleton | ε, capacity and `ledger=None` assertions fire; checkpointing round-trips |
| 5 | **L4 + L5 only** | both train end to end on `tiny_instance()` and reach the **decision-6** TV threshold — the machinery works before anything is claimed |
| **6** | **CALIBRATION GATE — only L4 and L5 exist yet, and the whole gate runs on the *tuning* suite** | (a) **β eligibility** recomputed (Phase-2 decision 19) — which candidates are admissible at all; then per rung `i` of decision 4's ladder: (b) **`Nᵢ`** from ceiling `cᵢ`; (c) **β sweep at `Nᵢ`**, freeze `βᵢ` (Phase-2 decision 22); (d) **sanity check** at `(Nᵢ, βᵢ)` (decision 6). Pass ⇒ adopt; fail ⇒ next rung; fail at the last ⇒ **Gate 2 inconclusive, phase stops**. Every rung recorded. **Nothing proceeds until this is written into `§6`.** |
| 7 | L6 + per-trajectory consistency measurement (G10) | p95 normalised error ≤ 0.05 |
| 8 | L7, capacity-matched L6, GAFlowNet (G11) | capacity assertion passes; `c_t` reaches 0 at `N`; all three train to completion at `N` |
| 9 | L7b, then L1, L2, L3 | L7b reported as an ablation; L1–L3 in the separate table of G2 |
| 10 | P3.6 Gate-2 harness + hierarchical paired bootstrap + report | the full matrix runs inside the 27,000-evaluation budget of decision 2 |

**Steps 2–5 are the correctness spine and must be finished before any learner is compared.** The lesson Phase 2 recorded: sampler-versus-DP agreement is exactly the kind of error that produces plausible curves and meaningless numbers.

**Why the calibration gate is step 6 and not step 2.** R1 put it second, which was **impossible** — it called for an L5 throughput pilot before the adapter, the policy, the sampler, the trainer or L5 existed. The gate needs a working L5, so it lands after step 5. It must still land **before** step 7, because β and `N` have to be frozen before L6, L7 or GAFlowNet trains once or the §6b amendment procedures are contaminated by learner results.

**Why (b) precedes (c).** Phase-2 decision 22 fixes the β winner as the candidate minimising mean exact TV *"at the same fixed trajectory budget the Gate-2 primary comparison uses"* — **the sweep needs `N`**. R2 froze β first, which made that budget undefined at the moment it was required. There is no circularity in the corrected order: `N` is a wall-clock quantity and rollout cost does not depend on β, so `N` can be fixed at the current default β and the sweep then run at that `N`.

**What is signed now versus later.** The calibration *procedure*, the *ceiling* (decision 5) and the *sanity threshold* (decision 6) are all signed off **now**, before any learner runs. Only the numerical `N` and the frozen β are written at the end of step 6. A value produced by a procedure fixed in advance is not a decision made after seeing results; a threshold written afterwards would be.

---

## 5. Exit criteria

**The machinery is correct**
1. A constant-logit policy through the adapter reproduces `UniformPolicy`'s terminal distribution to **≤ 1e-12 per terminal in float64** — the adapter introduces no distortion (G5, fix F6). **The dtype is part of the criterion.** Training runs in float32, which carries ~1e-7 of relative precision, so 1e-12 is unreachable there and asserting it would be asserting something false. Criterion 1 asks whether the *adapter* distorts the distribution; entangling that with the dtype the learners happen to use would answer a different question. The float32 agreement (≤ 1e-6, measured) is reported separately as a property of the dtype — and is orders of magnitude below any Gate-2 TV difference, which is the argument that it does not threaten the comparison.
2. Illegal actions receive exactly zero probability after masking, on every state of `tiny_instance()` — masking **before** the softmax, per P3.2, so an illegal action cannot carry a small positive mass into the DP.
3. Sampled terminal frequencies from a fixed policy match `policy_distribution` on `tiny_instance()` within the sampling floor of Phase-2 G8 at 200,000 rollouts, with a per-terminal z-test at 5σ.
4. `logZ_θ` at convergence on `tiny_instance()` matches `log Z` from `Target` within **decision 25**'s tolerance — the one place the conditional partition function has a known answer (G6).
5. **The `Δd` gate holds in both directions** (G5): an L6 forward pass is numerically unchanged when the `Δd` block is zeroed, and an L7 pass is changed. One direction alone proves nothing — without the first, `Δd` may be leaking to the control; without the second, L7 may be ignoring it, and either way Gate 2 compares two arms that differ by zero.
6. No module in `graft/setgen/learners/` imports `StateGraph`, `LatticeInstance`, or an atom id (fix F6, asserted by an import test).
7. Phase-2 exit criterion 21 still passes: `graft.synth` imports no ML library.

**The comparison is fair**
8. Capacity: every matched pair is matched on **live** trainable parameters per **decision 11** — the control never smaller, at the narrowest admissible width — and the achieved counts, the dead-parameter counts and the achieved excess all appear in the report (G4).
9. ε and its schedule, the replay flag, `u_weights`, β, `r_fail`, `K`, `checker_budget`, the pool and the action space are **identical across every arm**, asserted at trainer start (G8, plan §5.1).
10. Seeds are exactly `{13, 42, 7}` for every arm — the frozen set of `CLAUDE.md` §6 (**[ANALYSIS]** — the multi-seed rule is the project's own protocol discipline; its ACL 2018 test-selection authority does not prescribe seed counts) — **and a run that is not a Gate-2 run returns no verdict**: `contribution_3_supported` is null with every failed clause named, never a boolean. Admissibility is the conjunction of everything that defines the experiment, because every one of them is a `run_matrix` parameter with a default: the full **decision-1 roster**; exactly those **seeds**; the frozen **20-instance main suite** by instance identity (decision 8, criterion 25); the **probe suite** present (criterion 23); and **`N` and β taken from an adopted, non-`--quick` calibration record** and matching it (decision 4, Phase-2 decision 22). R5's harness accepted any roster and any seed subset; R6's first pass checked only those two, so a run on one tiny instance at an uncalibrated `N` still returned a boolean. "The config refuses to shorten" described an intention rather than a mechanism.
11. The trajectory budget `N` is identical across every arm, was produced by **decision 4's calibration ladder on the tuning suite**, and is recorded in `§6` before any L6/L7/GAFlowNet run (G3). Every rung tried is recorded, not only the one used.
12. **The decision-6 sanity threshold is evaluated on the tuning suite at each rung of decision 4's ladder.** If it still fails at the final rung, Gate 2 is recorded **inconclusive**, never as a negative result for Contribution 3 (G3). The write-up states that `N` was set by a bounded adaptive rule reading L4/L5 only — not by throughput alone.
13. Training spends zero `terminal_checks`; the trainer runs with `ledger=None` (G9).
14. GAFlowNet's intrinsic-reward coefficient `c_t` is reported at every checkpoint and is **exactly 0 at the final checkpoint** — the condition Theorem 1's unbiasedness rests on (G11). Without it the control is not sampling `p*` and its TV is measured against a target it was not aiming at.

**Gate 2**
15. At the adopted `(N, β)`, L4 and L5 reach exact TV below decision 6's level **on the main suite** — the architecture's "the machinery works" criterion. Distinct from criterion 12, which is the *tuning-suite* calibration check that selected `(N, β)` in the first place.
16. **Terminal consistency, per trajectory**, measured and normalised exactly as **decision 13** defines, on the common frozen trajectory set of **decision 14**, for L6, L7 and L7b — measured on the **raw** `φ_θ` (decision 23b: on the redistributed `φ̃` it is identically 0), with `FAIL` audited on its own line, never pooled, and at least one `FAIL` trajectory per instance — **which blocks the verdict when it is missing**, rather than being recorded and ignored as in R5 (G10).
17. **L7 versus capacity-matched L6 on the predeclared primary** — exact TV at fixed budget `N`, three seeds, hierarchical paired bootstrap (fix F12, decision 20). Secondaries: trajectories-to-threshold, and best-of-K valid-set utility at `checker_budget` — over **fix F5's portfolio, 1 greedy + 7 sampled**, not eight stochastic draws, since the greedy candidate is the one a reward-maximiser returns and Phase 4's S5 is compared against the same object.
18. **L7 versus capacity-matched GAFlowNet** on the same primary — plan §4.5.4's required control (decision 1).
19. **Decision 15** holds: L6 and L7 both pass decision 13's band **and** L7 exceeds L6 by no more than the declared non-inferiority margin. This is the hard constraint of plan §4.5.4 — a TV win must not be bought by relaxing LED's regulariser — expressed so that third-decimal noise cannot decide it (G10).
20. L7b is reported alongside as an ablation, never as a substitute for L7's result (fix F11).
21. L1, L2, L3 reported in a **separate table** on metrics that fit them, with TV descriptive and labelled as such (G2).
22. Audits carried into every table, per Phase-2's handoff item 7 and the architecture's Phase-3 exit criterion 5: `FAIL` rate, **equivalent-action collision rate (0)**, **unconstructible-valid-terminal rate (0)**, `p*(FAIL)` beside TV and never subtracted, the environment's structural and visitation-weighted `Δd` densities, the target-mass profile, and the `neither`-mass at the run's β.
23. The probe suite is evaluated once, at the end, as a held-out environment, and the write-up states the Gate-2 result **under its declared `Δd` density** (G7). **The probe carries its own audit block** — its structural and visitation-weighted `Δd` densities, target-mass profile, β and fingerprints — not only its TV: §6b's density row makes the conclusion conditional on the signal density it was measured under, and "both or neither" is unreadable if only one of the two suites reports one.

**Budget and reproducibility**
24. The whole matrix completes inside decision 2's evaluation budget. Phase 2 measured ~0.0002 s for the DP and ~0.049 s for FCS. **The per-evaluation cost is the DP**: FCS runs once per (arm, seed) at final θ — 27 `divergence_report` calls, 540 FCS computations, ~26 s across the matrix — because it is a correctness proxy with a standard error and its use is one number per run with a spread across seeds, not a curve. An earlier version of this criterion called FCS "the binding term", which only holds if it runs at every evaluation; it does not, and decision 2's arithmetic is written for what the code does.
25. Every run records the `environment_fingerprint` and `target_fingerprint` of the suite it used (Phase-2 decision 21), so no result computed against one environment can be silently compared with another.

**The decision**
26. A written decision, before results are inspected: **Contribution 3 is supported only if L7 beats capacity-matched L6 *and* capacity-matched GAFlowNet, both on the primary under the paired test.** Beating L6 alone is not sufficient — plan §4.5.4 makes GAFlowNet a required control precisely because "an intermediate signal helps" is already published. If either comparison fails, **C3 is not supported and the thesis consolidates on Contribution 1** (v1.2 §9 fallback). The sentence is in the plan now, not written after the numbers are known.
27. **Collision rate and unconstructible-valid-terminal rate are reported, both expected 0** — the architecture's Phase-3 exit criterion 5 requires them alongside the `FAIL` rate, and Phase 2 supplies both from `graft.synth.audits` at zero marginal cost.
28. **Gate-2 item 3 is discharged only in part, and the report says so.** The plan's item 3 names Tier 1 *and* Tier 2 (FM · DB · FL-DB · FL-SubTB · GAFlowNet). Phase 3 runs Tier 1 plus GAFlowNet; **FM, DB, FL-DB and FL-SubTB remain deferred**, with FL's row discharged by the measurement of decision 24 rather than by training. Recording it as a **deliberate partial discharge** is the honest form; claiming item 3 closed would not be.

---

## 6. Decisions to lock before writing code — **FIVE RULED; THE REST AWAIT SIGN-OFF**

**Ruled: 1** (11 Aug 2026, the Tier-1 roster) and **6, 11, 29, 30** (12 Aug 2026 —
the L4/L5 sanity threshold, capacity matching, GRPO's group size and `logZ_θ`'s
learning rate; reasoning in `PHASE3_DECISIONS.md` §6.8b). Every other row is
`[recommended]` or `[fill at step 6]`, and the build adopted the recommended ones
as written and recorded that it did.

**This table is normative.** Where a restatement elsewhere disagrees with it, the table wins and the restatement is a bug. Restatements point at a decision number rather than repeating its value — the mitigation Phase 2 adopted after five review rounds in which most defects were a fix landing in three places out of four.

Values marked **[fill at step 6]** are produced by the calibration gate, by a procedure that is signed off now. Values marked **[recommended]** are engineering choices awaiting a ruling; none is paper-backed, and each says so.

| # | Decision | Value | Cost if changed later |
|---|---|---|---|
| 1 | **Tier-1 roster** | **RULED: Option B** — L1–L7, L7b and **GAFlowNet**; the FL-GFN row discharged by the two-part report of G1, which disproves *the proposed deficit potential* and states the availability limitation — **never** "FL-GFN is measurably inapplicable" | A required control of plan §4.5.4 goes unrun; or the retired overreach of plan §4.5.2 returns |
| 2 | **Evaluation budget** | 9 evaluated arms × 3 seeds × **≤ 52** evaluations × 20 instances = **≤ 28,080**; ≤ 1 h total ⇒ **≤ 0.128 s** per evaluation. **52, not 50**: the trainer evaluates once at trajectory 0 — the random-initialisation baseline — then at each of decision 7's `C = 50` checkpoints, and once more at the end whenever the last checkpoint did not land exactly on `N`. `budget_for` rounds `N` to a multiple of the batch, not of `N // C`, so the trailing read fires in practice (verified: `N = 4,992, C = 50 → 52`; `N = 4,800 → 51`). `C` is unchanged; the count of *evaluations* is what the budget is spent on. The FCS proxy runs **once per (arm, seed)** at final θ, not per checkpoint — it has a standard error and one seed of three supplies no spread — which is **27 `divergence_report` calls over 20 instances each, so 540 FCS computations, ~26 s across the matrix**. An earlier version of this cell said "27 calls, ~26 s", which is internally inconsistent at ~0.049 s each; the time was right and the call count was not. The per-evaluation cost is the DP | a budget quoted for seven arms while nine run; or an evaluation count that is off by the baseline and trailing reads |
| 3 | **What counts as one trajectory** | one root-to-terminal rollout, `FAIL` included | a learner that dead-ends often buys extra gradient steps |
| 4 | **Trajectory budget `N`** | **[fill at step 6]** — decision 4's ladder (G3), **entirely on the tuning suite**: per rung, `Nᵢ` from ceiling `cᵢ` → **β sweep at `Nᵢ`** → freeze `βᵢ` → sanity check; at most two doublings, then **inconclusive**. **Every *learned* quantity reads L4/L5 only** — β selection and the decision-6 threshold — **never L6/L7/L7b/GAFlowNet.** **`Nᵢ` is sized from the *slowest* arm's measured throughput, which is not a learned quantity** (R6): a rate in trajectories per second is a property of the machine and the architecture, available before a gradient step means anything and unable to move in L7's favour. Sizing on L5 alone put `N` on the cheapest arm and spent it on the most expensive — measured, L5 ~2,700 traj/s against the LED arms' ~900, so an `N` fitting L5 in 1 h bought L7 2.9 h, and the rung's own guard reads only `beta_sweep` and `sanity_check`, which run L4 and L5 exclusively. **Recorded in the write-up as adaptive, not throughput-only** | F12 stays unfalsifiable; `N` chosen where the proposed method looks best; β frozen at a budget the comparison does not use; or a ceiling that six of the nine arms break while the ladder certifies it |
| 5 | **Wall-clock ceiling `c₀` and the ladder's cost** | **[recommended]** `c₀ = 1 h` per run, rungs **1 h → 2 h → 4 h**. Costed below; **[ANALYSIS]** an affordability choice, not a claim that 1 h suffices — insufficiency is what decision 6 detects and the ladder answers. **The ceiling is enforced on adoption, measured per run**: a rung whose *slowest single run* exceeds it is never adopted and ends the ladder (verdict `over_ceiling`), because an overrun says the throughput estimate that produced `N` was wrong and the rung did not spend the budget it reports — and escalating would only raise the ceiling further. It is **not** passed to the trainer as a hard stop: truncation spends fewer than `N` trajectories and criterion 11 requires `N` identical across arms (R6) | the ladder's top rung silently exceeds the phase, discovered at the last rung; or an unfunded `N` written into this table because the mean of the rung's runs cleared a bar one of them broke |
| 6 | **RULED, 12 Aug 2026 — unchanged at 0.10; §2.3's option 1.** **L4/L5 sanity threshold** | **[signed *before* calibration]** mean exact TV ≤ 0.10 on the **tuning** suite for both L4 and L5. Drives decision 4's escalation; failure at the last rung ⇒ Gate 2 **inconclusive** (G3). **[ANALYSIS]** engineering, not paper-backed | a compute limitation reported as a scientific conclusion about C3; or the primary budget selected on scored instances |
| 7 | **Checkpoints** | `C = 50`, inherited frozen from Phase-2 criterion 11 | the evaluation budget is meaningless without it |
| 8 | **Run shape** | one model per (arm, seed) across all 20 main-suite instances, `logZ_θ` conditioned on the instance; **unweighted** mean TV over instances, spread reported (G6) | 20× the runs, or a mean the largest lattice dominates |
| 9 | **Train/eval split** | train and score on **main**; **probe** held out, read once at the end; the write-up says "fitting", never "generalises" (G7) | a fitting result presented as generalisation |
| 10 | **Exploration** | **[recommended]** ε = 0.05, **constant, no decay**, identical for every arm; replay **off** for all main runs (G8). **[ANALYSIS]** the constancy matters more than the value | the comparison measures exploration tuning |
| 11 | **RULED, 12 Aug 2026.** **Capacity matching** | **live** trainable params — nominal minus the weights an arm's zeroed `Δd` block leaves ungradiented (`Trainer.dead_capacity_of`); control **never smaller**; control widened to the **narrowest admissible width**; achieved counts, dead counts and the achieved excess all printed (G4). **The 1% is retired as unachievable, not missed** (R6): the dead block is `12·hidden` and one width step is ~`2.4·hidden`, so the block is ~half a step at hidden 64, 128 and 256 alike, and the smallest width that closes it overshoots by ~1.4–2.2% at every scale. Removing L7's extra parameters instead is not available — they *are* the mechanism under test. Measured (re-measured 13 Aug 2026 with the post-§6.10 `dead_capacity_of`, which counts `LogZHead`'s pad in every arm): L6 at hidden 65 (+1.44% live), GAFlowNet at hidden 83 (+1.90% live; +2.17% before the pad was counted) | an uninterpretable Contribution-3 result; or a control that is smaller in trainable capacity while reporting a 0.00% match |
| 12 | **Metric routing** | exact TV primary for the flow family only; L1/L2/L3 in a separate table with TV descriptive (G2) | one table implying three methods failed at a task two never attempted |
| 13 | **Terminal consistency** | **per trajectory**, normalised by the **per-instance valid-terminal `log R` range** with a `range < 0.1` exclusion guard; final checkpoint; mean and **p95 ≤ 0.05**; `FAIL` audited separately (G10). **[ANALYSIS]** the 0.05 is engineering, not paper-backed | per-terminal normalisation swings the effective tolerance 216x on the measured suite |
| 14 | **Consistency trajectory set** | **[recommended]** one **common, frozen** set per instance, shared by L6/L7/L7b: 2,000 under `UniformPolicy` seed `20260815`, plus 500 under `ForcedContinuationPolicy` seed `20260816` for the `FAIL` line, asserting >= 1 `FAIL` trajectory (G10). **[ANALYSIS]** counts are engineering; the *sharing* and the seeds are what make the three p95 values comparable | per-arm sampling makes three p95 values that are not comparable |
| 15 | **The L7 hard constraint** | **both** L6 and L7 must pass the decision-13 band, **and** `p95(L7) - p95(L6) <= 0.01` — a **non-inferiority margin**, not an ordering. **[ANALYSIS]** plan §4.5.4 requires the regulariser retained and consistency acceptable, not that L7 be numerically tidier; 0.030 vs 0.031 is noise | a TV win bought by weakening LED's regulariser; or a pass/fail decided by third-decimal noise |
| 16 | **Training meters nothing** | trainer runs `ledger=None`; `checker_budget` is inference-only (G9) | `checker_budget` exhausted in epoch 1, measuring the wrong axis |
| 17 | **ML boundary** | `requirements-ml.txt`; the structural test narrowed per package, not deleted (P3.0) | Phases 0–2 stop running on a bare interpreter |
| 18 | **Backward policy** | uniform over **removable** atoms, re-exported from Phase 2, never reimplemented | mass on parents that do not exist under the closure rule |
| 19 | **GAFlowNet specification** | base objective **augmented TB** (so L4 is its unaugmented counterpart); edge-based; `r(s→s′) = c_t·‖max(Δd, 0)‖₁`; `c_t = c₀·(1 − t/N)` reaching **exactly 0** at `N`; `c₀` **[recommended]** 0.1, never tuned against TV; **policy featurised as L6's (`delta_d = False`)** — `Δd` reaches the loss, never the policy (G11). **The objective is Eq. 4 as written**: `Z Π P_F = R(x) Π (P_B + r/F(s′))`, so the correction to TB's residual is `Σ log(1 + r/(F·P_B))` and **the `P_B` divisor is load-bearing** — dropping it (R5's build did) implements an intrinsic reward of `P_B·r`, attenuated by the removable-atom count and increasingly so with set size, which is not the declared `r` and weakens a required control (R6). **[ANALYSIS]** `c₀` is engineering; the *decay to exactly zero* and the placement of `r/F` beside `P_B` are the paper's, not choices | the control targets `R + r` rather than `R`; it silently becomes GAFlowNet **plus** L7's mechanism; or it is not GAFlowNet at all and plan §4.5.4's required control goes unrun |
| 19b | **GAFlowNet unbiasedness claim** | zero-intrinsic **tail**: `c_t = c₀·max(0, 1 − t/(0.8N))`, exactly 0 for the final 20% of `N`; final augmented-TB loss reported; the write-up calls it a **finite-budget empirical control**, never a provably unbiased sampler (G11) | Theorem 1's second assumption (`L = 0`) asserted rather than approached, which no finite run establishes |
| 19a | **`Δd` routing, per arm** | policy-visible (`delta_d = True`): **L7, L7b only**. Loss-only: **GAFlowNet** (intrinsic reward, policy featurised as L6's). Neither: **L1–L6**. Asserted by the zeroing test of criterion 5 for every arm (G5, G11) | GAFlowNet silently becomes GAFlowNet **plus** L7's mechanism, and the required control isolates nothing |
| 20 | **"L7 beats L6", executable** | **hierarchical paired bootstrap** over seeds (outer) and instances (inner), 10,000 resamples, seed `20260814`; L7 wins iff the **one-sided 95% upper bound on `TV_L7 − TV_control` is below 0**. Applied identically against L6 and against GAFlowNet | an unfalsifiable decision rule, which is what fix F12 exists to retire |
| 21 | **Trajectories-to-threshold** | the secondary is trajectories to reach the decision-6 TV level; arms that never reach it are reported as censored, never as the budget | a secondary silently favouring whichever arm was measured last |
| 22 | **Gate-2 scope** | Phase 3 discharges Gate-2 items 3 and 4; item 5 is Phase 8 (G2) | Phase 3 claiming to close a gate it only partly covers |
| 23 | **Optimisation protocol** | **[recommended]** one frozen protocol for every arm, no per-arm search: Adam, lr 3e-4, batch 32 trajectories, grad-norm clip 1.0. **[ANALYSIS]** engineering, not paper-backed; uniformly suboptimal *is* the point | a comparison of tuning effort rather than of objectives |
| 23a | **LED configuration** | **[EVIDENCE]** from LED-GFN Appendix C, adopted verbatim and **identical for L6, L7 and L7b**: potential network architecture identical to the policy; **potential learning rate 0.001**; **dropout probability 0.10** (its rule for trajectory length < 10; ours is at most 9 transitions); **`N = 8`** decomposition iterations per round; mini-batch `B2 = B1` (**provenance nuance, 13 Aug 2026**: `N` and `B2` are stated in the bag-generation paragraph and inherited by RNA — molecule generation, whose settings the set task inherits, states only the no-buffer choice, so for the set task these two are the nearest-stated-value reading rather than a verbatim lookup); **no replay buffer** — Appendix C states molecule generation uses the round's own samples and set generation inherits its decomposition settings, and set generation is the closest published task to this one, so Algorithm 1's buffer is a per-task choice rather than part of the method (R6). The potential's loss is **Eq. 5 as written**: `(ℰ(x)/T − Σ z·φ/C)²` with `C = Σz` the *realised* kept count — not the expected keep rate `1/(1−p)` against an unnormalised energy, which is what R5's build used. The `1/T` is not cosmetic here: trajectories run 2–9 transitions, a ~20× spread in weight, and LED's own set benchmark is fixed-length so it never sees the difference. There are **no** "two regulariser coefficients" — LED has one squared loss with dropout as the variance regulariser (G12). **[ANALYSIS]** transferring the paper's task-specific values to this environment is a judgement; the values themselves are not | L6, L7 and L7b silently trained under different regularisation, making the capacity-matched comparison meaningless |
| 23b | **LED variant** | **LED-DB** (its Eq. 4), matching the architecture's "DB-style objective", identically for L6/L7/L7b (G12). **The variant is the paper's reported one**: Appendix B.1 gives exactly two ways of preserving the optimal policy under an inaccurate `φ_θ` — a correction term on the terminal flow, which it names `LED-GFN*`, or **uniform redistribution of the decomposition error `ℰ(x) − Σφ` across the trajectory's transitions**, which is what "LED-GFN" denotes in Figures 4, 5 and 11. **Redistribution is adopted; `LED-GFN*` stays out.** R5 read the alternatives as "plain versus correction term" and specified plain — a third form the paper never runs, standing in for the baseline plan §5.1 marks mandatory (R6). Consequence to hold onto: `Σφ̃ = ℰ(x)` exactly, so **decision 13 must be measured on the raw `φ_θ`**, or the band is identically 0 and decision 15's hard constraint measures nothing | one arm preserving the optimal policy under an inaccurate `φ_θ` while its control does not; or "vanilla LED-GFN" naming a variant with no published behaviour |
| 26 | **L7b auxiliary loss** | target = `d(X(τ))`, the deficit vector of the terminal the rollout actually reached; MSE over the six components; weight **[recommended]** `λ_aux = 0.1`; head discarded at evaluation. **[ANALYSIS]** the weight is engineering; the *forward-looking* target is fix F11's requirement, not a choice | the degenerate copy-task head fix F11 retired, returning through an unspecified target |
| 27 | **State-flow head** | one `StateFlowHead` (`log F(s)`), shared by **L5, L6, L7, L7b and GAFlowNet**; checkpointed with the policy; **counted in capacity matching** (G12) | LED-DB and augmented TB are unimplementable, and every matched pair is wrong by the size of a head |
| 25 | **`logZ_θ` tolerance** | **[recommended]** ≤ 1% relative to `log Z` from `Target`, on `tiny_instance()` at convergence. **[ANALYSIS]** engineering, not paper-backed; it is a machinery check, not a Gate-2 metric | a normative number living only in an exit criterion, which is how §6 stops being the single source |
| 24 | **FL-GFN discharge** | exhaustive residual `Φ(X) − β·U(X)` over every valid terminal of the main suite (P3.55), reported as mean/p95/max/non-zero count, with the availability limitation stated as **argued, not measured** (G1). **Measured 11 Aug 2026**: 8,638/8,638 terminals off tolerance at both eligible β; mean 0.373 / p95 0.909 / max 1.554 / R² 0.910 at β = 4 | decision 1 promises a measurement that nothing produces |
| 30 | **RULED, 12 Aug 2026 — `logz_lr_mult = 1`, the papers' 10x measured and declined.** **`logZ_θ`'s learning rate** | **[was recommended]** `logz_lr_mult = 10` — one multiplier over decision 23's `lr`, identical for every arm, so it is not per-arm tuning. **[EVIDENCE]** both objectives this environment rests on prescribe it: TB (NeurIPS 2022) §3 "we found it helpful to set a higher learning rate for Z than for the parameters of `P_F` and `P_B`", and SubTB (ICML 2023) Appendix C verbatim "for Z, use a learning rate of 10× the learning rate for forward logits". R5's build ran one Adam at `lr` over policy + `LogZHead` + flow — neither paper's protocol, and declared nowhere (R6). **The shipped default is 1.0 until this is signed**, because the evidence is mixed and adopting 10 would move decision 25 to accommodate a change made here: measured on the 5-instance tuning suite (2 seeds, 150k) 10× improves TV mid-curve 0.4755 → 0.4499 and final 0.3427 → 0.3248 while *worsening* mean \|log Z error\| 0.607 → 0.675; measured on `tiny_instance()` it breaks decision 25's 1% outright at 1.58% — though that lattice has **one** instance, so `instance_repr` is constant and the head never discriminates, which is the whole situation the multiplier addresses. The regime it is for is real: `log Z` spans 0.86 nats across the 20 main instances while their `instance_repr` vectors sit 0.052–0.227 apart in L2 | the partition function trains at a rate two published sources say is too low, in the one place `log F(s₀) = log Z` is an identity rather than a free parameter — and the departure goes unrecorded |
| 29 | **RULED, 12 Aug 2026.** **GRPO's group size** | `G = 8`, from **architecture §3.2**, which specifies "G = 8 samples/query" for L3. It is **not** the batch size: decision 23 freezes that at 32 trajectories for every arm, so a batch carries four groups and the advantage is standardised within each. R5's build standardised over the whole batch, making `G = 32` — a lower-variance baseline than the architecture specifies, so a *stronger* L3 rather than a weaker one, which is exactly why it read as correct (R6). Lives in `TrainSpec.grpo_group` so it appears in every artefact's shared-protocol block. **[EVIDENCE]** the group-relative construction is GRPO's (DeepSeekMath); `G = 8` is the architecture's number | L3 is not the baseline the architecture describes, and the departure is undeclared — unlike the two in `PHASE3_DECISIONS.md` §1.4, which are |
| 28 | **SubTB's λ** | **[recommended]** `λ = 0.9`, identical for every arm, no per-arm search. **[ANALYSIS]** — SubTB (ICML 2023) has **no single default**: 0.9 is its *hypergrid* value (Appendix A, selected from {0.8, 0.9, 0.99}), while its other tasks use 1.9 (bit sequences, Appendix C; AMP, Appendix D) and 0.99 (GFP, Appendix E), so 0.9 is one of the paper's task-tuned values adopted here untested, and applying any of them to *this* environment is a judgement. *(Corrected 13 Aug 2026 — this cell previously described 0.9 as the paper's experiment-wide value, which its own appendices refute.)* **Added by the build**: R5's decision 23 freezes the optimiser, lr, batch and clipping and omits λ, which is a free parameter of exactly that kind | L5 becomes a comparison of λ tuning rather than of the SubTB objective; or the one arm whose hyperparameter is unlisted is the one that sets `N` and β |

**The ladder's cost, so the schedule implication is visible at the point of decision.** 27 matrix runs (9 arms × 3 seeds); 12 calibration runs per rung (β sweep: L5 × 3 seeds × 2 eligible candidates = 6; sanity: L4/L5 × 3 seeds = 6). At `c₀ = 1 h`:

| Rung | Ceiling | Ladder this rung | Ladder cumulative | Matrix at this ceiling | Total if adopted here |
|---|---|---|---|---|---|
| 0 | 1 h | 12 GPU-h | 12 GPU-h | 27 GPU-h | **1.6 GPU-days** |
| 1 | 2 h | 24 GPU-h | 36 GPU-h | 54 GPU-h | **3.8 GPU-days** |
| 2 | 4 h | 48 GPU-h | 84 GPU-h | 108 GPU-h | **8.0 GPU-days** |

**[ANALYSIS]** R3 recommended `c₀ = 6 h` with two doublings, which permits 24 h per run — 27 GPU-days for the matrix alone, exceeding this phase's entire estimate before development, the β sweeps or calibration. That was an arithmetic failure in a decision whose whole purpose is to bound the schedule. At `c₀ = 1 h` the worst case is 8 GPU-days, which fits alongside ~3 weeks of development; the effort line at the top of this plan is stated accordingly.

**Why the bootstrap is hierarchical.** Twenty instances are evaluated per (arm, seed), and they are **not independent samples of method quality** — they share the seed, the trained weights and the generator. Resampling 60 (seed, instance) pairs as if they were 60 independent draws would understate the interval, which is the same clustering error the LoCoMo caveat guards against elsewhere in this project. Seeds are the outer unit because that is what the significance protocol replicates. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing* (ACL 2018) is the project's significance-testing authority; fixing this before any run is the project's own **[ANALYSIS]** discipline.

**Nothing here may be filled in after step 7.** Exactly three things are written later, and each has a named source: **decision 4's numeral** and **the frozen β** come from the calibration gate at step 6, and **decision 23a's LED coefficients** come from reading the pinned reference implementation — a lookup, not a choice, and it must be done before step 7. Every other value, *including decision 6's threshold and decision 5's ceiling*, is fixed **now** — a threshold written after the run it judges is not a threshold. A value written after L6, L7 or GAFlowNet has trained is a decision made with learner results in view, which §6b's second procedure forbids.

---

## 6b. Declared risks

| Risk | If it bites |
|---|---|
| **[HYPOTHESIS]** L7 does not beat capacity-matched L6 | This is a **designed outcome**, not a failure of the phase. Contribution 3 is not supported; the thesis consolidates on Contribution 1 (v1.2 §9). The phase succeeds by producing a trustworthy negative |
| L7 beats L6 but **not** GAFlowNet | Contribution 3's novelty claim collapses to "an intermediate signal helps", which GAFlowNet already published. The honest write-up reports it as such |
| The `Δd` density on the main suite (0.44–0.50 visitation-weighted, `PHASE2_DECISIONS.md` §6) is a **best case** by construction | A Gate-2 win is a win *under a declared signal density*, and the probe suite is what says whether it survives a sparse one. Both or neither (Phase-2 §6 sign-off condition 3) |
| Nine arms × 3 seeds does not fit the schedule | Cut **Tier-1 breadth before rigour**: L1/L2/L3 answer a secondary question (G2) and are the honest thing to defer; the flow family and GAFlowNet are not |
| `N` chosen too small to separate any method | A null result that cannot distinguish "no effect" from "no budget" is the worst outcome available. Caught by **decision 6's threshold at each rung of decision 4's ladder**; failure at the final rung makes Gate 2 **inconclusive**, never a negative verdict on C3 |
| The ladder's upper rung overruns the schedule | Decision 5 carries the GPU-day arithmetic for every rung, so the overrun is visible when the ceiling is chosen rather than discovered at the last rung. If it still bites, cut breadth before rigour (row above) |
| GAFlowNet's intrinsic reward does not reach zero | Its target is `R + r`, not `R`, so its exact TV is measured against a distribution it never aimed at — and the resulting "L7 beats GAFlowNet" would be an artefact. Caught by criterion 14; the schedule is linear-to-zero for exactly this reason (G11) |

**Amendments.** A change to a *band* on the Phase-2 instrument follows `GRAFT_PHASE2_BUILD.md` §6b's first procedure. A change to a *decision rule* here — the roster, `N`, the primary metric — follows its **second** procedure: new plan version, Gate-0 re-sign-off, and **no learner results inspected beforehand**. In Phase 3 that last clause is nearly always violated after step 6, which is why decisions 1 and 4 are placed before it.

---

## 7. Explicitly not in Phase 3

No search algorithm · no real data · no answerability gate · no distilled utility head · no Stage-B encoder · **no modification to `H`, `U`, `R`, the masks, `d(s)`, the lattice generator or the exact evaluator.** If a Gate-2 result appears to require changing the instrument, that is an amendment under `GRAFT_PHASE2_BUILD.md` §6b with its contamination rule — never an implementation-time adjustment.

**Tier-2 and Tier-3 learners stay out, with one named exception.** Look-Ahead featurisation, OP-GFN, COFlowNet, FM, DB and FL-DB/FL-SubTB remain stubs.

> **Exception: GAFlowNet.** The research plan's Gate 2 item 3 places it in **Tier 2**, and decision 1 promotes it into Phase 3 anyway. The promotion is **solely** because plan §4.5.4 makes capacity-matched GAFlowNet a **required control** for Contribution 3 — not because Tier 2 is opening. It is scored only as C3's control; it is not a Tier-1 method, and it does not license any other Tier-2 row.

**[ANALYSIS]** Recorded as an explicit exception because R1's blanket "no Tier-2 learner" contradicted its own decision 1, and a scope rule that contradicts a decision in the same document is how Tier 2 reopens by accident — the failure `CLAUDE.md` §10 says the tiering "only works if Tiers 2 and 3 stay deferred".

---

## 8. What Phase 4 will need from this, verbatim

```python
from graft.setgen.features import SyntheticFeaturizer
from graft.setgen.policy   import Policy, LogZHead
from graft.setgen.rollout  import sample_trajectories
from graft.setgen.trainer  import Trainer, TrainSpec
from graft.setgen.gate2    import Gate2Report, run_matrix
```

### Requirements this phase places on Phase 4

1. **S5 is "sample K from the trained sampler, `H`-filter, rank by `scorer`."** It consumes a checkpoint produced here; the checkpoint format must be loadable without the trainer.
2. **All five search methods score with exact `U`** on the lattice (fix F13) — the distilled head substitutes only at Phase 9. Phase 4's table is therefore measured under a *perfect scorer*: fair across methods, optimistic relative to deployment, and the write-up must carry that caveat.
3. **`K = 8` and `checker_budget = 32`** are the same constants used here and in the search comparison (fix F5). One constant, everywhere.
4. **S3 and S4 bypass the `ADD` masks** and are `H`-filtered afterwards, which is why closure is a checker sub-check and not only a mask rule (fix F10).
5. **Gate 3 is the kill-shot** — if training-free PCST or submodular greedy matches the learned sampler at equal budget, Stage D's claim narrows before the expensive real-data phases begin. Phase 4 must be able to run whether or not Gate 2 passed.
