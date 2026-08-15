# Phase 3 — what the build decided

Companion to `GRAFT_PHASE3_BUILD.md` (R6). That document says what Phase 3 must
do; this one records what building it settled, what the build found that five
rounds of reading did not, and what is **still open** and must not be lost.

Date: 12 August 2026 · Status: **code complete, calibration gate not run**
Tests: **590 passing** (518 at the end of Phase 2; +50 for Phase 3; +19 for the R6 review round)

> **§2.2's convergence table and §2.4's capacity table were measured before the
> R6 corrections of §6 below and are stale.** Every flow arm's objective changed
> (GAFlowNet's augmented term, LED's decomposition loss and its redistribution)
> and L6's width changed. They are kept because the *reasoning* around them still
> holds — §2.3's risk in particular — and marked rather than deleted so nobody
> quotes them. Re-measuring is cheap and belongs with step 6.

Same convention as `PHASE0_DECISIONS.md`, `PHASE1_DECISIONS.md` and
`PHASE2_DECISIONS.md`: **[EVIDENCE]** = a named paper, venue stated ·
**[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = judgment made here.

---

## 0. What exists, and what does not

| Build step | State |
|---|---|
| 1 · ML boundary | **Done** (Phase 2). `graft.core` and `graft.synth` import no ML library, asserted per package in a subprocess; `graft.setgen` may |
| 2 · FL-GFN discharge + adapter + heads | **Done and measured** — §2 below |
| 3 · rollout sampler | **Done**, agrees with the exact DP (criterion 3) |
| 4 · trainer skeleton | **Done** — `graft/setgen/trainer.py` |
| 5 · L4 + L5 | **Done and converged** on `tiny_instance()`: TV 0.028 and 0.032 at 50k trajectories, 0.0015 and 0.0053 at 200k, against decision 6's 0.10 |
| **6 · CALIBRATION GATE** | **NOT RUN.** The script exists and is wired end to end (`scripts/phase3_calibrate.py`); the run needs 12 GPU-hours at decision 5's first rung. **`N` and β are not frozen, so §6 is not complete and no L6/L7/GAFlowNet result may be quoted** |
| 7 · L6 + consistency | **Code done**, consistency measurable; the *band* is unmeasured until step 6 fixes `N` |
| 8 · L7, matched L6, GAFlowNet | **Code done**, capacity matching verified; not trained at a frozen `N` |
| 9 · L7b, L1, L2, L3 | **Code done** |
| 10 · Gate-2 harness | **Code done** — `graft/setgen/gate2.py`, matrix + hierarchical paired bootstrap + report, runs end to end on a reduced roster |

**The one thing to be clear about.** Every arm is implemented, tested and
converges; the *comparison* has not been run, because running it before step 6
would contaminate the §6b amendment procedure — β and `N` must be frozen before
L6, L7 or GAFlowNet trains once. That is a stated ordering constraint of the
plan, not an omission here.

---

## 1. Five departures from the plan

### 1.1 `PotentialHead` is a separate network, not a head on the policy trunk

R5's P3.2 lists `PotentialHead` among "the heads every objective needs", and the
first implementation gave it the policy's `h_s`. That is wrong for a reason
decision 23a itself supplies: the potential trains under **its own optimiser at
lr 0.001**, three times the shared protocol's 3e-4. A shared trunk would let
LED's inner loop drag the policy along with it at that rate, eight times per
gradient step — the policy would be trained by an objective it does not have.

LED Appendix C says the potential's architecture is "identical to that of
GFlowNet policy". *Identical architecture* is a separate network of the same
shape. It now is one: `trunk` + scorer, same factory, same widths, **disjoint
weights**, asserted by `test_the_potential_is_its_own_network_not_a_head_on_the_policy`.

**Cost of reverting:** L6, L7 and L7b would train their policies with a hidden
second objective, and the capacity match would be right while the comparison was
not.

### 1.2 The terminating transition is a first-class step

Neither the plan nor LED's paper says what to do with `STOP` when it is an
explicit action, because in the fixed-length settings both were written for the
terminal *is* the last state. Here it is not: a valid rollout ends by choosing
`STOP` from `x`, and a dead-ended one ends because nothing is legal.

Every trajectory therefore carries `len + 1` transitions, and the terminating one
has its own `log P_F` (`log P_F(STOP | x)`, or `0` for a forced dead-end
termination), its own `log P_B` (`0`), and its own `φ` slot. Without it,
`P_F(STOP | x)` never appears in any balance condition, every prefix of a
trajectory scores identically to the trajectory itself, and LED's decomposition
identity does not close.

### 1.3 One `g[i]` coordinate for the whole balance family

TB, SubTB, LED-DB and augmented TB are four expressions of one identity, and
writing four independent implementations of "the partial sum from `i` to `j`" is
four chances to get it wrong in a way no loss curve reveals. `Batch.g(terminal)`
computes `log F(s_i) − Σ_{k<i} log P_F + Σ_{k<i} log P_B` once; every objective is
a difference of two of its entries.

**The terminal boundary is the argument, and that is the whole LED difference.**
`log R(x)` for the balance family; `0` for LED, whose reward is carried by
`Σ φ` instead. Telescoping LED's Eq. 4 forces exactly that, and it is checked
(`test_the_led_boundary_is_zero_and_the_balance_boundary_is_log_r`).

### 1.4 L3's group return is `log R`, not `R`

**A declared departure, and it makes the baseline stronger rather than weaker.**
On this environment `R` spans `r_fail = 1e-6` to `exp(β·U_max) ≈ 2.6e3` — nine
orders of magnitude. GRPO standardises returns within the sampled group, and over
raw `R` the standard deviation is set by the single largest member: every other
trajectory receives an advantage of about `−1/G` regardless of quality, and the
method degenerates into "imitate the best sample". `log R` is the same monotone
ordering on a scale the normalisation can resolve.

**Also declared: no KL to a reference policy.** GRPO's KL term regularises toward
a pretrained reference; there is none here, every arm starts from random
initialisation, and inventing one would give L3 a component no other arm has.

**Third, added in R6: L3 trains on ε-mixture rollouts with no importance
correction.** `rollout.py` justifies exploring off-policy on the grounds that
trajectory-balance-family objectives are off-policy-capable — true, and it does
not cover GRPO, which is an on-policy policy gradient. Trajectories drawn from
the ε = 0.05 mixture and scored with `P_F` give a biased gradient. The bias is
small and decision 10 makes ε identical across every arm, so it does not threaten
the comparison's fairness; it simply is not what the stated justification says,
and unlike the first two departures it does **not** run in the direction that
strengthens the baseline. Declared rather than corrected: correcting it would
give L3 an importance-weighting component no other arm has, which is the same
objection that keeps the KL term out.

All three belong in the write-up.

### 1.5 `sample_trajectories` and the trainer may see a `StateGraph`; nothing in
`learners/` may

P3.1 says `features.py` is "the only module in `setgen/` that may touch a
`StateGraph`", and that was already inaccurate when written — P3.3's own surface
is `sample_trajectories(policy, graph, n, rng)`. The rule that is binding, and
the one exit criterion 6 actually states, is about `learners/`.

The test now states the real invariant and adds a guard the plan did not ask for:
`_ADAPTER_LAYER` is a **closed list**, and a new top-level `setgen/` module fails
`test_the_adapter_layer_is_a_closed_list` until someone records which side of the
boundary it is on. `policy.py` is deliberately *not* on the exemption list — it
is the F6 interface itself and imports nothing but torch, so it is scanned like a
learner.

---

## 2. What the build measured

### 2.1 The FL-GFN discharge (decision 24) — measured, at every eligible β

Fit over **every valid terminal of the main suite** (8,638 of them, 20
instances), with every coefficient and a free per-instance constant:

| β | mean \|residual\| | p95 | max | R² | terminals off tolerance |
|---|---|---|---|---|---|
| 4 | 0.3733 | 0.9085 | 1.5537 | 0.9104 | **8,638 / 8,638** |
| 8 | 0.7465 | 1.8169 | 3.1074 | 0.9104 | **8,638 / 8,638** |

Two named special cases at β = 4: `uniform_omega` (every obligation weighted
equally, size terms off) gives mean 0.868, p95 1.824, R² 0.019; `deficits_only`
frees the deficit weights and pins the size terms.

**What this licenses, in the wording decision 1 fixes.** It disproves *the
deficit potential of research-plan §4.5.2* — not FL-GFN. Even the best-fitting
member of that family, with every coefficient and every per-instance offset free,
misses terminal identity at **every** terminal, at **every** eligible β. That is
materially stronger than any single instantiation could support, and it removes
the "you chose the wrong weights" objection. The complementary limitation — that
no justified informative scalar extension of `sufficiency` to partial states is
currently available — is **argued, not measured**, and is reported as such.

The residual scaling exactly doubles from β = 4 to β = 8 while R² is unchanged,
which is what a family that cannot represent `β·U` at *any* temperature looks
like: it is fitting the same shape and being asked to hit a target twice as tall.

### 2.2 Every arm converges on `tiny_instance()`

Exact TV against decision 6's 0.10 threshold. **One seed (13), one 17-state
lattice** — these are machinery checks, not results, and nothing in the ordering
below should be read as a finding about any objective.

**Stale as of R6** — measured before the objectives were corrected (§6). Kept
for §2.3's argument, not for its numbers.

| Arm | 20k | 50k | 200k |
|---|---|---|---|
| L4 (TB) | 0.118 | 0.028 | 0.0015 |
| L5 (SubTB) | 0.127 | 0.074 | 0.0053 |
| L6 (LED-DB) | 0.109 | 0.035 | 0.026 |
| **L7** | **0.167** | **0.053** | **0.020** |
| L7b | 0.170 | 0.030 | 0.018 |
| GAFlowNet | 0.097 | 0.017 | 0.0009 |

`logZ_θ` reaches 6.0677 against `Target`'s exact 6.0460 at 50k (0.36% relative,
inside decision 25's 1%) and 6.0465 at 200k (0.008%).

**After the R6 corrections**, same instance, same seed, same caveats — every arm
still converges inside decision 6's threshold, GAFlowNet's `c_t` reaches exactly
0 (criterion 14), and nothing diverged under the changed objectives:

| Arm | 50k, post-R6 |
|---|---|
| L4 (TB) | 0.028 |
| L5 (SubTB) | 0.074 |
| L6 (LED-DB) | 0.046 |
| **L7** | **0.025** |
| L7b | 0.036 |
| GAFlowNet | 0.068 |

This is a machinery check that the corrected objectives train, and **nothing
more**. It is not the tuning suite, not the main suite, not three seeds, and not
at a frozen `N`.

### 2.3 **L7 is the slowest of the flow family to converge here, and that is a live risk for C3**

The bolded row is the finding, and it points at the proposed method rather than
at LED generally. At 20,000 trajectories L7 sits at 0.167 — the worst of the six
— while GAFlowNet is already at 0.097 and L6 at 0.109; at 50,000 it is still last
of the LED group. Its extra input features are extra parameters to fit through,
and on a lattice this small the cost of fitting them has not been repaid at these
budgets.

**Why that is dangerous rather than merely interesting.** Decision 4 sets `N`
from **L4 and L5 alone**, deliberately, so the primary budget is not selected on
the proposed method's results. If the adopted `N` lands where L7 has not
converged, Gate 2 returns a negative verdict on Contribution 3 that is about
budget rather than mechanism — and criterion 26 would record C3 as unsupported
without anything having gone wrong with the hypothesis. This is §6b's "`N` chosen
too small to separate any method" row arriving through a specific mechanism, and
it lands on the arm the whole gate is about.

**An honest reading of the evidence's weight.** One seed, one instance, 17
states. It is enough to justify watching the risk and not enough to act on. Two
things would make it real: the same ordering across `{13, 42, 7}`, and the same
ordering on the tuning suite rather than on `tiny_instance()`. Both are step 6's
outputs.

**R6 unsettles this without resolving it, and the ordering matters.** On the
same instance and seed after the corrections, L7 is no longer last — it is first
of the six at 50k (§2.2). **That does not retire the risk**, for exactly the
reason this section already gives: one seed on a 17-state lattice was never
enough to establish the ordering, so it is not enough to overturn it either. What
it does mean is that the evidence §2.3 was written on no longer exists, and a
ruling on option 2 taken *because of* the old table would be a ruling on stale
data.

A procedural note, recorded rather than glossed: the post-R6 numbers above are
**learner results**, and §6b's second procedure asks that decision-rule changes
be made without inspecting any. They were produced as a regression check that the
corrected objectives train at all — on `tiny_instance()`, which is neither the
tuning suite decision 4 reads nor the main suite the arms are scored on — and
they are the same object §2.2 already recorded before the fixes. That is the
honest description; whether it constrains a decision-6 ruling is the project's
call, not the builder's.

**Not resolved here, because resolving it is a decision-rule change** (§6b's
second procedure: new plan version, Gate-0 re-sign-off, and **no learner results
inspected beforehand**) — and the table above *is* a learner result, so changing
decision 4 in response to it is precisely what that procedure exists to prevent
doing casually. Three options, in increasing cost:

1. **Accept it.** Report the `N` decision 4 produces. Gate 2 compares the arms
   *to each other* at equal budget, which is what fix F12 asks, so the comparison
   stays valid — what suffers is the absolute TVs, and a null result would then
   need the "inconclusive" caveat criterion 12 already provides.
2. **Raise decision 6's threshold**, so the ladder escalates until L4/L5 are
   further converged and `N` is larger. Changes one number, reads no L6/L7
   result, and is therefore the cheapest option that is still procedurally clean
   — but it must be done **before** step 6, not after seeing its output.
3. **A separate budget for the potential-bearing arms.** Defensible on the
   grounds that they train two networks rather than one, and it breaks exit
   criterion 11 ("the trajectory budget `N` is identical across every arm"),
   which is load-bearing for fix F12. Not recommended.

**Option 1 is what the current code does**, because it is the only one that
changes no decision at all.

> **RULED 12 August 2026 — option 1, decision 6 unchanged at 0.10. §6.8b carries
> the reasoning and supersedes this section's "worth a ruling before step 6
> runs".** Everything above is the *pre-ruling* analysis and is kept because the
> risk it describes is real and a later reader may want to overturn the ruling
> knowingly. Two things to carry forward with it. First, §6.8b's decisive
> argument is not in this section at all: if `N` is enough for the flow family
> and **L7 alone** is still short, that is the finding fix F12 asks for — the
> primary is "improvement at a *fixed training budget*" precisely because C3's
> claim is about credit assignment, so an arm needing more budget to reach the
> same TV is evidence *against* better credit assignment, not an artefact. A
> threshold tightened to protect L7 from that would be unfalsifiable in the one
> direction it exists to be falsifiable. Second, the evidence *this* section rests
> on no longer exists: after R6, L7 is first of six at 50k rather than last
> (§2.2). Neither table settles the ordering — one seed on a 17-state lattice was
> never enough — which is exactly why the ruling was taken on the argument rather
> than on either table. **Nothing blocks the calibration gate on this item.**

### 2.4 Capacity matching — **superseded by §6.4; the nominal match was wrong**

At the default width 64 on a 12-dimensional feature environment:

| Arm | Nominal parameters | Note |
|---|---|---|
| L4 (policy + logZ) | 24,834 | |
| L5, GAFlowNet (+ flow) | 33,219 | |
| L6, **L7** (+ potential) | 52,484 | identical **nominally** |
| L7b (+ deficit head) | 61,194 | ablation, not matched |

This section originally read: "**L6 and L7 match to 0.00%**, not 'within 1%' …
That is the strongest form decision 11 can take, and it removes any argument that
L7's win came from extra parameters."

**That was wrong, and §6.4 records why.** The `Δd` block being *present and
zeroed* rather than absent gives byte-identical shapes, but 768 of L6's
parameters then read an identically-zero input and never receive a gradient. The
0.00% was a match of nominal counts concealing a **1.46% deficit in trainable
capacity for the control** — outside decision 11's tolerance and on the wrong
side of its directional clause. The counts above are still correct as *nominal*
counts; they are simply not the quantity decision 11 is about.

### 2.5 The calibration gate wires up end to end

A three-second wiring run (`--quick --rungs 3`) reproduces Phase 2's eligibility
measurement (`{4, 8}` eligible, `{1, 2}` not), measures throughput (~2,180
trajectories/s on the tuning suite, this machine), derives `N` from the ceiling,
sweeps β over the eligible candidates on L5 × 3 seeds, selects β = 4, runs the
decision-6 sanity check, fails it at that absurd budget, and records
**`inconclusive`** rather than adopting. That is the correct behaviour, and it is
the branch criterion 12 exists to protect.

---

## 3. Two defects the build found that reading did not

Same category as Phase 2's: both would have produced plausible numbers.

### 3.1 `Δd` on repeated states (found before this session's build, recorded here)

`action_reprs([s, s])` scattered per-edge `Δd` into a state-keyed row map, so the
last occurrence won and every earlier one kept a **zeroed** `Δd` block. Unique
queries — the exact DP, the sampler — never hit it. **Training batches hit it
constantly**, because a batch of trajectories revisits states by design.

The failure points the wrong way: L7 would train on mostly-zeroed features and
look like "`Δd` does not help", which is a **false negative for Contribution 3**.
Fixed with a dense `[n_states, n_atoms, 6]` gather, which is duplicate-safe by
construction, built only for the arms that use it.

### 3.2 A dead end has no logits, and asking for them is NaN, not an error

Building the batch naively queries every trajectory's terminal state for
`log P_F`. For a `FAIL` trajectory that state is a dead end: no legal `ADD`, a
masked `STOP`, so `log_softmax` over its row is all `−inf` and the gather is NaN.
The NaN then flows into the loss, `backward()` poisons every parameter, and the
run continues producing numbers.

The batch builder now excludes exactly those slots from the logits query and sets
their `log P_F = 0`, which is correct rather than defensive: a dead end
terminates with probability 1.

---

## 4. Open — do not lose these

| # | Open item | Why it matters |
|---|---|---|
| 1 | **`N` and β are not frozen.** Step 6 has not run. | Nothing past step 7 may be quoted; §6's `[fill at step 6]` cells are still empty |
| 2 | **§6's `[recommended]` values are not signed.** The build adopted them as written and says so here. | A value adopted by a builder is not a value signed by the project. Decisions **5, 10, 14, 15, 19 (`c₀`), 23, 25, 26** and 28. *(Decision 6 was on this list; it is **ruled and signed** — §6.8b, and its §6 row reads `[signed before calibration]`. Five decisions are ruled in total: 1, 6, 11, 29, 30.)* |
| 3 | **SubTB's λ is in no decision table.** Set to 0.9 — SubTB (ICML 2023)'s *hypergrid* value (Appendix A); the paper's other tasks use 1.9 and 0.99, so there is no single paper default — in `TrainSpec.subtb_lambda`. | Decision 23 freezes the optimiser, lr, batch and clipping; λ is a hyperparameter of exactly that kind and the plan does not list it. **[ANALYSIS]** as applied here. It belongs in §6 before Gate 2 is read |
| 4 | **The terminal convention was never written down.** Plan §4.1 line 333 says "pick one and write it down"; nobody did. The code uses the **terminating-edge** convention (`F(s) ≠ R(s)` at a state that both stops and continues), verified empirically in Phase 2. | Carried from `PHASE2_DECISIONS.md`. Phase 3 now depends on it in four objectives |
| 5 | **Gate-0 re-sign-off for the Phase-2 β amendment** is still outstanding. | Recorded in the Phase-2 handoff |
| 6 | **The 0.001 `neither`-margin.** One main-suite instance clears the 0.5 band by 0.001 at β = 4. | A weight change, a feature change or a NumPy `default_rng` stream change forces a regeneration. `PHASE2_DECISIONS.md` §4.2 |
| 7 | **The Gate-0 data contract** must be written before Phase 5. | `CLAUDE.md` §7; the memory note |
| 8 | ~~**§2.3's `N`-versus-LED question.**~~ **RULED 12 Aug 2026: option 1, accept** — §6.8b. | Ruled before step 6 ran |
| 9 | ~~**Decision 11's tolerance moved; decisions 29 and 30 are new**~~ **RULED 12 Aug 2026, §6.8b** (§6.4, §6.6, §6.10). All need the same ruling every `[recommended]` cell needs. | The 1% is unachievable by width, so the criterion became minimality plus the directional clause. Decision 30 ships its mechanism at a **default of 1.0** — the papers say 10, one measurement agrees on TV and disagrees on `log Z` error, and adopting 10 would move decision 25's tolerance to fit a change made here. All were changed **before** any L6/L7/GAFlowNet result was inspected — §6b's second procedure is satisfied on the ordering, not on the paperwork |
| 11 | **`instance_repr` is thin for what `logZ_θ` must do** (§6.11). Recorded, not fixed. | 12 informative dims, instances 0.052–0.227 apart in L2, `log Z` spanning 0.86 nats. Widening it touches every capacity number, so it waits — but it is the next place to look if decision 30 is signed and `log Z` error stays high |
| 12 | **Decision 5's ladder is wall-clock on a named machine**, and Phase 3 is a CPU workload (`TrainSpec.device` defaults to `"cpu"`; the sampler is numpy; the networks are ~52k-parameter MLPs). `--device` now exists on both scripts. | The GPU-day table and the README's "the CUDA build is not incidental" describe a phase that never touches CUDA. Harmless until someone budgets against it |
| 10 | **§2.2 and §2.4's measurements are stale** and marked as such. | Every flow objective changed and L6's width changed. Re-measure with step 6 |

---

## 5. What Phase 4 gets from this, unchanged

* `policy(state_repr, action_reprs) → logits` — frozen, and the search methods of
  Phase 4 consume the trained policies through the same interface.
* `Environment`, `TrainSpec`, `Trainer`, `Batch` — the shared protocol, so a
  Phase-4 search baseline is compared under the same budget accounting.
* `gate2.paired_bootstrap` — Gate 3's comparison is the same statistical object
  with different arms.
* `audit_block` — every Phase-4 table carries the same audits, from the same
  Phase-2 source.
* **`Trainer.save_checkpoint` / `policy.load_policy`** — §8 requirement 1's
  "loadable without the trainer", now real. `load_policy` lives in
  `graft/setgen/policy.py`, which imports nothing but torch, so S5 consumes a
  Gate-2 checkpoint without the training stack. `run_matrix(checkpoint_dir=...)`
  writes one per (arm, seed). **Read `blob["delta_d"]`**: an L7 policy behind an
  L6 featurisation reads a zeroed `Δd` block, which is silently wrong rather than
  an error.
* **`gate2.best_of_k`** — plan §6.4's Stage-D primary, already implemented here
  as criterion 17's secondary, so Phase 4's S1–S5 report the same object.
* **`ledger=None` does not carry over.** Phase 4 *is* the inference path, and
  `checker_budget = 32` terminal checks per query is enforced there
  (`would_exceed()` before spending, never observed after). Phase 3 meters
  nothing precisely because it is not that path.

---

## 6. The R6 review round — what a paper-by-paper reading found

An external review of the built code, checked against the PDFs in
`Research papers/01_GFlowNet/` rather than against the code's own algebra. Seven
findings, six upheld. **Three were defects in *controls*, and all three ran in
the direction that flatters the proposed method** — which is `CLAUDE.md` §5's
recorded failure mode ("overreaching on what a paper establishes") arriving in
code instead of prose. Every one of them reads as correct until the equation is
opened, and every test that covered them checked the implementation against
itself.

The distinction that matters for triage, and which the review did not draw:
**§6.1 and §6.4 are asymmetric** — they touch the controls only, so they can
manufacture a win. **§6.2 and §6.3 are symmetric** — L6, L7 and L7b share
`led_db_loss` and `decomposition_loss` verbatim, so an error there hits all three
identically and cannot bias the paired test. What those two break is the
*identity of the mandatory baseline*: Gate 2 would have compared
"checker-conditioned X against X" where X is not published LED-GFN, so neither a
positive nor a negative result would have spoken to plan §4.5.4 as written.

### 6.1 GAFlowNet was not Eq. 4 — the `P_B` divisor was missing

`Generative Augmented Flow Networks` (ICLR 2023), p.5, Eq. 4:

```
Z Π P_F(s_{t+1}|s_t) = R(x) Π ( P_B(s_t|s_{t+1}) + r(s_t→s_{t+1}) / F(s_{t+1}) )
```

Factoring `P_B` out to express this against TB's own residual gives
`Σ log(1 + r/(F·P_B))`. The build subtracted `Σ log(1 + r/F)`, which is
`Σ log(P_B + P_B·r/F)` — an intrinsic reward of `P_B·r` rather than `r`. `P_B` is
uniform over removable atoms, so the augmentation was attenuated by the removable
count, and by more of it as the set grows; it is not even a rescaling of `c₀`.

Two things bound the damage, and both belong in the write-up rather than being
used to wave it away. `c_t` reaches exactly 0 by `0.8N`, so the *final* target was
still `R` and exact TV was never measured against the wrong distribution. And the
direction of harm is a prior rather than a certainty — a weaker exploration bonus
could plausibly help TV on an environment this dense. The statement that does not
need a prior: **it was not GAFlowNet, so plan §4.5.4's required control had not
been run.**

Fixed in `learners/gaflownet.py`; the docstring stated the wrong identity too, so
the error was in the reading, not in the transcription. `test_augmented_tb_is_
gaflownet_equation_4_not_its_own_algebra` evaluates Eq. 4 directly in the
probability domain, in float64, sharing no algebra with the loss.

### 6.2 The decomposition loss was not Eq. 5

LED-GFN Eq. 5, p.5:

```
ℓ_LS(τ) = E_{z~Bern(λ)} [ ( ℰ(x)/T − (Σ_t z_t·φ_θ(s_t→s_{t+1})) / C )² ],  C = Σ_t z_t
```

The build used `(Σ z·φ/(1−p) − ℰ)²`: the *expected* keep rate instead of the
realised count `C`, and no `1/T` on the energy. The second is the one that bites
here. `1/T` scales each trajectory's residual, and this environment's
trajectories run 2–9 transitions — roughly a 20× spread in weight. LED's own
set-generation benchmark is **fixed-length**, so the difference is invisible
there, which is exactly why it had to be read off the equation rather than
inferred from the task being similar. That inference is `CLAUDE.md` §5's first
pattern, one level down from where it was last caught.

### 6.3 L6 was not the LED-GFN of the paper's experiments

Appendix B.1 gives two ways to preserve the optimal policy under an inaccurate
`φ_θ`: a correction term on the terminal flow, which it names `LED-GFN*`, or
**uniform redistribution of the decomposition error across the trajectory's
transitions**, which it states is what its implementation does and what
"LED-GFN" denotes in Figures 4, 5 and 11.

Decision 23b read the alternatives as "plain versus correction term" and chose
plain — a third form the paper never runs. Plan §5.1 marks LED-GFN **mandatory**
and §4.5.4 makes capacity-matched *vanilla* LED-GFN the control for C3, so the
control was a variant with no published behaviour.

`redistribute` now implements the paper's form. **One consequence had to be ruled
rather than ported blind**: `Σφ̃ = ℰ(x)` exactly, per trajectory, by construction.
Measured on `φ̃`, decision 13's consistency error is identically 0 for every arm —
a band that always passes, which would have voided decision 15's hard constraint
silently. `consistency_error` is therefore measured on the **raw** `φ_θ`, which is
also the right object: the regulariser plan §4.5.4 asks to be preserved is Eq. 5,
and Eq. 5 trains the raw potential.

**A related claim that did not survive checking.** The review also held that the
potential's batches must come from a replay buffer, per Algorithm 1. Appendix C
says otherwise per task: bag and RNA generation reuse a buffer their base
implementation already had, **molecule generation uses the round's own samples**,
and **set generation inherits molecule generation's decomposition settings**. Set
generation is the closest published task to this one. The trainer's behaviour was
already the paper's; what was missing was the citation, now in `l6_led.py`.

### 6.4 The capacity match was nominal, and 1% turned out to be unachievable

L6's `Δd` block is present and zeroed, so `N_DEFICIT × hidden` weights per
`action_repr` consumer — `Policy.scorer` and `PotentialHead.net`, 768 at hidden
64 — read an identically-zero input. Their gradient is `δ·x_j = 0` on every
example forever. Counted nominally, L6 and L7 matched to 0.00% while L6 held
**1.46% less trainable capacity**: outside decision 11's tolerance and on the
wrong side of its directional clause.

`policy.match_capacity`'s own docstring already named this failure mode — "a
match on paper and dead capacity in fact" — as the reason GAFlowNet is widened
rather than padded. `capacity_matched_arm` then did precisely that for L6, four
lines below.

**And decision 11's 1% is not reachable.** The dead block is `12·hidden`; one
width step is ~`2.4·hidden`; the ratio is 0.50, 0.52, 0.53 at hidden 64, 128, 256
— a property of the architecture, not of the width. So the narrowest width that
closes the gap always overshoots by about half a step, ~1.4–2.2%, at every scale.
The alternative of removing L7's extra parameters is not available: those weights
**are** the mechanism under test.

The criterion is therefore minimality plus the directional clause, which invents
no number and cannot need bumping: the control is the **narrowest width that is
not smaller**, verified rather than assumed, with the achieved excess reported.
The argument survives intact and is in fact stronger — if L7 wins, it wins against
a strictly larger control.

| Arm | Hidden | Nominal | Dead | Live | vs L7 |
|---|---|---|---|---|---|
| L7 | 64 | 52,484 | 512 | 51,972 | — |
| L6 (matched) | 65 | 54,019 | 1,300 | 52,719 | **+1.44%** |
| GAFlowNet (matched) | 83 | 54,119 | 1,162 | 52,957 | **+1.90%** |

*(Re-measured 13 Aug 2026 with the post-§6.10 `dead_capacity_of`, which counts
`LogZHead`'s `STATE_EXTRA_DIMS·hidden` pad in **every** arm — including L7,
whose dead count is no longer 0. Widths and the +1.44% are unchanged;
GAFlowNet's excess moved +2.17% → +1.90% because its dead block grew more than
L7's. The earlier row counted only the zeroed-`Δd` blocks.)*

### 6.5 Gate 2 could return a verdict from a run that was not Gate 2

`run_matrix` accepted any roster and any seed subset and still emitted
`contribution_3_supported` as a boolean — computed by a hierarchical bootstrap
whose outer resample had a single cluster when one seed ran. Criterion 10's "the
config refuses to shorten" described an intention with no mechanism behind it.

Refusing by raising would force every wiring test to run a full roster it does
not need, so the run proceeds and the **verdict** is withheld: null, with the
reason recorded beside it. Four related gaps closed with it — FAIL coverage was
measured and never read (criterion 16); best-of-K was named as criterion 17's
secondary and never built, leaving L1–L3 with no metric that fits them at all
(decision 12); the held-out probe read of criterion 23 had no code path; and no
model was ever persisted, so a Gate-2 run discarded every set of weights it
trained and Phase 4's S5 had nothing to consume (§8 requirement 1).

### 6.6 GRPO's group was the batch

Architecture §3.2 specifies "G = 8 samples/query" for L3; decision 23 freezes the
batch at 32. `group_advantage` standardised over the whole batch, making `G = 32`
— four groups pooled into one baseline. It is a **lower-variance** baseline than
the architecture specifies, so a *stronger* L3, which is why it read as correct.
Now four groups of eight, with `G` in `TrainSpec.grpo_group` so it appears in
every artefact's shared-protocol block (decision 29).

Not upheld, and correctly so: the review declined to reject the absent clipping.
With one update per batch and `π_old = π`, the ratio is identically 1 and the
clip is inert.

### 6.7 The calibration gate's budget

`budget_for` promised rounding to a batch multiple and did not do it, and it
billed the ceiling for training time only. The second is the substantive one: the
throughput pilot evaluates twice while a rung evaluates 51 times, so the measured
rate over-predicted, `N` came out too large, and every rung would have overrun
the ceiling it was derived from with nothing measuring it. Throughput and
evaluation cost are now measured separately and `N` solves
`N/rate + (C+1)·eval = ceiling`.

**The ceiling is deliberately still not passed to the trainer as
`wall_clock_ceiling`**, which is where this fix departs from what the review
asked for. A truncated run spends fewer than `N` trajectories, and criterion 11
requires `N` identical across every arm — so a hard stop would silently break the
one thing the budget exists to hold fixed. The run is allowed to finish and the
overrun is **recorded** instead (`measured_s_per_run`, `ceiling_respected`).

`graft/tests/test_setgen_calibration.py` is new: `budget_for` produces fix F12's
primary budget and had no test, which is why two defects lived in it.

### 6.8 A second pass on the fixes — three more, all "recorded but not acted on"

The same review, re-run against the first pass of R6 fixes. Nothing it found was a wrong
computation; all three were a value being measured, written into the artefact,
and then not used by the thing it existed to constrain. Worth naming as a class,
because it is not the class §6.1–§6.4 belonged to and it will recur elsewhere.

**An over-ceiling rung could still be adopted.** `ceiling_respected` was computed
and the adoption branch read `rung["sanity"]["passed"]` alone, so the field was
decorative — the exact shape of the FAIL-coverage defect in §6.5, one file over.
Worse, it was an *average* over the rung's runs: decision 5's ceiling is a
per-run budget, and a mean lets one slow seed clear a bar it individually broke.
Now `beta_sweep` and `sanity_check` return their slowest run, the rung is judged
on the max, and an overrun ends the ladder with verdict **`over_ceiling`** rather
than adopting or escalating. Escalating would be worse: the next rung's ceiling
is larger, and an overrun means the throughput estimate that produced `N` was
wrong, so the rung did not spend the budget it reports.

**Best-of-K was not fix F5's portfolio.** F5 fixes `K = 8` *and its composition*
— 1 greedy + 7 sampled. §6.5 built the metric and sampled all eight, which
measures a different object from the one Phase 9 ships and Phase 4's S5 is
compared against; the greedy candidate is precisely the one a reward-maximiser
would return, which is the comparison the portfolio argument (plan §9, Robust
Scheduling) turns on. Greedy is now a parameter of `sample_trajectories` rather
than a second walk in `gate2` — the rollout module's header gives the reason.

One claim written while fixing it did not survive its own check, and is recorded
because the habit is the point: the docstring initially said the greedy row
leaves the sampled rows identical to a plain call at the same seed, on the
reasoning that the random draw still happens. It does not — the draw is sized by
the live set, so a greedy rollout terminating at a different step shifts every
later draw. Measured, then corrected in place.

**Admissibility checked two clauses of five.** §6.5 closed the roster and the
seeds; a run on **one tiny instance, at an uncalibrated `N` and β, with no probe
suite** still returned a scientific boolean. Every input that defines the
experiment is a `run_matrix` parameter with a default, so each needs its own
clause: the frozen main suite by **instance identity** (decision 8, criterion 25
— not by count, which 20 arbitrary instances would pass), the probe suite's
presence (criterion 23), and `N` and β **matching an adopted, non-`--quick`
calibration record** (decision 4, Phase-2 decision 22). The last is what makes
§6's "nothing proceeds until this is written into §6" enforceable rather than
procedural.

The probe also gained its own `audit_block`. It was scored and reported as a
bare TV, while criterion 23 requires the result stated *under its declared `Δd`
density* and §6b's risk row says "both or neither" — which is unreadable when
only the training suite reports a density.

Reference-suite identity is computed once and cached (`_frozen_identities`): both
suites are deterministic functions of seeds frozen at Gate 0, and the check went
from 51 s to well under one when it stopped regenerating them per clause.

### 6.8b The four rulings, 12 August 2026

Taken together because they interact, and recorded with the reasoning so a later
reader can overturn any of them knowingly.

**Decision 6 stays at 0.10 — §2.3's option 1, accept.** The argument that changed
is not the stale table; it is that §2.3 conflates two situations. If `N` is so
small that *nothing* separates, that is genuinely inconclusive — and it is now
caught twice, by decision 6 on the tuning suite and by criterion 15 on the main
suite, both routing to `inconclusive` rather than to a negative. But if `N` is
enough for the flow family and L7 alone is still short, **that is the finding fix
F12 asks for**: the primary is deliberately "improvement at a *fixed training
budget*" precisely because C3's claim is about credit assignment, so an arm that
needs more budget to reach the same TV is evidence against better credit
assignment, not an artefact of the budget. Tightening the threshold to protect L7
from that would make the rule unfalsifiable in the one direction it is supposed
to be falsifiable. Option 1 also changes no decision, which is what §6b prefers.

*The cost of this ruling is real and should be stated:* sizing `N` from the
slowest arm (§6.9) makes `N` ~2.9× smaller in trajectories than the old L5-based
sizing would have produced at the same ceiling. L7 therefore trains on fewer
trajectories than the pre-R6 plan implied. That is the honest consequence of the
ceiling meaning something; the alternative was a ceiling that six of nine arms
broke.

**Decision 30 — `logz_lr_mult = 1`, the papers' 10× measured and declined.** The
decisive point is one the review did not make: **the multiplier is identical
across every arm, so it cannot bias the L7-vs-L6 or L7-vs-GAFlowNet comparison at
all.** It moves absolute TVs, which feed the machinery thresholds, not the paired
test. So this is a low-stakes choice, and at low stakes the conservative option
wins: 10× breaks decision 25's signed 1% tolerance outright, and moving a signed
tolerance to accommodate a change made here is the wrong direction whatever the
papers say. The defect was that the build did something neither paper does *and
declared it nowhere*; the mechanism and decision 30 close that, and the number is
now a recorded, reversible choice with its measurements attached.

**Decisions 11 and 29 adopted as implemented.** Decision 11's minimality rule is
not a preference — width granularity makes the 1% unachievable at any scale
(§6.4), so it is the only form the criterion can take. Decision 29 restores the
architecture's own `G = 8`; it was never a choice, only an omission.

### 6.9 A third pass — two blocking, and the same defect twice

Both blocking findings are *a rule the plan states that no code path evaluated*.
That is a third class, distinct from §6.1–§6.4's misread equations and §6.8's
recorded-but-unread values: here the value was never computed at all.

**`N` was calibrated on the cheapest arm and spent by the most expensive.** The
pilot ran L5. Measured on the tuning suite: L5 ~2,700 trajectories/s, the LED arms
~900 — a **2.9× gap**, because decision 23a's `N = 8` decomposition iterations
mean L6/L7/L7b take eight potential steps per policy step. So an `N` sized to fit
L5 inside a 1 h ceiling buys L7 a **2.9 h** run, and decision 5's GPU-day table
understates the matrix by that factor at every rung. **Nothing could detect it**:
the rung's ceiling guard reads `beta_sweep` and `sanity_check`, and both run L4
and L5 exclusively, so the ladder certified a ceiling six of the nine arms break.

The fix turns on a distinction decision 4 did not draw. Its rule is "reads L4/L5
only; never L6/L7/L7b/GAFlowNet", and its purpose is that the primary budget must
not be selected on the proposed method's **results**. A rate in trajectories per
second is not a result: it is a property of the machine and the architecture,
available before a gradient step means anything, and it cannot move in L7's
favour. So the pilot now times **every** arm at its matched width and sizes `N`
from the slowest, while β selection and the decision-6 threshold still read L4
and L5 alone — which is where the contamination risk actually lives. The ceiling
check gains a predicted worst-arm term from those same rates, so the guard sees
the arms it was blind to without training them.

Measured, capacity-matched, on the tuning suite: L2 6,606 · L1 5,536 · L3 3,641 ·
L4 3,016 · L5 2,946 · GAFlowNet 2,398 · L7 1,080 · L6 1,079 · **L7b 1,034**.

**Criterion 15 was implemented nowhere.** "At the adopted `(N, β)`, L4 and L5
reach exact TV below decision 6's level **on the main suite**" appeared in the
exit criteria and in no code path; `_verdict` conjoined six clauses and none was
this. Decision 6's check runs on five *tuning* instances during calibration; the
matrix runs twenty *main* instances through one conditional `logZ` head, and
passing the first does not imply the second. A matrix in which the machinery
failed on the scored suite still emitted a scientific boolean.

Closing it forced §6b's other open question with it — **item 8's**: `band_ok`
pooled L6's and L7's consistency, so vanilla LED-GFN failing to decompose its own
energy at this `N` came out as `contribution_3_supported = False`. The verdict now
has **three outcomes**. `inadmissible` and `inconclusive` both return null, and
the instrument clauses — criterion 15, *L6's* band, GAFlowNet's `c_t`, FAIL
coverage — route to `inconclusive`, because each is a statement about the harness.
Only the two comparisons, **L7's own** band and the non-inferiority margin can
make the answer `False`. That is criterion 12's reasoning applied past
calibration: a null that cannot distinguish "no effect" from "the instrument did
not work" is the worst outcome available.

**Also in this pass.** `logZ_θ` trained at the shared rate against both papers'
protocol (§6.10). The retired capacity claim survived in three module docstrings
because the guard read twelve `.md` files and no source — a correction that landed
in four documents missed three modules, in the medium where this project keeps
most of its reasoning; `check_plan_consistency.py` now scans 84 sources as well,
and exempts itself, since the blocklist necessarily contains every retired string.
And there was no Gate-2 runner: `run_matrix` was called only by tests and
`retarget` was stranded in `scripts/`, so the last mile — calibration record →
retarget → suites → matrix → report — existed only as instructions to an operator.
It is `Environment.at_beta` and `scripts/phase3_gate2.py` now.

### 6.10 `logZ_θ`'s learning rate — mechanism shipped, value not adopted

**[EVIDENCE]** Both objectives this environment rests on prescribe a faster rate
for the partition function. TB (NeurIPS 2022) §3: *"we found it helpful to set a
higher learning rate for Z than for the parameters of P_F and P_B."* SubTB (ICML
2023) Appendix C, verbatim: *"for Z, use a learning rate of 10× the learning rate
for forward logits."* The build ran one Adam at `lr` over policy + `LogZHead` +
flow — neither paper's protocol, and declared nowhere. **The declaration gap is
the defect**, and decision 30 plus the separate parameter group close it.

**The value is not adopted, and the honest reason is that the evidence is mixed.**

| | `mult = 1` | `mult = 10` |
|---|---|---|
| tuning suite, 2 seeds, 150k — TV at 40% | 0.4755 | **0.4499** |
| tuning suite — final TV | 0.3427 | **0.3248** |
| tuning suite — mean \|log Z error\| | **0.607** | 0.675 |
| `tiny_instance()` — decision 25's 1% | passes | **fails, 1.58%** |

TV moves the way the papers imply; `log Z` error moves against them, which is the
opposite of what the review that raised this measured at 400k. And `tiny_instance()`
has **one** instance, so `instance_repr` is constant and the head never has to
discriminate — the entire situation the multiplier exists for is absent there,
which is an argument against reading that row as decisive rather than a reason to
ignore it.

**The shipped default is therefore 1.0**, reproducing every number already
measured, with 10 recorded as decision 30's `[recommended]` value. Adopting it
would require moving decision 25's tolerance to accommodate a change made here,
and a normative tolerance that moves to fit a new default is not a tolerance.

**§6.11 is the related finding this does not fix.** `instance_repr` is
`atom_feat.mean(0)` padded with `STATE_EXTRA_DIMS` zeros: 12 informative
dimensions, of which the 20 main instances' vectors sit **0.052–0.227 apart in
L2** (per-dimension std 0.014–0.056), while `log Z` spans **0.86 nats** across
them. The head must be steep on near-collinear inputs, which is exactly the regime
decision 30 addresses and exactly why one lr for both is a poor fit. Widening the
representation is a model change touching every capacity number, so it is recorded
as a limitation rather than taken now — but if decision 30 is signed and `log Z`
error stays high, this is where to look next, not at the multiplier.

*(While measuring it: `LogZHead`'s first layer reads those `STATE_EXTRA_DIMS`
zeros, so it carries a dead block of its own in **every** arm.
`Trainer.dead_capacity_of` now counts it — the function claims to return
parameters that can never train and was under-reporting. It disturbs no match:
both sides of a matched pair carry it, and the residual difference is 8 parameters
between L6 at width 65 and L7 at 64.)*
