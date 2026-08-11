# Phase 3 — what the build decided

Companion to `GRAFT_PHASE3_BUILD.md` (R5). That document says what Phase 3 must
do; this one records what building it settled, what the build found that five
rounds of reading did not, and what is **still open** and must not be lost.

Date: 11 August 2026 · Status: **code complete, calibration gate not run**
Tests: **568 passing** (518 at the end of Phase 2; +50 for Phase 3)

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

Both belong in the write-up. A departure in a *baseline* must run in the
direction that strengthens it, and these do.

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
changes no decision at all. Option 2 is worth a ruling before step 6 runs.

### 2.4 Capacity matching lands exactly for L6 and needs a search for GAFlowNet

At the default width 64 on a 12-dimensional feature environment:

| Arm | Trainable parameters | Note |
|---|---|---|
| L4 (policy + logZ) | 24,834 | |
| L5, GAFlowNet (+ flow) | 33,219 | |
| L6, **L7** (+ potential) | 52,484 | **identical** |
| L7b (+ deficit head) | 61,194 | ablation, not matched |

**L6 and L7 match to 0.00%**, not "within 1%", and the reason is a design choice
worth keeping: the `Δd` block of `action_repr` is *present and zeroed* under L6
rather than absent, so the two arms have byte-identical parameter shapes. That is
the strongest form decision 11 can take, and it removes any argument that L7's
win came from extra parameters.

GAFlowNet carries no `PotentialHead` and is ~37% smaller at equal width, so
`match_capacity` widens its trunk until it is `>= target` and inside 1%. The
search rounds **up**, never down: "within 1%" is a tolerance, "the control is
never smaller" is a direction, and a smaller control turns any win into "L7 had
more capacity".

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
| 2 | **§6's `[recommended]` values are not signed.** The build adopted them as written and says so here. | A value adopted by a builder is not a value signed by the project. Twelve cells: decisions 5, 6, 10, 14, 15, 19 (`c₀`), 23, 25, 26 |
| 3 | **SubTB's λ is in no decision table.** Set to 0.9 — the value SubTB (ICML 2023) carries through its own experiments — in `TrainSpec.subtb_lambda`. | Decision 23 freezes the optimiser, lr, batch and clipping; λ is a hyperparameter of exactly that kind and the plan does not list it. **[ANALYSIS]** as applied here. It belongs in §6 before Gate 2 is read |
| 4 | **The terminal convention was never written down.** Plan §4.1 line 333 says "pick one and write it down"; nobody did. The code uses the **terminating-edge** convention (`F(s) ≠ R(s)` at a state that both stops and continues), verified empirically in Phase 2. | Carried from `PHASE2_DECISIONS.md`. Phase 3 now depends on it in four objectives |
| 5 | **Gate-0 re-sign-off for the Phase-2 β amendment** is still outstanding. | Recorded in the Phase-2 handoff |
| 6 | **The 0.001 `neither`-margin.** One main-suite instance clears the 0.5 band by 0.001 at β = 4. | A weight change, a feature change or a NumPy `default_rng` stream change forces a regeneration. `PHASE2_DECISIONS.md` §4.2 |
| 7 | **The Gate-0 data contract** must be written before Phase 5. | `CLAUDE.md` §7; the memory note |
| 8 | **§2.3's `N`-versus-LED question.** Option 1 is what the code does. | Decide before step 6 runs, not after |

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
* **`ledger=None` does not carry over.** Phase 4 *is* the inference path, and
  `checker_budget = 32` terminal checks per query is enforced there
  (`would_exceed()` before spending, never observed after). Phase 3 meters
  nothing precisely because it is not that path.
