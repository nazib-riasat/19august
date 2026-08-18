# Phase 4 — what the build decided and measured

Companion to `GRAFT_PHASE4_BUILD.md` (§6 ruled 12 Aug 2026). That document says
what Phase 4 must do; this records what building it settled, what the build
measured that reading did not, and **three sub-decisions of the ruled table that
measurement overturned** — every one before any Gate-3 number existed, and none
of them reading a learner result, so `GRAFT_PHASE2_BUILD.md` §6b's second
procedure is satisfied on the ordering.

Date: 13 August 2026 · Status: **Stage A built, green and run; Stage B unblocked
15 Aug 2026** — Phase 3's matrix ran (`PHASE3_DECISIONS.md` §7) and its 27
checkpoints sit at `artefacts/checkpoints/`; the Stage-B composition has not yet
run (G7's two-stage exit)
Tests: **639 passing**, of which 44 are Phase 4's (`graft/tests/test_search.py`)

Same convention as the other DECISIONS files: **[EVIDENCE]** = a named paper,
venue stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** =
judgment made here.

---

## 0. What exists

| Module | State |
|---|---|
| P4.1 `search/base.py` — protocol, `SearchResult`, ledger discipline, `h_filter`, closure completion, admissible ground set | **Done** |
| P4.2 `search/relevance.py` — decision 1's two variants | **Done** |
| P4.3 `search/s1_greedy.py`, `s2_beam.py` | **Done**, 0 checks each |
| P4.4 `search/s3_submodular.py` — Eq. 1 with all five constants | **Done** |
| P4.5 `search/s4_pcst.py` — Mapping A forest, completion, `C_e` calibration, **exact solver** | **Done** |
| P4.6 `search/s5_portfolio.py` — checkpoint consumer, no trainer import | **Code done; checkpoints exist since 15 Aug 2026** (27 at `artefacts/checkpoints/` — G7's blocker lifted) |
| P4.7 `search/gate3.py` — matrix, negated bootstrap, broadcast, `p*` ceiling, audits | **Done** |
| Gate-3 **Stage A** on the 20-instance main suite, both relevance variants | **Run** — §2 |
| Gate-3 **Stage B** | **Unblocked 15 Aug 2026** (step 6 adopted and the matrix ran — `PHASE3_DECISIONS.md` §7); not yet run, and S5's arm-selection rule is unruled (§5 item 1) |

---

## 1. The three overturned sub-decisions

All three sit inside ruled decision 8 or its consequences. Each was overturned by
a measurement, not an argument, and each measurement is a property of the
environment or of a third-party library — **no learner result was inspected**.

### 1.1 `pcst_fast` is pinned and **cannot be used**: its Windows wheel is wrong

Decision 8 ruled *"`pcst_fast`, pinned. The platform risk an earlier draft raised
is not real: a prebuilt `cp311-win_amd64` wheel exists (1.0.10, verified by
download)"*. The wheel exists. It does not work.

**Measured**: against a brute-force exact PCST reference (enumerate connected
vertex subsets, MST by Kruskal, compare the *objective value* so a tie-break is
not counted as a failure) on **60 random graphs**, `pcst_fast` 1.0.10's
`cp311-win_amd64` wheel returned the wrong optimum on **59**. The failure
signature is unambiguous: every returned array has the right *length* and every
element equal to its **first** — `[0, 0]` where `{0, 1}` is the unique optimum,
`[2, 2, 2, 2]` where `{2, 3, 4, 5}` is. A buffer-lifetime bug in the binding;
silent, never a crash.

The ruling verified that the wheel *existed*, which is a different claim from
its being *correct*. G8 named the alternative in advance — *"implement the PCST
objective directly and verify it against `pcst_fast` on a handful of instances
offline"* — and that verification is exactly what found this.

**Adopted: an exact solver, and the discrepancy shipped as a test.** Exactness is
affordable rather than heroic because **uniform edge costs collapse the
problem**: with one `C_e` per reference link a spanning tree of a connected `S`
costs `C_e·(|S| − 1)`, so G-Retriever's Eq. 7 becomes

```
value(S) = Σ_{n∈S} prize(n) − C_e·(|S| − 1)
```

over connected vertex sets — a maximum-weight connected subgraph problem.
Measured component sizes on the tuning suite are 1–13 atoms (2–4 components per
instance), so enumerating each component's connected subsets is instant; the
bound is asserted at `MAX_COMPONENT_ATOMS = 20` rather than assumed.

**Cost to reverse:** none scientifically — the solver is exact where `pcst_fast`
is approximate — but `pcst_fast` is now an installed dependency used **only** by
the regression test that documents its breakage.

### 1.2 `C_e`'s selection criterion: the median-at-the-cap straddle

Decision 8 selects `C_e` by *"median post-completion output size closest to
`max_atoms` without exceeding it"*. Implemented literally, that admits a median
**equal to** the cap — which puts half the outputs **over** it.

**Measured** on the tuning suite, obligation relevance:

| `C_e` | median size | breach rate (> `max_atoms`) |
|---|---|---|
| 0.25 | 9.0 | 0.525 |
| **0.50** | **8.0 = the cap** | **0.500** |
| 1.00 | 4.0 | 0.350 |
| 2.00 | 3.0 | 0.175 |

The literal criterion picks `C_e = 0.5` and hands S4 a **50% size-rejection
rate** — the straddle the plan's own amendment warned about, reached through the
letter of its replacement.

**Adopted: select on the post-completion breach rate, ties to the larger
median.** The breach rate measures the property the criterion was always *for*
(S4's candidates surviving `H`), admits no straddle, and stays purely structural
— it reads no `U`, no best-of-K, no Gate-3 number. Ties break toward the larger
median deliberately: a bigger subgraph is more evidence, and `CLAUDE.md` §8 wants
S4 dangerous.

**Frozen result**, calibrated on the tuning suite before S4 touched main:

| variant | `C_e` | achieved median | breach rate |
|---|---|---|---|
| obligation | **2.0** | 3.0 | 0.175 |
| informed | **2.0** | 3.0 | 0.000 |

Honest reading: under obligation relevance **no grid value is breach-free**, and
the best available still rejects 17.5% on size. That is a structural finding
about Mapping A under a proxy relevance, not a tuning failure — and the grid is
predeclared, so extending it would be inventing a new predeclared object.

### 1.3 Decision 5's one live gate is **size-confounded**, and predetermined here

G5 measured three candidate Gate-3 criteria and retired all three, leaving one:
*"if S5's portfolio diversity does not exceed the training-free arms', the flow
method is not producing a portfolio at all"*. Measurement says that criterion is
a **fourth** predetermined rule.

**Two measurements.** First, mean pairwise Jaccard distance falls monotonically
with set size for **uniformly random** portfolios — 8 random sets from a 24-atom
pool:

| set size | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| mean pairwise Jaccard distance | 0.945 | 0.923 | 0.896 | 0.873 | 0.850 | 0.825 | 0.792 |

*(Values from the shipped seeded control (`DIVERSITY_CONTROL_SEED`, 2,000 draws
per size pair, symmetric since the post-fix audit); a 3,000-draw MC agrees
within ±0.005. An earlier version of this table quoted a separate
high-precision estimate whose provenance was unrecorded.)*

So a method returning smaller sets scores higher diversity *for free*. Second, on
the main suite S4 under **informed** relevance returns size-3.78 sets and scores
**0.551** raw — above `p*`'s own **0.4517**. Since a converged S5 samples ≈ `p*`,
its diversity ceiling is ~0.45, so **S5 cannot beat S4 on the ruled metric**,
whatever it learned.

**Adopted: implement the ruled metric faithfully and report a size control
beside it** — `excess_diversity = observed − E[random portfolio at the same set
sizes from the same pool]`, seeded and frozen (`DIVERSITY_CONTROL_SEED`). The
ruling stands; the control is what says how much of any gap is mechanical. This
is the "report both, declare separately" pattern Phase 2 §6 used for the `Δd`
density.

**[ANALYSIS] Recommendation for Phase 9, not taken here:** the size-controlled
form (or a comparison at matched set size) is what makes a diversity criterion a
statement about methods rather than about set sizes. Changing the *gate* is a
§6b decision-rule amendment and belongs with Phase 9's re-ask of the best-of-K
comparison under the noisy scorer, not smuggled in as an implementation choice.

*(The estimator was corrected after the review round — §1.4 F1 — and the finding
survives it: under the pinned duplicates-included convention S4-informed scores
**0.483**, still above `p*`'s **0.4506**.)*

---

## 1.4 The adversarial review round — what five lenses found

Five independent review lenses (paper fidelity, algorithmic correctness, plan
conformance, test quality, integration), each followed by a **refutation** pass
instructed to kill its own findings. What survived:

**F1 — the diversity estimator floated against its own pin. MAJOR, fixed.**
Decision 5 pins *"the C(K,2) unordered pairs of the returned sets, **duplicate
pairs included**"*, and `jaccard_diversity` was being handed `SearchResult.sets`,
which is **deduplicated for ranking**. Every method returned duplicate-free sets
(0/80 method×instance runs returned a repeat), so the pinned convention was
unreachable at its only call site. It moved the gate's boundary: sweeping a
sampler from converged to collapsed, the shipped estimator **passed** a sampler
that had lost 27% of its modes where the pinned one **failed** it. `SearchResult`
now carries `portfolio` — the returned candidates *with multiplicity* — and
diversity, its random baseline and the excess are all computed over it. Visible
in §2: S1's diversity falls 0.162 → 0.059, which is the honest number for eight
openers funnelling to 2.45 distinct sets.

**F2 — "provably inert" was not a proof. MAJOR (claim), fixed.** The singleton
fallback's inertness rested on *"the chain's first pick is the argmax
singleton"*. `_chain` starts from a **forced opener**, so that holds on **20 of
160** main-suite chains. The conclusion survives as a *measurement* — 0/160 fires
at `max_atoms = 8` — but it fires **60/160 at a budget of 3**, so inertness is a
property of a frozen value `CLAUDE.md` §6 prices for change, not a theorem. This
is §5's "asserting math without checking it", in a docstring I wrote. The claim
is restated, and the test now asserts **both** halves, so it can distinguish
"inert" from "deleted" — which the previous version could not (removing the whole
fallback block passed 39/39).

**F3 — a self-referential test, and `α` unprotected. MAJOR, fixed.**
`test_s3_objective_on_a_hand_computable_case` recomputed the implementation's own
internals (`obj.sim`, `obj.cap`, `obj._repr_total`, `obj._div_total`) and
asserted equality — proving only that an expression equals itself. Worse, on the
chosen atom three of its four terms were `0 == 0`. Mutations that survived
39/39: **deleting the saturation `α` entirely**, doubling `rel_term`, inverting
the `QueryCov` predicate — so decision 7's own named risk (*"α is silently
defaulted"*) was unprotected. Replaced by a three-atom toy pool with orthogonal
unit features whose every expected value is arithmetic done on paper, plus a
test that `α` restrains near-duplicates.

**That new test immediately found a real bug**: with no active obligation slots
`qry_term` returned 1.0 for the **empty** set, so `F(∅) = w_qry = 0.5` — breaking
the `F(∅) = 0` normalisation the paper states for every term. Narrow (lattice
questions always have active slots) but real, and exactly the class of defect a
self-referential test cannot see.

**F4 — `upper_95` published on the negated scale. MINOR, fixed.**
`higher_is_better_bootstrap` re-negated `mean_difference` but not `upper_95`, so
a strictly better `a` reported `mean +0.4` against an "upper" bound of `−0.4`.
No verdict was ever wrong (`wins` is defined on the negated scale), but the point
estimate sat on the far side of its own interval. Now published as
`lower_95_on_a_minus_b` — the correct one-sided bound in the reader's frame —
with the negated value kept under an explicit name for the `wins` audit trail.

**F5 — the exactness reduction was missing its precondition. MINOR, fixed.**
"Uniform edge costs collapse the problem" is true only **after** Mapping A drops
edge prizes: under Eq. 7 as written, an edge with `prize(e) > C_e` is worth
including even when it closes a cycle, so the optimum is not a tree — which is
precisely why the paper needs its virtual-node construction. The code was
correct; the stated derivation omitted the load-bearing condition.

**F6 — the gold-spans-components test measured a graph nobody uses. MINOR,
fixed.** It called `reference_components(gold, pool)` — components *induced on
gold alone* — which is >1 on **100%** of instances, so `assert spans >= 1` could
never fail. Decision 2's 8/20 figure is on the pool and admissible graphs; the
test now measures the admissible graph S4 is actually handed, and asserts the
rate is neither 0 nor universal.

**F7 — per-method structural numbers were computed and dropped. MINOR, fixed.**
`completion_rate`, `max_atoms_breaches`, `h_rejections` and `E[U]` of returned
sets reached the rows and were discarded by `summary()`, so criteria 11c/12/12d
were computed but not *reported*. Now aggregated. `target_fingerprint` was
likewise absent from the report although criterion 14 names both digests — the
ceiling and global-max rows are `p*`-derived, so a `u_weights` move would have
passed unseen. Now recorded per instance.

**F8 — `nanmean` over a self-selected population. MINOR, disclosed.** S3's
obligation best-of-K is a mean over the **12** instances it survived, not 20. The
refutation pass established the bias runs *against* S3 rather than for it (the
unpaired reading gives −0.0161 against the paired −0.0135), so this is a
disclosure issue, not a correction: `best_utility_scored_on` and
`best_utility_mean_failures_as_zero` are now published beside every mean.

**Refuted, and deliberately not acted on: S3's unit cost divisor.** The
paper-fidelity lens argued that since a greedy move adds a whole *closure*, the
cost-scaled ratio should divide by the atoms actually consumed (73% of moves
consume more than one). The refutation pass measured the consequence: true-cost
greedy **collapses S3's validity** from 12/20 to **3/20** instances with any
valid set, and the quoted improvement was a `nanmean` over 3 successes against
one over 12. `cost = 1.0` is also the ruled cell. Shipped S3 is the *more*
dangerous baseline by ~4× on `CLAUDE.md` §8's own criterion, so the unit divisor
stays — with §3.3's "cost-scaling degenerates" now qualified rather than asserted.

---

## 2. Gate-3 Stage A — measured

Main suite, 20 instances, exact `U` as the scorer (fix F13), `K = 8`.
**`E[best-of-K | p*]` = 1.8865, mean global max `U` = 1.9245** — both reproducing
G9's independently measured values exactly, from this implementation.

Diversity is over the **portfolio with multiplicity**, which is the convention
decision 5 pins; `best-of-K` is a mean over the instances a method survived, with
that count and the failures-as-zero reading published beside it (§4.1).

**Obligation relevance** (`C_e = 2.0`):

| method | best-of-K | scored on | as-zero | distinct | diversity | excess div. | checks | size |
|---|---|---|---|---|---|---|---|---|
| S1 greedy | **1.9245** | 20/20 | 1.9245 | 2.45 | 0.059 | −0.759 | **0** | 7.08 |
| S2 beam | **1.9245** | 20/20 | 1.9245 | 8.00 | 0.225 | −0.578 | **0** | 7.26 |
| S3 submodular | 1.8704 | **12/20** | 1.1222 | 1.20 | 0.145 | −0.322 | 4.90 | 8.00 |
| S4 PCST | 0.5181 | 20/20 | 0.5181 | 1.15 | 0.033 | −0.923 | 4.85 | 1.15 |

**Informed relevance** (`C_e = 2.0`):

| method | best-of-K | scored on | as-zero | distinct | diversity | excess div. | checks | size |
|---|---|---|---|---|---|---|---|---|
| S1 greedy | **1.9245** | 20/20 | 1.9245 | 2.45 | 0.059 | −0.759 | **0** | 7.08 |
| S2 beam | **1.9245** | 20/20 | 1.9245 | 8.00 | 0.225 | −0.578 | **0** | 7.26 |
| S3 submodular | 1.8789 | 20/20 | 1.8789 | 2.25 | 0.222 | −0.517 | 4.60 | 8.00 |
| S4 PCST | **1.7217** | 20/20 | 1.7217 | 4.50 | **0.483** | −0.448 | 5.00 | 3.78 |

**What reproduces the plan, from this code:** greedy is globally optimal on
**20/20** (S1's best-of-K *is* the global max); S1 returns **2.45** distinct
(range 2–3, 20/20); S2 returns **8.00** distinct with diversity **0.225**; the
ceiling sits **0.0380 below** greedy; the spend splits **0 / ~4.9** exactly along
the mask-driven vs direct-builder line.

**The budget curve** (criterion 10's first half, closed 13 Aug 2026; levels
`{1, 2, 4, 8}` per G3 — the frozen 32 binds on nobody here). Two readings per
method, both published (§1.4 F8's rule): the conditional mean over surviving
instances, and failures-as-zero, which is monotone whenever the candidate sets
nest (S1/S3's openers and S4's top-k sweep are prefixes of the next level).
Informed relevance, main suite:

| method | best-of-k, failures-as-zero @ {1,2,4,8} | scored on | checks @ {1,2,4,8} |
|---|---|---|---|
| S1 greedy | 1.9245 · 1.9245 · 1.9245 · 1.9245 | 20 at every k | 0 · 0 · 0 · 0 |
| S2 beam | 1.9245 · 1.9245 · 1.9245 · 1.9245 | 20 at every k | 0 · 0 · 0 · 0 |
| S3 submodular | 0.6571 · 1.2120 · 1.4130 · 1.8789 | 7 → 14 → 16 → 20 | 1.0 · 1.7 · 2.95 · 4.6 |
| S4 PCST | 0.6960 · 0.8172 · 0.9345 · 1.7217 | 20 at every k | 1.0 · 1.4 · 2.2 · 5.0 |

The ceiling rises 1.5963 → 1.7415 → 1.8342 → 1.8865. **The shape is G3's
predicted finding**: flat at the optimum for the mask-driven arms (greedy's
first opener already finds the global max), climbing steeply for the direct
builders — S3 needs the full portfolio to survive on every instance at all, and
S4's utility more than doubles from k = 1 to 8. A *conditional-mean* dip (S3
under **informed** relevance: 1.8775 → 1.7314 from k = 1 to 2 as its scored
population grows; the obligation variant's conditional mean happens to be
monotone, since its population holds at 7 across that step) is a population
effect — more instances survive at higher k and drag the mean down — not a
search regression, which is why the zero-filled column is the one the shape
claims are made on.

**What the build found that reading did not:**

1. **Decision 1's two variants are load-bearing, and the informed one is what
   makes S4 a real baseline.** Under the proxy S4 scores 0.518 with size-1.15
   outputs; under the informed variant it scores **1.7217** with 4.50 distinct
   sets — the strongest portfolio of any training-free arm. Reporting only the
   proxy would have shipped a strawman, which is the failure G1 exists to
   prevent.
2. **S3 returns nothing valid on 8/20 instances under the proxy**, and the
   rejection categories say why: duplicate `answer` bindings (sub-check 9) and
   temporal contradictions. The submodular objective has **no analogue** of "one
   binding per slot", so a direct builder cannot see the constraint. That is a
   genuine measurement of the gap between a published packer's objective and
   formal proof validity — and it is exactly the cost the plan intends S3 to pay
   for bypassing the masks. **Not patched.** `h_rejections` is reported per
   method.
3. **`checker_budget = 32` is inert**, as G3 predicted: the most any method
   spends is 5 of 8. The budget curve's levels are `{1, 2, 4, 8}` and the frozen
   32 constrains nobody here; it binds at Phase 9.

---

## 3. Decisions the plan left to the implementation

*(§3.1 and §3.2 were built as judgement calls and flagged for a ruling; **both
were RULED 13 Aug 2026 — affirmed as implemented**, by the project's delegate at
the owner's instruction. The reasoning below is the ruling's basis; the
alternative in each case manufactures a strawman, which plan §5.1's matched-
baseline discipline and G1's "defining `rel` badly is how baselines get
accidentally weakened" both forbid.)*

### 3.1 Direct builders get the **admissible-atom** ground set — RULED

S3 and S4 bypass the `ADD` masks, so without a per-atom pre-filter they spend
their whole portfolio on atoms carrying a *permanent* violation — a retired edge,
a quarantined assertion. Phase-2 exit criterion 17 requires those to be *"present
in the snapshot and asserted **unreachable**… negative cases for sub-checks 4 and
7, **not selectable evidence**"*, and the `ADD` masks prune them for S1/S2/S5 at
every state.

Handing S3 and S4 the same view is the fair comparison, not a concession — and
Phase-4 decision 2 already rules it for S4's graph, so extending it to S3 is
consistency rather than novelty. It costs **zero** terminal checks: per-atom
admissibility is construction-time validity (Phase-0 G1, G6).

Measured: 12–14 admissible atoms out of pools of 20–25.

### 3.2 Closure completion applies to S3, not only S4 — RULED

Decision 2 gives S4 an explicit completion step. Without the same repair S3
returned **0 valid sets on every instance** (measured: 40 closure violations
across 11 candidates on the tuning suite), because a direct builder selects an
edge atom without its endpoints.

Closure is a **feasibility rule of the environment**, not part of anyone's
objective, and denying S3 the repair while granting it to S4 would decide S3's
strength by omission. Completion is shared in `base.close_under_refs`.

### 3.3 Feasibility is tested on the **completed** set

The paper's own greedy adds *"the feasible snippet"* subject to *"cost(S) ≤ B and
a snippet cap"*. What S3 actually returns is the *closed completion* of its
selection, so a candidate is feasible when its **completion** fits `max_atoms`.
Testing feasibility pre-completion would put every candidate over the cap and
into `H`'s size rejection — the straddle of §1.2 in a second place.

Atom **cost stays 1** (the ruled cell): cost-scaling still degenerates, and S1
and S3 still differ by objective rather than by search strategy.

### 3.4 The Lin–Bilmes singleton fallback is **measured** inert at this cap

Build step 4's done-when was *"singleton fallback fires at least once in the
suite"*, which does not happen at `max_atoms = 8`: measured **0 fires / 160
chains**.

**The first version of this section called that provable, and it is not** — see
§1.4 F2. The claimed argument needed the chain's first pick to be the argmax
feasible singleton; `_chain` starts from a *forced opener* (decision 3), so that
holds on 20 of 160 chains. Shrink only the budget and the fallback fires:
**60/160 at 3**, 11/160 at 2. Inertness is therefore a property of a frozen value
`CLAUDE.md` §6 explicitly prices for change, and the test asserts both halves so
it can tell an inert fallback from a deleted one.

### 3.5 QueryCov and Div are re-defined, and labelled

The paper's `QueryCov` is *"a set-cover over distinct query content terms"* and
`Div` is *"concave-over-documents"*. The lattice has neither. **[ANALYSIS]**
`QueryCov` becomes a coverage function over the **obligation slots** (Phase-1
G4's four active slots) — monotone submodular exactly as the paper's is — and
`Div`'s "documents" become the atom's resolved `Source` node, with unresolvable
atoms sharing one bucket rather than each becoming its own (which would hand
unsourced atoms a free diversity bonus).

`Repr` uses the project's **own** clamped-cosine similarity matrix — the same one
`U.redundancy` uses, cached per pool (Phase-2 G7). Sharing it is deliberate: S3's
coverage term and the reward's redundancy term are the same objective family,
which is what makes them comparable at all.

Verified: `F(∅) = 0`, `F(V) = Σ weights = 2.2`, and no submodularity violation in
200 random nested pairs.

### 3.6 The search package joins `_ADAPTER_LAYER`, deliberately

P4.1 anticipated this: search modules are **not** learners, and S3/S4 *must* see
the pool and its `refs`. The Phase-3 closed-list guard went red the moment the
package appeared — working as designed — and the package is now its own recorded
list (`_SEARCH_LAYER`), so a *learner* smuggled in under `search/` cannot become
exempt from exit criterion 6 by filing location alone.

---

## 4. Exit criteria — all sixteen

**44 tests** in `graft/tests/test_search.py` (two more landed with the post-fix
audit: a QueryCov fixture with active slots, and a call-site guard pinning
Stage-A diversity to the multiplicity portfolio — the two coverage gaps that
audit's mutation pass confirmed). The review round (§1.4 F7) found
criteria **10 (the budget curve's shape), 11c, 12 and 12d** computed and reported
but with no test that would fail on regression — the earlier claim that "every
criterion has an enforcing test" was false at that point. Closed 13 Aug 2026:
`budget_curve` computes criterion 10's first half (measured in §2) and two
dedicated guards (`test_the_budget_curve_is_computed_over_the_declared_levels`,
`test_the_structural_numbers_survive_into_the_summary`) fail if the curve, the
structural aggregates or the per-instance fingerprints drop out of the artefact.

The three tests worth naming:

* **Criterion 1** runs its `H` verification with `ledger=None`. `checker.H`
  increments `terminal_checks` on every ledgered call, so verifying through the
  live ledger would charge the mask-driven arms a check each and erase the
  0-versus-1 family difference criterion 6 exists to expose.
* **Criterion 2**'s named case — a PCST output that is *connected but not
  closed* — is a regression test built from a real edge atom and one endpoint.
* **Criterion 11**'s bootstrap is checked in **both** directions: the negation is
  what stops `wins = upper < 0` reporting the winner as the loser on a
  higher-is-better metric.

---

## 5. Open

| # | Item | Why it matters |
|---|---|---|
| 1 | **Stage B has not run** — though its blocker lifted 15 Aug 2026: Phase 3's 27 checkpoints exist and load (`PHASE3_DECISIONS.md` §7.5). S5's row is still empty, this is still **not** Gate 3 (G7, criterion 15), and **no document names which arm's checkpoints fill S5** — that ruling is owed before Stage B runs, and Gate 2 being `inconclusive` removes the obvious "winning arm" default | The table is labelled Stage A and says so in its own `caveats` block |
| 2 | **Decision 5's live condition is predetermined** under the ruled metric (§1.3) | Phase 9 should adopt the size-controlled form; changing the gate is a §6b amendment |
| 3 | **Decision 8's text still says "`pcst_fast`, pinned"** and "median … without exceeding" | Both overturned in §1.1/§1.2; the plan cells are annotated, but the *ruling* wants re-signing |
| 4 | S3's `h_rejections` show the objective cannot express sub-check 9 | A real finding for the write-up, not a defect to patch |
| 5 | `pcst_fast` is installed but used **only** by the test that documents its breakage | If it is ever removed, the regression test skips rather than fails |
| ~~6~~ | ~~The budget curve over `{1, 2, 4, 8}` is not computed~~ **CLOSED 13 Aug 2026** — `gate3.budget_curve`, measured in §2, guarded by `test_the_budget_curve_is_computed_over_the_declared_levels` | — |
| ~~7~~ | ~~Criteria 11c/12/12d are reported but not test-guarded~~ **CLOSED 13 Aug 2026** — `test_the_structural_numbers_survive_into_the_summary` | — |
| 8 | §3.1 and §3.2's judgement calls | **RULED 13 Aug 2026, affirmed as implemented** (see §3's header note) |

---

## 6. What Phase 9 gets

* `SearchModule` — the same protocol, with the distilled head substituted for
  exact `U` (fix F13). The methods do not change; their scorer does.
* **The training-free frontier under a perfect scorer**: best-of-K 1.9245
  (S1/S2), the `p*` ceiling at 1.8865, and the measured fact that greedy is
  globally optimal — the optimistic bound S5 must clear under a *noisy* scorer.
* Ledger discipline already exercised on the inference path, with the
  0-versus-1 family split measured rather than assumed.
* **The question Phase 4 could not answer, and why** (G9): under fix F13's
  perfect scorer this environment cannot discriminate on best-of-K. Phase 9 is
  where Robust Scheduling's precondition holds, and where the comparison should
  be re-asked.
