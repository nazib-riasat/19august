# GRAFT — Phase 4 Build Plan: the five Tier-1 search algorithms (`graft/setgen/search/`)

**Gate 3's synthetic kill-shot, run early and on purpose.**

Date: 12 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 4, fixes F5/F10/F13) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §5.2, §6.4, Gate 3) · `GRAFT_PHASE3_BUILD.md` §8 · `PHASE3_DECISIONS.md` §5
Effort: ~1.5 weeks **[ANALYSIS]** — an estimate, not a measurement. Cheap in compute: four of the five methods are training-free.
Status: **§6 unsigned.** Nothing here has run.

Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps are numbered **G1–G8**, matching the Phase-0/1/2/2.5/3 convention.

---

## 0. What this phase is for, and what it is not

**The question.** Does the learned set constructor beat *training-free* search at
equal budget? `CLAUDE.md` §8 names the two most likely to embarrass the project —
**submodular greedy** (0.451 F1 against 0.429 for a tuned heuristic on multi-hop
HotpotQA) and **PCST** (G-Retriever, NeurIPS 2024, whose connected subgraph is
structurally the closest classical object to a proof) — and says: *run them
early*. This is that.

**Why it comes before the data phases.** Plan §7 calls Gate 3 *"the highest-risk
gate"*: if PCST, MIP or submodular greedy matches the learned constructor at
equal budget, **Stage D's learning contribution does not hold** and the project
consolidates around Contribution 1. Learning that on a lattice, in 1.5 weeks,
costs a fraction of learning it in Phase 9 on real data after Phases 5–8 are
built on the assumption it held.

**Designed so a negative result is as publishable as a positive one**, exactly as
Phase 3 was. If S3 or S4 matches S5 here, that is a finding with a named
consequence, not a failure of the phase.

**Two caveats the write-up must carry, both inherited.** Fix F13: on the lattice
`U` is *exactly computable*, so all five methods score with a **perfect scorer** —
fair across methods, and **optimistic relative to deployment**, where Phase 9's
distilled head is noisy. And this environment is a lattice, not a memory graph:
Gate 3's synthetic table is a kill-shot, not a coronation.

---

## 1. Eight gaps this phase must close

### G1 — `relevance` has no source on the lattice, and it decides how strong S3 and S4 are [ANALYSIS]

S3's objective opens with `1.0·rel` and S4's PCST needs **node prizes**. Both are
*relevance*, which in the finished system comes from Stage C's hybrid retrieval —
**Phase 7, which does not exist.** Verified: a pool atom carries a 12-dimensional
feature vector and no relevance channel.

This is not a detail. `CLAUDE.md` §8 says these two baselines are the ones most
likely to embarrass the project, so **defining `rel` badly is how they get
accidentally weakened**, and a Gate-3 pass bought that way is worthless. Three
candidates, and the choice must be declared rather than defaulted into:

| Option | What it makes S3/S4 | Objection |
|---|---|---|
| **`rel` from `U`'s per-atom marginal** | maximally strong — the exact utility, through the objective | gives S3/S4 the *scorer* inside their own objective; they are then optimising the thing they are scored on, which no deployed retriever does |
| **`rel` from obligation match** (does this atom address an outstanding slot?) | realistic — the signal Stage C would actually produce | is a proxy, so a loss is arguable as "the proxy was weak" |
| **uniform `rel`** | degenerate | S3 collapses to coverage-only; S4's prizes carry no information |

**Recommended: obligation match, with the `U`-marginal variant run as a declared
"informed" arm.** Reporting both closes the escape hatch in either direction — if
S5 wins, it wins against a relevance-informed S3/S4; if S5 loses, the loss is not
explained away by a weak proxy. **[ANALYSIS]** the cost is one extra column, not
one extra method.

### G2 — PCST's connected output is **not** a closed set, and fix F10 says it is

Architecture fix F10 states: *"PCST's connected output maps to a closed set with
no conversion logic."* **That is false, and it is checkable in one line.**

Closure requires that every atom's `refs` are present. Connectivity in the
reference graph is strictly weaker. Measured on the first benchmark instance:

```
atom 58f7e4a0… (kind=edge) refs=[6d9cf606…, 870232c0…]
{58f7e4a0…, 6d9cf606…}  is CONNECTED     (the edge touches that endpoint)
                        is NOT CLOSED    (870232c0… is referenced and absent)
```

An edge atom with two referenced endpoints is connected to a subgraph containing
either one; closure needs **both**. So PCST needs an explicit **closure
completion** step, and completion has consequences the plan must fix: it adds
atoms, which can breach `max_atoms`, and it changes the set PCST's own objective
selected. Decide and record: complete-then-`H`-filter, or drop atoms whose
references PCST omitted. **[ANALYSIS]** Completion is the better default — dropping
an edge discards exactly the connective evidence PCST was chosen for — but it must
be the declared one, and the completion rate must be reported, because a high rate
means PCST's objective and this environment's feasibility rule disagree.

### G3 — A fixed checker budget is only an axis if every method can spend it [ANALYSIS]

The primary is *"best-of-K valid-set utility at a fixed checker-call budget"* and
the budget is 32 terminal checks. But:

| Method | Completed sets it naturally produces |
|---|---|
| S1 greedy | **1** — deterministic, one pass |
| S2 beam (width `b`) | up to `b` |
| S3 submodular greedy | **1** |
| S4 PCST | **1** |
| S5 portfolio | `K = 8` |

At a budget of 32, greedy spends **one** check and stops. Comparing "best of 1"
against "best of 8" and calling it *equal budget* is not a comparison — it is a
statement that one method was allowed to look 8 times and another once. Two
honest resolutions, and the plan must pick one:

1. **Give every method a way to consume the budget** — randomised tie-breaking
   with restarts for S1/S3, multiple PCST solves under prize perturbation for S4 —
   so all five return a portfolio and best-of-K is over comparable K.
2. **Report the budget each method actually spends**, plot performance against
   budget level as v1.2 §5.2 requires ("plot performance against several budget
   levels rather than reporting a single point"), and state plainly that the
   deterministic methods saturate at 1.

**Recommended: both.** (1) makes the headline number comparable; (2) is what the
plan already asks for and is what shows a reader *where* each method saturates.
**[EVIDENCE]** v1.2 §5.2's corrected protocol is explicitly "one primary budget,
plotted across levels", not a single point.

### G4 — "Three seeds" is vacuous for three of the five methods [ANALYSIS]

S1, S3 and S4 are **deterministic** given `(pool, obligations, scorer)`. The
frozen seed set `{13, 42, 7}` varies nothing in them, so three seeds means three
identical rows and any interval computed over them is zero-width — which would
make a paired test against S5 look impossibly significant.

State per method what a seed *is*:

| Method | What the seed varies |
|---|---|
| S1, S3 | tie-breaking among equal-gain atoms, **only if** G3's restart option is adopted; otherwise nothing |
| S2 | tie-breaking at beam boundaries |
| S4 | nothing, unless G3's prize perturbation is adopted |
| S5 | the sampler — the seed is the trained model *and* the draw |

**[ANALYSIS]** Where a method is deterministic, report it once and say so, rather
than printing three copies and inviting a variance estimate that does not exist.

### G5 — Gate 3 has no decision rule, which is the defect fix F12 retired for Gate 2

The architecture's Phase-4 exit is *"Explicit written decision: does the learned
sampler beat S3/S4 on the lattice?"* — with **no threshold, no test, no
predeclaration**. That is exactly the unfalsifiable rule fix F12 retired for
Gate 2, one gate over.

The remedy already exists and is built: `gate2.paired_bootstrap` is a hierarchical
paired bootstrap over seeds and instances, frozen at 10,000 resamples and seed
`20260814`. Phase 4 reuses it verbatim. **Predeclared here, before any run:**

> **S5's learning claim holds only if S5 beats S3 *and* S4 on the primary — the
> one-sided 95% upper bound on `best-of-K utility difference` — under the same
> paired test Gate 2 uses.** If either training-free method matches it, Stage D's
> claim narrows and that narrowing is written into the plan **before** Phase 9
> begins.

The asymmetry with Gate 2 is deliberate and worth stating: there, the proposed
method had to beat *controls*; here it must beat *baselines that cost nothing to
run*. **[EVIDENCE]** Dror et al. (ACL 2018) — a decision rule chosen after seeing
results is not a decision rule.

### G6 — Phase 4 is the first phase that meters, and the counting rule needs to be exact per method

`PHASE3_DECISIONS.md` §5: *"`ledger=None` does not carry over. Phase 4 **is** the
inference path, and `checker_budget = 32` terminal checks per query is enforced
there (`would_exceed()` before spending, never observed after)."* `Ledger` already
provides `from_config`, `would_exceed`, `count`, `remaining` and `query_scope`, and
`terminal_checks` is incremented **inside** the Phase-1 checker.

`CLAUDE.md` §6's two counting rules are the ones that go wrong quietly:

* `terminal_checks` counts **full validation of a completed candidate set only** —
  construction-time validity is incremental and free (Phase-1 G1). So S1's per-step
  `stop_allowed` costs nothing; only its final set costs 1.
* The budget must be **enforced, not observed** — `would_exceed()` *before*
  spending, so a method stops cleanly instead of overrunning and being noticed
  afterwards.

Consequence for S3 and S4, which bypass the `ADD` masks and build sets directly:
they get **no free incremental validity at all** and pay a full check per
candidate. That is a real cost difference between method families, and it belongs
in the table rather than in a footnote.

### G7 — S5 depends on a checkpoint that does not exist yet [ANALYSIS]

S5 is *"sample K from the trained sampler, `H`-filter, rank by scorer"*, and
Phase-3 §8 requirement 1 makes it a consumer of a Gate-2 checkpoint. Phase 3's
calibration and matrix have not run, so **no frozen-`N` checkpoint exists**.

`graft.setgen.policy.load_policy` is built and round-trips, so S5's *code* has no
blocker. Its *number* does. The phase therefore has a **two-stage exit**:

* **Stage A (now):** S1–S4 built, metered, and compared against each other. Every
  gap above closed. The Gate-3 table exists with the S5 row empty.
* **Stage B (after Phase 3's matrix):** S5's row filled from the checkpoints, and
  the predeclared G5 decision applied.

Recording this as a two-stage exit is what stops Stage A being reported as Gate 3.

### G8 — `pcst_fast` is not a dependency, and it may not need to be [ANALYSIS]

`requirements.txt` names `pcst_fast` only in a comment saying it "arrives" with
the ML extras; it is not pinned anywhere. Before adding a compiled dependency,
note the scale: `pool_cap = 64` and the measured lattice pools are **20–26 atoms**.
At that size an exact PCST is tractable directly, and the reference solver's value
is that it is *the standard one* rather than that it is fast.

Decide and record: pin `pcst_fast` (matching G-Retriever's formulation exactly, at
the cost of a build dependency) **or** implement the PCST objective directly and
verify it against `pcst_fast` on a handful of instances offline. **[ANALYSIS]** the
first is more defensible in a write-up; the second removes a platform risk on a
Windows laptop. Either is fine; leaving it undecided until S4 is being written is
not.

---

## 2. Scope

**In.** `graft/setgen/search/`: the `SearchModule` protocol, a declared relevance
function (G1), S1 greedy, S2 beam, S3 submodular greedy, S4 PCST with closure
completion (G2), S5 portfolio over a Phase-3 checkpoint, and the Gate-3 harness
reusing `gate2.paired_bootstrap` and `audit_block`. Ledger enforcement (G6).

**Out.** Real data. Stage C retrieval — S3/S4's `rel` is defined here as a
declared stand-in (G1), not built. The distilled utility head (Phase 9). Tier-2
search: **MIP, MCTS and diverse beam stay stubs** (v1.2 §2.4). **No change to
`H`, `U`, `R`, the masks, `d(s)`, the lattice generator, the exact evaluator, or
any Phase-3 learner.** Phase 4 consumes Phases 1–3's instruments; it does not
adjust them.

---

## 3. Modules

### P4.1 `search/base.py` — the protocol every method implements

```python
class SearchModule(Protocol):
    def run(self, env, obligations, scorer, ledger) -> SearchResult: ...
```

`SearchResult` carries the returned sets, the best-of-K utility, **the terminal
checks actually spent**, and whether the budget was exhausted. Spend is a returned
value, not a side effect to be read off a global, so a method that under-spends is
visible in its own row (G3, G6).

**What a search module may see, and why it differs from a learner.** Phase-3's
fix-F6 boundary keeps `learners/` away from `StateGraph`, `LatticeInstance` and
atom ids. **Search modules are not learners** — S3 and S4 *must* see the pool and
its `refs` to build sets directly, and fix F10 makes that legal precisely because
`H` re-checks closure afterwards. They sit on the adapter side of Phase-3's
`_ADAPTER_LAYER` list, and that list is a closed one, so adding this package is a
deliberate recorded act rather than a drift.

### P4.2 `search/relevance.py` — G1's declared stand-in

Two functions, both pure: `obligation_relevance(atom, obligations)` and
`utility_marginal_relevance(atom, env)`. The second is the "informed" variant and
must be *labelled* wherever it appears, because it hands S3/S4 the scorer.

### P4.3 `search/s1_greedy.py`, `search/s2_beam.py`

S1: iteratively `ADD` the argmax `scorer` gain; `STOP` when allowed and gain < 0.
S2: beam width `b = K` over partial sets, dedup by `canon_set_hash` (Phase 0).
Both go through the Phase-1 masks, so their construction-time validity is free
(G6) and their outputs are closed by construction.

### P4.4 `search/s3_submodular.py`

**[EVIDENCE]** objective, weights and algorithm from arXiv 2607.00725 (*provisional
venue, declared*): `F(S) = 1.0·rel + 0.5·query-coverage + 0.4·saturated
facility-location + 0.3·concave-over-source-diversity`, cost-scaled greedy with
the Lin & Bilmes (ACL 2011) singleton fallback, then `H`-filter.

**The weights are the paper's and are not tuned here.** Tuning them would make S3
a comparison of tuning effort, which is the objection v1.2 §5.2 raises against
unequal hyperparameter budgets.

**One inherited guarantee does not transfer, and the write-up must not claim it.**
Nemhauser–Wolsey–Fisher (1978) gives `(1 − 1/e)` for a monotone submodular
function under a **cardinality** constraint. Under the closure rule the feasible
family is not a cardinality constraint — it is the closed sets of a dependency
relation — so the guarantee is **not established here**. Report S3 as the
published *algorithm*, not as an approximation-ratio-bearing one. **[ANALYSIS]**
this is `CLAUDE.md` §5's "asserting math without checking it", pre-empted.

### P4.5 `search/s4_pcst.py`

Node prizes from `relevance`, uniform edge costs, unrooted, over the **reference
graph** of the pool. Then **closure completion** per G2, then `H`-filter. Reports
the completion rate and the `max_atoms` breach rate.

### P4.6 `search/s5_portfolio.py`

`load_policy(checkpoint)` → sample `K` with `greedy=1` (fix F5's 1 greedy + 7
sampled, already implemented in `sample_trajectories`) → `H`-filter → rank by
`scorer`. **No trainer import** — Phase-3 §8 requirement 1 is that the checkpoint
loads without it, and this module is what proves it.

### P4.7 `search/gate3.py`

The matrix over five methods × instances × seeds, the ledger accounting, the
budget-level curve of G3, `gate2.paired_bootstrap` for the predeclared G5 rule,
and `gate2.audit_block` so every Phase-4 table carries the same audits from the
same Phase-2 source.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 1 | P4.1 protocol + ledger wiring | a method that overspends raises; `would_exceed()` is called before every check (G6) |
| 2 | P4.2 relevance, both variants | both are pure functions of `(atom, obligations \| env)` and are unit-tested against hand cases |
| 3 | P4.3 S1 + S2 | outputs are closed and `H`-valid by construction; spend is 1 and ≤ `b` |
| 4 | P4.4 S3 | reproduces the paper's objective on a hand-built example; singleton fallback fires at least once in the suite |
| 5 | P4.5 S4 + closure completion | **a connected-but-unclosed PCST output is completed and passes `H`** — the G2 case, as a regression test |
| 6 | **Gate-3 Stage A**: S1–S4 across the main suite, budget curve, audits | the table exists with S5 empty and says so |
| 7 | P4.6 S5 (code only, no checkpoint yet) | loads a Phase-3 checkpoint without importing the trainer |
| 8 | **Gate-3 Stage B**, *after Phase 3's matrix* | S5's row filled; G5's predeclared rule applied and the outcome written |

Steps 1–7 need nothing from Phase 3's runs. Step 8 is the only one that waits.

---

## 5. Exit criteria

**The methods are correct**
1. Every returned set passes `H` — the filter is not assumed, it is asserted per method per instance.
2. S1 and S2's outputs are closed *by construction* (they use the masks); S3 and S4's are closed *after* their filter/completion — and a test exercises the G2 case where PCST's raw output is connected but not closed.
3. S3 reproduces the arXiv 2607.00725 objective on a hand-computable example, with the paper's weights unmodified.
4. S5 loads a checkpoint through `graft.setgen.policy` alone, with no import of `graft.setgen.trainer` (Phase-3 §8 requirement 1), asserted by an import test.

**The comparison is fair**
5. `checker_budget = 32` and `K = 8` are the same constants Phase 3 used (fix F5), read from config, not restated.
6. The budget is **enforced** — `would_exceed()` before every spend — and each method's *actual* spend is reported beside its result (G3, G6).
7. Every method's seed semantics are declared, and a deterministic method is reported once rather than as three identical rows (G4).
8. The relevance variant is named in every table: `obligation` or `informed` (G1).
9. All five score with **exact `U`** (fix F13), and the write-up carries the "perfect scorer, optimistic relative to deployment" caveat.

**Gate 3**
10. Best-of-K valid-set utility at `checker_budget`, per method, across the main suite, **plotted across budget levels** and not only at 32 (v1.2 §5.2, G3).
11. The **predeclared G5 rule** is applied with `gate2.paired_bootstrap`: S5 must beat S3 **and** S4 on the one-sided 95% upper bound.
12. PCST's closure-completion rate and `max_atoms` breach rate are reported (G2).
13. Audits carried into every table from `gate2.audit_block` — `FAIL` rate, collision rate (0), unconstructible rate (0), `Δd` densities, target-mass profile, `neither`-mass at the run's β.
14. Every run records the `environment_fingerprint` and `target_fingerprint` of the suite it used (Phase-2 decision 21).
15. **The two-stage exit is honoured** (G7): a table without S5 is labelled Stage A and is not called Gate 3.

**The decision**
16. A written decision, before results are inspected: **if S3 or S4 matches S5 under the G5 rule, Stage D's learning claim narrows, and the narrowing is written into the plan before Phase 9 begins.** Plan §7 calls Gate 3 the highest-risk gate; this is the sentence that makes it act like one.

---

## 6. Decisions to lock before writing code — **UNSIGNED**

| # | Decision | Value | Cost if changed later |
|---|---|---|---|
| 1 | **Relevance on the lattice** | **[recommended]** obligation-match as primary, `U`-marginal as a declared "informed" variant, both reported (G1) | the two baselines `CLAUDE.md` §8 calls most dangerous are silently weakened, and a Gate-3 pass means nothing |
| 2 | **PCST closure policy** | **[recommended]** complete-then-filter, with completion rate and `max_atoms` breach rate reported (G2) | fix F10's "no conversion logic" is carried into code where it is false |
| 3 | **Budget spending** | **[recommended]** restarts/perturbation so every method can consume the budget, **and** the budget-level curve v1.2 §5.2 asks for (G3) | best-of-1 is compared with best-of-8 under the label "equal budget" |
| 4 | **Seed semantics** | declared per method; deterministic methods reported once (G4) | a zero-width interval makes a paired test against S5 look impossibly significant |
| 5 | **The Gate-3 decision rule** | S5 beats **S3 and S4** on the one-sided 95% upper bound, `gate2.paired_bootstrap`, seed `20260814`, 10,000 resamples — predeclared (G5) | Gate 3 repeats the unfalsifiable-rule defect fix F12 retired for Gate 2 |
| 6 | **Ledger discipline** | `would_exceed()` before every spend; completed-set checks only; spend reported per method (G6) | methods drift over budget and the comparison is unfair without anyone noticing |
| 7 | **S3's weights** | the paper's, unmodified, `[EVIDENCE]` provisional venue declared; **no `(1 − 1/e)` claim** (P4.4) | S3 becomes a comparison of tuning effort, or an unearned guarantee enters the write-up |
| 8 | **PCST solver** | pin `pcst_fast`, **or** implement directly and verify against it offline — decided before S4 is written (G8) | a compiled dependency lands mid-build on a Windows laptop |
| 9 | **Two-stage exit** | Stage A (S1–S4) may be reported; only Stage B is Gate 3 (G7) | a table without the learned sampler is read as the gate |

---

## 6b. Declared risks

| Risk | If it bites |
|---|---|
| **[HYPOTHESIS]** S3 or S4 matches S5 | **A designed outcome.** Stage D's learning claim narrows; the thesis leans on Contribution 1 and the five-ceiling protocol. This is the cheapest place in the project to learn it, which is why it is here and not in Phase 9 |
| S5 wins only because `rel` was weak | Decision 1's "informed" variant exists to remove that explanation. If S5 beats the informed variant too, the objection is closed |
| The lattice flatters the learned sampler | It is a proof lattice with designed modes, not a memory graph. Gate 3's synthetic table is a **kill-shot, not a coronation**, and the Phase-9 table under a noisy scorer is the one that reflects deployment (fix F13) |
| PCST's completion rate is high | Its objective and this environment's feasibility rule disagree, and S4's numbers describe a method doing something other than what G-Retriever does. Report it and say so |
| Phase 3's matrix is late | Stage A stands alone and is useful — it establishes the training-free frontier, which is the number S5 must clear whenever it arrives |

---

## 7. Explicitly not in Phase 4

No real data · no Stage C retrieval · no distilled utility head · no answerability
gate · **no MIP, MCTS, diverse beam or learned A\*** (Tier 2/3, v1.2 §2.4) ·
**no modification to `H`, `U`, `R`, the masks, `d(s)`, the lattice generator, the
exact evaluator, or any Phase-3 learner.**

If a Gate-3 result appears to require changing the instrument, that is
`GRAFT_PHASE2_BUILD.md` §6b's amendment procedure with its contamination rule —
never an implementation-time adjustment.

---

## 8. What Phase 9 gets from this

* `SearchModule` — the same protocol, with the real pool and Phase-9's distilled
  head substituted for exact `U` (fix F13). The methods do not change; their
  scorer does.
* The training-free frontier on the lattice: the number the learned sampler had
  to clear under a *perfect* scorer, which is the optimistic bound on what it
  must clear under a noisy one.
* Ledger discipline already exercised on the inference path, so Phase 9 inherits
  a metered comparison rather than inventing one.
* If Gate 3 narrowed Stage D's claim, **the narrowed claim** — written down before
  Phase 9 starts, rather than discovered inside it.
