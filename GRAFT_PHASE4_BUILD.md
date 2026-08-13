# GRAFT — Phase 4 Build Plan: the five Tier-1 search algorithms (`graft/setgen/search/`)

**Gate 3's synthetic kill-shot, run early and on purpose.**

Date: 12 August 2026 (built 13 August 2026)
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 4, fixes F5/F10/F13) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §5.2, §6.4, Gate 3) · `GRAFT_PHASE3_BUILD.md` §8 · `PHASE3_DECISIONS.md` §5
Effort: ~1.5 weeks **[ANALYSIS]** — an estimate, not a measurement. Cheap in compute: four of the five methods are training-free.
Status: **§6 RULED 12 Aug 2026. BUILT 13 Aug 2026 — Stage A run; Stage B waits on Phase 3's matrix.** Two decisions were not merely unsigned but *incomplete* — decision 3 named a mechanism without a procedure and decision 8 required a value it never gave — and both were made executable before the build. **Three sub-decisions were then overturned by measurement during the build** (decision 8's solver, decision 8's selection statistic, decision 5's live condition); each is annotated in place and recorded with its evidence in `PHASE4_DECISIONS.md` §1. Every one is a property of the environment or of a third-party library — **no learner result was inspected** — so `GRAFT_PHASE2_BUILD.md` §6b's second procedure is satisfied on the ordering.

Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps are numbered **G1–G9**, matching the Phase-0/1/2/2.5/3 convention.

---

## 0. What this phase is for, and what it is not

**The question.** Does the learned set constructor beat *training-free* search at
equal budget? `CLAUDE.md` §8 names the two most likely to embarrass the project —
**submodular greedy** and **PCST** (G-Retriever, NeurIPS 2024, whose connected
subgraph is structurally the closest classical object to a proof) — and says:
*run them early*. This is that.

**One inherited number is cited too strongly, and it is corrected here.**
`CLAUDE.md` §8 and earlier drafts of this plan quote the submodular packer at
"0.451 F1 vs 0.429 for a tuned heuristic on multi-hop HotpotQA". That is one cell
of arXiv 2607.00725's Table 6 — the **single budget out of four** at which the
gap is significant, at one reader size:

| Budget | submod F1 | focused F1 | Δ | p |
|---|---|---|---|---|
| 96 | 0.374 | 0.392 | **−0.018** | 0.08 |
| 128 | 0.426 | 0.427 | −0.001 | 0.90 |
| **160** | **0.451** | **0.429** | **+0.022** | **< .05** |
| 224 | 0.472 | 0.459 | +0.013 | 0.14 |

At 7B the same comparison is **−0.010 (p = 0.45)**, and the paper's own abstract
concludes it *"reaches parity, winning outright only where evidence density is the
binding constraint."* So the honest statement is **parity with a hand-tuned
heuristic in most settings**, not a win. **[ANALYSIS]** This does not weaken
Gate 3 — the threat here is a *training-free* method matching a *learned* one,
which is a different comparison entirely — but quoting the winning cell as if it
were the result is `CLAUDE.md` §5's "overreaching on what a paper establishes",
and this plan reproduced it before the paper was reread.

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

**And one finding that changes what this phase can conclude — measured before
anything ran (G9).** Under fix F13's perfect scorer, plain greedy attains the
**global optimum on 30/30 instances** across all three suites, while a *flawless*
sampler drawing from `p*` reaches only 1.8865 of greedy's 1.9245 at `K = 8`. So
**best-of-K cannot be a learning verdict on this environment**: the comparison is
arithmetic, decided by `p*`'s shape, not by how well anything trained. G9 records
the measurement; G5 rewrites the decision rule around it. An earlier revision of
this plan would have run a test whose answer was fixed in advance and then, by its
own decision 16, narrowed Stage D's claim on the strength of it.

---

## 1. Nine gaps this phase must close

### G1 — `relevance` has no source on the lattice, and it decides how strong S3 and S4 are [ANALYSIS]

S3's objective opens with `w_rel·Rel` and S4's PCST needs **prizes** (on nodes
*and* edges — see P4.5). Both are *relevance*, which in the finished system comes
from Stage C's hybrid retrieval —
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

### G2 — Fix F10's PCST claim is true under one graph mapping and false under the other, and the architecture states neither

Architecture fix F10 says: *"PCST's connected output maps to a closed set with no
conversion logic."* Whether that holds depends entirely on **how the pool becomes
a graph**, which no document specifies. Measured arity across the main suite:

| Atom kind | Count | `refs` per atom |
|---|---|---|
| node | 306 | 0 |
| edge | 95 | **exactly 2** |
| binding | 60 | **1 (×20) or 2 (×40)** |

**Mapping A — atoms are PCST nodes, `refs` are PCST edges.** Then F10 is
**false**, checkable in one line on benchmark instance 0:

```
atom 58f7e4a0… (kind=edge) refs=[6d9cf606…, 870232c0…]
{58f7e4a0…, 6d9cf606…}  is CONNECTED     (the reference link is present)
                        is NOT CLOSED    (870232c0… is referenced and absent)
```

Connectivity is strictly weaker than closure, so this mapping needs an explicit
**closure completion** step — which adds atoms, can breach `max_atoms`, and
changes the set PCST's own objective chose.

**Mapping B — GRAFT's node-atoms are PCST nodes and its 2-ary atoms are PCST
edges.** This is the mapping that matches G-Retriever, which **assigns prizes to
nodes *and* edges** (NeurIPS 2024, §Prize-Collecting Steiner Tree: *"assigns
higher prize values to nodes and e[dges]"*). Under it F10 is **true for 2-ary
atoms by construction** — a subtree containing an edge contains both its
endpoints, which is exactly closure. But it fails elsewhere: a **1-ary binding
becomes a self-loop, and a tree contains no loops**, so **20 of the suite's 60
binding atoms could never be selected by S4 at all.**

**Neither mapping is free, and the plan must pick with the cost named.**

* Mapping A: F10's "no conversion logic" is wrong; completion is required and its
  rate must be reported, because a high rate means PCST's objective and this
  environment's feasibility rule disagree.
* Mapping B: faithful to G-Retriever and closure-free for 2-ary atoms, but S4
  becomes structurally unable to return a third of the bindings — which is not a
  fair baseline, it is a crippled one.

**Mapping A alone has the same defect, and it had to be measured rather than
assumed.** On the reference graph the pool is **not connected** — and the
population must be declared, because the two readings differ by a factor of
three *(declared 13 Aug 2026; the audit reproduced both)*: over the **whole
pool**, including the permanently inadmissible padding atoms, 6–15 components
per instance (mean 10.6); over **admissible atoms only**, 2–4 (mean 3.25). **S4
is handed the admissible-atom graph** — S1/S2's masks prune the padding atoms
automatically, so giving S4 the same information is what keeps the comparison
fair, and prizes on unselectable atoms would be wasted by construction; the
per-atom pre-filter is construction-time validity, spending zero
`terminal_checks` (G6). The decisive sub-claim holds on **both** populations:
**the gold proof spans more than one component on 8 of the 20 main instances**.
An unrooted PCST returning a single tree — G-Retriever's single-tree
configuration (its released code's `pcst_fast` call; the paper itself states
the guarantee as returning a *connected* subgraph) — therefore cannot return
the gold set on **40% of the suite**. Choosing Mapping A on the grounds that Mapping B is crippled, and
then not checking whether Mapping A is crippled too, is the error this gap exists
to prevent, committed inside the gap that prevents it.

**RULED: Mapping A, solved as a forest.** PCST runs **per connected component**
and the union of the per-component trees is the candidate, then closure
completion, then `H`. That is a **declared departure from G-Retriever**, which
solves for one tree, and the reason is measured rather than argued: with one tree
S4 is structurally unable to reach 40% of the gold sets, and `CLAUDE.md` §8 wants
this baseline dangerous.

**One consequence to record honestly:** under Mapping A the PCST "edges" are
*reference links*, not evidence, so G-Retriever's **edge prizes have no
counterpart and are dropped**. S4 is therefore *PCST applied to this environment*,
not G-Retriever's formulation transplanted — and the write-up must say so rather
than claim the latter. F10's sentence is correct only under Mapping B, which this
phase does not adopt.

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

**RULED: every method *attempts* `K = 8` candidates** — generated as below — **and reports how many distinct valid sets it actually returned** (decision 3; greedy genuinely yields ~2.45 and the table says so). *(Header aligned with the ruled decision 3 on 13 Aug 2026; it previously read "returns exactly K = 8", the pre-ruling wording the measurements below this line already refute.)*

| Method | Its eight candidates | Deterministic? |
|---|---|---|
| S1 greedy | 8 runs, each **forced to a different first atom** — the 8 highest-`rel` legal openers | yes |
| S2 beam | width `b = K = 8`; the **8 highest-scoring terminals encountered anywhere in the search**, deduped by `canon_set_hash` | yes |
| S3 submodular | 8 runs, same forced-distinct-opener rule | yes |
| S4 PCST | **8 values of G-Retriever's own top-`k` knob** — the paper's portfolio control | yes |
| S5 portfolio | `K = 8` = 1 greedy + 7 sampled (fix F5) | no — seeded |

**ε-greedy restarts were tried first and rejected on measurement.** An earlier
revision proposed 8 ε-restarts at the frozen ε = 0.05. Measured on the main
suite, that yields **2.10 distinct sets from 8 draws** — and every ε-deviation
is non-argmax *by construction*, so a deviated candidate never beats the clean
greedy run (in practice most deviations funnel straight back to the same
optimum, which is what the 2.10 measures). Since plain greedy reaches the global optimum (G9), one of the 8 restarts
is a clean greedy run with probability ≈ 0.9998, and **best-of-8 therefore equals
best-of-1 in value on every instance**. That is precisely the failure the earlier
revision rejected randomised tie-breaking for — "eight identical sets… invisible
in the table" — readmitted through a different constant.

**And forced-distinct openers, which replaced them, barely help: measured 2.45
distinct (range 2–3, 20/20) against ε-restarts' 2.10.** The cause is G9 itself —
greedy on exact `U` funnels to the same global optimum whatever atom it starts
from, so the opener does not survive to the terminal. Three mechanisms have now
been tried and all three fail for one reason, which is the finding: **on this
environment greedy has no portfolio to give.** So decision 3 stops trying to
manufacture one. Every method *attempts* `K = 8` and **reports how many distinct
valid sets it actually returned**; engineering greedy into eight forks would be
engineering the baseline, and a table saying 8 when the truth is 2.45 is the
label the last two attempts were really wrong about.

**S2's "eight survivors" was undefined and is now specified.** Terminals appear at
many layers of the DAG, so the final beam layer and the best terminals *seen* are
different objects; the latter is what a portfolio wants.

**The budget semantics, corrected (see G6).** Fix F5's two constants do different
jobs: `K = 8` is what a method **returns**, `checker_budget = 32` bounds what it
may **validate**. Because `stop_allowed` *is* `H`, mask-driven methods get
validity free and spend **0** checks; direct-build methods pay **1 per candidate**.
So the budget binds on S3 and S4 only — which is exactly the "real cost difference
between method families" G6 claims and the earlier revision's flat 8-for-everyone
accounting erased. The budget curve (v1.2 §5.2) is consequently informative for
S3/S4 and flat for S1/S2/S5, and *that shape is itself the finding*.

**But `checker_budget = 32` binds on nobody, and that has to be said rather than
implied.** The maximum any method spends is `K = 8` — one check per candidate for
the direct builders, zero for the rest. So the curve's **levels are `{1, 2, 4, 8}`**
and the frozen 32 is **inert on this environment**. `CLAUDE.md` §6 pins 32 as the
units the Stage-D primary is *defined* in, so the write-up must state that Phase 4
reports at a budget which constrains no method, rather than quoting "at
`checker_budget`" and implying it bit. It becomes binding at Phase 9, where a pool
of 64 and a noisy scorer make more candidates worth validating.

### G4 — "Three seeds" is vacuous for three of the five methods [ANALYSIS]

S1, S3 and S4 are **deterministic** given `(pool, obligations, scorer)`. The
frozen seed set `{13, 42, 7}` varies nothing in them, so three seeds means three
identical rows and any interval computed over them is zero-width — which would
make a paired test against S5 look impossibly significant.

State per method what a seed *is*:

| Method | What the seed varies | Reported |
|---|---|---|
| S1, S2, S3, S4 | **nothing** — all four are deterministic under G3's generation rules | **once** |
| S5 | the sampler: the seed is the trained model *and* the draw | three seeds |

**[ANALYSIS]** Where a method is deterministic, report it once and say so, rather
than printing three copies and inviting a variance estimate that does not exist.

**But the paired test needs matching shapes, and this contradicted decision 5.**
`gate2.paired_bootstrap` raises on a shape mismatch and expects
`[n_seeds, n_instances]`; reporting a deterministic method as one row while
requiring a paired test against S5's three rows is a contradiction *inside the
same ruled table*. **Resolution: a deterministic method is recorded once and
broadcast to the three seed rows for the test.** That is not a fudge — its seed
variance really is zero, and broadcasting is what makes the bootstrap see zero
instead of erroring. The interval is then driven entirely by instance resampling,
which is the truth about a deterministic method.

### G5 — Gate 3 has no decision rule, which is the defect fix F12 retired for Gate 2

The architecture's Phase-4 exit is *"Explicit written decision: does the learned
sampler beat S3/S4 on the lattice?"* — with **no threshold, no test, no
predeclaration**. That is exactly the unfalsifiable rule fix F12 retired for
Gate 2, one gate over.

The remedy exists and is built: `gate2.paired_bootstrap` — hierarchical over
seeds and instances, frozen at 10,000 resamples and seed `20260814`.

**But it cannot be reused "verbatim", and an earlier revision said it could.** Two
defects, both in the code as written:

* **Direction.** `wins = bool(upper < 0.0)` is built for exact TV, where *lower is
  better*. Best-of-K utility is *higher is better*, so passing `a = S5, b = S3`
  would declare S5 the winner exactly when it loses. `gate3.py` **negates before
  calling** and records that it does (P4.7).
* **Shape.** It raises on a shape mismatch — see G4's resolution.

**And the rule itself has to change, because G9 shows the old one was
predetermined.** Two parts, only one of which is a learning claim:

**Part 1 — reported, never tested: best-of-K against the `p*` ceiling.** Every
method's best-of-K is printed beside **`E[best-of-K | p*]`**, the closed-form
value a *flawless* sampler attains. G9 measures greedy at the global optimum and
the ceiling below it, so **no sampler can win this comparison and no conclusion is
drawn from losing it.** What the two distances mean:

| Distance | Reads as |
|---|---|
| S5 → ceiling | **a learning measurement** — how near training got to a perfect sampler |
| ceiling → greedy | **an artefact of the metric** under fix F13's perfect scorer; a property of `p*`, not of any method |

**Part 2 — tested: distinct valid proof sets at equal budget.** This is plan
§6.4's own Stage-D secondary — *"number and diversity of **distinct valid** proof
sets"* — which the earlier revision dropped entirely. It is where a portfolio is
*supposed* to beat a single-solution method, and unlike best-of-K it is not
decided in advance: greedy returns one set, a sampler returns a distribution.

**The distinct-*count* replacement was predetermined too, and measuring it before
writing it down is what caught that.** Decision 3 caps every method at `K = 8`;
beam-8 returns **8.00 distinct on 20/20** (there are 353–552 terminals to draw
from); and a flawless sampler's expectation is **E[distinct \| 8 draws of `p*`] =
7.78**, maximum 8. A rule reading "S5 must return *strictly more* than S1–S4"
therefore has `P(pass) = 0`. The first rule was rigged against S5 by arithmetic;
its replacement was rigged against S5 by a cap. Writing a third rule without
measuring it first would be the same mistake a third time.

**So the rules were measured before being written, and none survives as a gate.**

| Candidate criterion | Why it cannot be the gate |
|---|---|
| best-of-K vs any rival | greedy is globally optimal (G9); a flawless sampler is 0.038 short. Arithmetic |
| distinct-set **count** | capped at `K = 8`; beam saturates it and the `p*` ceiling is 7.78 |
| distinct-set **diversity** (mean pairwise Jaccard **distance** over the C(K,2) unordered pairs of the returned sets, **duplicate pairs included** — a collapsed sampler scores low, which is the point; convention pinned 13 Aug 2026, and this quantity is the one live gate, so the estimator may not float) | **live but near-definitional**: measured, `p*` spreads at **0.4506** (exact, under the pinned convention; the earlier 0.453 was a sampled estimate) against beam's 0.225 on 20/20 — but a GFlowNet is *defined* as sampling ∝ reward, so conditional on Gate 2 passing, S5 ≈ `p*` and this value is determined by Gate 2, not by Gate 3 |

> **RULED: Gate 3's synthetic stage is a diagnostic, not a verdict.** Under fix
> F13's perfect scorer this environment cannot host the architecture's Gate-3
> comparison, and **the gate's decision moves to Phase 9**, where the distilled
> head is noisy and Robust Scheduling's precondition actually holds. Phase 4
> reports; Phase 9 decides.
>
> **One necessary condition remains live and can fail here.** If S5's portfolio
> diversity does not exceed the training-free arms', then the flow method is not
> producing a portfolio at all on this environment and its Robust-Scheduling
> justification has no support even before the noisy-scorer test. That is a
> genuine failure — a mode-collapsed or under-trained sampler fails it — and it is
> the one thing Phase 4 may conclude on its own.
>
> **MEASURED AT BUILD TIME (13 Aug 2026): this condition is size-confounded and,
> on this environment, predetermined — a *fourth* rule the measurements retire.**
> Mean pairwise Jaccard distance falls monotonically with set size even for
> **uniformly random** portfolios (8 random sets from a 24-atom pool: 0.943 at
> size 2 down to 0.791 at size 8), so a method returning smaller sets scores
> higher diversity for free. Measured on the main suite, S4 under *informed*
> relevance returns size-3.78 sets and scores **0.483** under the pinned
> duplicates-included convention (0.551 under the review round's pre-fix
> deduplicated estimator), above `p*`'s own **0.4506** either way — and a
> converged S5 samples ≈ `p*`, so S5 cannot beat S4 on the
> ruled metric whatever it learned. The ruled metric is implemented faithfully
> and a **size control** is reported beside it (`excess_diversity` = observed −
> random-portfolio baseline at the same set sizes, seeded and frozen). Adopting
> the controlled form as the *gate* is a §6b decision-rule amendment and belongs
> with Phase 9's re-ask under the noisy scorer. `PHASE4_DECISIONS.md` §1.3.
>
> **No best-of-K comparison against any rival may be reported as a Gate-3
> failure — nor a best-of-K win as a Gate-3 pass.** It is arithmetic in both
> directions: the same perfect-scorer artefact that predetermines S5's loss to
> greedy would flatter S5 against a handicapped rival (S4's dropped edge prizes,
> completion losses). The ceiling row is the only frame either number may be
> read in.

**This is an amendment to the architecture's gate placement**, not an
implementation choice, and it is recorded as one: `GRAFT_EXECUTION_ARCHITECTURE_v1.md`
§Phase 4 puts Gate 3's synthetic exit here. The measurements forcing it are of
`p*`, `U`, greedy and beam — **no learner result was inspected** — so
`GRAFT_PHASE2_BUILD.md` §6b's second procedure is satisfied.

**What this costs, stated plainly.** Gate 3's synthetic stage can no longer answer
the architecture's question — *"does the learned sampler beat training-free search
at equal budget?"* — as posed, because on this environment under a perfect scorer
the answer is fixed. It answers a narrower one: *does the learned sampler produce
a genuinely better portfolio, and did training approach its own target?* The
original question survives intact for **Phase 9**, where the scorer is the noisy
distilled head — the proxy/true-evaluator regime that is the entire published
justification for a portfolio (Robust Scheduling, ICLR 2023). Phase 4's table is
then read as an **optimistic bound on the training-free frontier**, which is what
fix F13 always said it was.

A decision rule chosen after seeing results is not a decision rule — the
project's own **[ANALYSIS]** predeclaration discipline (its significance-testing
authority, Dror et al., ACL 2018, covers test selection and does not state this
aphorism). This one is being rewritten *before* any method
has run, from a property of the environment, and **no learner result has been
inspected**: G9's measurements are of `p*`, `U` and greedy, none of which is a
learner. `GRAFT_PHASE2_BUILD.md` §6b's second procedure is therefore satisfied.

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

**And the corollary an earlier revision got backwards.** `masks.py` defines
`stop_allowed(state)` as `state.ok()` — it **is** `H`. So a terminal reached
through the masks is `H`-valid *by construction*, and S1, S2 and S5 have nothing
left to validate: their honest spend is **0**, and the "`H`-filter" in
sample-then-filter has nothing to filter. Charging every method a flat 8 checks —
as the earlier revision's decision 3 did — was accounting fiction that erased the
very cost difference this gap exists to expose, and made `checker_budget = 32`
non-binding for everyone. Corrected in G3: mask-driven methods spend 0; S3 and S4
spend 1 per candidate and are the only arms the budget constrains.

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
note the scale: the synthetic profile sets **`pool_cap = 32`** (`graft/config/presets/synthetic.yaml`, not the real-data default of 64 — an earlier revision quoted the wrong one) and the measured lattice pools are **20–26 atoms**.
At that size an exact PCST is tractable directly, and the reference solver's value
is that it is *the standard one* rather than that it is fast.

Decide and record: pin `pcst_fast` (matching G-Retriever's formulation exactly, at
the cost of a build dependency) **or** implement the PCST objective directly and
verify it against `pcst_fast` on a handful of instances offline. **[ANALYSIS]** the
first is more defensible in a write-up; the second removes a platform risk on a
Windows laptop. Either is fine; leaving it undecided until S4 is being written is
not.

### G9 — Under a perfect scorer this environment does not discriminate on best-of-K, and greedy is optimal

**Measured on the frozen suites before any method was built.** Two facts, both
closed-form or exhaustive, neither involving a learner:

| Quantity | Value |
|---|---|
| mean global max `U` over valid terminals (20 main instances) | **1.9245** |
| **greedy on exact `U` reaches that global argmax** | **30 / 30 instances** (20 main, 5 probe, 5 tuning) |
| mean `E[best-of-8 \| p*]` — a *flawless* sampler | **1.8865** |
| gap to greedy at `K = 8` | **−0.0380** (range −0.0454 … −0.0319, negative on **20/20**) |
| gap at `K = 32` — the whole checker budget | −0.0066 |
| gap at `K = 256` | −0.0001 |
| `p*` mass on the single best terminal | **1.38 – 2.58%**, over 353–552 terminals |

`E[best-of-K | p*]` is closed-form from the already-built `Target`: sort terminals
by `U`, and with `F` the cumulative `p*`, `E = Σ u_i (F_i^K − F_{i−1}^K)`. It
costs nothing to compute and belongs in the table as a row.

**Why this happens.** `p* ∝ exp(β·U)` at β = 4 spreads mass across hundreds of
near-valued terminals, so eight draws are unlikely to contain the argmax — while
one greedy pass finds it for free. The sampler is not failing; it is *sampling*,
which is what it was asked to do.

**Why it matters more than a lost comparison.** Fix F13 sets
`scorer = exact U = the evaluation metric`. That equality removes the
proxy/true-evaluator gap which is the **entire published justification** for the
sample-then-filter portfolio — `CLAUDE.md` §4.2 calls Robust Scheduling
(ICLR 2023) *"the single best argument for using a flow method at all"*, and its
result is that diverse candidates under a **cheap proxy** beat proxy-optimisation
when the **true evaluator is expensive**. Phase 4 kept the pattern and deleted its
precondition. Under a perfect scorer, optimising the scorer directly is simply
correct, and greedy does it.

**What this phase does about it** — G5 restructures the decision rule; the ceiling
becomes a reported row; the tested claim moves to plan §6.4's distinct-valid-set
secondary. **What it does not do:** change the instrument. Introducing a noisy
proxy scorer for the search arms would restore Robust Scheduling's precondition
and is the scientifically stronger fix, but it is a `GRAFT_PHASE2_BUILD.md` §6b
amendment and is recorded here as the recommended one for Phase 9 rather than
smuggled in as an implementation choice.

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
Its eight candidates come from **eight forced-distinct first atoms** (G3), which
is deterministic and needs no ε.

S2: beam width `b = K = 8` over partial sets, dedup by `canon_set_hash` (Phase 0).
Its eight candidates are the **best terminals encountered anywhere in the
search**, not the final beam layer — terminals appear at many DAG layers, and the
final layer is a different object (G3).

Both go through the Phase-1 masks, so their construction-time validity is free and
their outputs are closed *and* `H`-valid by construction — hence **0 terminal
checks** (G6).

### P4.4 `search/s3_submodular.py`

**[EVIDENCE]** objective, weights and algorithm from arXiv 2607.00725 (*provisional
venue, declared*), Eq. 1: `F(S) = w_rel·Rel + w_qry·QueryCov + w_cov·Repr +
w_div·Div`, with the paper's values **`w_rel = 1.0, w_qry = 0.5, w_cov = 0.4,
w_div = 0.3`** and — omitted by earlier drafts of this plan and by the
architecture — the facility-location saturation parameter **`α = 0.3`**. Five
constants, not four.

**The weights are the paper's and are not tuned here** — tuning them would make S3
a comparison of tuning effort, the objection v1.2 §5.2 raises against unequal
hyperparameter budgets.

**But transplanting them is `[ANALYSIS]`, not `[EVIDENCE]`, and the label matters.**
Those five constants were fitted to *token-budgeted HotpotQA snippets*. Here all
four features are re-defined over lattice atoms and `rel` is an **admitted
stand-in** (G1). Calling the result "the paper's objective" is the overreach this
plan's own §0 corrects — in the same document. So: **the algorithm is
`[EVIDENCE]`; the weights as applied to these re-defined features are
`[ANALYSIS]`**, and the write-up says which is which.

**A consequence worth stating rather than discovering.** On this environment
those four terms re-weight components `U` already contains — obligation coverage,
facility-location redundancy, source quality. So S3 is greedy on a *distorted
copy* of the metric while S1 is greedy on the metric itself, and G9 shows S1 is
optimal. S3 cannot beat S1 on best-of-K by construction, which is why G5 moves the
tested claim off best-of-K entirely.

**"Cost-scaled" has no work to do on the lattice, and the plan should say so.**
The paper's greedy is cost-scaled **per token** under a **token budget `B`**, and
its Lin–Bilmes singleton fallback compares the greedy set against the best single
snippet — machinery for a *knapsack* constraint. GRAFT's lattice has no token
cost: the constraint is `max_atoms`, so **every atom costs 1**, marginal-gain-per-
cost collapses to marginal gain, and the fallback is near-vacuous at a cap of 8.

The consequence is worth stating plainly rather than discovering in the table:
**S1 and S3 become the same algorithm under different objectives** — greedy on
`U` versus greedy on `F`. That is still a legitimate and arguably cleaner
comparison, but S3 does not contribute a different *search strategy* here, and the
write-up must not imply it does.

**The guarantee: the source paper already declines it, and so does this phase.**
An earlier draft argued that Nemhauser–Wolsey–Fisher's `(1 − 1/e)` fails to
transfer because closure is not a cardinality constraint. That argument was aimed
at a claim nobody makes. arXiv 2607.00725 §4.2 states it directly:

> *"stronger constant-factor guarantees for the knapsack case require additional
> partial enumeration (Sviridenko, 2004; Nemhauser et al., 1978), which we do not
> perform — we use the algorithm for its empirical behaviour under a token budget,
> not for a guarantee."*

So there is no inherited guarantee to lose. GRAFT has one *additional* reason to
claim none: Gate 3 scores `U` on the **`H`-filtered** set, while any approximation
result would concern `F` on the unfiltered one — two different objectives and two
different objects. Report S3 as the published *algorithm*, exactly as its own
authors do.

### P4.5 `search/s4_pcst.py`

Prizes from `relevance`, unrooted, over Mapping A — the pool's **reference
graph** — solved **per connected component** and unioned (G2's forest ruling,
because the gold set spans components on 8/20 instances). Then **closure
completion**, then `H`-filter. Reports the completion rate, the `max_atoms` breach
rate, and the per-instance component count.

**Two details the architecture's one-line spec omits, both from the paper.**

*Prizes go on nodes **and** edges.* G-Retriever: *"assigns higher prize values to
nodes and e[dges]"*, and its appendix selects top-`k` for each separately
(WebQSP: `k = 3` nodes, `k = 5` edges). "Node prizes" alone is half the
formulation. Under Mapping A every atom is a node, so this resolves cleanly —
but it must be resolved deliberately, not by omission.

*Edge cost is a declared parameter, not a constant — and on this environment it
carries the size constraint.* The appendix sets `C_e = 1` for SceneGraphs and
`C_e = 0.5` for WebQSP: **tuned per dataset**. Worse for GRAFT, under Mapping A
the PCST "edges" are *reference links*, which are not evidence and have no cost
semantics of their own — at `C_e = 0` PCST degenerates and takes every
positive-prize atom in the component.

What `C_e` actually does here is control **output size**, which matters because
**PCST has no cardinality constraint and `max_atoms = 8` does**. G-Retriever says
so itself: *"By adjusting the prizes and costs on nodes and edges, users can
fine-tune the subgraph's extent."* So `C_e` cannot be a literal lifted from the
paper — it has to be calibrated to this environment's size regime.

**The procedure, which mirrors Phase 3's β/`N` calibration exactly** (decision 8):

1. Grid `C_e ∈ {0.25, 0.5, 1.0, 2.0}` — predeclared, and **anchored on the paper's
   own two values** rather than invented around them.
2. Evaluated on the **tuning** suite, never the main suite — the same separation
   Phase-2 decision 22 enforces for β.
3. Selected by the value with the **lowest POST-completion breach rate** — the
   share of outputs exceeding `max_atoms` — with ties broken toward the **larger**
   median, since a bigger subgraph is more evidence and `CLAUDE.md` §8 wants S4
   dangerous. Still purely *structural*, reading no `U`, no best-of-K and no
   Gate-3 number, so it cannot be tuned toward a result.
   *(Amended 13 Aug 2026, by measurement. This step previously selected on the
   median size, capped from above; taken literally that admits a median **equal
   to** the cap, and the build measured exactly that at `C_e = 0.5` under
   obligation relevance — median 8.0 = `max_atoms`, breach rate **0.500**, so
   half the outputs are rejected by `H` on size. That is the straddle the earlier
   amendment was written against, reached through the letter of its own
   replacement. The breach rate measures the property the criterion was always
   for and admits no straddle. `PHASE4_DECISIONS.md` §1.2.)* Calibrated **once per
   relevance variant** (decision 1 reports both, and S4's prizes come from `rel`,
   so each variant gets its own `C_e`, both recorded). **Frozen: `C_e = 2.0` for
   both variants** — obligation median 3.0 at breach 0.175 (no grid value is
   breach-free under the proxy, a structural finding about Mapping A rather than
   a tuning failure), informed median 3.0 at breach 0.000.

   **Post-, not pre-, and an earlier revision had it the wrong way round.**
   Closure completion only ever *adds* atoms. Targeting the cap *before*
   completion therefore puts roughly half the outputs *over* `max_atoms` after
   it, `H` rejects them on size, and S4 returns few or no valid candidates — while
   §6b lists "PCST's completion rate is high" as a declared risk. The earlier
   criterion selected for the failure the risk register warns about.
4. Frozen before S4 runs on main, and recorded in §6 beside the achieved median.

**[ANALYSIS]** This is the one thing Phase 4 calibrates, and it calibrates a
baseline *toward feasibility*, not toward performance — the direction that keeps
S4 dangerous. A `C_e` that made PCST return 30-atom sets would hand it a 100%
closure-breach rate and a free loss.

### P4.6 `search/s5_portfolio.py`

`load_policy(checkpoint)` → sample `K` with `greedy=1` (fix F5's 1 greedy + 7
sampled, already implemented in `sample_trajectories`) → `H`-filter → rank by
`scorer`. **No trainer import** — Phase-3 §8 requirement 1 is that the checkpoint
loads without it, and this module is what proves it.

### P4.7 `search/gate3.py`

The matrix over five methods × instances × seeds, the ledger accounting, the
budget-level curve of G3, and `gate2.audit_block` so every Phase-4 table carries
the same audits from the same Phase-2 source.

**Three things it must do that "reuse `paired_bootstrap` verbatim" does not.**

1. **Negate before calling.** The function's `wins = upper < 0` is written for
   exact TV (lower-better); every Phase-4 metric is higher-better. `gate3.py`
   negates and records that it did — reusing it unnegated would report the winner
   as the loser (G5).
2. **Broadcast deterministic arms** to the frozen seed count so shapes match
   (G4), with the zero seed-variance stated in the report rather than implied.
3. **Compute the `p*` ceiling row** — `E[best-of-K | p*]` per instance,
   closed-form from `Target` (G9), printed beside every method's best-of-K so the
   two distances of G5's Part 1 are readable off the table.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 1 | P4.1 protocol + ledger wiring | a method that overspends raises; `would_exceed()` is called before every check (G6) |
| 2 | P4.2 relevance, both variants | both are pure functions of `(atom, obligations \| env)` and are unit-tested against hand cases |
| 3 | P4.3 S1 + S2 | outputs are closed and `H`-valid by construction; each **attempts** `K = 8` — S1 by forced-distinct openers, S2 by the best terminals seen anywhere — **and reports its distinct-valid count** (measured ~2.45 for S1; the table says that, never 8 — decision 3, exit 12c); each spends **0** checks (decision 6). *(Corrected 13 Aug 2026 — this row predated decision 3's ruling and required the "8 genuinely distinct sets" the ruling forbids engineering; a builder following it either failed the step or engineered the baseline.)* |
| 4 | P4.4 S3 | reproduces the paper's objective on a hand-built example; the Lin–Bilmes singleton fallback is implemented and exercised by a **constructed unit case**, and recorded as **measured inert at the frozen cap** — 0 fires / 160 main-suite chains at `max_atoms = 8`, while the *same shipped code* fires 60/160 at a budget of 3, which is the constructed case. *(Corrected twice: the original done-when, "fires at least once in the suite", is unsatisfiable at the frozen cap; the 13 Aug replacement claimed inertness as a **theorem**, whose premise — the chain's first pick being the argmax singleton — is false for the forced-opener path used (true on 20/160 chains). Inertness is a property of a frozen value `CLAUDE.md` §6 prices for change, not a proof. `PHASE4_DECISIONS.md` §1.4 F2.)* |
| 5 | P4.5 S4: **per-component forest** + closure completion + **`C_e` calibration on the tuning suite** | **a connected-but-unclosed PCST output is completed and passes `H`** — the G2 case, as a regression test; a gold set spanning two components is reachable; and `C_e` is frozen from the **post-completion breach rate** (decision 8's amended statistic, ties to the larger median) before S4 touches main (decisions 2, 8) |
| 6 | **Gate-3 Stage A**: S1–S4 across the main suite, budget curve, audits, **the `p*` ceiling row and plan §6.4's secondaries** | the table exists with S5 empty and says so; best-of-K appears beside the ceiling, never beside a rival alone (decisions 5, 10) |
| 7 | P4.6 S5 (code only, no checkpoint yet) | loads a Phase-3 checkpoint without importing the trainer |
| 8 | **Gate-3 Stage B**, *after Phase 3's matrix* | S5's row filled; G5's predeclared rule applied and the outcome written |

Steps 1–7 need nothing from Phase 3's runs. Step 8 is the only one that waits.

---

## 5. Exit criteria

**The methods are correct**
1. Every returned set passes `H` — the filter is not assumed, it is asserted per method per instance. **The assertion runs with `ledger=None`.** `checker.py` increments `terminal_checks` on *every* `H` call that receives a ledger, so running this verification against the live ledger would charge S1/S2/S5 eight checks each and erase the 0-versus-1 family difference criterion 6 exists to expose. Out-of-band verification is what Phase 2's exhaustive enumeration already does for the same reason.
2. S1 and S2's outputs are closed *by construction* (they use the masks); S3 and S4's are closed *after* their filter/completion — and a test exercises the G2 case where PCST's raw output is connected but not closed.
3. S3 reproduces the arXiv 2607.00725 objective on a hand-computable example, with the paper's weights unmodified.
4. S5 loads a checkpoint through `graft.setgen.policy` alone, with no import of `graft.setgen.trainer` (Phase-3 §8 requirement 1), asserted by an import test.

**The comparison is fair**
5. `checker_budget = 32` and `K = 8` are the same constants Phase 3 used (fix F5), read from config, not restated.
6. The budget is **enforced** — `would_exceed()` before every spend — and each method's *actual* spend is reported beside its result: **0 for S1/S2/S5** (mask-driven, `H`-valid by construction) and **1 per candidate for S3/S4** (G3, G6). A table charging every method the same is the accounting fiction G6 retires.
7. Every method's seed semantics are declared, and a deterministic method is reported once rather than as three identical rows (G4).
8. The relevance variant is named in every table: `obligation` or `informed` (G1).
9. All five score with **exact `U`** (fix F13), and the write-up carries the "perfect scorer, optimistic relative to deployment" caveat.

**Gate 3**
10. Best-of-K valid-set utility per method across the main suite, **plotted over the budget levels `{1, 2, 4, 8}`** — and the write-up states that the frozen `checker_budget = 32` is **inert on this environment**, since no method can spend more than `K = 8` (G3, v1.2 §5.2).
11. The **predeclared G5 ruling** is applied: Phase 4 publishes a **diagnostic**, not a Gate-3 verdict, and the one live condition — S5's portfolio diversity exceeding the training-free arms' — is tested with `gate3.py` **negating** before `paired_bootstrap` and **broadcasting** deterministic arms to three rows (G4, G5, P4.7).
11b. **The `p*` ceiling row is present**, and best-of-K is reported against it rather than against a rival (G9). A best-of-K loss to greedy is recorded as arithmetic and **may not be reported as a Gate-3 failure**.
11c. **Plan §6.4's Stage-D secondaries are reported**, not dropped: distinct-valid-set count and diversity, `E[U]` of sampled sets, valid-terminal rate as an *efficiency* measure, and evidence-set size.
12. PCST's closure-completion rate and `max_atoms` breach rate are reported (G2).
12b. **`C_e` was calibrated by decision 8's procedure** — grid on the **tuning** suite, selected on the **post-completion breach rate** (ties to the larger median; decision 8's amended statistic), frozen before S4 touched main — and §6 records the chosen value beside its achieved breach rate and median. A `C_e` chosen after a Gate-3 number existed would be the same defect G5 closes for the decision rule.
12c. **The distinct-set count is reported per method, never assumed.** Every method *attempts* `K = 8`; measured **for S1** (masked greedy on exact `U`): **2.45** distinct (range 2–3, 20/20), because greedy funnels to one optimum, and the table says 2.45 rather than 8 (decision 3, G3). **S3's count is measured at build time, not borrowed from S1's** — S3 is *unmasked* greedy on `F` with a stand-in `rel`, a different objective and different feasibility mechanics, so its distinct-valid count (and its spend, after `canon_set_hash` dedup) can differ. *(Corrected 13 Aug 2026 — this criterion previously presented S1's measurement as covering S3.)*
12d. **S4's forest is honoured**: PCST solved per connected component, with the component count and the gold-spans-components rate reported (G2).
13. Audits carried into every table from `gate2.audit_block` — `FAIL` rate, collision rate (0), unconstructible rate (0), `Δd` densities, target-mass profile, `neither`-mass at the run's β.
14. Every run records the `environment_fingerprint` and `target_fingerprint` of the suite it used (Phase-2 decision 21).
15. **The two-stage exit is honoured** (G7): a table without S5 is labelled Stage A and is not called Gate 3.

**The decision**
16. A written decision, before results are inspected: **if S3 or S4 matches S5 under the G5 rule, Stage D's learning claim narrows, and the narrowing is written into the plan before Phase 9 begins.** Plan §7 calls Gate 3 the highest-risk gate; this is the sentence that makes it act like one.

---

## 6. Decisions to lock before writing code — **RULED 12 Aug 2026**

| # | Decision | Value | Cost if changed later |
|---|---|---|---|
| 1 | **RULED. Relevance on the lattice** | obligation-match as primary, `U`-marginal as a declared "informed" variant, **both reported in every table** (G1). Ruled rather than left recommended because this choice decides how strong the two baselines are, and a value that decides a baseline's strength cannot be settled while looking at that baseline's results | the two baselines `CLAUDE.md` §8 calls most dangerous are silently weakened, and a Gate-3 pass means nothing |
| 2 | **RULED. PCST graph mapping, forest, and closure** | Mapping A (atoms are PCST nodes, `refs` are PCST edges), **solved per connected component and unioned** — a declared departure from G-Retriever's single tree, forced by measurement: the pool has 6–15 components on the whole-pool refs graph and **2–4 on admissible atoms — the graph S4 is handed** (population declared 13 Aug 2026; the gold set spans more than one component on **8/20** instances on *both*), so a single tree cannot reach 40% of the golds. Under Mapping A the PCST edges are reference links, so **G-Retriever's edge prizes have no counterpart and are dropped** — S4 is PCST applied here, not G-Retriever transplanted. Plus complete-then-filter, with completion rate and `max_atoms` breach rate reported. Mapping B is faithful to G-Retriever and makes closure automatic, but turns 1-ary bindings into self-loops a tree cannot contain — **20 of 60 in the suite would be unselectable** (G2) | fix F10's "no conversion logic" is carried into code where it holds only under a mapping nobody declared; or S4 is crippled into a baseline that cannot express part of the answer space |
| 3 | **RULED. Portfolio size and budget semantics** | every method **attempts** `K = 8` and **reports how many distinct valid sets it actually returned** — it is not guaranteed 8 and must not be labelled 8. Measured: forced-distinct openers give **2.45** distinct (range 2–3, 20/20), barely above the ε-restarts' **2.10** they replaced, because G9's funnelling sends greedy to the same optimum whatever the opener. **Engineering 8 distinct out of greedy would be engineering the baseline**; greedy genuinely returns ~2.5 and the honest table says so. S2 reaches 8 by taking the best terminals seen; S5 by fix F5's 1 greedy + 7 sampled. **Spend is 0 for S1/S2/S5 and 1 per *distinct* candidate for S3/S4 — duplicates dedup by `canon_set_hash` *before* validation** (free, and it makes a direct builder's spend its distinct-candidate count ≤ 8; clause added 13 Aug 2026, the cell previously left dedup unstated, which alone decides whether S3's budget-curve point sits at 8 or ~2.5) — `stop_allowed` *is* `H` (G3, G6) | a portfolio of ~2.5 wearing a `K = 8` label, which is best-of-1 in value with extra bookkeeping |
| 4 | **RULED. Seed semantics** | **S1–S4 are all deterministic** under decision 3's generation rules and are **reported once**; S5 runs the frozen `{13, 42, 7}`. For the paired test a deterministic arm is **broadcast to three rows** — its seed variance really is zero, and broadcasting is what lets the bootstrap see zero rather than raise on a shape mismatch (G4). This resolves the contradiction an earlier revision left between decisions 4 and 5 | a zero-width interval makes a paired test against S5 look impossibly significant |
| 5 | **RULED. Gate 3's synthetic stage is a DIAGNOSTIC, not a verdict** | Measured before writing: best-of-K is arithmetic (G9); distinct-set **count** is capped at `K = 8`, which beam saturates on 20/20 while the `p*` ceiling is 7.78, so a strictly-more rule has `P(pass) = 0`; distinct-set **diversity** is live (`p*` **0.4506** exact under G5's pinned convention — duplicate pairs included — vs beam 0.225 on 20/20) but near-definitional, since a GFlowNet is *defined* as sampling ∝ reward and Gate 2 already measures the distance. **So the gate's decision moves to Phase 9**, where the distilled head is noisy and Robust Scheduling's precondition holds. **One live necessary condition stays here**: if S5's portfolio diversity does not exceed the training-free arms', the flow method is not producing a portfolio at all and its justification has no support — a mode-collapsed or under-trained sampler fails this. **No best-of-K comparison against a rival may be reported as a Gate-3 failure** (G5, G9) | a third predetermined rule; or a phase with no failure condition in either direction |
| 6 | **RULED. Ledger discipline** | `would_exceed()` before every spend; completed-set checks only; **spend reported per method and it differs by family — 0 for S1/S2/S5, 1 per candidate for S3/S4** (G6). `stop_allowed` *is* `H`, so mask-driven arms are valid by construction and have nothing to validate; charging them anyway erases the cost difference the budget exists to expose | methods drift over budget and the comparison is unfair without anyone noticing |
| 7 | **RULED. S3's constants, their label, and its guarantee** | **`[ANALYSIS]`, not `[EVIDENCE]`, for the weights as applied**: the algorithm is the paper's, but five constants fitted to token-budgeted HotpotQA snippets, transplanted onto four features re-defined over atoms with `rel` an admitted stand-in, is not fidelity (P4.4). **five** of the paper's values unmodified — `w_rel 1.0, w_qry 0.5, w_cov 0.4, w_div 0.3, α 0.3`; `[EVIDENCE]` provisional venue declared. **No approximation-ratio claim**: arXiv 2607.00725 §4.2 declines it itself ("we do not perform [partial enumeration] — we use the algorithm for its empirical behaviour... not for a guarantee"). Record that cost-scaling degenerates under unit atom costs, so **S1 and S3 differ by objective, not by search strategy** (P4.4) | S3 becomes a comparison of tuning effort; α is silently defaulted; or an unearned guarantee enters the write-up that the source paper itself refuses |
| 8 | **RULED — and two clauses OVERTURNED BY MEASUREMENT at build time (13 Aug 2026); see `PHASE4_DECISIONS.md` §1.1–§1.2. PCST solver and edge cost** | The 12 Aug ruling pinned `pcst_fast` on the ground that a prebuilt `cp311-win_amd64` wheel had been verified to exist, so the platform risk was settled. **That ruling is withdrawn: the wheel exists and is WRONG**: measured against a brute-force exact reference on 60 random graphs it returned the wrong optimum on **59**, every output array carrying the right length with every element equal to its first (a buffer-lifetime bug — silent, never a crash). The ruling verified that the wheel *existed*, which is not the same claim as its being correct; G8 named the alternative in advance ("implement the PCST objective directly and verify it against `pcst_fast`"), and that verification is what found this. **Adopted: an exact solver** — uniform edge costs make a spanning tree of a connected `S` cost `C_e·(\|S\|−1)`, so Eq. 7 is a maximum-weight connected subgraph problem, solved by enumerating each component's connected subsets (components measured at 1–13 atoms; bound asserted at 20). `pcst_fast` stays installed **only** for the regression test that documents its breakage. **`C_e` is calibrated, not picked**: grid `{0.25, 0.5, 1.0, 2.0}` anchored on the paper's two values, on the **tuning** suite, **once per relevance variant** (decision 1 reports both and the prizes come from `rel`) — a structural criterion reading no `U` and no Gate-3 number — then frozen and recorded with its achieved values. **The selection statistic is the post-completion BREACH RATE, ties to the larger median** — *not* "median closest to `max_atoms` without exceeding it", which measurement retired at build time: taken literally that admits a median **equal to** the cap, and at `C_e = 0.5` under obligation relevance the median is exactly 8.0 = `max_atoms` with a **0.500** breach rate — the straddle the clause was written against, reached through the letter of its replacement. The breach rate measures what the criterion was always for and admits no straddle. **Frozen: `C_e = 2.0` for both variants** (obligation: median 3.0, breach 0.175 — *no* grid value is breach-free under the proxy, which is a structural finding about Mapping A, not a tuning failure; informed: median 3.0, breach 0.000). See `PHASE4_DECISIONS.md` §1.2. **Post-, not pre-**: completion only adds atoms, so the earlier pre-completion criterion put ~half the outputs over the cap and selected for the very failure §6b lists as a risk (P4.5) | a compiled dependency lands mid-build on a Windows laptop; or S4 gets a tuning budget no other method has, chosen after seeing results |
| 9 | **RULED. Two-stage exit** | Stage A (S1–S4) may be reported; only Stage B is Gate 3 (G7) | a table without the learned sampler is read as the gate |
| 10 | **RULED. The `p*` ceiling is a reported row** | `E[best-of-K \| p*]`, closed-form from `Target`, printed beside every method's best-of-K. It splits an otherwise uninterpretable loss into S5-to-ceiling (**learning**) and ceiling-to-greedy (**an artefact of fix F13's perfect scorer**). Costs nothing; without it Gate 3 reports a predetermined number as a result (G9) | Gate 3 reports a predetermined number as if it were a result, and the phase's headline loses the only row that makes it interpretable |

---

## 6b. Declared risks

| Risk | If it bites |
|---|---|
| **[HYPOTHESIS]** S1–S4 match S5 **on distinct valid sets** | **A designed outcome.** Stage D's portfolio claim narrows; the thesis leans on Contribution 1 and the five-ceiling protocol. This is the cheapest place to learn it |
| S5 loses best-of-K to greedy | **Expected, and not a finding.** G9 measured it in advance: greedy is globally optimal on 30/30 and a flawless sampler is 0.038 short at `K = 8`. The ceiling row is what stops this being misread; decision 5 forbids reporting it as a Gate-3 failure |
| The lattice cannot discriminate on best-of-K at all | **True under fix F13, and recorded rather than fixed.** Making the search arms score with a *noisy proxy* would restore Robust Scheduling's precondition and make the comparison live — but that is a §6b amendment to how the instrument is used, and it is recommended for Phase 9 rather than taken here |
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
* **The question Phase 4 could not answer, and why.** G9 shows that under fix
  F13's perfect scorer this environment cannot discriminate methods on best-of-K:
  greedy is globally optimal and a flawless sampler is 0.038 short at `K = 8`.
  Phase 9 is where the architecture's original Gate-3 question becomes live,
  because its scorer is the **noisy distilled head** — the proxy/true-evaluator gap
  that is Robust Scheduling's precondition and the portfolio's only published
  justification. Phase 9 should therefore treat Phase 4's table as an optimistic
  bound and **re-ask the best-of-K comparison under the noisy scorer**, which is a
  §6b amendment recommended here and deliberately not taken in Phase 4.
