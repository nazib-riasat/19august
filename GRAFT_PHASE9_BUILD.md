# GRAFT — Phase 9 Build Plan: Stage D on real data (`graft/setgen/` + `graft/retrieve/` join)

**The learned evidence-set constructor leaves the lattice: real pools, real gold
proofs, a noisy scorer — and the Gate-3 decision that Phase 4 measured itself
unable to make.**

Date planned: 15 August 2026
Parent: `GRAFT_RESEARCH_PLAN_v1.md` v1.2 **§3.4** (Stage D), **§4** (the formal
objective, §4.5.4–§4.5.5 the C3 hypothesis and safe order), **§5.1–§5.2** (the
matched comparisons), **§6.4** (metric groups), **§7 Gate 3** ·
`GRAFT_EXECUTION_ARCHITECTURE_v1.md` Phase 9 (fixes F1, F3, F4, F6, F13) ·
`GATE0_CONTRACT.md` (signed 15 Aug 2026) items 3, 5, 10 ·
`PHASE3_DECISIONS.md` (the frozen learner stack this phase re-hosts) ·
`PHASE4_DECISIONS.md` §6 (the training-free frontier and the question Phase 4
could not answer) · `PHASE7_DECISIONS.md` §6 (the pool contract) ·
`PHASE8_DECISIONS.md` §2.5/§3.3 (the two-view score contract) ·
`DATASET_DECISION.md` §2 (Phase-9 rows), §4 (costs)
Effort: **~2–3 weeks solo [ANALYSIS]** — an estimate; the incremental
environment (G1) is the schedule risk and is deliberately first in the build
order, because everything else consumes its trajectories.
Status: **planned; §6 SIGNED 16 August 2026 (delegated). No code exists** beyond
what Phases 3, 4, 7 and 8 already provide. Build steps 0–5 are unblocked and
need no GPU; step 2 needs the corpora fetched; **step 6's blocker lifted 15 Aug
2026 — Phase-3 step 6 froze β = 4.0 (`artefacts/phase3_calibration.json`,
`PHASE3_DECISIONS.md` §7); the P9.7 runner remains to be written and run**.

Labels as everywhere: **[EVIDENCE]** = a named paper supports this, venue
stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = engineering
or mathematical judgment made here.

Gaps are numbered **G1–G12**, matching the Phase-0…8 convention.

---

## 0. What Phase 9 is for, and what it is not

**Purpose, in the architecture's own words:** swap the synthetic featurizer for
the real one; train; run the portfolio. The learners are untouched — fix F6's
frozen interface `policy(state_repr, action_reprs) → logits` is the whole point
of having built Phase 3 against an adapter. What changes is everything *around*
the learners: the environment steps a real pool instead of an enumerated
lattice, the reward's `sufficiency` is measured against real gold proofs
(fix F1), and the scorer that ranks delivered sets is a **distilled head**, not
exact `U` (fix F13).

**This is where two deferred decisions come due:**

1. **Gate 3's decision.** Phase 4 measured — before any method was built — that
   under fix F13's perfect scorer, greedy attains the global optimum on 30/30
   lattice instances while a flawless sampler is 0.038 short at K = 8, so the
   synthetic stage answers by arithmetic (`PHASE4_DECISIONS.md`,
   `CLAUDE.md` §8). **[EVIDENCE]** Robust Scheduling with GFlowNets
   (ICLR 2023) — the single best argument for a flow method in this project —
   requires the true evaluator to be expensive and the sampling proxy *cheap and
   imperfect*; that precondition holds here, where the distilled head is noisy,
   and nowhere earlier. Phase 9 asks Gate 3's question where it can be answered.
2. **The fallback-abstain wiring.** Phase 8 reserved
   `OutputRecord.abstain_cause ∈ {gate, fallback}` and the
   `abstain_fallback` ledger counter, zero until Stage D lands (its G5). The
   portfolio's budget-exhaustion path is that counter's writer, and this phase
   wires it.

**Three preconditions, each named with what it blocks:**

| Precondition | State (15 Aug 2026) | Blocks |
|---|---|---|
| **β (and `r_fail` gap) frozen by Phase-3 step 6** | **CLEARED 15 Aug 2026** — the gate adopted β = 4.0 at rung 0 (`PHASE3_DECISIONS.md` §7); the record sits at `artefacts/phase3_calibration.json` and `training_blocked_reason()` returns `None`. *(This cell read "calibration gate has never run; β defaults to 4.0, eligible candidates {4, 8}" at the 15 Aug signature.)* | **every Phase-9 training run** — the reward `R = 1[H]·exp(β·U)` is identical across arms by construction (`CLAUDE.md` §6), and training before β freezes would either waste the runs or contaminate the freeze. **Building and fixture-testing all Phase-9 code is NOT blocked** |
| Datasets fetched + SHA-pinned (G10) | 2Wiki and HoVer not downloaded; MuSiQue-Ans inside `data/musique/musique_v1.0.zip`, unextracted by design | build step 2 |
| Scope-c ingestion (Gate-0 item 9: decided, not run) | not run | **Stage B only** — the conversational training track and every conversational number (G12) |

**What Phase 9 is not.** Not the reader, not the orchestrator, not the ceiling
oracles (Phase 10); not the system baselines (Phase 11); not a Gate-4 or any
LongMemEval/LoCoMo end-to-end number; not a place to retune any frozen value —
`K = 8`, `checker_budget = 32`, `pool_cap = 64`, `max_atoms = 16`,
`u_weights`, `r_fail = 1e-6`, the closure rule and seeds {13, 42, 7} are all
inherited frozen (`CLAUDE.md` §6). And not Gate 2: exact TV/KL/JS exist only on
the enumerable environment, and the plan is explicit that real data enters
through the frozen policy interface while distribution correctness stays
synthetic (`DATASET_DECISION.md` §2, Phase-3 row).

**The supervisor's constraint holds:** everything trained here is a GNN/MLP
policy or a small regression head. The SLM appears nowhere.

---

## 1. Twelve gaps this phase must close

### G1 — The state space is not enumerable, so the environment layer must be rebuilt incrementally; the losses must not notice [ANALYSIS]

Phase 3's whole training stack — `rollout.py`, the exact DP, checkpoint TV —
runs over Phase 2's **enumerated** `StateGraph`. Plan §3.4 is explicit about why
that cannot carry over: ~50 candidates at sets ≤ 16 is on the order of **10¹³
subsets**; "enumeration is impossible; incremental construction costs ~16 × 50 =
800 action scores." A real pool at `pool_cap = 64` is past every enumerable
bound.

**Decision — a step-wise environment over Phase 1's incremental layer, producing
Phase 3's exact `Batch` shape.** `core.incremental.IncrementalChecker` +
`core.masks` (`legal_adds`, `stop_allowed`, `is_dead_end`, `terminal_of`)
already define legal transitions without enumeration — that is what Phase 1
built them for. The new sampler walks them state by state, ε-greedy at the
frozen ε = 0.05, `P_B` uniform over **removable** atoms (fix F10, re-exported
never reimplemented — the Phase-3 rule), and emits `trainer.Batch` tensors:
padded `[n, L]` `log_pf`/`log_pb`/`valid`/`is_fail`, the terminating transition
as a **first-class step** with its own `log P_F(STOP|x)`, `log_pb = 0` and `φ`
slot (`PHASE3_DECISIONS.md` §1.2 — the terminal convention: `R` is the flow on
the terminating `STOP` edge, measured, and four objectives depend on it).
Because the losses consume `Batch` and the F6 logits interface only, **every
loss module in `graft/setgen/learners/` is reused unchanged** — that was the
entire purpose of fix F6, and an edit to any learner file in this phase is a
defect, not an adaptation.

What is genuinely lost and stays lost: exact TV at checkpoints. Real-data
checkpoint monitoring uses training loss and a small dev best-of-K probe
(G7's metric, cheap), recorded as *monitoring, never selection* — selection at a
fixed budget is fix F12's discipline and early-stopping a learner would break
the identical-budget comparison.

### G2 — "Stage-B encoder embeddings" do not exist on Wikipedia pools; every feature block declares its source, with presence flags [ANALYSIS]

The architecture's `AtomFeaturizer` row says atoms carry "Stage-B encoder
embeddings + retrieval channel scores + obligation-match flags + `d(s)`
deltas." Stage B exists only on conversation — a 2Wiki paragraph never passed
through D1–D4 commit. Pretending otherwise would fabricate features.

**Decision — the Phase-8 G8 pattern, verbatim:** each block carries a presence
flag, absent blocks are zeros-with-flag-cleared, and the vector is the same
width on every corpus so the arms differ by nothing but their data:

| Block | Wikipedia pools (Stage A) | Conversational pools (Stage B) |
|---|---|---|
| text embedding | **pinned bge-small** (`graphbuild.pins.EMBEDDER`, shared cache — decision 2 of Phase 7: one embedder project-wide) | same |
| retrieval channel scores | bm25 + dense, **raw and normalised views** (`PHASE8_DECISIONS.md` §3.3's lesson: the raw view carries absolute strength; min–max is fusion's, not a feature) | all five channels + optional scorer, from `assemble()`'s two score maps |
| Stage-B encoder embedding | **absent, flag cleared** | present (the trained encoder's node representation) |
| obligation-match flags | from synthesized obligations (G5) via Phase 1's deterministic `slot_status` | from parsed obligations |
| `d(s)` / `Δd` | computed by the incremental checker per state/transition — **the `delta_d` gate is L7/L7b-only**, exactly as `setgen/features.py` enforces on synthetic (decision 19a; a leak in either direction voids the C3 comparison) | same |

**[EVIDENCE]** the presence-flag mechanism is load-bearing for the same reason
as Phase 8's: without it the model can learn "block absent ⇒ corpus X," a
dataset classifier. With it, absence is an input.

### G3 — Training data is the abstract `(pool, obligations, gold_proof)` triple; no corpus-specific parser may appear above the boundary [ANALYSIS]

The architecture states the requirement and the enforcement: "the loader
consumes the abstract `(pool, obligations, gold_proof)` triple and no
corpus-specific parser may appear above it — **verified by an import-graph
test** — so a conversational-source variant is a drop-in if the transfer claim
fails." This is the Wikipedia→conversation escape hatch (`CLAUDE.md` §7) made
structural.

**Decision — one record, one registry, adapters below.** `ProofExample` =
(snapshot, pool, atom_scores, obligations, gold_atom_ids, meta). Corpus
adapters (`wiki2`, `musique_ans`, `hover`, and later `longmemeval`) each emit
it; everything above — featurizer, environment, trainer, distillation,
portfolio, Gate-3 harness — imports only the abstract record. The import-graph
test mirrors Phase 7's containment guards: no module above the boundary names a
corpus.

**Per-corpus gold, from `GATE0_CONTRACT.md` item 3.1:**

* **2WikiMultiHopQA** (COLING 2020; 167,454 train — `DATASET_DECISION.md` §2):
  evidence is a set of (subject, property, object) triples forming a reasoning
  path, with sentence-level supporting facts over a shipped distractor context.
  Gold atoms = the atoms derived from supporting sentences. *(The exact context
  shape — paragraph count, supporting-fact indexing — is **verified at wiring**,
  the Phase-8 discipline: the MuSiQue wiring found two facts the plan did not
  state, and this plan assumes the same will happen here.)*
* **MuSiQue-Ans** (TACL 2022; 19,938 train): 20 paragraphs, `is_supporting`
  flags, **and the decomposition DAG** — "the only one shipping the
  *obligations* element of the triple; 2Wiki's must be synthesised"
  (`DATASET_DECISION.md` §2). Already on disk inside the kept archive; extract
  and SHA-pin, no second download.
* **HoVer** (Findings EMNLP 2020; 18,171 claims, 3,035 four-hop): adopted
  because 2Wiki/MuSiQue are thin at ≥3 hops and "minimality and the closure
  rule only bite when proof sets are genuinely large" — but HoVer ships
  evidence, **not a per-claim distractor context**, so its pools need a declared
  distractor rule (G10). Built second, after the pair above runs end to end.

### G4 — `H` on Wikipedia pools binds a declared subset of its nine checks, and the write-up must never read "H-valid" as the conversational statement [ANALYSIS]

`H` is defined over the conversational schema. The adapter builds a **real**
per-question snapshot — `Turn` per source paragraph, `SourceSpan`, `Assertion`
(explicitly `eligible`; the schema's fail-closed default is `quarantined` and
stays that way for distractor-only corpora paths that never set it), `Claim`
node per paragraph/sentence atom, `conv_id` = question id (the LongMemEval
convention, reused). On such a snapshot the binding checks are *real*: **size,
closure, identity/duplicates, support-eligibility, scope**. Three are vacuous
by construction and are declared so: **temporal** (no `valid_during`
intervals), **binding** (no binding atoms — the same vacuity
`PHASE8_DECISIONS.md` measured), **retired** (nothing retired).

Consequence stated now, not discovered later: with coverage and sufficiency
living in `U` (not `H`), most non-empty subsets of an eligible pool are
`H`-valid, `STOP` is rarely masked, and **the reward discrimination on
Wikipedia pools comes almost entirely from `U`** — sufficiency against gold
(fix F1), coverage against synthesized obligations, redundancy, size. That is
the intended design (plan §4.1's degenerate-legal-sets warning is why
valid-terminal rate is an efficiency measure, not quality), but a reader of the
Gate-3 table needs it said once, here.

### G5 — Obligations must be synthesized on Wikipedia corpora, by a declared rule that is part of the fingerprint [ANALYSIS]

The frozen obligation parser is conversational GPU work (fix F2); Wikipedia
questions never pass through it. Synthesis rules, one per corpus, reported with
their losses (the Gate-0 item-1 adaptation discipline):

* **MuSiQue-Ans**: the decomposition DAG supplies per-hop sub-questions;
  `entity_anchor` = hop-1's subject entity string; `aggregate` from question
  form; no `time_constraint` (none expressible); `scope` = (question id,).
* **2Wiki**: `entity_anchor` = the first evidence triple's subject;
  the remaining slots default absent.
* **Declared loss:** synthesized obligations are *weaker* than parsed ones —
  they exist so `coverage` has something deterministic to measure
  (Phase 1's `slot_status`/`coverage`, unchanged) and so the entity-anchor
  flags in G2's feature block are computable. No claim is made that they match
  the frozen parser's distribution; fix F2's rule adapts as: **the synthesis
  rule ships in `pins` and its per-slot fill rates are in every artefact.**

### G6 — The reward on real data: fix F1's split, the frozen constants, and the β dependency [ANALYSIS]

`R = 1[H]·exp(β·U)` with `U`'s six terms exactly as Phase 1 built them —
**train-time `sufficiency` = fraction of the gold proof covered**, deterministic
against G3's `gold_atom_ids` (fix F1). **[EVIDENCE]** Graph-S3 (ACL 2026)
validates dense supervision from offline golden subgraphs — its Table 3
ablation gains +11.8 accuracy / +17.1 F1-macro over sparse final-answer reward
(the corrected reading; the +8.1/+9.7 headline is against seven baselines, a
different comparison — `CLAUDE.md` §5's own erratum).

* `u_weights` identical across every arm — or the comparison measures reward
  engineering (`CLAUDE.md` §6).
* **β comes frozen from Phase-3 step 6** (eligible candidates {4, 8}) — frozen
  15 Aug 2026 at 4.0 (§0's precondition table; `PHASE3_DECISIONS.md` §7). `Target.at_beta`'s `r_fail` check
  is synthetic-only, so this phase adds the real-data analogue: **measure the
  minimum valid-terminal reward over a sample of real pools at the frozen β and
  report the `r_fail` gap** — an `r_fail = 1e-6` that overlaps real valid
  rewards would corrupt `FAIL`'s semantics silently.
* `FAIL` semantics inherited exactly (fix F3, as twice amended): reached when
  construction can neither legally continue nor legally stop; licenses only "no
  valid proof found under this pool, policy, attempt count and budget," never
  non-existence.

### G7 — Gate 3 (real): the decision rule, predeclared here, before any learner result exists [ANALYSIS — the §6b-critical section]

Phase 4's ruling: the gate's decision moves to Phase 9, where the scorer is a
noisy proxy. `GRAFT_PHASE2_BUILD.md` §6b's second procedure requires
decision-rule changes to be made with **no learner results inspected
beforehand** — which is *now*, before any real-data run exists. The rule,
proposed for signature in §6:

* **Primary:** best-of-K valid-set utility at the fixed budget — K = 8,
  `checker_budget = 32` terminal `H` checks/query, enforced via
  `would_exceed()` (Phase-4 discipline; the 0-vs-1 check-family split measured
  there carries over) — computed for every trained arm and S1–S5 **on the same
  held-out pools, under the same distilled head** where a scorer is consumed
  (S1/S2/S5; S3/S4 run their native objectives plus the informed variant,
  Phase-4 decision-1 pattern, both reported). Utility for the *table* is exact
  train-time `U` (gold is available on held-out Wikipedia dev) — the head
  selects, the exact value scores; the gap between them is F13's caveat made
  measurable.
* **Verdict rule:** three seeds {13, 42, 7}, `gate2.paired_bootstrap` (higher-
  is-better negation per Phase-4 G4). **If the best training-free arm's
  interval overlaps or beats the best learned arm's, Stage D's learning claim
  narrows and the project consolidates on Contribution 1** — plan §7 Gate 3
  verbatim, `CLAUDE.md` §8.
* **Secondary, adopted here as Phase 4 recommended:** the **size-controlled
  diversity** form — `excess_diversity = observed − E[random portfolio at the
  same set sizes from the same pool]` (`DIVERSITY_CONTROL_SEED` frozen there).
  Phase 4 measured the raw metric size-confounded (S4-informed 0.483 against
  `p*`'s own 0.4506) and explicitly deferred the rule change to "Phase 9's
  re-ask" — this is that re-ask, taken at the §6b-clean moment.
* **[EVIDENCE]** the framing is Robust Scheduling with GFlowNets (ICLR 2023):
  diverse candidates sampled under a cheap proxy beat proxy-optimization when
  the true evaluator is expensive. **[EVIDENCE]** the test-selection protocol
  is Dror et al. (ACL 2018); the seed count and predeclaration are this
  project's own discipline (**[ANALYSIS]**), as everywhere.

### G8 — The distilled utility head: trained on visited terminals, fidelity reported beside every row, never inside `H` [ANALYSIS]

Fix F13's inversion: on real data gold is absent at inference, so S1/S2/S5 and
portfolio ranking consume a **small MLP** regressing train-time `U(X)` from a
permutation-invariant pooled embedding of the selected atoms' features
(mean ⊕ max pooling — sets have no order, plan §3.4's canonical-state rule).
Training data: terminal sets visited during arm training plus their exact `U`
(computable, gold known); the three P6.11 guards verbatim (seed reaches init,
early-stop restores argmin-dev, no-scorable-dev refuses — this component *may*
early-stop; it is not one of the compared arms). **Its held-out Spearman ρ
against exact `U` ships in the artefact beside every Gate-3 row**, because the
head's noise is the experimental condition Gate 3 is being asked under —
without that number the table cannot be read. Structural guarantees, both
tested: the head never appears in `H`'s import graph (v1.2 §4.4 — the
multiplicative-gate property dies otherwise, `CLAUDE.md` §4.2), and never sees
a gold field (the G3/Phase-8 quarantine pattern; `sufficiency` values reach it
as training *targets*, never as input features).

### G9 — Portfolio inference, the fallback counter, and the contested flag's honest split [ANALYSIS]

Fix F4 wired: **1 greedy rollout + K−1 = 7 sampled → `H`-filter
(`search.base.h_filter`, dedup by `canon_set_hash` first) → rank by the
distilled head → tie-break smaller set**. Budget exhaustion or a dead end on
all K attempts reaches `FAIL` → **increments the `abstain_fallback` ledger
counter Phase 8 reserved** — closing its G5 loop; `OutputRecord.abstain_cause`
= "fallback" is written by Phase 10's orchestrator, which consumes this
counter's event.

**The contested flag splits into what is computable here and what is not**
(the Phase-8 G7 transfer pattern): fix F4 flags output `contested` when top
valid sets imply **different answer bindings**. On gold-bearing corpora
(train/dev) the binding of a set is computable: the subset of its atoms whose
`text_norm` contains the gold answer or an alias — so Phase 9 implements
`binding_of(set, aliases)` and reports contested-rate **as an evaluation
diagnostic**. At deployment there is no gold; the architecture's own words are
"costs one comparison" — a reader-level check that belongs to Phase 10.
**Transferred by name to Phase 10:** the inference-time contested comparison.
Inventing a gold-free proxy here would be unstated machinery.

### G10 — Datasets: pins, licences, the subset decision, and HoVer's distractor rule [ANALYSIS]

All through the one SHA-verified reader (`graphbuild.loaders.load_split`,
which learned JSONL for Phase 8), registered in `DATASETS` (which is **not** in
`stage_b_fingerprint` — verified when MuSiQue-Full was added), raw files under
`data/phase9/raw/` (gitignored, the Phase-8 pattern).

* **Licences are re-verified at download against the primary source.**
  `DATASET_DECISION.md` §8 is explicit that its verification pass did not
  complete and every adoption is single-sourced; nothing here asserts a licence
  the fetch does not confirm.
* **MuSiQue-Ans** extracts from the kept archive (`data/musique/`, 272 MB —
  retained on 15 Aug 2026 for exactly this).
* **The training subset is a declared decision, not an accident of wall-clock.**
  Full 2Wiki train is 167,454 questions; pool prep is embedder-bound
  (`DATASET_DECISION.md` §4: ~4–8 h dev GPU, one-time, for the *planned*
  volume). §6 fixes a per-corpus subset size and a stratification rule
  (by hop count / decomposition depth), seeded — because an undeclared subset
  is an untracked experimental condition.
* **HoVer's pools need distractors the corpus does not ship.** Declared rule
  (§6): gold evidence atoms + distractors sampled from *other claims'* gold
  evidence within HoVer, seeded, up to `pool_cap` — in-corpus, so no new
  retrieval stack enters (boring-stack rule). A declared adaptation with its
  loss stated: these distractors are topically looser than a retriever's.
* **HotpotQA is not promoted.** It remains the training-free search testbed
  only — "it is the surface submodular/PCST were tuned on"
  (`DATASET_DECISION.md` §2, rejected-for-training row); Stage-D training on it
  is explicitly forbidden by that row and this plan keeps the prohibition.
* **Multi-proof credit convention adopted unconditionally**
  (`DATASET_DECISION.md` §2, FEVER/FEVEROUS row): wherever multiple complete
  gold sets exist, a prediction scores sufficient iff **at least one complete
  gold set is a subset of the predicted set**. On 2Wiki/MuSiQue (single gold
  set) it reduces to subset-of-gold-covered; it is declared now so the
  conversational track (where Tier-B gold is *a* minimal proof, not *the* —
  `PHASE7_DECISIONS.md` §3.2c) inherits a convention instead of improvising
  one. FEVER/FEVEROUS themselves stay Tier-2-by-name.

### G11 — The equivalent-action collision instrument carries over to real pools [EVIDENCE + ANALYSIS]

Plan §3.4's action item: instrument the collision rate; if zero, report; if
nonzero, apply the correction. **[EVIDENCE]** Symmetry-Aware GFlowNets
(ICML 2025) proves the bias (Thm 4.6) and corrects by terminal-reward scaling
(Cor 5.1); its uncorrected sampler over-produced the symmetric fragment 5,220 :
1,042. On set-states over atom *ids*, two distinct atoms cannot canonicalize to
the same child **state**; the real-pool analogue is **content-equivalent atoms**
(distinct ids, identical `content_key`) — `H`'s identity check refuses them
*within* one set, but across trajectories they still split flow. The pool-prep
report counts content-key duplicates per pool; the rate ships in the artefact,
and Cor 5.1's scaling is applied only if it is nonzero (expected zero on
paragraph-derived pools; measured, not assumed).

### G12 — Two stages, and what each may honestly claim [ANALYSIS]

* **Stage A (now, once β freezes): the Wikipedia track.** Train the ruled
  9-arm roster (decision B, `PHASE3_DECISIONS.md`: L1–L7, L7b, GAFlowNet) on
  2Wiki + MuSiQue-Ans pools; distill the head; run S1–S5; produce **the
  Gate-3 (real) table** — the architecture's exit criterion and this phase's
  decisive artefact. These are *real numbers*, not machinery numbers: the
  corpora are the declared training interface and the decision rule is G7's.
  What they are **not** is conversational-memory numbers, and the artefact
  says so in its own body.
* **Stage B (post-scope-c): the conversational track.** The same loader
  boundary consumes (Phase-7 `assemble` pools, cached parsed obligations,
  Tier-B gold as amended — `GATE0_CONTRACT.md` item 3.2). Training on it is
  the **transfer ablation** that turns the Wikipedia→conversation gamble into
  a measurement (`DATASET_DECISION.md` §2's LongMemEval-train-slice row);
  deferred by name, the Phase-8 criterion-13 pattern.
* The C3 comparison (L7/L7b vs capacity-matched L6 and GAFlowNet — plan
  §4.5.4's required control) is reported on this phase's real pools as
  *supporting evidence*; **its confirmatory test remains Gate 2 on the
  enumerable environment**, where exact TV exists. Phase 9 must not be written
  up as the C3 verdict.

---

## 2. Scope

**In:** `graft/setgen/` additions — Stage-D pins + fingerprint; the
`ProofExample` boundary + import-graph test; corpus adapters (2Wiki,
MuSiQue-Ans, HoVer-gated); the real `AtomFeaturizer` (F6); the incremental
environment + sampler emitting `Batch`; utility-head distillation; the K = 8
portfolio with fallback-counter wiring; `scripts/phase9_gate3.py`; tests.
Dataset registration in `graphbuild.pins.DATASETS` + `data/phase9/raw/`
gitignore entry.

**Out:** any learner-loss edit (G1 — reuse is the design); reader/orchestrator/
ceiling oracles (Phase 10); baseline systems (Phase 11); Tier-2 learners
(PPO, FL-DB, DB, FM, AgeMem variants) and Tier-2 search (MIP, MCTS, diverse
beam) — named registry stubs exist for them (architecture §14); any frozen
value; any conversational evaluation number (Stage B); Gate-2 claims.

---

## 3. Modules

### P9.0 `setgen/pins.py` — what Phase 9 freezes (do this first)

Dataset pins live in `graphbuild.pins.DATASETS` (single SHA-verified reader);
this module freezes everything else: per-corpus subset sizes + stratification
seed, obligation-synthesis rules (G5), pool-construction constants, featurizer
block names/order (the Phase-8 fingerprint lesson: **names and order, not block
labels** — two feature sets must not hash alike), distillation config, portfolio
constants (referencing config's K and `checker_budget`, never duplicating),
the Gate-3 decision rule of G7 **verbatim**, and the **stage-D fingerprint**
printed by `verify_handoff.py` beside the other four.

### P9.1 `setgen/proofs.py` — the source-agnostic boundary (G3, G4)

`ProofExample` construction: per-question snapshot builder
(Turn/SourceSpan/Assertion/Node, explicit eligibility), pool assembly through
**Phase 7's `build_pool`** (closure, cap, eligibility boundary — one mapping
project-wide, the Phase-7 rule), gold-atom resolution, the adapter registry.
The import-graph test lives with Phase 7's containment guards in
`test_structure.py`.

### P9.2 `setgen/corpora/` — the adapters (G3, G5, G10)

`wiki2.py`, `musique_ans.py`, `hover.py` (gated). Each: SHA-pinned load,
snapshot/pool/gold construction, obligation synthesis, channel scores via the
same bm25s + pinned-embedder arithmetic Phase 8's adapter used (raw + normalised
views, `PHASE8_DECISIONS.md` §2.5), pool-prep cache to disk (embedder cost is
one-time — `DATASET_DECISION.md` §4), and a wiring report: counts, per-slot
synthesis fill rates, content-key collision rate (G11), verify-at-wiring
findings.

### P9.3 `setgen/atomfeat.py` — the real featurizer (F6, G2)

Implements the same surface as `SyntheticFeaturizer` over `ProofExample`s:
`state_repr` (pooled selected-atom features + `d(s)` + size/budget scalars,
STATE_EXTRA-compatible), `action_reprs` (per-candidate features + `Δd` gated by
`delta_d` + STOP slot). Chunked batch-first. The only module that knows what a
feature is; learners see tensors.

### P9.4 `setgen/realenv.py` — the incremental environment + sampler (G1)

`RealEnvironment` (per-example `IncrementalChecker` lifecycle, mask queries,
FAIL detection) and `sample_real_batch(policy, examples, spec) → Batch`.
ε-exploration, uniform-removable `P_B`, terminating transition first-class,
rewards from `R = 1[H]·exp(β·U)` with fix F1 sufficiency. No ledger during
training (Phase-3 rule: metering is the inference path's).

### P9.5 `setgen/distill.py` — the utility head (G8)

Pooled-set MLP + trainer (P6.11 guards), terminal-collection hooks, held-out ρ
report, `scorer(sets) → scores` in the `SearchModule` shape Phase 4 consumes.

### P9.6 `setgen/portfolio.py` — Stage-D inference (G9)

Greedy + sampled rollouts under the ledgered budget, `h_filter`, head ranking,
size tie-break, `binding_of` + contested diagnostic (gold-bearing eval only),
`abstain_fallback` counter increment, per-query report (spend, distinct-valid
count — reported, never assumed K; Phase-4 decision 3).

### P9.7 `scripts/phase9_gate3.py` — the runner and the Gate-3 (real) artefact

Stage-A track: pool prep (cached) → train 9 arms × 3 seeds at the declared
budget → distill → run S1–S5 + all arms' portfolios on held-out dev pools →
the Gate-3 table with paired bootstrap, the size-controlled diversity
secondary, head-ρ beside every row, per-question listing,
`digest_of(deterministic_view(...))` determinism digest, honesty stamp,
manifest, stage-D fingerprint. Refuses to *train* while β is unfrozen (names
Phase-3 step 6); refuses the conversational track by name until scope-c exists.

### P9.8 `graft/tests/test_setgen_real.py` (+ `test_structure.py` additions)

Per §5. Fixture `ProofExample`s built by hand (the Phase-7/8 fixture pattern);
StubEmbedder throughout; the two structural guards (import-graph boundary,
head-outside-`H`); learner-files-untouched assertion (hash or import test);
trainer smoke on 2-epoch budgets.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 0 | P9.0 pins + fingerprint | importable bare; in `verify_handoff.py`; Gate-3 rule text frozen in pins |
| 1 | P9.1 proofs boundary | fixture ProofExample round-trips; import-graph test green; H-subset behaviour (G4) fixture-tested |
| 2 | P9.2 wiki2 + musique_ans | SHA-pinned loads; wiring reports written; pool-prep cache works on a 100-question slice; verify-at-wiring findings recorded |
| 3 | P9.3 featurizer + P9.4 environment | Batch shape byte-compatible with Phase 3's losses (one TB step runs on a fixture); terminal convention honoured; `delta_d` gate leak-tested both directions |
| 4 | 9-arm training smoke | every arm trains 2 epochs on fixtures, no NaN at the reward floor; capacity match (**Phase-3** decision 11's form — not this plan's, which is the budget) recomputed at real dims and printed; **every arm's throughput measured in trajectories/second and the slowest identified, then `N_real` derived from decision 11's ladder and written into pins** — before any scored run |
| 5 | P9.5 distillation + P9.6 portfolio | head trains with guards; ρ reported; portfolio spends ≤ 32 with the 0/1 family split; fallback counter increments on a forced dead-end fixture |
| 6 | P9.7 Stage-A runner **[β unblocked 15 Aug 2026; runner unwritten]** | full Wikipedia-track run: training → Gate-3 table, three seeds, artefact audited (per-question rows, digest) |
| 7 | HoVer adapter (gated) | added to the table as a third training corpus or explicitly deferred with reason |

Steps 0–5 are CPU + fixtures (no GPU, no downloads beyond step 2's fetch).
Step 6 costs: pool prep ~4–8 h dev GPU **one-time**; training ~33 h CPU serial
for 7×3 per `DATASET_DECISION.md` §4 — the ruled 9-arm roster scales that to
~42 h serial, runs independent, parallel across cores. **Stage B is
post-scope-c and is not in this table** (G12).

---

## 5. Exit criteria

**The machinery is sound**
1. Every learner file in `graft/setgen/learners/` is byte-identical to its
   Phase-3 state (asserted) — fix F6 held.
2. The sampler's `Batch` passes one gradient step of every arm on fixtures; the
   terminating transition carries its own slot (terminal convention).
3. The import-graph boundary holds: nothing above `proofs.py` names a corpus;
   nothing in `H`'s import graph names the distilled head; the head's inputs
   contain no gold field. All structural, all in `test_structure.py`.
4. `delta_d` reaches L7/L7b features and no other arm's, tested in both
   directions (Phase-3 criterion 5's form, real featurizer).
5. Pools are Phase-7 pools: closed, capped, eligibility-enforced, edge-refs
   validated — through `build_pool`, not a re-implementation.
6. G4's H-subset is fixture-tested: an ineligible atom is refused; a duplicate
   content_key is refused within a set; temporal/binding checks are confirmed
   vacuous on a Wikipedia fixture.

**The data is honest**
7. Both Tier-1 corpora load SHA-pinned with licences recorded at fetch; wiring
   reports carry counts and every verify-at-wiring finding; subset +
   stratification are the pinned ones.
8. Obligation-synthesis fill rates per slot are in the artefact (G5); the
   multi-proof credit convention is stated in pins (G10).
9. Content-key collision rate reported per corpus (G11); correction applied
   only if nonzero.

**The experiment is decidable**
10. The Gate-3 rule was frozen in pins **before step 6 ran** (git evidences
    order); the artefact carries the rule text it was scored under.
10a. **`N_real` is identical across all 27 runs and came from decision 11's
    ladder read on the *slowest* arm's measured rate** — fix F12's fixed-budget
    primary. Every rung tried is recorded, not only the adopted one, and the
    per-arm trajectories/second table ships in the artefact so a budget-bound
    arm is visible rather than inferred (the Phase-3 §2.3 failure mode: a
    negative C3 verdict that is about budget). The training subset is
    decision 2's pinned one, with its stratification counts reported.
11. Best-of-K at K = 8 / budget 32, exact-`U` scored, head-selected, three
    seeds, paired bootstrap — every trained arm and S1–S5, same held-out pools.
12. Head fidelity (held-out Spearman ρ) beside every row; the size-controlled
    diversity secondary beside the raw one.
13. Budget enforced not observed (`would_exceed`), spend per method in its own
    row, distinct-valid counts reported never assumed.
14. `r_fail` gap measured on real pools at the frozen β and reported (G6).
15. The fallback counter increments on budget exhaustion and appears in the
    artefact (G9) — Phase 8's reserved loop closed.

**Discipline**
16. Per-question listing, `digest_of(deterministic_view(...))` digest equal
    across two runs, manifest, stage-D fingerprint in `verify_handoff.py` —
    the Phase-8 audit's artefact standards, adopted from the start.
17. The artefact's honesty stamp states: Wikipedia-track numbers are the
    declared training-interface result, **not** conversational-memory numbers;
    C3's confirmatory test remains Gate 2 (G12).
18. **Deferred by name to Stage B (post-scope-c):** the conversational training
    track, the transfer ablation, every conversational number.
    **Transferred by name to Phase 10:** the inference-time contested
    comparison (G9); `abstain_cause` record-writing.

---

## 6. Decisions to lock before writing code — **SIGNED OFF 16 August 2026**

**Signed:** Nazib Riasat — 16 August 2026. *(Signed by the assistant at the
project owner's explicit instruction, recorded as delegated rather than
presented as the owner's own hand — the `GATE0_CONTRACT.md` and
`GRAFT_PHASE3_BUILD.md` §6 convention.)* All fourteen decisions are ruled **as
written in the table below**, including decisions 2 and 11, whose values were
set on the assistant's recommendation on 16 August 2026 and are no longer
provisional.

**Signed now, and the timing is the strongest this project will ever have.**
`GRAFT_PHASE2_BUILD.md` §6b's decision-rule procedure requires that **no learner
results be inspected** before a decision rule is set. Phase 9 has **no code at
all** — not a rollout, not a fixture, not a throughput measurement — so decision
7, the Gate-3 (real) rule that Phase 4 explicitly deferred to this phase, is
being fixed at a moment when there is literally nothing to have peeked at. Phase
4 could not say that of itself: its own G9 measurement existed before its §6 was
ruled, which is why that gate's *decision* moved here in the first place. After
build step 6 runs, any edit to decision 7 is contaminated whether or not the
instrument moved.

**Four things this signature does not cover**, each with its named source:

1. **β and the `r_fail` gap** (decision 6) — produced by **Phase-3 step 6's
   calibration gate**, not chosen here. The *procedure* is signed; the numerals
   are that gate's output. Eligible candidates remain {4, 8}.
2. **`N_real`'s numeral** (decision 11) — the **ceiling (2 h) and the ladder
   (50k → 100k → 200k) are signed**; the number is *derived* at build step 4
   from the slowest arm's measured trajectories/second. A rate is a property of
   the machine and the architecture, not a learner result, so deriving it later
   is §6b-clean — the Phase-3 decision-4 argument, reused deliberately.
3. **Licences** (decision 1, G10) — re-verified against the primary source **at
   download**. `DATASET_DECISION.md` §8 records that its verification pass did
   not complete, so signing decision 1 adopts the corpora *subject to* that
   check, and a licence that fails it retires the corpus rather than the check.
4. **Evidence status.** Most cells here are **[ANALYSIS]** — engineering and
   protocol judgment, not paper-backed. Signing changes ownership, not
   evidential class, and the write-up must still label them as such. The
   paper-backed ones say so and name the venue (decision 7's Robust Scheduling
   framing and Dror et al.; decision 4's presence-flag rationale; G11's
   Symmetry-Aware GFlowNets correction).

**This table is normative.** Where a restatement elsewhere disagrees with it,
the table wins and the restatement is a bug — the Phase-2 mitigation adopted
after five review rounds in which most defects were a fix landing in three
places out of four.

| # | Decision | Ruled value | Cost if changed later |
|---|---|---|---|
| 1 | Training corpora (Stage A) | 2Wiki + MuSiQue-Ans (Tier 1); HoVer gated third (G10); HotpotQA **never** for training; FEVER/FEVEROUS Tier-2-by-name | re-train everything; a promoted HotpotQA breaks the search-baseline testbed's independence |
| 2 | Subset + stratification | **2,000 train + 500 dev per corpus** (4,000 / 1,000 total), balanced 1:1 between 2Wiki and MuSiQue-Ans so neither dominates the Gate-3 table; **stratified** by 2Wiki question type (comparison / inference / compositional / bridge-comparison) and MuSiQue hop count (2/3/4); question-level dev split, drawn before training; seed `20260816`. **Escalation ladder declared now: 2,000 → 5,000 → 10,000 per corpus**, stepped only on a *learning-curve* reading (arms still improving at the budget ceiling), never on a Gate-3 result. **[ANALYSIS]** sized from a measured anchor — Phase 8 embedded MuSiQue-Full **cold** in 1,146 s (`artefacts/phase8_gate.json`; its `embed_cache` was empty, so nothing was reused), giving **~300 texts/s on the dev GPU**, which puts this subset's one-time pool prep at **~17 min** against ~13 h for the full corpora. Cheap prep is the point: Phases 7 and 8 both had to regenerate artefacts after audit, and a subset that costs a coffee break to rebuild keeps that affordable | an undeclared subset is an untracked condition; resizing after results is selection. The ladder is what makes a later increase a rule rather than a reaction |
| 3 | Environment | incremental, over Phase-1 `IncrementalChecker`/masks; Batch-shape compatible; losses untouched (G1) | any learner edit voids the Phase-3 comparison lineage |
| 4 | Featurizer blocks | G2's table verbatim, presence flags mandatory, `delta_d` gated to L7/L7b | the C3 control dies (leak) or the arms differ by more than the arm |
| 5 | Obligation synthesis | G5's per-corpus rules, in pins, fill rates reported | coverage term silently means different things per corpus |
| 6 | Reward | fix F1 sufficiency vs gold; frozen `u_weights`/β/`r_fail`; **no training before β freezes**; real-pool `r_fail` gap measured | reward engineering confound; or FAIL semantics silently corrupted |
| 7 | **Gate-3 (real) decision rule** | G7 verbatim — primary best-of-K@8/32 exact-`U`-scored under head selection, paired bootstrap, 3 seeds; consolidation clause; size-controlled diversity secondary | the §6b failure: a rule chosen after seeing results is not a rule |
| 8 | Distilled head | pooled mean⊕max MLP, P6.11 guards, ρ reported per row, structurally outside `H` and gold-free (G8) | Gate 3 unreadable (no noise measurement); or the multiplicative gate property dies |
| 9 | Portfolio | 1 greedy + 7 sampled, `h_filter`, head rank, size tie-break; fallback counter wired; contested = gold-alias diagnostic here, reader comparison Phase 10's (G9) | fix F4 unimplemented, or unstated gold-free machinery invented |
| 10 | Learner roster | the Phase-3 ruled nine (L1–L7, L7b, GAFlowNet), fresh policies, same TrainSpec discipline, capacity match recomputed at real dims | the roster decision re-litigated; capacity confound returns |
| 11 | Training budget | **The ceiling and the ladder are frozen here; `N_real` is derived, not guessed.** Ceiling **2 h per run**; ladder **`N` ∈ {50,000 → 100,000 → 200,000}** trajectories, at most two escalations then stop. `N_real` = the largest rung the **slowest arm** completes within the ceiling at its **measured** rate, taken at build step 4 — a rate in trajectories/second is a property of the machine and the architecture, not a learner result, so measuring it first is §6b-clean (the Phase-3 decision-4 argument, verbatim). **`N_real` is identical across all 27 runs** (9 arms × 3 seeds); checkpoints are monitoring only, **no early stop for any arm** (G1). **Planning midpoint 50,000** — **[ANALYSIS]**, projected not measured: Phase 3 clocked the LED arms at ~900 traj/s on the synthetic profile (`pool_cap` 32, `max_atoms` 8, ~16-dim atoms), and Phase 9's profile (64 / 16 / ~400-dim) is ~50–150× costlier per trajectory, so ~6–18 traj/s → 50,000 ≈ 1.4 h/run, **~37 h serial, ~5 h wall clock across this machine's 16 logical cores**. `DATASET_DECISION.md` §4 independently estimated "~33 h, worst case ~7 d" for this row, which brackets the same ladder | fix F12's fixed-budget primary breaks; or `N` sized on the cheapest arm again — Phase 3 did exactly that (L5 ~2,700 traj/s sized a budget the LED arms spent at ~900, buying L7 a 2.9× overrun its own ladder guard could not see), and the same ~3× L5:L7 gap will exist here |
| 12 | Collision instrument | count content-key duplicates; Cor 5.1 scaling iff nonzero (G11) | a biased sampler read as a method property |
| 13 | Stage split | A = Wikipedia track now (post-β); B = conversational post-scope-c, deferred by name; C3 verdict stays at Gate 2 (G12) | conversational or C3 claims made from the wrong table |
| 14 | Artefact standards | per-question rows, real digest, honesty stamp, fingerprint — the Phase-8 audit's findings as *requirements* here | the same six defects, one phase later |

---

## 7. Explicitly not in Phase 9

No learner-loss modifications; no reader, orchestrator, or ceiling oracles; no
system baselines; no Tier-2/3 learners or search methods (registry stubs only);
no LongMemEval/LoCoMo numbers; no Gate-2 or C3-confirmatory claims; no exact
TV on real pools (impossible and not approximated — **[EVIDENCE]** exact
evaluation belongs to enumerable spaces, Shen et al. ICML 2023 / When Do
GFlowNets Learn the Right Distribution?, ICLR 2025); no retuning of any frozen
value; no new embedder, index, or retrieval machinery (Stage C is frozen for
this phase); no training on HotpotQA.

---

## 8. What Phases 10 and 11 get from this, verbatim

* **The trained Stage-D stack behind two calls**: `portfolio.run(question
  inputs) → ranked valid sets + spend + fallback events`, and the distilled
  `scorer` — Phase 10's orchestrator wires gate → this → reader without
  learning what a learner is.
* **The `abstain_fallback` counter, live** — with `gate.decide()`'s "gate"
  cause, both of plan §4.2's abstention routes are now countable, so §6.4's
  fallback-trigger rate is reportable from the first end-to-end run.
* **The contested hook** (`binding_of` + the transferred reader-comparison
  criterion) for fix F4's inference-time form.
* **The Gate-3 (real) verdict** — either Stage D's learning claim survives at
  equal budget under a noisy scorer, or Phase 10/11 proceed with the project
  consolidated on Contribution 1; both outcomes are planned for.
* **The stage-D fingerprint**, sixth in the identity tuple printed by
  `verify_handoff.py`.
