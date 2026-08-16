# PHASE9_DECISIONS.md — what the Phase-9 build decided and measured

**Status: build steps 0–5 complete; step 2 (corpora) and step 4 (throughput +
`N_real`) complete as of 16 August 2026. Step 6 — the scored Stage-A run —
remains blocked on Phase-3 step 6 freezing β.**

**A second adversarial audit ran on 16 August 2026 and found five blockers —
three in the reviewed claims, two the sweep found on its own.
Read §7 before quoting anything from §2.2, the capacity table, or any C3 row.** One
of the three would have invalidated the L7-vs-L6 comparison the phase exists to
make: the step-4 script matched capacity under the tolerance §6.4 *retired*, and
its fallback handed `l6_led` **less** live capacity than L7. All confirmed
findings are fixed, regression-tested and re-measured; §7 is the record.

Companion to `GRAFT_PHASE9_BUILD.md` (§6 signed 16 Aug 2026). Where this file
and the build plan disagree, **this file wins and the plan is the record of what
was intended** — the Phase-4/5/6/7/8 convention.

---

## 1. The signed decisions the build overturned

### 1.1 Decision 9's `h_filter` clause is wrong, and the error ran against the proposed method

§6 decision 9 as signed reads: *"1 greedy + 7 sampled, `h_filter`, head rank,
size tie-break"*. The build did **not** implement the `h_filter` step.

**Why.** Phase 4 had already settled this in a docstring nobody re-read at
signing time. `search.base.h_filter`:

> *"This is the direct-builder path only. S1, S2 and S5 construct through the
> masks, where `stop_allowed` **is** `H`, so they are valid by construction and
> must not come through here — routing them through it would charge them a check
> each and erase the family difference G6 exists to expose."*

`s5_portfolio.py` reports `terminal_checks=0` for exactly that reason. The
Stage-D portfolio constructs through the same masks, so the signed text would
have charged it **8 terminal checks it does not owe** against a budget of 32.

**Why it matters more than the arithmetic.** The 0-vs-1 check-family split is
what Gate 3's budget row reports: constructive methods (S1/S2/S5 and this
portfolio) spend 0, direct builders (S3/S4) pay 1 per distinct candidate.
Charging the portfolio would have collapsed that distinction *and* handicapped
the learned method in the comparison it exists to win or lose fairly. It is the
`CLAUDE.md` §5 pattern — an error that flatters or penalises the proposed method
— running in the unusual direction, and it survived a signature.

**Provenance of the error:** the decision-9 text was written from fix F4's prose
("sample K, `H`-filter, rank") without checking it against Phase 4's measured
design. Fix F4's prose is right about the *pattern* and wrong about who pays.

**What was built instead.** Spend is 0 terminal checks; validity is by
construction; `would_exceed` is still consulted before any check the module
*does* make, so the budget is enforced-not-observed (exit criterion 13) even at
zero spend. An `audit_validity()` path re-checks delivered sets against batch
`H` **off the measured path** — and finds **0 of 8 invalid**, which is what
turns "valid by construction" from a claim into a test.

### 1.2 `abstain_fallback` is not a ledger meter, and should not become one

Decision 9 says the fallback counter is "wired". The natural reading is
`ledger.count("abstain_fallback")` — which raises, because `abstain_fallback` is
not in `ledger.METERS`.

That is correct and was left correct. `METERS` is a closed vocabulary of
**budget-enforced resources**: things that are spent, capped, and refused at the
cap. An abstention has no cap and is not spent — it is an *outcome*. Phase 8
reserved `abstain_fallback` as an **artefact field** and wrote it as one
(`{FALLBACK_COUNTER: 0, "note": "zero until Phase 9 wires it"}`). Phase 9 writes
it as one too: `PortfolioResult.extra["abstain_fallback"]`, aggregated by the
runner. Adding an uncapped meter to the enforcement path would have made
`would_exceed` meaningless for it.

### 1.3 The 2Wiki `entity_anchor` rule was a gold leak, and §6 decision 5 is amended

`pins.OBLIGATION_SYNTHESIS["wiki2"]["entity_anchor"]` as signed read **"subject
of the first evidence triple"**. The adversarial audit (§3a) called it a leak and
was right.

`evidences` is 2WikiMultiHopQA's **own annotation** — one of the fields its
evidence-F1 metric scores against. It exists on train and dev and **nowhere at
inference**, where the frozen obligation parser reads the *question*. Measured on
the full 12,576-row dev split, the first evidence subject:

| | rate |
|---|---|
| matches a **gold** paragraph title | **92.5%** |
| matches a **distractor** title | 9.3% |

So `entity_anchor_hit` was a near-exact gold-paragraph indicator — and it does
not stop at one feature. The anchor drives `d(s)`'s first deficit component, so
it reaches `state_repr` for **every arm**, the `Δd` block for L7/L7b, and,
through the pooled atom matrix, the distilled head — whose docstring guarantees
that no gold reaches an input. Every Stage-A Gate-3 number would have been
measured under supervision unavailable at deployment.

**The amended rule: the longest context title that appears in the question.**
Both halves are available at inference — the question is the query, and candidate
titles are what retrieval is ranking — so no annotation field is read. Measured
on the same dev split: **100.0% filled**. `supporting_facts` and `evidences` are
now untouched by `synthesise_obligations`, and a regression test asserts that
*stripping* `evidences` does not move the anchor — asserting the absence of a
dependency is the only way to keep it absent.

**One thing the fix does not do, stated because it would be easy to imply
otherwise.** The new anchor still coincides with a gold title on **100%** of dev
rows. That is no longer a *leak* — only the question and the candidate titles are
read — but it is a real property of 2Wiki as a training surface: its questions
name their bridge entity, and the bridge entity's paragraph is supporting by
construction. Any retriever at inference would exploit the same correlation,
which is what makes it legitimate signal rather than leakage; but a reader of a
Stage-A number should know that on this corpus the anchor feature sits close to a
gold oracle *by dataset design*. Same class of finding as
`PHASE7_DECISIONS.md` §3.1's "recall 1.000 by arithmetic": not a bug, and not
something to quote without the caveat. MuSiQue-Ans is unaffected — its anchor
comes from the decomposition DAG's hop-1 subject, a question decomposition rather
than an evidence label.

---

### 1.4 §6 decision 12's collision instrument is amended — the measured quantity changed underneath it

**Signed decision 12 measures "content-key duplicates per pool" and applies a
Symmetry-Aware correction if the rate is nonzero.** The build discovered the
rate is **structurally zero** — `content_key` is `(kind, target, label, *refs)`
and `target` derives from the assertion id, itself derived from text and spans,
so two atoms sharing a content key already share an `atom_id` and `AtomPool`
collapsed them before any pool existed. That is a **proof, not a measurement**,
and G11 is discharged by reporting it as one.

The build then substituted a quantity that *can* vary — **distinct atoms with
identical normalised text**, 3.86% of 2Wiki dev rows — and asserted that
Symmetry-Aware GFlowNets' Cor 5.1 corrects it. **That assertion is wrong, and
the substitution was made under a signed normative table without amending it.**
Both halves are corrected here.

**Why Cor 5.1 does not apply.** Read verbatim from the paper (ICML 2025), the
corollary scales a terminal by `|Aut(G)|` — the **graph automorphism group** —
under a node-by-node generation scheme, where orbit-equivalent construction
*actions* build the same graph. GRAFT's state is a set of atom **ids**: two
identical-text paragraphs carry different ids, different spans and different
provenance, so they are genuinely different terminals, not one object reached
twice. There is no automorphism group here to quotient. Applying a scalar would
be an ad-hoc correction wearing a citation, which is `CLAUDE.md` §5's catalogued
failure mode — and provenance-preservation is Contribution 1's own claim, so
collapsing two provenances would contradict the thesis.

**The bias is real, and it is still not ours to correct.** Measured by
enumerating a real pool's 20 valid singleton terminals: a duplicated content
class carries exactly **2.0×** the reward mass an equivalent de-duplicated one
would. So "it would be a harmless no-op" is *also* false, and both wrong
readings are recorded — the tempting fix and the tempting excuse. The 2× is
arithmetic of the corpus, not a sampler symmetry, and on the pinned 200-row
sample **all 9 duplicate groups are non-gold** (0/9 touch a gold paragraph), so
the practical exposure is small and now reported rather than asserted.

**Amendment adopted:** `content_key_collisions` stays, reported as the proof it
is; `equivalent_evidence_atoms` is a **dataset diagnostic** with no correction
attached; `COLLISION_INSTRUMENT["apply_when"]` reads *never on this MDP*. One
further pin was factually wrong and is fixed with it: it claimed "`H`'s
sub-check 2 refuses them within one set", which is true of content-key-identical
atoms and **false** of distinct-id atoms with identical text — measured,
`H([A, B])` returns `ok=True` with zero violations, because their content keys
differ.


## 2. What the build measured that reading could not settle

### 2.1 Document granularity on 2Wiki — decided by measurement, against the plan's phrasing

G3 says 2Wiki's *"gold atoms = the atoms derived from supporting sentences"*,
and `supporting_facts` really are `[title, sentence_idx]` pairs. That reads as
sentence-level documents. Measured over 40 dev rows at `pool_cap = 64`:

| granularity | closed pool size | gold survives the cap |
|---|---|---|
| **paragraph** | 29 – 30 | **40 / 40** |
| sentence | 34 – 64 | 28 / 40 |

Sentence-level puts the closed pool **on the cap** and drops a gold atom on 30%
of questions. An example with partial gold is not a harder instance — its
`sufficiency` can never reach 1.0, so its maximum achievable reward sits
silently below every other example's, and training on it biases the budget
toward whatever the pool happened to fit.

**Decision: documents are paragraphs, on both corpora.** The plan's
sentence-level intent is honoured as *a paragraph is gold iff it contains a
supporting sentence*. MuSiQue annotates paragraphs natively, so this also makes
the two Tier-1 corpora commensurable in a single training run — a Gate-3 row
that pools them compares questions rather than schemas.

### 2.2 `N_real` — derived, and the signed projection was wrong

Decision 11 froze the ceiling (2 h) and ladder (50k → 100k → 200k) and left
`N_real` to be derived at build step 4. It also carried an `[ANALYSIS]`
projection: ~50–150× the synthetic per-trajectory cost, giving **6–18 traj/s**.

Measured on real pools (24 examples, 29–60 atoms, the true distribution),
**after** §3a's Δd fix — which made the gradient-time Δd block dense over legal
actions and cost L7/L7b ~28% throughput, the correct cost since a sparse Δd is
not the feature C3 is about:

| | arm | traj/s |
|---|---|---|
| **slowest** | `l7b_aux` | **30.06** |
| fastest | `l2_imitation` | 513.32 |
| spread | | **17.08×** |

**Re-measured after the §7 audit, and the headline did not move.** The first
reading (29.67, `l7b_aux`) was taken on a *head slice* of each dev split and
from a *single* wall-clock sample per arm. §7.4 fixed both — the sample is
stratified, and each arm is now timed three times with the slowest taken. The
corrected figure is **30.06**, inside the noise of the original, and `N_real`
never moved. That is the useful result: the derivation turned out robust to a
defect that was genuinely present in it. §7.4 also records the measured
*direction* of the sampling error, which was the opposite of the obvious guess.

The real ratio is ~30×, not 50–150×. **The projection was wrong in the safe
direction, and it is left in `pins.py` rather than quietly corrected** — the
whole point of decision 11 is that `N_real` is derived, and this is the evidence
that a frozen guess would have been off by ~2×.

**All three rungs fit** (0.47 h / 0.94 h / 1.87 h against a 2 h ceiling), so
`N_real = 200,000` is adopted outright and the ladder never escalated — **but the
top rung uses 92% of the ceiling**, against 67% before the Δd fix. Three
consequences recorded rather than left to be discovered:

1. **The ceiling is no longer slack**, so it certifies nothing about convergence
   and any further per-trajectory cost drops the adopted rung to 100,000 — a
   *derivation* changing rather than a decision being revisited. If 200,000
   leaves L7b under-converged there is no rung above it, and adding one is a §6b
   decision-rule amendment. `PHASE3_DECISIONS.md` §2.3 is the standing warning:
   an `N` that under-serves the slowest arm hands Contribution 3 a negative
   verdict that is about budget.
2. 200,000 × 27 runs at 30.06/s is **~49.9 h serial**, ~6.2 h wall clock across
   this machine's 16 logical cores. `DATASET_DECISION.md` §4's independent
   estimate for this row is "~33 h, worst case ~7 d", which still brackets it.
3. **The slowest arm moved** from `l7_checker_led` to `l7b_aux` once the Δd fix
   landed — which is what should happen, since L7b carries L7's Δd cost plus an
   auxiliary head. The pre-fix ordering was an artefact of the sparse block.

The 19.77× spread is itself the justification for the "slowest arm" rule: sizing
on L1 would have bought L7b a twentieth of its budget — Phase 3's own 2.9×
overrun one phase later, and an order of magnitude worse.

### 2.3 Verify-at-wiring findings, all measured on the real splits

**2WikiMultiHopQA** (Apache-2.0, licence re-verified at download 16 Aug 2026;
COLING 2020), dev = 12,576 rows:

* Context is **always 10 paragraphs** (12,576/12,576), pre-split into sentences.
* **Every** supporting-fact title resolves into the context — 0 failures.
* **3.7% of rows (463) carry duplicate context titles.** Titles are therefore
  not unique keys within a question, and the first adapter keyed documents by
  title and was refused by `build_snapshot`'s duplicate guard. Documents are
  keyed by paragraph **index**. Crucially, **0 rows have a duplicated title
  among their supporting facts**, so `[title, idx]` is never ambiguous for gold:
  the corpus's ambiguity is real and never touches a label.
* The first evidence subject is a context title on only **82.3%** (10,346) of
  rows, so the synthesised `entity_anchor` is an entity *string* rather than a
  guaranteed title — which is exactly how `atomfeat` consumes it.
* The four question types match `pins.SUBSET["stratify"]["wiki2"]` exactly:
  compositional 5,236 · comparison 3,040 · bridge_comparison 2,751 ·
  inference 1,549.

**MuSiQue-Ans** (CC BY 4.0), dev = 2,417 rows:

* 20 paragraphs with `is_supporting` — paragraph-level gold natively.
* Hop questions are `"<subject> >> <relation>"`; later hops reference `"#1"`.
  `hop_subject` returns `None` for a `#k` subject: returning it would create an
  `entity_anchor` slot `coverage` counts as active and no atom can satisfy,
  inflating the denominator on exactly the multi-hop questions the corpus exists
  to supply.
* `answer_aliases` is frequently empty, so the adapter puts the answer itself
  first in the alias list `portfolio.binding_of` consumes.

**Both corpora, on 200-row stratified samples:** pool 28–30 (2Wiki) and 45–60
(MuSiQue-Ans), `gold_complete` **200/200**, `entity_anchor` fill **100%**.
Strata: 2Wiki compositional 84 / comparison 49 / bridge_comparison 43 /
inference 24; MuSiQue **104 / 63 / 33** across 2/3/4-hop.

G11's instrument reports two numbers after §3a's A8: `content_key_collisions`
is **0 and structurally must be** (ids are content-derived, so a collision would
already have been a shared `atom_id`), and `equivalent_evidence_atoms` — distinct
atoms carrying identical normalised text, which is what actually splits flow —
reads **9 across 200 2Wiki examples** and 0 on MuSiQue. Non-zero, so Cor 5.1's
terminal-reward scaling is now a live question for the Wikipedia track rather
than a discharged one; at 9 atoms in ~6,000 it is 0.15% and the decision of
whether to apply it belongs with the first scored run.

MuSiQue's ~60-atom closed pool against `pool_cap = 64` means **the cap barely
binds**, which is `PHASE7_DECISIONS.md` §3.1's saturation finding one stage
later: there, every conversation held 8–23 candidates against the same cap and
recall was 1.000 *by arithmetic*. Reported per split rather than discovered in a
result.

---

## 3. Defects the build found, all fixed and regression-tested

| # | Defect | Consequence had it shipped |
|---|---|---|
| 1 | `sample_real` did not clear `live` when every trajectory dead-ended | A correct all-`FAIL` rollout raised "the ADD mask is not shrinking" — a **misdiagnosed masking bug on exactly the case the abstain fallback exists for**. Found by the fallback test. |
| 2 | `log_r_range` was seeded from `hash()` | Python salts string hashing per process, so decision 13's normaliser would differ on every launch. "Frozen before training" has to survive a restart. Now a content digest. |
| 3 | `about_entity` edges were built without `provenance` | The schema refused them, correctly: *"a schema that permits an unsourced edge permits an unsourced proof"*. The document's own span is the warrant and is now supplied. |
| 4 | A mis-bound ternary in the channel-score block (`float(x), float(y) if size else (0,0)`) | Would have raised on an empty pool rather than returning zeros. |
| 5 | Console output used `Δ`/`φ` on a cp1252 Windows console | The runner crashed **after** writing its artefact — a successful run reporting failure. ASCII in the console, Unicode in the artefact. |
| 6 | `test_the_head_and_the_features_are_gold_free` grepped the source for `gold_atom_ids` | Failed on `distill.py`'s own docstring, which names the field *in order to say it must not be read*. A guard that forbids documenting the rule it enforces is a guard that gets deleted. Now reads the AST. |
| 7 | `test_scoring_agrees_with_the_phase8_implementation` used `approx` on nested dicts | `pytest.approx` does not recurse, so identical values compared unequal. A comparison that reports a difference where there is none is worse than none. Now compares leaf by leaf. |

---

## 3a. The 16 August 2026 adversarial audit — 16 confirmed findings

Four independent reviewers over distinct dimensions (plan-vs-code, batch
correctness, leakage, adapters), **each finding then adversarially refuted** by a
separate agent instructed to default to "not real". 25 findings raised, 20
adjudicated, **16 confirmed and 4 refuted**. I re-verified every confirmed
finding against the code myself before acting; the ones below are the ones that
changed behaviour.

### The two that would have produced a false negative for Contribution 3

**A1 — `Δd` reached L7/L7b's policy sign-inverted *and* sparse at gradient time.**
`build_real_batch` computed `child − parent` where `sample_real` computed
`parent − child`, and filled only the *taken* action's column where the sampler
filled every legal one. `ob.delta_deficit` is `before − after`; the synthetic
`_delta_table` is dense over all edges; `batch.delta` itself used the correct
sign — so the batch builder disagreed with **every other Δd producer in the
codebase**.

Both halves ran against C3. The inverted sign trained L7's policy to read
obligation *discharge* as accrual. The sparse block turned Δd from a
**comparison across candidate actions** into a label attached to the choice
already made, so "prefer the ADD that discharges more obligation" — the entire
C3 mechanism — was unlearnable. Neither would have shown up in a loss curve.
`features.py` lines 123–137 document this exact failure being found and fixed
once already on the synthetic side, with the same predicted consequence quoted
verbatim: *"a false negative for Contribution 3"*. That it recurred on the real
side, in a phase whose plan cites that docstring, is this build's strongest
argument that the adversarial pass is not optional.

*Fix:* one shared `legal_deltas()` helper called by both paths; a test asserts
there is exactly one implementation. **Cost, re-measured not estimated:** L7/L7b
lost ~28% throughput and the slowest arm changed identity (§2.2). That is the
correct cost — a sparse Δd is not the feature C3 is about.

**A2 — the 2Wiki `entity_anchor` was read from the gold evidence annotation.**
§1.3.

### The rest, by what they broke

| # | Finding | Fix |
|---|---|---|
| A3 | **The `{channel}_raw` columns carried normalised values.** The adapters key the normalised view under the bare channel name and the raw magnitudes under `{channel}_raw`; the featurizer read the bare name into the column *called* `_raw` and min-maxed it again. Min-max is idempotent, so both columns were bit-identical and the genuine BM25 magnitude — computed, carried on the example — was read by nothing. `PHASE8_DECISIONS.md` §3.3 one phase later, and the fingerprint could not see it because the *names* never moved. | Read `{channel}_raw` for the raw column, with a documented fallback for adapters that have no natural magnitude. |
| A4 | **One block-level presence flag where the contract promised per-channel.** Four of six channels were constant zero on the Wikipedia track while `channel_scores_present` read 1.0 — so the policy could not tell "did not run" from "ran and scored zero", and Stage B would have differed from Stage A by an uninstrumented change in feature *semantics*. | Per-channel `{channel}_present` columns. |
| A5 | **`scope_ok` was inverted and identically zero.** It read `1.0 if not scope`, and `scope` is always set — `build_snapshot` makes the question id the `conv_id` precisely so `H`'s scope sub-check binds. | Reads the same `resolve.conv_ids` walk the checker makes, so feature and checker cannot disagree. |
| A6 | **`stage_d_fingerprint()` imported torch.** `frozen_values()` pulled `BLOCK_FEATURES` from `atomfeat`, which imports torch — so the *module* imported clean on a bare interpreter and the *call* did not, which is the harder version to notice. `verify_handoff.py` makes that call. | Names lifted into a torch-free `featurenames.py`; `atomfeat` imports from it, so there is still one definition. |
| A7 | **`training_blocked_reason()` checked only that a file existed.** An empty file, a `--quick` calibration, or a record whose adopted β disagreed with the config all read as "frozen" and would have let a scored run start at the placeholder 4.0. | Reads the record: a rung must have been *adopted*, β must be one of the eligible `{4, 8}`, and it must equal the config's. `record` is now a parameter, because a function testable only by monkey-patching `Path` is a function with a missing parameter — the first version of its test recursed into itself. |
| A8 | **G11's collision instrument could never read non-zero.** `content_key` is `(kind, target, label, *refs)` and `target` is a node id minted from the assertion id, itself minted from text and spans — so two atoms sharing a content key would already share an `atom_id` and `AtomPool` would have collapsed them. "Measured, not assumed" was hollow: it was a *proof*, reported as a measurement. | Both are reported. `content_key_collisions` stays, so the structural proof is visible; `equivalent_evidence_atoms` — distinct atoms carrying identical normalised text, which is what actually splits flow and what Cor 5.1 corrects — is the measured one. **It reads 9 on 200 2Wiki examples**, so the instrument now detects something the old one could not. |
| A9 | **`wiki2` hardcoded `answer_aliases` to `[]`**, structurally disabling `portfolio.binding_of` and the contested diagnostic on that corpus: an empty alias list matches nothing, so every 2Wiki set bound nothing and `contested_rate` reported `bound: 0` unconditionally. | The answer itself is the alias list, matching the MuSiQue convention. |
| A10 | **`real_gold_batch` allocated a zero Δd tensor and passed it unfilled.** Dead today (no arm is both `supervised` and `delta_d`), live the moment a supervised C3 ablation is added — at which point it would read as "checker-conditioning adds nothing under supervision". | Filled from `legal_deltas`. |

### A11 — the subset was not a sample, and on one corpus that was severe

`load_examples` took `rows[:limit]`. **`musique_ans_v1.0_train.jsonl` is sorted by
hop count**: its first 2,000 rows are *all* 2-hop, against ~72% / 22% / 5% for
the file as a whole.

Decision 2 pins the subset as "stratified by hop / decomposition depth" for
exactly this reason, and the adapters ignored it. The cost is not cosmetic:
`DATASET_DECISION.md` §2 adopts HoVer because "2Wiki/MuSiQue/HotpotQA are all
thin at ≥ 3 hops" and *"minimality and the closure rule only bite when proof sets
are genuinely large"*. An all-2-hop MuSiQue subset removes precisely the
questions Stage D's minimality claim needs — silently, and in a way every
downstream number would have inherited.

*Fix:* a shared seeded `stratified_sample()`, proportional to the corpus mix,
shuffled within each stratum so the draw is not itself a head slice. Measured
after: MuSiQue **104 / 63 / 33** across 2/3/4-hop at limit 200, 2Wiki
proportional across its four types.

### The four refuted, kept because a refutation is a finding

One claimed the per-channel presence flag contradicted a frozen block-level
contract (it did not — the block-level flag was a loose comment, not the pin);
one misread `GATE3_RULE["arms"]` as asserting all five training-free arms consume
the head; one claimed `hop_subject` returns a whole question when no `>>` is
present (it returns the stripped head, which is correct); one claimed a
support-eligibility check binds vacuously. Each was traced and rejected with a
reason, which is the outcome the refutation stage exists to produce as much as a
confirmation is.

### What moved as a result

The feature vector changed twice (per-channel flags, and the raw/normalised
correction), so **the stage-D fingerprint moved twice** — which is what a
fingerprint binding feature *names and order* is for. `ATOM_WIDTH` went 534 →
540 and dims 542/541 → 548/547. `N_real` was re-derived twice and remains
**200,000**; the slowest arm moved from `l7_checker_led` to `l7b_aux`, which is
what should happen once L7b's genuine Δd cost is paid.


---

## 4. Departures from the plan, and one shared-code edit

1. **Decision 9's `h_filter`** — §1.1.
2. **Document granularity** — §2.1.
3. **`trainer.py` gained two hooks** (`_dims`, `_build_featurizers`). Everything
   else in `Trainer.__init__` is generic over the environment; only those two
   lines were not. They are methods so `RealTrainer` can substitute a real
   featurizer **without duplicating head construction** — which would put two
   implementations of capacity matching in separate files, and capacity matching
   is a ruled decision Gate 2 depends on. Behaviour-preserving; the suite is
   what verifies that, not this sentence. **No file in
   `graft/setgen/learners/` was touched** — asserted by `git diff` in
   `test_setgen_real.py`, which is fix F6's payoff made checkable.
4. **`atomfeat.py`, `realenv.py`, `distill.py` and `portfolio.py` are held to
   the learner containment standard** rather than exempted as adapters. They are
   the real-data siblings of `features.py` and `rollout.py`, so exempting them
   was the natural reading and the wrong one: that they can never reach the
   synthetic environment is precisely what Phase 9 rests on. Only `proofs.py` is
   exempt, because it constructs the real `AtomPool`.

---

## 5. What Stage A's numbers will and will not be

Recorded now, before any exist, because the honesty stamp is easier to write
before there is a result to be pleased with.

* Stage-A numbers are **real** — the corpora are the declared training interface
  and the decision rule is `pins.GATE3_RULE`, frozen before a single rollout
  existed. They are **not** conversational-memory numbers.
* **Phase 9 is not the C3 verdict.** L7/L7b vs capacity-matched L6 and GAFlowNet
  is reported here as *supporting evidence*; the confirmatory test remains
  Gate 2 on the enumerable environment, where exact TV exists. `exact_tv()`
  raises here rather than returning an approximation, because an estimate
  printed in the column Gate 2's exact numbers occupy would invite exactly the
  comparison that is invalid.
* **`H` binds five of its nine sub-checks** on a Wikipedia pool. Size, closure,
  identity, support-eligibility and scope are real — closure genuinely binds,
  because `about_entity` edge atoms reference both endpoints and are unaddable
  until both are selected, which a test asserts. Temporal, binding and retired
  are vacuous by construction and declared in `pins.VACUOUS_ON_WIKIPEDIA`. With
  sufficiency and coverage living in `U` rather than `H`, reward discrimination
  on this track comes almost entirely from `U` — the intended design, but it
  makes "H-valid" here a weaker statement than on the conversational track.

---

## 6. Status against the build order

| Step | State |
|---|---|
| 0 — pins + stage-D fingerprint | **done**; sixth fingerprint in `verify_handoff.py`, reports Stage-D training BLOCKED |
| 1 — proofs boundary | **done**; import-graph test green |
| 2 — wiki2 + musique_ans adapters | **done**; both SHA-pinned, licences verified at download, wiring reports written, subsets stratified (§3a A11) |
| 3 — featurizer + environment | **done**; 542/541 dims, all nine arms take a gradient step |
| 4 — throughput + `N_real` | **done, and re-run after the §7 audit**; `N_real = 200,000` (unchanged), slowest arm `l7b_aux` at **30.06 traj/s** (min of 3 timed repeats, stratified sample), 92% of ceiling |
| 5 — distillation + portfolio | **done**; head ≤ cap with P6.11 guards, portfolio spends 0 checks, fallback fires |
| 6 — Stage-A runner | **BLOCKED on β** (Phase-3 step 6). `pins.training_blocked_reason()` refuses it by name. |
| 7 — HoVer (gated) | not started; decision 1 gates it behind the Tier-1 pair running end to end |

**Tests: 1,185 passing** (four added by the §7 audit — two capacity regressions,
two contested counterexamples — plus one renamed in place, the
`match_capacity` default guard being folded into the capacity pair). Nothing
committed.

---

## 7. The 16 August 2026 audit — three blockers, and one refutation that mattered

Two sources, the standing convention: an external review of the finished
steps 0–5, and an eight-agent verification pass that reproduced every claim by
**execution** rather than reading and swept the rest of the phase fresh. All
five external claims **confirmed** — two of them worse than stated — plus
several findings neither source had. Everything below is fixed and
regression-tested; the step-4 artefact was re-measured after the fixes.

### 7.1 Blocker — the capacity match used the retired tolerance, and its fallback inverted decision 11

`scripts/phase9_measure.py` called `policy.match_capacity` with its **default
`tol=0.01`** — the very tolerance §6.4 retired as *unachievable by width* (the
dead `Δd` block is ~half a width step at every scale, so the narrowest closing
width always overshoots by ~1.4–2.2%). The raise therefore fired on the
**correct** width, and the `except` handler recorded hidden 64. Measured:

| arm | live capacity | vs L7's 220,932 |
|---|---|---|
| `l7_checker_led` (target) | 220,932 | — |
| `l6_led` **as shipped** | **220,164** | **−768, i.e. SMALLER** |
| `l6_led` at its correct width 65 | 224,319 | +1.53% |

A control with *less* trainable capacity than the proposed method is the one
thing decision 11's directional clause exists to forbid, and the error ran in
the direction that flatters L7 — the third time this project has caught a
control defect pointing that way (`PHASE3_DECISIONS.md` §6). The artefact even
recorded the exception text and continued.

**Fixed** by passing `tol=gate2.CAPACITY_SANITY_CEILING` (0.05 — the pathology
guard, deliberately far above the achievable overshoot "so that it never becomes
a number anyone tunes") and by *verifying both clauses per arm* rather than
assuming them: every row now carries `control_never_smaller` and
`narrowest_admissible`, and a directional violation **raises** instead of being
reported. Post-fix, all nine arms pass both.

**A deeper cause, found by the sweep and not by either claim:** the retired 1%
survived as an **executable default**, and `check_plan_consistency.py` reads
prose — it cannot see a float. `match_capacity`'s default is now `None`
(directional clause only); a ceiling is something a caller opts into.

**Reusing `gate2.capacity_matched_arm` was considered and rejected**: it
hardcodes `SyntheticFeaturizer.dims` and takes no dims parameter, so calling it
here would have matched Phase 9's capacity at *Phase 3's* dimensions. The
duplication is two lines; the alternative is a wrong number.

### 7.2 Blocker — Symmetry-Aware GFlowNets was cited for a phenomenon it does not cover

Recorded in full as **§1.4**, which amends signed decision 12. In short: the
pinned collision statistic is structurally zero (a proof, not a measurement);
the quantity substituted for it — distinct atoms with identical text — is a
**dataset property**, not an MDP symmetry; and Cor 5.1 corrects graph
automorphisms under node-by-node generation, which an id-defined set-state MDP
does not have. Both the tempting fix (apply a scalar) and the tempting excuse
(it would be a harmless no-op) are measured false: the duplicated class carries
exactly **2.0×** the reward mass, so the correction is neither justified nor
inert. No scalar is applied, the statistic stays as a diagnostic, and the
citation is withdrawn from it.

### 7.3 Blocker — the stage-D fingerprint could not converge, because it bound its own outputs

The reported symptom was drift: `pins.stage_d_fingerprint()` read `d6f7fe84…`
against the artefact's `2f40c3f…`. The **cause** was worse than staleness.
`frozen_values()` bound the whole `BUDGET` dict — including
`measured_rate_slowest`, `ceiling_utilisation` and `measured_on`, all *outputs
of a run* — inside a digest whose own docstring defines it as *"the config, not
the output"*. That made the identity **circular**: step 4 measures a rate → the
rate is written into pins → the fingerprint moves → the artefact step 4 just
wrote now disagrees with the code that wrote it. Re-running could never fix it.

**Fixed** by binding only the budget's identity half (`ceiling_s`, `ladder`,
`n_real`, `identical_across_arms`, `early_stop_arms`, `sized_from`). `n_real`
stays bound deliberately: a run at 50k and one at 200k are different
experiments. Verified — updating the measured rate now leaves the fingerprint
unchanged, and artefact and pins agree.

Four further pin-vs-code contradictions were confirmed and fixed, all inside the
fingerprint and therefore all certifying a design that was never built:

| pin | said | code did |
|---|---|---|
| `PORTFOLIO["filter"]` | `h_filter after dedup` | nothing — valid by construction (§1.1) |
| `COLLISION_INSTRUMENT["measure"]` | content-key duplicates | duplicate normalised text (§1.4) |
| `CORPORA["wiki2"]["gold"]` | supporting **sentences** | **paragraphs** (§2.1, decided by measurement) |
| `wiki2.py` docstring | "recorded in `pins.GRANULARITY`" | no such symbol exists |

### 7.4 The `N_real` workload — confirmed, and its stated direction refuted

`build_examples` took `load_split(...)[:half]` *before* the adapter could
stratify, defeating `corpora.stratified_sample` — which both adapters already
call, and which exists for exactly this. Measured on the dev splits, the head
slice of MuSiQue-Ans is **100% 2-hop** against a true 51.8 / 31.4 / 16.7 mix.
A11 recorded this failure for `train`; `dev` is sorted the same way.

**But the claimed mechanism is wrong, and the measurement says so.** The
intuition — an all-2-hop slice is cheap, so the rate was flattered — predicts
that fixing it would *lower* the rate. Measured over 5 repeats on `l7b_aux`:
head-slice **28.69** traj/s against stratified **33.04**. The wrong workload was
*slower*, so the bias ran in the **safe** direction and `N_real` was never at
risk from it. Recorded rather than dropped, because a plausible mechanism that
measurement contradicts is exactly what `CLAUDE.md` §5 asks to be written down.

**A second defect surfaced while checking it:** a single wall-clock sample is
noise. Successive single-run readings of the same arm gave 29.67 / 34.68 /
35.58 / 36.16 traj/s — a 22% spread, and the reason `pins.BUDGET` kept
disagreeing with the artefact it cites. The script now times each arm **three
times and takes the slowest**, the same conservatism the "slowest arm" rule
applies one level up: a re-run cannot make the budget optimistic.

**Re-measured, stratified, min-of-3: `l7b_aux` at 30.06 traj/s, `N_real` =
200,000, 92% of the ceiling.** Inside the noise of the original 29.67 — so the
derivation was robust to a defect genuinely present in it, which is a better
outcome than either the claim or the defence predicted.

Two smaller parts of the same claim: exit criterion 10a's *"stratification
counts reported"* was unmet — the adapters' wiring report was computed and
discarded — and is now in the artefact (`examples.strata`, showing MuSiQue at
7 / 3 / 2 across hops where the head slice was 12 / 0 / 0). And the stub
embedder was **refuted** as a concern for this measurement specifically:
embedding is one-time pool prep outside the timed region, pool sizes and dims
are identical either way, and the measured rate difference (3.5%) sits inside
the noise band.

### 7.5 The contested diagnostic measured neither of the things it claimed

`contested_rate` compared **atom-id sets** while its own docstring said it
compared *"the answer they resolve to rather than by atom identity"*. Both
counterexamples reproduce:

* two paragraphs carrying the same gold answer → `contested: True` — a **false
  positive**, and agreement-by-different-evidence is what a portfolio is *for*;
* one set carrying the answer and one carrying nothing → `contested: False` — a
  **false negative**, because unbound sets were dropped before comparing.

The deeper point is that fix F4's flag is **not computable here at all**: every
alias is an alias of the one gold answer, so every match resolves to the same
answer by construction. A rename alone could not fix the false negative and a
recompute alone could not make the name honest, so the fix does the minimum of
both. `answer_agreement` now reports what is measurable — whether the top sets
agree on *carrying* the answer (`answer_presence_disagreement`) and how many
distinct evidence groups carry it (`distinct_evidence`) — and states in its own
`note` that fix F4's contested flag is Phase 10's by name (the G7 transfer
pattern). Both counterexamples are now regression tests.

### 7.6 Smaller confirmed items

| item | resolution |
|---|---|
| `ceiling_utilisation` recorded 0.93 where the arithmetic gave 0.9364 | re-derived from the new measurement (0.92) |
| `CLAUDE.md`'s Phase-9 row carried pre-Δd-fix numbers (`l7_checker_led`, 41.83 traj/s, 13.99× spread) and said the G11 correction "is not applied" without saying it is structurally zero | both corrected |
| dead assignment `collisions = duplicate_text` | deleted |
| `phase9_measure.py` imported `capacity`, never used | dropped |
| a relative `--out` crashed the final print *after* the artefact was written — the identical defect Phase 7's runner had | `.resolve()` on both halves of the guard |
| nothing tested the Phase-9 capacity match, which is why the directional violation shipped | two regression tests: every arm ≥ L7 at real dims, and `match_capacity`'s default is no longer the retired number |

### 7.7 A citation defect found while auditing Phase 9's — and it was not Phase 9's

`realenv.py` cites Graph-S3's Table 3 as **+11.8 / +17.1**; `GATE0_CONTRACT.md`
cited the *same table* as **+8.1 / +9.7**. A page-level read of the PDF settles
it: **Phase 9 is correct and the signed contract was wrong.** +8.1 / +9.7 is the
**abstract's** figure against **seven baselines** (printed p. 25510); Table 3
(printed p. 25517) is the stepwise-vs-outcome-reward ablation, and +11.8 / +17.1
is its macro mean across the five dataset columns. Both pairs are true of
different comparisons; only the contract's attribution was wrong, and it is
fixed.

**Two refinements the read forced, applied at all five sites.** The 11.8 / 17.1
pair **appears nowhere in the paper** — it is computed from Table 3 by this
project — so it is now labelled *"macro-averaged from Table 3 by this project"*
rather than quoted as the paper's own. And Graph-S3's stage II is **GRPO**, not
supervised fine-tuning; `CLAUDE.md` §5's "supervised stepwise supervision"
understates it, the distinguishing feature being that the reward is
process-level and rule-based rather than outcome-based.

### 7.8 What the fresh sweep found that neither claim listed — including two blockers

The five reviewed claims were the starting point, not the finding. A
module-by-module sweep of the same code turned up two further blockers and four
HIGH defects, all reproduced by execution.

**Blocker — `Δd` is structurally correct and semantically near-empty on this
track, and it lands on Contribution 3.** §3a's A1 fixed `Δd`'s *sign* and
*density* at gradient time. What nobody measured afterwards is how much `Δd`
has to say on Wikipedia pools. Measured on real dev pools (8 examples/corpus,
uniform-policy walks, after the anchor fix below):

| corpus | legal ADDs with Δd ≠ 0 | states with no informative ADD |
|---|---|---|
| 2Wiki | **5.04%** | 7.2% |
| MuSiQue-Ans | **1.73%** | **39.5%** |

and **exactly one of the six deficit components ever varies** — `anchor`.
`value`, `time`, `source`, `binding` and `closure` are constant on every
reachable state, because a Wikipedia question synthesises no value type, no time
constraint and no source requirement, and Wikipedia pools carry no binding
atoms. L7's mechanism is *"prefer the ADD that discharges more obligation"*;
where nothing is discharged, L7 and L6 see the same state.

**Declared, not repaired.** Enriching the synthesised obligations until `Δd`
moved would be reward engineering **in the proposed method's favour** —
`CLAUDE.md` §5's catalogued failure, and the third time this audit found an
error pointing that way. The measurement is now
`pins.OBLIGATION_SIGNAL_ON_WIKIPEDIA`, inside the fingerprint and in
`ADAPTATION_LOSSES`, and it carries the reading a null licenses, **written
before the run**: a Stage-A "L7 ≈ L6" is consistent with *both* "checker
conditioning does not help" *and* "this corpus offered almost nothing to
condition on", and Stage A cannot separate them. That is the same shape as
`PHASE3_DECISIONS.md` §2.3's budget warning, and it is a second independent
reason `STAGE_SPLIT` already says C3's verdict is Gate 2's and not this phase's.

**Blocker — MuSiQue's `entity_anchor` was unsatisfiable on 70% of questions.**
`hop_subject`'s docstring promises to avoid emitting *"a slot that `coverage`
counts as active and no atom can ever satisfy"* — but it guarded only the `#k`
placeholder case. It emitted the raw subject **string**, while
`resolve.matches_anchor` compares for equality against an `Entity` node's name,
and those names are paragraph **titles**. Measured on the pinned dev split: the
raw subject equals a title on **30.3%** of rows, so **69.7% carried an anchor no
subset of the pool could satisfy** — inflating those questions' `coverage`
denominator and depressing their maximum achievable `U`, worst at high hop
counts (66.3% at 2-hop rising to 72.5% at 4-hop), i.e. hardest on exactly the
questions this corpus is here to supply. **Fixed** by resolving the subject to a
title (containment, longest-first) and emitting `None` when it resolves to none:
unsatisfiable anchors go to **0%**, with 28.5% honestly reporting no anchor.
Recorded as the **second amendment to signed decision 5**, the first being
§1.3's 2Wiki anchor — the same defect from the other direction.

**HIGH — `spearman` returned 1.0 for a diverged head.** `NaN` compares unequal
to itself, so the tie-detection never fired, every `NaN` took a distinct rank in
`argsort`'s stable order, and the result correlated perfectly with any monotone
target. Probed: `spearman(full(6, nan), arange(6))` → **1.0**. ρ is the number
that licenses reading a Gate-3 row at all (`DISTILL["report_rho"]`: *"a row
without its ρ cannot be read"*), so **the one instrument that detects a broken
scorer was guaranteed to look healthiest when it broke**. Two-line finite guard.

**HIGH — `train_head` returned the diverged weights as "selected".** With every
dev loss non-finite, `best_state` stays `None`, the restore is skipped, and the
function returns the diverged head with a plausible `dev_spearman`. Probed at
`lr=1e6`: `best_dev_loss=inf`, `best_epoch=-1`, non-finite parameters, returned
normally. That is P6.11's **third** guard — the one this function never
inherited — and it now refuses.

**HIGH — a mismatched featurizer silently flattened best-of-K.** `pool_sets`
skipped atoms it could not resolve, so a `HeadScorer` built on example A and
applied to example B scored every candidate identically, collapsing the ranking
to the tie-break alone (size ascending, then hash) with nothing raising.
`run_portfolio` takes `feat` and `scorer` as independent arguments and cannot
check they agree, so the check belongs in `pool_sets`: an empty set still pools
to zeros, a **non-empty set that resolves to nothing** now raises.

**MEDIUM — `audit_validity` was not an independent check.** Its docstring says
*"'Valid by construction' is a claim about the masks, and this is how it gets
tested rather than trusted"* — but it called `IncrementalChecker.ok()`, and
`masks.stop_allowed` **is** `state.ok()`. It re-asked the very question the
claim is made of, so it could not have disagreed. Now calls the batch
`checker.H`, an independent implementation of the same specification.

Three further mediums are recorded and **not** acted on, because each is a
convention Stage D's runner should set rather than this module: `run_portfolio`
declares an unused `ledger` parameter and a constant-`False` `budget_exhausted`
field; an abstained query emits `best_score = NaN`, which is loud rather than
silent under `np.mean` but has no defined FAIL utility yet; and the Gate-3
aggregation convention for abstentions is Phase 10's to fix alongside its
reader. They are listed here so the runner's author inherits them as decisions
rather than discovering them as bugs.

### 7.9 One process defect, and it is the reason several of these survived

`scripts/check_plan_consistency.py` was not reading three live documents —
`GRAFT_PHASE9_BUILD.md`, `PHASE9_DECISIONS.md` and `PHASE8_DECISIONS.md`. The
Phase-9 pair matters most, because this file **wins conflicts** with its own
build plan, so an unretired wording here outranks the corrected one everywhere
else. All three are registered (28 live documents), and **R23** retires the
seven wordings this audit corrected.

The deeper version of the same gap is worth stating plainly, because the guard
cannot close it: **the retired 1% tolerance survived as an executable default.**
`check_plan_consistency.py` reads prose and cannot see a float, so a retired
*number* is invisible to it in a way a retired *sentence* is not. That is why
§7.1's blocker shipped, and it is why the fix is paired with a test that reads
the default itself rather than with a retirement alone.
