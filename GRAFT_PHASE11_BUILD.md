# GRAFT — Phase 11 Build Plan: Reference-table evaluation on LoCoMo

**The phase where GRAFT gets a number that means something to somebody else.**
Phase 10 made the read path answer; this phase scores those answers in the units
a published table uses, puts GRAFT's cost on an axis a supervisor can read, and
does both without running a single baseline.

Date planned: 19 August 2026
Parent: `GRAFT_RESEARCH_PLAN_v1.md` v1.2 **§6.4** (metric groups, reporting
discipline), **§5.3** (system baselines), **§7 Gate 4** ·
`GRAFT_EXECUTION_ARCHITECTURE_v1.md` **Phase 11** (§11) ·
`GRAFT_PHASE10_BUILD.md` **§8** (what Phase 11 inherits) ·
`PHASE10_DECISIONS.md` **§5 A1** (the missing cost numerator) and **decision 7**
(abstentions excluded from means) · `DATASET_DECISION.md` **§1.2** (LoCoMo's 446
adversarial questions) · `CLAUDE.md` **§9** (what is safe to claim)
Reference paper: **Mem-T: Densifying Rewards for Long-Horizon Memory Agents**,
Yue et al., arXiv 2601.23014v2, 9 March 2026 (PKU + NTU) — **Provisional venue**
per §3 tiering; it supplies a *reference table*, and under the tiering rule it
may motivate a comparison but never carry a central claim alone.
Effort: **~1.5 days solo [ANALYSIS]**, of which **Stages A–C need no GPU and are
unblocked today**.
Status: **Stages A, B and C BUILT and green (19 Aug 2026); §6 UNSIGNED; nothing
run on GPU.** Plus three things this phase turned out to need that no earlier
phase had built: the LoCoMo loader, the end-to-end runner Phase 10 deferred, and
the distilled-head trainer. Suite 1,304 passed, 0 failed.

Labels as everywhere: **[EVIDENCE]** = a named paper supports this, venue
stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = engineering
or mathematical judgment made here.

Gaps are numbered **G1–G8**, matching the Phase-0…10 convention.

---

## 0. What blocks this, and what compute it needs

### 0.1 The dependency answer, checked rather than assumed

**Nothing blocks the code.** Every dependency named in `GRAFT_PHASE10_BUILD.md`
§8 is built: `System.run(conversation, question) → OutputRecord`, the frozen
prompt and decoding SHA, the per-query ledger, the five ceilings, the stage-E
fingerprint. `token_f1` already exists at `graft/reader/parse.py:125`. The meter
vocabulary already exists at `graft/ledger.py:41` and is exactly the right one —
`llm_calls`, `llm_tokens_in`, `llm_tokens_out`, `model_forwards`,
`wall_clock_ms`.

What is blocked is only the **LoCoMo numbers**, which need ingestion to have run.
Stages A–C are testable against Phase 10's existing R3 fixtures and do not wait
for it.

### 0.2 Compute

| Stage | Where | Cost |
|---|---|---|
| A — metering | CPU | ~3–4 h build, no run |
| B — metrics + reference table | CPU | ~3 h build, no run |
| C — report assembly | CPU | ~2 h build, seconds to run |
| D — the numbers | GPU | **no additional GPU** — consumes the reader pass already scheduled |

Stages A–C run alongside GPU ingestion for free. That is the whole reason this
phase is worth starting before the corpus finishes.

### 0.3 What Phase 11 is not

**It is not Gate 4.** See G1. The phrase "Gate 4" must not appear in this
phase's artefact except to name the item it fails to meet.

---

## 1. Eight gaps this phase must close

### G1 — This is not Gate 4, and the plan must say so in its own title [ANALYSIS]

Plan §7's Gate 4 has four items. Item 4 is: *"Re-run system baselines rather
than quoting published numbers (§5.3)."* This phase quotes published numbers, at
the project owner's explicit instruction, because the three-day deadline does not
fit ~8–15 h of additional baseline GPU time.

So Gate 4 is **not closed**, and the honest name for what this phase produces is
a **reference-table comparison**: GRAFT measured, reported beside numbers
measured by someone else, on a different backbone, with the differences declared.

The cost of pretending otherwise is precisely `CLAUDE.md` §5's catalogued
pattern — *overreaching on what a source establishes* — committed inside the
project's own results section, which is where it is least defensible and most
likely to be caught.

**The close:** the artefact's header states, in its own words, that Gate 4 item 4
is unmet and what meeting it would cost.

### G2 — The cost axis is metered but ungoverned, and its failure mode is silent [ANALYSIS]

**Corrected 19 August 2026.** An earlier draft of this gap asserted that no meter
is spent anywhere in the read path. That is `PHASE10_DECISIONS.md` §5 A1's
*finding*, not the current state: §5 is the audit record and lists findings
**with their fixes**. A1 was fixed on 16 August. Reading a fix record as a
defect record is the mirror image of `CLAUDE.md` §5's overreach pattern, and it
is recorded here rather than quietly deleted.

**What is actually built.** `Reader.generate` spends `llm_calls`,
`model_forwards`, `llm_tokens_in` and `llm_tokens_out`
(`graft/reader/read.py:202`); `Ledger.stage()` records `wall_clock_ms`; the
orchestrator opens a stage for Stage D, Stage E and the contested comparison;
the snapshot reaches `OutputRecord.ledger_snapshot` and `report()`; and
`scripts/phase10_read.py` opens a `query_scope` per question. Run R3's artefact
carries real values on 8 of its 10 records.

**What was missing.** Three things, all governance rather than plumbing:

1. **No regression guard, and the failure is silent by construction.** A1's
   damage was that zero-initialised meters make an *absent* measurement look
   like a *cheap* query. Nothing asserted against its return.
2. **The obvious guard is wrong.** Two of R3's ten records are legitimately
   all-zero: the gate route returns before Stage D and before any reader call,
   so nothing is spent and nothing should be. A guard reading "no all-zero
   snapshots" fails on correct behaviour, gets relaxed, and takes the real check
   with it. The rule has to be narrower — all-zero is permitted **only** where
   the gate declined.
3. **No per-query cost aggregate in G7's unit.** The meters were spent and then
   never summarised into tokens-and-calls per query, which is the form the
   comparison needs.

**The close:** `cost_report()` in `reader/orchestrator.py`, raising
`UnmeteredError` when a record had a wired ledger and spent nothing outside the
gate route; `aggregate()` carries a `cost` block so no summary can reach an
artefact without the axis; and the runner refuses an artefact containing any
*unledgered* query — an empty snapshot and an all-zero one being different
states, whose conflation is how A1 stayed invisible.

**Measured on R3** (10 synthetic fixtures, `is_wiring_test = True`, so an
instrument demonstration and not a result): 2 LLM calls per query, 513.2 tokens
in, 13.5 out, **526.8 total**, 55.6 model forwards, 1,054 ms wall clock.

### G3 — The metric set does not match the reference table [ANALYSIS]

Mem-T Table 2 reports **token-F1 and BLEU-1**, split four ways: single-hop,
multi-hop, temporal, open-domain, plus overall. GRAFT has `token_f1` and nothing
else — no BLEU-1, no category split.

**The close:** add `bleu1()` beside the existing `token_f1`; add the four-way
LoCoMo category mapping; report overall as the reference table does.

### G4 — Abstention silently inflates F1, and this is the easiest number in the phase to get wrong [ANALYSIS]

Phase-10 decision 7 excludes abstentions from means and counts them separately —
correct for GRAFT's own diagnostics, because an abstention is not a wrong answer.
But the reference table's F1 is computed over **every** question, because none of
those systems abstain.

So GRAFT's F1-over-answered is **a different statistic** from Mem-T's 49.38 and
must never appear in the same column. A system that abstains on 83% of questions
and is right on the rest would post a spectacular F1-over-answered and have
answered almost nothing.

This is the same class of error as `PHASE8_DECISIONS.md` §3.3 — a quantity that
is correct for its own purpose becoming wrong when a different consumer reads it.

**The close:** emit both. `f1_over_all` (abstention scored 0) is the comparable
column and is the one that may sit beside a reference number. `f1_over_answered`
travels with `coverage` and is labelled a GRAFT-internal diagnostic.

### G5 — Which reference row is the honest one must be pinned before GRAFT's number exists [ANALYSIS]

Mem-T §4.1: *"we use the same training data configuration by splitting the LoCoMo
dataset into a 1:1:8 train/validation/test split"*, following Memory-R1 (ACL
2026, in the library at `Research papers/INDEX.md:44`). Their 58.65 overall F1 is
therefore an **in-domain** number.

Their own table prices what that buys: **w/o training 49.38 → MoT-GRPO 58.65**,
a gain of **+9.27 F1 from training on LoCoMo**.

GRAFT is zero-shot on LoCoMo by declared design — Stage D trains on 2Wiki and
MuSiQue-Ans, Stage B on LongMemEval pilot labels, the gate on MuSiQue-Full, and
the reader is frozen and trains on nothing. The comparable row is therefore the
**untrained** one: **49.38 F1 / 44.11 BLEU-1**.

Choosing that row after seeing GRAFT's own number would be the
`GRAFT_PHASE2_BUILD.md` §6b failure. It is pinned here, before any code.

### G6 — Non-comparability must be machine-readable data, not prose [ANALYSIS]

Four declared differences, none of them small:

| Axis | GRAFT | Mem-T |
|---|---|---|
| Reader backbone | Qwen2.5-3B-Instruct, bf16, pinned by revision | Qwen3-4B |
| Embedder | bge-small-en-v1.5, pinned by revision | BGE-M3 |
| LoCoMo exposure | zero-shot | in-domain, 1:1:8 split |
| Question subset | all categories incl. adversarial | adversarial discarded (§A.1) |

The backbone row is the one that decides the comparison, and **the paper itself
supplies the evidence**: GAM loses **23.31 F1** on nothing but a backbone change
from gpt-4o-mini to Qwen3-4B (Table 2, and §4.2 draws attention to it). Backbone
sensitivity on this benchmark is larger than the entire gap between the best and
worst memory system in the table.

In prose these caveats get lost between draft and viva.

**The close:** they are a structured block attached to every reference row, and
`report.py` **raises** rather than emitting a comparison table without them.

### G7 — The cost axis needs one defined unit, and two separate axes [ANALYSIS]

Mem-T's cost figure is **~9,000 tokens per query** at its chosen 6 retrieval
steps, rising to ~21,000 at 10 steps (§4.3, sensitivity analysis), and its
Pareto claim against GAM is a 19.94–24.45% per-query reduction (Figures 3–4).

GRAFT's `serialization_budget_tokens = 512`, on a ladder of 160/512/1024, is
**evidence tokens packed into the prompt** — not the same unit as total inference
tokens, and quoting one against the other would be an apples-to-oranges cost
claim of exactly the kind §5 warns about.

Two further hazards. First, GRAFT makes **one** reader call per query with no
agentic loop, while Mem-T makes up to six plus its construction calls — and
**call count is backbone-independent in a way token count is not**, which makes
it the more robust headline. Second, GRAFT's ingestion cost (220 output tokens
per turn, measured) is an *offline* cost; folding it into a per-query inference
figure would flatter nobody and confuse everybody.

**The close:** the comparison unit is **total LLM tokens per query** (prompt +
output, summed over all calls) and **LLM calls per query**, both read from the
ledger. Ingestion is a separate, separately-labelled axis. GRAFT is reported at
all three ladder rungs, which is Phase 10 exit criterion 9 reused and yields this
project's own version of the paper's Pareto plot.

### G8 — Three LoCoMo metric conventions are in circulation and mixing them is silent [ANALYSIS]

The library and this paper between them use at least three: **token-F1 / BLEU-1**
(Mem-T, Mem0, A-Mem), **LLM-judge** (HyperMem 92.73; Mem0 72.90 — the number
`CLAUDE.md` §9 quotes for full-context), and plain accuracy. A 58.65 and a 92.73
are not the same axis, and nothing in a results table announces that.

**The close:** F1/BLEU-1 is pinned as this phase's convention. Every reference
row carries its metric name, and rows whose metric differs from the pinned one
are stored but **refused** by the comparison assembler.

---

## 2. Scope

**In.** Read-path metering; BLEU-1; LoCoMo category mapping; the reference table
as frozen data; the comparison assembler and its refusals; the cost/Pareto
table; the runner and its artefact.

**Out.** Every baseline adapter — see §7.

---

## 3. Modules

### P11.0 `reader/orchestrator.py`, `reader/read.py` — metering (G2) · *edit, not new*

Spend the five meters per query; snapshot into `OutputRecord.ledger_snapshot`.
Follow `graft/ingest/extractor.py`'s existing spending pattern rather than
inventing one. `wall_clock_ms` is measured around the reader call and around the
whole query separately, so retrieval and generation are distinguishable.

### P11.1 `reader/parse.py` — `bleu1()` (G3) · *edit, not new*

BLEU-1 beside the existing `token_f1`, sharing its SQuAD normalisation — the one
`PHASE10_DECISIONS.md` §1.1 fixed for the citation-marker defect. Reusing that
path is deliberate: a second normalisation would re-open the same bug.

### P11.2 `baselines/reference.py` — the published table as data (G1, G5, G6, G8)

Mem-T Table 2, frozen: value, metric name, source, arXiv id and version, table
number, backbone, embedder, training exposure, question subset. Plus the
non-comparability block of G6 and the G5 pin naming the untrained row as
comparable. No code executes a baseline; this module is data and its guards.

### P11.3 `baselines/categories.py` — LoCoMo categories (G3, G4)

Map LoCoMo questions to single-hop / multi-hop / temporal / open-domain.
Adversarial routes to its own bucket and is **structurally unable** to enter the
four-way split — the Phase-9 `test_setgen_real.py` AST-guard pattern, not a
substring check.

### P11.4 `diagnostics/report.py` — the assembler (G1, G4, G6, G7)

Builds GRAFT's row, the cost table, and the ceiling decomposition. Raises when
the non-comparability block is absent, when a reference row's metric differs
from the pinned convention, or when `f1_over_answered` is placed in a
reference-comparable column.

### P11.5 `scripts/phase11_report.py` — runner and artefact

Emits `artefacts/phase11_report.json` plus a markdown table. Carries the honesty
stamp naming every untrained or placeholder component the run used, per Phase 10
exit criterion 16.

---

## 4. Build order — four stages

### Stage A — metering · **unblocked, no GPU, start here**

G2 only. It is first because it answers the supervisor's question, is
independent of ingestion, and gates the value of every later GPU hour.

### Stage B — metrics and the reference table · **no GPU**

G3, G5, G6, G8. BLEU-1, categories, the table as data. Pure entry and guards; no
runs. Stage B's pins must land **before** Stage D produces a GRAFT number.

### Stage C — the assembler · **no GPU**

G1, G4, G7. Testable against Phase 10's R3 fixtures, so the whole reporting path
is green before LoCoMo exists.

### Stage D — the numbers · **needs ingestion; no additional GPU**

Consumes the reader pass already scheduled. Produces the artefact.

---

## 5. Exit criteria

**The cost axis has a numerator** — *criteria 1–2 and 4 met 19 Aug 2026 (Stage A)*
1. ✅ A record produced under a wired ledger that spent nothing is **refused**,
   unless the gate declined — the carve-out asserted in its own test, so the
   guard cannot be relaxed by a correct all-zero record.
2. ✅ An empty snapshot and an all-zero snapshot are distinguished and separately
   counted; the runner refuses an artefact carrying any unledgered query.
3. ✅ Stage C retrieval cost reaches the same per-query snapshot as generation,
   as its own `stage_c` ledger stage, with the dense channel's **question**
   encode counted and its **corpus** encode deliberately not — the latter is
   index construction amortised over every query, and charging it per query
   would make the first question look many times the cost of the second.
   Reported separately from Stage D and Stage E, which were already separate.
4. ✅ Ingestion cost is a separate axis from inference cost, asserted by test.

**The metrics match the reference table** — *5–7 met 19 Aug 2026; 8 needs runs*
5. ✅ `bleu1` and `token_f1` reproduce hand-worked fixtures, and share one
   normalisation so `PHASE10_DECISIONS.md` §1.1 cannot re-open in a second
   metric.
6. ✅ Adversarial is structurally excluded — `four_way_split` has no branch that
   would include it, and the subset is reached by its own function. Unknown
   category codes are **refused**, not bucketed.
7. ✅ `f1_over_all` and `f1_over_answered` both emitted, with `coverage`.
   `graft_row` has **no field** that could hold the answered-only view, and
   `comparison_table` checks the view tag as well — belt and braces, because G4
   is the easiest number in the phase to get wrong.
8. ⬜ GRAFT reported at all three budget rungs. **Code complete, needs three
   runs** — `locomo_eval.py --budget` takes one rung per run by construction, so
   this is the only criterion here that a build cannot close. ~35 min of reader
   time per rung per 1,000 questions.

**The comparison is honest** — *9–13 all met 19 Aug 2026 (Stage C)*
9. ✅ The comparable row is the **untrained** one (49.38 / 44.11), pinned and
   SHA'd in `reference.py` before GRAFT has a number. The digest covers the pin
   and the metric, not the whole table, so adding a context row leaves it alone
   while editing the judged row moves it.
10. ✅ Metric-convention and exposure mismatches are refused by
    `comparable_rows`, **and** `report.py` raises without the non-comparability
    block — both halves asserted.
11. ✅ Every reference number carries arXiv id, version and table number.
12. ✅ The artefact's **first key** is `what_this_is_not` = `GATE4_STATUS`,
    asserted by test: a reader who stops after one line already knows this is not
    Gate 4.
13. ✅ `build_report` **raises** without an honesty stamp, so no report can be
    assembled that fails to name the untrained Stage-D sampler and the
    MuSiQue-trained gate threshold.

**A caveat carried forward.** `LOCOMO_CATEGORIES`' integer codes are the mapping
in common use but are **not verified against the dataset file** — the corpus is
not downloaded yet. `verify_against_corpus` pays that debt at load time by
checking the adversarial bucket against `DATASET_DECISION.md` §1.2's
independently recorded **446**: on a full corpus a mismatch raises, because the
count and the mapping come from different sources and agreement between them is
evidence. On a subset it reports `inconclusive` rather than failing, so the
check stays usable while the corpus is scaled up.

**Discipline**
14. Full suite green.
15. **Deferred by name:** the three baseline adapters, and the requirement that
    they reuse this phase's prompt byte-identically (inherited from
    `GRAFT_PHASE10_BUILD.md` exit criterion 15 and not discharged here).

---

## 6. Decisions to lock before writing code — **UNSIGNED**

Signing needs the project owner's explicit instruction, as with
`GATE0_CONTRACT.md`, `GRAFT_PHASE3_BUILD.md` §6, `GRAFT_PHASE9_BUILD.md` §6 and
`GRAFT_PHASE10_BUILD.md` §6. **No Phase-11 code exists**, which per
`GRAFT_PHASE2_BUILD.md` §6b is the uncontaminated moment for decisions 2 and 4
in particular — both are decision-*rules* about how a number will be judged, and
both are worthless if fixed after the number exists.

| # | Decision | Ruling | Cost to change |
|---|---|---|---|
| 1 | What this phase closes | A **reference-table comparison**, not Gate 4; the artefact names the unmet item | Reversing means running baselines: ~8–15 h GPU |
| 2 | Which reference row is comparable | Mem-T **untrained**: 49.38 F1 / 44.11 BLEU-1 | Changing after seeing GRAFT's number is a contaminated §6b amendment |
| 3 | Metric convention | **token-F1 + BLEU-1**; LLM-judge rows stored but refused | Re-scoring; cross-paper conclusions move |
| 4 | Abstention in the comparable column | Scored **0** (`f1_over_all`) | Changes every reported F1 |
| 5 | Cost unit | **Total LLM tokens/query + LLM calls/query**, from the ledger; ingestion separate | Re-derives the Pareto table |
| 6 | Reader backbone | Stays **Qwen2.5-3B**; the difference is declared, not matched | Swapping voids the stage-E fingerprint and runs R1–R3, and moves the reader out of the size regime where the packing benefit exists at all (`CLAUDE.md` §4.2) |

**What this signature would not cover:** the LoCoMo question counts and category
totals (measured at ingestion, not guessable here), and GRAFT's own numbers,
which do not exist.

---

## 7. Explicitly not in this phase

The three Phase-11 adapters named in the architecture — **full-context**,
**matched-budget RAG**, and **Mem0** — are **deferred by name** at the project
owner's instruction, on a three-day deadline.

Recording the cost of that, because an undocumented omission is the thing this
project's §5 exists to prevent:

* **Gate 4 item 4 stays unmet.** G1.
* **Full-context is the comparator `CLAUDE.md` §9 calls non-negotiable** — Mem0's
  own table has it at 72.90 LLM-judge, above every memory system it tested, and
  §9 says omitting it "will read as evasion." Deferring it is a time decision,
  not a scientific one, and the write-up should say so in those words.
* **Matched-budget RAG is the cheapest of the three** — roughly 2 h, because
  Stage C already computes the retrieval and only the reader pass is new — and is
  the single addition that would most repair G1. Mem-T's own RAG row (41.59 F1)
  shows why it matters: RAG is not a strawman on this benchmark, beating four of
  the eight training-free memory systems in Table 2.

---

## 8. What the write-up can claim, and what it cannot

Chosen so the claims survive the missing baselines. A five-point F1 difference
does not survive a backbone change of the size the paper itself documents
(23.31). A 9× cost ratio and a within-system decomposition do.

**Primary — robust without baselines**

* **Cost.** LLM calls per query: GRAFT **1**, Mem-T **≤6 plus construction**.
  Tokens per query: **~1k against ~9k**. Both from the ledger, both reported at
  three budgets.
* **The five-ceiling decomposition.** Self-contained, needs no baseline, and is
  Contribution 4. It converts a disappointing end-to-end number from
  uninterpretable into localised — which is what it did on run R3, where ceiling
  5 read 1.0 against 5/10 end-to-end and the whole gap was attributable to
  Stage D.
* **The adversarial subset.** Mem-T §A.1 discards it, as do Mem0 and A-Mem, so
  there is nothing to be non-comparable *with*. `DATASET_DECISION.md` §1.2
  anticipated this; the paper is the third confirmation.

**Secondary — reported only with the non-comparability block attached**

* F1 and BLEU-1 by category against the untrained reference row.
* Accuracy against tokens at all three rungs — this project's own Pareto curve.

**Not claimable.** That GRAFT beats Mem-T, or any system in Table 2. The
backbone differs, the LoCoMo exposure differs, the question subset differs, and
no baseline was re-run. `CLAUDE.md` §9's list of what is not safe to claim
applies unchanged.


---

## 9. The run sequence — built 19 August 2026, nothing run

Every item below is code that exists and is tested. Times are from measured
rates, not estimates: 135.67 turns/h for ingestion (`artefacts/phase5_pilot.json`)
and 2 LLM calls / ~527 tokens / ~1.05 s per query for the read path (run R3).

### Free, do these first (seconds to minutes, CPU)

| # | Command | Cost | Why first |
|---|---|---|---|
| 1 | `locomo_ingest.py probe` | seconds | **Already run: verdict OK.** 10 samples, 1,986 questions, 446 adversarial, 5,882 turns, every timestamp parsed, SHA pinned. This is what stops a structural mismatch surfacing at GPU hour 30 |
| 2 | `locomo_ingest.py plan` | seconds | **Already run.** The per-conversation stopping table |
| 3 | `train_head.py --real-embedder` | ~10–30 min CPU | The distilled utility head. Without it best-of-K ranks by noise; `PHASE10_DECISIONS.md` §1.4 is why ranking needs it at all. **The whole return on the first hour** |
| 4 | `locomo_eval.py --smoke --questions 20` | ~1 min CPU | Stub reader over the real graph. Proves the join before any reader time |

### The GPU work

| # | Command | Cost | Notes |
|---|---|---|---|
| 5 | `locomo_ingest.py ingest --extract-only` | **3.1 h per conversation**, 43.4 h for all 10 | Resumable per turn. Stop whenever; questions belong to conversations |
| 6 | `locomo_ingest.py ingest --verify-only` | ~minutes | NLI was 8.56 s across 248 turns. Deferred deliberately so extraction runs unattended |
| 6b | `locomo_stageb.py` | **CPU, ~minutes** | **Added 19 Aug 2026** (`PHASE11_DECISIONS.md` §1.8): Stage-B structural commit over a copy of the Stage-A log — Stage C retrieves over *nodes*, and only the Phase-6 stand-in constructor creates them. The eval refuses a nodeless graph by name |
| 7 | `locomo_eval.py --head artefacts/utility_head.pt` | **~35 min per 1,000 questions** | 2 calls × ~527 tokens × ~1.05 s. Resumable per question |

### Operating points, from the measured plan table

| Conversations | Turns | Ingest | Questions | Adversarial | Eval |
|---|---|---|---|---|---|
| 3 | 1,451 | 10.7 h | 497 | 112 | ~18 min |
| 5 | 2,760 | 20.3 h | 999 | 237 | ~35 min |
| **7** | **4,124** | **30.4 h** | **1,347** | **288** | **~48 min** |
| 10 | 5,882 | 43.4 h | 1,986 | 446 | ~70 min |

**Seven is the recommendation** — real calendar slack, and a question count no
reviewer calls thin. But ingestion is resumable and questions are
per-conversation, so the number is decided by when you stop, not in advance.

### Four cost decisions taken in the runner, each stamped

1. **Obligations deterministic, not LLM-parsed** — saves ~1.1 h; the temporal
   filter is fail-open, so a missing constraint weakens retrieval rather than
   corrupting validity. `--parse-obligations` opts in.
2. **Gate features recorded, gating off** — the gate is a small model over
   Stage-C outputs and the reader is the whole cost, so a threshold, the full
   risk–coverage curve and abstention accuracy are all available *post hoc*
   without a second reader pass. Default off also matches the reference table,
   whose systems never abstain.
3. **Gold from LoCoMo evidence markers** via `recall.tier_a_gold`, stamped Tier A.
4. **Ceilings opt-in** — ceiling 5 reads a second time, roughly doubling reader
   cost. Worth one subset, not the corpus.

### What is still not built

* **The Phase-9 policy ladder** (`N_real` = 200,000, ~2 h per arm per seed).
  Answers Gate 3; not needed for an end-to-end number. Deferred by name.
* **The three baseline adapters.** §7.
* **Phase 8 Stage B** — a conversationally *trained* gate. The recorded features
  plus `EVAL_PREVALENCES["locomo"]` make post-hoc thresholding possible from a
  MuSiQue-trained gate, which is the cheap substitute, not the same thing.
