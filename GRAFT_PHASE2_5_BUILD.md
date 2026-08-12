# GRAFT — Phase 2.5 Build Plan: the annotation feasibility spike

**One measurement, one week, one number.** Gate 0 item 8 — *how many annotations
are actually feasible* — answered by measuring instead of estimating.

Date: 12 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 2.5) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §7 Gate 0, risk #15) · `CLAUDE.md` §7
Effort: **1 week of work spread over ≥ 4 calendar days** — the self-agreement re-annotation needs a ≥ 2-day gap (G4)
Status: **§6 unsigned.** Nothing here has run.

Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (judgment made here).

Gaps are numbered **G1–G7**, matching the Phase-0/1/2/3 convention.

---

## 0. What this phase is for, and what it is not

**The question.** Contribution 1 — the learned graph constructor — is the
project's most defensible result *and* its fallback if Gate 2 or Gate 3 fails
(plan §9). It needs supervision for four decoders. Two of them, **D1 (mention
resolution) and D2 (claim-pair relation), have no off-the-shelf source**: the
architecture names them as such, and `CLAUDE.md` §7 records that **no dataset
provides conflict/supersession annotation at all**, which makes D2 the binding
constraint on C1.

So the question is not "is the architecture right". It is: **can one person, in
the time available, produce enough D1 and D2 labels for Gate 1 to mean
anything?** If the answer is no, the scope must narrow *here* — by cutting
schema or question types — which is exactly what Gate 0's stop condition asks
for, rather than later by quietly weakening the evaluation.

**Why now, before Phases 4–8.** Three facts, from the architecture:

1. Stage B is the strongest and most supervisor-aligned contribution.
2. Annotation infeasibility is the largest named risk (v1.2 risk #15) and Gate 0's
   stop condition depends on it.
3. A Gate-2 pass establishes that the *learners* work. It establishes **nothing**
   about whether real conversational proof supervision can be obtained.

One week here converts the project's biggest unknown into a number before
Phases 5–8 are built on the assumption that the answer is yes.

**This plan is deliberately short, and that is a design decision.** The
architecture's own warning is that this phase's main failure mode is *growing
into an early Phase 6*. Phase 2 and Phase 3's plans run past 600 lines because
they specify instruments that later phases are held to. This one specifies a
measurement that is thrown away once its number exists. A long plan here would
be the first symptom of the failure it is warning about.

---

## 1. Seven gaps the architecture leaves open

### G1 — The go/no-go has no denominator [ANALYSIS]

Deliverable 5 is "annotations achievable per week × weeks available, **against
the volume Phase 6 needs**". Nothing anywhere states that volume. Without it the
phase produces a numerator and calls it an answer.

**The required volume is a statistical-power question, not a guess.** Gate 1
compares three encoders (E1 GraphMixer-style, E2 HGT, E3 proposed) on D1's
end-to-end mention-resolution score and D2's macro-F1, over three seeds. The
test set must be large enough that the paired interval is narrower than an
effect worth acting on. So Phase 2.5 must derive and record:

* `n_test` — from a power argument on the **primary** Stage-B metric (v1.2 §6.4:
  the end-to-end D1 score, action *and* entity ID), for a difference large enough
  to change the Gate-1 decision;
* `n_train` — the count needed for the decoders to be trainable at all, stated
  as an assumption with its basis, not derived;
* both **per decoder**, because D1 and D2 cost different amounts per item (G2).

**The honest output may be a range.** If `n_test` is defensible only under
assumptions this phase cannot settle, record the assumptions and the range rather
than a single number. A stated range beats a false point estimate.

### G2 — "Minutes per item" is unmeasurable until "item" is defined [ANALYSIS]

Deliverable 4 measures minutes-per-item. The two decoders' items are not
comparable and must be timed separately:

| | One item is | Cost driver |
|---|---|---|
| **D1** | one *mention* → one of `LINK_EXISTING(id)` / `CREATE_NEW_ENTITY` / `NON_ENTITY` / `DEFER`, **plus the entity ID when linking** | scanning the candidate entity list; the ID is what makes it slow, and v1.2 §6.4 makes it part of the primary metric |
| **D2** | one *claim pair* → one of `INDEPENDENT` / `DUPLICATE` / `CONFLICT` / `SUPERSEDES` | reading two claims **and** their time intervals; supersession needs the temporal relation, not just the text |

Timing must be **per item, wall-clock, including the reading the item requires** —
not total-session time divided by count, which hides the fact that D2 items
arrive in bursts around a single re-read.

### G3 — The corpus is not pinned, and is not in the repo [ANALYSIS]

The architecture says "~50 real turns" without saying whose. `Research papers/INDEX.md`
§7 selects **LongMemEval** as primary and **LoCoMo** as secondary, and neither is
in this repository.

**Pinned here: LongMemEval-S**, because it is the plan's primary end-to-end
benchmark (v1.2 §6.1) and its **knowledge-updates subset is the closest thing
that exists to D2 supervision** (`CLAUDE.md` §7). Sampling for the spike must
**over-sample that subset deliberately** — a uniform sample of conversational
turns yields almost no `CONFLICT` or `SUPERSEDES` pairs, and a measurement of how
long it takes to label 50 `INDEPENDENT` pairs answers nothing.

Recorded as a dependency: the dataset must be fetched, its licence checked, and
its release SHA written down. **[ANALYSIS]** over-sampling is right for a *timing*
measurement and would be wrong for a *training* set; the phase measures cost per
item, not class balance, and the write-up must say so.

### G4 — Self-agreement forces a calendar constraint [ANALYSIS]

Deliverable 4 wants self-agreement on a 20-item subset re-annotated "after a
delay of at least two days". One annotator, so no inter-annotator agreement is
available; self-agreement is the honest substitute and **it is weaker** — it
measures the stability of one person's reading, not the clarity of the
guidelines to anyone else.

Consequence: **this phase cannot be compressed into consecutive working hours.**
Annotate on day 1, re-annotate on day 4. The build order below is written around
that gap rather than pretending it away.

Report Cohen's κ, not raw agreement: with a 4-way decision and a skewed class
distribution, raw agreement flatters. **[EVIDENCE]** it is the standard
convention the plan's own annotation-protocol requirement (Gate 0 item 7) implies.

### G5 — This is the project's first GPU workload, and the VRAM budget is unverified

Deliverable 1 runs Stage-A extraction on ~50 turns, which means loading
**Qwen2.5-7B-Instruct at 4-bit** for the first time. Architecture fix F7 sized the
VRAM budget against **32 GB**; the machine has **8.5 GB** (`chatcontext1.md` §4,
still an open item).

So this phase incidentally becomes the **first real test of F7**, and that is worth
capturing rather than discovering in Phase 5: record peak VRAM, whether 4-bit 7B
loads at all, and tokens/second. If it does not fit, that is a Phase-5 finding
delivered four phases early and at the cost of one afternoon.

**Not in scope:** fixing it. Recording it.

### G6 — Annotations need a storage contract, because Phase 6 consumes them [ANALYSIS]

Throwaway annotation lives in a spreadsheet and dies there. These labels are the
seed of Gate 1's training set, so they need a format Phase 6 can load. **JSONL,
one object per item, with the Phase-0 id conventions** — `mention_id` / `span_id`
/ `assertion_id` — so a label can be traced back to the turn and offsets it came
from. Two files, `d1_labels.jsonl` and `d2_labels.jsonl`, plus a `provenance.json`
naming the corpus SHA, the extractor, and the config hash.

**No new schema in `graft/schemas.py`.** Phase 2.5 writes files; it does not
extend the data model. That is Phase 5/6's decision and making it here would be
the "growing into Phase 6" failure.

### G7 — Annotating bad extractions measures the wrong thing [ANALYSIS]

D1 and D2 items are derived from Stage-A output. If extraction is poor, the
annotator spends the measured minutes deciding whether a *garbage* mention is an
entity — and the resulting minutes-per-item describes a pipeline nobody will run.

**A floor, checked before annotation starts:** manually audit span-grounding on
the first 20 extracted assertions. If precision is below the Gate-0 threshold, the
timing measurement is not valid and the extraction prompt is fixed first. This is
a *gate on the measurement*, not a Phase-5 quality bar — the number to beat is
"good enough that the annotator is judging real mentions".

---

## 2. Scope

**In.** A thin Stage-A slice on ~50 LongMemEval turns; D1 and D2 annotation
guidelines v0; ~100 D1 and ~50 D2 annotated items; per-item timing; self-agreement
on 20 items after a ≥ 2-day gap; the required-volume derivation of G1; the go/no-go
figure; the VRAM/throughput record of G5.

**Out.** Every encoder. Every decoder. Any training. Any change to
`graft/schemas.py`. The Phase-6 commit pipeline. D3 and D4 annotation — they have
off-the-shelf supervision interfaces (DialogRE/Re-DocRED, TORQUE/MATRES) and are
not the binding constraint. Anything that would still exist after this phase's
number is known.

---

## 3. Build order

| Day | Step | Done when |
|---|---|---|
| 1 | Fetch LongMemEval-S; record SHA and licence; sample ~50 turns, over-sampling knowledge-updates (G3) | the sample is reproducible from a recorded seed |
| 1 | Load the 4-bit 7B extractor; run Stage-A on the sample; record peak VRAM and tokens/s (G5) | extraction completes, or the VRAM failure is recorded as a Phase-5 finding |
| 1 | Span-grounding floor on the first 20 assertions (G7) | precision recorded; if below the floor, fix the prompt and re-run before proceeding |
| 2 | Write D1 and D2 guidelines v0 — the four actions, the four relations, and a worked example of each | a second person could apply them; that is the bar, even though no second person exists |
| 2–3 | Annotate ~100 D1 items and ~50 D2 pairs, **timing each item** (G2) | `d1_labels.jsonl`, `d2_labels.jsonl`, `provenance.json` (G6) |
| 3 | Derive `n_test` and `n_train` per decoder (G1) | the power argument and its assumptions are written down |
| **≥ 4** | Re-annotate the 20-item subset; compute Cohen's κ (G4) | κ recorded per decoder |
| 4 | The go/no-go figure and the scope-reduction options if it is negative | `PHASE2_5_DECISIONS.md` exists |

---

## 4. Exit criteria

1. **The Gate-0 item-8 number is a measurement with a stated method** — items per
   hour per decoder, from timed items, not an estimate.
2. `n_test` and `n_train` are derived per decoder with their assumptions stated,
   so the go/no-go is a comparison of two numbers rather than of one number and a
   feeling (G1).
3. **The go/no-go figure is written down with its arithmetic**: achievable per
   week × weeks available, against the required volume, per decoder.
4. Cohen's κ on the 20-item re-annotation, per decoder, with the ≥ 2-day gap
   recorded (G4) — and labelled as **self**-agreement, weaker than IAA.
5. Guidelines v0 exist and are specific enough to hand to a second annotator.
6. Labels are in Phase-6-loadable JSONL with provenance back to turn and offsets
   (G6).
7. Span-grounding precision on the audited 20 is recorded, and the timing
   measurement is declared valid or invalid against the G7 floor.
8. Peak VRAM, load success and extraction throughput for the 4-bit 7B are
   recorded against fix F7's assumption (G5).
9. **If the answer is negative, the scope reduction is chosen here** — narrower
   schema, fewer question types, or D2 deferred — and written into the plan.
   Deferring the decision is the one outcome this phase may not produce.
10. Nothing in `graft/` gained an encoder, a decoder or a training loop.

---

## 5. What a negative answer costs, and what it buys

**[ANALYSIS]** If the volume is unreachable, three reductions are available, in
increasing cost to the thesis:

| Reduction | What it protects | What it gives up |
|---|---|---|
| **Narrow D2 to `CONFLICT` vs `not-conflict`** | keeps the conflict-detection story, which is what the temporal/update claim rests on | the four-way relation decoder; the D2-grouped-vs-split ablation |
| **Defer D2 entirely; ship D1 + D3 + D4** | keeps open-world entity resolution, which is C1's strongest published gap (Learn to Not Link) | conflict and supersession — a large part of what makes the graph *provenance-preserving* |
| **Fewer question types end-to-end** | keeps all four decoders | the breadth of the Gate-4 comparison |

The reason to know this *now* rather than at month 4 is that all three change what
Phases 5–8 build. Discovering it after they are built means rebuilding them.

---

## 6. Decisions to lock before annotating — **UNSIGNED**

| # | Decision | Value | Cost if changed later |
|---|---|---|---|
| 1 | **Corpus** | LongMemEval-S, knowledge-updates over-sampled for D2 (G3) | timings describe a distribution the project will not annotate |
| 2 | **Sample size** | ~50 turns → ~100 D1 items, ~50 D2 pairs, per the architecture | too few to time reliably; too many and the spike becomes the annotation |
| 3 | **Item definitions** | G2's table — D1 includes the entity ID, D2 includes the interval | minutes-per-item is not comparable to the volume it is divided into |
| 4 | **Agreement statistic** | Cohen's κ, self-agreement, ≥ 2-day gap, 20 items (G4) | raw agreement flatters a skewed 4-way decision |
| 5 | **Span-grounding floor** | audited on 20; below it the timing is void (G7) | the measurement describes annotating noise |
| 6 | **Storage** | JSONL + `provenance.json`, no change to `graft/schemas.py` (G6) | the labels are unusable by Phase 6, which is the only consumer |
| 7 | **Required volume** | derived in-phase per G1, assumptions stated | the go/no-go has no denominator |

**Not decided here, deliberately:** the full Gate-0 data contract (v1.2 §7 items
1–10). Item 8 is what this phase measures; the other nine are written before
Phase 5 and are a separate piece of work. Conflating them is how a one-week spike
becomes a month.

---

## 7. What Phase 5 gets from this

* A go/no-go number for annotation volume, with method.
* Guidelines v0 for D1 and D2, already tested on real items by one annotator.
* ~150 labelled items — small, but real, and in a format Phase 6 can load.
* A measured answer to whether the 4-bit 7B extractor fits in 8.5 GB, which fix
  F7 assumed against 32 GB and nobody has checked.
* If the answer is negative: a chosen, recorded scope reduction, before Phases
  5–8 are built on the assumption that it was positive.
