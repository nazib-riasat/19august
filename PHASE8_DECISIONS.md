# Phase 8 — what the build decided

**Stage A built and green, 15 August 2026.** All eight modules, the runner, 32
tests. **Stage B — the conversational deletion-pair track and every decisive
evaluation number — is deferred by name** (exit criterion 13) and waits on
scope-c ingestion.

Parent: `GRAFT_PHASE8_BUILD.md` (§6 signed 15 Aug 2026) · `GRAFT_RESEARCH_PLAN_v1.md` v1.2 §4.2, §4.4, §6.4, §2.2 · `GATE0_CONTRACT.md` items 5, 9, 10 · `DATASET_DECISION.md` §1.2, §2 · `PHASE7_DECISIONS.md` §6–§7
Status: **live.** This file wins conflicts with `GRAFT_PHASE8_BUILD.md`, per the Phase-5/6/7 convention.

Labels as everywhere: **[EVIDENCE]** · **[HYPOTHESIS]** · **[ANALYSIS]**.

---

## 1. §6 adopted as signed

Decisions 1–10 and 12 built as written. Decision 11 (scope corpus) is Gate-0
item 9's and its ingestion has not run, which is exactly why the build lands in
two stages.

**Decision 1's prerequisite was closed the same day**: MuSiQue-Full downloaded
from the official archive, SHA-pinned per split, verified at 49,628 rows. What
the wiring found is in `GRAFT_PHASE8_BUILD.md` §6's sub-section.

---

## 2. Four departures from the build plan

### 2.1 `GateDecision` lives in `schemas.py`, not `graft/gate/`

G7 names `GateDecision(...)` as the gate's return type. It is defined in
`schemas.py` because it **crosses a module boundary** — Phase 10's orchestrator
consumes it — and criterion 12 makes `schemas.py` the single home of the data
model. `graft/gate/` therefore defines no dataclass at all, asserted by a test
mirroring the Phase-6/7 ones.

`SCHEMA_VERSION` **does not move**. Its stated rule is that it bumps when an
*existing record's interpretation* changes; adding `OutputRecord.abstain_cause`
with a `None` default and a brand-new class changes no existing record's
reading, only the shape. Recorded here rather than left implicit, because a
future reader could reasonably want the opposite convention — and bumping would
have moved every manifest fingerprint for a reason unrelated to the science.

### 2.2 The deletion is applied to channel outputs, not to `eligible_nodes`' scope

G1 says "excluded from `eligible_nodes`' scope". The channels call
`eligible_nodes` internally and Stage C is frozen for this phase (§2), so a true
scope-level exclusion would be a Stage-C change. The deletion is applied to the
channels' *outputs* instead, which is identical whenever no channel truncated at
its `top_k` — on the pilot it never does (8–23 candidates against 64). Rather
than assert that, each pair records `candidates_in_scope` and
`max_channel_hits` so the condition is **checkable per question**.

The property that matters is unaffected and is enforced: the twin's pool is
verified to contain **no** gold atom, and `deletion_pairs` raises rather than
counts if it ever does — a negative that is secretly answerable is the one
defect the whole training signal cannot survive.

### 2.3 The MuSiQue adapter builds report dicts, not a fake `AtomPool`

G8 says a MuSiQue question's "pool" is its paragraph set. Constructing real
`CandidateAtom`s over a fabricated snapshot would mean inventing a graph;
instead the adapter emits the *report shapes* `build_features` reads. The
features therefore come through **the same code path** (exit criterion 3)
without a synthetic graph existing anywhere.

### 2.4 `contrast_pair_accuracy` is new — it is not in the build plan

Added after the runner's first output contradicted the plan's own reasoning.
See §3.2.

### 2.5 **The channel-score features read RAW scores — a decision-3 amendment**

**Amended 15 Aug 2026, by measurement, and recorded rather than made silently**
— decision 3's cost column is "features are the claim; silent change = new
gate", so this is exactly the change that must not slip in as a code edit.

*What moved.* The `channel_scores` block was `{channel}_max` and
`{channel}_mean` over the **normalised** scores, following the architecture's own
"max/mean channel scores" wording. It now reads **raw** (pre-normalisation)
scores as `{channel}_raw_max`, `{channel}_raw_top3`, `{channel}_raw_mean`, plus
`{channel}_norm_mean` for the distribution's shape.

*What did not move.* **Decision 3's fusion arithmetic is untouched.** Min–max
per channel per question, weights 1.0, max-union, atom-id ties — all exactly as
frozen. `retrieve.fuse.assemble` gained `raw_channel_scores` **alongside**
`channel_scores`, purely additively; nothing downstream of the fusion contract
moved, and Stage C's own recall numbers are unaffected.

*Why.* See §3.3 — the measurement is the entire justification.

*Label.* **[ANALYSIS]**, and deliberately so. Departing from the architecture's
"max/mean" wording is this build's judgement, backed by the within-pair
measurement below and by nothing in the literature. No paper is cited for it
because none supports it; what supports it is a number.

*Cost.* The **stage-G fingerprint must move**, because the gate's feature set is
part of its identity and two runs with different features must not share one.

**It did not, and that was a second defect** — found by audit the same day and
fixed. `frozen_values()` bound only the five *block names*, which the amendment
did not change, so the pre- and post-amendment feature sets hashed identically
(`3908933be1e1cf30` both before and after). A fingerprint that cannot distinguish
two models is the one thing a fingerprint exists to prevent. It now binds the
actual feature **names and order** (`BLOCK_FEATURES`), `TOP_K` and `TARGET_RISK`.
This file previously asserted the fingerprint had moved; it had not, and the
claim is corrected here rather than quietly deleted.

*Two features were dropped, not merely changed.* `fused_max` is 1.0 for every
question with any channel hit — the fused score *is* the max over normalised
channels, so its maximum is 1.0 by construction — and a "raw fused mean" would
duplicate the normalised one exactly (both 0.453 on the same 400 pairs). Shipping
either as padding would have inflated the feature count without adding
information.

---

## 3. What the build measured

### 3.1 **MuSiQue's negatives are subtler than the plan describes** — measured, 400 dev pairs

The plan says unanswerability "comes from removing supporting paragraphs". What
the corpus actually does:

| | |
|---|---|
| Answerable rows' supporting paragraphs | **2**, always |
| Unanswerable twins' supporting paragraphs | **1** in 230 cases, **0** in 170 |
| Pairs with an identical paragraph *count* | **397 / 400** |

Two consequences, both recorded in `adapt_musique.ADAPTATION_LOSSES` so they
travel with the artefact:

1. **MuSiQue substitutes distractors rather than deleting**, so the whole
   `pool_shape` block is near-identical across the label. On this corpus the
   gate can only learn from `channel_scores`. `pool_shape` and `saturation`
   become discriminative on **conversation**, where a deletion twin really is
   smaller — so a feature-importance reading here does **not** transfer.
2. **The usual negative breaks one hop and leaves the other's evidence in
   place.** That is a *subtler* negative than this project's own conversational
   recipe (`labels.py` removes an evidence session outright), and it cuts
   helpfully: a partially-evidenced question that still cannot be answered is
   much closer to a deployed unanswerable. It partly offsets the adaptation loss
   `labels.py` declares — and it means a gate trained here must not be assumed to
   have learned "little evidence ⇒ unanswerable".

### 3.2 **An AURC gap between the arms is *not* evidence of leakage — the build's own note was wrong**

The runner initially recorded: *"with_question vs pool_only is expected to tie;
a gap would indicate a bug in the masking."* The first smoke run showed a gap
(`with_question` 0.0890 against `pool_only` 0.0998), so the claim was checked
rather than believed:

* the question-embedding block is **identical across the label for 40/40**
  sampled dev pairs;
* the `pool_only` mask selects **no** `q_emb` column;
* so the block provably carries **zero within-pair signal**.

The masking was correct and the note was wrong. **AURC is a *global* ranking over
the whole evaluation set**, so the embedding can improve cross-question
calibration — learning which questions have systematically higher channel scores
— without ever separating a twin. That is not leakage.

**The fix is a better instrument, not a corrected sentence.**
`riskcov.contrast_pair_accuracy` scores the leakage-immune question directly:
*within a pair, does the gate rank the answerable twin above its unanswerable
twin?* The question string is byte-identical inside a pair, so it can only be
won on pool-side features. Ties count as failures.

**And it immediately earned its place.** On the 64-pair stub smoke:

| arm/model | AURC (natural) | pair accuracy |
|---|---|---|
| pool_only/lr | 0.0998 | **0.500** |
| pool_only/mlp | 0.0968 | **0.604** |
| with_question/lr | 0.0890 | **0.521** |
| with_question/mlp | **0.0887** | **0.417** |

AURC looks healthy at ~0.09 while pair accuracy sits at chance — and the *best*
AURC arm has the *worst* pair accuracy. On a stub embedder with 64 pairs that is
the correct result (the dense channel is random noise), but it is exactly the
situation where AURC alone would have read as success. **The decisive leakage
measurement remains the zero-shot LoCoMo transfer** (G2), which is Stage B.

### 3.3 **Min–max normalisation made 10 of 13 channel features constant** — found by diagnosis, fixed, measured

**The first real run scored 0.52 within-pair accuracy against a 0.50 chance
line** (4,000 pairs, real embedder, three seeds, std 0.003–0.005 — a stable
near-chance result, not noise). AURC read a healthy 0.052 at the same time,
which is precisely the trap §3.2 had just finished describing.

**The diagnosis.** Min–max normalisation is *scale-invariant per question*: it
puts the top-scoring item at exactly 1.0 for every question, always. So of the
13 features in the block, **10 were constant** across all 400 measured dev rows:

| Feature | Value | Why |
|---|---|---|
| `bm25_max`, `dense_max`, `fused_max` | **1.0** always | min–max puts the top item at 1.0 by construction |
| `entity_*`, `expand_*`, `scorer_*` (6) | **0.0** always | absent channels on MuSiQue — expected |
| `channels_present` | **2.0** always | always bm25 + dense |

Three features varied — `bm25_mean`, `dense_mean`, `fused_mean` — with win-rates
0.517, 0.490 and **0.453** (that last one *inverted*). The gate was not
under-trained; it was reading three near-useless numbers, and no amount of extra
data can fix a constant feature.

**The signal was there the whole time.** Same 200 pairs, raw scores, win-rate
against 0.5 = no signal:

| Raw statistic | Win-rate |
|---|---|
| `bm25_max` | 0.665 |
| `bm25_top3` | 0.740 |
| `dense_max` | 0.495 |
| **`dense_top3`** | **0.795** |

**Why `top3` and not just the maximum** — and this is the part that is about
*this corpus*, not about retrieval generally. MuSiQue's answerable questions
carry **2** supporting paragraphs and its usual negative removes only one of them
(§3.1), so "is there *a* strong match" is weak evidence while "are there
*several*" is the discriminating question. `dense_max` at 0.495 against
`dense_top3` at 0.795 is that difference measured. It is also the same reason
this project selects evidence **sets** rather than single atoms.

**After the fix**, per-feature within-pair separation on 400 dev pairs:

| Feature | Before | After |
|---|---|---|
| `dense_raw_mean` | ~0.49 | **0.895** |
| `dense_raw_top3` | — | 0.868 |
| `bm25_raw_mean` | ~0.52 | 0.760 |
| `dense_raw_max` | 0.495 | 0.745 |

**And end to end.** Three seeds, identical budget, at two scales:

| arm/model | before (4k) | after (4k) | **after (full, 20k)** | AURC (full) |
|---|---|---|---|---|
| pool_only/lr | 0.519 | 0.668 | **0.788** | 0.0385 |
| pool_only/mlp | 0.520 | 0.735 | **0.784** | 0.0390 |
| with_question/lr | 0.523 | 0.726 | **0.801** | **0.0363** |
| with_question/mlp | 0.521 | 0.744 | **0.794** | 0.0432 |

Full run: 19,938 train pairs / 2,417 dev pairs, 1,319 s on the dev GPU. **~0.80
within-pair accuracy against a 0.50 chance line**, seed std 0.002–0.023.

**Scale mattered — but only after the features worked, and the order matters.**
This file's own §3.3 diagnosis predicted that more data would *not* help, and at
4k that was right for the stated reason: no quantity of rows fixes a constant
feature. Once the features carried signal, 5× the pairs bought **+0.06**
(0.74 → 0.80). Both readings are true of their own regime, and recording the
sequence is the point: *diagnose the representation before buying more data* —
the reverse order would have spent 22 GPU-minutes to reproduce 0.52.

**Two further readings from the full run.**

1. **The MLP stops earning its capacity at scale.** LR matches or beats it on
   both arms (0.788 vs 0.784; 0.801 vs 0.794) with a seed std **5–10× smaller**
   (0.002–0.003 against 0.017–0.023). Decision 2's dev-selected headline is
   therefore **LR**, the simpler arm — the control arguing *against* added
   capacity, which is the direction a control is most useful in and the one
   Phase 3's capacity lesson was learned from.
2. **`with_question` leads by 0.013–0.017**, consistent across seeds. This is
   **not yet a leakage finding** and must not be read as one: the embedding is
   provably inert *within* a pair (§3.2), so the gain is cross-question
   calibration, exactly the mechanism that section identified. The decisive test
   is the zero-shot LoCoMo transfer, which is Stage B, and decision 4's rule
   already governs the outcome — if `with_question` wins on dev and loses the
   transfer, **`pool_only` is the reported gate**.

**Two readings worth keeping.**

1. **The LR control earned its place.** `pool_only` LR 0.668 against MLP 0.735:
   decision 2's capacity comparison now has a real answer, where before both arms
   sat at chance and the comparison was vacuous. That is the Phase-3 lesson —
   a control is what tells you whether the capacity bought anything — paying off.
2. **`with_question` gains ~nothing once the pool features work** (MLP 0.744 vs
   0.735, inside noise). Before the fix it *appeared* to lead. That is consistent
   with G2's leakage argument rather than against it: the embedding is provably
   inert within a pair, and it only looked informative while the pool features
   were carrying no signal at all. The decisive leakage test remains the
   zero-shot LoCoMo transfer, which is Stage B.

**The generalisable lesson, because it is not MuSiQue-specific.** A normalisation
that is *correct for its own purpose* can destroy the information a **different**
consumer needs. Min–max is right for fusion — taking a `max` across channels
requires a shared scale — and wrong as a feature, because the gate needs absolute
evidence strength and normalisation deletes exactly that. The fix was to expose
both views, not to change the fusion.

---

## 4. What is not done

| Item | State |
|---|---|
| ~~**The MuSiQue track end to end** (step 6)~~ | **RUN 15 Aug 2026** — full 19,938 train / 2,417 dev pairs, 1,319 s on the dev GPU. Results in §3.3. *(This row said "not run" after the run had completed; corrected in the same audit that found the metric defects.)* |
| **Stage B: the conversational track** | deferred by name (criterion 13). `labels.deletion_pairs` is written and fixture-tested; it needs scope-c ingestion |
| **LoCoMo adversarial + LongMemEval's 30** | evaluation-only and **not touched**. LoCoMo is the primary abstention testbed (`DATASET_DECISION.md` §1.2's 15× argument) and is Stage B |
| **Orchestrator integration** | transferred to Phase 10 *by name* (criterion 14, G7). `decide()` is frozen, pure and tested; Phase 10 wires it |
| **The fallback counter** | reserved and zero. `OutputRecord.abstain_cause ∈ {gate, fallback}` exists so §6.4's fallback trigger rate is reportable from the first Stage-D run rather than retrofitted |

---

## 5. What Phases 9 and 10 get, unchanged

* **`gate.decide()`** — frozen, pure, no I/O, no model loading; threshold and
  model are arguments, so a `GateDecision` replays from an artefact.
* **`abstain_cause`** — the two-way vocabulary, guarded: a cause on a
  non-abstention is refused, and the gate can only ever produce `gate`.
* **`riskcov`** — reused unchanged by Phase 10/11 on end-to-end outputs, and it
  imports no torch, so the reader path does not acquire an ML dependency through
  the abstention metrics.
* **`contrast_pair_accuracy`** — the leakage-immune metric, for any future
  contrast-pair evaluation.
* **The stage-G fingerprint**, fifth in the identity quintuple printed by
  `verify_handoff.py`.
