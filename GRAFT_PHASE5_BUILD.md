# GRAFT — Phase 5 Build Plan: Stage A, ingestion and extraction (`graft/ingest/`)

**Turns → verified, span-grounded, support-gated assertions in the event log.**

Date: 13 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 5, fixes F2/F7/F9) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §3.1, §4.4, Gate 0) · `PHASE0_DECISIONS.md` §2.5/§2.10 · `PHASE1_DECISIONS.md` §3.2/§5.4 · `PHASE2_5_DECISIONS.md` (the spike's measurements)
Effort: ~1.5–2 weeks solo **[ANALYSIS]** — an estimate; the extractor bakeoff (G2) and the pilot run are the schedule risks.
Status: **built, green and run (14 Aug 2026) — `PHASE5_DECISIONS.md` is the record and wins any conflict with this file.** §6 is **adopted**, with decision 2 frozen by measurement on **candidate B** and decisions 3/4 amended by measurement (rows below carry the amendments). The bakeoff's first run was aborted and its instrument corrected before anything froze (§G2, decisions §1.6); the re-run froze B at 1.7% parse failure, and the live pilot ran 248 turns end to end. **Every machine-measurable exit criterion is met.** What remains is human: the four audit worksheets (§5 of the decisions file) and the Phase-2.5 timed pass (Gate-0 item 8) — *parallel* work, not a blocker for building, but a blocker for **Gate 1**, and §0 says where that dependency bites.

Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps are numbered **G1–G11**, matching the Phase-0/1/2/2.5/3/4 convention.

---

## 0. What Phase 5 is for, and what it is not

**The founding requirement returns.** Phases 1–4 ran on a synthetic lattice where
every atom was born grounded. Phase 5 is where extraction error enters the
system, and the architecture's two-layer design (fix F9) exists for exactly this
moment: the **audit layer** stores everything — which is what makes extraction
quality measurable at all — and the **active layer** admits only assertions the
support gate marks eligible, so an invented claim can never become retrievable
evidence. `H`'s seventh sub-check already reads the eligibility flag (Phase 1);
Phase 5 is what finally *writes* it from real data.

**[EVIDENCE]** the risk being engineered against is no longer hypothetical:
*How Memory Management Impacts LLM Agents* (ACL 2026) empirically documents
error propagation through stored memory — the risk `CLAUDE.md` §7 records as
upgraded from `[ANALYSIS]` to `[EVIDENCE]`. And the motivating number for the
whole project sits at this stage's output quality: ALCE (EMNLP 2023) found even
the best models lack complete citation support 50% of the time on ELI5, and
VeriCite (SIGIR-AP 2025) showed **verification, not generation** is what fixes
it (citation F1 77.73 → 68.91 with the NLI check removed).

| Downstream | Blocked on Phase 5 by |
|---|---|
| Phase 6 (Stage B, Gate 1) | eligible assertions are its entire input; D1/D2 items derive from Stage-A mentions and claims |
| Ceiling 1 (extraction) | "are gold statements represented as grounded assertions?" is unmeasurable until this exists |
| Gate-0 items 1–7, 9, 10 | the contract is written against *this* phase's concrete record shapes; item 8 is the Phase-2.5 human pass |
| Phase 8 (gate) / Phase 9 | the learned obligation parser (fix F2) ships here and both consume it |

**Not in Phase 5:** entity nodes, edges, or any of D1–D4 (Phase 6); retrieval
(Phase 7); the answerability gate (Phase 8); the reader (Phase 10); any change
to `H`, `U`, `R`, the masks, `d(s)` or the synthetic environment. Stage A is
**hybrid, not deterministic** (plan §3.1): storage, hashing and offsets are
deterministic; extraction, coreference and entailment are learned and carry
error — and the plan's naming discipline applies, so nothing in this phase may
be described as verifying *truth* (§4.4).

---

## 1. Eleven gaps this phase must close

### G1 — The exit criterion cites a Gate-0 threshold that does not exist [ANALYSIS]

The architecture's exit criterion says span-grounding precision is *"audited
manually against the Gate-0 threshold"*. No document declares that threshold,
and the Gate-0 contract (plan §7, items 1–10) is still unwritten — `CLAUDE.md`
§7 and the project memory both name it the binding prerequisite for Phases 5–9.

Two resolutions, both here:

1. **Gate-0 drafting is a named deliverable of this phase** (P5.0): items 1–7,
   9 and 10 are draftable now against concrete Phase-5 record shapes; item 8's
   number comes from the Phase-2.5 human pass and its cell says "pending"
   rather than blocking the other nine. *Nothing in Phase 5 trains*, so the
   plan's "nothing is trained before this is signed off" is not violated by
   building; **Gate 1 may not run** until the contract is signed.
2. **The threshold is declared: manual span-support precision ≥ 0.90 on a
   50-assertion audited sample** of the production extractor's pilot output.
   **[ANALYSIS]** — tighter than the spike's 0.80 floor (`PHASE2_5_DECISIONS.md`
   A2) deliberately: the spike's floor guarded a *timing measurement*; this one
   guards Phase 6's *training data*, and a 1-in-10 unsupported-assertion rate
   is already the level the support gate exists to catch, not to admit. The
   audit protocol is part of the declaration: an assertion counts as supported
   iff its grounded span, read alone plus the turn it came from, textually
   commits to the assertion's `text_norm` — the same reading discipline the
   spike's G7 audit used, written down so a second auditor could apply it.

### G2 — The spike falsified the "just prompt it" assumption, and the extractor is unpinned [ANALYSIS]

The architecture pins Qwen2.5-**7B**-Instruct 4-bit; the machine has 8 GB
(`CLAUDE.md` §7's unresolved hardware row); the spike ran **3B bf16** as a ruled
spike-local deviation and measured what matters:

| Spike measurement | Value | Consequence for Phase 5 |
|---|---|---|
| JSON parse failure | **9/58 turns (15.5%)** | those turns yielded *nothing* — a silent 15% evidence loss that would poison ceiling 1 before anything downstream runs |
| grounding failures | ~8.9% of extracted objects | dropped and counted, per the plan — acceptable if reported |
| mention yield | 0.59/turn | a third of the plan's assumption; D1 volume planning must use the measured rate |
| VRAM / speed | 6.5 GB peak · 18.4 tok/s | 3B bf16 fits with ~1.6 GB headroom; the 7B-4bit question is still unmeasured |

A production Stage A cannot ship a 15.5% silent turn loss. Three candidate
configurations, none free:

| Candidate | For | Against |
|---|---|---|
| **A: 3B + bounded repair-retry** (greedy parse → on failure, one reprompt with the parse error + "JSON only", then count) | no new dependency; measured stack | retries cost throughput; failure floor unknown |
| **B: 3B + grammar-constrained decoding** | never *malformed*: every step is a valid JSON prefix. *(Corrected 13 Aug 2026, by measurement: the original cell claimed the parse-failure rate is zero by construction, which is false under a finite token budget — nothing makes the object **close** within `max_new_tokens`, and a truncated object does not parse. B's 3-turn check produced exactly that failure, cause `truncated_at_token_cap`. The guarantee is prefix validity, not completion, and the truncation failure mode is shared with A.)* | a new dependency (boring-stack rule: written justification required); constrained decoding can degrade content quality |
| **C: 7B-4bit + repair-retry** | the architecture's own pick; larger models emit valid JSON more reliably | unmeasured on this GPU (F7's original question); ~2× slower |

**Decision — a predeclared bakeoff, run once, then frozen.** On a fixed
60-turn calibration slice (the spike's 58-turn sample + 2 held-back turns,
disjoint from the pilot's audit sample — enforced in code: the slice definition
lives in `graft.ingest.bakeoff.calibration_slice` and the pilot's span/NLI
audit draws exclude any assertion touching a slice turn), all three candidates
run with the same prompt and the pick is made by a rule declared **before** the
run:

1. parse-failure rate **< 2%** (hard filter);
2. among survivors, highest **grounded assertions per minute**;
3. tie-break: higher G1 span precision on a 20-assertion sub-audit.

This is instrument calibration — no learner exists anywhere near it — and the
predeclaration is the project's own protocol discipline (its significance-
testing authority, Dror et al. ACL 2018, covers test selection; fixing the rule
in advance is the project's own **[ANALYSIS]** rule). The winner's model id,
revision hash, quantization, prompt SHA and decoding config are frozen in §6
and stamped into every run manifest (G11).

**The first run (13 Aug 2026) was aborted and the instrument corrected before
anything froze** (`pins.EXTRACTOR` was still `None`; the aborted artefact is
preserved as `artefacts/phase5_bakeoff_AB.json`). Four corrections, all
measured, recorded in `PHASE5_DECISIONS.md` §1.6: the harness windowed one flat
60-turn list across ten different users' conversations (71% of context turns
were foreign) — the slice is now grouped per conversation with the production
context recipe, rolling summary included; the unclipped *m* = 10 window over
essay-length turns measured 5,017 input tokens/turn and overflowed the 8 GB
card — context turns are now head-clipped (`CONTEXT_CLIP_CHARS`, decision 4's
declared adaptation); `max_new_tokens` 600 was inferred to be truncating every
failing generation and was raised to 1,024, then — after the run at 1,024
returned `no_survivor` — to **2,048**, under a prediction written down *before*
the re-run and confirmed by it (decisions §2.1a/§2.1b); and truncations are now
counted per generation, because a repaired first-attempt truncation vanished
from the per-turn count. **The inference about the cap was itself falsified**:
candidate A's failures turned out to be 14 malformed against 0–1 truncated, so
the raise was right for the stated reason (a cap is a runaway guard, never a
content limit) and wrong about A. The rule itself is unchanged throughout.

### G3 — "Asynchronously refreshed" summary vs. the boring stack [ANALYSIS]

**[EVIDENCE, qualified]** Mem0 (ECAI 2025) extracts with the previous *m* = 10
messages plus an asynchronously refreshed conversation summary — the context
recipe the architecture adopts. "Asynchronous" in a one-process, no-services
stack (architecture §0.4) would be an invented scheduler.

**Decision.** The *content* recipe is kept and the *scheduling* is a declared
adaptation: the summary is refreshed **synchronously every `s = 10` turns** by
the same extractor model (one extra LLM call per 10 turns, metered), capped at
512 tokens, and the extraction context is `summary + previous min(m, 10) turns
+ current turn`. Summaries are **derived state, not evidence**: they are cached
in the run directory keyed by `(conv_id, turn_ix)`, never written to the event
log — the log stores what was said and what was extracted, and a summary is
recomputable from the log plus the frozen config. Mem0 supports the recipe as
*an implemented design, not an optimum* (the plan's own caveat), so `m` and `s`
are §6 constants, not tuning knobs.

### G4 — Nothing yet writes real data to the event log, and idempotence is undefined [ANALYSIS]

Phases 0–1 built the log, the stores and the ops (`turn.add`, `span.add` landed
with Phase-1 G6); no code path has ever written a *corpus* through them, and the
assertion write path — assertion record + four flags + eligibility — must be
audited/extended where missing (the replay stores already read assertions and
eligibility, so the op vocabulary is completed here, not invented).

**Idempotence is the decision that has to be made now**: re-running ingestion
over the same corpus must not duplicate evidence. All ids are content-derived
(Phase 0), so duplication is *detectable*; the rule adopted is **skip-and-count
at the turn level** — a `turn_id` already present in the log short-circuits that
turn's entire pipeline, and the skip count is a reported metric. A crashed run
is therefore resumed by re-running it (the log is crash-safe by Phase-0
construction, and the torn-write test already exists).

Reads during ingestion are pinned: the support gate and any lookups run against
`GraphStore.at(snapshot_id)` semantics, so a record's eligibility verdict is
reproducible from the log alone.

### G5 — The fuzzy grounder mis-bounds spans, measured [ANALYSIS]

The spike's exact-then-fuzzy grounder produced one **mis-bounded span**
(`d1_0022`: a leading `"s "` glued to *Sankeien Garden*) — a fuzzy window
landing one word off. Phase 6 trains on these offsets, so a mis-bound is not
cosmetic.

**Decision — the grounding ladder, frozen:**

1. exact substring match;
2. case- and whitespace-normalised exact match (offsets mapped back);
3. fuzzy window (`difflib`, acceptance ≥ 0.85) **with word-boundary snapping**:
   the matched window is expanded/trimmed to the nearest word boundaries unless
   the quote itself starts or ends mid-word;
4. failure: dropped and counted, per the plan.

Every pilot span reports which rung grounded it; **all rung-3 spans in the pilot
are manually audited** (they are few — the spike measured ~9% grounding
*failures* and rung-3 successes are rarer still), and the mis-bound rate is a
reported number with the same status as the quarantine rate: a signal, never a
threshold to quietly tune.

### G6 — The NLI verifier is named by style, not pinned [ANALYSIS]

The architecture says *"off-the-shelf NLI cross-encoder (TRUE-style)"*.
**[EVIDENCE]** the *pattern* is VeriCite's (SIGIR-AP 2025): an NLI model was the
cost-effective verifier against LLM judges (citation F1 80.05 vs 73.01 for an
8B LLM). But TRUE's own reference model is a T5-11B — unusable in 8 GB — so the
concrete pin is **[ANALYSIS]**: a DeBERTa-v3-class NLI cross-encoder from the
sentence-transformers cross-encoder family, pinned by model id + revision in §6.

Semantics, frozen here:

* **premise** = the grounded span(s), in turn order, plus nothing else —
  entailment is *by the span*, not by the conversation, or the flag stops
  meaning what `entailed_by_span` says;
* **hypothesis** = `text_norm`;
* score → `entailed_by_span = (bool at tau_nli, score)`; `tau_nli = 0.8` is
  **frozen** (Gate-0 value, Phase 0) — Phase 5 *audits* it on a ~50-pair
  hand-labelled pilot sample (agreement at the threshold, reported) and does
  **not** retune it; a miscalibrated threshold is a reported finding and a
  Gate-0 amendment, never an implementation-time adjustment;
* the verifier **never blocks storage** (architecture): an unentailed assertion
  is stored with its flag false — the support gate, not the verifier, decides
  eligibility.

**F7 discipline:** the extractor and the NLI model are never resident together.
Ingestion runs stage-sequentially — extract the whole slice, free the extractor,
then verify — which the 8 GB card makes mandatory rather than stylistic.

### G7 — The learned obligation parser exists only as a `NotImplementedError` [ANALYSIS]

Fix F2 routes real questions through `parse(mode="learned")`; Phase 1 built the
interface and the audit hook (`slot_level_scores`, corrected in Phase-1 §5.5)
and left the implementation to this phase.

**Decision.** The extractor LLM fills the typed slots from the question plus
its `question_date`: `entity_anchor`, `value_type`, `time_constraint`,
`needs_source`, `aggregate`, `scope`. Two constraints inherited from Phase 0
land here and are easy to violate silently:

* **relative dates resolve against `question_date` and widen to their natural
  granularity** — "last May" becomes the half-open month interval, because a
  zero-width instant contains nothing (`PHASE0_DECISIONS.md` §2.5's explicit
  requirement on Stage A);
* **the unbounded-constraint rate is a reported number** — an unbounded interval
  scores `temporal_correctness = 1.0` by the G5 (Phase 1) convention, so a
  parser that emits them freely quietly disables a reward term (the Phase-1
  plan's own warning, now measurable).

The audit: ~50 LongMemEval questions hand-labelled for all six slots
(30 minutes of work, not a Gate-0-scale annotation), scored with the existing
`slot_level_scores`, and the number is **reported wherever coverage is
reported** — fix F2's own words.

### G8 — The full corpus does not fit this machine, and pretending otherwise would surface in week 6 [ANALYSIS]

LongMemEval-S is ~500 questions × ~50 haystack sessions each. At the spike's
measured 18.4 tok/s a full ingestion is **weeks of wall-clock on this laptop**
— the architecture's own exit criterion ("extraction throughput measured —
budget check for full corpora") exists precisely to surface this before Gate 1
is scheduled.

**Decision — measure, then size, then decide at Gate 0.** The pilot (G10)
measures turns/hour end-to-end. `scripts/phase5_pilot.py` then emits a **sizing
memo**: projected wall-clock for (a) the full corpus, (b) evidence sessions +
`d` sampled distractor sessions per question, (c) a `q`-question subset — on
this machine and, per `README.md`'s existing Kaggle path, on a T4/P100 session
budget. The corpus *scope* for Gates 1 and 4 is *Gate-0 item 9's decision*,
made with the memo in hand — not this phase's, and not made by silently letting
the ingestion run forever. **[ANALYSIS]** the one commitment made here: the
knowledge-update evidence sessions are in every candidate scope, because D2's
supervision (the binding constraint on C1, `CLAUDE.md` §7) lives there.

### G9 — The extraction schema needs freezing, and the plan promises multi-span cross-turn provenance the spike never exercised [ANALYSIS]

Plan §3.1: *"Provenance may be multi-span and cross-turn. A claim assembled
across turns records every supporting span, not one."* The spike's schema was
single-turn, single-quote. Phase 5 freezes the production schema:

* `mentions: [{text}]` — exact substrings of the current turn;
* `assertions: [{kind ∈ {claim, value, event, time} (frozen vocabulary — the
  spike validated it), text_norm (self-contained: pronouns and relative dates
  resolved from context), quotes: [{turn_offset ∈ {0..−m}, text}]}]` — each
  quote names which context turn it comes from, `0` = current;
* grounding resolves each quote against *its* turn; an assertion is grounded
  iff **every** quote grounds; `Assertion.spans` then carries multi-span
  cross-turn provenance exactly as Tier B's schema always permitted;
* `asserted_by` = the *current* turn's speaker (the assertion event is the
  utterance being processed), with the usual four-flag discipline — grounding
  is not truth, and no field named `true` exists (Phase-0 §P0.2).

**Tier B freezes here.** `Turn`, `SourceSpan`, `Assertion` were "amendable until
Phase 5" (Phase-0 G5); this is Phase 5. The only amendment taken: none —
the spike exercised the shapes and they held. `SCHEMA_VERSION` moves only if a
later gap forces a field, and the freeze is recorded in `PHASE5_DECISIONS.md`
either way.

### G10 — The pilot must be one declared object, not a vibe [ANALYSIS]

Every exit criterion below is measured **on the pilot**, so the pilot is part of
the experimental record:

> **Pilot :=** the Phase-2.5 sample's 10 questions with their **full** evidence
> sessions (no windowing — ~250 turns), ingested end-to-end through the frozen
> G2 winner; plus the ~50-question obligation-parser audit set (G7); plus the
> 50-assertion span audit (G1) and the ~50-pair NLI audit (G6) drawn from its
> output by seeded sample.

Corpus, SHA and licence are already pinned (`scripts/phase2_5/common.py`). The
pilot is re-runnable from one command and its report is one JSON artefact plus
the sizing memo.

### G11 — An LLM stage breaks the project's byte-identity guarantee, and the honest move is to say where [ANALYSIS]

`README.md` promises identical `reproducibility` blocks and log digests across
machines — true for Phases 0–4, which are numpy and enumerations. A bf16 LLM
forward pass is not bit-stable across GPU architectures, and one flipped token
changes an extraction, its ids, and the digest.

**Decision.** Per-machine determinism is asserted and tested (greedy decoding,
fixed prompts, two runs → identical digests **on one machine**). Cross-machine
identity is **explicitly not promised for Stage A**; what must match across
machines instead is the *configuration identity*: model id + revision hash +
prompt SHA + decoding config, all stamped into the manifest (the same
hash-equality discipline architecture §10.1 already mandates for the reader).
`verify_handoff.py` gains an ingestion fingerprint that binds the config, not
the output. Recorded as a limitation in the write-up, not discovered as a
mystery in Phase 11.

---

## 2. Scope

**In.** `graft/ingest/`: corpus adapter, rolling summary, extractor wrapper with
the G2 bakeoff, span grounder, NLI verifier, support gate, per-turn pipeline
with idempotent event-log writes and full metering; the learned obligation
parser (fix F2) and its audit; the Gate-0 contract draft (items 1–7, 9, 10);
Tier-B schema freeze; the pilot runner and sizing memo; tests.

**Out.** Everything Stage B and later (§0). No new schema outside the freeze.
No retuning of `tau_nli` or any frozen value. No corpus-scope decision (Gate
0's, informed by G8's memo). **`graft.core` and `graft.synth` still import no ML
library** — the per-package boundary of Phase 3 extends: `graft.ingest` may
import torch/transformers; the structural test is narrowed per package again,
never deleted.

---

## 3. Modules

### P5.0 Gate-0 draft and the Tier-B freeze (do this first)

Write `GATE0_CONTRACT.md` — plan §7's ten items, with item 8 marked *pending
the Phase-2.5 human pass* and every other item filled against Phase-5's concrete
record shapes (which labels supervise D1–D4 and from where — the
`Research papers/INDEX.md` §7 selection is the input; splits chronological +
user-level; negative-example policy; annotation guidelines v1 seeded from the
spike's flagged items; the G1 threshold; the predeclared primary metrics, which
plan §2.4 already tables). Freeze Tier B (G9). **Gate 1 was blocked on this
document being signed (it signed 15 Aug 2026); building Phase 5's code never was.**

### P5.1 `ingest/corpus.py`

LongMemEval-S → a deterministic `Turn` stream: conv/session/turn ids from the
2.5 conventions, session dates → `ts` (ISO, per Phase-0 §2.5's split between
human-auditable `Turn.ts` and comparable `Interval`s), speaker from role.
Reuses the pinned SHA check. Emits per-question metadata (type, evidence ids)
as sidecar, not as evidence.

### P5.2 `ingest/summary.py`

G3's synchronous rolling summary. One public function:
`context_for(conv, turn_ix) -> (summary, window_turns)`. Cache keyed by
`(conv_id, turn_ix)`; a summary is recomputable and never logged.

### P5.3 `ingest/extractor.py`

The G2 bakeoff harness and the frozen winner behind one interface:
`extract(turn, context) -> RawExtraction` (G9's schema), with the repair-retry
policy inside, `llm_calls`/`llm_tokens_*` metered, prompt registry hashed.
Greedy decoding always. The obligation-parser prompt (P5.7) lives in the same
registry so one SHA covers every prompt this phase runs.

### P5.4 `ingest/grounding.py`

G5's four-rung ladder. `ground(quote, turn_text) -> (start, end, rung) | None`,
plus `ground_assertion` mapping every quote to its context turn (G9). Reports
per-rung counts and the boundary-snap audit list.

### P5.5 `ingest/nli.py`

G6's pinned cross-encoder. `verify(assertions) -> scores`, batched,
stage-sequential (never co-resident with the extractor), threshold application
separate from scoring so the audit can sweep the score distribution without
re-running the model.

### P5.6 `ingest/support.py`

Fix F9's gate, deliberately thin: `eligibility(assertion) -> eligible |
quarantined` under `support_policy = strict` (`entailed_by_span` true at
`tau_nli` **and** every span grounded). The quarantine rate — total and broken
out by cause (parse-repaired, rung-3-grounded, NLI-below-threshold, grounding
failure) — is a headline pilot metric: *"a high rate is an extraction-quality
signal, never grounds for quietly lowering the threshold"* (architecture F9).

### P5.7 `ingest/oblparse.py` + the `graft/core/obligations.py` learned-mode hook

G7. The core module keeps only the dispatch (`mode="learned"` → a callable
registered by `graft.ingest`), so `graft.core` still imports no ML library and
the routing table of plan §4.4 stays a module boundary.

### P5.8 `ingest/pipeline.py`

The per-turn orchestration: skip-if-ingested (G4) → context (P5.2) → extract
(P5.3) → ground (P5.4) → **store everything** (audit layer) → verify (P5.5,
second pass) → gate (P5.6) → eligibility written to the log. Stage-sequential
across the slice, not per turn, for F7. One `ledger.stage()` per stage;
`metrics.jsonl` rows per turn.

### P5.9 `scripts/phase5_pilot.py`

Runs G10's pilot, emits the report (every G-gap's measured number), the G8
sizing memo, and the audit worksheets (span, NLI, obligation slots) as CSVs a
human fills — the same pattern as the 2.5 annotate CLI, minus the timing.

### P5.10 `graft/tests/`

See §5. The write path is testable **without a GPU**: a `ReplayExtractor` that
serves the spike's recorded extractions lets the pipeline, grounding, gating,
idempotence and replay tests run on the stock CI/Kaggle image; only the bakeoff
and the live pilot need the model.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 0 | P5.0 Gate-0 draft + Tier-B freeze | the contract exists with 9/10 items filled; item 8 says what it waits on |
| 1 | P5.1 corpus adapter | the 2.5 sample's turns re-derive byte-identically from the pinned corpus |
| 2 | P5.8 pipeline + P5.4 grounder, on the `ReplayExtractor` | idempotence: second ingestion changes no digest; every stored span resolves on replay |
| 3 | P5.3 extractor + **G2 bakeoff** on the calibration slice | the pick rule's table is produced; winner frozen into §6 |
| 4 | P5.2 summary | context assembly matches the Mem0 recipe at the declared cadence |
| 5 | P5.5 NLI + P5.6 gate | quarantine rate computed on replayed spike data; stage-sequential asserted |
| 6 | P5.7 obligation parser | slot audit runs against the 50 hand labels; unbounded-constraint rate reported |
| 7 | P5.9 pilot end-to-end + audits + sizing memo | every §5 criterion has its measured number |

Step 2 before step 3 is deliberate: the write path is the part Phase 6 depends
on, and it must not wait on GPU work.

---

## 5. Exit criteria

**The record is sound**
1. Every stored assertion's spans resolve against the raw turns on a fresh
   replay of the event log (architecture's criterion, as a test).
2. Idempotence: re-ingesting the pilot changes neither `EventLog.digest()` nor
   the graph digest; the skip count equals the turn count (G4).
3. Tier B is frozen; both `SCHEMA_VERSION`s recorded; round-trip property tests
   still green (G9).
4. Provenance is non-empty by construction end-to-end: no assertion without
   spans, no eligibility without a stored verdict (Phase-0 §2.10 honoured on
   real data).

**Extraction quality is measured, not asserted**
5. Parse-failure rate on the pilot **< 2%** with the frozen G2 winner, and the
   bakeoff table (all three candidates, all three rule stages) is in the report.
6. Manual span-support precision **≥ 0.90** on the 50-assertion audit (G1), by
   the declared protocol, worksheet preserved.
7. Grounding: per-rung counts reported; all rung-3 spans audited; mis-bound
   rate reported (G5).
8. The quarantine rate is reported, total and by cause, and appears in the
   report beside the sentence that it is a quality signal, not a knob (F9).
9. NLI audit: agreement with ~50 hand labels at the frozen `tau_nli` reported;
   the score distribution saved so the audit is re-inspectable (G6). `tau_nli`
   unchanged.
10. Mention and assertion yields per turn reported against the spike's 0.59 —
    the number D1 volume planning uses (G2).

**The parser and the flags**
11. `parse(mode="learned")` works end-to-end; slot-level scores on the 50-label
    audit reported per slot (precision/recall/F1 for optional, accuracy for
    boolean — Phase-1 §5.5's corrected metric); unbounded-constraint rate
    reported (G7).
12. Relative dates widen to natural-granularity half-open intervals; a test
    covers "last May", "in 2023", "yesterday" against a fixed `question_date`
    (Phase-0 §2.5).

**Discipline**
13. `graft.core` and `graft.synth` still import no ML library; `graft.ingest`
    is the only new package on the allow side, asserted per package (P3.0's
    pattern).
14. `llm_calls` / `llm_tokens_in` / `llm_tokens_out` metered inside the
    extractor wrapper, never by callers; the pilot report carries totals and
    per-turn medians (G10 of Phase 0's ledger, finally exercised).
15. Two pilot runs on this machine produce identical log digests; the manifest
    carries model revision + prompt SHA + decoding config; the cross-machine
    limitation is stated in the report (G11).
16. Throughput measured; the sizing memo exists with all three scope options
    costed on both hardware targets (G8).

---

## 6. Decisions to lock before writing code — **ADOPTED 14 Aug 2026**

*The build adopted all fourteen rows as recommended. Four carry amendments made
**by measurement**, each recorded in `PHASE5_DECISIONS.md` with what was
measured and when — rows 2, 3 and 4 below, plus the ruling that withdrew
candidate C before the run (§1.1). Every amendment landed while
`pins.EXTRACTOR` was still `None`, i.e. before the value it influences was
frozen.*

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | Gate-0 threshold (item in the contract) | manual span-support precision ≥ **0.90** / 50-assertion audit, protocol as G1 declares | Phase-6 training data quality floor moves; ceiling 1 reads differently |
| 2 | Extractor config | **FROZEN 14 Aug 2026 on candidate B** — Qwen2.5-3B-Instruct @ `aa8e7253`, bf16, greedy, grammar-constrained, no repair policy. The predeclared rule decided it at stage 1: B 1.7% parse failure against A's 23.3%, sole survivor of the < 2% filter. **Candidate C was withdrawn by ruling before the run** (the extractor is 3B, §1.1 of the decisions), and stays in the declared table carrying a `withdrawn` field so the candidate set reads as the one that was declared | re-run the pilot; every downstream extraction artefact regenerates |
| 3 | Summary cadence | synchronous, `s = 10` turns, ≤ 512 tokens **enforced at generation** (`max_new_tokens=SUMMARY_MAX_TOKENS`; the word-count backstop alone admits ~725 tokens — words ≤ tokens, measured 13 Aug 2026), cached write-through, not logged (G3) | context recipe changes → extraction outputs change → re-pilot |
| 4 | Context window | `m = 10` previous turns + summary (Mem0's recipe, qualified evidence), **window turns head-clipped to `CONTEXT_CLIP_CHARS = 600` chars** — a declared adaptation, 13 Aug 2026: Mem0's *m* = 10 is over short chat messages, and unclipped LongMemEval essay turns measured 5,017 input tokens/turn, overflowing the 8 GB card (9,515 MB peak, 12× throughput collapse). The current turn is never clipped; grounding resolves against full text | same as 3 |
| 5 | Grounding ladder | exact → normalised-exact → fuzzy ≥ 0.85 with word-boundary snapping → drop-and-count; rung reported per span (G5) | offsets shift under any change → Phase-6 features and D1 items move |
| 6 | NLI pin | one DeBERTa-v3-class cross-encoder, id + revision frozen; premise = grounded spans only; audit-not-retune at `tau_nli = 0.8` (G6) | eligibility flips across the corpus; Phase 6 input changes |
| 7 | Support policy | `strict` (frozen since Phase 0); quarantine causes broken out (F9) | the active graph's contents change — Phase 6 onward re-runs |
| 8 | Idempotence | skip-and-count at `turn_id`; crash-resume = re-run (G4) | duplicate evidence in the log is unrecoverable without a migration |
| 9 | Extraction schema | G9's, with `kind ∈ {claim, value, event, time}` frozen and cross-turn quotes via `turn_offset` | Tier-B thaw + log migration |
| 10 | Pilot | G10's definition, verbatim | every §5 number loses its denominator |
| 11 | Corpus scope for Gates 1/4 | **not decided here** — Gate-0 item 9, taken with G8's sizing memo; knowledge-update evidence sessions in every candidate | deciding it here, without the memo, is exactly the unforced guess this phase exists to avoid |
| 12 | Reproducibility claim | per-machine digest identity asserted; cross-machine = config-identity only, manifest-stamped (G11) | a false promise in the write-up, found by a reviewer with two machines |
| 13 | ML boundary | `graft.ingest` may import ML; core/synth boundaries unchanged, per-package test | Phases 0–4 stop running on a bare interpreter |
| 14 | Obligation-parser audit set | ~50 questions, hand-labelled, all six slots; scored by `slot_level_scores`; reported wherever coverage is (F2) | the parser ships unaudited behind a paper claim |

---

## 7. Explicitly not in Phase 5

No entity resolution, no D1–D4, no encoders, no training, no retrieval, no
answerability gate, no reader, no `tau_nli` retuning, no corpus-scope decision,
no change to `H`, `U`, `R`, masks, `d(s)`, the lattice or the evaluator, and no
second notion of "supported" anywhere — the flag the NLI writes is the flag `H`
reads, one implementation, same as every other shared quantity in this project.

---

## 8. What Phase 6 will need from this, verbatim

* The event log with **eligible** assertions carrying resolved multi-span
  provenance, four flags and eligibility verdicts — Stage B's entire input.
* **Mentions, as `mention.add` log events** (`span_id`, `turn_id`, grounded
  text, rung), read back with `graft.ingest.pipeline.mentions_of`. Not graph
  state — no `Mention` node exists until D1 decides — and deliberately outside
  `GRAPH_OPS`, so replay ignores them. *(Added 14 Aug 2026: before this, a
  mention survived only as a bare `span.add` plus a per-turn count,
  indistinguishable from a quote span, and D1's items were unrecoverable from
  the permanent record.)*
* `GATE0_CONTRACT.md` signed (item 8 filled from the human pass) — Gate 1's
  entry condition.
* The D1/D2 item-derivation path (the 2.5 tooling, upgraded to read Phase-5
  extraction) with guidelines v1 seeded from the spike's 21 flagged items.
* The measured yields (mentions/turn, assertions/turn, quarantine rate) that
  size Gate-1's annotation batches under item 8's items/hour.
* The frozen extractor + prompt registry, because Phase 6's LLM-prompted-linking
  baseline (architecture §6.4) must run against the *same* extraction it
  competes with.
