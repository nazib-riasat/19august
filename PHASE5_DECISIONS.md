# Phase 5 — what the build decided, and what it measured

Date: 13 August 2026
Parent: `GRAFT_PHASE5_BUILD.md` (G1–G11, §6) · `GRAFT_EXECUTION_ARCHITECTURE_v1.md`
(Phase 5, fixes F2/F7/F9) · `GRAFT_RESEARCH_PLAN_v1.md` v1.2 §3.1, §4.4, §7 ·
`PHASE2_5_DECISIONS.md` (the spike's measurements) · `GATE0_CONTRACT.md` (drafted
here, as P5.0)
Status: **code built and green (805 tests, 14 Aug 2026); the Gate-0 contract
drafted with nine of ten items filled; the G2 bakeoff has run and decision 2 is
frozen on candidate B (§2.1b); the live pilot has run on 248 turns (§2.2a).
Every machine-measurable exit criterion is met. What remains is four human
audits and Gate-0 item 8 — none of them machine-fillable.** A post-build audit (13–14 Aug 2026, §7) confirmed ten
reported defects and nine fresh ones — including two blockers in the bakeoff
harness itself — and every confirmed finding is fixed and regression-tested;
the bakeoff's first run was aborted and its instrument corrected before
anything froze (§1.6).

Same convention as the other DECISIONS files: **[EVIDENCE]** = named paper, venue
stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = judgment
made here.

---

## 1. The §6 table, as adopted

`GRAFT_PHASE5_BUILD.md` §6 was **unsigned** when the build started. The build
adopted all fourteen decisions as recommended, with **one ruling and four
implementation-level resolutions** recorded below. Everything else is the plan's
own text and is not restated here — restating a decision is how it acquires a
second version.

### 1.1 The ruling — candidate C is withdrawn, and the 7B question stays open

**Ruled by the project owner, 13 August 2026: the extractor is
Qwen2.5-3B-Instruct.** Bakeoff candidate C (Qwen2.5-7B-Instruct, 4-bit) is
withdrawn before the run, extending the Phase-2.5 amendment
(`PHASE2_5_DECISIONS.md` §1) from the spike to the whole project.

Three consequences, recorded rather than glossed:

1. **Whether the architecture's 4-bit 7B fits in 8 GB remains unmeasured.** That
   was fix F7's original question and it is now a permanent open item, not a
   pending one. The architecture's extractor row is superseded for this project;
   the architecture itself is not edited (the protected-docs rule).
2. **The bakeoff decides between two parse-failure strategies on one model**,
   which is the question that actually matters here — the spike's 15.5% silent
   turn loss is the defect, and repair-retry (A) and grammar-constrained decoding
   (B) are the two ways to fix it. It is a *narrower* bakeoff than G2 declared and
   the artefact says so.
3. **C stays in the declared candidate table**, carrying a `withdrawn` field and
   unselectable by any code path. Deleting a predeclared candidate after the fact
   would leave a table that reads as though only two were ever considered — which
   is the shape of a post-hoc decision rule, the thing fix F12 exists to prevent.

### 1.2 Four implementation-level resolutions the plan left open

**(a) `turn.add` is appended *last*, after that turn's spans and assertions.**
G4 says "a crashed run is resumed by re-running it", and turn-level skip-and-count
is decision 8 — but the two together have an ordering hazard the plan does not
name. With the turn written first, a crash between the turn and its assertions
leaves a turn in the log with no assertions, the re-run **skips** it, and that
evidence is lost permanently with nothing to notice. Written last, the same crash
leaves spans and assertions whose turn is absent — which every reader already
fails closed on, since `H`'s scope sub-check refuses a provenance chain it cannot
resolve — and the re-run reprocesses the turn and repairs it. Content-derived ids
make the repair idempotent at the graph level: the replayed snapshot is
identical, and only the log carries the few duplicated lines that record a crash.

**The trade is stated because it is real**: a mid-crash log can hold an orphaned
span. It fails closed; the alternative fails silent.

**(b) Pass 2 works from the log, not from a run-local list.** The same hazard,
one level up: a run that dies between extraction and verification has stored
assertions with no verdict, and a re-run skips every turn. So "unverified" is
defined as *stored with no `assertion.set_eligibility` event*, whichever run
stored it. Without this, crash-resume leaves assertions quarantined-by-omission —
a silent recall loss, again with nothing to notice.

**(c) The op vocabulary is completed with `assertion.set_flags`.** G4 says the
assertion write path — record, four flags, eligibility — is completed here, "not
invented". The verifier writes `entailed_by_span` and `entailed_score`; the gate
writes eligibility; they are **two events**, because architecture fix F9 is
explicit that the verifier never blocks storage and the *gate* decides
eligibility. One combined event would erase that authority boundary in the one
place it is permanently recorded. `DictGraphSnapshot.set_flags` takes *named*
changes rather than a whole `AssertionFlags`, because the four flags are
independent by design (plan §3.1) and are written by different components at
different times — a whole-object write would let one component silently reset
another's.

**(d) The quarantine "causes" are two different kinds of thing, and the report
says so.** P5.6 asks for the rate broken out by cause — *parse-repaired,
rung-3-grounded, NLI-below-threshold, grounding failure*. Under the strict policy
exactly one of those, **NLI-below-threshold**, is a *gating cause*.
*Parse-repaired* and *rung-3-grounded* are **provenance attributes**, reported
cross-tabulated against the verdict, which is strictly more informative: "of the
quarantined assertions, how many came from a repaired parse" is a testable
hypothesis about where extraction error concentrates. *Grounding failure* is a
**drop**, not a quarantine: such an assertion is never stored, because
`Assertion` refuses to exist without spans, so counting it inside the quarantine
denominator would inflate a rate that is supposed to measure the *gate* and would
double-count a loss the grounding row already reports. It is its own row.

### 1.3 Two decisions the plan states but that are easy to implement wrongly

**`scope` is metadata and is never asked of the model.** G7 lists six slots for
the learned parser. `Obligations.scope` is a tuple of `conv_id`s that `H`'s
sub-check 5 rejects evidence against — so a hallucinated id would not produce a
wrong answer, it would produce a **silently over-restricted proof search**,
rejecting correct evidence for being outside a scope nobody asked for. The parser
therefore fills five slots and takes `scope` from the question's own binding, and
the audit reports it as metadata-derived rather than as parser quality.

**Relative dates are resolved in code, not by the model.** The model emits the
*phrase*; `graft/ingest/timeexpr.py` widens it to a half-open interval at natural
granularity. Three conventions are declared there because the language does not
settle them: `"last <month>"` is the most recent occurrence **strictly before**
the question's own month; `"last week"` is the previous **ISO calendar week**;
an open-ended expression (`"since 2022"`) resolves to a genuinely **unbounded**
interval — which is exactly what the unbounded-constraint rate counts.

### 1.4 The G5 grounding repair, and why it is measured rather than ruled

Decision 5 says "fuzzy window with word-boundary snapping … unless the quote
itself starts or ends mid-word". **That rule cannot be applied as written**:
whether a quote starts mid-word is not knowable from the quote alone — `"arden"`
is a legitimate mid-word quote of *Garden*. So the ladder *generates* the
boundary-snapped candidate windows and keeps whichever scores highest against the
quote the model actually wrote, ties going to the unsnapped window.

That is the same decision, made measurable: it fixes the spike's one measured
mis-bound span (dropping an orphaned leading `"s "` raises the ratio) and cannot
damage a genuinely mid-word quote (snapping would lower it). Both directions are
regression tests.

The fuzzy search also changed: the spike swept windows at `step = m // 4`, which
can miss the true window outright and is the *other* reason its span landed a
word off. The ladder sweeps coarse at `m // 8` and then refines at step 1 in the
winner's neighbourhood, which cannot miss by more than the coarse step and then
covers exactly that.

### 1.5 One dependency conflict, resolved the boring way

`xgrammar` (candidate B) declares `transformers<5`; this project pins
`transformers==5.15.0` and `pip` prints a dependency-conflict warning. Verified on
this machine: xgrammar imports, `xgrammar.contrib.hf.LogitsProcessor` resolves,
and `GrammarCompiler.compile_json_schema` compiles the Phase-5 extraction schema
against a Qwen2.5 tokenizer. **The pin is kept and the conflict is documented**
(`requirements-ingest.txt`), because letting pip downgrade `transformers` would
silently change the environment *the other candidate runs in*, and a bakeoff whose
harness differs per candidate measures the harness.

The verifier is loaded through `transformers` directly rather than through
`sentence-transformers`: a cross-encoder checkpoint *is* a sequence-classification
model, and a second library to call the same weights is a dependency bought for a
convenience wrapper.

### 1.6 The aborted bakeoff run, and the four instrument corrections

**The first G2 run (13 Aug 2026) was aborted mid-flight and nothing was frozen
from it** — `pins.EXTRACTOR` was `None` throughout, so these are instrument
corrections under the predeclaration discipline, not post-hoc changes to a
decision. The aborted artefact is preserved as
`artefacts/phase5_bakeoff_AB.json`; candidate A's row from it is quoted below
**as instrument diagnosis, never as candidate evidence**.

What the aborted run measured, and what each number forced:

| Measured | Value | Correction forced |
|---|---|---|
| input tokens/turn at *m* = 10, unclipped | **5,017** | context-window turns are head-clipped to `CONTEXT_CLIP_CHARS = 600` chars (decision 4's declared adaptation — Mem0's *m* = 10 is over short chat messages, not LongMemEval essays). The current turn is never clipped; grounding always resolves against full text |
| peak VRAM / throughput | **9,515 MB on an 8,151 MB card**; 18.4 → **1.6 tok/s** (12×) | same fix — the driver spilled to system memory |
| candidate A's token decomposition | 12,915 out-tokens over 78 generations, ≈ **15 × 600 + 63 × 62** | `max_new_tokens` 600 → **1024**. **Both halves of this row were later falsified and it is kept as the inference it was, not as a measurement** (§2.1, §2.1b): the identity is approximate, not exact — 15 × 600 + 63 × 62 = 12,906, and the 62 is a rounded mean (3,915 / 63 = 62.14) — and the hypothesis it carried, that A's failures were token-cap truncations, is **false**: at 1,024 A's rate was unchanged at 25% with **14 malformed against 1 truncated**, and at 2,048 it was 23.3% with **14 malformed and 0 truncated**. The raise was still right for a different, stated reason (the cap is a runaway guard, never a content limit — §2.1a), and the `truncated_at_token_cap` counter is what turned the inference into a measurement instead of leaving it standing |
| candidate B's 3-turn check | 1 parse failure, cause `truncated_at_token_cap` | G2's zero-by-construction parse-failure claim for B **falsified and corrected** in the plan, `pins.py` and `extractor.py`: a grammar guarantees prefix validity, not that the object closes within the budget |

And two harness defects the audit (§7) found in the same run, fixed with the
above: the flat 60-turn slice windowed **ten different users' conversations
into one stream** (55/60 prompts carried foreign context; 71% of all context
turns were foreign; the held-back turns sat out of corpus order; cross-turn
quotes could ground into a foreign user's turn — inflating stage 2's own
metric), and the summary block production prompts carry was absent. The slice
is now built by `graft.ingest.bakeoff.calibration_slice` — grouped per
conversation, corpus-ordered, held-back turns at their in-session positions —
and the harness runs the production recipe (window within conversation +
rolling summary via the candidate's own model), with model load excluded from
the timed loop. **The rule is unchanged.** One slice property is declared in
the artefact rather than repaired: each session contributes its *windowed*
turns, not the full session stream, so summaries are built over that windowed
stream.

**Consequence for candidate B, found while wiring the summary in:** the grammar
processor attached to *every* generation, including `complete()` — the path the
summary and the obligation parser use — so a B win would have forced summaries
and obligation replies into extraction-schema JSON, silently, in production.
The grammar is now scoped to extraction calls only (`_decoding_kwargs`,
regression-tested). No reviewer had caught this one.

---

## 2. What the build measured

*(GPU runs. Each row states whether it is measured, pending or withdrawn.)*

### 2.1 The extractor bakeoff (G2, decision 2)

**Status: the corrected run completed 14 Aug 2026
(`artefacts/phase5_bakeoff.json`). The rule fired and returned
`no_survivor` — neither candidate clears the 2% ceiling, so decision 2 stays
unfrozen and `pins.EXTRACTOR` stays `None`.** The aborted run's artefact is
`artefacts/phase5_bakeoff_AB.json`; its numbers diagnose the instrument (§1.6)
and are never quoted as candidate evidence.

**The measured table**, 60-turn slice, same prompt, same context recipe, same
rolling summary for both.

**One property of the slice, stated because the artefact records it and a
reader would otherwise assume otherwise:** `summary_calls = 0` for *both*
candidates. The summary is wired into the harness and identical for both, but
the slice's largest conversation group is exactly 10 turns and the refresh
point is `s = 10`, so it never fires. The comparison is therefore fair and the
recipe is production's, but **the bakeoff did not exercise a non-empty summary
block** — the live pilot did (21 calls, §2.2a). What stays untested at
selection time is how either candidate behaves with a summary in the prompt;
the risk is bounded by the grammar being scoped to extraction calls only
(§1.6), so B's constraint never applies to summary generation. Recorded as a
limitation of the slice, not repaired: re-cutting the slice to reach a refresh
point would change the predeclared calibration set after seeing results.

| | **A** — repair-retry | **B** — grammar-constrained |
|---|---|---|
| parse failure | **25.0%** (15/60) | **3.3%** (2/60) |
| — malformed JSON | **14** | **0** |
| — truncated at the token cap | 1 | 2 |
| repairs spent | 23 | 0 (no repair policy) |
| grounded assertions | 61 | **73** |
| assertions extracted → grounded | 64 → 61 (**95.3%**) | 81 → 73 (**90.1%**) |
| mentions / turn | 1.08 | 1.13 |
| wall clock | 1,225 s | **779 s** |
| **grounded assertions / minute** (stage 2) | 2.99 | **5.62** |
| throughput | 15.6 tok/s | 17.0 tok/s |
| peak VRAM | 6,689 MB | 6,465 MB |

**Four findings, in order of how much they change:**

1. **The verdict is `no_survivor`, and the rule was honoured rather than
   adjusted.** B missed the ceiling by **two turns**. That was recorded, not
   repaired — see §2.1a for what the project owner then ruled and on what
   grounds.
2. **The truncation hypothesis of §1.6 was wrong for candidate A, and the cause
   counter is what showed it.** The aborted run's arithmetic suggested A's
   failures were all token-cap truncations; at 1,024 tokens A's rate is
   *unchanged* at 25%, and the breakdown is **14 malformed against 1 truncated**.
   A 3B model with a repair reprompt does not reliably emit valid JSON on these
   prompts. The hypothesis was stated as an inference and it did not survive
   measurement, which is the outcome the counter was added for.
3. **B's guarantee holds exactly as narrowly as §1.6 restated it.** Zero
   malformed outputs in 60 turns; both of its failures are truncations. "Never
   malformed" is measured true; "parse failure 0 by construction" stays false.
4. **B trades grounding precision for volume, and stage 3 is where that would be
   settled.** B extracts more (81 vs 64) but grounds a *lower* share of it
   (90.1% vs 95.3%) — the "constrained decoding can degrade content quality"
   risk G2's own table named, now a number rather than a worry. The tie-break
   stage is a human span sub-audit, and this is what it would be adjudicating.

Two smaller measurements worth carrying forward:

* **mentions/turn is 1.08–1.13, roughly double the spike's 0.59.** D1 volume
  planning in Gate-0 item 8 was sized on 0.59; the corrected context recipe
  nearly doubles the yield, which moves the arithmetic in the favourable
  direction. The spike's figure should not be quoted as current.
* **The torch-native mask penalty did not materialise.** §1.5 warned that
  candidate B's Triton-free bitmask would count against it on stage 2's
  throughput metric; B was in fact *faster* than A (779 s vs 1,225 s), because A
  spends 23 repair generations. The caveat is withdrawn as immaterial here.

The rule, declared before the run and applied by `graft.ingest.bakeoff.decide`:
parse-failure rate **< 2%** (hard filter) → highest **grounded assertions per
minute** among survivors → a 20-assertion span sub-audit as tie-break, which is a
**human** step and is returned as a `tie` verdict rather than resolved by machine.

### 2.1a The token-budget raise, and why it is not tuning

**Ruled by the project owner, 14 August 2026, after the `no_survivor` verdict:
raise `max_new_tokens` and re-run both candidates.** The rule itself is
untouched — the 2% ceiling, the throughput metric and the human tie-break all
stand. What changed is one instrument constant, and the honesty of that change
rests on three things being true, all of which are checkable here:

1. **The argument is the reason, not the outcome.** The principle
   `max_new_tokens` answers is *the cap is a runaway guard, never a content
   limit*. It exists to stop a repetition loop; at any value where it binds on
   real extractions it is silently deleting evidence — and a truncated turn
   yields **nothing**, indistinguishable downstream from a turn with nothing to
   say, which is the precise failure this phase exists to remove. At 1,024 it
   demonstrably binds on real content (2/60). **2048** is ~10× the measured mean
   output (192–220 tokens) and covers ~20 assertions at the schema's ~70 tokens
   each — denser than any turn in the pinned corpus — while a repetition loop
   still terminates.
2. **The expected effect was written down before the re-run.** B's failures are
   *all* truncations, so B was predicted to fall to ~0%; A's are 14 malformed
   against 1 truncated, so A was predicted to stay near 25%. **The raise was
   expected to change the verdict, and saying so in advance is what separates it
   from tuning until a candidate passes.** The prediction is in
   `graft/ingest/pins.py` beside the constant, timestamped, and the re-run either
   confirms it or does not.
3. **Nothing was frozen when it changed.** `pins.EXTRACTOR` was `None`
   throughout, exactly as in §1.6.

**What this does not license.** A third raise, argued from a third residual,
would be tuning — the reason above is a *bound*, and once the cap provably does
not bind on legitimate content there is no further reason available. If the
re-run leaves either candidate failing on truncations at 2,048, that is evidence
about the extractor or the schema, not about the budget.

### 2.1b The re-run at 2,048 — decision 2 is frozen on candidate B

**Both predictions held**, which is what the predeclaration in §2.1a was for:

| | prediction | measured at 2,048 |
|---|---|---|
| A | "stays near 25%" | **23.3%** (14/60) — causes now **14 malformed, 0 truncated** |
| B | "falls to ~0%" | **1.7%** (1/60) — cause: 1 truncation, **0 malformed** |

| | **A** | **B** |
|---|---|---|
| parse failure | 23.3% | **1.7%** ✅ clears the 2% ceiling |
| grounded assertions | 61 | **74** |
| extracted → grounded | 64 → 61 (95.3%) | 82 → 74 (90.2%) |
| grounded / minute | 2.52 | **4.99** |
| wall clock | 1,452 s | **890 s** |
| throughput | 13.3 tok/s | 16.0 tok/s |
| peak VRAM | 7,407 MB | 6,465 MB |

**Verdict: `winner`, candidate B — decided by stage 1, not stage 2.** B was the
*sole* survivor of the parse-failure filter, so the throughput comparison had
nothing to compare against. `decide()` reported "stage 2" for the
single-survivor case; it was corrected, because the artefact is the permanent
record of *how* a frozen decision was reached and "decided on throughput" would
have been false. Regression-tested
(`test_a_sole_survivor_is_decided_by_stage_one_not_stage_two`).

**The correction landed *after* the run, and the frozen artefact was not
regenerated** — `artefacts/phase5_bakeoff.json` therefore still carries
`"decided_by": "stage 2 (grounded assertions per minute)"`. That string is
stale and this paragraph is the correction of record. It is left rather than
rewritten because an artefact edited by hand after the fact is worth less than
an artefact that says what the run said, and the *decision* is unaffected:
with one survivor, both readings select B. Re-running the bakeoff to refresh
one string would cost ~2.5 GPU-hours and change nothing else. *(Found by the
second audit pass, 14 Aug 2026 — the doc had described the artefact rather
than reading it.)*

**Decision 2 is frozen**: `pins.EXTRACTOR = dict(EXTRACTOR_CANDIDATES["B"])`,
transcribed by hand from `artefacts/phase5_bakeoff.json` — Qwen2.5-3B-Instruct at
revision `aa8e7253`, bf16, greedy, **grammar-constrained decoding**, no repair
policy. Exit criterion 5 is met: 1.7% < 2%.

**Three things carried forward rather than smoothed over:**

1. **B's one residual failure is a truncation at 2,048 tokens.** By §2.1a's own
   terms that is now evidence about the extractor or the schema — a single
   pathological turn — and explicitly *not* grounds for a third raise.
2. **A's failures are 14 malformed and 0 truncated.** At a cap that cannot bind,
   the 3B model with a repair reprompt still fails to emit valid JSON on 23% of
   turns. That is the clean form of the finding G2 was built to get: the spike's
   15.5% was not a token-budget artefact, and prompting alone does not fix it.
3. **B grounds a lower share of what it extracts** — 90.2% against A's 95.3%.
   G2's own table named this risk ("constrained decoding can degrade content
   quality"); it is now a measured 5-point gap and it is what the human span
   sub-audit should be read against. The winner was *not* chosen on grounding
   precision, and the write-up must not imply it was.

**A conflict this creates for the pilot, stated now rather than discovered
later**: the frozen extractor has no repair policy, so `MAX_REPAIRS` and the
repair prompt are dead code on the production path. They stay — candidate A
remains a declared candidate and the repair path is regression-tested — but the
pilot's `repairs` column will read 0 by construction, not by luck.

Two properties of the rule worth stating because they were choices:

* **stage 2 is a rate, not a count.** The candidates differ in throughput by
  construction, so "most assertions" would reward whichever candidate is slowest
  to be honest about its failures; assertions per minute is also the number the
  sizing memo needs.
* **a candidate that fails the ceiling is out however fast it is**, and *every*
  candidate failing it is a Gate-0 finding about extraction feasibility on this
  machine — not a reason to raise the ceiling after seeing the numbers.

### 2.2 The pilot (G10, decision 10)

**Status: both have run. The live pilot completed 14 Aug 2026 on the frozen
candidate-B extractor — §2.2a. The replay pilot below stays in the record as the
plumbing check it always was.**

`artefacts/phase5_pilot_replay.json` exercises the write path end to end on the
spike's recorded extractions and a fixed-score verifier. It measures **plumbing,
not quality**, and the artefact carries that sentence as its first limitation.
Every number below reproduced identically across the audit's fixes; the
regeneration additionally exercises the **fresh-log determinism leg** (two full
runs into two logs → identical log and graph digests — the check `--twice`
previously could not perform, §7) and the audit-draw exclusion (the span and
NLI worksheets are empty in replay mode *by construction*: the replay slice IS
the calibration slice). What it establishes:

| Quantity | Value | Reading |
|---|---|---|
| turns processed | 58 | the spike's windowed slice |
| assertions stored | 78 | every one with resolvable spans on a fresh replay |
| spans | 112 | 78 assertion spans + 34 mention spans. **No multi-span assertion**, and none was possible: the spike's recording is single-quote per assertion, so this run could not exercise the shape whatever the pipeline does. *(Corrected 14 Aug 2026 — this row previously claimed one cross-turn multi-span assertion. It was measured wrong: the replay log holds 78 assertions, all with exactly one span. See the note below, which is the finding that correction exposed.)* |
| mentions/turn | **0.586** | independently reproduces the spike's 0.59, through a different code path |
| assertions/turn | 1.345 | 78/58, as recorded |
| re-ingestion | log digest **unchanged**, graph digest **unchanged**, 58/58 turns skipped | exit criterion 2, measured |
| grounding rungs | 112 exact | **an artefact of the recording, not a measurement**: the spike stored the *recovered* quote text, so every quote grounds exactly by construction. The live pilot is where the rung distribution is measured |

**Multi-span cross-turn provenance has never been produced by a real run, and
that is a finding rather than a footnote.** Plan §3.1 requires that "a claim
assembled across turns records every supporting span, not one", G9 built the
schema for it (`turn_offset`), and the write path handles it —
`test_cross_turn_provenance_is_stored_as_two_spans_on_two_turns` exercises it
end to end. But measured over **both** real logs: the replay pilot has 0 of 78
(structurally impossible, above) and the **live pilot has 0 of 222** — candidate
B emitted no multi-quote assertion in 248 turns. So the capability is *tested*
and *unexercised by data*, which are different claims and only the first is
currently true of a run.

Three consequences, none of them "fix the extractor":

* **the write-up may say the schema and pipeline support multi-span
  provenance; it may not say the corpus produced any**, and no ceiling-1 or
  Phase-6 number may be attributed to it;
* the prompt asks for extra quotes only when "the fact genuinely needs support
  from an earlier turn" — a deliberately conservative instruction, and 0/222 is
  consistent with it being followed. Whether the *right* rate is 0% is a
  question for the span-support audit, not a defect to patch by loosening the
  prompt;
* **Phase 6 should not assume multi-span input exists** when sizing D1/D2
  items, and should re-check this number if the corpus scope grows.

The corpus numbers the sizing memo rests on, computed from the pinned file:
**246,930 turns over 500 questions**; **10,960 turns in evidence sessions alone**;
54 sessions per question on average.

### 2.2a The live pilot — measured, 14 Aug 2026

`artefacts/phase5_pilot.json`, ingestion fingerprint `bf176a37cfb47c36`.
G10's declared object exactly: the Phase-2.5 sample's 10 questions with their
**full** evidence sessions — **248 turns**, no windowing — through the frozen
candidate-B extractor, plus the 50-assertion span audit, the ~50-pair NLI audit
and the ~50-question obligation set, all drawn by seeded sample and all
**disjoint from the bakeoff's calibration slice**.

**The record is sound (criteria 1–4, 15).**

| | Measured |
|---|---|
| turns processed | 248 |
| assertions stored | 222 (**0.895/turn**) |
| spans | 397 |
| mentions | 187 (**0.754/turn**) |
| **idempotence** — re-ingest the same log | log digest **unchanged**, graph digest **unchanged**, 248/248 skipped |
| **determinism** — the same work twice into two *fresh* logs | log digests **identical**, graph digests **identical** |

The determinism leg is the one that matters and the one that could not be
checked before: 248 turns of grammar-constrained bf16 generation, run twice from
cold, produced byte-identical `(seq, op, payload)` streams. **Per-machine**
determinism (G11) is therefore measured, not asserted. Cross-machine identity
remains explicitly not promised; the fingerprint above is what must match.

**Extraction quality (criteria 5, 7, 10).**

| | Measured | Reading |
|---|---|---|
| parse failure | **1.6%** (4/248) | consistent with the bakeoff's 1.7%; all 4 are truncations at 2,048 |
| grounding rungs | **exact 375 · normalised 4 · fuzzy 30** | **91.7%** land on rung 1, over **409 groundings** — not over the 397 *distinct* spans in the graph: a quote and a mention can resolve to the same offsets and are then one span but two groundings. 30 fuzzy spans are the G5 audit set, **all** of them on the worksheet |
| assertions dropped, ungrounded | **108** | **333 extracted → 222 stored, 108 dropped, 3 duplicates within a turn** — the three dispositions now account for every extraction (the duplicate counter was added 14 Aug after the second audit found 333 − 222 − 108 = 3 vanishing through an uncounted `continue`). **A third of extracted assertions cite something not in their turn.** The single largest loss in Stage A, and it is a *drop*, not a quarantine — never stored, never counted in the gate's denominator |
| mentions/turn | 0.754 | between the spike's 0.59 and the bakeoff slice's 1.13 |
| throughput | **135.7 turns/hour** end to end | the sizing memo's input |

**The support gate (criterion 8).**

| | Measured |
|---|---|
| eligible | **151 (68.0%)** |
| **quarantined** | **71 (32.0%)** |
| gating cause | `nli_below_threshold` × 71 — the only one, as the strict policy implies |
| cross-tab by worst rung | exact 131/59 · fuzzy 18/11 · normalised 2/1 (eligible/quarantined) |
| cross-tab by repaired extraction | 0/0 — **zero by construction**: candidate B has no repair policy (§2.1b) |

**A 32% quarantine rate is the headline extraction-quality signal, and F9's
sentence ships beside it**: it is never grounds for lowering `tau_nli`. Two
readings are available and the audit is what separates them — either the
extractor is producing claims its own spans do not carry, or the NLI model is
strict at 0.8 on span-level premises. **The ~50-pair hand-labelled agreement
audit is exactly the instrument for telling those apart, and it is unfilled.**
No claim about which it is belongs in the write-up before that worksheet is done.

Worth stating plainly: the fuzzy-rung quarantine rate (11/29 = 38%) is *higher*
than the exact-rung rate (59/190 = 31%), in the direction one would expect if
weaker grounding produced weaker entailment — but on 29 spans that difference is
not a finding, and it is recorded as a number to re-read at corpus scale, not as
evidence.

**The obligation parser (criteria 11, 12).**

| | Measured |
|---|---|
| questions parsed | 48 |
| slot parse failures | **0** |
| questions carrying a time expression | 27.1% (13/48) |
| **time expressions the resolver could not parse** | **69.2%** (9/13) |
| unbounded-constraint rate | **2.1%** (1/48) |

**The 69% unresolved rate is the finding here, and it is about
`graft.ingest.timeexpr`'s vocabulary, not about the model** — the parser emitted
a phrase, and the declared vocabulary did not cover it. That is precisely why an
out-of-vocabulary expression returns `None` and is *counted* rather than guessed
at: a resolver that guessed would have produced 13 confident intervals and no
signal that 9 of them were invented. The unbounded rate is low (1 of 48), so the
Phase-1 G5 hazard — a parser quietly disabling `temporal_correctness` by emitting
unbounded intervals — is **not** what is happening; the constraint is being
dropped entirely instead, which `coverage` sees as an inactive slot. Both are
losses and they are different losses.

**The sizing memo (G8, criterion 16).** At the measured **135.7 turns/hour**:

| Scope | Turns | Dev GPU |
|---|---|---|
| a — full corpus | 246,930 | **1,820 h** — not a candidate |
| b — evidence sessions only | 10,960 | **80.8 h** |
| b′ — evidence + 2 distractor sessions/question | 20,798 | 153.3 h |
| b′ — evidence + 5 distractor sessions/question | 35,555 | 262.1 h |
| c — 50 questions, evidence only | 1,096 | **8.1 h** |
| c — 100 questions, evidence only | 2,192 | 16.2 h |
| c — 200 questions, evidence only | 4,384 | 32.3 h |

This is what G8 existed to surface, and it surfaces early rather than in week 6:
**the full corpus is ~76 days of continuous GPU on this machine.** Option (c) at
100–200 questions is the only scope that is comfortably runnable here, and the
knowledge-update evidence sessions must be inside whichever is chosen. **The
choice is still Gate-0 item 9's, taken with this memo and item 8's items/hour —
not Phase 5's.**

### 2.3 The NLI verifier (G6, decision 6)

**Measured.** `cross-encoder/nli-deberta-v3-base` at revision
`6c749ce3`, loaded through `transformers`, entailment index read from
`config.id2label` — which is **1**, not 0, on this checkpoint
(`{0: contradiction, 1: entailment, 2: neutral}`). A hard-coded index would have
produced a plausible, wrong `entailed_score` with nothing crashing, which is why
the index is read rather than assumed.

Smoke behaviour on hand-written pairs: 0.998 for a span that entails its
assertion, 0.001 for a contradicted one, 0.002 for an unrelated one. That is a
sanity check on the wiring, **not** an evaluation — the evaluation is the
~50-pair hand-labelled agreement audit at `tau_nli`, which is a pending human
step and is listed in §5.

---

## 3. Exit criteria — where each one stands

| # | Criterion | State |
|---|---|---|
| 1 | every stored assertion's spans resolve on a fresh replay | **green**, as a test |
| 2 | idempotence: digests unchanged, skip count = turn count | **green**, as a test and on the replay pilot |
| 3 | Tier B frozen, both `SCHEMA_VERSION`s recorded, round-trips green | **green** — the freeze is in `schemas.py`, the amendment taken was none |
| 4 | provenance non-empty and no eligibility without a stored verdict | **green**, as a test |
| 5 | parse-failure rate < 2% with the frozen winner; bakeoff table in the report | **green**: candidate B at **1.7%** clears the 2% ceiling, and all three candidate rows (A, B, withdrawn C) are in the report. Decision 2 is frozen on B (§2.1b). The intermediate `no_survivor` run at a 1,024-token cap is preserved as `artefacts/phase5_bakeoff_cap1024.json` |
| 6 | manual span-support precision ≥ 0.90 on 50 assertions | **pending a human.** The worksheet is emitted and populated — 50 assertions drawn by seeded sample from live pilot output, disjoint from the calibration slice — with the protocol on it. Nothing else can fill it |
| 7 | per-rung counts; all rung-3 spans audited; mis-bound rate reported | **measured**: exact 375 · normalised 4 · fuzzy 30 (**91.7%** of 409 groundings on rung 1). All **30** rung-3 spans are on `audit_fuzzy_spans.csv` — every one, not a sample, as G5 requires. The mis-bound rate itself is the human column and is **pending** |
| 8 | quarantine rate reported, total and by cause, with F9's sentence beside it | **green** — reported by `QuarantineTally`, asserted by a test |
| 9 | NLI audit vs ~50 hand labels at the frozen `tau_nli`; distribution saved; `tau_nli` unchanged | score distribution saved (`nli_scores.jsonl`, 222 rows); 50-pair worksheet emitted, stratified around the threshold. **Hand labels pending** — and they are the instrument that decides whether the 32% quarantine rate is extraction error or a strict verifier. `tau_nli` unchanged; no code path can change it |
| 10 | mention and assertion yields per turn reported against the spike's 0.59 | **green**: replay reproduces the spike exactly (0.586), and the corrected bakeoff measures **1.08 (A) / 1.13 (B)** mentions per turn under the production context recipe — roughly double. Gate-0 item 8's D1 arithmetic was sized on 0.59 and should be re-read against this |
| 11 | `parse(mode="learned")` end to end; slot-level scores; unbounded-constraint rate | **run on 48 questions**: 0 slot parse failures, unbounded-constraint rate **2.1%**, and a finding — **69% of emitted time expressions fall outside `timeexpr`'s declared vocabulary** and are counted as unresolved rather than guessed. Slot-level P/R/F1 needs the gold columns, **pending a human** |
| 12 | relative dates widen to natural-granularity half-open intervals; "last May" / "in 2023" / "yesterday" tested | **green** |
| 13 | `graft.core` and `graft.synth` import no ML library; `graft.ingest` is the only new package on the allow side | **green**, plus a stronger guard: importing `graft.ingest` itself pulls in neither torch nor transformers |
| 14 | `llm_calls` / `llm_tokens_in` / `llm_tokens_out` metered inside the wrapper | **green** — counted in `LlmExtractor`, never by callers. *(Amended 14 Aug: the NLI verifier's forwards were absent from the ledger entirely — the verify stage reported zero on every meter but wall-clock. `NliVerifier` now meters `model_forwards` internally, one unit per batched forward.)* |
| 15 | two runs on this machine give identical digests; manifest carries revision + prompt SHA + decoding; cross-machine limitation stated | **green on replay, measured** — but only since 14 Aug: the original `--twice` re-ran over the *same* log, so every turn skipped and the comparison could not fail (it re-tested criterion 2, not 15). It now runs the whole pilot a second time into a **fresh log** and compares digests; the replay artefact carries both legs passing. The live two-run check is still the pilot flag, at the cost of a genuine second pass |
| 16 | throughput measured; sizing memo with all three scopes on both hardware targets | **green**: **135.7 turns/hour** end to end, and the memo costs all three scopes (§2.2a). The headline: the full corpus is **1,820 h** — ~76 days — on this machine, which is what G8 existed to surface before Gate 1 was scheduled |

---

## 4. Open items, honestly

| Item | Blocks | Note |
|---|---|---|
| ~~The G2 bakeoff has not been frozen~~ — **CLOSED 14 Aug 2026** | — | Decision 2 is frozen on **candidate B** (§2.1b); `pins.EXTRACTOR = dict(EXTRACTOR_CANDIDATES["B"])` and `require_extractor()` returns it. The row is struck rather than deleted: it was live through two aborted/corrected runs and its fail-closed behaviour is what kept an unmeasured extractor out of every downstream artefact in the meantime. *(This row still read "not frozen" after the freeze; caught by the second audit pass.)* |
| **Multi-span cross-turn provenance is unexercised by data** | what the write-up may claim | 0 of 222 assertions in the live pilot, 0 of 78 in the replay. The schema, the pipeline and a test support it; no run has produced one. §2.2's note has the three consequences |
| **15/500 corpus instances repeat a haystack `session_id`** | nothing now | identical content, *different* `haystack_dates` (e.g. `d23cf73b`). Measured 14 Aug: **zero** of the duplicated sessions are evidence sessions, and no current ingestion path mints a duplicate `turn_id` — but a Gate-0 item-9 scope that samples *distractor* sessions (option b′) can hit them, and must dedupe or disambiguate `turn_id`s before ingestion. Recorded here so the scope decision inherits the caveat |
| **Four human audits** | exit criteria 6, 7, 9, 11 | span support, rung-3 boundaries, NLI agreement, obligation slots. The pilot emits all four worksheets; none is machine-fillable, and a machine pass would be Phase 2.5's bootstrap-labels mistake one phase later |
| **Gate-0 item 8** | signing `GATE0_CONTRACT.md`, and therefore Gate 1 | the Phase-2.5 human timed pass. Nothing in Phase 5's code needs it |
| **The corpus scope** | Gates 1 and 4 | Gate-0 item 9's decision, taken with the sizing memo. The one commitment made in advance: the knowledge-update evidence sessions are in every candidate scope |
| **Whether a 4-bit 7B fits in 8 GB** | nothing now | fix F7's original question, now permanently unmeasured by ruling (§1.1) |
| **The `has_answer` signal is unused by Stage A** | nothing now | it is a gold label; the ingestion path deliberately never sees it, and it enters only through the Gate-0 item-3 proof-set definition |
| **Cross-machine determinism for Stage A** | the write-up | explicitly not promised (G11). What must match is the ingestion fingerprint — model id, revision, decoding config, prompt SHA — which `verify_handoff.py` now prints |

---

## 5. The pending human passes

**Both GPU runs are done** (§2.1b, §2.2a) — the commands that produced them are
below for the record, not as instructions:

```bash
# ALREADY RUN. Re-running either OVERWRITES its artefact, and re-running the
# pilot into the same --run-dir REGENERATES the four worksheets, discarding any
# human column already filled in them. Copy the filled sheets out first.
python scripts/phase5_bakeoff.py --out artefacts/phase5_bakeoff.json
python scripts/phase5_pilot.py --out artefacts/phase5_pilot.json --twice
```

**DONE — all four filled and returned 15 August 2026** (50 / 50 / 30 / 48 rows,
100% filled). Labels are **human**; an assistant repaired the CSV structure only
and wrote no judgment column. Results in §5a; the blank originals are kept at
`artefacts/phase5_pilot/filled_backup/`. The column specification each was filled
against:

| Worksheet | Column to fill | Criterion |
|---|---|---|
| `audit_span_support.csv` | `supported` | ≥ 0.90 over 50 assertions (Gate-0 decision 1). *Supported* iff the grounded span, read alone plus its turn, textually commits to `text_norm` |
| `audit_fuzzy_spans.csv` | `well_bounded` | **every** rung-3 span, not a sample; the mis-bound rate is reported, never tuned toward |
| `audit_nli.csv` | `human_entailed` | agreement at `tau_nli = 0.8`, stratified near the threshold. **Audit, not retune** |
| `audit_obligation_slots.csv` | the `gold_*` columns | scored with `graft.core.obligations.slot_level_scores`; reported wherever coverage is reported (fix F2) |

### 5a. What the audits measured — 15 August 2026

Measured against the **frozen** Gate-0 thresholds: the contract was signed
earlier the same day, so these are audits *against* fixed targets, not inputs to
setting them.

**1. Span support — PASSES, with a denominator that had to be ruled.**

| Scope | Result | |
|---|---|---|
| **Eligible assertions only** | **35 / 35 = 1.000** | ✅ against Gate-0's ≥ 0.90 |
| All 50 audited rows | 39 / 50 = 0.780 | ✗ |

**Ruled: the eligible-only reading is the criterion's.** Quarantined assertions
never enter the active graph (fix F9), so a *precision* statement about
retrievable evidence is over what was admitted. The all-rows figure is not a
precision failure but a different quantity, and the 4-of-15 quarantined rows that
*were* supported are a **false-quarantine recall loss** — recorded here because
it is a real cost of `tau_nli` = 0.8 and belongs beside finding 2.

**2. NLI — 84% agreement overall, and a probable verifier defect underneath it.**

Near-threshold 24/25 = 0.960; far 18/25 = 0.720. The error is asymmetric: **7
false negatives against 1 false positive.**

**Do not read this as "`tau_nli` is too strict" yet.** Several of the seven are
near-verbatim restatements scoring ~0.001 — for example premise *"Taking care of
your leather boots is crucial to extend their lifespan and prevent damage. Here
are some tips…"* against hypothesis *"Taking care of your leather boots is
crucial to extend their lifespan and prevent damage."*, scored **0.001**, while a
structurally identical near-threshold row scores 0.9968. A verbatim prefix cannot
be a 0.001 entailment. That points at premise/hypothesis pairing, ordering or
truncation inside the verifier, **not at the threshold** — and the contract's
"audit, never retune" rule bites precisely here: the honest next step is to
reproduce the seven, not to move 0.8.

**3. Fuzzy spans — 9 / 30 well-bounded = 0.300, so a 70% mis-bound rate.**

The population, not a sample (every rung-3 span, per G5). Reported, never tuned
toward. The failures are visible and coarse: one span is `sparent`, a fragment of
*"parents"*; many begin mid-clause (`"warm light."`, `"candid moments."`,
`"'re sharp and well-lit."`). This is a ceiling-1 finding and the largest single
quality defect Stage A has produced.

**4. Obligation slots — and the number that matters is `entity_anchor` at 17%.**

| Slot | Exact match |
|---|---|
| `entity_anchor` | **8 / 48 = 0.167** |
| `value_type` | 21 / 48 = 0.438 |
| `time_expression` | 4 / 21 = 0.190 |
| `needs_source` | 44 / 48 = 0.917 |
| `aggregate` | 27 / 48 = 0.562 |

Exact string match is a harsh metric on a free-text slot — *"painting classes"*
against gold *"painting projects"* scores 0 while being nearly right — so 0.167
is a **floor**, not the parser's true quality.

**It is nonetheless the same defect Phase 7 measured from the other end.**
`PHASE7_DECISIONS.md` §3.2 found the entity channel matching an anchor in only
3 of 10 questions and considered amending decision 7's normalised-exact rule.
This says the anchors are **mostly wrong at source**, so the fault is upstream in
the parser rather than in the matcher — which weakens the case for amending
decision 7 and strengthens leaving it alone (§3.2a already recommended waiting).
Two independent measurements, one cause.

---

## 6. Handoff to Phase 6

* **The event log** with eligible assertions carrying resolved multi-span
  provenance, four flags and eligibility verdicts — Stage B's entire input.
  `graft.graphstore.ReplayGraphStore.at()` reads it; `is_eligible` is what `H`'s
  seventh sub-check has been reading since Phase 1.
* **Mentions, as `mention.add` log events** (`span_id`, `turn_id`, grounded
  text, rung), read back with `graft.ingest.pipeline.mentions_of` — D1's item
  source. Deliberately outside `GRAPH_OPS`: no `Mention` node exists until D1
  decides, so replay ignores the op. *(Added 14 Aug — §7's finding 3: before
  this, D1's items were unrecoverable from the permanent record.)*
* **`GATE0_CONTRACT.md`**, drafted here at nine of ten items. *(Since overtaken:
  item 8 measured GO and item 9 decided 15 Aug 2026, and the contract **signed**
  the same day — Gate 1 is unblocked. This bullet is the state Phase 5 handed
  over, kept as the historical record.)*
* **The measured yields** (mentions/turn, assertions/turn, quarantine rate) that
  size Gate-1's annotation batches under item 8's items/hour.
* **The frozen extractor and prompt registry**, because Phase 6's
  LLM-prompted-linking baseline (architecture §6.4) must run against the *same*
  extraction it competes with — one prompt-registry SHA covers every prompt this
  phase runs, including the obligation parser's.
* **`graft.ingest.corpus`**, which is also how Phase 6 addresses turns: the
  Phase-2.5 turn-id convention is unchanged, so the spike's flagged items and
  guidelines carry forward against Phase-5 records.

---

## 7. The 13–14 Aug 2026 audit — every finding, its verdict, and what moved

Two inputs: a code review the project owner brought in, and an independent
adversarial audit run against the tree (four verification agents over the
review's claims, four fresh lenses, every fresh finding given to a dedicated
refuter told to kill it). **Nothing was fixed on a claim's say-so** — each was
confirmed against the code, several by execution — and the refuted ones are
listed because knowing what was *not* wrong is part of the record.

### 7.1 The review's claims — all ten confirmed, all fixed

| # | Claim | What the verification added | Fix |
|---|---|---|---|
| 1 | the bakeoff mixed unrelated conversations | worse than reported: 21 sessions from **10 users**, 55/60 prompts with foreign context, 71% of context turns foreign, held-back turns out of corpus order, and cross-turn quotes could ground into a *foreign user's turn* — polluting stage 2's own metric | `calibration_slice` groups per conversation in corpus order; harness runs the production recipe (§1.6) |
| 2 | calibration/pilot disjointness declared but not implemented | measured: all 60 slice turns inside the pilot's 248, no exclusion anywhere; the declaration was structurally unsatisfiable without a draw-side filter | the span/NLI audit draws exclude any assertion touching a slice turn (`from_calibration`); one slice definition, imported by both scripts |
| 3 | mentions not preserved for Phase 6 | "unreferenced span = mention" confirmed unsafe (a mention span can coincide with a quote span) | `mention.add` log events + `mentions_of`; outside `GRAPH_OPS` by design |
| 4 | `--twice` tested idempotence, not determinism; resumed reports partial with no marker | the old check *could not fail* | fresh-log second run compares digests (both legs in the artefact); `partial_report` flag + limitation line on resumed runs |
| 5 | truncation counter counted only the final attempt | candidate-asymmetric: repair (A/C's policy) hid exactly A/C's truncations | `truncations` counted per generation end-to-end (`RawExtraction` → bakeoff row → pilot report) |
| 6 | NLI forwards absent from the ledger | verify stage reported zero on every meter but wall-clock | `NliVerifier(ledger=...)` meters `model_forwards` internally |
| 7a | "512-token" summary cap was 512 *words*; the docstring's conservatism argument was backwards | measured on the pinned tokenizer: 512 words ≈ 725 tokens | enforced at generation (`max_new_tokens=SUMMARY_MAX_TOKENS`); word cap demoted to a documented backstop |
| 7b | oblparse trace `parse_ok` constant-True | counter was right, per-question record lied | captured before the fallback installs |
| 8i | unbounded rendered context + 600-token cap | the §1.6 measurements | `CONTEXT_CLIP_CHARS = 600`; `max_new_tokens = 1024` |
| 8ii/iii | plan still credited B with a zero-by-construction parse failure | the code was already corrected; the plan was not | plan G2 cell corrected; wording retired in the consistency guard |

### 7.2 Fresh findings that survived refutation — all fixed

| # | Finding | Fix |
|---|---|---|
| 1 | `normalise()` broke on lowercase-expanding unicode (`'İ'.lower()` is two chars appended as one element against one offset entry) — rung-2 offsets shifted one char at score 1.0, IndexError possible at turn end; **13 real corpus turns affected** | per-character offset mapping; invariant `len(index) == len(norm)+1` restored and tested against the İstanbul case |
| 2 | `timeexpr.resolve()` **raised** on calendar-invalid model output (`"2023-06-31"`, `"9999"`, `"past 3000 years"`) — a hallucinated slot could kill the pilot after the GPU work was spent | invalid phrases resolve to `None`, counted as unresolved; a malformed `question_date` (caller data) still raises |
| 3 | `"between March and June"` asked in May 2023 resolved each bare month's year independently → `[Jun 2022 start, Mar 2023 end)` hulled into an interval covering *neither* endpoint | the second endpoint reads forward from the first (≤ 2 year-shifts); still-incoherent expressions return `None`, never a knowingly wrong hull |
| 4 | the fuzzy refinement was locked to the single coarse winner — a decoy on the coarse grid beat the true window (measured: a one-typo "forty euros" quote bound to the "ninety euros" sentence at 0.95) | top-4 coarse neighbourhoods refined; the residual is exactly what the G5 mis-bound audit measures. Plus: tied windows differing only by an edge blank are trimmed |
| 5 | crash-resume broke the rolling-summary chain (previous summary read from cache with silent `""` default; cache flushed only at exit) — a resumed run's prompts, extractions and digest differed from an uninterrupted run's | missing predecessors recomputed recursively; write-through flush after every refresh |
| 6 | `nli_worksheet` hardcoded `tau = 0.8` — a Gate-0 amendment would leave the tau audit stratifying around the stale copy | `tau` is a parameter fed from `cfg.tau_nli`, recorded per row |
| 7 | both scripts emitted bare `NaN` into artefacts (an errored candidate's rates), making them invalid RFC-8259 JSON | `runtime.json_sanitize` (NaN/±inf → `null`) + `allow_nan=False` so an unsanitized path fails loudly |
| 8 | whitespace-only quotes grounded at rung 1 score 1.0 — a blank span as recorded provenance | `strip()` filters at the parser *and* the grounder |
| 9 | `register()`'s plain-string path passed `question_date=""` — any time-bearing question through `core.obligations.parse(mode="learned")` crashed | the resolve call fails soft on the register path, counted as unresolved |

And one found here, by neither input (§1.6): **candidate B's grammar attached to
every generation**, so summaries and obligation parses would have been forced
into extraction-schema JSON had B won. Scoped to extraction calls; tested.

### 7.3 Refuted — investigated, deliberately not "fixed"

* **Bakeoff load-time bias beyond the tie band** — the measured artefact shows
  the load delta is far inside stage 2's 1% band. *(The load is excluded from
  the timed loop anyway now, as hygiene while the harness was rebuilt.)*
* **NLI silent CUDA→CPU fallback unattributable** — in the pilot's only live
  path the extractor would crash first on a dead CUDA; the device is now in the
  artefact config as a courtesy record.
* **Shared audit RNG defeats re-derivation** — the convention it tested against
  does not exist; per-sheet streams were adopted anyway as part of the
  exclusion rework, which changed the draws regardless.
* **Span-audit protocol "exists only in a docstring"** — it is in three
  normative documents including the Gate-0 contract.
* **Obligation-audit stratification truncation** — arithmetically impossible at
  the corpus's fixed 6 question types. *(The audit did get its own
  `OBLIGATION_AUDIT_N` pin: two decisions, two constants.)*
* **Duplicate `turn_id`s from repeated haystack sessions** — cannot occur on
  any current path (zero duplicated sessions are evidence sessions); recorded
  as a §4 caveat for the item-9 scope decision instead.

### 7.4 What the audit changed in the frozen-value record

`max_new_tokens` 600 → **1024** and `CONTEXT_CLIP_CHARS = 600` (new) — both
instrument corrections made while `pins.EXTRACTOR` was `None`, i.e. before any
value they influence was frozen; both are in `ingestion_fingerprint`, so no two
runs under different regimes can carry the same identity. `SUMMARY_MAX_TOKENS`
is unchanged in value and now enforced in the unit the plan declared.

### 7.5 The second pass — auditing the fixes themselves (14 Aug 2026)

The first pass fixed ~20 defects; this one audited **the fixed tree**, on the
principle that a fix is a change and a change is where defects come from. Three
lenses (write path, docs-vs-code, fix-delta verification), every finding given
to a dedicated refuter. **Eighteen findings, twelve confirmed, six refuted.**
Two of the twelve were introduced *by* the first pass's own fixes, which is the
result that justifies the pass.

**Confirmed in code — four, all fixed and mutation-tested:**

| Finding | Why it matters | Fix |
|---|---|---|
| **The unicode fix made the rung-2 offset map non-injective.** One original character now maps to several normalised ones, so `index[j + n]` lands on that character's *start* whenever a match ends inside an expansion — storing a span one character short, or of **zero width**, at score 1.0 | introduced by 7.2 #1; silent, and rung 2 has no score to warn on | `orig_end()` walks to the first strictly greater offset; a zero-width result falls through to rung 3 instead of being stored |
| **The run-scoped assertion dedup made a resumed run's graph differ from an uninterrupted one's.** `_write_assertion` takes `asserted_by` and `t_created` from the turn that *writes*, which the content-derived id does not cover — so an uninterrupted run kept the first turn's speaker and timestamp, a resumed run the last turn's, and replay is last-write-wins | it **falsifies this file's own §1.2(a)** ("the replayed snapshot is identical") on the crash path that claim is about | dedup scoped to the turn — its stated intent — so both orders end on the same final write. The 3 within-turn duplicates the pilot measured are now **counted** (`assertions_duplicate_within_turn`) instead of vanishing |
| **A resumed draft reported `from_repair=False`** — a clean extraction it cannot know about, since the repair count is nowhere in the log | `cross_tab_repaired_extraction` read as measured zeros | `None` = unrecoverable, counted in `repair_provenance_unknown`, mirroring the rung's existing `"resumed"` disclosure |
| **The write-through summary flush was not atomic**, and `_load_cache` raised on a torn file | introduced by 7.2 #5: a crash mid-flush killed the *next* run at construction — on exactly the path the write-through fix exists to serve, and against the class's own "deleting it changes throughput and nothing else" | temp-file + `os.replace`; a torn cache is a counted miss (`corrupt_cache`), not a crash |

**Confirmed in the record — seven, all corrected above:** §2.1b described the
frozen artefact as recording "stage 1" when it records "stage 2" (the code fix
landed after the run; the artefact is left as the run wrote it and the
discrepancy is now stated); §1.6's token identity is approximate, not exact, and
the hypothesis it carried was **falsified** by the re-run; §2.2 credited the
replay pilot with a cross-turn multi-span assertion it does not contain — and
correcting it exposed that **no real run has ever produced one**, now a §4 open
item; §2.2a's "330 extracted" and "92.2%" were 333 and 91.7% over a groundings
denominator; §4 still said the bakeoff was unfrozen after it was frozen; §5
listed both completed GPU runs as pending commands, without warning that
re-running the pilot **regenerates the four worksheets and discards any human
column already filled**; and `GRAFT_PHASE5_BUILD.md` had not been touched by the
runs at all — status, §6's UNSIGNED header and decision row 2 were all stale.

**Confirmed as a test gap — one, closed:** `LlmExtractor.extract`'s repair and
truncation loop had **no test whatsoever**, so reverting the per-generation
truncation counter survived the entire suite; the same was true of every
`ledger.count` call site. Both are now driven by the real methods against a fake
tokenizer and model — no GPU, no weights — and both mutations now fail.

**Refuted — six, deliberately not acted on:** the `between` year-shift applied
to explicit-year endpoints (the guard *is* reachable and the alternative is
worse); the bakeoff's summary never firing on the slice (real, but a disclosed
slice property — re-cutting the slice after seeing results is the thing
predeclaration forbids); stage 3 having no instrument (the artefact's
`samples[:20]` **is** the declared 20-assertion sub-audit); `GATE0_CONTRACT.md`
still quoting the spike's 0.59 mentions/turn (the document is unsigned and §2.2a
supersedes it in the same breath); criterion 14's metering being untested (a
real gap, but its stated harm was wrong — the extraction counters exist
independently in the report — and it was closed anyway with the test above);
and the pilot artefact omitting a `decoding` block (the `ingestion_fingerprint`
covers `DECODING` by construction, which is the promise G11 actually makes).

**Nothing measured changed.** The replay pilot's log and graph digests are
byte-identical before and after all four code fixes — they close hazards on
paths the clean data never took, which is the outcome to want from a fix pass.
