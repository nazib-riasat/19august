# PHASE11_DECISIONS.md — what the Phase-11 build decided and measured

**Status.** Stages A, B and C built and green, 19 August 2026. Nothing run on
GPU. Suite 1,304 passed, 0 failed; `scripts/check_plan_consistency.py` clean.

**Authority.** This file wins conflicts with `GRAFT_PHASE11_BUILD.md`, the
convention `PHASE9_DECISIONS.md` and `PHASE10_DECISIONS.md` already carry.

**Read §1 before quoting any Phase-11 number, and §1.1 first** — it is the one
that was wrong in a way that would have shaped the whole phase.

---

## 1. What the build found that reading did not

### 1.1 G2's premise was false: the cost axis was already metered

The plan's first draft of G2 asserted that no meter is spent anywhere in the read
path, and made closing that the phase's highest-priority item. **It was already
closed.** `PHASE10_DECISIONS.md` §5 is an *audit record* — it lists findings
**with their fixes** — and A1 was fixed on 16 August. Reading a fix record as a
defect record is the mirror image of `CLAUDE.md` §5's overreach pattern, and it
is kept here rather than quietly deleted because the failure mode is
reusable: **a document that records "we found X and fixed it" reads, on a fast
scan, exactly like one that records "X is broken."**

What is actually built: `Reader.generate` spends `llm_calls`, `model_forwards`,
`llm_tokens_in` and `llm_tokens_out` (`graft/reader/read.py:202`);
`Ledger.stage()` records `wall_clock_ms`; the orchestrator opens a stage per
phase; the snapshot reaches `OutputRecord.ledger_snapshot` and `report()`; the
runner opens a `query_scope` per question. Run R3's artefact carries real values
on 8 of its 10 records.

**Measured on R3** — 10 synthetic fixtures, `is_wiring_test = True`, so an
instrument demonstration and **not a result**: 2 LLM calls per query, 513.2
tokens in, 13.5 out, **526.8 total**, 55.6 model forwards, 1,054 ms wall clock.

### 1.2 The obvious regression guard would have been wrong, and would have been relaxed

A1's real danger is that zero-initialised meters make an *absent* measurement
look like a *cheap* query, so the guard is what matters, not the spending. The
natural guard — "no all-zero ledger snapshot may reach an artefact" — **fails on
correct behaviour**: 2 of R3's 10 records are legitimately all-zero, because the
gate route returns before Stage D and before any reader call, so nothing is spent
and nothing should be.

A guard that fires on correct behaviour gets relaxed, and takes the real check
with it. The rule is therefore narrower: **all-zero is permitted only where the
gate declined.** `UnmeteredError` in `reader/orchestrator.py`, with the carve-out
asserted in its own test so it cannot be widened by accident.

A second distinction fell out of the same work: an **empty** snapshot and an
**all-zero** snapshot are different states. The first says nobody was counting;
the second says counting happened and came to nothing. Conflating them is how A1
stayed invisible. They are separately counted, and only one is fatal at the
`aggregate` layer — unit tests legitimately run the read path unledgered, and
`scripts/phase10_read.py` owns the stricter half.

### 1.3 Three flags were cosmetic, and one of them would have produced a false record

Found by a self-scan for parsed-but-unread argparse destinations, 19 August 2026.

* **`--parse-obligations` was read in exactly one place: the stamp text.** The
  code used the empty obligation either way, so passing it made the artefact
  claim *"obligations LLM-parsed"* for a run that parsed none. **Removed, not
  fixed**, because it cannot be a flag: fix F7's `ModelSlot` refuses to hold the
  extractor and the reader at once, so LLM obligation parsing is a separate
  stage-sequential pass. Deferred by name. A label that disagrees with what ran
  is worse than an absent capability.
* **`--ceilings` was dead.** *Implemented*, not removed, because `report.py`
  lists the five-ceiling decomposition as a **primary** claim — one that needs no
  baseline — and a primary claim whose switch does nothing is not a claim.
* **`--device` was dead**, so `--device cpu` silently ran on cuda. Wired to
  `Reader(device=...)`, default `None` so the pin governs.

### 1.4 `--help` crashed on Windows on six runners — and the first explanation of why was wrong

`argparse` prints `description=__doc__`, the docstrings carry `→` and `β`, and
Windows consoles default to cp1252. Reproduced on **six** runners: the three
built in this phase plus `phase5_bakeoff.py`, `phase9_measure.py` and
`verify_handoff.py`. `scripts/phase3_calibrate.py` had already hit it and set the
`stream.reconfigure(encoding="utf-8", errors="replace")` convention; all six now
carry it.

**The severity claim attached to it was false, and is kept here because the
error is this project's own catalogued pattern.** The first write-up said the
guard mattered mainly because *"one curly apostrophe in a LoCoMo answer would
kill a 35-minute reader pass"*. Checked at the boundary, as `CLAUDE.md` §5 says
to: **U+2019 and U+2014 are cp1252 0x92 and 0x97 and encode fine.** The claim was
asserted, not measured.

What LoCoMo actually holds outside cp1252, measured on the parsed corpus: **18
occurrences of 11 distinct characters** — 8 × U+200B ZERO WIDTH SPACE, one
U+200D, one U+FE0F, and 8 emoji — across **7 turns and 1 gold answer**. (A first
scan of the *raw file* found zero, because JSON stores them as ASCII `\u200b`
escapes; the parsed layer is the one that matters.)

And the risk is smaller again: **no print path in these runners emits corpus
text.** `locomo_eval.py` prints ids, outcomes, counts and metrics;
`locomo_ingest.py` prints sample ids and counts; `probe` goes through
`json.dumps`, which escapes non-ASCII by default. So the guard is insurance
against a future debug print of an answer, not a live crash averted.

**Two lessons, both mechanical.** A `--help` crash is a real symptom of an
encoder that also sits on the run's output path, so fixing it is right. But
"worth more than the three defects above" was a severity ranking invented to
match a story rather than derived from a measurement — and the wrong version had
already been written into six source files, where a false comment outlives a
false paragraph.

### 1.5 The LoCoMo category mapping was a guess, and the corpus verified it

`baselines/categories.py`'s integer codes were the convention in common use
across LoCoMo evaluation code, unverified against the dataset file, which had not
been downloaded. Rather than assert them, the mapping shipped with
`verify_against_corpus`, which checks the adversarial bucket against
`DATASET_DECISION.md` §1.2's **independently recorded 446**.

The corpus arrived the same day and `scripts/locomo_ingest.py probe` reported
**zero findings**: 10 samples, 1,986 questions, **446 adversarial**, 5,882 turns,
every timestamp parsed, 9 of 2,815 evidence markers unresolved (0.3%, all
image-only turns). Agreement between a count and a mapping from two independent
sources is evidence; the mapping is now verified, with the measured distribution
written into the module:

| code | name | count |
|---|---|---|
| 1 | multi_hop | 282 |
| 2 | temporal | 321 |
| 3 | open_domain | 96 |
| 4 | single_hop | 841 |
| 5 | adversarial | 446 |

**One discrepancy, recorded rather than smoothed:** `DATASET_DECISION.md` §1
records 5,875 turns; this loader counts **5,882**, because it skips turns with
empty `text` (image-only turns carry a `blip_caption`) and the recorded figure
evidently counted a slightly different set. Immaterial to any claim; recorded
because an unexplained 7-turn gap later reads as a defect.

### 1.6 Throughput: the cost was where the measurement said, not where the reasoning said

Added 19 August 2026, after the project owner's instruction to use the hardware
fully wherever it saves time. Two wins, and the second one was nearly abandoned
on a wrong hypothesis.

**Win 1 — the eval path was re-indexing the corpus per question.** ``stage_c``
constructed ``BM25Channel`` and ``DenseChannel`` on every call, and both build an
index over the conversation in their constructor: BM25 tokenises it,
``DenseChannel._build`` **embeds** it. LoCoMo carries 96 to 260 questions per
conversation, so the corpus was being re-embedded up to 260 times to answer 260
questions about it. ``ChannelCache`` builds once per conversation; a one-entry
cache suffices because the question list is contiguous in ``conv_id``, and it is
*keyed* rather than assumed so an interleaving caller gets rebuilds instead of
silently wrong pools. Asserted by test: warm and cold produce **identical** pools,
and the per-question *question* encode is still metered — that is a real per-query
cost and must not disappear along with the index.

**Win 2 — batched extraction, and the hypothesis that was wrong by ~180×.** The
reasoning said the per-step grammar work would dominate: 220 decode steps, each
masking a 151,936-token vocabulary through xgrammar's Triton-free
``torch_native`` backend, which the module's own docstring already flags as
carrying a platform penalty. If that were true, batching could not help — the
mask is per row per step and scales with the batch.

Measured instead, on CPU in seconds and before any code was written:

| per decode step | cost | per 220-token generation |
|---|---|---|
| ``apply_token_bitmask_inplace`` (batch 1) | 0.29 ms | 0.06 s |
| ``fill_next_token_bitmask`` | 1.70 ms | 0.37 s |
| ``accept_token`` | 0.019 ms | ~0.00 s |

**0.43 s of a 26.5 s turn — 1.6%.** The other 98% is streaming 6.18 GB of weights
per decode step, which is exactly what a batch amortises. So batching is the right
lever, for a reason the reasoning had inverted.

**Why batching is sound**, checked rather than assumed: turn *N*'s prompt is built
by ``summary.summary_for`` and ``context_window``, both of which read the **raw
turn list**. No turn's prompt depends on another turn's extraction *output*, so
extraction is embarrassingly parallel within a conversation and the only serial
thing about it was ``GrammarLogitsProcessor``'s single matcher.

What was built: ``BatchedGrammarLogitsProcessor`` (one matcher **and one bitmask
row** per batch row — a shared matcher decodes row 2 against row 1's parse
position, a shared bitmask row masks every sequence with the first one's allowed
set); ``_generate_batch`` (**left**-padded, because ``generate`` appends right and
a right-padded prompt would put pads between prompt and continuation *and* make
the processor read a pad as the last sampled token); ``extract_batch`` (batches
the 98.3% happy path, falls back to the **existing audited** single-stream repair
loop for the 1.7% the pilot measured, rather than reimplementing it); and
``IngestPipeline.extract_slice_batched``, where storage stays per turn so
``turn.add`` is still appended last and crash-resume stays turn-granular.

``batch_size=1`` **routes to** ``extract_slice``, not to a one-row batched path:
the default has to be the code that produced `PHASE5_DECISIONS.md` §2's frozen
record, not a lookalike.

**The one claim batching makes that cannot be reasoned about.**
``ingestion_fingerprint`` hashes ``model_id``, ``revision``, ``dtype``,
``quantization``, ``repair`` and ``constrained`` — **not** batch size — so a
batched run is the same *experiment* by that definition. But different matmul
shapes reduce in a different order, so a near-tie argmax can flip and one turn can
decode differently in a batch than alone. `CLAUDE.md` §5's standing lesson is to
check that at the boundary, so ``scripts/locomo_ingest.py verify-batch`` extracts
the same turns both ways and reports the speedup **and** the identical-extraction
rate. A mismatch is not a failure; it is the number that makes the
speed-versus-byte-identity trade a decision instead of an accident.

**One of this section's own tests was silently vacuous.** The test asserting that
a batched slice builds the same graph compared ``snap.assertion_ids()`` — a method
that does not exist — so the content comparison was a no-op and only ``counts()``
was checked. Two different graphs can share counts; they cannot share
``state_digest()``, which is what it compares now. This is the same failure the
project has caught before (a guard that passes because it checks nothing), and it
appeared here inside the very test meant to protect a throughput change.

### 1.7 The prompt was re-frozen for metric-format alignment — at the last clean moment

Added 19 August 2026, at the project owner's instruction, after §1.4-style
boundary-checking of how the reference systems prompt. **Timing is the whole
argument**: no decisive LoCoMo run exists, only wiring-test runs R1–R3 (each
stamped `is_wiring_test`), so amending the prompt now is a pre-registered
decision. The identical edit one day later, after the first scored run, would be
tuning on evaluation data and unrecoverably contaminated.

**The asymmetry being corrected.** Mem-T trains with F1 *inside its reward*
(`Perform(v) = F1(v)`, their Eq. 10) and its FinishTool prompt demands "the
concise answer following the Final Result Format". GRAFT trains nothing on
LoCoMo and nothing against F1 — so the only lever it has, the one every
reference system also uses, is answer *format*. A frozen reader that loses
token-F1 to formatting is measuring prose style, not memory.

Three additions to `PROMPT_TEMPLATE`, each tied to a scoring failure invisible
in the answer's correctness:

| addition | failure it prevents |
|---|---|
| date rule ("8 May 2023") | 321 temporal questions have day-month-year golds; an ISO answer is *right* and scores ~0 token-F1 — the metric measures a calendar convention |
| multi-item rule (comma-separated) | multi-hop golds are frequently lists; token recall punishes naming one of two cities |
| three labelled format examples | a 3B instruct model anchors on demonstrations, not rules; examples are generic names and labelled "these are not evidence" |

Plus one parser change: a leading `Answer:` echo is stripped at *extraction*
(the scoring rule untouched) — normalisation strips articles but not the word
"answer", so every echoed prefix cost F1 precision for a formatting artefact.

**Cost, recorded:** `PROMPT_SHA` moved (4a8abf10… → e023ea71…), so the stage-E
fingerprint moved and R1–R3's artefacts carry the old one. Nothing result-grade
is voided. **What this does not do:** the metric implementation is still this
project's own, not the paper's script (§ the same-code gap) — format alignment
narrows the formatting loss, not the scorer-drift risk.

### 1.8 The integration-gap class: fixtures that pass while the pipeline misses a stage

Found 19 August 2026, when the first full LoCoMo ingestion produced a graph
reading ``nodes: 0`` — and Stage C retrieves over nodes, so the queued eval
would have spent ~35 minutes of reader time producing 1,986 fallback
abstentions that read as total system failure and mean "a stage was skipped".

**The mechanism.** `graft/tests/test_locomo_eval.py` drives the whole read path
over `test_retrieve.py`'s fixture — which has nodes **pre-built**. So the join
tests were green while the real pipeline lacked the stage that creates nodes:
Phase 6's stand-in constructor (`graphbuild.standin.construct`), which turns
eligible assertions into Claim/Value nodes via exact-match mention linking. The
pilot's graph had gone through it (`artefacts/phase6/events.jsonl`); the LoCoMo
chain never did. **A fixture that supplies a stage's output is a test that
cannot detect the stage's absence.**

**A sweep for the same class found one more instance.** The read-path stamp
promised gate features "for post-hoc thresholding" — but `scripts/phase8_gate.py`
**never persists the trained gate model** (no `torch.save` anywhere; only the
in-training best-state restore). Features without a model are half the promise:
applying a threshold post hoc requires retraining the MuSiQue gate (seeded and
deterministic, minutes of CPU — so recoverable, not lost). Checked and clean:
the eval's embedder auto-loads (`embed._encode` calls `self.load()`), and the
utility head's `ATOM_WIDTH` is guarded at load.

**Fixes, 19 Aug 2026:**

* `scripts/locomo_stageb.py` — the missing stage as a runner. Works on a **copy**
  of the Stage-A log because `construct()` appends into the log it is given and
  is **not idempotent**; refuses a destination already carrying Stage-B ops
  unless `--fresh`. CPU-only, exact-match linking, no embedder, no trained
  decoder — entirely inside audited Phase-6 machinery. **Not yet run.**
* `locomo_eval.pick_run_dir` prefers the Stage-B log when present (explicit
  `--run-dir` always wins), and `require_nodes` **refuses** a nodeless graph by
  naming the missing stage — the guard the fixture could not be.
* The stamp's post-hoc-gating line now states the retrain requirement instead of
  implying a checkpoint exists. For Opus 5: add `torch.save` of the winning
  arm's state dict to `scripts/phase8_gate.py` (plus a loader), so the promise
  becomes literal.
* `locomo_ingest.py`: a `--verify-only` pass no longer clobbers the extract
  run's artefact (the 19 Aug run lost its per-conversation JSON summary that
  way; the log and `progress.json` retained everything).

**What to expect when the stage runs, recorded so it is not misread as a
defect:** the stand-in links assertions through each turn's *first mention's
entity*, and the LoCoMo log carries 1,422 mentions over 5,882 turns — so node
count will sit **well below** the 2,268 eligible assertions. That is G5's
documented behaviour; the unlinked assertions stay in the log, recoverable by a
trained D1.

---

### 1.9 Runs 1–3: what two full evaluations measured, and the five changes between them

**Run 1 (20 Aug 2026) — overall F1 7.48 / BLEU-1 6.01, coverage 0.534.**
1,986 questions, 70 min, `--head --ceilings`. Two defects had to be fixed before
it would complete at all, both recorded above their fix: `cost_report` read the
ledger's *cumulative* totals, so per-query cost inflated ~1000× over 1,986
questions; and ceilings 4/5 serialised **gold** against the *capped* retrieval
pool, which crashed at question 2 with a `KeyError`. The second is
`PHASE10_DECISIONS.md` §5 A3's class arriving a third time — a *retrieval*
shortfall reaching a *packing* measurement — and it only became reachable when
the Stage-B coverage fix tripled the eligible-node count and pools began
saturating. Ceilings now serialise against an uncapped per-conversation pool;
ceiling 3 keeps the capped pool and is where a retrieval shortfall belongs.

**Run 2 — overall F1 15.17 / BLEU-1 12.46, coverage 0.638.** Three changes,
each measured against run 1 on identical questions rather than projected:

| change | what it was |
|---|---|
| training-free selection | top-`max_atoms` by Stage-C's question-conditioned fused score, closure-completed, `H`-checked (`portfolio.relevance_select`) |
| session dates | each claim line carries the date of the turn it is sourced from |
| parse hygiene | bracket-wrapped `[INSUFFICIENT EVIDENCE]` (13 records) and empty-normalising answers (172 of 1,009) become abstentions |

Temporal moved most — **2.58 → 8.81**, from the dates alone. Open-domain barely
moved (8.64 → 8.83), which is the right shape: those questions turn on neither
dates nor span selection.

**The projection that came with those changes was wrong by ~3×**, and it is
recorded because the failure is reusable. It predicted coverage → 0.8,
answered-F1 → 0.45–0.55, overall 35–44; measured on a 150-question slice
*before* committing GPU time, coverage was 0.587 and answered-F1 0.180. Running
the slice first is what turned a target into a measurement. **No change in run 2
or run 3 was made to reach a number**, and the frozen metrics
(`normalise_answer`, `token_f1`, `bleu1`) were not touched by any of them.

**Adversarial abstention went backwards, 0.581 → 0.473, for a mechanical
reason.** The training-free selector returns a set whenever one is `H`-valid;
the random sampler frequently dead-ended into `fallback`, which scored as a
correct abstention on unanswerable questions. Run 1 was partly being *rewarded
for failing to construct*. Neither figure reflects judgment — the component that
would supply it is the Phase-8 gate, off in both runs, with `gate_features`
recorded per row so a threshold costs no GPU.

**Ceiling 5 is the wall, and it did not move**: 0.125 → 0.137 exact, ~0.26
token-F1. Handed a *perfect* gold proof at unbounded budget, the frozen 3B
reader tops out there. Run 2's end-to-end 0.152 sits close to it, so the run-2
changes largely closed the gap *to the reader* and left the reader where it was.
That is a finding about any system using a 3B reader on LoCoMo, not only this
one — and it is the number that bounds how much run 3 can buy.

**Run 3 — five changes, built 20 Aug 2026, not yet run.** Post-error-analysis
engineering; metrics unchanged.

1. **Raw-turn evidence tier.** Top-3 dialogue turns by the same half-BM25
   half-dense fusion, appended as **uncitable context** after the numbered
   claims. Every reference system on LoCoMo shows its reader raw text; this one
   showed only extractor-derived claims, and 46% of questions have no gold atom
   because extraction or the quarantine removed the evidence before retrieval
   could see it. The turns are already stored as provenance, so this shows the
   reader what the graph was built *from* and adds no new store and no new
   claim. They carry no `[c#]` id, so `claim_map`, `resolve_citations` and
   citation precision are untouched.
2. **Junk-answer hygiene** (`locomo_eval.clean_answer`), at the runner and
   deliberately not in `reader/parse.py` — the core parser is the frozen read
   path whose rules ride in `stage_e_fingerprint`, while this is a reporting
   decision about one corpus's observed shapes. The ordering trap is recorded in
   its docstring: unwrapping before the junk test turns `[c12]` into the
   respectable-looking token `c12`.
3. **Date format** — `13 October 2023`, not ISO. Prompt rule 4 asks for "day
   month year" and run 2 fed the reader ISO, which it copied straight into
   answers scored against `9 October 2022`-style golds. Evidence and instruction
   have to agree about format or the instruction loses.
4. **Prompt examples swapped** — run 2's reader emitted `Rome, Lisbon [c1][c4]`
   verbatim as an answer. Examples are now values LoCoMo cannot contain
   (Tbilisi / 3 April 2019 / Oslo, Nairobi) plus a rule 7 forbidding copying.
   **`PROMPT_SHA` moves `e023ea71…` → `8121eb22…` and `stage_e_fingerprint`
   `a30f9b52…` → `9914b172…`**, so runs 1–2 and run 3 are different instruments
   and no number crosses between them. Run-1 and run-2 artefacts are preserved
   under `*_v1_prefix.*` / `*_run2.*` rather than overwritten.
5. **Stamp**: `policy_trained` had been read from the *head*, so a trained
   utility head stamped the untrained Stage-D sampler as trained — while the
   same stamp's `notes` said it was untrained. `head_trained` is now its own
   field, and `selection` records which Stage-D path ran. Both
   `training_free_relevance` and an untrained policy trip `is_wiring_test`.

**Two expectations to hold loosely.** Many temporal golds are relative phrases
("the Friday before…") that no date rendering can capture, so temporal will not
close fully. And ceiling 5 bounds the whole table: if run 3 lands well short,
the residual is the extraction and reader ceiling, which the decomposition
reports rather than something further iteration removes.

### 1.10 Run 4: the raw tier was half a dialogue move, and the fix was selected offline

Run 3 measured **30.18 F1 / 25.46 BLEU-1** at coverage 0.802 — roughly double run
2 on every answerable category — and the change that did it was the raw-dialogue
tier. That makes *how well the tier retrieves* the thing worth improving next,
and it is measurable without the reader, which is what this section is about.

**The diagnosis, carried over rather than re-derived.** A CPU replay of run 3's
corpus split the questions by whether the raw tier had shown the reader at least
one of the question's own gold evidence turns, and read **answered-F1 0.496 with
against 0.154 without** — a 3.2× gap on the same reader, same prompt, same
graph. Those two numbers were measured in the session that ran the replay and
are quoted here as handed over, not re-measured; what *is* re-measured below is
the retrieval side of the same claim. The replay was necessary only because run
3's rows kept `raw_turns_included` as a **count**, which is FIX 4 below.

**Independent corroboration of the premise.** `scripts/raw_tier_grid.py` scores
run 3's own configuration at **0.4344** gold-turn coverage over 1,531 answerable
questions: the tier missed the evidence entirely on 57% of them. That is
consistent with a large conditional gap and is measured on a different quantity
than the replay, so the two do not lean on each other.

**Why a window, and not simply more turns.** A retrieved turn is half a dialogue
move. LoCoMo answers routinely sit in the *reply* to the turn that matched the
question, or in the setup line before it, so a top-k of isolated turns hands the
reader the question's vocabulary and withholds the sentence that answers it. The
grid separates the two hypotheses cleanly, and the window is the load-bearing
half:

| k | radius | cap | gold-turn coverage | Δ vs run 3 | mean texts |
|---|---|---|---|---|---|
| 3 | 0 | 15 | 0.4344 | — | 3.00 |
| 6 | 0 | 15 | 0.5291 | +0.0947 | 6.00 |
| 3 | 1 | 15 | 0.6584 | **+0.2240** | 8.09 |
| **6** | **1** | **15** | **0.7276** | **+0.2933** | **14.08** |
| 6 | 1 | 12 | 0.6976 | +0.2632 | 11.34 |
| 8 | 1 | 15 | 0.7289 | +0.2946 | 14.25 |
| 6 | 2 | 15 | 0.7100 | +0.2756 | 13.73 |

Doubling k buys +0.095; adding a ±1 window to the *original* k buys +0.224. The
grid also says where it stops: k=8 adds 0.0013 over k=6, and radius 2 is
**worse** than radius 1 because the cap starts dropping whole windows.

**The selection rule was written into the script before the grid ran** — highest
coverage, and among rows within 0.005 of it the one showing fewest texts — so
the row was chosen by the rule rather than the rule by the row. It selects
**k=6, radius 1, cap 15**, which is the configuration run 4 ships. Artefact:
`artefacts/raw_tier_grid.json`.

**This is the pre-registration, and it is the point of the section.** LoCoMo has
no dev split and every full run costs ~50 GPU-minutes, so picking a variant by
running the reader on each and keeping the winner would be tuning on the test
set. Coverage is a *retrieval* property the reader never touches: a variant that
cannot retrieve the evidence cannot be rescued by any prompt, and one that can
may still fail for reader reasons — which is the separation the five-ceiling
protocol exists to preserve. Nothing here moves ceiling 5.

**The prerequisite the plan did not contain, found while building it.**
`ChannelCache.turns_for` sorted a conversation's turns by `turn_id` — and turn
ids are `locomo/{conv}/session_{n}/{ix}` **strings**, so the sort is
lexicographic: `session_10` before `session_2`, `session_1/10` before
`session_1/2`. Measured on the pinned corpus, conv-26's timestamps are **not
monotone** under it. This is precisely the reordering `locomo.session_keys`
documents guarding against, reappearing one layer up because the ids were sorted
as text instead of the keys as integers. It was invisible in run 3, which only
ever ranked turns by score and never read a neighbour off an index — and fatal
the moment run 4 does, since the turn before `session_2/1` would be
`session_2/10`. Replaced by `chrono_key` = `(ts, session_id, turn_index,
turn_id)`, and neighbours are taken by position within a turn's **own session**
so a window can never straddle a session boundary and splice two exchanges weeks
apart into what reads as one. The `snap` fixture cannot catch this — its ids
happen to sort identically both ways — so the test builds the real id shape by
hand and asserts the old key's behaviour explicitly.

**The four changes, and one consequence each.**

1. **Window-expanded raw tier.** k 3→6, ±1 same-session neighbours, deduped,
   ordered **chronologically** (a dialogue reads by clock, and the temporal
   category needs it), capped at 15 texts with whole windows dropped
   lowest-scored-first. The unit dropped is the window: dropping a neighbour off
   a kept seed would leave exactly the half-move the window exists to complete.
   Each line keeps its `(8 May 2023) Speaker:` header.
2. **Budget rebalance.** Claims tier **512→256**, total evidence **1024→1280**.
   Once the raw tier carries the answering text the claims are the *citation*
   layer — they supply the `[c#]` ids `H` validated and the reader cites — so
   half the old cap buys the raw tier double the room inside a total that barely
   moves. Projected ~1.2k tokens/query, still ~7× under the reference system's
   ~9k, so the cost claim is intact. **256 and 1280 are not `BUDGET_LADDER`
   rungs**; the ladder (160/512/1024) is the declared cost-reporting axis and is
   untouched. The old `--evidence-budget` help called 1024 "a pre-declared
   BUDGET_LADDER rung", which conflated the total-evidence cap with the
   claims-serialisation ladder; corrected in place.
3. **`--fresh` truncates the rows file.** It unlinked only on the auto-numbered
   branch and appended on the explicit one. Run 3's committed JSONL carries
   **2,136 lines for 1,986 questions** — 150 stale rows from the pre-check
   slice. The artefact was clean, because scoring reads the in-memory results
   and the totals are exactly 1,540 + 446; the rows file was not, and anything
   aggregating it by line rather than by question id reads a run that never
   happened.
4. **Raw-turn ids in every row, not a count.** `raw_evidence_block` now returns
   the surviving turns, so the row carries `raw_turn_ids`. This is what turns
   the §1.10 diagnosis from a full CPU replay into a join against
   `locomo.evidence_turn_ids`. The general form: a diagnostic that records a
   *cardinality* where the question will be about *identity* costs a re-run to
   answer, and the cost is paid later, by someone who did not choose it.

**Ordering interaction, recorded because it is easy to miss.**
`raw_evidence_block` used to drop "from the tail first — the ranking already put
the most relevant first". Once the turns are chronological the tail is the
*latest* turn, not the least relevant one, so the same line of code would have
started dropping by recency while its docstring claimed relevance. The block now
takes a relevance `rank` and drops by it while keeping survivors in clock order;
without a rank the old tail-first behaviour is preserved for the ranked-list
callers. The test distinguishes the two rules rather than asserting the new one.

**Fingerprints move again.** The claims budget is part of what the stage-E
fingerprint covers, so **run 4 is a different instrument from run 3** and no
number crosses between them — the same rule §1.9 applied to runs 1–2 vs run 3.

**Two things run 4 does not fix, stated in advance.** Ceiling 5 — the frozen 3B
reader — bounds the whole table, and run 2 measured it at 0.137 exact / ~0.26
token-F1. Coverage 0.7276 is an upper bound on what the raw tier can contribute,
not a prediction of F1. And **run 3 was executed without `--ceilings`**
(`ceilings: null`, and 0 of 1,986 rows carry one), so the only ceiling table the
project has was measured under `PROMPT_SHA e023ea71…` on a 1,037-question
subset. Run 4 should carry `--ceilings`, or the decomposition that is half the
defensible claim stays attached to a retired instrument.

## 2. Departures from the plan as written

| §6 ref | As planned | What was built | Why |
|---|---|---|---|
| G2 | "no meter is spent in the read path" | metering confirmed present; the phase built the **governance** instead | §1.1 — the premise was a fix record misread as a defect record |
| criterion 3 | "retrieval and generation latency separately reported" | `assemble(ledger=..., stage=...)` plus dense-channel forward metering | Retrieval ran outside `answer()`, so its cost never entered the per-query snapshot. Mem-T's ~9k *includes* retrieval, so a generation-only figure was the flattering half |
| decision 1 | `--parse-obligations` opts in | flag removed, deferred by name | §1.3 — fix F7 makes it a separate pass, not a flag |
| §6 | Phase 11 signs its own §6 | **still UNSIGNED** | Needs the project owner's explicit instruction, per the `GATE0_CONTRACT.md` / Phases 3, 9, 10 convention |

**No decision was overturned by measurement**, because Phase 11 has run nothing
on GPU. Every finding above is a defect in the build, not a result.

---

## 3. Decisions taken, with what each costs

| # | Decision | Cost to change |
|---|---|---|
| D1 | This phase closes a **reference-table comparison**, not Gate 4 | Reversing = running baselines, ~8–15 h GPU |
| D2 | The comparable reference row is Mem-T **untrained**, 49.38 F1 / 44.11 BLEU-1, pinned and SHA'd before GRAFT had a number | Changing after seeing GRAFT's number is a contaminated §6b amendment |
| D3 | Metric convention is **token-F1 + BLEU-1**; LLM-judge rows stored but refused | Re-scoring; cross-paper conclusions move |
| D4 | Abstentions score **0** in the reference-comparable column | Changes every reported F1 |
| D5 | Cost unit is **total LLM tokens + LLM calls per query**, from the ledger; ingestion is a separate offline axis | Re-derives the Pareto table |
| D6 | Reader stays **Qwen2.5-3B**; the backbone difference is declared, not matched | Voids the stage-E fingerprint and runs R1–R3, and moves the reader out of the size regime where the packing benefit exists (`CLAUDE.md` §4.2) |
| D7 | Gate features are **recorded per question, gating off by default** | Re-running the reader pass to add an abstention analysis |
| D8 | Gold proof atoms are **Tier A**, from LoCoMo's evidence markers via `recall.tier_a_gold` | A different gold tier changes ceilings 4 and 5 |
| D9 | `EVAL_PREVALENCES` is **outside** the stage-G fingerprint | Moving it retrospectively marks `PHASE8_DECISIONS.md` §2's numbers as a different gate's |

### 3.1 D9, expanded — the one that could have caused pointless churn

Adding LoCoMo's base rate (446/1,986 = 0.2246) to `gate/pins.py`'s `PREVALENCES`
moved the stage-G fingerprint, because `frozen_values()` binds that dict. That
would have marked every number the MuSiQue Stage-A run reported as a different
gate's — **churn with no measurement behind it**, since a new evaluation
target's base rate changes neither the trained model nor any number that run
produced. It would also mean every future evaluation dataset invalidates the
trained gate's identity, which is the wrong dependency direction.

So it lives in `EVAL_PREVALENCES`, outside the fingerprint, asserted by test.

**The line that must hold:** this is one *published dataset statistic*, not label
access. The threshold must still be chosen on MuSiQue dev **reweighted** to this
prevalence — never by reading LoCoMo's own risk–coverage curve, which is fitting
to evaluation data and is the leak SubgraphRAG exposed in RoG.

---

## 4. What this phase built that no earlier phase had

Three things turned out to be missing once the phase was wired end to end:

* **`graft/ingest/locomo.py`** — the evaluation corpus loader, on
  `ingest/corpus.py`'s interface. Includes `probe()`, which checks every
  structural assumption in seconds of CPU. Ingesting LoCoMo costs ~43 GPU hours;
  discovering a structural mismatch at hour 30 is the expensive failure.
* **`scripts/locomo_eval.py`** — the end-to-end runner. `scripts/phase10_read.py`
  drives the read path over *hand-built fixtures* ("in the shape Stage C would
  deliver"), which is what made R1–R3 wiring tests. This one drives it over a
  real ingested graph, which is the only way a LoCoMo number exists.
  `graft/tests/test_locomo_eval.py` exercises the whole join on a stub reader, so
  a wiring error surfaces before 1,986 questions of reader time rather than after.
* **`scripts/train_head.py`** — the distilled utility head. `PHASE10_DECISIONS.md`
  §1.4 measured `sufficiency(X, ∅) = 1.0`, so `U` is vacuous at inference and
  best-of-K ranks by noise without a trained head. This is a two-layer MLP under
  a 200,000-parameter cap: **minutes of CPU**, against ~2 h per arm per seed for
  the Phase-9 policy ladder that answers Gate 3 and is not needed for an
  end-to-end number. It refuses LoCoMo examples structurally, because a head
  fitted on LoCoMo would void the zero-shot declaration invisibly — the weights
  look identical either way.

A 3-epoch smoke run of the head reported **dev Spearman ρ = 0.5928** against
exact `U`. Not a result — 63 rows, 8 examples — but the instrument runs.

---

## 5. What is still not built, by name

* **The Phase-9 policy ladder** (`N_real` = 200,000, ~2 h per arm per seed).
  Answers Gate 3. Not needed for an end-to-end number.
* **The three baseline adapters** — full-context, matched-budget RAG, Mem0.
  Deferred at the project owner's instruction on a three-day deadline;
  `GRAFT_PHASE11_BUILD.md` §7 records what each omission costs, including that
  Gate 4 item 4 stays unmet and that matched-budget RAG is the ~2 h addition that
  would most repair it.
* **Phase 8 Stage B** — a conversationally *trained* gate. The recorded features
  plus `EVAL_PREVALENCES["locomo"]` make post-hoc thresholding from a
  MuSiQue-trained gate possible, which is the cheap substitute, not the same
  thing.
* **LLM obligation parsing** — §1.3.
* **§6's signature** — §2's last row.
