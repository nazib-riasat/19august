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

### 1.11 Run 5: three system changes, and the metric block that makes a run reportable

Run 4 measured **39.43 F1 / 33.70 BLEU-1** at coverage 0.925. This section
records what changed for run 5 and — the larger half — what was added so that a
run's numbers can be read at research-paper standard rather than quoted as bare
means. **Nothing here has been run**: run 5 is built and awaiting the owner's go.

#### The three system changes (A1–A3)

**A1 — the raw tier is now k=5 with a ±2 window, cap 25.** Run 4 shipped k=6/±1
at cap 15. The offline grid handed over for run 5 measures the new shape at
**0.76 gold-turn coverage against 0.75** for the incumbent — a small margin, and
recorded as such rather than dressed up.

That margin is worth reading against run 4's own grid, which found radius 2
**worse** than radius 1 (0.7100 vs 0.7276). The two do not contradict: run 4's
row measured radius 2 *at cap 15*, where a wider window is dropped whole to fit,
so the number described the cap and not the radius — §1.10 said as much at the
time ("a wider window, which the cap starts to fight"). Raising the cap to 25
lets the radius vary alone, and the deeper-but-fewer configuration then wins.
The run-5 candidates have been added to `scripts/raw_tier_grid.py`'s `GRID`
— `(5,2,25)`, `(6,2,25)`, `(6,1,25)`, `(4,2,25)` — so the comparison is
reproducible rather than only reported. **The 0.76/0.75 figures are quoted as
handed over, not re-measured here.**

**A2 — one prompt rule, and it is post-hoc.** Rule 4 now reads: *"If the question
is hypothetical or asks for a likely preference or outcome, answer with the most
likely short inference from the evidence — still a phrase, never a sentence."*

This was written **after seeing run 4's open-domain result (9.89 F1 against
46.63 single-hop)**, and that is disclosed rather than presented as foresight.
It is a post-hoc change, labelled run-5, and the honest reading of any
open-domain improvement in run 5 is that a change targeted at the category
improved the category. The rule restates the phrase contract deliberately: an
inference licence that permitted prose would undo rule 2 for exactly the
category it was written to help, and the test asserts the restatement rather
than assuming it.

**What run 4's intervals say about whether this can be measured at all.** The
bootstrap CI on open-domain F1 is **9.89 [5.4, 14.9]** at n=96 — a band nearly
ten points wide. A run-5 open-domain result inside that band is not evidence the
rule worked. This is the first case in the project where the new B1 machinery
changes what may be concluded, and it argues *against* the change's author.

**Fingerprints move:**

```
PROMPT_SHA   8121eb22…  ->  2dfef30b…
stage-E      9914b172…  ->  63c5a1ba…
```

Runs 1–2, run 3, run 4 and run 5 are **four different instruments**. No number
crosses between them.

**A3 — `raw_turns_included` is the ids.** Run 3 stored a count; run 4 added the
ids under a *second* key (`raw_turn_ids`) and left `raw_turns_included` an int,
so the field a reader reaches for first still answered "how many" when every
question asked of it was "which". The ids are the inclusion record now and
`raw_turns_count` rides beside them. The general form, stated because it has now
occurred twice one key apart: **a diagnostic that records a cardinality where
the question will be about identity costs a re-run to answer, and the cost is
paid later, by someone who did not choose it.**

*Note for anyone joining run-4 rows: they carry `raw_turn_ids` (list) and
`raw_turns_included` (int). Run-5 rows carry `raw_turns_included` (list) and
`raw_turns_count` (int).*

#### A4 — the junk triage, which found nothing to fix

All 40 `junk_answer` rows from run 4 were printed and classified. **No change
was made, and that is the finding.** Every one is bracket-fragments-only —
`[3][c1][c2]`, `[5],[6],[7],[8]`, `[ c5 ]`, `[č7]` — a reader emitting citation
syntax and no answer content.

The one shape that could have hidden a recoverable answer is a bare number
followed by citations (`[27][c5]`, `[31][c2]`, `[19][c3]`): 14 rows, and if the
number were the answer, `clean_answer` would be destroying correct output — the
same defect the module already fixed once for `[25 May 2023][c1]`. It was
checked against gold: **0 of 14 match.** The bare integers are citations in a
mangled format. `clean_answer` is classifying all 40 correctly, the rate is 2.0%,
and extending it would have been a change made because a change was expected.

#### The metric block (PART B)

Everything below lands in the artefact under `report_metrics`, is derived from
the rows alone, and therefore **costs no GPU and is recomputable from a rows
file after the fact**. Each metric ships with its definition string, because a
number whose definition lives only in a decisions document is a number that will
be misquoted.

| | What | The refusal that makes it honest |
|---|---|---|
| **B1** | 95% percentile bootstrap on F1 and BLEU-1, overall and per category — 10,000 resamples, seed 13, reported as `39.43 [37.5, 41.4]` | **No comparative test is computable.** Published systems' per-item outputs do not exist, only their table means, so there is no paired sample and no variance for them. The artefact says in writing that an overlapping interval against a quoted mean *is not a test*. |
| **B2** | Citation rate, citation resolution rate, span-grounded answer rate — ALCE's (EMNLP 2023) answer/citation separation | Span-grounding's denominator is **cited** answers, not all answers: "every citation resolves" is vacuously true of an answer citing nothing, and counting those would reward not citing. A hallucinated `[c9]` counts against resolution. NLI-backed precision is **declined, not approximated** — a resolution rate reported as precision overstates verifiability by exactly the entailment gap VeriCite measures. |
| **B3** | Selective QA: risk–coverage, AURC, adversarial abstention at the operating point (Geifman & El-Yaniv, NeurIPS 2017) | The threshold is the **MuSiQue-dev one, transferred unchanged**. An oracle threshold chosen on LoCoMo's own curve is computed and named `oracle_threshold_leaked_do_not_report`, so the transfer gap is visible as a diagnostic and cannot be quoted as a result. |
| **B4** | `--ceilings-sample N` — seeded (13) stratified sample across 4 categories × 10 conversations | Stratified, not uniform: ceilings 1–2 are conversation-level and the categories differ ninefold in size, so a uniform 100 would be mostly single-hop from the largest conversations and the mean would describe the sample's composition. |
| **B5** | Cost per query, throughput, and the hardware it was measured on; Pareto pairs against the reference table | **Only Mem-T publishes a token count.** Every other row is `not_published` and is *not* estimated. GRAFT's cost advantage is the one axis it actually wins; a frontier drawn through invented denominators would put it on fabricated ground. |
| **B6** | Corpus SHA, prompt SHA, stage-E fingerprint, config hash, seeds, decoding, determinism | Cross-machine byte-identity is **not promised**. Greedy is deterministic given identical inputs, dtype, kernels and batch composition — four conditions, one of which has already surprised this project. |

**Two structural changes came out of writing this.** `scripts/phase8_gate.py`
now persists the winning arm (`phase8_gate.pt`) and exposes `load_gate` — the
checkpoint gap §1.8 named, which until now made "apply the gate" mean "retrain
it first". Selection is by the **declared** primary (AURC) and the saved seed is
the **median** of three by that primary, not the best: a best-seed checkpoint
would make every downstream number an optimistic outlier. And the runner's
argparse is extracted into `build_parser()`, so the run-5 defaults are asserted
by a test rather than checked by reading — which is how run 4's
`--evidence-budget` help came to describe a `BUDGET_LADDER` rung it was not.

**The claims section is updated.** Primary is now cost, citations, the
selective-QA curve, and the ceilings — the four axes that need no baseline or
that no reference row reports. F1/BLEU-1 stay secondary and now travel with
their intervals.

#### What has not been run

`scripts/phase8_gate.py` has not been re-run, so no `phase8_gate.pt` exists yet
and `scripts/locomo_gate_posthoc.py` will refuse with the command to produce one.
The B4 ceiling sample has not been run at N=100. Run 5 itself has not been run.
The A1 token-per-query projection (~1.3k, under a 1,400 ceiling) was
**verified on 21 Aug 2026 and missed**: measured mean 1,493 on a 20-question
smoke, because the projection counted only the evidence block and not the ~210
tokens of prompt scaffolding around it. See §1.12.

### 1.12 The gate transfers at chance — and three defects found getting there

Run 5's build (§1.11) was followed by executing the three cheap steps it
unblocked. This is their record. **The headline is a negative result and it is
the most informative thing in this section.**

#### The gate, trained and persisted (B3-i)

`scripts/phase8_gate.py`, 31.4 min CPU, 39,876 train / 4,834 dev contrast pairs
from MuSiQue-Full, 435 features, 3 seeds × 2 models × 2 arms. Re-run once after
a fix; both runs produced **identical** AURC to four decimals, which is the
seeding working.

| arm/model | AURC (natural) | std | contrast pair-acc | features |
|---|---|---|---|---|
| pool_only/lr | 0.0405 | 0.0002 | 0.7876 | 50 |
| pool_only/mlp | 0.0395 | 0.0003 | 0.7840 | 50 |
| with_question/lr | 0.0387 | 0.0002 | 0.8010 | 435 |
| **with_question/mlp** | **0.0382** | 0.0006 | 0.7944 | 435 |

Pair accuracy ~0.78–0.80 against a 0.5 chance line. On its own corpus the gate
works.

#### Defect 1 — the winning arm cannot be applied to LoCoMo, and the names all matched

The first checkpoint saved only the winner, `with_question/mlp`. **LoCoMo eval
rows carry all 435 feature names — including 384 `q_emb_*` columns that are
every one exactly 0.0**, because the runner records
`gate_blocks_present.question_embedding: False` and never populates them.

The dangerous property is that a feature-*name* comparison passes cleanly. All
435 names match, in order. Only the values are empty, so the gate would have
read 384 zeros it was trained to use and returned entirely normal-looking
probabilities. This is `PHASE8_DECISIONS.md` §3.3 one corpus later: a feature
block meaning something different to its consumer than to its producer.

Three fixes: **every arm is persisted** (saving only the winner made the
applicable arm unavailable without a 32-minute retrain); each checkpoint carries
`requires_blocks`, matched against the row's own `gate_blocks_present`; and a
second, value-level guard that trusts the numbers rather than the flag. The
value guard is the one that actually fired — the first checkpoint predated
`requires_blocks` — which is why there are two.

#### Defect 2 — a defect that was not one, and the correction is the point

Ranking the gate on LoCoMo, `predict()` appeared to return **0.0 for all 1,986
questions**. That was written up as a float32 sigmoid underflowing at logits of
−81, destroying the ranking and making the reported AURC of 0.2181 a curve over
ties.

**It is false.** `predict` returns **1,974 distinct values across 1,986 rows and
no exact zeros**; the smallest is 7.66e-36, comfortably inside float32's
subnormal range. The "collapse" was a `:.4f` print format rendering 1.3e-14 as
`0.0000` — in a diagnostic written to inspect the very quantity it then
misreported. **The AURC computed before the "bug" was found was correct, and is
unchanged after.**

This is §1.1's pattern exactly, and it is recorded rather than quietly reverted
because the failure mode is what generalises: *a formatting artefact read as a
finding, in the tool built to look for findings.* The project has now done this
twice. The cheap defence both times would have been to print one raw value
beside the formatted one.

The logit ranking was **kept**, with an honest justification replacing the false
one: rank-based metrics are invariant to the monotone sigmoid, so it is
lossless and keeps headroom off-distribution — and computing the probability in
float64 alongside is what makes `probability_max`, and therefore the unreachable
threshold below, visible in the artefact at all. `graft/gate/model.py` is
correct and was not touched.

#### Defect 3 — a real bug, found by the test written for the false one

Writing the regression test for defect 2 exposed a genuine bug in the new
`auroc`: with `argsort` tie-breaking, **a gate that scores every question
identically reads as AUROC 0.0 or 1.0 depending only on input order** — a system
with no ranking at all scoring as a perfect one. Fixed to average ranks over
ties, so no ranking reads as exactly 0.5. That degenerate case is precisely the
one a transfer study is most likely to hit, which is what makes it worth the
line.

#### The result: the MuSiQue gate does not transfer to LoCoMo

`scripts/locomo_gate_posthoc.py` on run-4's 1,986 rows, `pool_only/mlp`, CPU,
no reader:

| | |
|---|---|
| **AUROC (threshold-free separation)** | **0.5050** — chance is 0.500 |
| AURC, reweighted to prevalence 0.2246 | 0.2181 |
| Transferred threshold (MuSiQue dev) | 0.6039 |
| Highest probability on LoCoMo | **1.32e-14** |
| Coverage at the transferred threshold | **0** |

Two independent readings, agreeing. **AUROC 0.5050 says the gate cannot order a
LoCoMo answerable question above an adversarial one** — 0.5 percentage points
above chance, on 1,540 × 446 pairs. And the MuSiQue-dev threshold is not merely
badly calibrated but **unreachable**: every LoCoMo probability is below 1.3e-14
against a threshold of 0.60, so the gate declines all 1,986 questions and
coverage is 0 by construction.

Adversarial abstention at that point is 1.000 and false abstention is also
1.000. Neither is a result; they are the same fact twice — the gate answers
nothing.

**This is the Wikipedia→conversation transfer claim being measured, and failing.**
`CLAUDE.md` §7 lists it as declared and untested; Phase 9 discharged the
*plumbing* and said explicitly that the hypothesis was untouched. This is the
hypothesis, on the gate, and the answer is no. It is a clean negative: the gate
is good on its own corpus (pair accuracy 0.79), the features are live and
sensibly scaled on LoCoMo, and the failure is squarely in the transfer.

**What is not claimed.** That a gate cannot work on LoCoMo — only that *this*
one, trained on MuSiQue contrast pairs, does not. The obvious next move is the
conversational track (Phase-8 Stage B), which trains on LongMemEval's own
evidence-deletion pairs and is deferred by name rather than blocked. The oracle
threshold chosen on LoCoMo's own curve is computed and stored as
`oracle_threshold_leaked_do_not_report`, so the gap is visible as a diagnostic
and cannot be quoted as performance.

#### Step 3 — the token check failed its target, and the target was wrong

The run-5 configuration was to be verified under 1,400 tokens/query on a
20-question smoke. **Measured: mean 1,493, median 1,530, max 1,619.**

The cause is arithmetic, not a leak. `--evidence-budget 1280` caps the
*evidence block*, and it is enforced exactly; but tokens/query counts the whole
prompt, and the instructions, format examples, the new rule 4, the question and
the answer add ~210 tokens on top. The ~1.3k projection omitted the scaffolding.

**`--evidence-budget` was left at 1280.** Lowering it to hit a derived target
would be tuning an explicitly specified parameter to rescue an estimate, and the
claim the target protects survives: 1,493 against the reference system's ~9,000
is **6.0× fewer tokens**, down from run 4's 7.88× but intact. Recorded here so
the ratio moving is on the record rather than noticed later in a table.

### 1.13 Run 5 measured — and the intervals immediately earn their keep

`results/locomo_eval3.json` / `locomo_eval_rows3.jsonl`, 1,986 questions,
99.2 min, `--ceilings`, `PROMPT_SHA 2dfef30b…`, stage-E `63c5a1ba…`.

| | Run 4 | **Run 5** | Δ |
|---|---|---|---|
| overall F1 | 39.43 | **41.52** [39.6, 43.5] | +2.09 |
| overall BLEU-1 | 33.70 | **35.38** [33.5, 37.3] | +1.68 |
| coverage | 0.925 | **0.942** | +0.017 |
| single-hop F1 | 46.63 | **50.24** [47.5, 53.1] | +3.61 |
| multi-hop F1 | 29.75 | **27.44** [23.8, 31.1] | **−2.31** |
| temporal F1 | 37.91 | **39.55** [36.1, 43.0] | +1.64 |
| open-domain F1 | 9.89 | **13.02** [7.6, 19.0] | +3.13 |
| adversarial abstention | 0.193 | **0.175** | −0.018 |
| tokens/query | 1,142 | **1,416** | ratio 7.88 → **6.35** |
| ceiling 5 (reader) | 0.1224 | 0.1345 | +0.012 |

**§1.11 predicted this section's most important sentence before the run, and it
was right.** It said the open-domain interval was so wide that a run-5 result
inside it would not be evidence the A2 prompt rule worked. Measured: **9.89
[5.4, 14.9] → 13.02 [7.6, 19.0]**. The intervals overlap across more than half
their width. **The prompt rule is not established by this run.** The point
estimate moved in the intended direction on the category it targeted, which is
exactly the shape of result that a post-hoc change produces whether or not it
works, and n=96 cannot separate the two.

Recorded because the discipline only counts when it costs something: the change
was mine, the movement flatters it, and the instrument says no.

**Multi-hop went down**, 29.75 → 27.44, and its intervals overlap too
([25.9, 33.7] vs [23.8, 31.1]) — so that is not established either. It is the
one category where a *deeper but narrower* raw tier (k 6→5, radius 1→2) has an
obvious mechanism to hurt: a multi-hop question needs evidence from two places,
and five seeds reach fewer places than six however deep each one goes. Worth a
targeted grid row before k is reduced again; not worth a claim now.

**What is established**: overall and single-hop. Single-hop's intervals
([43.8, 49.5] → [47.5, 53.1]) barely overlap and its n=841 is the only category
with the power to say so; overall F1's ([37.5, 41.4] → [39.6, 43.5]) likewise
sit mostly apart. The gain is real and it is concentrated where the window
mechanism should help — one retrieved turn plus its reply.

#### The citation metrics, measured for the first time (B2)

| | |
|---|---|
| citation rate | 0.774 |
| citation resolution rate | **0.787** |
| span-grounded answer rate | 0.735 |
| citations emitted / unresolved | 1,932 / **411** |

**411 of 1,932 citations — 21% — name no claim that was shown.** That is a
reader-ceiling finding of exactly the kind ALCE (EMNLP 2023) is about, it is
measured on this project's own axis rather than borrowed, and **no row in the
reference table reports anything comparable.** It is now the strongest of the
primary claims after cost, because it is a property of the system that a
baseline comparison cannot take away.

It also puts a number on something the five-ceiling table could not: roughly
one answer in four is not fully traceable to evidence the checker validated,
*even when the answer is correct*. Nothing in F1 shows this.

#### Cost

1,416 tokens/query against the reference system's ~9,000 — **6.35×**, down from
run 4's 7.88×. Throughput 1,201 q/h on a single RTX 5050 Laptop, 8 GB, bf16.
The cost claim is intact but it has now moved twice in one direction, and the
raw tier is what is buying the accuracy. That trade is the finding, not a
footnote: run 3 → run 5 is +11.3 F1 for +52% tokens, and every step of it came
from showing the reader more raw dialogue.

#### The ceilings are unchanged, which is the point

0.674 / 1.000 / 0.737 / 0.998 / **0.1345**, on the same 1,037-question
eligible subset as run 4. Ceilings 1–4 are identical to four decimals — they are
properties of the graph and the packer, and neither moved because neither was
touched. Ceiling 5 moved +0.012, within noise on 1,037 questions.

**The reader remains the binding constraint**, and run 5's end-to-end 41.52
against a reader ceiling that permits ~0.26 token-F1 on gold proofs says the
retrieval side has now closed most of the distance it can. Further raw-tier
work has little room left above it.

### 1.14 Matched-budget RAG: the first re-run baseline — 12 Sep 2026

**Gate 4 item 4 is now partially met.** `scripts/locomo_rag_baseline.py` re-runs
matched-budget RAG rather than quoting it: same frozen reader, same
`PROMPT_TEMPLATE` (SHA stamped), same decoding, same **1,280-token** total
evidence budget as run 5, same retriever over raw turns (`top_raw_turns` +
`expand_windows`, run 5's k = 5 / radius 2 / cap 25), same chronological
ordering, same `clean_answer`, same `build_report` / `report_metrics` — all
**imported from the runner**, so the baseline cannot drift from the system it is
compared with. The one system difference: no graph, no claims tier, no `H`, no
Stage D; the whole budget is raw dialogue. One design decision: RAG passages are
numbered `[c1]…` so prompt rule 5 is satisfiable (GRAFT's raw tier is uncitable
because its ids belong to checker-validated claims; a baseline has none).
F1/BLEU strip citations, so this moves nothing. Three chunks with cool-downs,
1,986/1,986 questions, 31 min GPU. `results/locomo_rag_baseline.json`.

| | RAG (re-run) | GRAFT run 5 | Mem-T RAG (quoted) |
|---|---|---|---|
| overall F1 | 39.87 | **41.52** | 41.59 |
| overall BLEU-1 | 33.95 | **35.38** | — |
| tokens / query | 1,252 | 1,416 | ~9,000 |
| adversarial abstention | 0.195 | 0.175 | — |

**Paired on the same 1,540 answerable questions: +1.65 F1, 95% CI [+0.24,
+3.05], P(≤ 0) = 0.011.** GRAFT wins 267, RAG 240, tie 1,033. By category:
single-hop **+3.86 [+1.42, +6.43]** is the only interval clearing zero;
multi-hop +1.45 [−0.78, +3.63], temporal +0.56 [−1.95, +3.00], open-domain
+0.54 [−3.17, +4.80]. **Read: at matched budget, adding checker-validated claims
to RAG gives a small, significant gain concentrated in single-hop, at ~13% more
tokens. It does not measurably help the reasoning categories.**

**The instrument validated itself.** The re-run RAG lands 1.7 points from
Mem-T's published RAG row with a smaller backbone — independent evidence the
harness scores the way the field does.

**Also on 12 Sep:** Stage C's GNN scorer trained (`artefacts/stage_c_scorer.pt`,
237,443 params, 20 epochs, dev loss 0.0046 on 2 held-out questions — on the
**LongMemEval pilot's** distant signal, never LoCoMo; 8 training questions, so
modest by construction); §6 of this phase signed, delegated and marked
contaminated; Gate 1 ran and lost (`PHASE6_DECISIONS.md` §9); D3/D4 heads
pretrained (§9.4 there).

**Where the thesis now rests.** Gate 2 `inconclusive`, Gate 1 negative at 125
items: neither learned contribution is supported. What stands was never gated on
learning — the controlled cost result above, the five-ceiling decomposition, and
the adversarial subset. Full-context is the baseline still owed (§5).

### 1.15 A second dataset row — 2Wiki dev — and a mislabelled run caught before it was recorded (12 Sep 2026)

**2WikiMultiHopQA dev, 500 questions stratified by type, full read path.**
`scripts/wiki2_eval.py`: `wiki2.build_one` (paragraph claims, `about_entity`
edges from titles, five-channel fused scores) -> `RealEnvironment` -> `answer()`
with the frozen reader, the trained utility head as scorer, and Stage D by
`training_free_relevance`. Claims tier only -- 2Wiki has no dialogue and its
claims *are* the paragraphs. SQuAD token-F1 / EM through `normalise_answer`.
Subset drawn by `stratified_sample` over `row["type"]` at the pinned seed, so it
is reproducible and not a head slice. 9 min GPU. `results/wiki2_eval.json`.

| | n | F1 | EM | coverage |
|---|---|---|---|---|
| **overall** | 500 | **28.21** | 12.20 | 0.870 |
| bridge-comparison | 109 | 36.09 | 9.17 | 0.72 |
| comparison | 121 | 30.24 | 6.61 | 0.83 |
| compositional | 209 | 23.65 | 17.70 | 0.93 |
| inference | 61 | 25.74 | 9.84 | 1.00 |

820 tokens/query, 1 LLM call. **Gold-complete on 100% of pools** -- the answer
was always in the pool, so the gap is Stage D selection plus the 3B reader, not
retrieval. 65 abstentions, all `fallback`. The EM/F1 inversion is the readable
part: compositional has the lowest F1 and the highest EM -- short entity answers
the reader nails or misses -- while comparison answers earn partial credit.

**What this is.** A *within-system* row. Published 2Wiki systems train on 2Wiki
train; GRAFT's head saw 200 rows and everything else is training-free. It is not
a controlled comparison and `is_wiring_test` stays True. **Why it matters
anyway:** 2Wiki is the source domain of the Wikipedia->conversation transfer
claim -- Stage D trains here -- so 28 F1 here is the floor LoCoMo's 41.5 should
be read against.

**A run mislabelled as LongMemEval, caught before it entered this record.** Item
D of the remaining-work table was "LongMemEval eval". The only ingested
LongMemEval graph is the 10-question Phase-5 pilot, and the runner chained under
that label -- `scripts/phase10_read.py` -- evaluates its own **hand-built Ada
Lovelace fixtures**, as its docstring says ("wiring this runner to [the pilot
graph] is Stage D of the plan, blocked on scope-c"). What ran was a re-run of
Phase 10's R3 wiring test: 10 fixtures, 5 answered, all five ceilings 1.0,
stamped `WIRING TEST`. It is filed as `results/phase10_read_fixtures_rerun.json`
and is **not a dataset row**. Recorded because the failure is reusable: a
correct script under a wrong label is the `PHASE7_DECISIONS.md` §7 smoke-quoted-
as-measured class, and the artefact name is where it was caught.

**Consequence.** There is no LongMemEval row and none is reachable without
scope-c ingestion (~32 h GPU) or a purpose-built runner over a 10-question
graph, which would be noise. A conversational gate (Phase 8 Stage B) is blocked
on the same ingestion. Dataset rows in hand: LoCoMo (41.52, plus the controlled
RAG comparison of §1.14) and 2Wiki (28.21).

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
  thing. **§1.12 measured that substitute: the MuSiQue gate does not transfer**
  (coverage 0 at the transferred threshold), so this is now the only route to an
  adversarial-abstention number above the 0.17–0.19 the ungated runs sit at --
  and it is blocked on scope-c ingestion (§1.15).
* **A LongMemEval row** — blocked on the same ingestion; the 10-question pilot
  graph is the only LongMemEval data ingested (§1.15).
* **LLM obligation parsing** — §1.3.
* **§6's signature** — §2's last row.
