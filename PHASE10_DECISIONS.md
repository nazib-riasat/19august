# PHASE10_DECISIONS.md — what the Phase-10 build decided and measured

**Status: Stages A, B and C complete as of 16 August 2026. Runs R1, R2 and R3
have run. Stage D — ceilings 1–2 at Tier B and every end-to-end number — remains
deferred on three separate blockers.**

**Read §1 before quoting any Phase-10 number.** Three of its findings changed a
result, and two of those were caught by measurement rather than by reading. §5 is
the adversarial-audit record: eight confirmed findings, two of them blockers, one
of which left the project's *claimed* cost axis with no numerator at all.

Companion to `GRAFT_PHASE10_BUILD.md` (§6 signed 16 Aug 2026). Where this file
and the build plan disagree, **this file wins and the plan is the record of what
was intended** — the Phase-4…9 convention.

---

## 1. What the build measured that reading could not settle

### 1.1 Citation markers counted as answer tokens — run R2

`normalise_answer` is SQuAD-style, so it strips punctuation. Run R2 handed the
frozen reader the gold proof and it answered **`"London\n[c1]"`** against gold
`"London"` — *correct* — and ceiling 5 scored it **False**.

Punctuation removal turned `[c1]` into the bare token `c1`, so the normalised
answer was `"london c1"`.

**Why this one mattered more than its size.** The error was **systematic and
one-directional**: it could only ever mark a *correct* answer **wrong**, and it
fired on every **cited** answer — which is every answer the prompt asks for. It
would have depressed ceiling 5, every end-to-end answer number, and the
orchestrator's contested check, uniformly and invisibly.

This module's own docstring had asserted the opposite — *"`normalise_answer`
already discards brackets when scoring"* — which is true of the brackets and
false of what they contain. Citation markers are now stripped **before**
normalisation, and `normalise_answer` carries the finding.

### 1.2 The prompt asked for a sentence; every benchmark's gold is a span

The first template said *"Answer in one short sentence."* Every benchmark this
project scores against — LongMemEval, 2WikiMultiHopQA, MuSiQue, LoCoMo — has a
**short span** as gold, so exact match against a sentence is ~0 **by
construction**, and decision 10's local exact/F-measure pair would have measured
verbosity rather than correctness.

Caught on the first smoke run, **before the prompt SHA froze into the stage-E
fingerprint**. The template now asks for the shortest answering phrase and
forbids a full sentence. Timing was the whole value here: after a reported run,
changing the prompt is a new experiment.

### 1.3 Run R3's "gold proof" was two atoms in content-hash order

The runner's fixtures never set `is_gold`, so `ProofExample.gold_atom_ids` came
back empty and the runner **silently fell back** to `sorted(pool.ids())[:2]` —
atoms in *content-hash* order. Measured: that selected one `Claim` node and one
`about_entity` edge, so on several questions the "gold proof" handed to ceiling 5
contained **no answer-bearing claim at all**. The reader correctly abstained, and
ceiling 5 read **0.6** as though that were a reader limitation.

It was noise. With gold set properly, ceiling 5 reads **1.0**.

**The fallback now raises instead of guessing.** A ceiling computed against
arbitrary atoms is not a ceiling, and silently producing one is worse than
producing none — the same reasoning `PHASE7_DECISIONS.md` §7 records about a
`--smoke` artefact quoted as measured.

### 1.4 `U` is vacuous at inference, so the orchestrator refuses to rank without the head

Measured while building the orchestrator: **`sufficiency(X, ∅) = 1.0`**. With no
gold, every set scores maximally sufficient and `U` collapses to its four
remaining terms.

That is exactly the circularity fix F1 exists to break — *"the reward must be
computable during training and some ranking signal must exist at inference, where
no gold exists"* — and the answer is the distilled utility head, never
`env.utility`.

So `answer()` **raises `UnrankableError`** when Stage D returns valid sets and no
head is supplied, rather than falling back to a number that looks like a utility
and is not one. The refusal is the only thing that keeps fix F1's split intact
against a caller in a hurry.

---

## 2. Departures from the signed §6

| # | Decision as signed | What was built | Why |
|---|---|---|---|
| 9 | answer equivalence = "SQuAD normalisation" | same, **with punctuation deleted rather than replaced by a space** | The official SQuAD script does `"".join(ch for ch in text if ch not in exclude)`. Substituting a space split one token into several and the article filter then ate any single `a` the split produced. §5 A2 |
| 9 | pin declared order `(casefold, strip_articles, strip_punctuation, …)` | `(casefold, delete_punctuation, strip_articles, …)` | The **pin** was wrong, not the code: stripping articles first leaves `"the-dog"` intact. The pin is what the fingerprint hashes, so it had to move |
| 2 | reader pinned by `model_id` | `model_id` **and** `revision` | A model id is a moving target; `graphbuild.pins.EMBEDDER` already pinned both. "Same frozen reader for every compared system" is unenforceable if the fingerprint binds a label rather than a checkpoint |
| 3 | the gold ordering is a diagnostic whose **gap** measures what honesty costs | added `serialize.ordering_gap` | Both orderings existed; the gap did not, so the decision was half-kept — the harder half to notice, because the code looked complete. §5 A4 |

**No decision was overturned.** That is a change from Phase 9, where four were —
and the reason is worth naming: Phase 10's §6 was signed against an architecture
section that had already been corrected twice (G1, G3), so the traps had been
found at planning time rather than at build time.

---

## 3. Runs

| Run | Cost | Result |
|---|---|---|
| **R1** — reader determinism | ~24 s load + 3 generations | **Bit-identical across 3 repeats.** Greedy decoding *is* reproducible here — measured, not assumed (G9). Peak VRAM **6.317 GB** on an 8 GB card, so fix F7's constraint is real |
| **R2** — ceiling 5 on the pilot | ~5 min | Found §1.1. Post-fix ceiling 5 = **1.0** |
| **R3** — end-to-end smoke | 36–119 s for 10 questions | The architecture's exit criterion. Found §1.3. Per-stage cost now real: `stage_e` 1 reader call / 200 tokens in / 6 out / 515 ms; `contested` **its own line** at 312 tokens |

**No training run anywhere in this phase.** The reader is frozen and post-hoc
faithfulness monitoring is declined (decision 11).

### 3.1 The five-ceiling decomposition worked, on a real case

R3's most useful output is not a ceiling value — the fixtures are trivial and
every ceiling reads 1.0 — but the **attribution**:

* ceiling 5 = **1.0**: given the 2-atom gold proof, the reader answers 10/10.
* end-to-end **answered 5/10**: 2 blocked by the placeholder gate, 3 where the
  reader declined Stage D's set.
* Stage D delivered a mean of **9 atoms** against a gold proof of **2**.

So the entire gap is attributable to Stage D's set selection, with an untrained
policy — which is exactly what Contribution 4 exists to produce, and it produced
it on the first run.

**One query is worth quoting.** On q3 the untrained policy delivered a
**one-atom** set. It was **formally valid** — `H` passed, `stop_allowed` *is* `H`
— and **semantically insufficient**, and the reader declined it. That is v1.2
§4.1's naming discipline observed in the wild: *"A set can satisfy every schema,
ID, interval and scope constraint and still be semantically insufficient."* The
system produced the case and handled it correctly on both sides.

---

## 4. What the read path is, structurally

* **Two abstention routes, never summed.** `gate` and `fallback` are different
  events with different owners; `aggregate()` reports the split and there is no
  single summed rate to quote. `PHASE5_DECISIONS.md` §1's quarantine-cause lesson,
  two phases on.
* **A third route exists and did not get a new cause.** The reader can decline a
  formally valid set (`INSUFFICIENT EVIDENCE`). `ABSTAIN_CAUSES` is a frozen
  two-member vocabulary, so this is recorded as `fallback` with the distinction
  kept in `stages.reader_declined` — widening a frozen vocabulary from inside the
  orchestrator would have been the wrong fix.
* **Abstentions are excluded from means and counted separately** (decision 7).
  Imputing 0 makes an abstaining system look like a wrong-answering one; dropping
  the query makes abstention free.
* **Fix F7 by construction, not by ordering.** The gate decision is computed
  *before* the reader loads and passed in, so no code path holds a gate model and
  a 3-billion-parameter reader at once. `ModelSlot` refuses a second concurrent
  slot, and a test asserts it.
* **`graft.reader` is the sixth ML-allowed package**, with the tightest
  containment yet: torch reaches `read.py` and nothing else.
  **`graft.diagnostics` was deliberately not admitted** — the ceilings take their
  reader as an injected callable, so Contribution 4's instrument stays computable
  on a bare interpreter.

---

## 5. The 16 August 2026 adversarial audit — 8 confirmed, 2 refuted

Four independent reviewers over distinct dimensions, each finding then
adversarially refuted by a separate agent. **18 of 32 verifier agents were cut
short by a session limit**, so this is a *partial* pass and the count is a floor,
not a total — stated because an audit reported as complete when it was not is
itself a finding.

### The two blockers

**A1 — no ledger meter was ever spent in the read path, and no snapshot reached
any artefact.** G12 and exit criterion 11 require every `OutputRecord` to carry a
per-stage snapshot with `wall_clock_ms`, `model_forwards`, `llm_tokens_in/out`
and `terminal_checks`. Measured: `grep` for `.count(` across `graft/reader/`,
`graft/diagnostics/` and the runner returned **nothing**, and
`grep -c ledger_snapshot artefacts/phase10_read.json` returned **0**.

Worse than absent — **falsely zero**. `Ledger.stage()` initialises every meter to
0 and only wall clock is auto-accumulated, so a query that invoked a
3-billion-parameter reader twice reported `llm_tokens_in = 0`. A false zero reads
as a measurement.

**Why it was the most consequential finding in the phase.** `CLAUDE.md` §9 forbids
claiming accuracy over full-context and requires the project to win on
*"latency and token cost"*. As built, **neither axis was measured for GRAFT**, so
Phase 11's matched-budget comparison had no numerator on GRAFT's side — the one
comparison the project's fallback position depends on.

*Fixed* by metering **inside the wrapper**, matching this project's own standing
convention (`ingest/extractor.py` lines 505–508, `ingest/nli.py` line 168): the
`Reader` now holds an optional ledger and counts where the tokens are;
`run_portfolio` spends `model_forwards` for the policy work; `ReadResult.report()`
carries the snapshot, which is what the runner persists. Post-fix, an answered
query reports `stage_d` 44 forwards / 115 ms, `stage_e` 1 call / 200 tokens in /
6 out / 515 ms, and `contested` **its own** 312 tokens / 548 ms.

`terminal_checks` stays **0** and must: the portfolio constructs through the masks
where `stop_allowed` *is* `H`, so it owes none (`PHASE9_DECISIONS.md` §1.1). That
is a documented true zero, and the audit correctly flagged its own reviewer for
calling it false.

**A2 — `normalise_answer` replaced punctuation with a space where SQuAD deletes
it.** The official script does `"".join(ch for ch in text if ch not in exclude)`.
Substituting split one token into several, and the article filter then ate any
single `a` the split produced. Measured:

| input | as built | official SQuAD |
|---|---|---|
| `U.S.A.` | `u s` | `usa` |
| `Wal-Mart` | `wal mart` | `walmart` |
| `O'Brien` | `o brien` | `obrien` |
| `1,000` | `1 000` | `1000` |

So `answers_equivalent("U.S.A.", "USA")` was **False** where SQuAD says True.
**The same one-directional class as §1.1** — it can only mark a correct answer
wrong — reaching `answers_equivalent`, `token_f1`, ceiling 5 and the contested
check. Post-fix, all eight probes agree with the official script exactly.

### The six majors

| # | Finding | Fix |
|---|---|---|
| A3 | **Ceiling 5 was serialised at the live token budget**, so a gold proof that did not fit was scored as a **reader** failure when it was a **packing** failure | Serialised at an unbounded budget; `survives_live_budget` reports ceiling 4's separate question **beside** it. This one is the worst in kind: v1.2 §6.3's entire argument is that one number is uninterpretable because it could be any of five stages, and a ceiling absorbing the one below it re-creates that problem *inside the instrument* |
| A4 | **Decision 3's honest-vs-gold ordering gap was computable by no code path.** Both orderings existed; the comparison did not | `serialize.ordering_gap` returns both serialisations plus their rank correlation |
| A5 | `READER` pinned no revision | `revision` added, defaulting to `None` with the instruction to resolve it before any reported run |
| A6 | **Budget truncation was first-fit, not tail truncation** — measured kept indices `[0,1,2,3,5]`, skipping 4 — contradicting the method's own docstring. The U-shape's premise (Lost in the Middle, TACL 2024) is that head and tail carry the most weight, so first-fit spends the budget on the positions the evidence says matter *least* | `break` rather than `continue`; and the token-reclaim pass, which reintroduced first-fit by another route, now also breaks at the first atom that does not fit |
| A7 | The citation-strip fix matches only `[cN]`, so another citation form would re-introduce §1.1 | Accepted as scoped: the regex is anchored deliberately so prose containing the letter c is not a citation. Recorded rather than widened — a looser pattern would create false citations, which is the worse error |
| A8 | `resolve_citations` omits the pool when resolving provenance, so a cited **binding** atom resolves to no spans | Latent: no binding atoms exist on this track (`pins.VACUOUS_ON_WIKIPEDIA`). Recorded, not fixed, because the fix is untestable until bindings exist |

### One finding my own regression test then found

Writing the A6 regression exposed that **both** closure-repair sites tested
membership in `dropped_set` rather than in `kept`. Those differ whenever a
reference points at an atom that was *never a candidate*: `dropped_set` holds only
what the budget refused, so an edge whose endpoint was absent from the **input**
set passed the check and shipped a dangling citation.

Stage D emits closed sets, so this cannot arise on the live path — but the
serializer is also handed hand-built and gold sets, and **a guard that holds only
when its input is already correct is not a guard.**

It had been masked because the test fixture itself sliced a non-closed subset:
the repair's bug and the fixture's malformedness cancelled out. Both fixed.

---

## 6. Status against the build order

| Stage | State |
|---|---|
| **A** — pins, serializer, parser, ceilings 3–4, tests | **done**; stage-E fingerprint is the seventh in `verify_handoff.py` |
| **B** — reader wrapper, `ModelSlot`, R1, R2, ceiling 5 | **done** |
| **C** — orchestrator, ledger, contested, conventions, R3 | **done**; the architecture's exit criterion has run |
| **D** — ceilings 1–2 at Tier B, end-to-end at scope-c (R4, R5) | **deferred by name** on three blockers: Phase-9 step 6 (a trained Stage-D policy), scope-c ingestion, Phase-8 Stage B (a conversational gate threshold) |

**Tests: 1,235 passing.** Nothing committed.

**Everything R3 printed is stamped a wiring test**, because two consumed
artefacts are untrained or out-of-domain. `ReadPathStamp.is_wiring_test` is a
**disjunction, not a score** — any one placeholder invalidates the run — and both
the per-query records and the aggregate carry the warning.
