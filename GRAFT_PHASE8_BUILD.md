# GRAFT — Phase 8 Build Plan: the answerability gate (`graft/gate/`)

**The decoupled abstention decision — a small trained classifier that says
whether a sufficient proof exists in the graph for this question, run *before*
Stage D, evaluated as selective prediction.**

Date planned: 15 August 2026
Parent: `GRAFT_RESEARCH_PLAN_v1.md` v1.2 **§4.2** (the decoupled design and the
math that killed the alternative), §4.4 (routing: answerability is a learned
quantity and never enters `H`), §6.4 (the selective-prediction metric group),
§2.2 (the gate is half of **Contribution 2**) ·
`GRAFT_EXECUTION_ARCHITECTURE_v1.md` Phase 8 · `GATE0_CONTRACT.md` (signed
15 Aug 2026) items 5, 9, 10 · `DATASET_DECISION.md` §2 (Phase-8 rows), §1.2 ·
`PHASE7_DECISIONS.md` §6–§7 (the handoff this phase consumes, as corrected)
Status: **§6 SIGNED 15 Aug 2026; Stage A BUILT and green the same day** —
`graft/gate/` (pins, features, labels, adapt_musique, model, riskcov, decide) +
`scripts/phase8_gate.py` + 32 tests. Decisions 1–10 and 12 built as written; 11
is Gate-0 item 9's and its ingestion has not run, which is why the build is
staged. **Step 6 (the MuSiQue track end to end) has not been run** — CPU work,
currently deferred — and **Stage B is deferred by name** (criterion 13).
`PHASE8_DECISIONS.md` is the record and wins conflicts with this file. **The one prerequisite the
plan assumed was missing is now in place** (15 Aug 2026): MuSiQue-Full is
downloaded from the official archive, SHA-pinned per split in
`graphbuild.pins.DATASETS["musique_full"]`, and readable through
`loaders.load_split(..., root=loaders.MUSIQUE_ROOT)` — see the sign-off note
under §6 for what was verified.
Labels as everywhere: **[EVIDENCE]** = a named paper supports this, venue
stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = engineering
or mathematical judgment made here.

---

## 0. What Phase 8 is for, and what it is not

Plan §4.2 withdrew `ABSTAIN` as a flow action because the math is fatal: under
reward-proportional sampling `P(ABSTAIN|q) = R_abstain / (R_abstain + Σ_X R(X))`,
so abstention probability falls as the *number of alternative valid proofs*
rises — even at identical answerability — and no tuning fixes a denominator
that moves per question (**[EVIDENCE]** the sampling law is GFlowNet
Foundations, JMLR 2023; the unsoundness argument is the plan's own, §4.2).
The corrected design is four steps, and Phase 8 builds steps 1 and 4:

1. **a separate trained classifier** predicts whether a sufficient proof exists
   in the current snapshot for `q` — *this phase*;
2. Stage D runs only when the gate says yes — *Phase 9/10 wiring*;
3. budget exhaustion falls back to abstain, with the rate logged — *Phase 9/10
   wiring; this phase reserves the counter* (G5);
4. **selective-prediction evaluation** — risk–coverage, selective accuracy,
   abstention recall, false-abstention rate on answerable questions — *this
   phase builds the instrument and produces the dev curve*.

**[EVIDENCE]** Abstention is a first-class evaluated ability, not an edge case:
LongMemEval (ICLR 2025) scores it as one of five core long-term-memory
abilities. **[EVIDENCE]** The evaluation methodology is selective prediction
per Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*
(NeurIPS 2017): a confidence threshold traded against risk on a risk–coverage
curve, chosen on dev.

**What Phase 8 is not.** It is not the read-path orchestrator — that is
Phase 10's fix F7, and the architecture's own exit criterion ("gate integrated
into the read-path orchestrator") **cannot be met inside this phase** because
the orchestrator does not exist yet. That criterion is split here, explicitly
(G7): Phase 8 freezes and tests the callable the orchestrator will consume;
Phase 10 owns the integration. It is also not a reader, not a retriever, and
not a Stage-D component: the gate consumes Stage C's outputs and produces one
bit plus one probability.

**The supervisor's constraint holds trivially:** the gate is a logistic
regression or a 2-layer MLP (architecture Phase 8 — "deliberately small; the
gate's value is the *decision protocol*, not capacity"). The SLM is not
involved at all.

---

## 1. Nine gaps this phase must close

### G1 — "Answerable" has no single label source, and the three corpora mean three different things by it [ANALYSIS]

What the gate must predict is *GRAFT's* notion: **a sufficient proof exists in
the current graph snapshot** (plan §4.2 step 1). No public dataset labels that.
Three families approximate it, and the mapping from each is a **declared
adaptation** (the Gate-0 item-1 discipline: reported with its losses, never
presented as native supervision):

* **MuSiQue-Full contrast pairs** (**[EVIDENCE]** MuSiQue, TACL 2022):
  answerable/unanswerable pairs whose *question strings are byte-identical* —
  unanswerability comes from removing supporting paragraphs, so a classifier
  **cannot** score answerability from wording and is forced onto pool-side
  features. That is exactly the declared feature set (G2), which is why
  `DATASET_DECISION.md` §2 makes it the primary training interface (49,628
  questions). The adaptation loss: MuSiQue pools are Wikipedia paragraph sets,
  not GRAFT atom pools — G8 owns that mapping.
* **Evidence-deletion contrast pairs on LongMemEval's train slice**
  [ANALYSIS, recipe **[EVIDENCE-adjacent]**]: for a question whose evidence
  sessions are in the graph, build the pool twice — once normally
  (answerable), once with the evidence sessions' assertions excluded from the
  candidate scope (unanswerable **by construction**). This is FEVEROUS's
  documented construct-negatives-by-deleting-evidence recipe
  (`DATASET_DECISION.md` §2 names it for exactly this purpose), applied to
  conversation. It is conversation-native, costs zero annotation, and gives the
  gate contrast pairs whose features come through the *real* Stage-C stack.
  The adaptation loss: deletion-unanswerable is cleaner than
  natural-unanswerable — deployed unanswerables are not made by deleting
  evidence, and the write-up says so.
* **Natural abstention labels, evaluation only**: LongMemEval's 30 abstention
  questions and **LoCoMo's 446 adversarial questions**. Never trained on.
  `DATASET_DECISION.md` §1.2's arithmetic is the reason LoCoMo is the primary
  abstention testbed: at n = 30 one flipped decision moves a rate by 3.3
  points, which cannot carry a contribution; 446 is the 15× fix that costs
  nothing, since LoCoMo is already ingested for Phase 5. Mem0 (ECAI 2025) and
  Chain-of-Memory (ACL 2026) both *exclude* the adversarial category, so
  reporting it is a differentiator, not a comparability loss.

**Decision for §6:** train on MuSiQue-Full pairs + LongMemEval-train deletion
pairs; evaluate on LoCoMo adversarial (primary), LongMemEval's 30 (in-domain
check, interval-reported, never a point estimate), and the answerable test
slice (false-abstention). Each mapping's dropped/merged label classes reported
with the loader, per the item-1 pattern.

### G2 — The feature contract, and where each feature comes from [ANALYSIS]

The architecture's list, made concrete against Phase 7's *corrected* handoff
(`PHASE7_DECISIONS.md` §7.1 — per-channel scores exist in `assemble()`'s
report only since fix B1; this phase is their first consumer):

| Feature block | Source | Note |
|---|---|---|
| obligation-slot coverage | `Obligations` + the pool: anchor matched an entity? a `value_type`-compatible atom present? time constraint satisfiable against pool intervals? | fix F2's rule applies: the parser's measured slot quality (`PHASE5_DECISIONS.md` §5a — `entity_anchor` exact-match 0.167) is reported beside any coverage number |
| max/mean channel scores | `assemble()` report `channel_scores`, per channel over pool atoms | the audit-restored contract; absence of a channel key = 0.0 |
| pool shape | `pool_size`, node/edge/support counts, `cap_skipped`, `hits_refused_ineligible` from the pool report | `cap_skipped > 0` is a real answerability signal: evidence may exist beyond the cap |
| saturation | `closed_atoms_in_scope` vs `pool_cap` (closed-atom units, §7.2) | tells the gate whether retrieval selected at all |
| question embedding | the shared pinned bge-small (`graphbuild.pins.EMBEDDER`), same cache | **ablation-armed** — see below |
| tier-A gold coverage | **FORBIDDEN** | gold fields never reach the gate (G3) |

**The question embedding is a leakage risk and gets an ablation arm, not a
guess.** On MuSiQue pairs it is provably inert (byte-identical questions). On
natural eval sets, "unanswerable-sounding" wording could carry the label — the
gate would learn the *benchmark's writing style*, not answerability. Two arms,
`with_question` / `pool_only`, same seeds; if `with_question` wins on dev but
loses on the zero-shot LoCoMo transfer, that *is* the leakage measurement, and
the reported gate is `pool_only`. [ANALYSIS]

### G3 — The gold quarantine extends to the gate, structurally [ANALYSIS]

A gate that could read `has_answer` or `answer_session_ids` would *be* the
label. Phase 7's structural test quarantines gold to `retrieve/recall.py`; the
same test pattern extends to every `graft/gate/` module: **feature code may
see (obligations, pool, scores, reports); label derivation lives in one named
loader module** (`gate/labels.py`), which is the only gate module allowed the
gold sidecar, mirroring `recall.py`'s boundary. Asserted over the source, not
promised (Phase-7 exit criterion 7's pattern).

### G4 — Threshold selection under prevalence shift [ANALYSIS]

The threshold is chosen on a dev risk–coverage curve (**[EVIDENCE]** Geifman &
El-Yaniv, NeurIPS 2017). But the training prevalence is ~50% unanswerable
(contrast pairs are 1:1 by construction) while LongMemEval's natural rate is
6% (30/500) and scope c's stratified slice carries proportionally few. A
threshold picked at balanced prevalence and applied at 6% inflates false
abstention. So: the dev curve is computed **at both prevalences** (natural
reweighting of the same dev predictions — no extra data), the threshold is
picked on the natural-prevalence curve, and both curves ship in the artefact.
Calibration (Brier/ECE) is reported per §6.4's Stage-B convention; no
post-hoc calibrator in Tier 1 (a knob without an instrument).

### G5 — Two abstention paths, never one count [ANALYSIS]

Plan §4.2 has two distinct abstain routes: the gate says no (step 2), and the
budget-exhaustion fallback (step 3). Flattening them repeats the
quarantine-cause mistake `PHASE5_DECISIONS.md` §1 catalogues — two different
reasons, one inflated rate. Phase 8 defines the record shape now:
`OutputRecord.abstain_cause ∈ {gate, fallback}` reserved, the fallback counter
named in `gate/pins.py`, wired when Stage D lands. §6.4's "post-hoc fallback
trigger rate" is reportable only if the distinction exists from day one.

### G6 — Class balance: natural for evaluation, constructed for training, both reported [ANALYSIS]

Training pairs are 1:1 by construction — that is the point of contrast pairs,
and it is *stated*, not corrected by resampling (the Gate-0 item-6 rule: class
weights, not resampling, and the weights reported). Evaluation keeps natural
prevalence. No natural-frequency claim may be made from the training balance —
the same clause item 6 attaches to D2's over-sampled pool.

### G7 — The architecture's exit criterion names an orchestrator that does not exist yet [ANALYSIS]

"Gate integrated into the read-path orchestrator" is Phase 10 work (fix F7).
Declared split: Phase 8 freezes `gate.decide(question, obligations, pool,
scores, report) → GateDecision(p_answerable, answerable, features, threshold)`
— a pure function of Stage-C outputs, no model loading inside — and Phase 10
consumes it unchanged. The Phase-8 exit criteria (§5) carry the callable's
freeze and tests; the integration criterion transfers to Phase 10's list *by
name*, so it cannot silently vanish between plans.

### G8 — MuSiQue pools are not GRAFT pools: the feature adapter is the contract [ANALYSIS]

The gate's features must be computable identically for a MuSiQue contrast pair
and a conversational pool, or training and inference read different objects.
The adapter: a MuSiQue question's "pool" is its paragraph set (gold +
distractors; the unanswerable twin lacks the removed supporting paragraphs);
BM25/dense channel scores are computed over paragraphs by the same
`bm25s`/pinned-embedder stack; entity/temporal/slot-coverage features take
their **declared absent value** (0.0 with a presence flag), exactly as a
conversational question with no parsed slots does in Phase 7's runner
("degrade visibly, never silently"). The adaptation is one module
(`gate/adapt_musique.py`), its losses enumerated in its docstring and the
artefact: no entity channel, no temporal channel, no obligations parse.
**The presence flags prevent the degenerate shortcut** — without them the gate
could learn "features absent ⇒ MuSiQue ⇒ 50% unanswerable", a dataset
classifier rather than an answerability classifier. [ANALYSIS]

### G9 — What this phase can honestly measure today, and the numbers it must refuse [ANALYSIS]

The Phase-7 honesty stamp carries over whole: the pilot graph is
stand-in-constructed, 10 questions, saturated pools. A gate trained on
MuSiQue-Full pairs is *trainable today* (the corpus is on disk and
SHA-pinnable through `graphbuild.loaders.load_split`); the LongMemEval
deletion pairs and every conversational evaluation number wait for **scope-c
ingestion** (Gate-0 item 9: decided, not yet run). The build therefore lands
in two stages like Phase 4: **Stage A (now)** — modules, tests, the MuSiQue
track, the dev risk–coverage instrument, wiring smoke on the pilot graph with
per-question listing; **Stage B (post-scope-c)** — the conversational
training track and the decisive evaluation. No number from Stage A's pilot
wiring may be quoted as a gate quality result, and the artefact says so in its
own body.

---

## 2. Scope

**In:** `graft/gate/` (features, labels, MuSiQue adapter, deletion-pair
construction, model, risk–coverage instrument, decide callable, pins),
`scripts/phase8_gate.py`, tests, the G7 interface freeze.

**Out:** the orchestrator (Phase 10) · any Stage-D coupling (Phase 9) · the
reader · any retuning of `pool_cap`, `tau_nli`, fusion arithmetic or any other
frozen value · LLM-based answerability (an LLM judge here would put a second
model's opinion inside Contribution 2's control) · post-hoc calibrators
(Tier 2, by name, if Brier/ECE says so).

---

## 3. Modules

### P8.0 `gate/pins.py` — what Phase 8 freezes (do this first)

Model class and sizes, feature-block names and order, threshold rule,
training-budget dict (the `graphbuild.pins.TRAINING` shape), the two-arm
ablation names, `abstain_cause` vocabulary, and a **stage-G fingerprint**
(config-not-output, the G11 convention) printed by `verify_handoff.py` beside
the ingestion/Stage-B/Stage-C ones.

### P8.1 `gate/features.py` — the feature vector

`(obligations, pool, atom_scores, assembly_report) → (vector, names,
presence_flags)`. Reads `channel_scores` (Phase 7 fix B1), slot coverage,
pool shape, saturation. Deterministic order; names shipped with the vector so
an ablation is a column mask, not a re-featurise. **No gold field** (G3's
structural test).

### P8.2 `gate/labels.py` — the one gold-reading module

Deletion-pair construction on conversation (G1): for each train-slice question,
the answerable pool via the normal Phase-7 stack and the unanswerable twin with
evidence-session assertions excluded from `eligible_nodes`' scope. Splits via
`graphbuild.train.split_questions` (user-level, stratified, seed 20260813 —
Gate-0 item 5; a later update never leaks into an earlier graph).

### P8.3 `gate/adapt_musique.py` — the declared adaptation (G8)

MuSiQue-Full contrast pairs → `(features, label)` through the same
`features.py` code path, absent blocks flagged. SHA-pinned loading through
`graphbuild.loaders.load_split`; the loader artefact reports pair counts and
every dropped field.

### P8.4 `gate/model.py` — LR and the 2-layer MLP

Both declared in pins; LR is the baseline arm, the MLP the capacity arm
(architecture: "deliberately small"). The three P6.11 guards verbatim: seed
reaches initialisation, early stopping restores argmin-dev, no scorable dev
item refuses. Class weights (G6) in pins.

### P8.5 `gate/riskcov.py` — the selective-prediction instrument

Risk–coverage curve, **AURC** (primary, §6 decision 8), selective accuracy,
abstention recall, false-abstention rate on answerable questions, Brier/ECE;
dual-prevalence curves (G4); bootstrap intervals (Dror et al., ACL 2018 for
test selection; three seeds {13, 42, 7}).

### P8.6 `gate/decide.py` — the frozen callable (G7)

`decide(...)` → `GateDecision`. Pure inference: threshold from pins, model
passed in, no I/O. What Phase 10 wires; what Phase 9's fallback path reports
against (`abstain_cause`).

### P8.7 `scripts/phase8_gate.py` — runner + artefact

Trains the declared arms on the available tracks, produces the dev
risk–coverage artefact with the honesty stamp (G9), per-question listing,
determinism digest (`runtime.deterministic_view` — the Phase-7 pattern),
manifest, and the stage-G fingerprint.

### P8.8 `graft/tests/test_gate.py`

Per §5. Bare-interpreter with `StubEmbedder` except the MLP's smoke.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 0 | P8.0 pins + fingerprint | importable bare; in `verify_handoff.py` |
| 1 | P8.1 features | vector + names + flags deterministic on the pilot graph; gold-quarantine test green |
| 2 | P8.2 labels + P8.3 adapter | deletion twin provably unanswerable-by-construction on a fixture; MuSiQue pairs load SHA-pinned, byte-identical questions asserted |
| 3 | P8.4 model | guards tested; LR and MLP train on a fixture |
| 4 | P8.5 risk–coverage | curve + AURC verified against a hand-computed toy; dual-prevalence reweighting tested |
| 5 | P8.6 decide + P8.7 runner | one command, one artefact, digest stable across two runs |
| 6 | MuSiQue track end to end | trains within budget on this machine; dev curve produced; ablation arms both run |

Steps 0–5 need no GPU (LR/MLP on CPU; embeddings cached). Step 6 is hours, not
days. The conversational track is **Stage B, post-scope-c**, and is listed in
§5 as deferred-by-name.

---

## 5. Exit criteria

**The gate is sound**
1. `decide()` frozen, pure, tested; `GateDecision` carries `p_answerable`,
   the decision, the threshold and the feature names.
2. Gold fields structurally unreachable from every gate module except
   `labels.py` (G3), asserted over source.
3. Feature vectors are deterministic and identical across the MuSiQue adapter
   and the conversational path for shared blocks (G8), with presence flags.
4. The deletion twin of an answerable fixture question is unanswerable by
   construction and its pool lacks every gold atom (G1).

**Selective prediction is measured, honestly**
5. Risk–coverage + AURC + selective accuracy + abstention recall +
   false-abstention rate + Brier/ECE, three seeds, paired bootstrap, intervals
   (plan §6.4's group, in full).
6. Dual-prevalence curves in the artefact; threshold picked on the
   natural-prevalence curve (G4), rule stated in pins.
7. LongMemEval's 30 in-domain abstention questions reported with an interval,
   never a point estimate (`DATASET_DECISION.md` §1.2).
8. The two abstain causes are distinct record fields; the fallback counter
   exists and is zero until Phase 9 wires it (G5).

**Discipline**
9. Both ablation arms (`pool_only`, `with_question`) run under identical
   budgets; the leakage reading (G2) is in the artefact.
10. Class weights, training balance and every adaptation loss reported (G6,
    G8); no natural-frequency claim from constructed balance.
11. Every artefact number is per-question-listed under the G9 honesty stamp;
    pilot-graph numbers are machinery numbers, and the artefact says so.
12. Determinism digest equal across two runs; stage-G fingerprint in
    `verify_handoff.py`.
13. **Deferred by name to Stage B (post-scope-c):** the conversational
    deletion-pair training track and every decisive evaluation number.
14. **Transferred by name to Phase 10:** "gate integrated into the read-path
    orchestrator" (G7).

---

## 6. Decisions to lock before writing code — **SIGNED 15 August 2026**

**Signed:** Nazib Riasat — 15 August 2026.
*(Signed by the assistant at the project owner's explicit instruction, recorded
as delegated rather than presented as the owner's own hand — the same form used
for `GATE0_CONTRACT.md` and `GRAFT_PHASE3_BUILD.md` §6 earlier the same day.)*

**Every row's "Recommended" column is adopted as written**, with two
qualifications recorded rather than glossed.

* **Decision 11 is not adopted, because it is not this document's to adopt.**
  The scope corpus is Gate-0 item 9's; item 9 is decided (scope c, 200
  questions) but **ingestion at that scope has not run**. Nothing in build steps
  0–5 depends on it — the MuSiQue track is independent of the conversational
  corpus — which is why exit criterion 13 already defers the conversational
  track by name.
* **Decision 1's prerequisite was missing at signing and was closed the same
  day.** MuSiQue-Full is now downloaded, pinned and readable — see the
  sub-section below for what the wiring verified, including two things that were
  *not* obvious from the plan.

**Why the three sharp rows were signed as recommended, and not softened:**

* **#4 — the question embedding is an ablation arm, not a default.** This is the
  one with a real failure mode: on natural eval sets "unanswerable-sounding"
  wording can carry the label, so the gate would learn the *benchmark's writing
  style* rather than answerability. The recommended form already handles it the
  right way round — two arms under identical budgets, and `pool_only` is the
  **reported** gate if `with_question` wins on dev but loses the zero-shot LoCoMo
  transfer. That disagreement *is* the leakage measurement rather than a
  disappointment, which is what makes it worth keeping as a declared arm.
* **#6 — AURC is the primary.** One primary fixed in advance is item 10's
  discipline, and AURC is the selective-prediction metric the risk–coverage
  framing implies (**[EVIDENCE]** Geifman & El-Yaniv, NeurIPS 2017). The named
  safety secondary — false-abstention rate on answerable questions — is what
  stops a gate scoring well by abstaining freely.
* **#1 — training sources.** LoCoMo's 446 adversarial questions and
  LongMemEval's 30 stay **evaluation-only**; the cost column's "a leaked eval set
  is unrecoverable" is the reason, and it is the same rule
  `DATASET_DECISION.md` §3 verified across the literature.

**Adopted in the Phase-3/Phase-7 shape**: taken as written and recorded as such,
rather than leaving code to rest on an unsigned table. Decisions 3, 5, 7, 8 and
12 inherit frozen values or project-wide protocol (the seeds, the honesty stamp,
fix F2's reporting rule) and are not re-decided here.

### Decision 1's corpus, as wired — 15 August 2026

Fetched from the **official** archive named by the paper's own repo
(`StonyBrookNLP/musique`, `musique_v1.0.zip`, 272,049,578 bytes, dated May 2022),
licence **CC BY 4.0** read from that repository. Pinned per split in
`graphbuild.pins.DATASETS["musique_full"]`; files live under
`data/phase8/raw/musique/` (gitignored) and are read with
`load_split("musique_full", split, root=loaders.MUSIQUE_ROOT)`.

**Row counts: 39,876 train + 4,834 dev + 4,918 test = 49,628** — exactly
`DATASET_DECISION.md` §2's figure, which is the cheapest available confirmation
that the *Full* variant was fetched and not *Ans*.

**Two things the wiring found that the plan did not say:**

1. **`id` is the pair key, not a row key.** Both members of a contrast pair carry
   the **same** `id`. Any `{row["id"]: row}` therefore keeps one twin and
   discards the other — halving the corpus and destroying the contrast decision 1
   exists for. `loaders.musique_pairs` groups on it deliberately and a test
   demonstrates the trap (4 rows → 2 naive keys).
2. **MuSiQue ships JSON Lines; the Phase-6 three ship JSON.** `load_split` gained
   a `.jsonl` branch rather than a second reader, so the SHA check stays on one
   path.

**Verified on the real dev split**: 2,417 answerable + 2,417 unanswerable, 2,417
exact contrast pairs, **zero malformed**, and every pair's two question strings
byte-identical — the property G2's leakage argument rests on, now checked rather
than assumed. Registering the corpus did **not** move `stage_b_fingerprint`
(`DATASETS` is not in `frozen_values()`), so no Phase-6 run was re-identified.

**MuSiQue-Ans is deliberately left unextracted.** It is in the same archive and
is *Phase 9's* Stage-D source; the archive is kept at `data/musique/` so Phase 9
needs no second 272 MB download.

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | Training sources | MuSiQue-Full contrast pairs + LongMemEval-train deletion pairs (G1); LoCoMo adversarial + LongMemEval-30 evaluation-only | re-train; a leaked eval set is unrecoverable |
| 2 | Model class | LR baseline arm + 2-layer MLP arm, both ≤ the pins budget; report both, headline the dev-selected one | capacity confound in Contribution 2's control |
| 3 | Feature set | G2's table verbatim, presence flags mandatory | features are the claim; silent change = new gate |
| 4 | Question-embedding arm | ablation, not default; `pool_only` is the reported gate if transfer disagrees with dev (G2) | benchmark-style leakage read as skill |
| 5 | Threshold rule | dev risk–coverage at natural prevalence (G4) | every downstream abstention number moves |
| 6 | Primary metric | **AURC** on the primary eval; false-abstention rate on answerable questions as the named safety secondary | item-10 discipline: one primary, fixed in advance |
| 7 | Seeds / significance | {13, 42, 7}, paired bootstrap, intervals (frozen project-wide) | protocol invalidated |
| 8 | Class handling | natural eval prevalence; constructed 1:1 training balance stated; class weights not resampling (G6) | inflated macro numbers |
| 9 | Calibration | report Brier/ECE; no calibrator in Tier 1 | a knob without an instrument |
| 10 | Abstain-cause vocabulary | `{gate, fallback}` (G5) | flattened rates, the §1 lesson again |
| 11 | Scope corpus | **not decided here** — item 9's scope c, ingestion pending | the unforced-guess failure |
| 12 | Honesty stamp | Phase-7's discipline verbatim | machinery numbers quoted as results |

---

## 7. Explicitly not in Phase 8

No orchestrator; no reader calls; no Stage-D coupling beyond the reserved
fallback counter; no LLM judge of answerability; no post-hoc calibrator; no
retuning of any frozen value; no iterative retrieve-then-decide loop (the gate
is one pass over Stage C's output, by design and by plan §4.2); no quoting of
any pilot-graph gate number as a quality result.

---

## 8. What Phases 9 and 10 get from this, verbatim

* **`gate.decide()`** — frozen signature, threshold inside pins, pure.
* **`abstain_cause`** — the two-way vocabulary and the fallback counter Stage D
  increments on budget exhaustion (plan §4.2 step 3), so §6.4's fallback
  trigger rate is reportable from the first Stage-D run.
* **The risk–coverage instrument** — Phase 10/11's abstention evaluation reuses
  `riskcov.py` unchanged on end-to-end outputs; LoCoMo adversarial is the
  primary testbed there for the same G1 reasons.
* **The stage-G fingerprint**, fourth in the identity quadruple printed by
  `verify_handoff.py`.
