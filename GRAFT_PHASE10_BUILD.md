# GRAFT — Phase 10 Build Plan: Stage E — reader, orchestrator, ceiling oracles

**The read path becomes a system.** Stage C retrieves, the gate decides, Stage D
selects, and — for the first time — something *answers*. This is also where the
five-ceiling protocol (Contribution 4) stops being a design and becomes five
runnable functions.

Date planned: 16 August 2026
Parent: `GRAFT_RESEARCH_PLAN_v1.md` v1.2 **§3.5** (Stage E), **§4.2**
(the decoupled abstention), **§6.3** (the five ceilings), **§6.4** (metric
groups, "End-to-end" and "Reporting discipline") ·
`GRAFT_EXECUTION_ARCHITECTURE_v1.md` **Phase 10** (§10.1–10.3) and **fix F7** ·
`GATE0_CONTRACT.md` items 3, 5, 10 · `PHASE8_DECISIONS.md` (the gate this wires,
and its criterion-14 transfer) · `PHASE9_DECISIONS.md` §7.8 (three runner
conventions transferred here by name) · `DATASET_DECISION.md` §4 (the reader's
measured cost)
Effort: **~2 weeks solo [ANALYSIS]** across four stages, of which **Stage A
needs no GPU and is unblocked today**.
Status: **planned; §6 SIGNED 16 August 2026 (delegated). No code.** `graft/reader/`,
`graft/diagnostics/` and `graft/baselines/` exist as empty Phase-0 scaffolds
carrying only their docstrings.

Labels as everywhere: **[EVIDENCE]** = a named paper supports this, venue
stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = engineering
or mathematical judgment made here.

Gaps are numbered **G1–G12**, matching the Phase-0…9 convention.

---

## 0. What blocks this, and what compute it needs

### 0.1 The dependency answer, checked rather than assumed

**Phase 10's code has no blocking dependency — not even Phase 3.** That is a
better position than the question assumed, and it is worth stating precisely,
because three *different* things get called "Phase 10" and only the last is
blocked:

| | Needs | Blocked? |
|---|---|---|
| **The code** — serializer, parser, reader wrapper, orchestrator, five oracles | interfaces that all exist today | **No.** Buildable and fixture-testable now |
| **The smoke test** the architecture sets as the exit criterion | the Phase-5 pilot (248 turns, on disk) + reader forward passes | **No.** Needs GPU *inference*, not training |
| **The decisive numbers** — end-to-end accuracy, abstention P/R/F1, citation P/R | a trained Stage-D policy, a conversational gate, scope-c data | **Yes — three separate blockers, listed below** |

**The three real blockers, and none of them is one thing:**

1. **Phase-3 step 6 → Phase-9 step 6.** β freeze unblocks Stage-D training,
   which produces the policy the orchestrator calls. *(β froze 15 Aug 2026 —
   `PHASE3_DECISIONS.md` §7 — so this chain now waits on Phase-9 step 6 alone.)*
   Until then the orchestrator
   runs with an **untrained** policy — which exercises the wiring and *nothing
   else*, and every artefact must say so.
2. **Scope-c ingestion** (Gate-0 item 9: decided, not run). Ceilings 1 and 2 need
   a real committed graph with gold; the pilot's 248 turns are enough for a
   smoke test and not for a number.
3. **Phase-8 Stage B.** The gate is trained on MuSiQue contrast pairs only
   (`PHASE8_DECISIONS.md`); its conversational threshold does not exist yet, so
   the orchestrator's gate step is a *declared placeholder* until it does.

**Consequence for the staging in §4:** Stages A and B are unblocked today,
Stage C is buildable against stubs and becomes meaningful as each blocker
clears, and Stage D is deferred by name. That ordering is the plan's main
structural decision.

### 0.2 Compute — **no training anywhere, and every run needs explicit permission**

**Phase 10 takes no gradient step.** The reader is frozen (v1.2 §3.5 and the
supervisor's standing constraint), the serializer and parser are deterministic,
and the five oracles are diagnostics. There is **no trainable parameter in this
phase**, which is why the one optional component that *would* need one is
declined in §6 decision 11 rather than deferred.

What it does need is **GPU inference**, and that is a different thing from
training. Every event is listed here with its cost so it can be authorised
individually:

| # | Run | Cost (dev RTX 5050) | Stage | Needed for |
|---|---|---|---|---|
| R1 | Reader determinism probe — one prompt × 3 repeats | **< 2 min** | B | G9: is greedy decoding bit-reproducible here? |
| R2 | Ceiling-5 oracle on the pilot (~10 questions, gold proof → reader) | **~5 min** | B | the reader ceiling, and the prompt freeze |
| R3 | End-to-end smoke on the pilot (~10 questions, full read path) | **~10 min** | C | the architecture's exit criterion |
| R4 | Ceiling 5 at scope-c (~200 questions) | **~0.5–1 h** | D | a reportable reader ceiling |
| R5 | End-to-end at scope-c (~200 questions) | **~1–2 h** | D | every end-to-end number |

*(R4/R5 sizing from `DATASET_DECISION.md` §4's Phase-10 row: "GRAFT 0.5–1 h" for
~100 questions on this card.)*

**I will ask before each of R1–R5 and will not start one otherwise.** R1–R3 are
minutes and unblocked; R4–R5 wait on §0.1's blockers anyway.

### 0.3 What Phase 10 is not

Not the system baselines (Phase 11 — full-context, matched-budget RAG, Mem0).
Not Gate 4. Not a benchmark protocol: v1.2 §6.3 says the protocol around the
ceilings "stays out of this document", and it stays out of this one. Not a place
to retune a frozen value, and not a place to change `H`, `U`, the gate threshold
rule or the portfolio.

---

## 1. Twelve gaps this phase must close

### G1 — There is no token budget anywhere in the project, and ceiling 4 is defined as a fraction of it [ANALYSIS]

Ceiling 4 asks *"does a sufficient proof survive the evidence/token budget and
serialization?"* — and `Config` has no such field. Verified: `config/schema.py`
carries `K`, `checker_budget`, `pool_cap`, `max_atoms`, `tau_nli` and the source
tiers, and nothing about tokens. The architecture's §12 table ("values that must
be frozen at Gate 0") does not list one either.

So ceiling 4 currently has no denominator, and worse, **the packing comparison
in `CLAUDE.md` §8 is stated at a budget this project cannot express**: that
section's corrected reading of arXiv 2607.00725 Table 6 turns entirely on the
**160-token** condition being the one where the submodular packer wins and the
other three where it does not.

**Decision (§6 decision 1):** add `serialization_budget_tokens` to `Config`,
frozen at Gate 0's discipline, and report every ceiling-4 number **at more than
one budget** — because a single budget is exactly how that §8 erratum happened.
Candidate ladder in §6; the 160-token cell is included precisely so the
project's own quoted comparison is reproducible in its own units.

### G2 — The prompt is the experiment's identity, and it does not exist yet [ANALYSIS]

v1.2 §3.5 requires "same frozen SLM, same prompt template, same decoding budget
for **every** compared system", and the architecture makes it a hash rather than
a promise: "The (prompt, decoding-config) SHA is stamped into every
`OutputRecord`". `OutputRecord.config_hash` exists; the prompt does not.

**Decision:** the template is written once, pinned in `reader/pins.py`, and its
SHA joins the **stage-E fingerprint** (the seventh). Two properties are
structural rather than intended: the template takes *no* corpus-specific or
system-specific branch (Phase 11's baselines must reuse it byte-identically, or
§3.5's requirement is unenforceable), and it is stamped into every record so a
run at a different prompt is a visibly different experiment.

### G3 — "Answer-binding evidence first and last" is not computable at inference, and this is the third time this class of error has appeared [ANALYSIS]

The architecture's serializer row says: *"Ordering: anchor and answer-binding
evidence at the beginning and end, connective evidence in the middle"*, citing
**[EVIDENCE]** Lost in the Middle (TACL 2024) — models use context edges far
better than the middle.

The U-shaped placement is sound. **"Answer-binding" is not available at
inference**: knowing which atom binds the answer requires the answer. That is
exactly the trap `PHASE9_DECISIONS.md` §1.3 caught in the 2Wiki anchor rule and
Phase 9's G9 caught in the contested flag — twice already, both times in a rule
that read as innocuous prose.

**Decision (§6 decision 3):** the ordering key is built **only** from
inference-computable signals, declared in pins:

| Position | Signal | Available at inference? |
|---|---|---|
| head | obligation-anchor match (`atomfeat`'s `entity_anchor_hit` rule, reused not reimplemented) | yes — the parser reads the question |
| head | highest fused retrieval score | yes — Stage C's own output |
| middle | edge atoms and structural companions ("connective evidence") | yes — atom kind |
| tail | second-highest fused score | yes |

A **gold-bearing ordering is implemented too, and only as a diagnostic**: the
difference between the two ordering's ceiling-4 and reader scores *is* the
measurement of how much the honest ordering costs. Reporting the diagnostic
without the honest one, or vice versa, is the failure this gap exists to
prevent.

### G4 — Citation correctness needs a resolvable chain, and the schema already has one [ANALYSIS + EVIDENCE]

**[EVIDENCE]** ALCE (EMNLP 2023) established reporting answer correctness and
citation correctness **separately**, and supplies this project's motivating
number: on ELI5, even the best models lacked complete citation support **50% of
the time**. **[EVIDENCE]** VeriCite (SIGIR-AP 2025) shows *verification* rather
than generation drives citation quality — removing its NLI check dropped
citation F1 **77.73 → 68.91** while answer correctness barely moved
(41.63 → 41.59).

The chain `claim id → atom → node → assertion → span → turn` already exists and
is what `H`'s support and scope sub-checks walk. Phase 10 must not invent a
second one.

**Decision:** claim ids are `[c1]…[cn]` assigned in **serialized order**, and the
record stores the id→atom map so a citation resolves to a character span in a
real turn. Citation precision/recall is computed against the *cited atom's*
provenance, never against the answer string.

### G5 — Ceilings 1 and 2 need gold that does not exist on conversation, and saying so is the honest close [ANALYSIS]

Ceiling 1 (extraction) asks whether gold evidence statements are represented as
grounded assertions; ceiling 2 (graph) asks whether the built graph contains a
sufficient proof. Both need **gold proof annotation on conversation**, which
`CLAUDE.md` §7 records as the binding constraint on Contribution 1 and which
Gate 0 funds only partially.

What exists: LongMemEval's `has_answer` turn indicators, which
`PHASE7_DECISIONS.md` §3.1 already uses for Tier-A recall, and the Gate-0 item
3.2 Tier-B minimal-proof rule as amended.

**Decision:** ceilings 1 and 2 are implemented against the **Tier-A/Tier-B**
definitions Phase 7 already froze, not against a new annotation, and each
reports **which tier it was computed at**. A ceiling computed at Tier A is a
weaker statement than one at Tier B, and a number that does not say which is
uninterpretable — the same discipline `PHASE9_DECISIONS.md` applies to
"H-valid" on Wikipedia pools.

### G6 — Fix F7 is a resource property, and it must be enforced rather than intended [ANALYSIS]

F7: *"Extractor (7B), reader (3B), embedder, GNNs cannot all be resident.
Stage-sequential execution enforced by the orchestrator: models are
context-managed (load → run stage → free)."* Assigned to P0 **and P10**; P0 built
the scaffolds, P10 owns the enforcement.

The machine makes this real rather than theoretical: `CLAUDE.md` §7 records that
the actual card is an **RTX 5050 with 8 GB**, not the 32 GB the header assumes.
Reader (3B bf16 ≈ 6.2 GB) plus embedder plus a resident GNN does not fit.

**Decision:** one `ModelSlot` context manager, a **test that asserts two LLM
slots are never open at once**, and a reported peak-VRAM figure per query. The
test is the deliverable — "never two resident" is otherwise a sentence.

### G7 — Three conventions Phase 9 deliberately left to this phase's runner [ANALYSIS]

`PHASE9_DECISIONS.md` §7.8 lists them explicitly *"so the runner's author
inherits them as decisions rather than discovering them as bugs"*:

1. `run_portfolio` declares an unused `ledger` parameter and a constant-`False`
   `budget_exhausted` field — Phase 10 either uses them or removes them.
2. An abstained query emits `best_score = NaN` — loud under `np.mean`, but there
   is **no defined FAIL utility** for aggregation.
3. **The Gate-3 aggregation convention for abstentions** has no owner.

**Decision (§6 decision 7):** an abstention is **excluded from utility means and
counted separately**, never imputed as 0 and never silently dropped. Imputing 0
makes an abstaining system look like a wrong-answering one; dropping it makes
abstention free. Both are reported: mean utility **over answered queries**, and
the abstention rate beside it, with its cause split (`gate` vs `fallback`) —
which is the split Phase 8 reserved the vocabulary for.

### G8 — The contested comparison, transferred here by name [ANALYSIS]

Fix F4 flags an output `contested` when top valid sets imply **different answer
bindings**. Phase 9 implemented the gold-bearing half as a diagnostic and
transferred the inference-time half here, because at deployment there is no gold
and the architecture's own words are "costs one comparison" — a *reader-level*
check.

**Decision:** the comparison is one extra reader call on the runner-up valid set
when the portfolio returns ≥ 2 distinct sets, and `outcome = "contested"` (the
third member of `OUTCOMES`, unused until now) when the two answers disagree
under the answer-equivalence rule of decision 9. **It is costed**: one extra
reader call per contested-eligible query, ledgered, and reported as its own
latency and token line — never folded into the base cost.

### G9 — Reader determinism is assumed everywhere and measured nowhere [ANALYSIS]

Greedy decoding is specified, and greedy is deterministic *given* identical
inputs, dtype, kernels and batch composition. This project has already been
bitten by the surrounding assumptions twice: `ingestion_fingerprint` hashes the
extractor's `dtype` because bf16 and fp16 give different digests
(`DATASET_DECISION.md`), and Phase 5's determinism was measured per machine
rather than promised.

**Decision:** R1 measures it — one prompt, three repeats, byte-compare — and the
result is *reported*, not asserted. If it is not bit-reproducible, that is a
finding about the reader and a caveat on every downstream number, not a bug to
chase.

### G10 — Answer scoring has no rule, and picking one after seeing outputs is the §6b failure [ANALYSIS]

v1.2 §6.4's End-to-end group names "LongMemEval accuracy broken out by the five
abilities" and "LoCoMo score" without fixing a scorer. LongMemEval's own release
uses an LLM judge; this project's hybrid budget reserves paid API for the
LLM-prompted-linking baseline, not for judging.

**Decision (§6 decision 10):** the **primary** is the benchmark's own metric,
computed by its own script where one ships, so the number is comparable to
published ones. A **local exact/F1 pair** is reported beside it as the
reproducible-without-API floor. The rule is fixed **before any end-to-end run**,
and the choice of judge is not made after seeing which one flatters the system.

### G11 — SynCheck: declined, with the cost that justifies declining it [EVIDENCE]

**[EVIDENCE]** SynCheck (EMNLP 2024) detects unfaithful sentences at **> 0.85
AUROC** across six long-form RAG tasks, and its FOD decoding improves
faithfulness by **> 10%**. v1.2 §3.5 already states the terms: *"It is not free:
it requires a trained monitor and inference-time compute, and FOD alters
decoding. If used, it is part of the inference algorithm and must share the same
compute budget as competing methods, with latency reported."*

**Decision: declined for Phase 10**, on three grounds that are recorded so the
decision can be reversed knowingly. It needs a **trained monitor** — the only
trainable component this phase would otherwise contain, and the supervisor's
constraint is that the SLM stays frozen and learning happens in the GNN/NN
stack. FOD **alters decoding**, which breaks the byte-identical
`(prompt, decoding-config)` hash G2 makes the enforcement mechanism for §3.5.
And it would have to be given to every Phase-11 baseline too, or the comparison
is unmatched. `CLAUDE.md` §5 already records "SynCheck described as free" as a
caught error; this is the same finding, one phase later, acted on.

### G12 — The per-query ledger is the only place the cost claim can be made honestly [ANALYSIS]

v1.2 §6.4 asks for "latency (p50, p95) · context tokens to the SLM · model and
checker call counts", and `CLAUDE.md` §9 makes cost a **claimed axis** — the
project may not claim to beat full-context on accuracy and must win on
"verifiability, abstention correctness, latency and token cost" instead. Mem0's
graph variant is the cautionary case: it lost on single- and multi-hop at ~2×
tokens and ~3.2× search latency.

`Ledger` already has per-query scoping and per-stage accounting (verified:
`query_scope()`, `_stages`). Phase 10 must open one scope per query, one stage
per pipeline step, and put the snapshot in the record — which `OutputRecord`
already has a field for.

**Decision:** every `OutputRecord` carries a complete `ledger_snapshot` with
per-stage `wall_clock_ms`, `model_forwards`, `llm_tokens_in/out` and
`terminal_checks`. A record without one is a record whose cost claim cannot be
audited.

---

## 2. Scope

**In:** `graft/reader/` (pins, serializer, reader wrapper, answer parser,
citation resolver) · `graft/diagnostics/` (the five ceiling oracles, one
function each) · `graft/orchestrate.py` or `graft/reader/orchestrator.py` (fix
F7's read path) · the stage-E fingerprint in `verify_handoff.py` ·
`serialization_budget_tokens` in `Config` · `scripts/phase10_read.py` ·
`graft/tests/test_reader.py`, `test_ceilings.py`, `test_orchestrator.py`.

**Out:** every Phase-11 baseline (full-context, matched-budget RAG, Mem0) · the
benchmark protocol and Gate 4 · SynCheck/FOD (G11) · any retraining of the gate,
the scorer, the Stage-D policy or the decoders · any change to `H`, `U`, the
gate's threshold rule, the portfolio or any frozen value · the conversational
Stage-B gate (Phase 8's own deferral).

---

## 3. Modules

### P10.0 `reader/pins.py` — what Phase 10 freezes (do this first)

The prompt template and its SHA (G2), the decoding config, the serialization
ordering rule (G3) and its declared signals, the claim-id format, the token
budget ladder (G1), the answer-scoring rule (G10), the abstention aggregation
convention (G7), and the **stage-E fingerprint** printed by `verify_handoff.py`
beside the other six. Importable on a bare interpreter — the Phase-9 §7.3 lesson:
a fingerprint that pulls torch breaks the guarantee its own docstring makes.

### P10.1 `reader/serialize.py` — `ProofSerializer` (G1, G3, G4)

`ProofSet + snapshot → (text, claim_map, report)`. Stable `[c1]…` ids in
serialized order, quoted source spans, token-capped at the pinned budget, U-shaped
ordering from inference-computable signals only. `report` carries what ceiling 4
consumes: which atoms survived the budget, which were dropped, and at what
position each landed.

### P10.2 `reader/read.py` — the frozen reader (G2, G6, G9)

Qwen2.5-3B-Instruct, bf16, greedy, one template. **The weights are already
local** (`models--Qwen--Qwen2.5-3B-Instruct` in the HF cache — the Phase-5
extractor is the same model), so no download is needed. Context-managed loading
per F7. The only module in `graft.reader` that imports torch, on the Phase-8
containment pattern.

### P10.3 `reader/parse.py` — `AnswerParser` + citation resolution (G4)

Answer text and cited claim ids out of the generation; ids resolved through the
claim map to atoms, and through the existing provenance chain to spans. Refuses
a citation it cannot resolve rather than dropping it — an unresolvable citation
is a finding about the reader, and silently discarding it would make citation
precision look better than it is.

### P10.4 `reader/orchestrator.py` — the read path (fix F7, G6, G7, G8, G12)

`answer(question, snapshot_id) → OutputRecord`. Stage C → gate → Stage D → E,
one ledger query scope, one stage scope per step, `ModelSlot` discipline,
`abstain_cause` written from the two routes, contested comparison, and the
untrained-artefact refusal: if the Stage-D policy is untrained or the gate
threshold is the MuSiQue one, the record says so and the runner stamps it.

### P10.5 `diagnostics/ceilings.py` — the five oracles (G5)

One function each, per v1.2 §6.3, each returning a rate **and** the tier it was
computed at. Ceiling 3 delegates to Phase 7's `recall` instrument rather than
recomputing it — including its `saturation()` guard, without which a recall of
1.000 means nothing (`PHASE7_DECISIONS.md` §3.1).

### P10.6 `scripts/phase10_read.py` — the runner and the artefact

Smoke on the pilot; ceilings 1–5 with their tiers; per-query ledger rows;
determinism digest; honesty stamp naming which artefacts were untrained; the
stage-E fingerprint; manifest.

---

## 4. Build order — **four stages, ordered by what unblocks them**

### Stage A — the deterministic core · **unblocked, no GPU, start here**

| Step | Build | Done when |
|---|---|---|
| A0 | P10.0 pins + stage-E fingerprint + `serialization_budget_tokens` | importable bare; seventh fingerprint prints; prompt SHA stable |
| A1 | P10.1 serializer | round-trips a fixture `ProofSet`; budget truncation is deterministic; ordering uses **no** gold field (asserted by AST test, the Phase-9 pattern) |
| A2 | P10.3 parser + citation resolution | a hand-written generation parses; an unresolvable citation raises; the chain reaches a real span |
| A3 | ceilings **3 and 4** | ceiling 3 delegates to Phase 7 and inherits its saturation guard; ceiling 4 reports at every budget on the ladder |
| A4 | tests | `test_reader.py` + `test_ceilings.py` green; full suite green |

**Cost: zero GPU, zero permission needed.** Everything here runs on fixtures and
the existing pilot artefacts.

### Stage B — the frozen reader · **needs R1 + R2 (~7 min GPU inference, no training)**

| Step | Build | Done when |
|---|---|---|
| B0 | P10.2 reader wrapper + `ModelSlot` | loads, generates, frees; peak VRAM reported |
| B1 | **R1** determinism probe | three repeats byte-compared; result *reported* either way (G9) |
| B2 | ceiling **5** | **R2**: gold proof → reader on the pilot; the reader ceiling is a number |
| B3 | the prompt freeze | template SHA in pins; a change to it moves the stage-E fingerprint |

### Stage C — the orchestrator · **buildable now against stubs; R3 when you allow it**

| Step | Build | Done when |
|---|---|---|
| C0 | P10.4 orchestrator | wires C → gate → D → E on fixtures; **two-LLM-slot test fails if violated** (G6) |
| C1 | ledger + record | per-stage accounting in every record; `abstain_cause` written from both routes |
| C2 | contested comparison (G8) | fires on ≥ 2 distinct valid sets; costed and ledgered separately |
| C3 | G7's conventions | abstention excluded from means and counted; `budget_exhausted` and the unused `ledger` param resolved |
| C4 | **R3** end-to-end smoke on the pilot | the architecture's exit criterion; artefact stamps every untrained artefact by name |

### Stage D — the decisive numbers · **deferred by name; blocked per §0.1**

| Step | Blocked on | Produces |
|---|---|---|
| D0 | scope-c ingestion | ceilings **1 and 2** at their declared tier |
| D1 | Phase-3 step 6 → Phase-9 step 6 | end-to-end with a *trained* Stage-D policy |
| D2 | Phase-8 Stage B | a conversational gate threshold, so abstention numbers mean something |
| D3 | all three | **R4 + R5** — the five-ceiling table and the end-to-end row Phase 11 compares against |

---

## 5. Exit criteria

**The machinery is sound**
1. The serializer reads **no gold field** — asserted structurally, the Phase-9
   `test_setgen_real.py` pattern (AST, not substring; that guard's first version
   failed on its own docstring).
2. Claim ids are stable under re-serialization of the same set, and every id
   resolves to a character span in a real turn.
3. Budget truncation is deterministic and reports what it dropped.
4. **Two LLM slots are never open simultaneously** — a test, not a sentence (G6).
5. The prompt SHA is in the stage-E fingerprint and in every `OutputRecord`.
6. `reader/pins.py` imports without torch.

**The diagnostics are honest**
7. All five ceilings run; each reports **the tier it was computed at** (G5).
8. Ceiling 3 delegates to Phase 7's instrument and carries its saturation flag.
9. Ceiling 4 is reported at **every budget on the ladder**, never one (G1).
10. Reader determinism is *measured and reported*, not assumed (G9).

**The read path is auditable**
11. Every `OutputRecord` carries a complete per-stage `ledger_snapshot` (G12).
12. `abstain_cause` is written from both routes and the two are never summed.
13. Abstentions are excluded from utility means and counted separately (G7).
14. The contested comparison is costed as its own ledger line (G8).
15. **Transferred by name to Phase 11:** the baseline adapters, and the
    requirement that they reuse this phase's prompt byte-identically.

**Discipline**
16. Honesty stamp names every untrained or placeholder artefact the run used —
    an untrained Stage-D policy, a MuSiQue-trained gate threshold — so no smoke
    number can be read as a result.
17. Determinism digest equal across two runs; manifest; stage-E fingerprint.
18. **Deferred by name:** Stage D's four steps and their blockers.

---

## 6. Decisions to lock before writing code — **SIGNED OFF 16 August 2026**

**Signed:** Nazib Riasat — 16 August 2026. *(Signed by the assistant at the
project owner's explicit instruction, recorded as delegated rather than
presented as the owner's own hand — the `GATE0_CONTRACT.md`,
`GRAFT_PHASE3_BUILD.md` §6 and `GRAFT_PHASE9_BUILD.md` §6 convention.)* All
twelve decisions are ruled **as written in the table below**, including decision
9, whose value was filled at signing.

**Signed with no Phase-10 code in existence**, which `GRAFT_PHASE2_BUILD.md` §6b
makes the only uncontaminated moment for the rules that matter here. Three of
these are decision-*rules* rather than engineering defaults, and each is fixed
before anything it judges exists:

* **decision 3** — the serialization ordering key. This is the phase's
  §6b-critical cell: an ordering chosen after seeing which one reads better is
  not an ordering rule, and the honest-vs-gold gap it mandates is only a
  measurement if both were declared first.
* **decision 10** — the answer-scoring rule, fixed *before any end-to-end run*,
  so the judge is not chosen after seeing which one flatters the system.
* **decision 1** — ceiling 4's budget ladder, fixed before any packing number,
  because `CLAUDE.md` §8's own erratum was a packing claim quoted at one
  budget.

**Five things this signature does not cover**, each with its named source:

1. **The benchmark's own metric** (decision 10's primary) is *whichever script
   the benchmark ships* — a lookup at wiring time, not a choice made here. If a
   benchmark ships none, that is a finding to record, and the local exact/F1
   pair becomes the primary with the substitution stated.
2. **Every number produced before the §0.1 blockers clear.** A smoke run with an
   untrained Stage-D policy and a MuSiQue-trained gate threshold is a wiring
   test; decision 12 makes the runner stamp it, and the signature does not turn
   a stamped smoke number into a result.
3. **Reader determinism** (G9) — measured by R1, not assumed by this table. If
   it is not bit-reproducible that is a caveat on every downstream number, and
   signing does not pre-empt the measurement.
4. **Peak VRAM under fix F7.** The 8 GB card is `CLAUDE.md` §7's recorded
   discrepancy; decision 6 mandates the test and the report, not a promise that
   the budget fits.
5. **Evidential class.** Most cells here are **[ANALYSIS]** — engineering and
   protocol judgment. Signing changes ownership, not evidential status, and the
   write-up must still label them. The paper-backed ones name their venue:
   decision 3's U-shaped placement (Lost in the Middle, TACL 2024), decision 4's
   answer/citation separation (ALCE, EMNLP 2023) and its verification finding
   (VeriCite, SIGIR-AP 2025), decision 9's normalisation (SQuAD, EMNLP 2016),
   and decision 11's declined monitor (SynCheck, EMNLP 2024).

**This table is normative.** Where a restatement elsewhere disagrees with it, the
table wins and the restatement is a bug — the Phase-2 mitigation adopted after
five review rounds in which most defects were a fix landing in three places out
of four. **And where `PHASE10_DECISIONS.md` eventually disagrees with this
table, that file wins** — four of Phase 9's signed decisions were overturned at
its build, and pretending a signature is a prediction is how that becomes a
surprise instead of a record.

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | **Token budget** | add `serialization_budget_tokens`; ladder **{160, 512, 1024}** and report ceiling 4 at all three. 160 is included so `CLAUDE.md` §8's own quoted comparison is reproducible in this project's units; a single budget is how that erratum happened | ceiling 4 has no denominator; or a packing claim at one cherry-picked budget — §5's own pattern |
| 2 | **Prompt + decoding** | one template, pinned, SHA in the fingerprint and every record; greedy, bf16; **no corpus- or system-specific branch** | §3.5's matched-comparison requirement becomes unenforceable, and Phase 11's baselines are not comparable |
| 3 | **Ordering** | G3's inference-computable key; gold-bearing ordering implemented **as a diagnostic only**, and the gap between them reported | the third repeat of an inference-unavailable signal reaching a live path |
| 4 | **Claim ids** | `[c1]…` in serialized order; id→atom map in the record; citations scored against the cited atom's provenance | citation P/R measures the answer string rather than the citation |
| 5 | **Ceiling tiers** | ceilings 1–2 at Phase 7's Tier-A/Tier-B definitions; **every ceiling reports its tier** | a Tier-A number read as a Tier-B claim |
| 6 | **F7 enforcement** | `ModelSlot` context manager + a test that two LLM slots never coexist; peak VRAM per query reported | an 8 GB card OOMs mid-run, or the constraint is "enforced" by convention |
| 7 | **Abstention aggregation** | excluded from utility means, counted separately, cause split reported | imputing 0 makes abstaining look like answering wrongly; dropping makes abstention free |
| 8 | **Contested** | one extra reader call on the runner-up when ≥ 2 distinct valid sets; `outcome = "contested"`; costed and ledgered separately | fix F4's inference half stays unimplemented, or its cost hides inside the base number |
| 9 | **Answer equivalence** | **SQuAD-style normalisation, then exact match, then alias set.** Normalisation = casefold · strip articles (`a`/`an`/`the`) · strip punctuation · collapse whitespace. A prediction matches if the normalised strings are equal **or** the normalised prediction equals any normalised alias the corpus ships. **[EVIDENCE]** this is the normalisation SQuAD (EMNLP 2016) introduced and HotpotQA, 2WikiMultiHopQA and MuSiQue all inherit, so a local number computed this way is commensurable with published ones. **One rule, two consumers** — the contested check (G8) and the local scorer (decision 10) call the same function, because two equivalence rules in one pipeline is how a system disagrees with itself about whether two answers are the same | two different equivalence rules in one pipeline; or a local score not comparable with any published one |
| 10 | **Answer scoring** | benchmark's own metric as primary (its own script where one ships), local exact/F1 beside it; **fixed before any end-to-end run** | a judge chosen after seeing which one flatters the system |
| 11 | **SynCheck / FOD** | **declined** (G11) — needs a trained monitor, alters decoding, and would have to be given to every baseline | reversible, but reversing it re-opens the frozen-decoding hash and the matched-budget requirement |
| 12 | **Untrained-artefact stamping** | the runner refuses to print an unstamped number while any consumed artefact is untrained or placeholder | a smoke number read as a result — `PHASE7_DECISIONS.md` §7's committed `--smoke` artefact, one phase later |

---

## 7. Explicitly not in Phase 10

No Phase-11 baselines; no Gate 4; no benchmark protocol; no SynCheck/FOD; no
training of anything; no retuning of `H`, `U`, `beta`, `K`, `checker_budget`,
`pool_cap`, `max_atoms`, `tau_nli`, the gate threshold rule or the portfolio; no
second provenance chain (G4 reuses the checker's); no second recall instrument
(ceiling 3 delegates); no conversational gate training (Phase 8's Stage B).

---

## 8. What Phase 11 gets from this

* **`System.run(conversation, question) → OutputRecord`** — the interface every
  baseline implements, with GRAFT as its first implementation.
* **The frozen prompt and decoding SHA**, which every baseline must reuse
  byte-identically. This is what makes v1.2 §3.5's "same frozen SLM, same
  prompt, same budget" checkable rather than promised.
* **The per-query ledger**, so the latency and token-cost axes `CLAUDE.md` §9
  says the project must win on are measurable for every system on equal terms.
* **The five ceilings**, which is what turns a bad end-to-end number from
  uninterpretable into localised — and is Contribution 4.
* **The stage-E fingerprint**, seventh in the identity tuple.
