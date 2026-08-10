# GRAFT — Execution Architecture v1.1

**Component-by-component build plan, in build order, ready to code.**

Date: 8 August 2026 (v1.1 — correctness review round; new fixes F9–F13 in §1, new Phase 2.5)
Derives from: `GRAFT_RESEARCH_PLAN_v1.md` (v1.2). This document adds nothing to the science; it turns v1.2 into an ordered, code-level construction plan. Where v1.2 and this document disagree, v1.2 wins and this document has a bug.
Evidence labels are inherited from v1.2: **[EVIDENCE]** (a named paper supports this), **[HYPOTHESIS]** (this project tests it), **[ANALYSIS]** (engineering/mathematical judgment made here).

---

## 0. Scope and pinned decisions

### 0.1 In scope
The system (Stages A–E, the deterministic core, the answerability gate, the synthetic correctness harness) and the **Tier-1 baselines** from v1.2 §2.4: 3 Stage-B encoders + LLM-prompted linking, 7 Stage-D learning methods, 5 Stage-D search methods, thin system-baseline adapters.

### 0.2 Out of scope (per user instruction)
Benchmark evaluation protocols, metric tables, dataset preparation, annotation guidelines, paper writing. **Two carve-outs stay in scope because code depends on them:**
1. the **enumerable synthetic environment** — it is the Gate-2 correctness harness, not an eval dataset;
2. **data interfaces** — every learned component's input/output schema is defined here, with its supervision source named in one line, so training code can be written now.

### 0.3 Pinned decisions (from user, 8 Aug 2026)

| Decision | Value | Consequence |
|---|---|---|
| Compute | 1× RTX 5090 (32 GB) | All models sized to fit one card; stages run sequentially, never two LLMs resident at once |
| LLM access | Hybrid | Local model for bulk extraction; small API budget reserved for the LLM-prompted-linking baseline so that comparison stays faithful to what Mem0/Zep actually do |
| Frozen reader | Qwen2.5-3B-Instruct | **[EVIDENCE]** 3B is the size class where evidence packing demonstrably helps, and Qwen2.5 is the exact reader family of the study that shows it ([arXiv 2607.00725](https://arxiv.org/abs/2607.00725): +F1 at 3B, null at 7B, reversed at 14B). Never fine-tuned, ever. |
| Scope | Tier 1 only | Tier 2/3 exist as registry stubs (§14), one paragraph each |

### 0.4 The boring-stack rule [ANALYSIS]

One Python 3.11 process, one GPU, no services. Anything that can be a pure function is a pure function.

| Need | Choice | Why |
|---|---|---|
| Tensors / training | PyTorch | — |
| Heterogeneous GNN | PyTorch Geometric (`HGTConv` is built in) | HGT is the Tier-1 relational encoder |
| Text embeddings | `sentence-transformers`, **bge-small-en-v1.5** | **[EVIDENCE-adjacent]** the embedder used by the packing study whose reader we adopt (arXiv 2607.00725); small enough to be free |
| Sparse retrieval | `bm25s` (or `rank_bm25`) | in-process, no server |
| Dense search | exact cosine via `torch.topk` | pools are tens of thousands of atoms at most — **no FAISS, no ANN index** [ANALYSIS] |
| Local extractor LLM | Qwen2.5-7B-Instruct, 4-bit (bitsandbytes or AWQ) | ~6–8 GB; fits alongside nothing else, loaded only during Stage-A runs |
| Reader | Qwen2.5-3B-Instruct, bf16 | ~6.5 GB; loaded only during Stage-E runs |
| NLI verifier | off-the-shelf NLI cross-encoder (TRUE-style) | **[EVIDENCE]** VeriCite found an NLI model the cost-effective verifier vs. LLM judges (citation F1 80.05 vs 73.01 for an 8B LLM) |
| PCST solver | `pcst_fast` | the G-Retriever formulation's standard solver [ANALYSIS] |
| Graph store | append-only JSONL event log + in-memory typed graph | §Phase 0; no database server |
| Config | one YAML → frozen dataclasses; SHA-256 config hash logged on every run | reproducibility without a framework |

Anything not on this list needs a written justification before it enters the repo.

---

## 1. Flaws found while making v1.2 executable — and their fixes

These are places where v1.2 is correct as research design but under-specified or silently broken as code. Each fix lands in a named phase.

| # | Flaw | Fix | Lands in |
|---|---|---|---|
| F1 | **`U`'s `sufficiency` term is circular at run time.** v1.2 §4.1 says sufficiency is "learned or annotated," but the reward must be computable during training and *some* ranking signal must exist at inference, where no gold exists. | Split it: **train-time sufficiency is deterministic against gold evidence** (fraction of the gold proof covered — computable on 2WikiMultiHopQA-style data, where gold proofs are (subject, property, object) triples; **[EVIDENCE]** Graph-S3 validates exactly this pattern of dense supervision from offline golden subgraphs, +15.6% acc / +17.2% F1 over sparse final-answer reward). **Inference-time ranking uses a small utility head** distilled from train-time `U` (Phase 9). The learned head never touches `H` (prohibited by v1.2 §4.4). | P1, P9 |
| F2 | **The obligation parser exists nowhere.** `coverage` is "deterministic given the obligation parser" (v1.2 §4.1) and `d(s)` drives Contribution 3 — but no component produces obligations. | New module `ObligationParser` (Phase 1): question → typed slots (entity anchor, requested value type, time constraint, source requirement, aggregate flag). Exact on the synthetic environment; extractor-LLM-derived on real questions, with slot-level precision audited before use. | P1, P5 |
| F3 | **Dead-ended trajectories break the declared target.** A sampler that exhausts legal `ADD`s while `STOP` is masked produces a trajectory with no terminal reward — undefined in every log-space loss. But giving it a positive *floor* reward while `p*` is normalized over **valid terminals only** puts policy mass on an outcome the target does not contain, so exact TV/JS would compare two distributions with different support and measure nothing. | **A single absorbing `FAIL` terminal**, reached on budget exhaustion with `H = 0`, carrying a fixed config reward `R_fail`. The target becomes `p*` over `{valid terminals} ∪ {FAIL}` with `Z = Σ_valid R(X) + R_fail`; the exact evaluator obtains `p_θ(FAIL)` by complement, with no path enumeration. **`FAIL` is not "an invalid proof was returned"** — it is reached when **construction can neither legally continue nor legally stop**, which is exactly the inference-time abstain fallback (Phase 9.4), so write-path and read-path semantics agree. *(Amended twice. Originally "budget exhaustion", which Phase 1 found too narrow — a dead end also arises from an empty pool, a pool exhausted below the cap, or every remaining atom failing a per-atom check. And the claim a dead end licenses is the modest one: **no valid proof was found under this pool, policy, attempt count and budget** — not that none exists, since another action sequence from the same root may reach one. An intermediate draft asserted the non-existence reading, which would let a Phase-9 abstention claim what the search never established. See `GRAFT_PHASE2_BUILD.md` G4.)* The safety property is therefore stated precisely as **no formally invalid proof set is ever returned** (guaranteed by `STOP`-masking), *not* as "invalid outcomes have zero probability". `p*(FAIL)` shrinking over training is a free convergence diagnostic. [ANALYSIS] | P0, P1, P2 |
| F4 | **The inference-time selection rule among K valid sets is unspecified.** Best-of-K needs a "best." | Rank valid sets by the distilled utility head; tie-break by smaller set. If the top valid sets imply **different answer bindings**, flag the output `contested` (kept from the original GRAFT pipeline; costs one comparison). | P9 |
| F5 | **K and the checker budget are unpinned**, yet the Stage-D primary metric is "best-of-K at fixed checker budget" (v1.2 §2.4). | Config defaults, declared before any comparison run: **K = 8** (1 greedy + 7 sampled), **checker budget = 32 calls/query**, sweepable `{4, 8, 16}` in configs but the primary is K = 8. Same K is the "number of returned sets" in the search comparison — one constant, used everywhere. | P0 config |
| F6 | **Policy/encoder coupling risk.** Phase-3 learners are developed on the synthetic environment (simple states); if the policy net assumes those features, Phase 9 forces a rewrite. | One frozen interface from day one: `policy(state_repr, action_reprs) → logits`. On synthetic data `state_repr` comes from an MLP featurizer; on real data from the Stage-B graph encoder. The learners never know which. | P3 |
| F7 | **VRAM collision.** Extractor (7B), reader (3B), embedder, GNNs cannot all be resident. | Stage-sequential execution enforced by the orchestrator: models are context-managed (load → run stage → free). No stage requires two LLMs. [ANALYSIS] | P0, P10 |
| F8 | **D2 needs claim *pairs*, but nothing proposes them.** Comparing a new claim against every existing claim is O(N) LLM-free but O(N) encoder passes per claim. | Pair proposer: new assertion is compared only against the **top-s similar existing claims sharing an entity anchor** (s = 10). **[EVIDENCE-adjacent]** Mem0 uses exactly this pattern — top s = 10 similar memories retrieved before its update decision. | P6 |
| F9 | **Unsupported assertions can reach the active evidence graph.** Phase 5 stores an assertion failing NLI with `entailed_by_span=False` and never blocks it; Phase 6's commit pipeline validates only "formally checkable violations"; Phase 1's checker has no entailment-flag check. Open path: extractor invents a claim → NLI marks it unsupported → stored → committed → retrieved as evidence. | **Two layers.** *Audit layer* — Phase 5 unchanged; everything is stored, which is what makes extraction quality measurable at all. *Active-evidence layer* — a **commit gate** marks assertions failing the support policy `quarantined`: present in the event log, absent from the active graph. `H` gains a **seventh sub-check** rejecting any atom that references a quarantined assertion. **[ANALYSIS]** *Computing* entailment is learned and stays out of `H` (v1.2 §4.4); *reading an already-stored flag* is deterministic and structurally identical to the existing `t_invalid` check. Default policy strict (`entailed_by_span=True` at `tau_nli`). The **quarantine rate** is a reported Phase-5 metric — a high rate is an extraction-quality signal, never grounds for quietly lowering the threshold. Restores the project's founding requirement that extracted facts be supported by the raw turn before becoming usable evidence. | P1, P5, P6 |
| F10 | **Structural closure is undefined.** v1.2 §4.3 requires the decision explicitly; the architecture never made it. Atoms may be nodes, edges or bindings, and nothing says what happens when an edge is selected without its endpoints. Blast radius: `H`, `ADD` masks, set-size accounting, `P_B`, the exact DP, PCST conversion, the collision audit. | **Rule: an atom may be added only once every atom it references is already selected** — an edge requires both endpoints, a binding requires its referents, nodes reference nothing. Mask cost is one membership test per reference. Consequences: `P_B` is uniform over *removable* atoms (already anticipated by v1.2 §12); the exact DP enumerates **closed** subsets, not all subsets; PCST's connected output maps to a closed set with no conversion logic; `H` must also check closure directly, because S3/S4 build sets without going through the `ADD` masks. **This retires an open question** — any closed terminal with `|X| ≤ max_atoms` is constructible by adding nodes first then edges, so the unconstructible-valid-terminal rate is **0 by construction**, turning v1.2 §4.3's "measure it" into a regression test with a known answer. [ANALYSIS] | P0, P1, P2, P4 |
| F11 | **The proposed auxiliary head is degenerate.** L7 fed `d(s)` into the encoder and then predicted `d(s)` from the resulting representation. The target is already in the input, so the objective is satisfiable by copying — it teaches nothing about credit assignment, which is the entire claim of Contribution 3. | L7 keeps **one** mechanism: `Δd` as input features to `φ_θ` and the policy. The auxiliary head is demoted to an ablation arm **L7b** with a **forward-looking** target — the terminal deficit reachable from `s`, supervised by the realized rollout, which is genuinely not in the input. Narrower and honest; v1.2 already labels Contribution 3 `[HYPOTHESIS]` with the claim entirely empirical, so nothing real is lost, and the capacity-matched control becomes meaningful. [ANALYSIS] | P3 |
| F12 | **The L7 success criterion was unquantified.** "If L7 ≤ L6" never said ≤ on what — exact TV, best-of-K utility, convergence speed, sample efficiency, or final reward — leaving the decision rule unfalsifiable. | Predeclared before any run (Phase 3 exit): **primary** = exact TV at a *fixed training budget*, three seeds, paired bootstrap; **secondaries** = trajectories-to-threshold and best-of-K utility at `checker_budget`. Fixed-budget improvement is the correct frame precisely because the claim is about credit assignment — the framing FL-GFN and LED-GFN both use. **[EVIDENCE]** Dror et al., ACL 2018: a decision rule chosen after seeing results is not a decision rule. | P3 |
| F13 | **Phase 4 forward-references a Phase 9 component.** S1/S2/S5 were specified to rank with the "utility head", but Phase 9.3 is where that head is first trained. Phase 4 could not run as written. | On the synthetic lattice `U` is *exactly computable* (every term deterministic, gold known), so all five search methods score with **exact `U`**; the distilled head substitutes only at Phase 9, on real data, where gold is absent. Stated consequence: Phase-4 results are measured under a **perfect scorer** — fair across methods, but optimistic relative to deployment, so the Gate-3 synthetic table must be read with that caveat. | P4, P9 |

---

## 2. The two orders

**Runtime data path** (what the finished system does):

```
WRITE PATH (per turn)
turn ─► A: ingest + extract + span-ground + NLI-verify + support-gate
      ─► B: eligible assertions → encode graph → D1..D4 propose → schema-validate → commit (versioned)

READ PATH (per question)
question ─► ObligationParser
         ─► C: hybrid retrieval → candidate pool (≤64 atoms) 
         ─► Gate: answerable? ──no──► ABSTAIN
         │yes
         ─► D: sample K evidence sets → H-filter → rank by scorer
         │        └─ none valid / FAIL ──► ABSTAIN (fallback, logged)
         ─► E: serialize best set → frozen Qwen2.5-3B → answer + citations
```

**Build order** (what you code, in order) — deliberately different. Phases 1–4 need zero real data and retire the highest-risk science first; Phases 5–8 are the data pipeline; Phases 9–11 join them.

| Phase | Builds | Plan gate it serves | Rough solo effort |
|---|---|---|---|
| 0 | Scaffold, schemas, event log, config, budget ledger | Gate 0 (encodes its outputs) | 1 wk |
| 1 | Deterministic core: `H`, `U`, reward, masks, obligation parser | Gate 2 prerequisite | 1–1.5 wk |
| 2 | Enumerable synthetic environment + exact evaluator | Gate 2 harness | 1 wk |
| **2.5** | **Annotation feasibility spike** (risk control, not architecture) | **Gate 0 item 8, measured not estimated** | 1 wk |
| 3 | Set-construction policy + all 7 Tier-1 learners (on synthetic) | **Gate 2** | 3 wk |
| 4 | All 5 Tier-1 search algorithms (on synthetic) | **Gate 3 (synthetic kill-shot)** | 1.5 wk |
| 5 | Stage A: ingestion, extraction, grounding, verification | Gate 1 prerequisite | 1.5 wk |
| 6 | Stage B: encoders, D1–D4, commit pipeline, LLM-linking baseline | **Gate 1** | 3 wk |
| 7 | Stage C: hybrid retrieval + pool builder | Gate 3 prerequisite | 1.5 wk |
| 8 | Answerability gate | — | 0.5 wk |
| 9 | Stage D on real data: featurization, training, portfolio inference | Gate 3 (real) | 2 wk |
| 10 | Stage E: serialization, reader, parsing; read-path orchestrator; ceiling oracles | Gate 4 prep | 1 wk |
| 11 | System-baseline adapters (full-context, matched RAG, Mem0) | Gate 4 | 0.5–1 wk |

Total ≈ 19 weeks of build for one person, excluding annotation and evaluation runs. If annotation (Gate 0/1 labels) is late, Phases 5–8 can be developed against pilot slices without blocking Phases 1–4 — but Phase 2.5 must still run on schedule, because its entire purpose is to discover that lateness early.

---

## Phase 0 — Scaffold and data contracts

**Purpose.** Every later phase imports its types from here. Nothing else may define a schema.

### Repo layout

```
graft/
  config/           # YAML configs; frozen dataclasses; config-hash util
  schemas.py        # every dataclass below; JSON (de)serialization
  eventlog.py       # append-only JSONL log; snapshot = byte/line offset
  graphstore.py     # in-memory typed graph built by replaying the log
  ledger.py         # budget ledger: checker calls, model calls, tokens, wall-clock
  core/             # Phase 1: checker, utility, reward, masks, obligations
  synth/            # Phase 2: ProofLattice env + exact evaluator
  setgen/           # Phase 3: policy interface, learners; Phase 4: search
  ingest/           # Phase 5: Stage A
  graphbuild/       # Phase 6: Stage B encoders + decoders + commit
  retrieve/         # Phase 7: Stage C
  gate/             # Phase 8
  reader/           # Phase 10: Stage E
  baselines/        # Phase 11 system adapters
  diagnostics/      # ceiling oracles, audits
  tests/
```

### Core schemas (`schemas.py`) — the Gate-0 contract, in code

```python
Turn        {turn_id, conv_id, session_id, speaker, ts, text}          # immutable
SourceSpan  {span_id, turn_id, start, end}                             # char offsets, immutable
Assertion   {assertion_id, kind: claim|value|event|time,
             text_norm, spans: [span_id],                              # multi-span, cross-turn allowed
             flags: {asserted_by: speaker, entailed_by_span: (bool, score),
                     externally_verified: bool=False,
                     current_under_update_policy: bool},
             t_created}
Node        {node_id, ntype ∈ {Entity, Claim, Value, Event, TimeInterval,
             Source, Mention, Turn, SourceSpan, Conflict}, payload}
Edge        {edge_id, etype ∈ {mentioned_in, asserted_by, about_entity, has_value,
             valid_during, supported_by, same_as, contradicts, supersedes,
             derived_from, retired_by},
             src, dst, t_created, t_invalid: ts|None, superseded_by: edge_id|None,
             provenance: [span_id]}
Obligations {entity_anchor: str|None, value_type: str|None,
             time_constraint: Interval|None, needs_source: bool, aggregate: bool}
CandidateAtom {atom_id, kind: node|edge|binding, refs, feat: np.ndarray}
ProofSet    {atoms: frozenset[atom_id], bindings: {slot: atom_id}}
OutputRecord{answer|abstain|contested, citations: [span_id], proofset,
             ledger_snapshot, config_hash}
```

**Design commitments, paper-backed:**
- **Edges are invalidated, never deleted** (`t_invalid`, `superseded_by`). **[EVIDENCE]** Zep's bi-temporal model (t_created/t_expired/t_valid/t_invalid) with edge invalidation is the published precedent, and its temporal-reasoning gains (+38–48% relative over full-context) are the payoff.
- **Provenance pointers on every edge.** **[EVIDENCE]** SEEM (ACL 2026) anchors memory structures with explicit provenance pointers; Stage A's four flags implement v1.2 §3.1's assertion-not-truth rule.
- **Snapshot pinning**: a read runs against `GraphStore.at(snapshot_id)` where `snapshot_id` is an event-log offset. Reads are reproducible byte-for-byte.
- **The budget ledger is global and mandatory**: every checker call, LLM call, token, and model forward increments it. The Stage-D primary metric and the search comparison are *defined* in terms of this ledger (v1.2 §2.4, §5.2), so it cannot be an afterthought.

### Config defaults frozen here (change only at Gate 0, never after seeing results)

```yaml
beta: 4.0                 # reward temperature — sweep on synthetic env only, then freeze
u_weights: {suff: 1.0, cov: 0.5, src: 0.25, temp: 0.5, red: 0.25, size: 0.1}
r_fail: 1.0e-6            # reward of the single FAIL terminal (fix F3); frozen with beta
K: 8                      # portfolio size, = search "returned sets"
checker_budget: 32        # terminal H checks per query (primary budget axis)
pool_cap: 64              # max candidate atoms into Stage D
max_atoms: 16             # H_max, max selected atoms per proof set
tau_nli: 0.8              # entailment acceptance threshold, audited in Phase 5
support_policy: strict    # eligible iff entailed_by_span=True at tau_nli (fix F9)
seeds: [13, 42, 7]        # three seeds for every trained method (Dror et al., ACL 2018)
```

**Exit criterion.** Round-trip tests: every schema serializes → deserializes identically; event-log replay reconstructs an identical graph; ledger counts survive a crash-resume.

---

## Phase 1 — Deterministic core (`core/`)

**Purpose.** The pure-function heart: formal validity `H`, executable utility `U`, reward `R`, action masks, obligation parsing. Everything downstream calls these; nothing downstream redefines them.

### 1.1 `checker.py` — formal validity `H`

`H(X: ProofSet, q: Obligations, G: GraphSnapshot) → (bool, trace)`

Implements **exactly** the left column of v1.2 §4.4, as independent named checks, each returning a violation record:

1. node/edge type legality against schema;
2. ID uniqueness / duplicate detection by hash;
3. temporal interval arithmetic and ordering (bindings must respect `valid_during` at question time);
4. no reference to retired/invalidated evidence (`t_invalid` set at pinned snapshot);
5. scope and access constraints;
6. set-size and budget limits;
7. **support eligibility** — no atom references a `quarantined` assertion (fix F9). This is a stored-flag read, exactly like check 4, and therefore deterministic: *computing* entailment is learned and excluded from `H`, *reading the flag it already produced* is not;
8. **structural closure** — every atom's referenced atoms are present in the set (fix F10). Enforced by the `ADD` masks during policy construction, and checked here because S3 and S4 build sets directly and bypass those masks entirely.

**Prohibition, enforced by construction [v1.2 §4.4]:** `H` imports no model of any kind. There is no code path by which a learned score reaches `H`. Entailment, sufficiency, authority, and answerability are handled by `U`, the gate, and Stage B respectively — the routing table in v1.2 §4.4 is implemented as module boundaries.

### 1.2 `obligations.py` — obligation parser *(fix F2)*

`parse(question, mode) → Obligations`
- `mode="exact"`: synthetic environment — obligations are generated with the instance, zero error.
- `mode="learned"`: real questions — extractor LLM fills the typed slots (Phase 5 prompt); slot-level precision is audited on a labeled pilot before any downstream use, and the audit number is reported wherever coverage is reported [ANALYSIS].

### 1.3 `utility.py` — executable `U` *(fix F1)*

`U(X, q, G, gold: GoldProof|None, w) → float`, with each term a separate function:

| Term | Definition | Character |
|---|---|---|
| `sufficiency` | **train:** fraction of gold proof atoms covered by `X` (gold = evidence triples, 2Wiki-style; **[EVIDENCE]** Graph-S3's offline-golden-subgraph supervision) | deterministic given gold |
| | **inference:** not part of `U` — the distilled utility head (Phase 9) ranks instead | learned, outside `H` |
| `coverage` | fraction of obligation slots addressed by `X` | deterministic given parser |
| `source_quality` | metadata lookup from `Source` node type | deterministic |
| `temporal_correctness` | interval-arithmetic agreement of bindings with constraint | deterministic |
| `redundancy` | facility-location marginal-gain overlap between atoms (per the v1.2 fix — **not** the degenerate `max(0, S(X\e)−S(X)+ε)` form) | deterministic |
| `size` | `|X| / max_atoms` — a single normalized atom count, one weight (Phase-0 plan G3) | deterministic |

**All six terms return values in [0, 1]** (Phase-0 plan G4), so `U` is bounded and `β` scales a bounded quantity. Asserted in unit tests; without it the Phase-3 `β` sweep is uninterpretable.

### 1.4 `reward.py`

```
R(X | q, G) = 1[H(X,q,G)] · exp(β · U(X,q,G))       # v1.2 §4.1 — indicator is multiplicative
R(FAIL)     = r_fail                                 # single absorbing terminal, config constant (fix F3)
```

`FAIL` is reached when construction can neither legally continue nor legally stop — budget exhaustion with `H = 0` is the common case, not the definition (see `GRAFT_PHASE2_BUILD.md` G4). It is a genuine terminal of the MDP and a member of the target's support, so `Z = Σ_{valid X} R(X) + r_fail`. `r_fail` is small enough that `p*(FAIL)` is a negligible share of total mass, and is frozen at Gate 0 alongside `beta`.

**No other invalid set is reachable as a terminal**, because `STOP` is masked whenever `H = 0` (§1.5). The safety property is therefore *no formally invalid proof set is ever returned* — stronger and more precise than a claim about probabilities, and unaffected by `FAIL` carrying positive mass.

### 1.5 `masks.py`

`legal_adds(state) → mask` — excludes repeats, budget violations, formally illegal atoms, and **atoms whose referenced atoms are not yet selected** (closure rule, fix F10). `stop_allowed(state) = H(state)`, evaluated incrementally so that construction costs no `terminal_checks` (Phase-0 plan G1). **No `ABSTAIN` action exists** (v1.2 §3.4/§4.2); the `FAIL` terminal of §1.4 is reached by budget exhaustion, never by an action.

**Exit criterion.** Property-based tests: *no formally invalid set is ever a reachable terminal other than `FAIL`*; all eight checker sub-checks have positive and negative unit cases; every `U` term returns a value in [0, 1]; masks never permit an atom whose references are absent; every zero-legal-action state with `STOP` masked transitions to `FAIL` — budget exhaustion is the common cause, not the only one (Phase-1 exit criterion 5).

---

## Phase 2 — Enumerable synthetic environment (`synth/`)

**Purpose.** The Gate-2 correctness harness: the only place the learned terminal distribution can be compared to the target *exactly*. **[EVIDENCE]** exact/enumerable evaluation is the standard instrument in the GFlowNet training literature (Shen et al., ICML 2023, evaluate on enumerable spaces; When Do GFlowNets Learn the Right Distribution?, ICLR 2025, builds correctness metrics precisely because sampled proxies mislead).

### 2.1 `lattice.py` — ProofLattice generator

Config-driven instance generator: a universe of 20–30 typed atoms with (a) required anchors, (b) dependency pairs (atom legal only if partner selected — exercises `ADD`-mask logic), (c) conflicting pairs (exercise `H`), (d) temporal toys (exercise interval checks), (e) distractors. Valid terminal sets capped at |X| ≤ 8, total valid terminals ≤ ~5,000 per instance. Obligations generated with the instance (`mode="exact"`).

### 2.2 `exact.py` — exact evaluator

- Enumerate all valid terminals; compute `Z = Σ_{valid X} R(X) + r_fail` and the target over **`{valid terminals} ∪ {FAIL}`**: `p*(X) = R(X)/Z`, `p*(FAIL) = r_fail/Z` (fix F3). The support of `p*` and the support of `p_θ` are then identical, which is the only condition under which a TV/JS number means anything.
- **Exact policy terminal probability by closed-subset DP [ANALYSIS]:** in a set-building MDP every trajectory to terminal `X` passes only through subsets of `X`, and under the closure rule (fix F10) only through **closed** subsets — so `p_θ(X)` is dynamic programming over the closed sub-lattice of `X`, at most `O(2^{|X|}·|X|)` per terminal and cheaper in practice, trivial at |X| ≤ 8. `p_θ(FAIL) = 1 − Σ_{valid X} p_θ(X)`, obtained by complement with no path enumeration. *(Phase 2 refines this numerically: the enumerated state graph labels dead ends, so `p_θ(FAIL)` is accumulated directly and the complement becomes a partition check — the complement subtracts two quantities agreeing to ~12 digits and cannot resolve `p*(FAIL) ≈ 2.5e-12` at float64. See `GRAFT_PHASE2_BUILD.md` G2/G6. Outside an enumerable environment the complement remains the only option, which is what this line is about.)* This is what makes **exact TV/JS** computable over the full support, satisfying v1.2 §6.4's "do not substitute a proxy when an exact result is available."
- Audits, all exact: **unconstructible-valid-terminal rate — expected 0 under fix F10, retained as a regression test**; `FAIL` rate; equivalent-action collision rate (**[EVIDENCE]** the collision audit is required because uncorrected equivalent actions bias sampling — Symmetry-Aware GFlowNets, ICML 2025, L₁ ≈ 0.12 uncorrected vs ≈ 0.01 corrected; if collisions ≠ 0, apply their terminal reward scaling).

**Exit criterion.** On a hand-checkable 6-atom instance: enumerated `p*` matches manual computation and `Σ p* = 1` **including `p*(FAIL)`**; for a uniform-random policy the DP result matches a Monte-Carlo estimate within tolerance and `Σ_valid p_θ + p_θ(FAIL) = 1`; the unconstructible rate is 0; all audits run.

---

## Phase 2.5 — Annotation feasibility spike (risk control, not architecture)

**Purpose.** Answer Gate 0 item 8 — *how many annotations are actually feasible* — with a measurement instead of an estimate, at week 4 rather than week 10.

**Why here [ANALYSIS].** Three facts make late discovery expensive. Stage B is the strongest and most supervisor-aligned contribution (v1.2 §9). Annotation infeasibility is the largest named risk (v1.2 risk #15) and Gate 0's stop condition depends on it. And a Gate-2 pass on the synthetic lattice establishes that the learners work — it establishes **nothing** about whether real conversational proof supervision can be obtained. One week here converts the project's biggest unknown into a number before three weeks are committed to seven learners.

**Deliverables.**
1. Stage-A extraction running end to end on ~50 real turns — a thin slice of Phase 5, not the component.
2. Annotation guidelines v0 for **D1 and D2**, the two decoders with no off-the-shelf supervision.
3. ~100 D1 decisions and ~50 D2 pairs annotated.
4. **Measured minutes-per-item**, plus self-agreement on a 20-item subset re-annotated after a delay of at least two days (or inter-annotator agreement if a second annotator exists).
5. A go/no-go figure: annotations achievable per week × weeks available, against the volume Phase 6 needs.

**Exit criterion.** The Gate-0 item-8 number is a measurement with a stated method, not an estimate. If it shows the required volume is unreachable, scope is reduced **here** — by narrowing the schema or the question types — which is exactly what v1.2 Gate 0's stop condition asks for, rather than later by quietly weakening the evaluation.

**Deliberately not in this phase.** No encoders, no decoders, no training, no Phase-6 architecture. This is a measurement, and the main way it fails is by growing into an early Phase 6.

---

## Phase 3 — Set-construction policy and the 7 Tier-1 learners (`setgen/`)

**Purpose.** All Stage-D learning methods, built and verified against exact TV on the ProofLattice **before any real data exists**. This is Gate 2.

### 3.1 The frozen policy interface *(fix F6)*

```python
class StateEncoder(Protocol):        # swapped between synthetic and real
    def encode(pool, selected_mask, d_s, budget_left) -> Tensor   # h_s
class Policy(nn.Module):
    def action_logits(h_s, atom_feats) -> Tensor   # ADD-per-atom ∪ STOP; masked before softmax
```
- Synthetic `StateEncoder`: sum/mean pooling + MLP.
- Real `StateEncoder` (Phase 9): Stage-B graph-encoder embeddings.
- `d_s` = checker obligation vector (which obligations satisfied so far) — **a feature, never an energy** (v1.2 §4.5.4).
- Backward policy: **uniform over *removable* atoms** (fixed) — amended from "selected atoms", which contradicted fix F10 and is wrong under the closure rule: an atom referenced by another selected atom cannot be removed, so uniform-over-selected puts mass on parents that do not exist. [ANALYSIS] Any fixed `P_B` leaves the target realizable; learned `P_B` is Tier 2 (v1.2 ablation).
- Conditional partition function: `logZ_θ(q)` head over the pooled instance embedding — conditional GFlowNet per **[EVIDENCE]** GFlowNet Foundations (JMLR 2023).

### 3.2 The seven learners (one file each, one shared trainer)

| # | Learner | Spec | Backing |
|---|---|---|---|
| L1 | **Supervised stepwise** | per-step multi-label CE: any not-yet-selected gold atom is positive; STOP positive when gold complete | **[EVIDENCE]** Graph-S3 (ACL 2026): stepwise supervision beats sparse terminal reward for graph retrieval (+15.6/+17.2) |
| L2 | **Canonical set imitation** | CE over gold sets with sampled insertion orders (order-marginalization by permutation sampling) | v1.2 §5.1; [ANALYSIS] implementation |
| L3 | **GRPO** | G = 8 samples/query; group-relative advantage `(r−mean)/std`; clipped policy-gradient; no critic, no reference model (documented adaptation from token policy to graph-action policy, as v1.2 §5.1 requires) | **[EVIDENCE]** DeepSeekMath (GRPO source); Memory-R1 (ACL 2026) precedent for GRPO on memory operations |
| L4 | **Trajectory Balance** | standard TB with `logZ_θ(q)` | **[EVIDENCE]** Malkin et al., NeurIPS 2022 |
| L5 | **SubTB(λ)** | λ = 0.9 default (config) | **[EVIDENCE]** Madan et al., ICML 2023 |
| L6 | **LED-GFN** | learnable transition potentials `φ_θ(s→s′)` used as local credit in a DB-style objective, with LED's two regularizers kept intact: (a) terminal consistency `Σφ = −log R(X)`, (b) trajectory-variance minimization. Follow the reference implementation (`hsjang0/LED-GFN`) | **[EVIDENCE]** LED-GFN (ICLR 2024 Oral) — built for exactly our case: intermediate energy unavailable and terminal utility non-additive |
| L7 | **Proposed: checker-conditioned LED** | = L6 with **one** addition and nothing else: `φ_θ` and the policy receive the checker's obligation delta `Δd = d(s)−d(s′)` as input features. **LED's terminal-consistency regularizer is untouched** (v1.2 §4.5.4 hard constraint). Capacity-matched control: widen L6's hiddens to equal L7's parameter count | **[HYPOTHESIS]** — Contribution 3; the entire claim is empirical |
| L7b | *Ablation arm only, not part of L7's definition* | L7 **+** an auxiliary head predicting the **terminal** deficit reachable from `s`, supervised by the realized rollout. Explicitly **not** a head predicting `d(s)`: `d(s)` is already an encoder input, so predicting it is a copy task that teaches nothing about credit assignment (fix F11) | [ANALYSIS] |

Shared trainer: on-policy sampling with ε-uniform exploration; optional prioritized replay flag (half of each replay batch from the top reward decile — **[EVIDENCE]** Shen et al., ICML 2023) default **off** for main runs, **on** only as the labeled ablation, so objective effects stay separable (v1.2 §5.1).

**Exit criterion = Gate 2.**
1. L4/L5 reach small exact TV on the lattice (sanity: the machinery works).
2. L6's learned potentials satisfy terminal consistency within tolerance.
3. **L7 vs capacity-matched L6, on a criterion predeclared before any run** (fix F12). Because the claim is about *credit assignment*, the criterion is improvement **at a fixed training budget**, not final converged performance — the framing FL-GFN and LED-GFN both use:
   - **Primary:** exact TV to target at a fixed number of sampled training trajectories, three seeds, paired bootstrap.
   - **Secondary 1:** trajectories required to reach a fixed TV threshold (sample efficiency).
   - **Secondary 2:** best-of-K valid-set utility at `checker_budget`.
   - **Decision rule:** L7 must beat capacity-matched L6 on the **primary** under the paired test. If it does not, **Contribution 3 is not supported and the thesis consolidates on Contribution 1** (v1.2 §9 fallback). This is the cheapest possible place to learn that.
4. L7b reported alongside as an ablation, never as a substitute for L7's result.
5. Audits: `FAIL` rate, collision rate, unconstructible rate (expected 0) all reported.

---

## Phase 4 — The 5 Tier-1 search algorithms (`setgen/search/`)

**Purpose.** Search baselines against the same pools, first on the lattice — running the **Gate-3 kill-shot early**: if a training-free method matches the learned samplers here, that is a project-level decision point, not a footnote.

All five implement `SearchModule.run(pool, obligations, scorer, budget) → [ProofSet]`, drawing on the shared ledger; the primary budget axis is **terminal `H` checks**, plotted across levels, wall-clock and FLOPs reported separately (v1.2 §5.2).

**What `scorer` is here, and why Phase 4 is not blocked on Phase 9 (fix F13).** On the synthetic lattice `U` is *exactly computable* — every term is deterministic and gold is known — so all five methods score with **exact `U`**. The distilled utility head (Phase 9.3) substitutes for it only on real data, where gold is absent. Two consequences to state in the write-up: the comparison here is fair (identical scorer for every method), and it is **optimistic** relative to deployment, because every method enjoys a perfect scorer. The Gate-3 synthetic table must be read with that caveat; the Phase-9 real-data table is the one that reflects a noisy scorer.

**S3 and S4 bypass the `ADD` masks** — they construct sets directly and are `H`-filtered afterwards, which is precisely why structural closure is a checker sub-check and not only a mask rule (fix F10; Phase 1.1 check 8).

| # | Method | Spec | Backing |
|---|---|---|---|
| S1 | Greedy | iteratively ADD the argmax `scorer` gain; STOP when allowed and gain < 0 | baseline |
| S2 | Beam | width b = K over partial sets, score = `scorer`; dedup by canonical set hash | baseline |
| S3 | **Submodular greedy** | `F(S) = 1.0·rel + 0.5·query-coverage + 0.4·saturated-facility-location + 0.3·concave-over-source-diversity`, cost-scaled greedy + Lin–Bilmes singleton fallback, then `H`-filter | **[EVIDENCE]** objective, weights, and algorithm from arXiv 2607.00725 (provisional venue, declared); guarantee from Nemhauser–Wolsey–Fisher 1978; applied form Lin & Bilmes, ACL 2011 |
| S4 | **PCST** | node prizes = relevance scores, uniform edge costs, `pcst_fast`; the connected result maps to a proof set; `H`-filter | **[EVIDENCE]** G-Retriever (NeurIPS 2024) formulates textual-subgraph retrieval as PCST — the closest classical algorithm to a connected proof subgraph |
| S5 | GFlowNet portfolio | sample K from the trained sampler, `H`-filter, rank by `scorer` | **[EVIDENCE]** the sample-then-filter pattern is the Robust Scheduling (ICLR 2023) result: diverse candidates under a cheap proxy beat proxy-optimization when the true evaluator is expensive |

**Exit criterion = Gate 3 (synthetic part).** Best-of-K utility at the fixed checker budget for S1–S5, three seeds, recorded. Explicit written decision: does the learned sampler beat S3/S4 on the lattice? If not, Stage D's claim narrows before the expensive real-data phases begin.

---

## Phase 5 — Stage A: ingestion and extraction (`ingest/`)

**Purpose.** Turns → verified, span-grounded assertions. Hybrid per pinned decision: local 7B for everything bulk.

| Component | Spec | Backing |
|---|---|---|
| `TurnIngestor` | writes immutable `Turn` + raw text to the event log | v1.2 §3.1 |
| `RollingSummary` | per-conversation summary refreshed asynchronously; extraction context = previous **m = 10** turns + summary | **[EVIDENCE]** Mem0 (ECAI 2025) uses exactly this context recipe |
| `Extractor` | Qwen2.5-7B-Instruct (4-bit), JSON-schema-constrained output: mentions, claims, values, time expressions, intra-turn relations — each with char offsets; also fills `ObligationParser` slots for questions | [ANALYSIS] model choice; schema per Phase 0 |
| `SpanGrounder` | verifies each extraction's offsets against the raw turn (exact, then fuzzy window); failures are dropped and counted | v1.2 §3.1 |
| `NLIVerifier` | off-the-shelf NLI cross-encoder scores `entailed_by_span`; threshold `tau_nli`; sets the flag, **never blocks storage** — an unentailed assertion is stored with `entailed_by_span=False` | **[EVIDENCE]** VeriCite: NLI verification is what drives citation quality (F1 77.73 → 68.91 without it) and an NLI model beat LLM verifiers on cost |
| `SupportPolicy` | classifies each stored assertion **eligible** or **quarantined** against `support_policy` (default strict: `entailed_by_span=True` at `tau_nli`). Both are written to the event log; **only eligible assertions are handed to Phase 6** (fix F9) | **[ANALYSIS]** the audit/active split — it is what stops an invented claim from becoming retrievable evidence, and it restores the project's founding requirement that extracted facts be supported by the raw turn before they are usable |

Assertions carry the four flags of v1.2 §3.1 — `asserted_by`, `entailed_by_span`, `externally_verified` (default false), `current_under_update_policy`. Grounding ≠ truth, encoded in the schema.

**Exit criterion.** On a pilot slice: span-grounding precision audited manually against the Gate-0 threshold; every stored assertion's spans resolve; **quarantine rate reported** — a high rate is an extraction-quality signal, never grounds for quietly lowering `tau_nli`; extraction throughput measured (budget check for full corpora).

---

## Phase 6 — Stage B: learned graph construction (`graphbuild/`)

**Purpose.** Contribution 1. Assertions → typed, versioned graph via propose → validate → commit.

### 6.1 Encoders (the Tier-1 ladder)

| Encoder | Spec | Backing |
|---|---|---|
| E1 MLP baseline | GraphMixer-style: MLP link-encoder + neighbor mean-pool + MLP head | **[EVIDENCE]** GraphMixer (ICLR 2023) matched or beat RNN/attention temporal GNNs — the simple baseline that must be beaten, or the thesis needs to know early |
| E2 HGT | 2-layer `HGTConv`, node/edge-type-dependent attention + relative temporal encoding | **[EVIDENCE]** HGT (WWW 2020) — the closest published encoder to this graph type (typed nodes, typed edges, time) |
| E3 Proposed | HGT backbone + GRAFT feature set: bge-small text embeddings, provenance flags, time deltas, degree | **[HYPOTHESIS]** — must beat E1 and E2 or Contribution 1's encoder story is "HGT suffices" |

### 6.2 Decoders (v1.2 §3.2, four decoders)

- **D1 mention resolution**: candidates = top-k existing entities (name-normalized + embedding) ∪ {`CREATE_NEW_ENTITY`, `NON_ENTITY`, `DEFER`}. Scored against the mention embedding. **[EVIDENCE]** the Missing-Entity/Non-Entity split is Learn to Not Link (Findings ACL 2023); emerging-entity difficulty is RAED (EMNLP 2025). Supervision interface: ConEL-2/NEL-style conversational EL labels.
- **D2 claim-pair relation**: `INDEPENDENT | DUPLICATE | CONFLICT | SUPERSEDES`, mutually exclusive, over pairs from the **pair proposer** (fix F8: top-s = 10 same-anchor similar claims, Mem0 pattern). Supervision interface: Gate-0 annotated conversational labels (the known annotation cost).
- **D3 relation type**: multi-relational link prediction over schema relations. Supervision interface: DialogRE/Re-DocRED-style labels.
- **D4 temporal**: validity-interval prediction per claim. Supervision interface: TORQUE/MATRES-style labels.
- Authority: metadata-derived, not learned (v1.2 §3.2).
- **Calibration**: temperature scaling on dev for all decoder confidences. **[EVIDENCE]** Guo et al., ICML 2017.

### 6.3 Commit pipeline

`eligible assertions only (fix F9) → propose (decoder outputs) → schema-validate (Phase-1 validator; rejects only formally checkable violations) → commit (event-log append)`. Quarantined assertions never reach the proposer, so no unsupported claim can enter the active evidence graph — they remain in the event log for audit. Supersession sets `t_invalid` + `superseded_by`; nothing is deleted (Zep precedent). A `DEFER` from D1 parks the mention in a revisit queue keyed by entity anchor.

### 6.4 LLM-prompted-linking baseline (the API budget's purpose)

Prompt an API model (GPT-4o-mini class) with the mention + candidate list + recent context; parse the same four-way action + ID. This is what Mem0/Zep actually do, so the comparison is faithful; calls are ledgered and capped. **[EVIDENCE]** Mem0's extraction/update pipeline is LLM-prompted (GPT-4o-mini).

**Exit criterion = Gate 1 (pilot).** On the annotated pilot: E1/E2/E3 × decoders compared on the **end-to-end D1 metric** (action *and* entity ID — v1.2 §6.4, three-number report), D2–D4 macro-F1, constraint-violation rate ≈ 0 by construction, corruption-after-sequential-updates audit runs green. Stop-or-redesign rule of v1.2 Gate 1 applies.

---

## Phase 7 — Stage C: hybrid candidate retrieval (`retrieve/`)

**Purpose.** Recall, not final selection (v1.2 §3.3). Union of cheap channels, capped pool.

| Channel | Spec |
|---|---|
| BM25 | `bm25s` over assertion text |
| Dense | bge-small embeddings, exact cosine top-k |
| Entity match | exact/alias hit on obligation anchor → that entity's local edges |
| Temporal filter | drop atoms whose intervals contradict the obligation's time constraint |
| Graph expansion | ≤ 2-hop from matched entities. **[EVIDENCE]** Beyond Static Retrieval (arXiv, provisional): 2 rounds is the cost–benefit optimum; the bottleneck is ranking, not coverage |
| GNN scorer | small query-conditioned relational GNN (≤ 8M params) scoring atoms against the question. **[EVIDENCE]** GFM-RAG (NeurIPS 2025): an 8M-param query-conditioned GNN does multi-hop retrieval in one pass, 0.107 s vs 3.162 s iterative; GNN-RAG (Findings ACL 2025): GNN retrieval beats LLM retrieval on multi-hop by 8.9–15.5 F1. Trained on gold-evidence pairs (interface: 2Wiki/MuSiQue supervision) |

Fusion: score-normalized weighted union → dedup by atom ID → cap at `pool_cap = 64` [ANALYSIS]. **Per-channel and union recall instrumented from day one** — if the pool misses the proof, nothing downstream can recover it (v1.2 §3.3), so this number is the ceiling for Phases 8–10.

**Exit criterion.** Sufficient-proof recall@pool measured on the pilot; per-channel ablation table produced; latency within the Mem0-class envelope (search p50 well under 0.5 s — **[EVIDENCE]** Mem0 reports 0.148 s flat / 0.476 s graph as the interactive bar).

---

## Phase 8 — Answerability gate (`gate/`)

**Purpose.** The decoupled abstention decision (v1.2 §4.2 — the flow has no abstain action, and the gate is why).

- **Features** [ANALYSIS]: obligation-slot coverage by the pool (anchor found? value-type candidate present? time constraint satisfiable?), max/mean channel scores, pool size, question embedding.
- **Model**: logistic regression or 2-layer MLP — deliberately small; the gate's value is the *decision protocol*, not capacity.
- **Threshold**: chosen on a dev risk–coverage curve. **[EVIDENCE]** selective-prediction methodology per Geifman & El-Yaniv (NeurIPS 2017).
- **Supervision interface**: answerable/unanswerable contrast pairs (MuSiQue-Full-style: pairs differing only in whether evidence exists) + benchmark abstention labels (LongMemEval evaluates abstention as a first-class ability).
- **Wiring**: gate says no → `OutputRecord(abstain)`; gate says yes but Stage D returns no valid set within budget → `abstain` via fallback, with the fallback rate logged (v1.2 §4.2 step 3).

**Exit criterion.** Risk–coverage curve produced on dev; gate integrated into the read-path orchestrator; false-abstention rate on answerable dev questions reported.

---

## Phase 9 — Stage D on real data (`setgen/` + `retrieve/` join)

**Purpose.** Swap the synthetic featurizer for the real one; train; run the portfolio.

1. **`AtomFeaturizer`**: candidate atoms carry Stage-B encoder embeddings + retrieval channel scores + obligation-match flags + `d(s)` deltas. Plugs into the frozen Phase-3 interface (fix F6) — learners are untouched.
2. **Training data**: (pool, obligations, gold proof) triples from gold-evidence corpora (2Wiki-style triples; MuSiQue decomposition), reward per Phase 1 with gold-derived sufficiency (fix F1). The Wikipedia→conversation transfer is a *declared* claim tested later, exactly as the plan's dataset section will specify. **Source-agnosticism is enforced, not asserted:** the loader consumes the abstract `(pool, obligations, gold_proof)` triple and no corpus-specific parser may appear above it — verified by an import-graph test — so a conversational-source variant is a drop-in if the transfer claim fails.
3. **Utility head distillation** (fix F1/F4/F13): a small MLP regresses train-time `U(X)` from the pooled set embedding. It is the `scorer` that substitutes for exact `U` at real inference, where gold is absent — used to rank valid sets and by S1/S2/S5 as the search scorer. It never touches `H`.
4. **Portfolio inference**: 1 greedy + K−1 = 7 sampled sets → `H`-filter → rank by utility head → tie-break minimal size → contested flag if top valid sets disagree on the answer binding (fix F4). Budget exhaustion reaches the `FAIL` terminal, which carries `r_fail` during training and maps to the fallback-abstain path at inference — the same outcome under both readings (fix F3).
5. **Bounded pools are principled, not just cheap** — **[EVIDENCE]** Generalization and Distributed Learning of GFlowNets (ICLR 2025): GFlowNet generalization bounds degrade with state-space size; `pool_cap` and `max_atoms` keep the state space small.

**Exit criterion.** All 7 learners train stably on real pools (loss curves, no NaN at reward floor); best-of-K at `checker_budget` computed for L1–L7 and S1–S5 on the same pools, three seeds — the Gate-3 (real) table.

---

## Phase 10 — Stage E: reader, orchestrator, ceiling oracles (`reader/`, `diagnostics/`)

### 10.1 Reader path

- **`ProofSerializer`**: stable claim IDs `[c1]…`, quoted source spans, token-capped. Ordering: anchor and answer-binding evidence at the beginning and end, connective evidence in the middle. **[EVIDENCE]** Lost in the Middle (TACL 2024): U-shaped position curve — models use context edges far better than the middle.
- **`Reader`**: Qwen2.5-3B-Instruct, bf16, greedy decoding, one fixed prompt template. The (prompt, decoding-config) SHA is stamped into every `OutputRecord` — v1.2 §3.5's "same frozen SLM, same prompt, same budget for every compared system" is enforced by hash equality, not by promise.
- **`AnswerParser`**: extracts answer + cited claim IDs; citation-to-span resolution for the final record.

### 10.2 Read-path orchestrator (fix F7)

`answer(question, snapshot_id) → OutputRecord` — wires C → gate → D → E with context-managed model loading (never two LLMs resident), full ledger accounting per query.

### 10.3 Ceiling oracles (`diagnostics/`) — the five-ceilings hooks, minimal

Cheap oracle modes, one function each (v1.2 §6.3): (1) extraction ceiling — gold statements represented as grounded assertions?; (2) graph ceiling — oracle search over the graph finds a sufficient proof?; (3) candidate ceiling — gold ⊆ pool?; (4) packing ceiling — gold survives serialization budget?; (5) reader ceiling — reader given the gold proof. These are diagnostics the gates consume; the benchmark protocol around them stays out of this document.

**Exit criterion.** End-to-end smoke test on the pilot: write path ingests a conversation, read path answers/abstains with citations; ceilings 2–5 runnable; config-hash equality verified across two runs.

---

## Phase 11 — System-baseline adapters (`baselines/`) — thin, last

All behind one interface: `System.run(conversation, question) → OutputRecord` (same record type, same reader, same ledger).

| Adapter | Spec |
|---|---|
| Full-context | entire history → Qwen2.5-3B with honest truncation policy. Non-negotiable comparator — **[EVIDENCE]** Mem0's own table has full-context at 72.90 LLM-judge, above every memory system it tested |
| Matched-budget RAG | BM25+dense top-k over raw turns, packed to the **same token budget** as GRAFT's serialized proofs |
| Mem0 | via its OSS library (API-backed; draws on the hybrid API budget; ledgered) |

Marked eval-adjacent: build only when Gate-4 comparisons are actually scheduled.

---

## 12. Values that must be frozen at Gate 0 (before any comparison run)

| Value | Default | Where used |
|---|---|---|
| `beta`, `u_weights` | §Phase 0 | reward; identical for every learner (v1.2 §5.1 — else the comparison measures reward engineering) |
| `K = 8`, `checker_budget = 32` | §Phase 0 | Stage-D primary metric **and** search comparison — one constant everywhere (fix F5) |
| `pool_cap = 64`, `max_atoms = 16` | §Phase 0 | state-space bound |
| `tau_nli`, gate threshold | audited/dev-chosen, then frozen | Phases 5, 8 |
| `r_fail` | 1e-6 | `FAIL` is a member of the target support; changing it re-derives `p*` (fix F3) |
| closure rule | atom addable only when every referenced atom is selected | masks, `H` check 8, `P_B`, the exact DP (fix F10) |
| `support_policy` | strict: `entailed_by_span=True` at `tau_nli` | commit gate and `H` check 7 (fix F9) |
| L7-vs-L6 decision rule | exact TV at fixed training budget, 3 seeds, paired | Gate 2 (fix F12) — must be fixed *before* any run |
| seeds = {13, 42, 7} | §Phase 0 | every trained method (**[EVIDENCE]** Dror et al., ACL 2018 — predeclared protocol) |
| splits: chronological + user-level | Gate-0 contract | all training |

---

## 13. What this architecture deliberately does **not** contain

- No microservices, no database server, no ANN index, no distributed training, no experiment-tracking platform beyond JSONL + config hashes.
- No learned backward policy, no FL-DB, no GAFlowNet, no PPO, no MIP/MCTS/diverse-beam, no CompGCN/R-GCN, no SynCheck monitor — all Tier 2/3 (§14).
- No benchmark evaluation protocol, no annotation guidelines, no dataset preparation — per user instruction, with the two carve-outs of §0.2.

## 14. Tier-2/3 registry stubs (one paragraph, total)

Three registries make later additions refactor-free: `LossRegistry` (a Tier-2 learner = one new loss module against the frozen policy interface — FL-DB and GAFlowNet each need only a transition-energy/intrinsic-reward hook already present as `d(s)` plumbing), `SearchRegistry` (MIP via OR-Tools, MCTS, diverse beam implement the same `SearchModule` protocol), `EncoderRegistry` (CompGCN/R-GCN swap behind `StateEncoder`). SynCheck, if ever used, is a post-reader monitor slot in the orchestrator with its cost ledgered (v1.2 §3.5).

---

## 15. Citation ledger (everything load-bearing above, grouped)

**Objective & learners:** GFlowNet Foundations (JMLR 2023) · Trajectory Balance (NeurIPS 2022) · SubTB(λ) (ICML 2023) · LED-GFN (ICLR 2024 Oral) · Shen et al. (ICML 2023) · Symmetry-Aware GFlowNets (ICML 2025) · When Do GFlowNets Learn the Right Distribution? (ICLR 2025) · Generalization & Distributed Learning of GFlowNets (ICLR 2025) · Robust Scheduling with GFlowNets (ICLR 2023) · DeepSeekMath/GRPO (arXiv 2024) · Memory-R1 (ACL 2026) · Graph-S3 (ACL 2026).
**Search:** G-Retriever/PCST (NeurIPS 2024) · Nemhauser–Wolsey–Fisher (1978) · Lin & Bilmes (ACL 2011) · What Survives Into Context (arXiv 2607.00725, *provisional*).
**Graph & memory:** HGT (WWW 2020) · GraphMixer (ICLR 2023) · Zep (arXiv, *vendor*) · SEEM (ACL 2026) · Mem0 (ECAI 2025) · Learn to Not Link (Findings ACL 2023) · RAED (EMNLP 2025).
**Retrieval:** GFM-RAG (NeurIPS 2025) · GNN-RAG (Findings ACL 2025) · Beyond Static Retrieval (arXiv, *provisional*).
**Reader & verification:** Lost in the Middle (TACL 2024) · VeriCite (SIGIR-AP 2025) · ALCE (EMNLP 2023).
**Methodology:** Geifman & El-Yaniv (NeurIPS 2017) · Guo et al. (ICML 2017) · Dror et al. (ACL 2018).
Reader-size condition: arXiv 2607.00725 (*provisional, declared*).
