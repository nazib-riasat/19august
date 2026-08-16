# GRAFT — Dataset Decision

**Every dataset the project will use, which phase uses it for what, what it costs, and on which machine.**

Date: 15 August 2026
Parent: `GRAFT_RESEARCH_PLAN_v1.md` v1.2 §6.1 · `GATE0_CONTRACT.md` items 1–3, 9 · `GRAFT_EXECUTION_ARCHITECTURE_v1.md` Phases 5–11 · `CLAUDE.md` §7
Status: **Recommendation — and item 9 has since taken it (15 Aug 2026): scope c, 200 questions, extending to scope b′ for final numbers**, recorded in `GATE0_CONTRACT.md` item 9, which signed the same day. §5 below is the reasoning that decision adopted; the rest of this document remains the dataset reference it was.
Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (engineering judgment made here).

**Method.** Four parallel surveys over the 103-paper library plus outside literature, covering ~50 candidate datasets; every size, licence and label semantics traced to a paper section, a dataset card, or a file on disk. An independent verification pass over the fourteen adopted candidates was launched and **did not complete** (session limit) — so the adoption reasons below are single-sourced, and §8 lists exactly what that leaves unchecked.

---

## 1. Two findings that change the plan

### 1.1 `CLAUDE.md` §7 is falsified for the supersession half

The row reads *"Conflict/supersession annotation — **No dataset provides it.**"* That is **no longer true**, and the correction is actionable rather than cosmetic.

**Memora** (arXiv 2026, provisional) ships a machine-readable memory trace: every session carries `operation ∈ {add, update, delete}` with `operation_details.item` naming the memory record acted on, plus `created_at`. An `update` op is a literal (antecedent, successor) pair — **exactly D2's `SUPERSEDES` class, on conversation, already labelled**. Volume: 103.2 memory ops per persona on the weekly track, 13% of them updates. Licence Apache-2.0, no restriction.

What it does **not** fix: `CONFLICT`. Memora's schema has no notion of two claims that disagree without one replacing the other, so the class the live measurement says is missing (**zero CONFLICT pairs across 49 annotated items** — `chatcontext1.md` §3a) is still unsupplied. And its conversations are LLM-simulated with a 5% human spot-check, so it encodes a synthetic prior on how updates happen.

**Decision:** adopt Memora as **tertiary evaluation + free `SUPERSEDES` supervision** at 3 weekly personas. Amend `CLAUDE.md` §7 to say *conflict* annotation has no source — which is the true and narrower claim.

### 1.2 The abstention evidence base is 30 questions, and there is a 15× fix that costs nothing

Contribution 2's decoupled answerability gate is evaluated by a risk–coverage curve (Geifman & El-Yaniv, NeurIPS 2017). LongMemEval supplies **30** abstention questions. At n = 30 one flipped decision moves the score 3.3 points; that cannot carry a contribution.

**LoCoMo has 446 adversarial (unanswerable) questions** and is already being ingested for other reasons — marginal cost zero. Mem0 and Chain-of-Memory both *exclude* the adversarial category "to ensure fair comparison", so reporting it is a differentiator, not a comparability loss.

**Decision:** LoCoMo's adversarial subset becomes the **primary abstention testbed**; LongMemEval's 30 stay as the in-domain check, reported with an interval, never a point estimate.

---

## 2. The finalised dataset list, by phase

### Phase 5 — Stage A ingestion (the corpora that get extracted)

| Dataset | Role | Volume ingested | Licence |
|---|---|---|---|
| **LongMemEval-S** | **Primary evaluation corpus.** Ingested; its train slice supervises D1/D2 and Stage C; its test slice is held-out end-to-end eval | Scope decision, §4 | MIT (SHA-pinned in `graft/ingest/corpus.py`) |
| **LoCoMo** | **Secondary zero-shot evaluation.** Never trained on, in any component. Its 446 adversarial questions are the abstention testbed (§1.2) | 5,875 turns | **CC BY-NC 4.0 — non-commercial.** Thesis use fine; do not release a derived dataset commercially |
| **Memora**, weekly track, 3 personas | **Tertiary eval** (FAMA directly scores whether invalidated memories leak into answers = Contribution 1's claim) **+ free `SUPERSEDES` pairs** | 7,486 turns | Apache-2.0 |

### Phase 6 — Stage B decoder supervision

| Decoder | Primary source | Secondary / optional | Note |
|---|---|---|---|
| **D1** mention resolution | **GRAFT's own LongMemEval annotations** — the only source of all four actions and the only one on conversation | **NEL / Learn to Not Link** (Findings ACL 2023): 9,924 entries, 3,331 NIL split into *Missing Entity* (17%) vs *Non-Entity Phrase* (83%) — **the only dataset anywhere that labels `CREATE_NEW_ENTITY` vs `NON_ENTITY` per instance**. `ConEL-2` (CIKM 2022, MIT) for conversational mention detection | NEL has no `DEFER` and its 83% Non-Entity share is a construction artefact — report it, never as a natural prior. **NEL's repo has no LICENSE file** — resolve before use |
| **D2** claim-pair relation | **GRAFT's own annotations** (only source of all four classes) | **Memora** `update` ops → `SUPERSEDES` (§1.1) | **The blocker is the proposer, not the labels**: zero CONFLICT pairs surfaced in 49 items. More annotation hours cannot fix a proposer that never proposes the pair |
| **D3** relation type | **DialogRE** (ACL 2020) — 1,788 Friends dialogues, 36 relations, genuinely conversational (>90% of triples involve a speaker) | **Re-DocRED** (EMNLP 2022, MIT) | Native-label training through the frozen interface (decision 7); the 27.1% / 4.88% figures are **schema-application** losses, not training losses. Re-DocRED's `same_as` half is **dead code** — P1449/P742/P1477 have zero instances in train |
| **D4** temporal validity | **GRAFT-internal**: left endpoint = the asserting turn's timestamp (free); right endpoint = derived from D2's `SUPERSEDES` via the commit path | **TORQUE** (Apache-2.0) as the native temporal interface. **EventTime** (ACL 2016) is the *only* text-grounded corpus emitting begin/end **intervals** — 1,498 events, α = 0.617 | **The re-framing that matters: D4's right endpoint is a function of D2's `SUPERSEDES`.** No amount of temporal-ordering data (TORQUE, MATRES, TDDiscourse) can supply it — they order events, which is not what D4 emits |

### Phase 7 — Stage C retrieval

| Dataset | Role |
|---|---|
| **LongMemEval distant signal** (`answer_session_ids`) | GNN scorer training (Gate-0 item 2). **Sound choice — but the contract's stated reason is wrong**: it says repairing the coarse signal needs turn-level annotation "which is item 8's budget", yet the corpus already ships `has_answer` per turn and `scripts/phase2_5/sample_turns.py` reads it. The real justification is that `has_answer` defines the *evaluation* target, so training on it would collapse Tier-A recall into a self-fulfilling number |
| **TopiOCQA** (TACL 2022, CC-BY 4.0) — *optional* | 45,450 turns of conversational passage relevance; an external sanity check that the query encoder works before it meets LongMemEval |

### Phase 8 — answerability gate

| Dataset | Role |
|---|---|
| **MuSiQue-Full** contrast pairs | **Primary.** 49,628 questions; the question string is byte-identical across each pair, so the classifier *cannot* score answerability from wording and is forced onto pool-side features — exactly the declared feature set |
| **LoCoMo adversarial** (446) | **Primary evaluation** (§1.2) |
| **LongMemEval abstention** (30) | In-domain check, evaluation only, reported with an interval |
| **QuAC** (EMNLP 2018, CC BY-SA 4.0) — *optional* | ~20K naturally-arising conversational unanswerables; 600× LongMemEval's 30. The questioner genuinely could not see the passage, unlike SQuAD 2.0's adversarial construction |
| **IIRC** / **FEVEROUS NEI** — *optional* | 30% unanswerable from genuinely absent evidence; FEVEROUS labels insufficiency **relative to an evidence set**, GRAFT's exact framing, and documents the construct-negatives-by-deleting-evidence recipe |

### Phase 9 — Stage D evidence-set learner

| Dataset | Role | Why |
|---|---|---|
| **2WikiMultiHopQA** | **Primary training.** 167,454 train questions; evidence = a **set** of (subject, property, object) triples | The only source whose annotation is simultaneously a set and a typed relational path. Human evidence-F1 78.81 vs baseline 14.94 = the headroom the claim needs |
| **MuSiQue-Ans** | **Primary training.** 19,938 train; gold supporting-paragraph sets **+ the decomposition DAG** | The only one shipping the *obligations* element of the `(pool, obligations, gold_proof)` triple; 2Wiki's must be synthesised |
| **HoVer** (Findings EMNLP 2020) | **Adopt — the strongest new find.** 18,171 train claims, of which **3,035 are 4-hop**, evidence spanning up to 4 articles | 2Wiki/MuSiQue/HotpotQA are all thin at ≥3 hops. Minimality and the closure rule only bite when proof sets are genuinely large |
| **FEVER / FEVEROUS** — *optional but strategically important* | 185,445 claims with **296,712 evidence sets** — ~1.6 **alternative complete proofs per claim** | **Contribution 2's target `p*` is a distribution over the set of valid terminals.** Every other corpus annotates *one* gold proof, which can only teach a point target. This is the only corpus at scale whose annotation says "these distinct sets are each complete." **Adopt FEVEROUS's scoring convention unconditionally**: a prediction scores 1 iff at least one complete gold set is a subset of the predicted set — GRAFT currently has no published convention to cite for multi-proof credit |
| **LongMemEval train-slice proofs** | **The conversational track** — `has_answer` turns → eligible assertions → closed → H-minimised (contract item 3.2) | Turns the Wikipedia→conversation transfer gamble into a **measured ablation** instead of a bet |
| **HotpotQA** — *optional* | Testbed for the training-free Gate-3 baselines only | Keep it in that role — it is the surface submodular/PCST were tuned on, so the comparison runs on their home turf. **Do not** promote it to Stage-D training: much of it is solvable single-hop |
| **QASPER** / **StrategyQA** — *optional diagnostics* | QASPER is the only corpus where a human was *instructed* to select the **minimal** evidence set — the only external check on GRAFT's H-minimisation. StrategyQA annotates evidence **per reasoning step** = per obligation | Both small; diagnostics, not training sources |

### Phase 3 — unchanged

**Synthetic ProofLattice only.** Gate 2's metric is exact TV to the declared target, computable only on an enumerable environment [EVIDENCE: Shen et al. ICML 2023; *When Do GFlowNets Learn the Right Distribution?* ICLR 2025]. Swapping in real data makes Contribution 3's predeclared decision rule unfalsifiable. Real data enters at Phase 9 through the frozen policy interface (fix F6).

---

## 3. What the training/evaluation boundary looks like when finished

```
TRAINED ON                                    EVALUATED ON (never trained)
──────────────────────────────────────────    ──────────────────────────────────
2Wiki + MuSiQue + HoVer      → Stage D  ─┐
LongMemEval train slice      → Stage D  ─┤    LongMemEval test slice  (in-domain)
LongMemEval train slice      → Stage B  ─┼──► LoCoMo, entire          (zero-shot)
LongMemEval distant signal   → Stage C  ─┤    Memora weekly           (supersession)
MuSiQue-Full + QuAC          → Phase 8  ─┘
NEL, DialogRE, Re-DocRED, TORQUE → D1/D3/D4 pre-training
Synthetic lattice            → Phase 3
```

**[EVIDENCE] Why this shape publishes.** The modal accepted protocol has three rules, verified across the library: (1) LoCoMo/LongMemEval are evaluation-only for 6 of 7 memory systems; (2) learned components train on a *different* dataset's official train split (GFM-RAG, GNN-RAG, Graph-S3, G-Retriever, RoG); (3) the **strongest** shape adds transfer numbers — GFM-RAG reports in-domain **and** zero-shot on 7 unseen datasets (§4.6); SubgraphRAG reports trained-on-one-applied-to-other; Memory-R1 (ACL 2026) trains on 152 LoCoMo questions and evaluates zero-shot on MSC and LongMemEval. The cautionary tale is also in the library: RoG passed ICLR 2024 training jointly on WebQSP+CWQ, then SubgraphRAG exposed it as label leakage (>50% of test questions in the other's train set). **Declare splits up front, user-level, never joint.**

---

## 4. GPU time per phase, across the three machines

**One measured anchor.** 135.7 turns/hour end to end, Phase-5 live pilot, 248 turns, dev RTX 5050. Extraction is 99.87% of that wall clock, so the whole pipeline scales with GPU throughput. Single-stream decode is bandwidth-bound (Qwen2.5-3B at bf16 reads 6.18 GB per forward pass), so the projection is the bandwidth ratio: **T4 ×0.781 each, RTX 5090 ×4.667.** Two T4s do not speed up one stream; they run two workers, so Kaggle's usable figure is **×1.562**.

| Phase | What actually runs on the GPU | RTX 5050 8 GB (dev) | Kaggle 2×T4 16 GB | RTX 5090 32 GB |
|---|---|---|---|---|
| **Phase 3** — synthetic learners (Gate 2) | **nothing.** `TrainSpec.device = "cpu"`, numpy sampler, ~52k-param MLPs | **39 h serial** (12 calibration + 27 matrix, 1 h ceiling each); the 39 runs are independent, so 16 cores cut it to **~3 h wall clock** | ~10 h wall clock (4 cores) — and it spends GPU quota on an idle GPU | **39 h serial** — GPU irrelevant; wall clock depends only on the host's core count |
| **Phase 5** — Stage A ingestion · LongMemEval 200 q (4,384 turns) | extractor (Qwen2.5-3B bf16) + NLI | **32.3 h** | 20.7 h ⛔ | **6.9 h** |
| ⤷ LongMemEval evidence-only (10,960) | " | 80.8 h | 51.7 h ⛔ | 17.3 h |
| ⤷ LongMemEval evidence + 2 distractors (20,798) | " | 153.3 h | 98.1 h ⛔ | **32.9 h** |
| ⤷ LoCoMo, entire (5,875) | " | **43.3 h** | 27.7 h ⛔ | **9.3 h** |
| ⤷ Memora weekly ×3 personas (7,486) | " | **55.2 h** | 35.3 h ⛔ | **11.8 h** |
| ⤷ LongMemEval full haystack (246,930) | " | 1,820 h | 1,165 h ⛔ | 390 h — *not a candidate on any machine* |
| **Phase 6** — Stage B, Gate-1 matrix (4 arms × 3 seeds) | bge-small embedder + encoders (E1 0.05M / E2 3.76M / E3 4.26M) | **~1–3 h** ~ | ~1–3 h | **~1–2 h** ~ — models are tiny; graph replay is CPU-bound |
| ⤷ D3/D4 pre-training (DialogRE, Re-DocRED, TORQUE) | same encoders, native label sets | ~2–6 h ~ | ~2–6 h | ~1–4 h ~ |
| **Phase 7** — Stage C GNN scorer (≤8M) | one forward per question-pool, ≤64 atoms | **< 1 h** | < 1 h | **< 1 h** — too small to differentiate |
| **Phase 8** — answerability gate | **the embedder, over MuSiQue paragraphs** — *not* "none" (corrected 15 Aug 2026) | **~1–4 h** ~ CPU, **~20–40 min** on the dev GPU | ~1–4 h ⛔ | **~15–30 min** |
| ⤷ *of which*: LR / 2-layer MLP training, 2 arms × 3 seeds | nothing — CPU | **minutes** | minutes | minutes |
| **Phase 9** — Stage D pool prep (2Wiki + MuSiQue + HoVer) | bge-small embedder over the pools, one-time | **~4–8 h** ~ | ~3–5 h | **~1–2 h** ~ |
| ⤷ Stage D training, 7 learners × 3 seeds | **nothing** — CPU, same as Phase 3 | **~33 h**, worst case ~7 d | slower (4 cores) | ~33 h — no benefit |
| ⤷ Stage D conversational-track ablation | nothing (CPU) | **+ ~1 d** | slower | + ~1 d |
| **Phase 10/11** — eval, LongMemEval test (~100 q) | frozen Qwen2.5-3B reader | GRAFT **0.5–1 h**; full-context 2–5 h | ⛔ | GRAFT **~10 min**; full-context 0.5–1 h |
| ⤷ eval, LoCoMo (1,986 q × 4 systems) | " | **~16–52 h** | ⛔ | **~3.5–11 h** |

`~` = estimated, no measured epoch time exists.

**Phase 8's row was wrong until 15 Aug 2026 and the correction is worth keeping.**
It read "GPU: none · < 1 h", which is true of the *classifier* and false of the
*phase*: `GRAFT_PHASE8_BUILD.md` G8 has the MuSiQue adapter compute BM25/dense
channel scores "over paragraphs by the same `bm25s`/pinned-embedder stack", and
MuSiQue-Full ships ~20 paragraphs across 49,628 questions — order 10⁶ paragraph
slots, several hundred thousand of them unique after the content-keyed cache
dedups the contrast twins. **The embedding pass is the entire cost; training is
minutes.** The lesson is the row's own: "GPU: none" was read off the model class
rather than off the feature pipeline that feeds it.
**⛔ = the number is what the hardware *would* deliver; the machine cannot run the frozen configuration at all,** so the figure is unusable for any comparable run — see below.

### Why Kaggle is marked ⛔ on every LLM row

**A T4 cannot execute the frozen extractor or the frozen reader at all.** Turing is compute capability 7.5; bfloat16 requires ≥ 8.0. The only fallback is fp16, and `ingestion_fingerprint()` hashes the extractor dict **including `dtype`** — verified locally, bf16 gives `bf176a37…` and fp16 gives `df7f3e22…`. Configuration identity is the one cross-machine property the project actually promises, so a Kaggle run silently produces a differently-identified corpus that cannot be compared with the pilot.

Quota compounds it: 12 h/session and 30 h/week mean the 51.7 h evidence-only ingestion needs ≥ 5 sessions across ~2 weeks — **slower in calendar time than the dev laptop's 3.4 days.**

Kaggle's two defensible roles: Phase 0–2 test runs (zero ML dependencies, by design) and a **≤ 1 h instrumented calibration session** to replace the projections in this table with a measurement.

### Reading the table

- **The 5090 only matters for Phase 5 and Phase 10/11.** Those are the LLM rows, and they are ~85% of total GPU wall clock. Phases 3, 6, 7, 8 and Stage-D training barely move — three of them do not use a GPU at all.
- **The critical path at the recommended scope** (200 q + LoCoMo + Memora + eval) is **~9–10 days of continuous GPU on the dev card, ~2 days on a 5090.** Phase 3 and Stage-D training (CPU) overlap it entirely.
- **A 5090 makes scope b′ affordable** — 32.9 h against 153.3 h — which is the difference between a retrieval claim with distractors and one without. Note 575 W TGP and a 1000 W minimum PSU.

**A defect this exposed:** `scripts/phase5_pilot.py` sets `kaggle_factor = 1.0`, so every row of the sizing memo has `hours_kaggle_projected == hours_dev_gpu` exactly. Exit criterion 16 ("all three scopes on **both** hardware targets") is green on a table whose second column carries no information. The assumption is honestly labelled in the docstring; the artefact still reads as two measurements.

---

## 5. Recommended scope decision (Gate-0 item 9)

**Scope c at 200 questions (4,384 turns, 32.3 h)** for the first full run, extending to **scope b′ (evidence + 2 distractor sessions, 153.3 h)** for the final numbers if the calendar allows or a 5090 arrives.

Reasoning: evidence-only sessions make retrieval artificially easy — Stage C's recall is measured against a haystack that contains almost nothing but the answer, and ceiling 3 becomes uninformative. Distractor sessions are what make the retrieval claim mean something. 200 questions is also the point where the end-to-end test slice (~40 held-out questions at 60/20/20) stops being embarrassing.

---

## 6. Rejected, with reasons

| Dataset | Why rejected |
|---|---|
| **BEAM** (ICLR 2026) | **The strongest benchmark found and still a no.** Only one covering knowledge-update + contradiction-resolution + abstention + temporal reasoning together — and "contradiction resolution" is the CONFLICT class nothing else supplies. Killed by cost: the smallest usable unit is one 100K-token conversation ≈ 24.7 h, and a meaningful sample is months. **Revisit if a 5090 lands.** |
| **AIDA-CoNLL, MSNBC, AQUAINT, WNED-*** | 0% NIL, or NIL unlabelled — they can only teach `LINK_EXISTING`. AIDA's raw text also sits behind a NIST agreement, which breaks the SHA-pinned loader discipline |
| **ZESHEL** | "Zero-shot" means unseen *entities*, not open-world NIL |
| **DocRED** | Superseded by Re-DocRED (its false negatives cost ~13 F1 for recall-oriented models) |
| **TACRED, CrossRE** | Sentence-level; wrong granularity |
| **MATRES, TimeBank-Dense, TDDiscourse, TempEval-3** | All emit event *ordering*, not validity *intervals* — the wrong quantity for D4 (§2, D4 row) |
| **YAGO11k / Wikidata12k** | Do store intervals, but are pure KG triples with no text to ground them |
| **DocNLI, ContraDoc, WikiContradiction** | Document contradiction ≠ conversational claim-pair CONFLICT; no antecedent/successor structure |
| **MSC, PersonaMem, DMR, MemoryAgentBench, DialSim, PerLTQA, MemoryBank, EverMemBench** | No supersession/conflict labels, or no evidence labels, or redundant with LoCoMo at added ingestion cost |
| **LongMemEval-M** | Scaling the existing primary (1.5M tokens/question), not adding a capability |
| **RULER / LongBench / InfiniteBench / ZeroSCROLLS** | Long-context stress tests, not conversational memory |
| **INSCIT, OR-QuAC** | Evidence is *external Wikipedia*; GRAFT's atoms come from the conversation's own turns. Structurally the wrong shape |
| **SQuAD 2.0** | Adversarially-written negatives teach reading comprehension, not retrieval coverage. MuSiQue-Full's identical-question pairs isolate the right signal |

---

## 7. Consequential side-findings

1. **Novelty risk to Contribution 1.** *Supersede* (arXiv 2606.27472, 25 Jun 2026) builds a trainable RL environment on LongMemEval's knowledge-update subset whose reward **is** supersession-correctness, and its authors claim "no existing work sits in the intersection". It does not do provenance-preserving graph construction, entity creation, or non-destructive versioning — **C1 survives** — but the claim must now be stated against it, and it is three weeks old. Add to §5.3's related work.
2. **A code defect — found and FIXED 15 Aug 2026.** `torque_items` parsed only the **train** shape (a list of annotator HITs). TORQUE's dev file is a dict keyed by passage id, whose `question_answer_pairs` are keyed by question text and whose `events` is a single mapping rather than a list — three shape differences, and the loader raised `AttributeError: 'str' object has no attribute 'get'` on dev. Reproduced, normalised in `_torque_passages` / `_torque_qa_pairs` / `_spans_of`, regression-tested against both shapes. Dev now loads 1,483 items (434 default-questions) beside train's 24,523.
3. **Re-DocRED's `same_as` mapping is dead code.** P1449/P742/P1477 have zero instances in train, so all 4,194 mapped instances are `has_value`. The mapping table implies a coverage it does not have.
4. **The corpus already ships what the contract says needs annotating.** Gate-0 item 2's stated reason for the coarse Stage-C signal is wrong (§2, Phase 7). Fix the justification, keep the decision.

---

## 8. What is not verified

Honesty section, because the verification pass did not finish.

- **The independent adversarial check over the 14 adopted datasets did not run** (session limit). Sizes, licences and label semantics come from one reading each. Before committing to a *new* dataset — NEL, HoVer, FEVER/FEVEROUS, EventTime, Memora, QuAC — re-verify its splits and licence against the primary source.
- **NEL has no LICENSE file.** Resolve before use.
- **Three licences unresolved:** TempEL, TimeQA, EventTime.
- **The 5050's 384 GB/s bandwidth is derived, not published**, and one major spec database contradicts it by 1.7×. Every projection inherits that uncertainty.
- **Kaggle's quota accounting for 2 GPUs is unverified** — if billed per-GPU-hour rather than per session-hour, every 2×T4 figure doubles in quota cost.
- **The 25.8% MBU calibration is pessimistic for Linux targets**: it includes prefill, and the grammar-constrained decode runs xgrammar's slow `torch_native` backend because **Triton has no Windows build**. A Linux 5090 may beat the 4.67× projection.
- **Phase-6 epoch wall-clock has never been measured** — the `~1–3 h` figure is an estimate from parameter counts and item volumes.
- **Cheapest way to retire most of this:** one instrumented Kaggle T4 session running the existing bakeoff harness on the pinned 60-turn calibration slice, under a deliberately non-frozen fp16 config marked as an instrument run. Under 1 h of quota, and it replaces `kaggle_factor = 1.0` with a measurement.
