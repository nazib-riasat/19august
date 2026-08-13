# GRAFT — Research Plan v1.2

**Provenance-preserving temporal graph memory with a checker-conditioned evidence-set learner**

Date: 7 August 2026 (v1.2 — internal consistency pass; see §1.7 and §1.8)
Supersedes: `GRAFT2_9_EPISETFLOW_PIPELINE.md` (learning core), and the merged plan audited in `GRAFT_MERGED_PLAN_EVIDENCE_CROSSCHECK.md`
Related: `GRAFT2_9_EVIDENCE_REVIEW.md` (component-by-component literature audit)

---

## 0. Document status and evidence rules

This is an implementation plan, not a results claim.

Three labels are used throughout and are never mixed:

| Label | Meaning |
|---|---|
| **[EVIDENCE]** | A specific published paper supports this. Venue is stated. |
| **[HYPOTHESIS]** | A design decision this project will test. No paper establishes it. |
| **[ANALYSIS]** | A mathematical or logical consequence derived here, not taken from a paper. |

Venue tiering, applied consistently:

- **Primary evidence:** ICLR, ICML, NeurIPS, ACL/EMNLP main, TACL, JMLR.
- **Qualified evidence:** Findings papers, ECAI, SIGIR-AP, regional venues. Usable, labelled.
- **Provisional evidence:** arXiv preprints and workshop papers. May motivate an experiment; never the sole support for a central claim.

**No paper establishes that this system will beat its baselines.** Papers justify components and motivate hypotheses. Superiority is established only by this project's own controlled experiments, or not at all.

---

## 1. Corrections adopted before implementation

This section records what changed relative to earlier drafts and why. Three of these correct errors in my own earlier analysis; they are stated plainly so they are not reintroduced.

### 1.1 The terminal reward was mathematically wrong

**Earlier (wrong):** `flow_stop = exp(β × checker_score)`, with score 0 for anything failing a hard check.

**The error [ANALYSIS]:** `exp(β·0) = 1`. A hard-invalid set would receive reward **1**, not 0 — a positive reward, and in a low-utility regime possibly a *competitive* one. This is a straightforward algebra error and it inverts the intended safety property.

**Corrected form** (§4.1): a multiplicative hard-validity indicator, not an additive score of zero.

### 1.2 A deficit **vector** is not an FL-GFN energy

**Earlier (wrong):** "use FL-DB where the intermediate energy is the proof-deficit vector."

**The error [EVIDENCE]:** [FL-GFN (ICML 2023)](https://arxiv.org/html/2302.01687) Assumption 4.1 requires a **scalar** function `ℰ : S → ℝ`, with transition energy `ℰ(s→s′) = ℰ(s′) − ℰ(s)`, and Proposition 4.2's guarantee holds only when the terminal energy agrees with `−log R`. A six-component vector is not that object.

A scalar potential *built from* the deficit vector can qualify — but only if it satisfies the terminal boundary condition. §4.5 works through when it does and when it does not.

### 1.3 Graph-S3 does not validate the FL-GFN bridge

**Earlier (wrong):** pairing FL-GFN theory with Graph-S3 results as "theory + same-domain empirics" for the proposed method.

**The error [EVIDENCE]:** [Graph-S3 (ACL 2026)](https://aclanthology.org/2026.acl-long.1169/) trains an **LLM-based** retriever with synthetic stepwise supervision against offline-extracted golden subgraphs. It uses no GFlowNet, no FL-DB, no learned energy, and no terminal-distribution target. Its reported **+8.1% accuracy / +9.7% F1 over seven baselines** supports one thing: *dense stepwise supervision beats sparse final-answer reward for agentic graph retrieval, in its own setting.*

That makes Graph-S3 a **strong supervised baseline this project must run**. It does not validate any GFlowNet construction.

### 1.4 The FL-GFN set experiment is not this task's shape

**Earlier (overstated):** "set generation — your exact task shape."

**Correction [EVIDENCE + ANALYSIS]:** FL-GFN's set-generation benchmark uses **fixed** set sizes (|S| = 20 / 60 / 80) with an **additive** per-element energy. Evidence selection here has variable stopping, hard validity predicates, temporal and authority interactions, and non-additive sufficiency. The result makes FL-GFN a mandatory baseline. It does not transfer.

### 1.5 A deterministic checker is not a "true evaluator"

**Earlier (overstated):** framing the checker as the true metric in the [Robust Scheduling (ICLR 2023)](https://openreview.net/forum?id=ZBUthI6wK9h) analogy.

**Correction [ANALYSIS]:** in Robust Scheduling the expensive true metric was *measured hardware runtime* — ground truth. A rules engine verifies exactly the rules encoded in it. Type constraints, ID uniqueness, interval arithmetic and schema legality are formally checkable. Entailment, factual correctness and source authority are not. A neural scorer at temperature 0 is repeatable, not formal.

Robust Scheduling therefore supports a **proxy-mismatch hypothesis** (diverse candidates can beat proxy optimization when the evaluator is expensive), not a claim that a GFlowNet is uniquely required. Diverse beam, MCTS and ensembles also produce candidate portfolios.

### 1.6 Other corrections adopted

| Correction | Basis |
|---|---|
| SynCheck is not free — it adds a trained monitor and inference cost; FOD changes decoding and must share the compute budget | [SynCheck, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.527/) |
| Published scores from different memory systems are **not** a controlled comparison; re-run with matched reader, prompt, history, budget and judge | [ANALYSIS] |
| Remove all "X% of the plan is sound" estimates and all predictions of likely wins | [ANALYSIS] — those were engineering judgments stated with unearned confidence |
| Split the single "graph ceiling" into five ceilings | §6.3 |
| Use FL-**DB** / FL-**SubTB**; there is no valid "FL-TB" | [FL-GFN, ICML 2023](https://arxiv.org/html/2302.01687) states the transformation applies to DB-based objectives, not TB |
| Add **LED-GFN** as a mandatory baseline — and, per §4.5, as the better-matched method | [LED-GFN, ICLR 2024 Oral](https://openreview.net/forum?id=P15CHILQlg) |
| Stage A is hybrid, not "mostly deterministic" — extraction, coreference and entailment are learned | [ANALYSIS] |
| Allow multi-span and cross-turn provenance | [ANALYSIS] — cross-turn claims cannot always be grounded in one current-turn span |
| CompGCN OpenReview ID used here is `BylA_C4tPr` (verified via search); the alternative could not be resolved either way and the point is dropped as immaterial | — |

### 1.7 Corrections adopted in v1.1

A second review round produced six further corrections. Four of them **reduce** the work; two add a small amount. Net effect on executable complexity is negative — see §2.4.

| # | Correction | Effect |
|---|---|---|
| 1 | **The abstention reward is not self-calibrating.** Under reward-proportional sampling, `P(ABSTAIN\|q) = R_abstain / (R_abstain + Σ_X R(X))`, so abstention probability falls as the number of alternative valid proof sets rises — *even when answerability is unchanged*. Replaced by a separate answerability gate (§4.2) | **Removes** the `R_abstain` tuning problem |
| 2 | **The `NIL` head conflates two different decisions** — a genuinely new entity (which must be created) and a non-entity phrase (which must not). Replaced by an open-world action space (§3.2) | **Merges** two heads into one decoder |
| 3 | **The masking/reachability argument was wrong.** Only `STOP` is masked when formal validity fails; `ADD` stays available, so traversing an invalid partial set does not make a valid terminal unreachable. The `H`-monotonicity proof obligation is withdrawn (§4.3) | **Removes** a proof obligation |
| 4 | **Simultaneous budget matching on scored states, model calls, wall-clock and FLOPs is impossible.** One primary budget, plotted across levels; others reported separately (§5.2) | **Makes the search comparison executable** |
| 5 | **"FL-DB is not applicable" was too strong.** Correct statement: no useful, justified scalar extension is currently available, so FL-GFN's local-credit advantage may vanish or mislead; LED-GFN is better matched (§4.5.2) | Precision only |
| 6 | **Add GAFlowNet** as a baseline using the same obligation-derived intermediate signal, and **HGT** as a Stage-B encoder baseline (§5.1, §3.2) | **Adds** two baseline rows |
| 7 | **Self-RAG removed as evidence for answer abstention.** Self-RAG learns whether to *retrieve*, then critiques generated text; its PopQA/PubHealth numbers concern retrieving less, not abstaining from answering | Corrects a mis-citation |
| 8 | **Predeclare one primary metric per stage** with paired significance testing. **[EVIDENCE]** [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/) sets out test selection for NLP; testing many metrics against many baselines without a predeclared decision rule is a multiple-comparison problem | Process discipline |

### 1.8 Corrections adopted in v1.2 — internal consistency

A consistency pass found five places where v1.1 contradicted itself. All were specification defects in this document, not literature disagreements. All are now fixed.

| # | Defect | Fix |
|---|---|---|
| 1 | §3.4 still listed **`ABSTAIN` as a Stage-D action** although §4.2 had withdrawn it from the flow | `ABSTAIN` removed from the action set; §3.4 now points to the decoupled gate |
| 2 | §4.4 allowed a **learned scorer to contribute to `H`**, which would stop `H` being a formal-validity predicate and turn the multiplicative gate in §4.1 into a soft threshold | Explicit prohibition, plus a routing table sending every learned quantity to `U`, the answerability gate, or a Stage-B decoder |
| 3 | **D1's four-way macro-F1 did not score the entity ID.** A model choosing `LINK_EXISTING` but linking to the wrong entity scored as correct — and a wrong merge is the most damaging Stage-B error (risk #1) | D1 now reports three numbers; the **end-to-end** score (action *and* ID) is the Stage-B primary metric |
| 4 | **Stage D's primary metric was valid-terminal rate**, which §4.1 itself argues is degenerate — a method can saturate it with legal-but-weak sets. Plain `E[U]` is also wrong: it rewards mode collapse, so a reward-maximizing baseline could win it while scoring *against* the claimed property | Primary is now **best-of-K valid-set utility at a fixed checker budget** (the §9 Robust Scheduling framing). Valid-terminal rate demoted to an efficiency secondary; exact TV on the enumerable environment is a Gate-2 pass/fail precondition, not a headline |
| 5 | §9 still described the current design as the **"eight-head model"** | Corrected to the four-decoder model (D1–D4). *(The "eight-head constructor" reference in §2.4 is historical — it describes what v1.0 listed — and correctly stays.)* |

---

## 2. Thesis and claim boundaries

### 2.1 Thesis statement

> A learned, provenance-preserving temporal heterogeneous graph constructor, combined with a checker-conditioned evidence-set learner, evaluated against matched neural, RL, GFlowNet and combinatorial-search baselines under fixed reader and compute budgets.

This is deliberately phrased so that it is true whether the proposed learner wins or loses. The scientific contribution is the controlled comparison plus the ceiling decomposition, not a promised victory.

### 2.2 What may be claimed

| Claim | Status |
|---|---|
| **Open-world, provenance-preserving incremental graph construction** — the specific combination of entity *creation* (not just linking), span-level provenance, temporal validity, conflict detection and non-destructive supersession, evaluated component-wise | **Contribution 1** — new architecture, must be ablated |
| A formal-validity-gated terminal reward for evidence-set GFlowNets with a decoupled answerability gate | **Contribution 2** — new formulation, must be verified on an enumerable environment |
| Checker-conditioned potential learning (LED-GFN extended with checker-derived obligation features) | **Contribution 3 [HYPOTHESIS]** — see §4.5.4 |
| Five-ceiling evaluation protocol for memory-graph QA | **Contribution 4 — evaluation protocol**, pending a fuller novelty search |

**[ANALYSIS] Contribution 1 must be scoped carefully.** "Learned incremental graph construction from text" is *not* new — [Learning Dynamic Belief Graphs to Generalize on Text-Based Games (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/1fc30b9d4319760b04fab735fbfed9a9-Abstract.html) trains a transformer sequence-to-sequence model that constructs and updates a *belief* knowledge graph from raw text observations at every step, and reports 24.2% average improvement over text-based baselines on 500+ TextWorld games. The novelty here is the **combination** listed above, not the act of building a graph neurally. State it that way.

### 2.3 What must not be claimed

- That order-invariant set states are novel. [GFlowNet Foundations (JMLR 2023)](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf) covers sets and DAG state spaces; set generation is a standard FL-GFN benchmark.
- That local credit from intermediate signals is novel. That is [FL-GFN (ICML 2023)](https://arxiv.org/html/2302.01687), [LED-GFN (ICLR 2024)](https://openreview.net/forum?id=P15CHILQlg) and [GAFlowNets (ICLR 2023)](https://openreview.net/pdf?id=urF_CBK5XC0).
- That constrained/masked GFlowNets are novel. [Robust Scheduling (ICLR 2023)](https://openreview.net/forum?id=ZBUthI6wK9h) and [Let the Flows Tell (NeurIPS 2023)](https://arxiv.org/abs/2305.17010) both use feasibility-masked MDPs.
- Reward-proportional sampling, unless proved for the actual objective used and checked empirically.
- That the checker verifies truth (§1.5).
- Any predicted margin over any baseline.

### 2.4 Scope control — the tiered experiment set

**[ANALYSIS] This section exists because v1.0 was not executable by one person.** It listed roughly fifteen learning configurations, ten search methods, seven end-to-end systems, an eight-head constructor and a new method. Attempting all of it simultaneously is the main delivery risk in this project — larger than any technical risk in §8.

Everything is therefore tiered. **Tier 1 is the thesis. Tiers 2 and 3 are only run if Tier 1 lands and time remains.**

| Stage | **Tier 1 — must run** | Tier 2 — if time | Tier 3 — only on demand |
|---|---|---|---|
| **B encoders** | MLP baseline (GraphMixer-style) · one relational GNN (**HGT**, or CompGCN) · proposed | the other relational GNN · bi/cross-encoder linker | R-GCN, GNN variants |
| **B comparison** | LLM-prompted linking (what Mem0/Zep do) | — | — |
| **C retrieval** | BM25+dense hybrid · query-conditioned GNN | PPR/HippoRAG-style | — |
| **D learning** | supervised stepwise · canonical set imitation · GRPO · TB · SubTB · **LED-GFN** · proposed | PPO · FL-DB · **GAFlowNet** · DB · FM | OP-GFN · COFlowNet · FlowRL-adapted · AgeMem progressive RL / step-wise GRPO |
| **D search** | greedy · beam · submodular greedy · **PCST** · GFlowNet sampling | **MIP** · diverse beam · MCTS | learned A\* |
| **E systems** | full-context · matched-budget RAG · Mem0 · proposed | Mem0ᵍ · Zep · SEEM | Memory-R1 |
| **Diagnostics** | five ceilings · enumerable-environment distribution check | complementary-evidence probe set · SynCheck | symmetry-collision audit |

**Tier 1 totals: 3 encoders, 7 learning methods, 5 search methods, 4 systems.** That is a full thesis and is achievable. Tier 3 exists so that a reviewer request has a named answer, not so that it gets run by default.

**[EVIDENCE] Predeclare the decision rule before running anything.** One primary metric per stage, fixed in advance, with paired significance testing and uncertainty intervals over multiple seeds. [Dror et al. (ACL 2018)](https://aclanthology.org/P18-1128/) gives the protocol for test selection in NLP. Without a predeclared rule, running this many comparisons guarantees some baseline is beaten by chance.

Suggested primary metrics — fix these and do not change them after seeing results:

| Stage | Primary metric |
|---|---|
| B | **end-to-end mention-resolution score** — the four-way action must be right **and**, when it is `LINK_EXISTING`, the entity ID must also be right (§6.4) |
| C | sufficient-proof recall@k |
| D | **best-of-K valid-set utility at a fixed checker-call budget** (§6.4) |
| E | LongMemEval accuracy, with abstention scored |

**[ANALYSIS] Why Stage D is not "valid-terminal rate."** `H` is formal validity only, so a method can saturate valid-terminal rate by emitting many legal-but-weak proof sets — exactly the degenerate case §4.1 warns about. Nor is plain expected utility `E[U]` right: it rewards mode-seeking, so a reward-maximizing PPO baseline could win it by collapsing onto a single proof, scoring *against* the property being claimed. Best-of-K under a fixed checker budget measures **portfolio quality**, which is the Robust Scheduling framing in §9 and the one setting where a flow method should beat a reward-maximizer.

**Correctness gate, not a headline metric.** Exact TV to the declared target on the enumerable environment (§6.4) is a **pass/fail precondition at Gate 2**. Distribution matching cannot be claimed without it, but it is not the thesis metric and must not be swapped in for one.

---

## 3. Architecture

Five stages. Stages B and D are the learned contributions. Stages A, C and E are supporting infrastructure with strong published precedent.

```
turn ──► A: immutable evidence + contextual extraction
          └─► B: learned incremental graph construction  ◄── Contribution 1
                └─► temporal heterogeneous graph (versioned, non-destructive)

query ──► C: hybrid high-recall candidate retrieval
           └─► D: evidence-set generation (masked policy + checker)  ◄── Contributions 2/3
                 └─► E: frozen SLM verbalization + post-hoc verification
```

### 3.1 Stage A — Immutable evidence and contextual extraction

**Stored permanently and never mutated:** raw turn text, conversation/session ID, turn ID, speaker, timestamp, exact character/token span offsets.

**Extracted with a bounded context window** (previous turns plus a rolling summary), because pronouns, ellipsis and relative dates are not resolvable from an isolated turn. **[EVIDENCE, qualified]** [Mem0 (ECAI 2025)](https://arxiv.org/html/2504.19413v1) uses the previous *m* = 10 messages plus an asynchronously refreshed conversation summary during extraction. This supports contextual extraction as an implemented design; it does not establish that any particular window is optimal.

**Provenance may be multi-span and cross-turn.** A claim assembled across turns records every supporting span, not one.

**Four independent flags per assertion — never collapsed into one "true" bit:**

```
asserted_by_source        : who said it, where, when
entailed_by_span          : the span textually supports the assertion
externally_verified       : corroborated outside the conversation (usually false)
current_under_update_policy : not superseded or retired as of this snapshot
```

**[EVIDENCE]** Grounding is not truth. Citation-faithfulness work separates the two explicitly: [ALCE (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.398/) evaluates citation quality separately from correctness, and [SynCheck (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.527/) monitors faithfulness to context rather than factual accuracy. "I was born on Mars" is entailed by its span and false in the world; the schema must be able to say so.

**This stage is hybrid, not deterministic.** Storage, hashing and offsets are deterministic. Extraction, coreference and entailment are learned and carry error.

### 3.2 Stage B — Learned incremental graph construction — **Contribution 1**

**Schema.** Node types: `Turn`, `SourceSpan`, `Mention`, `Entity`, `Claim`, `Value`, `Event`, `TimeInterval`, `Source`, `Conflict`, `Certificate`. Relation types: `mentioned_in`, `asserted_by`, `about_entity`, `has_value`, `valid_during`, `supported_by`, `same_as`, `contradicts`, `supersedes`, `derived_from`, `retired_by`.

**[EVIDENCE]** Keeping episodic/provenance structure alongside semantic relations, rather than storing bare entity–entity triples, has published precedent: [Zep (preprint)](https://arxiv.org/abs/2501.13956) uses an episode subgraph beneath a semantic entity subgraph with a four-timestamp bi-temporal model and edge *invalidation* rather than deletion; [SEEM (ACL 2026)](https://aclanthology.org/2026.acl-long.277/) anchors Episodic Event Frames with explicit provenance pointers and reports gains on LoCoMo and LongMemEval.

#### Open-world mention resolution — corrected action space

**The v1.0 design had a single `NIL` head meaning "refers to no existing entity." That is wrong for an evolving memory graph [ANALYSIS].** It collapses two decisions with opposite consequences:

- a **real new entity** mentioned for the first time — the system must *create a node*;
- a **non-entity phrase** — the system must create nothing.

In a conversational memory graph the first case is the common one. Classifying it as "NIL" and creating nothing loses the entity permanently.

**Corrected action space (one decoder, four actions):**

```text
LINK_EXISTING(entity_id)   # resolves to a node already in the graph
CREATE_NEW_ENTITY          # a genuinely new entity; instantiate a node
NON_ENTITY                 # not an entity mention at all
DEFER_OR_UNRESOLVED        # insufficient evidence now; revisit on later turns
```

**[EVIDENCE]** [Learn to Not Link (Findings of ACL 2023)](https://aclanthology.org/2023.findings-acl.690/) partitions unlinkable mentions into exactly *Missing Entity* and *Non-Entity Phrase*, and shows both types must appear in training data for NIL accuracy. *(Findings venue.)* The task of linking entities absent from a fixed inventory is separately recognized as hard: [RAED (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1746/) targets "the extremely challenging Emerging Entity Linking task."

**[ANALYSIS]** `DEFER_OR_UNRESOLVED` has no direct published precedent and is a hypothesis. It exists because a mention may be resolvable only after later turns arrive. Ablate it; if it does not pay for itself, drop it and use three actions.

#### Decoder grouping — four decoders, not eight heads

**[ANALYSIS]** v1.0 listed eight heads. Several are coupled — duplicate vs. conflict vs. supersession all depend on the *same claim pair* and the *same time interval*, so predicting them independently discards the constraint that they are mutually exclusive. Grouping them reduces parameters, removes contradictory outputs, and cuts the ablation grid.

| Decoder | Output | Notes |
|---|---|---|
| **D1 — mention resolution** | `LINK_EXISTING` / `CREATE_NEW_ENTITY` / `NON_ENTITY` / `DEFER` | replaces the entity-linking and NIL heads |
| **D2 — claim-pair relation** | `INDEPENDENT` / `DUPLICATE` / `CONFLICT` / `SUPERSEDES` | single mutually-exclusive decision over a claim pair |
| **D3 — relation type** | which schema relation connects two nodes | standard multi-relational link prediction |
| **D4 — temporal** | validity interval for a claim | |
| *(authority)* | source type and evidence strength | **mostly metadata-derived, not learned** — derive from source type where possible and only learn the residual |

The grouping itself is a **[HYPOTHESIS]**: a joint decoder over a shared encoder is testable and is not guaranteed to beat separate models. **Ablate D2 as grouped vs. split** — that is the one ablation that matters here.

**Encoder baselines [EVIDENCE]:**
- **[HGT (WWW 2020)](https://arxiv.org/abs/2003.01332)** — Heterogeneous Graph Transformer, with node- and edge-type dependent attention parameters and **relative temporal encoding** for dynamic heterogeneous graphs, plus HGSampling for scale. Reported 9–21% over prior GNN baselines on the 179M-node Open Academic Graph. **This is the closest published encoder to GRAFT's graph type** (typed nodes, typed edges, time) and should be the primary relational baseline.
- [CompGCN (ICLR 2020)](https://openreview.net/forum?id=BylA_C4tPr) — jointly embeds nodes and relation types for multi-relational graphs.
- [GraphMixer (ICLR 2023)](https://arxiv.org/abs/2302.11636) — MLP link-encoder, mean-pooling node-encoder, MLP classifier; matched or beat RNN/self-attention temporal GNNs on temporal link prediction with faster convergence. **Caveat:** validated on temporal link prediction, *not* on a heterogeneous assertion/conflict/update graph. A strong simple baseline, not a proven ceiling.
- Joint entity-and-relation extraction precedent: [GraphRel (ACL 2019)](https://aclanthology.org/P19-1136/), [TaG (ACL 2023)](https://aclanthology.org/2023.acl-long.607/), [HGERE (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.467/) — the last explicitly targets error propagation in marker-based pipelines.

**Commit discipline: propose → validate → commit, for formally checkable constraints only.**

**[ANALYSIS]** A schema validator can reject an edge type that cannot exist, a duplicate ID, or an interval that violates arithmetic. It **cannot** reject a semantically wrong `same_as` without another learned or human signal. Do not describe this gate as making neural predictions safe. It bounds a specific, enumerable class of damage.

**Non-destructive versioning.** Superseded edges are invalidated, never deleted, so a wrong supersession is recoverable. **[EVIDENCE]** [Zep (preprint)](https://arxiv.org/abs/2501.13956) uses exactly this mechanism.

### 3.3 Stage C — Hybrid high-recall candidate retrieval

Union of: BM25, dense similarity, exact entity match, temporal filtering, 1–2 hop graph expansion, and query-conditioned relational GNN scoring. Purpose is **recall**, not final proof selection.

**[EVIDENCE] Why a GNN, and why not only a GNN:**
- [GNN-RAG (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.856/): GNN retrieval beat LLM-based retrieval by **8.9–15.5 F1 points** on multi-hop and multi-entity questions, using **9× fewer KG tokens**. *(Findings venue.)*
- [GFM-RAG (NeurIPS 2025)](https://arxiv.org/html/2502.01113v2): an 8M-parameter, 6-layer query-conditioned GNN reached Recall@5 of 87.1 / 58.2 / 95.6 on HotpotQA / MuSiQue / 2Wiki at **0.107 s** vs 3.162 s for iterative IRCoT+HippoRAG — **but explicitly names noisy and incomplete KG-index quality as its principal dependency.**
- [HippoRAG (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf): single-step Personalized-PageRank retrieval matched or beat iterative IRCoT at **10–20× cheaper and 6–13× faster**.
- **[EVIDENCE, provisional]** [Beyond Static Retrieval (arXiv 2509.25530)](https://arxiv.org/html/2509.25530v1): across HippoRAG2, RAPTOR, GFM-RAG and GraphRAG, the bottleneck was **ranking, not coverage** (gold recall ≈95% at K=100 but buried below top-10); two iterations was the cost-benefit optimum, 3+ showed diminishing returns, and iteration slightly *hurt* simple comparison questions through over-retrieval. *Preprint — motivates the two-stage design, not load-bearing.*

**Deliverable of this stage:** a candidate pool of roughly 30–100 atoms, plus a measured **required-node / required-edge Recall@k**. If the proof is not in the pool, no downstream learner can recover it.

### 3.4 Stage D — Evidence-set generation — **Contributions 2/3**

**State** = the canonical set of atoms selected so far. Two orders reaching the same atoms are the same state.

**[EVIDENCE]** This is native to the formalism, not a contribution: [GFlowNet Foundations (JMLR 2023)](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf) covers sets and DAGs directly, and set generation is a standard benchmark in [FL-GFN (ICML 2023)](https://arxiv.org/html/2302.01687).

**Actions:** `ADD(atom)`, `STOP`. Masked: repeats, budget violations, formally illegal atoms, and `STOP` when the partial set fails formal validity (§4.3).

**There is no `ABSTAIN` action.** Abstention is decoupled from the flow entirely and handled by a separate gate before Stage D runs, plus a budget-exhaustion fallback after it — see §4.2 for why an in-graph abstention terminal is unsound.

**Policy:** GNN encoder over the graph and candidate pool; separate forward and backward policies. The GNN **encodes and scores actions**; it does not enumerate sets.

**[ANALYSIS]** With ~50 candidates and sets of ≤16, there are on the order of 10¹³ subsets. Enumeration is impossible; incremental construction costs ~16 × 50 = 800 action scores. Any description of the GNN "trying all combinations" is wrong and must not appear in the write-up.

**Equivalent-action check.** If two distinct atoms canonicalize to the same child state, an uncorrected sampler is biased. **[EVIDENCE]** [Symmetry-Aware GFlowNets (ICML 2025)](https://arxiv.org/abs/2506.02685) proves the bias (forward/backward ratio equals the automorphism-group-size ratio, Thm 4.6) and corrects it by scaling terminal rewards (Cor 5.1); the uncorrected sampler produced 5,220 cyclohexane fragments per 5,000 sampled molecules against 1,042 corrected. *(An earlier draft cited an L₁ figure of ≈0.12 vs ≈0.01 here; that number could not be located in the paper and its L₁ figures are plotted on a ×10³ axis, so it has been replaced by the verifiable count.)*. **Action:** instrument the collision rate. If zero, report that and move on. If nonzero, apply the correction. *(The commonly cited "Baking Symmetry into GFlowNets" is a NeurIPS 2023 AI4Science **workshop** paper on ≤7-node synthetic graphs — do not use it as primary support.)*

### 3.5 Stage E — Frozen SLM and output verification

Same frozen SLM, same prompt template, same decoding budget for **every** compared system. Evidence serialized with stable claim IDs and source spans. Answer correctness and citation correctness reported separately.

**[EVIDENCE]** [ALCE (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.398/) established the separation and the motivating failure: on ELI5, even the best models lacked complete citation support **50% of the time**. [VeriCite (SIGIR-AP 2025)](https://arxiv.org/html/2510.11394v1) showed verification, not generation, drives citation quality — removing its NLI check dropped citation F1 from **77.73 → 68.91** while answer correctness barely moved (41.63 → 41.59). *(SIGIR-AP is the Asia-Pacific regional venue, not SIGIR main.)*

**Post-hoc verification is optional and costed.** **[EVIDENCE]** [SynCheck (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.527/) detects unfaithful sentences at **>0.85 AUROC** across six long-form RAG tasks; its FOD decoding improves faithfulness by **>10%**. **It is not free:** it requires a trained monitor and inference-time compute, and FOD alters decoding. If used, it is part of the inference algorithm and must share the same compute budget as competing methods, with latency reported.

**[EVIDENCE, provisional] Reader size is a fixed experimental condition, not a proven design law.** [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1) found careful evidence packing helped a 3B reader (+0.022 F1, p<0.05), was null at 7B (−0.010, p=0.45), and **reversed at 14B (−0.029, p=0.013)** — in one setting, one dataset family, one embedder. This makes reader size a condition to declare and test, not evidence that a small SLM is universally correct.

---

## 4. The formal objective

This section exists because two earlier versions of it were wrong. Everything here is stated so it can be checked.

### 4.1 Terminal reward

Let `X` be a completed evidence set for question `q` against graph snapshot `G`.

- `H(X,q,G) ∈ {0,1}` — **formal-validity** indicator: every formally checkable obligation passes.
- `U(X,q,G) ∈ ℝ` — scalar utility distinguishing formally valid sets by sufficiency, coverage, source strength, temporal correctness, redundancy and size.

**Naming discipline [ANALYSIS].** `H` is called **formal validity**, never "valid proof." A set can satisfy every schema, ID, interval and scope constraint and still be *semantically insufficient* to answer the question. Semantic sufficiency is a separate, learned or annotated quantity that lives inside `U`, not inside `H`. The write-up must use "formally valid" and "sufficient" as distinct words throughout.

```
R(X | q,G)  =  1[ H(X,q,G) = 1 ] · exp( β · U(X,q,G) )
```

**[EVIDENCE]** The requirement that terminal sampling probability be proportional to a **nonnegative** terminal reward is standard: [GFlowNet Foundations (JMLR 2023)](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf), [Trajectory Balance (NeurIPS 2022)](https://arxiv.org/abs/2201.13259).

**[ANALYSIS]** The indicator must be **multiplicative**. Setting `U = 0` on failure yields `exp(0) = 1`, a positive and possibly competitive reward. This was the error in §1.1.

**[ANALYSIS]** If every valid set receives the same `U`, the target is uniform over valid sets. That yields diversity but provides **no pressure toward stronger or smaller proofs.** `U` must genuinely discriminate among valid sets, or the objective is doing less than intended.

**Terminal convention must be stated explicitly in the paper:** with a terminating edge to a sink, the flow on that edge equals `R`; with a terminal-state convention, the terminal state flow equals `R`. Pick one and write it down.

#### `U` must be published as an executable function, not a description

**[ANALYSIS] This is a Gate-0 blocker.** v1.0 described `U` in words. A word description cannot be trained against, reproduced, or held fixed across baselines. Publish the table below, fully filled in, **before any Stage-D training run**.

```text
U = w₁·sufficiency
  + w₂·coverage
  + w₃·source_quality
  + w₄·temporal_correctness
  − w₅·redundancy
  − w₆·size
```

| Term | Range & normalization | Provenance | Frozen across baselines? |
|---|---|---|---|
| `sufficiency` | — | **learned or annotated** — this is the semantically hard one; specify the model and its training data | must be |
| `coverage` | — | fraction of question obligations addressed; deterministic given the obligation parser | must be |
| `source_quality` | — | metadata-derived from source type | must be |
| `temporal_correctness` | — | deterministic interval arithmetic | must be |
| `redundancy` | — | set-coverage overlap; **do not** use the v0 formula `max(0, S(X\e) − S(X) + ε)`, which is near-constant when the proof score is monotone in evidence | must be |
| `size` | — | \|V\| and \|E\| counts | must be |

For every row, state: exact range and normalization; whether it is deterministic, metadata-derived, learned or human-labelled; how any model behind it is trained; and how `w₁…w₆` and `β` are chosen **without touching test data**.

**[ANALYSIS] `U` is frozen across every row of the learning comparison (§5.1).** If different learning methods see different rewards, the comparison measures reward engineering, not learning.

**[ANALYSIS] If every formally valid set receives the same `U`, the target is uniform over valid sets.** That yields diversity but provides **no pressure toward stronger or smaller proofs.** `U` must genuinely discriminate, or the objective is doing less than intended.

### 4.2 Abstention — decoupled, not a reward terminal

**v1.0 proposed `ABSTAIN` as a legal terminal action with a positive calibrated reward `R_abstain(q,G) > 0`. That design is unsound and is withdrawn.**

**[ANALYSIS] Why it fails.** Under the GFlowNet target, with a single abstention terminal:

```text
P(ABSTAIN | q) =  R_abstain / ( R_abstain + Σ_{X ∈ formally valid sets} R(X) )
```

This follows directly from reward-proportional sampling over terminals ([GFlowNet Foundations, JMLR 2023](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf)). The consequence is that **abstention probability depends on how many alternative valid proof sets exist, not on whether the question is answerable.** A question with one valid proof gets a *higher* abstention rate than a question with ten equally good proofs, at identical answerability. That is precisely backwards, and no amount of tuning `R_abstain` fixes it — the denominator moves per question.

It gets worse if each construction prefix produces a *distinct* abstention terminal: abstention mass is then multiplied by the number of prefixes.

**Corrected design — decouple the two decisions:**

1. **Answerability gate.** A separate trained classifier predicts whether a sufficient proof exists in the current graph snapshot for question `q`.
2. **Run the evidence-set GFlowNet only when the gate predicts evidence exists.** Its terminal distribution is then over formally valid proof sets only — no abstention terminal, no `R_abstain` hyperparameter.
3. **Post-hoc fallback.** If construction fails to reach any formally valid terminal within budget, abstain. This recovers the case the gate got wrong.
4. **Evaluate as selective prediction:** selective accuracy, risk–coverage curve, abstention recall, and **false-abstention rate on answerable questions**.

**[ANALYSIS] This is simpler than v1.0, not more complex.** It removes a hyperparameter, removes a calibration sweep, removes the distinct-abstention-terminal ambiguity, and replaces them with one small classifier and a standard selective-prediction evaluation. A joint formulation remains possible, but only with explicitly normalized target masses over the answer and abstention groups — do not attempt that first.

**[EVIDENCE]** Abstention is a first-class evaluated ability, not an edge case: [LongMemEval (ICLR 2025)](https://arxiv.org/abs/2410.10813) evaluates abstention as one of five core long-term-memory abilities, alongside information extraction, multi-session reasoning, temporal reasoning and knowledge updates.

**[CORRECTION]** v1.0 cited Self-RAG here. That citation is withdrawn. [Self-RAG (ICLR 2024)](https://selfrag.github.io/) learns whether to *retrieve* and then critiques generated content; its PopQA/PubHealth numbers concern the cost of retrieving less, not answer abstention. It is not evidence for abstention policy and must not be cited as such.

### 4.3 Masking

`STOP` is masked whenever `H = 0` for the current partial set.

**[ANALYSIS]** This is what keeps invalid sets out of the terminal distribution *without* requiring `log 0` in balance losses.

**[CORRECTION] v1.0 stated the reachability requirement wrongly and demanded a proof that is not needed.**

v1.0 claimed that if a construction order passes through a partial set with `H = 0`, the valid terminal becomes unreachable, and therefore `H` must be proved monotone under set inclusion.

That is false. **Only `STOP` is masked when `H = 0`; `ADD` remains available.** Traversing a formally invalid partial set is fine — the policy simply cannot stop there. The `H`-monotonicity proof obligation is **withdrawn**.

**The actual obligations [ANALYSIS] — narrower, and checkable by construction:**

1. **Constructibility.** Every formally valid terminal must have at least one construction order that respects the **`ADD` masks** (not the `STOP` mask) and the evidence budget. The binding constraints are therefore the `ADD` masks and `max_selected_atoms`, not `H`.
2. **Structural closure must be specified.** State explicitly whether selecting an edge automatically selects its endpoint nodes. If it does, the budget must account for the induced nodes, or long edges become unaffordable and their terminals unconstructible. If it does not, partial sets can contain dangling edges and `H` must reject them.
3. **Dead ends.** With abstention decoupled (§4.2), abstention is no longer an in-graph action, so a state whose `ADD` set is exhausted and whose `STOP` is masked *is* a dead end. Handle it by budget-exhaustion fallback to the post-hoc abstain path in §4.2, and **report the rate at which this happens** — a high rate means the `ADD` masks are too tight.

**Measure, don't assume:** report the **unconstructible-valid-terminal rate** on the enumerable environment (§6.4), where it can be computed exactly.

### 4.4 What the checker may and may not decide

| Formally checkable (deterministic) | Not formally checkable (learned or metadata) |
|---|---|
| Node/edge type legality against schema | Entailment between a span and a claim |
| ID uniqueness, duplicate detection by hash | Factual correctness in the world |
| Temporal interval arithmetic and ordering | Source authority and reliability |
| Deleted / retired evidence not referenced | Semantic correctness of an entity link |
| Scope and access constraints | Whether a proof is *semantically* sufficient |
| Set-size and budget limits | Relevance to the question's intent |

**[ANALYSIS] Nothing in the right-hand column may enter `H`.** `H` is a deterministic predicate over the left-hand column only. If a learned scorer were allowed to contribute to `H`, then `H` would stop being a predicate, `1[H=1]` would stop being a hard gate, and the entire safety argument for the multiplicative reward in §4.1 would degrade into a soft threshold with a tunable cutoff.

Everything learned is routed elsewhere and measured separately:

| Learned quantity | Where it goes | How it is measured |
|---|---|---|
| Entailment between span and claim | `U` (contributes to `sufficiency`) | precision/recall of the entailment model, reported independently |
| Semantic sufficiency of a proof | `U` (`sufficiency` term) | §4.1 table row; annotated per Gate 0 item 4 |
| Source authority and reliability | `U` (`source_quality`), metadata-derived where possible | reported independently |
| Whether the question is answerable at all | the **answerability gate** (§4.2) | risk–coverage, selective accuracy |
| Semantic correctness of an entity link | Stage B decoder D1 | end-to-end mention-resolution score (§6.4) |

The paper must state this boundary explicitly and must never describe the right-hand column as verified.

### 4.5 Local credit: choosing between three options

This is the central technical decision, and the earlier drafts got it wrong.

#### 4.5.1 What FL-GFN actually requires

**[EVIDENCE]** [FL-GFN (ICML 2023)](https://arxiv.org/html/2302.01687), Assumption 4.1 and Proposition 4.2:

```
ℰ : S → ℝ                       (scalar, defined on ALL states)
ℰ(s→s′) = ℰ(s′) − ℰ(s)          (transition energy)
ℰ(X_terminal) = −log R(X)       (up to an irrelevant common constant)
```

Under these conditions the reparametrized flow `F̃(s) = e^{ℰ(s)}F(s)` yields per-transition credit while sampling the **same** target distribution, and supports training on incomplete trajectories. FL-GFN applies to **DB-based** objectives — use **FL-DB** or **FL-SubTB**. There is no valid "FL-TB."

#### 4.5.2 Whether GRAFT's utility satisfies this — worked through

Take the deficit-based potential from the original design:

```
Φ(s) = − Σ_j ω_j d_j(s) − λ_v|V_s| − λ_e|E_s|
```

Set `ℰ = −Φ`. The terminal boundary condition demands `ℰ(X) = −log R(X)`, i.e. `Φ(X) = β·U(X)` at every valid terminal.

**[ANALYSIS] The failure case.** At a valid terminal every *mandatory* deficit is zero by construction, so `Φ(X) = −λ_v|V_X| − λ_e|E_X|`. That ranks valid proofs **by size only**. If `U` also contains sufficiency, source strength or temporal precision terms, then `Φ(X) ≠ β·U(X)` and Proposition 4.2 does not apply. The guarantee is lost silently — nothing crashes, the numbers just stop meaning what they are claimed to mean.

**[ANALYSIS] The fix, if FL-DB is wanted.** Do not define `Φ` and `U` separately. Define `U` **as** a state function evaluable on partial sets, and set `Φ(s) := β·U(s) − (penalty terms)` so that at terminals the identity holds exactly by construction. This is possible only for utility components that are meaningful on partial sets.

**Decision rule [ANALYSIS]:**

| Nature of the utility | Correct tool |
|---|---|
| Additive over atoms | FL-DB works well; this is FL-GFN's designed case |
| Non-additive but genuinely evaluable on partial states | FL-DB applicable, but energy fluctuation along the trajectory may mislead |
| Contains terminal-only global properties (e.g. "the proof is sufficient") | **No useful scalar extension is available → FL-GFN's advantage may vanish or mislead; LED-GFN is better matched** |

**[CORRECTION] The third row is a statement about availability, not about applicability.** v1.0 said FL-DB is "not applicable." That is too strong. FL-GFN *requires* a scalar extension of the energy to intermediate states with terminal consistency; its benefit arises when that extension is **informative**. The accurate statement is:

> For GRAFT's sufficiency term, no useful and justified scalar extension to partial sets is currently available. FL-GFN's local-credit advantage may therefore disappear or actively mislead. LED-GFN is better matched.

**GRAFT's utility is mixed.** Size and source strength are roughly additive. Temporal correctness is per-binding. **Sufficiency is a global, non-additive property of the whole set.** So GRAFT sits in the third row for its most important component — which makes FL-DB a baseline to run and report, not a method to build on.

#### 4.5.3 LED-GFN is the better-matched method

**[EVIDENCE]** [LED-GFN (ICLR 2024, Oral)](https://openreview.net/forum?id=P15CHILQlg) states its motivating problem precisely: FL-GFN's evaluation of intermediate energies "may be too expensive or impossible to evaluate and can provide misleading training signals under large energy fluctuations along the sequence of actions."

Its method: decompose the **terminal** energy into a sum of **learnable** potentials over transitions, use those as local credit, with no intermediate-energy evaluation required. The potential function is regularized to (a) preserve the ground-truth terminal energy and (b) minimize variance along the trajectory. Training with the learned potential **can preserve the optimal policy**. Verified on five problems including unstructured set generation, maximum independent sets, molecular graphs and RNA sequences.

**[ANALYSIS]** That problem statement is a direct description of GRAFT's situation: intermediate proof sufficiency is not evaluable, and the terminal utility is non-additive. LED-GFN is therefore not merely "another baseline" — it is the published method designed for this regime. This is a factual match of problem settings, **not** a prediction that it will win.

#### 4.5.4 The proposed contribution, stated as a hypothesis

**Contribution 3 [HYPOTHESIS]: checker-conditioned potential learning.**

LED-GFN learns transition potentials from terminal energy alone. GRAFT has something LED-GFN's settings do not: a **deterministic checker that emits partial obligation state at every intermediate step for negligible cost** — which mandatory obligations are currently satisfied, which are outstanding.

The hypothesis: conditioning LED-GFN's learned potential on the checker's obligation vector `d(s)` — as policy input features, as an auxiliary supervised target, and/or as a variance-reduction regularizer on the learned potential — improves credit assignment over vanilla LED-GFN.

Note carefully what this does and does not assert. `d(s)` is a **feature and auxiliary target**, not an energy. No theorem is claimed. The claim is empirical, the baseline is LED-GFN, and the verification environment is enumerable (§6.4).

**Hard constraint on the implementation [ANALYSIS]:** LED-GFN's guarantee rests on regularizing the learned potential to **preserve the ground-truth terminal energy** (and to minimize trajectory variance). If checker-derived features are allowed to modify the learned potential, **that terminal-consistency regularizer must be retained unchanged.** Dropping or weakening it to let the checker features have more influence silently discards the property that makes LED-GFN correct. This must be verified on the enumerable environment, not assumed.

**Required control:** Contribution 3 must beat **capacity-matched** vanilla LED-GFN and **capacity-matched GAFlowNet** (§5.1). Adding checker features also adds parameters; a win that disappears under parameter matching is not a win.

This is a narrower claim than earlier drafts made. It is also the first version that is actually checkable.

#### 4.5.5 Safe research order

1. Ordinary conditional **TB / SubTB** with the exact terminal reward of §4.1.
2. Add supervised **deficit prediction** as an auxiliary head (dense signal, no theoretical claim).
3. **FL-DB / FL-SubTB** — only after a terminal-consistent scalar energy is defined per §4.5.2, or reported as inapplicable with the reason.
4. **LED-GFN** as the direct learned-decomposition competitor.
5. **Checker-conditioned LED-GFN** (Contribution 3) — claimed as a contribution only after terminal-distribution correctness is verified on an enumerable environment.

---

## 5. Baselines

Three separate comparisons answering three different questions. Never merged into one table.

### 5.1 Comparison A — matched learning algorithms

**Matched across all rows where the algorithm permits:** graph encoder and size, candidate pool, action space, terminal utility and checker, training examples, approximate reward/checker call budget, frozen SLM, random seeds.

| Method | Include | Basis / caveat |
|---|:--:|---|
| Supervised next-action / subgraph training | ✔ | **[EVIDENCE]** [Graph-S3 (ACL 2026)](https://aclanthology.org/2026.acl-long.1169/) makes stepwise supervised graph retrieval a strong baseline: +8.1% acc, +9.7% F1 over 7 baselines in its own setting |
| Canonical set imitation | ✔ | Tests whether distribution training adds anything over imitating one gold set |
| PPO | ✔ | Reward-maximizing baseline. Use the **same graph-action policy**, not an LLM agent from another paper |
| GRPO | ✔ | Same matched-policy requirement. *Search-R1 is arXiv-only and trains an LLM search agent — implementation inspiration, not venue-backed evidence for this task* |
| Progressive RL | ✔ | **[EVIDENCE]** [AgeMem (ACL 2026)](https://aclanthology.org/2026.acl-long.981/) — three-stage progressive RL for memory operations. Implement per the paper; its stages change more than a loss function, so document the adaptation |
| Step-wise GRPO | ✔ | **[EVIDENCE]** AgeMem, designed for "sparse and discontinuous rewards induced by memory operations" |
| Flow Matching (FM) | ✔ | **[EVIDENCE]** [Foundations (JMLR 2023)](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf). A **distinct objective** from DB — report as its own row, not merged |
| Detailed Balance (DB) | ✔ | **[EVIDENCE]** [Foundations (JMLR 2023)](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf). Also the base objective FL-DB modifies |
| **GAFlowNet** | ✔ | **[EVIDENCE]** [ICLR 2023 Spotlight](https://openreview.net/pdf?id=urF_CBK5XC0) — incorporates **intermediate intrinsic rewards** (edge- and state-based augmented flows) for sparse-reward exploration, with an asymptotically unbiased treatment. **Run it with the same obligation-derived intermediate signal Contribution 3 uses.** It is the direct published alternative for "use the obligation signal as an intermediate reward," and Contribution 3 must beat it |
| Trajectory Balance | ✔ | **[EVIDENCE]** [NeurIPS 2022](https://arxiv.org/abs/2201.13259) |
| SubTB(λ) | ✔ | **[EVIDENCE]** [ICML 2023](https://arxiv.org/abs/2209.12782) — frames DB/TB as opposite ends of a bias-variance tradeoff |
| **FL-DB / FL-SubTB** | **Mandatory** | **[EVIDENCE]** [FL-GFN, ICML 2023](https://arxiv.org/html/2302.01687). Beating TB/SubTB alone proves nothing — FL-GFN already does, on set generation. Do **not** construct "FL-TB" |
| **LED-GFN** | **Mandatory** | **[EVIDENCE]** [ICLR 2024 Oral](https://openreview.net/forum?id=P15CHILQlg). The correct competitor for non-additive terminal utility (§4.5.3) |
| Prioritized replay / guided TB | ✔ ablation | **[EVIDENCE]** [Shen et al., ICML 2023](https://arxiv.org/abs/2305.07170) — keep objective effects separable from replay effects |
| OP-GFN | Conditional | **[EVIDENCE]** [ICLR 2024](https://arxiv.org/abs/2310.00386). Relevant if only an ordering over proofs is trusted rather than a calibrated scalar. Different target — label separately |
| COFlowNet | Conditional | **[EVIDENCE]** [ICLR 2025](https://iclr.cc/virtual/2025/poster/28047). Include **only** if training uses logged trajectories without online checker evaluation; otherwise not a matched setting |
| FlowRL-style objective | Secondary | **[EVIDENCE]** [ICLR 2026](https://arxiv.org/abs/2509.15207) reports +10.0% over GRPO and +5.1% over PPO **on math/code with LLM token policies**. Motivates a distribution-matching baseline; **cannot predict this project's outcome.** Document the graph-action adaptation |
| Checker-conditioned LED-GFN | Proposed | Contribution 3, per §4.5.4 |

### 5.2 Comparison B — search algorithms under a declared budget

Greedy · Beam · Diverse beam ([Vijayakumar et al., AAAI 2018]) · MCTS ([MCTS-RAG](https://aclanthology.org/2025.findings-emnlp.672.pdf), [AirRAG](https://aclanthology.org/2025.findings-emnlp.1030.pdf), both Findings of EMNLP 2025) · Submodular greedy · Learned A\* (Neural A\*, ICML 2021) · **PCST** · **MIP** · Raw GFlowNet sampling · GFlowNet sampling + checker selection

**[EVIDENCE] The two strongest additions:**
- **PCST** — [G-Retriever (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html) formulates textual-graph retrieval as Prize-Collecting Steiner Tree optimization. It selects a **connected** subgraph — structurally the closest classical algorithm to a proof subgraph.
- **MIP** — [ARM (ACL 2025)](https://aclanthology.org/2025.acl-long.1463/) retrieves all-at-once by jointly optimizing object selection for relevance and compatibility, rather than iterating.

**[ANALYSIS] Fairness cannot mean "identical trained model."** Learned A\* and MCTS need a value/heuristic; PCST, MIP and submodular greedy optimize their own objectives. Fairness therefore means:

- same input candidate pool and graph snapshot;
- same terminal checker and utility where compatible;
- **one declared primary budget** — see below;
- **both** an algorithm-native configuration and, where possible, a matched-score configuration;
- **equal hyperparameter-tuning budget for every method**, stated in the paper;
- raw GFlowNet samples reported separately from post-filtered delivered answers.

**[CORRECTION] v1.0 required matching scored states, model calls, wall-clock *and* FLOPs simultaneously. That is impossible** — the four move independently across a differentiable planner, a tree search, an ILP solver and a sampler. Trying to satisfy all four is the fastest way to make this comparison unrunnable.

**Corrected protocol [ANALYSIS]:**

1. **Choose one primary budget: checker/model evaluations.** It is the quantity all ten methods actually consume, and it is the one GRAFT's design is trying to economize.
2. **Plot performance against several budget levels** rather than reporting a single point. A method that wins at one budget and loses at another is a finding, not a failure.
3. **Report wall-clock and FLOPs separately**, as observed costs, not as matched conditions.
4. **Do not compare against "unlimited MCTS."** There is no defined fair stopping condition.

**[ANALYSIS] Neural A\* is an *adapted* baseline, not a validated proof-search method.** [Neural A\* (ICML 2021)](https://proceedings.mlr.press/v139/yonetani21a.html) is a differentiable planner trained from expert paths on spatial path-planning problems. Porting it here requires a new state definition, cost function, heuristic target and stopping rule — all of which are your design choices, not the paper's. Label it as adapted, describe the adaptation, and expect a reviewer to discount it accordingly. It sits in **Tier 3** (§2.4) for this reason.

**On submodular selection [ANALYSIS + EVIDENCE, provisional]:** A submodular objective has diminishing marginal returns and cannot directly express *superadditive* interaction — the case where two atoms are individually useless and jointly sufficient. But coverage features can still favour items covering different needs, and [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1) does exactly that, reporting **0.451 F1** for a training-free cost-scaled greedy submodular packer vs 0.429 (tuned heuristic) and 0.410 (MMR) on multi-hop HotpotQA at a 160-token budget. **[CORRECTION, 12 Aug 2026]** That budget is *the only one of four* at which the packer's edge over the tuned heuristic is significant — the others are −0.018 (p=0.08), −0.001 (p=0.90) and +0.013 (p=0.14), and at a 7B reader the gap is −0.010 (p=0.45). The paper's own conclusion is **parity**: it "reaches parity, winning outright only where evidence density is the binding constraint". Naming the budget, as this paragraph already did, was necessary but not sufficient; the reader also needs to know it is the winning one. The narrow hypothesis below is unaffected, and so is Gate 3 — which asks whether a *training-free* method matches a *learned* one, a comparison that table does not make. The correct hypothesis is therefore narrow: *some* proof synergies may escape a *chosen* submodular objective. **Test it on purpose-built complementary-evidence cases — and also on ordinary LongMemEval/LoCoMo cases, so the evaluation is not designed only to favour the proposed method.** That preprint is provisional evidence and a baseline, never a load-bearing conclusion.

### 5.3 Comparison C — end-to-end systems, re-run not quoted

Full-context · matched-budget RAG · [Mem0 and Mem0ᵍ (ECAI 2025)](https://arxiv.org/html/2504.19413v1) · [Zep (preprint)](https://arxiv.org/abs/2501.13956) · [Memory-R1 (ACL 2026)](https://aclanthology.org/2026.acl-long.583/) · [SEEM (ACL 2026)](https://aclanthology.org/2026.acl-long.277/) · this system

**[ANALYSIS] Two rules:**

1. **Re-run, do not quote.** Published scores from different papers use different readers, prompts, histories, token budgets and judges. They are not a controlled comparison. Where a system cannot be re-run, report its published number in a clearly separated table labelled as uncontrolled.
2. **This comparison answers a different question.** Memory-R1 and AgeMem *train LLM agents*; this project freezes the SLM. System-level numbers therefore cannot isolate the value of the GNN or the search learner. Comparison A does that; Comparison C establishes whether the whole system is competitive.

**[EVIDENCE] Full-context is non-negotiable.** In Mem0's own table, full-context scored **72.90** LLM-judge overall on LoCoMo — above Mem0 (66.88), Mem0ᵍ (68.44) and Zep (65.99). And Mem0's graph variant lost to its flat variant on single-hop (65.71 vs 67.13) and multi-hop (47.19 vs 51.15) while winning on temporal (58.13 vs 55.51), at ~2× tokens and ~3.2× search latency. **A graph does not automatically improve every memory task.** Omitting full-context will read as evasion.

---

## 6. Evaluation

### 6.1 Benchmarks

| Benchmark | Role | Basis |
|---|---|---|
| **[LongMemEval (ICLR 2025)](https://arxiv.org/abs/2410.10813)** | Primary end-to-end | 500 questions over five abilities — extraction, multi-session reasoning, temporal reasoning, knowledge updates, **abstention**. Directly matched to this system's design targets |
| **LoCoMo ([ACL 2024](https://aclanthology.org/2024.acl-long.747/))** | Secondary end-to-end | Community standard; enables comparison with Mem0's published table (uncontrolled) |
| **Enumerable synthetic graph** | Distribution correctness | The only setting where the learned terminal distribution can be checked exactly |
| **Complementary-evidence probe set** | Targeted diagnostic | Purpose-built cases where a proof requires a specific atom *pair* (§5.2) |

**[EVIDENCE + ANALYSIS] LongMemEval supplies evidence statements and sessions. It does not supply a gold heterogeneous graph, gold entity links, gold relation labels, or minimal proof subgraphs.** Stage B cannot be evaluated on it without additional annotation. Budget for a manually verified annotation subset and publish the annotation policy — this is a real cost, not a footnote.

### 6.2 Known benchmark caveats to state in the paper

- LoCoMo scores whether the **answer** was right, not whether **retrieval** was right — an integration test where component metrics are needed. Report component metrics separately.
- Report F1 and BLEU-1 alongside LLM-judge scores, as Mem0 does, rather than judge scores alone.
- LongMemEval-S at ~115k tokens per question partly fits inside current context windows, so full-context must be reported on it explicitly.

### 6.3 The five ceilings

A single "graph ceiling" conflates five distinct failure sources. Measure each.

| # | Ceiling | Question |
|---|---|---|
| 1 | **Extraction** | Are all gold evidence statements represented as correctly grounded assertions? |
| 2 | **Graph** | Does the constructed graph contain a sufficient proof, including required links and temporal/update relations? |
| 3 | **Candidate** | Does Stage C retrieve every atom needed by at least one sufficient proof? |
| 4 | **Packing** | Does a sufficient proof survive the evidence/token budget and serialization? |
| 5 | **Reader** | What does the frozen SLM achieve when handed a gold proof? |

**[ANALYSIS]** Without this decomposition, a bad end-to-end number is uninterpretable — it could be extraction, linking, retrieval, packing or the reader. With it, every result points at a specific stage. This is cheap to compute and is a genuine methodological contribution.

**[EVIDENCE, provisional] Precedent for ceiling-style diagnostics:** [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1) introduced "answer-in-context" — whether the gold answer survives into the packed reader context — and found it predicts answer F1 far better than retrieval recall (ΔR² = +0.17 over recall; F1 0.61 vs 0.20 conditional on it, **including among questions where retrieval was perfect**). Ceiling 4 is the direct analogue.

### 6.4 Metric groups

**Graph construction (Stage B).** Extraction and span-grounding P/R/F1 · D2 four-way macro-F1 · D3 relation macro-F1 · D4 temporal interval accuracy or overlap · calibration (Brier / ECE) · formal graph-constraint violation rate · **downstream corruption rate after sequential updates**. Chronological and user-level splits only — a later update must never leak into an earlier graph.

**D1 needs three numbers, not one [ANALYSIS].** A four-way action score alone is not enough: a model that correctly chooses `LINK_EXISTING` but attaches to the *wrong entity* would score as correct, and a wrong merge is the single most damaging Stage-B error — it corrupts every proof built on it afterwards (risk #1, §8). Report:

1. **Four-way action macro-F1**, with `CREATE_NEW_ENTITY` and `NON_ENTITY` broken out separately (the two decisions v1.0 conflated; they have opposite failure costs).
2. **Linking accuracy@1 conditional on `LINK_EXISTING`** — given the model chose to link, did it link correctly?
3. **End-to-end mention-resolution score** — a prediction counts as correct only if the action is right *and*, for `LINK_EXISTING`, the entity ID is right. **This is the Stage-B primary metric (§2.4).**

**Selective prediction (answerability gate).** Selective accuracy · risk–coverage curve · abstention recall · **false-abstention rate on answerable questions** · post-hoc fallback trigger rate (§4.2).

**Retrieval (Stage C).** Required-node Recall@k · required-edge Recall@k · sufficient-proof recall · candidate pool size · latency.

**Set generation (Stage D).** **Primary: best-of-K valid-set utility at a fixed checker-call budget** — sample K sets, keep those passing `H`, report the utility of the best one. Secondaries: expected terminal utility `E[U]` of sampled sets · **valid-terminal rate, now reported as an *efficiency* measure** (legal sets produced per checker call, not as evidence of quality) · evidence set size · number and diversity of **distinct valid** proof sets · training sample efficiency · unconstructible-valid-terminal rate (§4.3) · equivalent-action collision rate (§3.4).

Fix K and the checker budget before running, and use the same values for every method in the §5.1 and §5.2 comparisons.

**Distribution correctness.** On the enumerable environment, report **exact TV / KL / JS** to the declared target — do not substitute a proxy when an exact result is available. Additionally report the FCS proxy and GNN representation-collision diagnostics.

**[EVIDENCE]** [When do GFlowNets learn the right distribution? (ICLR 2025, Spotlight)](https://openreview.net/forum?id=9GsgCUJtic) shows the impact of an imbalanced edge scales with the flow through it, proposes FCS as a tractable correctness proxy, and — critically here — proves that **GNN representation limits constrain which graph distributions a GFlowNet can approximate.** Two proof states the encoder cannot distinguish cannot receive the different flows the target requires. Test a more expressive encoder or path/positional features as an ablation rather than assuming message passing suffices.

**[EVIDENCE]** [Generalization and Distributed Learning of GFlowNets (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/000eba875068854d5ff003b1fa534cd6-Abstract-Conference.html) gives the first data-dependent generalization bounds for GFlowNets and shows generalization degrades as state-space size grows — supporting bounded active graphs as a principled choice, not just an efficiency trick.

**End-to-end.** LongMemEval accuracy broken out by the five abilities · LoCoMo score · **abstention P/R/F1 and false-abstention rate on answerable questions** · ALCE-style citation precision/recall · latency (p50, p95) · context tokens to the SLM · model and checker call counts.

**Reporting discipline [ANALYSIS].** Same frozen reader and judge across all matched comparisons. Multiple random seeds with uncertainty intervals for every trained method. State wins **per metric and per budget** — never "beats everything."

---

## 7. Gated implementation order

Each gate can stop or redirect the project cheaply. That is the point.

### Gate 0 — The data contract *(nothing is trained before this is signed off)*

**[ANALYSIS] This gate is new in v1.1 and it is the real blocker.** The plan's biggest remaining weakness is not the architecture — it is that the plan does not yet say **where every supervision and reward signal comes from.** Without this, Stage-B training, Stage-D reward and the five ceilings are all unreproducible.

**What LongMemEval actually gives you.** Its released schema provides questions, answers, sessions, evidence-session IDs and turn-level `has_answer` indicators ([dataset](https://github.com/xiaowu0162/longmemeval); [ICLR 2025 paper](https://arxiv.org/abs/2410.10813)). **It does not provide** entity links, relation labels, conflict labels, supersession links, temporal intervals, or minimal proof subgraphs. Every one of those must be annotated.

**Deliverables of Gate 0 — all written down before any training run:**

| # | Item |
|---|---|
| 1 | Which labels supervise **each** of D1–D4 (§3.2), and where they come from |
| 2 | Which labels train Stage-C relevance scoring |
| 3 | How canonical proof sets are obtained for set imitation and Graph-S3-style stepwise supervision |
| 4 | How the `sufficiency` and `coverage` terms in `U` are labelled (§4.1) — the hardest item on this list |
| 5 | **User-level and chronological** train/dev/test splits, so a later update cannot leak into an earlier graph |
| 6 | Negative-example construction and class balance, especially for `CREATE_NEW_ENTITY` vs `NON_ENTITY` |
| 7 | Annotation guidelines, inter-annotator agreement measurement, and adjudication procedure |
| 8 | **How many annotations are actually feasible** given available time and people |
| 9 | Dataset selection: primary benchmark, component-label source, and the enumerable synthetic environment spec |
| 10 | Predeclared primary metric per stage and the significance protocol (§2.4) |

**Stop condition:** if item 8 shows the required annotation volume is not achievable, reduce scope *here* — by narrowing the schema or the question types — not later by quietly weakening the evaluation.

### Gate 1 — Is the graph worth learning?

1. Annotate the auditable dev/eval subset per the Gate-0 contract.
2. Measure **extraction** and **graph** ceilings (1 and 2).
3. Train and compare (Tier 1, §2.4): string/embedding similarity · GraphMixer-style MLP · **HGT** (or CompGCN) · the proposed four-decoder model · LLM-prompted linking (what Mem0 and Zep actually do).
4. Ablate: D2 grouped vs. split; `DEFER_OR_UNRESOLVED` on vs. off.

**Stop or redesign if** the learned constructor does not improve component accuracy, **or** if oracle use of the graph cannot support the target questions.

### Gate 2 — Is the terminal objective correct?

1. Publish the exact formal-validity predicate `H`, the **fully executable** utility `U` (§4.1 table, every row filled), the answerability-gate specification (§4.2), the structural-closure rule (§4.3), and the terminal convention.
2. Build the enumerable environment.
3. Compare (Tier 1) TB · SubTB · LED-GFN · proposed, and (Tier 2) FM · DB · FL-DB · FL-SubTB · GAFlowNet, on **exact** TV/KL/JS distribution error plus FCS.
4. Verify the **constructibility** and dead-end obligations (§4.3) and the equivalent-action collision rate (§3.4).
5. Train and calibrate the answerability gate; report the risk–coverage curve.

**Do not promote the deficit idea into an FL energy** until terminal consistency is demonstrated per §4.5.2 — or until it is reported as unavailable, with the reason, and LED-GFN is used instead.

**Capacity matching is mandatory here.** Contribution 3 adds parameters over LED-GFN; compare at matched capacity or the result is uninterpretable.

### Gate 3 — Are the search baselines strong?

1. Run greedy · beam · diverse beam · MCTS · submodular · learned A\* · **PCST** · **MIP** under declared budgets.
2. Report **candidate**, **packing** and **reader** ceilings (3, 4, 5).
3. Evaluate on the complementary-evidence probe set **and** on ordinary LongMemEval/LoCoMo cases.

**[ANALYSIS] This is the highest-risk gate.** PCST, MIP and submodular greedy are training-free. If any of them matches the learned set constructor at equal budget, the learning contribution for Stage D does not hold, and the project should consolidate around Contribution 1. **[CORRECTION, 12 Aug 2026]** That test must be run **where the scorer is a proxy**, not where it is the evaluation metric itself. On the enumerable environment, fix F13 makes the scorer exactly `U`, and it is then measured that plain greedy attains the global optimum on 30/30 instances while a flawless reward-proportional sampler falls 0.038 short at K = 8 — so on the lattice this criterion narrows Stage D's claim automatically, by arithmetic, whatever the sampler learned. Gate 3's *decision* therefore belongs to the real-data stage, whose distilled utility head is the cheap proxy Robust Scheduling's argument requires; the synthetic stage is a diagnostic that establishes the training-free frontier.

### Gate 4 — End-to-end with a frozen reader

1. Freeze SLM, prompt and decoding settings **before** the final comparison.
2. Evaluate answer, abstention, citation, cost and latency separately.
3. Treat SynCheck/FOD as an explicitly costed inference variant sharing the same budget.
4. Re-run system baselines rather than quoting published numbers (§5.3).

---

## 8. Risk register

| # | Risk | Basis |
|---|---|---|
| 1 | **Error propagation.** A false merge or wrong supersession corrupts every later proof. Provenance and non-destructive versioning bound the damage; they do not fix prediction error | [ANALYSIS]; [HGERE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.467/) on pipeline error propagation |
| 2 | **Checker incompleteness.** Formal checks cannot establish entailment, truth or authority | [ANALYSIS] (§4.4) |
| 3 | **Candidate bottleneck.** No set learner recovers atoms Stage C excluded | [EVIDENCE, provisional] [Beyond Static Retrieval](https://arxiv.org/html/2509.25530v1): ranking, not coverage, is the bottleneck |
| 4 | **Non-additive terminal utility.** Exactly where the simple FL-GFN argument is weakest | [EVIDENCE] [LED-GFN, ICLR 2024](https://openreview.net/forum?id=P15CHILQlg) names unavailable/misleading intermediate energy as the practical problem |
| 5 | **GNN representation limits** cap the representable distribution | [EVIDENCE] [When do GFlowNets learn the right distribution?, ICLR 2025](https://openreview.net/forum?id=9GsgCUJtic) |
| 6 | **Unfair baseline transfer.** PPO/GRPO memory agents, FlowRL token policies and graph-action policies are different systems | [ANALYSIS] |
| 7 | **Annotation cost.** LongMemEval is not a gold graph-construction corpus | [ANALYSIS] (§6.1) |
| 8 | **Graph memory may lose to flat memory and to full-context** on common question types | [EVIDENCE, qualified] [Mem0, ECAI 2025](https://arxiv.org/html/2504.19413v1) |
| 9 | **Reader-size scope.** The packing benefit is conditional and reversed at 14B in one study | [EVIDENCE, provisional] [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1) |
| 10 | **Sparse-reward instability.** Hard-validity gating concentrates reward on few terminals; near-zero flows destabilize log-space losses | [EVIDENCE] [SubTB, ICML 2023](https://arxiv.org/abs/2209.12782) and [GAFlowNets, ICLR 2023](https://openreview.net/pdf?id=urF_CBK5XC0) are both motivated by sparse-reward difficulty |
| 11 | **Over-abstention** unmeasured. A system that abstains constantly has perfect precision and no value | [ANALYSIS]; mitigated by the risk–coverage evaluation in §4.2 |
| 12 | **No guaranteed numerical win.** No cited paper supports a confidence prediction | [ANALYSIS] |
| 13 | **Scope overrun — the largest delivery risk.** v1.0 specified more experiments than one person can run. Mitigated by the Tier 1/2/3 split (§2.4); the mitigation only works if Tier 2 and 3 are genuinely deferred rather than attempted early | [ANALYSIS] |
| 14 | **Multiple comparisons.** Many metrics × many baselines without a predeclared decision rule will produce spurious "wins" | [EVIDENCE] [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/) |
| 15 | **Annotation infeasibility.** Gate 0 item 8 may show the required labels cannot be produced in the available time | [ANALYSIS] (§7 Gate 0) |

---

## 9. What is genuinely promising, and what is not

**Most likely to produce a defensible result:** learned incremental graph construction (Contribution 1). It matches the supervisor's NN/GNN focus, is evaluable without the SLM, and decomposes into standard tasks with standard metrics. The literature establishes that open-world entity resolution, temporal/update prediction and heterogeneous link prediction are real subproblems. **It does not establish that the proposed four-decoder model (D1–D4, §3.2) wins.** That is the experiment.

**Legitimate higher-risk contribution:** the evidence-set learner. **[EVIDENCE]** GFlowNets are designed to sample composite objects in proportion to reward ([Foundations, JMLR 2023](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf)), and [Robust Scheduling (ICLR 2023)](https://openreview.net/forum?id=ZBUthI6wK9h) shows diverse candidates sampled under a cheap proxy can beat proxy optimization when the real evaluator is expensive.

**[ANALYSIS]** But this is only convincing if **all three** of the following are demonstrated:

1. questions genuinely admit multiple materially different valid proof sets;
2. diversity improves robustness or downstream success **under a fixed evaluation budget**;
3. the learned sampler approximately matches its declared terminal target on the controlled environment.

If only one proof is ever needed, and the delivered output is always the maximum-scoring set, then beam, PCST, MIP or MCTS are reasonable preferences and a reviewer will say so. **The GFlowNet benefit must be measured, not assumed.**

**Fallback.** If Gate 2 or Gate 3 fails, consolidate around Contribution 1 plus the five-ceiling analysis, with supervised or PCST/MIP set selection. That is still a complete, publishable systems-and-analysis paper. Structure the work so this fallback survives.

---

## 10. References

### Primary venues

**GFlowNets**
- [GFlowNet Foundations — JMLR 2023](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf)
- [Trajectory Balance — NeurIPS 2022](https://arxiv.org/abs/2201.13259)
- [Learning GFlowNets from Partial Episodes (SubTB) — ICML 2023](https://arxiv.org/abs/2209.12782)
- [Better Training of GFlowNets with Local Credit and Incomplete Trajectories (FL-GFN) — ICML 2023](https://arxiv.org/html/2302.01687)
- [Learning Energy Decompositions for Partial Inference in GFlowNets (LED-GFN) — ICLR 2024 Oral](https://openreview.net/forum?id=P15CHILQlg)
- [Generative Augmented Flow Networks — ICLR 2023 Spotlight](https://openreview.net/pdf?id=urF_CBK5XC0)
- [Towards Understanding and Improving GFlowNet Training — ICML 2023](https://arxiv.org/abs/2305.07170)
- [When do GFlowNets learn the right distribution? — ICLR 2025 Spotlight](https://openreview.net/forum?id=9GsgCUJtic)
- [Generalization and Distributed Learning of GFlowNets — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/000eba875068854d5ff003b1fa534cd6-Abstract-Conference.html)
- [Symmetry-Aware GFlowNets — ICML 2025](https://arxiv.org/abs/2506.02685)
- [Order-Preserving GFlowNets — ICLR 2024](https://arxiv.org/abs/2310.00386)
- [COFlowNet — ICLR 2025](https://iclr.cc/virtual/2025/poster/28047)
- [Robust Scheduling with GFlowNets — ICLR 2023](https://openreview.net/forum?id=ZBUthI6wK9h)
- [Let the Flows Tell — NeurIPS 2023 Spotlight](https://arxiv.org/abs/2305.17010)
- [FlowRL — ICLR 2026](https://arxiv.org/abs/2509.15207)

**Graphs, retrieval, memory**
- [HGT: Heterogeneous Graph Transformer — WWW 2020](https://arxiv.org/abs/2003.01332)
- [CompGCN — ICLR 2020](https://openreview.net/forum?id=BylA_C4tPr)
- [GraphMixer — ICLR 2023](https://arxiv.org/abs/2302.11636)
- [Learning Dynamic Belief Graphs to Generalize on Text-Based Games (GATA) — NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/1fc30b9d4319760b04fab735fbfed9a9-Abstract.html)
- [RAED: Retrieval-Augmented Entity Description Generation for Emerging Entity Linking — EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1746/)
- [Neural A\*: Path Planning using Neural A\* Search — ICML 2021](https://proceedings.mlr.press/v139/yonetani21a.html)
- [G-Retriever (PCST) — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)
- [HippoRAG — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf)
- [GFM-RAG — NeurIPS 2025](https://arxiv.org/html/2502.01113v2)
- [SubgraphRAG — ICLR 2025](https://arxiv.org/abs/2410.20724)
- [ARM (MIP) — ACL 2025](https://aclanthology.org/2025.acl-long.1463/)
- [Graph-S3 — ACL 2026](https://aclanthology.org/2026.acl-long.1169/)
- [AgeMem — ACL 2026](https://aclanthology.org/2026.acl-long.981/)
- [SEEM — ACL 2026](https://aclanthology.org/2026.acl-long.277/)
- [Memory-R1 — ACL 2026](https://aclanthology.org/2026.acl-long.583/)
- [LoCoMo — ACL 2024](https://aclanthology.org/2024.acl-long.747/)
- [LongMemEval — ICLR 2025](https://arxiv.org/abs/2410.10813) · [released dataset schema](https://github.com/xiaowu0162/longmemeval)
- [The Hitchhiker's Guide to Testing Statistical Significance in NLP — ACL 2018](https://aclanthology.org/P18-1128/)
- [Self-RAG — ICLR 2024](https://selfrag.github.io/) — *cited only for adaptive retrieval. **Not** evidence for answer abstention (§4.2).*
- [GraphRel — ACL 2019](https://aclanthology.org/P19-1136/) · [TaG — ACL 2023](https://aclanthology.org/2023.acl-long.607/) · [HGERE — EMNLP 2023](https://aclanthology.org/2023.emnlp-main.467/)
- [ALCE — EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/)
- [SynCheck — EMNLP 2024](https://aclanthology.org/2024.emnlp-main.527/)
- [Lost in the Middle — TACL 2024](https://aclanthology.org/2024.tacl-1.9/)

### Qualified (Findings / regional / non-A\* venues)

- [Learn to Not Link — Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.690/)
- [GNN-RAG — Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.856/)
- [MCTS-RAG](https://aclanthology.org/2025.findings-emnlp.672.pdf) · [AirRAG](https://aclanthology.org/2025.findings-emnlp.1030.pdf) — Findings of EMNLP 2025
- [Mem0 — ECAI 2025](https://arxiv.org/html/2504.19413v1)
- [VeriCite — SIGIR-AP 2025](https://arxiv.org/html/2510.11394v1)

### Provisional (arXiv / workshop — motivating only)

- [Zep — arXiv 2501.13956](https://arxiv.org/abs/2501.13956) — vendor-authored preprint
- [What Survives Into Context — arXiv 2607.00725](https://arxiv.org/abs/2607.00725) — single-author preprint; v2 retitled *Recall Is Not Enough*
- [Beyond Static Retrieval — arXiv 2509.25530](https://arxiv.org/html/2509.25530v1)
- [Search-R1 — arXiv 2503.09516](https://arxiv.org/pdf/2503.09516)
- [Baking Symmetry into GFlowNets](https://openreview.net/forum?id=CZGHAeeBk3) — NeurIPS 2023 AI4Science workshop; superseded here by Symmetry-Aware GFlowNets (ICML 2025)
- [Benchmarking GFlowNets against MCMC](https://journal.ut.ac.ir/article_106220_565b5b56aeb6a2e1813f39d7ffebcd62.pdf) — non-A/A\* journal; not relied upon
