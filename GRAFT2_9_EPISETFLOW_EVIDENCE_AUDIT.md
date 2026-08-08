# Evidence Audit and Redesign Report for GRAFT-2.9 EpiSetFlow

**Audited artifact:** `GRAFT2_9_EPISETFLOW_PIPELINE.md`  
**Review date:** 6 August 2026  
**Scope:** GFlowNet formulation, temporal agent memory, graph retrieval, evidence-set search, proof/citation verification, learning baselines, search baselines, and evaluation design

## Executive conclusion

GRAFT-2.9 contains a promising research direction, but the current document is **not yet a mathematically justified GFlowNet formulation and is not yet empirically supported as an integrated system**. The most defensible contribution is narrower than the present novelty claim:

> A provenance-preserving, temporally scoped evidence-set constructor in which a learned policy proposes diverse sets, while a non-neural checker alone decides whether a set may be used for an authoritative read or write.

Four parts of the plan are strongly motivated by published evidence:

1. **Raw evidence plus temporal/provenance metadata.** Zep/Graphiti, SEEM, LongMemEval, Mem0, and related memory systems all show that source links, update handling, time, and episodic structure matter. The papers do not establish that a knowledge graph is always best, but they do support keeping source evidence and lifecycle information separate from derived memory ([Zep](https://arxiv.org/abs/2501.13956), [SEEM](https://aclanthology.org/2026.acl-long.277/), [LongMemEval](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf), [Mem0](https://arxiv.org/abs/2504.19413)).
2. **Constrained construction and a deterministic authorization boundary.** Constrained GFlowNet work demonstrates the value of putting hard feasibility rules in the action space, and citation-verification work shows that generation quality alone is insufficient for reliable attribution ([SynFlowNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7495fa446f10e9edef6e47b2d327596e-Abstract-Conference.html), [ALCE](https://aclanthology.org/2023.emnlp-main.398/), [VeriCite](https://doi.org/10.1145/3767695.3769505), [SynCheck](https://aclanthology.org/2024.emnlp-main.527/)). A checker can guarantee compliance with its encoded rules; it cannot by itself guarantee real-world truth.
3. **Set-valued, diverse output.** GFlowNets are explicitly designed to sample composite objects, including sets and graphs, approximately in proportion to a nonnegative reward, and can represent multiple modes rather than return only one optimum ([GFlowNet Foundations](https://jmlr.org/papers/v24/22-0364.html), [Trajectory Balance](https://arxiv.org/abs/2201.13259)). This is a reasonable match to the desire for several alternative valid evidence sets.
4. **A hybrid retrieval stack.** Leading GraphRAG results support combinations of lexical/dense retrieval, graph structure, learned GNN scoring, path extraction, and sometimes iterative retrieval. No single search family dominates every query type ([HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html), [GNN-RAG](https://aclanthology.org/2025.findings-acl.856/), [GFM-RAG](https://papers.neurips.cc/paper_files/paper/2025/hash/33ca0b1102b54c191a9a45a05adafaf4-Abstract-Conference.html), [ARM](https://aclanthology.org/2025.acl-long.1463/)).

Five central claims do **not** follow from the cited literature or from the current equations:

1. The two conservation laws do not define how exploratory mass becomes authoritative mass. There is no authoritative root/source term or formal cross-channel transition. The all-zero authoritative flow is therefore a solution, and standard reward-proportional GFlowNet results do not apply.
2. Adding arbitrary local exploratory source rewards changes the flow network. Without a derived terminal measure, it is not the standard single-source GFlowNet objective and does not inherit the usual proportional-sampling guarantee.
3. The proposed deficit-potential loss does not automatically preserve the terminal distribution. Published local-credit guarantees require a terminal energy that extends consistently to intermediate states or decomposes over transitions ([FL-GFN](https://arxiv.org/abs/2302.01687)).
4. “Closure” cannot mean completeness with respect to an open world. It can only be a certificate relative to a pinned graph snapshot, a declared candidate generator, and explicit proof obligations.
5. A deterministic certificate guarantees only that its program returned `true`. Truth, entailment, source quality, entity resolution, and temporal correctness remain empirical properties of the checker and its inputs.

**Recommendation:** continue the work only as a gated research programme. First implement a simpler certified supervised/set-imitation system. Add an ordinary conditional GFlowNet only after exact small-state tests establish reward-proportional sampling. Treat the dual epistemic formulation as a later research hypothesis that must be formally specified and independently validated. If it does not beat the supervised and optimization baselines under equal budgets, retain the simpler system.

## 1. Review method and evidential standard

The source document was read section by section, including its objects, canonical states, dual mass equations, proof obligations, counterfactual credit, closure residual, training schedule, inference procedure, complexity claims, baselines, ablations, success gates, and novelty statement.

The evidence search prioritized peer-reviewed work in leading machine-learning, NLP, and information-retrieval venues: JMLR, NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, and their Findings tracks. Directly relevant 2025–2026 papers from other peer-reviewed venues were also included. Preprints and under-review papers were used only as provisional or conditional evidence and are labelled as such. Venue labels are reported factually; venue prestige is not treated as evidence that a result transfers to GRAFT.

The audit uses three confidence levels:

- **Supported:** the design choice is directly consistent with repeated or close empirical/theoretical evidence.
- **Plausible but unproven:** adjacent literature motivates the choice, but no paper validates it in this setting.
- **Unsupported or contradicted as written:** the claim does not follow from its cited work, conflicts with the proposal's own equations, or has credible counterevidence.

No located paper evaluates the complete EpiSetFlow architecture. All conclusions about the integrated design are therefore analytical inferences from component evidence, not reported experimental results.

## 2. What the pipeline actually proposes

The document defines two set-generation tasks.

- **EditSetFlow** constructs a set of write atoms for memory changes and then commits it atomically.
- **EvidenceSetFlow** constructs a bounded set of evidence nodes, edges, and bindings for answering a query.

Both use canonical set states: different insertion orders that yield the same set map to the same state. A temporal GNN scores graph content. A set policy adds atoms or stops. An obligation vector and closure residual determine whether stopping is allowed. The plan separates exploratory flow, (F_X), from authoritative flow, (F_A), and requires a deterministic certificate \(\kappa\) before an authoritative transition. Terminal utilities combine correctness, sufficiency, minimality, risk, and cost. Counterfactual removal scores attempt to assign local evidence credit. Incremental graph updates and optional adaptive \(\tau\) messages aim to reduce cost.

This is best understood as six coupled claims, which need separate validation:

1. a temporal/provenance memory representation is beneficial;
2. a GNN can rank useful graph evidence or edit candidates;
3. sequential set construction is better than independent or one-shot selection;
4. GFlowNet training is better than supervised learning, RL, search, or optimization;
5. the certificate is sound and useful;
6. incremental execution preserves decisions while reducing cost.

End-to-end gains cannot identify which of these claims is true. The experiment design must isolate them.

## 3. Component-by-component verdict

| Pipeline component | Verdict | What the literature supports | Required change |
|---|---|---|---|
| Canonical set state | **Supported engineering choice; not novel by itself** | GFlowNet foundations explicitly include set construction; FL-GFN describes set states with element-insertion transitions. Symmetry work argues that unhandled isomorphic actions can waste samples or distort learned flows, although its experiments are synthetic and the work is a workshop/preprint rather than a main-track result ([Foundations](https://jmlr.org/papers/v24/22-0364.html), [FL-GFN](https://arxiv.org/abs/2302.01687), [Baking Symmetry](https://arxiv.org/abs/2406.05426)). | Keep canonical hashing and aggregate all paths to the same state. Claim implementation discipline, not a new GFlowNet principle. |
| Temporal GNN candidate scorer | **Plausible and strongly motivated** | GNN-RAG and GFM-RAG show strong learned graph retrieval; Graph-S3 shows value from step-level supervision. Their settings are KGQA/document retrieval, not long-term memory authority ([GNN-RAG](https://aclanthology.org/2025.findings-acl.856/), [GFM-RAG](https://papers.neurips.cc/paper_files/paper/2025/hash/33ca0b1102b54c191a9a45a05adafaf4-Abstract-Conference.html), [Graph-S3](https://aclanthology.org/2026.acl-long.1169/)). | Keep, but compare against lexical+dense, MLP triple scoring, PPR, and no-GNN variants. Do not assume graph propagation is always beneficial. |
| Raw evidence, source spans, bitemporal lifecycle | **Supported** | Zep keeps episodic raw content alongside semantic graph data and models valid/transaction time. LongMemEval directly tests temporal reasoning, updates, and abstention. SEEM uses provenance pointers and reverse provenance expansion ([Zep](https://arxiv.org/abs/2501.13956), [LongMemEval](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf), [SEEM](https://aclanthology.org/2026.acl-long.277/)). | Make raw evidence immutable; store derived assertions separately; pin both valid time and transaction/snapshot time in every certificate. |
| Exploratory versus authoritative frontiers | **Plausible systems separation; unproven learning law** | Memory papers show error propagation, stale facts, and the need for selective write/delete behaviour. No reviewed paper establishes two conserved GFlowNet masses for epistemic authority ([Memory-R1](https://aclanthology.org/2026.acl-long.583/), [Memory management study](https://aclanthology.org/2026.acl-long.27/), [Memora](https://arxiv.org/abs/2604.20006)). | Retain two queues/statuses in the system. Until a theorem exists, do not call both quantities GFlowNet mass. Use one proposal distribution and a separate certification gate. |
| Deterministic certificate \(\kappa\) | **Supported boundary, limited guarantee** | Constrained construction supports exact action rules; attribution work supports post-generation verification. None shows that a deterministic program can establish truth from noisy extractions ([SynFlowNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7495fa446f10e9edef6e47b2d327596e-Abstract-Conference.html), [ALCE](https://aclanthology.org/2023.emnlp-main.398/), [VeriCite](https://doi.org/10.1145/3767695.3769505)). | Specify certificate semantics. Test checker soundness/completeness separately. Say “certified under rules R on snapshot t,” not “authoritative truth.” |
| Proof obligations | **Supported as an explicit contract; exact list unvalidated** | ALCE decomposes citation correctness into support/recall and citation precision; LongMemEval separates temporal/update/abstention capabilities. RARR and SynCheck show value in explicit attribution and faithfulness checks ([ALCE](https://aclanthology.org/2023.emnlp-main.398/), [RARR](https://aclanthology.org/2023.acl-long.910/), [SynCheck](https://aclanthology.org/2024.emnlp-main.527/)). | Keep the obligation interface, but validate each obligation's checker and marginal benefit. Distinguish claim entailment, source identity, time validity, conflict handling, and bounded closure. |
| STOP mask until certified | **Supported for hard feasibility** | Constrained generation can encode domain rules in allowable actions; standard GFlowNet guarantees are over the resulting DAG ([SynFlowNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7495fa446f10e9edef6e47b2d327596e-Abstract-Conference.html), [FL-GFN background](https://arxiv.org/abs/2302.01687)). | Keep. Ensure at least one path reaches a terminal or abstention state and every non-root state has a valid backward path. |
| Deficit potential \(\Phi\) | **Plausible feature; unsupported as a distribution-preserving reward** | FL-GFN supplies a rigorous local-credit route only when terminal energy extends to all states or decomposes over transitions ([FL-GFN](https://arxiv.org/abs/2302.01687)). | Use deficits first as features/auxiliary labels. If used in flow equations, derive a partial energy whose terminal value exactly matches the intended terminal reward. |
| Leave-one-out necessity/redundancy | **Useful diagnostic; insufficient credit model** | ALCE uses removal-style citation precision, but it is a citation metric, not a complete cooperative credit assignment method. Data Shapley reports that Shapley valuation can be more informative than leave-one-out in its supervised-data setting ([ALCE](https://aclanthology.org/2023.emnlp-main.398/), [Data Shapley](https://proceedings.mlr.press/v97/ghorbani19c.html)). | Fix the formulas; add group-removal and exact minimal-subset labels where feasible. Treat the BCE as auxiliary, not as a conservation guarantee. |
| Frozen SLM verbalizer | **Plausible control, not a faithfulness guarantee** | ALCE, RARR, SynCheck, and VeriCite all show that generated text can be unsupported even when evidence is present; verification improves but does not eliminate the problem. | Constrain the verbalizer to certified bindings, require claim-level citations, and measure entailment and unsupported-claim rate. Freezing weights is not enough. |
| Incremental delta update | **Plausible with strict conditions** | Dynamic GNN work can update affected representations exactly or approximately under architecture-specific propagation rules; it does not justify arbitrary freezing ([InstantGNN, IJCAI 2022](https://www.ijcai.org/proceedings/2022/0438.pdf), [InkStream, IPDPS 2025](https://www.comp.nus.edu.sg/~tulika/IPDPS25.pdf)). | Define the affected set layer by layer, include feature/lifecycle deletions, and compare decisions against full recomputation. Report disagreement, not only latency. |
| Atomic write-set commit | **Sensible systems requirement; not validated by the cited ML papers** | Set-level certification is compatible with all-or-none application, but the reviewed memory papers do not establish GRAFT's transaction protocol. | Precheck the entire edit set, write under one transaction/version, and test crash, retry, rollback, and concurrent-update cases. Never infer transaction correctness from model accuracy. |
| Route and calibration losses | **Underspecified** | The objective names \(\mathcal L_{\mathrm{route}}\) and \(\mathcal L_{\mathrm{cal}}\) but does not define labels, predicted events, or estimands. | Define route classes and the probability being calibrated. Report Brier/ECE and risk–coverage on held-out data; adding a loss called “calibration” is not evidence of calibration. |
| State features: authority, query, deficit, budget | **Plausible feature design; unvalidated combination** | Temporal/query features and explicit step supervision are supported separately by LongMemEval and Graph-S3, but no reviewed paper validates this exact state vector. | Keep as an initial feature set and ablate each group. Prevent certificate outputs unavailable at decision time from leaking into policy inputs. |
| Adaptive \(\tau\) | **Undefined and unsupported in this document** | No operational definition, estimator, compression rule, or cited validation connects \(\tau\) to EpiSetFlow correctness. | Remove from the first implementation. Reintroduce only with a precise definition and an accuracy/cost ablation. |
| Defaults: 1 greedy + 3 stochastic, 16 atoms, 5 hops | **Unsupported constants** | Graph and iterative retrieval papers show query- and dataset-dependent effects; iteration can help multi-hop queries and hurt simple queries ([IRCoT](https://aclanthology.org/2023.acl-long.557/), [ARM](https://aclanthology.org/2025.acl-long.1463/), [Beyond Static Retrieval, preprint](https://arxiv.org/abs/2509.25530)). | Treat all as development hyperparameters. Tune under a declared compute/context budget and report sensitivity curves. |

## 4. Formal audit of the GFlowNet design

### 4.1 What standard results require

The conventional formulation uses a directed acyclic state graph with a unique initial state, terminal objects, a normalized forward policy, and a nonnegative terminal reward \(R(x)\). The target is

\[
P_F^\top(x) \propto R(x).
\]

Flow matching, detailed balance, trajectory balance, or appropriate subtrajectory constraints can imply this result under their stated conditions. The guarantees are idealized: full support and global minimization appear in the theory, while practical training can be underdetermined and sample-inefficient ([FL-GFN preliminaries](https://arxiv.org/abs/2302.01687), [Trajectory Balance](https://arxiv.org/abs/2201.13259), [GFlowNet training analysis](https://proceedings.mlr.press/v202/shen23a.html), [SubTB](https://proceedings.mlr.press/v202/madan23a.html)).

EpiSetFlow does not yet provide all of those objects for the authoritative channel.

### 4.2 The missing exploratory-to-authoritative law

The document gives an exploratory balance of the form

\[
\sum_{p \in \mathrm{Pa}(s)} F_X(p\!\to\!s) + R_X(s)
= \sum_{c \in \mathrm{Ch}(s)}F_X(s\!\to\!c)+F_X(s\!\to\!\mathrm{STOP}),
\]

and an authoritative balance gated by \(\kappa\):

\[
\sum_{p}\kappa(p,s)F_A(p\!\to\!s)
=\sum_c\kappa(s,c)F_A(s\!\to\!c)+F_A(s\!\to\!\mathrm{STOP}).
\]

The prose says exploratory states can “discover” authority, but the equations do not contain a projection edge, a transfer term, an authoritative root flow, or a terminal boundary condition tying \(F_A\) to a reward. Consequently:

- \(F_A=0\) satisfies the authoritative equation;
- the scale and induced terminal distribution of \(F_A\) are unidentified;
- the exploratory reward cannot determine the authoritative distribution;
- standard DB/TB/SubTB results cannot be invoked for the pair of equations.

This conclusion follows directly from the stated equations. COFlowNet does not fill the gap: it is an **offline** GFlowNet method that conservatively restricts flow through state-action regions unsupported by a fixed dataset. It does not define epistemic authority, deterministic proof transfer, or two interacting probability masses ([COFlowNet, ICLR 2025](https://openreview.net/forum?id=tXUkT709OJ)).

### 4.3 Two defensible reformulations

**Preferred first implementation: one GFlowNet plus an external gate.**

1. A proposal GFlowNet samples candidate sets \(X\) from a canonical set DAG.
2. A deterministic checker returns \(C(X,q,G_t)\in\{\text{valid},\text{invalid},\text{abstain}\}\) and a certificate.
3. Only `valid` sets enter authoritative consumption; invalid sets are rejected; abstention is an explicit terminal decision.
4. The learned distribution is described as a proposal distribution, not an authority distribution.

This is simple, testable, and avoids making an unsupported probabilistic claim.

**Research formulation: a single augmented DAG.** Define states \((X,z)\), where \(z\in\{E,A\}\), and explicit certified edges

\[
(X,E) \rightarrow (X,A) \quad\text{iff}\quad C(X,q,G_t)=\text{valid}.
\]

The augmented graph needs one root, normalized forward and backward policies with correct support, terminal reward boundaries, and conservation on **all** edges including certification edges. This would turn “two frontiers” into a precise state variable. A new theorem or a reduction to an existing GFlowNet theorem would still be required before claiming proportional sampling over authoritative sets.

### 4.4 Local source reward changes the target

Ordinary GFlowNet flow is injected at the source and removed according to terminal reward. The proposed \(R_X(s)\) injects mass at intermediate states. This is a generalized multi-source flow network, not automatically the standard terminal-reward construction. Unless the authors derive the terminal measure produced by all source terms, the meaning of “sampled proportional to utility” is ambiguous.

For early credit, FL-GFN provides the closer published construction. It assumes a terminal energy \(\mathcal E(x)\) that extends to intermediate states and uses transition differences

\[
\mathcal E(s\to s')=\mathcal E(s')-\mathcal E(s).
\]

Under that structure, it derives a modified balance objective while retaining the target distribution ([FL-GFN](https://arxiv.org/abs/2302.01687)). EpiSetFlow should either follow this derivation or leave the deficit signal outside the flow equation.

### 4.5 Reward must be nonnegative

The document's terminal utility is a signed linear combination. Standard GFlowNet theory requires a nonnegative reward. A defensible transformation is

\[
R(X\mid q,G_t)=\mathbf 1[C(X,q,G_t)=\text{valid}]\exp(U(X\mid q,G_t)/T).
\]

In practice, exact zero rewards interact badly with log-ratio losses. It is cleaner to mask actions so invalid terminal states are unreachable, preserve a certified abstention terminal, and apply the positive reward only on feasible terminals. Temperature \(T\) changes the target distribution and must be reported.

### 4.6 Potential shaping is not established

The document defines \(\Phi(s)\) from obligation deficits and uses \(\Phi(s')-\Phi(s)\) as a shaped gain, while asserting terminal ranking is unchanged. Telescoping alone is insufficient if terminal potentials differ or if the added terms enter an auxiliary loss that shares parameters with the flow policy. The exact terminal target must remain unchanged, or the modification must be derived as a valid intermediate-energy factorization. FL-GFN is evidence that this can be done under explicit assumptions; it is not evidence that every deficit potential works.

Recommended sequence:

1. train TB or SubTB with terminal rewards only;
2. use obligation deficits as input features and supervised auxiliary labels;
3. measure whether they improve convergence without changing exact small-space terminal frequencies;
4. only then test an FL-GFN-compatible energy extension.

SubTB is particularly relevant because it trades bias and variance across partial trajectories and reported better convergence/stability on long constructions ([SubTB](https://proceedings.mlr.press/v202/madan23a.html)). Shen et al. also found that prioritized replay and objectives that improve substructure credit can materially affect training, reinforcing the need to report the training recipe rather than only the loss name ([Shen et al., ICML 2023](https://proceedings.mlr.press/v202/shen23a.html)).

### 4.7 Backward policy requirements

A deterministic “remove the last canonical atom” backward rule is mathematically possible only if it is a normalized policy over valid parents and every sampled terminal can reach the root. It also selects one reverse trajectory among potentially many. This may reduce variance, but it changes credit allocation and can remove the benefit of recognizing multiple construction paths.

The first comparison should include:

- uniform removal among all valid parents;
- deterministic canonical removal;
- learned backward policy;
- learned backward policy with full-support regularization.

Report terminal-distribution error, training variance, mode discovery, and wall-clock cost. “Efficient backward operation” should not be presented as inherently superior before this ablation.

### 4.8 Order-Preserving GFlowNets is not support for set canonicalization

Order-Preserving GFlowNets concerns preserving a total or partial **preference order over candidate rewards** when the exact reward values are unavailable or unreliable. It is not primarily about permutation invariance of element-insertion sequences ([Order-Preserving GFlowNets, ICLR 2024](https://arxiv.org/abs/2310.00386)). Cite GFlowNet set construction and symmetry work for canonical states; cite OP-GFN only if the terminal target is learned from ordinal human/expert preferences.

### 4.9 FlowRL and COFlowNet are conditional baselines, not direct foundations

FlowRL matches a reward-induced distribution for LLM reasoning and reports gains over PPO/GRPO on mathematical reasoning and code. It is relevant when the output is an LLM token trajectory and when reward-distribution matching is the research question. It does not validate graph evidence sets, proof closure, or temporal memory writes ([FlowRL, ICLR 2026](https://openreview.net/forum?id=lObnTKbm9U)).

COFlowNet is relevant only if EpiSetFlow is trained offline from a fixed logged dataset and extrapolation outside dataset support is a primary risk. Its “conservative” constraint must not be reinterpreted as factual conservatism or deterministic authority ([COFlowNet](https://openreview.net/forum?id=tXUkT709OJ)).

### 4.10 Post-sampling choice changes the delivered distribution

The read procedure proposes one greedy and several stochastic constructions and then chooses a set using validity, minimality, sufficiency, and risk. Even if the stochastic policy exactly sampled \(P(X)\propto R(X)\), taking the best or first accepted set from a batch creates a different order-statistic/rejection distribution. Mixing a greedy result into the batch changes it again.

Therefore the paper must distinguish:

- **proposal distribution:** raw samples from the learned forward policy;
- **accepted distribution:** proposals conditioned on checker acceptance;
- **delivered decision rule:** the deterministic or stochastic rule that selects one set for use.

Only the first can directly inherit an ordinary GFlowNet proportionality statement. The second can be described by conditioning if acceptance is explicit; the third needs its own evaluation and must not be called a reward-proportional sample. If the operational goal is simply the best certified set under four draws, compare that decision rule directly with beam search, diverse beam, and optimization.

## 5. Memory and temporal evidence

### 5.1 What Mem0 supports—and what it does not

Mem0 uses extraction followed by `ADD`, `UPDATE`, `DELETE`, or `NOOP`; its graph variant adds entities and relations. This strongly supports comparing EpiSetFlow writes against operation-classification and graph-memory baselines. On LoCoMo, however, the graph variant is not uniformly better: it helps some open-domain and temporal categories while hurting or failing to improve others. The paper's graph advantage is modest in aggregate, and its evaluation omits the adversarial/unanswerable subset for which labels were unavailable. Therefore Mem0 does not establish graph superiority, safe abstention, or deterministic truth ([Mem0](https://arxiv.org/abs/2504.19413)).

Mem0's latency and token savings are useful engineering evidence, but they are system-specific comparisons against full-context or competing memory pipelines. EpiSetFlow must measure its own ingestion, update, read, and p95 latency; those numbers cannot be transferred.

### 5.2 Memory-R1 is the closest learning baseline

Memory-R1 trains a Manager for memory operations and an Answer Agent for memory use. It compares supervised fine-tuning, PPO, and GRPO and uses a distillation step over retrieved memories. It reports strong results on LoCoMo with transfer to MSC and LongMemEval, but the best RL algorithm is not universal across every table, and supervised variants remain competitive. The authors train the two agents separately because of sparse reward and optimization difficulty ([Memory-R1, ACL 2026](https://aclanthology.org/2026.acl-long.583/)).

Consequences for GRAFT:

- Memory-R1 must be an end-to-end matched baseline, not reduced to generic “PPO” and “GRPO” rows.
- Separate write and read policies are the evidence-backed default. Shared encoders can be ablated, but joint training should not be assumed stable.
- Outcome-only RL must be compared with supervised operation labels and evidence-level/process supervision.

### 5.3 AgeMem clarifies two listed baselines

AgeMem introduces a progressive curriculum over long-term memory construction, short-term distractor control, and integrated management, using actions such as add, update, delete, retrieve, summarize, and filter. Its “step-wise GRPO” assigns the terminal group-normalized advantage to earlier actions; it is not an independently established local-credit algorithm for arbitrary set construction ([AgeMem, ACL 2026](https://aclanthology.org/2026.acl-long.981/)).

The baseline table should therefore say “AgeMem progressive curriculum” and “AgeMem step-wise GRPO implementation,” using the same action space and base model where possible. Treating the names as generic checkboxes would make the comparison irreproducible.

### 5.4 Temporal graphs are useful, but graph complexity has credible counterevidence

Zep/Graphiti's bitemporal graph, episode store, hybrid BM25/cosine/graph retrieval, and contradiction invalidation support GRAFT's temporal and provenance fields. Its source pointers are not evaluated as citation guarantees, and its strongest reported benchmark comparisons have limitations. It is evidence for architecture hypotheses, not proof of authority ([Zep](https://arxiv.org/abs/2501.13956)).

Recent peer-reviewed alternatives warn against assuming that a complex entity graph is necessary:

- Chain-of-Memory reports that lightweight memory construction plus dynamic utilization can outperform complex memory structures at much lower token and latency cost ([Chain-of-Memory, ACL 2026](https://aclanthology.org/2026.acl-long.534/)).
- StructMem argues that entity resolution and symbolic graph construction can be expensive and fragile, and uses event-centred hierarchical memory instead ([StructMem, ACL 2026](https://aclanthology.org/2026.acl-short.12/)).
- HyperMem uses a hypergraph to represent higher-order relations rather than reducing all structure to pairwise edges ([HyperMem, ACL 2026](https://aclanthology.org/2026.acl-long.1627/)).
- SEEM explicitly joins relational semantic memory to episodic evidence with provenance pointers, making it a particularly close alternative to the proposed source-binding design ([SEEM](https://aclanthology.org/2026.acl-long.277/)).

The correct conclusion is not “graphs work” or “graphs do not work.” It is that graph structure can help relational and multi-hop retrieval, while construction errors and cost can offset the gain. GRAFT needs no-graph, event-hierarchy, pairwise graph, and—if high-order proof interactions matter—hypergraph ablations.

### 5.5 Error propagation makes certification worth testing

Published memory studies show that blindly storing or replaying experience can propagate errors and that selective add/delete policies can improve downstream performance. Memora focuses on forgetting obsolete memories; the ACL 2026 memory-management study reports substantial degradation from misaligned experience following and gains from selective operations ([Memory management study](https://aclanthology.org/2026.acl-long.27/), [Memora preprint](https://arxiv.org/abs/2604.20006)).

This supports a write gate and retirement semantics. It does **not** establish that GRAFT's proposed obligations are the right gate. Required tests include false acceptance of poisoned writes, false rejection of valid updates, stale-memory use, wrong-entity merges, deleted-source reuse, and rollback after certificate invalidation.

## 6. Graph retrieval and evidence-set search

### 6.1 Learned graph retrieval is a strong candidate generator

GNN-RAG scores KG nodes and extracts shortest paths connecting question entities to high-scoring answers. It reports large gains over several KG reasoning methods while using fewer KG tokens, but also finds that combining its retrieval with RoG can help, exposing generalization limitations in either method alone ([GNN-RAG](https://aclanthology.org/2025.findings-acl.856/)).

GFM-RAG pretrains a relatively small graph foundation model over many graphs, then transfers it to document retrieval. It reports strong Recall@5 and efficiency across multi-hop datasets compared with dense and graph baselines ([GFM-RAG](https://papers.neurips.cc/paper_files/paper/2025/hash/33ca0b1102b54c191a9a45a05adafaf4-Abstract-Conference.html)). This supports pretrained/shared graph scoring as a baseline and candidate generator. It does not show that a task-specific temporal GNN plus GFlowNet is necessary.

Graph-S3 uses synthetic stepwise supervision to train graph retrieval and reports that step supervision improves over sparse final reward ([Graph-S3](https://aclanthology.org/2026.acl-long.1169/)). This is directly relevant counterevidence to starting with complex flow/RL training: a supervised step-labelled retriever may capture much of the benefit.

### 6.2 Strong simple and optimization baselines are mandatory

SubgraphRAG uses lightweight parallel triple scoring and adjustable subgraph size, demonstrating that a simple learned selector can be competitive ([SubgraphRAG, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html)). G-Retriever formulates textual-graph retrieval as Prize-Collecting Steiner Tree selection, directly matching the desire for a small connected evidence subgraph ([G-Retriever, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)). ARM uses constrained alignment plus mixed-integer optimization to retrieve a connected set of heterogeneous objects all at once and reports better cost/performance than iterative agentic baselines on its tasks ([ARM, ACL 2025](https://aclanthology.org/2025.acl-long.1463/)).

These papers make an independent-atom scorer, MLP triple scorer, PCST/MIP solver, and exact enumeration on small candidate pools necessary baselines. Without them, any EpiSetFlow gain could come from the checker, graph features, or combinatorial objective rather than from flow learning.

### 6.3 Iterative retrieval helps conditionally

IRCoT interleaves a reasoning sentence with retrieval and reports large retrieval and answer improvements on several multi-hop QA datasets ([IRCoT, ACL 2023](https://aclanthology.org/2023.acl-long.557/)). KiRAG similarly uses iterative knowledge-driven retrieval and reports improved early-rank recall on multi-hop tasks ([KiRAG, ACL 2025](https://aclanthology.org/2025.acl-long.929/)). R3-RAG trains a reason-and-retrieve policy with outcome and relevance process rewards; its ablations support both RL and process feedback in that setting ([R3-RAG, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.554/)). FLARE shows that deciding when and what to retrieve during generation can be better than retrieving once on long-form tasks ([FLARE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.495/)).

But iterative search is not universally superior. ARM reports cases where iterative agents forget prior information, loop, or incur higher cost, and proposes retrieve-all-at-once structural alignment instead ([ARM](https://aclanthology.org/2025.acl-long.1463/)). The supplied “Beyond Static Retrieval” study reports gains on bridge/multi-hop questions but limited or negative effects on simpler comparison questions, plus noise and diminishing returns; as of this audit it is an arXiv preprint/ICLR submission, not established main-track evidence ([preprint](https://arxiv.org/abs/2509.25530), [OpenReview submission](https://openreview.net/forum?id=d8xbed6XAB)).

The EpiSetFlow frontier should therefore be compared with both static/global selection and iterative expansion. Query-adaptive routing is a hypothesis to test, not a conclusion to assume.

### 6.4 PPR, paths, and planned traversal cover different inductive biases

- HippoRAG combines a knowledge graph with Personalized PageRank and reports strong multi-hop results at substantially lower online cost than iterative retrieval in its experiments ([HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html)).
- RoG plans relation paths, retrieves executable paths, then reasons over them; it is a strong interpretable path baseline ([RoG, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e2aeb66481dd63a32421bf032b70384-Abstract-Conference.html)).
- KG2RAG expands semantically retrieved seeds using KG relations and organizes the result, providing a simple graph-guided expansion baseline ([KG2RAG, NAACL 2025](https://aclanthology.org/2025.naacl-long.449/)).
- A*Net learns a priority function and prunes graph traversal for knowledge-graph completion. It is relevant to the listed “learned A*” search baseline, but its original task is link prediction, not proof-bearing QA; adaptation must use the same evidence reward and budget ([A*Net, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html)).

These methods should not be collapsed into one “graph search” row.

### 6.5 Context packing is a separate stage

ALCE shows that retrieval recall can be much higher than downstream citation correctness and that simply adding more passages eventually plateaus, indicating limited multi-passage use ([ALCE](https://aclanthology.org/2023.emnlp-main.398/)). The supplied 2026 budget-constrained packing paper introduces answer-in-context and reports conditional gains from submodular packing only under a particular combination of multi-hop complementarity, sufficient retrieval, a binding but non-extreme budget, and a small reader. It is a recent single-author preprint with narrow dataset coverage, so its findings should be treated as provisional rather than general ([What Survives Into Context](https://arxiv.org/abs/2607.00725)).

Accordingly, report four separate quantities:

1. relevant evidence retrieved into the candidate pool;
2. sufficient evidence selected into the certified set;
3. evidence surviving serialization/truncation into the reader context;
4. claims actually supported by cited evidence in the answer.

Optimizing only Recall@k or evidence-set reward cannot establish the final two.

## 7. Proof, closure, and citation reliability

### 7.1 A proof certificate must have scoped semantics

The certificate should be a reproducible record, not merely a Boolean:

```text
certificate_id
query_contract_hash
graph_snapshot_id / transaction_time
valid_time_interval(s)
selected_atom_ids
immutable source-span hashes
entity-resolution decisions
claim-to-source entailment results
conflict-resolution rule and alternatives considered
candidate-generator version and bounds
obligation results
checker version
terminal decision: valid | abstain | invalid
```

This design is consistent with the provenance and citation literature, but the exact schema is a GRAFT design recommendation. Its success must be measured.

### 7.2 Closure must be bounded

The current residual includes unprojected support, hypotheses, disputes, lag, and budget. A zero vector cannot show that no relevant fact exists outside the bounded graph or candidate inventory. Define instead:

> **Snapshot-relative closure:** every obligation in contract \(Q\) is either satisfied or explicitly discharged by an abstention reason, with respect to graph snapshot \(G_t\), candidate generator \(g_v\), depth/budget \(B\), and checker \(C_v\).

This is auditable. “Complete evidence” or “world closure” is not.

LongMemEval's explicit abstention category supports making insufficient evidence a first-class outcome rather than forcing an answer ([LongMemEval](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf)). Self-RAG likewise shows benefits from adaptive rather than indiscriminate retrieval, but its reflection tokens are learned judgments, not deterministic proof ([Self-RAG, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html)).

### 7.3 Citation verification should be claim-level

ALCE evaluates citation recall (whether answer claims are supported) and citation precision/necessity; RARR retrieves evidence, attributes output, and revises unsupported text; SynCheck detects fine-grained faithfulness problems and uses them during decoding; VeriCite adds evidence verification and reports improved citation quality ([ALCE](https://aclanthology.org/2023.emnlp-main.398/), [RARR](https://aclanthology.org/2023.acl-long.910/), [SynCheck](https://aclanthology.org/2024.emnlp-main.527/), [VeriCite](https://doi.org/10.1145/3767695.3769505)).

These results support the pipeline's proof-carrying direction. They also show why a set-level `valid` label is too coarse. Each atomic answer claim needs:

- one or more cited source spans;
- an entailment decision;
- correct entity and time binding;
- contradiction/conflict status;
- an unsupported/abstain path.

VeriCite is a peer-reviewed SIGIR-AP 2025 paper, not a SIGIR main-conference result. The iterative GraphRAG and 2026 packing papers are preprints. Their evidence should be weighted accordingly.

### 7.4 Leave-one-out credit has a structural failure

Suppose either \(e_1\) or \(e_2\) independently proves a claim. For \(X=\{e_1,e_2\}\), removing either leaves the score unchanged. The proposed necessity score assigns both zero, even though every minimal proof needs one of them. Conversely, an evidence item may be necessary only jointly with another item. Individual removal cannot identify all such interactions.

There is also a formula problem. With

\[
D(e;X)=\max(0,S(X\setminus\{e\})-S(X)+\epsilon),
\]

an exactly neutral removal becomes positively redundant solely because of \(\epsilon\). When \(N=D=0\), the sigmoid target is 0.5, which is not a clear include/exclude label.

Recommended hierarchy:

1. exact minimal-subset labels when candidate sets are small;
2. deterministic checker-based labels for whether an atom participates in at least one minimal certificate;
3. individual leave-one-out as a cheap diagnostic;
4. pair/group removal for known multi-hop bundles;
5. Shapley/Banzhaf-style approximation only if the extra computation is justified.

Data Shapley found advantages over leave-one-out in a different domain—training-data valuation—so it motivates but does not validate Shapley evidence credit here ([Data Shapley](https://proceedings.mlr.press/v97/ghorbani19c.html)).

## 8. Revised architecture recommended for the first publishable study

### 8.1 Data model

Keep three layers:

1. **Immutable episodic/source layer:** raw text/event, source identity, immutable span hash, transaction time, valid-time assertions, access/retirement status.
2. **Derived semantic layer:** entities, relations, normalized values, temporal intervals, confidence, contradiction links, and pointers to all supporting raw spans.
3. **Certificate layer:** query/write contract, selected atom set, checker results, snapshot, model/checker versions, and outcome.

Never overwrite raw evidence. Updates create new derived versions and invalidate or close old valid-time intervals. This design follows the strongest common ground across Zep, SEEM, LongMemEval, and selective memory-management work.

### 8.2 Candidate generation

Generate a recall-oriented bounded pool using a union of:

- BM25/lexical retrieval;
- dense retrieval;
- temporal/entity filters;
- one graph method, initially PPR or supervised GNN scoring;
- source provenance expansion for derived assertions;
- optional one- or two-round bridge expansion on queries classified as multi-hop.

Deduplicate by immutable source identity and canonical semantic atom. Record which generator contributed every atom. The pool boundary becomes part of the closure certificate.

### 8.3 Selection and certification

Implement selectors in this order:

1. independent supervised atom scorer + greedy/ILP;
2. permutation-invariant set predictor or autoregressive set imitation;
3. ordinary conditional GFlowNet over canonical sets;
4. only after exact validation, experimental dual/augmented EpiSetFlow.

After every proposed set, run the same deterministic checker. The checker must be held constant across selectors so that search quality is not confounded with certification quality.

### 8.4 Reward and termination

Use a feasible-set reward such as

\[
U(X)=w_s\operatorname{suff}(X)+w_f\operatorname{faith}(X)
-w_n|X|-w_c\operatorname{cost}(X)-w_r\operatorname{risk}(X),
\]

\[
R(X)=\exp(U(X)/T),\qquad X\in\mathcal X_{\text{certified}}.
\]

Do not let a weighted utility compensate for a failed hard obligation. Invalid source, wrong snapshot, unresolved required conflict, or hypothesized-only support must mask termination. Include a certified abstention terminal with its own calibrated cost.

### 8.5 Verbalization

The answer model receives only selected source spans plus machine-readable bindings. It must emit claim IDs and citation IDs. A post-verbalization checker reruns claim-level entailment and temporal/entity binding. If checks fail, revise from certified material or abstain. This is closer to the attribution evidence than assuming that a frozen SLM will remain faithful.

### 8.6 Incremental execution

Cache embeddings with explicit dependencies. On an edit:

1. find nodes/edges/features and certificates touched by the change;
2. propagate invalidation through every GNN layer's actual receptive field;
3. invalidate temporal/conflict/retirement certificates even if their text embedding is unchanged;
4. recompute affected candidate pools and proofs;
5. periodically compare with full recomputation.

An incremental result is acceptable only if set decisions, certificate decisions, and answer claims stay within declared tolerances of full recomputation. Exact dynamic-GNN speedups are architecture-dependent, as InstantGNN and InkStream make clear.

## 9. Required learning baselines

Every baseline must use the same train/validation/test split, graph snapshot, candidate pool where applicable, certificate checker, reader/verbalizer, maximum selected atoms, context budget, model-size class, and accounting of LLM calls. Otherwise the comparison does not isolate the learning algorithm.

### 9.1 Minimum matched baselines

| Family | Baseline | Why it is necessary | Required output |
|---|---|---|---|
| Non-learned | heuristic obligation completion | Tests whether explicit rules solve most cases | one certified set |
| Supervised independent | atom BCE/ranking scorer | Lowest-complexity learned selector | scores + greedy/top-k set |
| Supervised set | Set Transformer | Permutation-invariant interactions among atoms ([Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)) | one-shot set |
| Supervised set | bipartite-matching set loss | Avoids imposing an arbitrary target order; set generation has been trained this way in NLP ([Set Generation for KB Population, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.760/)) | one-shot set |
| Imitation | canonical-order autoregressive | Directly matches the proposed Stage 2 | sequential set |
| Imitation | randomized-order autoregressive | Tests sensitivity to arbitrary canonical order | sequential set |
| Structured prediction | scorer + exact enumeration/ILP on small pools | Gives an oracle-quality search control | optimum under stated utility |
| RL | PPO with same actions/reward | Standard reward-maximization comparison | stochastic policy |
| RL | GRPO with same actions/reward | Direct match to Memory-R1/AgeMem-style training | stochastic policy |
| Memory RL | full Memory-R1 Manager + Answer Agent | Closest learned memory-operation baseline ([Memory-R1](https://aclanthology.org/2026.acl-long.583/)) | writes + answers |
| Memory RL | AgeMem curriculum/step-wise GRPO | Tests the named progressive baseline as published ([AgeMem](https://aclanthology.org/2026.acl-long.981/)) | memory actions + answers |
| Retrieval RL | R3-RAG-style process reward | Tests reason/retrieve RL with local relevance feedback ([R3-RAG](https://aclanthology.org/2025.findings-emnlp.554/)) | retrieval trajectories |
| GFlowNet | Flow Matching / Detailed Balance | Foundational local objectives | terminal samples |
| GFlowNet | Trajectory Balance | Established whole-trajectory objective ([TB](https://arxiv.org/abs/2201.13259)) | terminal samples |
| GFlowNet | SubTB with tuned \(\lambda\) | Strong partial-trajectory baseline ([SubTB](https://proceedings.mlr.press/v202/madan23a.html)) | terminal samples |
| GFlowNet | FL-GFN | Correct comparator for local/partial energy ([FL-GFN](https://arxiv.org/abs/2302.01687)) | terminal samples |
| GFlowNet | ordinary conditional set GFlowNet | Tests whether epistemic dual mass adds anything | terminal samples |
| Proposed | EpiSetFlow variants | Tests each added mechanism | terminal samples + certificates |

### 9.2 Conditional rather than universal baselines

- **COFlowNet:** include only for fixed logged/offline training with incomplete state-action coverage.
- **FlowRL:** include when token-level reasoning trajectories and distribution matching are comparable to the proposed output.
- **Order-Preserving GFN:** include only when the reward supervision is ordinal.
- **MCMC:** include on enumerable/synthetic or compact spaces with a matched target distribution. A supplied 2025 journal comparison finds performance depends on peak structure and reports high training cost for GFlowNets, but it uses small synthetic HyperGrid settings and is not top-venue evidence; it motivates a diagnostic baseline, not a general conclusion ([GFN versus MCMC](https://journal.ut.ac.ir/article_106220_565b5b56aeb6a2e1813f39d7ffebcd62.pdf)).

### 9.3 Training controls that must be reported

- parameter count and initialization;
- number of reward/checker evaluations;
- on-policy versus replay samples;
- replay prioritization;
- exploration policy and full-support mechanism;
- backward-policy form;
- reward temperature and clipping;
- number of trajectories/updates and wall-clock time;
- hyperparameter selection budget;
- seed-level results.

Shen et al. show that replay and parameterization materially affect GFlowNet learning, so hiding these details would make an objective-level comparison misleading ([GFlowNet training analysis](https://proceedings.mlr.press/v202/shen23a.html)).

## 10. Required search and retrieval baselines

### 10.1 Candidate retrieval

1. BM25 only.
2. Dense retriever only.
3. BM25+dense hybrid with a fixed fusion rule.
4. Hybrid + temporal/entity filter.
5. HippoRAG/PPR-style graph diffusion.
6. GNN-RAG node scoring + shortest paths.
7. GFM-RAG or a task-matched pretrained graph retriever.
8. SubgraphRAG-style parallel triple scorer.

Measure pool Recall@k, bridge-evidence recall, source recall, wrong-entity rate, stale/deleted-evidence rate, and candidate-generation cost before set selection.

### 10.2 Static/global set selection

1. top-k by independent relevance;
2. greedy marginal utility;
3. MMR/diversity selection;
4. beam search;
5. diverse beam with a stated diversity penalty;
6. exact enumeration for small candidate pools;
7. ILP/MIP maximizing relevance plus compatibility, following ARM's class of solver;
8. PCST/Steiner selection following G-Retriever;
9. budgeted submodular packing, explicitly labelled as a conditional/preprint-derived baseline.

### 10.3 Iterative and path search

1. KG2RAG seed expansion and organization;
2. RoG planned relation paths;
3. IRCoT;
4. KiRAG;
5. FLARE for long-form active retrieval;
6. R3-RAG learned iterative policy;
7. adapted A*Net priority traversal;
8. MCTS only when it receives the same reward/value oracle and exact expansion budget;
9. EpiSetFlow sampling.

For iterative methods, report rounds, retrieved objects per round, duplicate rate, loop rate, LLM calls, and evidence-rank movement. Compare them against ARM-style retrieve-all-at-once selection. This directly tests the literature's conflicting results on iterative versus global search.

### 10.4 Memory-system baselines

On write/read memory tasks, include at minimum:

- full-context/no-memory reader;
- recency window;
- vector RAG;
- Mem0 base and Mem0 graph;
- Zep/Graphiti-style temporal graph retrieval if reproducibly available;
- Memory-R1;
- AgeMem;
- one lightweight/event-centric alternative such as StructMem or Chain-of-Memory;
- one provenance-aware alternative such as SEEM.

Not every paper's code may be compatible with the same environment. If a system cannot be reproduced, report that fact and implement only clearly labelled component baselines; do not present an approximate reimplementation under the original name.

## 11. Evaluation programme

### 11.1 Phase A: exact synthetic set validation

Construct tasks with 8–20 candidate atoms so all terminal sets and rewards can be enumerated. Include:

- single sufficient proofs;
- multiple disjoint valid modes;
- substitutable evidence;
- complementary two- and three-hop bundles;
- distractors;
- conflicting/stale/deleted evidence;
- no-answer cases;
- symmetric construction orders.

Primary metrics:

- total variation, Jensen–Shannon divergence, and KL where defined between empirical terminal frequency and normalized reward;
- Spearman correlation of reward and sample frequency;
- valid-mode coverage and time-to-first-discovery per mode;
- conservation residuals;
- invalid STOP rate;
- certificate false accept/reject rates;
- calibration of abstention.

This phase is a prerequisite for any claim that EpiSetFlow samples authoritative sets proportionally to reward. A high downstream QA score cannot replace it.

### 11.2 Phase B: evidence selection under fixed candidate pools

Use gold and realistically retrieved candidate pools on HotpotQA, 2WikiMultiHopQA, MuSiQue, and an applicable KGQA dataset. Freeze the reader. Compare only selectors.

Metrics:

- evidence-node and edge precision/recall/F1;
- exact sufficient-set rate;
- minimal-certified-set rate and excess atoms;
- answer-in-context;
- answer EM/F1;
- citation entailment, citation recall, and citation precision;
- distinct valid sets, pairwise Jaccard diversity among valid sets, and valid mode coverage;
- tokens, checker calls, latency, and memory.

Diversity must be conditioned on validity and utility. Counting distinct invalid or low-quality sets is not useful.

### 11.3 Phase C: temporal memory writes and reads

Use LongMemEval and LoCoMo, plus adversarial update/deletion tasks. LongMemEval covers information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention; LoCoMo supplies long conversational dependencies ([LongMemEval](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf), [LoCoMo, ACL 2024](https://aclanthology.org/2024.acl-long.747/)).

Add controlled cases for:

- correction of a previously stored value;
- two valid values over non-overlapping time intervals;
- source deletion/retirement;
- same-name entities;
- malicious or low-authority sources;
- unsupported hypotheses;
- conflict without resolvable precedence;
- incomplete evidence requiring abstention;
- memory poisoning followed by replay.

Write metrics:

- operation accuracy for add/update/delete/noop;
- exact edit-set and atom F1;
- false authority promotion;
- invalid merge/split rate;
- stale fact survival;
- certificate invalidation latency;
- atomic-commit rollback correctness.

Read metrics:

- strict answer accuracy;
- temporal and entity-binding accuracy;
- claim-level support;
- stale/deleted evidence use;
- abstention precision/recall and risk–coverage;
- proof sufficiency/minimality;
- p50/p95 latency and token cost.

### 11.4 Phase D: incremental equivalence and cost

For every update, run both incremental and full recomputation on a sampled audit stream. Report:

- node embedding error where approximation is allowed;
- candidate-pool disagreement;
- selected-set exact match and Jaccard;
- certificate decision disagreement;
- final-answer disagreement;
- invalidation misses;
- update throughput and p50/p95 latency;
- memory overhead.

The delta method passes only if its accuracy/certificate tolerances and speed threshold are declared before evaluation.

### 11.5 Statistical protocol

Use multiple seeds for learned systems, confidence intervals for paired differences, and paired resampling for dataset-level answer metrics. Report all planned primary outcomes and correct for multiple primary comparisons. Also report failure counts for safety-critical events; a mean answer score can hide rare false-authority promotions.

## 12. Essential ablations

The source document's ablation list is directionally good, but it should be reorganized around causal questions:

1. **Representation:** no graph vs event hierarchy vs pairwise graph; no provenance vs source pointers; no temporal edges vs bitemporal.
2. **Candidate scorer:** lexical/dense vs MLP vs temporal GNN vs pretrained graph model.
3. **Set learner:** independent scorer vs one-shot set vs imitation vs ordinary GFlowNet vs EpiSetFlow.
4. **Epistemic mechanism:** no gate; gate only; proposal+gate; augmented dual-state flow. This isolates whether the value comes from certification rather than two masses.
5. **Credit:** terminal-only TB; SubTB; deficit auxiliary; FL-GFN partial energy; leave-one-out; group-removal.
6. **Backward policy:** uniform, canonical deterministic, learned.
7. **Closure:** obligations only; bounded closure; falsely claimed global closure must not be evaluated as a legitimate variant.
8. **Search:** static, retrieve-all-at-once optimization, iterative, query-adaptive routing.
9. **Diversity:** greedy only; independent stochastic samples; GFlowNet samples, all at equal candidate/checker budget.
10. **Verbalizer:** unconstrained; frozen; citation-constrained; post-verified.
11. **Incremental execution:** full recomputation; exact affected-set update; approximate delta.
12. **Budget sensitivity:** selected atoms, hops, trajectories, context tokens, and wall-clock budget.

The most important ablation is **ordinary conditional GFlowNet plus the same certificate versus the proposed dual formulation**. Without it, the paper cannot attribute a gain to the epistemic flow idea.

## 13. Complexity audit

The current asymptotic section is incomplete because it omits several potentially dominant terms:

- candidate generation over the corpus/graph;
- temporal and entity filtering;
- number of GNN layers and affected-neighbourhood expansion;
- proof-checker and entailment calls;
- leave-one-out or group counterfactual evaluations;
- serialization/context packing;
- number of sampled trajectories and rejected invalid sets;
- database reads, cache invalidations, and certificate recomputation.

A more honest decomposition is

\[
C_{\text{turn}}=C_{\text{retrieve}}+C_{\text{GNN}}+C_{\text{select}}
+C_{\text{cert}}+C_{\text{pack}}+C_{\text{read}}.
\]

For stochastic selection,

\[
C_{\text{select}}\approx N_{\text{traj}}\,H\,(C_{\text{policy}}+C_{\text{mask}})

\]

plus terminal checking and rejected trajectories. Counterfactual credit costs at least one re-evaluation per removed atom unless cached or approximated, and group removal can be combinatorial.

For delta execution, cost depends on the actually affected nodes/edges at every message-passing layer, not simply the size of the directly edited neighbourhood. InstantGNN and InkStream obtain efficiency through architecture-specific incremental propagation; they do not justify freezing all apparently untouched encodings.

Report empirical scaling against candidate-pool size, graph degree, proof length, update batch size, and number of requested samples. Big-O notation alone will not establish the claimed deployment advantage.

## 14. Revised success gates

Proceed from one stage to the next only if all earlier gates pass.

### Gate 1: checker validity

- certificate schema is deterministic and replayable;
- near-zero rule nondeterminism;
- false acceptance and false rejection measured on adversarial labelled cases;
- every accepted claim has immutable source provenance and snapshot/time binding;
- abstention exists for unresolved obligations.

### Gate 2: ordinary set learning

- supervised set/imitation model beats independent top-k on proof sufficiency or cost at fixed validity;
- graph model beats lexical+dense and simple MLP under total ingestion+query cost, or is removed;
- incremental method stays within declared full-recompute disagreement tolerance.

### Gate 3: standard GFlowNet

- exact synthetic terminal distribution matches normalized reward within a predeclared tolerance;
- produces more high-utility valid modes than beam/diverse beam/optimization at equal reward-call budget;
- cost is acceptable relative to simpler selectors.

### Gate 4: epistemic extension

- formal augmented-state or coupled-flow definition is complete;
- no degenerate zero/scale solution;
- exact synthetic authoritative distribution is validated;
- improves a predeclared metric over ordinary conditional GFlowNet plus the identical certificate;
- false-authority promotion does not increase.

### Gate 5: end-to-end value

- improves temporal memory and evidence-grounded QA across more than one dataset family;
- claim-level citation/faithfulness gains survive equal token, model, and retrieval budgets;
- p95 latency and update cost are reported, including graph construction;
- failure analysis covers false closure, stale evidence, conflicts, and poisoning.

If Gate 4 fails, publish or deploy the simpler proposal+certificate system rather than preserving dual mass for novelty.

## 15. Claims that are supportable after appropriate experiments

Potentially defensible:

- “We formulate evidence and memory actions as canonical set construction with immutable provenance and bitemporal scope.”
- “A learned proposal policy generates multiple candidate evidence/edit sets; a deterministic checker exclusively controls authorization under a declared rule set.”
- “We evaluate sufficiency, minimality, claim-level attribution, abstention, temporal updates, and diverse valid-mode coverage under matched compute budgets.”
- If proven and verified: “We define an augmented-state GFlowNet whose certified terminals are sampled approximately in proportion to a nonnegative utility.”

Not currently defensible:

- “Canonical set construction is novel.”
- “COFlowNet provides authoritative or safety-constrained flow.”
- “The two conservation equations guarantee authoritative reward-proportional sampling.”
- “A zero closure residual proves that the evidence is complete.”
- “Deterministic certification guarantees factual truth.”
- “Potential shaping preserves terminal ranking” without a derivation and tests.
- “Graphs outperform simpler memory.”
- “One greedy plus three stochastic samples, 16 atoms, five hops, or adaptive \(\tau\) are evidence-based defaults.”
- “Order-Preserving GFlowNets solves permutation invariance.”

## 16. Bottom-line decision

The plan should proceed, but in a reduced and falsifiable form.

**Build now:** immutable source/provenance store; bitemporal derived memory; explicit proof obligations; bounded closure; deterministic certificate; abstention; hybrid candidate retrieval; supervised GNN and set-imitation baselines; exact small-state harness; claim-level citation verification.

**Build after those pass:** ordinary conditional GFlowNet with TB/SubTB and a positive terminal reward; uniform/deterministic/learned backward ablation; diversity evaluation under matched budgets.

**Do not build yet:** dual exploratory/authoritative flow loss, arbitrary intermediate source reward, deficit shaping inside conservation, adaptive \(\tau\), or unsupported fixed sampling/hop defaults.

**Research before claiming:** a precise augmented-state/coupled-flow theorem; checker scope and error model; group evidence credit; full-versus-incremental equivalence; proof that any local credit leaves the terminal target unchanged.

This ordering preserves the strongest idea in GRAFT-2.9—the separation of neural proposal from rule-bound authorization—while preventing the paper from depending on claims that the present equations and literature do not support.

## 17. Annotated reference map

### GFlowNets

1. Bengio et al. [GFlowNet Foundations](https://jmlr.org/papers/v24/22-0364.html), JMLR 2023 — composite sets/graphs, reward-proportional target, flow foundations.
2. Malkin et al. [Trajectory Balance](https://arxiv.org/abs/2201.13259), NeurIPS 2022 — trajectory-level balance and credit assignment.
3. Madan et al. [Learning GFlowNets from Partial Episodes](https://proceedings.mlr.press/v202/madan23a.html), ICML 2023 — SubTB and partial-trajectory bias/variance.
4. Pan et al. [Better Training with Local Credit and Incomplete Trajectories](https://arxiv.org/abs/2302.01687), ICML 2023 — FL-GFN and the intermediate-energy assumptions.
5. Shen et al. [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html), ICML 2023 — underdetermination, replay, parameterization, practical training.
6. Chen and Mauch. [Order-Preserving GFlowNets](https://arxiv.org/abs/2310.00386), ICLR 2024 — ordinal reward/preferences, not set permutation invariance.
7. Ma et al. [Baking Symmetry into GFlowNets](https://arxiv.org/abs/2406.05426), workshop/preprint — isomorphic actions and canonical representations; synthetic evidence.
8. Cretu et al. [SynFlowNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7495fa446f10e9edef6e47b2d327596e-Abstract-Conference.html), ICLR 2025 — domain constraints embedded in a construction process.
9. Liu et al. [COFlowNet](https://openreview.net/forum?id=tXUkT709OJ), ICLR 2025 — conservative offline GFlowNet under dataset-support limitations.
10. [FlowRL](https://openreview.net/forum?id=lObnTKbm9U), ICLR 2026 — reward-distribution matching for LLM reasoning.

### Agent memory

11. Packer et al. [Mem0](https://arxiv.org/abs/2504.19413), ECAI 2025 — add/update/delete/noop memory and graph variant.
12. [Memory-R1](https://aclanthology.org/2026.acl-long.583/), ACL 2026 — RL memory Manager and Answer Agent; PPO/GRPO/SFT comparisons.
13. Rasmussen et al. [Zep/Graphiti](https://arxiv.org/abs/2501.13956), preprint — bitemporal knowledge graph, episodic provenance, hybrid retrieval.
14. Wu et al. [LongMemEval](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf), ICLR 2025 — long-term memory abilities including updates, temporal reasoning, and abstention.
15. Maharana et al. [LoCoMo](https://aclanthology.org/2024.acl-long.747/), ACL 2024 — very long-term conversational memory benchmark.
16. [AgeMem](https://aclanthology.org/2026.acl-long.981/), ACL 2026 — progressive RL and step-wise GRPO memory management.
17. [Chain-of-Memory](https://aclanthology.org/2026.acl-long.534/), ACL 2026 — lightweight construction and dynamic utilization; graph-cost counterpoint.
18. [StructMem](https://aclanthology.org/2026.acl-short.12/), ACL 2026 — event-centred structured memory without fragile entity graph construction.
19. [SEEM](https://aclanthology.org/2026.acl-long.277/), ACL 2026 — semantic/episodic memory and provenance expansion.
20. [HyperMem](https://aclanthology.org/2026.acl-long.1627/), ACL 2026 — hypergraph memory for higher-order associations.
21. [How Memory Management Impacts LLM Agents](https://aclanthology.org/2026.acl-long.27/), ACL 2026 — error propagation and selective memory management.
22. [Memora](https://arxiv.org/abs/2604.20006), Findings of ACL 2026/preprint record — forgetting obsolete memory.

### Graph and iterative retrieval

23. Jin et al. [GNN-RAG](https://aclanthology.org/2025.findings-acl.856/), Findings of ACL 2025 — GNN node retrieval and path extraction.
24. Luo et al. [GFM-RAG](https://papers.neurips.cc/paper_files/paper/2025/hash/33ca0b1102b54c191a9a45a05adafaf4-Abstract-Conference.html), NeurIPS 2025 — pretrained graph foundation retriever.
25. He et al. [G-Retriever](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html), NeurIPS 2024 — PCST subgraph retrieval for textual graphs.
26. Luo et al. [RoG](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e2aeb66481dd63a32421bf032b70384-Abstract-Conference.html), ICLR 2024 — relation planning and executable KG paths.
27. [SubgraphRAG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html), ICLR 2025 — lightweight parallel triple scoring.
28. [HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html), NeurIPS 2024 — graph memory retrieval with PPR.
29. Zhu et al. [KG2RAG](https://aclanthology.org/2025.naacl-long.449/), NAACL 2025 — semantic seeds, graph expansion, and organization.
30. Trivedi et al. [IRCoT](https://aclanthology.org/2023.acl-long.557/), ACL 2023 — iterative reasoning and retrieval.
31. Jiang et al. [FLARE](https://aclanthology.org/2023.emnlp-main.495/), EMNLP 2023 — active retrieval during long-form generation.
32. [KiRAG](https://aclanthology.org/2025.acl-long.929/), ACL 2025 — iterative knowledge-driven retrieval.
33. [R3-RAG](https://aclanthology.org/2025.findings-emnlp.554/), Findings of EMNLP 2025 — RL retrieval with process reward.
34. Chen et al. [ARM](https://aclanthology.org/2025.acl-long.1463/), ACL 2025 — retrieve-all-at-once alignment and MIP selection.
35. Zhu et al. [A*Net](https://papers.nips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html), NeurIPS 2023 — learned priority traversal for KG completion.
36. [Graph-S3](https://aclanthology.org/2026.acl-long.1169/), ACL 2026 — stepwise-supervised graph retrieval.
37. Guo et al. [Beyond Static Retrieval](https://arxiv.org/abs/2509.25530), 2025 preprint/ICLR submission — conditional benefits and noise pitfalls of iterative GraphRAG.
38. Bala. [What Survives Into Context](https://arxiv.org/abs/2607.00725), 2026 preprint — answer-in-context and conditional submodular packing.

### Attribution and set modelling

39. Gao et al. [ALCE](https://aclanthology.org/2023.emnlp-main.398/), EMNLP 2023 — automatic citation evaluation and retrieval-generation gap.
40. Gao et al. [RARR](https://aclanthology.org/2023.acl-long.910/), ACL 2023 — research, attribution, and revision.
41. [SynCheck](https://aclanthology.org/2024.emnlp-main.527/), EMNLP 2024 — fine-grained faithfulness monitoring and guided decoding.
42. Qian et al. [VeriCite](https://doi.org/10.1145/3767695.3769505), SIGIR-AP 2025 — citation verification.
43. Asai et al. [Self-RAG](https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html), ICLR 2024 — adaptive retrieval and self-reflection.
44. Lee et al. [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), ICML 2019 — permutation-invariant set modelling.
45. Ye et al. [Set Generation for Knowledge Base Population](https://aclanthology.org/2021.emnlp-main.760/), EMNLP 2021 — order-free matching loss for generated sets.
46. Ghorbani and Zou. [Data Shapley](https://proceedings.mlr.press/v97/ghorbani19c.html), ICML 2019 — cooperative valuation and empirical comparison with leave-one-out in data valuation.

### Incremental graph computation

47. Zheng et al. [Instant Graph Neural Networks for Dynamic Graphs](https://www.ijcai.org/proceedings/2022/0438.pdf), IJCAI 2022 — incremental graph representation updates.
48. Wu et al. [InkStream](https://www.comp.nus.edu.sg/~tulika/IPDPS25.pdf), IPDPS 2025 — architecture-aware incremental GNN inference with exact outputs for supported aggregators.
