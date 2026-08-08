# Evidence cross-check of the merged GRAFT plan

Date checked: 7 August 2026

## Scope and evidence rule

This report audits the merged plan in `pasted-text.txt`. I checked its central claims against the primary papers, not against summaries or blogs. I use papers from ICLR, ICML, NeurIPS, ACL, EMNLP and JMLR as the main evidence. I label arXiv-only work, Findings papers and workshop papers separately; they may motivate an experiment, but they are not used as the only evidence for a central claim.

The most important conclusion is simple:

> The overall project is viable, but the merged plan should not be implemented unchanged. Its graph-construction and retrieval stages are reasonable research plans. Its proposed GFlowNet reward and its central Forward-Looking GFlowNet argument are not mathematically correct as written.

No published paper proves that the complete proposed system will beat the listed baselines. Papers can justify components and hypotheses; superiority must be established by the project's own controlled experiments.

## Executive verdict

### Keep

- Preserve every raw turn, timestamp, speaker and supporting span.
- Store an assertion with provenance, rather than treating every supported statement as objectively true.
- Include explicit NIL/no-link prediction in open-world entity linking.
- Learn graph construction and incremental linking with neural models, while enforcing genuinely formal constraints outside the neural model.
- Use hybrid, high-recall candidate retrieval before expensive evidence-set construction.
- Treat the final evidence object as an unordered set or subgraph, not as an answer sentence.
- Freeze the SLM and use it only after evidence selection.
- Include PCST and MIP-style retrieval baselines.
- Include FL-GFN, and also LED-GFN, in the learning comparison.
- Measure graph, retrieval, packing and reader ceilings separately.

### Correct before implementation

- Replace the merged plan's hard-failure reward. `exp(beta * 0) = 1`, not zero.
- Do not call a proof-deficit **vector** an FL-GFN energy. The FL-GFN theorem requires a scalar state energy that agrees with the terminal reward.
- Do not say Graph-S3 validates the proposed FL-GFN energy. It validates synthetic stepwise supervision for its own LLM graph-retrieval agent.
- Do not describe the FL-GFN set experiment as the “exact task shape.” It is a fixed-size, synthetic, additive-energy set problem; the proposed proof task is variable-size, constrained and generally non-additive.
- Do not claim the checker is a true evaluator unless it really verifies every semantic condition. A deterministic rules engine verifies only the rules encoded in it.
- Do not call SynCheck free. It leaves the SLM weights frozen, but adds a trained monitor and inference cost.
- Do not use published scores from different memory systems as a controlled comparison. Rerun systems with the same reader, prompts, history, token budget and judge where possible.
- Remove the “85% right” estimate and the predictions of likely wins. Those numbers are not evidence-based.

## Claim-by-claim audit

| Merged-plan claim | Verdict | What the paper actually supports |
|---|---|---|
| “Supported by the turn” is not the same as “true.” | Supported | Entailment or span grounding establishes support by a source, not external truth. Citation-faithfulness work such as [ALCE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/) and [SynCheck, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.527/) separates grounding/faithfulness from general factual correctness. |
| NIL/no-link must be handled. | Supported, with wording correction | [Learn to Not Link, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.690/) shows that missing-entity and non-entity NIL examples materially affect NIL accuracy. An explicit NIL output is a well-supported design choice for an open-world graph, although no paper proves that it must be a separate neural “head.” |
| Extraction should use a context window. | Reasonable, not proved optimal | [Mem0, ECAI 2025](https://arxiv.org/html/2504.19413v1) uses recent messages plus a summary. That supports contextual extraction as an implemented design, not the claim that its exact window is optimal. Cross-turn claims may need provenance from several spans, not only one current-turn span. |
| PCST and MIP are necessary strong retrieval baselines. | Supported | [G-Retriever, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html) formulates textual-graph retrieval as Prize-Collecting Steiner Tree selection. [ARM, ACL 2025](https://aclanthology.org/2025.acl-long.1463/) is a strong retrieve-all-at-once alignment method; its full method uses joint optimization. These are meaningful non-neural or optimization-oriented competitors. |
| SynCheck gives over 0.85 AUROC and FOD gives over 10% faithfulness improvement. | Supported, but not free | Those results are stated in [SynCheck, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.527/). They are results across the paper's six long-form RAG tasks. SynCheck still requires its monitor and inference-time computation; FOD changes decoding. |
| FL-GFN beats DB, TB and SubTB on set generation, with a larger advantage on larger sets. | Supported only in the paper's setting | [FL-GFN, ICML 2023](https://arxiv.org/html/2302.01687) reports this on a didactic task that builds fixed-size sets of 20, 60 or 80 elements, with an additive fixed energy for each element. It is a mandatory baseline, but this result does not establish performance on proof selection. |
| The FL-GFN set task has the exact shape of evidence selection. | Incorrect | Evidence selection has variable stopping, hard validity conditions, temporal/authority interactions and non-additive proof sufficiency. The FL-GFN experiment uses exactly fixed set sizes and an additive terminal energy. |
| `flow_stop = exp(beta * checker_score)`, with score 0 for failed hard checks. | Incorrect | A failed set receives `exp(0) = 1`. [GFlowNet Foundations, JMLR 2023](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf) requires terminal flow to match a nonnegative reward and defines energy as `-log R`. Hard-invalid stopping must be masked or assigned true zero mass under an objective that handles it safely. |
| The proof-deficit vector can be the FL-GFN intermediate energy. | Incorrect as stated | [FL-GFN](https://arxiv.org/html/2302.01687) Assumption 4.1 requires a scalar `E:S -> R`, with transition energy `E(s')-E(s)`, and terminal energy matching `-log R`. A vector is useful as a feature, but it is not the scalar energy required by Proposition 4.2. |
| FL-GFN theory plus Graph-S3 results jointly validate the proposed method. | Incorrect inference | [Graph-S3, ACL 2026](https://aclanthology.org/2026.acl-long.1169/) trains an LLM-based retriever using synthetic golden subgraphs and stepwise supervision, reporting average gains of 15.6% accuracy and 17.2% F1 over seven baselines. It does not use a GFlowNet, FL-DB, a proof-deficit energy or a certified terminal distribution. It supports a strong supervised stepwise baseline, not the proposed theorem bridge. |
| Submodular packing scores 0.451 versus 0.429 and 0.410, and behaves differently by reader size. | Numerically reported, but provisional | These figures appear in the July 2026 arXiv preprint [What Survives Into Context](https://arxiv.org/abs/2607.00725). It is not yet an A/A* peer-reviewed publication. Its HotpotQA result is conditional on one packing formulation, a 160-token budget and specific readers. Use it as a baseline and hypothesis, not as a load-bearing conclusion. |
| Submodularity cannot represent complementary evidence. | Too broad | A submodular objective has diminishing marginal returns and cannot directly express positive superadditive interaction. However, coverage features can still favour items that cover different needs; the cited preprint does exactly this. The correct hypothesis is that **some** proof synergies may not be captured by a chosen submodular objective. Test this rather than assume it. |
| The reader-size result proves that a small frozen SLM is the right choice. | Unsupported generalization | The arXiv study reports a benefit at 3B, no benefit at 7B and a reversal at 14B in its setting. That does not prove a universal size threshold or choose the SLM for this project. Reader size must be an explicitly fixed condition, with any generalization tested separately. |
| Robust Scheduling shows the architecture requires a GFlowNet. | Overstated | [Robust Scheduling with GFlowNets, ICLR 2023](https://openreview.net/forum?id=ZBUthI6wK9h) shows that diverse candidates sampled using a cheap proxy can outperform pure proxy optimization when evaluated on expensive target hardware. This supports a proxy-mismatch hypothesis. It does not show that a GFlowNet is uniquely required, or that this project's checker is a complete true evaluator. Diverse beam, MCTS and ensembles can also produce candidates. |
| GraphMixer is a pure-MLP baseline. | Partly correct | [GraphMixer, ICLR 2023](https://arxiv.org/abs/2302.11636) has an MLP temporal-link encoder, a mean-pooling node encoder and an MLP link classifier. It is a good simple temporal-link baseline, but it was not validated on this heterogeneous assertion/conflict/update graph. |
| The CompGCN OpenReview ID is wrong. | Incorrect | The cited ID is valid: [Composition-based Multi-Relational Graph Convolutional Networks, ICLR 2020](https://openreview.net/forum?id=B8U54mHknbRi). The merged plan's citation audit contains an error here. |
| LongMemEval is a suitable primary end-to-end benchmark. | Supported, with a missing annotation issue | [LongMemEval, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf) has 500 questions covering extraction, multi-session reasoning, temporal reasoning, updates and abstention. It supplies evidence statements/sessions, but not necessarily gold entity links, all graph relations or minimal proof subgraphs. A graph-construction evaluation therefore needs additional annotation or a separate dataset. |
| A small enumerable graph plus FCS is appropriate. | Supported, with an addition | [When do GFlowNets learn the right distribution?, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a48f8928f78a58399ef0049453c14b02-Abstract-Conference.html) introduces a tractable correctness proxy and shows that GNN representation limits constrain which graph distributions a GFlowNet can approximate. On a truly enumerable state space, also report exact terminal-distribution TV/KL or JS divergence; do not use FCS as a substitute for an exact result that is available. |
| FlowRL's gains predict a win over PPO/GRPO for proof sets. | Unsupported transfer | [FlowRL, ICLR 2026](https://openreview.net/pdf?id=lObnTKbm9U) reports gains on math and code reasoning while training LLM token policies. It does not evaluate graph construction or evidence-set diversity. It can motivate a distribution-matching baseline, but it cannot predict this project's outcome. |

## The two mathematical problems that must be fixed

### 1. Define a valid terminal reward

Let `X` be a completed evidence set for question `q` and graph `G`. Let:

- `H(X,q,G)` be a hard-validity indicator: all required formal checks pass.
- `U(X,q,G)` be a scalar utility that distinguishes valid sets by sufficiency, coverage, source quality, temporal correctness, redundancy and size.

A valid form is:

```text
R(X | q,G) = 1[H(X,q,G)=1] * exp(beta * U(X,q,G))
```

This matches the standard GFlowNet requirement that terminal sampling probability be proportional to a nonnegative terminal reward ([GFlowNet Foundations, JMLR 2023](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf); [Trajectory Balance, NeurIPS 2022](https://arxiv.org/abs/2201.13259)).

Practical correction:

- Mask the `STOP` action when the partial set is formally invalid. This avoids taking logarithms of zero in common balance losses.
- Provide a separate, legal `ABSTAIN` terminal with a positive, calibrated reward. Otherwise, unanswerable questions have no valid terminal.
- If all valid sets get exactly the same utility, the target is uniform over valid sets. That may produce diversity, but it does **not** train the model to prefer stronger or smaller proofs.
- State the terminal convention explicitly. With a terminating edge to a sink, the flow on that edge equals the reward; with a terminal-state convention, the terminal state flow equals the reward.

The checker also needs a precise boundary. Type constraints, duplicate IDs, temporal interval arithmetic and allowed relation schemas can be deterministic. Entailment, factuality and source authority are usually learned or metadata-dependent. A repeatable neural score is not a formal proof merely because temperature is zero.

### 2. Use FL-GFN only with a terminal-consistent scalar energy

[FL-GFN, ICML 2023](https://arxiv.org/html/2302.01687) requires:

```text
E : states -> real numbers
E(s -> s') = E(s') - E(s)
E(X_terminal) = -log R(X_terminal)   [up to an irrelevant common constant]
```

The proposed deficit object may be a vector such as:

```text
d(s) = [missing_support, missing_time, missing_authority,
        unresolved_conflict, uncovered_claims, budget_remaining]
```

That vector can safely be:

- an input to the GNN policy;
- the target of auxiliary supervised heads; or
- mapped to a scalar potential by a learned or fixed function.

However, calling `d(s)` the FL energy does not invoke the FL-GFN theorem. If all deficits become zero at every valid terminal, a deficit-only energy also gives every valid terminal the same energy, losing any ranking by proof quality or cost.

The closest top-tier paper missing from the merged plan is [Learning Energy Decompositions for Partial Inference in GFlowNets, ICLR 2024 Oral](https://openreview.net/forum?id=P15CHILQlg). LED-GFN was designed for cases where intermediate energy is unavailable, expensive or misleading. It learns transition potentials while preserving the terminal policy under its construction. Because the proposed checker utility is non-additive and mainly terminal, LED-GFN is at least as important a baseline as ordinary FL-GFN.

Therefore the safe research order is:

1. Train an ordinary conditional TB or SubTB GFlowNet using the exact terminal reward.
2. Add supervised deficit prediction as auxiliary dense credit.
3. Run FL-DB and FL-SubTB only when a scalar state energy with the correct terminal boundary has been defined.
4. Run LED-GFN as the direct learned-decomposition baseline.
5. Call a new proof-deficit FL method a contribution only after proving its terminal consistency and checking the learned terminal distribution on an enumerable environment.

Note that the FL-GFN paper explicitly says its transformation applies to objectives other than TB. Use **FL-DB** or **FL-SubTB**, not “FL-TB.”

## Corrected architecture

### Stage A: immutable evidence and contextual extraction

Keep the raw turn, conversation/session ID, speaker, timestamp and exact character or token spans. Extract contextual assertions, entities, relations, events and temporal expressions using a bounded context. Store the distinction among:

- `asserted_by_source`;
- `entailed_by_span`;
- `externally_verified`;
- `current_under_update_policy`.

This stage is hybrid, not “mostly deterministic.” Storage, hashing and span offsets are deterministic; extraction, coreference and entailment are generally learned. Allow multi-span and cross-turn provenance when the asserted fact cannot be supported by the current turn alone.

### Stage B: learned incremental graph construction

The main idea is reasonable: encode the existing heterogeneous graph and score candidate additions, links and update operations. CompGCN is a legitimate relational encoder baseline because it jointly represents nodes and relations ([CompGCN, ICLR 2020](https://openreview.net/forum?id=B8U54mHknbRi)). GraphMixer is a legitimate simple temporal baseline ([GraphMixer, ICLR 2023](https://arxiv.org/abs/2302.11636)).

But the proposed eight heads—entity, NIL, relation, duplicate, conflict, supersession, time and authority—are a new architecture, not an architecture established by one cited paper. Present them as a hypothesis and ablate them. Some labels may be coupled: duplicate versus update versus contradiction often depends on the same pair of claims and time. A joint decoder or shared encoder with calibrated task heads is testable; it is not guaranteed to outperform separate models.

Use “neural proposes, constraints validate, commit” only for constraints that are truly checkable. A schema validator can reject an impossible edge type. It cannot reliably reject a semantically wrong entity link without another learned or human signal.

Report component metrics before end-to-end QA:

- extraction and span-grounding precision/recall/F1;
- entity-link accuracy and NIL precision/recall/F1;
- relation macro-F1;
- duplicate/conflict/update macro-F1;
- temporal interval accuracy or overlap;
- calibration (for example, Brier score or ECE);
- formal graph-constraint violation rate;
- downstream corruption after incremental updates.

Use chronological splits so that a future update cannot leak into construction of an earlier graph.

### Stage C: high-recall candidate retrieval

The merged hybrid approach is defensible: combine sparse text, dense text, entity match, temporal filters and limited graph expansion, then rerank. The purpose of this stage is recall, not the final proof decision.

Include the following strong competitors:

- BM25 and dense retrieval;
- hybrid sparse+dense retrieval;
- query-conditioned relational GNN scoring;
- G-Retriever-style PCST;
- ARM-style retrieve-all-at-once/joint optimization;
- the July 2026 submodular packer as a clearly labelled preprint baseline.

Measure required-node and required-edge Recall@k. If the candidate pool does not contain the proof, the later learner cannot recover it.

### Stage D: evidence-set generation

Use one conditional policy over a canonical partial set. Adding the same atoms in different orders must reach the same set state; do not treat orderings as different final answers. Mask repeats, budget violations and formally illegal actions.

Recommended first implementation:

- GNN graph/candidate encoder;
- separate forward policy and backward policy;
- actions: add an evidence atom, stop when valid, or abstain;
- ordinary TB/SubTB terminal learning first;
- proof-deficit vector as policy features and auxiliary labels;
- deterministic checks only for formal obligations;
- frozen SLM completely outside the learning reward.

Only promote FL-DB/FL-SubTB to the proposed method after the scalar energy issue is solved. LED-GFN should be compared because it directly addresses learned local credit. The [ICLR 2025 GFlowNet correctness paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a48f8928f78a58399ef0049453c14b02-Abstract-Conference.html) also makes GNN expressivity a central risk: two proof states that the encoder cannot distinguish cannot receive the different flows the target distribution requires. Test a more expressive encoder or positional/path features as an ablation rather than assuming message passing is sufficient.

### Stage E: frozen SLM and output verification

Keep the same frozen SLM, prompt template and decoding budget for every system. Serialize selected evidence with stable claim IDs and source spans. Evaluate answer correctness separately from citation correctness.

SynCheck is a valid optional monitor, but report its extra latency and state whether it only flags, reranks, abstains or triggers another decoding attempt. If it changes decoding, it is part of the inference algorithm and must share the same compute budget as competing methods.

## Corrected baseline design

The merged plan says to “fix everything else and swap only the training objective.” That is possible for several objectives, but not for every named system. Use two separate comparisons.

### A. Matched component comparison

Use the same graph encoder, candidate pool, action space, terminal utility, frozen SLM and maximum training/inference budget wherever the algorithm permits.

| Learning method | Include? | Reason/caveat |
|---|---:|---|
| Supervised next-action/subgraph training | Yes | Strong direct baseline; [Graph-S3, ACL 2026](https://aclanthology.org/2026.acl-long.1169/) makes stepwise supervised retrieval especially important. |
| Canonical set imitation | Yes | Tests whether generative distribution training adds value beyond imitating one gold set. |
| PPO | Yes | Reward-maximizing baseline. Use the same graph-action policy rather than comparing only to an LLM agent from another paper. |
| GRPO | Yes | Same matched-policy requirement. Search-R1 is arXiv-only and trains an LLM search agent, so it is implementation inspiration, not A/A* proof for this graph task. |
| Progressive RL and step-wise GRPO | Yes, if faithfully adapted | [AgeMem, ACL 2026](https://aclanthology.org/2026.acl-long.981/) motivates these curricula for memory operations. Their stages change more than a loss function; document the adaptation. |
| Flow Matching/Detailed Balance | Yes | Standard GFlowNet baselines from [GFlowNet Foundations, JMLR 2023](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf). |
| Trajectory Balance | Yes | Standard complete-trajectory baseline from [Trajectory Balance, NeurIPS 2022](https://arxiv.org/abs/2201.13259). |
| SubTB(lambda) | Yes | Strong credit-assignment baseline from [Subtrajectory Balance, ICML 2023](https://arxiv.org/abs/2209.12782). |
| FL-DB and FL-SubTB | Mandatory | Direct local-credit competitors from [FL-GFN, ICML 2023](https://arxiv.org/html/2302.01687). Do not create FL-TB. |
| LED-GFN | Mandatory | Direct competitor when intermediate energy must be learned; [ICLR 2024 Oral](https://openreview.net/forum?id=P15CHILQlg). |
| Guided training/prioritized replay | Yes, as an ablation | [Towards Understanding and Improving GFlowNet Training, ICML 2023](https://proceedings.mlr.press/v202/shen23a.html) studies training choices. Keep objective and replay effects separable. |
| OP-GFN | Conditional | [Order-Preserving GFlowNets, ICLR 2024](https://openreview.net/forum?id=VXDPXuq4oG) is relevant if only an ordering or multi-objective preference is trusted. It does not target the same explicit scalar reward, so label it separately. |
| COFlowNet | Conditional | [COFlowNet, ICLR 2025](https://openreview.net/forum?id=tXUkT709OJ) is an offline GFlowNet. Include it if training uses only logged trajectories without online checker evaluation; otherwise it is not a matched setting. |
| FlowRL-style objective | Secondary/adapted | [FlowRL, ICLR 2026](https://openreview.net/pdf?id=lObnTKbm9U) is valuable distribution-matching work, but its published task is LLM math/code reasoning. Clearly explain the graph-action adaptation. |
| Proposed checker-consistent deficit method | Only after formal definition | It must preserve the exact terminal target and be evaluated against FL and LED. |

“Baking Symmetry into GFlowNets” is relevant to unordered constructions, but the available publication is a NeurIPS AI4Science workshop oral, not an A/A* main-track paper. Use a canonical set state in the implementation; do not use this workshop paper as the sole basis for a central claim.

### B. Search comparison

Compare:

- greedy;
- beam;
- diverse beam;
- MCTS;
- submodular greedy;
- learned A*;
- PCST;
- MIP/joint optimization;
- raw GFlowNet sampling;
- GFlowNet candidate generation followed by checker selection.

It is not literally possible to hold the trained model identical for all of these. Learned A* and MCTS may require a value/heuristic; PCST, MIP and submodular selection use their own objectives. Fairness therefore means:

- the same input candidate pool and graph snapshot;
- the same terminal checker/utility where compatible;
- the same maximum number of scored states, model calls, wall-clock or FLOP budget;
- both an algorithm-native configuration and, where possible, a matched-score configuration;
- raw GFlowNet samples reported separately from post-filtered delivered answers.

Do not compare against “unlimited MCTS.” An unlimited-compute comparison has no defined fair stopping condition.

### C. System comparison

Run full-context, matched-budget RAG, Mem0, a graph-memory system, Zep, Memory-R1, SEEM and the proposed system where reproducible. These system-level results answer a different question from the component-level algorithm comparison. Memory-R1 and AgeMem train LLM agents, while this project freezes the SLM; published system numbers therefore cannot isolate the value of the GNN or search learner.

## Evaluation and ceilings

The merged plan's single “graph ceiling” should be split into five ceilings:

1. **Extraction ceiling:** are all gold evidence statements represented by correctly grounded assertions?
2. **Graph ceiling:** does the constructed graph contain a sufficient proof, including required links and temporal/update relations?
3. **Candidate ceiling:** does Stage C retrieve every atom needed by at least one sufficient proof?
4. **Packing ceiling:** does a sufficient proof survive the evidence/token budget and serialization?
5. **Reader ceiling:** what does the frozen SLM achieve when given a gold proof?

[LongMemEval](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf) provides evidence statements and sessions that help with these measurements. It does not automatically provide the gold heterogeneous graph or every minimal proof set. Build a manually verified graph/proof annotation for an evaluation subset, and publish the annotation policy.

Recommended result groups:

- **Graph construction:** head-level F1, NIL F1, calibration, graph violations and performance after sequential updates.
- **Retrieval:** required-node/edge Recall@k, sufficient-proof recall, candidate size and latency.
- **Set generation:** valid-terminal rate, expected terminal utility, evidence size, number and diversity of distinct valid proof sets, training sample efficiency.
- **Distribution correctness:** exact TV/KL/JS on an enumerable synthetic environment plus FCS; include representation-collision diagnostics for the GNN.
- **End to end:** LongMemEval accuracy by the five abilities, LoCoMo score, abstention precision/recall/F1, ALCE-style citation precision/recall, latency, tokens and model/checker calls.

Use the same frozen reader and judge across matched comparisons. Report several random seeds and uncertainty intervals for trained methods. A win should be stated per metric and budget, not as “beats everything.”

## What is genuinely promising

The strongest research contribution is probably **learned incremental graph construction**, because it aligns with the supervisor's NN/GNN focus and can be evaluated without relying on the final SLM. NIL-aware entity decisions, temporal/update relation prediction and heterogeneous link prediction form a coherent learned task. The papers support the importance of these subproblems, but not the claim that the exact eight-head model will win; that is the experiment.

The evidence-set GFlowNet is a legitimate higher-risk contribution because a proof may have several distinct valid evidence sets, and GFlowNets are designed to sample composite objects in proportion to reward ([GFlowNet Foundations](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf)). It is most convincing if the project demonstrates all three of the following:

- the answer has multiple materially different valid proof sets;
- diversity improves robustness or downstream success under a fixed evaluation budget;
- the learned sampler approximately matches its declared terminal target on a controlled environment.

If only one proof is needed and the final output always takes the maximum-scoring set, a reviewer may reasonably prefer beam, PCST, MIP or MCTS. The GFlowNet benefit must therefore be measured, not assumed.

## Main risks and negative evidence

1. **Error propagation.** A false merge or wrong supersession edge can corrupt every later proof. Raw provenance and non-destructive versioning reduce damage, but do not solve prediction error.
2. **Checker incompleteness.** Formal checks cannot by themselves prove semantic truth, entailment or source authority.
3. **Candidate bottleneck.** A sophisticated GFlowNet cannot recover facts excluded by Stage C.
4. **Non-additive terminal utility.** This is exactly where the simple FL-GFN argument is weakest; LED-GFN identifies unavailable or misleading intermediate energy as a practical problem.
5. **GNN representation limits.** The ICLR 2025 correctness paper proves that encoder expressivity can prevent a GFlowNet from representing the desired graph distribution.
6. **Unfair baseline transfer.** PPO/GRPO memory agents, FlowRL token policies and graph-action policies are different systems. A matched adaptation and a published-system comparison must be reported separately.
7. **Annotation cost.** LongMemEval is an end-to-end benchmark, not a complete gold graph-construction corpus. The project needs reliable component labels.
8. **No guaranteed numerical win.** No cited paper supports the merged plan's confidence predictions. The project should define falsifiable gates rather than promise victory.

## A defensible implementation order

### Gate 1: verify that the graph is worth learning

- Define the graph schema and annotation rules.
- Annotate a manually auditable development/evaluation subset for entities, NIL, relations, conflicts, updates, time and provenance.
- Measure extraction and graph ceilings.
- Train and compare simple similarity, bi/cross-encoder, GraphMixer, R-GCN/CompGCN-family and the proposed joint model.

Stop or redesign if the learned graph does not improve component accuracy or if oracle use of the graph cannot support the target questions.

### Gate 2: verify the terminal objective

- Publish the exact hard-validity predicate, scalar utility, abstention rule and terminal reward.
- Create a small environment whose terminal proof sets can be enumerated.
- Compare DB, TB, SubTB, FL-DB, FL-SubTB and LED-GFN using exact distribution error and FCS.

Do not move the proof-deficit idea into FL energy until its scalar terminal consistency is shown.

### Gate 3: establish strong search baselines

- Run greedy, beam, diverse beam, MCTS, submodular, learned A*, PCST and MIP under declared budgets.
- Report candidate, packing and reader ceilings.
- Test specifically constructed complementary-evidence cases, but also normal LongMemEval/LoCoMo cases so the test is not designed only to favour the proposed method.

### Gate 4: end-to-end test with a frozen reader

- Freeze the SLM, prompt and decoding settings before the final comparison.
- Evaluate answer, abstention, citation, cost and latency separately.
- Treat SynCheck/FOD as an explicitly costed inference variant.

## Final decision

Proceed with the project, but revise the thesis claim to something that the experiments can establish:

> A learned, provenance-preserving temporal heterogeneous graph constructor and a checker-conditioned evidence-set learner are evaluated against matched neural, RL, GFlowNet and combinatorial-search baselines under fixed reader and compute budgets.

The strongest safe core is Stage B plus the ceiling analysis. Stage D remains a good research question, but the merged plan's current “FL-DB + proof-deficit vector” is not yet a valid paper-backed method. The appropriate research comparison is ordinary terminal-reward GFlowNet versus FL-GFN versus LED-GFN, followed by a carefully defined new method if its energy or potential is mathematically consistent.

There is no paper-backed basis for saying the system will beat every named baseline. There is a paper-backed basis for saying the experiment is well motivated, the baseline set is strong after the corrections above, and the result will be scientifically meaningful whether the proposed learner wins or loses.

## Primary evidence used

- [GFlowNet Foundations — JMLR 2023](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf)
- [Trajectory Balance — NeurIPS 2022](https://arxiv.org/abs/2201.13259)
- [Subtrajectory Balance — ICML 2023](https://arxiv.org/abs/2209.12782)
- [Better Training of GFlowNets with Local Credit and Incomplete Trajectories — ICML 2023](https://arxiv.org/html/2302.01687)
- [Learning Energy Decompositions for Partial Inference in GFlowNets — ICLR 2024 Oral](https://openreview.net/forum?id=P15CHILQlg)
- [When do GFlowNets learn the right distribution? — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a48f8928f78a58399ef0049453c14b02-Abstract-Conference.html)
- [Robust Scheduling with GFlowNets — ICLR 2023](https://openreview.net/forum?id=ZBUthI6wK9h)
- [Order-Preserving GFlowNets — ICLR 2024](https://openreview.net/forum?id=VXDPXuq4oG)
- [COFlowNet — ICLR 2025](https://openreview.net/forum?id=tXUkT709OJ)
- [FlowRL — ICLR 2026](https://openreview.net/pdf?id=lObnTKbm9U)
- [Graph-S3 — ACL 2026](https://aclanthology.org/2026.acl-long.1169/)
- [G-Retriever — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)
- [ARM — ACL 2025](https://aclanthology.org/2025.acl-long.1463/)
- [LongMemEval — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf)
- [CompGCN — ICLR 2020](https://openreview.net/forum?id=B8U54mHknbRi)
- [GraphMixer — ICLR 2023](https://arxiv.org/abs/2302.11636)
- [SynCheck — EMNLP 2024](https://aclanthology.org/2024.emnlp-main.527/)
- [ALCE — EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/)

## Evidence used only with qualification

- [Learn to Not Link — Findings ACL 2023](https://aclanthology.org/2023.findings-acl.690/): peer-reviewed Findings paper, not ACL main track.
- [What Survives Into Context — arXiv 2607.00725](https://arxiv.org/abs/2607.00725): July 2026 preprint; useful provisional baseline, not A/A* venue evidence.
- [Search-R1 — arXiv 2503.09516](https://arxiv.org/abs/2503.09516): preprint; not used as main venue-backed evidence.
- [Zep temporal knowledge graph architecture — arXiv 2501.13956](https://arxiv.org/abs/2501.13956): preprint/system evidence; not proof that the proposed graph learner will reproduce its results.
- [Baking Symmetry into GFlowNets](https://openreview.net/forum?id=CZGHAeeBk3): NeurIPS 2023 AI4Science workshop oral, not a NeurIPS main-track paper.
