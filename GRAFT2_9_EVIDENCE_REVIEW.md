# GRAFT-2.9 / EpiSetFlow — Evidence Review

**What this is:** a component-by-component audit of `GRAFT2_9_EPISETFLOW_PIPELINE.md` against published work.
**Question answered:** for each design decision, is there published evidence that it works, evidence that it fails, or no evidence either way?
**Rule I followed:** every claim below is tied to a specific paper, or is explicitly labelled as *analysis* (my own reasoning about the document's math, not a paper's finding), or is labelled *no evidence found*. Nothing is asserted because it "sounds right".

**Date of review:** 6 August 2026.

---

## 0. How to read this

Each component gets one of five verdicts:

| Verdict | Meaning |
|---|---|
| **SUPPORTED** | Published, peer-reviewed evidence that this mechanism works, in a setting close enough to matter. |
| **SUPPORTED-BUT-WEAKER-THAN-CLAIMED** | It works, but the document overstates its novelty or its effect size. |
| **CONDITIONAL** | It works only under conditions the paper spells out. GRAFT must check those conditions hold. |
| **CONTRADICTED** | Published evidence points the other way, or shows the benefit reverses. |
| **NO EVIDENCE** | I could not find published work for or against. Not the same as "wrong" — it means this is research risk that must be measured, not assumed. |

A section labelled **⚠ SPEC DEFECT** is a problem I found in the document's own mathematics, independent of the literature.

---

## 1. Corrections to the reading list — read this first

Five items on the supplied list are mis-attributed. This matters because the plan's credibility depends on citing correctly, and a reviewer will check.

| As listed | Actual |
|---|---|
| *"What Survives Into Context: Budget-Constrained Multi-Hop RAG and Submodular Evidence Packing"* | Title is **"What Survives Into Context: A Diagnostic for Budget-Constrained Multi-Hop RAG and When Submodular Evidence Packing Improves It"** (arXiv:2607.00725 v1, 1 Jul 2026). **v2 (5 Aug 2026) was retitled "Recall Is Not Enough: A Reader-Context Diagnostic for Budget-Constrained Retrieval-Augmented Generation".** Single-author arXiv preprint (Ananto Nayan Bala, AUST). **Not an A/A\* venue, not peer-reviewed.** Its content is excellent and directly relevant — but cite it as a preprint. |
| *"Baking Symmetry into GFlowNets"* | **NeurIPS 2023 AI-for-Science workshop paper** (Ma, Bengio, Bengio, Zhang), synthetic experiments on graphs of ≤7 nodes. Not archival, not A\*. **Use [Symmetry-Aware GFlowNets, ICML 2025](https://arxiv.org/abs/2506.02685) instead** — same problem, main-track venue, with theorems. |
| *"Benchmarking GFlowNets against MCMC: The Role of Peak Structure"* | Published in the **Journal of Advances in Computing** (Univ. of Tehran), Vol. 57(2), Dec 2025. **Not an A/A\* venue.** I could not extract its numeric results (the PDF is image-based), so I treat its conclusions as low-confidence and have not leaned on them anywhere below. |
| *"Zep: A Temporal Knowledge Graph Architecture for Agent Memory"* | **arXiv preprint (2501.13956), authored by the vendor (Zep AI).** No peer-reviewed venue found. Its numbers are usable but should be labelled as vendor-reported — especially since [Mem0 (ECAI 2025)](https://arxiv.org/html/2504.19413v1) reports operational difficulty reproducing Zep (see §4.6). |
| *"VeriCite … (SIGIR-AP 2025)"* | Correct venue, but SIGIR-AP is the Asia-Pacific regional SIGIR conference, **not** SIGIR main. Don't present it as A\*. |

Two more notes:
- **GFM-RAG** was accepted at **NeurIPS 2025**, not ICML (per the authors' repository). The version you linked is the arXiv v2.
- **Mem0** is confirmed at **ECAI 2025** (Frontiers in AI and Applications, vol. 413).
- The plan's own reference #5 (Bu et al., hierarchical GFlowNet data synthesis) checks out: **[Findings of ACL 2025, pp. 15931–15958](https://aclanthology.org/2025.findings-acl.821/)**.
- The plan's reference #7 (Let the Flows Tell) checks out: **NeurIPS 2023 spotlight**.

---

## 2. Executive summary

### 2.1 The one-line verdict

The **systems half** of GRAFT-2.9 (certified writes, bi-temporal graph, deterministic proof checking, minimal evidence sets, frozen-SLM verbalization) is well-supported by published work and is the part most likely to produce a defensible paper. The **learning half** (EpiSetFlow) is the risky part: two of its five claimed innovations are already standard in the GFlowNet literature, one has a real mathematical gap in the document, and the composite objective has no published precedent as a whole.

The single most important finding is a **specification defect, not a literature disagreement**: as written, §8's flow-conservation losses are never connected to §9's terminal utility. See §3.9 below. Everything else is secondary to fixing that.

### 2.2 Component verdict table

| § | Component | Verdict |
|---|---|---|
| 2.3 | Canonical (order-invariant) set states | **SUPPORTED-BUT-WEAKER-THAN-CLAIMED** — native to GFlowNets since 2021 |
| 3 | Dual epistemic mass (F_X / F_A) | **NO EVIDENCE** — no precedent found; nearest neighbours are multi-objective / augmented flows |
| 3.3, 8.3 | Certification-gated authoritative flow (κ ∈ {0,1}) | **SUPPORTED** as action masking; **⚠ two spec gaps** |
| 5.2, 8.5 | Proof-deficit potential Φ | **CONTRADICTED as implemented** — the invariance claim requires FL-GFN-style reparametrization, not an auxiliary head |
| 6 | Counterfactual (leave-one-out) evidence credit | **SUPPORTED** in principle; **CONDITIONAL** on cost; **⚠ redundancy term is near-vacuous** |
| 7 | Closure residual | **NO EVIDENCE** — plausible; aggregate queries over incomplete KGs are a known-hard case |
| 8.2, 8.3 | Flow-Matching-style parent/child conservation | **CONTRADICTED** — FM is the weakest-credit objective in the literature; TB/SubTB/FL-DB dominate it |
| 8.4 | Squared log-residual loss | **CONDITIONAL** — standard, but shown not to be the best choice |
| 8.7 | Six-term composite objective | **NO EVIDENCE** — no theoretical guarantee survives the combination |
| 9.3 | Feasibility-first (utility → infeasible) | **CONDITIONAL** — creates the sparse-reward regime where GFlowNets are documented to struggle |
| 11.4 | Sample-then-filter with deterministic checker | **SUPPORTED** — this is the strongest argument in the whole plan (see §3.14) |
| 12 | Deterministic backward policy | **SUPPORTED** at the optimum; **CONDITIONAL** on efficiency; ablation correctly required |
| 13 | Incremental delta training | **NO EVIDENCE** — nearest work is ICLR 2025 distributed GFlowNet training |
| 15.3 | Budget: 4 sets × 16 atoms × 5 hops | **CONDITIONAL** — latency-feasible, but 5 GNN hops needs justification |
| — | Graph memory over flat memory | **CONTRADICTED for single/multi-hop; SUPPORTED for temporal** (Mem0, ECAI 2025) |
| — | Minimal sufficient proof sets | **CONDITIONAL** — benefit reverses with stronger readers (arXiv 2607.00725) |
| — | Proof → frozen SLM → citations | **SUPPORTED** (ALCE EMNLP 2023; VeriCite SIGIR-AP 2025; Self-RAG ICLR 2024) |
| — | Single-shot GNN retrieval vs. iterative | **SUPPORTED** (GFM-RAG NeurIPS 2025; HippoRAG NeurIPS 2024; GNN-RAG Findings ACL 2025) |
| 18 | Baseline list | **INCOMPLETE** — missing every memory-system baseline; see §5 |

---

# Part A — The learning core (EpiSetFlow)

## 3.1 What GFlowNets actually buy you, per the literature

Before auditing individual claims, it's worth being precise about the *documented* advantage, because the plan leans on it repeatedly.

The advantage is **diversity of high-reward candidates**, and it only exists when many trajectories map to the same object. [Shen et al. (ICML 2023)](https://arxiv.org/abs/2305.07170) state this sharply: when trajectories and objects are one-to-one, GFlowNets *reduce to discrete-action soft Q-learning*; the distinct benefit applies specifically when "many trajectories for each x" exist, where "autoregressive or standard RL methods learn biased generative probabilities, while GFlowNets do not."

GRAFT's set-construction DAG is exactly that regime — a set of *k* atoms has *k!* construction orders. So the plan is in the right regime for GFlowNets to matter. **But the same paper shows this is also the regime where credit assignment is worst.** Its Remark 3 says that with many trajectories per object, "trajectory balance and maximum entropy GFlowNets inadequately credit substructures of x associated with high reward," and its Proposition 5.4 shows TB updates actively *prefer* to increase flow through non-important substructures. Empirically, on their SIX6 task, TB failed to reach the target mean even after sampling 800,000 points — more than 12× the size of the entire 65,536-state space.

**Implication for GRAFT:** the plan is right that this is a GFlowNet-shaped problem, and right that ordinary GFlowNets are not enough (§1). But the fix in the literature is *better credit assignment through the flow parametrization* (FL-GFN, guided TB), not the auxiliary-head approach the plan takes. This distinction recurs throughout the review.

**Second documented advantage, more directly useful to GRAFT:** [Robust Scheduling with GFlowNets (ICLR 2023)](https://arxiv.org/abs/2302.05446) found that GFlowNets' key benefit is robustness to a **misspecified proxy metric** — sampling a diverse set of high-quality candidates, then evaluating them with the expensive true metric, beat optimizing the proxy directly. **This is precisely GRAFT's architecture**: neural utility = cheap proxy; deterministic proof/closure checker = expensive true metric. This is the single best citation available for justifying EpiSetFlow's existence, and the plan does not currently use it. See §3.14.

---

## 3.2 §2.3 Canonical set states — **SUPPORTED-BUT-WEAKER-THAN-CLAIMED**

**The plan's claim:** order-invariant canonical set states are innovation #1, removing "arbitrary action-order dependence from the scientific object."

**What the literature says.** This is not new. [GFlowNet Foundations (JMLR 2023)](https://jmlr.org/papers/v24/22-0364.html) explicitly establishes that GFlowNets "naturally handle structured objects like sets and directed acyclic graphs." The whole point of the flow formalism — as opposed to autoregressive modelling — is that a state is a node in a DAG that many trajectories reach, and the flow decomposition sums over them correctly. [FL-GFN (Pan et al., ICML 2023)](https://arxiv.org/html/2302.01687) uses *set generation* as one of its three headline benchmarks (|S|=20/60/80 with universes of 30/80/100), where "agents sequentially build sets by adding elements." Canonical hashing of a selected-atom set is standard implementation practice, not a contribution.

**Where a real problem remains, and the plan does not address it.** [Symmetry-Aware GFlowNets (ICML 2025)](https://arxiv.org/abs/2506.02685) shows that bias arises not from equivalent *states* but from equivalent *actions* — distinct actions leading to the same child. Their Theorem 4.6 gives the forward/backward probability ratio as |Aut(G)|/|Aut(G′)|, and Corollary 5.1 shows the fix is to scale terminal rewards by the automorphism-group size. Uncorrected, vanilla GFlowNets showed L₁ error ≈0.12 on their synthetic graph task vs ≈0.01 with reward scaling; on fragment-based molecule generation, uncorrected sampling produced 5,220 cyclohexane instances per 5,000 samples vs 1,042 corrected.

**Applies to GRAFT if and only if** two distinct edit or evidence atoms can canonicalize to the same child state — e.g. two overlapping source spans that normalize to the same evidence node, or two alias paths that resolve to the same verified link. Given that §10.2 explicitly includes entity resolution and alias handling, this is likely, not hypothetical.

**Recommendation.**
1. Drop "order-invariant set states" from the list of contributions. Present it as correct engineering, citing Foundations and FL-GFN.
2. Add an explicit test: instrument how often two legal atoms produce the same canonical child. If nonzero, apply the SA-GFN reward-scaling correction and cite ICML 2025.
3. If you keep the claim in any form, the honest version is *"we make the set-lattice canonicalization explicit and measure equivalent-action collapse, which prior GFlowNet applications to sets do not."*

---

## 3.3 §3 Dual epistemic mass (F_X and F_A) — **NO EVIDENCE**

**Searched for and did not find:** any GFlowNet variant maintaining two separate flow functions with different transition-admissibility rules over the same DAG. The nearest published relatives are all doing something else:

- **[Multi-Objective GFlowNets (ICML 2023)](https://proceedings.mlr.press/v202/jain23a/jain23a.pdf)** — one flow, conditioned on a preference vector that scalarizes multiple objectives. Not two flows.
- **[Distributional GFlowNets with Quantile Flows (ICLR 2024)](https://arxiv.org/pdf/2302.05793)** — learns a *distribution* over flow values, for stochastic rewards. Not two epistemic channels.
- **[GAFlowNets (ICLR 2023 spotlight)](https://openreview.net/pdf?id=urF_CBK5XC0)** — adds intrinsic-motivation intermediate rewards as *augmented* edge- and state-based flows, and is the closest formal analogue to §8.2's optional `R_X(s)` source term. It provides an asymptotically unbiased treatment. **The plan should cite this; §8.2's `R_X(s)` is otherwise an unjustified injection of flow mass.**

**This is genuinely novel, which cuts both ways.** It is the strongest candidate for the paper's contribution. It is also the component with zero external validation, so it must carry the heaviest experimental burden. The plan's ablation #1 ("one mass instead of dual mass") and success gate #8 ("dual-mass ablation shows a meaningful safety benefit") are correctly specified — that gate is the load-bearing experiment for the entire paper.

**Analysis — an unresolved question in the formalism.** Two flows over one DAG only has a clean interpretation if you can say what the pair *jointly* samples. Three readings are possible and the document does not choose:
- (a) F_A is a GFlowNet on the sub-DAG induced by κ=1 edges, and F_X is a separate GFlowNet on the full DAG. Then they are two independent models and "dual mass" is really "two models plus a shared encoder" — defensible, easy to explain, and easy to ablate.
- (b) They are coupled, and mass *transfers* between channels ("project authoritative mass", §15.1 step 5). Then a conservation law governing the transfer is needed, and none is given.
- (c) F_A is a re-weighting of F_X. Then it isn't a second flow at all.

Reading (a) is the only one I can find precedent for (it is just two conditional GFlowNets). **Recommend committing to (a) explicitly**, and dropping the "transfer/projection" language from §15.1 unless you can write the transfer law down.

---

## 3.4 §3.3 and §8.3 Certification-gated authoritative flow — **SUPPORTED**, with two spec gaps

**Supported.** Constraining a GFlowNet to a feasible sub-DAG by masking illegal actions is standard and published. [Robust Scheduling with GFlowNets (ICLR 2023)](https://arxiv.org/abs/2302.05446) and [Let the Flows Tell (NeurIPS 2023 spotlight)](https://arxiv.org/abs/2305.17010) both construct MDPs where only feasible transitions are available. The plan is correct in §20 not to claim "first constrained GFlowNet."

Multiplying edge flows by κ ∈ {0,1} is mathematically equivalent to deleting those edges from the DAG. Nothing breaks, *provided* the two conditions below hold.

**⚠ SPEC DEFECT 1 — no authoritative source flow.** §8.3 sums over `Pa(s)` for every state. At the initial state s₀ that sum is empty, so the equation forces total authoritative flow to zero unless a source term Z_A is defined. §8.2 has `R_X(s)` for the exploratory channel; §8.3 has no counterpart. **Fix:** define Z_A(c) as a learned per-context authoritative partition function (the standard Z of Trajectory Balance).

**⚠ SPEC DEFECT 2 — reachability under masking is not checked.** If certification zeroes edges such that a valid terminal proof set becomes unreachable in the authoritative sub-DAG — for example, the only construction orders that reach a certified set pass through an intermediate state where some atom is not yet certified — then F_A cannot assign it mass, no matter how good the utility. This is a set-lattice-specific hazard: certification of atom *e* may depend on *other* atoms already in the set (§12 says "sole source span: non-removable until dependent binding is removed", which confirms the dependency exists). **Fix:** either prove the authoritative sub-DAG is closed under the certification predicate (i.e. κ is monotone in set inclusion), or measure the unreachable-terminal rate empirically. This is a cheap, checkable experiment and belongs in the paper.

**Related caution.** [When do GFlowNets learn the right distribution? (ICLR 2025 Spotlight)](https://openreview.net/forum?id=9GsgCUJtic) shows that the accuracy impact of an imbalanced edge scales with the *total flow through it*, and — critically for GRAFT — that "violations to the balance might be associated with the limited expressiveness of the GNN that parameterizes the policy network." Since GRAFT parameterizes its policy with a heterogeneous temporal GNN over a graph with typed relations, **the expressiveness of PC-AGT-GNN is a hard ceiling on which distributions EpiSetFlow can represent at all.** The paper proposes a tractable goodness-of-fit metric (FCS) for exactly this; it is a better diagnostic than "number of distinct proof sets found" and should be adopted.

---

## 3.5 §5.2 and §8.5 Proof-deficit potential Φ — **CONTRADICTED as implemented**

**The plan's claim (§5.2):** "Because this is a potential difference, it provides dense guidance without changing the ranking of terminal utilities when applied consistently."

**The relevant results.**

- In RL, [Ng, Harada & Russell (ICML 1999)](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html) prove that potential-based shaping rewards `F(s,a,s′) = γΦ(s′) − Φ(s)` are exactly the class preserving the optimal policy. **This theorem is about MDP optimal policies. GFlowNets do not optimize an MDP return — they match a distribution. The theorem does not transfer.** Citing it here would be an error.
- The correct GFlowNet analogue exists and is stronger: [FL-GFN (Pan et al., ICML 2023)](https://arxiv.org/html/2302.01687). It assumes an energy ℰ defined on *all* states (Assumption 4.1), defines transition energies ℰ(s→s′) = ℰ(s′) − ℰ(s), and **reparametrizes the flow itself** as `F̃(s) ≝ e^{ℰ(s)}F(s)`, so F̃ depends only on future energy. The detailed-balance constraint becomes `F̃(s)P_F(s′|s) = F̃(s′)P_B(s|s′)e^{−ℰ(s→s′)}`. Their Proposition 4.2 then guarantees that at zero loss, P_F still samples proportionally to the reward — *the same target distribution*, with denser credit. Empirically, the gap over DB/TB/SubTB grows with trajectory length, and an FL-GFN trained **only on partial trajectories with no terminal rewards** performed close to one trained on complete trajectories.

**The gap.** GRAFT's §8.5 does *not* do this. It trains an auxiliary head `ĝ_θ(s,a,s′)` to regress `Φ(s′) − Φ(s)` as a separate squared-error term added to the total loss. Φ never enters the flow parametrization. Consequently:

1. **The invariance claim in §5.2 is unsupported.** An auxiliary regression head added to a composite loss changes the gradient field of the shared encoder. There is no theorem saying the terminal ranking survives.
2. **The credit-assignment benefit is not obtained.** FL-GFN's speedup comes from the energy appearing *inside the balance constraint*. A head that predicts Φ-differences but doesn't feed them into the flow gives the flow no extra signal.

**Recommendation — this is the highest-value single change in the plan.** Adopt the FL-GFN reparametrization directly: set ℰ(s) = −Φ(s) (the plan's Φ is already a state-defined potential with obligation terms and size penalties, which satisfies Assumption 4.1's "state-space extension" case), and use the FL-DB objective. You then get:
- a published theorem that the target distribution is unchanged (Prop. 4.2),
- dense per-transition credit, which is what §5.2 wanted,
- the ability to train on incomplete trajectories, which matters for §13's delta training,
- a much cleaner story: *"we instantiate forward-looking GFlowNets with a proof-obligation energy"* — a real, defensible, citable contribution.

Keep `L_deficit` as an *auxiliary interpretability head* if you want it, but stop claiming invariance for it.

---

## 3.6 §6 Counterfactual evidence credit — **SUPPORTED in principle, CONDITIONAL on cost**

**Supported.** Leave-one-out ablation is the reference standard for context attribution. [ContextCite (NeurIPS 2024)](https://arxiv.org/abs/2409.00729) uses it as the oracle for its top-1 metric and defines it exactly as the plan's N(e): ablate a source, measure the log-probability drop. Using per-source necessity as a training signal for what to include is therefore well-grounded, and the plan's §6.4 list of removal types (source span, entity link, temporal edge, conflict evidence, closure member, exploratory bridge) is a genuinely nice refinement — ContextCite ablates undifferentiated text chunks; GRAFT can ablate *typed* structure.

**Cost is the binding constraint.** [AttriBoT (arXiv 2411.15102)](https://arxiv.org/html/2411.15102v1) quantifies it: exact LOO needs **|C|+1 forward passes**, ≈ `2PT|C|(|C|−1)` FLOPs, and for long contexts "can be orders of magnitude more expensive than generating the response itself." ContextCite reports typical contexts of 32–166 sources.

The plan's §17 already flags this (O(H) proof checks per terminal set) and proposes batched LOO, dependency-aware removal, incremental recomputation, and uncertainty-based sampling. **Every one of those has a published counterpart, and citing them turns a hand-wave into a method:**

| Plan's mitigation (§17) | Published counterpart |
|---|---|
| batched leave-one-out | KV-caching of shared prefixes — **~1.6×** (AttriBoT) |
| dependency-aware removal | hierarchical attribution (coarse groups → refine) — **~H×** (AttriBoT) |
| sampling only high-uncertainty atoms | sparse linear surrogate from **32 ablations** even for a 98-source context (ContextCite) |
| — | proxy models, R = 0.88–0.94 correlation with target attributions (AttriBoT) |

AttriBoT reports **>300× combined speedup** while staying more faithful to target-model LOO than prior attribution methods; on HotpotQA, **90% recall keeping only half the sources**.

**Critical branch point the plan leaves open.** §6.1 says S is "a deterministic or fixed-model proof score." These are completely different cost regimes:
- **Deterministic checker** → each LOO is a cheap symbolic re-check, H=16 removals × M=4 sets = 64 checks/query is trivially affordable, and the AttriBoT machinery is unnecessary.
- **Fixed-model score** → you are in the AttriBoT regime and need every trick above.

**Decide this in the paper.** I recommend deterministic: it is cheaper, it makes N(e) exactly reproducible, and it aligns with the plan's own "no neural authority promotion" principle (§3.4).

**Documented limitation to acknowledge.** ContextCite's linear surrogate assumes sources interact roughly additively and explicitly warns it "may fail when interactions between sources are significant." **Proof subgraphs are the case where interactions are large by construction** — an entity link is worthless without the source span it binds. So GRAFT should expect single-atom LOO to systematically under-credit atoms in mutually-dependent pairs. The plan's §12 already notices this ("sole source span: non-removable until dependent binding is removed"); make it a stated limitation, and consider *pairwise* removal for known dependency pairs.

**⚠ ANALYSIS — the redundancy term D(e) is close to vacuous as defined.** §6.1 gives `N(e) = max(0, S(X) − S(X\e))` and §6.2 gives `D(e) = max(0, S(X\e) − S(X) + ε)`. If the proof score S is monotone non-decreasing in evidence (removing evidence never *improves* a proof score — which is what "proof score" implies), then `S(X\e) − S(X) ≤ 0` always, so `D(e) ∈ [0, ε]`, and D is essentially the constant ε wherever N(e) = 0 and exactly 0 wherever N(e) > 0. The term `−βD(e)` in §6.3 therefore contributes an almost-constant shift rather than a redundancy signal. The document half-notices this ("or use zero necessity plus semantic overlap as a redundancy indicator"). **Take the alternative:** define redundancy from *set coverage overlap*, which is what the submodular-selection literature does (see §4.4) — e.g. marginal gain of e given the rest of X under a facility-location/coverage function. That gives a real, non-degenerate redundancy signal and connects directly to the minimality objective.

---

## 3.7 §7 Closure residual — **NO EVIDENCE**, and the underlying task is known-hard

I found no published work on decomposed residual prediction for aggregate-query closure over a conversational graph. The idea — replace a binary "closed/not-closed" label with a six-component residual vector for denser supervision — is a reasonable generalization of the multi-head auxiliary-supervision pattern, but it is unvalidated.

**What is documented is that the underlying task is hard.** Complex/aggregate query answering over *incomplete* knowledge graphs is a mature subfield with known limitations: the standard ranking objective is computationally infeasible for queries with multiple free variables (exponential in the number of free variables), benchmarks have been restricted to tree-like or single-cycle queries, and many neural CQA models are not probabilistically calibrated ([One Model, Any Conjunctive Query, arXiv 2409.13959](https://arxiv.org/html/2409.13959)). A recent systematic comparison found no neural CQA model consistently beats a training-free path-counting relaxation on count queries.

**Implication.** The plan's insistence (§7) that `‖r_C(s)‖₁ = 0` is *necessary but not sufficient*, with a deterministic certificate still required, is the correct and defensible design — it sidesteps the calibration problem that sinks neural CQA. Keep that framing prominently. But treat "residual heads give stronger supervision than a binary label" as a **hypothesis to be ablated** (the plan's ablation #5 does this correctly), not as an established result.

---

## 3.8 §8.2 / §8.3 Flow-Matching-style conservation — **CONTRADICTED by the objective literature**

The plan writes conservation as a per-state balance over `Pa(s)` and `Ch(s)`. That is **Flow Matching (FM)**, the original 2021 objective. The entire arc of GFlowNet training research since then has been away from it:

| Objective | Finding |
|---|---|
| FM, DB | [Trajectory Balance (Malkin et al., NeurIPS 2022)](https://arxiv.org/abs/2201.13259): flow matching and detailed balance are "analogous to temporal difference learning" and are "prone to inefficient credit propagation across long action sequences." TB improves convergence, diversity, and robustness to long sequences and large action spaces. |
| TB | High gradient variance. [SubTB(λ) (Madan et al., ICML 2023)](https://arxiv.org/abs/2209.12782) frames DB and TB as "opposite ends of a gradient bias-variance tradeoff" and interpolates, giving faster convergence and enabling longer action sequences with sparser rewards. |
| SubTB | [FL-GFN (Pan et al., ICML 2023)](https://arxiv.org/html/2302.01687) beats DB, TB, *and* SubTB on set generation, bit sequences and molecules, with the gap widening at larger scale and longer trajectories. |
| TB (again) | [Shen et al. (ICML 2023)](https://arxiv.org/abs/2305.07170): even TB systematically undersamples the target mean; their Sub+PRT+SSR combination found **867 modes vs. 140 for TB baseline** by round 10,000 on sEH, and reached the target mean **9× faster**. |

**GRAFT's setting is the worst case for FM:** H = 16 atoms means trajectories of length 16, and 16! construction orders per terminal set. Also, enumerating `Ch(s)` for the FM loss costs O(K) per state where K is the candidate-atom count per step — the plan's own complexity analysis (§17) assumes K is large enough to be worth caching.

**Recommendation.** Rewrite §8.2/§8.3 in **SubTB(λ) or FL-DB form**, not FM form. Concretely, FL-DB is the natural fit because it composes with the Φ recommendation in §3.5 above, and because FL-GFN explicitly supports **incomplete trajectories** — which §13's delta training needs anyway. This is one change that fixes three sections at once.

---

## 3.9 ⚠ SPEC DEFECT 3 — the objective is never anchored to the terminal utility

**This is the most serious finding in the review.**

Read §8.2 through §8.7 and §9 together. §9 defines terminal utilities `U_Q` and `U_W`. §8.2 and §8.3 define conservation equations containing a terminating flow `F^stop(s)`. **No equation anywhere states that `F^stop(s)` equals (a monotone transform of) `U`.** §8.7's final objective has six terms; none of them contains U.

In Flow Matching as originally defined, and in the flow-network formalism of [GFlowNet Foundations (JMLR 2023)](https://jmlr.org/papers/v24/22-0364.html), the terminating flow at a terminal state **must equal the reward**. That boundary condition is the *only* thing that makes the trained sampler reward-proportional. Without it:

- The conservation losses `L_X` and `L_A` in §8.4 are minimized by *any* internally consistent flow assignment, including the trivial one where all flows are equal.
- The trained policy has no reason to prefer high-utility proof sets.
- The claim in §0 that the system distributes "probability mass across multiple high-quality compositional objects" does not follow from the stated objective.

I read this as an **omission** rather than a design intent — §9's existence implies U was meant to be the reward. But as written the objective is unanchored, and any reviewer who works through the math will find it.

**Fix:** state the boundary condition explicitly, e.g. `F^stop(s) = R(s) = exp(β · U(s))` for terminal s, with `R(s) = 0` for infeasible states (§9.3). Then define how β is chosen (see §3.11 on temperature) and how `U = −∞` / infeasible is handled numerically. If you adopt TB or SubTB per §3.8, this appears naturally as the log-reward term in the balance equation and the defect disappears.

---

## 3.10 §8.4 Squared log-residual loss — **CONDITIONAL**

The `[log(ε + F_in) − log(ε + F_out)]²` form is standard practice. But [Beyond Squared Error (Hu et al., arXiv 2410.02596)](https://arxiv.org/abs/2410.02596) shows the choice has been made without theoretical justification: different regression losses correspond to different divergence measures with distinct **zero-forcing** (exploitation, higher reward) versus **zero-avoiding** (exploration, diversity) behaviour. They propose Shifted-Cosh, Linex(1/2) and Linex(1), reporting improvements in convergence speed, diversity and robustness on hyper-grid, bit-sequence and molecule tasks.

**Why this matters specifically for GRAFT.** The plan wants *different* behaviour from its two channels — exploratory mass should be zero-avoiding (broad candidate coverage), authoritative mass should be zero-forcing (concentrate on certified, minimal proofs). Beyond Squared Error gives you a principled, published way to instantiate that asymmetry with **two different loss functions on the two channels**. That would turn "dual mass" from an unmotivated architectural split into a mechanism with a stated inductive bias.

Also note the ε-inside-log form interacts badly with sparse rewards: near-zero flows make the log highly sensitive and destabilize gradients — a documented GFlowNet failure mode in sparse-reward settings. Given §9.3 makes many terminal states infeasible (reward 0), GRAFT will be in exactly that regime.

---

## 3.11 §8.7 The six-term composite objective — **NO EVIDENCE**

`L = L_X + λ_A L_A + λ_D L_deficit + λ_cf L_cf + λ_route L_route + λ_cal L_cal`.

The document is candid: "This is not standard Subtrajectory Balance copied unchanged." True — but that is a cost, not a feature. **Every published guarantee cited in §23 is a guarantee about a single objective at zero loss.** TB's guarantee (Malkin et al.), FL-DB's Prop. 4.2, the DB/FM conditions in Foundations — all are statements of the form "if *this* loss is zero everywhere, the policy samples proportional to reward." None survives being one of six weighted terms whose weights trade off against each other.

**Consequences to state honestly in the paper:**
- EpiSetFlow has **no reward-proportionality guarantee**. The plan's §20 already says "avoid claiming mathematically equivalent reward-proportional sampling unless proved" — good. Extend that to a positive statement: *"EpiSetFlow is a heuristic composite objective; we do not claim reward-proportional sampling and we measure distributional fit empirically."*
- The right empirical instrument is the [ICLR 2025 Spotlight FCS metric](https://openreview.net/forum?id=9GsgCUJtic), or the Anderson–Darling / target-mean diagnostics of [Shen et al.](https://arxiv.org/abs/2305.07170), not "number of distinct proof sets."
- Six λ hyperparameters over an already-expensive pipeline is a real tuning-budget risk. **Recommend a staged ablation that adds one term at a time**, so the paper can report which terms actually pay for themselves. Right now §18's ablation list is structured around *features* (dual mass, canonical states…), not around *loss terms* — add the latter.

---

## 3.12 §9.3 Feasibility-first infeasible utility — **CONDITIONAL**

Setting utility to infeasible on scope leakage, deleted evidence, hypothesis-only bindings, uncertified closure, invalid retirement, or missing grounding is exactly right for a reliability-critical system, and it is the design that makes the "authorize narrowly" thesis real.

**The documented cost:** this manufactures a sparse-reward landscape, and sparse rewards are a known GFlowNet weak point. In sparse settings the vast majority of trajectories carry near-zero reward, the model assigns near-zero flow, and the log in the loss becomes highly sensitive — producing large losses and gradient instability. [SubTB(λ)](https://arxiv.org/abs/2209.12782) was motivated in part by extending GFlowNets to "longer action sequences and sparser reward landscapes." [GAFlowNets (ICLR 2023)](https://openreview.net/pdf?id=urF_CBK5XC0) exist specifically because "GFlowNets only learn from rewards of the terminal states, which can limit its applicability" in sparse-reward tasks.

**Mitigations with published support, in priority order:**
1. **FL-DB with the Φ energy** (§3.5) — turns the sparse terminal signal into dense per-transition signal. Directly addresses the cause.
2. **Prioritized replay** — [Shen et al.](https://arxiv.org/abs/2305.07170) form each replay batch with α% from the top β percentile of reward (they use α=50%, β=90%). Cheap, and it maps onto the plan's §14 Stage 5 mixture, which currently has *no* citation. Cite this.
3. **Local search** — [Local Search GFlowNets (ICLR 2024 spotlight)](https://openreview.net/forum?id=6cFcw1Rxww) backtrack-and-reconstruct around high-reward samples, addressing over-exploration of wide sample spaces. A natural fit for GRAFT: backtrack from a valid proof set and reconstruct to find a *smaller* valid one.
4. **Pessimistic backward policy** — [PBP-GFN (NeurIPS 2024)](https://arxiv.org/abs/2405.16012) targets under-exploitation of high-reward objects when training trajectories are few; validated on eight benchmarks including *structured set generation*, which is GRAFT's exact object type.

---

## 3.13 §12 Deterministic backward policy — **SUPPORTED at the optimum, CONDITIONAL in practice**

**The good news.** In the GFlowNet formalism the target distribution is realizable for *any* fixed backward policy; P_B determines *which* flow function realizes it, not *whether* one exists. So a hand-designed removable-atom distribution is not a correctness risk. It is also genuinely more interpretable — the plan's weights (redundant evidence high, exploratory edge medium, necessary anchor low, sole source span non-removable) are a readable prior, and reducing parameters is a real benefit.

**The caution.** An entire research line exists because P_B materially affects *learning efficiency*: [PBP-GFN (NeurIPS 2024)](https://arxiv.org/abs/2405.16012) and [Optimizing Backward Policies via Trajectory Likelihood Maximization (ICLR 2025)](https://arxiv.org/pdf/2410.15474) both show measurable gains from choosing P_B well. A fixed heuristic P_B is a bet that your prior is good enough. The plan is right that "a learned backward-policy ablation remains required" (§12, ablations #8/#9).

**One design coupling worth noting.** The removal weights depend on N(e) (redundancy → high removal weight; necessary anchor → low). So P_B depends on the counterfactual scores from §6, which are computed on terminal sets. **Make the ordering explicit**: either P_B uses a *cached* necessity estimate from a previous epoch, or it uses the auxiliary inclusion head p_θ(e ∈ X). The document does not say which, and the difference affects whether P_B is stationary during training.

---

## 3.14 §11.4 Sample-then-filter — **SUPPORTED, and this is your best argument**

§11.4 samples one greedy plus several diverse high-mass proof sets, then selects by valid → minimal → sufficient → low-risk, using the deterministic checker.

[Robust Scheduling with GFlowNets (ICLR 2023)](https://arxiv.org/abs/2302.05446) is the direct precedent and the strongest available justification for using a GFlowNet at all here. Their finding: evaluating candidates on the true target is expensive, so prior methods optimize a fast proxy and get bad results when tested for real; GFlowNets' diverse high-quality candidate sets proved **robust against misspecified proxies**, "confirming the hypothesis that a diverse set of high-quality candidates is essential for robustness."

Map that onto GRAFT: the neural utility U is the cheap proxy; the deterministic proof/closure checker is the expensive true metric. **A reward-maximizing learner (PPO/GRPO) optimizes the proxy and fails when the checker disagrees; a flow learner produces a diverse portfolio, of which at least one member passes.** This is a clean, falsifiable, published-precedent argument — and the plan currently doesn't make it. It should be the framing of the paper's introduction.

Supporting evidence from the LLM side: [FlowRL (ICLR 2026)](https://arxiv.org/abs/2509.15207) makes exactly this argument in the LLM-reasoning setting — that PPO and GRPO "tend to over-optimize dominant reward signals while neglecting less frequent but valid reasoning paths, thus reducing diversity" — and reports **+10.0% over GRPO and +5.1% over PPO on math benchmarks**, with consistent gains on code reasoning. [Flow of Reasoning (ICML 2025)](https://arxiv.org/abs/2406.05673) makes the same case with 15 training examples.

**Caveat to state:** neither FlowRL nor FoR is a memory/graph system, so they support the *mechanism* (distribution matching beats reward maximization for diverse valid solutions), not the *application*. Say so.

---

## 3.15 §13 Incremental delta training — **NO EVIDENCE**

Freezing unaffected encodings, updating only local GNN state, reusing replay states by dependency hash, and invalidating only states touching changed lifecycle/deletion data — I found no published GFlowNet work on incremental retraining under a changing state space.

The nearest published work is [Generalization and Distributed Learning of GFlowNets (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/000eba875068854d5ff003b1fa534cd6-Abstract-Conference.html), which gives the **first data-dependent generalization bounds for GFlowNets** and shows (via PAC-Bayesian inequalities) that **generalization degrades as state-space size grows**. Their Subgraph Asynchronous Learning (SAL) trains on smaller subnetwork components and aggregates, outperforming centralized training on mode coverage and distribution matching.

**Two consequences.**
1. **A supporting result you should cite:** bounded active graphs (§15.2 step 3) and delta-local training aren't just efficiency tricks — smaller state spaces are *provably* better for GFlowNet generalization. This is real theoretical backing for §13's premise, and the plan is missing it.
2. **An unaddressed risk:** flows learned on stale neighbourhoods. If the graph changes and you freeze unaffected encodings, the conservation constraints that held before may no longer hold, because a state's set of legal children can change when the graph changes (new candidate atoms appear). The plan invalidates states "touching changed lifecycle or deletion data" — but a *new* verified link creates new legal actions at states that touch nothing deleted. **Recommend:** invalidate on any change to the legal-action set of a cached state, not just on lifecycle/deletion changes, and report the measured invalidation rate.

---

# Part B — The systems core

## 4.1 Graph memory vs. flat memory — **CONTRADICTED for single/multi-hop, SUPPORTED for temporal**

This is the most important system-level finding, because GRAFT is a graph-memory system and the best available head-to-head study says graph memory is *not* uniformly better.

[Mem0 (ECAI 2025)](https://arxiv.org/html/2504.19413v1) ran what it describes as the first broad head-to-head comparison of memory approaches on LoCoMo. Comparing its own flat variant (Mem0) against its graph variant (Mem0ᵍ), LLM-judge scores:

| Question type | Mem0 (flat) | Mem0ᵍ (graph) | Graph helps? |
|---|---|---|---|
| Single-hop | **67.13** | 65.71 | ✗ −1.4 |
| Multi-hop | **51.15** | 47.19 | ✗ −4.0 |
| Temporal | 55.51 | **58.13** | ✓ +2.6 |
| Open-domain | 72.93 | **75.71** | ✓ +2.8 |
| **Overall** | 66.88 | **68.44** | ✓ +1.6 |

Their own reading: "the addition of graph memory in Mem0ᵍ does not provide performance gains" for multi-hop, "indicating potential inefficiencies or redundancies in structured graph representations"; and "relational structure provides limited utility when retrieval target occupies single turn."

**Cost of the graph:** ~14,000 tokens per conversation vs ~7,000 for flat (2×); search latency p50 0.476s vs 0.148s (3.2×); total p95 2.590s vs 1.440s.

**And the uncomfortable number:** full-context (no memory system at all) scored **72.90** overall — higher than both Mem0 variants (66.88 / 68.44) and higher than Zep (65.99). Memory systems in that study won on *cost and latency*, not on accuracy.

**What GRAFT must do with this.**
- **Do not claim graph memory is better per se.** It isn't, on that evidence.
- **Aim the claim where the evidence points:** temporal reasoning, lifecycle/supersession, conflict, scope, and *attribution* — which is exactly what GRAFT's typed temporal graph and proof obligations are built for, and exactly where flat memory is weakest. This is a well-supported positioning.
- **Report full-context as a baseline and expect to lose to it on raw accuracy** on LoCoMo-scale conversations. Win on latency, token cost, verifiability, and abstention correctness instead — and say so up front.

[Zep (arXiv preprint)](https://arxiv.org/pdf/2501.13956) points the same way. On LongMemEval with gpt-4o it reports 71.2% vs 60.2% full-context at 2.58s vs 28.9s and 1.6k vs 115k context tokens — but the per-type breakdown shows it *loses* to full-context on single-session-assistant (80.4% vs 94.6%) and wins hugely on temporal-reasoning (62.4% vs 45.1%) and preference (56.7% vs 20.0%). **Same shape: structure wins on temporal/preference, loses on direct recall.**

## 4.2 Bi-temporal modelling and invalidation — **SUPPORTED**

GRAFT's lifecycle checker, supersession, and "deleted evidence must not be used" rule have a direct published analogue in Zep/Graphiti's four-timestamp model (t_created, t_expired, t_valid, t_invalid) with **edge invalidation rather than deletion**, preserving history while excluding stale facts from retrieval. The temporal-reasoning gains above (+38–48% relative) are the associated evidence.

Zep's own stated limitations are ones GRAFT inherits and should pre-empt: entity resolution in unstructured dialogue is hard; temporal extraction depends on LLM reasoning and fails on implicit/relative time references; community detection and traversal overhead grow with graph size; and extraction errors propagate through the graph. GRAFT's deterministic checkers address the *use* of bad facts but not their *creation* — §10.2's checkers are the mitigation, and their precision/recall must be measured and reported separately, not folded into end-task accuracy.

## 4.3 Single-shot GNN retrieval vs. iterative — **SUPPORTED**

GRAFT's design (bounded active graph, GNN encoding, then bounded set construction) sits on the right side of a well-evidenced trade-off.

- [GFM-RAG (NeurIPS 2025)](https://arxiv.org/html/2502.01113v2): an **8M-parameter, 6-layer query-dependent GNN** performs multi-hop reasoning in a single inference pass — Recall@5 of 87.1 (HotpotQA), 58.2 (MuSiQue), 95.6 (2Wiki) vs 83.0 / 57.6 / 93.9 for IRCoT+HippoRAG, at **0.107s vs 3.162s** retrieval latency. 18.9% average improvement over HippoRAG across seven unseen datasets without fine-tuning.
- [HippoRAG (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf): single-step retrieval comparable to or better than iterative IRCoT while **10–20× cheaper and 6–13× faster**.
- [GNN-RAG (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.856/): GNN retrieval beats LLM-based retrieval by **8.9–15.5 F1 points** on multi-hop and multi-entity questions, using **9× fewer KG tokens** than long-context inference, and matches GPT-4 with a 7B tuned LLM.

**And the counter-evidence on iterative retrieval, which supports GRAFT's bounded design.** [Beyond Static Retrieval (arXiv 2509.25530)](https://arxiv.org/html/2509.25530v1) evaluates iterative retrieval across HippoRAG2, RAPTOR, GFM-RAG and GraphRAG on HotpotQA/2Wiki/MuSiQue/PopQA. Findings:
- Iteration helps most on **bridge-style multi-hop** questions.
- It gives "little to no benefit on simple Comparison questions, and can even lead to slight performance drops due to over-retrieval and noise accumulation."
- **Two iterations is the cost-benefit optimum; 3+ shows diminishing returns.**
- Single-hop (PopQA) barely moves: 0.419 → 0.435 EM.
- The real bottleneck is **ranking, not coverage**: gold recall is ~95% at K=100, but bridge documents stay buried below top-10.

**Reading for GRAFT.** `max_graph_hops: 5` and `stochastic_sets: 3` are in the right ballpark, but the plan should (a) justify 5 hops empirically rather than by fiat, and (b) recognize that its real advantage over one-shot GNN retrieval must come from the *proof obligations* driving expansion toward missing evidence — i.e. `h_deficit` doing the work of "bridge-document ranking." That's a testable, differentiating claim and worth stating.

## 4.4 Minimal sufficient proof sets — **CONDITIONAL, with a documented reversal**

§11.4 selects the minimal sufficient proof; §9.1 penalizes |V_Q| and |E_Q|; the plan lists "submodular proof selection" as a search baseline. The best evidence on exactly this is [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1) (preprint — see §1).

Its packer maximizes a weighted sum of relevance + query coverage + saturated facility-location representativeness + concave-over-documents diversity, under a token budget, via **cost-scaled greedy with Lin–Bilmes singleton fallback** — the standard constant-factor template for budgeted monotone submodular maximization. That is exactly the algorithm GRAFT would use.

**Headline result (HotpotQA, 3B reader, 160-token budget, 3 seeds):**

| Policy | F1 | EM | Tokens |
|---|---|---|---|
| naive packed | 0.400 | 0.306 | 151.1 |
| MMR (λ=0.7) | 0.410 | 0.318 | 151.7 |
| focused heuristic | 0.429 | 0.331 | 152.1 |
| **submodular** | **0.451** | **0.359** | **145.5** |
| oracle | 0.601 | 0.487 | 141.7 |

**But the paper's central contribution is the boundary conditions.** All four must hold for submodular packing to beat the simpler heuristic:
1. **Multi-hop complementary structure** — fails on single-pass tasks (RAGBench CovidQA: −0.010 F1, p=0.30).
2. **Retrieval actually surfaces the evidence** — fails on MuSiQue (all-gold@5 = 0.184; benefit +0.011, p=0.34), and tripling retrieval depth didn't help.
3. **Binding but not extreme budget** — inverted-U peaking near 160 tokens: at B=96, Δ=+0.004 (p=0.81); at B=160, +0.035 (p=0.04); at B=224, +0.017 (p=0.26).
4. **Reader is the bottleneck** — and this one **reverses**: 3B reader +0.022 (p<0.05); 7B −0.010 (p=0.45); **14B −0.029 (p=0.013)**.

**Direct implications for GRAFT.**
- GRAFT uses a **frozen small language model** (§15.2 step 9). That is condition 4 satisfied — the small reader is where minimality pays. **This is a genuine, evidence-backed reason for GRAFT's SLM choice, and the plan should make it explicitly** rather than treating the SLM as merely a cost decision.
- Corollary and warning: **if you later swap in a larger verbalizer, the minimality benefit may vanish or invert.** State this as a scope condition.
- The paper also introduces **answer-in-context** (does the gold answer survive into the packed context?) as a diagnostic that predicts F1 far better than recall — ΔR² = +0.17 over recall; F1 of 0.61 vs 0.20 conditional on it; and among *retrieval-perfect* questions (all gold in top-5), F1 was still 0.61 vs 0.20. **GRAFT should adopt an analogous metric — "does the certified answer binding survive into the final proof set?" — as a first-class intermediate metric.** It isolates construction quality from checker quality and from SLM quality, which the plan currently cannot do.
- Complementary support for why packing matters at all: [Lost in the Middle (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/) shows a U-shaped position curve — models use evidence at the beginning and end of context far better than in the middle. Small, ordered proof sets sidestep this.

## 4.5 Proof → frozen SLM → citations — **SUPPORTED**

- [ALCE (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.398/) established automatic citation evaluation (fluency, correctness, citation quality via NLI) and the headline failure: on ELI5, **even the best models lack complete citation support 50% of the time.** This is the number that justifies GRAFT's entire "proof-carrying" premise, and should be quoted in the introduction.
- [VeriCite (SIGIR-AP 2025)](https://arxiv.org/html/2510.11394v1) validates the deterministic-verification design directly. Its three-stage pipeline (generate → NLI-verify each statement against cited passages, discard unsupported → select evidence → refine) improved citation F1 by an average of **11.41%** across five LLMs. Its ablation is the key evidence: removing NLI verification drops citation F1 from **77.73 → 68.91** while barely touching answer correctness (41.63 → 41.59) on ASQA/Llama3-8B. **Verification is what buys citation quality — that is GRAFT's proof-checker thesis, independently confirmed.** They also found an NLI model (TRUE) beat both an 8B LLM verifier (73.01 citation F1) and DeepSeek-R1 (79.22) on the cost-effectiveness trade-off — supporting GRAFT's choice of cheap deterministic checkers over LLM judges.
- **Two VeriCite limitations GRAFT inherits.** (i) On ELI5, VeriCite *underperformed* baselines on answer correctness across all five models — verification-first pipelines can cost you on non-factoid, open-ended questions. GRAFT's `open-domain` route needs a plan for this. (ii) The authors state their evidence-selection stage "may be suboptimal for multi-hop scenarios requiring cross-passage information integration" — per-passage independent verification struggles when a claim is supported only *jointly*. GRAFT's connected proof subgraph is a plausible fix and would be a legitimate contribution, but it must be evaluated against VeriCite on multi-hop, not just asserted.
- [Self-RAG (ICLR 2024)](https://selfrag.github.io/) supports the adaptive-retrieval and critique-token pattern that GRAFT's abstention certificate (§5.3) resembles, and reports that retrieving *less* costs 40% relative on PopQA but only ~2% on PubHealth — i.e. abstention/retrieval policy is highly task-dependent. GRAFT's abstention gate must be evaluated per route.

## 4.6 Latency — **CONDITIONAL, currently plausible**

Published interactive-memory latencies to beat:

| System | Search p50 | Total p95 |
|---|---|---|
| Mem0 (flat) | 0.148s | 1.44s |
| Mem0ᵍ (graph) | 0.476s | 2.59s |
| Zep | 0.513s | 2.93s |
| RAG (k=1) | 0.251s | 1.63s |
| Full-context | — | 17.1s |
| LangMem | 17.99s | 60.4s |

GRAFT's §15.3 budget (1 greedy + 3 stochastic sets, ≤16 atoms, ≤5 hops) plus deterministic checking plus (optionally) leave-one-out counterfactual scoring must land under ~2.6s p95 to be competitive with graph-memory peers, and under ~1.4s to match flat memory. **This is achievable but tight**, and it is why the deterministic-vs-model choice for S (§3.6) matters so much.

Note the plan's §17 instruction to "report empirical latency and checker cost separately" is exactly right and should be non-negotiable — Mem0's paper singles out LangMem's 18s p50 as rendering it "impractical for interactive applications," and reports that Zep "often failed to answer queries correctly" immediately after writes, with results improving hours later, attributing this to asynchronous background processing. **GRAFT's atomic-commit design (§10.3) directly avoids that failure mode, and that is a concrete, citable systems advantage worth claiming.**

## 4.7 GNN depth / `max_graph_hops: 5` — **CONDITIONAL**

Deep message-passing GNNs suffer over-smoothing: node features converge toward a constant as depth grows, and message-passing GNN performance is widely reported to deteriorate after a few layers. Against that, [GFM-RAG](https://arxiv.org/html/2502.01113v2) successfully uses **6 layers** of query-dependent message passing for multi-hop retrieval, and query conditioning is likely why — relation embeddings and entity representations update *per query*, which resists the collapse.

**Reading:** 5 hops is defensible **if** GRAFT's PC-AGT-GNN is query/context-conditioned in the GFM-RAG sense (§4's `h_q` suggests it is). Say so explicitly and cite GFM-RAG. Add a depth ablation (2/3/5 hops) — it is cheap and pre-empts an obvious reviewer question. Do not present 5 as a tuned constant without that.

## 4.8 Upstream KG quality — the risk the plan under-weights

Every downstream guarantee in GRAFT is conditional on the graph being right, and the plan's §10.2 checkers only validate *proposed* atoms against sources — they cannot recover facts that extraction never produced. The literature is consistent that this is the dominant error source: KGs are "frequently incomplete due to information loss during construction and the difficulties in extracting all relevant triples, especially in noisy or complex scenarios" ([IJCAI 2025](https://www.ijcai.org/proceedings/2025/0901.pdf)); Zep names extraction fidelity and entity resolution among its top limitations; GFM-RAG names KG-index quality (entity extraction and resolution) as its main dependency.

**Recommendation:** report **recall of the certification pipeline** (what fraction of genuinely-supported claims get certified) alongside precision. A system that certifies almost nothing has perfect precision and is useless. The plan's §19 gates measure "reduces unsupported or wrong-entity answers" (precision-side) and "does not worsen false closure" — but nothing guards against over-abstention. **Add a gate: abstention rate must not increase on questions that are answerable from certified evidence.**

---

# Part C — Baselines

## 5.1 Learning baselines — audit of §18

The plan lists: supervised only; canonical set imitation; PPO; GRPO; progressive RL; step-wise GRPO; standard conditional GFlowNet; TB; SubTB; EpiSetFlow.

**Good.** PPO/GRPO are the right reward-maximizing comparators, and their inclusion is what makes success gate #3 meaningful. Precedent for the comparison in the memory setting exists: [Memory-R1 (ACL 2026)](https://arxiv.org/abs/2508.19828) trains a Memory Manager (ADD/UPDATE/DELETE/NOOP) and an Answer Agent with outcome-driven PPO **and** GRPO, from **152 training QA pairs**, reporting on LLaMA-3.1-8B vs Mem0: F1 30.41 → 45.02, BLEU-1 22.22 → 37.51, LLM-judge 45.68 → 62.74. Its ablation shows memory distillation (filtering 60 retrieved candidates before answering) lifts F1 40.95 → 45.02, and that GRPO converges faster early while both reach comparable final performance.

**Missing, and each is a likely reviewer request:**

| Missing baseline | Why it's needed | Citation |
|---|---|---|
| **FL-GFN / FL-DB** | The closest published method to what GRAFT's §5.2 is *trying* to do. If FL-DB alone matches EpiSetFlow, the composite objective is unjustified. **This is the most important missing baseline.** | [Pan et al., ICML 2023](https://arxiv.org/html/2302.01687) |
| **Guided TB + prioritized replay + SSR** | The published "improved GFlowNet training" package; up to 10× sample efficiency, 6× more modes. Without it, "we beat standard GFlowNet" is a weak claim. | [Shen et al., ICML 2023](https://arxiv.org/abs/2305.07170) |
| **FlowRL** | Distribution matching *without* GRAFT's task-specific machinery; the honest "flow-matching but generic" control. | [ICLR 2026](https://arxiv.org/abs/2509.15207) |
| **Memory-R1 (PPO and GRPO variants)** | The published RL-for-memory system. Directly comparable on the write side. | [ACL 2026](https://arxiv.org/abs/2508.19828) |
| **Local Search GFlowNet** | Backtrack-and-reconstruct; the natural competitor for finding *minimal* valid proof sets. | [ICLR 2024 spotlight](https://openreview.net/forum?id=6cFcw1Rxww) |
| **PBP-GFN** | Validated on structured set generation — GRAFT's exact object type. Also the right comparator for ablation #9. | [NeurIPS 2024](https://arxiv.org/abs/2405.16012) |
| **Order-Preserving GFN (OP-GFN)** | Removes the reward-temperature/exponent tuning problem that GRAFT will hit (§3.10, §3.12), by training from *rankings*. Given that GRAFT's utility is a hand-weighted sum of 5–7 terms, an order-only objective is a serious alternative, not a strawman. | [ICLR 2024](https://arxiv.org/abs/2310.00386) |
| **COFlowNet** | If any part of training is offline (§14 Stages 3–4 use gold/synthetic sets), the offline-GFlowNet failure mode applies: limited data → insufficient state-space exploration → poor OOD candidates. COFlowNet's unsupported-flow restriction is the published remedy. | [ICLR 2025](https://iclr.cc/virtual/2025/poster/28047) |
| **Search-R1-style RL retrieval** | GRPO-trained interleaved reason-and-retrieve with outcome-only rewards; the current strong baseline for learned retrieval policies. | [arXiv 2503.09516](https://arxiv.org/pdf/2503.09516) |

**Also missing: the ablation that decides the paper.** The plan's §18 ablation list is organized by *feature*. Add a **loss-term ladder**: `L_X` alone → `+L_A` → `+L_deficit` → `+L_cf` → `+L_route` → `+L_cal`. With six λ's, a reviewer will ask which terms pay for themselves, and "we ablated dual mass" does not answer that.

## 5.2 Search baselines — audit of §18

The plan lists: greedy; beam search; diverse beam; MCTS; submodular proof selection; learned A\* traversal.

**This list is well-chosen.** Notes and additions:

| Baseline | Status | Note |
|---|---|---|
| Greedy | ✓ | Keep. GRAFT's own §11.4 samples one greedy set, so this is nearly free. |
| Beam / diverse beam | ✓ | Diverse beam search: Vijayakumar et al., AAAI 2018. Fine as listed. |
| MCTS | ✓ | Published RAG instantiations now exist and should be the concrete form: [MCTS-RAG (Findings of EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.672.pdf) and [AirRAG (Findings of EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1030.pdf). Note MCTS is far more expensive than flow sampling — that cost gap is part of your argument. |
| Submodular proof selection | ✓✓ | **The most important search baseline.** Implement the cost-scaled greedy + Lin–Bilmes fallback from [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1) with relevance / query-coverage / facility-location / diversity terms, and Nemhauser-style budgeted greedy guarantees. If a training-free submodular packer matches EpiSetFlow on minimality, that is the result that kills the paper — so run it early, not last. |
| Learned A\* | ✓ | Cite Neural A\* (Yonetani et al., ICML 2021) for the learned-heuristic form. Worth noting GRAFT's `h_deficit` is naturally an A\* heuristic — this baseline is closer to GRAFT than it looks and deserves a fair implementation. |

**Additions worth including:**
- **Personalized-PageRank retrieval** ([HippoRAG, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf)) — training-free graph retrieval, the standard strong non-learned baseline.
- **Single-pass GNN scoring** ([GFM-RAG](https://arxiv.org/html/2502.01113v2), [SubgraphRAG, ICLR 2025](https://arxiv.org/abs/2410.20724)). SubgraphRAG is the sharpest control for GRAFT: a **lightweight MLP with parallel triple scoring**, with subgraph size adjustable to the downstream LLM. It is the "simple is effective" hypothesis in executable form, and if it matches GRAFT's read side at a fraction of the cost, the burden shifts back to GRAFT.
- **Path-based LLM retrieval** ([RoG, ICLR 2024](https://arxiv.org/abs/2310.01061)) — plan-then-retrieve relation paths; the KGQA-standard interpretable baseline.
- **Iterative retrieval (IRCoT-style), capped at 2 rounds** — per [Beyond Static Retrieval](https://arxiv.org/html/2509.25530v1)'s finding that 2 iterations is the cost-benefit optimum. Running it uncapped would be a strawman.
- **MMR** — the classic diversity baseline; it appears in the packing comparison above (0.410 F1 vs 0.451 submodular on HotpotQA) and is cheap to include.

## 5.3 Memory-system baselines — **entirely missing from §18**

§18 has no system-level comparators at all. GRAFT is a memory system; it will be reviewed as one. Required:

- **Full-context** (no memory). **Non-negotiable** — it scored 72.90 J on LoCoMo, above every memory system Mem0 tested. Omitting it will read as evasion.
- **Mem0 and Mem0ᵍ** ([ECAI 2025](https://arxiv.org/html/2504.19413v1)) — the flat-vs-graph pair, with latency and token cost.
- **Zep** ([preprint](https://arxiv.org/pdf/2501.13956)) — the temporal-KG comparator, the closest architectural relative to GRAFT.
- **Memory-R1** ([ACL 2026](https://arxiv.org/abs/2508.19828)) — the learned-memory-operations comparator; overlaps GRAFT's EditSetFlow almost exactly (ADD/UPDATE/DELETE/NOOP vs GRAFT's eight edit atoms).
- **A-Mem, LangMem, OpenAI memory** — reported in Mem0's table (48.38 / 58.10 / 52.90 J), so they cost nothing extra to cite as context.
- **Plain RAG at matched token budget** — Mem0 reports RAG k=2/8192 tokens at 60.53 J with 2.31s p50. A structured system must beat a dumb one at equal budget.

## 5.4 Ablations — audit of §18's twelve

The twelve listed ablations are well-targeted. Three notes:

- **#7 "terminal-only reward" is the ablation that decides whether EpiSetFlow earns its complexity.** If terminal-only reward with a good objective (TB/SubTB) matches the full system, everything in §5–§8 is unjustified. Prioritize it.
- **#3 "no canonical set state"** — per §3.2, this may show little effect, because GFlowNets already handle multi-trajectory states natively. Expect a null result and plan to report it honestly; a null here is fine, it just means dropping the novelty claim.
- **Add: "no equivalent-action correction"** (i.e. SA-GFN reward scaling on vs. off), per §3.2. If GRAFT's atoms can collide under canonicalization, this ablation has a real, published expected effect size.

## 5.5 Benchmarks and metrics — the plan is silent, and shouldn't be

The document never names a benchmark. Given the domain, the defaults are **LoCoMo** ([ACL 2024](https://aclanthology.org/2024.acl-long.747/)), **LongMemEval**, and **MSC** (the three used by Memory-R1). Known problems with both defaults, which GRAFT should address rather than inherit:

- **LoCoMo measures whether the *answer* was right, not whether *retrieval* was right** — an integration test where GRAFT needs a unit test. GRAFT should report proof-set-level metrics separately (certified-binding survival, evidence precision/recall, minimality) — analogous to the answer-in-context diagnostic in §4.4.
- **LLM-as-judge reliability is contested.** A non-peer-reviewed community audit reports ~6.4% answer-key errors and that the judge accepts a large share of deliberately wrong answers; I flag this as **unverified** and have not relied on it, but it argues for reporting F1/BLEU-1 alongside judge scores (as Mem0 does) rather than judge scores alone.
- **LongMemEval-S at ~115k tokens/question fits inside current context windows**, so it partly measures context capacity rather than memory. Report full-context on it explicitly, as Zep did.
- **Add citation-quality metrics**: ALCE-style citation recall/precision and VeriCite's citation F1. GRAFT's entire selling point is proof-carrying answers; it should be evaluated with the field's citation metrics, not just QA accuracy. This is also how you win against full-context — full-context cannot produce minimal grounded citations.
- **Report calibration/abstention explicitly**: abstention rate, false-abstention rate on answerable questions, and contested-detection rate. §5.3 and §11.4 make abstention a first-class output; nothing in §19 measures whether it is *correctly* used.

---

# Part D — Novelty claims audit

§20's "claims to avoid" list is sensible and should be kept verbatim. Refinements based on this review:

| Claim in §20 | Assessment |
|---|---|
| "order-invariant edit and evidence sets" as a contribution | **Weaken or drop.** Native to GFlowNets since [Foundations (JMLR 2023)](https://jmlr.org/papers/v24/22-0364.html); set generation is a standard FL-GFN benchmark. Reframe as "explicit canonicalization with measured equivalent-action collapse." |
| "separate exploratory and authoritative flows" | **Keep — this is the real contribution.** No precedent found. Must be carried by the dual-mass ablation and success gate #8. |
| "authoritative flow admitted only through deterministic certification" | **Keep, but position correctly.** Action masking on a feasible sub-DAG is standard ([Robust Scheduling, ICLR 2023](https://arxiv.org/abs/2302.05446); [Let the Flows Tell, NeurIPS 2023](https://arxiv.org/abs/2305.17010)). What is new is *what* the mask encodes (source/identity/temporal/scope certificates) and that it gates a *second* flow channel. |
| "proof-deficit … provide local credit" | **Cannot be claimed as-is.** Local credit from intermediate energies is [FL-GFN (ICML 2023)](https://arxiv.org/html/2302.01687) and [GAFlowNets (ICLR 2023)](https://openreview.net/pdf?id=urF_CBK5XC0). The novelty is the *specific* proof-obligation energy, not the mechanism. Say "we instantiate forward-looking GFlowNets with a proof-obligation energy." |
| "counterfactual evidence signals" | **Weaken.** Leave-one-out attribution is [ContextCite (NeurIPS 2024)](https://arxiv.org/abs/2409.00729). The novelty is *typed structural* removal (§6.4) and using it as a training target for set inclusion. That is a real but narrow contribution — claim exactly that. |
| "closure-residual" supervision | **Keep as novel, label as unvalidated.** No precedent found either way. |
| "for conversational graph memory" | **Keep — this is your strongest defensible framing.** I found no prior GFlowNet application to agent memory or memory-edit operations. Combined with the Robust-Scheduling proxy-robustness argument (§3.14), this is the paper. |

**One claim you should *add*, because it is well-supported and currently unstated:** *a diverse-candidate sampler is the right learner when the true objective is an expensive deterministic verifier and the learned utility is only a proxy* — [Robust Scheduling with GFlowNets (ICLR 2023)](https://arxiv.org/abs/2302.05446), corroborated in the LLM setting by [FlowRL (ICLR 2026)](https://arxiv.org/abs/2509.15207) and [Flow of Reasoning (ICML 2025)](https://arxiv.org/abs/2406.05673). This converts EpiSetFlow from "a GFlowNet variant we designed" into "the learner the architecture demands."

---

# Part E — Risk register and recommended sequencing

## 6.1 Risks, ordered by severity

| # | Risk | Severity | Evidence | Fix |
|---|---|---|---|---|
| 1 | **Objective not anchored to terminal utility** (§8 vs §9) | **Critical** | Analysis; boundary condition required by FM/DB formalism ([JMLR 2023](https://jmlr.org/papers/v24/22-0364.html)) | State `F^stop(s) = exp(β·U(s))`, or adopt TB/SubTB where it appears natively |
| 2 | **Φ as auxiliary head, not reparametrization** (§5.2, §8.5) | **Critical** | [FL-GFN Prop. 4.2](https://arxiv.org/html/2302.01687); Ng et al. 1999 doesn't transfer to distribution matching | Adopt FL-DB with ℰ = −Φ |
| 3 | **FM-style conservation is the weakest objective** (§8.2/8.3) | High | [TB (NeurIPS 2022)](https://arxiv.org/abs/2201.13259); [SubTB (ICML 2023)](https://arxiv.org/abs/2209.12782); [FL-GFN](https://arxiv.org/html/2302.01687) | Rewrite in SubTB(λ) or FL-DB form |
| 4 | **Graph memory may lose to flat memory and to full-context** | High | [Mem0 (ECAI 2025)](https://arxiv.org/html/2504.19413v1): graph −4.0 on multi-hop; full-context 72.90 > all memory systems | Reposition claims onto temporal / lifecycle / attribution / latency |
| 5 | **Minimality benefit may reverse with a stronger verbalizer** | High | [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1): 3B +0.022, 14B **−0.029 (p=0.013)** | State the frozen-SLM scope condition explicitly; test at ≥2 reader sizes |
| 6 | **No guarantee survives the six-term composite** (§8.7) | High | Every cited guarantee is single-objective | Say so plainly; measure fit with [FCS (ICLR 2025)](https://openreview.net/forum?id=9GsgCUJtic) |
| 7 | **Sparse rewards from feasibility-first** (§9.3) | Medium-High | Documented GFlowNet weak point; motivation for SubTB and GAFlowNets | FL-DB + prioritized replay (α=50%, β=90%, [Shen et al.](https://arxiv.org/abs/2305.07170)) |
| 8 | **GNN expressiveness caps representable distributions** | Medium-High | [ICLR 2025 Spotlight](https://openreview.net/forum?id=9GsgCUJtic) | Report FCS; ablate GNN capacity |
| 9 | **Counterfactual scoring blows the latency budget** | Medium | [AttriBoT](https://arxiv.org/html/2411.15102v1): LOO = \|C\|+1 passes | Commit to a deterministic S; apply hierarchical + cached LOO |
| 10 | **Redundancy term D(e) near-vacuous** (§6.2) | Medium | Analysis | Redefine via coverage/facility-location marginal gain |
| 11 | **Authoritative sub-DAG reachability unproven** (§8.3) | Medium | Analysis | Prove κ monotone in set inclusion, or measure unreachable-terminal rate |
| 12 | **Delta training invalidation rule too narrow** (§13) | Medium | Analysis; [ICLR 2025 generalization bounds](https://proceedings.iclr.cc/paper_files/paper/2025/hash/000eba875068854d5ff003b1fa534cd6-Abstract-Conference.html) support the premise | Invalidate on legal-action-set change, not just lifecycle/deletion |
| 13 | **Equivalent-action bias if atoms collide** (§2.3) | Medium | [SA-GFN (ICML 2025)](https://arxiv.org/abs/2506.02685): L₁ 0.12 → 0.01 | Measure collision rate; apply automorphism reward scaling if nonzero |
| 14 | **Over-abstention unmeasured** (§19) | Medium | [Self-RAG](https://selfrag.github.io/): retrieving less costs 40% relative on PopQA | Add a false-abstention gate |
| 15 | **Upstream extraction/entity-resolution errors dominate** | Medium | [Zep](https://arxiv.org/pdf/2501.13956); [GFM-RAG](https://arxiv.org/html/2502.01113v2); [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0901.pdf) | Report certification recall, not just precision |
| 16 | **§16 adaptive τ underspecified** | Low (deferred) | — | §14 Stage 6 already defers it. Keep it out of the paper. |

## 6.2 Recommended sequencing

**Before any training runs — fix the math (1 week).**
1. Write the terminal boundary condition (risk 1).
2. Convert §8 to FL-DB with ℰ = −Φ (risks 2, 3, 7 in one move).
3. Define Z_A; prove or measure authoritative reachability (risk 11).
4. Redefine D(e) via coverage (risk 10).
5. Decide deterministic vs. model-based S (risk 9).

**Kill-shot experiments — run these before building the full system.** Each can end the project cheaply, which is the point.
- **E1 — Submodular packer vs. EvidenceSetFlow.** Training-free cost-scaled greedy per [arXiv 2607.00725](https://arxiv.org/html/2607.00725v1). If it matches on minimality and validity, EpiSetFlow's read side is unjustified.
- **E2 — Terminal-only reward** (ablation #7) with TB/SubTB. If it matches, §5–§8 are unjustified.
- **E3 — FL-DB alone vs. full EpiSetFlow.** Isolates whether the six-term composite adds anything over the one published mechanism that already does most of what §5.2 wants.
- **E4 — Full-context and Mem0/Mem0ᵍ on the target benchmark.** Establishes the accuracy ceiling and the latency/token floor before any GRAFT claim is framed.
- **E5 — Reader-size sweep** (small SLM vs. mid vs. large) on minimality benefit, to establish the scope condition from risk 5 *before* it becomes a reviewer's discovery.

**Then, if E1–E3 survive:** dual mass (the actual contribution) gets the full experimental budget, with the loss-term ladder and the FCS diagnostic.

**Fallback, per §19:** the plan's own instruction — "If it fails, retain supervised PC-AGT-GNN plus set imitation" — is sound. Note that this fallback plus deterministic certification plus proof-carrying citations is *already* a publishable systems paper on the evidence in Part B, with ALCE/VeriCite-style citation metrics as the headline. The learning contribution is upside, not the foundation. Structuring the work so that the systems paper survives an EpiSetFlow failure is the right risk posture.

---

# References

**Memory systems**
- Chhikara et al. **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.** ECAI 2025 (FAIA vol. 413). [arXiv:2504.19413](https://arxiv.org/html/2504.19413v1)
- Rasmussen et al. **Zep: A Temporal Knowledge Graph Architecture for Agent Memory.** arXiv preprint (vendor-authored, not peer-reviewed). [arXiv:2501.13956](https://arxiv.org/pdf/2501.13956)
- Yan et al. **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning.** ACL 2026. [ACL Anthology](https://aclanthology.org/2026.acl-long.583/) · [arXiv:2508.19828](https://arxiv.org/abs/2508.19828)
- Maharana et al. **Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo).** ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.747/)

**Graph retrieval / RAG**
- Luo et al. **GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation.** NeurIPS 2025. [arXiv:2502.01113](https://arxiv.org/html/2502.01113v2)
- Mavromatis & Karypis. **GNN-RAG: Graph Neural Retrieval for Efficient LLM Reasoning on Knowledge Graphs.** Findings of ACL 2025. [ACL Anthology](https://aclanthology.org/2025.findings-acl.856/)
- Gutiérrez et al. **HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs.** NeurIPS 2024. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf)
- Li, Miao & Li. **Simple Is Effective: The Roles of Graphs and LLMs in KG-Based RAG (SubgraphRAG).** ICLR 2025. [arXiv:2410.20724](https://arxiv.org/abs/2410.20724)
- Luo et al. **Reasoning on Graphs (RoG): Faithful and Interpretable LLM Reasoning.** ICLR 2024. [arXiv:2310.01061](https://arxiv.org/abs/2310.01061)
- Guo et al. **Beyond Static Retrieval: Opportunities and Pitfalls of Iterative Retrieval in GraphRAG.** arXiv preprint. [arXiv:2509.25530](https://arxiv.org/html/2509.25530v1)
- Bala. **What Survives Into Context: A Diagnostic for Budget-Constrained Multi-Hop RAG and When Submodular Evidence Packing Improves It.** arXiv preprint (v2 retitled *Recall Is Not Enough*). [arXiv:2607.00725](https://arxiv.org/html/2607.00725v1)
- Liu et al. **Lost in the Middle: How Language Models Use Long Contexts.** TACL 12 (2024) 157–173. [ACL Anthology](https://aclanthology.org/2024.tacl-1.9/)

**Citation / attribution / verification**
- Gao et al. **Enabling Large Language Models to Generate Text with Citations (ALCE).** EMNLP 2023. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.398/)
- **VeriCite: Towards Reliable Citations in RAG via Rigorous Verification.** SIGIR-AP 2025. [arXiv:2510.11394](https://arxiv.org/html/2510.11394v1)
- Cohen-Wang et al. **ContextCite: Attributing Model Generation to Context.** NeurIPS 2024. [arXiv:2409.00729](https://arxiv.org/abs/2409.00729)
- **AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution.** [arXiv:2411.15102](https://arxiv.org/html/2411.15102v1)
- Asai et al. **Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection.** ICLR 2024. [Project page](https://selfrag.github.io/)

**GFlowNets — foundations and objectives**
- Bengio et al. **GFlowNet Foundations.** JMLR 24 (2023). [JMLR](https://jmlr.org/papers/v24/22-0364.html)
- Malkin et al. **Trajectory Balance: Improved Credit Assignment in GFlowNets.** NeurIPS 2022. [arXiv:2201.13259](https://arxiv.org/abs/2201.13259)
- Madan et al. **Learning GFlowNets from Partial Episodes for Improved Convergence and Stability (SubTB(λ)).** ICML 2023. [arXiv:2209.12782](https://arxiv.org/abs/2209.12782)
- Pan et al. **Better Training of GFlowNets with Local Credit and Incomplete Trajectories (FL-GFN).** ICML 2023. [arXiv:2302.01687](https://arxiv.org/html/2302.01687)
- Pan et al. **Generative Augmented Flow Networks (GAFlowNets).** ICLR 2023 (spotlight). [OpenReview](https://openreview.net/pdf?id=urF_CBK5XC0)
- Shen et al. **Towards Understanding and Improving GFlowNet Training.** ICML 2023. [arXiv:2305.07170](https://arxiv.org/abs/2305.07170)
- Hu et al. **Beyond Squared Error: Exploring Loss Design for Enhanced Training of GFlowNets.** [arXiv:2410.02596](https://arxiv.org/abs/2410.02596)

**GFlowNets — theory, symmetry, exploration**
- da Silva, de Souza da Silva, Mesquita et al. **When Do GFlowNets Learn the Right Distribution?** ICLR 2025 (Spotlight). [OpenReview](https://openreview.net/forum?id=9GsgCUJtic)
- Silva, Souza, Rivasplata, Garg, Kaski, Mesquita. **Generalization and Distributed Learning of GFlowNets.** ICLR 2025. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/000eba875068854d5ff003b1fa534cd6-Abstract-Conference.html)
- Kim, Lee & Oh. **Symmetry-Aware GFlowNets (SA-GFN).** ICML 2025. [arXiv:2506.02685](https://arxiv.org/abs/2506.02685)
- Ma, Bengio, Bengio & Zhang. **Baking Symmetry into GFlowNets.** NeurIPS 2023 AI4Science **workshop**. [arXiv:2406.05426](https://arxiv.org/abs/2406.05426)
- Kim et al. **Local Search GFlowNets.** ICLR 2024 (spotlight). [OpenReview](https://openreview.net/forum?id=6cFcw1Rxww)
- **Pessimistic Backward Policy for GFlowNets (PBP-GFN).** NeurIPS 2024. [arXiv:2405.16012](https://arxiv.org/abs/2405.16012)
- **Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization.** ICLR 2025. [arXiv:2410.15474](https://arxiv.org/pdf/2410.15474)
- Chen & Mauch. **Order-Preserving GFlowNets.** ICLR 2024. [arXiv:2310.00386](https://arxiv.org/abs/2310.00386)
- Zhang et al. **COFlowNet: Conservative Constraints on Flows Enable High-Quality Candidate Generation.** ICLR 2025. [ICLR page](https://iclr.cc/virtual/2025/poster/28047)
- Jain et al. **Multi-Objective GFlowNets.** ICML 2023. [PMLR](https://proceedings.mlr.press/v202/jain23a/jain23a.pdf)
- Zhang, Rainone, Peschl & Bondesan. **Robust Scheduling with GFlowNets.** ICLR 2023. [arXiv:2302.05446](https://arxiv.org/abs/2302.05446)
- Zhang et al. **Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets.** NeurIPS 2023 (spotlight). [arXiv:2305.17010](https://arxiv.org/abs/2305.17010)
- Bu et al. **Enhanced Data Synthesis for LLM through Reasoning Structures Generated by Hierarchical GFlowNet.** Findings of ACL 2025. [ACL Anthology](https://aclanthology.org/2025.findings-acl.821/)
- Fard. **Benchmarking GFlowNets against MCMC: The Role of Peak Structure.** J. Advances in Computing 57(2), 2025 — *non-A/A\* venue; results not independently verifiable from the PDF.* [PDF](https://journal.ut.ac.ir/article_106220_565b5b56aeb6a2e1813f39d7ffebcd62.pdf)

**Flow methods for LLMs / RL comparators**
- Zhu et al. **FlowRL: Matching Reward Distributions for LLM Reasoning.** ICLR 2026. [arXiv:2509.15207](https://arxiv.org/abs/2509.15207)
- Yu et al. **Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples.** ICML 2025. [arXiv:2406.05673](https://arxiv.org/abs/2406.05673)
- Hu, Jain, Elmoznino, Kaddar, Lajoie, Bengio & Malkin. **Amortizing Intractable Inference in Large Language Models.** ICLR 2024 (oral). [arXiv:2310.04363](https://arxiv.org/abs/2310.04363)
- Jin et al. **Search-R1: Training LLMs to Reason and Leverage Search Engines with RL.** [arXiv:2503.09516](https://arxiv.org/pdf/2503.09516)
- Ng, Harada & Russell. **Policy Invariance Under Reward Transformations.** ICML 1999, 278–287. *(Cited here only to note it does* not *transfer to GFlowNets.)*

**Search baselines**
- Vijayakumar et al. **Diverse Beam Search.** AAAI 2018.
- Yonetani et al. **Path Planning using Neural A\* Search.** ICML 2021.
- **MCTS-RAG.** Findings of EMNLP 2025. [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.672.pdf)
- **AirRAG: Autonomous Strategic Planning and Reasoning Steer RAG.** Findings of EMNLP 2025. [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1030.pdf)
- Nemhauser, Wolsey & Fisher (1978); Lin & Bilmes (ACL 2011) — budgeted monotone submodular maximization and the greedy guarantee underlying the packer in arXiv:2607.00725.

**Other**
- Cong et al. **Do We Really Need Complicated Model Architectures for Temporal Networks? (GraphMixer).** ICLR 2023. [arXiv:2302.11636](https://arxiv.org/abs/2302.11636) — *relevant as a caution: on temporal link prediction, a pure-MLP baseline matched or beat RNN/self-attention temporal GNNs. Include a simple temporal baseline for PC-AGT-GNN.*
- **How to Mitigate Information Loss in Knowledge Graphs for GraphRAG.** IJCAI 2025. [PDF](https://www.ijcai.org/proceedings/2025/0901.pdf)
- **One Model, Any Conjunctive Query: GNNs for Answering Queries over Incomplete Knowledge Graphs.** [arXiv:2409.13959](https://arxiv.org/html/2409.13959)
