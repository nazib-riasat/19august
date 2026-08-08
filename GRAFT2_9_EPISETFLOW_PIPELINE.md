# GRAFT-2.9 — EpiSetFlow Pipeline

## Proof-Carrying Authority-Gated Temporal GNN with Epistemic Set-Flow Learning

**Status:** Final task-specific redesign of the GFlowNet-inspired training stage  
**Supersedes:** `GRAFT2_8_PROOFFLOW_FINAL_PIPELINE.md`  
**Core graph model:** Proof-Carrying Authority-Gated Temporal Heterogeneous GNN (`PC-AGT-GNN`)  
**New learning mechanism:** Epistemic Set-Flow Learning (`EpiSetFlow`)  
**Write learner:** `EditSetFlow`  
**Read learner:** `EvidenceSetFlow`  
**Deterministic modules:** authority checker, lifecycle checker, proof checker, closure certificate  
**Answer model:** frozen small language model  
**Optional efficiency:** adaptive τ transit on low-risk paths  
**Inference:** no weight updates; only external graph state evolves

---

# 0. Research Position

GRAFT-2.9 does not present an ordinary GFlowNet transplanted into conversational memory.

It retains one useful principle from GFlowNets:

> distribute probability mass across multiple high-quality compositional objects rather than optimizing only one trajectory.

It changes the learning problem in five task-specific ways:

1. **Order-invariant set states**  
   The learned objects are canonical edit sets and evidence sets, not arbitrary action sequences.

2. **Dual epistemic mass**  
   Exploratory mass and authoritative mass obey different transition rules.

3. **Certification-controlled authority**  
   Neural actions cannot move probability mass into the authoritative channel unless deterministic source, identity, temporal, and scope checks pass.

4. **Counterfactual evidence credit**  
   Each selected evidence component is scored by how much the proof fails when that component is removed.

5. **Proof-deficit and closure-deficit learning**  
   Partial states receive task-specific dense supervision from unsatisfied proof obligations and unresolved closure obligations.

The system should be described as:

> a GFlowNet-inspired, task-specific set-flow learning framework.

It should not claim to have invented flow matching or reward-proportional compositional sampling.

---

# 1. Why Ordinary GFlowNet Is Not Enough

A standard conditional GFlowNet would:

- define a construction DAG;
- assign a terminal reward;
- train forward/backward policies with Flow Matching, Trajectory Balance, or Subtrajectory Balance;
- sample terminal objects approximately in proportion to reward.

That formulation leaves several GRAFT-specific problems unresolved:

- evidence order should not affect the proof object;
- exploratory and authoritative evidence are not equivalent;
- an invalid verified link must not receive positive authority flow;
- terminal answer reward is too coarse to identify which evidence mattered;
- closure failure may arise from missing candidates, lag, scope, or dispute;
- dynamic conversational updates should not require relearning the complete graph distribution;
- multiple proof subgraphs can share redundant evidence but differ in minimality.

EpiSetFlow makes these requirements part of the state representation and objective.

---

# 2. Learned Objects

## 2.1 Write object

For a new conversational turn, construct a canonical edit proposal set:

\[
X_W =
\{
x_1,\ldots,x_k
\}
\]

Possible edit atoms:

```text
ADD_CLAIM_PROPOSAL
ATTACH_SOURCE_SPAN
ADD_HYPOTHESIZED_LINK
PROPOSE_VERIFIED_LINK
PROPOSE_TEMPORAL_UPDATE
PROPOSE_CONFLICT
PROPOSE_SUPERSESSION
MARK_UNRESOLVED
```

The persistent graph is changed only after the complete set is checked.

## 2.2 Read object

For a question, construct a canonical evidence set:

\[
X_Q =
(V_Q,E_Q,B_Q)
\]

where:

- \(V_Q\): evidence, claim, and entity nodes;
- \(E_Q\): selected graph relations;
- \(B_Q\): proposed answer bindings.

The selected set is rendered as a connected proof subgraph after construction.

## 2.3 Canonical state

A partial state is identified by its selected atoms, sorted and hashed canonically:

\[
s =
\operatorname{Canon}
(
\{x_1,\ldots,x_t\}
)
\]

Two action sequences that select the same atoms correspond to the same state.

This removes arbitrary action-order dependence from the scientific object.

---

# 3. Dual Epistemic Mass

## 3.1 Exploratory mass

\[
F_X(s\mid c)
\]

represents flow assigned to candidate structures that may contain:

- support claims;
- hypothesized links;
- unresolved conflicts;
- uncertain temporal relations.

Exploratory mass controls:

- search;
- candidate expansion;
- alternative generation;
- uncertainty.

## 3.2 Authoritative mass

\[
F_A(s\mid c)
\]

represents flow assigned to structures that satisfy answer-authority requirements.

Authoritative mass may contain only:

- asserted source-grounded claims;
- verified identity paths;
- admissible temporal/lifecycle state;
- authorized scope;
- non-deleted evidence.

## 3.3 Asymmetric transition law

An exploratory transition is allowed when type and scope constraints pass:

\[
s
\xrightarrow[\text{explore}]{a}
s'
\]

An authoritative transition requires a deterministic certificate:

\[
s
\xrightarrow[\text{authorize}]{a,\kappa(a)}
s'
\]

where:

\[
\kappa(a)\in\{0,1\}
\]

is produced by the external checker.

If \(\kappa(a)=0\):

- no authoritative flow is transferred;
- the atom may remain exploratory;
- or the action is rejected.

## 3.4 No neural authority promotion

The neural network predicts:

- which candidate to inspect;
- which source span to request;
- which link to propose;
- which proof atom to add.

It does not decide that a claim or link is authoritative.

---

# 4. EpiSetFlow State Representation

For partial state \(s\), the `PC-AGT-GNN` produces:

\[
h_s =
[
h_s^A
\Vert
h_s^X
\Vert
h_q
\Vert
h_{\mathrm{deficit}}
\Vert
h_{\mathrm{budget}}
]
\]

where:

- \(h_s^A\): authoritative selected-subgraph representation;
- \(h_s^X\): exploratory selected/frontier representation;
- \(h_q\): query or write-context representation;
- \(h_{\mathrm{deficit}}\): unsatisfied proof/closure obligations;
- \(h_{\mathrm{budget}}\): remaining node, edge, and latency budget.

Forward action scores are:

\[
\ell(a\mid s,c)
=
\operatorname{MLP}_{\mathrm{flow}}
[
h_s
\Vert
h_a
\Vert
h_{\mathrm{relation}}
]
\]

Invalid actions are masked before normalization.

---

# 5. Proof Obligations

## 5.1 Obligation vector

For a query, define:

\[
d(s)
=
[
d_{\mathrm{entity}},
d_{\mathrm{value}},
d_{\mathrm{source}},
d_{\mathrm{time}},
d_{\mathrm{conflict}},
d_{\mathrm{closure}}
]
\]

Each component measures an unsatisfied obligation.

Examples:

- no verified anchor for the requested entity;
- value claim exists but source span is missing;
- historical question lacks matching temporal interval;
- conflicting claim is unrepresented;
- aggregate answer lacks source-inventory coverage.

## 5.2 Deficit potential

\[
\Phi(s)
=
-
\sum_j
\omega_j d_j(s)
-
\lambda_v |V_s|
-
\lambda_e |E_s|
\]

A transition receives shaped local gain:

\[
g(s,a,s')
=
\Phi(s')-\Phi(s)
\]

Because this is a potential difference, it provides dense guidance without changing the ranking of terminal utilities when applied consistently.

## 5.3 Stopping rule

`STOP` is masked unless:

- all mandatory obligations are satisfied; or
- the state contains a valid abstention certificate.

For closure routes, `STOP_CLOSED` is masked unless the deterministic closure checker accepts.

---

# 6. Counterfactual Evidence Credit

## 6.1 Necessity score

For terminal evidence set \(X\) and atom \(e\in X\):

\[
N(e;X,q)
=
\max
\left(
0,
S(X,q)-S(X\setminus\{e\},q)
\right)
\]

where \(S\) is a deterministic or fixed-model proof score.

A high necessity score means the atom contributes uniquely.

## 6.2 Redundancy score

\[
D(e;X,q)
=
\max
\left(
0,
S(X\setminus\{e\},q)-S(X,q)+\epsilon
\right)
\]

or use zero necessity plus semantic overlap as a redundancy indicator.

## 6.3 Inclusion target

The model receives atom-level supervision:

\[
y_e
=
\sigma
\left(
\alpha N(e)-\beta D(e)
\right)
\]

This trains evidence inclusion directly rather than relying only on terminal answer reward.

## 6.4 Counterfactual types

Run removal tests for:

- source span;
- entity link;
- temporal edge;
- conflict evidence;
- closure member;
- exploratory bridge.

A hypothesized bridge may be useful for retrieval while having zero authoritative necessity.

---

# 7. Closure Residual

For aggregate route, define:

\[
r_C(s)
=
[
u_{\mathrm{unprojected}},
u_{\mathrm{support}},
u_{\mathrm{hypothesis}},
u_{\mathrm{dispute}},
u_{\mathrm{lag}},
u_{\mathrm{budget}}
]
\]

The closure residual is zero only when every component is zero.

\[
\|r_C(s)\|_1=0
\]

is necessary, but not sufficient, for closure; the deterministic certificate must still pass.

Train the closure head to predict each residual component.

This gives stronger supervision than a single terminal “closed/not closed” label.

---

# 8. Set-Flow Conservation Objective

## 8.1 Parent and child sets

For canonical state \(s\):

- \(\operatorname{Pa}(s)\): states produced by removing one selected atom;
- \(\operatorname{Ch}(s)\): states produced by adding one legal atom.

## 8.2 Exploratory conservation

\[
\sum_{p\in\operatorname{Pa}(s)}
F_X(p\rightarrow s)
+
R_X(s)
=
\sum_{s'\in\operatorname{Ch}(s)}
F_X(s\rightarrow s')
+
F_X^{\mathrm{stop}}(s)
\]

where \(R_X(s)\) is optional local exploratory source flow for newly discovered supported candidates.

## 8.3 Authoritative conservation

\[
\sum_{p\in\operatorname{Pa}(s)}
\kappa(p\rightarrow s)
F_A(p\rightarrow s)
=
\sum_{s'\in\operatorname{Ch}(s)}
\kappa(s\rightarrow s')
F_A(s\rightarrow s')
+
F_A^{\mathrm{stop}}(s)
\]

No failed certification transition contributes authoritative flow.

## 8.4 Log-residual losses

\[
\mathcal L_X
=
\sum_s
\left[
\log
\left(
\epsilon+
F_X^{\mathrm{in}}(s)
\right)
-
\log
\left(
\epsilon+
F_X^{\mathrm{out}}(s)
\right)
\right]^2
\]

\[
\mathcal L_A
=
\sum_s
\left[
\log
\left(
\epsilon+
F_A^{\mathrm{in}}(s)
\right)
-
\log
\left(
\epsilon+
F_A^{\mathrm{out}}(s)
\right)
\right]^2
\]

## 8.5 Deficit-consistency loss

\[
\mathcal L_{\mathrm{deficit}}
=
\sum_{(s,a,s')}
\left[
\hat g_\theta(s,a,s')
-
\left(
\Phi(s')-\Phi(s)
\right)
\right]^2
\]

## 8.6 Counterfactual inclusion loss

\[
\mathcal L_{\mathrm{cf}}
=
-\sum_{e}
\left[
y_e\log p_\theta(e\in X)
+
(1-y_e)\log(1-p_\theta(e\in X))
\right]
\]

## 8.7 Final objective

\[
\mathcal L_{\mathrm{EpiSetFlow}}
=
\mathcal L_X
+
\lambda_A\mathcal L_A
+
\lambda_D\mathcal L_{\mathrm{deficit}}
+
\lambda_{\mathrm{cf}}\mathcal L_{\mathrm{cf}}
+
\lambda_{\mathrm{route}}\mathcal L_{\mathrm{route}}
+
\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}
\]

This is not standard Subtrajectory Balance copied unchanged.

---

# 9. Terminal Utility

## 9.1 Read utility

For feasible proof set \(X_Q\):

\[
U_Q
=
w_p S_{\mathrm{proof}}
+
w_e S_{\mathrm{evidence}}
+
w_b S_{\mathrm{binding}}
+
w_t S_{\mathrm{temporal}}
+
w_z S_{\mathrm{abstention}}
-
\lambda_n |V_Q|
-
\lambda_m |E_Q|
\]

## 9.2 Write utility

\[
U_W
=
w_g S_{\mathrm{grounding}}
+
w_l S_{\mathrm{link}}
+
w_t S_{\mathrm{temporal}}
+
w_c S_{\mathrm{conflict}}
+
w_o S_{\mathrm{edit}}
-
\lambda_k |X_W|
\]

## 9.3 Feasibility-first rule

Reliability-critical failures are not compensated by high utility.

Set terminal utility to infeasible when:

- scope leakage occurs;
- deleted evidence is used;
- an authoritative binding relies only on a hypothesized link;
- closure is claimed without a valid certificate;
- an invalid retirement is proposed;
- source grounding is absent for an asserted value.

---

# 10. EditSetFlow

## 10.1 Purpose

Generate several valid edit proposal sets for a new turn.

## 10.2 Two-phase construction

### Phase E1 — Exploratory proposal

Construct:

- claim proposals;
- candidate entity links;
- temporal candidates;
- conflict candidates.

### Phase E2 — Certification projection

For each candidate:

- attach exact evidence;
- run source-grounding checker;
- run identity checker;
- run lifecycle checker;
- assign verified, hypothesized, support, or rejected state.

## 10.3 Atomic commit

Select the highest-utility checked edit set and commit it atomically.

Keep alternative unresolved links only in the repair plane.

---

# 11. EvidenceSetFlow

## 11.1 Purpose

Construct multiple compact proof sets.

## 11.2 Two coupled frontiers

### Exploratory frontier

May follow:

- hypothesized coreference;
- support claims;
- uncertain predicate aliases.

### Authoritative frontier

Contains only:

- verified entity paths;
- asserted claims;
- exact source spans;
- admissible temporal/lifecycle information.

## 11.3 Bridge behavior

An exploratory edge may lead to an authoritative source.

The final proof contains:

- the authoritative source;
- a flag that discovery used an exploratory bridge;
- no claim that the bridge itself authorized the answer.

## 11.4 Selection

At inference, sample:

- one greedy proof set;
- several diverse high-mass alternatives.

Choose:

1. valid;
2. minimal;
3. sufficient;
4. low-risk.

When valid proof sets imply different answers, return contested or abstain.

---

# 12. Efficient Backward Operation

A fully learned backward policy is optional.

Use a deterministic removable-atom distribution:

\[
P_B(s\mid s')
=
\frac{
w_{\mathrm{remove}}(e)
}{
\sum_{e'\in\operatorname{Removable}(s')}
w_{\mathrm{remove}}(e')
}
\]

Recommended weights:

- redundant evidence: high removal weight;
- exploratory edge: medium;
- necessary authoritative anchor: low;
- sole source span: non-removable until dependent binding is removed.

This reduces model parameters and makes parent-state semantics interpretable.

A learned backward-policy ablation remains required.

---

# 13. Incremental Delta Training

Conversational memory changes locally.

For new graph delta \(\Delta_t\):

1. freeze unaffected node encodings;
2. update only affected local GNN states;
3. construct edit sets over delta plus bounded neighborhood;
4. reuse replay states whose dependency hashes remain valid;
5. invalidate only proof/edit states touching changed lifecycle or deletion data.

Approximate per-turn cost depends on the delta graph rather than full history.

---

# 14. Training Schedule

## Stage 1 — Supervised PC-AGT-GNN

Train:

- candidate generation;
- entity/reference repair;
- temporal and lifecycle tasks;
- conflict;
- proof-edge scoring;
- closure residual prediction.

## Stage 2 — Canonical set imitation

Convert gold trajectories to canonical terminal sets.

Train inclusion and route heads without preserving arbitrary action order.

## Stage 3 — EpiSetFlow warm start

Use:

- synthetic worlds;
- deterministic high-precision edit sets;
- gold proof sets;
- human difficult cases.

## Stage 4 — Counterfactual credit

Generate leave-one-out and structured-removal examples.

Train necessity and redundancy.

## Stage 5 — Mixed set-flow refinement

Suggested development starting mixture:

```text
40% gold/certified sets
30% on-policy sampled sets
20% counterfactual variants
10% high-error prioritized replay
```

These are starting settings, not fixed scientific constants.

## Stage 6 — Optional adaptive τ

Only after the full-message method passes.

---

# 15. Inference Procedure

## 15.1 Write

1. pin current graph manifest;
2. encode delta neighborhood;
3. generate several exploratory edit sets;
4. certify atoms;
5. project authoritative mass;
6. select highest-utility valid set;
7. commit atomically.

## 15.2 Read

1. pin `SnapshotManifest`;
2. parse task contract;
3. build bounded active graph;
4. sample candidate evidence sets;
5. evaluate proof obligations;
6. run deterministic proof checker;
7. run closure checker when required;
8. select minimal sufficient proof;
9. send proof to frozen SLM;
10. replay answer bindings.

## 15.3 Default sample budget

```yaml
greedy_sets: 1
stochastic_sets: 3
max_selected_atoms: 16
max_graph_hops: 5
```

Profile and tune on development data.

---

# 16. Adaptive τ Integration

τ remains a partial efficiency mechanism.

Use τ only for:

- exploratory frontier propagation;
- low-ambiguity local paths;
- repeated cached traversals.

Use full messages for:

- authoritative certification;
- conflict;
- closure;
- high-entropy entity resolution;
- long paths;
- high-degree bottlenecks.

τ cannot carry authority metadata, source IDs, lifecycle, scope, or deletion status.

---

# 17. Complexity

Let:

- \(n_q,m_q\): bounded active graph size;
- \(K\): candidate atoms per construction step;
- \(H\): maximum selected atoms;
- \(M\): sampled sets;
- \(d\): GNN width.

Graph encoding:

\[
O(n_qd^2+Lm_qd)
\]

Set construction:

\[
O(MHKd)
\]

with cached action-atom representations.

Counterfactual scoring naïvely costs:

\[
O(H)
\]

proof checks per terminal set.

Reduce this with:

- batched leave-one-out checking;
- dependency-aware removal;
- exact incremental proof recomputation;
- sampling only high-uncertainty atoms.

Incremental write cost uses the graph delta:

\[
O(n_\Delta d^2+Lm_\Delta d+MHKd)
\]

Report empirical latency and checker cost separately.

---

# 18. Required Baselines

## Learning baselines

- supervised only;
- canonical set imitation;
- PPO;
- GRPO;
- progressive RL;
- step-wise GRPO;
- standard conditional GFlowNet;
- Trajectory Balance;
- Subtrajectory Balance;
- `EpiSetFlow`.

## Search baselines

- greedy selection;
- beam search;
- diverse beam;
- MCTS;
- submodular proof selection;
- learned A* traversal.

## EpiSetFlow ablations

1. one mass instead of dual mass;
2. no deterministic certification projection;
3. no canonical set state;
4. no proof deficits;
5. no closure residual;
6. no counterfactual credit;
7. terminal-only reward;
8. learned backward policy;
9. deterministic backward policy;
10. no incremental delta reuse;
11. no alternative proof sampling;
12. no authoritative/exploratory frontier split.

---

# 19. Success Gates

EpiSetFlow becomes a primary contribution only if:

1. it beats canonical set imitation;
2. it beats standard GFlowNet/SubTB;
3. it beats PPO/GRPO on a reliability-sensitive metric;
4. it produces more distinct valid proof sets;
5. it reduces unsupported or wrong-entity answers;
6. it does not worsen false closure;
7. it generalizes beyond synthetic templates;
8. dual-mass ablation shows a meaningful safety benefit;
9. counterfactual credit improves minimal proof quality;
10. inference latency remains practical.

If it fails, retain supervised PC-AGT-GNN plus set imitation.

---

# 20. Novelty Claim

## Recommended claim

> We introduce EpiSetFlow, a task-specific set-flow learning framework for conversational graph memory. Unlike trajectory-centric policy optimization or a standard single-mass GFlowNet, EpiSetFlow learns order-invariant edit and evidence sets with separate exploratory and authoritative flows. Authoritative flow is admitted only through deterministic certification, while proof-deficit, closure-residual, and counterfactual evidence signals provide local credit for constructing minimal verifiable proof subgraphs.

## Honest attribution

State explicitly:

> EpiSetFlow is inspired by generative flow networks and flow-conservation learning, but modifies the state representation, mass semantics, transition admissibility, and credit assignment for proof-carrying conversational memory.

## Claims to avoid

- first constrained GFlowNet;
- first hierarchical GFlowNet;
- first GFlowNet for graphs;
- first GFlowNet for reasoning;
- mathematically equivalent reward-proportional sampling unless proved;
- guaranteed calibration;
- guaranteed superiority over PPO or GRPO.

---

# 21. Final Pipeline

```text
Conversation turn
  -> exact raw and source-span commit
  -> typed temporal graph delta
  -> dual-channel PC-AGT-GNN
  -> EditSetFlow exploratory construction
  -> deterministic certification projection
  -> authoritative edit-set flow
  -> atomic graph commit

Question
  -> valid snapshot
  -> task contract
  -> bounded active graph
  -> dual-channel PC-AGT-GNN
  -> EvidenceSetFlow
       exploratory candidate set
       authoritative certified set
       proof deficits
       closure residual
       counterfactual evidence credit
  -> deterministic proof and closure checks
  -> minimal sufficient proof
  -> frozen SLM verbalization
  -> answer replay
  -> answer / partial / contested / abstain

Optional:
  -> adaptive τ only on low-risk exploratory paths
```

---

# 22. Paper Positioning

## Suggested title

**GRAFT: Epistemic Set-Flow Learning for Proof-Carrying Conversational Graph Memory**

Alternative:

**Explore Broadly, Authorize Narrowly: Dual-Mass Set Flows for Long-Term Conversational Memory**

## One-sentence thesis

> The system learns diverse graph repairs and evidence sets through exploratory flow, but transfers mass into answer-authoritative proofs only when exact source, identity, temporal, and scope checks certify the selected structure.

---

# 23. Key References to Distinguish

1. Bengio et al. **GFlowNet Foundations.** JMLR 2023 / arXiv:2111.09266.
2. Madan et al. **Learning GFlowNets From Partial Episodes for Improved Convergence and Stability.** ICML 2023.
3. Pan et al. **Better Training of GFlowNets with Local Credit and Incomplete Trajectories.** ICML 2023.
4. Shen et al. **Towards Understanding and Improving GFlowNet Training.** ICML 2023.
5. Bu et al. **Enhanced Data Synthesis for LLM through Reasoning Structures Generated by Hierarchical GFlowNet.** Findings of ACL 2025.
6. Zhang et al. **COFlowNet: Conservative Constraints on Flows Enable High-Quality Candidate Generation.** ICLR 2025.
7. Zhang et al. **Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets.** NeurIPS 2023.
8. Malkin et al. **Trajectory Balance: Improved Credit Assignment in GFlowNets.**
9. Li et al. **Order-Preserving GFlowNets.**
10. Research on symmetry-aware/canonical GFlowNet state representations must be acknowledged when discussing order-invariant set states.
