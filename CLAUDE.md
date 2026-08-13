# CLAUDE.md — GRAFT project context

**Purpose of this file.** The documents in this folder record *what* was decided. This file records *why*, *what was rejected*, and *what it costs to change your mind*. It exists so that a fresh session — human or Claude — does not silently re-derive, re-litigate, or reverse a decision that took several rounds of review to reach.

**Authority.** The project has been handed over in full. **Nothing here is an order.** Every decision below comes with its reasoning and its cost-to-reverse. Change whatever you judge should change — just do it knowingly, and record it, because several of these choices are coupled to each other and to experiments that would need re-running.

**Setup assumed.** 1× RTX 5090 (32 GB), hybrid LLM access (local Qwen2.5-7B-Instruct 4-bit for bulk extraction, small paid-API budget reserved for the LLM-prompted-linking baseline), frozen reader Qwen2.5-3B-Instruct. If the hardware changes, §6 marks which decisions move with it.

---

## 1. What the project is

A conversational memory system that stores raw turns plus a **provenance-preserving temporal heterogeneous graph**, builds that graph with learned neural decoders, retrieves a **minimal formally-valid evidence set** for each question using a GFlowNet-family set constructor under a deterministic checker, and hands that set to a **frozen** small language model that only verbalizes.

Two claimed contributions, one methodological, one hypothesis:

| | Claim | Risk |
|---|---|---|
| **C1** | Open-world provenance-preserving incremental graph construction (entity *creation*, span provenance, temporal validity, conflict, non-destructive supersession) | Low — most likely to produce a defensible result |
| **C2** | Formal-validity-gated terminal reward for evidence-set GFlowNets with a decoupled answerability gate | Medium |
| **C3** | Checker-conditioned potential learning (LED-GFN + obligation features) | **High — explicitly a hypothesis** |
| **C4** | Five-ceiling evaluation protocol | Methodological, pending novelty search |

The supervisor's constraint that shaped everything: **focus on GNN/NN learning and training; the SLM stays frozen.**

---

## 2. Document map

| File | Status |
|---|---|
| `GRAFT_RESEARCH_PLAN_v1.md` (v1.2) | **Live — the science.** Wins any conflict with other docs. |
| `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (v1.1) | **Live — the build plan.** 12 phases, in build order. |
| `GRAFT_PHASE0_BUILD.md` | **Live.** Phase 0's spec. Built and green as of 8 Aug 2026 — all 13 exit criteria. |
| `GRAFT_PHASE1_BUILD.md` | **Live.** Phase 1's spec. The deterministic core; closes nine gaps (G1–G9). Built and green as of 8 Aug 2026 — all 16 exit criteria. |
| `PHASE1_DECISIONS.md` | **Live.** What Phase 1 decided, its six departures from the plan, the three defects the tests found that reading did not, and the five invariants a post-build review found unenforced. Read before touching `H`, `U` or `d(s)`. |
| `GRAFT_PHASE2_BUILD.md` | **Live.** Phase 2's spec. The enumerable environment and the exact evaluator; closes ten gaps (G1–G10). §6 signed off 9 Aug 2026. Built and green as of 11 Aug 2026 — all 21 exit criteria. |
| `PHASE2_DECISIONS.md` | **Live.** What Phase 2 decided, its five departures, and what the build found that reading did not — including that the `neither`-mass band fails at two of the four predeclared β candidates. Read before touching the generator, `Target` or `ActionPolicy`. |
| `chatcontext1.md` | **Live — the session handoff.** Where the project is, what to read in what order, what is open. Start here on a cold session. |
| `GRAFT_PHASE2_5_BUILD.md` | **Live.** The annotation feasibility spike: one measurement answering Gate-0 item 8. Seven gaps (G1–G7). **Tooling built and run 13 Aug 2026** — corpus pinned (LongMemEval-S, MIT, SHA in `scripts/phase2_5/common.py`), extraction slice run on Qwen2.5-**3B** (a ruled deviation from the architecture's 7B, sized to the actual 8 GB GPU), items + guidelines + bootstrap labels exist. **The Gate-0 item-8 number still needs the human timed pass** — `PHASE2_5_DECISIONS.md` §5. |
| `PHASE2_5_DECISIONS.md` | **Live.** What the spike decided (two amendments to §6: κ subset = 20 *per decoder*; span-grounding floor = 0.80), what it measured, why the bootstrap labels do not answer Gate-0 item 8, and the exact commands for the human timed pass. |
| `GRAFT_PHASE3_BUILD.md` | **Live — the current component.** Phase 3's spec: the nine learner arms, the frozen policy interface and the Gate-2 harness; closes twelve gaps (G1–G12), revision **R6**. §6 is normative and partly signed off — decision 1 is ruled, the rest are recommended. **Steps 1–5 and 7–10 built and green as of 12 Aug 2026; step 6, the calibration gate, has not run**, so `N` and β are not frozen. |
| `PHASE3_DECISIONS.md` | **Live.** What Phase 3 decided, its five departures, the FL-GFN measurement, the two defects the build found that five review rounds did not, and **§6 — the R13 paper-by-paper review**, whose three control defects all ran in the direction that flatters the proposed method. **Read §2.3 before running the calibration gate** — L7 is the slowest arm to converge, and an `N` sized on L4/L5 could hand Contribution 3 a negative verdict that is about budget. §2.2 and §2.4's measurements are stale by R13 and marked so. |
| `PHASE4_DECISIONS.md` | **Live.** What the Phase-4 build decided and measured. **Read §1 before touching S4 or the Gate-3 rule**: three sub-decisions of the ruled §6 table were overturned by measurement — `pcst_fast`'s Windows wheel is silently wrong (59/60 wrong optima, so an exact solver is adopted), `C_e`'s median criterion straddles the cap at a 0.500 breach rate (selection moved to the breach rate), and decision 5's one live Gate-3 condition is size-confounded and predetermined here (S4-informed scores 0.483 under the pinned convention against `p*`'s own 0.4506). §1.4 records the adversarial review round — eight findings fixed, incl. a diversity estimator floating against its own pin and a self-referential test that had left the saturation constant unprotected. §2 carries Stage A's measured table, which reproduces every G9 figure independently. |
| `GRAFT_PHASE4_BUILD.md` | **Live — Stage A built and run 13 Aug 2026; Stage B waits on Phase 3's matrix.** The five Tier-1 search algorithms and Gate 3's synthetic kill-shot; nine gaps (G1–G9), **§6 ruled 12 Aug 2026**. **G9 is the one to read**: measured before any method was built, greedy on exact `U` is globally optimal on 30/30 instances while a *flawless* sampler reaches only 1.8865 of greedy's 1.9245 at K = 8 — so best-of-K under fix F13's perfect scorer is arithmetic, not learning, and the old rule would have narrowed Stage D's claim on an artefact. The tested claim moved to plan §6.4's distinct-valid-set secondary. Six of its claims were corrected by a paper-by-paper audit (12 Aug 2026) that also amended fix F10, architecture rows S3/S4 and `CLAUDE.md` §8: F10's "connected output maps to a closed set" holds only under a graph mapping that makes a third of the bindings unselectable; the architecture asserted an approximation guarantee the source paper declines; and Gate 3 had no decision rule — the defect fix F12 retired for Gate 2. Needs nothing from Phase 3's runs except S5's row. |
| `GRAFT_PHASE5_BUILD.md` | **Live — the next component.** Stage A: ingestion, extraction, grounding, verification, the support gate, the learned obligation parser (fix F2), and the Gate-0 contract draft as its step 0. Eleven gaps (G1–G11); **§6 unsigned, nothing run.** Two gaps are measurements the spike already made: the 3B extractor's 15.5% JSON parse failure forces a predeclared extractor bakeoff (G2), and the full corpus does not fit this machine, so the corpus scope is Gate-0 item 9's decision, taken with G8's sizing memo. Cross-machine byte-identity is explicitly not promised for Stage A (G11). |
| `PHASE0_DECISIONS.md` | **Live.** What Phase 0 actually decided, including the eleven questions the build plan left to the implementation, and the multi-machine workflow rules. Read before changing a config default. |
| `README.md` | **Live.** Setup on a new machine, Kaggle, and what must match across machines. |
| `Research papers/` (103 PDFs, `INDEX.md`, `papers.csv`) | **Live.** Titles verified against publisher records. |
| `GRAFT2_9_EPISETFLOW_PIPELINE.md` | Superseded — the original design. Useful only as history; §4.1 lists what was cut from it. |
| `GRAFT2_9_EVIDENCE_REVIEW.md` | Historical — the literature audit that killed most of §4.1. |
| `GRAFT_MERGED_PLAN_EVIDENCE_CROSSCHECK.md` | Historical — the review round that produced v1.1 of the plan. |

Changelogs live in the docs themselves: research plan §1.7 (v1.1) and §1.8 (v1.2); architecture §1 flaw table F1–F13.

---

## 3. Working rules that produced these documents

These are habits, not policies, but they are why the docs hold up.

**Three labels, never mixed.** `[EVIDENCE]` = a named paper supports this, venue stated. `[HYPOTHESIS]` = this project tests it, no paper establishes it. `[ANALYSIS]` = engineering or mathematical judgment made here, not from literature. Infrastructure choices are mostly `[ANALYSIS]` and are labelled honestly rather than dressed up.

**Venue tiering.** Primary = ICLR/ICML/NeurIPS/ACL/EMNLP main/TACL/JMLR. Qualified = Findings, ECAI, SIGIR-AP, regional. Provisional = arXiv preprints and workshop papers — may motivate an experiment, never carry a central claim alone. Two load-bearing sources are provisional and flagged everywhere they appear: the evidence-packing study (arXiv 2607.00725) and Beyond Static Retrieval (arXiv 2509.25530).

**No paper establishes that this system will beat its baselines.** Papers justify components and motivate hypotheses. Superiority is established by controlled experiment or not at all. Every earlier draft that predicted wins had those predictions removed.

**Cite the verified published title, never the short name.** Six titles in the paper library were wrong before being corrected (ARM, TaG, HGERE, SEEM, SynCheck, and the packing preprint). Short names — SubTB, FL-GFN, LED-GFN, SEEM, ConEL-2 — belong in prose, not in a bibliography.

---

## 4. Decisions, with reasoning and cost to reverse

### 4.1 Ideas that were cut — and the specific reason each one died

These are the ones most likely to come back if the reasoning is lost. Each was in a serious draft.

| Cut | Why it died | If you want it back |
|---|---|---|
| **Dual epistemic mass** (separate exploratory flow `F_X` and authoritative flow `F_A`) | No precedent found anywhere in the GFlowNet literature. Two unresolved spec gaps: no authoritative source flow `Z_A` was ever defined, and reachability under certification-masking was never proved. The nearest published relatives (Multi-Objective GFlowNets, Distributional GFlowNets, GAFlowNets) all do something else. | It was the most *novel* idea in the project. Reviving it means writing the transfer law between channels and proving sub-DAG reachability. Budget several weeks; it is a research project of its own. |
| **`ABSTAIN` as a flow action with reward `R_abstain`** | The math is fatal. Under reward-proportional sampling, `P(ABSTAIN\|q) = R_abstain / (R_abstain + Σ_X R(X))`. Abstention probability therefore falls as the *number of alternative valid proofs* rises — even at identical answerability. A question with one valid proof abstains more than one with ten. No tuning fixes it; the denominator moves per question. | Only with explicitly normalized target masses over the answer and abstention groups. The decoupled gate that replaced it is simpler and removes a hyperparameter. |
| **FL-DB with the proof-deficit vector as "energy"** | FL-GFN (ICML 2023) Assumption 4.1 requires a **scalar** `ℰ: S → ℝ` whose terminal value equals `−log R`. A six-component vector is not that object. Worse: at a valid terminal all mandatory deficits are zero, so a deficit-only potential collapses to ranking proofs **by size alone**, and Proposition 4.2's guarantee silently stops applying — nothing crashes, the numbers just stop meaning what they claim. | Possible if you define `U` as a state function evaluable on partial sets and set `Φ(X) = β·U(X)` at terminals *by construction*. But proof sufficiency is a global non-additive property, which is precisely LED-GFN's stated regime. |
| **The `d(s)`-predicting auxiliary head** | `d(s)` is already an encoder input, so predicting it from `h_s` is a copy task. It teaches nothing about credit assignment, which is the entire claim of C3. | Kept as ablation arm **L7b** with a *forward-looking* target (terminal deficit reachable from `s`), which genuinely is not in the input. |
| **Order-invariant set states as a contribution** | Native to GFlowNets since 2021. GFlowNet Foundations (JMLR 2023) covers set-valued state spaces directly; set generation is a standard FL-GFN benchmark. Correct engineering, not novelty. | Don't. The honest version — "we make canonicalization explicit and measure equivalent-action collision" — is already in the plan. |
| **Redundancy term `D(e) = max(0, S(X\e) − S(X) + ε)`** | Near-vacuous by construction. If the proof score is monotone in evidence, `S(X\e) − S(X) ≤ 0` always, so `D ∈ [0, ε]` — an almost-constant shift, not a signal. | Replaced by facility-location coverage overlap, which is non-degenerate and connects to the submodular baseline. |
| **Eight independent prediction heads** | Duplicate / conflict / supersession all depend on the *same* claim pair and *same* time interval. Predicting them independently discards the fact that they are mutually exclusive, and produces contradictory outputs. | Now four decoders (D1–D4). Ablate D2 grouped-vs-split if you want the evidence. |
| **`R_floor` for dead-ended trajectories** | It put policy mass on an outcome the target `p*` did not contain (`p*` was normalized over valid terminals only), so exact TV/JS would have compared two distributions with different support and measured nothing. | Replaced by a single absorbing `FAIL` terminal that is a *member of the target support*. `FAIL` = "no proof found within budget" = the inference-time abstain fallback, so write-path and read-path semantics agree. |
| **Flow-Matching-style parent/child conservation** | FM is the weakest-credit objective in the literature. Trajectory Balance (NeurIPS 2022) shows FM/DB propagate credit inefficiently over long sequences; SubTB and FL-GFN both beat them. With 16 atoms per set (16! orders), this is FM's worst case. | Use TB/SubTB/LED-GFN, all of which are in the Tier-1 learner set. |

### 4.2 Decisions that were kept, and the evidence behind them

| Decision | Reasoning |
|---|---|
| **Frozen SLM at 3B (Qwen2.5-3B-Instruct)** | Not a cost compromise. The evidence-packing study found careful evidence selection helps a 3B reader (+0.022 F1, p<0.05), is null at 7B, and **reverses at 14B** (−0.029, p=0.013) — and its reader family was Qwen2.5, so the evidence transfers directly. A small reader is the regime where the minimality contribution matters. *(Provisional source, flagged.)* **If you swap in a bigger reader, the minimality benefit may vanish or invert.** |
| **Sample-then-filter portfolio** (K candidates → `H`-filter → rank) | Robust Scheduling with GFlowNets (ICLR 2023): diverse candidates sampled under a *cheap proxy* beat proxy-optimization when the true evaluator is expensive. That is exactly this architecture — learned utility = proxy, deterministic checker = expensive evaluator. **This is the single best argument for using a flow method at all**, and it should frame the paper's introduction. |
| **`H` contains no learned component, ever** | If a learned scorer can enter `H`, then `H` stops being a predicate, `1[H]` stops being a hard gate, and the multiplicative safety property degrades into a soft threshold. Entailment, sufficiency, authority and answerability are routed to `U`, the gate, and Stage B respectively — enforced as module boundaries, not discipline. |
| **Non-destructive versioning** (`t_invalid`, `superseded_by`; nothing deleted) | Zep's bi-temporal edge-invalidation model, whose temporal-reasoning gains were +38–48% relative over full-context. Also makes a wrong supersession recoverable. *(Vendor preprint — flagged.)* |
| **Open-world entity actions** `LINK_EXISTING / CREATE_NEW_ENTITY / NON_ENTITY / DEFER` | A single "NIL" head conflates a *real new entity* (must create a node) with a *non-entity phrase* (must create nothing). In a growing memory graph the first is the common case, and classifying it as NIL loses the entity permanently. Learn to Not Link (Findings ACL 2023) partitions unlinkable mentions into exactly Missing-Entity vs Non-Entity. |
| **Support gate before the active graph** | Restores the project's founding requirement. Without it: extractor invents a claim → NLI marks it unsupported → stored → committed → retrieved as evidence. Audit layer keeps everything; active layer takes only eligible assertions. |
| **Bounded pools** (`pool_cap=64`, `max_atoms=16`) | Not just cheap. Generalization and Distributed Learning of GFlowNets (ICLR 2025) gives data-dependent bounds that degrade with state-space size. |
| **Structural closure**: an atom is addable only once every atom it references is selected | Chosen over the alternatives because it is the cheapest (one membership test per reference), keeps every partial state structurally sound, and **proves the unconstructible-valid-terminal rate to 0** — nodes-first construction always works — turning a required measurement into a regression test. |
| **Build Phases 1–4 before the data pipeline** | They need zero real data, and they retire the highest-risk science first. Two kill-shots (Gate 2, Gate 3-synthetic) land by roughly week 7 instead of month 4. |

---

## 5. Errors already caught — and the pattern behind them

Recorded because the *pattern* matters more than the individual mistakes. All of these survived at least one review round before being caught.

| Error | Correction |
|---|---|
| `exp(β·0) = 1`, not 0 — the hard-failure reward was written as `U = 0` | The validity indicator must be **multiplicative**: `R = 1[H]·exp(β·U)` |
| A deficit *vector* was called an FL-GFN energy | FL-GFN requires a scalar with terminal consistency; see §4.1 |
| Graph-S3's headline gains were used to validate the FL-DB bridge — and were quoted at nearly double their published size (+15.6%/+17.2% against the paper's +8.1%/+9.7%) | Graph-S3 trains an **LLM** retriever with supervised stepwise supervision — no GFlowNet, no energy. It is a strong *baseline*, not a validation. The figures were corrected to the paper's on 13 Aug 2026, after an independent PDF re-read; the stale pair had survived one review round that fixed only its *interpretation* |
| "FL-GFN's set benchmark has the exact task shape" | It has fixed-size sets with additive energy; ours are variable-size, constrained, non-additive |
| The deterministic checker was called the "true evaluator" | Robust Scheduling's true metric was *measured hardware runtime*. A rules engine verifies only the rules encoded in it |
| SynCheck described as "free" | It needs a trained monitor and inference compute; FOD alters decoding |
| Self-RAG cited as evidence for answer abstention | It learns whether to *retrieve*, then critiques generated text. Withdrawn |
| A monotonicity proof was demanded for `H` under masking | Wrong — only `STOP` is masked; `ADD` stays available, so traversing an invalid partial set is fine. The obligation was withdrawn |
| Two citations were flagged as wrong that were correct | The CompGCN OpenReview ID and the COFlowNet proceedings URL were both fine. Verify before flagging |

**The pattern: overreaching on what a paper establishes, and asserting math without checking it.** Both are cheap to avoid and expensive to leave in. When something feels like it follows from a paper, check whether the paper's setting actually matches; when something feels algebraically obvious, evaluate it at a boundary case.

---

## 6. Frozen values and what changing each costs

These were frozen so that comparisons stay controlled. They are yours to change — the third column is the price.

| Value | Default | Cost to change |
|---|---|---|
| `beta` | 4.0 (swept on the synthetic lattice in Phase 3, then frozen). Candidates `{1, 2, 4, 8}`, but the argmin runs over the **eligible** ones only — those whose main-suite target-mass bands pass at that β. Measured on the frozen suite: `{4, 8}` eligible, `{1, 2}` not | Re-run everything after the sweep |
| `u_weights` | `{suff 1.0, cov 0.5, src 0.25, temp 0.5, red 0.25, size 0.1}` | Must be **identical across all seven learners**, or the comparison measures reward engineering rather than learning |
| `U` term ranges | all normalized to [0, 1] | β becomes uninterpretable; reward re-derivation |
| `size` term | `\|X\| / max_atoms`, one weight | Reward changes → all learners re-run |
| `r_fail` | 1e-6 | `FAIL` is in the target support; changing it re-derives `p*` and every Gate-2 result |
| `K` | 8 | Same constant is the "returned sets" count in the search comparison — change both or neither |
| `checker_budget` | 32 **terminal** `H` checks/query | The Stage-D primary metric is defined in these units |
| `pool_cap` / `max_atoms` | 64 / 16 | State-space bound; interacts with the generalization argument |
| `tau_nli` | 0.8 | Changes active-graph contents → Phase 6 onward |
| closure rule | references-before-atom | Masks, `H`, `P_B`, exact DP all change → Phases 1–4 re-run |
| seeds | {13, 42, 7} | Significance protocol invalidated |

**Hardware-coupled** (move with the machine, not the science): extractor size/quantization, reader precision, stage-sequential model loading, batch strategy. Phases 0–4 use no GPU models at all.

**Two counting rules that are easy to get wrong.** `terminal_checks` counts full validation of a *completed* candidate set only — construction-time validity is maintained incrementally and is free. And the budget must be **enforced**, not merely observed (`would_exceed()` before spending), or methods drift over budget and the comparison is unfair without anyone noticing.

---

## 7. What is not decided

| Open | Notes |
|---|---|
| **Eval and dataset sections** | **The selection exists; the integration does not.** `Research papers/INDEX.md` §7 already records it with roles — 2WikiMultiHopQA and MuSiQue for proof-set supervision, HotpotQA as the submodular/PCST testbed, DialogRE/ConEL-2/Re-DocRED/TORQUE/MATRES for D1–D4, MSC and Memora as additional benchmarks, plus a "still missing" list. What is missing is that none of it is wired into the research plan's §6.1 or the Gate-0 contract, and dataset *names* are not a contract: the public labels do not map exactly onto GRAFT's schema, which is why the architecture says "-style" and "supervision *interface*" everywhere. *(An earlier version of this row said the analysis "was never written into a document". That was wrong.)* |
| **Conflict/supersession annotation** | **No dataset provides it.** LongMemEval's knowledge-updates subset is closest and is small. This is the binding constraint on C1. *(Phase 2.5 exists for a slightly broader reason than an earlier version of this row said: the architecture names **D1 and D2** as "the two decoders with no off-the-shelf supervision", and Phase 2.5's stated purpose is measuring Gate-0 item 8 — how many annotations are feasible — rather than conflict labels alone.)* |
| **Wikipedia→conversation transfer** | Stage D trains on 2Wiki/MuSiQue-style proofs and is evaluated on conversational memory. A declared, untested claim. The training loader is deliberately source-agnostic so a conversational variant is a drop-in if it fails. |
| **`r_fail` after the β sweep** | If β moves substantially in Phase 3, the gap between `r_fail` and the valid reward range moves with it. One-line re-check, belongs in the β-sweep task — and `Target.at_beta` now performs it automatically. |
| **The terminal convention was never written down** | Plan §4.1 requires it: *"with a terminating edge to a sink, the flow on that edge equals `R`; with a terminal-state convention, the terminal state flow equals `R`. Pick one and write it down."* Nobody did. The code has picked one — measured on `tiny_instance()`, a state that is both a valid stop and has children gives `F(s) = 181.49` against `R(s) = 14.88`, so `R` is the flow on the terminating `STOP` edge. This matters now: Phase 3's `StateFlowHead` serves LED-DB and augmented TB, both stated over `F(s)`, and a loss assuming `F(X) = R(X)` would be wrong by the continuation flow — misreported as a decomposition failure. Recording it is three lines. |
| **The assumed hardware does not match the machine** | §0 above says "1× RTX 5090 (32 GB)"; the development machine is an **RTX 5050 Laptop GPU with 8 GB**. Phases 0–4 use no GPU models so nothing built is affected, but fix F7's VRAM budget (7B extractor + 3B reader + embedder) was reasoned against 32 GB. Unresolved because "Setup assumed" may name a target machine rather than this one. |
| **How close the main suite sits to its own `neither` band** | At the frozen β = 4 the twenty main-suite instances span 0.463–**0.499** against a hard band of 0.5 — one clears it by 0.001. A weight change, a feature change or a NumPy release that moves the `default_rng` stream could push an instance over and force a regeneration for no scientific reason. Two exits, both costly: record the fragility and accept a possible regeneration, or move the generator for headroom — a change to a frozen instrument at the full §6b price. **Not decided.** `PHASE2_DECISIONS.md` §4.2. |
| **Three baselines found late, not yet in the plan** | **HyperMem** (ACL 2026) reports 92.73% LLM-judge on LoCoMo — the closest structured-memory competitor. **Chain-of-Memory** (ACL 2026) gets 7.5–10.4% gains at ~2.7% of the token cost, which is the strongest published argument against building a heavy graph at all. **How Memory Management Impacts LLM Agents** (ACL 2026) empirically documents error propagation, upgrading that risk from `[ANALYSIS]` to `[EVIDENCE]`. All three are in the paper library; none are in the plan's §5.3 yet. |

---

## 8. Where the project is designed to fail cheaply

Four gates exist to kill bad directions early. They are the reason the build order looks the way it does.

| Gate | Question | If it fails |
|---|---|---|
| **Gate 0 / Phase 2.5** | Is the required annotation volume actually achievable? Measured, not estimated. | Narrow the schema or question types **here**, not later by weakening the evaluation |
| **Gate 1** | Does the learned graph constructor beat a GraphMixer-style MLP and LLM-prompted linking? | C1 is in trouble; GraphMixer (ICLR 2023) beat RNN/attention temporal GNNs, so this is not a strawman |
| **Gate 2** | Does checker-conditioned LED beat **capacity-matched** LED-GFN on exact TV at a fixed training budget, 3 seeds, paired test? | **C3 is not supported; consolidate on C1.** This is the cheapest place in the project to learn that |
| **Gate 3** | Does the learned sampler beat training-free PCST and submodular greedy at equal budget — **under a proxy scorer**? | Stage D's claim narrows before the expensive real-data phases begin. **Amended 12 Aug 2026:** the *synthetic* stage cannot ask this. With fix F13's exact-`U` scorer, greedy is globally optimal on 30/30 instances and a flawless sampler is 0.038 short at K = 8, so the lattice answers by arithmetic. Phase 4 is a diagnostic; the decision is Phase 9's, where the distilled head is noisy |

**The two baselines most likely to embarrass the project**, both training-free and both cheap to run: **submodular greedy** and **PCST** (G-Retriever, NeurIPS 2024 — returns a *connected* subgraph, structurally the closest classical algorithm to a proof). Run them early.

*(Corrected 12 Aug 2026. This row previously quoted the submodular packer at 0.451 F1 against 0.429, unconditionally. That is **one cell of arXiv 2607.00725's Table 6** — the only budget of four where the gap is significant, at a 3B reader. The others are −0.018 (p=0.08), −0.001 (p=0.90) and +0.013 (p=0.14); at 7B it is −0.010 (p=0.45); and the paper's own abstract concludes it "reaches parity, winning outright only where evidence density is the binding constraint". The research plan §5.2 always named the 160-token budget, so it was the compression **into this file** that dropped the condition. **It changes nothing about Gate 3** — the threat is a training-free method matching a *learned* one, which that table does not measure — but a cherry-picked cell is §5's own pattern, in §5's own document.)*

---

## 9. Claims audit — what the evidence supports

**Safe to claim:** open-world provenance-preserving graph construction as a *combination*; a formal-validity-gated reward with a decoupled gate; the five-ceiling protocol; that no prior GFlowNet work targets agent memory.

**Not safe:**
- That graph memory is better than flat memory. Mem0 (ECAI 2025) — its own graph variant *lost* on single-hop (65.71 vs 67.13) and multi-hop (47.19 vs 51.15), winning only on temporal (58.13 vs 55.51), at ~2× tokens and ~3.2× search latency.
- That the system beats full-context. In the same table, **full-context scored 72.90 — above every memory system tested.** Full-context is a non-negotiable baseline, and omitting it will read as evasion. Win on temporal reasoning, verifiability, abstention correctness, latency and token cost instead.
- That "learned incremental graph construction from text" is new. GATA (NeurIPS 2020) builds and updates a belief graph from raw text at every step, +24.2% on 500+ TextWorld games.
- Reward-proportional sampling, unless proved for the actual objective used *and* checked empirically on the enumerable environment.
- Any predicted margin over any baseline.

**The motivating number for the whole project:** ALCE (EMNLP 2023) — on ELI5, even the best models fail to fully support their claims **50% of the time**. And VeriCite's ablation shows verification, not generation, is what fixes it: citation F1 77.73 → 68.91 with the NLI check removed, while answer correctness barely moved.

---

## 10. Working agreements

Carried over because they kept the documents honest:

- **Verify citations before relying on them.** Every venue and title in `Research papers/INDEX.md` was checked against the publisher record. Two "errors" I reported turned out to be my mistakes, not the library's.
- **Label evidence honestly.** Infrastructure decisions are `[ANALYSIS]`. Saying so costs nothing and protects the parts that really are paper-backed.
- **Say what is not known.** The docs contain explicit "no evidence found" entries. Those are findings, not gaps.
- **Scope control is a real risk, not a formality.** An earlier draft specified ~15 learning configs, 10 search methods and 7 systems — undeliverable by one person. The Tier 1/2/3 split exists for that reason and only works if Tiers 2 and 3 stay deferred.
- **Do not over-engineer.** The architecture has an explicit boring-stack rule: one process, one GPU, no database, no ANN index, no experiment platform. Anything outside it needs a written justification.
