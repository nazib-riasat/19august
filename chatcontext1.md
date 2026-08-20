# Session handoff — GRAFT, Phases 0–11 (+2.5)

**Rewritten 11 Aug 2026; amended 13–16 Aug 2026; amended 19 Aug 2026 (last:
Phases 10 and 11 built, LoCoMo loaded and verified).** Read this plus `CLAUDE.md`
and you can continue without re-deriving anything.

**19 August 2026 — read this before §1, which predates it.** The project is now
built through **Phase 11**, and the remaining work is *runs*, not code:

* **LoCoMo is in place and verified.** `data/locomo/locomo10.json`, SHA pinned in
  `graft/ingest/locomo.py`, checked on every load. `scripts/locomo_ingest.py
  probe` reported **zero findings**: 10 conversations, 1,986 questions, 446
  adversarial, 5,882 turns, every timestamp parsed. The 446 independently
  **verified** the category mapping in `graft/baselines/categories.py`, which had
  been an unverified convention hours earlier.
* **Phase 11 exists**: `GRAFT_PHASE11_BUILD.md` + `PHASE11_DECISIONS.md`. It is a
  **reference-table comparison, not Gate 4** — published Mem-T numbers
  (arXiv 2601.23014v2 Table 2) are quoted rather than re-run, at the project
  owner's instruction on a three-day deadline. Read `GRAFT_PHASE11_BUILD.md` G1.
* **The end-to-end runner now exists.** `scripts/phase10_read.py` drives the read
  path over *hand-built fixtures*, which is what made R1–R3 wiring tests;
  `scripts/locomo_eval.py` drives it over a real ingested graph, and
  `graft/tests/test_locomo_eval.py` exercises the whole join on a stub reader.
* **The critical path changed.** It is **no longer P9.7.** The Phase-9 policy
  ladder (~2 h per arm per seed) answers Gate 3 and is *not* needed for an
  end-to-end number. What Stage D needs to rank at all is the **distilled utility
  head** (`PHASE10_DECISIONS.md` §1.4), and `scripts/train_head.py` fits it in
  **minutes of CPU**. Both are deferred-or-cheap; the real cost is ingestion.
* **Run order and costs** are in `GRAFT_PHASE11_BUILD.md` §9, with a
  per-conversation stopping table. Headline: ~3.1 GPU h per conversation,
  ~35 min per 1,000 questions of reader time, and both stages resume per unit.
* **Suite: 1,304 passed, 0 failed;** `scripts/check_plan_consistency.py` clean
  over 32 live documents.

Sections §1–§7 below are the 11–16 August state and are still accurate about
Phases 0–9. Where they say the critical path is P9.7, the bullet above supersedes
them.

**If you read only one paragraph.** Every phase through 9 is built and green.
**Phase-3 step 6, the calibration gate, RAN 15 Aug 2026** (CPU cluster job
985953): **β = 4.0 and `N` = 1,997,088 adopted at rung 0**, and the Gate-2
matrix ran in the same job — outcome **`inconclusive`** on two instrument
clauses (not a negative for C3; `PHASE3_DECISIONS.md` §7). β is frozen,
`artefacts/phase3_calibration.json` is in place, and Stage-D training is
mechanically unblocked — so the critical path is now **Phase-9 step 6's runner
(P9.7, unwritten) plus GPU time and human annotation.**

**Two habits this project runs on, before you touch anything.** Nothing is
committed — work lives in the working tree by standing instruction. And every
phase is adversarially audited after it is built: the audits have repeatedly
found blockers that reading did not, including two that would have produced a
**false negative for Contribution 3** with no symptom in any loss curve. Budget
for the audit as part of building, not as an optional extra.

---

## 1. Where the project actually is

| | State |
|---|---|
| **Phase 0** — scaffold, schemas, event log, ledger, config | **Built, green.** All 13 exit criteria. |
| **Phase 1** — `H`, `U`, `R`, masks, obligations, `d(s)` | **Built, green.** All 16 exit criteria. |
| **Phase 2** — ProofLattice + exact evaluator | **Built, green.** All 21 exit criteria. |
| **Phase 2.5** — annotation feasibility spike | **Tooling built and run 13 Aug 2026** (`scripts/phase2_5/`, `data/phase2_5/`): corpus pinned, 58 turns extracted with Qwen2.5-**3B** (ruled deviation from the 7B), G7 span floor passed at 0.85, items + guidelines + machine-assisted bootstrap labels exist. **Item 8 answered 15 Aug 2026 — GO** (§3a): 16.7 h for 3,625 items, D1 κ 0.829 / D2 κ 0.813, both real inter-annotator agreement. Guidelines v1 supersede v0; the spike's own batches are archived, their candidate ids having resolved to nothing in the real graph. |
| **Phase 3** — 9 learner arms + Gate-2 harness | **Code complete; CALIBRATED AND MATRIX RUN 15 Aug 2026** (job 985953): step 6 adopted rung 0 — β = 4.0, `N` = 1,997,088; the 27-run Gate-2 matrix returned **`inconclusive`** (criterion 15 and L6's band failed — instrument clauses, so null, never negative; `PHASE3_DECISIONS.md` §7). A 13 Aug audit added five harness guards (tv_threshold pinned, truncation refused, realized-spend instrument clause, probe-β check, `budget_for` 52 evals) — all latent-hazard fixes, no result invalidated. |
| **Phase 4** — five Tier-1 search methods + Gate-3 harness | **Stage A built, green and RUN (13 Aug 2026); Stage B unblocked 15 Aug 2026** — Phase 3's matrix ran and its 27 checkpoints sit in `artefacts/checkpoints/`; the Stage-B composition has not yet run and S5's arm-selection rule is unruled (`PHASE3_DECISIONS.md` §7.5). `graft/setgen/search/`. **Three sub-decisions of the ruled §6 table were overturned by measurement** — `PHASE4_DECISIONS.md` §1. |
| **Phase 5** — Stage A ingestion | **Built, green, and run (14 Aug 2026)** — `graft/ingest/`, Gate-0 contract drafted 9/10, **G2 bakeoff frozen on candidate B** (1.7% parse failure vs A's 23.3%), **live pilot 248 turns end to end**. Every machine-measurable exit criterion met. A two-source audit confirmed 19 defects + 1 found in-house — two of them blockers in the bakeoff harness itself — all fixed and regression-tested (`PHASE5_DECISIONS.md` §7); the bakeoff's first run was aborted and its instrument corrected before anything froze (§1.6), and the token cap was raised once more under a written-down prediction after a `no_survivor` verdict (§2.1a/b). **Remaining: four human audit worksheets + Gate-0 item 8.** Two findings to carry forward: **a third of extracted assertions fail to ground** (108/333) and the **quarantine rate is 32%** — the NLI audit is the instrument that separates "extractor overreaches" from "0.8 is strict". |
| **Phase 6** — Stage B graph construction | **Built, audited, green (14 Aug 2026)** — `graft/graphbuild/` (validate, items, candidates, embed, features, encoders, decoders, loaders, llmlink, commit, standin, gate1). A two-source audit confirmed **21 defects (2 blockers)**, refuted 8; all fixed and regression-tested (`PHASE6_DECISIONS.md` §7). Smoke run rebuilt: 187 D1 items (177 with construction-time candidates), 120 D2 items, 130 edges, corruption audit green over real commits. **The trainer (P6.11) is now built** and the decisive path exists; a review of it found 9 more defects (2 blockers — gold labels joined by a non-durable id, and every LINK label naming the spike's entity namespace), all fixed (`PHASE6_DECISIONS.md` §8). Gate 1 waits on the **Gate-0 signature alone**: the pilot batch (15 Aug) supplies 40 D1 + 49 D2 human labels in the graph's own namespace (40/40 span-join, 9/9 LINK reachable), and a second audit fixed 8 more defects (`PHASE6_DECISIONS.md` §8.5 — incl. a cwd-dependent test set and a filename-sort gold collision). |
| **Phase 7** — Stage C retrieval | **ALL SEVEN STEPS BUILT, green and run (15 Aug 2026)** — `graft/retrieve/` (pool, entity, temporal, expand, bm25, dense, fuse, recall, **scorer**, pins) + `scripts/phase7_retrieval.py`. `bm25s` adopted (G2), `graft.retrieve` is the fourth ML-allowed package with a **stricter** containment guard (non-scorer modules import no torch at all). **The scorer is built but NOT trained** — 237,443 params against decision 9's 8M cap, all three P6.11 guards inherited, `--scorer` loads a checkpoint or the stack runs on five channels and says so. **Two findings, both in `PHASE7_DECISIONS.md` §3**: Tier-A recall is 1.000 on 9/10 pilot questions *by arithmetic* (8–23 candidates per conversation against `pool_cap = 64` — the pool is the whole conversation), and the entity channel matched an anchor in only **3 of 10** questions — which Phase 5's §5a audit then traced upstream to the parser (`entity_anchor` exact 17%), settling it as a fix-F2 problem rather than a decision-7 one. **Audited the same day — `PHASE7_DECISIONS.md` §7**: the external audit's 8 claims all confirmed (2 blockers: per-channel scores discarded against architecture §9.1; the scorer wired to score the already-capped pool), ~20 further defects from an 8-agent sweep, 15 suspected channel defects refuted by probe; all confirmed defects fixed, the artefact was a smoke run and is regenerated on the pinned embedder, the Stage-C fingerprint moved (layers pinned), and the determinism digest is verified equal across two runs. |
| **Phase 8** — answerability gate | **§6 SIGNED and Stage A BUILT, green and run (15 Aug 2026)** — `graft/gate/` (pins, features, labels, adapt_musique, model, riskcov, decide) + `scripts/phase8_gate.py`, 32 tests. `graft.gate` is the **fifth** ML-allowed package with the tightest containment yet: torch reaches `model.py` and nothing else. **MuSiQue-Full downloaded and SHA-pinned** (official archive, CC BY 4.0, 49,628 rows verified). **Three findings in `PHASE8_DECISIONS.md` §3, and §3.3 is the one that changed a result**: min–max normalisation is scale-invariant per question, so it made **10 of 13 channel features constant** and the first real run sat at chance (0.52) while AURC looked healthy (0.05); reading **raw** scores instead took within-pair accuracy to **0.67–0.74** (+0.22, ~14 SE). Recorded as a decision-3 amendment — the fusion arithmetic is untouched, `raw_channel_scores` was added *alongside* it, and the stage-G fingerprint moved. §3.1: MuSiQue *substitutes* distractors rather than deleting, so `pool_shape` is non-discriminative there. §3.2: an AURC gap between the arms is **not** evidence of leakage — the build's own note said it was, the measurement refuted it, and the fix was a better instrument (`contrast_pair_accuracy`). **Stage B deferred by name** (criterion 13, needs scope-c). |
| **Phase 9** — Stage D on real data | **BUILT (16 Aug 2026), §6 SIGNED, twice audited.** `graft/setgen/` gained `pins, featurenames, proofs, atomfeat, realenv, distill, portfolio` + `corpora/{scoring,wiki2,musique_ans}`, plus `scripts/phase9_measure.py`. **Build steps 0–5 done, corpora fetched and SHA-pinned, `N_real` derived.** The incremental environment runs over Phase-1's `IncrementalChecker` and **all nine arms train on real pools with `graft/setgen/learners/` byte-identical** — fix F6's payoff, asserted by a `git diff` test. **`N_real = 200,000`**, derived from the slowest arm (`l7b_aux`, ~30 traj/s) and now using ~93% of the 2 h ceiling. **§6 was signed with zero Phase-9 code in existence**, which is the cleanest §6b position the project has had — decision 7, the Gate-3 (real) rule Phase 4 deferred here, was fixed with nothing to have peeked at. **Two audit rounds: §3a (16 confirmed findings, 2 blockers) and §7 (5 blockers, 3 external + 2 the sweep found alone).** Read `PHASE9_DECISIONS.md` §7 before quoting any Phase-9 number, and §1 before touching `portfolio.py`, the anchor rules or the collision instrument — **four signed decisions were overturned at the build**. **Step 6, the scored Stage-A run, is the only thing left; β froze 15 Aug 2026 and `training_blocked_reason()` now returns `None` — what remains is writing and running the P9.7 runner (~50 CPU-h serial, ~6 h across 16 cores).** |
| **Phase 10** — Stage E: reader, orchestrator, ceilings | **Stages A, B and C BUILT, green, and run (16 Aug 2026)** — `graft/reader/` (pins, serialize, parse, read, orchestrator) + `graft/diagnostics/ceilings.py` + `scripts/phase10_read.py`, 43 new tests. `graft.reader` is the **sixth** library-allowed package with the tightest containment yet (torch reaches `read.py` alone); `graft.diagnostics` deliberately **not** admitted, so Contribution 4's instrument stays computable on a bare interpreter. **Runs R1, R2, R3 complete** — reader determinism **bit-identical**, peak video memory **6.317 GB on an 8 GB card** (fix F7 is a live constraint), and the end-to-end smoke ran in 51 s. **Read `PHASE10_DECISIONS.md` §1 before quoting anything**: citation markers were counting as answer tokens (systematic, one-directional, on every cited answer), the prompt asked for a sentence where gold is a span, and the runner's "gold proof" was two content-hash-ordered atoms making a reported ceiling into noise. All fixed. §4 has the five-ceiling decomposition attributing the whole end-to-end gap to Stage D. **Stage D deferred by name** on Phase-9 step 6, scope-c and Phase-8 Stage B. |
| Phase 11 | Not started. Thin baseline adapters; build only when Gate-4 comparisons are scheduled. |

**Phase 10 build state (16 Aug 2026):** **Stages A, B and C built and green; runs R1, R2 and R3 have run.** `graft/reader/` gained `pins, serialize, parse, read, orchestrator`; `graft/diagnostics/ceilings.py` carries all five ceilings; `scripts/phase10_read.py` is the runner. **`Config` gained `serialization_budget_tokens`** — G1's finding that ceiling 4 had *no denominator anywhere in the project*. `graft.reader` is the **sixth** ML-allowed package with the tightest containment yet (torch reaches `read.py` alone), and `graft.diagnostics` was deliberately **not** admitted so Contribution 4's instrument stays computable without a GPU. **The stage-E fingerprint is the seventh** in `verify_handoff.py`. **No training anywhere in the phase** — the reader is frozen and SynCheck is declined with its cost recorded (decision 11).

R1 measured greedy decoding **bit-identical** across three repeats and the reader at **6.317 GB peak on an 8 GB card**, so fix F7 binds and `ModelSlot` refuses a second concurrent model. R3 is the architecture's exit criterion and **demonstrated Contribution 4 working**: ceiling 5 = 1.0 while end-to-end answered 5/10, the whole gap attributable to Stage D's untrained policy — including one query where a **one-atom formally-valid set** was correctly declined as insufficient, which is v1.2 §4.1's naming discipline observed in the wild.

An adversarial audit confirmed **8 findings, 2 blockers** (`PHASE10_DECISIONS.md` §5) — **18 of 32 verifiers were cut short by a session limit, so 8 is a floor, not a total.** Read `PHASE10_DECISIONS.md` §1 before quoting any Phase-10 number. **Stage D is deferred by name** on three blockers: Phase-9 step 6 (a trained Stage-D policy), scope-c ingestion, and Phase-8 Stage B (a conversational gate threshold).

**1,185 tests, ~4 min, all passing. Nothing skipped, nothing xfailed.**
`scripts/check_plan_consistency.py` runs inside the suite over **28 live
documents with 111 retired wordings**. Each phase's DECISIONS file carries its
own record; the retirements are how a corrected wording stays corrected.

**Six fingerprints** must agree across machines for two numbers to be comparable:
`config_hash`, ingestion, stage-B, stage-C, stage-G (gate) and **stage-D**
(Stage D). `scripts/verify_handoff.py` prints all of them; with
`artefacts/phase3_calibration.json` in place (15 Aug 2026) it reports Stage-D
training as **unblocked**.

**Three things to know before touching Phase 4** (all in `PHASE4_DECISIONS.md` §1,
all measured, none reading a learner result):

1. **`pcst_fast` is installed and must not be used.** Its `cp311-win_amd64`
   wheel returned the wrong optimum on **59 of 60** random graphs — every output
   array the right length with every element equal to its first. S4 uses an exact
   solver; the library survives only in the regression test that documents the bug.
2. **`C_e`'s selection moved to the breach rate.** "Median closest to `max_atoms`
   without exceeding it" admits a median *equal* to the cap: at `C_e = 0.5` the
   median is 8.0 = `max_atoms` and **half** the outputs are rejected on size.
3. **Decision 5's one live Gate-3 condition is size-confounded.** Mean pairwise
   Jaccard distance falls monotonically with set size even for *random*
   portfolios, and S4-informed already scores 0.483 against `p*`'s own 0.4506
   (pinned duplicates-included convention) — so S5 cannot beat it whatever it
   learned. The ruled metric ships; a size control ships beside it.

**Read `PHASE3_DECISIONS.md` §6 first if you are touching a learner.** An
external review on 12 Aug 2026 found three defects in *controls* — GAFlowNet's
Eq. 4, LED's Eq. 5 and its decomposition-error redistribution, and the L6/L7
capacity match — and **all three ran in the direction that flatters the proposed
method**. They are fixed. The convergence and capacity tables in §2.2 and §2.4
predate the fixes and are marked stale.

The suite got slower because Phase 3 added `test_setgen_convergence.py`, which
actually trains: it is the only place a *learned* quantity is asserted, and steps
2–5 exist precisely because a broken sampler or adapter yields plausible curves
and meaningless numbers.

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pytest -q
python scripts/check_plan_consistency.py
python scripts/verify_handoff.py --preset synthetic
```

Python **3.11** (`py -3.11`); the system default here is 3.14, which PyTorch has
no wheels for. `.venv` is never copied between machines — `PHASE0_DECISIONS.md`
§3.1.

**Git state.** `master`, no remote, linear solo history.

---

## 2. Read these, in this order

**Tier 1 — you cannot work without these (~1 hour).**

1. `CLAUDE.md` — why decisions are what they are, what was cut, cost to reverse.
   §4.1 (ideas that died and why), §5 (errors already caught), §6 (frozen values)
   and §8 (the four gates) are the load-bearing sections.
2. `GRAFT_RESEARCH_PLAN_v1.md` (v1.2) — the science. **Wins any conflict with
   every other document.** §4.1 (reward), §4.3 (masking), §4.5 (local credit),
   §5.1 (matched baselines), §6.3 (five ceilings), §7 (the gates).
3. `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (v1.1) — 12 phases in build order, and
   the F1–F13 fix table that phase plans cite constantly.
4. `GRAFT_PHASE9_BUILD.md` + `PHASE9_DECISIONS.md` — **the thing being built
   right now** (Stage D on real data). §6 of the plan is normative and signed;
   **`PHASE9_DECISIONS.md` wins conflicts with it**, and four of its signed
   decisions were already overturned at the build. Read `PHASE9_DECISIONS.md`
   §7 before quoting any Phase-9 number — a second audit found five blockers,
   one of which had handed a *control* less capacity than the proposed method.
5. `GRAFT_PHASE3_BUILD.md` — §6 is normative, signed, and **now filled**: step 6
   ran 15 Aug 2026 and decision 4's cell carries β = 4.0 and `N` = 1,997,088.
   Nothing stands between Phase 9 and a scored run but the P9.7 runner.

**Tier 2 — read before touching the corresponding code.**

5b. `PHASE0_DECISIONS.md` — before changing a config default.
6. `PHASE1_DECISIONS.md` — before touching `H`, `U` or `d(s)`.
7. `PHASE2_DECISIONS.md` — before touching the generator, `Target` or
   `ActionPolicy`. §4.2 (the β eligibility finding) and §5 (three post-build
   fixes) matter most.
7b. `PHASE3_DECISIONS.md` — **before touching any learner.** §7 first (the
   15 Aug 2026 run record: gate adopted, Gate 2 `inconclusive`), then §2.3
   (whose warned risk — an `N` sized on L4/L5 leaving slower arms short — is now
   measured fact on the main suite) and §4.
8. `GRAFT_PHASE0_BUILD.md` / `GRAFT_PHASE1_BUILD.md` / `GRAFT_PHASE2_BUILD.md` —
   the gap sections (G*n*) explain why each module is shaped as it is.
9. `README.md` — setup on a new machine, Kaggle, what must match across machines.
10. `Research papers/INDEX.md` — 103 papers with verified titles and venues.
    **Also holds the dataset selection** (§7), which is the thing `CLAUDE.md` §7
    wrongly says was never written down.

**Tier 3 — historical. Do not treat as live.**

`GRAFT2_9_EPISETFLOW_PIPELINE.md`, `GRAFT2_9_EVIDENCE_REVIEW.md`,
`GRAFT2_9_EPISETFLOW_EVIDENCE_AUDIT.md`, `GRAFT_MERGED_PLAN_EVIDENCE_CROSSCHECK.md`.
These describe superseded designs and *should* contain retired wording;
`scripts/check_plan_consistency.py` excludes them for that reason.

**Code, in dependency order.** `graft/schemas.py` → `graft/config/schema.py` →
`graft/core/{checker,masks,obligations,utility,reward,incremental,resolve}.py` →
`graft/synth/{lattice,enumerate,exact,policies,audits}.py` →
`graft/setgen/{features,policy,rollout,flgfn_probe,trainer,gate2}.py` →
`graft/setgen/learners/*.py` → `graft/ingest/` → `graft/graphbuild/` →
`graft/retrieve/` → `graft/gate/` → **`graft/setgen/{pins,featurenames,proofs,
atomfeat,realenv,distill,portfolio}.py` + `graft/setgen/corpora/`** (Phase 9).
Every module's docstring carries its own reasoning; they are not decoration and
are the fastest way in.

**Start with `trainer.py`'s docstring** — it defines the `g[i]` coordinates all
four flow objectives are written in, and none of the learner files makes sense
without it. **Then `realenv.py`'s**, which is how those same objectives run on
real pools without a single learner file changing (fix F6's payoff, asserted by
a `git diff` test).

**The five ML packages and their containment.** `setgen`, `ingest`, `graphbuild`,
`retrieve`, `gate` are the only packages allowed to import torch, each with a
guard in `test_structure.py`. Phase 9 tightened its own: `atomfeat`, `realenv`,
`distill` and `portfolio` are held to the *learner* standard (they may not reach
the synthetic environment at all) rather than exempted as adapters — only
`proofs.py` is exempt, because it constructs the real `AtomPool`.

---

## 3. What to do next

**Updated 16 August 2026: Track A is DONE.**

Every phase through 9 is built. **The calibration gate and the Gate-2 matrix ran
15 Aug 2026 on a CPU cluster (job 985953)** — β and `N` are frozen and
transcribed; Gate 2 is `inconclusive` on instrument clauses
(`PHASE3_DECISIONS.md` §7). The critical path is now **Phase-9 step 6's runner**
(Track G) plus the GPU and human tracks below.

| Track | State | Next action |
|---|---|---|
| **A — Phase 3 calibration** | **DONE 15 Aug 2026** (job 985953) — adopted rung 0: β = 4.0, `N` = 1,997,088; the matrix ran too, `inconclusive` on instrument clauses (§7) | transcribed 16 Aug 2026; nothing left on this track |
| **B — Phase 5 GPU** | pilot-scale done; scope c decided (200 q, 32.3 h) | the scope-c re-ingestion — GPU time only, nothing blocks it |
| ~~**C — human**~~ | **CLOSED 15 Aug 2026** — item 8 GO, item 9 decided, Gate 0 signed, four worksheets filled | — |
| **D — Phase 6** | code complete, twice audited | Gate 1 needs D2 labels + D1 link labels re-annotated against the 187-item batch |
| **E — Phase 7** | all seven steps built, green, run, audited | a scope **with distractors**; without it the recall instrument measures nothing (`PHASE7_DECISIONS.md` §3.1) |
| **F — Phase 8** | §6 signed, Stage A built + run (pair accuracy 0.78–0.80, AURC 0.038–0.041) | Stage B, deferred by name — needs scope-c |
| **G — Phase 9** | **§6 signed; build steps 0–5 done, corpora fetched, `N_real` derived. Twice audited.** | **step 6, the scored Stage-A run — UNBLOCKED** (Track A done; `training_blocked_reason()` returns `None`). Write P9.7 (`scripts/phase9_gate3.py`) and run it: ~50 CPU-h serial, ~6 h across 16 cores |

**What Phase 9 already has, so a fresh session does not re-derive it:**

* **Both Tier-1 corpora on disk and SHA-pinned.** 2WikiMultiHopQA fetched from
  `Alab-NII/2wikimultihop`, licence **re-verified at download as Apache-2.0**
  (COLING 2020); MuSiQue-Ans extracted from the kept archive (TACL 2022,
  CC BY 4.0). Both under `data/phase9/raw/` (gitignored), registered in
  `graphbuild.pins.DATASETS`, read through the one SHA-verified loader.
* **All nine arms train on real pools with `graft/setgen/learners/`
  byte-identical** — asserted by `git diff` in `test_setgen_real.py`.
* **`N_real = 200,000`**, derived from the slowest arm (`l7b_aux`, ~30 traj/s)
  against the 2 h ceiling — and it now uses **~93% of that ceiling**, so the
  headroom is gone: any further per-trajectory cost drops the rung to 100,000.
* **A sixth fingerprint.** `verify_handoff.py` prints stage-D beside the other
  five and, with the calibration artefact in place, reports Stage-D training as
  unblocked.

**Two things Phase 9 deliberately does not have.** Exact TV — it needs an
enumerable state space, so `RealTrainer.exact_tv()` **raises** rather than
returning an approximation, because an estimate printed in the column Gate 2's
exact numbers occupy would invite exactly the comparison that is invalid. And a
C3 verdict — L7/L7b vs capacity-matched L6 and GAFlowNet is *supporting
evidence* here; the confirmatory test stays at Gate 2.

**Track A does *not* contend with the GPU tracks** — `TrainSpec.device` defaults
to `"cpu"` ([trainer.py:139](graft/setgen/trainer.py:139)) and
`DATASET_DECISION.md` §4 lists Phase 3's GPU usage as "nothing", so the
calibration gate's ~12 h are **CPU** hours and its 39 runs are independent
(~3 h wall clock on 16 cores). *(An earlier version of this table said A and D
contend for the same 8 GB card and cannot run simultaneously. That was wrong, and
it was costing free parallelism.)* C is human time and contends with nothing.

**So A and B can run at the same time, and should.** A is CPU-only and B is
GPU-only; running them together is the shortest path to both Gate 2 and a
scored Stage D.

### 3a. The annotation pass — DONE, Gate-0 item 8 is GO (15 Aug 2026)

**Item 8 is measured and the stop condition does not fire.** Four annotators:
Sabbir and Nazib on D1, Meherin and Sakib on D2.

| | Required | Measured rate | Hours (slower annotator) |
|---|---|---|---|
| **D1** | 625 test + 1,500 train = 2,125 | 228-735 items/h | **9.3 h** |
| **D2** | 300 test + 1,200 train = 1,500 | 203-352 items/h | **7.4 h** |
| | | | **16.7 h; 8.4 h each split two ways** |

Costed at the *slower* annotator on each decoder, not the mean. 18.1 h at the
pessimistic end of the power grid. **No scope reduction is required on
feasibility grounds.**

**Agreement — item 7, and it took three attempts.**

| Attempt | D1 | D2 | What it established |
|---|---|---|---|
| v0 guidelines | κ 0.262 | κ 0.179 | **failed.** Five of ten D1 disagreements shared one cause: a rule v0 never stated |
| v1 guidelines | **κ 0.829** | κ 0.517 | D1 passes; D2's batch was contaminated (16/20 pairs seen by one annotator only) |
| v1, fresh pair | — | **κ 0.813** | D2 settled, class-varied batch, neither annotator had seen any of it |

Both finals are **real inter-annotator agreement between two people**, so
`GATE0_CONTRACT.md` item 7's old caveat ("say self-agreement, never IAA") no
longer binds.

**A fourth D2 measurement exists on disk and must NOT be quoted.** The
`*_clean` batch gave raw agreement 0.850 with **κ exactly 0**: one annotator
used a single label on all 20 items, so chance agreement equalled observed and
κ collapsed. That is the kappa paradox — zero class variance, not disagreement.
It is why the decisive D2 run used the class-varied batch instead.

**Guidelines v1** (`GUIDELINES_D{1,2}_v1.md`) supersede v0 and carry five rules,
each followed by the disagreement that produced it. The load-bearing one, which
v0 never stated: *the candidate list is the graph's memory, not the annotator's*
— if the entity is in the list, link, regardless of whether you saw it created.

**Gold sets:** `labels/d{1,2}_labels_adjudicated_*.jsonl`,each row carrying its own
provenance (`both_agreed` / `adjudicated`). **Caveat recorded in every
adjudicated row and both guideline docs: the adjudications were
assistant-derived and human-accepted, not resolved item-by-item by both
annotators.** Weaker than item 7 specifies.

**The first attempt is archived, not deleted** —
`data/phase2_5/archive_pass1_v0/` and `labels/superseded/`, each with a README.
Its labels are **not** carried into gold: v1 changes the answer to whole classes
of item (rule 1 alone flips 10 of the original 40 D1 labels), and mixing two
labelling standards behind one gold column would be worse than having fewer
labels.

**Two tooling flags on `scripts/phase2_5/annotate.py`:** `--items <tag>` selects
a named batch so two batches' labels cannot be mixed; `--second-annotator
<name>` records a different person's pass under their own name, reports
inter-annotator rather than self-agreement, and drops the 2-day gap check
(independence comes from being two people). A markdown form
(`D2_ANNOTATION_FORM.md`) exists for annotators without the CLI — it produces
labels but **no timing**.

**Two findings that are not about the annotators**, both for Phase 6:

1. **The pair proposer surfaced zero `CONFLICT` across 49 items**, 34 of them
   knowledge-update by design. D2's rare classes are the binding constraint on
   C1, so a pool that does not surface them is a Gate-1 problem arriving early.
2. **A question is stored as an assertion** — *"Can you provide more information
   on social identity theory?"* appears as claim text in a D2 pair. A question
   asserts nothing; this is a Stage-A extraction defect polluting the pair pool.

### 3b. The corpus scope — DECIDED, item 9 is closed (15 Aug 2026)

**Scope c, 200 questions — 4,384 turns, 32.3 h on the dev GPU (6.9 h on a
5090).** Costed alternatives: full corpus 1,820 h (not a candidate at any scale
here) · evidence sessions only 80.8 h · evidence + 2 distractors 153.3 h. One
constraint fixed throughout: the knowledge-update evidence sessions are in every
candidate scope, because that is where D2's supervision lives.

**Extend to scope b′ (evidence + 2 distractor sessions, 153.3 h) for the final
numbers**, calendar and hardware permitting — full reasoning in
`DATASET_DECISION.md` §5 and `GATE0_CONTRACT.md` item 9.

**Why the extension is not optional decoration.** Phase 7's own measurement gives
the concrete reason: on the pilot's scope-b-shaped graph, every conversation held
8–23 eligible candidates against `pool_cap = 64`, so every pool was the whole
conversation and Tier-A recall read 1.000 by arithmetic — measuring no retrieval
at all (`PHASE7_DECISIONS.md` §3.1). Scope c alone would very likely reproduce
that at the decisive scale, since it is still evidence-only; distractor sessions
are the one lever that changes it.

**What is and is not done.** The scope is *decided*; ingestion at that scope has
**not** run. `GATE0_CONTRACT.md`'s Sign-off table records the decision, and the
contract **signed the same day** (delegated, recorded as such) — Gate 1 is
unblocked.

---

**Step 6, the calibration gate — RUN 15 Aug 2026 (job 985953), non-`--quick`:
adopted at rung 0, β = 4.0, `N` = 1,997,088; the matrix followed in the same job
and returned `inconclusive`.** The run record is `PHASE3_DECISIONS.md` §7; what
follows below is the pre-run briefing, kept as the prediction record.

```bash
python scripts/phase3_calibrate.py --out artefacts/phase3_calibration.json
```

The script implements decision 4's ladder in decision 4's order: β eligibility →
`N` from the wall-clock ceiling by an L5 throughput pilot on the **tuning** suite
→ β sweep at that `N` → freeze β → the L4/L5 sanity check at decision 6's 0.10.
Pass ⇒ adopt; fail ⇒ next rung; fail at the last ⇒ **Gate 2 inconclusive**, never
a negative verdict on C3.

**Rung 0 should adopt — measured forecast, 12 Aug 2026.** L5 on the tuning
suite, seed 13, at the default β: TV **0.386** at 100k trajectories, **0.191** at
400k, **0.0990** at 1.2M. So L5 crosses decision 6's 0.10 at ~1.2M, and rung 0's
ceiling buys **N ≈ 4.0M** — 3.3x headroom. The ladder should not escalate, which
puts calibration at ~4.5 h rather than the ~31 h a walk to rung 2 would cost.
A scheduling forecast only: one seed, one arm, and decision 6 needs L4 **and** L5
averaged over three. It reads L5 on the tuning suite, which is exactly the read
the ladder itself performs, and decision 6 was already ruled before it was taken.
*(Outcome, 15 Aug 2026: rung 0 did adopt — but at `N` = 1,997,088, not ≈ 4.0M;
the cluster node's 555 traj/s on the slowest arm is a different machine's
arithmetic, and the ceiling's named machine is now that node. The forecast's
directional call — no escalation — held.)*

Budget: **12 GPU-hours** at rung 0 (`c₀ = 1 h`), rising to 84 cumulative if the
ladder goes to its top rung. `--quick --rungs 3` is a three-second wiring check
and its output must never be written into §6.

**Then, and only then:** write `N` and the frozen β into `GRAFT_PHASE3_BUILD.md`
§6, and run the matrix — `graft.setgen.gate2.run_matrix(envs, spec)`. After any
L6/L7/GAFlowNet run the §6b amendment procedures are contaminated by learner
results, which is why nothing may be adjusted afterwards. *(Both happened — the
matrix ran 15 Aug 2026, the transcription landed 16 Aug 2026, and the
contamination clause is now in force: no §6 band or threshold may move cleanly.)*

**Nothing is left to rule before the gate runs.** The last such item — whether to
raise decision 6's threshold so `N` lands where L7 has converged — was **ruled on
12 August 2026: option 1, accept, threshold unchanged at 0.10**
(`PHASE3_DECISIONS.md` §6.8b; §2.3 is the pre-ruling analysis and says so).

Read §6.8b anyway, for the argument, because it is the one that makes a null
result on C3 interpretable: if `N` is enough for the flow family and **L7 alone**
is still short, that is the finding fix F12 asks for — the primary is improvement
at a *fixed training budget*, so an arm needing more budget to reach the same TV
is evidence against better credit assignment rather than an artefact of the
budget. A threshold tightened to protect L7 would be unfalsifiable in the one
direction it exists to be falsifiable. The genuinely inconclusive case — `N` so
small that *nothing* separates — is caught twice, by decision 6 on the tuning
suite and criterion 15 on the main suite, both routing to `inconclusive` rather
than to a negative verdict.

*(This paragraph previously instructed the reader to rule option 2 before running
the gate. That was already false when written: the ruling predates it.)*

### Built in Phase 3

| Step | Module | What it is |
|---|---|---|
| 1 | `requirements-ml.txt`, narrowed structural test | the ML boundary: `graft.core` and `graft.synth` still import no ML library; `graft.setgen` may |
| 2 | `graft/setgen/flgfn_probe.py` | the FL-GFN discharge — numpy only, no learner needed |
| 2 | `graft/setgen/features.py` | `SyntheticFeaturizer`, the fix-F6 boundary; the only module that reads an atom id |
| 2 | `graft/setgen/policy.py` | `Policy`, `LogZHead`, `StateFlowHead`, `PotentialHead`, `DeficitHead`, `capacity`, `match_capacity` |
| 3 | `graft/setgen/rollout.py` | `sample_trajectories`; re-exports `uniform_backward` rather than reimplementing it |
| 4 | `graft/setgen/trainer.py` | `TrainSpec` / `Environment` / `Batch` / `Arm` / `Trainer`. Owns ε, `N`, the checkpoint schedule, capacity, the seeds, `ledger=None`, **and the `g[i]` coordinates every flow objective is written in** |
| 5, 7–9 | `graft/setgen/learners/` | all nine arms. L6 and L7 share `led_db_loss` verbatim; the entire difference is `delta_d=True` on L7's featurizer |
| 10 | `graft/setgen/gate2.py` | `run_matrix`, `paired_bootstrap`, `consistency_report`, `audit_block`, and decision 26's verdict applied |
| 6 | `scripts/phase3_calibrate.py` | **the gate — wired, not run** |

### Measured results already in hand

**The FL-GFN discharge** (decision 24). Best-fitting member of the deficit-potential
family over all 8,638 main-suite terminals, every coefficient and a per-instance
constant free: RMS **0.4666**, R² 0.910, all 8,638 terminals off tolerance, at
every eligible β. Named special cases: `deficits_only` RMS 0.4717, `uniform_omega`
RMS 1.0298.

**This disproves that potential family. It does not show FL-GFN is inapplicable** —
plan §4.5.2 retired that stronger reading once already, and the two-part claim
string ships with the numbers so they cannot be separated later.

---

## 4. The environment, and one discrepancy

`torch 2.11.0+cu128`, verified working: `sm_120` in `torch.cuda.get_arch_list()`
and a real matmul executes. **A cu124 build would install, import, report
`cuda.is_available() == True`, and then fail at the first kernel launch** — this
GPU is Blackwell. `requirements-ml.txt` records that trap and the check that
catches it.

**`CLAUDE.md` line 7 says "1× RTX 5090 (32 GB)". The actual machine is an
RTX 5050 Laptop GPU with 8 GB.** Phases 0–4 use no GPU models, so nothing built
so far is affected. But fix F7 (VRAM collision — 7B extractor + 3B reader +
embedder, stage-sequential) was reasoned against 32 GB and is far tighter at 8.
**Not corrected**, because "Setup assumed" may describe a target machine rather
than this laptop. Someone has to say which.

---

## 5. Open items, honestly

| Item | Blocks | Where |
|---|---|---|
| **Gate 2 returned `inconclusive` (15 Aug 2026)** — criterion 15 failed (L4 0.126 / L5 0.166 vs 0.10 on the main suite) and L6's 0.05 band failed (p95 0.115); comparisons measured but withheld (L7 beat L6, lost to GAFlowNet — the exploration-bonus control is currently the best flow arm on TV) | any C3 claim; any clean re-run — §6b's "no learner results inspected" is now unsatisfiable, so a re-instrumented Gate 2 needs a new plan version + Gate-0 re-sign-off with the contamination stated, or the thesis consolidates on C1 | `PHASE3_DECISIONS.md` §7 |
| ~~**The terminal convention was never written down**~~ — **CLOSED 16 Aug 2026** | nothing | plan §4.1 says "pick one and write it down"; nobody did. The code *has* picked one — measured on `tiny_instance()`, a state that is both a valid stop and has children gives `F(s) = 181.49` vs `R(s) = 14.88`, so `R` is the flow on the terminating `STOP` edge, **not** the terminal state flow. Phase 3 now depends on it in four objectives and honours it: the terminating transition is a first-class step with its own `log P_F(STOP \| x)`, `log P_B = 0` and `φ` slot, so no loss assumes `F(X) = R(X)`. **Recorded 16 Aug 2026** in `realenv.build_real_batch`'s docstring, which is where it is now load-bearing: the real sampler emits the terminating transition as a first-class step for the same reason. |
| ~~**Phase-3 §6 not fully signed off**~~ — **CLOSED, signed 15 Aug 2026** | nothing | 1 (11 Aug), 6/11/29/30 (12 Aug, §6.8b), and the nine remaining `[recommended]` cells — 5, 10, 14, 15, 19's `c₀`, 23, 25, 26, 28 — **signed 15 Aug 2026 as written**, before the calibration gate ran, which §6b's "no learner results inspected beforehand" makes the last uncontaminated moment. Ownership changed, behaviour did not. **`N` and β remain `[fill at step 6]`**, and every signed value is still **[ANALYSIS]** — signing does not upgrade evidential status. |
| **SubTB's λ is in no decision table** | L5, and therefore β and `N` | set to 0.9 in `TrainSpec.subtb_lambda` — SubTB (ICML 2023)'s *hypergrid* value (its other tasks use 0.99–1.9, so it is not a single paper default), and **[ANALYSIS]** as applied to this environment. Added to §6 as decision 28 by the build; **signed 15 Aug 2026** with the other `[recommended]` cells (see the Phase-3 §6 row above — an earlier version of this row still said "not signed" beside it). L5 is the arm that selects β *and* sizes `N`, so it is the worst arm to have an unlisted hyperparameter. |
| ~~**L7 is the slowest flow arm to converge**~~ — **CLOSED, ruled 12 Aug 2026** | nothing; **it does not block the calibration gate** | option 1 (accept; decision 6 stays at 0.10). `PHASE3_DECISIONS.md` §6.8b carries the reasoning, §2.3 the pre-ruling analysis. The decisive argument: if `N` suffices for the flow family and L7 alone is short, **that is the finding fix F12 asks for** — the primary is improvement at a *fixed* budget, so an arm needing more budget is evidence against better credit assignment, not an artefact. The convergence table it was raised on is also stale: after R6, L7 is *first* of six at 50k, not last. *(This row previously read "option 2 must be ruled before step 6 runs", which was already false when written.)* |
| ~~**Gate-0 re-sign-off** for the β eligibility amendment~~ — **CLOSED, re-signed 15 Aug 2026** | nothing | `GATE0_CONTRACT.md`'s own "Re-sign-off" section. All three §6b decision-rule steps checked: step 1 (**the actual gap** — the rule text had been amended in decisions 19/22 since 11 Aug while the plan header still read "§6 signed off (9 Aug 2026)", asserting a signature over text it predated) is now recorded in `GRAFT_PHASE2_BUILD.md`'s header; step 2 is the signature; step 3 holds because **the calibration gate has never run** and decision 22's sweep *is* the learner result §6b names. Instrument untouched — no band moved, suites byte-identical, `environment_fingerprint` unchanged. |
| **The 0.001 margin** | possibly a regeneration | one main-suite instance clears the `neither` band at β = 4 by 0.001. Both exits cost something; deliberately undecided. `PHASE2_DECISIONS.md` §4.2. |
| ~~**Gate-0 data contract** (v1.2 §7 items 1–10)~~ — **CLOSED, SIGNED 15 Aug 2026** (delegated, recorded as such) | nothing — Gate 1 is unblocked | **all ten items filled and decided.** Item 8 measured 15 Aug 2026 — **GO**, 16.7 h, D1 κ 0.829 / D2 κ 0.813 (§3a). **Item 9 decided 15 Aug 2026 — scope c, 200 questions (§3b); nothing recorded here remains open.** Only the signature — a human act — is outstanding. D1 and D2 still have no off-the-shelf supervision. |
| **The D2 pair pool surfaced no CONFLICT at all** | D2's rare classes, and therefore C1 | 0 of 49 items, with 34 drawn from knowledge-update questions. A proposer property, not a labelling one (§3a) — Phase 6's pair proposer, not Phase 2.5's annotation. |
| **A question is stored as an assertion** | D2's pair pool, and ceiling 1 | "Can you provide more information on…" appears as claim text in a D2 pair. A question asserts nothing and should not enter a memory graph — a Stage-A extraction defect (§3a). |
| ~~**Gate-0 item 3.2's Tier-B definition is degenerate**~~ — **FIXED, amended 15 Aug 2026** | nothing | The original `H`-minimisation stripped every proof to one atom and lost evidence on 5 of 9 pilot questions, because `H` is formal validity only. **Item 3.2 now restricts removal to structural atoms and exempts gold from `max_atoms`** — evidence_dropped is 0 everywhere and no question is voided. A `U`-sufficiency variant was rejected first as provably vacuous (it makes Tier B ≡ Tier A). `GATE0_CONTRACT.md` item 3, `PHASE7_DECISIONS.md` §3.2c |
| **Stage C's recall instrument has nothing to measure at the pilot scope** | any ceiling-3 number | 8–23 eligible candidates per conversation against `pool_cap = 64`, so Tier-A recall is 1.000 by arithmetic on 9/10 questions and a channel returning its input unchanged would score identically. Flagged per question by `recall.saturation()`. Item 9 is now decided (scope c, §3b) but scope c is still evidence-only and will likely reproduce this; **scope b′'s distractor sessions are the one lever that changes it**, and that extension is not yet run. `PHASE7_DECISIONS.md` §3.1 |
| **The entity channel matched an anchor in 3 of 10 questions** | Stage C's entity channel, and Phase 8's anchor feature | one miss is decision 7's normalised-exact rule (`"new black Converse…"` vs the graph's `"black converse…"`); six are the **category-vs-instance** line — the same one behind both residual D1 disagreements at Gate-0 item 8. Not amended: changing a matching rule because a run was unflattering is the §6b failure. `PHASE7_DECISIONS.md` §3.2 |
| **Decision 11's tolerance moved; decision 29 is new** | signing off §6 | R13 found the 1% capacity tolerance unachievable by width (the dead block is ~half a width step at every scale), so the criterion is now minimality plus "control never smaller". Decision 29 fixes GRPO's `G = 8`. Both were changed **before** any L6/L7/GAFlowNet result was inspected, which is what §6b's second procedure is about — but neither is signed. |
| **`N_real` sits at ~93% of its ceiling** | any further per-trajectory cost | 200,000 trajectories at the slowest arm's ~30/s is ~1.87 h against a 2 h ceiling. The ladder has no rung above 200,000, so a cost increase does not escalate — it *demotes* to 100,000. That is a derivation moving, not a decision to revisit, but it means the ceiling now certifies nothing about convergence. `PHASE9_DECISIONS.md` §2.2 |
| **2Wiki's `entity_anchor` is near-oracle by dataset design** | how any Stage-A number is read | the anchor rule was a gold leak and is fixed (§1.3) — it now reads only the question and the candidate titles. But it *still* coincides with a gold title on 100% of dev rows, because 2Wiki questions name their bridge entity and that entity's paragraph is supporting by construction. Legitimate signal, not leakage; a caveat that must travel with the number. Same class as Stage C's "recall 1.000 by arithmetic" |
| **MuSiQue's anchor is unsatisfiable on 28.5% of rows, honestly** | `U`'s coverage denominator | the raw hop-1 subject matched a paragraph title on only 30.3%, so 69.7% of questions carried an anchor no subset of the pool could satisfy. Fixed by resolving to a title; the residual 28.5% now report **no anchor** rather than an impossible one. `PHASE9_DECISIONS.md` §1.3 / `musique_ans._resolve_to_title` |
| **The Cor 5.1 citation is withdrawn, and G11's instrument is a diagnostic only** | nothing; it is recorded | signed decision 12's collision statistic is **structurally zero** (a proof, not a measurement). The quantity substituted for it — distinct atoms with identical text — is a *dataset* property, not an MDP symmetry, so Symmetry-Aware GFlowNets' Cor 5.1 does not cover it. No scalar is applied. `PHASE9_DECISIONS.md` §1.4 / §7.2 |
| **A retired *number* is invisible to the consistency guard** | any retired numeric threshold | `check_plan_consistency.py` reads prose and cannot see a float, which is how the retired 1% capacity tolerance survived as an executable default and shipped a blocker. The fix pairs each such retirement with a test that reads the default itself. `PHASE9_DECISIONS.md` §7.9 |
| **Three Stage-D runner conventions are deliberately unset** | Phase 10's runner | `run_portfolio` declares an unused `ledger` and a constant-`False` `budget_exhausted`; an abstained query emits `best_score = NaN`; and the Gate-3 aggregation convention for abstentions has no owner yet. Listed so the runner's author inherits them as decisions rather than discovering them as bugs. `PHASE9_DECISIONS.md` §7.8 |
| **Three late-found baselines** | plan §5.3 | HyperMem, Chain-of-Memory, *How Memory Management Impacts LLM Agents* — in the library, not in the plan. |

---

## 6. How this project stays correct

These are the habits that produced the parts that hold up. They are not optional
decoration; skipping them is how the defects below happened.

**Label every claim.** `[EVIDENCE]` (named paper, venue stated) / `[HYPOTHESIS]`
(this project tests it) / `[ANALYSIS]` (judgment made here). Never blur them.

**Verify a citation before relying on it.** Cite the verified published title,
never the short name. Several numbers in this project's history survived multiple
review rounds and turned out not to be in the paper.

**Do not fabricate numbers.** This has been caught three times: an FCS literal
invented past the digits actually printed, a "70% restates INDEX" statistic with
nothing behind it, and a `torch==2.5.1` pin that would have failed on this GPU.
Every literal in a test should be independently re-derivable, and several now are
(the FCS reference is recomputed in exact rational arithmetic in its own test).

**A correction lands in four or five places, and three is the usual score.**
Decisions are restated in the gap section, the module spec, the exit criterion,
the normative table and the next phase's handoff. Phase 3 hit this in three
consecutive review rounds. Two mitigations are in place: **restatements point at
a decision number instead of repeating its value**, and
`scripts/check_plan_consistency.py` runs inside the test suite with **111 retired
wordings across 28 live documents** that fail the build if they reappear.

**The consistency script only catches textual recurrence.** It does not catch
semantic contradictions, and several rounds of real defects passed it cleanly.
It also **cannot see a number**: the retired 1% capacity tolerance survived as an
executable default and shipped a blocker (`PHASE9_DECISIONS.md` §7.1). A retired
threshold needs a test that reads the default, not a retirement alone.

**Two of this project's own guards were themselves wrong, and both were caught
by writing the guard's *inverse*.** A gold-quarantine test grepped source text
and failed on the docstring that explains the rule; a cross-implementation
agreement test used `pytest.approx` on nested dicts, where it does not recurse,
and reported a difference between identical values. A guard that fails on correct
code is a guard someone deletes.

**Say what is not known.** "No evidence found" is a finding. So is "this band is
a guess calibrated on Phase-1 fixtures, not on the lattice".

---

## 7. Defects found by review that should not be re-introduced

The reasoning matters more than the string; every one is now a registered
retirement in `scripts/check_plan_consistency.py`.

| Defect | Correct position |
|---|---|
| `exp(β·0) = 1` — an invalid set scoring 1 | the validity indicator is **multiplicative** |
| `p*(FAIL)` called a TV floor | `FAIL` is in **both** distributions, so TV = 0 is reachable |
| "A dead end proves no proof exists" | it licenses only *no proof found under this pool, policy, attempt count, budget* |
| Per-terminal DP for `p_θ` | one forward pass over the policy-independent state graph |
| `environment_fingerprint` carrying β | it must be β-independent, or freezing β after Gate 0 changes the identity of frozen suites |
| `validate_bands` failing open on an unknown scope | it raises; every other decision on that path fails closed |
| Counting source tiers over the whole pool | count tier **keys** among atoms whose `Source` resolves |
| LED consistency measured per terminal | LED decomposes per **trajectory**; one terminal has many |
| GAFlowNet with `Δd` in its policy | `Δd` reaches its **loss** only, or it becomes L7 with extra steps |
| `c_N = 0` proving unbiasedness | Theorem 1 also assumes `L = 0`; call it a finite-budget empirical control |
| Scattering per-edge `Δd` by a state→row map | duplicate states in one batch silently lose their features — a **false negative** for Contribution 3 |
| Asserting 1e-12 agreement for a float32 network | float32 carries ~1e-7; test the adapter in float64 and report the dtype separately |
| Min–max normalising a score *before* it becomes a feature | normalisation is scale-invariant per question and destroys absolute strength; carry **raw and normalised** and let each consumer choose. Caught twice — Phase 8 §3.3, then again in Phase 9's `{channel}_raw` columns, which held normalised values under a raw name |
| One presence flag for a whole block of channels | "did not run" and "ran and scored zero" must be distinguishable inputs, or a later stage differs by feature *semantics* rather than by its data |
| Δd computed in two places | it drifted in **both** directions at once — sign inverted *and* sparse where the other was dense. One helper, called by both paths; a test asserts there is exactly one implementation |
| Δd filled only at the taken action | it is a **comparison across candidate actions**; filled at the chosen one it is a label on a decision already made, and the C3 mechanism becomes unlearnable — with no symptom in any loss curve |
| Synthesising an obligation from a corpus's own annotation | annotations do not exist at inference. Derive from the question, and regression-test that *removing* the annotation does not move the result |
| Emitting an anchor no atom can satisfy | it inflates `coverage`'s denominator and depresses achievable `U`, unevenly and hardest on the multi-hop questions the corpus is there to supply. Resolve it to something the pool contains, or emit nothing |
| A head slice as a training subset | `musique_ans` train is **sorted by hop count**, so the first N rows are all 2-hop — removing exactly the questions the minimality claim needs. Stratify, seeded |
| Matching capacity on a *nominal* count, or under a retired tolerance | both put a control **below** the proposed method — the one thing decision 11's directional clause forbids. Verify both clauses per arm and raise on a violation |
| A fingerprint that binds its own outputs | it cannot converge. Bind the configuration, never the result |
| Citing a correction for a phenomenon it does not cover | Cor 5.1 corrects graph automorphisms under node-by-node generation; identical *text* in a dataset is not an MDP symmetry |
