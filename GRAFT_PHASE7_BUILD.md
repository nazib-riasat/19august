# GRAFT — Phase 7 Build Plan: Stage C, hybrid candidate retrieval (`graft/retrieve/`)

**A question → a capped pool of candidate atoms that contains the proof, measured. Recall, not final selection.**

Date: 14 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 7) · `GRAFT_RESEARCH_PLAN_v1.md` v1.2 §3.3, §2.4, §6.3–§6.4 · `GATE0_CONTRACT.md` (items 2, 3, 9) · `PHASE6_DECISIONS.md` §5 (the handoff) · `PHASE5_DECISIONS.md` §2.2a (the obligation parser's measured rates) · `CLAUDE.md` §6 (`pool_cap = 64` is frozen)
Effort: ~1.5–2 weeks solo **[ANALYSIS]** — an estimate; the GNN scorer is the schedule risk, and it is deliberately last in the build order.
Status: **§6 adopted 15 Aug 2026** (decisions 1–10 and 12 as recommended; 11 deferred to Gate-0 item 9, which decided scope c the same day). **All seven steps built and green** — steps 0–5 first, then step 6, the scorer, once `GATE0_CONTRACT.md` was signed. **The scorer is built but NOT trained**: one of step 6's three "done when" conditions is unmet and stays unmet until GPU/CPU time is spent on it. `PHASE7_DECISIONS.md` is the record and wins conflicts with this file.

Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps are numbered **G1–G10**, matching the Phase-0/1/2/2.5/3/4/5/6 convention.

---

## 0. What Phase 7 is for, and what it is not

**Stage C is a recall stage, and the plan says so twice** (§3.3: "Purpose is
recall, not final proof selection"). Its one deliverable is a candidate pool of
roughly 30–100 atoms per question with a **measured** sufficient-proof recall —
because *if the proof is not in the pool, nothing downstream can recover it*.
Ceiling 3 (plan §6.3) is this stage's number, and every Stage-D result in
Phases 8–10 is bounded by it.

**[EVIDENCE] Why hybrid channels and a small GNN, not an LLM retriever and not
iteration:**

* GNN-RAG (Findings ACL 2025): GNN retrieval beat LLM-based retrieval by
  **8.9–15.5 F1** on multi-hop and multi-entity questions at **9× fewer KG
  tokens**. *(Findings venue.)*
* GFM-RAG (NeurIPS 2025): an **8M-parameter** query-conditioned GNN reached
  Recall@5 of 87.1/58.2/95.6 on HotpotQA/MuSiQue/2Wiki in **one pass at
  0.107 s**, against 3.162 s for iterative IRCoT+HippoRAG — and explicitly
  names **noisy, incomplete KG-index quality as its principal dependency**,
  which is exactly why ceilings 1–2 are reported beside ceiling 3.
* HippoRAG (NeurIPS 2024): single-step PPR retrieval matched or beat iterative
  IRCoT at 10–20× cheaper — the Tier-2 alternate if the GNN disappoints.
* **[EVIDENCE, provisional]** Beyond Static Retrieval (arXiv 2509.25530):
  across four graph-RAG systems the bottleneck was **ranking, not coverage**
  (gold recall ≈95% at K=100 but buried below top-10); two iterations was the
  cost–benefit optimum; iteration *hurt* simple questions through
  over-retrieval. Motivates the ≤2-hop bound; flagged provisional, not
  load-bearing.

**The latency envelope is a sanity bar, not a comparison.** Mem0 (ECAI 2025)
reports search at 0.148 s flat / 0.476 s graph; the architecture requires p50
well under 0.5 s. Those are *published, uncontrolled* numbers (§5.3's re-run
rule) — Stage C measures its own p50/p95 on this machine and the envelope says
only "interactive, not batch".

| Downstream | Blocked on Phase 7 by |
|---|---|
| Phase 8 (gate) | its features are obligation-slot coverage **by the pool**, channel scores, pool size |
| Phase 9 (Stage D real) | the real `AtomFeaturizer` consumes retrieval channel scores per atom (fix F6's feature set, architecture §9.1) — and Stage D's whole environment is the pool this stage emits |
| Ceiling 3 | unmeasurable until a pool exists |
| Phase 10 | latency and token budgets inherit the pool size |

**Not in Phase 7:** final evidence selection (Stage D); the answerability gate
(Phase 8); any change to `H`, `U`, `R`, masks, the lattice, Stage A or Stage B;
any frozen value — `pool_cap = 64` is inherited frozen (`CLAUDE.md` §6), not
re-decided. No iterative multi-round retrieval (the ≤2-hop expansion is the
whole concession to it, per the provisional evidence above).

---

## 1. Ten gaps this phase must close

### G1 — "Sufficient-proof recall" needs gold proofs that exist on conversation, and they arrive in two tiers [ANALYSIS]

The primary metric (plan §2.4: sufficient-proof recall@k) presupposes a gold
proof set per question. `GATE0_CONTRACT.md` item 3 defines the conversational
one — *eligible assertions whose spans lie in `has_answer` turns of the
question's evidence sessions, closed under the structural refs rule, minimised
by removing any atom whose deletion keeps `H` true* — but the H-minimised form
needs per-question obligations and a signed contract. *(Written while Gate 0 was
unsigned; it signed 15 Aug 2026 and Tier B is live — item 3.2 as amended, see
`PHASE7_DECISIONS.md` §3.2c.)*

**Decision — a two-tier recall instrument, both tiers built now:**

1. **Tier A (runs today): `has_answer`-derived required-atom recall.** Gold =
   the pool-representable atoms of eligible assertions grounded in
   `has_answer` turns, closed under refs. No `H`, no minimisation — an
   *over-estimate* of the required set, so recall against it is
   **conservative** (a pool that covers the superset covers every minimal
   proof inside it). This is ceiling 3's measurable proxy and is labelled as
   the proxy it is.
2. **Tier B (one flag, post-Gate-0): H-minimised sufficient-proof recall** per
   item 3's full definition, computed with the frozen checker once the
   contract signs. The instrument takes the definition as a callable, so
   Tier B is a gold-set swap, not a rewrite.

**The gold labels never touch the channels.** `has_answer` and
`answer_session_ids` enter through the recall instrument's sidecar only — the
same quarantine `graft.ingest.corpus.question_meta` already enforces. A channel
that could see them would make ceiling 3 a fiction; a structural test asserts
`graft.retrieve`'s modules import no gold field.

### G2 — `bm25s` is a new dependency, and it is currently on the forbidden-import list [ANALYSIS]

The architecture names `bm25s` for the BM25 channel. The boring-stack rule
(§0.4) requires a written justification, and `graft/tests/test_structure.py`
lists `bm25s` in `ML_LIBRARIES` — importable only inside `ML_ALLOWED_PACKAGES`.

**Decision.** Adopt `bm25s`, justified: it is a numpy-backed, in-process BM25
with no index server and no persistence daemon — the boring-stack shape — and
re-implementing BM25 by hand is a correctness risk in a *baseline channel*,
the weakened-control direction R13 warned about. `graft.retrieve` joins
`ML_ALLOWED_PACKAGES` (it needs torch for the scorer regardless), the
allow-list assertion is updated deliberately against its failing test — the
intended friction — and the containment guard extends: importing
`graft.retrieve`'s non-scorer modules must not pull in torch.

### G3 — What each channel gets as "the question" [ANALYSIS]

Five channels, three different query representations, and leaving this
implicit is how a channel quietly reads something it should not:

* **BM25 and dense** get the **raw question text**, nothing else.
* **Entity match** gets `Obligations.entity_anchor` — the learned parser's
  slot (fix F2, built in Phase 5) — matched against entity canonical names
  *and alias sets* (the shape Phase 6 §5 hands over), normalised by the one
  shared `normalise_name`. **No anchor → empty channel**, never a fallback to
  keyword guessing: a guessed anchor is a hallucinated scope one field over.
* **Temporal filter** gets `Obligations.time_constraint`. Phase 5 measured the
  resolver's vocabulary coverage at **69% unresolved among time-bearing
  questions** (`PHASE5_DECISIONS.md` §2.2a) — so most of the time there is no
  constraint, and the filter must be a **no-op without one**.
* **Graph expansion and the GNN scorer** get the entity channel's seeds and
  the question embedding respectively.

The obligation parser runs **once per question** and its slots are recorded in
the artefact — the parser's measured quality is part of Stage C's error budget,
and fix F2's rule (slot scores reported wherever coverage is reported) follows
the value here.

### G4 — The temporal filter must fail open, or it deletes the graph [ANALYSIS]

The architecture says "drop atoms whose intervals contradict the obligation's
time constraint". On the current corpus **no `valid_during` edges exist on real
data** (D4's conversational application is Gate-1-time work,
`PHASE6_DECISIONS.md` §2.2), so a filter that dropped atoms *lacking* an
interval would empty every pool.

**Decision.** The filter removes only atoms that **provably contradict**: an
atom is dropped iff it carries a `valid_during` interval and
`interval.overlaps(constraint)` is false (`Interval.overlaps`, frozen since
Phase 0). No interval → pass. No constraint → the filter never runs. Both
fail-open directions are the same principle the support gate applies fail-closed
— each errs toward the side whose mistake is recoverable downstream: a
too-large pool costs budget; a wrongly emptied one costs the answer.

### G5 — Fusion needs declared arithmetic, and recall comes before ranking [ANALYSIS]

"Score-normalized weighted union → dedup → cap at 64" (architecture) has three
knobs nobody declared: the normalisation, the weights, and the tie-break.

**Decision — frozen in §6, deliberately boring:** per-channel min–max
normalisation to [0, 1] over that question's results (a rank-shape transform —
BM25 scores and cosines share no scale); **all channel weights = 1.0**; an
atom's fused score = the **max** of its normalised channel scores (union
semantics: one strong channel suffices — recall is the purpose, and averaging
would punish an atom only one channel could know about); ties broken by atom
id. The GNN scorer enters as a sixth score under the same normalisation. These
are §6 constants, not tuning knobs; the per-channel recall table (G7) is what
reveals a bad choice, and changing them later is a declared amendment.

### G6 — The GNN scorer's training signal is already contracted, and it is the distant one [ANALYSIS]

The architecture row says "trained on gold-evidence pairs (interface:
2Wiki/MuSiQue supervision)", but `GATE0_CONTRACT.md` item 2 — written later,
against the real records — is more specific: **Stage C trains on the coarse
distant signal** (an atom is relevant iff it derives from a turn in the
question's `answer_session_ids`) **and is evaluated on the fine one**
(required-atom recall, G1). The contract wins: the distant signal is
conversation-native, so the Wikipedia→conversation transfer declaration
(`CLAUDE.md` §7) is not spent here, and the labels exist today.

Constraints carried from the architecture: **≤ 8M parameters** (the GFM-RAG
scale point), query-conditioned, one forward pass per question (no iteration),
consuming the shared pinned embeddings and the typed graph through
`features.encoder_metadata()` — the fix-F6 shape: `score(question_repr,
pool_graph) → per-atom scores`, so Stage D's featurizer can consume the scores
without knowing the scorer. Training is GPU work and shares the queue with
Phase-3 calibration; the channel stack must run and be measured **without**
the scorer (five channels are training-free), so the scorer is additive, last
in the build order, and its absence is a declared configuration, not a crash.

### G7 — Per-channel recall is the diagnostic contract, instrumented before any fusion exists [ANALYSIS]

Plan §3.3: per-channel and union recall "instrumented from day one". Concretely:
every channel emits `(atom_id, score)` behind one protocol; the instrument
reports, per question and pooled: recall of each channel alone at its own k,
union recall pre-cap, union recall post-cap (**the** number — what the cap
costs is visible as the pre/post difference), pool size, and per-channel
latency. The ablation table (architecture's exit criterion) is this instrument
run with channels toggled — one loop, not seven scripts.

### G8 — What exactly is in the pool: the atom contract, and closure under refs [ANALYSIS]

Stage D's environment is an `AtomPool` (frozen in Phase 1), and the closure
rule — an atom is addable only once every atom it references is selected — is
frozen project-wide (`CLAUDE.md` §6). A pool whose atoms reference atoms
*outside* the pool would make valid proofs unconstructible, silently reviving
the failure the closure rule proved to zero in Phase 1.

**Decision — the graph→pool mapping, declared:**

* **node atoms** for assertion-backed nodes (Claim/Value/Event) whose
  assertions are **eligible** (fix F9's boundary holds at retrieval too) and
  for the Entity/TimeInterval nodes their edges reference;
* **edge atoms** for live edges among selected nodes, `refs` = their endpoint
  atoms;
* retrieval scores attach to assertion-backed node atoms; referenced
  structural atoms (entities, intervals, their edges) ride along at score 0 —
  they are *support*, not hits;
* **closure is enforced at pool assembly**: selecting an atom pulls its refs
  in, and the cap applies to the closed set — so `pool_cap = 64` counts what
  Stage D actually receives. A closure test is an exit criterion, mirroring
  Phase 1's unconstructible-terminal regression.

### G9 — The numbers this phase can honestly produce today are plumbing numbers [ANALYSIS]

Three bounds, stated before any recall number exists so none of them is
discovered as an excuse later:

1. **The pilot graph is stand-in-constructed** (`PHASE6_DECISIONS.md` §2.1):
   links are a deterministic rule, not a trained D1. Recall on it exercises
   the machinery; it does not measure Stage C's quality on a real graph.
2. **Ten questions.** The pilot has 10; the decisive corpus is Gate-0 item 9's
   scope decision. Every number is per-question-listed, never presented as a
   distribution over 10 points.
3. **Ceiling 1 already took 55%** (Phase 5: 333 extracted → 151 eligible).
   Ceiling 3 is reported **beside** ceilings 1–2 (the §6.3 decomposition), or
   a retrieval failure will be misread into a stage that did not cause it.

So Phase 7's runs carry the same `smoke`-style honesty stamp as Phase 6's:
machinery numbers now, decisive numbers when the scope corpus and a Gate-1
graph exist.

### G10 — Expansion needs a fan-out bound, not just a hop bound [ANALYSIS]

"≤ 2-hop from matched entities" bounds depth; it does not bound width. A hub
entity (the user, in a memory graph, links to *everything*) makes 2-hop
expansion the whole graph, and the cap then silently decides recall by
truncation order. **Decision:** expansion walks **live** edges only, ≤ 2 hops,
with a per-hop fan-out cap (§6; default 32) that keeps the *most recent* edges
by `t_created` when it binds — recency as the tie-break is [ANALYSIS], declared
because a memory graph's hubs skew old. The number of times the fan-out cap
binds is reported: a cap that binds often is the signal to revisit, visibly.

---

## 2. Scope

**In.** `graft/retrieve/`: the atom-pool contract with closure (P7.0); five
training-free channels — BM25, dense, entity match, temporal filter, 2-hop
expansion (P7.1–P7.5); fusion + cap (P7.6); the two-tier recall instrument,
ceiling 3 and the ablation loop (P7.7); the GNN scorer behind a frozen
interface, trained on the item-2 distant signal (P7.8); the runner script and
artefact (P7.9); tests (P7.10). `graft.retrieve` joins the per-package ML
allow-list (G2).

**Out.** Final selection, `U`, `H`, the gate, the reader; iterative retrieval;
any frozen value (`pool_cap = 64`, `tau_nli`, the closure rule are inherited);
PPR/HippoRAG-style retrieval (Tier 2 — named alternate, built only if the GNN
scorer disappoints at Gate-3-real time); any Stage-A/B change.

---

## 3. Modules

### P7.0 `retrieve/pool.py` — the atom contract (do this first)

G8's mapping: snapshot → `CandidateAtom`s → `AtomPool`, closure-enforcing,
eligibility-only, cap-aware. No ML import; testable against the Phase-2
lattice's snapshots as well as real ones (the protocol is the same on purpose).

### P7.1 `retrieve/bm25.py`

`bm25s` over eligible assertions' `text_norm`, index built once per graph
snapshot and cached in memory (the corpus is small; an index *file* is a
premature artefact). Returns `(atom_id, score)`.

### P7.2 `retrieve/dense.py`

Exact cosine top-k over the **shared pinned embedder's** vectors
(`graphbuild.embed` — one embedder project-wide, or channel fusion is
incomparable; its disk cache means Stage C never re-embeds what Stage B
embedded). Exact, not ANN: `pool_cap`-scale corpora need no index, and the
boring-stack rule forbids one without a measured reason.

### P7.3 `retrieve/entity.py`

Anchor → normalised match over entity names *and aliases* → that entity's
local live edges → their assertion-backed endpoints. Empty anchor → empty
result (G3). Conversation-scoped by the question's `conv_id`, the same
wrong-merge guard the candidate generator applies.

### P7.4 `retrieve/temporal.py`

G4's fail-open filter, applied to the fused pool (it is a filter, not a
channel — it can only remove).

### P7.5 `retrieve/expand.py`

G10's bounded walk from entity-channel seeds. Live edges only; hop ≤ 2;
fan-out cap with recency tie-break; binding counts reported.

### P7.6 `retrieve/fuse.py`

G5's declared arithmetic: min–max per channel, weights 1.0, max-union, atom-id
ties, dedup by atom id, closure via P7.0, cap at the frozen 64.

### P7.7 `retrieve/recall.py`

G1's two-tier instrument + G7's per-channel/ablation loop + ceiling 3,
emitting one table. Gold sidecar quarantined here and nowhere else.

### P7.8 `retrieve/scorer.py`

G6's query-conditioned GNN (≤ 8M, one pass), `score(question, pool_graph) →
per-atom scores` behind the fix-F6 shape; trained on the item-2 distant signal
with the shared `TRAINING`-style budget dict; its absence is a legal
configuration of the fusion (five channels stand alone).

*(Amended 15 Aug 2026 — P6.11 now exists, so the conventions this row gestured
at are code, not intent: reuse `graphbuild.train.split_questions` (user-level,
stratified, seed 20260813) for any question-level split, and inherit the
trainer's three guards — the seed reaches **initialisation** (`build_arm`
pattern), early stopping **restores** the argmin-dev state, and a loop with no
scorable dev item **refuses** rather than returning its initialisation.
Reinventing any of these is how Phase 6's caught defects come back.)*

### P7.9 `scripts/phase7_retrieval.py`

Runs the stack over the pilot log's committed graph: per-question channel
table, union/capped recall (Tier A), latency p50/p95 per channel, obligation
slots as parsed, fan-out binding counts, one JSON artefact with the G9 honesty
stamp. *(As built: the Gate-0 contract signed 15 Aug 2026, so both tiers run and
are reported side by side; the by-name refusal served its purpose and is gone.)*

### P7.10 `graft/tests/`

Per §5. Everything except the scorer runs on a bare interpreter with stub
embeddings; the scorer's forward/backward smoke needs the ML environment (the
Phase-3/6 pattern).

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 0 | P7.0 pool contract | closure enforced and tested; eligibility boundary tested; cap counts the closed set |
| 1 | P7.3 entity + P7.4 temporal + P7.5 expansion (no ML) | anchor→pool path works on the pilot graph; fail-open semantics tested; fan-out cap binds visibly |
| 2 | P7.1 BM25 + P7.2 dense | both channels return normalised scores over eligible assertions; the dense channel hits the shared cache |
| 3 | P7.6 fusion + cap | union → dedup → closure → cap reproduces byte-identically across two runs |
| 4 | P7.7 recall instrument | Tier-A recall + per-channel ablation table on the pilot graph, honesty-stamped |
| 5 | P7.9 runner | one command, one artefact, latency measured |
| 6 | P7.8 scorer | ≤ 8M verified by `parameter_count`; trains one epoch on the distant signal within 8 GB; fusion accepts its scores as a sixth channel |

Steps 0–5 need no GPU and no new training; step 6 is the only contended work
and everything before it is independently useful — the same de-risking shape as
Phase 6's build order.

---

## 5. Exit criteria

**The pool is sound**
1. Every emitted pool is closed under refs; the closure regression test exists
   (G8) and `pool_cap = 64` counts the closed set.
2. Only eligible assertions' atoms enter any pool (fix F9 at retrieval),
   negative-tested.
3. Two runs over the same graph and questions produce identical pools and
   identical artefacts (the determinism the ledger's digest discipline expects).
   *(Amended 15 Aug 2026: "identical artefacts" is measured as an identical
   `determinism.digest` — the artefact minus its declared volatile keys
   (`runtime.VOLATILE_KEYS`: wall-clock latencies and the host environment).
   Byte-identity is impossible while criterion 14 embeds latencies; the digest
   is the testable surface, and the exclusion list is itself asserted by test.)*

**Recall is measured, honestly**
4. Tier-A required-atom recall per question and pooled, on the pilot graph,
   labelled with G9's three bounds; Tier B *(live since the 15 Aug 2026
   signature; its pre-signature state was a by-name refusal)* reported beside
   it, with required-node/required-edge Recall@k for both tiers (plan §3.3, §6.4).
5. The per-channel and union(pre-cap/post-cap) table exists — the ablation
   loop, not per-channel scripts (G7).
6. Ceiling 3 is reported beside ceilings 1–2, never alone (G9).
7. The gold sidecar (`has_answer`, `answer_session_ids`) is unreachable from
   channel code — asserted structurally, not promised (G1).

**The channels behave as declared**
8. The temporal filter passes atoms without intervals and questions without
   constraints (G4), tested in both directions.
9. The entity channel returns nothing on a missing anchor; anchor matching
   goes through the one shared normalisation and the alias sets (G3).
10. Expansion respects hop ≤ 2 and the fan-out cap; the cap's binding count is
    in the artefact (G10).
11. Fusion arithmetic is the declared one (min–max, weights 1.0, max-union,
    id ties) and is byte-stable (G5).

**Discipline**
12. `bm25s` adopted with the G2 justification; `graft.retrieve` on the
    allow-list; non-scorer modules import no torch (containment guard).
13. The scorer is ≤ 8M parameters (asserted), one forward per question, behind
    the frozen interface; the stack runs and is measured without it (G6).
14. Latency p50/p95 per channel and end-to-end, measured on this machine, in
    the artefact beside the envelope note (G0's re-run rule: Mem0's numbers
    are uncontrolled and are not compared against).
15. `verify_handoff.py` gains the Stage-C fingerprint: channel weights,
    normalisation, caps, scorer config, and the shared embedder pin.
16. Obligation slots used by the channels are recorded per question, with the
    parser's measured unresolved rate quoted beside them (fix F2's rule).

---

## 6. Decisions to lock before writing code — **ADOPTED 15 Aug 2026**

**Every row's "Recommended" column is adopted as written**, with two
qualifications recorded rather than glossed:

* **Decision 11 is not adopted, because it was not this document's to adopt.**
  The scope corpus is Gate-0 item 9's. **Item 9 is now decided** (scope c, 200
  questions; `GATE0_CONTRACT.md` item 9, 15 Aug 2026), but ingestion at that
  scope has not run, so steps 0–5 ran on the pilot's 10 questions under G9's
  honesty stamp. Nothing in steps 0–5 depends on the scope; the recall
  *numbers* do, which is exactly why they are labelled machinery numbers and why
  they will need re-running once scope-c ingestion exists.
* **Decision 9 is adopted but not built.** The scorer trains, and
  `GATE0_CONTRACT.md`'s first line is that nothing is trained before it signs.
  Adopting the row fixes the interface the other modules are written against
  (fusion accepts a sixth channel; its absence is a legal configuration);
  building it waits for the signature.

Adopted by the build, in the Phase-3 shape (`PHASE3_DECISIONS.md` §4 item 2):
**the build took them as written and says so**, rather than leaving the code
resting on an unsigned table. Decisions 2, 4 and the `pool_cap` in 3 are
*inherited frozen* (`CLAUDE.md` §6, `PHASE6_DECISIONS.md`), not re-decided here.

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | `bm25s` | adopt, with G2's written justification; `graft.retrieve` joins the ML allow-list | a hand-rolled BM25 baseline — the weakened-control shape R13 catalogued |
| 2 | Embedder | the **shared** pinned bge-small (`graphbuild.embed`), same cache | two embedders → channel fusion scores incomparable; Stage-B/C vectors diverge |
| 3 | Fusion arithmetic | min–max per channel per question · all weights 1.0 · max-union · atom-id ties (G5) | every recall number moves; the ablation table is the instrument that justifies any change |
| 4 | Pool mapping + closure | G8's mapping verbatim; closure at assembly; support atoms at score 0 | Stage D's environment changes shape → Phase 9 re-runs |
| 5 | Temporal filter semantics | provable-contradiction-only, fail-open both ways (G4) | silent pool emptying on the corpus that lacks intervals |
| 6 | Expansion bounds | hops ≤ 2 (provisional evidence, flagged) · fan-out cap 32/hop · recency tie-break · binding count reported (G10) | recall changes untraceably; hub blowups return |
| 7 | Anchor source | `Obligations.entity_anchor` only; no anchor → empty channel (G3) | a guessed anchor is a hallucinated scope — the failure `scope`'s metadata rule exists to prevent |
| 8 | Recall gold | the two-tier instrument (G1): Tier A `has_answer`-derived closed superset now, Tier B H-minimised post-Gate-0, same callable slot | recall numbers that mean different things in different documents |
| 9 | Scorer | ≤ 8M query-conditioned GNN, one pass, item-2 **distant** training signal, fix-F6 interface, optional in fusion (G6) | the architecture's 2Wiki interface would spend the transfer declaration Stage D needs |
| 10 | Latency protocol | p50/p95 per channel + end-to-end via the ledger's wall-clock, on this machine; Mem0's envelope quoted as uncontrolled context only | an accidental cross-paper latency comparison — §5.3's re-run rule violated |
| 11 | Scope corpus | **not decided here** — Gate-0 item 9's, as everywhere; the pilot's 10 questions are plumbing (G9) | the unforced guess G8 (Phase 5) exists to avoid |
| 12 | Honesty stamp | Phase-6's discipline verbatim: pilot-graph numbers are machinery numbers, per-question-listed, never quoted as results | the bootstrap-labels mistake, two phases later |

---

## 7. Explicitly not in Phase 7

No iterative retrieval loops; no ANN index, vector database or search server;
no PPR channel (Tier 2, by name, if the scorer disappoints); no reranking-for-
packing (that is Stage D's job under `U`); no gate features (Phase 8 consumes
this stage, not the reverse); no retuning of `pool_cap`, `tau_nli` or any
frozen value; no gold field visible to any channel; and no quoting of any
number produced on the stand-in-constructed pilot graph.

---

## 8. What Phases 8 and 9 will need from this, verbatim

* **`AtomPool`s with closure and the cap already applied** — Stage D's
  environment, in the exact shape the Phase-1 masks and checker consume.
* **Per-atom channel scores** (five + optional scorer, normalised) — the real
  `AtomFeaturizer`'s retrieval features (architecture §9.1), riding on the
  atoms so fix F6's interface stays one call.
* **Obligation-slot coverage of the pool** — Phase 8's gate features are
  "anchor found? value-type candidate present? time constraint satisfiable?",
  all computable from the pool + the recorded slots.
* **The recall instrument and its gold callable** — ceiling 3 at every later
  gate, with both tiers live since the 15 Aug 2026 signature.
* **The Stage-C fingerprint**, alongside the ingestion and Stage-B ones — the
  triple that makes a cross-machine run's identity checkable end to end.
