# GRAFT — Phase 6 Build Plan: Stage B, learned graph construction (`graft/graphbuild/`)

**Eligible assertions → a typed, versioned, provenance-preserving temporal graph, via propose → validate → commit. Contribution 1's machinery, and Gate 1's harness.**

Date: 14 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 6, fixes F8/F9) · `GRAFT_RESEARCH_PLAN_v1.md` v1.2 §3.2, §2.4, §6.3–§6.4, Gate 1 · `GATE0_CONTRACT.md` (items 1, 3, 5, 6, 8, 9) · `PHASE5_DECISIONS.md` §6 (the handoff) · `PHASE2_5_DECISIONS.md` (guidelines, bootstrap labels, power analysis)
Effort: ~2–2.5 weeks solo **[ANALYSIS]** — an estimate; the two schedule risks are the external-dataset loaders (G6) and the D2 annotation batch, which is human work outside this phase's control.
Status: **built, audited and green (14 Aug 2026) — `PHASE6_DECISIONS.md` is the record and wins any conflict with this file.** §6 was adopted as recommended, with the amendments the audit forced recorded there (decision-3 endpoint widening §1.3a, the supersession scope §1.3b, decision 15's feature variants). A two-source post-build audit confirmed 21 defects — two blockers in the item/snapshot machinery — all fixed and regression-tested (`PHASE6_DECISIONS.md` §7). **Gate 1 itself is a run this phase cannot perform** — G1 says exactly what it waits on. *(Amended 19 Aug 2026: this sentence used to continue "and the driver additionally refuses a decisive run by name until the multi-seed trainer loop (the largest unbuilt piece) exists". The trainer was built on 14 Aug as **P6.11** — `graft/graphbuild/train.py`, recorded in `PHASE6_DECISIONS.md` §8, which wins conflicts with this file — so the paragraph was asserting a blocker and its own removal in the same breath. The driver still refuses a decisive run by name, but on the conditions build-order row 9 lists: an unsigned contract, missing decoder labels, or a stub embedder.)*

Labels inherited: **[EVIDENCE]** (named paper, venue stated) · **[HYPOTHESIS]** (this project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps are numbered **G1–G12**, matching the Phase-0/1/2/2.5/3/4/5 convention.

---

## 0. What Phase 6 is for, and what it is not

**Contribution 1 lives here.** The claim (`CLAUDE.md` §1, §9) is open-world
provenance-preserving incremental graph construction **as a combination**:
entity *creation* distinguished from non-entities, span provenance on every
edge, temporal validity, conflict detection, and non-destructive supersession.
It is deliberately *not* the claim that learned graph construction from text is
new — **[EVIDENCE]** GATA (NeurIPS 2020) builds and updates a belief graph from
raw text at every step, and the claims audit records that boundary. What no
prior system does is the combination under a deterministic checker's
provenance discipline.

**[EVIDENCE] Why the construction step is worth learning at all.** The risk it
manages is measured, not hypothesized: *How Memory Management Impacts LLM
Agents* (ACL 2026) documents error propagation through stored memory —
`CLAUDE.md` §7 records that upgrade from `[ANALYSIS]` to `[EVIDENCE]` — and a
wrong merge is the single most damaging Stage-B error because it corrupts every
proof built on it afterwards (plan §8 risk #1). The commit pipeline's
audit/active split (fix F9) and non-destructive supersession (Zep's
edge-invalidation precedent — vendor preprint, flagged as everywhere) are the
two structural answers.

| Downstream | Blocked on Phase 6 by |
|---|---|
| Phase 7 (Stage C) | retrieval channels read the committed graph: entity match walks `about_entity` edges, temporal filtering reads `valid_during`, graph expansion needs typed edges |
| Phase 9 (Stage D real) | the real `AtomFeaturizer` consumes Stage-B encoder embeddings (fix F6's other half) |
| Ceiling 2 (graph) | "does the constructed graph contain a sufficient proof?" is unmeasurable until a graph exists |
| Gate 1 | the kill-shot for C1 — runs on this phase's harness once its entry conditions (G1) are met |

**Not in Phase 6:** retrieval (Phase 7); the answerability gate (Phase 8); any
change to `H`, `U`, `R`, masks, `d(s)`, the lattice, the evaluator, or the
Phase-5 write path; any change to a frozen value; no retuning of anything the
Gate-0 contract fixes. Stage B is **learned and carries error** (plan §3.1's
naming discipline applies — nothing here verifies *truth*, and the schema
validator "bounds a specific, enumerable class of damage", plan §3.2, never
"makes neural predictions safe").

---

## 1. Twelve gaps this phase must close

### G1 — Gate 1 cannot run in this phase, and pretending otherwise would repeat the bootstrap-labels mistake [ANALYSIS]

Gate 1's four entry conditions, and their state as of 14 Aug 2026:

1. **`GATE0_CONTRACT.md` signed** — **blocked**, on item 8 (the Phase-2.5
   human timed pass, `PHASE2_5_DECISIONS.md` §5).
2. **Human D1/D2 labels** at the item-8-budgeted volumes — **blocked**;
   planning figures D1 `n_test` ≈ 627, D2 ≈ 300–600 pairs
   (`data/phase2_5/power.json`). Only the spike's machine-assisted bootstrap
   labels exist, and they carry `answers_gate0_item8: false`.
3. **A frozen extractor** — **met.** Phase-5 decision 2 is frozen on candidate
   B (`PHASE5_DECISIONS.md` §2.1b).
4. **An ingested corpus** — **partly met.** The live pilot's log is real
   Stage-A output at pilot scale: **248 turns, 222 assertions of which 151 are
   eligible, 397 spans, 187 mentions** (`artefacts/phase5_pilot/`). What is
   *not* met is the item-9 **scope** corpus, which the sizing memo (§2.2a)
   prices and Gate-0 item 9 chooses.

**Decision.** Phase 6 builds *everything* — schema freeze, items, encoders,
decoders, commit pipeline, baselines, the Gate-1 harness with its predeclared
decision rule — and exercises it end to end on the pilot log, which is enough
graph to construct: 151 eligible assertions and 187 mentions is a real, if
small, Stage-B input. Labels are the binding constraint, not data. Runs on
**bootstrap labels** are **smoke runs**: they prove the machinery works and are
*never quoted as results* — the same discipline that kept those labels out of
Gate-0 item 8 — and the artefact carries a `smoke: true` stamp so the
distinction survives being read six weeks later. Gate 1 is then one command on
the day conditions 1 and 2 hold.

**Three Phase-5 measurements change what Phase 6 should expect**, and they
belong here rather than in a footnote (`PHASE5_DECISIONS.md` §2.2a):

1. **108 of 333 extracted assertions (32%) failed to ground** and **32% of what
   was stored was quarantined** by the support gate. Stage B therefore receives
   roughly **45% of what the extractor proposed** — 151 eligible from 333
   extracted. Consequence: ceiling 2 (does the graph contain a sufficient
   proof?) is *bounded by ceiling 1*, and the two must be reported together or
   a graph failure will be misread as a construction failure.
2. **No multi-span assertion exists in any real run** — 0 of 222 live, 0 of 78
   replay. The schema, the write path and a test support cross-turn provenance;
   the corpus has not produced it. **Phase 6 must not assume multi-span input
   when sizing D1/D2 items**, and P6.1 re-measures it if the scope grows.
3. **D2's claim pool is thinner than the spike suggested** — 0.895 stored
   assertions per turn, of which 68% are eligible. The pair proposer works over
   that, which is exactly what G5's recall measurement exposes. This mirrors Phase 3 exactly (built and green;
calibration gate since run — 15 Aug 2026, `PHASE3_DECISIONS.md` §7), and the
same rule applies: **no
proposed-vs-baseline comparison may be inspected before the decisive run** —
what the smoke run may verify is plumbing (losses decrease, shapes match,
artefacts write), not rankings.

### G2 — The encoder dependency crosses the boring-stack line [ANALYSIS]

The architecture pins E2 as "2-layer `HGTConv`" — that class lives in
**torch-geometric**, a new dependency, and §0.4 requires a written
justification. It is this: HGT's node- and edge-type-dependent attention with
relative temporal encoding (**[EVIDENCE]** HGT, WWW 2020 — reported 9–21% over
prior GNN baselines on the 179M-node OAG; the closest published encoder to
this graph type) is the *point of comparison* E2 exists to provide, and
re-implementing typed attention by hand is a correctness risk in the exact
place a subtle bug would masquerade as "the baseline is weak" — the failure
mode Phase 3's capacity-match defect already demonstrated in this project.

Three conditions, so the dependency stays contained:

* `torch_geometric` is installed **pure-Python** (no `pyg_lib`, no
  `torch_scatter` — since PyG 2.3 the core message-passing classes run on
  plain torch, and the compiled extensions have no reliable Windows wheels).
  A build-time check asserts `HGTConv` imports and runs forward on this
  machine before E2 is declared runnable; if it does not, E2 falls back to
  **CompGCN** (ICLR 2020, the architecture's named alternate) implemented on
  plain torch, and the swap is recorded as a departure.
* it enters `requirements-ml.txt`'s layer only — `graft.core`, `graft.synth`,
  `graft.ingest` never import it (the P3.0/P5 per-package boundary test
  extends to `graft.graphbuild`: **only** this package, `graft.setgen` and
  `graft.ingest` may import torch-family libraries). Note `torch_geometric` is
  *already* on the forbidden-import list that test enforces, and
  `ML_ALLOWED_PACKAGES` is pinned by an equality assertion — so admitting
  `graft.graphbuild` is a deliberate one-line edit with a test that fails until
  it is made, which is the intended shape.
* **E1 imports none of it.** The GraphMixer-style baseline (**[EVIDENCE]**
  GraphMixer, ICLR 2023 — MLP link-encoder + mean-pool + MLP head matched or
  beat RNN/attention temporal GNNs; caveat recorded in the plan: validated on
  temporal link prediction, not on a heterogeneous assertion graph) is plain
  torch, so the mandatory baseline survives any PyG breakage.

### G3 — The graph schema is frozen *here*, and it has no endpoint contract yet [ANALYSIS]

`schemas.py` says `Node` and `Edge` are "amendable until Phase 6". This is
Phase 6. Three things resolve:

1. **A discrepancy nobody has recorded.** Plan §3.2 lists **eleven** node
   types; `NODE_TYPES` has **ten**. The missing one is **`Certificate`**, and
   it appears nowhere in the built code, in any DECISIONS file, or in any
   other plan section — Phase 1 produced `CheckResult`, not a certificate
   node, and no component writes or reads one. The research plan wins
   conflicts (`CLAUDE.md` §2), so this cannot be closed by silence in either
   direction. **Recommendation (§6 decision 2): do not instantiate it, and
   record the drop as a departure with its reason** — a node type nothing
   writes is a schema promise the graph cannot keep, and adding an unwritten
   type to a *frozen* vocabulary would make the freeze meaningless. If a later
   phase finds a use (a checker certificate persisted as evidence would be the
   obvious one), that is a deliberate amendment with a `SCHEMA_VERSION` bump,
   which is exactly what the freeze exists to force.
2. **The type vocabularies otherwise as they stand** — `EDGE_TYPES` (11)
   matches plan §3.2 verbatim; the ten node types match it minus row 1's
   `Certificate`. No addition is needed and none is taken.
3. **An endpoint table** — nothing anywhere says which edge types may connect
   which node types, and without it "schema-validate" (architecture §6.3) has
   nothing formal to check. The table is declared in §6 (decision 3) and
   enforced by the commit validator (G8). `mentioned_in: Mention → Turn`,
   `about_entity: {Claim,Value,Event} → Entity`, `supported_by: {Claim,Value,
   Event,Entity,Conflict} → SourceSpan`, `valid_during: {Claim,Value,Event} →
   TimeInterval`, `asserted_by: {Claim,Value,Event} → Source`, `has_value:
   Entity → Value`, `same_as: Entity → Entity`, `contradicts: Claim → Claim`,
   `supersedes: Claim → Claim`, `derived_from: {Conflict} → Claim`,
   `retired_by: Claim → Turn`. **[ANALYSIS]** derived from plan §3.2's prose;
   any relation the build finds it needs outside this table is a *recorded
   amendment*, not a silent extension.

`SCHEMA_VERSION` moves only if a field changes; the endpoint table is a new
constant beside the vocabularies, not a field change.

### G4 — D1's candidate generator is underspecified, and its recall is D1's own ceiling [ANALYSIS]

The architecture says "top-k existing entities (name-normalized + embedding)".
Unspecified: `k`; the normalization; the embedding pin; cold start; and the
fact that **a linker cannot link to a candidate that was never proposed** — so
candidate recall bounds every D1 number and must be reported beside them
(exactly the ceiling discipline of plan §6.3, one level down).

**Decision.** `k = 10` — the same top-*s* = 10 the pair proposer inherits from
Mem0's retrieval-before-update pattern (**[EVIDENCE-adjacent]**, fix F8), one
constant with one precedent rather than two knobs. Name normalization is
`graft.ingest.grounding.normalise` — the project already owns exactly one
casefold/whitespace-collapse and a second one is how "Sankeien  Garden" and
"sankeien garden" become different entities. Embeddings per G7's pin.
Cold start: an empty candidate list is *legal* (the first mention of a
conversation can only be `CREATE_NEW_ENTITY` / `NON_ENTITY` / `DEFER`), and
the D1 head must be trained with empty-candidate examples present, or it
learns that linking is always possible. **Candidate recall@k on gold links is
a reported number in every D1 table.**

### G5 — The pair proposer is D2's ceiling, and the spike already found its failure mode [ANALYSIS]

Fix F8: a new claim is compared against the top-*s* = 10 similar existing
claims **sharing an entity anchor**. Consequences to make explicit:

* **proposer recall on gold pairs is reported** wherever D2 is reported — a
  conflict never proposed is a conflict never classified, and no downstream
  macro-F1 can see the loss;
* **negatives come from the proposer, never from random pairing**
  (`GATE0_CONTRACT.md` item 6): a random negative is trivially `INDEPENDENT`
  and inflates macro-F1 on the class nobody cares about;
* the spike's measured flood — assistant list-replies produce piles of
  `INDEPENDENT` advice pairs (`PHASE2_5_DECISIONS.md`, guidelines v1) — is
  handled at the **proposer** (anchor-sharing already suppresses most of it;
  the residual is measured on the pilot items, not patched blind);
* the anchor for a claim with no linked entity yet is its D1 output — so D2
  runs *after* D1 in the per-turn commit loop, and a `DEFER`ed mention parks
  its claims' pairing too (the revisit queue, G8).

### G6 — D3/D4 have no loaders, and "supervision interface" needs to mean something concrete [ANALYSIS]

`GATE0_CONTRACT.md` item 1 sources D3 from DialogRE (ACL 2020; scored with its
progressive **F1_c**) and Re-DocRED (EMNLP 2022 — **not DocRED**, whose false
negatives punish recall-oriented models by ~13 F1), and D4 from TORQUE
(EMNLP 2020, MATRES' annotation scheme, ACL 2018). None of their label sets
maps 1:1 onto GRAFT's eleven relations — which is why every parent document
says "-style" and "interface".

**Decision — what "interface" means here, so it cannot drift.** D3 and D4
train and evaluate **on the external datasets' native label sets**, through
the *same frozen encoder interface* the GRAFT decoders use (the fix-F6
pattern, applied to Stage B: `decoder(pair_repr | node_reprs) → logits`, with
the dataset adapter below the interface and no dataset-specific code above
it). What this establishes is that the shared encoder + typed-decoder
machinery carries real relational/temporal signal — the plan's actual Gate-1
question for D3/D4. Their *GRAFT-schema* application (writing `contradicts` /
`valid_during` edges on conversation) uses the declared mapping in §6
(decision 7), and **every mapping loss is reported with the loader**: which
native classes were dropped, which merged, per the contract's own rule that
adaptations are "never presented as native supervision". Datasets are pinned
by SHA at download (the corpus.py pattern), licences recorded, raw files
gitignored.

### G7 — The embedding model is named by size, not pinned [ANALYSIS]

The architecture says "bge-small text embeddings" (E3's feature set) and
Stage C will reuse the same embedder. Pin it here once: model id
`BAAI/bge-small-en-v1.5`, revision hash frozen in §6, loaded through
`transformers` directly (the Phase-5 precedent — a second library for the
same weights is a dependency bought for a wrapper), CLS-pooled and
L2-normalized per the model card, **F7 discipline**: never resident with the
extractor or the NLI model; embeddings for stored objects are computed in a
batch pass and cached on disk keyed by content id, so encoder training never
re-runs the embedder. **[ANALYSIS]** the pin; bge-small's published numbers
are the model card's, not independently verified here.

### G8 — The commit pipeline's "formally checkable constraints" need an enumerated list [ANALYSIS]

Architecture §6.3: propose → schema-validate → commit, rejecting "only
formally checkable violations". The list, so the validator is a contract
rather than a mood (plan §3.2's warning: it **cannot** reject a semantically
wrong `same_as`, and must not be described as making predictions safe):

1. endpoint typing per G3's table;
2. id determinism and uniqueness (`graft.ids` derivations; a commit that would
   overwrite a different payload under the same id is a violation);
3. **provenance non-empty** on every edge (`Edge` already enforces it) and
   every provenance span resolvable in the snapshot;
4. **eligibility**: a node backed by an assertion (`ASSERTION_BACKED_NTYPES`)
   may enter the active graph only if `is_eligible` — fix F9's boundary,
   enforced at commit, so a quarantined assertion cannot become retrievable
   evidence through a decoder's enthusiasm;
5. interval arithmetic: `t_invalid` strictly after `t_created`;
   `superseded_by` implies `t_invalid` (already in `Edge`); a `valid_during`
   target must be a well-formed half-open interval;
6. no self-loops on `same_as` / `contradicts` / `supersedes`; no duplicate
   live edge of the same type between the same endpoints;
7. supersession writes `edge.invalidate` + the new edge — **nothing is ever
   deleted** (Zep precedent, flagged), and a `DEFER` parks the mention in a
   revisit queue keyed by entity anchor with no graph write at all.

Violation rate ≈ 0 **by construction** is then an exit criterion the same way
Phase 1 proved unconstructible-valid-terminals to 0: the pipeline refuses,
counts, and reports.

### G9 — "Corruption after sequential updates" is an exit criterion with no definition [ANALYSIS]

The architecture's Gate-1 row requires a "corruption-after-sequential-updates
audit"; no parent document defines it. Declared here, over the knowledge-update
questions' evidence streams (the only place updates are labelled):

Process each stream in turn order through the full commit pipeline. After
every commit, assert three properties on the snapshot:

* **(a) supersession is effective** — a superseded claim's edge is not
  `is_live`, and the superseding claim is;
* **(b) history is intact** — `ReplayGraphStore.at(seq)` *before* the update
  still returns the old fact live (bi-temporal recoverability, the reason
  nothing is deleted);
* **(c) no collateral flips** — the set of live edges that changed across the
  commit is exactly the committed edge plus its declared invalidations;
  anything else is corruption, counted per step.

The audit reports corruption counts per stream and runs green as a *harness
property* on synthetic decoder outputs before any learned decoder exists —
then on real decoder outputs at Gate 1, where (c) becomes a measured error
rate of the *decoders*, not the pipeline. **[ANALYSIS]**; the bi-temporal
property is Zep's, flagged.

### G10 — Gate 1's decision arithmetic must be declared before any comparison exists [ANALYSIS]

Phase 3 declared Gate 2's rule before training; the same discipline here,
against the Gate-0 contract's item 10:

* **Primary**: end-to-end mention-resolution score (action right, and for
  `LINK_EXISTING` the entity id right — plan §6.4's three-number D1 report
  beside it: four-way macro-F1 with `CREATE_NEW_ENTITY`/`NON_ENTITY` broken
  out, linking accuracy@1 given `LINK_EXISTING`).
* **Test**: McNemar's paired test on per-item correctness of the primary
  score, proposed vs each baseline, α = 0.05 two-sided — the pairing the
  power analysis in `data/phase2_5/power.json` was computed for (its D1
  `n_test` planning figure ≈ 627 at δ = 0.05, ψ = 0.2). **[EVIDENCE]** test
  selection per Dror et al. (ACL 2018); the α and the pairing are this
  project's own predeclaration, labelled as such.
* **Comparisons**: proposed (E3+D1..D4) vs (i) string/embedding-similarity
  linker (no learning), (ii) E1 MLP, (iii) E2 HGT, (iv) LLM-prompted linking.
  Seeds {13, 42, 7} for every trained method; identical training budget and
  early-stopping rule per §6 decision 9 (the Phase-3 capacity lesson:
  comparisons at unmatched budget are uninterpretable).
* **Stop-or-redesign** (plan Gate 1, verbatim): if the learned constructor
  does not improve component accuracy, **or** if oracle use of the graph
  cannot support the target questions (ceiling 2), C1 is in trouble — the
  answer is consolidation, not a weaker evaluation.
* D2–D4: macro-F1 (D2 four-way; D3 F1_c on DialogRE, micro-F1 on Re-DocRED;
  D4 TORQUE's exact-match/F1), with proposer recall (G5) and candidate recall
  (G4) printed in the same table, and calibration (Brier/ECE) after
  temperature scaling — **[EVIDENCE]** Guo et al. (ICML 2017) — fitted on dev
  only.

### G11 — The LLM-prompted-linking baseline spends real money and must be reproducible anyway [ANALYSIS]

Architecture §6.4: prompt a GPT-4o-mini-class API model with the mention, the
candidate list and recent context; parse the same four-way action + id. This
is what Mem0/Zep actually do (**[EVIDENCE]** Mem0's update pipeline is
LLM-prompted, ECAI 2025), which is the whole point — the baseline is faithful
to the deployed pattern, not a strawman.

Rules: the prompt lives in the **Stage-B prompt registry** (`graphbuild.prompts`), whose SHA is a component of `stage_b_fingerprint` — per-stage registries, because the Phase-5 registry's SHA is baked into frozen Phase-5 artefacts and must not move for a Stage-B edit; "one SHA covers every prompt" holds per stage; the model is pinned by provider version
string in the artefact; every call is **ledgered** (`llm_calls`,
`llm_tokens_*`) and **capped** by a §6-declared budget, enforced with
`would_exceed` before each call (the Phase-4 lesson: budgets are enforced, not
observed); every reply is **cached to disk keyed by prompt hash**, the cache
is committed as the run's record, and re-runs replay it — so the baseline's
numbers are reproducible after the budget is spent, exactly like the
`ReplayExtractor`. A local-model stand-in is *not* an acceptable substitute
(it would change the comparison's meaning), but the smoke path (G1) runs on a
recorded cache fixture, so no test spends money.

### G12 — Incremental training data must not leak the future through the graph [ANALYSIS]

Item 5's splits are user-level and chronological, but Stage B's training
examples are *(graph-state, item, label)* triples, and the graph state is the
leak vector: a D1 example at turn *t* whose candidate list was computed
against the *final* graph has seen entities created after *t* — and a D2
supersession label is trivially predictable if the superseding claim is
already in the graph. **Every training example is featurized against
`ReplayGraphStore.at(seq_t)`** — the snapshot as of that item's turn — which
the event log gives exactly and cheaply at pilot scale. If full-corpus
featurization measures slow, the remedy is the graphstore's own
TODO(Phase 6) checkpoint cache, keyed by seq — an optimization with an
existing name, not a new design.

---

## 2. Scope

**In.** `graft/graphbuild/`: schema freeze + endpoint table + commit
validator; D1/D2 item derivation from the Phase-5 log (the 2.5 tooling
upgraded); the pinned embedder with its disk cache; candidate generator and
pair proposer; encoders E1/E2/E3; decoders D1–D4 with temperature
calibration; the commit pipeline with DEFER queue, supersession and the
corruption audit; D3/D4 external-dataset loaders with declared mappings; the
LLM-prompted-linking baseline with cached, ledgered, capped calls; the
string/embedding-similarity baseline; the Gate-1 harness with its predeclared
rule; `scripts/phase6_gate1.py`; tests.

**Out.** Running Gate 1 (G1's conditions 1–2, both human-blocked); retrieval; the answerability
gate; reader work; any Phase-5 change beyond consuming its log; any frozen
value; `H`, `U`, `R`, masks, the lattice, the evaluator. **`graft.core`,
`graft.synth`, `graft.ingest` import no PyG/torch beyond their existing
boundaries; `graft.graphbuild` joins `graft.setgen` on the ML side of the
per-package test.**

---

## 3. Modules

### P6.0 Schema freeze, endpoint table, commit validator (do this first)

Freeze `Node`/`Edge` (G3): the endpoint table as a constant in `schemas.py`
beside `EDGE_TYPES`; `graft/graphbuild/validate.py` implementing G8's seven
checks over a `GraphSnapshot` + proposed commit, returning violations in
`CheckResult` style (reasons, never bare bools). No ML imports. The
Phase-1 validator precedent applies: reject only what is formally checkable.

### P6.1 `graphbuild/items.py` — D1/D2 items from the permanent record

`mentions_of(log)` → D1 items (mention span, turn, conversation, snapshot
seq); assertion pairs via the proposer → D2 items. Upgrades the Phase-2.5
item derivation to read Phase-5 logs, keeping the 2.5 item-id conventions so
the spike's guidelines and flagged examples carry forward
(`PHASE5_DECISIONS.md` §6). Emits annotation batches in the `annotate.py` CLI
shape — the human labels Gate 1 needs are collected with the tool the
annotator has already used, sized by the **live pilot's** measured yields —
0.754 mentions/turn and 0.895 assertions/turn over 248 turns
(`PHASE5_DECISIONS.md` §2.2a) — against item 8's items/hour. *(The spike's
0.59 and the bakeoff slice's 1.13 are both superseded for planning; the
pilot's is the production recipe's rate.)*

### P6.2 `graphbuild/embed.py` — the pinned embedder

G7's pin, batch pass, disk cache keyed by content id, F7 stage-sequential.
`embed_texts(texts) -> np.ndarray` plus cache management; nothing else knows
the model exists.

### P6.3 `graphbuild/candidates.py` — D1 candidates and the F8 pair proposer

G4 and G5. `candidates_for(mention, snapshot, k)` (name-normalized exact +
embedding top-k over `Entity` nodes, empty list legal) and
`pairs_for(claim, snapshot, s)` (top-s similar same-anchor claims). Both
report their recall against gold when gold is supplied — the two ceilings, in
the same module as the mechanisms that cause them.

### P6.4 `graphbuild/encoders.py` — E1, E2, E3

E1: GraphMixer-style (MLP link-encoder over feature vectors + neighbor
mean-pool + MLP head), plain torch. E2: 2-layer `HGTConv`, PyG (G2's
conditions; CompGCN fallback recorded if the platform check fails). E3: the
HGT backbone + the GRAFT feature set (bge-small embeddings, provenance flags,
time deltas, degree) — **[HYPOTHESIS]**, must beat E1 and E2 or the encoder
story is "HGT suffices" and the plan says so. One frozen interface:
`encode(snapshot_features) -> node_reprs`, so decoders never know which
encoder produced their inputs (fix F6's pattern).

### P6.5 `graphbuild/decoders.py` — D1–D4 and calibration

Four decoders per plan §3.2's grouping (the eight-heads design is dead —
`CLAUDE.md` §4.1 — and **D2 grouped-vs-split is the one ablation that
matters**, kept in the harness). D1 scores candidates ∪
{CREATE_NEW_ENTITY, NON_ENTITY, DEFER} — the Learn-to-Not-Link partition
(Findings ACL 2023) with RAED (EMNLP 2025) as the emerging-entity difficulty
evidence, ConEL-2/CREL (CIKM 2022) for the personal-entity convention ("my
car" creates). `DEFER` is **[ANALYSIS]**, no published precedent, ablated
on/off per the plan. D2: four-way mutually-exclusive pair decision. D3/D4:
typed-relation and temporal heads on the shared interface. Temperature
scaling on dev for every head (Guo et al., ICML 2017); Brier/ECE reported.
Class weights, never resampling (item 6); authority stays metadata-derived,
not learned.

### P6.6 `graphbuild/commit.py` — propose → validate → commit

The per-turn loop over eligible assertions (fix F9 enforced at entry *and*
re-checked by the validator): D1 on new mentions → entity nodes or DEFER
queue; D2 over proposed pairs → `contradicts`/`supersedes`/duplicate merges
(supersession = `edge.invalidate` + new edge, nothing deleted); D3/D4 edges
where their mapped confidence clears the declared floor; every commit through
P6.0's validator; every write an event-log op (`node.add`, `edge.add`,
`edge.invalidate` — the ops replay already handles). G9's corruption audit
lives here, runnable on synthetic decoder outputs.

### P6.7 `graphbuild/loaders.py` — D3/D4 external supervision

G6: pinned downloads (SHA + licence recorded), native-label training through
the shared interface, the declared GRAFT mapping with its losses tabulated in
the artefact. The loader is the *only* dataset-specific code; above it, the
trainer sees `(item, label)` streams.

### P6.8 `graphbuild/llmlink.py` — the LLM-prompted-linking baseline

G11: registry prompt, pinned provider model, ledgered + capped + cached
calls, replay-from-cache for tests and re-runs.

### P6.9 `graphbuild/gate1.py` + `scripts/phase6_gate1.py` — the harness

G10's predeclared rule, executable: trains all arms at the declared budget,
seeds {13, 42, 7}, McNemar on the primary, the three-number D1 report,
D2–D4 tables with their ceilings printed alongside, ceiling 2 (the graph
ceiling: does the committed graph contain a sufficient proof for each pilot
question, per the Gate-0 item-3 conversational proof definition), the
corruption audit, calibration plots, one JSON artefact. Runs end to end in
smoke mode on bootstrap labels (G1's discipline: plumbing only, no quotable
numbers, and the artefact is stamped `smoke: true` so it cannot be quoted by
accident).

### P6.11 `graphbuild/train.py` — the multi-seed trainer *(added 14 Aug 2026)*

**This module was missing from §3's original list, and its absence was a defect
in this plan rather than in the build.** Ten modules were named, none of them a
trainer, and build-order step 5 asked only for "one smoke epoch per arm" — which
the learning tests satisfy. So encoders forwarded, decoders backwarded and
calibration fitted, and nothing drove them through `TRAINING`'s budget across
arms and seeds, which is what a *comparison* needs. Recorded here rather than
left as an open item.

Decision 10 made executable: user-level stratified splits (item 5, seed
20260813), one budget dict for every arm, seeds {13, 42, 7} reaching
**initialisation** as well as the loop, early stopping that *restores* the best
state, class weights from train only, and the no-learning similarity control
tuning its one threshold on the same dev split. Two counters the honesty of the
numbers depends on: **unreachable gold** (the candidate generator never proposed
the right entity — a real ceiling, kept in the test set as a failure) and
**stale gold** (the label names an entity outside this graph's namespace — a
label-provenance problem, excluded from both splits and named).

### P6.10 `graft/tests/`

Per §5. Everything except actual encoder training runs on a bare interpreter:
the validator, items, candidates/proposer (embedding calls stubbed), commit
pipeline + corruption audit (stub decoders), loaders (fixture files), llmlink
(recorded cache), harness arithmetic (McNemar on constructed tables). Encoder
forward/backward smoke tests are marked for the ML environment, the Phase-3
pattern.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 0 | P6.0 freeze + validator | endpoint table declared; seven checks enforced; violation = reasons, not bool; no ML import |
| 1 | P6.1 items | D1/D2 items derive from the live-pilot log (187 mentions, 151 eligible assertions); 2.5 ids carry forward; annotation batches emit in the CLI shape |
| 2 | P6.3 candidates + proposer (stub embeddings) | recall-vs-gold plumbing works on bootstrap labels; empty-candidate cold start is an exercised path |
| 3 | P6.6 commit + P6.0 integration, stub decoders | replayed commits are idempotent; G9's audit green on synthetic outputs; F9 boundary tested (a quarantined assertion cannot commit) |
| 4 | P6.2 embedder + P6.4 encoders | HGTConv platform check passes (or CompGCN fallback recorded); E1/E2/E3 forward on pilot-scale graphs in 8 GB |
| 5 | P6.5 decoders + calibration | one smoke epoch per arm on bootstrap labels: losses decrease, temperature fits on dev, no NaN |
| 6 | P6.7 loaders | DialogRE/Re-DocRED/TORQUE pinned, mapped, losses tabulated; one smoke epoch each |
| 7 | P6.8 llmlink | cache-replay path tested; budget enforcement raises before overspend |
| 8 | P6.9 harness | smoke run end to end on the live-pilot log + bootstrap labels, artefact stamped `smoke: true`; Gate 1 is one command awaiting G1's conditions 1–2 |
| 9 | P6.11 trainer | every arm trains under one budget at three seeds; the decisive path exists and **refuses** without a signed contract, both decoders' labels, or the pinned embedder |

Step 0–3 before any torch import is deliberate: the commit pipeline is what
Phase 7 reads and what the corruption audit certifies, and none of it needs a
learner to exist.

---

## 5. Exit criteria

**The graph is sound by construction**
1. Endpoint table frozen; every commit validated; constraint-violation rate
   on the pilot corpus = 0, with the refusal reasons tabulated (G8).
2. Nothing is ever deleted: supersession invalidates and links; `at(seq)`
   before any update still serves the old fact (G9 property b, as a test).
3. The corruption audit runs green on synthetic decoder outputs (G9), and its
   per-step collateral check is a regression test.
4. Fix F9 holds at commit: a quarantined assertion cannot enter the active
   graph through any decoder path (negative test).
5. Replay determinism: rebuilding the graph from the log twice gives
   identical `state_digest`s; committed ops replay through the existing
   `GRAPH_OPS` with no new op invented silently.

**The learning machinery is real, and honestly smoke-only**
6. E1/E2/E3 forward and train one epoch on pilot-scale graphs within 8 GB;
   parameter counts and training budget recorded per arm (G10's budget rule).
7. D1 items derive from `mention.add` events alone (no re-extraction), and
   candidate recall@k is computed and printed beside every D1 number (G4).
8. Pair-proposer recall and the list-reply flood rate are measured on the
   pilot items (G5).
9. D3/D4 train on their native label sets through the shared interface; the
   GRAFT mapping's losses (dropped/merged classes) are tabulated in the
   loader artefact (G6).
10. Temperature scaling fits on dev only; Brier/ECE before/after reported
    (G10).
11. The Gate-1 harness runs end to end in smoke mode; its artefact is stamped
    `smoke: true` and quotes no comparison; the predeclared McNemar rule and
    seeds are in the artefact *before* any decisive run (G1, G10).
12. The LLM baseline's calls are ledgered, capped, and cached; the cached
    replay reproduces the run byte-identically; tests spend nothing (G11).

**Discipline**
13. Per-package ML boundary extended: `graft.graphbuild` may import
    torch/PyG; `graft.core`/`graft.synth`/`graft.ingest` boundaries
    unchanged, asserted per package (G2).
14. Every training example's features are computed against `at(seq_t)` — a
    test constructs a future-entity trap and asserts the candidate list
    cannot see it (G12).
15. All three annotation batches (D1, D2, adjudication) emit in the 2.5 CLI
    shape, sized against item 8's planning figures (P6.1).
16. `verify_handoff.py` gains the Stage-B fingerprint (embedder id+revision,
    encoder config, endpoint-table hash) alongside Phase 5's ingestion
    fingerprint.

---

## 6. Decisions to lock before writing code — **ADOPTED 14 Aug 2026**

*All fourteen rows adopted as recommended (`PHASE6_DECISIONS.md` §1), with two
measured amendments recorded there — decision 3's endpoint rows for
`supersedes`/`contradicts` widened to all assertion-backed types (§1.3a), and
the supersession scope widened to every currency edge (§1.3b) — plus one row
added by the audit, decision 15 below.*

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | PyG dependency | adopt `torch_geometric` pure-Python for E2's `HGTConv`, platform-checked at build; CompGCN fallback recorded as a departure if the check fails (G2) | E2 re-implemented by hand — the exact place a subtle bug reads as "weak baseline" |
| 2 | Embedder pin | `BAAI/bge-small-en-v1.5`, revision frozen at build, transformers-direct, CLS+L2-norm, disk cache (G7) | every embedding-derived feature and candidate list changes; Stage C inherits the swap |
| 3 | Endpoint table | G3's table, frozen with the schema | commit validator loses its contract; Phase-7 traversals meet unexpected edges |
| 4 | D1 candidates | `k = 10`, normalise() reuse, empty list legal, recall@k always reported (G4) | D1 numbers move with k; cold-start behaviour changes training data |
| 5 | Pair proposer | fix F8 verbatim: top-`s = 10` same-anchor; negatives from proposer only; recall reported (G5) | D2's ceiling moves; class balance changes → re-annotation risk |
| 6 | D2 rare-class handling | class weights, never resampling; the over-sampled pool declared (item 6) | quoted class balances become unreproducible |
| 7 | D3/D4 mapping | native-label training through the shared interface; declared GRAFT mapping with tabulated losses; DialogRE F1_c / Re-DocRED micro-F1 / TORQUE EM+F1 (G6) | "native supervision" misclaim — the contract's own red line |
| 8 | Calibration | temperature scaling on dev per head; Brier/ECE reported (Guo et al., ICML 2017) | decoder confidences uninterpretable; commit floors (row 9) meaningless |
| 9 | Commit confidence floors | D3/D4 edges commit only above a dev-chosen threshold after calibration; D1/D2 always commit their argmax action (they are decisions, not annotations) | graph density and ceiling 2 both move |
| 10 | Training budget | identical optimizer/epoch budget and early-stop rule across E1/E2/E3 and all Gate-1 arms; parameter counts reported; seeds {13, 42, 7} (G10) | the Phase-3 capacity lesson repeated at Stage B |
| 11 | Gate-1 rule | G10 verbatim: McNemar α = 0.05 on the primary, predeclared before any decisive run | post-hoc rule choice — fix F12's failure mode |
| 12 | LLM baseline budget | GPT-4o-mini-class, ledgered, cache-committed; hard cap declared at sign-off (a $ number, spent once) (G11) | an unreproducible baseline, or silent overspend |
| 13 | Corruption audit | G9's three properties, verbatim | Gate 1's "audit runs green" is unfalsifiable |
| 14 | Smoke discipline | bootstrap-label runs stamped `smoke: true`, never quoted; no proposed-vs-baseline inspection before Gate 1 proper (G1) | Gate 2's contamination rule violated one gate earlier |
| 15 | Feature variants *(added 14 Aug 2026 — the audit found E3 byte-identical to E2 because no feature builder existed)* | `base` (E1/E2): bias + log-degree + node-level sinusoidal RTE at day/week/month/year periods — a declared adaptation of HGT's per-edge relative temporal encoding, since PyG's `HGTConv` has no edge-time input; `graft` (E3): base + the four provenance flags + the pinned bge embedding. Only **live** edges enter message passing; reverse relations are an encoder convention (`features.encoder_metadata`), never schema edges | E3-vs-E2 stops being attributable to the feature set; a time-blind control repeats R13's weakened-baseline direction |

---

## 7. Explicitly not in Phase 6

No retrieval channels, no answerability gate, no reader, no Stage-D coupling
(Phase 9 consumes the encoder through fix F6's interface later), no `tau_nli`
or any frozen-value change, no Phase-5 write-path change, no schema field
additions beyond the freeze, no second implementation of name normalization,
no Gate-1 execution before its four entry conditions hold, and no quoting of
any number produced under `smoke: true`.

---

## 8. What Phase 7 will need from this, verbatim

* **The committed graph in the event log** — typed nodes and edges with
  provenance, versioning and eligibility respected — readable through
  `ReplayGraphStore.at()`, which is already Stage C's read interface.
* **The endpoint table**, because graph expansion's 2-hop walks are typed.
* **The embedder and its cache** (G7's pin) — Stage C's dense channel uses
  the same vectors; two embedders would make channel-fusion scores
  incomparable.
* **Entity nodes with alias sets** from D1's normalize-and-link decisions —
  the entity-match channel's index.
* **`valid_during` edges** — the temporal filter's substrate.
* **The Gate-1 verdict**, whenever its conditions allow it to run: Phase 7
  proceeds on the constructed graph either way (retrieval needs *a* graph),
  but Gate 1's stop-or-redesign rule decides whether the *learned* + proposed
  encoder story continues into Phase 9's featurizer.
