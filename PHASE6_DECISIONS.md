# Phase 6 — what the build decided, and what it measured

Date: 14 August 2026
Parent: `GRAFT_PHASE6_BUILD.md` (G1–G12, §6) · `GRAFT_EXECUTION_ARCHITECTURE_v1.md`
(Phase 6, fixes F8/F9) · `GRAFT_RESEARCH_PLAN_v1.md` v1.2 §3.2, §2.4, §6.3–§6.4,
Gate 1 · `GATE0_CONTRACT.md` (items 1, 3, 5, 6, 8, 9) · `PHASE5_DECISIONS.md` §6
Status: **built, audited and green (958 tests, 14 Aug 2026). Gate 1 has NOT run
and cannot** — two of its four entry conditions are human-blocked (G1). **The
trainer loop is now built** (P6.11, §8) and the decisive path exists; it
refuses without a signed contract, both decoders' labels, or the pinned
embedder. A two-source post-build audit (§7)
confirmed **21 defects — two of them blockers — and refuted 8**; every
confirmed finding is fixed and regression-tested, and the smoke run was rebuilt
from a driver that constructed no edges into a turn-ordered constructor that
exercises the write path end to end (§2.1).

Same convention as the other DECISIONS files: **[EVIDENCE]** = named paper, venue
stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = judgment
made here.

---

## 1. The §6 table, as adopted

§6 was **unsigned** when the build started; the build adopted all fourteen
decisions as recommended, as Phase 5 did. Four are recorded below because the
build learned something about them, and one is a departure.

### 1.1 Decision 1 — PyG is adopted, and the platform check passed

G2 made E2's dependency conditional on a build-time check, because
`torch_geometric`'s compiled extensions have no reliable Windows wheels and the
failure mode is a missing kernel at the first scatter rather than an import
error. **Measured, 14 Aug 2026:**

```
HGTConv forward AND backward execute
WITH_PYG_LIB = False, WITH_TORCH_SCATTER = False
```

Pure-Python, exactly as the condition required. **E2 is HGT; the CompGCN fallback
is not taken**, and `encoders.hgt_available()` re-runs the check rather than
trusting this record. `build_encoder("E2")` **refuses** if it ever fails —
falling back silently would report a number for an encoder that never ran.

One dependency arrived that nobody chose: `torch_geometric` 2.8 hard-imports
`xxhash`. Pinned so the manifest does not move for a reason no one picked.

### 1.2 Decision 2 — the `Certificate` node type is dropped (a departure)

Plan §3.2 lists **eleven** node types; `NODE_TYPES` has **ten**. The missing one
is `Certificate`, and it appears nowhere else — not in the built code, not in any
DECISIONS file, not in any other plan section. Phase 1 produced `CheckResult`,
not a certificate node, and no component writes or reads one.

The plan wins conflicts (`CLAUDE.md` §2), so this could not be closed by silence
in either direction. **It is dropped deliberately**: a node type nothing writes
is a schema promise the graph cannot keep, and adding an unwritten type to a
*frozen* vocabulary would make the freeze meaningless. Pinned by a test asserting
`len(NODE_TYPES) == 10`, so restoring it is a deliberate edit against a failing
assertion plus a `SCHEMA_VERSION` bump — which is the friction the freeze exists
to create.

### 1.3 Decision 3 — the endpoint table, and what it does *not* do

Eleven rows, derived from plan §3.2's prose, frozen beside the vocabularies as
`schemas.ENDPOINT_TABLE`. Before it, "schema-validate" had nothing formal to
check: a decoder could propose `valid_during: Entity → Turn` and the pipeline had
no ground to refuse it on.

**What it bounds is narrow, and the write-up must say so.** Plan §3.2's own
words: the validator bounds *a specific, enumerable class of damage*. It cannot
reject a semantically wrong `same_as` — a merge of two different people with
similar names is type-correct, provenance-bearing, id-unique and completely
wrong. "Makes neural predictions safe" is the overreach `CLAUDE.md` §5
catalogues, and the module docstring says the opposite in its first paragraph.

### 1.3a Decision-3 amendment — `supersedes`/`contradicts` widened to all
assertion-backed types (14 Aug 2026, by measurement)

As first derived, both relations were `Claim → Claim`. But D2 decides over
*assertion pairs of every kind* — the pair proposer walks Claim/Value/Event
anchors, and plan §3.2's "claim pair" uses claim in the assertion sense — and
the canonical knowledge update the corpus actually contains ("my weight is 70kg"
→ "68kg") is a **Value** pair. Measured on the live pilot: **20 of 151 eligible
assertions (13.2%) are value/event-kind**, and under the narrow rows every
supersession or conflict among them was refused on endpoint typing — the update
case Contribution 1 exists for, structurally unrepresentable. Both rows now
admit `{Claim, Value, Event}` on both sides; `SELF_LOOP_FORBIDDEN` is unchanged.
The endpoint-table hash moved with the amendment, as it is designed to.

### 1.3b The supersession scope — every currency edge, not only `about_entity`
(14 Aug 2026, by measurement)

`Committer.supersede` originally invalidated only the old claim's
`about_entity` edges; the audit measured a superseded claim keeping its
`valid_during` edge **live**, so the retired fact kept answering temporal
queries and `H` kept accepting it as current. It now invalidates **every live
edge incident to the superseded node except the history relations**
(`supersedes`, `contradicts`), which stay live deliberately: a superseded claim
was still contradicted by what contradicted it, and retiring the record of a
disagreement is the erasure non-destructive versioning exists to prevent.

### 1.4 Two implementation decisions the plan left open

**(a) Entity ids are conversation-scoped.** Two users who both mention "my car"
have two different cars. A globally-keyed entity would merge them — a wrong
merge, the most damaging Stage-B error (plan §8 risk #1), introduced by the *id
scheme* rather than by a decoder. Cross-conversation identity is what `same_as`
is for, and that is a decision a decoder makes and the log records.

**(b) `d2_items` returns nothing without a proposer**, rather than falling back
to all-pairs. `GATE0_CONTRACT.md` item 6 requires negatives from the proposer
only: a random negative is trivially `INDEPENDENT` and would inflate macro-F1 on
the class nobody cares about. An empty batch is an honest "no proposer was
supplied"; a fabricated one would poison the class balance the contract fixes.

### 1.4a Three further implementation decisions, made with the audit fixes

**(c) Alias growth is the one permitted payload evolution.** §1.5 promised
"alias sets accumulate through D1's linking decisions", but the validator's id
check refused *any* payload change under an existing id — so the promise was
unimplementable as written. The check now admits exactly one evolution: an
``Entity`` rewrite identical in every field except a **superset** of aliases
(`validate._is_alias_growth`). Growth cannot change identity — the id derives
from the canonical name, which must be unchanged — replay is last-write-wins so
resubmission is monotone, and anything else under the same id is still a
collision. `Committer.add_aliases` is the write path; the raw surface form is
also kept as an alias at creation when it differs from the canonical.

**(d) One assertion, one live node.** The kind→ntype mapping applied twice
(`link_existing(a, "claim")` then `create_entity(a, "value", …)`) derived two
node ids for one assertion and passed every per-node check — one fact counted
twice, with two live `about_entity` edges. The id check now refuses an
assertion-backed node whose assertion already backs a *different* node.

**(e) Atomicity is by resubmission, not by write — worded, ordered, and
tested.** A `Commit` is N log appends with no transaction; the appends are
deliberately ordered adds-then-invalidations so a torn commit leaves the
*recoverable* state (old and new both live and visible) rather than the silent
one (a fact invalidated with its successor missing), and re-submitting the same
commit converges — identical rewrites are idempotent, a live edge equal to
itself is not a duplicate. The regression test walks a torn supersession to
convergence. *(The earlier docstring said "atomic write", which the audit
correctly called false.)*

### 1.5 Two defects the build found in its own work

**The validator caught the node builder.** `entity_node` derived its id from the
*normalized* name but stored the *raw* one in the payload, so "Fitbit Charge 3"
and "fitbit charge 3" produced one id with two payloads — refused for id
collision on the first smoke run over the pilot log. Fixed by storing the
normalized name as canonical and letting **alias sets accumulate through D1's
linking decisions** (the shape Phase 7 is promised in §8). The alternative —
putting the raw form in the id — is worse: it makes two surface forms two
entities, which is the duplicate-entity bug normalization exists to prevent.
*The check that caught it is check 2 of seven, doing exactly its job.*

**The smoke run contaminated the Phase-5 pilot log.** It appended `node.add`
events to `artefacts/phase5_pilot/events.jsonl`, so the digest recorded in
`PHASE5_DECISIONS.md` §2.2a stopped reproducing — a silently invalidated
experimental record. The log was repaired by truncating at the first `node.add`
(Phase 5 wrote none) and **verified back to its recorded digest
`1ede59c5…`**. The script now copies the Stage-A log into
`artefacts/phase6/` and builds there: **a Stage-A log is an input, and inputs are
copied, not extended.**

---

## 2. What the build measured

### 2.1 The smoke run (G1, decision 14) — rebuilt 14 Aug 2026

`artefacts/phase6_gate1_smoke.json`, built from the live-pilot log by
`graphbuild.standin.construct` — the turn-ordered stand-in constructor that
replaced the first driver after the audit (§7) found it committed **bare entity
nodes only**: zero edges, zero D2 items, every candidate list empty, and a
corruption audit green over a graph with nothing in it to corrupt.

| | First run | **Rebuilt** | Reading |
|---|---|---|---|
| D1 items | 187, **0** with candidates | **187, 177 with construction-time candidates** | candidates are drawn from the constructor's own snapshot, per mention, before that mention's commit — G12 by construction, and each item pins the exact position (`stage_b_seq`) |
| D2 items | **0** (sequence arithmetic, misread as a finding) | **120** | pairs proposed at link time, against anchors that exist because the constructor links |
| commits | 187 accepted / 0 refused | **242 / 0** | entities, alias growths, and `about_entity` links |
| graph | 111 nodes, **0 edges** | **241 nodes, 130 edges** | the shape Phase 7 is promised, actually present |
| corruption audit | green over node-only commits | **GREEN over 242 commits including every edge write** | the audit now audits something |
| comparisons | None, suppressed | None, suppressed | unchanged — the stamp suppresses, never labels |

**The first table's "D2 = 0 is a real finding" reading was wrong, and is
retracted.** The zero was sequence arithmetic: items were derived *before*
construction, against Stage-A positions that precede every Stage-B commit, so
the proposer could never see an anchor on any corpus. G5's ordering point (D2 is
downstream of D1) stands, but the measured zero demonstrated the harness defect,
not the ordering.

**`run_gate1` suppresses the comparison rather than labelling it.** A comparison
that existed under a `smoke: true` stamp would be quotable by anyone who read
past the stamp — the bootstrap-labels mistake one phase later. Per-arm reports
are computed *when arms are supplied*; the smoke run supplies none, because no
trained arm exists — `arms: []` and `per_arm: {}` in the artefact are the
honest reading of that. *(An earlier version of this section said the per-arm
reports "are computed" as though the smoke run exercised them; it did not, and
the harness's report shape is exercised by `test_graphbuild_learning.py`
instead.)*

**The decisive path is a named refusal, not a hidden smoke run.** The first
driver hardcoded `smoke=True` on every path, so "Gate 1 when conditions hold"
silently produced another smoke artefact. The driver now refuses a decisive run
even with all four entry conditions met, naming the missing piece: **the
multi-seed trainer loop is not built** (§4). There is deliberately no code path
that can produce a quotable Gate-1 number today.

### 2.2 The external datasets (G6, decision 7) — and the mapping loss

All three fetched, SHA-pinned, licences read from source on 14 Aug 2026:

| Dataset | Decoder | Licence | Native classes | Mapped | **Instance coverage** |
|---|---|---|---|---|---|
| DialogRE | D3 | **non-commercial research only** | 35 | 6 | **27.1%** |
| Re-DocRED | D3 | MIT | 96 | 6 | **4.88%** |
| TORQUE | D4 | Apache-2.0 | — | structural | — |

*(Corrected 14 Aug 2026: the first version of this table quoted 33/5/24.5% and
94/6/4.8% — numbers computed over the smoke artefact's 200-document prefix and
presented as though they covered the corpora. These are the **full-split**
figures — DialogRE 5,963 items, Re-DocRED 85,932 — and the regenerated artefact
now parses the full splits, `limited_to_first_n_docs: null`.)*

**This is the number G6 exists to expose, and it is worse than "some loss".**
Mapping Re-DocRED onto GRAFT's schema discards 90 of 96 Wikidata properties and
95% of labelled instances.

**TORQUE's mapping is structural and stops there, said plainly:** its answers
are event spans, not intervals, and **no conversion from a TORQUE answer to a
`valid_during` interval exists or is claimed**. D4 trains and is scored on the
native span-selection task; the `valid_during` *commit path* now exists
(`Committer.valid_during`, floor-gated per decision 9), but what feeds it on
conversation is Gate-1-time work with its own declared mapping, not a TORQUE
by-product. It is not a defect in the mapping: GRAFT's eleven
relations are about *provenance, identity, time and conflict*, and these datasets
annotate *interpersonal and biographical facts*. It is the concrete content of
"supervision **interface**" — and it is why D3/D4 **train and are scored on the
native label sets**, with the GRAFT mapping a separate, declared, lossy
application whose cost is published beside it. `GATE0_CONTRACT.md` item 1's red
line is that an adaptation is never presented as native supervision; this table
is what keeps that checkable.

`P31` (instance-of) and `P279` (subclass-of) are deliberately **not** mapped to
`same_as`: doing so would assert that a dog is identical to the concept "dog",
which is the merge error the whole validator exists around. Regression-tested.

### 2.3 Capacity, and a direction worth flagging now — remeasured 14 Aug 2026

*(The first version of this section reported E2 = E3 = 428,300 — measured over a
toy two-type metadata, and **equal because E3 was structurally identical to E2**:
no feature builder existed anywhere, so the proposed encoder was the baseline.
The audit confirmed it; decision 15 and `graphbuild.features` are the fix.)*

Measured at the shared `hidden = 128`, over the **full schema metadata**
(`features.encoder_metadata()` — every endpoint-table relation plus its `rev_`
counterpart, which is why HGT's per-type/per-relation parameters dwarf the toy
figure):

| Arm | Input | in_dim | Parameters |
|---|---|---|---|
| E1 (MLP) | `base` features | 10 | **50,816** |
| E2 (HGT) | `base` features | 10 | **3,760,188** |
| E3 (HGT + GRAFT features) | `graft` features | 398 | **4,256,828** |

Three readings, all of which belong beside any future Gate-1 number:

* **E2's 74× over E1 is architecture-intrinsic**, not a knob: HGT allocates
  parameters per node type and per relation (that is its published mechanism),
  and an MLP has none of that. Decision 10 requires the *budget* identical and
  the counts *reported* — it does not require capacity parity across
  architecture families, which would mean crippling one of them.
* **E3's +496,640 over E2 is exactly the input projection of the feature set**
  (10 types × (398−10) × 128) — so the capacity delta is mechanically
  attributable to the one mechanism E3 adds, which is the L7 discipline this
  arm was held to.
* The direction still favours the proposed arm, which is the direction Phase 3's
  R13 audit found three control defects running in. Phase 3's own resolution
  (minimality plus "control never smaller") is the precedent if Gate 1 needs
  one.

---

## 3. Exit criteria — where each one stands

| # | Criterion | State |
|---|---|---|
| 1 | endpoint table frozen; every commit validated; violation rate 0 with reasons tabulated | **green, and now over a real graph** — 242/242 accepted on the pilot *including 130 edge writes and alias growths* (the first run's 187/187 was entity-nodes-only, so five of seven checks were never exercised by it); the table carries the 14 Aug amendment (§1.3a) |
| 2 | nothing deleted; `at(seq)` before an update still serves the old fact | **green**, as a test |
| 3 | corruption audit green on synthetic decoder outputs; collateral check a regression test | **green** — and on the rebuilt smoke run it audits 242 commits with edges, not 187 node-only writes |
| 4 | F9 holds at commit: a quarantined assertion cannot enter through any decoder path | **green**, negative test at both validator and pipeline level |
| 5 | replay determinism; no new op invented silently | **green** — ops ⊆ `GRAPH_OPS` ∪ {`mention.add`}, the one documented non-graph op |
| 6 | E1/E2/E3 forward and train within 8 GB; parameter counts recorded | **green** — forward *and* backward, **each arm on its declared feature variant** (decision 15): E3 consumes the GRAFT set and is no longer parameter-identical to E2 (§2.3) |
| 7 | D1 items from `mention.add` events alone; candidate recall@k printed beside every D1 number | **green — actually true now.** *(This row was previously marked green while the code recovered mentions by span set-difference, the retired heuristic; a false green, caught by the audit.)* 187 items from `mention.add`, 177 with construction-time candidates; the recall *number* needs gold links |
| 8 | proposer recall and the list-reply flood rate measured on pilot items | **machinery green, items exist** — 120 D2 items from the rebuilt constructor; the recall/flood *numbers* need gold pairs |
| 9 | D3/D4 train on native label sets; mapping losses tabulated | **green** — full-split losses in the artefact (§2.2); the first table's prefix-derived numbers are corrected |
| 10 | temperature scaling fits on dev only; Brier/ECE before/after | **green** |
| 11 | harness runs end to end in smoke mode; artefact stamped and quotes nothing; rule predeclared | **green, with the claim scoped honestly**: the smoke run exercises the *write path* end to end (items → candidates → commits → audit); encoders/decoders/calibration are exercised by the learning tests, and the decisive path refuses by name (§2.1) |
| 12 | LLM baseline ledgered, capped, cached; replay reproduces; tests spend nothing | **green** — refuses without a signed budget; the gate refuses the call that *would* overshoot (not merely after); a cache miss in replay mode raises; the prompt is in the Stage-B registry and its SHA is in the fingerprint |
| 13 | per-package ML boundary extended to `graft.graphbuild`; other boundaries unchanged | **green**, plus the containment guard: importing `graft.graphbuild` pulls in no ML library |
| 14 | every training example featurized against `at(seq_t)`; a future-entity trap is refused | **green** — every item carries both pins (`snapshot_seq` for Stage A, `stage_b_seq` for the construction log), candidates are drawn from the constructor's own snapshot, and the trap test exists: the first mention's candidate list is empty, a later same-surface mention sees the earlier entity, and a same-surface mention in *another conversation* sees nothing |
| 15 | all three annotation batches emit in the 2.5 CLI shape | **partly** — D1/D2 items emit in the 2.5 shape (with construction-time candidates); the adjudication batch is not built |
| 16 | `verify_handoff.py` gains the Stage-B fingerprint | **green** — embedder, constants, budget, commit floors, the endpoint-table hash, and (since 14 Aug) the Stage-B prompt-registry SHA |

---

## 4. Open items, honestly

| Item | Blocks | Note |
|---|---|---|
| **Gate 1 cannot run** | the C1 verdict | two of four entry conditions are human-blocked: a signed Gate-0 contract (item 8) and human D1/D2 labels — **D1 annotation has started** (`d1_labels_Sabbir.jsonl` exists); D2 has not. `scripts/phase6_gate1.py` evaluates all four (requiring *both* decoders' label files, since 14 Aug) and refuses by name |
| ~~No trainer loop~~ — **CLOSED 14 Aug** | — | Built as P6.11 (§8), with the nine defects its own review found already fixed. The driver's decisive path exists and refuses on the remaining entry conditions |
| **Every `LINK_EXISTING` gold label is stale** | D1's link supervision | the 34 human labels name entity ids from the spike's positional namespace; **0 of 7 link labels** name a node in the current graph, so the arms can currently learn only the three non-link actions. Re-annotating the **current** D1 batch (187 items, emitted by P6.1) is what makes link supervision exist — and it is the same batch Gate 1 needs anyway |
| **Only 17 of 34 human labels transfer** | the usable label count | labels were collected against the spike's extraction; candidate B proposes different spans. Not a defect — a cost of having changed extractor, measured and reported by `load_d1_gold` |
| ~~Criterion 14's trap test~~ — **CLOSED 14 Aug** | — | `test_candidates_cannot_see_the_future` + the cross-conversation variant, on the real constructor |
| ~~D2 items are 0 on the smoke run~~ — **RETRACTED 14 Aug** | — | the zero was the harness deriving items before construction (§2.1); the rebuilt constructor yields 120 |
| **The LLM baseline has never been called** | Gate 1's fourth arm | by design: `max_usd` is `None`, and decision 12 requires a dollar cap declared at sign-off. The cache-replay path is tested; no test spends money |
| **Capacity asymmetry across arms** | reading any Gate-1 result | §2.3's remeasured table: E2 is 74× E1 (architecture-intrinsic), E3 is E2 + exactly the feature projection |
| **The adjudication batch is not built** | exit criterion 15's last third | D1/D2 item batches emit in the 2.5 CLI shape; the disagreement-adjudication batch does not exist yet |
| **Mention/`mentioned_in` nodes are deliberately uncommitted** | nothing now | mentions live in the log (`mention.add`) and D1 items derive from there; committing `Mention` nodes awaits a real D1 whose decisions give them edges worth having. Recorded so the unwritten node type is a decision, not an oversight — the `Certificate` discipline applied consistently |

---

## 5. Handoff to Phase 7

* **The committed graph** in the event log, readable through
  `ReplayGraphStore.at()` — already Stage C's read interface.
* **The endpoint table**, because graph expansion's 2-hop walks are typed.
* **The embedder and its cache** (`graphbuild.embed`) — Stage C's dense channel
  must use the same vectors, or channel-fusion scores are incomparable.
* **Entity nodes with alias sets**, canonical-name keyed and conversation-scoped.
* **The Stage-B fingerprint**, alongside Phase 5's ingestion fingerprint.
* **The Gate-1 verdict, whenever its conditions allow it to run.** Phase 7
  proceeds on the constructed graph either way — retrieval needs *a* graph — but
  Gate 1's stop-or-redesign rule decides whether the learned constructor is what
  Phase 7 should be reading.

---

## 7. The 14 Aug 2026 post-build audit — every finding, its verdict, what moved

Two sources, deliberately independent: an adversarial audit workflow (four
lenses over the built modules, every finding handed to a refuter told to kill
it, most verdicts settled by *running* the code against the real pilot log) and
the project owner's own review. Between them: **21 confirmed defects — two
blockers — and 8 findings investigated and refuted.** Everything confirmed is
fixed and regression-tested; the suite grew from 900 to 924.

### 7.1 The two blockers — one number serving two streams

`d1_items` computed candidates against `at(mention.seq)` and `d2_items`
proposed pairs against `at(eligibility.seq)` — both **Stage-A** sequence
numbers, and every Stage-B commit appends *after* the whole Stage-A prefix. So
the candidate list was empty for every mention **on every corpus**, and the
pair proposer could never see an anchor: reproduced on the smoke run's own log
(mention seqs 0–1046; first `node.add` at seq 1498). The G12 guard was real,
pinned to the wrong stream.

**The fix is structural**: candidate generation and pair proposal moved into
the constructor (`graphbuild.standin.construct`), which processes turns in
Stage-A order and draws from **its own snapshot at that moment** — past-only by
construction. Items carry both pins: `snapshot_seq` (Stage A) and `stage_b_seq`
(the construction log position the candidates were actually drawn from,
reproducible via `at()`). The trap test closes exit criterion 14.

### 7.2 Confirmed in the built modules — fixed, each with a named test

| Finding | Fix |
|---|---|
| **Mentions recovered by span set-difference** — the exact heuristic Phase 5 built `mention.add` to retire; a mention coinciding with a quote span vanished, a crash-repaired one duplicated (both reproduced through the real write path; the pilot log agreed 187=187 by luck) | `mention_records` reads `mention.add` alone, deduped on span id; exit criterion 7's false green corrected |
| **`supersedes`/`contradicts` unrepresentable for value/event kinds** — 13.2% of the pilot's eligible assertions; the corpus's own weight-update case refused on endpoint typing | decision-3 amendment (§1.3a): both relations widened to `{Claim, Value, Event}` |
| **`supersede()` retired only `about_entity`** — a superseded claim's `valid_during` stayed live and kept answering temporal queries | every live incident edge invalidated except the history relations (§1.3b) |
| **One assertion → two live nodes** via the kind→ntype mapping, zero violations | one-assertion-one-node guard in the id check (§1.4a-d) |
| **"Atomic write" claim false** — N unguarded appends | reworded + ordered adds-then-invalidations + torn-commit convergence test (§1.4a-e) |
| **Alias accumulation promised (§1.5) but unimplementable** — the id check refused all payload change | `_is_alias_growth` exception + `Committer.add_aliases`; raw surface kept as alias at creation (§1.4a-c) |
| **Candidate scan unscoped by conversation** — one user's mention could link to another user's entity, and the entity payload carried no `conv_id` to filter on | `PAYLOAD_CONV_ID` in the payload; `candidates_for` filters by the mention's conversation; cross-conversation trap test |
| **D2 sides ordered by id-hash** — broke the guidelines' "claim_b is the later session" invariant that the CONFLICT-vs-SUPERSEDES rule depends on; ~half the collected labels would have meant the opposite | ordered by `(session_date, turn_id, assertion_id)` |
| **O(N²) log replay** in item derivation | snapshot hoisted |
| **E3 was byte-identical to E2** — no feature builder existed anywhere, so the proposed encoder *was* the baseline, and both were time-blind (HGT's architecture row names relative temporal encoding as load-bearing) | `graphbuild.features` (decision 15): `base` = bias+log-degree+RTE for E1/E2, `graft` = base+provenance flags+pinned embedding for E3; RTE declared as a node-level adaptation of Hu et al.'s per-edge mechanism; capacity table remeasured (§2.3) |
| **`rev_about_entity` invented ad hoc in a test** — HGTConv starves source-only node types, and the reverse-relation workaround lived nowhere exportable | `features.encoder_metadata()`: endpoint table expanded to concrete triples plus `rev_` counterparts, documented as an encoder convention, never schema edges |
| **LLM prompt in no registry** while the docstring claimed Phase-5 registry membership — a prompt edit would have moved no recorded hash | `graphbuild.prompts` registry, SHA folded into `stage_b_fingerprint`; the per-stage form of "one SHA covers every prompt", chosen because the Phase-5 SHA is baked into frozen artefacts |
| **Budget gate authorised the overshooting call** — it checked only money already spent | refuses when `spent + est_max_usd_per_call > cap` |
| **`d1_report` zip-truncated mismatched lengths** — an arm answering half the items scored over the half it chose | length guard raises |
| **Smoke run exercised almost nothing** (§2.1's table) and the driver had **no decisive path** — `smoke=True` hardcoded everywhere; the labels check passed on any single non-bootstrap file | `standin.construct` + the rebuilt driver: real edges, named refusal on the missing trainer, both decoders' label files required |
| **§2.2's dataset numbers were prefix-derived** (200 docs) but presented as corpus figures | full-split numbers (35/6/27.1%; 96/6/4.88%), artefact parses full splits |

### 7.3 Refuted — investigated, deliberately not "fixed"

* **"Check 7 never requires a supersedes edge to invalidate"** — the harmful
  shape is unreachable: `Committer.supersede` is the only production
  constructor and computes invalidations from every live edge itself; a bare
  supersedes after the target was already retired is legal history, and
  enforcing the inverse would refuse it.
* **"D2's DUPLICATE has no commit path"** — DUPLICATE's graph effect is a
  non-write (the duplicate is not instantiated twice); `submit(Commit)` is the
  general API and no decoder-orchestration layer exists yet for *any* action,
  so nothing is selectively missing.
* **"Re-invalidation destructively overwrites `t_invalid`"** — no built caller
  can name a dead edge; `supersede` guards with `is_live`.
* **"`supported_by`/`mentioned_in`/`retired_by` unconstructible"** — they
  construct fine when the endpoint node ships in the same commit, which is the
  documented `_ntype_of` route.
* **"The corruption audit passes a supersession cycle"** — the cycle is
  unproposable: a superseded claim's anchor dies with its live edge, so the
  pair proposer can never re-propose the reverse pair.
* **"`duplicate_edge` is unreachable / re-assertion refused"** — a restated
  fact arrives as a *new* assertion id (spans differ), commits cleanly; the
  refusal of a byte-identical re-add is the guard, not a bug.
* **"Torn commit leaves a permanently forbidden state"** — the built driver
  rebuilds its log per run; and with the §1.4a-e ordering, the torn state is
  recoverable by resubmission (now tested).
* **"Positional item ids break label durability"** — the 2.5 id convention is
  prescribed by the plan, and `span_id` on every item is the durable key.

One user-raised item was **considered and declined**: a validator check that
ids equal their prescribed content derivation. The id constructors are the only
production writers, a derivation check cannot know the inputs for every node
type, and the damage a hand-rolled id could do (duplicate content under two
ids) is already what the duplicate-edge and one-assertion-one-node checks catch
at the level where it matters.

### 7.4 What the audit changed in the frozen-value record

The **endpoint table** carries the §1.3a amendment (its hash moved, as
designed); `stage_b_fingerprint` now also covers the **Stage-B prompt-registry
SHA**; `PAYLOAD_CONV_ID` joined the payload vocabulary; `LLM_BASELINE` gained
`est_max_usd_per_call`; and decision 15 (the two feature variants) is new. No
Phase-5 or earlier frozen value moved, and the Phase-5 pilot log's digest still
matches its recorded artefact — verified after every wave.

---

## 8. The trainer (P6.11), and the review that followed it — 14 Aug 2026

### 8.1 Why it was missing, which is the part worth recording

`GRAFT_PHASE6_BUILD.md` §3 listed ten modules and **none of them was a
trainer**; build-order step 5 asked only for "one smoke epoch per arm: losses
decrease, temperature fits on dev, no NaN", which `test_graphbuild_learning.py`
satisfies. So the build did exactly what the plan specified, and what was
missing was the *specification*. §4 recorded it as an open item — which reads
as a scheduling fact and hid a plan defect. Nothing external blocked it: the
labels, the items, the graph and the budget dict all already existed.

The plan now carries **P6.11** and a ninth build-order row.

### 8.2 What the trainer is

Decision 10 made executable: one `TRAINING` dict for every arm; seeds
{13, 42, 7} reaching **initialisation** as well as the loop (the first version
seeded only the loop, so three seeds would have estimated variance over dropout
alone — caught by its own determinism test); early stopping that **restores**
the argmin-dev-loss state rather than merely stopping; class weights from
**train only**, never resampling; and the no-learning similarity control tuning
its single threshold on the same dev split, so "tuned on dev, scored on test"
is uniform rather than an exemption granted to the simple method.

`adjudication_items` closes exit criterion 15's last third: the disagreement
batch item 7 asks for, carrying both passes' labels and a blank column, with an
`action_level_disagreement` flag because the κ convention collapses
`LINK_EXISTING(<id>)` to its action while an id-only disagreement is still a
real thing to resolve.

### 8.3 The review — nine confirmed, fifteen refuted

An adversarial review of the new code (three lenses, every finding handed to a
refuter, most verdicts settled by execution) found **nine real defects in code
written hours earlier — two of them blockers** — and refuted fifteen. All nine
are fixed and regression-tested.

**The blocker, in two halves.** Gold labels were joined to items by
`item_id` — and positional ids are re-issued from zero by every derivation. The
spike's `d1_0000` is "Gitzo" in session `_2`; the Phase-6 log's is "Lowepro
ProTactic 450 AW" in session `_1`; **zero of 34** spike items name the same
mention as the Phase-6 item with the same id. Every label would have attached
to a different mention. The key is now `span_id(turn_id, start, end)` — a fact
about the corpus rather than about emission order — and **17 of 34** labels
transfer, which is itself reported: labels collected against one extractor's
mentions only partly survive a change of extractor.

Fixing that exposed the second half: of the labels that do transfer, every
`LINK_EXISTING` one names an entity id from the **spike's namespace**
(`e_41698283_000`, minted positionally by `build_items.py`), and Phase 6 derives
entity ids by content hash — **0 of 7 exist in this graph**. Under the first
fix they would have been silently counted as candidate-generator recall misses,
blaming the generator for a label-provenance problem. They are now excluded from
both splits and named as `stale_link_labels_excluded`, with the distinction
between *unreachable* gold (a real ceiling) and *stale* gold (a stale label)
made explicit in code and in the report.

**The other seven, each with its regression test:**

| Finding | Fix |
|---|---|
| `--smoke --decisive` skipped **all four entry conditions** and wrote an artefact stamped `smoke: false` with a full McNemar table — the condition block was `if not args.smoke` | the two flags are mutually exclusive, enforced by argparse |
| `--decisive` silently used the **stub embedder**, filling 384 of E3's 398 input dimensions with meaningless noise — the arm's entire declared difference from E2 would have been hash values | refuses without `--real-embedder`, by the same argument `build_encoder` refuses a silent CompGCN substitution |
| The relative temporal encoding anchored at the **latest edge in the whole construction log** — i.e. another conversation's calendar; one conversation's newest node carried Δ = 203.9 days instead of 0, across 8 of the 10 dimensions E1 and E2 consume | features are timed against the **item's own turn** (`turn_ts`), which is what a *relative* encoding should mean; the cache keys on the reference |
| An empty or non-finite dev loss **silently restored the random initialisation** and reported it as a normal early stop — reachable, since `n_dev = round(n·0.2)` is 0 for n ≤ 2 | raises, naming the counts |
| A decisive run emitted a full comparison table at **any** test size including 0 — a p = 1.0 that reads as a measured null | `run_gate1` refuses an empty test split; the report carries `n_test` beside the power analysis's planning figure of 627 and the sentence that a non-significant result at this n means "not measured", never "no difference" |
| `GATE1_RULE` predeclares five arms; the decisive path builds four | the artefact records `arms_declared` and `arms_omitted` by name, so the difference from the verbatim rule block is stated rather than inferred |
| Splits were never stratified — D1 items carry no `question_type` | the driver reads types from the pinned corpus and passes them; unavailability is recorded, not silently ignored |
| `epochs=0` raised `UnboundLocalError` | counters bound before the loop |

**Refuted and deliberately not acted on** (fifteen; the pattern is worth
keeping): the dead `fold(a)==fold(b)` conjunct in the adjudication skip
condition, the similarity arm's tie-break on a flat dev curve, `adjudication_items`
joining on `item_id` (both passes are over the *same* batch by construction, so
the id is the right key there), `batch_size` being declared and unconsumed (real,
but full-batch on dozens of items is the same computation), the claim that the
D1 split moves D2's rare classes (D2 is not split or trained by this code), and
the claim that early stopping is inert (measured: it fires, and restoration is
what makes dev selection real).

### 8.4 What this does not change

No frozen value moved. The Phase-5 pilot log still matches its recorded digest.
Gate 1 still cannot run: two entry conditions are human-blocked, and the driver
now refuses on a third ground — the pinned embedder — as well.
