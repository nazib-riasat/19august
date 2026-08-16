# Phase 7 — what the build decided

**All seven steps built and green, 15 August 2026.** Steps 0–5 first; **step 6,
the GNN scorer, was built later the same day once `GATE0_CONTRACT.md` was
signed** — the signature was its only blocker. **It has not been trained**: that
is GPU/CPU work and is deferred, so exactly one of step 6's three "done when"
conditions is unmet (§5).

**Audited the same day — read §7 before quoting anything from §3.** A
two-source post-build audit (an external review whose eight claims all
confirmed, plus an eight-agent verification pass) found **two contract
blockers** — per-channel scores were computed and discarded, so the Phase-8/9
feature contract was unimplementable from the handoff, and the scorer was wired
to score the *already-capped* five-channel pool, so it could never surface an
atom the cheap channels' cap dropped — plus two instrument defects (saturation
measured in the wrong units; every "uncapped" pool bounded by a guessed finite
cap with its `cap_skipped` discarded), and the fact that **the committed
artefact was a `--smoke` run** quoted by §3.1 as measured. All confirmed
defects are fixed, regression-tested (1,057 tests green) and the artefact
regenerated on the pinned embedder; §3.1's conclusions survive, and §7 is the
itemised record.

Parent: `GRAFT_PHASE7_BUILD.md` (§6 adopted 15 Aug 2026) · `GRAFT_RESEARCH_PLAN_v1.md` v1.2 §3.3, §2.4, §6.3–§6.4 · `GATE0_CONTRACT.md` items 2, 3, 9 · `CLAUDE.md` §6
Status: **live.** This file wins conflicts with `GRAFT_PHASE7_BUILD.md`, per the Phase-5/6 convention.

Labels as everywhere: **[EVIDENCE]** · **[HYPOTHESIS]** · **[ANALYSIS]**.

---

## 1. §6 adopted, with two rows qualified

Decisions 1–10 and 12 are adopted **as written in the Recommended column**, in
the Phase-3 shape (`PHASE3_DECISIONS.md` §4 item 2): the build took them as
written and says so, rather than leaving code resting on an unsigned table.

Two rows are not simply adopted, and the difference is recorded rather than
glossed:

* **Decision 11 (scope corpus) is not this document's to adopt.** It is Gate-0
  item 9's. **Item 9 is now decided** — scope c, 200 questions
  (`GATE0_CONTRACT.md` item 9, 15 Aug 2026) — but ingestion at that scope has not
  run, so the build ran on the pilot's 10 questions under G9's honesty stamp.
  Nothing in steps 0–5 depends on the scope. The recall *numbers* do — see §3.1,
  which is the measured form of that, and which will need re-running once
  scope-c ingestion exists.
* **Decision 9 (the scorer) was adopted before the signature and built after
  it** — the ordering `GATE0_CONTRACT.md`'s first line requires. Adopting the row
  first fixed the interface the other modules were written against (fusion
  accepts a sixth channel; its absence is a legal configuration), which is what
  let steps 0–5 be built, run and measured while Gate 0 was still open.
  `pins.SCORER["built"]` is the machine-readable form, asserted by a test, and
  `verify_handoff.py` prints it. **Built is not trained** — §5.

Decisions 2, 4 and the `pool_cap` inside 3 are **inherited frozen**
(`CLAUDE.md` §6, `PHASE6_DECISIONS.md`), not re-decided here.

---

## 2. Four departures from the build plan

### 2.1 Channels emit `node_id`, not `atom_id` (G7's wording)

G7 says every channel emits `(atom_id, score)`. They emit `(node_id, score)`, and
the mapping to atom space happens once, at assembly, through
`pool.node_atom_id`.

**Why.** Closure runs on the *graph*: turning hits into a closed pool means
asking which entities and intervals their edges reference, which is only
answerable in node space. Channels emitting atom ids would force the pool builder
to invert the mapping — a reverse lookup needing the snapshot — to do identical
work. `node_atom_id` is pure and total, so the two spaces carry the same
information and the protocol is honoured in substance: one shape,
`Mapping[str, float]`, for every channel.

### 2.2 The temporal filter runs before assembly, not on the assembled pool

P7.4's row says "applied to the fused pool". It is applied to the fused *node
scores*.

**Why, and it is a correctness point rather than a preference.** Removing a node
atom from an already-closed `AtomPool` strands every edge atom that references
it, breaking the invariant `AtomPool.validate()` exists to hold — a temporal drop
would produce a malformed pool. Filtering first and closing afterwards yields a
pool that is closed by construction, which is what G8 promises Stage D.

### 2.3 Obligation slots are replayed from a cache, not parsed per run

Fix F2's parser is the frozen extractor, so obligations are GPU work. The runner
replays a cached CSV and `--parse` fills what is missing (10 LLM calls for the
pilot's 10 questions, 0 parse failures, 15 Aug 2026).

**Why.** Re-parsing every run spends the GPU for a deterministic answer, and —
worse — lets two runs of the same script read different slots for the same
question if the model or its pins ever move. **There is deliberately no
rule-based fallback**: `core.obligations.parse` admits only `exact` and `learned`
precisely so real questions cannot silently get exact-mode behaviour, and a
keyword parser here would reintroduce the guessed anchor decision 7 forbids one
field over. A question with no slots runs BM25 + dense only and **every row says
so** — `obligation_slots_present` is in the artefact per question.

### 2.4 `saturation()` is new — it is not in the build plan at all

Added because the first run produced ten recalls of exactly 1.000. See §3.1.

### 2.5 The scorer types relations by edge type, not by `encoder_metadata()`'s triples

G6 says the scorer consumes the typed graph "through `features.encoder_metadata()`".
It consumes the **frozen `EDGE_TYPES` vocabulary** instead —
`RELATIONS = EDGE_TYPES + rev_* + incident`, 23 relations — rather than
`encoder_metadata()`'s 40-plus `(src, rel, dst)` triples.

*Why.* A triple types a relation **by its endpoints**, which is what an encoder
over the whole heterogeneous graph needs and is redundant here: a pool's edge
atom already carries its own `label`, and its endpoints' types are already in
their own feature rows. Using triples would triple the relation-embedding table
to parameterise a distinction the input already encodes. Both derive from the
same frozen schema, so a `SCHEMA_VERSION` amendment reaches the scorer either
way — which is the property the G6 sentence is protecting.

### 2.6 `torch_geometric` is not imported, though Stage B's encoders use it

The pool is ≤ `pool_cap` = 64 atoms, so message passing is a handful of
`index_add_` calls on plain `torch`. PyG buys typed attention over large
heterogeneous graphs; at 64 nodes it would be a dependency bought for nothing —
the boring-stack rule, and the same reasoning that kept an ANN index out of the
dense channel. Stage B's choice to use it (Phase-6 decision 1) was made for E2's
HGT at whole-graph scale and does not transfer.

---

## 3. What the build measured

### 3.1 **The pilot graph cannot exercise retrieval, and every recall of 1.000 says so**

**Measured, 15 Aug 2026, `artefacts/phase7_retrieval.json`.** *(Provenance
corrected by §7: the first artefact was a `--smoke` run — StubEmbedder, not the
pinned bge-small — and this section quoted it as measured without saying so.
The artefact was regenerated the same day on the real embedder; every claim
below was re-checked against the regenerated run and holds. A smoke run now
writes to `phase7_retrieval_smoke.json` so this cannot recur silently.)*

Tier-A recall is **1.000 on 9 of 10 questions** (the tenth, `2133c1b5_abs`, is
the abstention question and has empty gold, scored `None` rather than 1.0).
Per-conversation eligible candidates:

| Conversation | 08e0 | 2133 | 3a70 | 4169 | 618f | 6d55 | b01d | db46 | faba | gpt4 |
|---|---|---|---|---|---|---|---|---|---|---|
| **candidates** | 12 | 13 | 18 | 11 | 14 | 23 | 10 | 9 | 8 | 12 |
| **closed atoms** | 32 | 34 | 45 | 28 | 36 | 59 | 27 | 25 | 21 | 30 |

Against `pool_cap = 64`. **Every conversation fits entirely inside the cap**, so
the pool *is* the conversation, nothing is ranked, nothing is excluded, and a
channel that returned its input unchanged would score identically. The second
row is the audit's units correction (§7): the cap counts **closed atoms**, and
the conclusion had to be re-established in those units — it survives, 21–59
against 64, but a conversation of ~30 candidates could have crossed 64 closed
atoms while the old node-count comparison still read "unexercised".

**This is Phase 4's G9 arriving one stage later, and it is recorded the same
way.** There, greedy on exact `U` was globally optimal on 30/30 lattice instances,
so best-of-K was arithmetic rather than learning and the gate's rule had to move
before it narrowed a claim on an artefact. Here, a recall of 1.000 says only that
the candidate set is smaller than the cap.

**It is also the measured form of `DATASET_DECISION.md` §5's argument.** That
section reasons that evidence-only sessions make retrieval artificially easy and
render ceiling 3 uninformative; this is that reasoning with a number on it.
Distractor sessions (scope b′) are what make `candidates > cap` true, and until it
is true this instrument measures its own plumbing.

**What was done about it:** nothing to the retrieval code, and a flag.
`recall.saturation()` reports `exercised: false` per question, the runner refuses
to print a recall without it, and the artefact carries a top-level
`saturation_warning`. Loosening `pool_cap` to force the cap to bind would be
tuning a frozen value to make a number look like a measurement.

### 3.2 **The entity channel matched an anchor in 3 of 10 questions, and one miss is the matching rule**

Measured on the same run, with the frozen parser's own anchors.

| Outcome | n | Questions |
|---|---|---|
| anchor matched an entity, hits returned | 2 | `08e075c7` (Fitbit Charge 3), `b01defab` (The Nightingale) |
| anchor matched, **no** assertion-backed endpoints | 1 | `db467c8c` (parents) |
| **anchor absent under normalised-exact, present under a looser rule** | 1 | `618f13b2` |
| category-vs-instance, or genuinely absent | 6 | the rest |

**`618f13b2` is the one to read.** The parser's anchor is
`"new black Converse Chuck Taylor All Star sneakers"`; the graph holds the entity
`"black converse chuck taylor all star sneakers"`. They differ by the leading
modifier `new`. Decision 7's normalised-exact rule misses an entity that is
**demonstrably in the graph**.

**The other six are a different failure and are not fixable by loosening.** The
anchors are categories — `plants`, `camera lens`, `projects led or currently
leading by me` — while the graph holds instances (`basil plant`, `nikon z6`).
That is **the same category-vs-instance line that produced both residual D1
disagreements** in the Gate-0 item 8 annotation pass (`yoga apps`,
`customer data`, `chatcontext1.md` §3a). One failure mode, two stages, and it is
now measured on each.

**Deliberately not fixed.** Decision 7 was ratified hours before this
measurement, and changing a matching rule *because a result was unflattering* is
the §6b failure the project guards against everywhere else. It is recorded here
with its reproduction so the amendment, if taken, is taken knowingly. Two
candidate directions, neither adopted: substring/containment matching (fixes 1 of
7, risks a hub anchor matching everything), and anchoring on the mention stream
rather than the entity name (larger, and it is really a D1 question).

**Confounded on this graph, and the confound is §3.1.** With `candidates < cap`
every atom is in the pool anyway, so the entity channel's `unique` count is 0
everywhere and its *marginal* contribution cannot be read off this run. The
match-rate finding stands on its own; the recall consequence does not.

#### 3.2a The two candidate amendments, measured — **for a ruling, not applied**

Measured 15 Aug 2026 on the same 10 questions and the same graph, so the ruling
is taken against numbers rather than against the intuition in §3.2. **Nothing was
changed**; the shipped rule is still A.

| Rule | Questions matching an entity | Questions yielding hits | Total entities | Total hits | Max entities one anchor |
|---|---|---|---|---|---|
| **A — exact** (shipped, decision 7) | 3/10 | **2/10** | 3 | 4 | 1 |
| **B — containment** (either direction) | 4/10 | **4/10** | 10 | 11 | 3 |
| **C — token-subset** (entity's tokens ⊆ anchor's) | 4/10 | **3/10** | 6 | 7 | 3 |

**Three things the measurement settles, and two it does not.**

Settled: *(i)* both B and C recover `618f13b2`, the case §3.2 identified as an
entity demonstrably present in the graph. *(ii)* **The hub blowup §3.2 warned
about did not appear** — the worst anchor under either rule matched 3 entities,
not the graph. That weakens this file's own stated objection to B, and is
recorded because it runs against the caution rather than for it. *(iii)* B also
rescues `db467c8c`, where A matched an entity that had **no assertion-backed
endpoints at all** — A scored a match and returned nothing.

Not settled, and both cut against acting: *(i)* **the hub risk is a scale
property and this graph has no hubs** — 6–16 entities per conversation, so "max
3" is close to no evidence about the corpus where a user entity links to
everything. G10's fan-out cap exists because that regime is expected, not
hypothetical. *(ii)* **Six of ten questions get nothing from the entity channel
under *any* of the three rules** — the category-vs-instance misses (`plants`,
`camera lens`, `projects led or currently leading by me`, `Alex`,
`current apartment in Shinjuku`, `road trip to the coast`). The dominant failure
is untouched by the amendment being considered, which is the strongest argument
that this is a D1/anchor-granularity question rather than a matching-rule one.

**And the whole table is confounded by §3.1**: at `candidates < cap` every one of
those hits is already in the pool via BM25 or dense, so B changes *which channel
found an atom*, not what Stage D receives. The recall consequence of this choice
is unmeasurable until a scope where the cap binds.

**Recommendation, if a ruling is wanted now: do nothing yet.** Not because B
looks wrong — it measures better than A on every column here — but because the
one number that would justify it (marginal recall at a binding cap) cannot be
produced at this scope, and the failure mode it fixes is the minority one. The
cheap, honest sequence is to re-run this table at scope b′ and rule then, with
`unique` counts that mean something.

### 3.2b Phase 5's own audit says the anchor problem is upstream — settling §3.2a

The four Phase-5 audit worksheets were filled and returned on 15 August 2026
(`PHASE5_DECISIONS.md` §5a). One number bears directly on this section:
**`entity_anchor` exact-match 8/48 = 0.167.**

That is the same defect §3.2 measured from the other end. §3.2 found the entity
channel matching an anchor in 3 of 10 questions and asked whether decision 7's
normalised-exact rule was too strict; §5a says the **anchors are mostly wrong at
source**. The fault is in the obligation parser, not in the matcher.

**This effectively settles §3.2a's recommendation**: amending decision 7 would be
tuning the consumer of a bad signal. Exact string match is a harsh metric on a
free-text slot — *"painting classes"* against gold *"painting projects"* scores
0 while being nearly right — so 0.167 is a floor rather than the parser's true
quality, and the conclusion is directional rather than quantitative. But the
direction is clear and it points away from decision 7. **Leave the matching rule
alone; the work is fix F2's parser.**

### 3.2c **Tier-B gold was degenerate as originally defined — item 3.2 amended, 15 Aug 2026**

**Built 15 August 2026, measured immediately, found broken, and the contract
amended the same day.** The amendment is recorded in `GATE0_CONTRACT.md` item 3;
this section keeps the reproduction because the defect is the reason the
amendment exists.

**After the amendment**, on the pilot's 10 questions: Tier-B gold retains
**every** evidence atom (`evidence_dropped` 0 on all 9 with valid gold,
`under_constrained` false throughout) while still genuinely minimising the
scaffolding — 9 → 4, 9 → 3, 6 → 2, 5 → 2, 3 → 1. No question is voided by
`max_atoms`.

**What was wrong, kept below**, because the failure is what justifies the two
clauses and a later reader will otherwise reinstate the simpler definition.

Item 3.2: *eligible assertions whose spans lie in `has_answer` turns, closed
under the structural refs rule, **minimised by removing any atom whose deletion
keeps `H` true***. Implemented literally, over the pilot's 10 questions:

| | |
|---|---|
| Questions with a valid Tier-A superset | 9 of 10 (the tenth is the abstention question — empty gold) |
| Tier-B gold size | **1 atom, on every one of the 9** |
| Questions where minimisation **discarded evidence** | **5 of 9** (up to 3 atoms dropped, on `41698283`) |
| Tier-B recall | 1.000 everywhere |

**Why, and it is not an implementation defect.** `H` is **formal validity
only** — sufficiency, entailment and answerability are routed to `U`, the gate
and Stage B by design (`CLAUDE.md` §4.2, plan v1.2 §4.4). Measured directly on
the fixture: the empty set fails only on `size`, and **every single node atom
passes `H` on its own** — including a bare `Entity` or `TimeInterval` node, which
asserts nothing. Stage-C pools also contain no `binding` atoms, so `CHECK_BINDING`
— the sub-check that would force a proof to fill an answer slot — is vacuous.
Nothing in `H` can express *"this question needs both claims"*, so minimisation
against `H` alone strips every proof to one atom.

**`GATE0_CONTRACT.md` item 10 already knew the premise and drew the conclusion
one field over.** Its "not the primary metric" note says: *not valid-terminal
rate — `H` is formal validity only, so a method saturates it with legal-but-weak
sets.* Item 3.2 then used `H` alone as the **minimisation criterion**, which is
the same saturability applied to the gold set instead of the metric. **The
contract is signed**, so this is an amendment to a signed document, not a draft
fix — which is exactly why it is reported here rather than quietly repaired.

**What the build did do, and why it is inside the spec.** The spec fixes the
*operation*, not the **order**, and the order decides which irreducible set you
land on. Two ordering choices, both recorded:

1. **Structural atoms are removed to fixpoint before any evidence atom is
   tried.** Without this the survivor on the fixture was `TimeInterval:n_ti` — a
   gold "proof" asserting nothing. Sorting alone was **not** enough, and that had
   to be measured rather than reasoned: a structural atom can be temporarily
   *blocked* by an edge referencing it, and by the time the edge is gone a single
   interleaved pass has already removed the evidence.
2. **`degenerate` and `under_constrained` are reported per question.** The first
   fires when nothing assertion-backed survives; the second when minimisation
   dropped evidence, i.e. when Tier-B gold is narrower than the question's actual
   evidence need. Both are in the artefact so no Tier-B number can be read clean.

**What the build did *not* do: invent a sufficiency criterion.** Adding one to
the minimisation would be writing a definition the contract does not contain, and
train-time `sufficiency` is defined *against gold* (item 4), so using it to build
gold is circular.

**A second, independent Tier-B problem, found while closing a test's coverage
gap.** `H`'s `size` sub-check rejects any set larger than `max_atoms` = 16. The
Tier-A closed set is *not* capped at 16 — it is capped at `pool_cap` = 64 by
construction, and deliberately so, since gold must be able to exceed what a pool
can hold. **So any question whose Tier-A closed set exceeds 16 atoms fails `H` on
the superset and has no Tier-B gold at all.** The pilot's largest was 9, so it
never fired; scope c's richer evidence sessions will reach it. The failure is
*reported* (`status: tier_a_superset_invalid`) rather than silent, which is the
only reason it would not read as a retrieval failure — but it means Tier-B
coverage will thin out exactly on the questions with the most evidence, which is
the opposite of the intended selection.

**The amendment adopted (item 3.2, 15 Aug 2026), and one rejected first.**

*Adopted, two clauses:* removal is **restricted to structural atoms** —
assertion-backed atoms are never removed — and **`max_atoms` does not bind a
gold set**, since it bounds what a *candidate* set may select and gold is not a
candidate set. Both are still "remove any atom whose deletion keeps `H` true",
over a stated subset and without a size bound that was never meant for gold.

*Rejected, and worth recording because it is the obvious fix:* minimise subject
to `H` **and** non-decreasing `U`-sufficiency against the Tier-A set. This is
**provably vacuous**. `sufficiency(X, gold) = |X ∩ gold| / |gold|`
(`graft/core/utility.py`), so measured against the Tier-A set *any* removal drops
it from 1.0 to below — nothing is removable and **Tier B becomes Tier A
exactly**. Two names for one number is the failure decision 8's cost column
names. (This file recommended it before checking `sufficiency`'s definition; the
check is what retired it.)

### 3.3 The temporal filter ran three times and dropped nothing

Three of ten questions carried a resolvable `time_constraint`; the filter applied
on all three and dropped 0 atoms — because **no `valid_during` edges exist on the
pilot graph**, exactly as G4 predicted (D4's conversational application is
Gate-1-time work, `PHASE6_DECISIONS.md` §2.2). This is the fail-open path doing
its job: the drop path is unexercised on real data and is covered by a fixture
that has an interval, so the semantics are tested in both directions rather than
only the one the corpus happens to reach.

The parser reported `time_expression_rate` 0.30 and `time_unresolved_rate` 0.00 on
these ten — better than Phase 5's 69%-unresolved on its stratified 49, and on ten
questions that difference is not a finding.

### 3.4 The fan-out cap never bound

0 binds across all 10 questions. Expected on a graph whose largest conversation
has 23 eligible nodes; the counter exists for the corpus where a hub entity makes
2-hop expansion the whole graph, and it is reported so that a cap doing real work
is visible rather than inferred (G10).

---

## 4. Two defects the build found that reading did not

### 4.1 `build_pool` admitted quarantined assertions when handed them directly

**A blocker for exit criterion 2, found by its own negative test.** Every channel
filters through `eligible_nodes`, so on the built-in paths nothing quarantined
could reach assembly — and `build_pool` therefore trusted whoever scored the
nodes. Passing a quarantined node id in directly admitted it to the pool.

G8 puts fix F9's boundary **at assembly**, not in the callers, and for the reason
this defect demonstrates: a guard that lives only in the callers is one new caller
away from being absent. Fixed — `build_pool` intersects its input with
`eligible_nodes(snapshot)` and reports `hits_refused_ineligible` **separately**
from `cap_skipped`, because two different reasons an atom did not make it,
flattened into one count, inflate whichever rate is being judged
(`PHASE5_DECISIONS.md` §1's quarantine-causes lesson).

### 4.2 `support_atoms` counted score-zero atoms, and min–max produces those

The pool report counted support atoms as `score == 0.0`. But G5's min–max maps a
channel's **lowest-scoring** retrieved atom to exactly 0.0 — so a genuine hit was
counted as support, and most often on pools where one channel dominates, which is
the case the G7 table is read to detect. Fixed to membership (`atom.target in
hit_score`) rather than value.

A third, milder one, fixed while writing the instrument rather than after: the
fusion report's `wins` counts strict argmax and ties go to the alphabetically
first channel, so a channel reaching the same atoms at the same normalised score
reads `wins: 0`. The entity channel is *permanently* in that position, since it
scores every hit 1.0 by design. `unique` — atoms no other channel returned — is
reported beside it, and is the count that answers "what would we lose by dropping
this channel".

---

## 5. What is not done

| Item | State |
|---|---|
| **P7.8, the scorer — built, NOT trained** | The one unmet "done when". Step 6's three conditions: **≤ 8M verified by `parameter_count`** ✅ (asserted, and `build_scorer` *refuses* an over-cap configuration); **fusion accepts its scores as a sixth channel** ✅ (tested, and wired through the runner's `--scorer`); **trains one epoch on the distant signal within 8 GB** ❌ — not run, because training is GPU/CPU work that is currently deferred. `train_scorer` exists and carries all three P6.11 guards; nothing has called it on real data |
| **The scorer's weights** | there are none. `--scorer` loads a checkpoint or the stack runs on five channels and says so. **There is deliberately no "build one on the fly" path**: an untrained scorer is noise, and adding noise as a sixth channel under `max` fusion raises some atom's fused score for no reason — a silently worse pool that every report would call a six-channel run |
| **Tier-B recall** | **built 15 Aug 2026, and item 3.2 amended the same day** (§3.2c) after the original definition proved degenerate. Now quotable: evidence is never dropped and `max_atoms` no longer voids evidence-rich questions. Tier A stays reported beside it as the conservative over-estimate |
| **Exit criterion 14, latency** | measured and in the artefact, but on a graph where every pool is the whole conversation — so the p50/p95 are plumbing numbers like everything else here |
| **Decision 7's match rate** | measured (§3.2), **and both candidate amendments measured beside it (§3.2a)**; not amended. B beats A on every column at this scope, but the number that would justify it — marginal recall at a binding cap — is unmeasurable until scope b′, and the dominant failure (category-vs-instance, 6/10) is untouched by either candidate. Re-run §3.2a's table at b′ and rule then |
| **The `unique` column's meaning** | uninterpretable on this graph (§3.2), because saturation makes every channel's marginal contribution zero. It becomes readable at scope b′ |

---

## 5a. The scorer, as built

**Shape.** `PoolScorer` — relation-typed message passing over the *pool*, two
layers, hidden 128 from the shared `pins.TRAINING`, **237,443 trainable
parameters** against decision 9's 8M cap — 3% of it. `build_scorer` raises rather than
returning an over-cap configuration: 8M is the GFM-RAG scale point and the claim
of comparable scale rests on it.

**[EVIDENCE] and what is not.** The scale point and the one-pass constraint are
GFM-RAG's (NeurIPS 2025: an 8M query-conditioned GNN, Recall@5 87.1/58.2/95.6 on
HotpotQA/MuSiQue/2Wiki in one 0.107 s pass against 3.162 s for iterative
IRCoT+HippoRAG); the case for a GNN over an LLM retriever is GNN-RAG's
(Findings ACL 2025, 8.9–15.5 F1 at 9× fewer KG tokens — qualified venue). **The
architecture here is [ANALYSIS]**: a faithful small implementation of
"query-conditioned, one pass", not a reimplementation of GFM-RAG's model. No
published number transfers to it and none is claimed.

**Query conditioning is a gate, not a concatenation.** The question vector gates
every message and is concatenated at the head. With concatenation alone the
propagation would be identical for every question and only the readout would
move — which is precisely what "query-conditioned" is not. A test asserts two
questions over one pool give different scores.

**Aggregation is mean, not sum.** A hub atom would otherwise receive a message
whose magnitude scales with its degree — which G10's fan-out cap bounds but does
not equalise.

**Loss is per-atom binary cross-entropy with a per-example `pos_weight`**, not a
softmax over the pool. The distant signal is not one-of-*n*: a question's
evidence sessions can make many atoms relevant at once, and a softmax would force
them to compete for one unit of mass they do not share. The weight is there
because the signal is sparse — a handful of relevant atoms in a pool of up to 64 —
and unweighted BCE reaches a low loss by predicting "irrelevant" everywhere,
which scores well and retrieves nothing.

**The gold boundary held.** Deriving the distant labels needs
`answer_session_ids`, which is a gold field, so `recall.distant_labels` and
`recall.evidence_turns` do it and the scorer receives plain numbers. The
structural test that quarantines gold to `recall.py` covers `scorer.py` and
passes — moving the derivation into the scorer would break it, correctly.

**`evidence_turns` vs `has_answer_turns` is item 2's distinction, now in code**:
session-level for *training*, turn-level for *evaluation*. `evidence_turns`
carries the corrected justification (`DATASET_DECISION.md` §7.4): the corpus does
ship `has_answer` per turn, so no annotation is needed — the real reason not to
train on it is that it defines the Tier-A evaluation target, so training on it
would collapse recall into a self-fulfilling number.

**All three P6.11 guards inherited, not reinvented**: the seed reaches
initialisation (`build_scorer`, the `build_arm` pattern), early stopping
**restores** the argmin-dev state, and a loop with no scorable dev example
**refuses**. Each has its own test, and each is a defect Phase 6 actually
shipped.

---

## 6. What Phase 8 and 9 get, unchanged

* **`AtomPool`s with closure and the cap applied** — Stage D's environment, in the
  exact shape the Phase-1 masks and checker consume, with the edge-atom refs
  correspondence now *enforced* rather than conventional (the check
  `CandidateAtom`'s docstring asked Phase 7 for by name).
* **Per-atom channel scores**, normalised, from `assemble()` — the real
  `AtomFeaturizer`'s retrieval features (architecture §9.1). *(True since §7's
  fix B1 and not before: when this bullet was first written, fusion computed the
  per-channel values and immediately collapsed them to one scalar, so the thing
  promised here did not exist in the handoff. `assemble()`'s report now carries
  `channel_scores` per atom, and the runner stores the fused `atom_scores`
  beside it.)*
* **Obligation slots per question**, recorded, with the parser's measured rates
  beside them (fix F2's rule).
* **The recall instrument and its gold callable** — ceiling 3 at every later gate,
  both tiers live since the signature, required-node/required-edge Recall@k for
  both (plan §3.3/§6.4), and `saturation()` — now in closed-atom units — to say
  whether the number meant anything.
* **The Stage-C fingerprint**, printed by `verify_handoff.py` beside the ingestion
  and Stage-B ones.

---

## 7. The post-build audit — 15 August 2026, same day as the build

Two sources, the Phase-5/6 convention: an external review of the finished
build, and an eight-agent verification pass that checked every one of its
claims against the code (executed probes, not readings) and swept the rest of
the package fresh. **All eight external claims confirmed** — several upgraded —
plus roughly twenty further defects found by the sweep; fifteen *suspected*
channel defects were refuted with executed probes (the five channels themselves
audited clean). Everything below is fixed and regression-tested unless marked
otherwise; the suite is 1,057 tests green, and `artefacts/phase7_retrieval.json`
was regenerated on the pinned embedder after the fixes.

### 7.1 The two contract blockers

**B1 — per-channel scores were computed and discarded.** Architecture §9.1's
Phase-8 features are "max/mean channel scores" and its Phase-9 `AtomFeaturizer`
row carries "retrieval channel scores" per atom — but `fuse()` collapsed every
channel's min–max value to one scalar, `build_pool` attached only that scalar,
and the runner received `atom_scores` and dropped it. §6's own bullet promised
what did not exist. *Fix:* `weighted_scores()` factored out as the single place
the fusion arithmetic is applied; `assemble()` reports `channel_scores` per
admitted atom (absence of a channel key = scored 0.0; structural and edge atoms
have empty maps); the runner stores `atom_scores` per question. Tested against
hand-computed values.

**B2 — the scorer scored the already-capped five-channel pool.** So the GNN
could never surface an atom the cheap channels' cap had dropped: under `max`
fusion it could only re-rank what was admitted — not a member of plan §3.3's
*union*, and the inverse of the cited evidence's ordering (GFM-RAG's GNN scores
**before** retrieval ranking, over the whole indexed graph). The integration
test did not catch it because it passed an **empty** scorer mapping into fusion
— a tautology. The wiring also leaked structural `Entity`/`TimeInterval` node
ids into the channel, inflating `hits_refused_ineligible` — the count that
exists to flag quarantine leakage. *Fix:* `scorer.channel_scores()` — the sixth
channel scores the **uncapped closed pool of the whole eligible scope**, one
forward pass as before, emitting assertion-backed node ids only; the runner
calls it; the cap is applied once, at assembly, after every channel has spoken.
Tested: a scorer hit the five channels never returned enters the final pool
under a binding cap.

### 7.2 The two instrument defects

**I1 — `saturation()` counted nodes against a cap defined over closed atoms.**
G8 applies `pool_cap` to the closed set; the flag compared eligible *node*
counts. A ~30-candidate conversation closing past 64 atoms would have read
"unexercised" exactly when the cap started binding. *Fix:* the flag now
computes the closed size of the full candidate scope (`closed_atoms_in_scope`)
and compares in the cap's own units; §3.1's table gained the closed-atom row
and its conclusion was re-established rather than assumed. A regression test
pins the fixture case (3 nodes → 9 atoms, cap 8: exercised, where the old
comparison said not).

**I2 — every "uncapped" pool was bounded by a guessed finite cap, and the
guess's failure signal was discarded.** The runner's pre-cap pool used
`64·(len(fused)+1)`, the gold builders `64·len(nodes)+64` — neither derived
from the closure it bounded, which grows with the *edges among* the chosen
nodes, O(n²) in the schema — and all three call sites threw away `cap_skipped`.
A silent truncation of the gold set flatters every method scored against it.
*Fix:* `pool.uncapped_pool()` computes the exact closed size first and asserts
`cap_skipped == 0` (guaranteed by monotonicity; the assert is the tripwire);
the pre-cap pool, both gold builders and `saturation()` all use it. One mapping
code path, as the module always claimed.

### 7.3 The artefact was a smoke run, quoted as measured

`phase7_retrieval.json` carried `"smoke": true` — StubEmbedder, not the pinned
bge-small — while §3.1 cited it as "Measured, 15 Aug 2026" with no mention of
either word, and `stage_c_frozen.embedder` in the same file named the real pin
it did not use. Both defaults wrote to the same path, so a smoke run silently
overwrote a real one. *Fix:* `--smoke` now defaults to its own
`phase7_retrieval_smoke.json`; the real artefact was regenerated on the pinned
embedder (all §3 claims re-checked and standing); §3.1 carries the provenance
note. Saturation, entity-match and temporal-filter findings were
embedder-independent; the dense-channel numbers were not, and are now real.

### 7.4 Metrics the plan requires that were missing

Plan §3.3 and §6.4 name **required-node Recall@k and required-edge Recall@k**
as separate Stage-C metrics; required-edge recall had no implementation at all
(atom ids are opaque hashes, so the kind split needs the pool object).
*Fix:* `recall.required_sets()` returns the gold sets split by kind, built
through `uncapped_pool`; `tier_b_gold`'s report carries the minimal set's
split; the runner reports required-node/-edge recall pre- and post-cap for
Tier A and post-cap for Tier B — which also closed a smaller gap, that Tier B
(the plan's §2.4 primary) reported no pre/post-cap pair at all.

### 7.5 Run identity, determinism, and the stale wording

* **Scorer runs were not reproducibly identified**: `--scorer` recorded only a
  boolean, and the layer count was a constructor literal outside the frozen
  config — two different checkpoints (or depths) shared one Stage-C identity.
  *Fix:* `pins.SCORER` gains `layers: 2` (constructor reads it); the artefact
  records `scorer_checkpoint = {path, sha256, parameters}` when active. The
  dead `channel_name` duplicate of `SCORER_CHANNEL` was removed at the same
  time. **The Stage-C fingerprint changed** — before any decisive run existed,
  which is the only cheap moment; the regenerated artefact carries the new one.
* **Exit criterion 3 ("identical artefacts") was untestable as written** — the
  artefact embeds wall clocks because criterion 14 requires them. *Fix:*
  `runtime.deterministic_view()` strips the declared `VOLATILE_KEYS`; the
  artefact records a `determinism.digest` over the rest; **two independent
  real-embedder runs produced equal digests** (verified 15 Aug 2026); the
  criterion is amended in the build plan and the exclusion list is itself
  asserted by test.
* **Stale wording, pinned by a test**: the honesty stamp and two artefact keys
  still said "Tier B refuses / refused until GATE0_CONTRACT.md signs" *while
  the same artefact carried nine Tier-B results*, `__init__.py` said the scorer
  "is not built", and `test_retrieve.py` **asserted the stale stamp text**, so
  fixing the falsehood would have failed the suite. All rewritten; the test now
  pins the corrected wording forward (`"refuses" not in stamp`), and R22
  retirements make the phrases unrevivable. The same R22 sweep closed the
  signature's own propagation debt — six live documents still called Gate 0
  unsigned or Gate 1 blocked on it, `GRAFT_PHASE3_BUILD.md`'s header still said
  only decision 1 was ruled against its own signed §6 heading, and
  `chatcontext1.md` contradicted itself about decision 28 in adjacent rows.

### 7.6 Smaller confirmed defects, all fixed

| Defect | Fix |
|---|---|
| The temporal filter — one of the five training-free channels — had no latency row (criterion 14) | `assemble()` times its three stages; the runner lifts `temporal` into the per-channel table |
| `read_slots` precedence inverted its own docstring: a stale `--parse` cache beat the authoritative Phase-5 audit worksheet | order flipped (cache first, worksheet last-wins); nothing fresh lost — `parse_missing` never re-parses covered questions |
| entity/expand emitted hits liveness-only: a **live** edge to a quarantined or cross-conversation node surfaced it at full score (the fixture's only such edge was dead, so the old negative test was vacuous — probed by the audit with the edge made live) | both channels filter through `eligible_nodes`; `build_pool` gains the same `conv_id` scope boundary at assembly (plan §8 risk #1); non-vacuous live-edge regression test added |
| `match_entities` ran twice per question — its cost landed in *both* the entity and expand latency rows | matched once; `entity_channel` accepts precomputed seeds; expand's row now times only the walk |
| `expand_channel` consumed a generator input twice (`seeds: 0` in the report while the walk ran) | seeds materialised once |
| Nearest-rank p95 off by one exactly when `0.95·n` is integral — fires at scope c's n = 200 | `ceil(0.95n)`, 1-based |
| An all-stopword corpus crashed `bm25s.index` (`max()` on an empty vocab, probed on 0.3.10) | narrow `except ValueError` → the channel goes quiet, matching the empty-corpus path |
| A snapshot with no corpus-joined turns wrote a zero-question artefact and exited 0 | the runner refuses |
| `channel_table` serialised the full missed-gold id list per channel per question | per-channel `missed_n` count; the full list survives in the tier rows |
| `train_scorer`'s restore-best guard was untested (deleting `load_state_dict` failed nothing); `has_answer_turns` — the function defining Tier-A gold — and the `support_atoms` membership fix had no tests; `assemble` was never tested with a temporal constraint; the fusion weights were only ever exercised at 1.0; the criterion-4 test self-disabled via `pytest.skip` | all six covered: restoration asserted by recomputed dev loss; the skip is now an assert |
| A relative `--out` crashed the final print after the artefact was written | `_rel()` falls back to the path as given |

### 7.7 Recorded, deliberately not fixed

* **min–max collapses graded within-channel scores at the fusion floor**: with
  any hop-1 hit present, every hop-2 expansion hit normalises to exactly 0.0
  (probed: `{a:1.0, b:0.5}` → `{a:1.0, b:0.0}`), and BM25's
  `drop_non_positive` puts its weakest *real* match at the floor likewise. Both
  are consequences of decision 3's declared arithmetic; changing them is a §6
  amendment, the G7 ablation table is the instrument that would justify one,
  and per-atom `channel_scores` now at least exposes the raw shape downstream.
* **The scorer's query-conditioning test is necessary, not sufficient** — a
  merely query-augmented model (head concatenation, no message gating) would
  pass it. The gating is architectural; recorded as a known test limitation.
* **`minimise()` is Θ(s²) `H`-checks** in the structural-atom count per
  question — offline gold construction, s ≤ ~50 at scope c, so ~1,000 checks
  per question and `ledger=None` keeps it off the metered budget. A scale note,
  not a defect.
* **`eligible_nodes` provenance-walks the whole snapshot per call**, several
  times per question. Milliseconds at pilot scale; linear in corpus size at
  scope c. Revisit only if the scope-c run's wall clock says so.
* **The runner's module docstring still carries the historical G9 numbers**
  ("10 questions", "ceiling 1 took 55%") as the declaration made before the
  first run; the artefact's stamp now self-describes instead of hard-coding
  either.

### 7.8 What the audit refuted

Fifteen suspected channel defects were probed and cleared, among them: the
recency tie-break keeps the newest edges and is total; hop counting is not off
by one; both text channels are deterministic, share one normalisation and one
candidate space, and cannot see gold fields; the BM25 pins match the library's
own defaults; an all-stopword *query* does not crash; `distant_labels` row
order provably matches `atom_features`; `tier_b_gold`'s `replace(cfg, ...)`
covers the only config read in `H`'s call graph. The channels as built were
sound; the defects were in the wiring, the instruments and the record.
