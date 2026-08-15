# Phase 7 — what the build decided

**Steps 0–5 built and green, 15 August 2026. Step 6 (the GNN scorer) is not
built: it trains, and `GATE0_CONTRACT.md` is unsigned.**

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
* **Decision 9 (the scorer) is adopted but not built.** Adopting the row fixes
  the interface the other modules are written against — fusion accepts a sixth
  channel and its absence is a legal configuration — while the model itself
  waits for the signature. `pins.SCORER["built"] is False` is the machine-readable
  form, asserted by a test, and `verify_handoff.py` prints it.

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

---

## 3. What the build measured

### 3.1 **The pilot graph cannot exercise retrieval, and every recall of 1.000 says so**

**Measured, 15 Aug 2026, `artefacts/phase7_retrieval.json`.** Tier-A recall is
**1.000 on 9 of 10 questions** (the tenth, `2133c1b5_abs`, is the abstention
question and has empty gold, scored `None` rather than 1.0). Per-conversation
eligible candidates:

| Conversation | 08e0 | 2133 | 3a70 | 4169 | 618f | 6d55 | b01d | db46 | faba | gpt4 |
|---|---|---|---|---|---|---|---|---|---|---|
| **candidates** | 12 | 13 | 18 | 11 | 14 | 23 | 10 | 9 | 8 | 12 |

Against `pool_cap = 64`. **Every conversation fits entirely inside the cap**, so
the pool *is* the conversation, nothing is ranked, nothing is excluded, and a
channel that returned its input unchanged would score identically.

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
| **P7.8, the GNN scorer** | not built. It trains; Gate 0 is unsigned. Interface frozen in `pins.SCORER`, absence is a legal configuration, `scorer.py` does not exist and a test asserts it |
| **Tier-B recall** | `tier_b_gold()` refuses by name. One callable swap once the contract signs |
| **Exit criterion 14, latency** | measured and in the artefact, but on a graph where every pool is the whole conversation — so the p50/p95 are plumbing numbers like everything else here |
| **Decision 7's match rate** | measured (§3.2), **and both candidate amendments measured beside it (§3.2a)**; not amended. B beats A on every column at this scope, but the number that would justify it — marginal recall at a binding cap — is unmeasurable until scope b′, and the dominant failure (category-vs-instance, 6/10) is untouched by either candidate. Re-run §3.2a's table at b′ and rule then |
| **The `unique` column's meaning** | uninterpretable on this graph (§3.2), because saturation makes every channel's marginal contribution zero. It becomes readable at scope b′ |

---

## 6. What Phase 8 and 9 get, unchanged

* **`AtomPool`s with closure and the cap applied** — Stage D's environment, in the
  exact shape the Phase-1 masks and checker consume, with the edge-atom refs
  correspondence now *enforced* rather than conventional (the check
  `CandidateAtom`'s docstring asked Phase 7 for by name).
* **Per-atom channel scores**, normalised, from `assemble()` — the real
  `AtomFeaturizer`'s retrieval features (architecture §9.1).
* **Obligation slots per question**, recorded, with the parser's measured rates
  beside them (fix F2's rule).
* **The recall instrument and its gold callable** — ceiling 3 at every later gate,
  Tier B ready the day the contract signs, and `saturation()` to say whether the
  number meant anything.
* **The Stage-C fingerprint**, printed by `verify_handoff.py` beside the ingestion
  and Stage-B ones.
