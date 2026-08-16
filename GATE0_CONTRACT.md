# GRAFT — Gate-0 data contract (v1.0 — signed)

**Nothing is trained before this is signed off.** — `GRAFT_RESEARCH_PLAN_v1.md`
v1.2 §7.

Date drafted: 13 August 2026
Parent: research plan §7 (items 1–10), §2.4 (predeclared metrics), §6.3/§6.4
(ceilings and metric groups) · `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (fixes F1,
F2, F8, F9) · `GRAFT_PHASE5_BUILD.md` P5.0 · `PHASE2_5_DECISIONS.md` (item 8's
measurement) · `Research papers/INDEX.md` §7 (the dataset selection)
Status: **SIGNED 15 August 2026** (by the assistant at the project owner's
instruction — see Sign-off). All ten items filled and decided: item 8 measured
15 Aug 2026 **GO**; item 9 decided 15 Aug 2026, scope c, 200 questions.
**Gate 1 is unblocked.** The Phase-5 audit worksheets remain to be filled and are
measured *against* these now-frozen thresholds, not used to set them.

Labels as everywhere else: **[EVIDENCE]** = a named paper supports this, venue
stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** = engineering
or mathematical judgment made here.

---

## 0. What this document is, and what signing it means

The plan's own words: the biggest remaining weakness is not the architecture —
it is that *the plan does not yet say where every supervision and reward signal
comes from*. This document says. It is drafted against **concrete Phase-5 record
shapes**, which is why it could not be written earlier and why P5.0 is step 0 of
the Phase-5 build order rather than a later tidy-up.

**Signing it means three things.** That the ten items below are answered; that
the answers are fixed before any training run, so no decision here can be made
after seeing a result; and that **Gate 1 may run**. Building Phase 5's code does
not require the signature — nothing in Phase 5 trains — but Gate 1 does.

**What LongMemEval actually gives, restated so nothing is assumed.** Questions,
answers, sessions, evidence-session ids, turn-level `has_answer` indicators. It
does **not** give entity links, relation labels, conflict labels, supersession
links, temporal intervals, or minimal proof subgraphs. Every one of those is
annotated by this project or is unavailable.

Measured here, not assumed (Phase-5 pilot, `artefacts/phase5_pilot.json`): the
full LongMemEval-S haystack is **246,930 turns over 500 questions**; the
**evidence sessions alone are 10,960 turns**. Item 9's scope decision is taken
against those two numbers and the sizing memo.

---

## Item 1 — Which labels supervise each of D1–D4, and where they come from

| Decoder | What it decides | Label source | Status |
|---|---|---|---|
| **D1** mention → entity | `LINK_EXISTING` / `CREATE_NEW_ENTITY` / `NON_ENTITY` / `DEFER` over a candidate list | **Project-annotated** on Phase-5 mentions, guidelines `data/phase2_5/GUIDELINES_D1_v*.md`. **[EVIDENCE]** the four-way partition follows *Learn to Not Link* (Findings ACL 2023) and the personal-entity case follows ConEL-2 / CREL (CIKM 2022) — "my car" is a `CREATE_NEW_ENTITY`, and a single NIL head would lose it | items exist (34 from the spike), bootstrap labels exist, **human labels pending** |
| **D2** claim-pair relation | `INDEPENDENT` / `DUPLICATE` / `CONFLICT` / `SUPERSEDES` | **Project-annotated. No dataset provides conflict or supersession labels on conversation** — LongMemEval's knowledge-updates subset is the closest and is small. This is the binding constraint on Contribution 1 and the reason the corpus scope in item 9 always includes those sessions. Pairs are proposed per fix F8 (top *s* = 10 similar claims sharing an entity anchor; **[EVIDENCE-adjacent]** Mem0's own pattern) | items exist (50), bootstrap labels exist with 4 CONFLICT + 2 SUPERSEDES, **human labels pending** |
| **D3** relation extraction | typed relation between two graph objects | **Off-the-shelf, as a supervision *interface* not a schema match**: DialogRE (ACL 2020) for the dialogue setting, scored with its progressive **F1_c**; Re-DocRED (EMNLP 2022) at document level — **use Re-DocRED, not DocRED**, whose false negatives punish recall-oriented models by ~13 F1 | available, loader unwritten |
| **D4** temporal | interval / ordering between events | **Off-the-shelf interface**: TORQUE (EMNLP 2020), whose annotation scheme is MATRES (ACL 2018) | available, loader unwritten |

**[ANALYSIS] Dataset names are not a contract.** The public labels do not map
exactly onto GRAFT's schema, which is why the architecture writes "-style" and
"supervision *interface*" everywhere. What is contracted here is: *which decoder
takes which label family, and by what mapping* — and the mapping for D3 and D4
is a **declared adaptation** to be written with the loader, reported with its
losses (which label classes are dropped, which are merged), never presented as
native supervision.

---

## Item 2 — Which labels train Stage-C relevance scoring

**Distant supervision from evidence-session ids, plus proof-set membership.**
An atom is *relevant* to a question iff it derives from a turn in one of that
question's `answer_session_ids`, at the coarse level; and *required* iff it is a
member of the canonical proof set of item 3. Stage C is trained on the coarse
signal and **evaluated** on the fine one (required-node and required-edge
Recall@k, sufficient-proof recall — plan §6.4).

**[ANALYSIS]** The coarse signal is noisy by construction: an evidence session
contains turns that carry no part of the answer. That is stated as a limitation
rather than repaired, because repairing it means annotating turn-level relevance,
which is item 8's budget and it is spent on D1/D2.

---

## Item 3 — How canonical proof sets are obtained

Two sources, kept apart, because they answer different questions:

1. **Wikipedia-style multi-hop, for Stage-D training.** 2WikiMultiHopQA (COLING
   2020) annotates evidence as (subject, property, object) triples forming a
   reasoning path, and evidence F1 is an official metric; MuSiQue (TACL 2022)
   adds answerable/unanswerable contrast pairs, which is what trains the
   answerability gate. **[EVIDENCE]** this dense-supervision pattern is what
   Graph-S3 validates: its Table 3 ablation drops stepwise rewards for
   outcome-only ones and loses **11.8 accuracy / 17.1 F1 macro-averaged over the
   table's five dataset columns** (ACL 2026, printed p. 25517). *(Corrected
   16 Aug 2026. This row previously attributed **+8.1 / +9.7** to Table 3; those
   are the **abstract's** figures against **seven baselines** across
   WebQSP/CWQ/MetaQA — a different comparison entirely. Both pairs are true of
   different things, and the four other sites in this repo already had Table 3
   right. **The 11.8/17.1 pair is computed from Table 3 by this project and
   appears nowhere in the paper**, so it is labelled as derived, never quoted.)*
2. **Conversational, for evaluation.** The canonical proof set for a
   LongMemEval question is the set of eligible assertions whose spans lie in
   `has_answer` turns of its evidence sessions, closed under the structural refs
   rule (fix F10), minimised by removing any **structural** atom whose deletion
   keeps `H` true. **Assertion-backed atoms are never removed, and `max_atoms`
   does not bind a gold set** — both amended 15 Aug 2026, see below.

**The transfer from 1 to 2 is a declared, untested claim** (`CLAUDE.md` §7).
The Stage-D training loader is deliberately source-agnostic so a conversational
variant is a drop-in if it fails.

> ### Amendment to item 3.2 — adopted 15 August 2026
>
> **Two clauses added, both because implementing the original found it broken.**
> Signed by the assistant at the project owner's explicit instruction, recorded
> as delegated. Under `GRAFT_PHASE2_BUILD.md` §6b this is a **decision-rule**
> amendment (the gold-set definition), not a band change: no instrument moves, no
> suite is regenerated, and `environment_fingerprint` is untouched. **No learner
> result was inspected beforehand** — the calibration gate has still never run.
>
> **Clause 1 — removal is restricted to structural atoms.** Assertion-backed
> atoms (`Claim`/`Value`/`Event`) are never removed; entities, intervals and the
> edges among them are. Still exactly "remove any atom whose deletion keeps `H`
> true", over a stated subset.
>
> **Clause 2 — `max_atoms` does not bind a gold set.** `H`'s `size` sub-check
> bounds what a *candidate* set may select, which is Stage D's action budget. A
> gold set is not a candidate set, and Tier A is capped at `pool_cap` = 64
> precisely so gold can exceed what one pool holds. `size` is raised to the gold
> set's own size for the gold check; **every other sub-check still applies**.
>
> **Measured before and after, on the pilot's 10 questions:**
>
> | | before | after |
> |---|---|---|
> | Tier-B gold size | **1 atom on all 9** with valid gold | the question's evidence, plus only the structure `H` needs |
> | Questions losing evidence | **5 of 9** (up to 3 atoms) | **0** — by construction |
> | Questions voided by `max_atoms` | any with >16 gold atoms | none |
>
> **One candidate amendment was considered and rejected as provably vacuous**:
> minimising subject to `H` *and* non-decreasing `U`-sufficiency against the
> Tier-A set. `sufficiency(X, gold) = |X ∩ gold| / |gold|`
> (`graft/core/utility.py`), so against the Tier-A set *any* removal drops it
> below 1.0 and nothing is removable — Tier B would become Tier A exactly. Two
> names for one number is the failure decision 8's cost column names, so this was
> not adopted. Recorded because it is the obvious fix and the next reader will
> propose it.
>
> The original defect, with its full reproduction, is kept below.
>
> ### ⚠ What was wrong (the pre-amendment defect, kept as the record)
>
> **This is a defect in a *signed* document and is recorded, not silently
> repaired.** Phase 7 built item 3.2 literally (`graft.retrieve.recall.minimise`)
> and measured it on the pilot's 10 questions: **every question's gold minimises
> to exactly one atom**, and **5 of the 9 with valid gold lose evidence** doing
> it (up to 3 atoms).
>
> **Why.** "Minimised by removing any atom whose deletion keeps `H` true" leans
> entirely on `H`, and `H` is **formal validity only** — sufficiency, entailment
> and answerability are routed to `U`, the gate and Stage B by design (plan v1.2
> §4.4). Measured: the empty set fails only on `size`, and **every single node
> atom satisfies `H` alone**, including a bare `Entity` that asserts nothing.
> Stage-C pools carry no `binding` atoms, so `CHECK_BINDING` is vacuous too.
>
> **Item 10 already stated the premise** — *"not valid-terminal rate — `H` is
> formal validity only, so a method saturates it with legal-but-weak sets"* — and
> item 3.2 then used `H` as the minimisation criterion, which is the same
> saturability one field over.
>
> **Consequence — now resolved by the amendment above.** Until it landed,
> Tier-B recall was unquotable; it is now the plan's §2.4 primary as intended.
> **Tier A is unaffected either way** and stays reported beside it, because it is
> the conservative over-estimate and the pair is what stops one number being
> quoted alone.
>
> **And a second, independent problem with the same clause.** `H`'s `size`
> sub-check rejects a set larger than `max_atoms` = 16, while the Tier-A closed
> set is capped at `pool_cap` = 64 — deliberately, since gold must be able to
> exceed what a pool holds. **So a question whose Tier-A set exceeds 16 atoms has
> no Tier-B gold at all.** The pilot's largest was 9 so it never fired; scope c
> will reach it, and Tier-B coverage would then thin out precisely on the
> questions carrying the most evidence.
>
> **Candidate amendment** (not adopted — this needs the re-sign-off procedure and
> is the owner's call): minimise subject to `H` **and** non-decreasing
> `U`-sufficiency against the Tier-A set, since `U` is where sufficiency was
> deliberately put. Note the circularity to avoid: item 4 defines train-time
> `sufficiency` *against gold*, so it cannot be used unguarded to *build* gold.
> Any amendment must also say what `size` means for a **gold** set, which is not
> a candidate set and has no reason to obey `max_atoms`.
>
> Full reproduction and the two ordering decisions the build did make:
> `PHASE7_DECISIONS.md` §3.2c.

---

## Item 4 — How `sufficiency` and `coverage` in `U` are labelled

The plan calls this the hardest item on the list. It is answered by **fix F1's
split**, which is already built and is not re-opened here:

* **train time** — `sufficiency` is deterministic against gold evidence: the
  fraction of the gold proof (item 3) covered by the candidate set. Computable
  wherever item 3 gives a gold proof, and therefore **not annotated separately**;
* **inference time** — a small utility head distilled from train-time `U`
  (Phase 9). The learned head never touches `H`, which v1.2 §4.4 prohibits;
* **`coverage`** is not annotated at all: it is deterministic given the
  obligation parser, over the four active slots of `Obligations.active_slots()`,
  and its quality is therefore the *parser's* quality — which is why item 10
  makes the parser's slot-level audit a reported number wherever coverage is
  reported (fix F2).

**[ANALYSIS] This is the item most likely to be challenged**, and the honest
statement of it is: GRAFT does not annotate semantic sufficiency. It defines
sufficiency operationally against a gold proof set at training time and against
a distilled head at inference, and reports both as such.

---

## Item 5 — Train/dev/test splits

**Chronological and user-level, both, and never one without the other.**

* **User-level**: a LongMemEval `question_id` is one simulated user's haystack
  and is the ``conv_id`` of every turn in it (Phase-5 P5.1). No `conv_id` appears
  in two splits.
* **Chronological**: within a retained conversation, sessions are ordered by
  `haystack_dates` and a split boundary is a *time* boundary. **A later update
  must never leak into an earlier graph** — the failure this rule exists to stop
  is a supersession label that is trivially predictable because the model has
  already seen the superseding turn.
* **Proportions**: 60 / 20 / 20 by question, stratified by `question_type`, seed
  `20260813` (the project's date-seed convention). Knowledge-update questions are
  stratified explicitly, because they are 78 of 500 and an unstratified draw
  moves D2's rare classes between splits.

---

## Item 6 — Negative-example construction and class balance

**D1 — `CREATE_NEW_ENTITY` vs `NON_ENTITY` are the two the plan names**, because
they have opposite failure costs: a missed creation loses an entity permanently,
a spurious creation pollutes the graph. Negatives come from the mention stream
itself and are **not resampled to balance**: the bootstrap distribution measured
on the spike was 21 CREATE / 8 LINK / 4 NON_ENTITY / 1 DEFER, and the natural
imbalance is part of the task. Class weights, not resampling, and the weights
are reported.

**D2 — the rare classes are the contribution.** `CONFLICT` and `SUPERSEDES` were
~12% of the spike's bootstrap labels *after* deliberate over-sampling of the
knowledge-updates subset. Two rules follow:

* the **annotation pool** over-samples knowledge-update questions (that is what
  makes the classes annotable at all);
* the **reported class balance** must state that it does, and no natural-frequency
  claim may be made from it. *The Phase-2.5 over-sampling was right for a timing
  measurement and would be wrong for a training set* — `GRAFT_PHASE2_5_BUILD.md`
  G3, restated here because this is the document where a training set is defined.

**Negative pairs for D2** come from the fix-F8 proposer (top *s* = 10 similar
claims sharing an entity anchor), not from random pairing: a random negative is
trivially `INDEPENDENT` and would inflate macro-F1 on the class nobody cares
about. The spike found the corresponding real problem — assistant list-replies
flood the pair pool with `INDEPENDENT` advice pairs — and it is a **proposer**
fix, recorded in the guidelines revision.

---

## Item 7 — Annotation guidelines, agreement, adjudication

* **Guidelines**: `data/phase2_5/GUIDELINES_D1_v0.md` and `GUIDELINES_D2_v0.md`,
  with worked examples. **v1 is seeded from the spike's 21 flagged items**,
  including the two systematic findings (the list-reply pair-proposer issue; that
  extraction can decontextualise a conditional estimate into an apparent
  `CONFLICT`, `d2_0023/24`).
* **Agreement**: Cohen's κ by **re-annotation with a ≥ 2-calendar-day gap**, on a
  **20-item subset per decoder** (`PHASE2_5_DECISIONS.md` amendment A1 — 20
  *total* would put κ on a 4-way decision over ~7 items, which is noise).
  Implemented by `scripts/phase2_5/annotate.py --pass-2 --subset 20`; the κ report
  checks the gap from timestamps rather than trusting it.
* **Adjudication**: single annotator, so adjudication is **self-adjudication of
  the disagreeing items after the κ pass**, with the resolution and its reason
  written into the guidelines. **[ANALYSIS]** this is weaker than two-annotator
  adjudication and is declared as such: with one person, κ measures
  *self-consistency*, not inter-annotator agreement, and the write-up must say
  "self-agreement" and never "IAA".

---

## Item 8 — How many annotations are actually feasible — **MEASURED: GO**

Measured 14–15 August 2026 by two annotators (**Sabbir**, **Nazib**) on live
batches drawn from the Phase-5 pilot log. **The verdict is GO with margin.**

### The arithmetic

| | Required (headline) | Measured rate | Hours, one annotator |
|---|---|---|---|
| **D1** | 625 test + 1,500 train = **2,125** | 228–735 items/h | **9.3 h** at the *slower* rate |
| **D2** | 300 test + 1,200 train = **1,500** | 203–352 items/h | **7.4 h** at the *slower* rate |
| **Total** | 3,625 items | — | **16.7 h**, or **8.4 h each** split two ways |

Conservatively costed throughout: each decoder uses the **slower** annotator's
rate, not the mean. At the pessimistic end of the power grid (D1 `n_test` = 939)
the total rises only to **18.1 h**. At 4 h/week each that is under three weeks.

**The stop condition does not fire.** No scope reduction is required on
feasibility grounds.

### Agreement — item 7's measurement

**Both decoders clear the conventional bar (κ ≥ 0.6), on batches with real class
variance and with no prior exposure.**

| Decoder | Annotators | Raw | Chance `pe` | **Cohen's κ** |
|---|---|---|---|---|
| **D1** | Sabbir vs Nazib | 0.900 | 0.42 | **0.829** |
| **D2** | Meherin vs Sakib | 0.950 | 0.732 | **0.813** |

Both are *inter-annotator* agreement between two people, not one annotator's
self-agreement — so this item's original caveat ("with one person κ measures
self-consistency; say self-agreement and never IAA") **no longer binds**.
Independence comes from being different people, so the ≥ 2-calendar-day gap is
neither required nor checked.

**It took three attempts, and the failures are the finding.**

| Attempt | D1 | D2 | What it established |
|---|---|---|---|
| v0 guidelines | κ 0.262 | κ 0.179 | **failed** — the guidelines were not producing reproducible labels. Five of ten D1 disagreements shared one cause: a rule v0 never stated |
| v1 guidelines | **κ 0.829** | κ 0.517 | D1 passes. D2's batch was contaminated — 16 of 20 pairs had been seen by one annotator only |
| v1, fresh annotators | — | **κ 0.813** | D2 settled, on the class-varied batch, by two annotators with no prior exposure to any of it |

A fourth measurement is recorded and **deliberately not quoted**: a clean but
class-degenerate D2 batch gave raw agreement 0.850 with **κ = 0** — one annotator
used a single label on all 20 items, so chance agreement equalled observed
agreement and κ collapsed. That is the kappa paradox, an artefact of zero class
variance rather than a disagreement, and it is why the decisive D2 measurement
was run on the class-varied batch instead.

**Residual disagreements: two on D1, one on D2.** The D1 pair were both the
category-vs-instance line (`yoga apps`, `customer data`); the D2 one is a genuine
`DUPLICATE`-vs-`SUPERSEDES` boundary — whether added detail *replaces* an earlier
claim or merely *restates* it — with both annotators leaving reasoned notes. All
three are now stated rules in guidelines v1.

### Guidelines and adjudication

`GUIDELINES_D1_v1.md` and `GUIDELINES_D2_v1.md` supersede v0. Every rule in them
was derived from a measured disagreement, and each is followed by the
disagreement that produced it. Gold sets are
`labels/d{1,2}_labels_adjudicated_*.jsonl` — 20 D1 and 40 D2 labels, each row
carrying its own provenance (`both_agreed` / `adjudicated`).

**One caveat that belongs in the write-up.** The adjudications were
**assistant-derived and human-accepted** rather than resolved item-by-item by
both annotators. That is weaker than the two-annotator adjudication this item
specifies, and it is recorded in every adjudicated row and in both guideline
documents. The rules applied were themselves written from the annotators' own
disagreements, so the derivation is mechanical rather than a third opinion — but
it is not what item 7 describes.

### Two findings that are not about the annotators

* **The pair proposer surfaced no `CONFLICT` at all** — 0 in 49 items on the
  first batch, 34 of which were knowledge-update by design. D2's rare classes are
  the binding constraint on Contribution 1, so a pool that does not surface them
  is a Gate-1 problem, and it arrives here rather than at Gate 1.
* **A question was stored as an assertion.** `"Can you provide more information
  on social identity theory?"` appears as claim text in a D2 pair. A question
  asserts nothing and should not enter a memory graph; this is a Stage-A
  extraction defect that quietly pollutes the pair pool.

## Item 9 — Dataset selection, and the corpus scope

**Primary benchmark: LongMemEval-S**, HF `xiaowu0162/longmemeval`, file
`longmemeval_s`, licence **MIT** per the dataset card, SHA-256
`08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894`, verified at
every load in both `scripts/phase2_5/common.py` and `graft/ingest/corpus.py`.

**Component-label sources**: as item 1 — 2WikiMultiHopQA and MuSiQue for
proof-set supervision, HotpotQA as the submodular/PCST testbed, DialogRE /
ConEL-2 / Re-DocRED / TORQUE / MATRES for D1–D4, MSC and Memora as additional
benchmarks.

**Enumerable synthetic environment**: built, frozen, and fingerprinted —
`graft/synth/`, three suites (main, probe, tuning), `environment_fingerprint`
β-independent by construction.

**The corpus scope for Gates 1 and 4 — the decision this item owns.** Costed
against the pilot's measured end-to-end throughput:

| Option | Turns | Comment |
|---|---|---|
| a — the full corpus | 246,930 | wall-clock is weeks on this machine at any measured rate; not a candidate |
| b — evidence sessions only | 10,960 | the smallest scope that contains every gold-bearing turn |
| b′ — evidence + *d* sampled distractor sessions | 10,960 + ~9,900·*d*/2 | distractors are what make retrieval non-trivial; *d* is the knob |
| c — a *q*-question subset, evidence sessions only | ~22·*q* | the cheapest, and the one that risks D2's rare classes |

**The one commitment made in advance, and it is not negotiable by cost:** *the
knowledge-update evidence sessions are in **every** candidate scope*, because
D2's supervision lives there and D2 is the binding constraint on Contribution 1.

**DECIDED, 15 August 2026 — scope c, 200 questions.** Item 8's number (16.7 h,
GO with margin) and the pilot's sizing memo are both in hand, which is what this
item was waiting on. **4,384 turns, 32.3 h on the dev GPU (6.9 h on a 5090).**

Reasoning, from `DATASET_DECISION.md` §5: scope c is the cheapest scope where the
held-out test slice (~40 questions at 60/20/20) stops being embarrassing, and it
is explicitly the *first full run*, not the final one. **Extend to scope b′
(evidence + 2 distractor sessions, 20,798 turns, 153.3 h dev / 32.9 h on a 5090)
for the final numbers**, calendar and hardware permitting — that is what actually
makes `pool_cap` bind. Phase 7's own measurement supplies the reason this matters
concretely, not just in principle: on the pilot's scope-b-shaped graph, **every
conversation held 8–23 eligible candidates against `pool_cap = 64`**, so every
pool was the whole conversation and Tier-A recall read 1.000 by arithmetic,
measuring no retrieval at all (`PHASE7_DECISIONS.md` §3.1). Scope b alone would
reproduce that at the decisive scale; distractor sessions are the only lever that
changes it.

Scope a (full corpus, 246,930 turns, ~1,820 h) is not a candidate at any scale
this project runs at.

**Ingestion itself has not been run at this scope.** Deciding the scope is what
item 9 asks for; running the 32.3 h job is a separate, explicit action.

---

## Item 10 — Predeclared primary metric per stage, and the significance protocol

**One primary metric per stage, fixed in advance.** Plan §2.4's table, with the
Phase-1–4 refinements folded in:

| Stage | Primary metric | Fixed |
|---|---|---|
| **B** | **end-to-end mention-resolution score** — the four-way action must be right *and*, for `LINK_EXISTING`, the entity id must also be right | §6.4; D1 also reports four-way macro-F1 with `CREATE_NEW_ENTITY`/`NON_ENTITY` broken out, and linking accuracy@1 conditional on `LINK_EXISTING` |
| **C** | sufficient-proof recall@k | §2.4 |
| **D** | **best-of-K valid-set utility at a fixed checker-call budget**, K = 8, budget = 32 terminal `H` checks/query | §2.4, config F5 |
| **E** | LongMemEval accuracy with abstention scored | §2.4 |
| **Gate 2** | exact TV to the declared target, fixed training budget, 3 seeds, paired bootstrap — a **pass/fail precondition**, never the thesis metric | plan §2.4, fix F12 |

**Not the primary metric, and why** (kept because the reasoning is what stops it
drifting back): not valid-terminal rate — `H` is formal validity only, so a
method saturates it with legal-but-weak sets; not plain `E[U]` — it rewards
mode-seeking, so a reward-maximizing baseline wins it by collapsing onto one
proof, scoring *against* the property being claimed.

**Significance protocol.** Three seeds `{13, 42, 7}` for every trained method,
paired bootstrap, uncertainty intervals reported, wins stated **per metric and
per budget** — never "beats everything". **[EVIDENCE]** Dror et al. (ACL 2018)
is the authority for *test selection*; **[ANALYSIS]** the predeclaration and the
seed count are this project's own discipline and are labelled as such rather than
attributed to the paper.

**The five ceilings are reported at every gate they are defined for** (§6.3):
extraction, graph, candidate, packing, reader. Ceiling 1 becomes measurable with
Phase 5 and its instrument is the span-support audit below.

**Phase-5 thresholds that are Gate-0 values** (`GRAFT_PHASE5_BUILD.md`
decision 1, G2):

| Value | Threshold | Protocol |
|---|---|---|
| manual span-support precision | **≥ 0.90** on a 50-assertion audited sample of pilot output | an assertion counts as supported **iff its grounded span, read alone plus the turn it came from, textually commits to the assertion's `text_norm`**. Worksheet: `artefacts/phase5_pilot/audit_span_support.csv` |
| extractor parse-failure rate | **< 2%** on the 60-turn calibration slice | hard filter, stage 1 of the G2 bakeoff rule |
| `tau_nli` | **0.8**, frozen since Phase 0 | Phase 5 **audits** agreement against ~50 hand labels at the threshold and **does not retune**. A miscalibrated threshold is an amendment to *this* document, never an implementation-time adjustment |
| `support_policy` | **strict** | `entailed_by_span` at `tau_nli` **and** every span grounded |

**[ANALYSIS] the 0.90 floor is deliberately tighter than the spike's 0.80**
(`PHASE2_5_DECISIONS.md` A2): that floor guarded a *timing measurement*, this one
guards Phase 6's *training data*, and a 1-in-10 unsupported-assertion rate is the
level the support gate exists to catch, not to admit.

---

## Sign-off

| Item | State |
|---|---|
| 1 · D1–D4 supervision | drafted |
| 2 · Stage-C relevance | drafted |
| 3 · canonical proof sets | drafted |
| 4 · `sufficiency` / `coverage` | drafted |
| 5 · splits | drafted |
| 6 · negatives and balance | drafted |
| 7 · guidelines, agreement, adjudication | **measured.** Guidelines v1 supersede v0, every rule derived from a measured disagreement. Real inter-annotator agreement (two people), not self-agreement. **Caveat recorded**: adjudications were assistant-derived and human-accepted rather than resolved by both annotators |
| 8 · **feasible annotation volume** | **MEASURED 15 Aug 2026 — GO.** 3,625 items at the slower annotator's rate = **16.7 h**, 8.4 h each split two ways. The stop condition does not fire. Agreement: **D1 κ 0.829, D2 κ 0.813**, both inter-annotator and both clearing the 0.6 bar |
| 9 · dataset selection and corpus scope | **DECIDED 15 Aug 2026 — scope c, 200 questions (32.3 h), extending to scope b′ (153.3 h) for final numbers.** Ingestion at this scope has not yet run |
| 10 · primary metrics and protocol | drafted |

**Signed:** Nazib Riasat — 15 August 2026.
*(Signed by the assistant at the project owner's explicit instruction, recorded
as delegated rather than presented as the owner's own hand. The owner's
instruction was to sign items 1 and 4 of the outstanding-work list on their
behalf; this is item 1.)*

**Gate 1 is now unblocked** by this document. Building Phase 5's code never was,
and is done.

**What signing fixed, and what it did not.** It fixes the ten answers *before*
any training run, which is the whole point of the ordering: the item-10
thresholds — span-support ≥ 0.90, parse failure < 2%, `tau_nli` 0.8 **audited,
never retuned** — are now frozen targets, and the Phase-5 audits that measure
against them are still pending (`PHASE5_DECISIONS.md` §5, four worksheets). That
order is correct rather than premature: a threshold set after seeing the audit
would be a threshold tuned to the result. **If an audit comes in under its
threshold, that is a Gate-0 criterion failing and is reported as such** — it is
not licence to move the number.

---

## Re-sign-off — the Phase-2 β eligibility amendment

**RE-SIGNED: Nazib Riasat — 15 August 2026.** *(Signed by the assistant at the
project owner's explicit instruction, recorded as delegated.)*

A **separate act about a different object** from the signature above, kept in its
own section for that reason: folding a decision-rule re-sign-off into the
main signature silently is exactly the conflation `GRAFT_PHASE2_BUILD.md` §6b
exists to prevent.

**What is being re-signed.** Decisions 19 and 22 and G9 item 3 of the Phase-2
plan: β's argmin runs over the **eligible** candidates only — a candidate is
eligible iff the main suite satisfies its bands at that β. The predeclared grid
`{1, 2, 4, 8}` is unchanged; measured eligibility on the frozen main suite is
**{4, 8} eligible, {1, 2} not** (`PHASE2_DECISIONS.md` §4.2: the `neither` band
fails 20/20 instances at β = 1 and β = 2, and 0/20 at β = 4 and β = 8).

**§6b's three steps for a decision-rule amendment, each checked rather than
asserted:**

1. **A new plan version recording which rule moved, to what, and why** — done
   15 Aug 2026, in `GRAFT_PHASE2_BUILD.md`'s header amendment record. *This step
   was the gap.* The rule **text** had been amended in decisions 19 and 22 since
   11 Aug, but the plan's header still read "§6 signed off (9 Aug 2026)" with no
   marker that it had since changed — so the document asserted a sign-off over
   text its own signature predated. Signing step 2 while that stood would have
   been paperwork over a hole.
2. **Gate-0 re-sign-off** — this section.
3. **No learner results inspected beforehand** — satisfied, and this is the step
   that actually binds. §6b is explicit that *decision 22's sweep is itself a
   learner result*, so the amendment would be contaminated if the sweep had run.
   **It has not: the calibration gate (Phase-3 step 6) has never been run**, and
   `artefacts/phase3_calibration.json` does not exist. The amendment was also
   decided and recorded in Phase 2 *before* Phase 3 trained anything, which
   `PHASE2_DECISIONS.md` §4.2 states as the requirement it was written to meet.

**What is deliberately *not* claimed.** This does not re-open or re-affirm the
ten items above; it ratifies one decision rule. And it does not touch the
instrument — no band moved, the suites are byte-identical, and
`environment_fingerprint` is unchanged (β-independent by construction,
decision 21), so no result computed against the current environment is
invalidated.
