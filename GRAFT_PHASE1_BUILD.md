# GRAFT — Phase 1 Build Plan: the deterministic core (`graft/core/`)

**`H`, `U`, `R`, the action masks, the obligation parser, and the obligation deficit vector `d(s)`.**

Date: 8 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 1) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §4) · `GRAFT_PHASE0_BUILD.md` (complete)
Effort: ~2 weeks solo
Status: ready to code once §6 is signed off

Labels inherited: **[EVIDENCE]** (named paper) · **[HYPOTHESIS]** (project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Phase 1 is where the project's central mathematical objects stop being prose and
become executable. v1.2 §4.1 calls publishing `U` as an executable function a
**Gate-0 blocker**; this phase discharges it. Most of what follows is
**[ANALYSIS]** and is labelled honestly — the four places where published work
actually constrains a choice are marked.

Gaps found while making this phase concrete are numbered **G1–G9** and are
referenced from code comments as "Phase-1 gap G*n*", matching the Phase-0
convention.

---

## 0. Why Phase 1 is next, and what blocks on it

| Phase | Blocked on Phase 1 by |
|---|---|
| 2 (synthetic env) | `exact.py` enumerates **valid** terminals and computes `R` over them. Neither exists without `H` and `U`. The closed-subset DP is over sets the masks declare constructible. |
| 3 (7 learners) | `d(s)` is a policy input for every learner and the *entire* definition of L7 (Contribution 3). `log R` is the target of every balance loss. |
| 4 (5 search algos) | S3/S4 build sets directly and are `H`-filtered afterwards — that filter is this phase. `scorer` on the lattice is exact `U` (fix F13). |
| 9 (Stage D real) | the distilled utility head regresses train-time `U`; without `U` there is no regression target. |

Two of these are load-bearing in a way worth stating plainly. **Gate 2's
decision rule is defined on `d(s)`** — L7 is "L6 plus `Δd` as input features",
so the dimension and semantics of `d` are part of the experimental condition.
And **Gate 2's exact TV is computed against `R`**, so any change to `U` after
Phase 3 re-runs every learner.

---

## 1. Nine specification gaps Phase 1 must close

Each was found by trying to write the code. Each is a decision that must be made
**now**, because it is frozen at Gate 0 and consumed by Phases 2–4.

### G1 — The empty proof set is formally valid and scores `R = 7.39` [ANALYSIS]

Run the eight sub-checks of architecture §1.1 against `X = ∅`. Type legality:
vacuous. Duplicates: vacuous. Temporal: vacuous. Retired evidence: vacuous.
Scope: vacuous. Size: `0 ≤ max_atoms`. Eligibility: vacuous. Closure: vacuous.
So `H(∅) = 1`, and therefore `stop_allowed(root) = True`.

Now score it. `sufficiency = 0`, `source_quality = 0`, `redundancy = 0`,
`size = 0`; `coverage` and `temporal_correctness` depend on the question, and
`coverage` depends on the `d_source` definition fixed in G4 — the two sections
must be read together, and the numbers below assume G4's **corrected**
`d_source`.

| Question shape | `coverage` | `temporal` | `U(∅)` | `R(∅)` |
|---|---|---|---|---|
| anchor + value + source, **no** time constraint | 0 | 1.0 (unbounded ⇒ vacuously perfect) | 0.5 | **7.39** |
| the same **with** a bounded time constraint | 0 | 0 | 0 | 1.0 |
| no active obligation slots at all | 1.0 (nothing to cover) | 1.0 | 1.0 | **54.60** |

The worst formally valid non-empty set scores `R = 0.247`. **The empty proof set
outranks the bottom third of the valid utility range in the common case, and
still beats the worst valid sets even in the harshest one.**

Under G4's *original* `d_source` (0 when there are no bindings) the first row
scores `U = 0.667`, `R = 14.39` instead — the null set is credited with
satisfying `needs_source` by binding nothing. Fixing `d_source` removes that
extra credit and is what makes the 7.39 above correct; the two defects are the
same defect seen twice, and both are fixed here.

This is the same failure shape as the `exp(β·0) = 1` error in v1.2 §1.1: a null
object landing in the middle of the reward range instead of at the bottom,
because "nothing is wrong with it" was confused with "it is good". It also lets
a trajectory terminate at step zero, returning a proof that cites nothing — which
the serializer cannot render and the reader cannot use.

**Decision.** Check 6 becomes `1 ≤ |X| ≤ max_atoms`. The empty set is not a legal
terminal; `stop_allowed(root)` is always `False`. A query whose pool is empty has
no legal `ADD` and a masked `STOP`, so it reaches `FAIL` — which is exactly the
"no proof found within budget" semantics that maps to the abstain fallback.
Constructibility is untouched: every valid terminal has at least one atom by
definition.

**Cost to reverse.** Changes the target support; re-derives `p*` and every Gate-2
result.

### G2 — Nothing resolves an atom, either to its record or to the graph object it denotes [ANALYSIS]

**Two distinct lookups are missing, and conflating them is the trap.**

*First:* the architecture's signature is
`H(X: ProofSet, q: Obligations, G: GraphSnapshot)`. But `ProofSet.atoms` is
`frozenset[str]` — bare ids — and every sub-check needs the atom's `kind`,
`refs` and `label`. Nothing in the signature can supply them.

*Second, and worse:* even holding the `CandidateAtom`, there is no field naming
the **graph object it denotes**. `CandidateAtom` is
`{atom_id, kind, refs, feat, label}`; `refs` are other *atom* ids — that is the
closure mechanism, deliberately pool-internal — and `atom_id` is a content hash
computed by `atom_id()`, which is a different function from `node_id()`,
`edge_id()` and `assertion_id()`. So no field carries a `node_id`, `edge_id` or
`assertion_id`.

Five specified lookups therefore cannot be written at all: check 3 needs the
edge's `valid_during` and `t_created`, check 4 needs its `t_invalid`, check 5
needs its provenance chain to a `conv_id`, check 7 needs an `assertion_id` to
pass to `is_eligible()`, and `U.source_quality` needs to resolve `asserted_by` to
a `Source` node. All five are specified as per-atom lookups against `G`.

**The failure mode if this is left open** is not a crash. The Phase-2 lattice can
synthesise everything in-pool and never touch `G`, so checks 3/4/5/7 would pass
their unit tests, ship green through Gate 2 and Gate 3, and first meet a real
snapshot in Phase 9 — which is the most expensive possible place to discover that
four of nine sub-checks were never exercised against the thing they check.

**Decision — two Tier-A amendments, same window:**

* **`AtomPool`** in `graft/schemas.py`, and a fourth parameter to `H`. It is not
  just a dict: it validates that **every `ref` resolves inside the pool** — an
  atom referencing outside can never satisfy closure, so it can never be legally
  added, and the only honest place to notice that is Stage C's pool builder
  rather than Phase 9 — and that **`|pool| ≤ pool_cap`**, the state-space bound
  the generalization argument rests on. **[EVIDENCE]** Generalization and
  Distributed Learning of GFlowNets (ICLR 2025) gives data-dependent bounds that
  degrade with state-space size. It also owns the reverse-reference index
  (`ref → atoms that reference it`), which the `ADD` mask needs to answer "which
  atoms just became legal" in O(1) rather than by rescanning the pool.
* **`CandidateAtom.target: str = ""`** — the `node_id` / `edge_id` /
  `assertion_id` this atom denotes; empty for atoms that denote nothing in the
  graph. `refs` and `target` are two different reference systems —
  pool-internal ordering versus graph-external denotation — and keeping them
  separate is the point.

`atom_id()` gains `target` as a component of its content key. Without it, check
2's "no two selected atoms share a content hash" is computed on an incomplete
key, and two atoms denoting different edges could hash identically.

Phase 0's own handoff rule applies verbatim: *"If Phase 1 needs a sixth import,
Phase 0 is incomplete — fix it there, not by widening Phase 1."* This is that
case, and the fix is cheap today because no experiment has consumed Tier A yet.

`AtomPool` is not just a dict. It validates the two invariants that are silent
bugs everywhere else:

* **every `ref` resolves inside the pool.** An atom whose reference points
  outside can never satisfy the closure rule, so it can never be legally added —
  it is unreachable dead weight, and the only honest place to notice that is
  Stage C's pool builder, not Phase 9;
* **`|pool| ≤ pool_cap`**, the state-space bound the generalization argument
  rests on. **[EVIDENCE]** Generalization and Distributed Learning of GFlowNets
  (ICLR 2025) gives data-dependent bounds that degrade with state-space size.

It also owns the reverse-reference index (`ref → atoms that reference it`), which
the `ADD` mask needs to answer "which atoms just became legal" in O(1) rather
than by rescanning the pool.

**Consequence.** `graft/schemas.py`'s `SCHEMA_VERSION` goes `0.1.0 → 0.2.0`.
Free now; a migration after Phase 5 writes real logs. Note there are **two**
constants with that name — see P1.0 for which amendments move which.

### G3 — Bindings are derived, not chosen, and nothing said so [ANALYSIS]

`ProofSet` carries `bindings: {slot → atom_id}`, but the action space is
`ADD(atom)` and `STOP` only (v1.2 §3.4). There is no `BIND` action, so nothing
can populate that field — yet `H`'s temporal check is defined over bindings and
`U`'s `temporal_correctness` grades them.

**Decision.** A binding **is** an atom: `CandidateAtom.kind == "binding"`, whose
`label` names the slot and whose `refs` name the referents. `ProofSet.bindings`
is therefore *derived* from the selected set, never set independently, by
`AtomPool.derive_bindings(atoms)`.

Two consequences that have to be checked rather than assumed:

* **a new sub-check (9): no two selected binding atoms claim the same slot.** Two
  atoms both binding `answer` is an internally contradictory proof, and it is
  formally detectable, so it belongs in `H` rather than being penalised in `U`;
* the closure rule already forces a binding's referents to be selected first, so
  a binding can never dangle. That is the rule doing useful work, and it is worth
  a regression test.

**Cost to reverse.** Changes what a terminal *is*; Phases 2–4 re-run.

### G4 — `d(s)` is undefined, and Contribution 3 is defined on it [ANALYSIS]

Architecture §3.1 passes `d_s` to the policy and calls it "the checker obligation
vector (which obligations satisfied so far) — **a feature, never an energy**".
L7 is precisely "L6 plus `Δd = d(s) − d(s′)` as input features". So the dimension
and semantics of `d` are part of the Gate-2 experimental condition — and nothing
in any document defines them.

**Decision — `d(s) ∈ [0,1]^6`, frozen here:**

| # | Component | Definition on a **partial** set |
|---|---|---|
| 1 | `d_anchor` | 1 if the obligation names an entity anchor and no selected atom resolves to it, else 0 |
| 2 | `d_value` | 1 if a `value_type` is required and no selected atom supplies a `Value` of that type, else 0 |
| 3 | `d_time` | fraction of the required interval not covered by the selected evidence's intervals; 0 when no constraint |
| 4 | `d_source` | **1 when `needs_source` is active and no binding has a resolvable span**; otherwise the fraction of bindings with no resolvable span |
| 5 | `d_binding` | 1 if no binding atom is selected, else 0 |
| 6 | `d_closure` | fraction of selected atoms with at least one unselected `ref` |

**Why these and not others.** Components 1–4 are the four active obligation slots
Phase 0 fixed in `Obligations.active_slots()`; 5 is what makes an answer
extractable at all; 6 is 0 under the `ADD` masks by construction but non-zero for
sets built directly by S3/S4, which is exactly why closure is also a checker
sub-check (fix F10).

**One implementation, two consumers.** `coverage` in `U` is
`1 − mean(d₁..d₄ over active slots)`. Computing them separately would let the
reward and the policy feature drift apart silently; they share one function and a
test asserts the identity.

**Why `d_source` is defined the way it is, and not the obvious way.** The natural
phrasing — "fraction of current bindings with no resolvable span, 0 when there
are no bindings" — is the empty-set defect of G1 in miniature: a set that binds
nothing has zero source deficit, so `needs_source` scores as *fully covered* by
selecting no binding at all. `d_binding` catches the missing binding, but
`d_binding` is component 5 and `coverage` averages only components 1–4, so the
reward genuinely credits the null case. Making the unmet-and-unbound case a full
deficit closes it. This is the same shape as G1 and G5: a vacuous truth being
scored as an achievement.

**Two things deliberately dropped, recorded so they are not re-derived.**
`d_conflict` from the original pipeline's six-vector has no corresponding slot in
`Obligations` and no question type that requires one — conflict handling lives in
Stage B's D2 decoder. And **"closure" in this project now means the structural
refs rule (fix F10), never the old aggregate-query closure certificate**; the
`GRAFT2_9` documents use the word the other way, and the collision is the kind of
thing that silently reintroduces a cut idea.

**`d` must be evaluable on partial sets.** That is its entire purpose — a dense
signal at every step where `U` is only meaningful at terminals. It is a feature
and an auxiliary target, **never an energy**: FL-GFN's Assumption 4.1 requires a
scalar with terminal consistency, and a six-component vector is not that object
(v1.2 §1.2). **[EVIDENCE]** LED-GFN (ICLR 2024 Oral) is the method built for the
regime where intermediate energy is unavailable, which is why `d` enters as
conditioning rather than as `ℰ`.

**Cost to reverse.** Changing the dimension or any component re-runs Gate 2,
because L7 vs capacity-matched L6 is defined on it.

### G5 — The temporal check and the temporal utility would be the same predicate [ANALYSIS]

Architecture §1.1 check 3 rejects bindings that violate the time constraint.
§1.3's `temporal_correctness` grades "interval-arithmetic agreement of bindings
with constraint". If both are the same predicate, then every set surviving `H`
scores `temporal_correctness = 1.0` and **the term is constant across the entire
valid space** — a fifth of `U`'s positive weight doing nothing, and precisely the
degeneracy v1.2 §4.1 warns about ("`U` must genuinely discriminate, or the
objective is doing less than intended").

**Decision — split them on hard-versus-graded, which is also the `H`/`U`
boundary:**

* **`H` check 3 rejects contradictions only.** A bound claim whose validity
  interval is *disjoint* from the constraint is formally wrong. Also: a selected
  `supersedes` edge pointing at an edge with a later `t_created` is a supersession
  running backwards in transaction time, which is formally wrong and cheap to
  detect.
* **`U`'s `temporal_correctness` grades precision:** the fraction of the required
  interval actually covered by the selected evidence's intervals,
  `|constraint ∩ ⋃ evidence| / |constraint|`. Non-contradictory but vague evidence
  scores low; tight, complete evidence scores 1.0.

This keeps `H` a formal predicate and puts the graded judgment in `U`, which is
the routing rule of v1.2 §4.4 applied to a term that was about to violate it.

**Two denominators that do not exist, both settled here.** `Interval` permits
`start == end` (Phase 0 defines it as the empty interval), and permits either
side to be `None`. Neither has a `|constraint|` the ratio can divide by.

* **Empty constraint → rejected at construction.** A time constraint no instant
  can satisfy is not an answerable obligation, so `Obligations.__post_init__`
  refuses it. This is a targeted validation on the `time_constraint` field, not
  a change to `Interval`'s semantics, which stay as Phase 0 froze them.
* **Unbounded constraint (either side `None`) → `temporal_correctness = 1.0`,
  and the instance is counted in a `temporal_unbounded` diagnostic.** The measure
  is infinite, so no ratio is meaningful; scoring it as satisfied at least
  matches the `H` side, which also cannot find a contradiction against an
  unbounded window. The cost is that the term does no work on such questions —
  which is why the Phase-2 lattice must generate **bounded** constraints (§8), so
  that exit criterion 9 tests something, and why Phase 5 must report the rate at
  which the obligation parser emits unbounded constraints on real data. A term
  that quietly stops discriminating is exactly the failure this gap exists to
  prevent, so the rate is a reported number rather than an assumption.

### G6 — Check 5, "scope and access constraints", has no data model [ANALYSIS]

v1.2 §4.4 lists scope among the formally checkable obligations, so the paper will
claim it. Nothing in the Phase-0 schema supports it: `Obligations` has no scope
field, and `GraphSnapshot` exposes no way to reach a `Turn`'s `conv_id` from an
atom.

The risk is real rather than hypothetical. GRAFT is a *multi-conversation* memory
graph — `Turn` carries `conv_id` and `session_id` — so a question about
conversation A being answered from conversation B's evidence is a genuine failure
mode, and one a memory paper will be asked about directly.

**Decision — implement it, minimally, fail-closed:**

* `Obligations` gains `scope: tuple[str, ...] = ()`, where empty means
  unrestricted (Tier-A amendment, same window as G2);
* `GraphSnapshot` gains `span(span_id)` and `turn(turn_id)`, so an atom's
  provenance chain can be resolved to a `conv_id`;
* check 5 passes trivially when `scope` is empty, and otherwise requires every
  selected atom's provenance to resolve inside it. **An unresolvable provenance
  chain under a non-empty scope is a violation, not a pass** — the check fails
  closed, so a missing implementation can never be mistaken for a clean result.

**This is cheap on the lattice and not cheap in Phase 0.** On the lattice `scope`
is empty and the two new methods return `None` — one line each. In Phase 0 the
same amendment touches `GraphSnapshot`, `_PROTOCOL_MEMBERS`,
`DictGraphSnapshot`'s storage and constructor, two new `GRAPH_OPS`
(`turn.add`, `span.add`), `ReplayGraphStore._apply`, and their tests. Budget it
as half a day, not a line.

**Alternative priced and rejected:** stub check 5 as always-true and defer to
Phase 6. Cheaper by roughly a day, but it ships a hole behind a claim, and the
project's own error register (`CLAUDE.md` §5) is mostly this failure mode.

### G7 — `source_quality` has no source of scores [ANALYSIS]

§1.3 says "metadata lookup from `Source` node type". There is no table of source
types and no scores anywhere.

**Decision.** A `source_tiers: {tier_name: score}` map in **config**, not in code
— it is part of the reward, so it must be frozen at Gate 0 and identical across
all seven learners for the same reason `u_weights` must be (v1.2 §5.1).

`source_quality(X) = mean over selected atoms of tier(atom)`, where `tier` is
resolved through `asserted_by` to a `Source` node and falls back to a configured
`default_tier`. Defaults: `{first_party: 1.0, corroborated: 0.75, reported: 0.5,
unknown: 0.25}`, `default_tier = unknown`.

**Gotcha this creates, and the test that catches it.** If every atom in an
environment resolves to the same tier, the term is constant and contributes
nothing. Phase 2's lattice generator must emit mixed tiers, and Phase 1's exit
criterion 9 (§5) requires every `U` term to take at least two distinct values on
the fixture set — so a degenerate term fails the build rather than quietly
wasting a fifth of the reward.

### G8 — The `trace` return type is undefined [ANALYSIS]

`H(...) → (bool, trace)`. A bare bool is not enough: the exit criterion requires
positive **and negative** cases for every sub-check, Phase 4's `H`-filter needs to
report *why* candidates failed, and Phase 2's audits are defined over failure
categories.

**Decision.** `H` returns `CheckResult(ok: bool, violations: tuple[Violation, ...])`,
where `Violation` carries `(check: str, atoms: tuple[str, ...], message: str)`.
Sub-checks are named constants so that failure categories can be counted without
string matching.

**Both types live in `graft/schemas.py`, not in `core/checker.py`.** The tempting
argument — "a module's own result format, like `Event` in `eventlog.py`" —
does not survive contact with the consumers: Phase 4's `H`-filter reports why
candidates failed and Phase 2's audits are defined over failure categories, so
violations genuinely cross module boundaries in a way a log line's own record
does not. Adding two Tier-A dataclasses with `to_dict`/`from_dict` costs
nothing, keeps `test_structure.py`'s allow-list at three entries instead of
eroding a rule the project leans on, and makes the trace serialisable — which
Phase 4 wants anyway, since a filtered candidate's failure category is worth
logging. Consequence: both need generators, because
`test_every_schema_class_has_a_generator` requires one per schema class.

**Short-circuit or collect?** Collect all violations by default; offer
`first_failure_only=True` for the hot path. The incremental checker uses the
short-circuit form; audits use the full one.

### G9 — Nothing says which call increments `terminal_checks` [ANALYSIS]

Phase 0 froze the meter and its rule ("incremented **inside** the checker, never
by callers"), but not which of the two checker paths spends it.

**Decision.**

* the batch `H(X, ...)` on a completed candidate set spends **exactly one**
  `terminal_check`;
* the incremental path spends `incremental_ops`, one per `ADD`, and is never
  capped — construction-time validity is maintained incrementally and is free
  (Phase-0 gap G1);
* `H` takes `ledger: Ledger | None`. `None` means "do not meter", which Phase 2's
  exhaustive enumeration needs — enumerating every terminal of a 5,000-terminal
  lattice would exhaust any per-query budget, and it is an offline audit, not a
  query.

**Two wiring consequences that follow immediately.** `terminal_checks` is capped
and `Ledger.count` *raises* on a capped spend outside `query_scope()`, so:
the Phase-3 training harness must open one query scope per instance, not one per
epoch; and Phase 1's own agreement property test (exit criterion 3, 10⁴
batch-versus-incremental pairs) must pass `ledger=None`, or it dies at call 33
with `BudgetExceeded`. Both are the ledger working as designed, and both are the
kind of thing that costs an afternoon if it is discovered rather than written
down.

**The loophole, stated so it is watched.** `ledger=None` is a way for a search
module to cheat the budget. Mitigation: the Phase-3/4 harness always constructs
search modules with a ledger, and a test asserts that every `SearchModule` in the
registry calls `H` with one. Named here because the whole comparison rests on it.

---

## 2. Scope

**In.** `H` and its nine sub-checks (batch and incremental, proved to agree),
`U` and its six terms, `R` and `log R`, the `FAIL` terminal, `legal_adds` /
`stop_allowed` / dead-end detection, the obligation parser's exact mode and its
learned-mode interface, `d(s)`, `derive_bindings`, the Phase-0 amendments G2/G6/G7
require, and **a minimal coherent fixture pool generator** (P1.7).

The fixture generator is in scope because it has to be. Exit criteria 7, 9 and 10
need pools with resolvable refs, mixed source tiers, non-trivial `feat` and a
gold proof, and Phase 0's `generators.py` cannot supply them: it emits atoms
whose `refs` are random tokens, so `AtomPool.validate()` rejects every one. The
alternative — deferring non-degeneracy (criterion 9) to Phase 2 — would let
`source_quality` and `temporal_correctness` stay silently constant through all of
Phase 1, which is the failure G5 and G7 exist to catch.

**Out.** The lattice generator and the exact evaluator (Phase 2), any policy or
learner (Phase 3), any search algorithm (Phase 4), the learned obligation parser
itself (Phase 5), the distilled utility head (Phase 9). **Phase 1 has zero ML
dependencies and imports no model of any kind** — this is not a preference, it is
v1.2 §4.4's prohibition enforced as an import test.

---

## 3. Modules

Seven units. Signatures are specification, not implementation.

### P1.0 The Phase-0 amendments (do these first)

**Additions.**

| Where | Change | Gap |
|---|---|---|
| `schemas.py` | `AtomPool`; `CandidateAtom.target`; `Obligations.scope`; `Obligations` rejects an empty `time_constraint`; `Violation`, `CheckResult` | G2, G5, G6, G8 |
| `ids.py` | `atom_id(kind, refs, target="", label="")` | G2 |
| `graphstore.py` | `GraphSnapshot.span` / `.turn`, plus `_PROTOCOL_MEMBERS`, `DictGraphSnapshot` storage, two `GRAPH_OPS`, `ReplayGraphStore._apply` branches | G6 |
| `config/` | `source_tiers`, `default_tier` | G7 |

**Both `SCHEMA_VERSION` constants move, and they are different constants.**
`graft/schemas.py:52` stamps every event-log line and the run manifest, and moves
because Tier A changed. `graft/config/schema.py:20` is validated for equality at
load, and moves because the config tree gained fields. Both go `0.1.0 → 0.2.0`.

**Expect the handoff fingerprints to change, once.** Adding `source_tiers` alters
`config_hash`, so all four values printed by `scripts/verify_handoff.py` shift.
That is a documented event, not a bug — but tell the team before they compare
numbers across the boundary, because a silently changed fingerprint is
indistinguishable from a broken environment.

**Surface.** `AtomPool(atoms: Iterable[CandidateAtom], cap: int)` ·
`pool[atom_id] -> CandidateAtom` · `pool.referencing(atom_id) -> tuple[str, ...]` ·
`pool.derive_bindings(atom_ids) -> dict[str, str]` · `pool.validate()`.

**`derive_bindings` belongs on `AtomPool`, not in `checker.py`.** It reads each
selected atom's `kind` and `label` and nothing else, so the pool is its natural
home — and putting it there resolves a build-order deadlock: `deficit()` needs
bindings for components 4 and 5, and would otherwise depend on a module built two
steps later.

**Gotcha.** `AtomPool` must reject an atom whose `ref` is not in the pool. It is
tempting to allow it and let the mask handle it — but such an atom is permanently
unaddable, so allowing it means Stage C can silently ship a pool where part of
the budget is unreachable and recall looks better than it is.

### P1.1 `graft/core/obligations.py`

**Responsibility.** Obligation parsing and the deficit vector.

**Surface.** `parse(question, mode) -> Obligations` · `deficit(state, pool, q, G) -> ndarray[6]` ·
`delta_deficit(before, after) -> ndarray[6]` · `slot_status(state, pool, q, G) -> dict[str, float]` ·
`DEFICIT_COMPONENTS: tuple[str, ...]`.

**Design notes.**
- `mode="exact"`: the synthetic instance carries its obligations, so this is a
  lookup with zero error. Say so plainly — the module is thin in Phase 1 and that
  is correct, not a shortcut.
- `mode="learned"`: raises `NotImplementedError` until Phase 5, with the audit
  hook (`slot_level_scores`) already defined so the number is reportable the
  moment the extractor exists. It reports precision, recall and F1 for the four
  optional slots and accuracy for the two boolean ones — one metric does not fit
  both families, and a slot-scoring function that only looks where gold is active
  measures recall while being called precision.
- `slot_status` is the single implementation shared by `d(s)` and `coverage`
  (G4).

**Gotcha.** `d` must be cheap. It is called at *every* construction step for
*every* learner; an implementation that rescans the pool turns a 16-step rollout
into an O(16·64) sweep per trajectory. Cache slot status incrementally alongside
the incremental checker.

### P1.2 `graft/core/checker.py` — formal validity `H`

**Surface.** `H(X, q, G, pool, *, ledger=None, first_failure_only=False) -> CheckResult` ·
`CHECKS: tuple[str, ...]`. (`Violation` and `CheckResult` come from `schemas.py`
per G8; `derive_bindings` from `AtomPool` per P1.0.)

**The nine sub-checks.**

| # | Check | Content | Incremental? |
|---|---|---|---|
| 1 | type legality | atom `kind` legal; referenced node/edge types legal against the schema vocabularies | per-atom |
| 2 | identity | every id resolves in the pool; no two selected atoms share a content hash (`frozenset` already excludes literal duplicates, so the real risk is a malformed pool double-counting evidence) | hash set |
| 3 | temporal | no bound claim whose validity is **disjoint** from the constraint; no `supersedes` edge pointing at a later `t_created` (G5) | scoped to affected binding |
| 4 | retired evidence | no atom's `target` is an edge with `t_invalid` set at the pinned snapshot | per-atom |
| 5 | scope | the `target`'s provenance resolves inside `q.scope`; fails closed when unresolvable (G6) | per-atom |
| 6 | size | `1 ≤ |X| ≤ max_atoms` (G1) | counter |
| 7 | support eligibility | no atom whose `target` is an assertion fails `G.is_eligible` (fix F9) | per-atom |
| 8 | structural closure | every atom's `refs` are selected (fix F10) | membership test |
| 9 | binding consistency | no two binding atoms claim the same slot (G3) | per-atom |

**Prohibition, enforced by construction.** `H` imports no model. There is no code
path by which a learned score reaches it. Entailment, sufficiency, authority and
answerability are routed to `U`, the gate and Stage B — the routing table of
v1.2 §4.4 is implemented as module boundaries and asserted by an import test.
Note the distinction that makes check 7 legal: *computing* entailment is learned;
*reading the flag it already produced* is a stored-field lookup, structurally
identical to check 4.

**Gotcha.** Checks 1–2 and 6–9 are cheap; 3 and 5 are not. Order them cheapest
first under `first_failure_only`, and *do not* let the ordering change which
violations a full audit reports.

### P1.3 `graft/core/incremental.py`

**Responsibility.** Maintain validity across `ADD` so that construction costs no
`terminal_checks` (Phase-0 gap G1, part 1).

**Surface.** `IncrementalChecker(pool, q, G, cfg)` · `.add(atom_id)` · `.undo()` ·
`.ok() -> bool` · `.state() -> ProofSet` · `.deficit() -> ndarray[6]`.

**Design notes.** Eight of nine sub-checks are per-atom or counter-based;
only 3 recomputes, and only over the affected binding. Amortised O(1) per `ADD`.
`.undo()` exists because beam search (S2) and local search backtrack, and
rebuilding from scratch would reintroduce the cost this class exists to remove.

**Gotcha, and the single most important test in this phase.** The incremental
checker and batch `H` **must agree, for every set, under every insertion order**.
Two implementations of one predicate that drift apart would make `stop_allowed`
disagree with the terminal filter — invalid sets reaching the terminal
distribution while every test still passes individually. Exit criterion 3 is a
property test over 10⁴ random `(set, order)` pairs.

### P1.4 `graft/core/utility.py` — executable `U`

**Surface.** `U(X, q, G, pool, gold, w) -> float` · `u_terms(...) -> dict[str, float]` ·
one public function per term.

| Term | Definition | Range | Character |
|---|---|---|---|
| `sufficiency` | `|X ∩ gold| / |gold|` | [0,1] | deterministic given gold. **[EVIDENCE]** Graph-S3 (ACL 2026) validates dense supervision from offline golden subgraphs (+15.6 acc / +17.2 F1 over sparse terminal reward) |
| `coverage` | `1 − mean(d₁..d₄)` over active slots; 1.0 when no slots are active | [0,1] | deterministic; shares `slot_status` with `d(s)` (G4) |
| `source_quality` | mean tier over selected atoms, from `config.source_tiers` (G7) | [0,1] | metadata-derived |
| `temporal_correctness` | `|constraint ∩ ⋃ evidence intervals| / |constraint|`; 1.0 when unbounded (G5) | [0,1] | deterministic interval arithmetic |
| `redundancy` | `(Σ_x F({x}) − F(X)) / Σ_x F({x})`, `F` facility location **over the pool** with similarity **clamped to [0,1]**; 0 for a singleton and when the denominator is 0 | [0,1) | deterministic, submodular |
| `size` | `|X| / max_atoms` | [0,1] | deterministic |

**On `redundancy`.** This is the v1.2 replacement for the degenerate
`max(0, S(X\e) − S(X) + ε)` form, which is near-constant when the proof score is
monotone in evidence. The form above is 0 for a perfectly complementary set and
approaches 1 as atoms become interchangeable — the signal actually wanted — and
it is the same objective family as the S3 submodular baseline, which makes the
two directly comparable. **[EVIDENCE]** Lin & Bilmes (ACL 2011) for the applied
form; Nemhauser–Wolsey–Fisher (1978) for the guarantee.

**The range only holds under two conditions, and both must be pinned or exit
criterion 7 fails non-deterministically.**
`F(S) = (1/|V|) Σ_{v∈V} max_{x∈S} sim(v, x)` is in [0,1] with
`red(X) ∈ [0, 1 − max_x F({x})/Σ_x F({x})] ⊆ [0,1)` **only if** `F` is monotone
submodular with `F(∅) = 0`.

* **Ground set `V` is the pool, not `X`.** With `V = X` the formula degenerates:
  every `v ∈ X` has `max_x sim(v,x) = sim(v,v) = 1`, so `F(X) = 1` identically.
* **Similarity is `max(0, cosine)`.** Raw cosines over arbitrary `feat` vectors
  are negative roughly half the time, which makes `F({x})` negative, breaks
  monotonicity from `F(∅) = 0`, and can drive the ratio outside [0,1] or flip
  its sign. Clamping restores both properties.
* **Zero denominator guard.** `Σ_x F({x}) = 0` (all-zero or mutually orthogonal
  features) yields `red = 0`, alongside the diagnostic counter below.

**Gold is required.** `gold=None` raises. Train-time `U` is deterministic against
gold (fix F1); inference ranks with the distilled head (Phase 9) and never calls
this function. Failing loudly is what stops a silent `sufficiency = 0` from
looking like a legitimately weak proof.

**Gotcha.** If any selected atom has a zero-length `feat`, `redundancy` is 0 and a
diagnostic counter increments. A misconfigured featurizer must be visible, not
silently worth 0.25 of the reward.

### P1.5 `graft/core/reward.py`

```
R(X | q, G)  = 1[H(X, q, G, pool)] · exp(β · U(X, q, G, pool, gold))
log R(X)     = β · U(X)  if H else −inf
R(FAIL)      = r_fail          log R(FAIL) = log(r_fail)
```

**Design notes.** The indicator is **multiplicative** — the v1.2 §1.1 error, kept
as a named regression test. `log_reward` exists because every Phase-3 balance
loss works in log space and must never compute `log(0)`; the masks guarantee no
invalid set is a terminal, and `FAIL` carries `log(r_fail)` explicitly.

`FAIL` is a genuine terminal and a **member of the target's support**, so
`Z = Σ_valid R(X) + r_fail`. It is reached only by budget exhaustion with `H = 0`,
never by an action — there is no `ABSTAIN` action (v1.2 §3.4/§4.2).

### P1.6 `graft/core/masks.py`

**Surface.** `legal_adds(state, pool, q, G, cfg) -> ndarray[bool]` ·
`stop_allowed(state) -> bool` · `is_dead_end(state) -> bool` ·
`Terminal` enum `{VALID, FAIL}`.

`legal_adds` excludes: already-selected atoms; atoms that would exceed
`max_atoms`; atoms failing a per-atom formal check (1, 4, 5, 7); and atoms whose
`refs` are not all selected. `stop_allowed = IncrementalChecker.ok()`.

**Design notes.** Only `STOP` is masked when `H = 0`; **`ADD` stays available**, so
traversing a formally invalid partial set is fine — the policy simply cannot stop
there. The `H`-monotonicity proof obligation was withdrawn for this reason
(v1.2 §4.3), and re-deriving it is a recurring temptation worth naming.

`is_dead_end` is true iff no legal `ADD` and `STOP` masked. **It has several
causes, not one:** budget exhaustion at `|X| = max_atoms` with `H = 0`; an empty
pool at `|X| = 0`; a pool exhausted below `max_atoms` with `H = 0`; and every
remaining atom failing a per-atom check or having unsatisfied `refs`. All
transition to `FAIL`. What distinguishes a healthy environment from
over-tight masks is *where* they occur, so the `|X|` distribution of dead ends is
a reported Phase-2 audit rather than a single rate.

### P1.7 `graft/tests/fixtures.py` — minimal coherent pools

**Responsibility.** Seeded generation of pools that `AtomPool.validate()` accepts
and that make every `U` term non-degenerate.

**Surface.** `coherent_pool(rng, n_nodes, n_edges, n_bindings) -> AtomPool` ·
`fixture_instance(rng) -> (AtomPool, Obligations, GraphSnapshot, ProofSet)`.

**Design notes.** Node atoms first, then edge atoms whose `refs` are node atoms
already generated, then binding atoms — the same nodes-first ordering that makes
closure satisfiable, used here to guarantee the pool validates. Each instance
emits a backing `DictGraphSnapshot` so that `target` resolves and checks 3/4/5/7
are exercised **against a snapshot rather than in-pool** — which is the whole
point of G2's second half. Mixed source tiers and non-degenerate `feat` vectors
are required, or criteria 9 and 10 cannot pass.

**This is not the lattice.** No dependency chains beyond `refs`, no conflicting
pairs, no enumerability, no exhaustive terminal set. It exists to exercise `H`,
`U` and the masks; Phase 2 builds the thing Gate 2 runs on. Keeping the two
separate matters, because a fixture generator that grows into an early
ProofLattice is the same failure mode Phase 2.5 was written to avoid.

**Gotcha.** Phase 0's `graft/tests/generators.py` cannot be reused for pools: its
`candidate_atom` gives edge and binding atoms `refs` of random tokens, so every
generated pool fails validation. It stays as-is for schema round-trips, which is
what it was written for.

### P1.8 `graft/tests/` — see §5

---

## 4. Build order inside Phase 1

Strictly sequential; each step testable before the next.

| Step | Build | Done when |
|---|---|---|
| 1 | Phase-0 amendments per P1.0: `AtomPool` (+`derive_bindings`), `CandidateAtom.target`, `atom_id` target key, `Obligations.scope`, empty-constraint rejection, `Violation`/`CheckResult`, `GraphSnapshot.span/turn` and its five knock-on sites, `source_tiers`, **both** `SCHEMA_VERSION`s | Phase-0 suite green again; `AtomPool` rejects an out-of-pool ref; new schema classes round-trip |
| 2 | `fixtures.py` (P1.7) | Generated pools validate; tiers and `feat` are mixed |
| 3 | `CHECKS` constants | Every sub-check name is a constant; no string matching anywhere |
| 4 | `obligations.py`: `slot_status`, `d(s)`, `parse(mode="exact")` | `d ∈ [0,1]^6` on fixture states; `coverage` identity holds; `d_source` is 1 on the unbound case |
| 5 | `checker.py`: nine sub-checks, batch `H` | Each check has a positive and a negative unit case, **checks 3/4/5/7 against a snapshot** |
| 6 | `incremental.py` | Agreement property over 10⁴ random (set, order) pairs, `ledger=None` |
| 7 | `masks.py` | Masks never permit an unresolved-ref atom; every dead end transitions to `FAIL` |
| 8 | `utility.py`: six terms | Every term in [0,1]; every term takes ≥2 distinct values on the fixtures |
| 9 | `reward.py` | `R(invalid) == 0` exactly; `R(FAIL) == r_fail`; `log R` never `log(0)` |

Step 1 is larger than it looks — four Phase-0 files and their tests — and steps 5
and 6 are the risk. Budget roughly a third of the phase on step 1 and a third on
5–6.

**Why `fixtures.py` comes before the checker.** Steps 5 and 8 both need pools
that validate, and writing the checker against ad-hoc hand-built pools produces
tests that pass for the wrong reason. It also forces `AtomPool`'s invariants to
be exercised the moment they exist.

---

## 5. Exit criteria

**Correctness**
1. Every one of the nine sub-checks has at least one positive and one negative unit case.
2. No formally invalid set is a reachable terminal other than `FAIL` — property test over random trajectories on a hand-built fixture pool.
3. **Incremental ≡ batch:** for 10⁴ random `(set, insertion order)` pairs, `IncrementalChecker.ok()` equals `H(set).ok`, and the violation sets match under full collection. Run with `ledger=None` — `terminal_checks` is capped at 32 and `Ledger.count` raises, so a metered run dies at call 33 (G9).
4. Masks never permit an atom whose `refs` are absent.
5. **Every dead end transitions to `FAIL`, and `VALID` and `FAIL` are the only terminal outcomes.** Not "the only dead end is budget exhaustion" — that is false, and the plan contradicts it in G1: an empty pool is a dead end at `|X| = 0`, as is a state where every unselected atom fails a per-atom check, as is a pool exhausted below `max_atoms` with `H = 0`. The `|X|` distribution of dead ends is a reported Phase-2 audit, because a mass of them at small `|X|` means the masks are too tight rather than the budget too small.
6. The empty set is not a legal terminal, and `stop_allowed(root)` is `False` (G1).

**Utility**
7. Every `U` term returns a value in [0, 1] on 10⁴ generated inputs.
8. `U` is order-invariant: the same set built in different orders scores identically.
9. **Non-degeneracy:** every one of the six terms takes at least two distinct values across the fixture set. Catches G5 and G7 at build time rather than in a Gate-2 result.
10. `redundancy` is 0 for a singleton, and strictly increases when an atom is replaced by a near-duplicate.

**Reward**
11. `R = 1[H]·exp(β·U)`; `R(invalid) == 0.0` exactly and `log_reward(invalid) == −inf`.
12. **Regression test for the v1.2 §1.1 error:** an invalid set never scores `R == 1`, at any `β`, including `U = 0`.
13. `R(FAIL) == r_fail` and `log R(FAIL) == log(r_fail)`.

**Metering**
14. One batch `H` on a completed set increments `terminal_checks` by exactly 1; a full 16-atom incremental construction increments it by 0 and `incremental_ops` by 16.

**Structure**
15. `graft.core` imports no ML library, no concrete graph store, and no module outside the Phase-0 handoff contract plus `AtomPool` — the v1.2 §4.4 prohibition as an import-graph test.
16. `d(s)` has fixed dimension 6, every component in [0, 1], and `DEFICIT_COMPONENTS` is exported so Phase 3 cannot hard-code an index.

---

## 6. Decisions to lock before writing code

Frozen at Gate 0; every later phase inherits them.

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | Empty set legal? | **No** — `1 ≤ |X| ≤ max_atoms` (G1) | Target support changes; `p*` re-derived; every Gate-2 result re-runs |
| 2 | Atom resolution | `AtomPool` in `schemas.py` Tier A **and** `CandidateAtom.target`; `atom_id` keyed on target (G2) | Log migration after Phase 5 |
| 3 | Bindings | derived from `binding`-kind atoms via `AtomPool.derive_bindings`; sub-check 9 forbids duplicate slots (G3) | Changes what a terminal is; Phases 2–4 re-run |
| 4 | `d(s)` | 6 components as tabled, with `d_source = 1` on the unmet-and-unbound case (G4) | **Re-runs Gate 2** — L7 is defined on `Δd` |
| 5 | Temporal split | `H` rejects disjoint intervals; `U` grades coverage of the constraint; empty constraint rejected at construction; unbounded constraint → 1.0 plus a diagnostic (G5) | Reward changes; all learners re-run |
| 6 | Scope | implemented, fail-closed, `scope=()` means unrestricted (G6) | Adds a checker sub-check; Phases 2–4 re-run |
| 7 | `source_tiers` | in config, `{first_party 1.0, corroborated 0.75, reported 0.5, unknown 0.25}` (G7) | Reward changes; all learners re-run |
| 8 | `redundancy` form | facility-location overlap ratio; ground set = pool; similarity `max(0, cosine)`; zero-denominator → 0 | Reward changes; all learners re-run |
| 9 | Trace type | `CheckResult(ok, violations)` in `schemas.py`, named check constants (G8) | Cheap |
| 10 | Metering | batch `H` = 1 `terminal_check`; incremental = `incremental_ops`; `ledger=None` for offline audits; one `query_scope` per instance in the Phase-3 harness (G9) | Every Phase 3–4 comparison re-runs |
| 11 | Both `SCHEMA_VERSION`s | `schemas.py` and `config/schema.py` both `0.1.0 → 0.2.0`; handoff fingerprints shift once | Log incompatibility |

**Resolved — the `source_quality` question.** Keep all six terms from the start
and have the lattice emit mixed source tiers. Dropping `source_quality` for
Phases 2–4 and adding it at Phase 9 would give a five-term reward to some
learners and a six-term reward to others, which is precisely the confound
v1.2 §5.1 forbids — and it is the kind that is invisible in a results table,
because nothing in the numbers says the reward changed underneath them. The extra
knob is the cheaper cost.

---

## 7. Explicitly not in Phase 1

No lattice generator · no exact evaluator · no policy · no learner · no search
algorithm · no learned obligation parser · no distilled utility head · no
PyTorch · no dataset code · no LLM calls.

If a `graft/core/` file imports `torch`, or a model of any kind, or
`ReplayGraphStore`, something has gone wrong — and exit criterion 15 will say so.

---

## 8. What Phase 2 will need from this, verbatim

```python
from graft.core.checker     import H, CHECKS
from graft.core.incremental import IncrementalChecker
from graft.core.masks       import legal_adds, stop_allowed, is_dead_end, Terminal
from graft.core.obligations import deficit, delta_deficit, DEFICIT_COMPONENTS, parse
from graft.core.reward      import reward, log_reward
from graft.core.utility     import U, u_terms
from graft.schemas          import AtomPool, CheckResult, Violation   # new in step 1
```

### Requirements this phase places on the Phase-2 lattice

Recorded here because they are consequences of Phase-1 decisions, and because two
of them are the difference between an audit that measures something and one that
reports zero having measured nothing.

1. **Each instance must carry a gold proof set.** `U`'s `sufficiency` is defined
   against it and Phase 4 scores with exact `U` (fix F13). Phase 2.1's current
   spec lists anchors, dependency pairs, conflicting pairs, temporal toys and
   distractors — but not gold.
2. **Instances must admit substitutable evidence and multiple disjoint valid
   modes.** v1.2 §9 makes "questions genuinely admit multiple materially
   different valid proof sets" one of three conditions for the GFlowNet argument
   to be convincing. The lattice is the one place that can be guaranteed by
   construction rather than hoped for.
3. **The generator must be able to emit colliding atoms** — two distinct atoms
   canonicalising to the same child state. Otherwise Phase 2.2's
   equivalent-action collision audit reports 0 having tested nothing.
   **[EVIDENCE]** Symmetry-Aware GFlowNets (ICML 2025): L₁ ≈ 0.12 uncorrected vs
   ≈ 0.01 corrected — the correction matters only if collisions exist, and
   whether they do is a measurement.
4. **Atoms must carry mixed source tiers and non-trivial `feat` vectors**, or
   `source_quality` and `redundancy` are constant and exit criterion 9 fails.
5. **Time constraints must be bounded on both sides.** An unbounded constraint
   scores `temporal_correctness = 1.0` by the G5 convention, so a lattice that
   emits them tests nothing in that term.
6. **Atoms must carry a `target` resolving into a backing snapshot.** The lattice
   *can* synthesise everything in-pool and never touch `G` — and if it does,
   checks 3, 4, 5 and 7 ship through Gate 2 and Gate 3 never having run against
   the thing they check (G2). Each instance must therefore emit a
   `DictGraphSnapshot` alongside its pool, including at least one invalidated
   edge and one quarantined assertion so checks 4 and 7 have negative cases.
7. **`max_atoms = 8` and `pool_cap = 32`** come from the `synthetic` config
   profile, not from the real-data defaults (`PHASE0_DECISIONS.md` §2.1).
8. **The lattice must construct its pool as `AtomPool(atoms, cap=cfg.pool_cap)`.**
   `AtomPool` accepts `cap=None` and does not enforce a bound of its own — the
   bound belongs at the construction site, because a required parameter would
   not stop Phase 2 or Phase 7 passing the *wrong* bound. An uncapped lattice
   pool is not a type error but it is a real one: the state space stops being
   bounded and the enumerability that Gate 2 rests on quietly goes with it.
   Phase 7's pool builder carries the same requirement.
