# GRAFT — Phase 1 Build Plan: the deterministic core (`graft/core/`)

**`H`, `U`, `R`, the action masks, the obligation parser, and the obligation deficit vector `d(s)`.**

Date: 8 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 1) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §4) · `GRAFT_PHASE0_BUILD.md` (complete)
Effort: ~1–1.5 weeks solo
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

Now score it. `sufficiency = 0`, `coverage = 0`, `source_quality = 0`,
`redundancy = 0`, `size = 0` — and `temporal_correctness = 1.0`, because a set
with no bindings vacuously contradicts no time constraint.

```
U(∅) = 1.0·0 + 0.5·0 + 0.25·0 + 0.5·1.0 − 0.25·0 − 0.1·0  =  0.5
R(∅) = exp(4 · 0.5) = 7.39
```

The worst formally valid non-empty set scores `R = 0.247`. **The empty proof set
outranks the bottom third of the valid utility range**, and a question whose
obligation slots are all inactive scores it at `U = 1.0`, `R = 54.6`.

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

### G2 — `H` has no way to resolve an atom id [ANALYSIS]

The architecture's signature is `H(X: ProofSet, q: Obligations, G: GraphSnapshot)`.
But `ProofSet.atoms` is `frozenset[str]` — bare ids — and every sub-check needs
the atom's `kind`, `refs` and `label`. Nothing in the signature can supply them.

**Decision.** Add **`AtomPool`** to `graft/schemas.py` as a Tier-A type, and give
`H` a fourth parameter. Phase 0's own handoff rule applies here verbatim: *"If
Phase 1 needs a sixth import, Phase 0 is incomplete — fix it there, not by
widening Phase 1."* This is that case, and the fix is cheap today because no
experiment has consumed Tier A yet.

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

**Consequence.** `SCHEMA_VERSION` goes `0.1.0 → 0.2.0`. Free now; a migration
after Phase 5 writes real logs.

### G3 — Bindings are derived, not chosen, and nothing said so [ANALYSIS]

`ProofSet` carries `bindings: {slot → atom_id}`, but the action space is
`ADD(atom)` and `STOP` only (v1.2 §3.4). There is no `BIND` action, so nothing
can populate that field — yet `H`'s temporal check is defined over bindings and
`U`'s `temporal_correctness` grades them.

**Decision.** A binding **is** an atom: `CandidateAtom.kind == "binding"`, whose
`label` names the slot and whose `refs` name the referents. `ProofSet.bindings`
is therefore *derived* from the selected set, never set independently, by
`derive_bindings(atoms, pool)`.

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
| 4 | `d_source` | fraction of current bindings with no resolvable span; 0 when there are no bindings |
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
  scores low; tight, complete evidence scores 1.0. Unbounded constraint → 1.0.

This keeps `H` a formal predicate and puts the graded judgment in `U`, which is
the routing rule of v1.2 §4.4 applied to a term that was about to violate it.

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

On the Phase-2 lattice `scope` is empty and the two new protocol methods return
`None`, costing one line each.

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

`Violation` and `CheckResult` are dataclasses defined in `core/checker.py`, which
means extending the Phase-0 structural allow-list. That is the same exception
already granted to `Event` in `eventlog.py` — a module's own result format, not
part of the persisted data model — and the allow-list entry carries that
rationale so the rule keeps its teeth.

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

**The loophole, stated so it is watched.** `ledger=None` is a way for a search
module to cheat the budget. Mitigation: the Phase-3/4 harness always constructs
search modules with a ledger, and a test asserts that every `SearchModule` in the
registry calls `H` with one. Named here because the whole comparison rests on it.

---

## 2. Scope

**In.** `H` and its nine sub-checks (batch and incremental, proved to agree),
`U` and its six terms, `R` and `log R`, the `FAIL` terminal, `legal_adds` /
`stop_allowed` / dead-end detection, the obligation parser's exact mode and its
learned-mode interface, `d(s)`, `derive_bindings`, and the Phase-0 amendments
G2/G6 require.

**Out.** The lattice generator and the exact evaluator (Phase 2), any policy or
learner (Phase 3), any search algorithm (Phase 4), the learned obligation parser
itself (Phase 5), the distilled utility head (Phase 9). **Phase 1 has zero ML
dependencies and imports no model of any kind** — this is not a preference, it is
v1.2 §4.4's prohibition enforced as an import test.

---

## 3. Modules

Seven units. Signatures are specification, not implementation.

### P1.0 `graft/schemas.py` — the Phase-0 amendments (do these first)

**Additions.** `AtomPool` (G2); `Obligations.scope` (G6); `GraphSnapshot.span` /
`.turn` (G6, in `graphstore.py`). `SCHEMA_VERSION → 0.2.0`.

**Surface.** `AtomPool(atoms: Iterable[CandidateAtom], cap: int)` ·
`pool[atom_id] -> CandidateAtom` · `pool.referencing(atom_id) -> tuple[str, ...]` ·
`pool.validate()`.

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
  hook (`slot_level_precision`) already defined so the number is reportable the
  moment the extractor exists.
- `slot_status` is the single implementation shared by `d(s)` and `coverage`
  (G4).

**Gotcha.** `d` must be cheap. It is called at *every* construction step for
*every* learner; an implementation that rescans the pool turns a 16-step rollout
into an O(16·64) sweep per trajectory. Cache slot status incrementally alongside
the incremental checker.

### P1.2 `graft/core/checker.py` — formal validity `H`

**Surface.** `H(X, q, G, pool, *, ledger=None, first_failure_only=False) -> CheckResult` ·
`Violation` · `CheckResult` · `CHECKS: tuple[str, ...]` · `derive_bindings(atoms, pool)`.

**The nine sub-checks.**

| # | Check | Content | Incremental? |
|---|---|---|---|
| 1 | type legality | atom `kind` legal; referenced node/edge types legal against the schema vocabularies | per-atom |
| 2 | identity | every id resolves in the pool; no two selected atoms share a content hash (`frozenset` already excludes literal duplicates, so the real risk is a malformed pool double-counting evidence) | hash set |
| 3 | temporal | no bound claim whose validity is **disjoint** from the constraint; no `supersedes` edge pointing at a later `t_created` (G5) | scoped to affected binding |
| 4 | retired evidence | no atom references an edge with `t_invalid` set at the pinned snapshot | per-atom |
| 5 | scope | provenance resolves inside `q.scope`; fails closed when unresolvable (G6) | per-atom |
| 6 | size | `1 ≤ |X| ≤ max_atoms` (G1) | counter |
| 7 | support eligibility | no atom references a quarantined assertion (fix F9) | per-atom |
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
| `redundancy` | `(Σ_x F({x}) − F(X)) / Σ_x F({x})` with `F` the facility-location coverage over `feat` cosines; 0 for a singleton | [0,1] | deterministic, submodular |
| `size` | `|X| / max_atoms` | [0,1] | deterministic |

**On `redundancy`.** This is the v1.2 replacement for the degenerate
`max(0, S(X\e) − S(X) + ε)` form, which is near-constant when the proof score is
monotone in evidence. The form above is 0 for a perfectly complementary set and
approaches 1 as atoms become interchangeable — the signal actually wanted — and
it is the same objective family as the S3 submodular baseline, which makes the
two directly comparable. **[EVIDENCE]** Lin & Bilmes (ACL 2011) for the applied
form; Nemhauser–Wolsey–Fisher (1978) for the guarantee.

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

`is_dead_end` is true iff no legal `ADD` and `STOP` masked. Under the closure
rule this can only happen at `|X| = max_atoms` with `H = 0`, i.e. budget
exhaustion — which transitions to `FAIL`. A dead end arising anywhere else means
the masks are too tight, and the rate is a reported Phase-2 audit.

### P1.7 `graft/tests/` — see §5

---

## 4. Build order inside Phase 1

Strictly sequential; each step testable before the next.

| Step | Build | Done when |
|---|---|---|
| 1 | Phase-0 amendments: `AtomPool`, `Obligations.scope`, `GraphSnapshot.span/turn`, `source_tiers` in config, `SCHEMA_VERSION → 0.2.0` | Phase-0 suite still green; `AtomPool` rejects an out-of-pool ref |
| 2 | `Violation` / `CheckResult` / `CHECKS`; structural allow-list extended | Every sub-check name is a constant; no string matching anywhere |
| 3 | `obligations.py`: `slot_status`, `d(s)`, `parse(mode="exact")` | `d ∈ [0,1]^6` on generated states; `coverage` identity holds |
| 4 | `checker.py`: nine sub-checks, batch `H` | Each check has a positive and a negative unit case |
| 5 | `incremental.py` | Agreement property over 10⁴ random (set, order) pairs |
| 6 | `masks.py` | Masks never permit an unresolved-ref atom; dead end ⇔ budget exhaustion |
| 7 | `utility.py`: six terms | Every term in [0,1]; every term takes ≥2 distinct values on the fixtures |
| 8 | `reward.py` | `R(invalid) == 0` exactly; `R(FAIL) == r_fail`; `log R` never `log(0)` |

Steps 4 and 5 are the risk. Budget half the phase there.

---

## 5. Exit criteria

**Correctness**
1. Every one of the nine sub-checks has at least one positive and one negative unit case.
2. No formally invalid set is a reachable terminal other than `FAIL` — property test over random trajectories on a hand-built fixture pool.
3. **Incremental ≡ batch:** for 10⁴ random `(set, insertion order)` pairs, `IncrementalChecker.ok()` equals `H(set).ok`, and the violation sets match under full collection.
4. Masks never permit an atom whose `refs` are absent.
5. The only zero-legal-action state is budget exhaustion, and it maps to `FAIL`.
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
| 2 | `AtomPool` home | `schemas.py` Tier A, `SCHEMA_VERSION → 0.2.0` (G2) | Log migration after Phase 5 |
| 3 | Bindings | derived from `binding`-kind atoms; sub-check 9 forbids duplicate slots (G3) | Changes what a terminal is; Phases 2–4 re-run |
| 4 | `d(s)` | 6 components as tabled (G4) | **Re-runs Gate 2** — L7 is defined on `Δd` |
| 5 | Temporal split | `H` rejects disjoint intervals; `U` grades coverage of the constraint (G5) | Reward changes; all learners re-run |
| 6 | Scope | implemented, fail-closed, `scope=()` means unrestricted (G6) | Adds a checker sub-check; Phases 2–4 re-run |
| 7 | `source_tiers` | in config, `{first_party 1.0, corroborated 0.75, reported 0.5, unknown 0.25}` (G7) | Reward changes; all learners re-run |
| 8 | `redundancy` form | facility-location overlap ratio, cosine over `feat` | Reward changes; all learners re-run |
| 9 | Trace type | `CheckResult(ok, violations)`, named check constants (G8) | Cheap |
| 10 | Metering | batch `H` = 1 `terminal_check`; incremental = `incremental_ops`; `ledger=None` for offline audits (G9) | Every Phase 3–4 comparison re-runs |

**Open question for you.** `source_tiers` adds a fourth reward-shaping knob
alongside `beta`, `u_weights` and the term normalisations. The alternative is to
drop `source_quality` from `U` entirely on the lattice (where source type is
synthetic anyway) and reintroduce it at Phase 9 when real `Source` metadata
exists. That would make `U` five-term for Phases 2–4 and six-term afterwards —
**which breaks the "identical reward across all seven learners" rule if any
learner is trained before and after the change.** My recommendation is to keep
all six terms from the start and make the lattice generate mixed tiers, because
the cost of the alternative is exactly the confound v1.2 §5.1 forbids.

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
from graft.core.checker     import H, CheckResult, Violation, CHECKS, derive_bindings
from graft.core.incremental import IncrementalChecker
from graft.core.masks       import legal_adds, stop_allowed, is_dead_end, Terminal
from graft.core.obligations import deficit, delta_deficit, DEFICIT_COMPONENTS, parse
from graft.core.reward      import reward, log_reward
from graft.core.utility     import U, u_terms
from graft.schemas          import AtomPool          # new in Phase 1 step 1
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
5. **`max_atoms = 8` and `pool_cap = 32`** come from the `synthetic` config
   profile, not from the real-data defaults (`PHASE0_DECISIONS.md` §2.1).
