# Phase 1 — decisions taken, with reasoning and cost to reverse

Date: 8 August 2026
Parent: `GRAFT_PHASE1_BUILD.md` · `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (v1.1) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2)
Predecessor: `PHASE0_DECISIONS.md`

Phase 1 is complete. **All 16 exit criteria pass; 347 tests, ~7 s.** `graft/core/`
is ~1,760 lines across seven modules.

This file records what the build decided that the plan did not, and where the
implementation departed from the plan. Every departure is listed — including the
three that were found by writing tests rather than by reading the spec.

---

## 1. The plan's ten §6 decisions — all implemented as written

| # | Decision | Where it lives |
|---|---|---|
| 1 | Empty set is not a legal terminal (`1 ≤ \|X\| ≤ max_atoms`) | `checker.set_level_violations`, sub-check 6 |
| 2 | `AtomPool` in `schemas.py` **and** `CandidateAtom.target`; `atom_id` keyed on target | `schemas.py`, `ids.py` |
| 3 | Bindings derived via `AtomPool.derive_bindings`; sub-check 9 forbids duplicate slots | `schemas.py`, `checker.py` |
| 4 | `d(s)` = 6 components, `d_source = 1` on the unmet-and-unbound case | `core/obligations.py` |
| 5 | `H` rejects disjoint intervals; `U` grades coverage; empty constraint refused; unbounded → 1.0 + diagnostic | `schemas.Obligations`, `core/obligations.covered_fraction` |
| 6 | Scope implemented, fail-closed, `scope=()` unrestricted | `checker.per_atom_violations` |
| 7 | `source_tiers` in config, four tiers | `config/schema.py` |
| 8 | Facility-location redundancy, ground set = pool, similarity clamped, zero-denominator guard | `core/utility.redundancy` |
| 9 | `CheckResult(ok, violations)` in `schemas.py`, named check constants | `schemas.py`, `checker.CHECKS` |
| 10 | Batch `H` = 1 `terminal_check`; incremental = `incremental_ops`; `ledger=None` for audits | `checker.H`, `incremental.add` |
| 11 | Both `SCHEMA_VERSION`s → `0.2.0` | `schemas.py`, `config/schema.py` |

The `source_quality` question was resolved before coding: **all six terms from
the start**, lattice emits mixed tiers.

---

## 2. Departures from the plan

Each is additive or a correction; none reverses a §6 decision.

### 2.1 A seventh core module, `core/resolve.py`

**Decision.** Atom→graph resolution lives in one module: target lookup, backing
assertions, provenance spans, conversation ids, intervals, source tier, anchor
and value-type matching.

**Why.** The checker, the deficit vector and the utility all walk the same three
paths. Three copies would drift, and the drift would be silent — a checker
resolving provenance one way and a utility another is not a crash, it is a
result nobody can interpret.

**Rule it keeps.** *Everything in `resolve` reads; nothing decides.* It returns
facts and, where a chain is broken, says so. What a broken chain **means** is the
caller's: `H` treats unresolvable provenance under a non-empty scope as a
violation, `U` treats an unresolvable source as the default tier. Keeping the
read and the judgment apart is what stops a lookup helper from becoming policy.

### 2.2 `GraphSnapshot` gained three members, not two

The plan listed `span` and `turn`. `assertion(assertion_id)` was also needed:
scope has to walk a node atom's provenance to a `conv_id`, and for a node the
only route to spans is the backing assertion. The alternative — duplicating
`Assertion.spans` into `Node.payload` at commit time — gives provenance two
sources of truth, which is worse than one more read method.

### 2.3 `H` takes `cfg`; `legal_adds` takes only the state

`H(X, q, G, pool, cfg, *, ledger, first_failure_only)` — sub-check 6 needs
`max_atoms`, and `pool.cap` is a different constant (`pool_cap`).

`legal_adds(state)` rather than `legal_adds(state, pool, q, G, cfg)`: the
`IncrementalChecker` already carries all four, and the plan's own neighbours
(`stop_allowed(state)`, `is_dead_end(state)`) assume it does.

### 2.4 The batch and incremental paths share one assembly function

Both call `checker.assemble` over a per-atom dict and a set-level dict. Exit
criterion 3 (10⁴ (set, order) pairs) then **confirms** agreement rather than
being the only thing holding it up — a test that carried the guarantee alone
would be one refactor away from silently stopping.

The split is also what "amortised O(1) per ADD" actually means here: the
per-atom checks do all the graph traversal and are memoised across `add`/`undo`;
the set-level checks are pure bookkeeping over ≤ `max_atoms` entries and are
recomputed, which at 16 atoms is cheaper than maintaining and having to prove
correct an incremental version of each.

### 2.5 Masks exclude every per-atom violation, including supersession

The plan's parenthetical said "(1, 4, 5, 7)". The supersession half of check 3 is
per-atom and permanent too, so an atom carrying it can never appear in a valid
set. Excluding it prunes branches that could only ever reach `FAIL`; including it
would let the policy spend budget on sets that can never stop.

### 2.6 The temporal contradiction is measured against the graph, not the selection

`resolve.validity_intervals` follows live `valid_during` edges out of the bound
node, rather than reading only what the proof selected.

**Found by writing a negative test.** As first implemented, the check read the
binding's referents' own intervals — and a binding's refs are a claim node and a
value node, neither of which carries one. The check could never fire.

The deeper reason it must use the graph: a proof that binds a claim the graph
says was valid only outside the asked-about window is contradictory whether or
not it selected the interval edge. A selection-dependent check would let a proof
escape by *omitting* evidence, which is the opposite of what a formal check is
for. `U`'s `temporal_correctness` stays selection-based, because it grades what
the proof actually presents.

---

## 3. Decisions the plan left to the implementation

### 3.1 `coverage` counts slots addressed, not one minus the graded deficit

**The plan's formula was `coverage = 1 − mean(d₁..d₄)`.** Taken literally,
`temporal_correctness = 1 − d_time` *exactly*, so `d_time` enters `U` twice and
temporal evidence carries **2–5× the weight of any other obligation slot**
depending on how many are active:

| active slots | temporal weight | any other slot |
|---|---|---|
| 1 | 1.000 | 0.500 |
| 2 | 0.750 | 0.250 |
| 3 | 0.667 | 0.167 |
| 4 | 0.625 | 0.125 |

Nothing in the plan intends that.

**Decision.** A slot counts as addressed when its deficit is **below 1** — some
evidence bears on it. `coverage` asks *is this obligation addressed at all*;
`temporal_correctness` asks *how precisely*. Two questions, two terms, which is
the same hard-versus-graded split as gap G5's separation of `H` from `U`.

G1's arithmetic is preserved exactly: the empty set has every deficit at 1, so
`coverage = 0`, and `U(∅) = 0.5`, `R = 7.39` still hold.

**Cost to reverse.** Reward changes; all learners re-run.

### 3.2 Payload conventions

`Node.payload` is free-form, so the keys the core reads are constants in
`schemas.py` rather than string literals: `assertion_id`, `name`, `aliases`,
`value_type`, `tier`. `ASSERTION_BACKED_NTYPES = (Claim, Value, Event)` names the
node types the support gate applies to.

**A node of an assertion-backed type carrying no `assertion_id` is a violation,
not a skip.** The point of check 7 is that unsupported claims cannot reach a
proof, so "no evidence against it" must not read as "supported". Phase 6 has to
honour this at commit time.

### 3.3 Scope has three states, not two

*Neutral* — an atom with no provenance at all (`Entity`, `Source`,
`TimeInterval`) is abstract and has no conversational origin. Scoping it would
make every structural atom fail under any scope.
*Outside* — provenance resolves to a conversation not in scope: a violation.
*Broken* — the chain cannot be resolved: **also a violation**. The atom was not
shown to be in scope, and failing closed is what stops a missing span from
reading as a clean result.

### 3.4 `AtomPool` also rejects reference cycles

The plan named the out-of-pool reference. A cycle is the same bug wearing a
different hat: each atom in it waits on the others, so none can ever be added.
Detection is an iterative DFS — iterative so a large pool cannot blow the stack.

`AtomPool` is a plain class rather than a dataclass, which keeps it out of the
`test_every_schema_class_has_a_generator` contract (it is a container, not a
record) and lets it own its index and invariants.

### 3.5 A lone node atom is formally valid

It looks like a bug and is not, so it has a named test.

A single `Entity` node violates no schema, id, interval, support or scope
constraint, so `STOP` is allowed. What makes it a useless proof is `U` —
near-zero sufficiency and coverage — which is exactly the division of labour
`H` and `U` are for. **A checker that rejected it would be smuggling semantic
judgment into a formal predicate**, which is v1.2 §4.4's prohibition in reverse.

Two of my own first-draft tests asserted the opposite and were wrong.

### 3.6 Binding atoms must name a slot and denote nothing

`kind == "binding"` requires a non-empty `label` (sub-check 9 is defined over it)
and an empty `target` (a binding names a slot and its referents; it is not a
graph object). Both enforced at construction.

---

## 4. What the tests found that reading did not

Three things surfaced only from running code, and all three would have been
invisible in a Gate-2 table.

**The temporal contradiction check could never fire** (§2.6). Caught by trying to
write its negative case.

**`temporal_correctness` was binary in practice.** It passed the plan's
non-degeneracy criterion — "at least two distinct values" — while taking only
{0, 1}, because every fixture interval either matched the question's constraint
exactly or missed it entirely. That is the term behaving as a *presence flag*,
which is precisely the collapse G5 split it out to avoid. Fixed by giving the
fixtures partially-overlapping windows, and by a second test asserting that the
two continuous terms take **strictly interior** values:

| term | distinct values | strictly interior |
|---|---|---|
| sufficiency | 9 | 7 |
| coverage | 5 | 3 |
| source_quality | 162 | 160 |
| temporal_correctness | 8 | 6 |
| redundancy | 9,404 | 9,330 |
| size | 17 | 15 |

**Dead ends occur only at `|X| = max_atoms`.** Under mask-respecting rollouts
with `p_stop = 0`, 110 of 480 trajectories reached `FAIL`, every one at the size
limit. That is the healthy signature — the *budget* ran out. A mass at small
`|X|` would mean the `ADD` masks are too tight, and Phase 2 reports this
distribution for the lattice rather than a single dead-end rate.

---

## 5. Handoff fingerprints moved once, as predicted

Adding `source_tiers` changed `config_hash`, and adding turns and spans to
`DictGraphSnapshot` changed `state_digest`:

| fingerprint | before | after |
|---|---|---|
| config hash | `9f6e93a6…` | `b7515c00…` |
| log digest | `878095bb…` | `878095bb…` (unchanged) |
| graph digest | `4b6b4438…` | `f258d5a6…` |
| manifest | `3319991…` | `4f9003c…` |

A documented event, not a bug — but tell the team before they compare numbers
across the boundary, because a silently changed fingerprint is indistinguishable
from a broken environment.

---

## 6. Handoff to Phase 2

```python
from graft.core.checker     import H, CHECKS
from graft.core.incremental import IncrementalChecker
from graft.core.masks       import Terminal, is_dead_end, legal_adds, stop_allowed
from graft.core.obligations import DEFICIT_COMPONENTS, deficit, delta_deficit, parse
from graft.core.reward      import log_reward, reward
from graft.core.utility     import U, u_terms
from graft.schemas          import AtomPool, CheckResult, Violation
```

Verified by a test, so it works on the day Phase 1 exits.

### What Phase 2's lattice must provide

Unchanged from `GRAFT_PHASE1_BUILD.md` §8, with one addition learned here:

1. a gold proof set per instance;
2. substitutable evidence and multiple disjoint valid modes;
3. atoms that can collide under canonicalisation, or the collision audit tests nothing;
4. mixed source tiers and non-trivial `feat`;
5. **bounded** time constraints — and **partially overlapping** ones, not only
   exact-or-disjoint, or `temporal_correctness` silently reverts to a presence
   flag (§4);
6. `target`s resolving into a backing snapshot, with at least one invalidated
   edge and one quarantined assertion;
7. `max_atoms = 8`, `pool_cap = 32` from the `synthetic` profile.

`graft/tests/fixtures.py` is a working reference for 1, 4, 5, 6 — but it is
**not** the lattice: no conflicting pairs, no dependency chains beyond `refs`, no
enumerability. Phase 2 builds the thing Gate 2 runs on, and a fixture generator
that grows into an early ProofLattice is the failure mode Phase 2.5 exists to
avoid.

### Still open, and not blocking

`beta = 4.0` remains a placeholder until the Phase-3 sweep. The
`r_fail_margin` check will refuse the config if that sweep pushes `beta` high
enough to make `FAIL` competitive — which is the automated form of the open item
in `CLAUDE.md` §7.
