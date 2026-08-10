# Phase 1 — decisions taken, with reasoning and cost to reverse

Date: 8 August 2026
Parent: `GRAFT_PHASE1_BUILD.md` · `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (v1.1) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2)
Predecessor: `PHASE0_DECISIONS.md`

Phase 1 is complete. **All 16 exit criteria pass; 366 tests, ~7 s.** `graft/core/`
is ~1,800 lines across seven modules.

This file records what the build decided that the plan did not, and where the
implementation departed from the plan. Every departure is listed — including the
three that were found by writing tests rather than by reading the spec (§4), and
the five invariants a post-build review found were not being enforced (§5).

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

## 5. Post-build review — five fixes and one rejection

A review after Phase 1 closed raised six findings. All six reproduced. Five were
fixed; one was rejected, and the reason matters more than the fix would have.

Every fix is an invariant enforced in a `__post_init__` that already existed.
None touches `U`, the reward, `d(s)`, or any frozen Gate-0 value, so no
experiment is invalidated.

### 5.1 `frozen=True` did not mean frozen

`@dataclass(frozen=True)` blocks *rebinding* a field. It does nothing about
mutating the object a field points at — so four mappings inside frozen
dataclasses were writable, and each sat somewhere that depended on it not
changing:

| Field | What depended on it |
|---|---|
| `Config.source_tiers` | `config_hash`, the identity of an experimental condition |
| `ProofSet.bindings` | a dict key in Phase 2's DP and Phase 4's beam dedup |
| `Node.payload` | a record read after the snapshot is built |
| `OutputRecord.ledger_snapshot` | a write-once record |

Reproduced on the first two: mutating `source_tiers` moved the config hash
`177b7c96…` → `2ef237c8…` mid-run; mutating `bindings` changed a `ProofSet`'s
hash and made its own dict entry unreachable.

All four now go through `_freeze_mapping` (`MappingProxyType`). One limitation
recorded rather than engineered around: a mappingproxy cannot be pickled. If a
later phase needs to pickle a `Config` or a `ProofSet`, the transport is
`to_dict()`/`from_dict()`, which already exist. A bespoke picklable frozen-dict
class would be more code for a speculative need.

`dataclasses` refuses an unhashable *default*, so `source_tiers` uses a factory
returning the module constant — safe precisely because that constant is itself
read-only now.

### 5.2 `ProofSet` identity was not the atom set

The review framed this as mutability. The deeper defect was that equality and
hashing included `bindings` at all.

`bindings` is *derived* — a pure function of `atoms` and the pool (gap G3). Two
proof sets over identical atoms whose bindings differed compared **unequal**,
which directly contradicts "two orders reaching the same atoms are the same
state", the canonicalisation the whole set formulation rests on. Since bindings
are a function of the atoms, any difference means one side was built wrongly, so
treating them as equal is both correct and robust to a caller who forgot to
derive.

Identity is now `atoms` alone. That fixes more than validation would have, and
is less code.

Nothing downstream read `ProofSet.bindings` anyway — `H` uses
`pool.binding_slots()`, `U` never touches it — so the field was inert *except*
in the key path, which is the one place it could do damage.

### 5.3 A superseded edge could still read as live

`Edge` permitted `superseded_by` without `t_invalid`, while `is_live` reads only
`t_invalid`. Confirmed: such an edge reported live.

The argument that settled it is not Zep's model but `is_live`'s own docstring,
which claims to be "the one place invalidation semantics live, so that no
downstream component reimplements invalidation semantics". If supersession
implied invalidation and `is_live` ignored it, that claim was already false and a
superseded fact would keep answering questions — in the subsystem this project
claims as a strength.

Now enforced at construction: `superseded_by ⇒ t_invalid is not None`. One field
stays authoritative, which is cheaper than teaching every reader about two. The
converse is deliberately not required — an edge may be retired with nothing
replacing it.

### 5.4 The support gate defaulted open

`AssertionFlags.entailed_by_span` defaults `False`, but `Assertion.eligibility`
defaulted `"eligible"` — so an assertion with no support was admissible evidence
by omission.

Every other decision on that path fails closed: `H`'s support sub-check rejects a
claim with no traceable assertion, the scope check rejects a broken provenance
chain, `is_eligible` returns `False` for an unknown id. This was the only
fail-open step, and it sat exactly where a skipped support-gate result would
land. Default is now `"quarantined"`, including in `from_dict` for records
written before the field existed. Forgetting to set it now costs recall, not
correctness.

### 5.5 The parser metric was recall wearing the name precision

`slot_level_precision` scored only items where *gold* had the slot active, so a
parser hallucinating an anchor on every blank question scored **1.0**. It also
omitted `aggregate` entirely.

That number was destined for a paper — architecture F2 requires it "reported
wherever coverage is reported" — which makes it the overreaching pattern
`CLAUDE.md` §5 catalogues, in our own code.

Replaced by `slot_level_scores`, returning a flat `{slot}.{metric}` dict. The fix
is not simply "count false positives": the two slot families need different
metrics. Optional slots (`entity_anchor`, `value_type`, `time_constraint`,
`scope`) get precision, recall and F1; boolean slots (`needs_source`,
`aggregate`) get accuracy, because `False` is a prediction rather than an
absence. A slot neither predicted nor imposed reports `nan` rather than a
flattering 1.0. The hallucinating parser above now scores precision 0.333.

A test asserts every field of `Obligations` is scored, so a slot added later
cannot go unaudited.

### 5.6 Rejected — `AtomPool` keeps accepting `cap=None`

The finding was that the `pool_cap` bound is optional, and the plan calls it
load-bearing for the generalization argument.

**Rejected, because it is the wrong layer.** Making `cap` required would add
`cap=64` to ~15 test call sites to buy a guarantee that does not hold: a type
that refuses to exist without a bound still cannot stop Phase 7 passing the
*wrong* bound. The guarantee belongs at the two sites where pools are built for
real, as a test there.

There is a genuine risk underneath, and it is Phase 2's: a lattice that builds
its pool without `cfg.pool_cap` has an unbounded state space, and the
enumerability Gate 2 rests on goes with it. That is now requirement 8 in
`GRAFT_PHASE1_BUILD.md` §8, where it can actually be checked.

### 5.7 Schema version

`graft.schemas.SCHEMA_VERSION` → **0.3.0**: §5.3 and §5.4 change how an existing
record would be *read*, so the on-disk interpretation moved even though its shape
did not. `graft.config.SCHEMA_VERSION` stays at 0.2.0 — the config tree did not
change. The two now diverge, which is the point of having versioned them
separately.

### 5.8 The environment report was a false alarm

The review reported `.venv` as stale, referencing a missing Python 3.11 against a
machine exposing only 3.14. Not reproducible here: 3.11.9 is installed, `py -0p`
lists it, and the venv runs clean. The review ran on a different machine.

That is `PHASE0_DECISIONS.md` §3.1 working as designed rather than a defect. A
venv records the absolute path of the interpreter that built it and **is not
portable**; the answer on a machine without 3.11 is `scripts/bootstrap.ps1`,
which fails with an explicit "install 3.11" message rather than half-working.
Worth noting too that `pyproject.toml` declares `>=3.11,<3.13`, so verifying
under a 3.12 runtime is *inside* the supported window — the only thing genuinely
unreproduced there is the exact pin set, which the review was right to flag.

---

## 6. Handoff fingerprints moved once, as predicted

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

## 7. Handoff to Phase 2

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
7. `max_atoms = 8`, `pool_cap = 32` from the `synthetic` profile;
8. **the pool constructed as `AtomPool(atoms, cap=cfg.pool_cap)`** — `AtomPool`
   accepts `cap=None` by design (§5.6), so an uncapped lattice pool is not a
   type error but it does dissolve the bounded state space Gate 2's
   enumerability rests on.

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
