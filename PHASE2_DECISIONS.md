# PHASE2_DECISIONS.md — what the Phase-2 build actually decided

**Status: built and green as of 11 August 2026. All 21 exit criteria.**
487 tests, ~25 s (Phase 0+1 were 367; Phase 2 adds 120). The count includes
the twelve regression tests added by the post-build review in §5.

Companion to `GRAFT_PHASE2_BUILD.md`, in the same relationship
`PHASE1_DECISIONS.md` has to the Phase-1 plan: the plan records *what* was
specified, this records what the implementation decided, where it departed, and
what the build found that reading did not.

Read this before touching the generator, `Target`, or the `ActionPolicy`
protocol.

---

## 1. The plan's §6 decisions — all 24 implemented as written

Every row of the normative table is in the code, and each is anchored by a test
that names it. Nothing in §6 was overridden. The non-obvious mappings:

| Decision | What it says | Where it lives |
|---|---|---|
| #3 | one forward DP over a precomputed graph | `exact.forward_mass`, reused by `policy_distribution`, the absorption audit and the `Δd` densities — three consumers, one pass |
| #4 | `uint64` bitmask identity | `StateGraph.mask`; `canon_set_hash` survives only in `enumerate.state_fingerprints` |
| #5 | collision audit expecting 0, SA-GFN **not** applied | `audits.collision_audit`, with G3's proof in its docstring and a test asserting the docstring says the correction does not apply |
| #8 | batch-first `action_log_probs(state_ix, graph)` | `policies.ActionPolicy`; a test counts the calls and asserts exactly one for the whole instance |
| #9 | oracle by flow decomposition, `r_fail` split uniformly | `policies.FlowOraclePolicy`; `F(root) == Z` is asserted directly |
| #10 | `at_beta` re-validates `r_fail_margin`; `validate_bands("main")` checks the mass bands | two separate methods on `Target`, with a test that `at_beta` does **not** raise on a β where the band fails |
| #15 | direct dead-end sum authoritative, complement as a ≤ 1e-9 check | `exact.partition_residual`; a test measures the complement's relative error on a full lattice and asserts it is >1000× worse than the direct sum's |
| #16 | conditional early share under `ForcedContinuationPolicy` | `audits.dead_end_absorption` |
| #20 | FCS frozen at `m=8`, 2000 subsets, seed 20260812 | `exact.fcs`, verified against `fcs_exact` **and** against an exact-rational recomputation |
| #21 | two fingerprints, environment excludes β | `enumerate.environment_fingerprint`, `exact.target_fingerprint` |
| #24 | planted structure asserted per instance | `audits.structural_assertions`, enforced at generation time by `band_report` |

---

## 2. Departures from the plan — five, all recorded

### 2.1 `LatticeSpec` gained knobs the plan did not name

The plan specifies the *bands*; it does not specify the levers that hit them.
Five were needed: `n_distractor_nodes`, `n_distractor_edges`, `n_chain_edges`,
`pool_size` and `enforce_neither_mass`. All are in `LatticeSpec.to_dict()` and
therefore inside `environment_fingerprint`, so a knob change cannot be silently
compared across machines.

**Cost to reverse:** re-generate all three suites; every fingerprint moves.

### 2.2 `enforce_neither_mass` is a *generation-time* flag, and main+tuning share it

The plan makes the `neither`-mass band "a hard band, **on the main suite at the
frozen β**", with the probe reported-only and the tuning suite "not scored on
it". That is unambiguous for `Target.validate_bands`, which is what the plan is
describing, and it is implemented exactly so: `validate_bands("main")` raises,
`validate_bands("tuning"|"probe")` reports.

At **generation** time the decision had to be made separately, and it is: main
and tuning are generated with identical settings and differ only in seed; only
the probe is generated distractor-heavy and without the `Δd` and `neither` bands.

**Why.** β is swept on the tuning suite and applied to the main one. If the two
were drawn from different environment families, that transfer would be an
unstated assumption stacked on top of an already-declared one. Making them the
same family is the weaker assumption, and a test asserts their specs are
identical modulo the label.

**Cost to reverse:** the β sweep's transfer argument changes; re-run the sweep.

### 2.3 `forward_mass` was factored out of `policy_distribution`

Not in the plan's surface. Three consumers need the same forward pass —
`p_θ`, dead-end absorption mass, and the visitation-weighted `Δd` density — and
three DPs would have been three chances to disagree about layer order. One
function, three callers, and a test asserting the direct `p_θ(FAIL)` equals
`f[dead].sum()`.

### 2.4 `divergence_report` was added

The plan requires, in three separate places, that FCS be reported alongside exact
TV with its standard error, that KL appear only when finite, and that `p*(FAIL)`
sit beside TV and never be subtracted from it. Those are three reporting rules
that drift apart the moment they live in three call sites. `exact.divergence_report`
is one call that returns all of them together.

### 2.5 `ActionPolicy` is `@runtime_checkable`

So Phase 3 can *test* the layering fix F6 requires — an adapter implements the
protocol, a learner never does — rather than promising it in a docstring.

---

## 3. Decisions the plan left to the implementation

### 3.1 `tiny_instance()` uses `max_atoms = 3`, not 8

The plan says "6 atoms, `p*` computed by hand". With `max_atoms = 8` and 6
atoms, no state can reach the size cap, so the *only* route to a dead end would
be every remaining atom being inadmissible — and the fixture would need extra
blocked atoms to arrange it. At `max_atoms = 3` the dead end is the natural one:
`{nX, nV, bBAD}` is invalid by sub-check 3 and sits exactly at the cap.

17 states, 15 valid terminals, 1 dead end. `k = 15 ≤ 100`, which is what
decision 11 needs for `TV < 0.02` at `N = 200,000` to be a real assertion rather
than a measurement of the sampling floor.

**tiny is exempt from every band** and is not produced by `generate()`. It is a
fixture for the evaluator, not a Gate-2 environment, and conflating the two is
how a fixture generator grows into an early lattice.

### 3.2 The pool is padded with **inadmissible** atoms

Exit criterion 10 wants 20–30 atoms; G1 wants ≤ 1e5 states. Those pull in
opposite directions if every atom is selectable, because free node atoms are what
the enumeration is exponential in.

The generator therefore keeps two atom populations: ~13 admissible atoms and
~7–13 that carry a permanent per-atom violation (a retired edge, a quarantined
assertion) and are pruned by the `ADD` mask at every state. Pool size and state
count become independent knobs.

This is not a trick to satisfy a counter. A real Stage-C pool contains
candidates the checker rejects, and the retired/quarantined atoms are exactly the
negative cases criterion 17 requires — present in the snapshot, asserted
unreachable.

### 3.3 The independent closure enumeration

Criterion 12 wants the unconstructible-valid-terminal rate. Measuring it with
`legal_adds` would make the audit agree with the thing it audits, so
`audits.closure_audit` walks the pool's `refs` directly in topological order and
calls **batch `H`**. Topological order is needed because the enumeration extends
subsets in index order and an atom's references must have smaller index, or
closed subsets would be skipped.

It enumerates only admissible atoms, which is a proof rather than a shortcut: a
per-atom violation makes `H` fail for any set containing that atom. The audit
checks that claim rather than assuming it (`blocked_atoms_passing_H` must be
empty).

Measured: closed subsets == reachable states, exactly, on every instance. The
rate is 0 because it cannot be anything else, which is what fix F10 promised.

### 3.4 `P_S` is sampled exactly, not by rejection

Corollary 1's `P_S(S; m) ∝ Σ_{x∈S} p_T(x)` normalises to `C(n−1, m−1)` because
`Σ_x p_T(x) = 1`. So drawing one distinguished element `x ~ p_T` and then `m−1`
others uniformly from the remaining `n−1` yields exactly `P_S` — no rejection, no
importance weights. The `m−1` others come from the `m−1` smallest of `n−1` i.i.d.
uniforms, which is a uniformly random `(m−1)`-subset.

Verified three ways: against `fcs_exact` (exhaustive over all 20 3-subsets),
against an exact-**rational** recomputation with `fractions.Fraction`, and by the
2,000-draw estimator landing within 4 SE of the literal.

### 3.5 `js` is in nats; `kl` returns `inf` rather than raising

TV and JS are bounded and always reported. KL is reported only when finite, and
the guard is exercised by a deterministic policy in the tests rather than assumed
away.

---

## 4. What the build found that reading did not

### 4.1 The `neither`-mass band is the binding constraint, and near-misses drive it

The plan's G10 estimate — 4,000 junk terminals against ten gold-like ones, ~23%
junk mass — assumed the competition was *distractor* sets. It is not. Measured on
the first working generator: mass by `|gold ∩ X|` was 0.23 at 5/5, **0.35 at
4/5** and 0.27 at 3/5. The `neither` bucket sat at 0.74, and almost none of it
was junk.

The mechanism: `sufficiency` is `|gold ∩ X| / |gold|`, so with `|P_A| = 5`,
`β = 4` and `w_suff = 1`, dropping one gold atom costs a factor of only
`exp(0.8) ≈ 2.2`, while the number of ways to drop one and refill the freed slot
grows by ~5×. Multiplicity beats the reward gap.

Three structural changes fixed it, each of which also makes the environment more
faithful rather than less:

1. **The two chains share the Value node and the validity window.** "Two
   substitutable claim chains reaching the same answer" is most naturally read
   as two *claims* about the same entity, value and period. It also leaves the
   requested value type and the requested window with **unique providers**, so
   dropping either costs a coverage slot outright instead of being repaired by
   the other chain's copy. Overlap is 3/7 = 0.43, inside decision 23's ≤ 0.5.
2. **Feature clusters are assigned, not sequential.** The atoms of each designed
   proof get distinct clusters and distractors reuse them, so complementary
   evidence scores `redundancy` near 0 and interchangeable distractors score
   high. Sequential assignment had made `redundancy` a function of pool position.
3. **The chain-edge count became a knob.** An edge atom addresses no obligation
   slot, so every `ADD` of one has `Δd = 0` *and* it is one more way to build a
   near-miss. Cutting them moves both bands the same way, which is why the `Δd`
   band and the `neither` band are satisfiable together at all.

Final: `neither` 0.46–0.50 on the main suite at β = 4, with ~75% of generation
attempts accepted.

**This band is genuinely tight, and that is a finding, not a defect.** It is
recorded here so a future β change is not made casually — see §4.2.

### 4.2 The `neither` band fails at half the predeclared β grid — resolve it *before* Phase 3 trains

**[ANALYSIS]** Measured across **all twenty** main-suite instances, not one:

| β | `neither` mass, 20 instances | instances failing the ≤ 0.5 band |
|---|---|---|
| β = 1 | 0.779 – 0.806 | **20 / 20** |
| β = 2 | 0.676 – 0.709 | **20 / 20** |
| β = 4 | 0.463 – 0.499 | 0 / 20 |
| β = 8 | 0.165 – 0.196 | 0 / 20 |

It is not marginal and it is not instance-dependent. Half the predeclared
candidate set of decision 19 is categorically unusable with this suite.

**The scheduling constraint, which an earlier version of this section got
wrong.** Decision 22 selects β by *running L5 across the tuning suite at a fixed
trajectory budget over three seeds* — those are learner results. §6b's amendment
procedure, point 6, says a relaxation made after inspecting any learner result is
contaminated, leaving only two honest exits: keep the failing band and report
that the environment could not be built, or restart the sweep. So "raise this at
the start of the sweep" was too weak. **The amendment has to be decided and
recorded before Phase 3 trains anything.**

Three options, and they are not equally viable:

1. **Amend the candidate set to `{4, 8}` before any sweep runs.** Feasible now
   and uncontaminated. It is itself an amendment to decision 19 and needs the
   §6b paperwork, but the recorded reason must be *"the instrument cannot
   resolve at β ∈ {1, 2}"* — a property of the environment measured before any
   learner existed — and never *"those β looked bad"*. **[EVIDENCE]** *The
   Hitchhiker's Guide to Testing Statistical Significance in Natural Language
   Processing* (ACL 2018) is the project's standing authority for a predeclared
   decision rule, and it is the reason the distinction between those two
   sentences is not pedantry.
2. **Regenerate the instrument so all four candidates pass.** This looks
   *infeasible*, not merely expensive, and saying otherwise would be dishonest.
   At β = 1 a missing gold atom costs a factor of `exp(1 · w_suff/|P_A|) =
   exp(0.2) ≈ 1.22`, against roughly 5× multiplicity in the ways to drop one gold
   atom and refill the freed slot — §4.1's arithmetic with β set four times
   lower. Clearing 0.5 there needs a lattice near the 200-terminal floor with
   almost no near-misses, which trades away the discriminating power the band
   exists to protect.
3. **Keep the grid and accept the outcome.** If 1 or 2 wins, §6b's second exit
   applies: report that the environment could not be built at that β. Honest,
   but it spends a full sweep to learn something already measured here.

**Recommendation: option 1, decided and written before Phase 3 starts.** The
justification is G10's own: at β = 1 roughly four fifths of the target mass sits
on terminals that complete no designed proof, so exact TV would be substantially
measuring whether learners can match a near-uniform tail — which is precisely the
failure the band was introduced to prevent, and it makes Gate 2 unable to resolve
what it is pointed at.

**The β = 4 margin is thin, and that belongs here too.** The main suite spans
0.463–0.499 against a hard band of 0.5: one instance clears it by **0.001**. A
weight change, a feature change, or a NumPy release that moves the `default_rng`
stream could push an instance over and force a regeneration for no scientific
reason. The generator was shaped until it cleared the band (§4.1) and was not
shaped to leave headroom. Two ways out, both with costs: record the fragility and
accept a possible regeneration, or move the generator to leave margin — which is
a change to a frozen instrument and carries the full §6b price. **Not decided
here**; it needs the same pre-Phase-3 timing as the grid question.

### 4.3 The alternative proof mode holds 6–10% of target mass

G10's declared risk was that mode B might fall below 1%, forcing the write-up to
narrow to "effectively unimodal". It does not: 0.079–0.101 on the main suite.
**The Gate-2 environment is genuinely bimodal** and the write-up may say so.

That is a consequence of §4.1's shared-value/shared-window change — with
`sufficiency(P_B) = 3/5` rather than `1/5`, mode B's per-terminal reward is
`exp(−4·0.4) ≈ 0.2` of mode A's rather than `exp(−4·0.8) ≈ 0.04`.

### 4.4 Dead ends occur *only* at the size cap

`|d| = max_atoms` on every instance, in every suite. The conditional early share
is **0.0**, against a band of ≤ 0.05. Phase 1 measured the same shape on its
fixtures (110 of 480 rollouts, every one at `max_atoms`), so this now holds on the
lattice too.

The reading Phase 1 asked for: the `ADD` masks are **not** too tight. Every dead
end is budget exhaustion, which is the benign cause.

### 4.5 The complement's error on `p_θ(FAIL)` is what the plan said — but only at scale

Decision 15 quotes ~2e-4 relative error for the complement. That figure is for
`p*(FAIL) ≈ 2.5e-12`. On `tiny_instance()`, `p*(FAIL) ≈ 2.4e-9` — three orders
larger — and the complement is accurate to ~8e-8 relative, which looks fine. A
test written against tiny would have proved nothing.

The test therefore lives on a full benchmark lattice, where the measured
complement error is >1e-6 relative and >1000× the direct sum's. **The direct sum
is right for the reason stated; the demonstration just has to be run at the right
scale.**

### 4.6 One test literal was fabricated, and caught

The FCS reference at `m = 3` was first written as `0.472855421686747` — digits
invented past the six that had actually been printed. The true value is
`0.4728548372381662`, confirmed by an independent exact-rational computation that
is now itself a test.

This is `CLAUDE.md` §5's pattern — "asserting math without checking it" —
reappearing in a test file. Recorded because the mechanism that caught it (a
second, independent derivation of every literal) is worth keeping.

### 4.7 `_similarity_matrix` had to return a read-only array

The G7 cache is keyed on pool identity, so the matrix is now *shared* between
every `redundancy` call on that pool. A caller mutating it would silently change
every later reward. `arr.flags.writeable = False`, with a test.

---

## 5. Post-build review — three fixes and one scheduling finding

A review after the build went green found four defects the 475-test suite did not
catch. All are **[ANALYSIS]** — engineering errors with engineering fixes; none
touches a claim any paper supports, and none invalidates a Phase-2 measurement.
The enumerator, the DP, the flow oracle and the audits were unaffected.

### 5.1 `environment_fingerprint` was not β-independent — **fixed**

Decision 21 requires the environment digest to exclude β. `LatticeSpec.to_dict()`
embedded the whole resolved `Config`, which carries `beta`, `u_weights`,
`r_fail` and `r_fail_margin` alongside the caps. Reproduced: identical pool,
identical obligations, identical snapshot, **identical enumerated-graph digest**
— change only β and the environment digest moved.

The consequence is exactly the one decision 21 names. `synthetic.yaml` pins
β = 4; the moment Phase 3 freezes a different value, every suite's environment
fingerprint would move although the environments are byte-identical, so "frozen
at Gate 0" would mean nothing — and a genuine structural divergence between two
machines could be dismissed as "β differs".

**Fix.** `ENVIRONMENT_CONFIG_FIELDS = ("profile", "pool_cap", "max_atoms")` — the
only config fields that change what is *built and enumerated*. The reward fields
stay bound through `target_fingerprint`, which carries `(β, u_weights, r_fail)`
explicitly and hashes the **computed** `p*`; that last part is what also binds
`source_tiers`, which decision 21 names in neither list. `r_fail_margin` is
deliberately in neither digest: it constrains which `(β, r_fail)` pairs the
loader accepts, not what any number is.

**The test that should have caught it could not.** It asserted
`environment_fingerprint(tiny, tiny_graph) == env` where `env` came from the same
two arguments three lines earlier — an assertion with no way to fail.
`Target.at_beta` builds a new `Config` on the *target* and never touches
`instance.spec.cfg`, so a test routed through it cannot see this class of leak.
Replaced by one that rebuilds the spec's `Config`, asserts the graph digest is
unchanged first (so it is testing one thing), and is parametrised over `beta`,
`r_fail` and `checker_budget`. A companion test raises `pool_cap` — which changes
neither the pool nor the enumeration — to confirm the narrowed payload is not
empty.

**Consequence for anyone holding earlier digests:** every `environment_fingerprint`
value printed before this fix is superseded. Nothing was frozen, so nothing is
lost.

### 5.2 `Target.validate_bands` failed open on an unrecognised scope — **fixed**

It gated on `scope == "main"` and returned the profile for anything else, so
`validate_bands("mian")`, `"MAIN"` or `""` silently skipped the hard band. In a
codebase where `Assertion.eligibility` defaults to `quarantined` and `H` rejects
an unresolvable provenance chain rather than passing it, this was the one
fail-open step on the path — and it sits on the call Phase 3's β sweep makes.
Now `ValueError` on anything outside `SCOPES`, with a test tying `SCOPES` to
`lattice.SUITE_SIZES` (an import would be a cycle).

For the record, the generation path was already guarded: `lattice._spec_for`
raises on an unknown scope. Only the target-side call was open.

### 5.3 The source-tier assertion was weaker than criterion 17 — **fixed**

Criterion 17 asks for ">= 2 distinct source tiers **among atoms whose `Source`
resolves**". The audit counted distinct `source_tier` *scores* over the whole
pool — and `source_tier` returns `default_tier` when nothing resolves, while
`unknown` is itself a tier carrying that same 0.25. So one resolved tier plus a
pile of defaulted atoms satisfied a criterion asking for two resolved ones.

Measured: on the real suites 3 tiers genuinely resolve and 6 atoms of ~24 fall
back to the default (Entity, TimeInterval and binding atoms have no
`asserted_by` path), so **no reported number was wrong** — the guard was simply
weaker than advertised and would have gone on passing if the generator ever
stopped varying tiers.

**Fix.** `resolve.source_node(atom, G)` was extracted from `resolve.source_tier`,
which now calls it — behaviour-preserving, and it puts "did the source resolve?"
and "what is it worth?" in separate functions instead of hiding the first inside
the second. The audit counts tier **keys** among atoms with a resolving source
and reports `atoms_without_resolved_source` beside it. The regression test builds
a snapshot with exactly one `Source` left live and asserts both that the retired
score-counting logic would still have passed and that the new one rejects.

### 5.4 The β grid must be resolved before Phase 3 trains — **not a code change**

See §4.2. Recorded here because it was found in the same review and because its
deadline is earlier than it looks: after the sweep runs it is too late to amend
without contamination.

---

## 6. Measured numbers, for the record

Main suite, 20 instances, seed `20260808`, β = 4:

| Quantity | Band | Measured |
|---|---|---|
| valid terminals | 200–5,000 | 353–552 |
| reachable closed states | ≤ 1e5 | 539–838 |
| state-graph edges | ≤ 2e6 | 2,028–3,157 |
| pool size | 20–30 | 20–26 |
| zero-`Δd`, structural | ≤ 0.6 | 0.471–0.504 |
| zero-`Δd`, visitation-weighted | ≤ 0.6 | 0.443–0.467 |
| distinct `d` values | ≥ 10 | 32–40 |
| sizes at which `d` varies | ≥ 3 | 8 |
| dead-end early share | ≤ 0.05 | 0.0 |
| `neither` mass | ≤ 0.5 | 0.463–**0.499** — one instance clears by 0.001; see §4.2 |
| mode-B mass | ≥ 1% (diagnostic) | 0.079–0.101 |
| template overlap | ≤ 0.5 | 0.429 |
| generation attempts | ≤ 200 | 1–3 (29 for 20 instances) |

Probe suite (5, seed `20260809`, distractor-heavy, no `Δd` band): terminals
3,141–3,685; zero-`Δd` structural 0.568–0.581; `neither` 0.642–0.677 — sparser
signal and a heavier tail, which is the point of it.

Tuning suite (5, seed `20260811`): same profile as main.

Budgets, on one RTX-5090 workstation:

| Budget | Ceiling | Measured |
|---|---|---|
| one exact evaluation (DP only) | ≤ 0.15 s | ~0.0002 s |
| FCS per instance | ≤ 0.05 s | ~0.049 s |
| enumeration + `Target` per instance | ≤ 60 s | ~0.07 s |
| all three suites, including rejections | ≤ 30 min | ~4.6 s |
| main-suite resident memory | ≤ 2 GB | ~2 MB |

The DP is ~750× inside its ceiling, so the G2 estimate of 0.29 h for Gate-2
evaluation is very conservative at this state count. **FCS, not the DP, is the
per-evaluation cost**, and it is at 98% of its own budget — if `n_subsets` or `m`
ever rises, that budget is what moves first.

---

## 7. Handoff to Phase 3

```python
from graft.synth.lattice   import LatticeInstance, LatticeSpec, benchmark_suite, generate, probe_suite, tiny_instance, tuning_suite
from graft.synth.enumerate import StateGraph, reachable_states, valid_terminals
from graft.synth.exact     import Target, fcs, policy_distribution, target_distribution, tv, js, kl
from graft.synth.policies  import ActionPolicy, FlowOraclePolicy, UniformPolicy
from graft.synth.audits    import run_audits
```

Verified by a test, so it works on the day Phase 2 exits. Two more that Phase 3
will want and the plan's §8 list predates:

```python
from graft.synth.exact     import divergence_report, forward_mass
from graft.synth.policies  import ForcedContinuationPolicy, uniform_backward
```

### What Phase 2 requires of Phase 3

Unchanged from the plan's §8, with one addition:

1. **An adapter implements `ActionPolicy`; the learners keep the F6 tensor
   interface.** No learner may take a `frozenset` of atom ids or a `StateGraph`.
   The protocol is `runtime_checkable`, so this is testable.
2. **`C = 50` exact-evaluation checkpoints per run.** Frozen; exit criterion 11's
   budget is derived from it.
3. **The backward policy is uniform over *removable* atoms.** `uniform_backward`
   reads them off the enumerated in-edges, which are exactly the removable atoms.
4. **The β sweep uses `Target.at_beta`**, on the tuning suite, and the frozen β is
   re-validated with `validate_bands("main")`. **Settle the candidate set before
   training anything** — §4.2 measures 20/20 main-suite failures at β ∈ {1, 2},
   and §6b's amendment procedure forbids relaxing a band after any learner result
   has been inspected. Once the sweep has run it is too late to amend without
   contamination.
5. **Exact TV converges to 0.** `p*(FAIL)` is a diagnostic beside it, never a
   floor. `divergence_report` returns them together so the rule cannot drift.
6. **The Gate-2 decision rule is predeclared and unchanged.**
7. **Every Gate-2 table carries the environment's `d`-density (both) and the
   target-mass profile.** The numbers are in §5 and in `instance.meta["bands"]`.
8. **New:** every Gate-2 table also carries the **`neither`-mass at the β it was
   run at**. §4.2 shows that number moving from 0.46 to 0.78 across the candidate
   grid, and a reader cannot interpret a TV without it.

---

## 8. Still open

* **Gate-0 data contract** (v1.2 §7 items 1–10) — unwritten, blocks Phases 5–9,
  not Phases 3–4.
* **Three late-found baselines** — HyperMem, Chain-of-Memory, *How Memory
  Management Impacts LLM Agents* — in the library, not in plan §5.3.
* **β remains a placeholder** until the Phase-3 sweep, and §4.2 is now a known
  hazard attached to it.
* **No reader-size experiment** exists, though the minimality claim's scope
  condition rests on a provisional single-author preprint.
