# GRAFT — Phase 2 Build Plan: the enumerable synthetic environment (`graft/synth/`)

**The ProofLattice, the exact evaluator, and the audits Gate 2 consumes.**

Date: 8 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 2) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §6.4) · `GRAFT_PHASE1_BUILD.md` §8 · `PHASE1_DECISIONS.md` §7
Effort: ~1.5–2 weeks solo
Status: ready to code once §6 is signed off

Labels inherited: **[EVIDENCE]** (named paper) · **[HYPOTHESIS]** (project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps found while making this phase concrete are numbered **G1–G9** and are
referenced from code as "Phase-2 gap G*n*", matching the Phase-0/1 convention.

---

## 0. What Phase 2 is for, and what it is not

**[EVIDENCE]** Exact evaluation on an enumerable space is the standard instrument
in the GFlowNet training literature: Shen et al. (ICML 2023) evaluate on
enumerable spaces, and *When Do GFlowNets Learn the Right Distribution?*
(ICLR 2025, Spotlight) builds correctness metrics precisely because sampled
proxies mislead.

This is **the only place in the project where the learned terminal distribution
can be compared to its declared target exactly.** Everywhere else, a distribution
claim is a sampled estimate.

It is therefore the Gate-2 harness, and Gate 2 is where the project is designed
to learn cheaply that Contribution 3 does not hold. That framing drives every
decision below: the lattice is not a toy dataset, it is a **measuring
instrument**, and an instrument that cannot resolve the thing it is pointed at is
worse than none — it produces a number.

| Phase | Blocked on Phase 2 by |
|---|---|
| 3 (7 learners) | exact TV is the **primary** Gate-2 criterion; `p*` and `p_θ` both live here. The β sweep runs on this lattice. |
| 4 (5 search algos) | S1–S5 run against these pools with exact `U` as the scorer (fix F13). |
| 9 (Stage D real) | the featurizer swap is validated by re-running the same learners; without a known-good environment there is no control. |

**Not in Phase 2:** any learner, any policy network, any training loop, any real
data, the annotation spike (Phase 2.5). Phase 2 builds the environment and the
ruler; Phase 3 brings something to measure.

---

## 1. Nine specification gaps Phase 2 must close

### G1 — Nothing controls the size of the enumeration [ANALYSIS]

The architecture says "a universe of 20–30 typed atoms … total valid terminals
≤ ~5,000 per instance". With `pool_cap = 32` and `max_atoms = 8` there are
**15,033,173** subsets of size ≤ 8. The target is 5,000 valid terminals — a
factor of ~3,000 apart. Closure and `H` do that pruning, but *how much* they
prune is a consequence of the reference structure the generator happens to emit,
and nothing in the spec makes it a controlled quantity.

Both failure directions are real:

* **too few valid terminals** — TV is measured over a trivial support and Gate 2
  passes or fails on noise;
* **too many** — the enumeration and the DP stop being affordable, and the β
  sweep (which re-derives `p*` at each β) becomes the bottleneck of Phase 3.

**Decision.** The generator targets a **declared band on two quantities**, and
verifies them by construction rather than hoping:

| Quantity | Band | Why this one |
|---|---|---|
| valid terminals `k` | 200 ≤ k ≤ 5,000 | below 200 the TV estimate is dominated by a handful of modes; above 5,000 the DP and the MC cross-check get expensive |
| reachable closed states | ≤ 2 × 10⁵ | this, not the terminal count, is what the DP actually costs (G2) |

Instances are generated, enumerated, and **rejected if out of band**, with the
seed and the rejection count recorded. Enumeration is needed anyway, so the
rejection loop is nearly free. An instance family that cannot hit the band after
a declared number of attempts is a generator defect and raises.

### G2 — The per-terminal DP is the wrong algorithm [ANALYSIS]

The architecture specifies "`p_θ(X)` is dynamic programming over the closed
sub-lattice of `X`, at most `O(2^{|X|}·|X|)` per terminal."

That is correct but it is per-terminal, and it repeats almost all of its work:
the forward reachability mass `f(S)` is a function of the **state**, and states
are shared across terminals. One forward pass over the reachable closed
sub-lattice yields **every** terminal's probability at once.

```
f(∅) = 1
for S in states, in increasing |S|:
    for a in legal_adds(S):        f(S ∪ {a}) += f(S) · P_F(a | S)
p_θ(X) = f(X) · P_F(STOP | X)      for every valid terminal X
p_θ(FAIL) = 1 − Σ_valid p_θ(X)
```

Cost is `|reachable closed states| × branching`, once — not
`|terminals| × 2^{|X|} × |X|`. That is why G1 bands the **state** count rather
than the terminal count: the state count is what is actually being paid for.

Correctness of the complement: every trajectory adds one atom per step and is
capped at `max_atoms`, so every trajectory terminates; under the masks it
terminates at a valid `STOP` or at a dead end, which is `FAIL`. So the two
outcomes partition the mass and `p_θ(FAIL)` needs no path enumeration — which is
exactly what architecture fix F3 requires.

### G3 — Equivalent-action collisions are impossible here, and the audit should say so [ANALYSIS]

v1.2 §3.4 and Phase 0's `ids.py` both treat the equivalent-action collision rate
as a **measurement**, with the Symmetry-Aware GFlowNets correction on standby.
On this action space it is not a measurement. It is a theorem.

> For a state `S` and distinct legal actions `a ≠ b`, the children are `S ∪ {a}`
> and `S ∪ {b}`. Since `a ∉ S` and `b ∉ S`, these sets differ, and
> `canon_set_hash` is injective on sets of ids. Two distinct legal actions
> therefore never produce the same child. ∎

Two atoms could only be the same action by sharing an `atom_id`, which
`AtomPool` rejects; and two atoms with identical content necessarily share an id,
because `atom_id` hashes exactly the fields `content_key` compares (Phase-1 gap
G2). `H`'s sub-check 2 remains as a guard against externally supplied pools.

**Decision.** The collision audit still runs — as a **regression test with a
known answer of 0**, in the same class as the unconstructible-valid-terminal rate
under fix F10. A non-zero result means the pool is malformed, not that a
correction is needed.

**What this changes in the write-up.** **[EVIDENCE]** Symmetry-Aware GFlowNets
(ICML 2025) measured L₁ ≈ 0.12 uncorrected versus ≈ 0.01 corrected — on a state
space quotiented by graph isomorphism. Ours is over *labelled* sets and is not
quotiented, so the correction does not apply. The honest sentence is "our action
space admits no equivalent actions by construction; we verify this" — **not** a
citation implying the correction was applied. Citing a correction one did not
need, as though one had, is the overreaching pattern `CLAUDE.md` §5 catalogues.

This also **removes** the SA-GFN correction from the Tier-1 workload.

### G4 — `FAIL` is unreachable unless the generator plants the two things that can fail [ANALYSIS]

Under mask-respecting construction, per-atom violations never enter a state — the
`ADD` mask excludes them (Phase-1 §2.5). So a *reachable* state can only be
invalid through a **set-level** check, and of the five, three cannot fire:
closure is maintained by the mask, identity is impossible with content-derived
ids, and size only bites at the cap.

**Exactly two ways remain**, and both must be planted deliberately:

1. **two bindings claiming one slot** (sub-check 9);
2. **a binding whose bound claim's validity is disjoint from the time
   constraint** (sub-check 3).

Phase 1 confirmed the shape empirically: with `p_stop = 0`, 110 of 480
mask-respecting rollouts reached `FAIL`, **every one at `|X| = max_atoms`**.

**Decision.** The generator plants both, and the FAIL-reachability audit is a
build-time requirement rather than an observation. If `FAIL` were unreachable,
`STOP`-masking would be doing nothing, the `FAIL` terminal would be decorative,
and fix F3's whole construction would be untested.

**The `p*(FAIL)` floor, stated once.** `p*(FAIL) = r_fail / Z`. With ~1,000 valid
terminals at `R ≈ e^{4·1.5} ≈ 400`, `Z ≈ 4 × 10⁵` and `p*(FAIL) ≈ 2.5 × 10⁻¹²`.
So a policy that never reaches `FAIL` carries a TV error of that size —
negligible, but it means **exact TV has a non-zero floor** and "TV → 0" should be
written as "TV → p*(FAIL)". Report the floor alongside the TV.

### G5 — `d(s)` must be informative on the lattice, or Gate 2 measures nothing [ANALYSIS]

**This is the most important gap in the phase.**

Gate 2's decision rule is: L7 beats capacity-matched L6 on exact TV at a fixed
training budget. L7 *is* L6 plus `Δd` as input features, and nothing else
(v1.2 §4.5.4). So the entire discriminating power of Gate 2 rests on `Δd`
carrying information on this environment. If it does not, Gate 2 returns a null
result that says nothing about Contribution 3 — and a null result that cannot
distinguish "the hypothesis is false" from "the instrument could not resolve it"
is the worst outcome available, because it looks like an answer.

Two degeneracies to rule out, measured on Phase-1 fixtures as a rough proxy
(the lattice does not exist yet, so these are indicative, not results):

**(i) Is `d(s)` determined by `|s|`?** If it were, conditioning on `Δd` would be
conditioning on a re-parameterised step counter, and L7 would differ from L6 by
nothing of substance. On the fixtures it is not — 7 to 29 distinct `d` values per
set size:

| `\|s\|` | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| distinct `d` | 7 | 18 | 25 | 26 | 18 | 8 |

**(ii) How often is `Δd` non-zero?** On the fixtures, **67% of transitions have
`Δd = 0`** — most adds discharge no obligation. That is realistic, and it is also
a ceiling on how much L7's extra input can possibly help: its signal is silent on
two transitions in three.

**Decision.** Both become **declared, banded, audited properties of the
lattice**, not observations after the fact:

| Property | Requirement |
|---|---|
| `d(s)` determined by `\|s\|` | must be **false**, at ≥ 3 distinct set sizes |
| fraction of transitions with `Δd = 0` | **≤ 0.6**, reported per instance |
| distinct `d` values reachable | ≥ 10 per instance |

If the generator cannot meet these, that is a finding about the environment to
report **before** three weeks go into seven learners — which is exactly what
building Phases 1–4 before the data pipeline is for.

### G6 — The exact evaluator needs a policy interface Phase 3 has not defined [ANALYSIS]

The DP consumes `P_F(a|s)` and `P_F(STOP|s)` for every reachable state. Phase 3's
frozen interface (fix F6) is `policy(state_repr, action_reprs) → logits` — a
*tensor* interface, one level below what the evaluator needs, and it does not
exist yet.

**Decision.** Phase 2 defines and freezes the evaluation-side protocol:

```python
class ActionPolicy(Protocol):
    def action_log_probs(self, state: IncrementalChecker) -> tuple[np.ndarray, float]:
        """(log P_F(ADD a_i | s) over pool.ids(), log P_F(STOP | s)).

        -inf on illegal actions; the finite entries sum to 1 in probability.
        """
```

Phase 2 ships `UniformPolicy` (uniform over legal actions, including `STOP` when
allowed) and `TemperedOraclePolicy` (softmax over exact terminal reward, for
sanity-checking that the evaluator can recover a known distribution). Phase 3's
learners adapt to this protocol; the evaluator never learns what is behind it.

**Freezing it here is the point.** If the evaluator were written against Phase
3's tensors, Gate 2 would only be able to evaluate Phase-3 learners — and the
uniform-policy exit criterion, which is what proves the evaluator itself correct,
would have nothing to run on.

### G7 — Two recomputations that make the β sweep and the enumeration expensive [ANALYSIS]

**`U` per terminal, not `R`.** `p*` depends on β only through `R = exp(β·U)`, and
`U(X)` is independent of β. Caching `U` for every valid terminal makes each
additional β in the Phase-3 sweep an `exp` over a vector — effectively free.
Caching `R` instead would re-derive `U` for 5,000 terminals at every β.

**The pool similarity matrix, per instance.** `utility.redundancy` calls
`_similarity_matrix(pool)` on every invocation, rebuilding a 32 × 32 matrix for
each of ~5,000 terminals. `AtomPool` defines no `__eq__`, so it is hashable by
identity and an `lru_cache` keyed on the pool works directly. **This is a
one-line Phase-1 amendment**, done as Phase 2 step 1 rather than left as a
30-million-flop tax per instance.

### G8 — The DP-versus-Monte-Carlo tolerance is unachievable as stated [ANALYSIS]

Phase 2's exit criterion says "for a uniform-random policy the DP result matches
a Monte-Carlo estimate within tolerance". No tolerance is given, and the obvious
one does not exist.

For an `N`-sample empirical over `k` outcomes, `E[TV] ≈ sqrt(k / (2πN))`:

| terminals `k` | N = 200,000 | TV floor |
|---|---|---|
| 50 | | 0.006 |
| 500 | | 0.020 |
| 5,000 | | **0.063** |

Asserting `TV < 0.01` on a 5,000-terminal lattice would need ~8 million
rollouts. The sampling floor, not the DP, would be what the test measures.

**Decision — split the two checks by instrument:**

* **DP correctness** is verified on a **small instance** (`k ≤ 100`), where
  `N = 200,000` gives a floor near 0.006 and `TV(p_DP, p_MC) < 0.02` is a real
  assertion. Plus a per-terminal z-test at 5σ.
* **On a full-size lattice**, assert the properties that do not depend on
  sampling: `Σ_valid p_θ + p_θ(FAIL) = 1` to floating-point tolerance, `p_θ ≥ 0`,
  and agreement with MC on the **top-20 highest-mass terminals only**, where per
  terminal `p` is large enough for the z-test to bite.

**Divergences reported.** TV and JS always (both bounded). KL only when finite,
with the guard stated: `KL(p*‖p_θ) = ∞` whenever `p_θ(X) = 0` for a valid `X`,
which a deterministic policy produces. Under fix F10 every valid terminal is
constructible, so a softmax policy keeps KL finite — but the guard is reported
rather than assumed.

### G9 — The Gate-2 benchmark must be frozen and content-hashed [ANALYSIS]

Gate 2 compares seven learners × three seeds. If each run generates its own
instances, the learners are compared on different environments and the
comparison measures instance luck.

**Decision.** A fixed **benchmark suite** of instances, generated once from
declared seeds, frozen at Gate 0, and carrying a content digest in the same style
as `config_hash` and `EventLog.digest`. `verify_handoff.py` gains a lattice
fingerprint so two machines can confirm they enumerated the same environment
before comparing any number.

Instance count: **20** — enough that per-instance variance averages out, small
enough that a full seven-learner sweep with periodic exact TV stays affordable.
Suite size and seeds are frozen at Gate 0 alongside β.

---

## 2. Scope

**In.** The ProofLattice generator with banded, audited structural properties;
exhaustive closed-subset enumeration; the exact evaluator (`p*`, `p_θ` by one
forward DP, TV/JS/KL); the `ActionPolicy` protocol with two reference policies;
five audits; a fixed hand-checkable instance; the frozen benchmark suite; and the
one-line Phase-1 caching amendment of G7.

**Out.** Any learner or policy network (Phase 3), any search algorithm (Phase 4),
the β sweep itself (Phase 3 — Phase 2 only makes it cheap), real data, the
annotation spike (Phase 2.5). **Phase 2 imports no ML library**; the evaluator is
numpy and dictionaries.

---

## 3. Modules

### P2.0 The Phase-1 amendment (do this first)

`lru_cache` on `utility._similarity_matrix`, keyed on pool identity (G7). One
line, plus a test that a second call on the same pool does no work. Everything
downstream enumerates thousands of terminals against one pool, so this is the
difference between an affordable enumeration and a wasteful one.

### P2.1 `graft/synth/lattice.py`

**Responsibility.** Generate `LatticeInstance`s whose structural properties are
declared rather than emergent.

**Surface.** `LatticeInstance` (pool, obligations, graph, gold, meta) ·
`generate(rng, spec) -> LatticeInstance` · `LatticeSpec` (the banded knobs) ·
`tiny_instance() -> LatticeInstance` · `benchmark_suite(seed, n) -> tuple[LatticeInstance, ...]`.

**Structural content, and which requirement each discharges:**

| Feature | Serves |
|---|---|
| required entity anchor, requested value type | `d_anchor`, `d_value`; coverage |
| dependency structure via `refs` (edges need endpoints, bindings need referents) | the `ADD` mask and the closed-subset lattice |
| **two substitutable claim chains** reaching the same answer | multiple disjoint valid modes — v1.2 §9's first condition for the GFlowNet argument |
| **duplicate-slot binding pairs** | sub-check 9; `FAIL` reachability (G4) |
| **temporally disjoint claims** | sub-check 3; `FAIL` reachability (G4) |
| bounded, **partially overlapping** intervals | `temporal_correctness` graded, not a presence flag (`PHASE1_DECISIONS.md` §4) |
| mixed source tiers | `source_quality` non-degenerate |
| clustered `feat` vectors | `redundancy` non-degenerate |
| ≥ 1 invalidated edge, ≥ 1 quarantined assertion | sub-checks 4 and 7 exercised against a snapshot |
| distractor atoms | admissible but low-utility; makes `U` discriminate |
| backing `DictGraphSnapshot` | Phase-1 gap G2 — `target` must resolve, or four sub-checks ship untested |

**Gotcha.** Instances are built with `AtomPool(atoms, cap=cfg.pool_cap)`, from the
`synthetic` profile (`pool_cap = 32`, `max_atoms = 8`). `AtomPool` accepts
`cap=None` by design (`PHASE1_DECISIONS.md` §5.6), so nothing stops an uncapped
lattice pool except this instruction and the test that checks it.

### P2.2 `graft/synth/enumerate.py`

**Responsibility.** The closed sub-lattice, exhaustively and once.

**Surface.** `reachable_states(instance, cfg) -> StateGraph` ·
`valid_terminals(instance, cfg) -> tuple[ProofSet, ...]` ·
`StateGraph` (states by size, legal actions per state, stop-allowed flag).

**Design notes.** Breadth-first over set sizes, keyed by `canon_set_hash`, using
`IncrementalChecker` for validity and `legal_adds` for successors — so the
enumeration walks exactly the space the policy will, and cannot drift from it.
`H` is called with `ledger=None`: enumerating a lattice would exhaust any
per-query budget, and it is an offline audit rather than a query (Phase-1 gap
G9).

**Gotcha.** The state count is the cost driver (G2) and the thing G1 bands. Build
the counter before the generator, so a runaway instance is caught while
generating rather than after.

### P2.3 `graft/synth/exact.py`

**Responsibility.** `p*`, `p_θ`, and the divergences.

**Surface.** `target_distribution(instance, cfg) -> Target` ·
`policy_distribution(policy, state_graph) -> np.ndarray` ·
`tv(p, q)` · `js(p, q)` · `kl(p, q)` · `Target` (terminals, `U` cache, `Z`,
`p*`, `p_fail_floor`).

**Design notes.** `Target` caches `U` per terminal, not `R` (G7), and exposes
`at_beta(beta)` returning `p*` for a new β without re-deriving `U` — which is
what makes the Phase-3 sweep affordable. One forward DP for `p_θ` (G2);
`p_θ(FAIL)` by complement.

### P2.4 `graft/synth/policies.py`

**Responsibility.** The `ActionPolicy` protocol (G6) and two reference
implementations.

**Surface.** `ActionPolicy` Protocol · `UniformPolicy` · `TemperedOraclePolicy(target, beta)`.

**Design note.** `TemperedOraclePolicy` exists so the evaluator can be checked
against a distribution whose answer is known in advance: a policy constructed to
sample proportionally to reward must produce `TV ≈ 0`. An evaluator that cannot
recover a distribution it was handed is broken in a way a uniform policy would
not reveal.

### P2.5 `graft/synth/audits.py`

**Responsibility.** The five numbers Gate 2 reports, one function each.

| Audit | Expected | Meaning if violated |
|---|---|---|
| unconstructible valid terminals | **0**, by fix F10 | the closure rule or the masks are wrong |
| equivalent-action collisions | **0**, by G3 | the pool is malformed |
| `FAIL` reachability and rate | reachable, low | `STOP`-masking is doing nothing (G4) |
| `d` informativeness | not `\|s\|`-determined; ≤ 0.6 zero-`Δd` | **Gate 2 cannot resolve L7 from L6** (G5) |
| mode structure | ≥ 2 disjoint modes; Jaccard spread | v1.2 §9's "multiple materially different proofs" is false here |

**Gotcha.** These are consumed by a gate, so they are reported *per instance and
aggregated*, never a single pooled number that a bad instance can hide inside.

### P2.6 `graft/tests/`

See §5.

---

## 4. Build order

| Step | Build | Done when |
|---|---|---|
| 1 | P2.0 similarity-matrix cache | second call on one pool is a cache hit |
| 2 | `enumerate.py` + the state counter | counts closed states and valid terminals on a hand-built pool |
| 3 | `tiny_instance()` with a written-out enumeration table | 6 atoms, `p*` computed by hand and asserted as literals |
| 4 | `exact.py`: `Target`, `p*`, divergences | `Σ p* = 1` including `p*(FAIL)` on the tiny instance |
| 5 | `policies.py` + the forward DP for `p_θ` | uniform policy: `Σ p_θ = 1`; DP matches MC on the tiny instance |
| 6 | `lattice.py` generator with band rejection | 20 instances inside both bands of G1 |
| 7 | `audits.py` | all five run; `d`-informativeness inside its band |
| 8 | frozen benchmark suite + lattice digest in `verify_handoff` | two runs produce the same digest |

Steps 3–5 are the correctness spine and should be finished before the generator
gets interesting — an evaluator verified on a hand-checked instance is what makes
every later number trustworthy. Step 6 is the one that can overrun (G1).

---

## 5. Exit criteria

**The evaluator is correct**
1. On `tiny_instance()`: enumerated `p*` matches the hand-computed table literal for literal, and `Σ p* = 1` **including `p*(FAIL)`**.
2. On `tiny_instance()` with `UniformPolicy`: `TV(p_DP, p_MC) < 0.02` at `N = 200,000`, and every terminal within 5σ (G8).
3. On `tiny_instance()` with `TemperedOraclePolicy` at the instance's own β: `TV < 10⁻⁹`. An evaluator that cannot recover a distribution it was handed is broken.
4. On every benchmark instance: `Σ_valid p_θ + p_θ(FAIL) = 1` to 1e-9, and `p_θ ≥ 0` everywhere.
5. `p_θ` is invariant to the order states are visited in the DP.
6. KL is reported only when finite, with the zero-support guard exercised by a deterministic policy.

**The environment is enumerable and controlled**
7. Every benchmark instance has 200 ≤ valid terminals ≤ 5,000 and ≤ 2 × 10⁵ reachable closed states (G1).
8. Enumeration and one full `p_θ` evaluation complete within a declared wall-clock budget per instance, reported (this bounds the Phase-3 sweep).

**The audits**
9. Unconstructible-valid-terminal rate is **0** on every instance (fix F10 as a regression test).
10. Equivalent-action collision rate is **0** on every instance, with the proof of G3 recorded in the module docstring rather than the correction applied.
11. `FAIL` is reachable on every instance, and the `p*(FAIL)` floor is reported alongside every TV.
12. **`d(s)` is not determined by `|s|`** at ≥ 3 distinct sizes, and the zero-`Δd` transition fraction is ≤ 0.6, on every instance (G5).
13. Every instance has ≥ 2 disjoint valid modes; the pairwise-Jaccard distribution over valid terminals is reported.

**Reproducibility**
14. The benchmark suite is deterministic: same seed → identical lattice digest, on a different machine.
15. `graft.synth` imports no ML library.

---

## 6. Decisions to lock before writing code

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | Terminal-count band | 200 ≤ k ≤ 5,000 | Re-generate the suite; every Gate-2 number re-runs |
| 2 | State-count ceiling | 2 × 10⁵ reachable closed states | Same, plus the Phase-3 sweep budget moves |
| 3 | `p_θ` algorithm | one forward DP over the closed sub-lattice (G2) | none — it is an implementation of the same quantity |
| 4 | Collision audit | regression test expecting 0; SA-GFN correction **not** applied, with the reason written down (G3) | a citation that would not survive review |
| 5 | `FAIL` reachability | planted via duplicate-slot bindings and disjoint-temporal bindings (G4) | `STOP`-masking untested; fix F3 undemonstrated |
| 6 | `d` informativeness band | not `\|s\|`-determined; zero-`Δd` ≤ 0.6 (G5) | **Gate 2 becomes unable to resolve Contribution 3** |
| 7 | Policy interface | `action_log_probs(state) -> (log p_add[], log p_stop)` (G6) | Phase 3's seven learners re-wire |
| 8 | Caching | `U` per terminal; similarity matrix per pool (G7) | β sweep cost multiplies by the number of β values |
| 9 | MC agreement | on a `k ≤ 100` instance at `N = 200,000`, `TV < 0.02`; top-20 only at full scale (G8) | a test that measures its own sampling floor |
| 10 | Benchmark suite | 20 instances, frozen seeds, content-hashed (G9) | learners compared on different environments |

**Open question for you.** The `d`-informativeness band of ≤ 0.6 zero-`Δd`
transitions is a **guess calibrated on Phase-1 fixtures, not on the lattice** —
they measured 0.67, just outside it. Two readings, and they lead to different
places:

*The band is a design target.* The generator is shaped until the lattice hits it
— fewer distractors, obligations that more adds can discharge. Gate 2 then tests
L7 in the regime where `Δd` is most informative, which is the **best case** for
Contribution 3. A win there is real but narrow; a loss is decisive.

*The band is a measurement.* Whatever the lattice naturally produces is reported,
and if it is 0.8 then Gate 2's power to resolve L7 is correspondingly low — which
is itself the finding, and arguably the honest one, since real pools will be
distractor-heavy too.

My recommendation is **both, declared separately**: build the lattice to hit ≤
0.6 so Gate 2 has resolving power, *and* report the zero-`Δd` fraction as a
first-class property of the environment, so that a Gate-2 win is stated as "under
a `Δd` density of X" rather than unconditionally. That costs one number in a
table and forecloses the reviewer question "would this hold where the signal is
sparser?", which has no good answer if it was never measured.

---

## 7. Explicitly not in Phase 2

No policy network · no learner · no training loop · no search algorithm · no
β sweep (only the machinery that makes it cheap) · no real data · no annotation ·
no PyTorch. If a `graft/synth/` file imports `torch`, something has gone wrong.

---

## 8. What Phase 3 will need from this, verbatim

```python
from graft.synth.lattice   import LatticeInstance, benchmark_suite, generate, tiny_instance
from graft.synth.enumerate import StateGraph, reachable_states, valid_terminals
from graft.synth.exact     import Target, policy_distribution, target_distribution, tv, js, kl
from graft.synth.policies  import ActionPolicy, UniformPolicy
from graft.synth.audits    import run_audits
```

### Requirements this phase places on Phase 3

1. **Every learner implements `ActionPolicy`.** The evaluator never learns what
   is behind it, which is what lets the same TV machinery score L1 through L7.
2. **The β sweep uses `Target.at_beta`**, not regeneration. Re-deriving `U` per β
   is the difference between a two-day sweep and a two-week one.
3. **Exact TV is reported against the `p*(FAIL)` floor**, not against zero.
4. **The Gate-2 decision rule is predeclared and unchanged**: exact TV at a fixed
   number of sampled training trajectories, three seeds, paired bootstrap
   (fix F12). Phase 2 supplies the metric; it does not get to redefine the rule
   after seeing it work.
5. **Every Gate-2 table carries the `d`-density of the environment it was
   measured on** (§6 open question).
