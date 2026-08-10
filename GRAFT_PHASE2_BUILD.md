# GRAFT — Phase 2 Build Plan: the enumerable synthetic environment (`graft/synth/`)

**The ProofLattice, the exact evaluator, and the audits Gate 2 consumes.**

Date: 8 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 2) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §6.4) · `GRAFT_PHASE1_BUILD.md` §8 · `PHASE1_DECISIONS.md` §7
Effort: ~1.5–2 weeks solo
Status: ready to code once §6 is signed off

Labels inherited: **[EVIDENCE]** (named paper) · **[HYPOTHESIS]** (project tests it) · **[ANALYSIS]** (engineering or mathematical judgment made here).

Gaps found while making this phase concrete are numbered **G1–G10** and are
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

## 1. Ten specification gaps Phase 2 must close

### G1 — Nothing controls the size of the enumeration [ANALYSIS]

The architecture says "a universe of 20–30 typed atoms … total valid terminals
≤ ~5,000 per instance". With `pool_cap = 32` and `max_atoms = 8` there are
**15,033,173** subsets of size ≤ 8. The target is 5,000 valid terminals — a
factor of ~3,000 apart. Closure and `H` do that pruning, but *how much* they
prune is a consequence of the reference structure the generator happens to emit,
and nothing in the spec makes it a controlled quantity.

Both failure directions are real:

* **too few valid terminals** — the target becomes small enough to match almost
  trivially, and TV differences between learners are then dominated by
  across-seed training variance rather than by anything about the environment.
  (Not sampling noise: exact TV has none, which is the whole point of
  enumerating. The risk is triviality and variance, which is a different
  argument and was stated wrongly in an earlier draft.);
* **too many** — the enumeration and the DP stop being affordable, and the β
  sweep (which re-derives `p*` at each β) becomes the bottleneck of Phase 3.

**Decision.** The generator targets a **declared band on three quantities**, and
verifies them by construction rather than hoping:

| Quantity | Band | Why this one |
|---|---|---|
| valid terminals `k` | 200 ≤ k ≤ 5,000 | below 200 the TV estimate is dominated by a handful of modes; above 5,000 the DP and the MC cross-check get expensive |
| reachable closed states | ≤ 1 × 10⁵ | this, not the terminal count, is what the DP actually costs (G2) |
| state-graph edges | ≤ 2 × 10⁶ | the evaluation pass is linear in edges, and the graph is held in memory (~24 MB/instance at this bound under the `uint64` state representation of G3, ~500 MB for the suite) |

Instances are generated, enumerated, and **rejected if out of band**, with the
seed and the rejection count recorded.

**Rejection is not free, and an earlier draft claiming it was had no budget
behind it.** At 200 attempts per instance and 60 s of enumeration each, twenty
instances is a 66-hour worst case. Three things bound it:

* **Early abort on the upper bounds.** Enumeration stops the moment the state or
  edge count passes its cap — an oversized instance is rejected in the time it
  takes to exceed the bound, not after a full sweep. **Undersized-terminal
  rejections still pay full enumeration**, and are not necessarily cheap: an
  instance can hold 10⁵ states while `H` leaves fewer than 200 valid terminals,
  so "few terminals" does not imply "small graph". That case is bounded by the
  suite budget below, not by being fast.
* **A total suite-generation budget of ≤ 30 minutes**, reported. Exceeding it is
  a generator that is not controlling its counts, which is a defect to fix rather
  than a wait to endure.
* **At most 200 attempts per instance**, after which it raises rather than
  silently widening the band.

If acceptance turns out to need many attempts, the fix is to make the generator
target the counts directly — chain length and distractor count are the levers —
not to raise the attempt limit.

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
p_θ(X)    = f(X) · P_F(STOP | X)          for every valid terminal X
p_θ(FAIL) = Σ_{dead-end states d} f(d)    # direct; see P2.3 for why not the complement
```

The architecture (§Phase 2.2) writes the last line as
`p_θ(FAIL) = 1 − Σ_valid p_θ(X)`. That is right in general and is the only option
outside an enumerable environment, but here the state graph labels dead ends and
the direct sum is both available and far better conditioned. **This is a declared
numerical refinement of the architecture, not a disagreement with it** — the
complement is retained as the partition check (P2.3, exit criterion 5).

Cost is `|reachable closed states| × branching`, once — not
`|terminals| × 2^{|X|} × |X|`. That is why G1 bands the **state** count rather
than the terminal count: the state count is what is actually being paid for.

Correctness of the complement: every trajectory adds one atom per step and is
capped at `max_atoms`, so every trajectory terminates; under the masks it
terminates at a valid `STOP` or at a dead end, which is `FAIL`. So the two
outcomes partition the mass and `p_θ(FAIL)` needs no path enumeration — which is
exactly what architecture fix F3 requires.

**The state graph is policy-independent, and that is what makes Gate 2
affordable.** States, legal actions and stop-flags depend only on the instance
and the masks; only `P_F` changes between policies. So the graph is enumerated
**once per instance**, stored as integer-indexed `(parent, action, child)`
arrays, and every subsequent evaluation is one numpy pass over that edge list.

The difference is not marginal. Gate 2 runs 7 learners × 3 seeds, with exact TV
at multiple training checkpoints:

| states | checkpoints | instances | pure-Python dict pass | numpy over a precomputed graph |
|---|---|---|---|---|
| 2 × 10⁵ | 50 | 20 | **35 h** | 0.58 h |
| 1 × 10⁵ | 50 | 20 | 17 h | 0.29 h |

Thirty-five hours of evaluation would have been discovered in Phase 3, as an
unexplained schedule overrun. The precomputed graph plus the tightened state band
of G1 puts the *estimate* at 0.29 h; the *budget* is set at ≤ 1 h in exit
criterion 11, from which the per-evaluation ceiling of 0.15 s is derived. Both
numbers are stated because quoting only the estimate is how a per-unit limit ends
up permitting several times the total anyone intended.

**Consequence for the policy interface.** With 10⁵ states per instance, querying
a policy state-by-state is the new bottleneck — 10⁵ Python calls into a network
per evaluation. The interface must therefore be **batch-first** (G6).

### G3 — Equivalent-action collisions are impossible here, and the audit should say so [ANALYSIS]

v1.2 §3.4 and Phase 0's `ids.py` both treat the equivalent-action collision rate
as a **measurement**, with the Symmetry-Aware GFlowNets correction on standby.
On this action space it is not a measurement. It is a theorem.

> For a state `S` and distinct legal actions `a ≠ b`, the children are the **sets**
> `S ∪ {a}` and `S ∪ {b}`. Since `a ∉ S` and `b ∉ S`, these sets differ. Two
> distinct legal actions therefore never produce the same child state. ∎

The statement is about **set equality**, not about hashes. An earlier draft of
this argument appealed to `canon_set_hash` being injective; it is not — it is a
64-bit truncation of SHA-256, so it is injective only with overwhelming
probability. The proof does not need it and must not lean on it.

Two atoms could only be the *same* action by sharing an `atom_id`, which
`AtomPool` rejects. Two atoms with identical content share an id **when ids are
content-derived**, which is a convention `CandidateAtom` does not enforce — it
accepts whatever `atom_id` it is handed. So the honest scope is: *for correctly
formed pools*, the collision rate is 0. `H`'s sub-check 2 stays as the guard for
pools built by hand or supplied externally.

**Decision — two parts.**

*State identity is exact, and cheap: a **bitmask over the pool**.* With
`pool_cap = 32`, a selected set is exactly a `uint64` whose bit *i* marks
`pool.ids()[i]`. That is an exact identity — no hashing, no collisions to reason
about — and it is 8 bytes per state against roughly 700 for a `frozenset` of
short strings, which is the difference between ~0.4 MB and ~70 MB per instance
and therefore between comfortably inside the 2 GB suite ceiling and over it.

`canon_set_hash` is retained as a **fingerprint** for logging and cross-machine
comparison, never as identity. An exact evaluator must not be able to merge two
distinct states because two 64-bit truncations happened to agree, however
unlikely — exactness is the entire point of this phase. A test asserts no two
enumerated states share a fingerprint.

**The bitmask is valid while `pool_cap ≤ 64`.** Beyond that it needs a wider
representation, and the assumption is asserted at `StateGraph` construction so
that a future `pool_cap` change fails loudly rather than silently truncating.

*The collision audit runs as a regression test with a known answer of 0*, in the
same class as the unconstructible-valid-terminal rate under fix F10, and defined
over **exact child-set equality**. A non-zero result means the pool is malformed,
not that a correction is needed.

**This amends a Phase-1 requirement.** `GRAFT_PHASE1_BUILD.md` §8 requirement 3
told Phase 2 that the generator "must be able to emit colliding atoms … otherwise
the collision audit reports 0 having tested nothing". That requirement was
written from the architecture's framing and is **withdrawn**: on this action
space no generator can emit them, and the audit reporting 0 is a proof
discharged, not a test that failed to fire. Amended in place rather than left as
two live documents disagreeing.

**What this changes in the write-up.** **[EVIDENCE]** Symmetry-Aware GFlowNets
(ICML 2025) measured a systematic bias toward low-symmetry objects — 5,220
cyclohexane fragments per 5,000 sampled molecules uncorrected, against 1,042
corrected — on a state space quotiented by graph **isomorphism**, with its
Theorem 4.6 stated over a graph-level policy and ratios of automorphism-group
sizes. Ours is over *labelled* sets and is not quotiented, so the correction does
not apply. (The paper reports no set-generation task at all.) The honest sentence is "our action
space admits no equivalent actions by construction; we verify this" — **not** a
citation implying the correction was applied. Citing a correction one did not
need, as though one had, is the overreaching pattern `CLAUDE.md` §5 catalogues.

This also **removes** the SA-GFN correction from the Tier-1 workload.

### G4 — `FAIL` needs a definition, and it is not "budget exhausted" [ANALYSIS]

The architecture describes `FAIL` as reached "on budget exhaustion with `H = 0`"
(fix F3). Phase 1 found that to be too narrow: a dead end is *any* state with no
legal `ADD` and a masked `STOP`, and its causes include an empty pool at
`|X| = 0`, a pool exhausted below the cap, and every remaining atom failing a
per-atom check — not only reaching `max_atoms`.

**Definition, used from here on:** `FAIL` is reached exactly when **construction
can neither legally continue nor legally stop**. Budget exhaustion is the common
case, not the meaning.

This is not wording. `FAIL` maps to the inference-time abstain fallback, so what
it means determines what an abstention in a Phase-9 `OutputRecord` is asserting.

**A dead end licenses one claim, and it is the modest one:**

> No valid proof was found under this candidate pool, this policy, this attempt
> count and this construction budget.

It does **not** license "no proof exists". A trajectory can dead-end while a
different action sequence from the same root reaches a valid terminal, and `K`
failed attempts are `K` samples, not a proof of non-existence. An earlier draft
of this section had it exactly backwards — it called the non-existence reading
"always true" — which would have licensed a Phase-9 abstention to assert
something the search never established.

The distinction is also where Phase 2 and Phase 9 genuinely differ: **here we
enumerate, so "no valid terminal exists" is a statement we can actually make**;
at inference we sample, and we cannot.

Everything below concerns the sub-case Phase 2 must *engineer*: making `FAIL`
reachable at all on a well-formed lattice.

Under mask-respecting construction, per-atom violations never enter a state — the
`ADD` mask excludes them (Phase-1 §2.5). So a *reachable* state can only be
invalid through a **set-level** check, and of the five, three cannot fire:
closure is maintained by the mask, identity is impossible with content-derived
ids, and size only bites at the cap.

**Two mechanisms remain that the generator must plant deliberately:**

1. **two bindings claiming one slot** (sub-check 9);
2. **a binding whose bound claim's validity is disjoint from the time
   constraint** (sub-check 3).

These are the two *planted* routes to an invalid reachable state — not an
exhaustive account of invalidity, and not the only causes of a dead end. The
empty root is a third invalid reachable state, by design (G1), which is why
`stop_allowed(root)` is `False`. And dead ends have further causes that Phase 1
enumerated: an empty pool, a pool exhausted below the cap, and every remaining
atom failing a per-atom check. What is true is that without these two, no
*mask-respecting* trajectory past the root could ever become invalid, so `FAIL`
would be unreachable and `STOP`-masking would be doing nothing.

Phase 1 confirmed the shape empirically: with `p_stop = 0`, 110 of 480
mask-respecting rollouts reached `FAIL`, **every one at `|X| = max_atoms`**.

**Decision.** The generator plants both, and the FAIL-reachability audit is a
build-time requirement rather than an observation. If `FAIL` were unreachable,
`STOP`-masking would be doing nothing, the `FAIL` terminal would be decorative,
and fix F3's whole construction would be untested.

**`p*(FAIL)` is not a TV floor.** `p*(FAIL) = r_fail / Z`; with ~1,000 valid
terminals at `R ≈ e^{4·1.5} ≈ 400`, `Z ≈ 4 × 10⁵` and `p*(FAIL) ≈ 2.5 × 10⁻¹²`.

`FAIL` is in **both** distributions — that is the whole reason fix F3 put it in
the target's support — so a policy that assigns it exactly `p*(FAIL)` achieves
**TV = 0**. The convergence target is 0 and stays 0.

The correct statement is conditional and much narrower: *a policy that cannot
reach `FAIL` carries `TV ≥ p*(FAIL)`.* Useful as a diagnostic — if measured TV
sits at exactly that value, the policy is never dead-ending — but it is not a
floor on the metric. Report `p*(FAIL)` **alongside** TV; never subtract it, and
never describe TV as bottoming out there. An earlier draft did both, which would
have propagated into every Gate-2 table and quietly contradicted the
`FlowOraclePolicy` criterion in §5.

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
| zero-`Δd` fraction, **structural** | ≤ 0.6, reported per instance |
| zero-`Δd` fraction, **visitation-weighted** under `UniformPolicy` | ≤ 0.6, reported per instance |
| distinct `d` values reachable | ≥ 10 per instance |

**Both densities, because they answer different questions.** The structural one
describes the environment; the visitation-weighted one describes the transitions
a learner actually samples, which is what Gate 2's comparison is made of. They
can differ substantially when mass concentrates on a region of the lattice, and
reporting only the flattering one would be a choice made after seeing them.

**Both need a formula, because "fraction of transitions" has several readings and
they do not agree.** Let `E` be the set of `ADD` edges `(s, a, s')`:

```
structural           = |{e ∈ E : Δd(e) = 0}| / |E|
visitation-weighted  = Σ_{e ∈ E, Δd(e)=0} f(s)·P_F(a|s)  /  Σ_{e ∈ E} f(s)·P_F(a|s)
```

Three exclusions, each of which would otherwise move the number:

* **`STOP` transitions are excluded.** `STOP` does not change the selected set, so
  `Δd = 0` for every one of them by construction. Counting them would inflate the
  zero fraction mechanically, and they are not where the policy's structural
  decisions are made.
* **Dead-end absorption is excluded** — it is not a transition.
* **Weights are visitation mass** `f(s)·P_F(a|s)`, normalised over `ADD` edges —
  which is global conditioning on "an `ADD` was taken", and is deliberate. The
  rejected alternative is **per-state renormalisation**: dividing each state's
  contribution by `P_F(ADD|s)` so every visited state contributes `f(s)`
  regardless of how likely it was to add. That upweights states where *stopping*
  is likely, and answers "what does a typical visited state look like" rather
  than "what does a typical sampled transition look like". L7 conditions on
  transitions, so the second is the question. (An earlier draft described this
  bullet's own formula as the rejected alternative, and got the reweighting
  direction backwards.)

If the generator cannot meet these, that is a finding about the environment to
report **before** three weeks go into seven learners — which is exactly what
building Phases 1–4 before the data pipeline is for.

### G6 — The exact evaluator needs a policy interface Phase 3 has not defined [ANALYSIS]

The DP consumes `P_F(a|s)` and `P_F(STOP|s)` for every reachable state. Phase 3's
frozen interface (fix F6) is `policy(state_repr, action_reprs) → logits` — a
*tensor* interface, one level below what the evaluator needs, and it does not
exist yet.

**Decision.** Phase 2 defines and freezes a **batch-first** evaluation protocol:

```python
class ActionPolicy(Protocol):
    def action_log_probs(
        self, state_ix: np.ndarray, graph: StateGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        """(log P_F(ADD a_j | s_i) as [n_states, n_atoms], log P_F(STOP | s_i) as [n_states]).

        `state_ix` indexes into `graph`; the states themselves are `uint64`
        bitmasks reachable as `graph.mask[state_ix]`.  Passing indices rather
        than materialised sets keeps the hot path free of Python objects.

        -inf on illegal actions; per row the finite entries sum to 1 in
        probability.  Never called on a dead-end state.
        """
```

Batch-first because of G2: at 10⁵ states per instance, a per-state call would put
10⁵ Python round-trips into a network on the critical path of every Gate-2
checkpoint. Batched calls are what keep evaluation at seconds rather than hours.
The implementation **may chunk internally** — 10⁵ states of network activations
is a memory constraint Phase 3 should not be forced into by an interface — so
"batched" means "not one call per state", not "exactly one call per instance".

**The learners do not implement this protocol. An adapter does.** Architecture
fix F6 freezes the learner-facing interface as
`policy(state_repr, action_reprs) → logits`, precisely so that a learner never
sees where its features came from: an MLP featurizer on synthetic data, the
Stage-B graph encoder on real data, and the learner cannot tell. If a Phase-3
learner implemented `ActionPolicy` directly it would take `frozenset`s of
**synthetic atom ids** and a `StateGraph`, coupling it to this environment — the
exact failure F6 exists to prevent, arriving through the back door of an
evaluation interface.

So the layering is explicit:

```
StateGraph ──► SyntheticFeaturizer ──► policy(state_repr, action_reprs) → logits
                    (adapter)                    (the learner, F6 interface)
           └────────── implements ActionPolicy ──────────┘
```

The adapter takes state **indices**, reads their `uint64` bitmasks from the
graph, featurizes them, calls the network once per chunk, and scatters the logits
back into the `[n_states, n_atoms]` layout the DP wants. Phase 3 writes it once
and every learner reuses it. Converting a bitmask to whatever representation a
featurizer wants is the adapter's job, and the only place atom ids appear.

**Dead-end states are never queried.** A dead end has no legal `ADD` and a masked
`STOP`, so *no probability distribution over its actions exists* — asking for one
is a category error, and returning all `-inf` would silently produce NaNs in the
DP. The evaluator routes such states' mass straight to `FAIL` and skips them.

### The oracle policy needs a flow construction, not a softmax

"Softmax over terminal reward" is not a specification. Actions lead to **partial
states**, not to terminals, and a terminal is reachable through many insertion
orders — so naively scoring an action by the reward of terminals below it
double-counts every terminal reachable through more than one child.

The correct construction is the standard flow decomposition against a **fixed
backward policy**, computed on the enumerated graph in decreasing `|S|`:

```
P_B(s | s')   = uniform over the removable atoms of s'          # architecture §3.1
r_dead        = r_fail / |dead-end states|                      # FAIL's allocation
F(s)          = R(s)·1[s is a valid stop]                       # terminating flow
              + r_dead·1[s is a dead end]                       # flow into FAIL
              + Σ_{s' ∈ Ch(s)} F(s') · P_B(s | s')              # child flow
P_F(s → s')   = F(s') · P_B(s | s') / F(s)
P_F(STOP | s) = R(s)·1[valid] / F(s)
```

Under this construction `P_F` samples terminals exactly in proportion to `R`,
which is what makes it a usable oracle.

**`FAIL` needs its own term in the recurrence, and this is the part that is easy
to get wrong.** `FAIL` is a *single absorbing terminal* reached from *many*
dead-end states, so the construction has to say how `r_fail` is divided among
them. Decision: **`r_fail` is split uniformly across dead-end states**, each
receiving terminating flow `r_dead = r_fail / |dead ends|`. Total flow into
`FAIL` is then exactly `r_fail`, `Z = Σ_valid R(X) + r_fail` as fix F3 requires,
and the oracle achieves `p_θ(FAIL) = p*(FAIL)` — which is what lets it reach
TV = 0 rather than the `p*(FAIL)` residual an earlier draft would have accepted
(G4).

**Why the middle term cannot be left to prose.** A dead end is not a valid stop
and has no children, so without `r_dead·1[dead end]` the recurrence gives it
`F(s) = 0` — and then `P_F(s' → s) = 0`, so the oracle never routes any mass to
`FAIL` at all. Worse, it would still *pass* a `TV < 10⁻⁹` check, because the
resulting error is only `p*(FAIL) ≈ 2.5 × 10⁻¹²`. A silently broken construction
that goes green is the failure mode this whole phase exists to prevent, so:

* `tiny_instance()` **must contain at least one reachable dead end**, or the
  `FAIL` path of the oracle is never executed by any test;
* the oracle criterion asserts `p_θ(FAIL)` against `p*(FAIL)` in **relative**
  terms, not only through an aggregate TV that a 10⁻¹² discrepancy cannot move.

Phase 2 therefore ships `UniformPolicy` (uniform over legal actions, `STOP`
included when allowed) and `FlowOraclePolicy` (the construction above). The
oracle is a **test instrument, not a baseline** — it consumes the exact
enumeration and so is unavailable at any scale where the learners matter.

**Freezing the protocol here is the point.** If the evaluator were written
against Phase 3's tensors, Gate 2 could only evaluate Phase-3 learners — and the
uniform-policy and oracle criteria, which are what prove the evaluator itself
correct, would have nothing to run on.

**One thing the interface must not foreclose, and nearly did.** v1.2 §6.4 names
GNN representational limits as a risk: **[EVIDENCE]** *When Do GFlowNets Learn
the Right Distribution?* (ICLR 2025) proves that the representation limits of
GNNs delineate which distributions a GFlowNet can approximate — two states the
encoder cannot distinguish cannot receive the different flows the target needs.
Its *remedy* is **Look-Ahead GFlowNets**: feed **children-state embeddings** into
the forward policy, which provably raises expressiveness.

Passing `graph` alongside `state_ix` is what keeps that available — the children
of any state are in the edge arrays. A Phase-3 adapter that featurised only the
current state would silently foreclose the published fix for a risk this project
has already written down, and would do so in a way no test would catch. The
adapter contract therefore states that children are reachable and that an LA-style
featuriser is a supported Tier-2 variant.

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

**Decision.** A fixed **benchmark suite** of instances, generated once from a
declared seed, frozen at Gate 0, and carrying a content digest in the same style
as `config_hash` and `EventLog.digest`. `verify_handoff.py` gains a lattice
fingerprint so two machines can confirm they enumerated the same environment
before comparing any number.

* **Main suite: 20 instances, generator seed `20260808`.** Enough that
  per-instance variance averages out, small enough that a full seven-learner
  sweep with periodic exact TV stays affordable. The seed is *not* one of
  `{13, 42, 7}` — those are the **training** seeds, and reusing one here would
  tie the environment to a run's randomness.
* **Probe suite: 5 instances, seed `20260809`, distractor-heavy**, generated
  *without* the G5 `Δd` band. Its purpose is to check whether a Gate-2 result
  survives where the `Δd` signal is sparse. Run once, at the end, not at every
  checkpoint — it is a robustness check on the conclusion, not part of the
  primary comparison.

* **Tuning suite: 5 instances, seed `20260811`.** Where Phase 3's β sweep runs.

**Why β needs its own suite.** β is still a placeholder, chosen by a Phase-3
sweep on "the synthetic lattice". If that sweep runs on the Gate-2 benchmark,
then β — a reward parameter shared by all seven learners — is selected on the
same instances the learners are scored on, and v1.2 §4.1's explicit requirement
that β be chosen "without touching test data" is violated by the plan's own
construction.

So the β lifecycle is declared here, in full, because two of its three parts were
missing:

1. **Candidate set, predeclared:** `β ∈ {1, 2, 4, 8}`. A sweep over an open range
   with the winner chosen afterwards is a decision rule selected after seeing
   results, which Dror et al. (ACL 2018) is cited elsewhere in this project to
   forbid.
2. **Selected on the tuning suite only.** The main and probe suites are not
   enumerated at any candidate β except the winner.
3. **Re-validated on the main suite before Gate 2 runs.** The chosen β is fed
   through `Target.at_beta`, which re-checks `r_fail_margin` *and* the
   target-mass bands (P2.3). If the main suite fails its bands at the frozen β,
   that is a regeneration under the amendment procedure of §6b — not a shrug.

All three suites are frozen at Gate 0 alongside the β candidate set.

### G10 — Two valid modes existing does not make the target multimodal [ANALYSIS]

The generator plants two substitutable claim chains, and Phase-1 §8 requirement 2
calls that "multiple disjoint valid modes". But `sufficiency` is exact atom
overlap with **one** gold proof (`utility.sufficiency`), so:

* the designated gold chain scores `sufficiency = 1`;
* the substitutable chain scores near 0, and loses `w_suff · β = 4` in log-reward
  — a factor of `e⁻⁴ ≈ 1/55` in mass against an otherwise comparable terminal;
* meanwhile *thousands* of formally valid distractor sets each carry small mass,
  and combinatorial multiplicity can make their **total** share large.

A back-of-envelope on the current weights: a junk set of 8 distractor nodes
scores `U ≈ 0.16`, `R ≈ 1.9`; the gold set scores `U ≈ 1.97`, `R ≈ 2600`. Four
thousand junk terminals against ten gold-like ones puts ~23% of the target mass
on junk. Nothing is *wrong* with that — `p*` is by definition proportional to
`R` — but it means exact TV would be substantially measuring whether learners can
match a near-uniform tail, which is not what anyone reading a Gate-2 table will
assume.

**Decision — measure where the mass actually is, and band it.** Five audits, all
cheap because `p*` is already in hand:

| Audit | Band |
|---|---|
| `p*` mass on each **designed** proof mode (completed) | **diagnostic**: below 1% for the alternative mode changes the claim, does not fail the build |
| `p*` mass on the **`neither` bucket** — completes no designed proof | ≤ 0.5 — a hard band |
| `p*` mass on terminals with `sufficiency = 0` | reported, no band |
| effective support size, `exp(H(p*))` | reported, no band |
| `p*` mass on the top-10 terminals | reported, no band |

**Why the gate moved off `sufficiency = 0`.** That threshold is too weak to catch
what it was aimed at: a terminal holding one gold atom and seven distractors has
*positive* sufficiency and escapes the band while being junk. The `neither`
bucket asks the question directly — does this terminal complete either designed
proof — and reuses a definition already needed above. `sufficiency = 0` mass is
still reported, as a descriptive cross-check.

**Why the mode mass is a diagnostic and the distractor mass is a gate.** The two
look alike and are not. The distractor share is something the generator
*controls* — fewer distractors, fewer junk terminals — so banding it is a
legitimate acceptance test. The alternative mode's share is a **consequence of
the frozen reward**: with one gold, mode B forfeits `w_suff·β = 4` in log-reward
— but *how much* it forfeits is not fixed at 4. Since `sufficiency` is the
fraction of gold atoms covered and the templates share the anchor, the binding
and possibly more, a terminal completing `P_B` has positive overlap with the gold
`P_A`, so the forfeit is

```
β · w_suff · (1 − |P_A ∩ P_B| / |P_A|)
```

which is 4 only when the templates are disjoint. An earlier draft asserted the
full `e⁻⁴` "whatever the generator does", which is wrong: overlap **is** a lever.

It is a lever that cuts both ways, and that is why the decision stands. Raising
`|P_A ∩ P_B|` raises mode-B mass and simultaneously makes the two modes *less
materially different* — and "materially different valid proofs" is the property
v1.2 §9 wants demonstrated. Tuning overlap upward to clear a 1% bar would be
buying the number by destroying the thing it is evidence for. That is the same objection
that rejected multi-gold two paragraphs down, and it applies here too.

So: report it, and if the alternative mode cannot reach 1%, **the write-up says
the Gate-2 environment is effectively unimodal** — a narrower claim, honestly
stated. Phase 2 does not block on it.

**Rejected: multiple acceptable gold proofs.** Redefining `sufficiency` as a max
over several golds would make the target genuinely bimodal — and would change a
**frozen reward term** in order to make the measuring instrument read the way we
would like. It is also unfaithful to the data the reward is modelled on:
2WikiMultiHopQA and MuSiQue supply one gold evidence path per question, so a
multi-gold `sufficiency` would be a synthetic-only convenience that does not
transfer to Phase 9.

Keep one gold; report the mass distribution; and if the alternative mode cannot
reach 1%, **say in the write-up that the Gate-2 environment is effectively
unimodal** rather than reshaping the reward until it is not. Gate 2 tests
distribution correctness, and does not require multimodality to do that — the
portfolio claim is Gate 3's, on best-of-K.

---

## 2. Scope

**In.** The ProofLattice generator with banded, audited structural properties;
exhaustive closed-subset enumeration; the exact evaluator (`p*`, `p_θ` by one
forward DP, TV/JS/KL); the `ActionPolicy` protocol with three reference policies;
the audit suite; a fixed hand-checkable instance; the frozen benchmark suite; and the
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
`LatticeSpec` (the banded knobs **and the `Config`**) ·
`generate(rng, spec) -> LatticeInstance` · `tiny_instance() -> LatticeInstance` ·
`benchmark_suite(spec) -> tuple[LatticeInstance, ...]` · `probe_suite(spec)`.

**`LatticeSpec` carries the `Config`.** Pool construction must use
`cfg.pool_cap` and enumeration must use `cfg.max_atoms`; a generator that took
only an `rng` would have to reach for a global or default them, and defaulting to
the *real* profile (64/16) instead of the synthetic one (32/8) would silently
produce an environment nobody can enumerate.

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
`StateGraph` (integer-indexed states by size, `(parent, action, child)` edge
arrays, stop-allowed and dead-end flags, `fingerprint()`).

**`fingerprint()` binds the environment, not the seed.** "Same seed → same
fingerprint" would be satisfied by hashing the seed, which proves nothing about
what was built. The digest covers the canonical serialisation of: the
`LatticeSpec` (bands, seeds and `config_hash`), every pool atom **including
`feat`**, the obligations, the backing snapshot's `state_digest()`, the gold set,
and both proof templates. A sensitivity test mutates one feature value, one
source tier and one interval bound, and asserts the digest moves each time.

**Design notes.** Breadth-first over set sizes, using `IncrementalChecker` for
validity and `legal_adds` for successors — so the enumeration walks exactly the
space the policy will and cannot drift from it. `H` is called with `ledger=None`:
enumerating a lattice would exhaust any per-query budget, and it is an offline
audit rather than a query (Phase-1 gap G9).

**States are `uint64` bitmasks over `pool.ids()`, not hashes and not
`frozenset`s** (G3): exact identity at 8 bytes, valid while `pool_cap ≤ 64` and
asserted as such. `canon_set_hash` is kept as a fingerprint for logging and
cross-machine comparison only.

**The graph is policy-independent** (G2), so it is built once per instance and
reused across every policy and every checkpoint. That is the difference between a
35-hour and a half-hour Gate 2.

**Gotcha.** The state and edge counts are the cost drivers and the things G1
bands. Build the counter before the generator, so a runaway instance is caught
while generating rather than after 200 of them exist.

### P2.3 `graft/synth/exact.py`

**Responsibility.** `p*`, `p_θ`, and the divergences.

**Surface.** `target_distribution(instance, cfg) -> Target` ·
`policy_distribution(policy, state_graph) -> np.ndarray` ·
`tv(p, q)` · `js(p, q)` · `kl(p, q)` · **`fcs(policy, state_graph)`** · `Target`
(terminals, `U` cache, `Z`, `p*`, `target_p_fail`).

**FCS belongs here, and an earlier draft of this plan omitted it entirely.**
v1.2 §6.4 requires it ("additionally report the FCS proxy") and Gate 2's own
definition in v1.2 §7 is "exact TV/KL/JS distribution error **plus FCS**".

The reason is not compliance. **[EVIDENCE]** *When Do GFlowNets Learn the Right
Distribution?* (ICLR 2025, Spotlight) defines Flow Consistency in Subgraphs as a
tractable goodness-of-fit metric, relates it to TV (Thm. 5, Cor. 1), and — in a
section titled *Inadequacy of commonly used evaluation protocols* — compares it
against number-of-modes, average reward of top-scoring samples and Shen's
accuracy, finding FCS "often the only metric accurately reflecting" convergence.

FCS exists for the case Phase 2 does **not** have: state spaces too large to
enumerate. So why compute it here? **Because this is the only place where FCS and
exact TV can both be computed on the same distribution, which makes it the only
place FCS can be *calibrated* as a proxy.** Phase 9 has no exact TV; if FCS is
never measured against ground truth on the lattice, any later FCS number is an
uncalibrated proxy standing in for a quantity nobody checked it against.

**`p_θ(FAIL)` is accumulated directly, and the complement is a consistency
check — not the other way round.** Two ways to get it:

```
direct      p_θ(FAIL) = Σ_{dead-end states d} f(d)     # sum of small positives
complement  p_θ(FAIL) = 1 − Σ_{valid X} p_θ(X)         # cancellation near 1
```

The complement is what architecture fix F3 specifies, and it is the *only* option
outside an enumerable environment — but here the state graph already labels dead
ends, so the direct sum is available and is far better conditioned. The
complement subtracts two quantities that agree to ~12 digits, so its absolute
error is bounded below by float64 representation near 1 (~2 × 10⁻¹⁶) regardless
of how carefully the sum is done. Measured: with `p*(FAIL) ≈ 2.5 × 10⁻¹²`, the
complement's relative error is **~2 × 10⁻⁴** with naive summation and no better
with `math.fsum`, because the loss is in `1 − x`, not in `Σ`.

So: **direct is the reported value**; the complement is asserted against it to
`≤ 10⁻⁹` absolute as the check that mass genuinely partitions between valid
terminals and `FAIL`. Any tolerance tighter than float64 allows would be a test
that cannot pass, dressed as rigour.

**Design notes.** `Target` caches `U` per terminal, not `R` (G7), and exposes
`at_beta(beta)` returning `p*` for a new β without re-deriving `U` — which is
what makes the Phase-3 sweep affordable. One forward DP for `p_θ` (G2);
`p_θ(FAIL)` by the **direct dead-end sum**, with the complement as a partition
check.

**`at_beta` re-runs *both* β-dependent checks, and the second one was missing.**
`p*` depends on β through `R = exp(β·U)`, so **the hard `neither`-mass band of
G10 is a function of β** — it was accepted at β = 4 and nothing recomputed it
afterwards. A suite frozen at β = 4 could therefore violate its own acceptance
band the moment Phase 3's sweep lands on a different value, with no check
failing anywhere. `at_beta` recomputes the target-mass bands as well as
`r_fail_margin`, and raises on either.

**`at_beta` re-runs the `r_fail_margin` check.** Phase 0 added a load-time
assertion that `r_fail < r_fail_margin · exp(β·U_min)`, precisely so a β sweep
cannot quietly promote `FAIL` into a competitive terminal. A sweep that moved β
through `at_beta` would walk straight past the loader and around that protection.
`at_beta` therefore constructs the corresponding `Config` and validates it,
raising on a β the loader would have refused — the check exists for this exact
caller.

### P2.4 `graft/synth/policies.py`

**Responsibility.** The `ActionPolicy` protocol (G6) and three reference
implementations.

**Surface.** `ActionPolicy` Protocol · `UniformPolicy` ·
`ForcedContinuationPolicy` · `FlowOraclePolicy(target, state_graph)` ·
`uniform_backward(state_graph)`.

`ForcedContinuationPolicy` is uniform over legal `ADD`s and takes `STOP` only
when no `ADD` remains. It exists for the dead-end absorption audit, where
`UniformPolicy` would stop early and report a clean profile by construction.

**Design note.** `FlowOraclePolicy` exists so the evaluator can be checked
against a distribution whose answer is known in advance: a policy constructed by
the flow decomposition of §G6 samples terminals exactly in proportion to `R`, so
it must produce `TV ≈ 0`. An evaluator that cannot recover a distribution it was
handed is broken in a way a uniform policy would never reveal.

It is built from the exact enumeration and is therefore a **test instrument, not
a baseline** — it is unavailable at any scale where the learners matter, and must
not appear in a results table as though it were a method.

### P2.5 `graft/synth/audits.py`

**Responsibility.** The numbers Gate 2 reports, one function each.

| Audit | Expected | Meaning if violated |
|---|---|---|
| unconstructible valid terminals | **0**, by fix F10 | the closure rule or the masks are wrong |
| equivalent-action collisions (exact child-set equality) | **0**, by G3 | the pool is malformed |
| state-fingerprint collisions | **0** | `canon_set_hash` collided; state identity is still exact, but the fingerprint is unusable for comparison |
| `FAIL` reachability | ≥ 1 reachable dead end | `STOP`-masking is doing nothing (G4) |
| **dead-end absorption mass by `\|X\|`** | **conditional** early share `Σ_{\|d\|<max_atoms−1} f(d) / Σ_{d∈D} f(d)` **≤ 0.05**, not absolute mass; full profile reported | dead ends at small `\|X\|` mean the **`ADD` masks are too tight**, not that the budget is small — the distinction Phase 1 asked Phase 2 to make |
| `d` informativeness, structural **and** visitation-weighted | not `\|s\|`-determined; both ≤ 0.6 zero-`Δd` | **Gate 2 cannot resolve L7 from L6** (G5) |
| **target mass by mode bucket** (completed chains) | reported; alt. mode ≥ 1% is a *diagnostic*, not an acceptance test | the target is effectively unimodal (G10) |
| target mass on the `neither` bucket | ≤ 0.5 | the distractor tail dominates what TV measures (G10) |
| `sufficiency = 0` mass, effective support `exp(H(p*))`, top-10 mass, Jaccard spread | reported, no band | descriptive |

**Absorption mass, not a modal bin, and exact rather than sampled.** An earlier
draft required only that the *modal* dead-end size be ≥ `max_atoms − 1`, which
passes even when a large share of dead-end mass sits at small `|X|` — the single
biggest bin can be late while the tail is unhealthy. Phase 1 asked for early dead
ends to be *exposed*, so the audit bands the **cumulative** early mass. And
because the state graph is enumerated, absorption mass is computed exactly by the
same forward DP rather than estimated from rollouts.

**The band is a conditional share, and the formula is written out because the two
readings disagree.** Let `D` be the dead-end states and `f` the forward mass
under `ForcedContinuationPolicy`:

```
early_share = Σ_{d ∈ D, |d| < max_atoms − 1} f(d)  /  Σ_{d ∈ D} f(d)   ≤ 0.05
```

Not the absolute `Σ_early f(d) ≤ 0.05`. The two can give opposite verdicts: an
instance whose *every* dead end is early but whose total dead-end mass is 10⁻³
passes the absolute form and fails the conditional one — and it is the conditional
one that answers the question the audit exists for, which is *where* failures
happen, not how many. The denominator is non-zero because criterion 14 requires at
least one reachable dead end.

**The measuring policy is named, because it is not `UniformPolicy`.**
`ForcedContinuationPolicy` is uniform over legal `ADD`s and takes `STOP` **only
when no `ADD` is legal** — the `p_stop = 0` regime Phase 1 used to find that its
own dead ends all sat at `max_atoms`. `UniformPolicy` includes `STOP` in its
support whenever it is allowed, so it stops early and barely reaches dead ends at
all; using it here would report a clean profile by construction.

**Modes are audited on the generator's own proof templates, not recovered by
clustering.** The generator *builds* two substitutable proofs, so it records each
one as a **complete minimal proof template** — `P_A` and `P_B`, each the full
atom set that proof needs, **including atoms shared with the other**: the entity
anchor, the answer binding, and any common structure. Completion is then a
predicate with no ambiguity:

```
completes_A(X) ≡ P_A ⊆ X          completes_B(X) ≡ P_B ⊆ X
```

| Bucket | Condition |
|---|---|
| mode A | `completes_A(X)` and not `completes_B(X)` |
| mode B | `completes_B(X)` and not `completes_A(X)` |
| mixed | both |
| neither | neither |

**Templates, not "the atoms unique to each chain".** An earlier draft used the
unique-atom sets, which is a different and wrong thing: a terminal can hold every
atom unique to chain A while lacking the shared anchor, and `H` would not object
— formal validity requires neither the anchor nor a binding. It would have been
bucketed as a completed mode A while proving nothing.

**Completion, not membership.** An earlier draft bucketed on "contains ≥ 1 atom
of `A*`", which would count a single chain-head node plus seven distractors as
mode-A mass — inflating exactly the number the audit exists to measure, with
terminals that prove nothing. A mode is a *finished* proof or it is not a mode.

`p*` mass is reported per bucket. Direct, unambiguous, and it needs no threshold.

**Two reasons an earlier clustering definition was dropped, both fatal.** It
connected terminals whose Jaccard similarity was **≤ 0.5** and called the
connected components modes — but that is the *dissimilarity* relation, so its
components group unlike proofs together, which is exactly backwards. And it first
subtracted a "mandatory core" defined as the intersection of all valid terminals,
which is very likely **empty**: formal validity does not require covering the
anchor or carrying a binding at all — a lone node atom is a valid terminal
(Phase-1 finding). So the subtraction was a no-op and the definition rested on
nothing.

Pairwise-Jaccard spread over valid terminals is retained as a **descriptive
secondary** measurement, reported and not banded.

**Gotcha.** These are consumed by a gate, so they are reported *per instance and
aggregated*, never a single pooled number that one bad instance can hide inside.

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
| 6 | `lattice.py` generator with band rejection | 20 instances inside all three bands of G1, within the 30-minute suite budget |
| 7 | `audits.py` | every audit runs; `d`-informativeness and dead-end absorption inside their bands |
| 8 | frozen benchmark suite + lattice digest in `verify_handoff` | two runs produce the same digest |

Steps 3–5 are the correctness spine and should be finished before the generator
gets interesting — an evaluator verified on a hand-checked instance is what makes
every later number trustworthy. Step 6 is the one that can overrun (G1).

---

## 5. Exit criteria

**The evaluator is correct**
1. On `tiny_instance()`: enumerated `p*` matches the hand-computed table literal for literal, and `Σ p* = 1` **including `p*(FAIL)`**.
2. On `tiny_instance()` with `UniformPolicy`: `TV(p_DP, p_MC) < 0.02` at `N = 200,000`, and every terminal within 5σ (G8).
3. On `tiny_instance()` with `FlowOraclePolicy`: **`TV < 10⁻⁹`** — literally zero to floating point, not `p*(FAIL)`. **And separately**, on the *directly accumulated* `p_θ(FAIL)`: `|p_θ(FAIL) − r_fail/Z| / (r_fail/Z) < 10⁻⁹`. `p*(FAIL) ≈ 2.5 × 10⁻¹²` cannot move an aggregate TV, so an oracle that never routes mass to `FAIL` would pass the TV check while being wrong (G6). The assertion is on the direct sum because the complement cannot resolve that value — its relative error is ~2 × 10⁻⁴ at float64, even with `fsum`. `tiny_instance()` must contain a reachable dead end, or neither assertion executes the path.
4. On a benchmark instance with `UniformPolicy` at **`N = 200,000` rollouts, MC seed `20260810`**: the **top-20 highest-mass terminals** agree within 5σ (G8). A full-support TV assertion is not made — at 5,000 terminals the sampling floor is 0.063, so such a test would measure its own noise.
5. On every benchmark instance: mass partitions — `|1 − Σ_valid p_θ − p_θ(FAIL)| ≤ 10⁻⁹` with `p_θ(FAIL)` taken from the direct sum — and `p_θ ≥ 0` everywhere. **The tolerance is set by the worst-case bound, not by the observed residual.** Scatter-add over up to 2 × 10⁶ edges has a worst case of `n·eps ≈ 4.4 × 10⁻¹⁰`; a measured one-pass residual is ~2 × 10⁻¹⁶ because the errors random-walk rather than conspire, but a criterion calibrated on the lucky case is a flaky test waiting to happen. An earlier draft said 10⁻¹², which the bound does not support.
6. `p_θ` agrees to **`≤ 10⁻¹²` absolute per terminal** under permutation of states within each `|S|` layer. Not bit-for-bit: float addition is not associative, so reordering a scatter-add changes the last bits and an exact-equality assertion would fail on a correct implementation. The DP requires ascending layer order; only intra-layer order is free, and asserting invariance to arbitrary visitation order would be asserting something false.
7. KL is reported only when finite, with the zero-support guard exercised by a deterministic policy.
    **FCS is reported alongside exact TV on every instance** (v1.2 §6.4, Gate 2 item 3). This is the only environment where both exist, so it is the only place the proxy can be calibrated against ground truth before Phase 9 has to rely on it.
8. `Target.at_beta` raises on a β the config loader would refuse, so a sweep cannot bypass the `r_fail_margin` check.

**The environment is enumerable and controlled**
9. Every benchmark instance has 200 ≤ valid terminals ≤ 5,000, ≤ 1 × 10⁵ reachable closed states and ≤ 2 × 10⁶ edges (G1).
10. **The pool is built to spec, asserted directly:** `instance.pool.cap == cfg.pool_cap == 32` and `20 ≤ len(instance.pool) ≤ 30` (the architecture's universe size). The state-count band of criterion 9 does **not** subsume this — a generator that passed the wrong cap could still happen to produce a small graph, and the failure would only surface in Phase 7 when the same mistake is made against real pools.
11. **Budgets, derived from the total rather than guessed per unit.** Gate 2 is 7 learners × 3 seeds × **C = 50 checkpoints** × 20 instances = **21,000 evaluations**. Total exact-TV evaluation is budgeted at **≤ 1 h**, fixing the per-evaluation ceiling at **≤ 0.15 s**; the estimate from G2's edge count is 0.05 s, so there is 3× headroom. `C = 50` is **frozen in the Phase-3 handoff** — the budget is meaningless without it, and a Phase 3 that quietly checkpointed 200 times would quadruple the number this criterion exists to bound. If `C` must change, the budget is `21,000 · (C/50) · 0.15 s` and the ceiling moves with it.
    Enumeration + `Target` construction ≤ 60 s per instance; **total suite generation including rejections ≤ 30 min** (G1); suite resident memory ≤ 2 GB — comfortable under the `uint64` state representation of G3, and not under a `frozenset` one.
    **The 0.15 s covers the numpy DP given precomputed log-probabilities, and nothing else.** The policy's batched forward pass is a property of the learner, not of the evaluator, and is measured and reported separately — folding it in would make Phase 2's budget depend on Phase 3's architecture.

**The audits**
12. Unconstructible-valid-terminal rate is **0** on every instance (fix F10 as a regression test).
13. Equivalent-action collision rate, by exact child-set equality, is **0** on every instance — with G3's proof in the module docstring and the SA-GFN correction *not* applied. Zero state-fingerprint collisions.
14. `FAIL` is reachable on every instance. Dead-end **absorption mass by `|X|`** is computed exactly under `ForcedContinuationPolicy` and reported in full, with the **conditional early share** `Σ_{d: |d| < max_atoms − 1} f(d) / Σ_{d ∈ D} f(d) ≤ 0.05` — the Phase-1 handoff, which asked Phase 2 to distinguish "budget too small" from "masks too tight", and which neither a modal-bin statistic nor an absolute-mass band would answer. `p*(FAIL)` is reported alongside every TV and **never subtracted from it**.
15. On every **main-suite** instance: **`d(s)` is not determined by `|s|`** at ≥ 3 distinct sizes, **≥ 10 distinct `d` values are reachable**, and both the structural and the visitation-weighted zero-`Δd` fractions are ≤ 0.6 (G5). The **probe suite is exempt by design** — it is generated without this band precisely to test robustness where the signal is sparse (G9), so applying the band there would require it to satisfy the condition it exists to violate.
16. Target mass is reported per mode bucket (A / B / mixed / neither) on **completed** chains, with `sufficiency = 0` mass, effective support size, top-10 mass and Jaccard spread. Mass on the **`neither`** bucket is ≤ 0.5 — **a hard band**. The alternative mode's ≥ 1% share is a **diagnostic that changes the claim, not an acceptance test that blocks the build** (G10).

**Reproducibility**
17. **The planted structure is asserted to exist, per instance.** Nothing else in this list does — and the current criteria can all pass on an instance whose deliberate mechanisms are broken. Criterion 14 needs only *one* reachable dead end, which a cap-induced one supplies even if both planted failure routes are dead. Worst of all, a malformed `P_B` produces zero alternative-mode mass, which G10 treats as a **diagnostic** — so a broken instance would be written up as "effectively unimodal" instead of rejected. Therefore, on every instance:
    - `P_A` and `P_B` are each formally valid, closed and reachable terminals, and `P_A ≠ P_B`;
    - the duplicate-slot mechanism and the temporal-disjoint mechanism **each independently** produce a reachable invalid state (checked separately, not "at least one dead end exists");
    - ≥ 2 distinct source tiers occur among atoms whose `Source` resolves, and every `feat` is non-zero with ≥ 2 distinct vectors;
    - the time constraint is bounded on both sides, and ≥ 1 claim interval **partially** overlaps it (neither disjoint nor containing);
    - every atom's `target` resolves in the backing snapshot;
    - ≥ 1 invalidated edge and ≥ 1 quarantined assertion are present and reachable.
18. **The lattice fingerprint binds the environment, not just the seed.** "Same seed → same fingerprint" is satisfied by hashing the seed alone, which would prove nothing. The digest covers the canonical serialisation of: the `LatticeSpec` (bands, seeds, and `config_hash`), every pool atom including `feat`, the obligations, the backing snapshot's `state_digest()`, the gold set, and both templates. **Sensitivity is tested**: mutating any one of those — a single feature value, one tier, one interval bound — must change the digest.
19. Both suites are deterministic: same seed → identical lattice fingerprint, on a different machine.
20. `graft.synth` imports no ML library.

---

## 6. Decisions to lock before writing code

**This table is normative.** Every value in it is also stated in §1 (the gap that
motivated it), §3 (the module that implements it), §5 (the criterion that checks
it) and sometimes §8 (the handoff that inherits it) — four or five copies each,
which is how five review rounds produced roughly forty defects, most of them a
fix landing in three places out of four. Where a restatement disagrees with this
table, **the table wins and the restatement is a bug**. `scripts/check_plan_consistency.py`
runs in the test suite and fails on the specific disagreements that have already
happened once.

| # | Decision | Recommended | Cost if changed later |
|---|---|---|---|
| 1 | Terminal-count band | 200 ≤ k ≤ 5,000; ≤ 200 rejection attempts | Re-generate the suite; every Gate-2 number re-runs |
| 2 | State and edge ceilings | 1 × 10⁵ states, 2 × 10⁶ edges | Same, plus the Phase-3 sweep budget moves |
| 3 | `p_θ` algorithm | one forward DP over a **policy-independent, precomputed** graph (G2) | 35 h of Gate-2 evaluation instead of 0.3 h |
| 4 | State identity | `uint64` bitmask over `pool.ids()`, asserted valid while `pool_cap ≤ 64`; `canon_set_hash` is a fingerprint only (G3) | an "exact" evaluator that can merge two states |
| 5 | Collision audit | regression test expecting 0 by child-set equality; SA-GFN correction **not** applied, reason recorded; Phase-1 §8 req. 3 withdrawn (G3) | two live documents disagreeing, and a citation that would not survive review |
| 6 | `FAIL` semantics | reachable via duplicate-slot and disjoint-temporal bindings; `p*(FAIL)` reported **beside** TV, never subtracted; TV target is **0** (G4) | every Gate-2 table carries a fictitious floor |
| 7 | `d` informativeness | **main suite only**: not `\|s\|`-determined at ≥ 3 sizes; ≥ 10 distinct `d` values; **both** structural and visitation-weighted zero-`Δd` ≤ 0.6. The probe suite is generated *without* this band, by design (G5, G9) | **Gate 2 becomes unable to resolve Contribution 3** |
| 8 | Policy interface | **batch-first** `action_log_probs(state_ix: np.ndarray, graph)` — indices, not materialised sets; never called at dead ends; an adapter implements it, never a learner (G6) | 10⁵ Python round-trips per checkpoint; Phase 3 re-wires |
| 9 | Oracle policy | flow decomposition against uniform `P_B`; `r_fail` split uniformly across dead ends (G6) | an oracle that cannot reach TV = 0, so the evaluator is unverifiable |
| 10 | Caching | `U` per terminal; similarity matrix per pool; `at_beta` **re-validates** `r_fail_margin` (G7) | β sweep cost multiplies, and a sweep can bypass the FAIL-competitiveness check |
| 11 | MC agreement | `k ≤ 100` instance at `N = 200,000`, `TV < 0.02`; top-20 only at full scale (G8) | a test that measures its own sampling floor |
| 12 | Suites | main 20 @ seed `20260808`; probe 5 @ seed `20260809`, distractor-heavy; both content-hashed (G9) | learners compared on different environments |
| 13 | Mode definition | bucket by **completion** of the generator's complete minimal proof templates `P_A` / `P_B` (which include shared atoms), not by unique-atom membership and not by clustering (G10) | partial chains counted as proof modes; or a clustering definition that groups *dissimilar* proofs and subtracts an empty core |
| 14 | Target-mass status | `neither`-bucket mass ≤ 0.5 is a **gate**; alt.-mode ≥ 1% is a **diagnostic**; one gold retained (G10) | either a build that cannot pass, or a Gate-2 table that reads as multimodal when it is not |
| 15 | `p_θ(FAIL)` | **direct** sum over dead ends is the reported value; complement is a partition check at ≤ 10⁻⁹ absolute, set by the worst-case `n·eps` bound over 2 × 10⁶ edges (G6) | a criterion float64 cannot satisfy, or an oracle bug that passes on aggregate TV |
| 16 | Dead-end audit | exact absorption mass by `\|X\|` under `ForcedContinuationPolicy`; **conditional** early share `Σ_{|d|<max_atoms−1} f(d) / Σ_{d∈D} f(d)` ≤ 0.05, not absolute mass | a modal-bin statistic that passes with an unhealthy tail |
| 17 | Evaluation budget | total ≤ 1 h across 21,000 evaluations ⇒ ≤ 0.15 s each, DP only; forward pass reported separately | a per-unit limit that silently permits 2.9 h |
| 18 | Full-scale MC | `N = 200,000`, seed `20260810`, top-20 within 5σ | "within 5σ" without an `N` is not a criterion |
| 19 | β lifecycle | candidates `{1, 2, 4, 8}` predeclared; swept on a **separate tuning suite** (seed `20260811`); re-validated on the main suite via `at_beta`, which checks `r_fail_margin` **and** the target-mass bands | β chosen on the same instances the learners are scored on, and a frozen suite silently violating its own acceptance band |
| 19b | FCS | reported beside exact TV on every instance, to calibrate the proxy Phase 9 must use without ground truth (v1.2 §6.4) | an uncalibrated proxy standing in for a quantity never checked against it |
| 20 | Lattice fingerprint | digest over spec, pool (with `feat`), obligations, snapshot digest, gold and both templates; sensitivity-tested | a fingerprint that only proves the seed was the same |
| 21 | Structural assertions | both templates valid and reachable; **each** planted failure mechanism separately reachable; tiers, features, intervals, targets, invalidated edge and quarantined assertion all asserted per instance | an instance whose planted mechanisms are broken passing every other criterion, and a malformed `P_B` being written up as "effectively unimodal" |

**PROPOSED — requires your sign-off; not locked until then.** Everything else in
§6 is a decision I have taken; this one is a recommendation awaiting a yes,
and the document's status line ("ready to code once §6 is signed off") means
exactly this item. Calling it "resolved pending sign-off" in an earlier draft was
having it both ways.

The `d`-informativeness band of
≤ 0.6 zero-`Δd` transitions is a **guess calibrated on Phase-1 fixtures, not on
the lattice** — they measured 0.67, just outside it. It is a hard main-suite
criterion (exit criterion 15), and the three things that makes it commit to are
stated here so the Gate-2 write-up cannot quietly overclaim:

1. **`≤ 0.6` is deliberately a best-case design condition**, chosen so Gate 2 has
   the resolving power to separate L7 from L6 at all. It is not an estimate of
   anything.
2. **It is not claimed to represent natural or real-data density.** Real pools are
   distractor-heavy; Phase-1 fixtures already measured 0.67, and Phase 9's pools
   will plausibly be worse.
3. **The probe suite controls the robustness claim.** A Gate-2 result on the main
   suite is a result *under a declared signal density*; whether it survives a
   sparse one is what the 5-instance distractor-heavy probe answers, and the
   write-up states both or neither.

The reasoning that led there, kept because the alternative is a live option:

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

## 6b. Declared risks — not defects, but not settled either

Four things in this plan are estimates that the build will confirm or refute.
Recorded so that a later overrun is a known risk materialising rather than a
surprise, per `CLAUDE.md`'s rule that "no evidence found" entries are findings.

| Risk | If it bites |
|---|---|
| The 0.05 s DP estimate and 0.15 s ceiling over 2 × 10⁶ edges may be optimistic in numpy | tighten the state/edge bands (G1), or drop `C` below 50 and re-derive the budget |
| The generator may not satisfy the terminal, state, edge, `Δd`, dead-end and `neither`-mass bands **simultaneously** within 200 attempts | relax the band that is least load-bearing — `neither`-mass before `Δd`, since G5 is what Gate 2's resolving power rests on — **under the amendment procedure below**, never as an implementation-time adjustment |
| The `≤ 24 MB`/instance graph estimate counts edge arrays; index and object overhead could push the suite past 2 GB | the `uint64` state representation of G3 already removes the dominant term (~70 MB → ~0.4 MB per instance); if it still bites, hold one instance resident at a time |
| The alternative proof mode may hold < 1% of target mass | **already handled as a diagnostic** (G10): the write-up narrows to "effectively unimodal" rather than the environment being reshaped. A research limitation, not a plan defect |

### Amending a band after it fails — the procedure, declared in advance

A band that the generator cannot satisfy may have to move. **That is changing the
measuring instrument, and it is exactly the move that has to be governed**,
because the same edit made after seeing a Gate-2 result is indistinguishable from
tuning the environment until the answer improves.

Any relaxation requires, in order:

1. a **new plan version** recording which band moved, to what, and why;
2. a **new config/spec version** — the band is part of the frozen set;
3. **regeneration of both suites** from their seeds;
4. a **new lattice fingerprint**, so no result computed against the old
   environment can be silently compared to one against the new;
5. **Gate-0 re-sign-off**;
6. **no learner results inspected beforehand.** If any have been, the relaxation
   is contaminated and the honest options are to keep the failing band and report
   that the environment could not be built, or to restart the sweep.

Point 6 is the one that matters and the one that is easy to skip, because by the
time a band is failing there is usually a half-finished run sitting there.

---

## 7. Explicitly not in Phase 2

No policy network · no learner · no training loop · no search algorithm · no
β sweep (only the machinery that makes it cheap) · no real data · no annotation ·
no PyTorch. If a `graft/synth/` file imports `torch`, something has gone wrong.

---

## 8. What Phase 3 will need from this, verbatim

```python
from graft.synth.lattice   import LatticeInstance, LatticeSpec, benchmark_suite, generate, probe_suite, tiny_instance
from graft.synth.enumerate import StateGraph, reachable_states, valid_terminals
from graft.synth.exact     import Target, policy_distribution, target_distribution, tv, js, kl
from graft.synth.policies  import ActionPolicy, FlowOraclePolicy, UniformPolicy
from graft.synth.audits    import run_audits
```

### Requirements this phase places on Phase 3

1. **An adapter implements `ActionPolicy`; the learners keep the F6 tensor
   interface.** Phase 3 writes one `SyntheticFeaturizer` that batches states,
   featurizes them, calls `policy(state_repr, action_reprs)` in chunks, and
   scatters the logits back. **No learner may take a `frozenset` of atom ids or a
   `StateGraph`** — that would couple it to the synthetic environment, which is
   the failure F6 exists to prevent. Chunking is the adapter's business; the
   interface does not mandate one call per instance.
2. **`C = 50` exact-evaluation checkpoints per run.** Frozen, because exit
   criterion 11's 1-hour budget is derived from it.
3. **The backward policy is uniform over *removable* atoms, not selected ones.**
   Under the closure rule an atom referenced by another selected atom cannot be
   removed, so "uniform over selected" puts mass on parents that do not exist.
   Architecture §Phase 3.1 said exactly that and contradicted its own fix F10;
   **it has been amended to "removable"**, so this handoff records a resolved
   inconsistency rather than a live one.
4. **The β sweep uses `Target.at_beta`**, not regeneration — and inherits its
   `r_fail_margin` re-validation, so a sweep cannot silently make `FAIL`
   competitive.
5. **Exact TV converges to 0.** `p*(FAIL)` is reported beside it as a diagnostic
   and is **not** a floor: `FAIL` is in both distributions, so a policy matching
   it exactly scores TV = 0.
6. **The Gate-2 decision rule is predeclared and unchanged**: exact TV at a fixed
   number of sampled training trajectories, three seeds, paired bootstrap
   (fix F12). Phase 2 supplies the metric; it does not get to redefine the rule
   after seeing it work.
7. **Every Gate-2 table carries the environment's `d`-density** — structural and
   visitation-weighted — and the **target-mass profile** (G10; note the §6 item
   still awaiting sign-off is the `Δd ≤ 0.6` band, not this one, which is
   decided).
   A result on the main suite is a result *under a declared signal density*, and
   the probe suite is where its robustness to a sparse one gets checked.
