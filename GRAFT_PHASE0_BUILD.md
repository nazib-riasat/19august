# GRAFT — Phase 0 Build Plan: Scaffold and Data Contracts

**The first component. Nothing else can be written until this exists.**

Date: 8 August 2026
Parent: `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 0) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2)
Effort: ~1 week solo
Status: ready to code

Labels inherited: **[EVIDENCE]** (named paper) · **[HYPOTHESIS]** (project tests it) · **[ANALYSIS]** (engineering judgment made here).
Phase 0 is infrastructure. Most of it is **[ANALYSIS]** and is labelled honestly as such — infrastructure choices are not paper-derived and should not pretend to be. The four places where published work does constrain the design are marked.

---

## 0. Why Phase 0 is first

Not a preference — a dependency chain:

| Phase | Blocked on Phase 0 by |
|---|---|
| 1 (deterministic core) | `H(X: ProofSet, q: Obligations, G: GraphSnapshot)` — three types, none of which exist yet |
| 2 (synthetic env) | `CandidateAtom`, `ProofSet`, canonical set hash (the collision audit is *defined* over that hash) |
| 3 (7 learners) | ledger (checker-call metering), config freeze (β, `u_weights` must be identical across learners or the comparison measures reward engineering — v1.2 §5.1) |
| 4 (5 search algos) | ledger **budget enforcement** — "fixed checker budget" is the comparison's controlled variable |

The ledger point is the one that actually bites. If Phases 3–4 are built with a ledger bolted on afterwards, every run made before it is uncomparable and must be re-run. Meter first, measure after.

---

## 1. Five specification gaps Phase 0 must close

Found while making Phase 0 concrete. Each is a decision that must be made *now* because it is frozen at Gate 0 and consumed by later phases.

### G1 — `stop_allowed = H(state)` makes the checker budget unsatisfiable [ANALYSIS]

Architecture §1.5 defines `stop_allowed(state) = H(state)`. Architecture §12 sets `checker_budget = 32` calls/query and `K = 8`, `max_atoms = 16`.

If every construction step evaluates `H`, one portfolio costs up to 8 × 16 = **128 checker calls** against a budget of 32. The primary metric is unmeasurable as specified.

**Resolution (three parts, all decided here):**

1. **`H` is implemented as an incremental validator.** Most sub-checks are incrementally maintainable per `ADD`: type legality (per-atom), duplicate detection (hash set), size limit (counter), retired-evidence (per-atom at pinned snapshot). Only cross-atom binding/temporal consistency needs recomputation, and that is scoped to the affected binding. Amortized O(1) per `ADD`. This is a **Phase-1 implementation constraint recorded in Phase 0** because the meter definition depends on it.
2. **Two meters, not one.**
   - `terminal_checks` — full validation of a *completed candidate set*. **This is the primary budget axis.** It is the "expensive true evaluator" in the Robust Scheduling framing (**[EVIDENCE]** ICLR 2023: diverse candidates under a cheap proxy beat proxy-optimization when the true evaluator is expensive — the budget must meter *that* evaluator).
   - `incremental_ops` — cached validity updates during construction. Logged and reported, never the primary axis. Charging these would make constructive methods artificially expensive against one-shot methods like PCST, and the architecture already reports wall-clock and FLOPs separately (§4).
3. **`checker_budget = 32` stands, and becomes binding in a useful way.** K = 8 is the *returned* portfolio size; 32 is how many terminal validations may be spent producing it. A sampler may generate 32 candidates, validate all, return the best 8. PCST spends ~1. That degree of freedom is the point of a budget — it is what makes the comparison about search efficiency rather than about who is allowed more tries.

### G2 — `GraphSnapshot` must be a Protocol, not a class [ANALYSIS]

Phase 1's checker takes a `GraphSnapshot`. The real graph store is Phase 6. If `GraphSnapshot` is a concrete class backed by the event log, **Phase 1 blocks on Phase 6** and the whole "build 1–4 with zero real data" strategy collapses.

**Resolution:** `GraphSnapshot` is a `Protocol` exposing only the read API the checker needs. The Phase-2 synthetic lattice implements it directly; the Phase-6 event-log store implements it later. Phases 1–4 depend on the Protocol alone and never import a concrete store.

### G3 — `u_weights.size` under-specifies the size term [ANALYSIS]

v1.2 §4.1 gives `U` six weights. Architecture §1.3 defines the size term as `λ_v·|V| + λ_e·|E|` — which needs **two** coefficients, so the config's single `size: 0.1` cannot express it.

**Resolution:** `size = |X| / max_atoms`, a single normalized atom count, one weight, six weights total as v1.2 specifies. Node/edge asymmetry is not motivated by anything in the plan and adds a free parameter to a reward that must stay frozen across seven learners.

### G4 — `U` term normalization is undeclared, which makes β meaningless [ANALYSIS]

v1.2 §4.1 requires "exact range and normalization" per term. Without it, β has no consistent interpretation and the Phase-3 β sweep is uninterpretable.

**Resolution, frozen here:** every `U` term outputs **[0, 1]**, so `U ∈ [−(w₅+w₆), (w₁+w₂+w₃+w₄)]` and β scales a bounded quantity. Recorded in config as a schema constraint, asserted in Phase 1 unit tests.

### G5 — Phase 0 as written builds schemas nobody consumes for 10 weeks [ANALYSIS]

Architecture §Phase 0 lists the full schema set including `Turn`, `SourceSpan`, `Assertion`, `Node`, `Edge`, `OutputRecord` — but Phases 1–4 (~6.5 weeks) run entirely on synthetic data and touch **none** of them. They need exactly four: `Obligations`, `CandidateAtom`, `ProofSet`, plus the `GraphSnapshot` Protocol.

Designing the graph schema in week 1 with no consumer is speculative, and because the event log is append-only, wrong guesses are expensive to migrate.

**Resolution — two tiers, one home.** The "nothing else may define a schema" rule holds: `schemas.py` remains the single home. But:
- **Tier A (frozen in Phase 0):** `Obligations`, `CandidateAtom`, `ProofSet`, `GraphSnapshot` Protocol, ledger and config records. Consumed immediately by Phases 1–4. Changing these later means re-running experiments, so they are frozen.
- **Tier B (declared in Phase 0, frozen at their consumer phase):** `Turn`, `SourceSpan`, `Assertion`, `Node`, `Edge`, `OutputRecord`. Written now as the architecture specifies, carrying `schema_version`, explicitly amendable until Phase 5/6/10 respectively. The event log records `schema_version` per line so a migration is possible rather than catastrophic.

---

## 2. Scope

**In:** config system, schemas (both tiers), deterministic IDs, event log, graph-store Protocol + replay implementation, budget ledger, run manifest and determinism control, test suite.

**Out:** every checker sub-check (Phase 1), the lattice generator (Phase 2), any model, any dataset, any LLM call. Phase 0 has **zero ML dependencies** — PyTorch is not imported.

---

## 3. Modules

Eight units. Each lists responsibility, public surface (signatures as specification, not implementation), and the gotchas that cause rework if missed.

### P0.1 `graft/config/`

**Responsibility.** One YAML → validated frozen dataclasses → SHA-256 hash. The hash is the identity of an experimental condition.

**Surface.** `load_config(path) -> Config` · `config_hash(cfg) -> str` · `Config` tree of frozen dataclasses mirroring architecture §12.

**Design notes.**
- Hash over the **canonical serialization of the resolved config** (sorted keys, normalized floats), never the YAML text — comments and whitespace must not change identity.
- Frozen dataclasses, `__setattr__` blocked. A config mutated mid-run silently invalidates the hash.
- Validation at load: `0 < beta`, `0 < r_fail`, all `u_weights ≥ 0`, `1 ≤ K ≤ checker_budget`, `max_atoms ≤ pool_cap`, `len(seeds) ≥ 3` (**[EVIDENCE]** Dror et al., ACL 2018 — multiple seeds are part of the predeclared protocol, so the config should refuse fewer), `support_policy ∈ {strict}`.
- Config carries `schema_version` and `frozen_at_gate0: bool`.

**Gotcha.** Defaults live in code; YAML overrides only. Otherwise a missing YAML key silently changes an experiment.

### P0.2 `graft/schemas.py`

**Responsibility.** Every dataclass in the system, both tiers, with explicit JSON round-trip.

**Surface.** Tier A: `Obligations`, `CandidateAtom`, `ProofSet`, `Interval`. Tier B: `Turn`, `SourceSpan`, `Assertion`, `Node`, `Edge`, `OutputRecord`. All with `to_dict()` / `from_dict()`.

**Design notes.**
- **Plain frozen dataclasses + hand-written serialization**, not pydantic. Justification against the boring-stack rule: the event log persists to disk permanently and append-only, so the on-disk format must be explicit and version-migratable. Hand-written `to_dict`/`from_dict` is ~10 lines per class and makes evolution visible in diffs. A validation library hides the format behind decorators, which is the wrong trade for a permanent log. [ANALYSIS]
- `ProofSet.atoms` is a `frozenset[str]` — **[EVIDENCE]** the object is a set, not a sequence (v1.2 §3.4; GFlowNet Foundations, JMLR 2023, covers set-valued state spaces directly). Any list-typed proof set is a bug.
- `Edge` carries `t_invalid` and `superseded_by`; **nothing is ever deleted** — **[EVIDENCE]** Zep's edge-invalidation model.
- `Edge.provenance: list[span_id]` is non-optional — **[EVIDENCE]** SEEM (ACL 2026) anchors memory structures with explicit provenance pointers. A schema that permits an unsourced edge permits an unsourced proof.
- `Assertion.flags` holds the four independent flags of v1.2 §3.1 (`asserted_by`, `entailed_by_span`, `externally_verified`, `current_under_update_policy`), plus an `eligibility ∈ {eligible, quarantined}` field set by the Phase-5 support gate (architecture F9). Quarantined assertions stay in the log and never reach graph construction. **No boolean named `true` or `verified` may exist** — grounding ≠ truth is enforced by the type, not by discipline.
- **`CandidateAtom.refs` is the closure mechanism** (architecture F10): an atom is legal to add only once every atom listed in its `refs` is already selected. Edges reference their endpoints, bindings reference their referents, nodes have empty `refs`. Getting this field right in Phase 0 is what lets Phase 1's masks and Phase 2's **closed-subset** DP be written without a later schema change.

**Gotcha.** Tier B classes must be marked in docstrings as amendable-until-Phase-N, or someone will treat them as frozen and design around a guess.

### P0.3 `graft/ids.py`

**Responsibility.** Deterministic content-derived identity. Reruns must produce byte-identical logs.

**Surface.** `span_id(turn_id, start, end)` · `atom_id(kind, refs)` · `canon_set_hash(atoms: frozenset[str]) -> str` · `edge_id(...)`.

**Design notes.**
- `canon_set_hash = sha256(sorted(atom_ids))` is **load-bearing in three places**: policy state identity (two orders reaching the same atoms are the same state — v1.2 §3.4), beam-search dedup (Phase 4 S2), and the equivalent-action collision audit (Phase 2). **[EVIDENCE]** the collision audit exists because uncorrected equivalent actions bias sampling — Symmetry-Aware GFlowNets, ICML 2025, L₁ ≈ 0.12 uncorrected vs ≈ 0.01 corrected. That audit is meaningless without a canonical hash, so it belongs here, not in Phase 2.
- All IDs are truncated SHA-256 hex (16 chars). No counters, no UUIDs — both break rerun determinism.

**Gotcha.** If `canon_set_hash` lands in Phase 2 or 3 instead, each phase invents its own and the audit silently measures nothing.

### P0.4 `graft/eventlog.py`

**Responsibility.** Append-only JSONL log. The single source of truth for graph state.

**Surface.** `EventLog.append(op, payload) -> seq` · `EventLog.replay(upto: int|None)` · `EventLog.snapshot_id() -> int` · `EventLog.open(path)` (crash-safe).

**Design notes.**
- Line format `{seq, ts, schema_version, op, payload}`. `seq` monotonic from 0; `snapshot_id` **is** a seq number.
- **Crash-resume:** on open, scan forward; a trailing line that fails JSON parse is a torn write — truncate it and resume from the last valid `seq`. Validate `seq` monotonicity during the scan. No hash chaining — over-engineering for a single-writer local file. [ANALYSIS]
- `fsync` policy configurable: off during Phase 1–4 development (no real data at risk), on for ingestion runs.

**Gotcha.** The exit criterion "ledger counts survive crash-resume" needs a *deliberate* torn-write test — truncate a file mid-line and reopen. Easy to skip, and it's the one failure that corrupts a long ingestion run.

### P0.5 `graft/graphstore.py`

**Responsibility.** The `GraphSnapshot` Protocol (G2) plus the replay-backed implementation.

**Surface.**
- `GraphSnapshot` Protocol: `node(id)` · `edge(id)` · `edges_of(node_id, etype=None)` · `is_live(edge_id) -> bool` · `ntype(id)` · `snapshot_id`.
- `ReplayGraphStore(log).at(snapshot_id) -> GraphSnapshot`.

**Design notes.**
- The Protocol is **the entire dependency surface for Phase 1**. Keep it minimal; every method added is a method the synthetic lattice must also implement.
- `is_live(edge_id)` encapsulates the `t_invalid` / `superseded_by` logic in one place, so no downstream component reimplements invalidation semantics.
- Naive replay is O(n) per snapshot. Fine at Phase 0 — there is no real data. **Write the checkpoint/cache TODO in the docstring and leave it**; premature caching here is exactly the over-engineering the architecture forbids.

**Gotcha.** Resist adding query helpers "while we're here." Every convenience method on the Protocol is a burden on the Phase-2 lattice implementation.

### P0.6 `graft/ledger.py`

**Responsibility.** Meter *and enforce* budgets. The Stage-D primary metric and the entire search comparison are defined in its units.

**Surface.** `Ledger.count(meter, n=1)` · `Ledger.stage(name)` (context manager) · `Ledger.query_scope()` (context manager; resets per-query counters) · `Ledger.remaining(meter) -> int` · `Ledger.would_exceed(meter, n) -> bool` · `Ledger.snapshot() -> dict`.

**Meters (frozen here, per G1).**

| Meter | Counts | Role |
|---|---|---|
| `terminal_checks` | full `H` on a completed candidate set | **primary budget axis**, cap 32/query |
| `incremental_ops` | cached validity updates during construction | reported, not capped |
| `model_forwards` | policy/encoder forward passes | reported |
| `llm_calls`, `llm_tokens_in/out` | extractor + API baseline | reported, cost tracking |
| `wall_clock_ms` | per stage | reported separately (architecture §4) |

**Design notes.**
- **Enforcing, not just observing.** `would_exceed` lets a search module stop cleanly at the cap. If the ledger only counts, methods drift over budget by accident and the comparison is silently unfair. This is the single most important behavioural requirement in Phase 0. [ANALYSIS]
- Per-query scoping is mandatory — the metric is *per query*, and a global counter cannot produce it.
- Single-threaded; no locking. One process, one GPU (architecture §0.4).

**Gotcha.** `terminal_checks` must be incremented **inside** the checker in Phase 1, not by callers. Caller-side counting always drifts.

### P0.7 `graft/runtime.py`

**Responsibility.** Determinism, run identity, artifact layout.

**Surface.** `set_seed(seed)` · `run_manifest(cfg, seed) -> dict` · `new_run_dir(cfg, seed) -> Path`.

**Design notes.**
- `set_seed` covers `random`, `numpy`, `torch`, `torch.cuda`; enables deterministic algorithms where available. (Imports torch lazily so Phase 0's test suite stays ML-free.)
- Manifest: `config_hash`, git SHA + dirty flag, seed, UTC timestamp, pinned package versions, hostname/GPU. **[EVIDENCE-adjacent]** Dror et al. (ACL 2018) require a predeclared protocol; a manifest is what makes "predeclared" checkable after the fact rather than asserted.
- Layout `runs/{utc}_{config_hash[:8]}_{seed}/` containing `manifest.json`, `events.jsonl`, `metrics.jsonl`. This is what the exit criterion "config-hash equality across two runs" actually tests.
- `requirements.txt` with exact pins; versions recorded in the manifest.

**Gotcha.** Record the git SHA **and** whether the tree is dirty. A dirty-tree run is not reproducible and should be visibly marked, not silently logged.

### P0.8 `graft/tests/`

Covered in §5.

---

## 4. Build order inside Phase 0

Strictly sequential; each step is testable before the next.

| Step | Build | Done when |
|---|---|---|
| 1 | Repo skeleton, `requirements.txt` with pins, empty package tree per architecture §Phase 0 | `import graft` works; test runner green on zero tests |
| 2 | `config/` — dataclasses, loader, validation, hashing | Two YAMLs differing only in comments hash identically; an out-of-range `beta` raises at load |
| 3 | `schemas.py` **Tier A only** (`Obligations`, `CandidateAtom`, `ProofSet`, `Interval`) | Round-trip property test passes on generated instances |
| 4 | `ids.py` | `canon_set_hash` invariant to insertion order; ID collisions absent on 10⁶ generated atoms |
| 5 | `ledger.py` | Budget enforcement test: a loop calling `would_exceed` stops at exactly 32 |
| 6 | `graphstore.py` — **Protocol first**, then a trivial in-memory dummy implementing it | A hand-built 3-node dummy satisfies the Protocol; Phase 1 can be written against it |
| 7 | `eventlog.py` + `ReplayGraphStore` | Replay reconstructs an identical graph; torn-write test recovers |
| 8 | `schemas.py` **Tier B** + `runtime.py` + manifest | Two runs with the same config and seed produce identical `manifest.json` except timestamp |

**Note on ordering.** Step 6 before step 7 is deliberate: the Protocol and a dummy implementation are all Phase 1 needs. If the event log slips, Phase 1 still starts on time. This is the practical payoff of G2.

---

## 5. Exit criteria

Phase 0 is done when all of these pass. Each is a real test, not a checkbox.

**Correctness**
1. Round-trip: every Tier-A and Tier-B schema serializes → deserializes identically (property-based, generated instances).
2. Event-log replay reconstructs a byte-identical graph from a 1,000-event synthetic log.
3. Torn-write recovery: truncate the log mid-line, reopen, confirm last valid `seq` recovered and the partial line discarded.
4. `canon_set_hash` is invariant across all permutations of a 6-atom set (720 permutations, exhaustive).

**Metering**
5. Budget enforcement: a module consuming `terminal_checks` in a loop halts at exactly `checker_budget`, never at 33.
6. Per-query scoping: two queries in one process report independent counters.
7. Ledger counters survive a crash-resume cycle.

**Reproducibility**
8. Comment-only YAML edits leave `config_hash` unchanged; any value change alters it.
9. Two runs, same config and seed → identical manifests apart from timestamp and wall-clock.
10. A dirty git tree is flagged in the manifest.

**Structural**
11. `graft.core` (Phase 1's namespace) can be written against the `GraphSnapshot` Protocol with no import of `graphstore`'s concrete class — verified by an import-graph assertion in tests.
12. No module outside `schemas.py` defines a dataclass that crosses a module boundary.
13. Phase 0 imports no ML library — verified by an import-graph test.

---

## 6. Decisions to lock before writing code

These are frozen at Gate 0 and every later phase inherits them. Confirm or amend now; changing any of them after Phase 3 means re-running experiments.

| # | Decision | Recommended | Consequence if changed later |
|---|---|---|---|
| 1 | Primary budget meter | `terminal_checks` only (G1) | Every Phase 3–4 comparison re-runs |
| 2 | `checker_budget` | 32/query | Same |
| 3 | `K` | 8 (returned portfolio; also "returned sets" in the search comparison) | Same, plus search table re-runs |
| 4 | `U` term ranges | all terms normalized to [0, 1] (G4) | β becomes uninterpretable; reward re-derivation |
| 5 | Size term | `|X| / max_atoms`, one weight (G3) | Reward changes → all learners re-run |
| 6 | `beta` | 4.0, swept on the synthetic lattice in Phase 3, then frozen | Re-run everything after the sweep |
| 7 | `u_weights` | per architecture §12, identical across all seven learners | v1.2 §5.1 violated — comparison measures reward engineering |
| 8 | ID scheme | truncated SHA-256, content-derived | Log incompatibility |
| 9 | Tier-A/Tier-B schema split | per G5 | Migration cost on the append-only log |
| 10 | Seeds | {13, 42, 7} | Significance protocol invalidated |
| 11 | `r_fail` | 1e-6 — `FAIL` is a member of the target support (architecture F3) | Target support changes → `p*` re-derived, every Gate-2 result re-run |
| 12 | Closure rule | an atom is addable only once every atom it references is selected (architecture F10) | Masks, `H`, `P_B` and the exact DP all change → Phases 1–4 re-run |
| 13 | Support policy | strict: `entailed_by_span=True` at `tau_nli` (architecture F9) | Active-graph contents change → Phase 6 onward re-run |

**One open question for you:** `beta = 4.0` is a placeholder in the architecture doc, to be swept on the lattice in Phase 3. That sweep must finish *before* any comparison run, and its result is then frozen. Confirm you want the sweep in Phase 3 rather than a fixed β from the start — the sweep costs perhaps two days of Phase 3 and is the standard practice in the GFlowNet literature, but it is a schedule item worth acknowledging now rather than discovering in week 8.

---

## 7. Explicitly not in Phase 0

No checker sub-checks · no utility terms · no lattice · no policy · no models · no PyTorch import · no dataset code · no LLM calls · no database · no ANN index · no experiment-tracking service · no caching layer in `ReplayGraphStore` · no learned anything.

If a Phase 0 file imports `torch`, something has gone wrong — except `runtime.set_seed`, which imports lazily and is excluded from the import-graph test.

---

## 8. What Phase 1 will need from this, verbatim

The handoff contract, so Phase 1 can start the moment Phase 0 exits:

```
from graft.schemas    import ProofSet, CandidateAtom, Obligations, Interval
from graft.graphstore import GraphSnapshot          # Protocol
from graft.ids        import canon_set_hash
from graft.ledger     import Ledger
from graft.config     import Config
```

Phase 1 writes `H`, `U`, `R`, masks, and the obligation parser against exactly these five imports and nothing else. If Phase 1 needs a sixth, Phase 0 is incomplete — fix it there, not by widening Phase 1.
