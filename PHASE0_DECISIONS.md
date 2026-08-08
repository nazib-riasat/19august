# Phase 0 — decisions taken, with reasoning and cost to reverse

Date: 8 August 2026
Parent: `GRAFT_PHASE0_BUILD.md` · `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (v1.1) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2)

This file does for Phase 0 what `CLAUDE.md` does for the project: it records
*why*, not just *what*, so that a later session does not silently reverse a
choice that something else depends on.

Phase 0 is complete. All 13 exit criteria pass (168 tests, ~3.5 s).

---

## 1. The thirteen §6 decisions — all confirmed as recommended

| # | Decision | Value | Confirmed because |
|---|---|---|---|
| 1 | Primary budget meter | `terminal_checks` only | The budget must meter the *expensive true evaluator* for the Robust Scheduling framing to hold. Charging incremental ops would make constructive methods artificially expensive against PCST. |
| 2 | `checker_budget` | 32/query | Enforced, not observed. `would_exceed()` before spending. |
| 3 | `K` | 8 | One constant, used as both portfolio size and "returned sets" in the search comparison (fix F5). |
| 4 | `U` term ranges | all [0, 1] | Without it β scales an unbounded quantity and the Phase-3 sweep is uninterpretable. |
| 5 | Size term | `\|X\| / max_atoms`, one weight | Node/edge asymmetry is unmotivated and adds a free parameter to a reward that must stay frozen across seven learners. |
| 6 | `beta` | 4.0, swept in Phase 3, then frozen | The sweep costs ~2 days and is standard practice; a β fixed blind over a reward spanning [−0.35, 2.25] would make Gate-2 TV numbers hard to read. |
| 7 | `u_weights` | per architecture §12, identical across learners | Otherwise the comparison measures reward engineering (v1.2 §5.1). |
| 8 | ID scheme | truncated SHA-256, content-derived, 16 hex chars | Counters and UUIDs break rerun determinism. 64 bits gives ~2.7e-8 collision probability at 10⁶ atoms; measured at 0. |
| 9 | Tier-A / Tier-B schema split | per G5 | Tier B carries an amendable-until-Phase-N note in its docstring. |
| 10 | Seeds | {13, 42, 7} | Config refuses fewer than three, and refuses duplicates. |
| 11 | `r_fail` | 1e-6 | Plus an automated margin check — see §2.7. |
| 12 | Closure rule | atom addable only when every referenced atom is selected | Enforced structurally: a `node` atom with non-empty `refs` raises at construction, which is what makes nodes-first construction always valid. |
| 13 | Support policy | strict | `support_policy` accepts only `"strict"`; a looser value is a config error, not a knob. |

---

## 2. Decisions Phase 0 had to make that the build plan did not settle

Each of these was forced by writing the code. They are listed with what it costs
to change your mind.

### 2.1 `max_atoms` and `pool_cap` follow the environment profile

**Decision.** They are per-profile, not global: `default.yaml` is 64/16,
`synthetic.yaml` is 32/8. Everything else — `beta`, `u_weights`, `r_fail`, `K`,
`checker_budget`, `seeds` — stays global.

**Why.** `size = |X| / max_atoms`, and Phase 2.1 caps lattice terminals at
|X| ≤ 8. Under a single global `max_atoms = 16` the lattice's size term could
only ever span [0, 0.5], so `U` would have a different range on the lattice than
on real data — and β is swept on the lattice and then frozen and carried
forward. That is precisely the situation gap G4 exists to prevent. Letting
`max_atoms` follow the environment keeps every term in [0, 1] and `U`'s range
identical in both profiles.

**What this does not fix.** β still transfers from the lattice to real data.
That remains a *declared assumption*, in the same class as the
Wikipedia→conversation transfer for Stage D, and it should be stated that way in
the write-up rather than treated as free.

**Cost to reverse.** Re-run the β sweep and every Phase 3–4 comparison.

### 2.2 `max_atoms` is one field, used twice

It is simultaneously the `H` size limit and the denominator of the size term.
One field means the two cannot drift apart — a class of bug that would be
invisible in results.

### 2.3 `CandidateAtom` identity is `atom_id`, and only `atom_id`

**Decision.** `@dataclass(frozen=True, eq=False)` with hand-written `__eq__` /
`__hash__` over `atom_id`. `feat` is copied to float32 and made read-only at
construction; it serialises as `{dtype, shape, data}`.

**Why.** A dataclass-generated `__eq__` over a numpy field returns an *array*,
which makes the type unusable in the sets and dict keys the whole pipeline is
built on. Identity by `atom_id` is not a workaround — `atom_id` is already
content-derived, so two atoms sharing one *are* the same atom. The copy matters
because freezing a caller's array in place would be a surprising side effect of
construction. float32 because it round-trips exactly through JSON, halves the
log, and matches what the embedder emits.

**Cost to reverse.** Low now; high once Phase 9 features exist on disk.

### 2.4 `atom_id` gains a `label` discriminator

**Decision.** `atom_id(kind, refs, label="")` rather than the build plan's
`atom_id(kind, refs)`.

**Why.** Without it, two differently-typed edges between the same pair of
endpoints get the same id — and Phase 0's own exit criterion requires no
collisions across 10⁶ atoms. The default keeps the two-argument call working.

**Also fixed here.** Reference order is significant: `("a","b")` is not
`("b","a")`, because an edge is directed. A caller with genuinely unordered
references must sort before calling.

### 2.5 `Interval` is half-open `[start, end)` over epoch seconds

**Decision.** Half-open; `None` on either side is unbounded; `start == end` is
the *empty* interval, not an instant.

**Why it had to be settled now.** Phase 1's interval arithmetic, Phase 2's
temporal toys and Phase 6's `valid_during` edges all depend on the convention,
and none of them may re-derive it. Half-open is the standard bi-temporal
convention and composes with Zep's timestamp model. Epoch seconds rather than
ISO strings because this type exists to be compared and intersected;
`Turn.ts` stays ISO because it is an immutable human-auditable record.

**Consequence Stage A must honour.** A point-in-time expression has to be
widened to a window at its natural granularity — a date becomes
`[day, next_day)` — because a zero-width half-open interval contains nothing.

**Cost to reverse.** Phases 1, 2, 6 all touch it.

### 2.6 `Obligations.active_slots()` fixes what `coverage` divides by

**Decision.** The four requirement slots (`entity_anchor`, `value_type`,
`time_constraint`, `needs_source`) count. `aggregate` does not — it selects a
route rather than naming something the evidence must supply. A question with no
active slots scores coverage 1.0 by convention.

**Why.** `U`'s coverage term is "fraction of obligation slots addressed", which
is undefined until *which* slots count is written down. Better decided once here
than invented in Phase 1.

### 2.7 `r_fail_margin` — the CLAUDE.md §7 open item, automated

**Decision.** New config field, default 1e-3. The loader refuses any config
where `r_fail >= r_fail_margin * exp(beta * U_min)`.

**Why.** `p*(FAIL) <= r_fail / exp(beta*U_min)` whenever at least one valid
terminal exists, so this bounds `p*(FAIL)` below 0.1%. `CLAUDE.md` §7 lists
"re-check `r_fail` after the β sweep" as an open item and a one-line manual
task; a manual one-line check is exactly the kind that gets skipped. At β = 4 the
defaults pass with three orders of magnitude to spare; at β = 20 the config is
refused at load, which is the behaviour that was wanted.

### 2.8 `GraphSnapshot` gains `is_eligible(assertion_id)`

**Decision.** Seven protocol members, not six.

**Why.** `H`'s support sub-check (fix F9) cannot be written without it. Reading
an already-stored flag is deterministic and belongs in the checker; *computing*
entailment is learned and stays out. Unknown assertion ids are ineligible — a
proof may not lean on something the snapshot has never seen. The Phase-2 lattice
implements it trivially (no quarantine in synthetic instances).

**Note.** The protocol is not `@runtime_checkable`, because a protocol with a
non-method member cannot be used with `isinstance` on Python 3.11.
`implements_graph_snapshot()` does the structural check instead.

### 2.9 Ledger: capped meters require an active query scope

Spending a terminal check outside `query_scope()` raises. The metric is *per
query*, and a global counter cannot reconstruct it. Durability comes from a
`ledger.checkpoint` op in the same append-only log as everything else, rather
than a second file — so the crash-resume criterion is satisfied by machinery
that already had to exist.

### 2.10 Provenance and spans are non-empty by construction

`Edge` with no provenance and `Assertion` with no spans both raise. A schema that
permits an unsourced edge permits an unsourced proof. If Phase 6 finds a purely
structural edge type that genuinely has no span, that is the moment to amend the
rule deliberately — not to pass an empty list today.

### 2.11 Unknown YAML keys raise

The build plan's gotcha is that a *missing* key silently changes an experiment.
The mirror-image failure is a typo'd key that silently does nothing, so both are
rejected. `u_weights` overrides merge onto the defaults rather than replacing
them, so tuning one weight does not zero the other five.

---

## 3. Decisions forced by the multi-machine workflow

The team works across several laptops and Kaggle and hands folders around. That
is not a packaging detail — it changes what "the same run" means.

### 3.1 The venv is created per machine, never copied

A venv hard-codes the absolute path of the interpreter that built it; copied to
another laptop it installs, imports, and then fails in ways that look like code
bugs. It cannot work on Kaggle at all. So: `.venv/` is gitignored,
`requirements.txt` carries exact pins, and `scripts/bootstrap.{ps1,sh}` rebuild
it in about a minute. **Ship the repo, recreate the environment.**

### 3.2 Python 3.11, with an upper bound that bites early

`requires-python = ">=3.11,<3.13"`. This machine's default interpreter is 3.14,
for which PyTorch has no wheels — a venv built on it would install Phase 0
cleanly and fail at Phase 3. Failing at `pip install` is the cheaper failure.
3.11 is also what Kaggle runs.

### 3.3 Phase 0 has no ML dependency, and a test enforces it

`PyYAML`, `numpy`, `pytest`. Nothing else. That is what lets the suite run
unchanged in a bare Kaggle notebook before any heavy install, and it is why the
property-test generators are hand-rolled rather than using `hypothesis`.
`test_structure.py` fails if `torch` is imported at module scope anywhere.

### 3.4 All event-log I/O is binary

Text mode on Windows translates `\n` into `\r\n` on write. The same logical log
written on a laptop and on Kaggle would then differ byte for byte, and the digest
would diverge for identical content. Binary mode removes the translation.
`.gitattributes` normalises source and data files to LF for the same reason.

### 3.5 `EventLog.digest()` excludes timestamps

Wall clock genuinely differs between two runs of the same work, so byte-identity
of the file is the wrong claim. What must match is the `(seq, op, payload)`
stream. This gives a teammate a one-line way to check that their run produced the
same log as yours.

### 3.6 The manifest is split into `reproducibility` and `environment`

`reproducibility` (config hash, git SHA + dirty flag, seed, interpreter and
package versions) must match **across machines**; `environment` (hostname, OS,
CPU, GPU, clock) is expected to differ and is recorded so a machine-dependent
result can be traced. This is a stronger criterion than the build plan's "two
runs on one machine", and it is the one the team actually needs.

### 3.7 Everything that reaches bytes is sorted first

Python randomises string hashing per process, so `list(frozenset_of_strings)`
yields a different order on every launch. `ProofSet.to_dict()`,
`canon_set_hash` and `canonical_json` all sort. Without this, two machines
produce different bytes for the same set and nothing above is true.

### 3.8 The 192 MB of PDFs are gitignored

`INDEX.md`, `papers.csv`, `README.md` and `download_papers.ps1` stay tracked; the
PDFs do not, so the repo is 432 KB and clones instantly. One script refills the
folder per machine.

---

## 4. Verifying a handoff

```bash
python scripts/verify_handoff.py
```

Prints four fingerprints — config hash, log digest, graph digest, manifest
fingerprint. Two people on the same commit must see identical values on all
four. If they do not, that is a real bug, not noise.

---

## 5. What Phase 0 deliberately does not contain

No checker sub-checks, no utility terms, no lattice, no policy, no models, no
dataset code, no LLM calls, no database, no ANN index, no caching layer in
`ReplayGraphStore`, no learned anything. `ReplayGraphStore` is O(n) per snapshot
and carries a TODO rather than a cache — premature caching here is exactly the
over-engineering the boring-stack rule forbids.

---

## 6. Handoff to Phase 1

```python
from graft.schemas    import ProofSet, CandidateAtom, Obligations, Interval
from graft.graphstore import GraphSnapshot          # Protocol
from graft.ids        import canon_set_hash
from graft.ledger     import Ledger
from graft.config     import Config
```

All five are re-exported from `graft` directly and are covered by a test that
fails if the contract narrows. Phase 1 writes `H`, `U`, `R`, the masks and the
obligation parser against exactly these. If it needs a sixth, Phase 0 is
incomplete — fix it there, not by widening Phase 1.

Three things Phase 1 inherits as already-settled that were open before:
the half-open interval convention (§2.5), what `coverage` divides by (§2.6), and
`is_eligible` for `H`'s seventh sub-check (§2.8).

**Still open, and not blocking Phase 1:** the Phase-2 lattice generator needs to
emit a gold proof set and to be able to produce substitutable evidence, multiple
disjoint valid modes and colliding atoms — otherwise the collision audit measures
nothing and Gate 2 passes on a distribution that does not exercise the property
the thesis rests on. That belongs in the Phase 2 build plan.
