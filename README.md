# GRAFT

Provenance-preserving temporal graph memory with a checker-conditioned
evidence-set learner.

- **The science:** [`GRAFT_RESEARCH_PLAN_v1.md`](GRAFT_RESEARCH_PLAN_v1.md) (v1.2). Wins any conflict.
- **The build plan:** [`GRAFT_EXECUTION_ARCHITECTURE_v1.md`](GRAFT_EXECUTION_ARCHITECTURE_v1.md) (v1.1), 12 phases. Per-phase specs: [Phase 5](GRAFT_PHASE5_BUILD.md) (built and run) · [Phase 6](GRAFT_PHASE6_BUILD.md) (code complete, twice audited) · [Phase 7](GRAFT_PHASE7_BUILD.md) (planned, §6 unsigned).
- **Why things are the way they are:** [`CLAUDE.md`](CLAUDE.md) — decisions, what was rejected, and the cost of changing your mind.
- **Decisions taken while building:** [`PHASE0_DECISIONS.md`](PHASE0_DECISIONS.md) · [`PHASE1_DECISIONS.md`](PHASE1_DECISIONS.md) · [`PHASE2_DECISIONS.md`](PHASE2_DECISIONS.md) · [`PHASE2_5_DECISIONS.md`](PHASE2_5_DECISIONS.md) · [`PHASE3_DECISIONS.md`](PHASE3_DECISIONS.md) · [`PHASE4_DECISIONS.md`](PHASE4_DECISIONS.md) · [`PHASE5_DECISIONS.md`](PHASE5_DECISIONS.md).
- **Which datasets, for what, at what cost:** [`DATASET_DECISION.md`](DATASET_DECISION.md) — every dataset by phase, plus a three-machine time table.
- **The data contract:** [`GATE0_CONTRACT.md`](GATE0_CONTRACT.md) — nine of ten items drafted, unsigned; Gate 1 is blocked on it.
- **Picking up a cold session:** [`chatcontext1.md`](chatcontext1.md) — current state, reading order, open items. Start there.

Current state: **Phases 0, 1, 2, 2.5 and 5 complete, Phase 4 Stage A complete**
— scaffold and data contracts; the deterministic core (`H`, `U`, `R`, masks,
obligations, `d(s)`); the enumerable synthetic environment with its exact
evaluator (ProofLattice, the forward DP for `p_θ`, TV/JS/KL/FCS, the three frozen
suites and the Gate-2 audits); the annotation-feasibility spike; the five
Tier-1 search methods with the Gate-3 harness; and Stage A ingestion
(`graft/ingest/` — extraction, grounding, NLI verification, the support gate,
the learned obligation parser, and the Gate-0 contract draft). **958 tests, all
passing.**

**Phase 5's GPU runs are done and its machine-measurable exit criteria are
met.** The G2 bakeoff froze the extractor on **candidate B** (grammar-
constrained; 1.7% parse failure against A's 23.3%) and the live pilot ran 248
turns end to end — `PHASE5_DECISIONS.md` §2. What remains is human: four audit
worksheets under `artefacts/phase5_pilot/` (span support, NLI agreement,
rung-3 boundaries, obligation slots) and Gate-0 item 8. Read
`PHASE5_DECISIONS.md` §7 for the audit record and §1.6/§2.1a for the two
instrument corrections before touching `graft/ingest/`.

**Phase 3 is code complete and uncalibrated.** Steps 1–5 and 7–10 of
[`GRAFT_PHASE3_BUILD.md`](GRAFT_PHASE3_BUILD.md) §4 are built and green: the ML
dependency boundary, the FL-GFN discharge (measured — 8,638/8,638 terminals off
terminal identity at every eligible β), the fix-F6 adapter, the policy and its
five heads, the trajectory sampler, the shared trainer, **all nine arms** (L1–L7,
L7b, GAFlowNet) and the Gate-2 harness with its hierarchical paired bootstrap.

A review on 12 Aug 2026 (plan revision **R6**) corrected three defects in the
*controls* — GAFlowNet's Eq. 4, LED-GFN's Eq. 5 and its decomposition-error
redistribution, and the L6/L7 capacity match, which was counting parameters no
gradient could reach. All three ran in the direction that flatters the proposed
method. `PHASE3_DECISIONS.md` §6 has the papers, the arithmetic and what was
*not* upheld.

**Step 6, the calibration gate, has not run.** `N` and β are therefore not
frozen, and no L6/L7/GAFlowNet result may be quoted — the plan's own ordering
constraint, so that β and `N` are fixed before the proposed method trains once.
Run it with:

```bash
python scripts/phase3_calibrate.py --out artefacts/phase3_calibration.json
```

Phase 3 needs torch — see `requirements-ml.txt`, and note that the CUDA build is
**not** incidental on a Blackwell GPU.

`python scripts/verify_handoff.py --preset synthetic` prints the config, log and
**lattice** fingerprints that must match across machines before any Gate-2 number
is compared.

**Phase 4 (`graft/setgen/search/`) is built and Stage A has run.** S1–S4 are
compared on the main suite under both relevance variants, beside the closed-form
`E[best-of-K | p*]` ceiling; S5's row waits on a Phase-3 checkpoint, which is
why the table is labelled **Stage A and is not Gate 3**. Read
[`PHASE4_DECISIONS.md`](PHASE4_DECISIONS.md) §1 first — three sub-decisions of
the ruled plan were overturned by measurement, including that `pcst_fast`'s
Windows wheel is silently wrong and S4 therefore uses an exact solver.

---

## Setting up on a new machine

Everything below assumes you have the repo folder. It takes about a minute.

**Windows**

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

**Linux / macOS**

```bash
bash scripts/bootstrap.sh
```

Then activate and check:

```bash
pytest -q
```

You need **Python 3.11** (3.12 works; 3.13+ does not — PyTorch has no wheels for
it, and Phase 3 would fail after everything else installed cleanly). Check what
you have with `py -0p` on Windows or `python3.11 --version` elsewhere.

### Never copy `.venv` between machines

A virtual environment records the absolute path of the interpreter that created
it. Copied to another laptop — or into Kaggle — it installs, imports, and then
fails in ways that look like code bugs. `.venv/` is gitignored for that reason.

**Ship the repo, recreate the environment.** `requirements.txt` carries exact
pins, so two people who bootstrap from the same commit get byte-identical
package versions, and every run records those versions in its manifest.

### Kaggle

Kaggle already runs Python 3.11 and gives you an environment; do not build a
venv there. Phase 0 has zero ML dependencies, so it runs on the stock image
unchanged:

```bash
!pip install -q -e /kaggle/working/CSE498R
```

Then `import graft` works in any cell. When Phase 3 adds torch, install
`requirements-ml.txt` on top — the Phase 0-4 code path never needs it.

---

## Working across machines

The things that make handing a folder around safe are already enforced, not
conventions to remember:

| Concern | How it is handled |
|---|---|
| Windows CRLF vs Linux LF | `.gitattributes` normalises to LF; the event log is written in binary mode so `\n` is never translated |
| "Did we run the same thing?" | `config_hash` covers the *resolved* config, not the YAML text — comments and whitespace cannot change an experiment's identity |
| "Did we get the same result?" | `EventLog.digest()` hashes the `(seq, op, payload)` stream, excluding timestamps. Same digest = same log |
| Machine-dependent results | `manifest.json` splits `reproducibility` (must match across machines) from `environment` (allowed to differ) |
| Uncommitted changes | the manifest records the git SHA **and** a dirty flag; a dirty run is visibly marked |
| Set/dict ordering | Python randomises string hashing per process, so everything that reaches bytes is sorted first |

Two runs of the same condition should produce identical `reproducibility` blocks
and identical log digests **on different machines**. If they do not, that is a
bug worth chasing, not noise.

---

## Layout

```
graft/
  config/        # frozen dataclasses, YAML overrides, config hash, presets
  schemas.py     # every dataclass; the only home for the data model
  ids.py         # content-derived ids; canon_set_hash
  eventlog.py    # append-only JSONL log, crash-safe
  graphstore.py  # GraphSnapshot protocol + replay-backed store
  ledger.py      # budget metering and enforcement
  runtime.py     # seeding, run manifest, artefact layout
  canonical.py   # the one way an object becomes bytes
  core/          # Phase 1  — the deterministic core
    checker.py     #   H: nine sub-checks, per-atom + set-level
    incremental.py #   validity across ADD/undo, agreeing with H by construction
    masks.py       #   legal_adds, stop_allowed, dead ends, Terminal
    obligations.py #   d(s), coverage, interval measure, the parser
    resolve.py     #   atom -> graph resolution, in one place
    utility.py     #   U and its six terms
    reward.py      #   R = 1[H]*exp(beta*U), and FAIL
  synth/         # Phase 2  — ProofLattice + exact evaluator
  setgen/        # Phase 3/4 — learners and search
  ingest/        # Phase 5  — Stage A
  graphbuild/    # Phase 6  — Stage B
  retrieve/      # Phase 7  — Stage C
  gate/          # Phase 8  — answerability gate
  reader/        # Phase 10 — Stage E
  baselines/     # Phase 11 — system adapters
  diagnostics/   # ceiling oracles and audits
  tests/
Research papers/ # 103-PDF library; INDEX.md and papers.csv are the index
```

## Configs

```python
from graft.config import load_config

cfg = load_config(preset="default")    # real-data profile
cfg = load_config(preset="synthetic")  # Phases 2-4 lattice profile
```

Defaults live in `graft/config/schema.py`; YAML may only override them, and an
unknown key raises rather than silently doing nothing.
