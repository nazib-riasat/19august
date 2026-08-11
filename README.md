# GRAFT

Provenance-preserving temporal graph memory with a checker-conditioned
evidence-set learner.

- **The science:** [`GRAFT_RESEARCH_PLAN_v1.md`](GRAFT_RESEARCH_PLAN_v1.md) (v1.2). Wins any conflict.
- **The build plan:** [`GRAFT_EXECUTION_ARCHITECTURE_v1.md`](GRAFT_EXECUTION_ARCHITECTURE_v1.md) (v1.1), 12 phases.
- **Why things are the way they are:** [`CLAUDE.md`](CLAUDE.md) — decisions, what was rejected, and the cost of changing your mind.
- **Decisions taken while building:** [`PHASE0_DECISIONS.md`](PHASE0_DECISIONS.md) · [`PHASE1_DECISIONS.md`](PHASE1_DECISIONS.md) · [`PHASE2_DECISIONS.md`](PHASE2_DECISIONS.md).

Current state: **Phases 0, 1 and 2 complete** — scaffold and data contracts; the
deterministic core (`H`, `U`, `R`, masks, obligations, `d(s)`); and the
enumerable synthetic environment with its exact evaluator (ProofLattice, the
forward DP for `p_θ`, TV/JS/KL/FCS, the three frozen suites and the Gate-2
audits). Phase 3, the seven learners, is next.

`python scripts/verify_handoff.py --preset synthetic` prints the config, log and
**lattice** fingerprints that must match across machines before any Gate-2 number
is compared.

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
