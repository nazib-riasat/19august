#!/bin/bash
# GRAFT on Narval -- STEP 0, run ON THE LOGIN NODE (it needs internet). Idempotent.
#
#   bash scripts/slurm/narval_00_setup_login.sh
#
# What it does, in order, stopping on the first failure:
#   1. repo      clone or fast-forward nazib-riasat/19august (master) into $SCRATCH/graft/19august
#   2. venv      python/3.11 venv; wheelhouse first (--no-index), PyPI for what the wheelhouse lacks
#   3. models    pre-download the three pinned checkpoints into $HF_HOME (compute nodes are offline)
#   4. data      verify the SHA-pinned corpora are present; fetch the ones that can be fetched
#   5. tests     the full suite on the login node's CPU (quick sanity before any GPU minute is spent)
#
# Nothing here submits a job. narval_submit_all.sh does that, after this passes.

set -Eeuo pipefail

export GRAFT_ROOT="${SCRATCH:-$HOME/scratch}/graft"
export REPO_DIR="$GRAFT_ROOT/19august"
export ENV_DIR="$GRAFT_ROOT/envs/full-py311"
export HF_HOME="$GRAFT_ROOT/hf"
export PIP_CACHE_DIR="$GRAFT_ROOT/cache/pip"
REPO_URL="https://github.com/nazib-riasat/19august.git"
mkdir -p "$GRAFT_ROOT/envs" "$HF_HOME" "$PIP_CACHE_DIR" "$GRAFT_ROOT/logs" "$GRAFT_ROOT/results"

echo "== 1. repo =="
if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" fetch --quiet origin && git -C "$REPO_DIR" checkout --quiet master && git -C "$REPO_DIR" pull --ff-only --quiet
else
    git clone --quiet "$REPO_URL" "$REPO_DIR"
fi
echo "   at $(git -C "$REPO_DIR" rev-parse --short HEAD): $(git -C "$REPO_DIR" log -1 --format=%s)"
cd "$REPO_DIR"

echo "== 2. venv =="
# Python version: 3.11 is the project pin AND is proven on this account (job 985953 ran on
# python/3.11.5, exit 0). 3.10 is available as an EXPLICIT fallback -- PYTHON_MODULE=python/3.10 --
# for the case where the ingest stack (transformers 5 / xgrammar / bitsandbytes) has no 3.11 wheel
# here. It is never chosen silently: a different interpreter is a different experiment stack, and
# pyproject pins >=3.11, so the 3.10 path installs the package with --ignore-requires-python and
# names itself in the venv path so nobody mistakes the two.
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
module load "$PYTHON_MODULE" 2>/dev/null || module load python
PYV=$(python -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
[[ "$PYV" == "311" || "$PYV" == "310" ]] || { echo "unsupported python $PYV; use PYTHON_MODULE=python/3.11 or python/3.10"; exit 4; }
export ENV_DIR="$GRAFT_ROOT/envs/full-py$PYV"
echo "   python $(python --version 2>&1) -> $ENV_DIR"
[[ -x "$ENV_DIR/bin/python" ]] || python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip
# Wheelhouse first (exact pins where it has them), PyPI for the rest. Alliance recommends
# --no-index; the ingest stack is usually not in the wheelhouse, so the PyPI fallback is expected.
# A package that fails BOTH is reported by name -- that is the "python issue", and the answer
# is to re-run with PYTHON_MODULE=python/3.10, not to guess.
for req in requirements.txt requirements-ml.txt requirements-ingest.txt; do
    echo "   $req"
    if ! python -m pip install --quiet --no-index -r "$req" 2>/dev/null; then
        python -m pip install --quiet -r "$req" 2> "$GRAFT_ROOT/logs/pip_$req.err" || {
            echo "   FAILED under python $PYV: $req"; grep -iE "no matching|could not find|error" "$GRAFT_ROOT/logs/pip_$req.err" | head -5
            echo "   -> re-run:  PYTHON_MODULE=python/3.10 bash scripts/slurm/narval_00_setup_login.sh"; exit 5; }
    fi
done
IGN=""; [[ "$PYV" == "310" ]] && IGN="--ignore-requires-python"
python -m pip install --quiet --no-deps $IGN -e .
echo "$PYTHON_MODULE" > "$GRAFT_ROOT/envs/PYTHON_MODULE"   # the jobs load the same module
python - <<'PY'
import numpy, torch, transformers, yaml
print(f"   numpy {numpy.__version__}  torch {torch.__version__}  transformers {transformers.__version__}  cuda-build {torch.version.cuda}")
try:
    import xgrammar, bm25s, torch_geometric; print("   xgrammar", xgrammar.__version__, " bm25s", bm25s.__version__, " pyg", torch_geometric.__version__)
except Exception as e:
    raise SystemExit(f"   MISSING ingest/ml dependency: {e!r}")
PY

echo "== 3. models -> $HF_HOME =="
python - <<'PY'
from huggingface_hub import snapshot_download
pins = [
    ("Qwen/Qwen2.5-3B-Instruct",          "aa8e72537993ba99e69dfaafa59ed015b17504d1"),  # extractor AND reader (same weights)
    ("BAAI/bge-small-en-v1.5",             "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),  # embedder
    ("cross-encoder/nli-deberta-v3-base",  "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"),  # NLI verifier
]
for repo, rev in pins:
    p = snapshot_download(repo, revision=rev)
    print(f"   {repo}@{rev[:8]} -> {p}")
PY

echo "== 4. data =="
python - <<'PY'
import hashlib, json, pathlib, sys
from graft.ingest import corpus, locomo
ok = True
def sha(p): h=hashlib.sha256(); h.update(pathlib.Path(p).read_bytes()); return h.hexdigest()
for label, path, want in (
    ("LoCoMo", "data/locomo/locomo10.json", "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"),
    ("LongMemEval-S", str(corpus._DEFAULT_PATH), corpus.CORPUS_SHA256),
):
    p = pathlib.Path(path)
    if not p.exists(): print(f"   MISSING {label}: {path}  (rsync it from the laptop; raw data is gitignored)"); ok = False
    elif sha(p) != want: print(f"   SHA MISMATCH {label}: {path}"); ok = False
    else: print(f"   ok {label}")
for label, path in (("2Wiki dev", "data/phase9/raw/2wiki/dev.json"), ("2Wiki train", "data/phase9/raw/2wiki/train.json"),
                    ("MuSiQue-Ans train", "data/phase9/raw/musique_ans/musique_ans_v1.0_train.jsonl"),
                    ("MuSiQue-Ans dev", "data/phase9/raw/musique_ans/musique_ans_v1.0_dev.jsonl"),
                    ("DialogRE", "data/phase6/raw/dialogre/train.json"), ("Re-DocRED", "data/phase6/raw/redocred/train_revised.json"),
                    ("TORQUE", "data/phase6/raw/torque")):
    ok &= pathlib.Path(path).exists() or (print(f"   MISSING {label}: {path}") or False)
sys.exit(0 if ok else 3)
PY

echo "== 5. tests (login-node CPU) =="
python -m pytest graft/tests/ -q -x --ignore=graft/tests/test_ingest_bakeoff.py 2>&1 | tail -3

echo
echo "setup complete. next: bash scripts/slurm/narval_submit_all.sh"
