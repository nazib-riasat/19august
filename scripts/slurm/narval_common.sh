#!/bin/bash
# Shared setup for every GRAFT job on Narval. Sourced, never executed.
#
# Layout (inherited from slurm_phase3_narval.sh, which ran as job 985953):
#   $SCRATCH/graft/
#     19august/            the repo (rsync'd or cloned from nazib-riasat/19august)
#     envs/full-py311/     the venv (Alliance wheelhouse first, PyPI fallback)
#     hf/                  HF_HOME -- models pre-downloaded on the login node
#     cache/pip/
#     logs/
#     results/<jobname>_<jobid>/   one RUN_DIR per job; nothing overwrites anything
#
# Every job: probe the node, write hw.json into RUN_DIR, refuse if the probe
# refuses (no CUDA on a GPU job, or a card without bf16), and export the knobs.

set -Eeuo pipefail

export GRAFT_ROOT="${SCRATCH:-$HOME/scratch}/graft"
export REPO_DIR="$GRAFT_ROOT/19august"
export ENV_DIR="$GRAFT_ROOT/envs/full-py311"
export HF_HOME="$GRAFT_ROOT/hf"
export PIP_CACHE_DIR="$GRAFT_ROOT/cache/pip"
export LOG_DIR="$GRAFT_ROOT/logs"
export RUN_DIR="${RUN_DIR:-$GRAFT_ROOT/results/${SLURM_JOB_NAME:-local}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

# Compute nodes have no internet: everything HF must already be in $HF_HOME.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module load python/3.11 2>/dev/null || module load python
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "GRAFT job ${SLURM_JOB_NAME:-?}  id ${SLURM_JOB_ID:-?}  node ${SLURMD_NODENAME:-?}  account ${SLURM_JOB_ACCOUNT:-?}"
echo "started $(date --iso-8601=seconds)   RUN_DIR $RUN_DIR"
trap 'echo "exit=$? at $(date --iso-8601=seconds)  RUN_DIR $RUN_DIR"' EXIT

# --- probe -> knobs -----------------------------------------------------------
PROBE_FLAGS="${PROBE_FLAGS:-}"
python scripts/slurm/hw_probe.py $PROBE_FLAGS --out "$RUN_DIR/hw.json" > "$RUN_DIR/hw.txt" || {
    echo "PROBE REFUSED -- see $RUN_DIR/hw.txt"; cat "$RUN_DIR/hw.txt"; exit 2; }
export EXTRACT_BATCH=$(python scripts/slurm/hw_probe.py $PROBE_FLAGS --knob extract_batch)
export EMBED_BATCH=$(python scripts/slurm/hw_probe.py $PROBE_FLAGS --knob embed_batch)
export READER_CTX_OK=$(python scripts/slurm/hw_probe.py $PROBE_FLAGS --knob reader_ctx_ok)
export DEVICE=$(python scripts/slurm/hw_probe.py $PROBE_FLAGS --knob device)
echo "knobs: extract_batch=$EXTRACT_BATCH embed_batch=$EMBED_BATCH reader_ctx_ok=$READER_CTX_OK device=$DEVICE"
