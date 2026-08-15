#!/bin/bash
# GRAFT Phase 3: calibration gate followed by the admissible Gate-2 matrix.
#
# Submit from a Narval login node:
#   sbatch slurm_phase3_narval.sh
#
# The account, partition, GPU type, and cluster user are inherited from the
# supplied demo.sh. Phase 3 itself is CPU-oriented: its sampler is NumPy and
# scripts/phase3_gate2.py constructs TrainSpec(device="cpu"). The A100 request
# is therefore retained only because the supplied account/partition is
# GPU-specific; it is not represented as useful Phase-3 compute.

#SBATCH --account=def-loutfouz_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=graft_phase3
#SBATCH --output=graft_phase3_%j.out
#SBATCH --error=graft_phase3_%j.err
#SBATCH --mail-user=nazib.riasat@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -Eeuo pipefail

EXPECTED_USER="nahian26"
EXPECTED_ACCOUNT="def-loutfouz_cpu"
REPOSITORY_URL="https://github.com/nazib-riasat/GRAFT_System.git"
GRAFT_REF="${GRAFT_REF:-7cf6a38}"

if [[ "${USER:-}" != "$EXPECTED_USER" ]]; then
    echo "ERROR: this job is configured for user $EXPECTED_USER, not ${USER:-unset}." >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ACCOUNT:-}" && "$SLURM_JOB_ACCOUNT" != "$EXPECTED_ACCOUNT" ]]; then
    echo "ERROR: expected Slurm account $EXPECTED_ACCOUNT, got $SLURM_JOB_ACCOUNT." >&2
    exit 1
fi

USER_SCRATCH="${SCRATCH:-/home/nahian26/scratch}"
GRAFT_ROOT="$USER_SCRATCH/graft"
REPO_DIR="$GRAFT_ROOT/GRAFT_System"
ENV_DIR="$GRAFT_ROOT/envs/phase3-py311"
CACHE_DIR="$GRAFT_ROOT/cache/pip"
LOG_DIR="$GRAFT_ROOT/logs"
RUN_DIR="$GRAFT_ROOT/results/phase3/${SLURM_JOB_ID}"
CALIBRATION_FILE="$RUN_DIR/phase3_calibration.json"
REPORT_FILE="$RUN_DIR/gate2_report.json"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"

mkdir -p "$GRAFT_ROOT" "$GRAFT_ROOT/envs" "$CACHE_DIR" "$LOG_DIR" \
    "$RUN_DIR" "$CHECKPOINT_DIR"

# Preserve a copy under scratch even though Slurm also writes the two files
# named by --output and --error in the submission directory.
exec > >(tee -a "$LOG_DIR/phase3_${SLURM_JOB_ID}.out") \
     2> >(tee -a "$LOG_DIR/phase3_${SLURM_JOB_ID}.err" >&2)

trap 'status=$?; echo "Phase 3 job exit=$status at $(date --iso-8601=seconds)"; echo "Results: $RUN_DIR"' EXIT

echo "GRAFT Phase 3"
echo "Job:       $SLURM_JOB_ID"
echo "User:      $USER"
echo "Account:   ${SLURM_JOB_ACCOUNT:-$EXPECTED_ACCOUNT}"
echo "Node:      ${SLURMD_NODENAME:-unknown}"
echo "Git ref:   $GRAFT_REF"
echo "Scratch:   $GRAFT_ROOT"
echo "Started:   $(date --iso-8601=seconds)"

module --force purge
module load python

echo "Python module: $(python --version 2>&1)"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'

if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone --branch master --single-branch "$REPOSITORY_URL" "$REPO_DIR"
else
    if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
        echo "ERROR: $REPO_DIR has local changes; refusing to overwrite them." >&2
        exit 1
    fi
    git -C "$REPO_DIR" fetch origin master
fi

git -C "$REPO_DIR" checkout --detach "$GRAFT_REF"
cd "$REPO_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    python -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
export PIP_CACHE_DIR="$CACHE_DIR"

python -m pip install --upgrade pip setuptools wheel
# Alliance Python modules expose the site wheelhouse. Exact project pins are
# required; failure is preferable to silently changing the experiment stack.
python -m pip install --no-index -r requirements.txt
python -m pip install --no-index torch
python -m pip install --no-deps -e .

python -c 'import numpy, torch, yaml; print("numpy", numpy.__version__); print("torch", torch.__version__); print("PyYAML", yaml.__version__)'

echo "Running Phase-3 regression tests..."
srun --cpu-bind=cores python -m pytest -q \
    graft/tests/test_flgfn_probe.py \
    graft/tests/test_setgen_calibration.py \
    graft/tests/test_setgen_convergence.py \
    graft/tests/test_setgen_gate2.py \
    graft/tests/test_setgen_learners.py \
    graft/tests/test_setgen_spine.py \
    graft/tests/test_synth_audits.py \
    graft/tests/test_synth_enumerate.py \
    graft/tests/test_synth_exact.py \
    graft/tests/test_synth_lattice.py \
    graft/tests/test_synth_policies.py

echo "Verifying the synthetic handoff..."
srun --cpu-bind=cores python scripts/verify_handoff.py --preset synthetic

echo "Running the non-quick Phase-3 calibration gate..."
srun --cpu-bind=cores python -u scripts/phase3_calibrate.py \
    --out "$CALIBRATION_FILE" \
    --device cpu

# phase3_calibrate.py already exits non-zero unless it adopts N and beta. Check
# the artefact independently before allowing any proposed arm to train.
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); r=json.loads(p.read_text()); assert not r.get("quick") and r.get("verdict") == "adopted" and r.get("adopted"), r' "$CALIBRATION_FILE"

echo "Checking the admissible Gate-2 matrix without training..."
srun --cpu-bind=cores python scripts/phase3_gate2.py \
    --calibration "$CALIBRATION_FILE" \
    --out "$REPORT_FILE" \
    --checkpoints "$CHECKPOINT_DIR" \
    --dry-run

echo "Running the full nine-arm, three-seed Gate-2 matrix..."
srun --cpu-bind=cores python -u scripts/phase3_gate2.py \
    --calibration "$CALIBRATION_FILE" \
    --out "$REPORT_FILE" \
    --checkpoints "$CHECKPOINT_DIR"

python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); r=json.loads(p.read_text()); v=r["verdict"]; assert v["outcome"] != "inadmissible", v; print("Gate-2 outcome:", v["outcome"]); print("Contribution 3 supported:", v["contribution_3_supported"])' "$REPORT_FILE"

echo "Phase-3 computation completed."
echo "Calibration: $CALIBRATION_FILE"
echo "Report:      $REPORT_FILE"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Next manual step: review the report and transcribe the adopted N, beta, and Gate-2 outcome into the Phase-3 decision documents."
