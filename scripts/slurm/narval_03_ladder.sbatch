#!/bin/bash
# GRAFT on Narval -- STEP 3 (I): the Phase-9 policy ladder as a 27-task array (9 arms x 3 seeds), CPU.
# Each task trains ONE arm x seed at N_real = 200,000 (~2 h at the laptop's 30 traj/s; faster here) and
# writes one checkpoint. Wall-clock = the slowest task, not 54 h. Uses the CPU allocation (def-*_cpu).
# Task 0 builds the pool cache first; the others wait on it via a lock file so 27 tasks do not all embed 4,000 pools.
#SBATCH --job-name=graft_ladder
#SBATCH --array=0-26
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL

PROBE_FLAGS="--cpu"
export RUN_DIR="${SCRATCH:-$HOME/scratch}/graft/results/phase9_gate3"   # shared by all tasks + assemble
source "$(dirname "$0")/narval_common.sh"

ARMS=(l4_tb l5_subtb l6_led l7_checker_led l7b_aux gaflownet l1_supervised l2_imitation l3_grpo)
SEEDS=(13 42 7)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "task $SLURM_ARRAY_TASK_ID -> arm $ARM seed $SEED"

# Pool cache: task 0 builds it (real embedder, CPU); everyone else waits up to 90 min for the marker.
CACHE="$RUN_DIR/cache"; MARK="$CACHE/.pools_ready"
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    rm -f "$MARK"
    srun python - <<'PY'
import sys, pathlib, os
sys.path.insert(0, os.getcwd())
import importlib.util
spec = importlib.util.spec_from_file_location("p9", "scripts/phase9_gate3.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from graft.setgen import pins
run = pathlib.Path(os.environ["RUN_DIR"])
for split, per in (("train", int(pins.SUBSET["train_per_corpus"])), ("dev", int(pins.SUBSET["dev_per_corpus"]))):
    ex, strata = m.load_pools(split, smoke=False, per_corpus=per, cache_dir=run / "cache")
    print(split, len(ex), strata)
PY
    touch "$MARK"
else
    for i in $(seq 1 180); do [[ -f "$MARK" ]] && break; sleep 30; done
    [[ -f "$MARK" ]] || { echo "pool cache never appeared; task 0 failed?"; exit 3; }
fi

srun python -u scripts/phase9_gate3.py train --arm "$ARM" --seed "$SEED" --device cpu --run-dir "$RUN_DIR"
echo "LADDER TASK DONE -- $ARM seed $SEED"
