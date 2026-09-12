#!/bin/bash
# GRAFT on Narval -- STEP 4: assemble Gate 3 from the 27 checkpoints (CPU, afterok:ladder).
#SBATCH --job-name=graft_gate3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

PROBE_FLAGS="--cpu"
export RUN_DIR="${SCRATCH:-$HOME/scratch}/graft/results/phase9_gate3"
source "$(dirname "$0")/narval_common.sh"

n=$(ls "$RUN_DIR/checkpoints"/*.pt 2>/dev/null | wc -l)
echo "checkpoints: $n / 27"
srun python -u scripts/phase9_gate3.py assemble --run-dir "$RUN_DIR" --head artefacts/utility_head.pt
echo "GATE 3 ASSEMBLED -- $RUN_DIR/gate3_real.json"
