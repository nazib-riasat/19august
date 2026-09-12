#!/bin/bash
# GRAFT on Narval -- STEP 4: distil the utility head at FULL spec, then assemble Gate 3. CPU, afterok:ladder.
#
# The laptop head (artefacts/utility_head.pt, rho 0.756) was distilled on 200 examples per corpus --
# a 10% slice of pins.SUBSET's 2,000. That was a laptop compromise, not a design choice, so it is
# redone here on the full subset before any Gate-3 row or final eval is scored against it.
#SBATCH --job-name=graft_gate3
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

PROBE_FLAGS="--cpu"
export RUN_DIR="${SCRATCH:-$HOME/scratch}/graft/results/phase9_gate3"
source "$(dirname "$0")/narval_common.sh"

echo "== distil the utility head on the full pinned subset (2,000 examples per corpus) =="
srun python -u scripts/train_head.py --examples 2000 --rollouts 40 \
    --out "$RUN_DIR/utility_head_full.pt" --report "$RUN_DIR/utility_head_full.json"
python -c "import json,sys; r=json.load(open(sys.argv[1])); print('   full head: rows', r.get('rows'), 'dev rho', r.get('dev_spearman'))" "$RUN_DIR/utility_head_full.json"

n=$(ls "$RUN_DIR/checkpoints"/*.pt 2>/dev/null | wc -l)
echo "== assemble Gate 3 ($n / 27 checkpoints) under the FULL head =="
srun python -u scripts/phase9_gate3.py assemble --run-dir "$RUN_DIR" --head "$RUN_DIR/utility_head_full.pt"
echo "GATE 3 ASSEMBLED -- $RUN_DIR/gate3_real.json ; head -> $RUN_DIR/utility_head_full.pt"
