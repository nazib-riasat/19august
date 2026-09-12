#!/bin/bash
# GRAFT on Narval -- STEP 5 (G): the full-context baseline -- the ENTIRE conversation (11k-22k tokens) in the prompt.
# Refuses on a card the probe says cannot hold a 22k prefill (needs >= 32 GB free). A100-40GB qualifies.
# Also the D3/D4 heads UNCAPPED (the laptop capped Re-DocRED to 600 docs and TORQUE to 800 passages).
# The 2Wiki row moved to step 7 so it is scored with the full-spec head.
#SBATCH --job-name=graft_fullctx
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source "$(dirname "$0")/narval_common.sh"
RES="$GRAFT_ROOT/results/baselines"; mkdir -p "$RES"

echo "== G: full-context baseline (whole conversation, no truncation) =="
if [[ "$READER_CTX_OK" != "True" ]]; then
    echo "REFUSED: probe says this card cannot hold a 22k-token prefill (see $RUN_DIR/hw.json). Not truncating."; exit 2
fi
srun python -u scripts/locomo_fullcontext_baseline.py --run-dir artefacts/locomo \
    --out "$RES/locomo_fullcontext_baseline.json" --rows "$RES/locomo_fullcontext_baseline_rows.jsonl"

echo "== D3/D4 uncapped (frozen bge vectors, full DialogRE / Re-DocRED / TORQUE) =="
srun python -u scripts/phase6_pretrain_d3d4.py --max-docs 0 --max-passages 0 --device "$DEVICE" \
    --out "$RES/d3d4_pretrain_uncapped.json" 2>&1 | tail -5
echo "BASELINES DONE -- $RES"
