#!/bin/bash
# GRAFT on Narval -- STEP 2 (C): Stage A over LongMemEval scope c (200 q, 4,452 evidence turns), batched, resumable.
# Batch size comes from the probe (A100-40GB -> 32). Laptop measured 135 turns/h single-stream; expect ~1-3 h here.
# Re-submitting the same job continues where it stopped (append-only log, ingested_turn_ids).
#SBATCH --job-name=graft_ingest_lme
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source "$(dirname "$0")/narval_common.sh"
OUT="$GRAFT_ROOT/results/longmemeval_scope_c"      # fixed path, NOT per-job: this is the resumable log
mkdir -p "$OUT"

srun python -u scripts/longmemeval_ingest.py --questions 200 --batch-size "$EXTRACT_BATCH" --device "$DEVICE" \
    --run-dir "$OUT" --out "$OUT/longmemeval_ingest.json"

echo "== Stage B (stand-in constructor) over the new log =="
srun python -u scripts/locomo_stageb.py --source "$OUT/events.jsonl" --run-dir "$OUT/stageb" \
    --out "$OUT/stageb.json" --fresh 2>&1 | tail -4
echo "INGEST DONE -- $OUT"
