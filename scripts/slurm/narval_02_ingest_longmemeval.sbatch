#!/bin/bash
# GRAFT on Narval -- STEP 2 (C): Stage A over LongMemEval scope c (200 q, 4,452 evidence turns).
# **4 GPUs, 4 shards, one node.** Each shard ingests every 4th question into its own log with its own
# verify pass; the shards then concatenate into one log. Measured on the pilot (12 Sep 2026): the
# merged graph is content-identical to a serial run. Batch size per GPU from the probe (A100 -> 32).
# Resumable: re-submitting continues each shard where it stopped.
#SBATCH --job-name=graft_ingest_lme
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source "$(dirname "$0")/narval_common.sh"
OUT="$GRAFT_ROOT/results/longmemeval_scope_c"; mkdir -p "$OUT"
N=${SLURM_NTASKS:-4}

echo "== ingest: $N shards on $N GPUs, batch $EXTRACT_BATCH each =="
for i in $(seq 0 $((N-1))); do
    srun --exact --ntasks=1 --gpus=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --output="$OUT/shard_$i.log" \
        python -u scripts/longmemeval_ingest.py --questions 200 --shard "$i/$N" --batch-size "$EXTRACT_BATCH" \
        --device cuda --run-dir "$OUT" --out "$OUT/shard_$i.json" &
done
wait
echo "== merge shards =="
python scripts/longmemeval_ingest.py --merge --run-dir "$OUT"

echo "== Stage B (stand-in constructor) over the merged log =="
srun --ntasks=1 python -u scripts/locomo_stageb.py --source "$OUT/events.jsonl" --run-dir "$OUT/stageb" \
    --out "$OUT/stageb.json" --fresh 2>&1 | tail -4
echo "INGEST DONE -- $OUT"
