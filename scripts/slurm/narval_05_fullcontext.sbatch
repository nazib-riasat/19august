#!/bin/bash
# GRAFT on Narval -- STEP 5 (G): the full-context baseline -- the ENTIRE conversation (11k-22k tokens) in the prompt.
# **4 GPUs, one node:** GPUs 0-2 run three question shards of full-context; GPU 3 runs the uncapped D3/D4
# pretrain concurrently. Refuses full-context on a card the probe says cannot hold a 22k prefill
# (needs >= 32 GB free); A100-40GB qualifies. Nothing is truncated.
#SBATCH --job-name=graft_fullctx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source "$(dirname "$0")/narval_common.sh"
RES="$GRAFT_ROOT/results/baselines"; mkdir -p "$RES"
if [[ "$READER_CTX_OK" != "True" ]]; then
    echo "REFUSED: probe says this card cannot hold a 22k-token prefill (see $RUN_DIR/hw.json). Not truncating."; exit 2
fi
ROWS="$RES/locomo_fullcontext_baseline_rows.jsonl"; OUT="$RES/locomo_fullcontext_baseline.json"

echo "== G: full-context, 3 shards on GPUs 0-2  |  D3/D4 uncapped on GPU 3 =="
for i in 0 1 2; do
    srun --exact --ntasks=1 --gpus=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --output="$RES/fullctx_shard_$i.log" \
        python -u scripts/locomo_fullcontext_baseline.py --run-dir artefacts/locomo --shard "$i/3" \
        --rows "$ROWS.shard_$i.jsonl" --out "$OUT.shard_$i.json" --fresh &
done
srun --exact --ntasks=1 --gpus=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --output="$RES/d3d4.log" \
    python -u scripts/phase6_pretrain_d3d4.py --max-docs 0 --max-passages 0 --device cuda \
    --out "$RES/d3d4_pretrain_uncapped.json" &
wait
cat "$ROWS".shard_*.jsonl > "$ROWS"
echo "   merged $(wc -l < "$ROWS") full-context rows; finalizing"
python -u scripts/locomo_fullcontext_baseline.py --run-dir artefacts/locomo --rows "$ROWS" --out "$OUT" 2>&1 | tail -4
tail -4 "$RES/d3d4.log"
echo "BASELINES DONE -- $RES"
