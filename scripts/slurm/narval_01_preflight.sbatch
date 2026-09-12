#!/bin/bash
# GRAFT on Narval -- STEP 1: preflight on a real GPU node. ~20 min. Nothing full-scale runs until this passes.
#   probe -> full test suite on-node -> 2-question smoke of every GPU pipeline with the real reader/extractor.
#SBATCH --job-name=graft_preflight
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

source "$(dirname "$0")/narval_common.sh"
SMK="$RUN_DIR/smoke"; mkdir -p "$SMK"

echo "== tests on-node =="
srun python -m pytest graft/tests/ -q -x 2>&1 | tail -3

echo "== smoke: LoCoMo eval (real reader, 3 q) =="
srun python -u scripts/locomo_eval.py --head artefacts/utility_head.pt --questions 3 --conversations 1 \
    --run-dir artefacts/locomo_stageb --out "$SMK/locomo.json" --rows "$SMK/locomo_rows.jsonl" --fresh 2>&1 | tail -3

echo "== smoke: full-context 22k on this card (2 q, largest conversation) =="
if [[ "$READER_CTX_OK" == "True" ]]; then
    srun python -u scripts/locomo_fullcontext_baseline.py --conv-offset 2 --conversations 1 --questions 2 \
        --out "$SMK/fc.json" --rows "$SMK/fc_rows.jsonl" --fresh 2>&1 | tail -3
else
    echo "   probe says reader_ctx_ok=False on this card -- full-context job will refuse; see hw.json"
fi

echo "== smoke: LongMemEval ingest (extractor+NLI, 1 question, batch $EXTRACT_BATCH) =="
srun python -u scripts/longmemeval_ingest.py --questions 1 --batch-size "$EXTRACT_BATCH" --device "$DEVICE" \
    --run-dir "$SMK/lme" --out "$SMK/lme.json" 2>&1 | tail -3

echo "== smoke: 2Wiki eval (real reader, 4 q) =="
srun python -u scripts/wiki2_eval.py --questions 4 --out "$SMK/w2.json" --rows "$SMK/w2_rows.jsonl" --fresh 2>&1 | tail -2

echo "== smoke: ladder train task (fixtures) =="
srun python scripts/phase9_gate3.py train --arm l7_checker_led --seed 13 --smoke --run-dir "$SMK/p9" 2>&1 | tail -1

echo "PREFLIGHT PASSED -- $RUN_DIR"
