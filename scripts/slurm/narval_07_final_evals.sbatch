#!/bin/bash
# GRAFT on Narval -- STEP 7: the FINAL evaluation rows, every one scored with the full-spec components.
# **4 GPUs, 4 question shards per eval, one node.** Rows files are per-question, so shards merge by
# concatenation; the runner is then re-invoked without --fresh, finds every question done, and writes the
# artefact through its own report path (no separate merge code to get wrong).
#   LoCoMo   full read path, ceilings, full head        (1,986 q)
#   RAG      matched-budget re-run (reader-only; same node, same stack as the GRAFT row)
#   2Wiki    full dev, 12,576 rows, full head
#SBATCH --job-name=graft_final
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source "$(dirname "$0")/narval_common.sh"
HEAD="$GRAFT_ROOT/results/phase9_gate3/utility_head_full.pt"
[[ -f "$HEAD" ]] || { echo "full head missing at $HEAD -- step 4 did not finish"; exit 3; }
RES="$GRAFT_ROOT/results/final"; mkdir -p "$RES"
N=${SLURM_NTASKS:-4}

# sharded <name> <rows> <out> <runner args...>: N shards in parallel, then merge + finalize.
sharded() {
    local name="$1" rows="$2" out="$3"; shift 3
    echo "== $name: $N shards =="
    for i in $(seq 0 $((N-1))); do
        srun --exact --ntasks=1 --gpus=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --output="$RES/${name}_shard_$i.log" \
            python -u "$@" --shard "$i/$N" --rows "$rows.shard_$i.jsonl" --out "$out.shard_$i.json" --fresh &
    done
    wait
    cat "$rows".shard_*.jsonl > "$rows"
    echo "   merged $(wc -l < "$rows") rows; finalizing through the runner's own report path"
    python -u "$@" --rows "$rows" --out "$out" 2>&1 | tail -6     # no --fresh: all done, writes the artefact
}

sharded locomo "$RES/locomo_eval_rows.jsonl" "$RES/locomo_eval.json" \
    scripts/locomo_eval.py --head "$HEAD" --ceilings --run-dir artefacts/locomo_stageb
sharded rag "$RES/locomo_rag_baseline_rows.jsonl" "$RES/locomo_rag_baseline.json" \
    scripts/locomo_rag_baseline.py
sharded wiki2 "$RES/wiki2_eval_full_rows.jsonl" "$RES/wiki2_eval_full.json" \
    scripts/wiki2_eval.py --questions 12576 --head "$HEAD"

echo "== paired comparison, GRAFT vs RAG, identical questions =="
python - <<'PY'
import json, os, numpy as np
from graft.reader.parse import token_f1
res = os.environ["RES"]
def rows(p): return {json.loads(l)["question_id"]: json.loads(l) for l in open(p, encoding="utf-8")}
g, r = rows(f"{res}/locomo_eval_rows.jsonl"), rows(f"{res}/locomo_rag_baseline_rows.jsonl")
ids = [q for q in g if q in r and not g[q]["adversarial"]]
f1 = lambda x: token_f1(x["answer_text"] or "", str(x["gold"])) if x["outcome"] == "answer" else 0.0
d = np.array([f1(g[q]) - f1(r[q]) for q in ids]); rng = np.random.default_rng(13)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
out = {"n": len(ids), "graft_f1": float(np.mean([f1(g[q]) for q in ids])), "rag_f1": float(np.mean([f1(r[q]) for q in ids])),
       "diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
       "p_le_0": float((boot <= 0).mean())}
json.dump(out, open(f"{res}/graft_vs_rag_paired.json", "w"), indent=1); print(out)
PY
echo "FINAL EVALS DONE -- $RES"
