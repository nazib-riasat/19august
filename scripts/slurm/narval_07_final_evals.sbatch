#!/bin/bash
# GRAFT on Narval -- STEP 7: the FINAL evaluation rows, every one scored with the full-spec components.
# afterok: gate3 (for the full head). These are the numbers the paper quotes; everything before them
# was either a laptop run (10% head) or a component build.
#   LoCoMo   full read path, ceilings, full head        (1,986 q, ~1-2 h)
#   RAG      matched-budget re-run, unchanged (reader-only, no head) -- re-run here so both rows share a node/stack
#   2Wiki    full dev, 12,576 rows, full head             (~2-4 h)
#SBATCH --job-name=graft_final
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source "$(dirname "$0")/narval_common.sh"
HEAD="$GRAFT_ROOT/results/phase9_gate3/utility_head_full.pt"
[[ -f "$HEAD" ]] || { echo "full head missing at $HEAD -- step 4 did not finish"; exit 3; }
RES="$GRAFT_ROOT/results/final"; mkdir -p "$RES"

echo "== LoCoMo, full head, ceilings =="
srun python -u scripts/locomo_eval.py --head "$HEAD" --ceilings --run-dir artefacts/locomo_stageb --fresh \
    --out "$RES/locomo_eval.json" --rows "$RES/locomo_eval_rows.jsonl"

echo "== matched-budget RAG baseline (same node, same stack) =="
srun python -u scripts/locomo_rag_baseline.py --fresh \
    --out "$RES/locomo_rag_baseline.json" --rows "$RES/locomo_rag_baseline_rows.jsonl"

echo "== 2Wiki dev, ALL rows, full head =="
srun python -u scripts/wiki2_eval.py --questions 12576 --head "$HEAD" --fresh \
    --out "$RES/wiki2_eval_full.json" --rows "$RES/wiki2_eval_full_rows.jsonl"

echo "== paired comparison, GRAFT vs RAG, identical questions =="
srun python - <<'PY'
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
