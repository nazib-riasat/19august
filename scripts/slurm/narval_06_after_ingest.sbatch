#!/bin/bash
# GRAFT on Narval -- STEP 6 (D, E, Gate 1 at volume): everything that needed the LongMemEval graph. GPU, afterok:ingest.
#   D  LongMemEval eval through the full read path (the LoCoMo runner, pointed at the new graph)
#   E  conversational answerability gate trained on the new corpus' contrast pairs
#   Gate 1 at volume: construct items over the scope-c graph, label with the assistant's labels? -- NO: items
#      only. Gate 1 needs labels for the new items; this job EXPORTS the batch and stops. Labelling is a
#      separate decision (see PHASE6_DECISIONS.md §9.5).
#SBATCH --job-name=graft_after_ingest
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

source "$(dirname "$0")/narval_common.sh"
export LME="$GRAFT_ROOT/results/longmemeval_scope_c"
export RES="$GRAFT_ROOT/results/longmemeval_eval"; mkdir -p "$RES"
[[ -f "$LME/stageb/events.jsonl" ]] || { echo "no Stage-B log at $LME/stageb -- ingest job did not finish"; exit 3; }

echo "== D: LongMemEval eval (full read path, real reader) =="
# The LoCoMo runner is corpus-specific in its question loader; the LongMemEval track uses the
# Phase-7 retrieval runner's corpus join. If a dedicated runner does not exist yet, this step
# records that fact instead of faking a row.
if python -c "import scripts.longmemeval_eval" 2>/dev/null || [[ -f scripts/longmemeval_eval.py ]]; then
    srun python -u scripts/longmemeval_eval.py --head artefacts/utility_head.pt --ceilings --run-dir "$LME/stageb" \
        --out "$RES/longmemeval_eval.json" --rows "$RES/longmemeval_eval_rows.jsonl"
else
    echo "   scripts/longmemeval_eval.py not present: D deferred -- graph is built at $LME/stageb, runner is the next build item"
fi

echo "== E: conversational gate (Phase 8 Stage B) =="
if grep -q -- "--corpus" scripts/phase8_gate.py 2>/dev/null; then
    srun python -u scripts/phase8_gate.py --corpus longmemeval --run-dir "$LME/stageb" --out "$RES/gate_conversational.json" 2>&1 | tail -5 \
        || echo "   phase8_gate.py has no conversational track yet: E deferred, contrast pairs are derivable from $LME/stageb"
else
    echo "   phase8_gate.py has no --corpus flag: E deferred, contrast pairs are derivable from $LME/stageb"
fi

echo "== Stage-C scorer retrained on scope c (the laptop version saw 8 questions) =="
srun python - <<'PY'
import json, os, pathlib, random, time, torch
from graft.config import Config
from graft.eventlog import EventLog
from graft.graphstore import ReplayGraphStore
from graft.graphbuild.embed import Embedder
from graft.ingest import corpus as corpus_mod
from graft.retrieve import recall
from graft.retrieve.pool import eligible_nodes, uncapped_pool
from graft.retrieve.scorer import atom_features, pool_adjacency, build_scorer, train_scorer
lme = pathlib.Path(os.environ["LME"]); res = pathlib.Path(os.environ["RES"])
cfg = Config(); snap = ReplayGraphStore(EventLog.open(lme / "stageb" / "events.jsonl", fsync=False)).at()
instances = corpus_mod.question_index(corpus_mod.load_corpus())
qs = sorted({t.conv_id for t in snap._turns.values()} & set(instances))
emb = Embedder(cache_dir=pathlib.Path("artefacts/phase6/embed_cache"))
def prep(qid):
    nodes = eligible_nodes(snap, qid)
    if not nodes: return None
    pool, _, _ = uncapped_pool(snap, {n: 1.0 for n in nodes}, config=cfg, conv_id=qid)
    feats, ids = atom_features(pool, snap, emb); ei, er = pool_adjacency(pool, ids)
    return {"features": feats, "edge_index": ei, "edge_rel": er, "query": emb.embed([instances[qid]["question"]])[0],
            "labels": recall.distant_labels(snap, pool, instances[qid], conv_id=qid)}
ex = [e for e in (prep(q) for q in qs) if e is not None]
rng = random.Random(13); rng.shuffle(ex); k = max(1, len(ex) // 5)
scorer = build_scorer(seed=13); t0 = time.perf_counter()
r = train_scorer(scorer, {"train": ex[k:], "dev": ex[:k]}, seed=13)
torch.save({"state_dict": scorer.state_dict(), "seed": 13, "train_questions": len(ex) - k, "dev_questions": k,
            "corpus": "LongMemEval-S scope c", "locomo_used": False,
            "result": {a: b for a, b in r.items() if not isinstance(b, (list, dict))}}, res / "stage_c_scorer_scopec.pt")
print(f"   scorer: {len(ex)} questions ({len(ex)-k} train / {k} dev), best dev loss {r.get('best_dev_loss')}, {time.perf_counter()-t0:.0f}s")
PY

echo "== Gate 1 at volume: export the item batch over the scope-c graph =="
srun python - <<'PY'
import json, shutil, tempfile, pathlib, os
from graft.eventlog import EventLog
from graft.graphbuild.standin import construct
from graft.graphbuild.embed import Embedder
from graft.graphbuild import pins
lme = pathlib.Path(os.environ["LME"]) if "LME" in os.environ else pathlib.Path(os.environ["GRAFT_ROOT"]) / "results/longmemeval_scope_c"
src = lme / "events.jsonl"; tmp = pathlib.Path(tempfile.mkdtemp()) / "events.jsonl"; shutil.copyfile(src, tmp)
emb = Embedder(cache_dir=pathlib.Path("artefacts/phase6/embed_cache"))
log = EventLog.open(tmp, fsync=False)
try:
    out = construct(log, embed=emb.embed, k=pins.K_CANDIDATES, s=pins.S_PAIRS)
finally:
    log.close()
res = pathlib.Path(os.environ["RES"]); res.mkdir(parents=True, exist_ok=True)
for name, items in (("d1_items_scopec.jsonl", out["d1_items"]), ("d2_items_scopec.jsonl", out["d2_items"])):
    (res / name).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    print(f"   {name}: {len(items)} items")
PY
echo "AFTER-INGEST DONE -- $RES"
