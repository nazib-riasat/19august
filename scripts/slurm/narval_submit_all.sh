#!/bin/bash
# GRAFT on Narval -- submit the whole remaining programme with dependencies. Run on the login node AFTER
# narval_00_setup_login.sh has passed.
#
#   bash scripts/slurm/narval_submit_all.sh                 # submit everything
#   bash scripts/slurm/narval_submit_all.sh --only ladder   # one job family
#   bash scripts/slurm/narval_submit_all.sh --dry           # print the sbatch lines, submit nothing
#
# Accounts are DISCOVERED, not hard-coded: GPU jobs go to the *_gpu allocation, CPU jobs to *_cpu,
# from `sacctmgr show assoc user=$USER`. If only one allocation exists it is used for both.
# The programme:
#   preflight (GPU) ─┬─> ingest (GPU)  ──> after_ingest (GPU)
#                    ├─> fullctx+D3D4 (GPU)
#                    └─> ladder x27 (CPU) ──> distil full head + gate3 (CPU) ──> final evals (GPU)
# Nothing full-scale starts unless preflight exits 0.

set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DRY=0; ONLY=""
while [[ $# -gt 0 ]]; do case "$1" in --dry) DRY=1;; --only) ONLY="$2"; shift;; esac; shift; done

accounts=$(sacctmgr -nP show assoc user="$USER" format=account | sort -u)
GPU_ACC=$(echo "$accounts" | grep -m1 '_gpu$' || true)
CPU_ACC=$(echo "$accounts" | grep -m1 '_cpu$' || true)
[[ -z "$GPU_ACC" ]] && GPU_ACC=$(echo "$accounts" | head -1)
[[ -z "$CPU_ACC" ]] && CPU_ACC="$GPU_ACC"
echo "accounts: gpu=$GPU_ACC cpu=$CPU_ACC   (all: $(echo $accounts | tr '\n' ' '))"
MAIL="${GRAFT_MAIL:-}"; MAILOPT=""; [[ -n "$MAIL" ]] && MAILOPT="--mail-user=$MAIL"

sub() {  # sub <account> <deps-or-empty> <script>  -> prints job id
    local acc="$1" dep="$2" script="$3"; shift 3
    local args=(--account="$acc" --parsable $MAILOPT)
    [[ -n "$dep" ]] && args+=(--dependency="$dep")
    if [[ "$DRY" == "1" ]]; then echo "sbatch ${args[*]} $script" >&2; echo "DRY"; return; fi
    sbatch "${args[@]}" "$script" | cut -d';' -f1
}
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

cd "$HERE/../.."
PRE=""
if want preflight; then PRE=$(sub "$GPU_ACC" "" "$HERE/narval_01_preflight.sbatch"); echo "preflight   $PRE"; fi
DEP=""; [[ -n "$PRE" && "$PRE" != "DRY" ]] && DEP="afterok:$PRE"

if want ingest; then
    ING=$(sub "$GPU_ACC" "$DEP" "$HERE/narval_02_ingest_longmemeval.sbatch"); echo "ingest      $ING"
    AFT=$(sub "$GPU_ACC" "afterok:$ING" "$HERE/narval_06_after_ingest.sbatch"); echo "after-ingest $AFT"
fi
if want baselines; then
    BASE=$(sub "$GPU_ACC" "$DEP" "$HERE/narval_05_fullcontext.sbatch"); echo "baselines   $BASE"
fi
G3=""
if want ladder; then
    LAD=$(sub "$CPU_ACC" "$DEP" "$HERE/narval_03_ladder.sbatch"); echo "ladder x27  $LAD"
    G3=$(sub "$CPU_ACC" "afterok:$LAD" "$HERE/narval_04_after_ladder.sbatch"); echo "gate3+head  $G3"
fi
if want final; then
    FDEP=""; [[ -n "$G3" && "$G3" != "DRY" ]] && FDEP="afterok:$G3"
    FIN=$(sub "$GPU_ACC" "$FDEP" "$HERE/narval_07_final_evals.sbatch"); echo "final evals $FIN   (LoCoMo + RAG + full 2Wiki, full-spec head)"
fi
echo
echo "watch:  squeue -u $USER          logs: $SCRATCH/graft/logs + ./graft_*.out"
echo "results land under $SCRATCH/graft/results/"
