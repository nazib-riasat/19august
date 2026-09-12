#!/usr/bin/env python
"""Hardware probe -- the job reads the node and configures itself; no human, no model.

    python scripts/slurm/hw_probe.py                    # print + write hw.json
    python scripts/slurm/hw_probe.py --knob extract_batch   # one number, for shell use

Every Slurm job in this kit runs the probe first and takes its knobs from the
result, so the same script is correct on an A100-40GB, an H100-80GB, or the
8 GB laptop card it was tested on.  What it decides and why:

* ``extract_batch`` -- the 3B extractor's batch size.  Measured on the 8 GB
  laptop: batch 8 leaves ~1.7 GB headroom.  Scaled by free memory in steps that
  stay well inside the card: <12 GB -> 8, <24 -> 16, <48 -> 32, else 48.  The
  fingerprint does not include batch size, so this is the same experiment
  (`PHASE5_DECISIONS.md`); batching was measured deterministic (12/12) on 12 Sep.
* ``embed_batch`` -- bge-small (130 MB) barely matters; 256 on >=16 GB.
* ``reader_ctx_ok`` -- whether a 22k-token full-context prompt is plausible: the
  dense prefill measured 14 GB at 8k on the laptop; require >= 32 GB free.
* ``workers`` -- ``SLURM_CPUS_PER_TASK`` minus one, min 1.

**Refusals, not fallbacks.**  A GPU without bf16 (compute capability < 8.0)
exits non-zero: `DATASET_DECISION.md` records that the fp16 fallback changes
``ingestion_fingerprint``, so a run there is a different experiment and the
decision belongs to a person.  CPU-only jobs pass ``--cpu`` and skip that check.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def probe(require_gpu: bool) -> dict:
    hw: dict = {
        "host": platform.node(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")
                  and k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK",
                            "SLURM_MEM_PER_NODE", "SLURM_GPUS_ON_NODE", "SLURM_JOB_PARTITION",
                            "SLURM_JOB_ACCOUNT", "SLURMD_NODENAME")},
        "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1),
        "python": sys.version.split()[0],
        "gpus": [],
    }
    try:
        import torch
        hw["torch"] = torch.__version__
        hw["cuda_available"] = bool(torch.cuda.is_available())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            hw["gpus"].append({
                "index": i, "name": p.name, "total_gb": round(total / 2**30, 1),
                "free_gb": round(free / 2**30, 1), "capability": f"{p.major}.{p.minor}",
                "bf16": bool(p.major >= 8),
            })
    except Exception as e:  # torch missing or broken -- report, don't crash the probe
        hw["torch_error"] = repr(e)[:200]
        hw["cuda_available"] = False

    scratch = os.environ.get("SCRATCH") or str(Path.home())
    try:
        du = shutil.disk_usage(scratch)
        hw["scratch_free_gb"] = round(du.free / 2**30, 1)
    except OSError:
        hw["scratch_free_gb"] = None

    # -- knobs -------------------------------------------------------------
    gpu = hw["gpus"][0] if hw["gpus"] else None
    free = gpu["free_gb"] if gpu else 0.0
    if not gpu:
        extract_batch = 1
    elif free < 12:
        extract_batch = 8
    elif free < 24:
        extract_batch = 16
    elif free < 48:
        extract_batch = 32
    else:
        extract_batch = 48
    hw["knobs"] = {
        "extract_batch": extract_batch,
        "embed_batch": 256 if free >= 16 else 64,
        "reader_ctx_ok": bool(gpu and free >= 32),
        "workers": max(1, hw["cpus"] - 1),
        "device": "cuda" if gpu else "cpu",
    }
    hw["refusals"] = []
    if require_gpu:
        if not gpu:
            hw["refusals"].append("GPU job but no CUDA device visible")
        elif not gpu["bf16"]:
            hw["refusals"].append(
                f"{gpu['name']} (cc {gpu['capability']}) has no bf16; the fp16 fallback "
                "changes ingestion_fingerprint (DATASET_DECISION.md) -- a person decides this"
            )
    return hw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="CPU-only job: do not require a GPU")
    ap.add_argument("--out", default=None, help="write hw.json here (default: $RUN_DIR/hw.json or ./hw.json)")
    ap.add_argument("--knob", default=None, help="print one knob value and exit")
    args = ap.parse_args()
    hw = probe(require_gpu=not args.cpu)
    if args.knob:
        print(hw["knobs"][args.knob]); return 0
    out = Path(args.out or os.path.join(os.environ.get("RUN_DIR", "."), "hw.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(hw, indent=1), encoding="utf-8")
    print(json.dumps(hw, indent=1))
    if hw["refusals"]:
        print("\nREFUSED:", *hw["refusals"], sep="\n  ", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
