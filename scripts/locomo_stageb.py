"""Stage-B structural commit over the LoCoMo Stage-A log. **CPU only, minutes.**

**The missing stage the 19 Aug 2026 ingestion run exposed.** Stage A writes
turns, spans, mentions and assertions; Stage C retrieves over **nodes**
(`retrieve.pool.eligible_nodes`). What turns eligible assertions into Claim/Value
nodes with entity links is Phase 6's stand-in constructor
(`graphbuild.standin.construct`) — deterministic, exact-match linking, **no
trained decoder and no embedder required**. The pilot's graph got its nodes this
way (`artefacts/phase6/events.jsonl`); the LoCoMo chain omitted the step, so its
graph read ``nodes: 0`` and every eval question would have abstained.

**Why the runner works on a COPY of the Stage-A log.** ``construct`` appends its
commits into the log it is given (``Committer(log)``), and it is **not
idempotent** — running it twice would create duplicate entities. So the Stage-A
log stays pristine at ``artefacts/locomo/events.jsonl``, the copy lives at
``artefacts/locomo_stageb/events.jsonl``, and this runner **refuses** a
destination that already contains Stage-B ops unless ``--fresh`` recopies from
Stage A.

**What to expect from the counts, so nobody misreads them as a defect.** The
stand-in links a turn's eligible assertions through the turn's **first mention's
entity**; a turn with no mentions contributes no Claim nodes (G5's
anchors-exist-only-for-linked-claims rule). The LoCoMo log carries 1,422
mentions over 5,882 turns, so the node count will be **well below** the 2,268
eligible assertions. That is the stand-in's documented behaviour, not data loss —
the assertions remain in the log, retrievable the moment a trained D1 links more
turns.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graft.eventlog import EventLog  # noqa: E402
from graft.graphbuild.standin import construct  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402

STAGE_B_OPS = ("node.add", "edge.add")


def _op_census(path: Path) -> dict[str, int]:
    ops: collections.Counter = collections.Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ops[json.loads(line)["op"]] += 1
    return dict(ops)


def main() -> int:
    # The cp1252 guard, per the convention scripts/phase3_calibrate.py set.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artefacts/locomo/events.jsonl",
                        help="the pristine Stage-A log; never written to")
    parser.add_argument("--run-dir", default="artefacts/locomo_stageb",
                        help="where the copied log is constructed over")
    parser.add_argument("--out", default="artefacts/locomo_stageb.json")
    parser.add_argument(
        "--fresh", action="store_true",
        help="delete the run-dir copy and recopy from Stage A before constructing. "
        "Required to redo a construction: construct() is not idempotent, and "
        "appending a second pass would duplicate every entity.",
    )
    args = parser.parse_args()

    source = REPO / args.source
    if not source.is_file():
        raise SystemExit(
            f"no Stage-A log at {source}; run scripts/locomo_ingest.py first"
        )

    run_dir = REPO / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / "events.jsonl"

    if dest.exists():
        census = _op_census(dest)
        if any(op in census for op in STAGE_B_OPS) and not args.fresh:
            raise SystemExit(
                f"{dest} already carries Stage-B commits ({ {k: v for k, v in census.items() if k in STAGE_B_OPS} }). "
                "construct() is not idempotent -- a second pass would duplicate "
                "every entity. Pass --fresh to recopy from Stage A and redo."
            )
        if args.fresh:
            dest.unlink()

    if not dest.exists():
        shutil.copyfile(source, dest)
        print(f"copied {source.name}: {source.stat().st_size:,} bytes -> {dest}")

    before = _op_census(dest)
    started = time.perf_counter()
    log = EventLog.open(dest, fsync=False)
    try:
        # embed=None deliberately: the stand-in's linking is exact normalised
        # match ("no embedding, no scores"), so the step is deterministic and
        # CPU-only, and stays inside the audited Phase-6 machinery.
        report = construct(log, embed=None)
    finally:
        log.close()
    elapsed = time.perf_counter() - started

    after = _op_census(dest)
    snap = ReplayGraphStore(EventLog.open(dest, fsync=False)).at()
    counts = snap.counts()

    artefact = {
        "what_this_is": (
            "Stage-B structural commit (Phase 6 stand-in constructor, exact-match "
            "linking, no trained decoder) over a COPY of the LoCoMo Stage-A log. "
            "The Stage-A log is untouched."
        ),
        "source": str(args.source),
        "log": str(dest.relative_to(REPO)),
        "ops_before": before,
        "ops_after": after,
        "graph_counts": counts,
        # **The keys `construct()` actually returns** (corrected 19 Aug 2026).
        # These read `report["d1"]` / `report["d2"]`, which do not exist -- the
        # constructor returns `d1_items` / `d2_items` -- so `.get` fell to `[]`
        # and every run recorded **0 D1 and 0 D2 items** regardless of what was
        # built.  Zero supervision is a result this project is already primed to
        # believe (`CLAUDE.md` §7 records the pair proposer surfacing zero
        # CONFLICT pairs), which is exactly why an instrument must not be able
        # to report it by accident.  Same defect class as the eval runner's
        # `sat.get("saturated")`.
        "d1_items": len(report["d1_items"]) if isinstance(report, dict) else None,
        "d2_items": len(report["d2_items"]) if isinstance(report, dict) else None,
        "seconds": round(elapsed, 1),
        "expectation_note": (
            "Every eligible assertion becomes a node (standin.py's coverage "
            "policy, 19 Aug 2026): anchored with an `about_entity` edge where "
            "its turn carries a mention, standalone and edge-less where it does "
            "not. So `assertion-backed nodes == eligible_assertions` is the "
            "expectation, while `edges << nodes` is the stand-in's exact-match "
            "linking being crude -- which is what a trained D1 is for. This note "
            "previously predicted `nodes << eligible assertions`, which was the "
            "behaviour before that policy and is no longer what the run does."
        ),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artefact, indent=2, default=str), encoding="utf-8")

    print(f"constructed in {elapsed:.1f}s")
    print(f"ops:   {before} -> +{ {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k,0) != before.get(k,0)} }")
    print(f"graph: {counts}")
    print(f"written: {args.out}")
    print()
    print(f"next: python scripts/locomo_eval.py --run-dir {args.run_dir} --head artefacts/utility_head.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
