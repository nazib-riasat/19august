"""Phase 7's runner: the five-channel stack over the pilot's Stage-B graph.

    python scripts/phase7_retrieval.py --smoke     # stub embedder, no GPU
    python scripts/phase7_retrieval.py             # the pinned bge-small

**Every number this produces is a machinery number** (G9), and the artefact says
so in its own body rather than in a commit message.  Three bounds, all declared
before the first run:

1. the graph is **stand-in-constructed** (`PHASE6_DECISIONS.md` §2.1) — links are
   a deterministic rule, not a trained D1;
2. there are **10 questions**, so everything is per-question-listed and nothing
   is presented as a distribution;
3. **ceiling 1 already took 55%**, so ceiling 3 is reported beside ceilings 1–2
   or a retrieval failure gets misread into a stage that did not cause it.

**Tier-A recall only.**  ``graft.retrieve.recall.tier_b_gold`` refuses by name
until ``GATE0_CONTRACT.md`` signs, and this script does not work around it.

**The obligation slots are replayed, not re-parsed.**  Fix F2's parser is the
frozen extractor, which is GPU work; its output for these questions was already
recorded by the Phase-5 pilot in ``audit_obligation_slots.csv``.  Replaying that
file keeps this runner CPU-only *and* keeps Stage C reading exactly the slots
Stage A produced — re-parsing could silently drift from the recorded audit and
make the two disagree about the same question.  The parser's measured unresolved
rate is quoted beside the slots (fix F2's rule, exit criterion 16).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from graft.config import load_config  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import corpus as corpus_mod  # noqa: E402
from graft.ingest.oblparse import obligations_from_slots  # noqa: E402
from graft.retrieve import pins as cpins  # noqa: E402
from graft.retrieve.bm25 import BM25Channel  # noqa: E402
from graft.retrieve.dense import DenseChannel  # noqa: E402
from graft.retrieve.entity import entity_channel, match_entities  # noqa: E402
from graft.retrieve.expand import expand_channel  # noqa: E402
from graft.retrieve.fuse import assemble  # noqa: E402
from graft.retrieve.pool import build_pool  # noqa: E402
from graft.retrieve.recall import (  # noqa: E402
    HONESTY_STAMP,
    ceilings,
    channel_table,
    has_answer_turns,
    recall_of,
    saturation,
    tier_a_gold,
)
from graft.runtime import json_sanitize, run_manifest  # noqa: E402

STAGE_B_LOG = REPO / "artefacts" / "phase6" / "events.jsonl"
SLOTS_CSV = REPO / "artefacts" / "phase5_pilot" / "audit_obligation_slots.csv"
SLOTS_CACHE = REPO / "artefacts" / "phase7_obligations.csv"
OUT = REPO / "artefacts" / "phase7_retrieval.json"


SLOT_FIELDS = (
    "question_id",
    "question",
    "question_date",
    "pred_entity_anchor",
    "pred_value_type",
    "pred_time_expression",
    "pred_needs_source",
    "pred_aggregate",
)


def read_slots(*paths: Path) -> dict[str, dict[str, Any]]:
    """The recorded Stage-A obligation parse, keyed by question id.

    Several files are merged because the slots for these questions live in two
    places: the Phase-5 pilot's audit worksheet holds a *stratified draw over the
    whole corpus* (49 questions, none of them necessarily this graph's), and
    ``--parse`` writes a cache for whatever is still missing.  Later paths win, so
    a freshly parsed slot supersedes a stale cached one.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                out[row["question_id"]] = {
                    "question": row["question"],
                    "question_date": row["question_date"],
                    "slots": {
                        "entity_anchor": row["pred_entity_anchor"] or None,
                        "value_type": row["pred_value_type"] or None,
                        "time_expression": row["pred_time_expression"] or None,
                        "needs_source": row["pred_needs_source"] == "True",
                        "aggregate": row["pred_aggregate"] == "True",
                    },
                }
    return out


def parse_missing(question_ids: list[str], instances: dict, cache: Path, device: str) -> int:
    """Run the **frozen** obligation parser over questions with no recorded slots.

    Fix F2's parser is the extractor LLM, so this is the only GPU work in steps
    0–5 and it is opt-in for that reason.  The result is cached to ``cache`` so a
    later CPU run replays it: re-parsing on every run would spend the GPU for a
    deterministic answer and, worse, would let two runs of the same script read
    different slots for the same question if the model or its pins ever moved.

    There is deliberately **no rule-based fallback**.
    ``graft.core.obligations.parse`` refuses any mode but ``exact`` and
    ``learned`` precisely so that real questions cannot silently receive
    exact-mode behaviour, and inventing a keyword parser here would reintroduce
    the guessed anchor that decision 7 forbids one field over.
    """
    from graft.ingest.extractor import build_extractor
    from graft.ingest.oblparse import ObligationParser

    extractor = build_extractor(device=device)
    parser = ObligationParser(lambda system, user: extractor.complete(system, user))
    rows: list[dict[str, Any]] = []
    extractor.load()
    try:
        for question_id in question_ids:
            meta = corpus_mod.question_meta(instances[question_id])
            obligations = parser.parse(
                meta["question"],
                meta["question_date"],
                scope=(question_id,),
                question_id=question_id,
            )
            rows.append(
                {
                    "question_id": question_id,
                    "question": meta["question"],
                    "question_date": meta["question_date"],
                    "pred_entity_anchor": obligations.entity_anchor or "",
                    "pred_value_type": obligations.value_type or "",
                    "pred_time_expression": parser.traces[-1]["time_expression"] or "",
                    "pred_needs_source": str(bool(obligations.needs_source)),
                    "pred_aggregate": str(bool(obligations.aggregate)),
                }
            )
    finally:
        extractor.close()

    existing = list(csv.DictReader(cache.open(encoding="utf-8", newline=""))) if cache.is_file() else []
    merged = {row["question_id"]: row for row in existing}
    merged.update({row["question_id"]: row for row in rows})
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SLOT_FIELDS))
        writer.writeheader()
        for key in sorted(merged):
            writer.writerow({k: merged[key].get(k, "") for k in SLOT_FIELDS})
    print(f"parsed {len(rows)} obligation slots -> {cache.relative_to(REPO)}")
    print(f"  parser report: {json.dumps(parser.report(), sort_keys=True)}")
    return len(rows)


def conv_ids_in(snapshot: Any) -> tuple[str, ...]:
    """The conversations this graph actually contains, from its turns.

    Derived rather than configured: the graph is whatever the pilot committed,
    and a hard-coded question list would silently disagree with it the first time
    the pilot is re-run at a different scope.
    """
    turns = getattr(snapshot, "_turns", {})
    return tuple(sorted({t.conv_id for t in turns.values()}))


def run(
    *,
    smoke: bool,
    graph_path: Path,
    slots_path: Path,
    cache_path: Path,
    out_path: Path,
    parse: bool = False,
    device: str = "cuda",
) -> int:
    config = load_config()
    log = EventLog.open(graph_path)
    snapshot = ReplayGraphStore(log).at()

    instances = corpus_mod.question_index(corpus_mod.load_corpus())
    questions = [q for q in conv_ids_in(snapshot) if q in instances]

    slots_by_q = read_slots(slots_path, cache_path)
    if parse:
        missing = [q for q in questions if q not in slots_by_q]
        if missing:
            parse_missing(missing, instances, cache_path, device)
            slots_by_q = read_slots(slots_path, cache_path)

    if smoke:
        from graft.graphbuild.embed import StubEmbedder

        embedder: Any = StubEmbedder(dim=int(cpins.EMBEDDER["dim"]))
    else:
        from graft.graphbuild.embed import Embedder

        embedder = Embedder(cache_dir=REPO / "artefacts" / "phase6" / "embed_cache")
        embedder.load()

    per_question: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {name: [] for name in cpins.CHANNELS}
    end_to_end: list[float] = []

    for question_id in questions:
        record = slots_by_q.get(question_id)
        meta = corpus_mod.question_meta(instances[question_id])
        if record is None:
            # **Degrade visibly, never silently.**  Without slots the anchored
            # channels are structurally empty -- decision 7's "no anchor -> empty
            # channel" -- so BM25 and dense carry the question alone.  That is a
            # legitimate configuration to measure, but it is *not* the five-channel
            # stack, and a recall number from it would be quoted as though it were
            # unless every row says which one it is.
            record = {
                "question": meta["question"],
                "question_date": meta["question_date"],
                "slots": {
                    "entity_anchor": None,
                    "value_type": None,
                    "time_expression": None,
                    "needs_source": False,
                    "aggregate": False,
                },
            }
        obligations, trace = obligations_from_slots(
            record["slots"], record["question_date"], scope=(question_id,)
        )
        slots_present = question_id in slots_by_q
        text = record["question"]
        started = time.perf_counter()

        timings: dict[str, float] = {}
        channels: dict[str, dict[str, float]] = {}

        mark = time.perf_counter()
        channels["bm25"] = BM25Channel(snapshot, question_id, config=config).query(text)
        timings["bm25"] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        channels["dense"] = DenseChannel(snapshot, embedder, question_id, config=config).query(text)
        timings["dense"] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        channels["entity"] = entity_channel(snapshot, obligations, question_id)
        timings["entity"] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        seeds = match_entities(snapshot, obligations.entity_anchor, question_id)
        expand_hits, expand_report = expand_channel(snapshot, seeds)
        channels["expand"] = expand_hits
        timings["expand"] = (time.perf_counter() - mark) * 1000.0

        pool, atom_scores, report = assemble(
            snapshot, channels, constraint=obligations.time_constraint, config=config
        )
        total_ms = (time.perf_counter() - started) * 1000.0

        for name, value in timings.items():
            latencies[name].append(value)
        end_to_end.append(total_ms)

        # Tier-A gold, and the pre/post-cap pair that makes the cap's cost visible.
        gold_closed, gold_reachable = tier_a_gold(
            snapshot, has_answer_turns(instances[question_id]), question_id
        )
        fused_kept = _fused_of(channels, snapshot, obligations)
        uncapped, _, _ = build_pool(
            snapshot, fused_kept, cap=max(1, 64 * (len(fused_kept) + 1))
        )

        per_question.append(
            {
                "question_id": question_id,
                "question": text,
                "question_type": meta["question_type"],
                "obligation_slots_present": slots_present,
                "obligation_slots": obligations.to_dict(),
                "obligation_trace": trace,
                "entity_seeds": list(seeds),
                "channels": channel_table(channels, gold_reachable, latency_ms=timings),
                "expansion": expand_report,
                "assembly": report,
                "pool_size": len(pool),
                "saturation": saturation(snapshot, question_id, config.pool_cap),
                "recall_tier_a": {
                    "post_cap": recall_of(pool.ids(), gold_closed),
                    "pre_cap": recall_of(uncapped.ids(), gold_closed),
                },
                "latency_ms": {"channels": timings, "end_to_end": round(total_ms, 3)},
            }
        )

    with_slots = sum(1 for r in per_question if r.get("obligation_slots_present"))
    exercised = sum(1 for r in per_question if r["saturation"]["exercised"])
    artefact = {
        "phase": 7,
        "smoke": bool(smoke),
        "honesty_stamp": HONESTY_STAMP,
        "questions_exercising_retrieval": exercised,
        "saturation_warning": (
            None
            if exercised == len(per_question)
            else (
                f"{len(per_question) - exercised} of {len(per_question)} questions have "
                "fewer eligible candidates than pool_cap, so their pool is the whole "
                "conversation and Tier-A recall is 1.0 by construction. These are "
                "plumbing numbers in the strongest sense: they show the stack runs, "
                "and they measure no retrieval at all. Distractor sessions "
                "(DATASET_DECISION.md 5, scope b') are what make the cap bind."
            )
        ),
        "questions_with_obligation_slots": with_slots,
        "questions_without_obligation_slots": len(per_question) - with_slots,
        "obligations_absent_reading": (
            "a question with no slots runs BM25 + dense only: decision 7 makes the "
            "entity channel empty without an anchor, and the temporal filter never "
            "runs without a constraint. Those rows measure a two-channel stack and "
            "must not be pooled with the four-channel ones. Use --parse to fill them."
        ),
        "tier": "A",
        "tier_b": "refused until GATE0_CONTRACT.md signs (retrieve.recall.tier_b_gold)",
        "graph": str(graph_path.relative_to(REPO)),
        "questions": len(per_question),
        "ceilings": ceilings(snapshot),
        "stage_c_fingerprint": cpins.stage_c_fingerprint(),
        "stage_c_frozen": cpins.frozen_values(),
        "obligation_parser_note": (
            "slots replayed from the Phase-5 pilot audit, not re-parsed. Phase 5 "
            "measured 69% unresolved among time-bearing questions "
            "(PHASE5_DECISIONS.md 2.2a); fix F2 requires that rate beside any "
            "coverage number derived from these slots."
        ),
        "latency": {
            "per_channel": {
                name: _percentiles(values) for name, values in sorted(latencies.items())
            },
            "end_to_end": _percentiles(end_to_end),
            "envelope_note": (
                "Mem0 (ECAI 2025) reports 0.148 s flat / 0.476 s graph. Those are "
                "published, uncontrolled numbers quoted as context only; nothing "
                "here is compared against them (plan 5.3's re-run rule)."
            ),
        },
        "per_question": per_question,
        # Seed is nominal: nothing in steps 0-5 samples. It is recorded anyway so
        # the manifest keeps one shape across phases, and so the seed is already
        # there when the scorer (which does sample) arrives.
        "manifest": run_manifest(config, config.seeds[0]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(json_sanitize(artefact), indent=1, sort_keys=True), encoding="utf-8"
    )
    print(
        f"wrote {out_path.relative_to(REPO)}  questions={len(per_question)}  "
        f"with_slots={with_slots}/{len(per_question)}"
    )
    for row in per_question:
        post = row["recall_tier_a"]["post_cap"]
        pre = row["recall_tier_a"]["pre_cap"]
        flag = " " if row["obligation_slots_present"] else "*"
        sat = "" if row["saturation"]["exercised"] else "  [unexercised]"
        print(
            f" {flag}{row['question_id']}: pool={row['pool_size']:3d} "
            f"cand={row['saturation']['candidates_in_scope']:3d} "
            f"gold={post['gold']:3d} recall_post={_fmt(post['recall'])} "
            f"pre={_fmt(pre['recall'])}{sat}"
        )
    if with_slots < len(per_question):
        print("  * = no obligation slots: BM25 + dense only. Re-run with --parse.")
    if artefact["saturation_warning"]:
        print(f"\n  WARNING: {artefact['saturation_warning']}")
    return 0


def _fused_of(channels, snapshot, obligations):
    """The fused, temporally-filtered node scores — the pre-cap input.

    Recomputed rather than threaded out of ``assemble`` so that the pre-cap pool
    is built from exactly the same arithmetic the capped one was, with only the
    cap differing.  That is what makes their difference *the cap's cost* and not
    the difference between two code paths.
    """
    from graft.retrieve.fuse import fuse
    from graft.retrieve.temporal import temporal_filter

    fused, _ = fuse(channels)
    kept, _ = temporal_filter(snapshot, fused, obligations.time_constraint)
    return kept


def _percentiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": round(statistics.median(ordered), 3),
        # Nearest-rank p95: with n = 10 the interpolated variants invent a value
        # between two measurements, and at this sample size the honest statistic
        # is an observed one.
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
    }


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="stub embedder; no GPU, no model")
    parser.add_argument(
        "--parse",
        action="store_true",
        help="run the frozen obligation parser for questions with no cached slots (GPU)",
    )
    parser.add_argument("--device", default="cuda", help="device for --parse")
    parser.add_argument("--graph", type=Path, default=STAGE_B_LOG)
    parser.add_argument("--slots", type=Path, default=SLOTS_CSV)
    parser.add_argument("--slots-cache", type=Path, default=SLOTS_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return run(
        smoke=args.smoke,
        graph_path=args.graph,
        slots_path=args.slots,
        cache_path=args.slots_cache,
        out_path=args.out,
        parse=args.parse,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
