"""Phase 6's harness: the Gate-1 run, and the smoke run that stands in for it.

    python scripts/phase6_gate1.py --smoke        # bootstrap labels, nothing quotable
    python scripts/phase6_gate1.py                # reports the entry conditions
    python scripts/phase6_gate1.py --decisive     # the real Gate-1 run, when they hold

**Gate 1 cannot run yet, and this script says so rather than approximating it**
(G1).  Its four entry conditions, as of 14 Aug 2026:

1. ``GATE0_CONTRACT.md`` signed — **blocked** on item 8 (the Phase-2.5 human
   timed pass);
2. human D1 **and** D2 labels — **blocked**; only the spike's machine-assisted
   bootstrap labels exist, and they carry ``answers_gate0_item8: false``;
3. a frozen extractor — **met** (Phase-5 decision 2, candidate B);
4. an ingested corpus — **partly met**: the live pilot's 248 turns are real
   Stage-A output, but the item-9 scope corpus is undecided.

**The decisive path exists** (`graphbuild.train`, built 14 Aug 2026): with all
four conditions met, ``--decisive`` trains every arm under the shared budget at
seeds {13, 42, 7}, predicts on the **test** split and hands the predictions to
``gate1.run_gate1(smoke=False)``, which computes the McNemar table.  Until the
conditions hold it refuses **by name** rather than silently producing another
smoke artefact — the failure the first version had, where ``smoke=True`` was
hardcoded on every path.

What ``--smoke`` establishes is plumbing, end to end through the *write* path:
the stand-in constructor (``graphbuild.standin``) builds the graph turn by turn
— entities, alias growth, ``about_entity`` links — computing every D1 item's
candidates against the constructor's own snapshot at that moment (G12), and
proposing D2 pairs at link time (G5).  The validator's refusals are counted by
category, the corruption audit runs over every commit, and the artefact is
stamped ``smoke: true`` with **no comparison in it at all** — ``gate1.run_gate1``
suppresses it rather than labelling it.  The learned pieces (encoders, decoders,
calibration) are exercised by ``test_graphbuild_learning.py``, not here: a
learned decoder in the smoke loop would make the audit's result depend on
training, which is the confound the stand-in exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from graft.config import load_config  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import corpus  # noqa: E402
from graft.graphbuild import gate1 as gate  # noqa: E402
from graft.graphbuild import pins  # noqa: E402
from graft.graphbuild.embed import Embedder, StubEmbedder  # noqa: E402
from graft.graphbuild.items import yields  # noqa: E402
from graft.graphbuild.loaders import loader_artefact  # noqa: E402
from graft.graphbuild.standin import construct  # noqa: E402
from graft.graphbuild.train import (  # noqa: E402
    ARM_VARIANTS,
    FeatureCache,
    build_arm,
    parse_d1_gold,
    predict_d1,
    read_labels,
    split_questions,
    train_d1_arm,
)
from graft.ids import span_id  # noqa: E402
from graft.ingest.pins import EXTRACTOR  # noqa: E402
from graft.runtime import json_sanitize, run_manifest  # noqa: E402

PILOT_LOG = REPO / "artefacts" / "phase5_pilot" / "events.jsonl"


def entry_conditions() -> dict[str, dict]:
    """Gate 1's entry conditions, evaluated rather than asserted, so the refusal
    names which one is missing instead of saying "not ready"."""
    contract = (REPO / "GATE0_CONTRACT.md").read_text(encoding="utf-8") if (
        REPO / "GATE0_CONTRACT.md"
    ).is_file() else ""
    signed = "**Signed:**" in contract and "*(unsigned)*" not in contract

    # Both decoders need human labels, not "any non-bootstrap file" (the first
    # version's check passed on a single stray jsonl).  Volume sufficiency is
    # item 8's judgment at sign-off, not a file count this script could know.
    labels_dir = REPO / "data" / "phase2_5" / "labels"
    human = {
        decoder: sorted(
            p.name
            for p in labels_dir.glob(f"{decoder}*.jsonl")
            if "bootstrap" not in p.name
        )
        for decoder in ("d1", "d2")
    }
    return {
        "gate0_contract_signed": {
            "met": signed,
            "detail": "item 8 (feasible annotation volume) needs the Phase-2.5 human timed pass",
        },
        "human_d1_d2_labels": {
            "met": bool(human["d1"]) and bool(human["d2"]),
            "detail": f"non-bootstrap label files: d1={human['d1'] or 'none'}, "
            f"d2={human['d2'] or 'none'} (both decoders required; volumes are "
            "judged against item 8 at sign-off)",
        },
        "frozen_extractor": {
            "met": EXTRACTOR is not None,
            "detail": f"pins.EXTRACTOR = {EXTRACTOR['candidate'] if EXTRACTOR else None}",
        },
        "ingested_corpus": {
            "met": PILOT_LOG.is_file(),
            "detail": f"pilot log at {PILOT_LOG.relative_to(REPO)}"
            if PILOT_LOG.is_file()
            else "no ingested log",
        },
    }


def load_d1_gold(graph_entity_ids: set[str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """Human D1 labels, keyed by **span id**, with the transfer rate reported.

    **``item_id`` is not a durable key across item generations, and using it
    would be silently catastrophic.**  Positional ids are re-issued from zero by
    every derivation: the spike's ``d1_0000`` is "Gitzo" in session `_2`, the
    Phase-6 log's ``d1_0000`` is "Lowepro ProTactic 450 AW" in session `_1`, and
    **zero of 34 spike items name the same mention as the Phase-6 item with the
    same id**.  Joining on ``item_id`` attaches every label to a different
    mention — measured, 14 Aug 2026, before this path ever ran.

    The durable key is the span: ``span_id(turn_id, start, end)``, which the
    spike items carry as their three components and Phase-6 items carry
    directly.  It survives re-derivation because it is a fact about the corpus,
    not about the order items happened to be emitted in.

    Bootstrap labels are excluded (``answers_gate0_item8: false``, machine
    assisted) — training a *decisive* arm on them would be the bootstrap-labels
    mistake two phases later. Pass 2 is excluded by ``read_labels``: it is the
    κ re-annotation of a subset, not a second opinion to train on.

    Returns ``(span_id -> label, report)``.  The report carries the **transfer
    rate**, which is a number worth seeing rather than a detail: labels
    collected against one extractor's mentions only partly survive a change of
    extractor — measured at **17 of 34** for the spike labels against candidate
    B's pilot mentions, because the two extractions do not propose the same
    spans.
    """
    items_by_id: dict[str, dict] = {}
    source_of: dict[str, str] = {}
    for path in sorted((REPO / "data" / "phase2_5").glob("d1_items*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item["item_id"])
            if item_id in items_by_id:
                # Positional ids restart at zero in every derivation, so two
                # item files silently collide — and the later one would re-key
                # every label to a different span, which is the failure the
                # span-keying fixed one level down.  Refuse instead.
                raise SystemExit(
                    f"REFUSED: item id {item_id} appears in both "
                    f"{source_of[item_id]} and {path.name}. Positional ids are "
                    "re-issued per derivation; two item files sharing an id "
                    "space would re-key labels to the wrong spans."
                )
            items_by_id[item_id] = item
            source_of[item_id] = path.name

    rows: list[dict] = []
    files: list[str] = []
    for path in sorted((REPO / "data" / "phase2_5" / "labels").glob("d1_*.jsonl")):
        if "bootstrap" in path.name:
            continue
        files.append(path.name)
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    by_item = read_labels(rows)
    gold: dict[str, str] = {}
    unresolved = 0
    stale_links: list[str] = []
    for item_id, label in by_item.items():
        item = items_by_id.get(item_id)
        if item is None:
            unresolved += 1
            continue
        action, entity = parse_d1_gold(label)
        if (
            action == "LINK_EXISTING"
            and graph_entity_ids is not None
            and entity not in graph_entity_ids
        ):
            # **A label naming an entity this graph does not contain is stale,
            # not wrong** — and the distinction decides whether it may be
            # scored.  The spike minted candidate ids positionally
            # (``e_41698283_000``, `scripts/phase2_5/build_items.py`) and
            # ``annotate.py`` baked them into the label string; Phase 6 derives
            # entity ids by content hash, so **0 of 7 spike link labels name a
            # node that exists here** (measured 14 Aug 2026).  Scoring those as
            # model failures would blame every arm for the labels' provenance,
            # and counting them as candidate-recall misses would blame the
            # generator — they are neither, so they are excluded and named.
            stale_links.append(item_id)
            continue
        gold[span_id(item["turn_id"], item["start"], item["end"])] = label
    return gold, {
        "label_files": files,
        "labels_read": len(by_item),
        "keyed_by": "span_id(turn_id, start, end) — item_id is re-issued per "
        "derivation and joining on it attaches labels to the wrong mentions",
        "labels_without_a_source_item": unresolved,
        "stale_link_labels_excluded": len(stale_links),
        "stale_link_reading": (
            "LINK_EXISTING labels naming an entity id outside this graph's "
            "namespace: collected against a different item generation, so they "
            "are neither a model failure nor a candidate-recall miss. Excluded "
            "from train AND test, and named here. Re-annotating the current D1 "
            "batch is what makes link supervision usable."
        ),
    }


def train_arms(log, d1_items, embedder, cfg) -> tuple[dict, list, dict]:
    """Train every Gate-1 arm at every seed; return test predictions and reports.

    **Predictions come from the test split only, and every arm sees the same
    items in the same order** — that is what makes the paired McNemar test
    valid, and `gate1.d1_report` now refuses a length mismatch rather than
    zip-truncating one.

    The multi-seed protocol: three seeds per arm (decision 10), and the arm's
    reported predictions are the **first seed's**, with every seed's dev loss and
    parameter count in the report.  Averaging predictions across seeds would
    invent an ensemble the plan never declared; reporting the spread and scoring
    one member is the honest form.
    """
    snapshot = ReplayGraphStore(log).at()
    entity_ids = {
        node_id
        for node_id, node in (getattr(snapshot, "_nodes", {}) or {}).items()
        if node.ntype == "Entity"
    }
    gold_by_span, gold_report = load_d1_gold(entity_ids)
    # Joined on the span, never on the item id (see load_d1_gold).
    labelled = [i for i in d1_items if i["span_id"] in gold_by_span]
    gold_labels = {i["item_id"]: gold_by_span[i["span_id"]] for i in labelled}
    gold_report["items_matched"] = len(labelled)
    gold_report["transfer_rate"] = (
        len(labelled) / gold_report["labels_read"] if gold_report["labels_read"] else 0.0
    )
    if not labelled:
        raise SystemExit(
            "REFUSED: no human D1 label survives the join to this graph's items "
            f"({gold_report}). A decisive run on zero supervision would report "
            "four arms agreeing on nothing."
        )

    # Question types for the stratified split (item 5): unstratified draws move
    # the rare classes between splits, and D1 items carry only the question id.
    types = {}
    try:
        index = corpus.question_index(corpus.load_corpus())
        types = {qid: index[qid]["question_type"] for qid in index}
    except (FileNotFoundError, ValueError) as exc:  # the corpus is gitignored
        gold_report["stratification"] = f"unstratified — corpus unavailable: {exc}"
    questions = sorted({i["question_id"] for i in labelled})
    splits = split_questions(questions, types)
    by_split = {
        part: [i for i in labelled if i["question_id"] in set(ids)]
        for part, ids in splits.items()
    }

    texts = [i["mention"] for i in labelled]
    vectors = embedder.embed(texts) if texts else []
    mention_vectors = {
        item["item_id"]: torch.from_numpy(vectors[ix]).float()
        for ix, item in enumerate(labelled)
    }

    gold_rows = [
        dict(zip(("action", "entity_id"), parse_d1_gold(gold_labels[i["item_id"]])))
        for i in by_split["test"]
    ]
    # The power analysis asks for D1 n_test ≈ 627 at δ = 0.05, ψ = 0.2
    # (`data/phase2_5/power.json`).  A decisive run far below it is not wrong,
    # but it must not be silent: an underpowered null reads exactly like a
    # measured null, and Gate 1's stop-or-redesign rule turns on that reading.
    reports_power = {
        "n_test": len(gold_rows),
        "planning_figure": 627,
        "reading": (
            "below the planning figure this run cannot detect a small effect; a "
            "non-significant McNemar here means 'not measured at this n', never "
            "'no difference'"
        ),
    }

    arms: dict[str, list] = {}
    reports: dict[str, Any] = {
        "gold": gold_report,
        "power": reports_power,
        "splits": {k: len(v) for k, v in by_split.items()},
        "arms": {},
    }

    similarity = build_arm("similarity")
    similarity.fit(by_split["dev"], gold_labels)
    arms["similarity"] = predict_d1(similarity, None, by_split["test"])
    reports["arms"]["similarity"] = {
        "threshold": similarity.threshold, "parameters": 0, "seeds": "n/a (no learning)",
    }

    for name in ("E1", "E2", "E3"):
        cache = FeatureCache(log, ARM_VARIANTS[name], embed=embedder.embed)
        per_seed = []
        first_predictions = None
        for seed in pins.TRAINING["seeds"]:
            arm = build_arm(name, seed=seed)
            per_seed.append(
                train_d1_arm(arm, cache, by_split, gold_labels, mention_vectors, seed=seed)
            )
            if first_predictions is None:
                first_predictions = predict_d1(
                    arm, cache, by_split["test"], mention_vectors
                )
        label = "proposed" if name == "E3" else name
        arms[label] = first_predictions or []
        reports["arms"][label] = {
            "encoder": name,
            "variant": ARM_VARIANTS[name],
            "parameters": per_seed[0]["parameters"],
            "per_seed": [
                {k: r[k] for k in ("seed", "epochs_run", "best_dev_loss")} for r in per_seed
            ],
            "budget": per_seed[0]["budget"],
            "unreachable_gold_in_train": per_seed[0]["train_items_unreachable"],
        }
    return arms, gold_rows, reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artefacts/phase6_gate1_smoke.json")
    # **Mutually exclusive, and that is a correctness guard, not tidiness.**
    # With both flags the entry-condition block (`if not args.smoke`) was
    # skipped while `smoke = not args.decisive` evaluated to False — a full
    # McNemar table stamped `smoke: false`, written against an unsigned
    # contract.  Found by the 14 Aug review, before the path ever ran.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="bootstrap labels; nothing quotable")
    mode.add_argument(
        "--decisive",
        action="store_true",
        help="the real Gate-1 run: trains every arm at seeds {13,42,7} and "
        "computes the McNemar table. Refuses unless all four entry conditions hold.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real-embedder", action="store_true")
    args = parser.parse_args()

    conditions = entry_conditions()
    unmet = [k for k, v in conditions.items() if not v["met"]]

    if not args.smoke:
        print("Gate 1 entry conditions:")
        for name, state in conditions.items():
            print(f"  [{'x' if state['met'] else ' '}] {name}: {state['detail']}")
        if unmet:
            print(
                f"\nREFUSED: {len(unmet)} entry condition(s) unmet — {', '.join(unmet)}.\n"
                "Gate 1 is one command on the day they hold. Use --smoke to exercise "
                "the machinery on bootstrap labels; its artefact quotes nothing."
            )
            return 1
        if args.decisive and not args.real_embedder:
            print(
                "\nREFUSED: --decisive without --real-embedder. The stub embedder "
                "fills 384 of E3's 398 input dimensions with meaningless noise, so "
                "the arm's entire declared difference from E2 (decision 15) would "
                "be hash values — the same failure `build_encoder` refuses for the "
                "HGT arm. A decisive run uses the pinned embedder."
            )
            return 1
        if not args.decisive:
            print(
                "\nEntry conditions hold. Re-run with --decisive to train every arm "
                "at seeds {13, 42, 7} and compute the Gate-1 table.\n"
                "The flag is explicit because a decisive run is the one whose "
                "numbers may be quoted, and fix F12's rule is that the decision "
                "rule was fixed first — it is in gate1.GATE1_RULE, predeclared."
            )
            return 1

    cfg = load_config()

    # **Stage B builds into its OWN log, never into Phase 5's.**  Found the hard
    # way on the first smoke run: appending node.add events to the pilot log made
    # the Phase-5 artefact's recorded digest stop reproducing, which would have
    # silently invalidated an experimental record.  The Stage-A log is an
    # *input*, and inputs are copied, not extended.
    run_dir = REPO / "artefacts" / "phase6"
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_b_log = run_dir / "events.jsonl"
    stage_b_log.unlink(missing_ok=True)
    shutil.copyfile(PILOT_LOG, stage_b_log)
    log = EventLog.open(stage_b_log)
    embedder = (
        Embedder(device=args.device, cache_dir=REPO / "artefacts" / "phase6" / "embed_cache")
        if args.real_embedder
        else StubEmbedder(pins.EMBEDDER["dim"])
    )

    print("constructing (stand-in decoder, turn order)...", flush=True)
    built = construct(
        log, embed=embedder.embed, k=pins.K_CANDIDATES, s=pins.S_PAIRS
    )
    d1, d2 = built["d1_items"], built["d2_items"]
    with_candidates = sum(1 for i in d1 if i["candidates"])
    print(
        f"  D1 items {len(d1)} ({with_candidates} with construction-time candidates) "
        f"· D2 items {len(d2)}",
        flush=True,
    )

    arms: dict = {}
    gold_rows: list = []
    training: dict = {}
    if args.decisive:
        print("training arms at seeds {13, 42, 7}...", flush=True)
        arms, gold_rows, training = train_arms(log, d1, embedder, cfg)

    artefact = gate.run_gate1(
        arms=arms,
        gold=gold_rows,
        smoke=not args.decisive,
        extras={
            "manifest": run_manifest(cfg, seed=cfg.seeds[0], root=REPO),
            "stage_b_fingerprint": pins.stage_b_fingerprint(),
            "endpoint_table_hash": pins.endpoint_table_hash(),
            "entry_conditions": conditions,
            "unmet_entry_conditions": unmet,
            "training": training,
            "source_log": str(PILOT_LOG.relative_to(REPO)),
            "stage_b_log": str(stage_b_log.relative_to(REPO)),
            "yields": yields(log),
            "items": {
                "d1": len(d1),
                "d1_with_candidates": with_candidates,
                "d2": len(d2),
            },
            "commit": built["commit"],
            "corruption_audit": built["corruption_audit"],
            "graph": built["graph"],
            "embedder": embedder.report() if hasattr(embedder, "report") else {"model": "stub"},
            "external_datasets": loader_artefact(limit=None),
        },
    )
    log.close()
    if hasattr(embedder, "flush"):
        embedder.flush()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_sanitize(artefact), indent=1, default=str, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )

    audit = built["corruption_audit"]
    print("\n=== smoke run ===")
    print(f"  D1 items            {len(d1)} ({with_candidates} with candidates)")
    print(f"  D2 items            {len(d2)}")
    print(f"  commits accepted    {built['commit']['accepted']}")
    print(f"  commits refused     {built['commit']['refused']}")
    print(f"  corruption audit    {'GREEN' if audit['green'] else 'FAILURES ' + str(audit['failures'])}")
    print(f"  graph               {built['graph']}")
    print(f"  comparisons         {artefact['comparisons']}")
    if artefact.get("smoke_notice"):
        print(f"\n  {artefact['smoke_notice']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
