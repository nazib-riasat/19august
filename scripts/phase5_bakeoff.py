"""The G2 extractor bakeoff — declared candidates, one predeclared rule, run once.

    python scripts/phase5_bakeoff.py --out artefacts/phase5_bakeoff.json
    python scripts/phase5_bakeoff.py --only A --limit 6      # a wiring check

**Three candidates were declared; C was withdrawn before the run** (project
owner, 13 Aug 2026: the extractor is Qwen2.5-3B, not the architecture's 7B).  So
the bakeoff decides between two *parse-failure strategies* on one model, which is
the question that matters here — the spike's 15.5% silent turn loss is the
defect, and repair-retry and grammar-constrained decoding are the two ways to fix
it.  The withdrawn row stays in the artefact: a predeclared candidate deleted
after the fact makes the table unreadable.

**The calibration slice is fixed, grouped by conversation, and disjoint from the
pilot's audit sample** (decision 2): the spike's 58 sampled turns plus the two
held-back turns that windowing dropped, taken deterministically from the pinned
corpus.  The slice definition lives in ``graft.ingest.bakeoff.calibration_slice``
— **one definition**, imported both here and by the pilot, which excludes these
turns from its audit draws; that is what makes the disjointness a property of the
code rather than a sentence in a plan.  The same groups, the same prompt and the
same production context recipe (window within conversation + rolling summary) run
for every candidate; nothing about the harness may differ between them or the
table measures the harness.

**The first run (13 Aug 2026) was aborted and its instrument corrected** — see
`PHASE5_DECISIONS.md`: the flat-list context mixed ten users' conversations, the
600-token cap truncated every failing generation, and the unclipped m = 10 window
overflowed the 8 GB card.  The aborted artefact is preserved as
``artefacts/phase5_bakeoff_AB.json``; nothing was frozen from it.

**The rule is applied by ``graft.ingest.bakeoff.decide`` and is not re-stated
here**, because a rule restated in two places is a rule with two versions.  What
this script owns is the artefact and the sub-audit worksheet stage 3 needs when
stage 2 ties.

**The winner is not written into the code by this script.**  It prints the
transcription line for ``graft/ingest/pins.py``, and a human moves it — the same
discipline as Phase 3's `N` and β, which the calibration gate measures and a
person writes into §6.  A script that edits its own frozen constants is a script
that can silently re-freeze them on a re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from graft.ingest import corpus  # noqa: E402
from graft.ingest.bakeoff import (  # noqa: E402
    SLICE_SIZE,
    calibration_slice,
    calibration_turn_ids,
    decide,
    run_candidate,
)
from graft.ingest.extractor import build_extractor  # noqa: E402
from graft.ingest.pins import (  # noqa: E402
    CONTEXT_CLIP_CHARS,
    CONTEXT_TURNS,
    DECODING,
    EXTRACTOR_CANDIDATES,
    RUNNABLE_CANDIDATES,
)
from graft.runtime import json_sanitize, run_manifest  # noqa: E402
from graft.config import load_config  # noqa: E402

#: The spike's sample, whose composition and seed are recorded in
#: ``PHASE2_5_DECISIONS.md`` §1.  Read from the artefact rather than re-sampled,
#: so the slice is literally the measured one.
SAMPLE = REPO / "data" / "phase2_5" / "sample.json"


def main() -> int:
    # **Measured, after a first version of this comment overstated it.**  These
    # docstrings carry U+2192 and U+03B2, Windows consoles default to cp1252, and
    # `--help` died on the description before printing a word of it -- that part
    # is reproduced, on six runners.
    #
    # The guard also covers `print` of *data*, but the original justification
    # ("a curly apostrophe would kill the run") was **wrong**: U+2019 and U+2014
    # are cp1252 0x92/0x97 and encode fine.  What LoCoMo actually holds outside
    # cp1252 is 18 occurrences of 11 characters -- 8 zero-width spaces and 9
    # emoji -- across 7 turns and 1 gold answer.  And no current print path in
    # these runners emits corpus text, so this is insurance against a future
    # debug print, not a live crash averted.  `scripts/phase3_calibrate.py` set
    # the convention; extended here 19 Aug 2026.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artefacts/phase5_bakeoff.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(RUNNABLE_CANDIDATES),
        help="run a subset; the artefact records which candidates were skipped",
    )
    parser.add_argument("--limit", type=int, default=None, help="wiring check only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    index = corpus.question_index(corpus.load_corpus())
    groups = calibration_slice(SAMPLE, index)
    if args.limit:
        kept: list = []
        remaining = args.limit
        for conv_id, turns in groups:
            if remaining <= 0:
                break
            kept.append((conv_id, turns[:remaining]))
            remaining -= len(turns[:remaining])
        groups = kept
    n_turns = sum(len(turns) for _, turns in groups)
    wanted = args.only or list(RUNNABLE_CANDIDATES)

    print(
        f"calibration slice: {n_turns} turns in {len(groups)} conversations; "
        f"candidates {wanted}",
        flush=True,
    )
    progress = None if args.quiet else (lambda line: print(line, flush=True))

    results = []
    for name in wanted:
        print(f"\n=== candidate {name}: {EXTRACTOR_CANDIDATES[name]['note']}", flush=True)
        results.append(
            run_candidate(
                name,
                groups,
                lambda n=name: build_extractor(n, device=args.device),
                context_turns=CONTEXT_TURNS,
                progress=progress,
            )
        )

    verdict = decide(results)
    cfg = load_config()
    artefact = {
        "phase": 5,
        "gap": "G2",
        "decision": 2,
        "manifest": run_manifest(cfg, seed=cfg.seeds[0], root=REPO),
        "slice": {
            "n_turns": n_turns,
            "declared_size": SLICE_SIZE,
            "source": "the Phase-2.5 sample plus held-back turns, from the pinned corpus",
            "corpus_sha256": corpus.CORPUS_SHA256,
            "conversations": [conv_id for conv_id, _ in groups],
            "turn_ids": [t.turn_id for _, turns in groups for t in turns],
            "context_recipe": (
                "production: window within conversation + rolling summary via the "
                "candidate's own model; window turns clipped to "
                f"{CONTEXT_CLIP_CHARS} chars; max_new_tokens "
                f"{DECODING['max_new_tokens']}. Declared slice property: each "
                "session contributes its windowed turns, not the full session "
                "stream, so summaries are built over that windowed stream."
            ),
        },
        "candidates_run": wanted,
        "candidates_skipped": sorted(set(RUNNABLE_CANDIDATES) - set(wanted)),
        "candidates_withdrawn": {
            name: config["withdrawn"]
            for name, config in sorted(EXTRACTOR_CANDIDATES.items())
            if config.get("withdrawn")
        },
        "partial": bool(args.limit) or len(wanted) < len(RUNNABLE_CANDIDATES),
        **verdict,
        "samples": {r.candidate: r.samples[:20] for r in results},
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_sanitize(artefact), indent=1, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )

    print("\n=== bakeoff table ===")
    header = f"{'cand':<5} {'turns':>5} {'parse fail':>11} {'grounded':>9} {'g/min':>8} {'tok/s':>7}"
    print(header)
    for row in verdict["table"]:
        if row["error"]:
            print(f"{row['candidate']:<5} did not run: {row['error']}")
            continue
        print(
            f"{row['candidate']:<5} {row['turns']:>5} "
            f"{row['parse_failure_rate']:>10.1%} {row['assertions_grounded']:>9} "
            f"{row['grounded_per_minute']:>8.2f} {row['tokens_per_second']:>7.1f}"
        )
    print(f"\nrule       {verdict['rule']}")
    print(f"verdict    {verdict['verdict']}  ({verdict['decided_by']})")
    if verdict["verdict"] == "winner":
        print(f"winner     {verdict['winner']}")
        print("\nTranscribe into graft/ingest/pins.py (decision 2), by hand:")
        print(f"    EXTRACTOR = dict(EXTRACTOR_CANDIDATES[{verdict['winner']!r}])")
    elif verdict["verdict"] == "tie":
        print(f"tied       {verdict['tied']} — stage 3 is the human span sub-audit")
    else:
        print(verdict["reading"])
    if artefact["partial"]:
        print("\nNOTE: this run is PARTIAL and must not be used to freeze decision 2.")
    print(f"\nexcluded from the pilot's audit draws: {len(calibration_turn_ids(groups))} turn ids")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
