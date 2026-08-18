"""Phase 10's runner: the read path end to end, and the five-ceiling table.

    python scripts/phase10_read.py --smoke                 # stub reader, no GPU
    python scripts/phase10_read.py --questions 10          # run R3: the real reader
    python scripts/phase10_read.py --questions 10 --no-contested

**This is the architecture's Phase-10 exit criterion**: *"End-to-end smoke test
on the pilot: write path ingests a conversation, read path answers/abstains with
citations; ceilings 2–5 runnable; config-hash equality verified across two
runs."*

**Nothing this prints is a result until the stamp says so.**  Three artefacts the
read path consumes are untrained or out-of-domain today — the Stage-D policy
(Phase-9 step 6 has not run), the gate threshold (Phase-8 trained on MuSiQue
contrast pairs, not conversation) and, under ``--smoke``, the reader itself.  Any
one of them makes the run a **wiring test**, and ``ReadPathStamp`` reports that as
a disjunction rather than a score.  `PHASE7_DECISIONS.md` §7 records a
``--smoke`` artefact being quoted as measured; this refuses to let that happen
quietly.

**Fix F7 is honoured by construction, not by ordering.**  The gate decision is
computed *before* the reader loads and passed in, so no code path holds a gate
model and a 3-billion-parameter reader at the same time.  Run R1 measured the
reader alone at 6.317 GB peak on an 8 GB card, so this is a real constraint and
``ModelSlot`` refuses a second concurrent slot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from graft.canonical import digest_of  # noqa: E402
from graft.config import load_config  # noqa: E402
from graft.diagnostics import ceilings as C  # noqa: E402
from graft.ledger import Ledger  # noqa: E402
from graft.reader import pins as rpins  # noqa: E402
from graft.reader.orchestrator import ReadPathStamp, aggregate, answer  # noqa: E402
from graft.reader.serialize import ProofSerializer  # noqa: E402
from graft.runtime import deterministic_view, json_sanitize, run_manifest  # noqa: E402
from graft.schemas import GateDecision, Obligations  # noqa: E402
from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer  # noqa: E402
from graft.setgen.distill import HeadScorer, build_head  # noqa: E402
from graft.setgen.policy import Policy  # noqa: E402
from graft.setgen.proofs import SourceDoc, build_example  # noqa: E402
from graft.setgen.realenv import RealEnvironment  # noqa: E402

OUT = REPO / "artefacts" / "phase10_read.json"

HONESTY_STAMP: dict[str, str] = {
    "purpose": (
        "the architecture's Phase-10 exit criterion: an end-to-end read-path "
        "smoke plus the five-ceiling table. NOT a Gate-4 result."
    ),
    "untrained": (
        "the Stage-D policy is UNTRAINED (Phase-9 step 6 has not run) and the "
        "gate threshold is Phase-8's MuSiQue-trained placeholder, not a "
        "conversational one. Every record carries a ReadPathStamp saying so."
    ),
    "ceilings": (
        "ceilings 1 and 2 are support-gate and graph-construction SURVIVAL rates, "
        "not oracle searches against annotated gold proofs -- that annotation does "
        "not exist on conversation (CLAUDE.md §7). Each reports its tier."
    ),
    "no_training": (
        "no gradient step is taken anywhere in this phase. The reader is frozen "
        "and post-hoc verification is declined (Phase-10 decision 11)."
    ),
    "budget": (
        "ceiling 4 is reported at EVERY rung of the ladder. CLAUDE.md §8 records "
        "that quoting a packing result at one cherry-picked budget was one of this "
        "project's own caught errors."
    ),
}


def build_questions(count: int) -> list[dict[str, Any]]:
    """Hand-built read-path fixtures in the shape Stage C would deliver.

    The pilot's own graph is a Phase-5 artefact and wiring this runner to it is
    Stage D of the plan (blocked on scope-c).  What this runner exercises is the
    **read path**, so the pool is constructed here through the same
    ``proofs.build_example`` boundary Phase 9 uses — one pool mapping
    project-wide, per the Phase-7 rule.
    """
    out: list[dict[str, Any]] = []
    for i in range(count):
        # **`is_gold` is set, and that is not cosmetic.** Run R3 was first written
        # without it, so `ProofExample.gold_atom_ids` came back empty and the
        # runner fell back to `sorted(pool.ids())[:2]` -- atoms in *content-hash*
        # order, which selected one Claim node and one `about_entity` edge. The
        # "gold proof" handed to ceiling 5 therefore contained no answer-bearing
        # claim on several questions, the reader correctly abstained, and ceiling 5
        # read 0.6 as though that were a reader limitation. It was noise.
        #
        # Only the first two documents state the birthplace and they are marked as
        # the gold proof; the rest are distractors. On real data this is Phase 7's
        # Tier-A / Tier-B definition, which is why every ceiling stamps its tier.
        docs = [
            SourceDoc(
                f"p{j}",
                f"Session {i}: Ada Lovelace was born in London in 18{15 + j}.",
                ("London",),
                is_gold=(j < 2),
            )
            for j in range(4)
        ] + [
            SourceDoc(f"d{j}", f"Session {i}: unrelated note number {j}.", ("Note",))
            for j in range(2)
        ]
        example = build_example(
            f"q{i}", docs,
            Obligations(entity_anchor="London", scope=(f"q{i}",)),
            {d.doc_id: 1.0 - 0.05 * k for k, d in enumerate(docs)},
        )
        out.append(
            {
                "question": "Where was Ada Lovelace born?",
                "gold_answer": "London",
                "aliases": ["London, England"],
                "example": example,
                # Phase-8's threshold, carried as the placeholder it is.
                "gate": GateDecision(
                    p_answerable=0.9 if i % 5 else 0.2,
                    answerable=bool(i % 5),
                    threshold=0.6266720890998840,
                    arm="pool_only",
                ),
            }
        )
    return out


def run(*, smoke: bool, questions: int, out_path: Path, contested: bool) -> int:
    config = load_config()
    started = time.perf_counter()
    items = build_questions(questions)

    reader: Any = None
    if smoke:
        def read_fn(evidence: str, question: str) -> str:
            # A deterministic stand-in. It answers correctly and cites, so the
            # wiring is exercised; it is NOT the reader and the stamp says so.
            return "London [c1]"
        count_tokens = None
        reader_report: dict[str, Any] = {"stub": True}
    else:
        from graft.reader.read import Reader

        reader = Reader()
        reader.__enter__()
        read_fn = reader.generate
        count_tokens = reader.count_tokens
        reader_report = reader.report()

    stamp = ReadPathStamp(
        policy_trained=False,
        gate_source="musique_placeholder",
        scorer_source="distilled_head",
        token_counter="approx_tokens" if count_tokens is None else "reader_tokenizer",
        ordering="u_shaped_inference_computable",
        notes=(
            "Stage-D policy untrained: Phase-9 step 6 has not run",
            "gate threshold from Phase-8 MuSiQue Stage A, not conversational",
        ) + (("reader is a stub",) if smoke else ()),
    )

    try:
        results = []
        per_question: list[dict[str, Any]] = []
        ceiling_rows: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            example = item["example"]
            env = RealEnvironment(example, config, range_samples=0)
            torch.manual_seed(config.seeds[0])
            featurizer = RealFeaturizer(
                example, Policy(*RealFeaturizer.dims(), hidden=16), config, delta_d=False
            )
            scorer = HeadScorer(build_head(ATOM_WIDTH, seed=config.seeds[0]), featurizer)
            ledger = Ledger.from_config(config)
            # The Reader is built once outside this loop and the Ledger is per
            # query, so the reference is re-pointed rather than re-constructed.
            # Metering lives inside the wrapper (`ingest/extractor.py`'s rule), so
            # this is the only wiring the runner owes it.
            if reader is not None:
                reader.ledger = ledger

            with ledger.query_scope(f"q{index}"):
                result = answer(
                    item["question"],
                    env=env, featurizer=featurizer, scorer=scorer, read_fn=read_fn,
                    gate_decision=item["gate"], aliases=item["aliases"],
                    rng=np.random.default_rng(config.seeds[0] + index),
                    ledger=ledger, config=config, stamp=stamp,
                    count_tokens=count_tokens, query_id=f"q{index}",
                    contested_check=contested,
                )
            results.append(result)
            per_question.append({"question_id": f"q{index}", **result.report()})

            # -- the five ceilings, per question ---------------------------
            serializer = ProofSerializer(
                example.snapshot, example.pool, config=config,
                count_tokens=count_tokens,
                counter_name="reader_tokenizer" if count_tokens is not None else None,
            )
            # **Refuses rather than guesses.** The first version fell back to
            # `sorted(pool.ids())[:2]` when gold was empty, which is content-hash
            # order and therefore arbitrary -- see `build_questions`. A ceiling
            # computed against arbitrary atoms is not a ceiling, and silently
            # producing one is worse than producing none.
            gold = example.gold_atom_ids
            if not gold:
                raise ValueError(
                    f"q{index} has no gold atoms; ceilings 4 and 5 are defined "
                    "against a gold proof and there is nothing to define them "
                    "against. Set `is_gold` on the answer-bearing documents."
                )
            ceiling_rows.append(
                {
                    "question_id": f"q{index}",
                    **C.all_ceilings(
                        snapshot=example.snapshot, conv_id=f"q{index}",
                        retrieved=sorted(example.pool.ids()), gold=gold,
                        serializer=serializer, obligations=example.obligations,
                        scores=example.atom_scores, question=item["question"],
                        read_fn=read_fn, gold_answer=item["gold_answer"],
                        aliases=item["aliases"], config=config,
                    ),
                }
            )
    finally:
        if reader is not None:
            reader.__exit__(None, None, None)

    summary = aggregate(results)
    elapsed = time.perf_counter() - started

    def _ceiling_mean(name: str) -> Any:
        vals = [
            row[name]["ceiling"]
            for row in ceiling_rows
            if row[name].get("available") and isinstance(row[name].get("ceiling"), (int, float, bool))
        ]
        return (sum(float(v) for v in vals) / len(vals)) if vals else None

    artefact = {
        "phase": 10,
        "stage": "C (orchestrator) + the five ceilings",
        "smoke": bool(smoke),
        "honesty_stamp": HONESTY_STAMP,
        "read_path_stamp": stamp.to_dict(),
        "stage_e_fingerprint": rpins.stage_e_fingerprint(),
        "prompt_sha": rpins.PROMPT_SHA,
        "budget_ladder": list(rpins.BUDGET_LADDER),
        "serialization_budget_tokens": config.serialization_budget_tokens,
        "post_hoc_verification": dict(rpins.POST_HOC_VERIFICATION),
        "reader": reader_report,
        "summary": summary,
        "ceiling_means": {
            f"ceiling_{i}": _ceiling_mean(name)
            for i, name in enumerate(
                ("1_extraction", "2_graph", "3_candidate", "4_packing", "5_reader"), start=1
            )
        },
        "per_question": per_question,
        "ceilings": ceiling_rows,
        "deferred_by_name": {
            "ceilings_1_2_at_tier_b": "needs scope-c ingestion and gold proof annotation",
            "trained_stage_d_policy": "Phase-9 step 6",
            "conversational_gate_threshold": "Phase-8 Stage B",
            "system_baselines": "Phase 11 — full-context, matched-budget RAG, Mem0",
        },
        "wall_clock_s": round(elapsed, 2),
        "manifest": run_manifest(config, config.seeds[0]),
    }
    artefact["determinism_digest"] = digest_of(
        deterministic_view(
            {
                k: artefact[k]
                for k in ("stage_e_fingerprint", "prompt_sha", "read_path_stamp", "ceiling_means")
            }
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(json_sanitize(artefact), indent=1, sort_keys=True), encoding="utf-8"
    )

    shown = out_path.resolve()
    print(f"wrote {shown.relative_to(REPO) if shown.is_relative_to(REPO) else shown}  ({elapsed:.1f}s)")
    print(f"  questions {summary['queries']}  answered {summary['answered']}  "
          f"contested {summary['contested']}  abstained {summary['abstained']}")
    print(f"  abstain by cause: {summary['abstain_by_cause']}")
    print(f"  mean proof size {summary['mean_proof_size']}  mean citations {summary['mean_citations']}")
    print(f"  contested extra reader calls: {summary['contested_extra_reader_calls']}")
    print(f"\n  {'ceiling':<16}{'mean':>10}   tier")
    for i, name in enumerate(
        ("1_extraction", "2_graph", "3_candidate", "4_packing", "5_reader"), start=1
    ):
        value = artefact["ceiling_means"][f"ceiling_{i}"]
        tier = ceiling_rows[0][name].get("tier", "-") if ceiling_rows else "-"
        shown_value = "n/a" if value is None else f"{value:.4f}"
        print(f"  {name:<16}{shown_value:>10}   {tier}")
    print(f"\n  stage-E fingerprint {rpins.stage_e_fingerprint()[:16]}  prompt {rpins.PROMPT_SHA[:16]}")
    if stamp.is_wiring_test:
        print("\n  *** WIRING TEST — no number above is a result. ***")
        for note in stamp.notes:
            print(f"      - {note}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="stub reader, no GPU")
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--no-contested", action="store_true", help="skip the G8 comparison")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return run(
        smoke=args.smoke, questions=args.questions,
        out_path=args.out, contested=not args.no_contested,
    )


if __name__ == "__main__":
    raise SystemExit(main())
