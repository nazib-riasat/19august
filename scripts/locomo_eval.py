"""LoCoMo end-to-end — Stage C → gate → Stage D → Stage E, scored against the
reference table.

The runner Phase 10 deferred.  ``scripts/phase10_read.py`` drives the read path
over **hand-built fixtures** ("in the shape Stage C would deliver"), which is
what made its R1-R3 runs wiring tests.  This one drives it over a real ingested
graph, which is the only way a LoCoMo number exists.

**Four decisions taken here to keep the run cheap, each recorded in the stamp
rather than left implicit.**  The deadline is three days and the reader is the
only expensive component, so every decision below buys GPU time back:

1. **Obligations are deterministic; LLM parsing is deferred by name.**  Parsing
   them costs one extra generation per question -- ~1.1 h over 1,986 questions --
   and the temporal filter is *fail-open* by design (Phase-7 decision), so a
   missing ``time_constraint`` is safe rather than wrong.  The cost is that the
   entity anchor is unset, so the entity and expand channels contribute nothing;
   `PHASE7_DECISIONS.md` §3.2 already measured that channel matching an anchor in
   only 3 of 10 questions, so this is a smaller loss than it sounds, and the stamp
   says so either way.

   **Deferred by name rather than offered as a flag**, because it cannot be one:
   fix F7's ``ModelSlot`` refuses to hold the extractor and the reader at once, so
   LLM obligation parsing is a *separate stage-sequential pass* that writes a
   cache this runner then reads.  A ``--parse-obligations`` flag existed in the
   first version and was cosmetic -- it changed the stamp text and nothing else,
   so an artefact could claim "obligations LLM-parsed" for a run that parsed
   none.  Removed rather than left: a label that disagrees with what ran is worse
   than an absent capability.

2. **Gate features are recorded for every question; gating is off by default.**
   The gate is a small model over Stage-C outputs with no LLM in it, while the
   reader is the whole cost.  So the features are written out per question and a
   threshold can be applied *post hoc* -- abstention accuracy, the full
   risk-coverage curve, both prevalences -- without a second reader pass.
   Default off also matches the reference table, whose systems never abstain.

3. **Gold proof atoms come from LoCoMo's own evidence markers**, through
   ``retrieve.recall.tier_a_gold``, which takes turn ids and knows nothing about
   which corpus produced them.  Tier A, stamped as Tier A: it is the
   evidence-derived closed superset, not an `H`-minimised proof.

4. **Ceilings are opt-in.**  Ceiling 5 serialises a gold proof and reads it, so
   ``--ceilings`` roughly doubles reader cost.  Worth one small subset, not the
   whole corpus.

**Resumable, like ingestion.**  Per-question rows are appended to a JSONL as they
complete and re-read on restart, so a killed run resumes rather than restarts and
the partial file is always a valid evaluation set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from graft.config import load_config  # noqa: E402
from graft.diagnostics import ceilings as C  # noqa: E402
from graft.diagnostics.report import build_report  # noqa: E402
from graft.eventlog import EventLog  # noqa: E402
from graft.gate import features as gfeatures  # noqa: E402
from graft.gate import pins as gpins  # noqa: E402
from graft.graphbuild.embed import Embedder  # noqa: E402
from graft.graphstore import ReplayGraphStore  # noqa: E402
from graft.ingest import locomo  # noqa: E402
from graft.ledger import Ledger  # noqa: E402
from graft.reader.orchestrator import ReadPathStamp, answer, cost_report  # noqa: E402
from graft.reader import pins as rpins  # noqa: E402
from graft.reader.serialize import ProofSerializer  # noqa: E402
from graft.retrieve.bm25 import BM25Channel  # noqa: E402
from graft.retrieve.dense import DenseChannel  # noqa: E402
from graft.retrieve.entity import entity_channel, match_entities  # noqa: E402
from graft.retrieve.expand import expand_channel  # noqa: E402
from graft.retrieve.fuse import assemble  # noqa: E402
from graft.retrieve.pool import eligible_nodes, uncapped_pool  # noqa: E402
from graft.retrieve.recall import saturation, tier_a_gold  # noqa: E402
from graft.schemas import Obligations  # noqa: E402
from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer  # noqa: E402
from graft.setgen.distill import HeadScorer, build_head  # noqa: E402
from graft.setgen.realenv import RealEnvironment  # noqa: E402
from graft.setgen.policy import Policy  # noqa: E402
from graft.setgen.proofs import ProofExample  # noqa: E402


def default_obligations(question_id: str) -> Obligations:
    """The cheap obligation: scope only, no anchor, no time constraint.

    Not a guess at the question's semantics -- deliberately the *empty*
    obligation, so nothing is asserted that was not parsed.  Phase-1 G5's
    convention scores an unbounded interval as ``temporal_correctness = 1.0``, and
    the temporal filter fails open, so this weakens retrieval rather than
    corrupting validity.
    """
    return Obligations(
        entity_anchor=None,
        value_type=None,
        time_constraint=None,
        needs_source=False,
        aggregate=False,
        scope=(question_id,),
    )


def next_iteration(results_dir: "Path") -> int:
    """The next free iteration number under results/, by scanning what exists.

    Owner's convention (19 Aug 2026): every eval lands in the project's
    ``results/`` folder as ``locomo_eval{N}.json`` + ``locomo_eval_rows{N}.jsonl``.
    Run 3's artefacts were moved there as iteration 1, so numbering continues
    from what the folder actually holds rather than from memory.
    """
    import re
    top = 0
    if results_dir.is_dir():
        for f in results_dir.glob("locomo_eval*.json"):
            m = re.match(r"locomo_eval(\d+)\.json$", f.name)
            if m:
                top = max(top, int(m.group(1)))
    return top + 1


def resolve_out_paths(args, repo: "Path") -> tuple["Path", "Path", int]:
    """``(artefact_path, rows_path, iteration)`` under results/, auto-numbered.

    **Resume survives the numbering**: if the newest iteration has rows but no
    final artefact, that run crashed mid-way -- reuse its number so the rows
    resume, exactly as before. ``--fresh`` always claims a new number. Explicit
    ``--out``/``--rows`` bypass the scheme entirely (tests use this).
    """
    results = repo / "results"
    results.mkdir(exist_ok=True)
    explicit_out = args.out != "artefacts/locomo_eval.json"
    explicit_rows = args.rows != "artefacts/locomo_eval_rows.jsonl"
    if explicit_out or explicit_rows:
        return repo / args.out, repo / args.rows, -1
    n = next_iteration(results)
    prev_rows = results / f"locomo_eval_rows{n - 1}.jsonl"
    prev_json = results / f"locomo_eval{n - 1}.json"
    if not args.fresh and n > 1 and prev_rows.is_file() and not prev_json.is_file():
        n -= 1  # crashed run: resume its rows under its own number
    rows = results / f"locomo_eval_rows{n}.jsonl"
    # --fresh reclaims a crashed iteration's number; its stale rows must go, or
    # the restart appends onto them (run-3's rows file carried 150 stale subset
    # rows for exactly this reason).
    if args.fresh and rows.is_file():
        rows.unlink()
    return results / f"locomo_eval{n}.json", rows, n


def pick_run_dir(requested: str, repo: "Path") -> str:
    """Prefer the Stage-B log when the caller asked for the plain default.

    The 19 Aug 2026 ingestion run exposed a missing stage: Stage A writes no
    nodes, Stage C retrieves over nodes, and the join between them is Phase 6's
    stand-in constructor (`scripts/locomo_stageb.py`). When that stage has run,
    its log is the one to evaluate over; an explicit --run-dir always wins.
    """
    if requested != "artefacts/locomo":
        return requested
    stageb = repo / "artefacts" / "locomo_stageb" / "events.jsonl"
    return "artefacts/locomo_stageb" if stageb.is_file() else requested


def require_nodes(snapshot) -> None:
    """Refuse a graph Stage C cannot retrieve from, and say which stage is missing.

    Without this, a Stage-A-only log evaluates to empty pools and 1,986
    fallback abstentions -- a 35-minute reader pass producing a number that
    looks like total system failure and means "a stage was skipped". The first
    version of this runner had exactly that hole: its join tests used a fixture
    with nodes pre-built, so they passed while the real pipeline lacked the
    stand-in construction (PHASE11_DECISIONS.md 1.8).
    """
    counts = snapshot.counts()
    if counts.get("nodes", 0) == 0:
        raise SystemExit(
            f"the graph has {counts.get('assertions', 0)} assertions but ZERO "
            "nodes -- Stage C retrieves over nodes, so every question would "
            "abstain. Run the missing stage first:\n"
            "  python scripts/locomo_stageb.py\n"
            "then re-run this eval with --run-dir artefacts/locomo_stageb."
        )


class ChannelCache:
    """One BM25 index and one corpus embedding per conversation, not per question.

    **The single largest waste in the first version of this runner.** Both
    channels build an index over the conversation's eligible nodes in their
    constructor -- ``BM25Channel._build`` tokenises the corpus,
    ``DenseChannel._build`` *embeds* it -- and the loop constructed both for every
    question. LoCoMo has 96 to 260 questions per conversation, so the corpus was
    being re-embedded up to 260 times to answer 260 questions about it.

    Safe to cache because both are pure functions of ``(snapshot, conv_id)``: the
    snapshot is a pinned event-log offset and does not move during a read pass.
    A **one-entry** cache is enough, and bounds memory to a single conversation's
    matrix, because the question list is built conversation by conversation and is
    therefore contiguous in ``conv_id``. Keyed rather than assumed, so a caller
    that interleaves conversations gets correctness at the cost of rebuilds
    instead of silently wrong pools.

    Cost that is *not* removed: the per-question **question** encode, which is a
    real per-query cost and stays metered as one (`graft/retrieve/dense.py`).
    """

    def __init__(self, snapshot, embedder, config) -> None:
        self.snapshot = snapshot
        self.embedder = embedder
        self.config = config
        self._conv_id: str | None = None
        self._bm25: Any = None
        self._dense: Any = None
        self.builds = 0
        self._turn_conv: str | None = None
        self._turns: Any = []
        self._turn_bm25: Any = None
        self._turn_matrix: Any = None
        self.turn_builds = 0

    def turns_for(self, conv_id: str):
        """``(turn_ids, texts, bm25 retriever, dense matrix)`` for one conversation.

        **The raw-dialogue tier** (run 3).  Every reference system on LoCoMo --
        Mem-T's own `Mraw` row, Mem0, matched-budget RAG -- shows its reader raw
        conversation text; this project showed it only extractor-derived claims.
        Measured cost of that: 46% of questions have no gold atom at all, because
        an extraction miss or a quarantine removes the evidence before retrieval
        can ever see it, and `H` cannot certify what was never stored.

        Raw turns are already held as provenance, so this adds no new store and
        no new claim -- it shows the reader what the graph was built *from*.
        Built once per conversation and cached, like the channels above.
        """
        if conv_id != self._turn_conv:
            import bm25s
            import numpy as np

            from graft.retrieve.pins import BM25

            turns = [
                t for t in self.snapshot._turns.values() if t.conv_id == conv_id
            ]
            turns.sort(key=lambda t: t.turn_id)
            texts = [t.text for t in turns]
            retriever = None
            if texts:
                tokens = bm25s.tokenize(
                    texts, stopwords=BM25["stopwords"], show_progress=False
                )
                retriever = bm25s.BM25(
                    method=BM25["method"], k1=BM25["k1"], b=BM25["b"]
                )
                try:
                    retriever.index(tokens, show_progress=False)
                except ValueError:
                    retriever = None
            matrix = (
                np.asarray(self.embedder.embed(texts), dtype=np.float32)
                if texts else np.zeros((0, 1), dtype=np.float32)
            )
            self._turn_conv = conv_id
            self._turns = turns
            self._turn_bm25 = retriever
            self._turn_matrix = matrix
            self.turn_builds += 1
        return self._turns, self._turn_bm25, self._turn_matrix

    def for_conv(self, conv_id: str, ledger):
        if conv_id != self._conv_id:
            self._bm25 = BM25Channel(self.snapshot, conv_id, config=self.config)
            self._dense = DenseChannel(
                self.snapshot, self.embedder, conv_id, config=self.config, ledger=ledger
            )
            self._conv_id = conv_id
            self.builds += 1
        else:
            # The ledger is per query, so the cached channel's reference has to be
            # re-pointed or the question encode would be charged to a dead scope.
            self._dense.ledger = ledger
        return self._bm25, self._dense


_WRAPPED = re.compile(r"^\[([^\[\]]+)\]$")
_FRAGMENT = re.compile(r"\[[^\]]*\]")
#: A bracketed token shaped like a claim id rather than like an answer: up to
#: three letters, optionally followed by digits.  ``c12``, ``ci`` and a
#: mis-transcribed ``č6`` all match; ``Audrey`` and ``2023-05-31`` do not.
_IDLIKE = re.compile(r"^[^\W\d_]{0,3}\d*$", re.UNICODE)


def _cleaned(result) -> dict:
    """The three outcome fields for one row, after :func:`clean_answer`.

    An abstention the runner *derives* keeps ``fallback`` as its cause: the two
    ``ABSTAIN_CAUSES`` are ``gate`` and ``fallback``, the gate is off in this
    phase, and inventing a third would break the closed vocabulary
    `PHASE5_DECISIONS.md` §1 keeps for exactly this reason. ``junk_answer``
    carries the distinction instead, so the rate is countable without flattening
    it into the cause split.
    """
    record = result.record
    if record.outcome != "answer":
        return {
            "outcome": record.outcome,
            "abstain_cause": record.abstain_cause,
            "answer_text": record.answer_text,
            "junk_answer": False,
        }
    text, abstained, junk = clean_answer(record.answer_text or "")
    if abstained:
        return {
            "outcome": "abstain",
            "abstain_cause": "fallback",
            "answer_text": record.answer_text,
            "junk_answer": junk,
        }
    return {
        "outcome": "answer",
        "abstain_cause": None,
        "answer_text": text,
        "junk_answer": False,
    }


def clean_answer(text: str) -> tuple[str, bool, bool]:
    """``(text, abstained, junk)`` -- runner-level hygiene on one generation.

    **Deliberately here and not in ``graft/reader/parse.py``.**  The core parser
    is the frozen read path every phase shares and its rules ride in
    ``stage_e_fingerprint``; this is a *reporting* decision about one corpus's
    observed failure shapes, so it lives with the runner that observed them.
    ``normalise_answer`` and ``token_f1`` are untouched -- nothing here changes
    how a scored string is scored, only which strings count as answers.

    Three shapes, all measured on run 2's 1,986 rows:

    * **wrapped answer** -- the whole answer inside one bracket pair
      (``[Audrey]``, ``[2023-05-31]``): the reader copied citation syntax onto
      the answer itself.  Unwrapped, because the content *is* the answer;
    * **bracketed abstention** -- ``[INSUFFICIENT EVIDENCE]``;
    * **junk** -- nothing survives once bracket fragments are removed
      (``[c12]``, ``[?]``, ``[ci]``).  172 of run 2's 1,009 answers were this.
      They can only ever score 0, and counting them as answers hid a failure
      inside the F1 mean instead of putting it in the abstention rate where it
      is attributable.  Reported as ``junk_answer`` so the rate stays visible.

    **Order matters and the first version had it wrong**: unwrapping before the
    junk test turned ``[c12]`` into the perfectly good-looking token ``c12``.
    The junk test therefore runs on the *original* string, and a wrapped payload
    is only rescued when it does not look like a claim id.
    """
    from graft.reader.parse import normalise_answer
    from graft.reader.pins import INSUFFICIENT

    raw = (text or "").strip()
    inner = _WRAPPED.match(raw)
    payload = inner.group(1).strip() if inner else raw

    if payload.casefold() == INSUFFICIENT.casefold():
        return payload, True, False

    # **The junk test removes only ID-LIKE fragments** (corrected after the
    # run-3 subset, 19 Aug 2026). The previous version stripped EVERY bracket
    # fragment, so ``[25 May 2023][c1]`` -- a correct date answer wearing
    # citation syntax -- lost its date along with its citation and died as junk;
    # 5 of the subset's 13 junk rows were recoverable answers of this shape.
    # A fragment counts as id-like when its inner text matches ``_IDLIKE``
    # (``c12``, ``č6``, ``ci``, ``7``) or normalises to nothing (``?``).
    # ``[25 May 2023]`` survives, so the row stays an answer -- and the returned
    # text is the ORIGINAL, untouched: ``normalise_answer`` already discards
    # brackets at scoring time, so no rewrite is needed and the stored
    # ``answer_text`` stays faithful to what the reader generated.
    def _idlike(m) -> bool:
        inner_text = m.group(0)[1:-1].strip()
        return bool(_IDLIKE.match(inner_text)) or not normalise_answer(inner_text)

    survivors = _FRAGMENT.sub(lambda m: " " if _idlike(m) else m.group(0), raw)
    if not normalise_answer(survivors):
        # A wholly-wrapped payload survives only if it does not look like an id.
        if inner and not _IDLIKE.match(payload) and normalise_answer(payload):
            return payload, False, False
        return payload, True, True
    return payload if inner else raw, False, False


def _minmax(values):
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def top_raw_turns(cache, conv_id: str, question: str, embedder, k: int = 3):
    """The ``k`` raw dialogue turns most relevant to ``question``.

    Half normalised BM25, half normalised dense cosine -- Stage C's own fusion
    arithmetic, on turns instead of claims. Min-max is applied per question, so
    the two channels are comparable before averaging; `PHASE8_DECISIONS.md` §3.3
    is the standing warning that this destroys cross-question scale, which is
    fine here because the ranking is *within* one question.

    Deterministic: ties break by ``turn_id``, so two runs on one machine select
    the same turns and the evidence block is reproducible.
    """
    import numpy as np

    turns, retriever, matrix = cache.turns_for(conv_id)
    if not turns:
        return []

    lex = np.zeros(len(turns), dtype=np.float64)
    if retriever is not None and question.strip():
        import bm25s

        from graft.retrieve.pins import BM25

        tokens = bm25s.tokenize(
            [question], stopwords=BM25["stopwords"], show_progress=False
        )
        idx, scores = retriever.retrieve(
            tokens, k=min(len(turns), len(turns)), show_progress=False
        )
        for position, score in zip(idx[0], scores[0]):
            lex[int(position)] = float(score)

    sim = np.zeros(len(turns), dtype=np.float64)
    if matrix is not None and getattr(matrix, "size", 0):
        q = np.asarray(embedder.embed([question]), dtype=np.float32)[0]
        sim = np.asarray(matrix, dtype=np.float32) @ q

    fused = 0.5 * _minmax(lex) + 0.5 * _minmax(sim)
    order = sorted(
        range(len(turns)), key=lambda i: (-float(fused[i]), turns[i].turn_id)
    )
    return [turns[i] for i in order[:k]]


def raw_evidence_block(turns, count_tokens, room: int) -> tuple[str, int]:
    """Render raw turns as uncitable context, inside ``room`` tokens.

    Returns ``(text, n_included)``.  Whole turns are dropped from the tail first
    -- a half-sentence is worse than one turn fewer, and the ranking already put
    the most relevant first.  Carries **no ``[c#]`` id**: these are context, not
    citable evidence, so ``claim_map`` and citation precision are untouched.
    """
    from graft.reader.serialize import format_date

    if not turns or room <= 0:
        return "", 0
    header = (
        "\n\nRaw dialogue excerpts (context; cite only the [c#] claims above):\n"
    )
    lines = [
        f"({format_date(str(t.ts))}) {t.speaker}: {t.text}".strip() for t in turns
    ]
    for keep in range(len(lines), 0, -1):
        block = header + "\n".join(lines[:keep])
        if count_tokens(block) <= room:
            return block, keep
    return "", 0


def stage_c(snapshot, embedder, question: str, conv_id: str, config, ledger, cache=None):
    """The five training-free channels, then assembly. Phase 7's order, imported.

    Channel construction is inside the ``stage_c`` ledger stage so retrieval cost
    lands in the same per-query snapshot as generation -- otherwise GRAFT reports
    only its generation latency while a retrieval-augmented competitor's per-query
    figure includes both.

    ``cache`` is a :class:`ChannelCache`. Passing ``None`` rebuilds per call, which
    is what the tests do deliberately -- a cache that is only ever exercised warm
    would not catch a cold-path regression.
    """
    obligations = default_obligations(conv_id)
    with ledger.stage("stage_c"):
        channels: dict[str, dict[str, float]] = {}
        if cache is None:
            bm25 = BM25Channel(snapshot, conv_id, config=config)
            dense = DenseChannel(snapshot, embedder, conv_id, config=config, ledger=ledger)
        else:
            bm25, dense = cache.for_conv(conv_id, ledger)
        channels["bm25"] = bm25.query(question)
        channels["dense"] = dense.query(question)
        seeds = match_entities(snapshot, obligations.entity_anchor, conv_id)
        channels["entity"] = entity_channel(snapshot, obligations, conv_id, seeds=seeds)
        expand_hits, _ = expand_channel(snapshot, seeds, conv_id=conv_id)
        channels["expand"] = expand_hits
        pool, atom_scores, report = assemble(
            snapshot, channels,
            constraint=obligations.time_constraint,
            config=config, conv_id=conv_id,
            # The caller owns the stage; `assemble` must not open a second one.
            ledger=ledger, stage=None,
        )
    return pool, atom_scores, report, obligations, channels


def build_example(qid, snapshot, pool, atom_scores, report, obligations, gold_atoms, embedder):
    """A ``ProofExample`` over a conversational pool.

    This is the drop-in `GRAFT_PHASE9_BUILD.md` promised: the learner side takes
    ``(pool, obligations, gold_proof)`` and knows nothing about the corpus, so no
    file in ``graft/setgen/`` moves to support conversation.
    """
    texts: dict[str, np.ndarray] = {}
    node_atoms = [a for a in pool if a.kind == "node"]
    if node_atoms:
        from graft.retrieve.pool import node_text

        raw = [node_text(snapshot, a.target) for a in node_atoms]
        vecs = np.asarray(embedder.embed(raw), dtype=np.float32)
        for atom, vec in zip(node_atoms, vecs):
            texts[atom.atom_id] = vec

    return ProofExample(
        example_id=qid,
        snapshot=snapshot,
        pool=pool,
        obligations=obligations,
        atom_scores=atom_scores,
        channel_scores=report.get("raw_channel_scores", {}),
        atom_feat=texts,
        # Tier A, and the stamp says so: the evidence-derived closed superset,
        # never an H-minimised proof.
        gold_atom_ids=frozenset(gold_atoms),
        gold_groups={},
        meta={"gold_tier": "A", "corpus": "locomo"},
    )


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
    parser.add_argument("--corpus", default="data/locomo/locomo10.json")
    parser.add_argument("--run-dir", default="artefacts/locomo", help="the ingest log's dir")
    parser.add_argument("--out", default="artefacts/locomo_eval.json")
    parser.add_argument("--rows", default="artefacts/locomo_eval_rows.jsonl")
    parser.add_argument("--questions", type=int, default=None, help="cap total questions")
    parser.add_argument("--conversations", type=int, default=None, help="cap conversations")
    parser.add_argument(
        "--device", default=None,
        help="override the reader's device. Defaults to None so the pin in "
        "reader/pins.py governs -- passing nothing changes nothing.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="stub reader, no GPU. Exercises every join; produces no result.",
    )
    parser.add_argument(
        "--ceilings", action="store_true",
        help="compute the five ceilings (Contribution 4). Ceiling 5 reads a gold "
        "proof, so this roughly doubles reader cost; questions with no gold atoms "
        "are skipped with a reason rather than scored against nothing.",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="serialization budget in tokens; defaults to the config's 512",
    )
    parser.add_argument("--fresh", action="store_true", help="ignore existing rows and restart")
    parser.add_argument(
        "--raw-turns", type=int, default=3,
        help="raw dialogue turns appended as uncitable context (0 disables). "
        "Every reference system on LoCoMo shows its reader raw conversation "
        "text; these are the turns the graph was built from, already stored as "
        "provenance.",
    )
    parser.add_argument(
        "--evidence-budget", type=int, default=1024,
        help="hard cap in reader tokens on the WHOLE evidence block (claims + "
        "raw turns). `--budget` caps the claims tier alone; this caps the total, "
        "and raw turns are dropped from the tail to fit. 1024 is a pre-declared "
        "BUDGET_LADDER rung.",
    )
    parser.add_argument(
        "--selection", default="training_free_relevance",
        choices=("training_free_relevance", "learned_portfolio"),
        help="how Stage D picks its set. `training_free_relevance` takes the top "
        "max_atoms by Stage-C's question-conditioned fused score, closure-completes "
        "and H-checks; `learned_portfolio` uses the GFlowNet sampler, which is "
        "UNTRAINED until Phase-9 step 6 runs (so it selects blind). Either way the "
        "honesty stamp records which, and both trip `is_wiring_test`.",
    )
    parser.add_argument(
        "--head", default=None,
        help="a trained utility head from scripts/train_head.py. Without it the "
        "head is randomly initialised, best-of-K ranks by noise, and the stamp "
        "says so -- PHASE10_DECISIONS.md §1.4 is why ranking needs it at all.",
    )
    args = parser.parse_args()

    config = load_config()
    if args.budget is not None:
        config = config.with_overrides(serialization_budget_tokens=int(args.budget))

    args.run_dir = pick_run_dir(args.run_dir, REPO)
    log_path = REPO / args.run_dir / "events.jsonl"
    if not log_path.is_file():
        raise SystemExit(
            f"no ingest log at {log_path}. Run `scripts/locomo_ingest.py ingest` first."
        )
    snapshot = ReplayGraphStore(EventLog.open(log_path, fsync=False)).at()
    require_nodes(snapshot)
    print(f"graph: {snapshot.counts()} (from {args.run_dir})")

    corpus = locomo.load_corpus(args.corpus)
    samples = list(corpus)
    if args.conversations is not None:
        samples = samples[: args.conversations]

    # Only conversations that actually reached the graph. Asking a question about
    # an un-ingested conversation would score 0 for a reason that has nothing to
    # do with the system, and silently drag the mean down.
    ingested = {t.conv_id for t in (getattr(snapshot, "turns", ()) or ())}
    if ingested:
        skipped = [str(s["sample_id"]) for s in samples if str(s["sample_id"]) not in ingested]
        samples = [s for s in samples if str(s["sample_id"]) in ingested]
        if skipped:
            print(f"skipping {len(skipped)} un-ingested conversation(s): {skipped}")

    questions: list[dict[str, Any]] = []
    by_conv: dict[str, Any] = {}
    for sample in samples:
        by_conv[str(sample["sample_id"])] = sample
        for q in locomo.questions_of(sample):
            q["gold_turns"] = locomo.evidence_turn_ids(sample, q["evidence"])
            questions.append(q)
    if args.questions is not None:
        questions = questions[: args.questions]

    out_path, rows_path, iteration = resolve_out_paths(args, REPO)
    if iteration > 0:
        print(f"results iteration {iteration}: {out_path.name} / {rows_path.name}")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if rows_path.is_file() and not args.fresh:
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["question_id"]] = row
        print(f"resuming: {len(done)} of {len(questions)} questions already scored")

    print(
        f"LoCoMo eval: {len(questions)} questions over {len(samples)} conversations, "
        f"budget {config.serialization_budget_tokens} tokens"
    )

    reader = None
    if args.smoke:
        def read_fn(evidence: str, question: str) -> str:
            return "London [c1]"
        count_tokens = None
        reader_report: dict[str, Any] = {"stub": True}
    else:
        from graft.reader.read import Reader

        # `--device` is honoured rather than merely parsed: it was dead in the
        # first version, so `--device cpu` silently ran on cuda. `None` keeps the
        # pin's own default, so passing nothing changes nothing.
        reader = Reader(device=args.device)
        reader.__enter__()
        read_fn = reader.generate
        count_tokens = reader.count_tokens
        reader_report = reader.report()

    embedder = Embedder()

    # The head is global (one set of weights); the featurizer stays per question,
    # which is what `distill.pool_sets` requires -- a HeadScorer built on one
    # example and applied to another scores every candidate identically.
    head_state = None
    head_rho = None
    if args.head:
        blob = torch.load(REPO / args.head, map_location="cpu", weights_only=False)
        if int(blob.get("atom_width", ATOM_WIDTH)) != int(ATOM_WIDTH):
            raise SystemExit(
                f"head was trained at atom_width {blob.get('atom_width')} but this "
                f"build is {ATOM_WIDTH}; its features do not mean the same thing"
            )
        head_state = blob["state_dict"]
        head_rho = blob.get("dev_spearman")
        print(f"utility head: {args.head} (dev rho vs exact U = {head_rho})")

    stamp = ReadPathStamp(
        # **False, and not derived from the head** (corrected 19 Aug 2026).
        # `policy_trained` means the Stage-D *sampler*; this read it from
        # `head_state`, so a trained utility head stamped the untrained policy
        # as trained -- contradicting this same stamp's own note two lines down.
        # Phase-9 step 6 has not run; the head is reported separately.
        policy_trained=False,
        head_trained=bool(head_state),
        gate_source="none_recorded_for_post_hoc",
        scorer_source="distilled_head",
        token_counter="approx_tokens" if count_tokens is None else "reader_tokenizer",
        ordering="u_shaped_inference_computable",
        selection=args.selection,
        notes=(
            "Stage-D policy untrained: Phase-9 step 6 has not run",
            "obligations deterministic (empty); LLM parsing deferred by name -- "
            "fix F7 forbids the extractor and reader being co-resident, so it is "
            "a separate stage-sequential pass, not a flag on this one",
            "gate NOT applied; features recorded per question. Post-hoc "
            "thresholding additionally needs the MuSiQue gate RETRAINED "
            "(seeded, minutes of CPU) or a persisted checkpoint -- "
            "scripts/phase8_gate.py saves none (PHASE11_DECISIONS.md 1.8) -- "
            f"then reweighted to prevalence {gpins.EVAL_PREVALENCES['locomo']}",
            "gold proof atoms are Tier A (LoCoMo evidence-derived closed superset), "
            "not H-minimised",
            f"utility head trained, dev rho vs exact U = {head_rho}"
            if head_state else
            "utility head UNTRAINED: best-of-K ranks by noise, no ranking claim holds",
        ) + (("reader is a stub",) if args.smoke else ()),
    )

    # One index build per conversation instead of per question -- see ChannelCache.
    channel_cache = ChannelCache(snapshot, embedder, config)
    # Uncapped per-conversation pools for the ceiling serializer, built at most
    # once each: closure is O(n^2) in the chosen nodes and this would otherwise
    # be rebuilt on all 1,986 questions.
    ceiling_pools: dict[str, Any] = {}

    started = time.perf_counter()
    results = []
    fh = rows_path.open("a", encoding="utf-8")
    try:
        for index, q in enumerate(questions):
            qid = q["question_id"]
            if qid in done:
                results.append(done[qid])
                continue
            conv_id = q["conv_id"]
            sample = by_conv[conv_id]

            # **A fresh ledger per question** -- `phase10_read.py`'s convention,
            # adopted here 19 Aug 2026.  `Ledger.snapshot()["totals"]` and
            # `["stages"]` are both cumulative over a ledger's whole life, so one
            # ledger spanning 1,986 questions made the n-th question's record
            # carry the sum of the first n.  `cost_report` now reads the
            # per-scope `query` counter and is immune either way, but the
            # per-stage table in each row's `ledger_snapshot` has no such scoped
            # view -- so the ledger itself has to be per query, or every stage
            # figure in the artefact is a running total wearing a per-query
            # label.  The scope stays: it is what `would_exceed` enforces
            # `checker_budget` against, per query (fix F5).
            ledger = Ledger.from_config(config)
            with ledger.query_scope(qid):
                pool, atom_scores, report, obligations, _ = stage_c(
                    snapshot, embedder, q["question"], conv_id, config, ledger,
                    cache=channel_cache,
                )
                sat = saturation(snapshot, conv_id, config.pool_cap)
                # Recorded, not applied -- see the module docstring's decision 2.
                vec, names, flags = gfeatures.build_features(
                    obligations=obligations, pool=pool, atom_scores=atom_scores,
                    assembly_report=report, snapshot=snapshot, saturation=sat,
                )
                gold_atoms, _ = tier_a_gold(snapshot, q["gold_turns"], conv_id)
                example = build_example(
                    qid, snapshot, pool, atom_scores, report, obligations,
                    gold_atoms, embedder,
                )
                env = RealEnvironment(example, config, range_samples=0)
                featurizer = RealFeaturizer(
                    example, Policy(*RealFeaturizer.dims(), hidden=16), config, delta_d=False
                )
                head = build_head(ATOM_WIDTH, seed=config.seeds[0])
                if head_state is not None:
                    head.load_state_dict(head_state)
                    head.eval()
                scorer = HeadScorer(head, featurizer)
                if reader is not None:
                    reader.ledger = ledger
                # -- raw dialogue tier (run 3) ------------------------------
                # Budgeted against the reader's own tokenizer, and against what
                # the claims tier actually spent -- not against its cap, which it
                # rarely reaches. `count_tokens is None` under `--smoke`, where a
                # word count is the honest stand-in rather than a silent skip.
                suffix, n_raw = "", 0
                if args.raw_turns > 0:
                    counter = count_tokens or (lambda t: len(t.split()))
                    turns = top_raw_turns(
                        channel_cache, conv_id, q["question"], embedder,
                        k=int(args.raw_turns),
                    )
                    # Against the claims tier's **cap**, not its actual spend:
                    # `answer()` serialises internally, so the runner cannot know
                    # the packed size without serialising twice. Reserving the cap
                    # guarantees the total stays inside `--evidence-budget` and
                    # costs only unused room on questions whose claims pack short.
                    room = int(args.evidence_budget) - int(
                        config.serialization_budget_tokens
                    )
                    suffix, n_raw = raw_evidence_block(turns, counter, max(0, room))

                result = answer(
                    q["question"], env=env, featurizer=featurizer, scorer=scorer,
                    evidence_suffix=suffix,
                    read_fn=read_fn, gate_decision=None, obligations=obligations,
                    atom_scores=atom_scores,
                    rng=np.random.default_rng(config.seeds[0] + index),
                    ledger=ledger, config=config, stamp=stamp,
                    count_tokens=count_tokens, query_id=qid,
                    contested_check=False,
                )

                # -- the five ceilings, opt-in (Contribution 4) --------------
                # **Skipped rather than raised on empty gold, unlike
                # `phase10_read.py`.** There it is a wiring error; here it is
                # expected: adversarial questions have no evidence by definition,
                # and 9 of 2,815 evidence markers name image-only turns the loader
                # skips. Raising would abort a legitimate run at the first
                # adversarial question. `all_ceilings` already reports
                # `available: False` with a reason rather than omitting a row, so
                # a four-row table never reads as five.
                ceilings = None
                if args.ceilings:
                    if not gold_atoms:
                        ceilings = {
                            "skipped": True,
                            "reason": (
                                "no gold atoms: adversarial question, or every "
                                "evidence marker resolved to a skipped turn"
                            ),
                        }
                    else:
                        # **The ceilings serialise GOLD, so their serializer must
                        # resolve gold -- which the capped pool cannot.**
                        # `tier_a_gold` builds from the *snapshot* (every eligible
                        # node in the gold turns, closed); the retrieval pool is
                        # capped at `pool_cap = 64`. Gold outside the pool is not
                        # an error, it is exactly what ceiling 3 measures, and
                        # ceiling 3 gets the real capped pool via `retrieved=`
                        # below. Handing ceilings 4 and 5 the capped pool made a
                        # *retrieval* shortfall surface inside a *packing*
                        # measurement -- `PHASE10_DECISIONS.md` §5 A3's exact
                        # failure, which is the conflation the five-ceiling
                        # protocol exists to prevent -- and it crashed with a
                        # `KeyError` rather than mis-scoring, which is the one
                        # piece of luck in it.
                        #
                        # The conversation-wide uncapped pool is a superset of
                        # any gold set drawn from that conversation (gold comes
                        # from `uncapped_pool` over a subset of the same nodes,
                        # and atom ids are content-derived, so they agree). Only
                        # the atoms passed in are ever serialised, so the extra
                        # atoms cost nothing but resolvability.
                        if conv_id not in ceiling_pools:
                            nodes = eligible_nodes(snapshot, conv_id)
                            ceiling_pools[conv_id], _, _ = uncapped_pool(
                                snapshot, {n: 1.0 for n in nodes},
                                config=config, conv_id=conv_id,
                            )
                        serializer = ProofSerializer(
                            snapshot, ceiling_pools[conv_id], config=config,
                            count_tokens=count_tokens,
                            counter_name=(
                                "reader_tokenizer" if count_tokens is not None else None
                            ),
                        )
                        ceilings = C.all_ceilings(
                            snapshot=snapshot, conv_id=conv_id,
                            retrieved=sorted(pool.ids()), gold=sorted(gold_atoms),
                            serializer=serializer, obligations=obligations,
                            scores=atom_scores, question=q["question"],
                            read_fn=read_fn, gold_answer=q["gold"],
                            aliases=(), config=config,
                            # Tier A, because that is what the gold is: LoCoMo's
                            # evidence-derived closed superset, never H-minimised.
                            tier="tier_a",
                        )

            row = {
                "question_id": qid,
                "conv_id": conv_id,
                "category": q["category"],
                "adversarial": q["adversarial"],
                "gold": q["gold"],
                # Runner-level hygiene, applied before the row is written so the
                # scored string and the recorded outcome agree. `clean_answer`
                # only reclassifies; it never rewrites a scoreable answer.
                **_cleaned(result),
                "citations": len(result.record.citations),
                "pool_size": len(pool.ids()),
                "gold_atoms": len(gold_atoms),
                "raw_turns_included": n_raw,
                # **The whole dict, not a derived flag.**  An earlier version
                # wrote `bool(sat.get("saturated"))`, and `saturation()` has no
                # such key -- it returns `exercised` (the cap binds and
                # retrieval selects) plus `candidates_in_scope`,
                # `closed_atoms_in_scope`, `pool_cap` and `reading`.  The
                # `.get` therefore read `None` on every question and the field
                # was constantly False, which is the diagnostic
                # `PHASE7_DECISIONS.md` §3.1 made mandatory reporting the one
                # answer it can never be allowed to give by accident.  Storing
                # the dict also keeps `reading`, the sentence the runner is
                # supposed to print beside any recall number.
                "saturation": sat,
                # The whole point of decision 2: a threshold can be applied to
                # these later without touching the GPU again.
                "gate_features": [float(x) for x in vec],
                "gate_feature_names": list(names),
                "gate_blocks_present": dict(flags),
                "ledger_snapshot": dict(result.record.ledger_snapshot),
                "ceilings": ceilings,
            }
            results.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if index % 25 == 0:
                rate = (index + 1) / max(time.perf_counter() - started, 1e-9)
                print(
                    f"  [{index + 1}/{len(questions)}] {qid} {result.record.outcome} "
                    f"({rate * 3600:.0f} q/h)",
                    flush=True,
                )
    finally:
        fh.close()
        if reader is not None:
            reader.__exit__(None, None, None)

    elapsed = time.perf_counter() - started

    # `cost_report` wants ReadResult-likes; the rows carry the snapshots, so a
    # tiny shim keeps one implementation of the cost arithmetic.
    class _R:
        def __init__(self, row):
            self.record = type("rec", (), {
                "ledger_snapshot": row["ledger_snapshot"],
                "outcome": row["outcome"],
                "abstain_cause": row["abstain_cause"],
            })()

    cost = cost_report([_R(r) for r in results])

    # Means over questions where the ceiling ran, per ceiling. A question whose
    # ceiling was skipped is excluded from that mean and counted, rather than
    # imputed -- the Phase-10 decision-7 rule applied to ceilings: imputing 0
    # makes a skipped question look like a failed one.
    ceiling_block = None
    if args.ceilings:
        names = ("1_extraction", "2_graph", "3_candidate", "4_packing", "5_reader")
        ran = [r for r in results if isinstance(r.get("ceilings"), dict)
               and not r["ceilings"].get("skipped")]
        skipped = sum(
            1 for r in results
            if isinstance(r.get("ceilings"), dict) and r["ceilings"].get("skipped")
        )
        means = {}
        for name in names:
            vals = [
                float(r["ceilings"][name]["ceiling"])
                for r in ran
                if r["ceilings"].get(name, {}).get("available")
                and isinstance(r["ceilings"][name].get("ceiling"), (int, float, bool))
            ]
            means[f"ceiling_{name}"] = (sum(vals) / len(vals)) if vals else None
        ceiling_block = {
            "means": means,
            "questions_with_ceilings": len(ran),
            "questions_skipped": skipped,
            "tier": "tier_a",
            "reading": (
                "Tier A gold is LoCoMo's evidence-derived closed superset, not an "
                "H-minimised proof; skipped questions are counted, never imputed"
            ),
        }

    report_body = build_report(
        results,
        cost=cost,
        ceilings=ceiling_block,
        backbone="stub" if args.smoke else rpins.READER["model_id"],
        embedder=getattr(embedder, "name", "bge-small-en-v1.5"),
        budget_tokens=config.serialization_budget_tokens,
        ladder=rpins.BUDGET_LADDER,
        honesty_stamp=stamp.to_dict(),
        ingestion_cost={
            "measured_turns_per_hour": 135.67,
            "output_tokens_per_turn": 220,
            "axis": "offline, per turn; never folded into per-query inference cost",
        },
    )
    report_body["run"] = {
        "utility_head": {
            "checkpoint": args.head,
            "trained": bool(head_state),
            "dev_spearman_vs_exact_u": head_rho,
            "reading": (
                "DISTILL['report_rho']: a ranking claim without this rho cannot be "
                "read, because 'the sampler won' and 'the scorer was nearly perfect' "
                "are different findings"
            ),
        },
        "conversations": len(samples),
        "questions": len(results),
        "wall_clock_s": round(elapsed, 1),
        "questions_per_hour": round(len(results) / max(elapsed, 1e-9) * 3600, 1),
        "smoke": bool(args.smoke),
        "reader": reader_report,
        "corpus_sha256": locomo.corpus_sha256(args.corpus),
        "channel_index_builds": channel_cache.builds,
        "channel_index_builds_avoided": max(0, len(results) - channel_cache.builds),
    }
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_body, indent=2), encoding="utf-8")

    sc = report_body["scores"]["overall"]
    print()
    print(f"overall F1 (over all, abstention=0): {sc['f1_over_all']}")
    print(f"overall BLEU-1:                      {sc['bleu1_over_all']}")
    print(f"coverage:                            {sc['coverage']}")
    print(f"adversarial abstention accuracy:     {report_body['adversarial']['abstention_accuracy']}")
    print(f"tokens/query:                        {cost['llm_tokens_total_per_query']}")
    print(f"written: {out} ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
