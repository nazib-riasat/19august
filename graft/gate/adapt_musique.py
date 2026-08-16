"""P8.3 — MuSiQue-Full contrast pairs → gate features (G8, decision 1).

**Why this corpus is the primary training interface.** MuSiQue-Full ships each
question twice with a **byte-identical question string** — unanswerability comes
from *removing supporting paragraphs*, not from rewording — so a classifier
**cannot** score answerability from wording and is forced onto pool-side
features, which is exactly the declared feature set (G2). **[EVIDENCE]** MuSiQue
(TACL 2022); the byte-identity was verified over the real dev split on 15 Aug
2026 (2,417 pairs, zero mismatches, `GRAFT_PHASE8_BUILD.md` §6).

**The adaptation, and its losses, enumerated** — the Gate-0 item-1 discipline is
that an adaptation is reported with what it drops and is *never* presented as
native supervision:

===========================  ==============================================
Block                        On MuSiQue
===========================  ==============================================
``slot_coverage``            **absent** — a paragraph set has no obligation
                             parse: no entity anchor, no value type, no time
                             constraint. Flag cleared.
``channel_scores``           bm25 + dense **present**, over paragraphs, via
                             the same ``bm25s``/pinned-embedder stack.
                             entity/expand/scorer are 0.0: there is no graph
                             to walk and no learned scorer here.
``pool_shape``               present, counted over paragraphs.
``saturation``               present, paragraphs against ``pool_cap``.
``question_embedding``       present when the arm asks for it — and
                             *provably inert* on this corpus, since the twins'
                             question strings are identical.
===========================  ==============================================

**The presence flags are what stop this becoming a dataset classifier.** Without
them the gate can learn *"slot features are zero ⇒ MuSiQue ⇒ 50% unanswerable"*
and score well while having learned nothing about answerability. With them,
"absent" is an explicit input that is *equally* present on both members of every
pair — so it carries no label information at all, by construction.

**Everything goes through ``gate.features.build_features``**, unchanged. A
second featuriser here would be the thing exit criterion 3 forbids: training and
inference reading different objects.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from graft.gate.features import build_features
from graft.gate.labels import ANSWERABLE, UNANSWERABLE

__all__ = [
    "ADAPTATION_LOSSES",
    "UNANSWERABILITY_CONSTRUCTION",
    "paragraph_scores",
    "pair_features",
    "adapt_pairs",
]

#: Reported in the artefact, not just here (exit criterion 10).
ADAPTATION_LOSSES: dict[str, str] = {
    "slot_coverage": "absent — a Wikipedia paragraph set has no obligation parse",
    "entity_channel": "absent — no entity graph to match an anchor against",
    "expand_channel": "absent — no typed edges to walk",
    "temporal_filter": "absent — paragraphs carry no valid_during intervals",
    "scorer_channel": "absent — the Stage-C GNN scores GRAFT pools, not paragraphs",
    "pool_units": "atoms→paragraphs: 'pool_size' counts paragraphs here, atoms on conversation",
    "pool_shape_non_discriminative": (
        "MuSiQue SUBSTITUTES distractors for the removed supporting paragraphs "
        "rather than deleting them, so both twins carry ~20 paragraphs and the "
        "whole pool_shape block is near-identical across the label — measured "
        "397/400 identical paragraph counts on dev. On this corpus the gate can "
        "therefore only learn from channel_scores; pool_shape and saturation "
        "become discriminative on conversation, where a deletion twin really is "
        "smaller. Reported because a reader comparing feature importances across "
        "the two tracks would otherwise read this as the block being useless."
    ),
}

#: **How MuSiQue actually makes a question unanswerable** — measured on 400 dev
#: pairs, 15 Aug 2026, because it is *not* what "removing supporting paragraphs"
#: suggests and it changes how the negatives should be read.
#:
#: Answerable rows carry 2 supporting paragraphs, always. Their twins carry
#: **1** in 230 cases and **0** in 170: the usual construction breaks *one hop*
#: of the chain and leaves the other hop's evidence in place.
#:
#: That is a **subtler** negative than this project's own conversational recipe
#: (`gate/labels.py`, which removes an evidence session outright), and it cuts in
#: the helpful direction: a partially-evidenced question that cannot actually be
#: answered is much closer to a deployed unanswerable than a question whose
#: evidence was wholly removed. It partly offsets the adaptation loss
#: ``labels.py`` declares — and it is the reason a gate trained here should not be
#: assumed to have learned "little evidence ⇒ unanswerable".
UNANSWERABILITY_CONSTRUCTION: dict[str, Any] = {
    "answerable_supporting": {"2": 400},
    "unanswerable_supporting": {"1": 230, "0": 170},
    "identical_paragraph_count": "397/400",
    "reading": "one hop of the chain is broken; partial evidence usually remains",
}


def paragraph_scores(
    question: str,
    paragraphs: Sequence[Mapping[str, Any]],
    embedder: Any,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """BM25 + dense over one question's paragraphs, in Stage-C's output shape.

    Returns ``(fused, per_channel, per_channel_raw)`` keyed by a stable
    per-paragraph id, mirroring ``assemble()``'s ``channel_scores`` and
    ``raw_channel_scores`` exactly — the two paths must hand the gate the same
    two views or exit criterion 3 fails. So the
    feature code sees exactly the mapping shapes ``assemble()`` produces —
    ``atom_id -> score`` and ``atom_id -> {channel: score}``.

    **The same normalisation Stage C declares** (G5, `retrieve.pins`): per-channel
    min–max to [0, 1] over this question's own results, then ``max`` across
    channels. Reimplemented over paragraphs rather than reused because
    ``fuse.fuse`` operates on graph node ids, but the *arithmetic* is the
    declared one — if it diverged, the channel-score features would mean
    different things in training and inference.
    """
    import bm25s

    from graft.retrieve.pins import BM25

    texts = [str(p.get("paragraph_text", "")) for p in paragraphs]
    ids = [f"p{i}" for i in range(len(paragraphs))]
    if not texts:
        return {}, {}, {}

    lexical: dict[str, float] = {}
    tokens = bm25s.tokenize(texts, stopwords=BM25["stopwords"], show_progress=False)
    retriever = bm25s.BM25(method=BM25["method"], k1=BM25["k1"], b=BM25["b"])
    try:
        retriever.index(tokens, show_progress=False)
        query = bm25s.tokenize(
            [question], stopwords=BM25["stopwords"], return_ids=False, show_progress=False
        )
        docs, scores = retriever.retrieve(query, k=len(texts), show_progress=False)
        for slot, score in zip(docs[0].tolist(), scores[0].tolist()):
            value = float(score)
            if not (BM25["drop_non_positive"] and value <= 0.0):
                lexical[ids[int(slot)]] = value
    except ValueError:
        # An all-stopword paragraph set leaves bm25s with an empty vocabulary
        # (the same 0.3.10 behaviour `retrieve.bm25` documents). The channel goes
        # quiet; the dense one still carries the question.
        lexical = {}

    matrix = np.asarray(embedder.embed(texts), dtype=np.float64)
    q = np.asarray(embedder.embed([question]), dtype=np.float64)[0]
    matrix = matrix / np.where(
        (n := np.linalg.norm(matrix, axis=-1, keepdims=True)) == 0.0, 1.0, n
    )
    qn = np.linalg.norm(q)
    q = q / (qn if qn else 1.0)
    dense = {ids[i]: float(v) for i, v in enumerate(matrix @ q)}

    def minmax(scores: Mapping[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    raw = {"bm25": lexical, "dense": dense}
    normalised = {"bm25": minmax(lexical), "dense": minmax(dense)}
    per_channel: dict[str, dict[str, float]] = {}
    per_channel_raw: dict[str, dict[str, float]] = {}
    fused: dict[str, float] = {}
    for name in sorted(normalised):
        for pid, value in normalised[name].items():
            per_channel.setdefault(pid, {})[name] = value
            # Fusion is over the **normalised** scores — decision 3's declared
            # arithmetic, unchanged. Only the *features* read the raw view.
            fused[pid] = max(fused.get(pid, 0.0), value)
        for pid, value in raw[name].items():
            per_channel_raw.setdefault(pid, {})[name] = value
    return (
        dict(sorted(fused.items())),
        {k: dict(sorted(v.items())) for k, v in sorted(per_channel.items())},
        {k: dict(sorted(v.items())) for k, v in sorted(per_channel_raw.items())},
    )


def pair_features(
    row: Mapping[str, Any],
    embedder: Any,
    *,
    pool_cap: int,
    question_vector: Sequence[float] | None = None,
    embed_dim: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, bool]]:
    """One MuSiQue row → the gate's feature vector, through ``build_features``.

    ``obligations=None`` and ``snapshot=None`` are what clear the
    ``slot_coverage`` flag — the adapter does not fabricate an obligation parse
    to fill the block, which would be exactly the silent degradation G8 forbids.
    """
    paragraphs = list(row.get("paragraphs", ()))
    fused, per_channel, per_channel_raw = paragraph_scores(
        str(row["question"]), paragraphs, embedder
    )
    n = len(paragraphs)
    report = {
        "channel_scores": per_channel,
        "raw_channel_scores": per_channel_raw,
        "pool": {
            "pool_size": n,
            "node_atoms": n,
            "edge_atoms": 0,
            "support_atoms": 0,
            "hits_offered": n,
            "hits_admitted": len(fused),
            "cap_skipped": max(0, n - pool_cap),
            "hits_refused_ineligible": 0,
        },
        "union_pre_cap": len(fused),
        "after_temporal": len(fused),
        "temporal": {"applied": False, "dropped": 0},
    }
    saturation = {
        "candidates_in_scope": n,
        "closed_atoms_in_scope": n,
        "pool_cap": int(pool_cap),
        "exercised": n > int(pool_cap),
    }
    return build_features(
        obligations=None,
        pool=(),
        atom_scores=fused,
        assembly_report=report,
        snapshot=None,
        saturation=saturation,
        question_vector=question_vector,
        embed_dim=embed_dim,
    )


def adapt_pairs(
    pairs: Iterable[Mapping[str, Any]],
    embedder: Any,
    *,
    pool_cap: int,
    with_question: bool = True,
    embed_dim: int | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Contrast pairs → ``[{question_id, vector, label, ...}, ...]`` plus a report.

    Both members of every pair are emitted, so the training balance is **1:1 by
    construction** (G6) — stated, never corrected by resampling, and never quoted
    as a natural frequency.

    ``with_question`` controls only whether the embedding block is filled; the
    vector width is identical either way, so the two arms differ by a column mask
    (the ``pool_only`` arm masks it out at train time as well, but filling it
    here keeps one cached featurisation usable by both).
    """
    rows: list[dict[str, Any]] = []
    seen = 0
    for pair in pairs:
        if limit is not None and seen >= limit:
            break
        seen += 1
        question = str(pair["question"])
        qvec = None
        if with_question:
            qvec = np.asarray(embedder.embed([question]), dtype=np.float64)[0]
        for side, label in (("answerable", ANSWERABLE), ("unanswerable", UNANSWERABLE)):
            vector, names, flags = pair_features(
                pair[side], embedder, pool_cap=pool_cap, question_vector=qvec, embed_dim=embed_dim
            )
            rows.append(
                {
                    "question_id": str(pair["id"]),
                    "side": side,
                    "label": label,
                    "vector": vector,
                    "names": names,
                    "flags": flags,
                }
            )
    labels = [r["label"] for r in rows]
    return rows, {
        "pairs": seen,
        "rows": len(rows),
        "answerable": sum(1 for x in labels if x == ANSWERABLE),
        "unanswerable": sum(1 for x in labels if x == UNANSWERABLE),
        "balance": "1:1 by construction (G6) — constructed, never a natural frequency",
        "adaptation_losses": dict(ADAPTATION_LOSSES),
        "unanswerability_construction": dict(UNANSWERABILITY_CONSTRUCTION),
        "question_embedding_note": (
            "inert on this corpus by construction: the two members of a pair carry "
            "byte-identical question strings, so the embedding block is identical "
            "across the label and can carry no label information here. The arm "
            "exists for the natural evaluation sets (G2)."
        ),
    }
