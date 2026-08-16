"""Channel scores over plain texts — Stage C's declared arithmetic, corpus-side.

**Why this exists rather than an import from Stage C.**  ``retrieve.fuse`` and
``retrieve.bm25`` operate on *graph node ids* over a built snapshot, and a
Stage-D adapter has to score documents **before** the snapshot exists in order to
rank them into the pool.  ``gate.adapt_musique.paragraph_scores`` is the Phase-8
sibling of this function and is keyed to MuSiQue's paragraph dicts; importing it
here would couple two phases' adapters through one corpus's schema.

**The arithmetic is the declared one and must stay so** (Stage C's G5, and
`PHASE8_DECISIONS.md` §2.5): per-channel min–max to [0, 1] over *this question's
own* results, then ``max`` across channels.  If it diverged, the channel-score
features would mean different things in training and at inference.
``test_setgen_corpora.py`` asserts this agrees with the Phase-8 implementation on
a shared input, so "the same arithmetic" is checked rather than asserted.

**Both views ship** — raw and normalised.  `PHASE8_DECISIONS.md` §3.3 is the
record of why: min–max is scale-invariant per question, so it made 10 of 13
channel features constant and put a real run at chance while AURC still read
healthy.  Fusion needs the shared scale; features need absolute strength.  The
fix is to expose both, never to change the fusion.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["score_texts", "minmax"]


def minmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Per-channel min–max over one question's own results.

    A flat channel maps to 1.0 everywhere rather than 0.0: the channel found
    nothing to distinguish, and mapping it to the floor would let it drag the
    ``max`` fusion down on every document it scored.  Phase 8's implementation
    makes the same choice, and the agreement test covers it.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def score_texts(
    question: str,
    ids: Sequence[str],
    texts: Sequence[str],
    embedder: Any,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """``(fused, per_channel_norm, per_channel_raw)`` keyed by document id.

    ``per_channel_*`` are keyed **channel → {doc_id: score}**, which is the shape
    ``proofs.build_example`` takes; Phase 8's sibling transposes it the other way
    for its own feature builder.  The transposition is the only difference.
    """
    import bm25s

    from graft.retrieve.pins import BM25

    if len(ids) != len(texts):
        raise ValueError(f"{len(ids)} ids against {len(texts)} texts")
    if not texts:
        return {}, {}, {}

    lexical: dict[str, float] = {}
    tokens = bm25s.tokenize(list(texts), stopwords=BM25["stopwords"], show_progress=False)
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
        # An all-stopword document set leaves bm25s with an empty vocabulary
        # (the 0.3.10 behaviour `retrieve.bm25` documents). The channel goes
        # quiet and the dense one carries the question.
        lexical = {}

    matrix = np.asarray(embedder.embed(list(texts)), dtype=np.float64)
    q = np.asarray(embedder.embed([question]), dtype=np.float64)[0]
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    matrix = matrix / np.where(norms == 0.0, 1.0, norms)
    qn = np.linalg.norm(q)
    q = q / (qn if qn else 1.0)
    dense = {ids[i]: float(v) for i, v in enumerate(matrix @ q)}

    raw = {"bm25": lexical, "dense": dense}
    normalised = {name: minmax(values) for name, values in raw.items()}

    fused: dict[str, float] = {}
    for name in sorted(normalised):
        for did, value in normalised[name].items():
            # Fusion is over the **normalised** scores — the declared arithmetic,
            # unchanged. Only the *features* read the raw view.
            fused[did] = max(fused.get(did, 0.0), value)
    return (
        dict(sorted(fused.items())),
        {k: dict(sorted(v.items())) for k, v in sorted(normalised.items())},
        {k: dict(sorted(v.items())) for k, v in sorted(raw.items())},
    )
