"""Everything Phase 7 freezes, importable without an ML library.

Same shape and same reason as ``graft.ingest.pins`` and ``graft.graphbuild.pins``:
this module is §6's decision table made executable, and it carries the **Stage-C
fingerprint** (exit criterion 15) that ``scripts/verify_handoff.py`` prints — so
it must stay importable on a bare interpreter, and every model wrapper below it
imports lazily.

**Values that belong to the config tree are absent here on purpose.**
``pool_cap`` is the one anybody will look for: it is frozen at 64 (`CLAUDE.md`
§6) and it lives in ``graft.config.schema.Config``, so it reaches a run's
identity through ``config_hash`` and not through this file.  Giving a frozen
value two homes is the failure mode `CLAUDE.md` §5 catalogues, and the run's full
identity is now the quadruple ``(config_hash, ingestion_fingerprint,
stage_b_fingerprint, stage_c_fingerprint)``.

**The embedder is referenced, not re-pinned.**  Decision 2 is that Stage C reuses
the *shared* bge-small pin from ``graphbuild.pins``; importing it here rather
than restating it is what makes "two embedders would make channel-fusion scores
incomparable" a property of the code instead of a warning in a docstring.
"""

from __future__ import annotations

from typing import Any

from graft.canonical import digest_of
from graft.graphbuild.pins import EMBEDDER

__all__ = [
    "EMBEDDER",
    "CHANNELS",
    "SCORER_CHANNEL",
    "CHANNEL_WEIGHTS",
    "NORMALISATION",
    "TIE_BREAK",
    "BM25",
    "EXPANSION",
    "SCORER",
    "frozen_values",
    "stage_c_fingerprint",
]

# --------------------------------------------------------------------------
# decision 3 — the fusion arithmetic (G5)
# --------------------------------------------------------------------------

#: The five training-free channels, in the order every table reports them.  The
#: scorer is deliberately **not** here: it is a sixth score under the same
#: normalisation, optional by G6, and its absence is a legal configuration rather
#: than a missing channel.
CHANNELS = ("bm25", "dense", "entity", "expand")

#: The learned channel's name in the fusion table.  Separate from
#: :data:`CHANNELS` because that tuple is "the channels that run without
#: training" — the set G6 requires to stand alone — and folding the scorer into
#: it would make its absence look like a missing channel rather than the legal
#: configuration decision 9 declares it to be.
SCORER_CHANNEL = "scorer"

#: **All weights 1.0, and that is the decision** — not a placeholder awaiting a
#: sweep.  Recall is the purpose of this stage (plan §3.3 says so twice), the
#: per-channel recall table (G7) is the instrument that would justify moving one,
#: and moving one before that table exists would be tuning against nothing.
#:
#: The scorer's weight is listed **explicitly rather than defaulted**.  ``fuse``
#: falls back to 1.0 for an unlisted channel, so omitting it would work and would
#: leave the one *learned* channel's weight the only one not in the fingerprint —
#: exactly the asymmetry decision 3 exists to prevent.
CHANNEL_WEIGHTS: dict[str, float] = dict(
    {name: 1.0 for name in CHANNELS}, **{SCORER_CHANNEL: 1.0}
)

#: Per-channel min–max to [0, 1] **over that question's own results**.  BM25
#: scores and cosines share no scale, so a raw union would be a comparison
#: between a term-frequency statistic and an inner product; min–max is the
#: rank-shape transform that makes ``max`` across channels mean anything.
#:
#: The degenerate case is declared rather than discovered: when a channel returns
#: a single result, or several with identical scores, ``max == min`` and the
#: normalisation would divide by zero.  Those score **1.0** — the channel
#: retrieved them and has no basis to separate them — which is the choice that
#: preserves recall.
NORMALISATION = "minmax_per_channel_per_question"

#: An atom's fused score is the **max** of its normalised channel scores.  Union
#: semantics: one strong channel suffices.  Averaging would punish an atom that
#: only one channel could possibly know about — the entity channel's whole
#: purpose — which is precisely backwards for a recall stage.
FUSION = "max"

#: Ties break by atom id, ascending.  Any total order would do; what matters is
#: that it is *stated*, so two runs cap the same pool.
TIE_BREAK = "atom_id"

# --------------------------------------------------------------------------
# decision 1 — the BM25 channel (G2)
# --------------------------------------------------------------------------

#: ``bm25s`` defaults, pinned explicitly rather than inherited.  The library's
#: defaults are the Robertson/Spärck-Jones values every BM25 baseline uses, and
#: writing them down is what stops a library upgrade from silently moving a
#: control.  **[ANALYSIS]** as applied here: no tuning was done and none should
#: be without the G7 table to justify it.
#:
#: ``method`` is ``lucene``, ``bm25s``'s own default and the variant with the
#: widest deployed precedent.  English stopwords because the corpus is English
#: conversation; the corpus language is not a free parameter this project varies.
BM25: dict[str, Any] = {
    "method": "lucene",
    "k1": 1.5,
    "b": 0.75,
    "stopwords": "en",
    "drop_non_positive": True,
}

# --------------------------------------------------------------------------
# decision 6 — the expansion bounds (G10)
# --------------------------------------------------------------------------

#: ``max_hops = 2`` is the concession to iterative retrieval and nothing more.
#: **[EVIDENCE, provisional]** Beyond Static Retrieval (arXiv 2509.25530) found
#: two iterations the cost–benefit optimum across four graph-RAG systems, and
#: found iteration *hurting* simple questions through over-retrieval — flagged
#: provisional per `CLAUDE.md` §3 and not load-bearing for any claim.
#:
#: ``fan_out`` bounds **width**, which a hop bound does not.  In a memory graph
#: the user is a hub that links to everything, so 2-hop expansion is the whole
#: graph and the pool cap would then decide recall by truncation order — an
#: invisible determinant of the headline number.  Keeping the **most recent**
#: edges when it binds is **[ANALYSIS]**, declared because a memory graph's hubs
#: skew old; the number of times it binds is reported so that a cap doing real
#: work is visible rather than inferred.
EXPANSION: dict[str, Any] = {
    "max_hops": 2,
    "fan_out": 32,
    "live_edges_only": True,
    "tie_break": "recency_by_t_created",
}

# --------------------------------------------------------------------------
# decision 9 — the scorer (G6).  Declared before Gate 0 signed; built 15 Aug
# 2026, once it had.
# --------------------------------------------------------------------------

#: The GNN scorer's frozen configuration.  It is in the fingerprint **before it
#: is built** on purpose: decision 9 fixes the interface the other modules are
#: written against, and a fingerprint that only started describing Stage C once
#: the scorer existed would give the five-channel runs an identity that silently
#: changed underneath them when it arrived.
#:
#: ``max_params`` is the GFM-RAG scale point (**[EVIDENCE]** NeurIPS 2025: an 8M
#: query-conditioned GNN reaching Recall@5 87.1/58.2/95.6 on
#: HotpotQA/MuSiQue/2Wiki in one 0.107 s pass, against 3.162 s for iterative
#: IRCoT+HippoRAG).  ``training_signal`` is the **distant** one, per
#: ``GATE0_CONTRACT.md`` item 2, which beats the architecture's 2Wiki row: the
#: distant signal is conversation-native, so the Wikipedia→conversation transfer
#: declaration (`CLAUDE.md` §7) is not spent here.
#: ``layers`` is here because it was **not** (15 Aug 2026 audit): the layer
#: count was a hard-coded constructor default, so two differently-deep scorers
#: would have carried the same Stage-C fingerprint — configuration identity is
#: the one cross-machine property Stage C promises, and depth was outside it.
#: (The fingerprint moved when this key landed; recorded in
#: `PHASE7_DECISIONS.md` §7 — nothing decisive had been produced under the old
#: one.)  ``hidden`` is deliberately absent: it is `graphbuild.pins.TRAINING`'s
#: value, referenced not re-pinned, the same rule as the embedder above.
SCORER: dict[str, Any] = {
    "max_params": 8_000_000,
    "passes": 1,
    "layers": 2,
    "query_conditioned": True,
    "training_signal": "distant_answer_session_ids",
    "optional_in_fusion": True,
    "built": True,
}


def frozen_values() -> dict[str, Any]:
    return {
        "embedder": EMBEDDER,
        "channels": list(CHANNELS),
        "scorer_channel": SCORER_CHANNEL,
        "channel_weights": dict(sorted(CHANNEL_WEIGHTS.items())),
        "normalisation": NORMALISATION,
        "fusion": FUSION,
        "tie_break": TIE_BREAK,
        "bm25": BM25,
        "expansion": EXPANSION,
        "scorer": SCORER,
    }


def stage_c_fingerprint(length: int | None = None) -> str:
    """Configuration identity for Stage C, printed by ``verify_handoff.py``.

    Binds the config, not the output — the same G11 distinction Phases 5 and 6
    drew.  Two machines will not produce bit-identical scorer weights; they must
    produce them from an identical setup, and they must fuse channels with
    identical arithmetic, or two recall numbers are not comparable.
    """
    return digest_of(frozen_values(), length)
