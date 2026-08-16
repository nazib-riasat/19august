"""P7.1 — the lexical channel, on ``bm25s`` (decision 1, gap G2).

**Why a library and not thirty lines of TF-IDF.**  BM25 is a *baseline channel*,
and `PHASE3_DECISIONS.md` §6 is this project's own catalogue of what happens to
baselines: three control defects, all of which ran in the direction that
flattered the proposed method.  A subtly wrong IDF or length normalisation would
not crash — it would quietly make the learned scorer look better in the one
comparison Stage C exists to report.  ``bm25s`` is numpy-only, in-process, with
no index server and no persistence daemon, so adopting it costs nothing the
boring-stack rule is protecting (`requirements-ml.txt` carries the full
justification).

**The index is held by its caller, not by a module-level cache.**  P7.1 says
"built once per graph snapshot and cached in memory", and the honest way to do
that is an object whose lifetime *is* the snapshot's: a global memo keyed by
``snapshot_id`` would serve a stale index the moment two snapshots of the same
log were live at once, and keying it by ``id()`` would be worse, since CPython
reuses addresses after collection.  The runner builds one and reuses it.

**Queries are tokenised to strings, not to ids.**  ``bm25s.tokenize`` returns
either, and both work — verified 15 Aug 2026, including the case that
distinguishes them, where a query's own vocabulary maps a term to an id that
denotes a *different* term in the corpus vocabulary (``bm25s`` remaps correctly
via the ``Tokenized.vocab`` dict).  Strings are used anyway because they cannot
depend on an id space at all, so the correctness of this channel does not rest on
a remapping detail of a third-party library.

**Non-positive scores are dropped** (``pins.BM25['drop_non_positive']``).
``retrieve`` returns ``k`` documents whether or not they share a term with the
query, so a zero score means "no lexical overlap", not "weakly relevant".
Keeping them would pad the channel with atoms it has no evidence for, and — since
G5's min–max maps a channel's minimum to 0.0 — would drag the normalisation's
floor onto documents BM25 never actually matched.
"""

from __future__ import annotations

from typing import Any

from graft.config.schema import Config
from graft.retrieve.pins import BM25
from graft.retrieve.pool import eligible_nodes, node_text

__all__ = ["BM25Channel"]


class BM25Channel:
    """A BM25 index over one snapshot's eligible assertion text.

    Not a dataclass: ``graft.retrieve`` defines none (criterion 12), and this
    holds a live index rather than a record anyway.
    """

    def __init__(self, snapshot: Any, conv_id: str | None = None, *, config: Config | None = None) -> None:
        self.snapshot = snapshot
        self.conv_id = conv_id
        self.config = config or Config()
        self.node_ids: tuple[str, ...] = ()
        self._retriever: Any = None
        self._build()

    def _build(self) -> None:
        import bm25s

        candidates = eligible_nodes(self.snapshot, self.conv_id)
        pairs = [(nid, node_text(self.snapshot, nid)) for nid in candidates]
        # An assertion-backed node with empty text cannot be lexically matched and
        # would occupy a corpus row that shifts every index behind it.  Dropped
        # here rather than scored 0 later, so the corpus and ``node_ids`` stay
        # positionally aligned -- the alignment every result row depends on.
        pairs = [(nid, text) for nid, text in pairs if text.strip()]
        self.node_ids = tuple(nid for nid, _ in pairs)
        if not pairs:
            self._retriever = None
            return
        tokens = bm25s.tokenize(
            [text for _, text in pairs],
            stopwords=BM25["stopwords"],
            show_progress=False,
        )
        retriever = bm25s.BM25(method=BM25["method"], k1=BM25["k1"], b=BM25["b"])
        try:
            retriever.index(tokens, show_progress=False)
        except ValueError:
            # A corpus whose every text tokenises to nothing (all stopwords or
            # single characters) leaves ``bm25s`` with an empty vocabulary, and
            # its indexer raises ``max()`` on an empty dict (probed on 0.3.10,
            # 15 Aug 2026 audit).  Such a corpus has no lexical content to match,
            # so the honest degradation is the same as an empty corpus: this
            # channel goes quiet and the other channels carry the question.
            self._retriever = None
            return
        self._retriever = retriever

    def query(self, text: str, top_k: int | None = None) -> dict[str, float]:
        """Raw question text → ``node_id -> BM25 score``.

        ``top_k`` defaults to ``pool_cap``.  A channel returning more than the
        pool can hold is not wasted — fusion takes the ``max`` across channels
        before capping, so a document ranked 60th here can still be first
        overall — but ``pool_cap`` is the scale at which that stops being true
        often enough to pay for, and it keeps every channel's *own* k on one
        declared number rather than five.
        """
        import bm25s

        if self._retriever is None or not text.strip():
            return {}
        k = int(self.config.pool_cap if top_k is None else top_k)
        k = max(1, min(k, len(self.node_ids)))
        query = bm25s.tokenize(
            [text], stopwords=BM25["stopwords"], return_ids=False, show_progress=False
        )
        documents, scores = self._retriever.retrieve(query, k=k, show_progress=False)
        out: dict[str, float] = {}
        for slot, score in zip(documents[0].tolist(), scores[0].tolist()):
            value = float(score)
            if BM25["drop_non_positive"] and value <= 0.0:
                continue
            out[self.node_ids[int(slot)]] = value
        return dict(sorted(out.items()))

    def report(self) -> dict[str, Any]:
        return {
            "indexed": len(self.node_ids),
            "method": BM25["method"],
            "k1": BM25["k1"],
            "b": BM25["b"],
            "stopwords": BM25["stopwords"],
        }
