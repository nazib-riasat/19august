"""P7.2 — the dense channel: exact cosine over the *shared* pinned embedder.

**One embedder for the whole project (decision 2), and it is not re-pinned here.**
``graphbuild.pins.EMBEDDER`` is the single pin and ``graphbuild.embed.Embedder``
the single loader; ``graft.retrieve.pins`` re-exports the pin rather than
restating it.  Two embedders would make channel-fusion scores incomparable and
would silently split Stage-B and Stage-C vector spaces — and because the cache is
keyed by *content plus model config*, sharing the loader means Stage C never
re-embeds what Stage B already embedded.

**Exact, not approximate.**  At ``pool_cap`` scale the corpus is hundreds of
short strings; an ANN index would be a dependency, a build step and a recall
approximation bought for a matrix multiply that takes microseconds.  §7 forbids
one without a measured reason, and there is none.

**True cosine, not a bare dot product.**  ``graphbuild.candidates`` ranks by ``@``
and is correct *because* the pin sets ``normalize: True`` — a comment there says
so.  This module divides by the norms anyway.  The pin is a dict someone can
edit, and a swapped pin with ``normalize: False`` would turn a similarity
ranking into a ranking by vector magnitude, with nothing failing and every recall
number moving.  One division per row is a cheap way not to depend on that.

**Negative scores are kept, unlike BM25's zeros.**  A cosine of −0.2 is a real
measurement of dissimilarity, whereas a BM25 zero is the structural artefact of
``retrieve`` returning ``k`` rows regardless of overlap.  The two channels differ
here for that reason, and the asymmetry is deliberate rather than an oversight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from graft.config.schema import Config
from graft.retrieve.pool import eligible_nodes, node_text

__all__ = ["DenseChannel"]


class DenseChannel:
    """Cosine similarity between the question and each eligible assertion.

    ``embedder`` is any object with ``embed(texts) -> (n, d)``: the pinned
    :class:`graft.graphbuild.embed.Embedder` in a real run, ``StubEmbedder`` in
    tests.  Injected rather than constructed here so that nothing in this module
    decides which model runs — the pin is Stage B's, and a channel that could
    pick its own encoder is the thing decision 2 forbids.
    """

    def __init__(
        self,
        snapshot: Any,
        embedder: Any,
        conv_id: str | None = None,
        *,
        config: Config | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.embedder = embedder
        self.conv_id = conv_id
        self.config = config or Config()
        self.node_ids: tuple[str, ...] = ()
        self._matrix: np.ndarray | None = None
        self._build()

    def _build(self) -> None:
        candidates = eligible_nodes(self.snapshot, self.conv_id)
        pairs = [(nid, node_text(self.snapshot, nid)) for nid in candidates]
        pairs = [(nid, text) for nid, text in pairs if text.strip()]
        self.node_ids = tuple(nid for nid, _ in pairs)
        if not pairs:
            self._matrix = None
            return
        matrix = np.asarray(self.embedder.embed([text for _, text in pairs]), dtype=np.float32)
        self._matrix = _unit(matrix)

    def query(self, text: str, top_k: int | None = None) -> dict[str, float]:
        """Raw question text → ``node_id -> cosine``, top-k by score.

        Ties break by ``node_id`` so the cut is reproducible: ``argsort`` is not
        stable across shapes, and two assertions with equal similarity are
        genuinely common on short conversational text.
        """
        if self._matrix is None or not text.strip():
            return {}
        k = int(self.config.pool_cap if top_k is None else top_k)
        k = max(1, min(k, len(self.node_ids)))
        query = _unit(np.asarray(self.embedder.embed([text]), dtype=np.float32))[0]
        scores = self._matrix @ query
        order = sorted(range(len(self.node_ids)), key=lambda i: (-float(scores[i]), self.node_ids[i]))
        return {self.node_ids[i]: float(scores[i]) for i in sorted(order[:k], key=lambda i: self.node_ids[i])}

    def report(self) -> dict[str, Any]:
        return {
            "indexed": len(self.node_ids),
            "embedder": getattr(self.embedder, "name", type(self.embedder).__name__),
            "dim": 0 if self._matrix is None else int(self._matrix.shape[1]),
        }


def _unit(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving a zero row as zero rather than as NaN.

    A zero vector has no direction, so its cosine against anything is undefined;
    returning zero scores it as maximally dissimilar, which is the honest reading
    and, unlike a NaN, does not poison the ``argsort`` that follows.
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
