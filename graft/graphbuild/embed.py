"""P6.2 — the pinned text embedder, with a disk cache (G7, decision 2).

**One embedder for the whole project, pinned by id *and* revision.**  Stage C
will reuse these exact vectors, and two embedders would make channel-fusion
scores incomparable — so the pin lives in ``graphbuild.pins`` and every consumer
comes through here.  Loaded via ``transformers`` directly: a cross-encoder or a
bi-encoder checkpoint is a plain HF model, and a second library to call the same
weights is a dependency bought for a convenience wrapper (the Phase-5 precedent).

**CLS pooling and L2 normalization, per the model card.**  Both matter
downstream: ``candidates.py`` ranks by a plain dot product, which *is* cosine
similarity only because the vectors are unit-norm.  If the pooling ever changes,
that dot product silently stops being a similarity.

**F7 discipline.**  Never resident with the extractor or the NLI model.  Stage B
computes embeddings for stored objects in one batch pass, caches them on disk
keyed by content id, and frees the model — so encoder *training* never re-runs
the embedder, which is what makes twenty epochs affordable on 8 GB.

**The cache is keyed by content, not by position.**  Two runs over the same log
hit the same keys, and a re-run after new turns arrive recomputes only the new
ones.  It is a cache and not a record: deleting it changes wall-clock and
nothing else, because the same pinned model on the same text is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from graft.canonical import digest_of
from graft.graphbuild.pins import EMBEDDER

__all__ = ["Embedder", "StubEmbedder", "cache_key"]


def cache_key(text: str, config: Mapping[str, Any] | None = None) -> str:
    """Content-derived key: the text *and* the model configuration.

    The config is in the key deliberately.  A cache keyed on text alone would
    serve bge-small vectors after someone swapped the pin, and the swap would be
    invisible — the exact failure the ingestion fingerprint exists to prevent one
    stage earlier.
    """
    spec = dict(config or EMBEDDER)
    return digest_of(
        {
            "text": text,
            "model": spec["model_id"],
            "revision": spec["revision"],
            "pooling": spec["pooling"],
            "normalize": spec["normalize"],
        },
        24,
    )


class Embedder:
    """The pinned bge-small encoder behind a two-method interface.

    ``embed(texts) -> (n, dim) float32`` is all any caller needs; nothing else in
    the project knows the model exists.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.config = dict(config or EMBEDDER)
        self.name = f"{self.config['model_id']}@{self.config['revision'][:8]}"
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._tok = None
        self._model = None
        self._memory: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    # -- cache -------------------------------------------------------------

    def _cache_path(self) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / "embeddings.npz"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if path is None or not path.is_file():
            return
        with np.load(path) as blob:
            self._memory = {k: blob[k] for k in blob.files}

    def flush(self) -> None:
        path = self._cache_path()
        if path is None or not self._memory:
            return
        np.savez_compressed(path, **self._memory)

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._model is not None:
            return
        revision = self.config["revision"]
        self._tok = AutoTokenizer.from_pretrained(self.config["model_id"], revision=revision)
        model = AutoModel.from_pretrained(self.config["model_id"], revision=revision)
        model.eval()
        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self._model = model.to(device)
        self.device = device

    def close(self) -> None:
        """Free the model — F7's stage-sequential rule."""
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        self._model = None
        self._tok = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "Embedder":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()
        self.close()

    # -- the interface -----------------------------------------------------

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts``, serving cache hits and batching the misses.

        Order-preserving, because ``candidates.py`` pairs rows back onto entities
        positionally — a reordering would attach one entity's vector to another's
        name, an error nothing downstream could see.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, int(self.config["dim"])), dtype=np.float32)

        keys = [cache_key(t, self.config) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._memory]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)

        if missing:
            fresh = self._encode([texts[i] for i in missing])
            for slot, i in enumerate(missing):
                self._memory[keys[i]] = fresh[slot]

        return np.stack([self._memory[k] for k in keys]).astype(np.float32)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self.load()
        assert self._tok is not None and self._model is not None
        out: list[np.ndarray] = []
        batch = int(self.config.get("batch_size", 32))
        for start in range(0, len(texts), batch):
            chunk = list(texts[start : start + batch])
            encoded = self._tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=int(self.config.get("max_length", 512)),
                return_tensors="pt",
            ).to(self._model.device)
            with torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state
            if self.config.get("pooling") == "cls":
                pooled = hidden[:, 0]
            else:  # pragma: no cover - the pin is CLS; kept so a swap is explicit
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            if self.config.get("normalize", True):
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def report(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "dim": int(self.config["dim"]),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cached_vectors": len(self._memory),
        }


class StubEmbedder:
    """Deterministic hash-based vectors.  No model, and no claim to be one.

    Exists so candidate generation, the pair proposer and the encoders can be
    tested on a bare interpreter.  Every test using it asserts a *plumbing*
    property — shapes line up, order is preserved, an exact match still wins —
    never a semantic one, because these vectors carry no meaning at all.
    """

    name = "stub"

    def __init__(self, dim: int = 384) -> None:
        self.dim = int(dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(digest_of(text, 8), 16) % (2**32)
            vec = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            out[i] = vec / (np.linalg.norm(vec) + 1e-12)
        return out

    def load(self) -> None:  # pragma: no cover - trivial
        return

    def close(self) -> None:  # pragma: no cover - trivial
        return

    def flush(self) -> None:  # pragma: no cover - trivial
        return
