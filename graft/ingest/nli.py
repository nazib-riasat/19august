"""P5.5 — the NLI verifier that writes ``entailed_by_span`` (decision 6, gap G6).

**[EVIDENCE]** the *pattern* is VeriCite's (SIGIR-AP 2025): an NLI model was the
cost-effective verifier against LLM judges — citation F1 80.05 against 73.01 for
an 8B LLM — and VeriCite's ablation is the motivating number for this whole
project's verification step (citation F1 77.73 → 68.91 with the NLI check
removed).  The architecture names the component "TRUE-style", but TRUE's own
reference model is a T5-11B, unusable in 8 GB, so the **concrete pin is
[ANALYSIS]**: a DeBERTa-v3-class cross-encoder, id and revision in
``graft.ingest.pins``.

Four semantics, frozen (G6), each of which is easy to violate silently:

* **premise = the grounded span(s), in turn order, and nothing else.**  Not the
  turn, not the conversation.  If the context were in the premise,
  ``entailed_by_span`` would stop meaning what its name says and the support gate
  would admit claims the span does not carry.
* **hypothesis = ``text_norm``.**
* **scoring and thresholding are separate calls.**  ``score`` returns the raw
  entailment probability and ``apply`` compares it to ``tau_nli``, so the pilot
  audit can sweep the distribution without re-running the model — and so the
  stored ``entailed_score`` is the model's number rather than a thresholded
  shadow of it.
* **the verifier never blocks storage** (architecture Phase 5).  An unentailed
  assertion is stored with its flag false; the *support gate*, not the verifier,
  decides eligibility.

``tau_nli = 0.8`` is a **frozen Gate-0 value**.  Phase 5 audits it on a
hand-labelled pilot sample and reports the agreement; a miscalibrated threshold
is a reported finding and a Gate-0 amendment, **never** an implementation-time
adjustment.  There is no code path in this module that can change it.

**F7 discipline.** The extractor and this model are never resident together.
Ingestion is stage-sequential across the slice — extract everything, free the
extractor, then verify — which the 8 GB card makes mandatory rather than
stylistic.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from graft.ingest.pins import NLI

__all__ = ["NliVerifier", "StubVerifier", "apply_threshold"]


def apply_threshold(score: float, tau_nli: float) -> bool:
    """``score >= tau_nli``.  One line, one home.

    Written out because "at ``tau_nli``" has two readings and the strict one
    would make a score exactly at the threshold ineligible.  The inclusive
    reading is the one the config's ``0 < tau_nli <= 1`` validation implies —
    ``tau_nli = 1.0`` must be satisfiable by a perfectly confident model.
    """
    return float(score) >= float(tau_nli)


class NliVerifier:
    """A pinned cross-encoder scoring ``P(entailment | premise, hypothesis)``.

    The entailment index is read from ``config.id2label`` rather than hard-coded.
    NLI checkpoints disagree about label order — this one is
    ``contradiction/entailment/neutral``, others are
    ``entailment/neutral/contradiction`` — and a hard-coded index would produce a
    plausible, wrong number on a model swap, with nothing crashing.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        device: str = "cuda",
        batch_size: int | None = None,
        ledger: Any = None,
    ) -> None:
        self.config = dict(config or NLI)
        self.name = f"{self.config['model_id']}@{self.config['revision'][:8]}"
        self.device = device
        self.batch_size = int(batch_size or self.config.get("batch_size", 16))
        #: Metered inside the component, exactly as the extractor meters its own
        #: ``llm_calls`` (exit criterion 14's argument): the architecture's
        #: ledger is global and mandatory, and until the 13 Aug 2026 audit the
        #: verify stage reported zero on every meter but wall-clock — extraction
        #: looked like the whole model cost of Stage A.
        self.ledger = ledger
        self._tok = None
        self._model = None
        self._entail_ix: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._model is not None:
            return
        revision = self.config["revision"]
        self._tok = AutoTokenizer.from_pretrained(self.config["model_id"], revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_id"], revision=revision
        )
        model.eval()
        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self._model = model.to(device)
        self.device = device

        labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
        want = str(self.config.get("entail_label", "entailment")).lower()
        if want not in labels:
            raise RuntimeError(
                f"{self.config['model_id']} has labels {sorted(labels)}, with no "
                f"{want!r} among them; the entailment index cannot be guessed and a "
                "wrong one would produce a plausible, wrong entailed_score"
            )
        self._entail_ix = labels[want]

    def close(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        self._model = None
        self._tok = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "NliVerifier":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- scoring -----------------------------------------------------------

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """``[(premise, hypothesis), ...]`` → entailment probabilities.

        Batched, and **order-preserving**: the caller pairs these back onto
        assertions positionally, so a reordering would attach one assertion's
        score to another's flag — an error that no test downstream could see.
        """
        import torch

        self.load()
        assert self._tok is not None and self._model is not None
        out: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = list(pairs[start : start + self.batch_size])
            if not chunk:
                continue
            encoded = self._tok(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                padding=True,
                truncation=True,
                max_length=int(self.config.get("max_length", 512)),
                return_tensors="pt",
            ).to(self._model.device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            if self.ledger is not None:
                # One batched forward = one ``model_forwards`` unit (the meter
                # counts forward passes, not items; the pair count is recoverable
                # as len(scores) from the artefact).
                self.ledger.count("model_forwards", 1)
            probs = torch.softmax(logits.float(), dim=-1)[:, self._entail_ix]
            out.extend(float(p) for p in probs.tolist())
        return out


class StubVerifier:
    """A fixed-score verifier for the write-path tests.

    Not a "mock NLI model" — it makes no entailment claim at all.  It exists so
    that idempotence, gating and replay can be tested on a bare interpreter, and
    every test using it asserts a *plumbing* property (the flag is stored, the
    verdict is written, the premise is the spans) rather than a quality one.
    """

    name = "stub"

    def __init__(self, scores: Mapping[str, float] | float = 1.0) -> None:
        self.scores = scores

    def load(self) -> None:  # pragma: no cover - trivial
        return

    def close(self) -> None:  # pragma: no cover - trivial
        return

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if isinstance(self.scores, Mapping):
            return [float(self.scores.get(hypothesis, 0.0)) for _, hypothesis in pairs]
        return [float(self.scores)] * len(pairs)


def score_drafts(verifier: Any, drafts: Iterable[Any]) -> list[float]:
    """Score ``DraftAssertion``s by their declared premise/hypothesis pair.

    The one place the G6 semantics are applied, so no caller can accidentally
    pass the turn text as the premise.
    """
    drafts = list(drafts)
    return verifier.score([(d.premise, d.text_norm) for d in drafts])
