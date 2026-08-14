"""P5.3 — one extraction interface, three model backends and a replay backend.

**The interface is one method** — ``extract(turn, context) -> RawExtraction`` —
so the pipeline, the bakeoff and the tests all drive the same object, and the
G2 winner is swapped in by changing a config dict rather than a call site.

**Metering lives inside the wrapper** (exit criterion 14).  ``llm_calls``,
``llm_tokens_in`` and ``llm_tokens_out`` are counted where the tokens are, not
by a caller who might forget on one of three paths — the same argument Phase 0
made for ``terminal_checks`` being counted inside the checker.

**Parse failure is a first-class outcome, not an exception.**  The spike's
measured 15.5% JSON parse failure rate produced turns that yielded *nothing*,
and nothing downstream could tell that apart from a turn with no extractable
content.  Here a failed parse returns a ``RawExtraction`` with ``parse_ok=False``
and empty lists, so the loss is counted where it happens.

**Greedy decoding, always** (G11).  Per-machine determinism is the only
reproducibility Stage A can promise, and sampling would spend it.

``ReplayExtractor`` is the reason the write path is testable without a GPU
(P5.10): it serves recorded extractions keyed by ``turn_id``, so idempotence,
grounding, gating and replay all run on the stock CI image and only the bakeoff
and the live pilot need the model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from graft.ingest import prompts
from graft.ingest.pins import (
    CONTEXT_CLIP_CHARS,
    DECODING,
    MAX_REPAIRS,
    require_extractor,
)
from graft.ingest.records import Quote, RawAssertion, RawExtraction
from graft.ledger import Ledger
from graft.schemas import ASSERTION_KINDS, Turn

__all__ = [
    "GrammarLogitsProcessor",
    "Extractor",
    "ExtractionContext",
    "ReplayExtractor",
    "LlmExtractor",
    "parse_extraction",
    "build_extractor",
]


class ExtractionContext:
    """What the model sees besides the current turn: a summary and a window.

    ``window`` is the previous ``min(m, 10)`` turns in order, oldest first, and
    ``offsets`` maps each ``turn_offset`` to its ``(turn_id, text)`` so the
    grounder can resolve a cross-turn quote against the turn it actually names
    (G9).  Building that map here rather than in the pipeline keeps one
    definition of "what offset -1 means".
    """

    __slots__ = ("summary", "window", "session_date")

    def __init__(
        self,
        summary: str = "",
        window: Sequence[Turn] = (),
        session_date: str = "",
    ) -> None:
        self.summary = summary or ""
        self.window = tuple(window)
        self.session_date = session_date or ""

    def offsets(self, current: Turn) -> dict[int, tuple[str, str]]:
        """``turn_offset -> (turn_id, FULL text)`` — never the clipped text.

        Grounding must resolve quotes against the raw turn (exit criterion 1),
        and a quote the model emits necessarily comes from the part of the turn
        it could see, so clipping the prompt does not clip the search space.
        """
        table = {0: (current.turn_id, current.text)}
        for back, turn in enumerate(reversed(self.window), start=1):
            table[-back] = (turn.turn_id, turn.text)
        return table

    @staticmethod
    def clip(text: str, limit: int = CONTEXT_CLIP_CHARS) -> str:
        """Head-clip one context turn to ``limit`` characters (decision 4's
        declared adaptation — see ``pins.CONTEXT_CLIP_CHARS`` for the measured
        reason).  The marker makes the clip visible to the model, so it does not
        quote across the cut believing the text continues."""
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " [clipped]"

    def rendered(self) -> str:
        """The context block of the prompt.  **Window turns are clipped; the
        current turn (rendered by the caller) never is** — it is the extraction
        target, and clipping it would hide the very text quotes must come from."""
        if not self.window:
            return "(none)"
        return "\n".join(
            f"[turn_offset {off - len(self.window)}] [{t.speaker}] {self.clip(t.text)}"
            for off, t in enumerate(self.window)
        )


class Extractor(Protocol):
    """``extract(turn, context) -> RawExtraction``, and a name for the manifest."""

    name: str

    def extract(self, turn: Turn, context: ExtractionContext) -> RawExtraction: ...


class GrammarLogitsProcessor:
    """Candidate B's decoding constraint, written here rather than imported.

    ``xgrammar.contrib.hf.LogitsProcessor`` exists and is the reference path, but
    on this machine it raises ``ImportError: Triton is not installed`` — xgrammar
    masks CUDA logits with a Triton kernel and **Triton has no Windows build**.
    That is a platform fact, not a bug, and it would have made candidate B
    unrunnable and therefore unmeasured.

    So the mask is applied through the library's own
    ``apply_token_bitmask_inplace(..., backend="torch_native")`` — xgrammar's
    documented Triton-free path.  The grammar, the matcher and the bitmask are
    all still the library's; what is written here is the twenty lines of
    per-step bookkeeping that ``contrib.hf`` would otherwise do.

    **The cost is recorded rather than hidden**: the torch-native mask is slower
    than the Triton kernel, so candidate B's throughput on this machine carries a
    platform penalty a Linux run would not have.  Stage 2 of the bakeoff rule
    ranks by throughput, so that penalty counts *against* B in the comparison —
    which is the conservative direction, and is stated in the artefact.

    **A fresh instance per generation, never reused.**  The matcher carries parse
    position; reusing one across turns would decode turn 2 against turn 1's state
    and silently produce malformed output.
    """

    def __init__(self, compiled_grammar: Any) -> None:
        import xgrammar as xgr  # type: ignore

        self._xgr = xgr
        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size = compiled_grammar.tokenizer_info.vocab_size
        self._bitmask = xgr.allocate_token_bitmask(1, self._vocab_size)
        self._started = False

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self._started:
            # Feed back the token just sampled, so the matcher's state matches
            # what the model actually emitted.
            self._matcher.accept_token(int(input_ids[0][-1].item()))
        else:
            self._started = True
        if not self._matcher.is_terminated():
            self._matcher.fill_next_token_bitmask(self._bitmask)
            self._xgr.apply_token_bitmask_inplace(
                scores,
                self._bitmask.to(scores.device),
                vocab_size=self._vocab_size,
                backend="torch_native",
            )
        return scores


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_extraction(raw: str) -> tuple[RawExtraction | None, str]:
    """A model reply → ``(RawExtraction, "")`` or ``(None, error)``.

    The error string is returned rather than logged because it is *fed back* to
    the model on the repair attempt: the failure is malformed JSON, and the
    parser's complaint is the one piece of information the model does not
    already have.

    Malformed *items* inside a well-formed object are dropped individually.  A
    single unparseable assertion should not discard the turn's other three —
    that would recreate the all-or-nothing loss this phase exists to remove —
    but it is also not silently repaired into a guess.
    """
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return None, "no JSON object found in the reply"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} at line {exc.lineno} column {exc.colno}"
    if not isinstance(obj, dict):
        return None, f"top-level value is a {type(obj).__name__}, not an object"

    mentions: list[str] = []
    for item in obj.get("mentions") or ():
        text = item.get("text") if isinstance(item, Mapping) else item
        if isinstance(text, str) and text.strip():
            mentions.append(text)

    assertions: list[RawAssertion] = []
    for item in obj.get("assertions") or ():
        if not isinstance(item, Mapping):
            continue
        kind = item.get("kind", "claim")
        if kind not in ASSERTION_KINDS:
            # The vocabulary is frozen (decision 9); an out-of-vocabulary kind is
            # recorded as the general case rather than dropped, because the
            # assertion's text and quotes are still usable evidence.
            kind = "claim"
        text_norm = item.get("text_norm") or ""
        if not isinstance(text_norm, str) or not text_norm.strip():
            continue
        quotes: list[Quote] = []
        raw_quotes = item.get("quotes")
        if raw_quotes is None and isinstance(item.get("quote"), str):
            # Tolerated, not blessed: the spike's single-quote schema.  Accepting
            # it is what lets the recorded spike extractions drive the write-path
            # tests without rewriting the file.
            raw_quotes = [{"turn_offset": 0, "text": item["quote"]}]
        for q in raw_quotes or ():
            if isinstance(q, str):
                q = {"turn_offset": 0, "text": q}
            if not isinstance(q, Mapping):
                continue
            text = q.get("text")
            # ``strip()``, matching the mention filter above: a whitespace-only
            # quote would "ground" against any blank run in the turn at rung 1
            # score 1.0, and a blank span would become recorded provenance
            # (found by the 13 Aug 2026 audit).
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                offset = int(q.get("turn_offset", 0))
            except (TypeError, ValueError):
                continue
            if offset > 0:
                continue
            quotes.append(Quote(offset, text))
        if not quotes:
            continue
        assertions.append(RawAssertion(kind, text_norm, quotes))

    return RawExtraction(mentions=mentions, assertions=assertions, raw=raw), ""


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


class ReplayExtractor:
    """Serves recorded extractions by ``turn_id``.  No model, no GPU.

    Its whole purpose is step 2 of the build order landing before step 3: the
    write path is what Phase 6 depends on, and it must not wait on GPU work.
    A turn with no recorded extraction returns an empty, ``parse_ok=True``
    extraction — the honest reading, since the recording is the ground truth of
    what that extractor produced.
    """

    name = "replay"

    def __init__(self, records: Mapping[str, Mapping[str, Any]], source: str = "") -> None:
        self._records = dict(records)
        self.source = source

    @classmethod
    def from_jsonl(cls, path: str | Path, key: str = "turn_id") -> "ReplayExtractor":
        records: dict[str, Mapping[str, Any]] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                records[row[key]] = row
        return cls(records, source=str(path))

    def __contains__(self, turn_id: object) -> bool:
        return turn_id in self._records

    def extract(self, turn: Turn, context: ExtractionContext) -> RawExtraction:
        del context
        row = self._records.get(turn.turn_id)
        if row is None:
            return RawExtraction()
        mentions = [
            m["text"] if isinstance(m, Mapping) else str(m) for m in row.get("mentions", ())
        ]
        assertions: list[RawAssertion] = []
        for item in row.get("assertions", ()):
            quotes_raw = item.get("quotes")
            if quotes_raw is None and item.get("quote"):
                quotes_raw = [{"turn_offset": 0, "text": item["quote"]}]
            quotes = [
                Quote(int(q.get("turn_offset", 0)), q["text"]) for q in (quotes_raw or ())
            ]
            if not quotes:
                continue
            kind = item.get("kind", "claim")
            assertions.append(
                RawAssertion(
                    kind if kind in ASSERTION_KINDS else "claim",
                    item.get("text_norm", ""),
                    quotes,
                )
            )
        return RawExtraction(mentions=mentions, assertions=assertions)


# --------------------------------------------------------------------------
# the model-backed extractor
# --------------------------------------------------------------------------


class LlmExtractor:
    """A causal LM behind the interface, with the repair-retry policy inside.

    **Loaded lazily and freeable** (fix F7).  The extractor and the NLI model are
    never resident together — mandatory rather than stylistic on an 8 GB card —
    so the pipeline runs stage-sequentially across the whole slice and calls
    :meth:`close` between stages.

    ``constrained=True`` is bakeoff candidate B.  It is implemented against
    ``transformers``' own ``GenerationConfig`` grammar support when the installed
    version exposes it, and otherwise refuses rather than silently degrading to
    candidate A — a candidate that quietly becomes another candidate would make
    the bakeoff table a fiction.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        device: str = "cuda",
        ledger: Ledger | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else require_extractor()
        self.name = f"{self.config['model_id']}@{self.config['revision'][:8]}"
        self.device = device
        self.ledger = ledger
        self._tok = None
        self._model = None
        self._grammar = None
        self._last_tokens: tuple[int, int] = (0, 0)
        self.load_seconds = 0.0
        self.peak_vram_mb: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        import time

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is not None:
            return
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "dtype": getattr(torch, self.config.get("dtype", "bfloat16")),
            "device_map": self.device,
            "revision": self.config["revision"],
        }
        if self.config.get("quantization") == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs.pop("dtype", None)
        self._tok = AutoTokenizer.from_pretrained(
            self.config["model_id"], revision=self.config["revision"]
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.config["model_id"], **kwargs)
        self._model.eval()
        self.load_seconds = time.perf_counter() - started
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.config.get("constrained"):
            self._grammar = self._build_grammar()

    def close(self) -> None:
        """Free the model.  Called between stages, not between turns."""
        try:
            import torch
        except ImportError:  # pragma: no cover - close() on a never-loaded object
            torch = None
        if torch is not None and self.device == "cuda" and torch.cuda.is_available():
            self.peak_vram_mb = round(torch.cuda.max_memory_allocated() / 2**20)
        self._model = None
        self._tok = None
        self._grammar = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "LlmExtractor":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- candidate B -------------------------------------------------------

    def _build_grammar(self) -> Any:
        """Compile the extraction schema into a decoding grammar (candidate B).

        **[ANALYSIS]** the boring-stack rule (architecture §0.4) requires a
        written justification for a new dependency, and this is it: candidate B's
        entire claim is *the output is never malformed JSON*, which cannot be
        approximated by prompting harder.  It is a bakeoff candidate rather than
        an adopted dependency — if B does not win, nothing in the shipped
        pipeline needs it.

        **What the grammar does not buy** (measured, 13 Aug 2026): it constrains
        every step to a valid JSON *prefix*, so the object is never malformed —
        but nothing makes it *close* within ``max_new_tokens``, and a truncated
        object does not parse.  G2's table called B's parse failure "0 by
        construction"; that is false under a finite token budget, and the
        remaining failure mode is shared with candidate A.
        """
        try:
            import xgrammar as xgr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "candidate B needs grammar-constrained decoding and no grammar "
                "backend is installed (`pip install xgrammar`). Refusing rather "
                "than falling back to candidate A: a candidate that silently "
                "becomes another candidate makes the bakeoff table a fiction."
            ) from exc

        config = self._model.config  # type: ignore[union-attr]
        info = xgr.TokenizerInfo.from_huggingface(self._tok, vocab_size=config.vocab_size)
        compiler = xgr.GrammarCompiler(info)
        return compiler.compile_json_schema(json.dumps(prompts.EXTRACTION_JSON_SCHEMA))

    # -- generation --------------------------------------------------------

    def _decoding_kwargs(
        self, *, constrained: bool, max_new_tokens: int | None = None
    ) -> dict[str, Any]:
        """The generation config for one call, as data.

        Split out of :meth:`_generate` so the two decisions it makes are
        testable without torch:

        * **the grammar applies to extraction calls only.**  ``complete`` serves
          the rolling summary and the obligation parser, and a grammar compiled
          from the *extraction* schema would force those replies into extraction
          JSON — under candidate B the summary would silently become schema junk
          in every prompt.  (Found 13 Aug 2026, while wiring the summary into
          the bakeoff harness; no reviewer had caught it.)
        * ``max_new_tokens`` may be overridden per call — the summary's 512-token
          cap (decision 3) is enforced here, at generation, rather than by the
          word-count backstop.
        """
        kwargs: dict[str, Any] = dict(DECODING)
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(max_new_tokens)
        if constrained and self._grammar is not None:
            # A fresh processor per generation — see GrammarLogitsProcessor.  The
            # expensive object is the *compiled grammar*, built once at load.
            kwargs["logits_processor"] = [GrammarLogitsProcessor(self._grammar)]
        return kwargs

    def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        constrained: bool = True,
        max_new_tokens: int | None = None,
    ) -> str:
        import torch

        assert self._tok is not None and self._model is not None
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        n_in = int(inputs["input_ids"].shape[1])

        kwargs = self._decoding_kwargs(
            constrained=constrained, max_new_tokens=max_new_tokens
        )
        kwargs["pad_token_id"] = self._tok.eos_token_id

        with torch.no_grad():
            out = self._model.generate(**inputs, **kwargs)
        generated = out[0][n_in:]
        n_out = int(generated.shape[0])

        if self.ledger is not None:
            self.ledger.count("llm_calls", 1)
            self.ledger.count("llm_tokens_in", n_in)
            self.ledger.count("llm_tokens_out", n_out)
        self._last_tokens = (n_in, n_out)
        return self._tok.decode(generated, skip_special_tokens=True)

    def complete(
        self, system: str, user: str, *, max_new_tokens: int | None = None
    ) -> tuple[str, int, int]:
        """One raw **unconstrained** completion, metered.

        Used by the summary and the obligation parser — never by extraction —
        so the grammar (candidate B) is deliberately not applied here; see
        :meth:`_decoding_kwargs`.
        """
        self.load()
        reply = self._generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            constrained=False,
            max_new_tokens=max_new_tokens,
        )
        n_in, n_out = self._last_tokens
        return reply, n_in, n_out

    # -- the interface -----------------------------------------------------

    def extract(self, turn: Turn, context: ExtractionContext) -> RawExtraction:
        self.load()
        summary_block = (
            f"CONVERSATION SUMMARY SO FAR:\n{context.summary}\n\n" if context.summary else ""
        )
        user = prompts.EXTRACT_USER.format(
            summary_block=summary_block,
            session_date=context.session_date or turn.ts,
            context=context.rendered(),
            speaker=turn.speaker,
            text=turn.text,
        )
        messages = [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ]

        calls = tokens_in = tokens_out = 0
        repairs = 0
        truncations = 0
        error = ""
        truncated = False
        cap = int(DECODING["max_new_tokens"])
        while True:
            reply = self._generate(messages)
            n_in, n_out = self._last_tokens
            calls += 1
            tokens_in += n_in
            tokens_out += n_out
            # ``truncated`` is the FINAL attempt's flag (it attributes the parse
            # failure's cause); ``truncations`` counts every capped generation,
            # including ones a successful repair would otherwise hide.
            truncated = n_out >= cap
            if truncated:
                truncations += 1
            parsed, error = parse_extraction(reply)
            if parsed is not None:
                parsed.parse_ok = True
                parsed.repairs = repairs
                parsed.llm_calls = calls
                parsed.tokens_in = tokens_in
                parsed.tokens_out = tokens_out
                parsed.truncated = truncated
                parsed.truncations = truncations
                return parsed
            if not self.config.get("repair") or repairs >= MAX_REPAIRS:
                return RawExtraction(
                    parse_ok=False,
                    repairs=repairs,
                    llm_calls=calls,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    error=error,
                    truncated=truncated,
                    truncations=truncations,
                    raw=reply,
                )
            repairs += 1
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": prompts.REPAIR_USER.format(error=error)},
            ]


def build_extractor(
    candidate: str | Mapping[str, Any] | None = None,
    *,
    device: str = "cuda",
    ledger: Ledger | None = None,
) -> LlmExtractor:
    """``'A'`` / a config dict / ``None`` (the frozen winner) → an extractor."""
    from graft.ingest.pins import EXTRACTOR_CANDIDATES

    if candidate is None:
        config: Mapping[str, Any] = require_extractor()
    elif isinstance(candidate, str):
        if candidate not in EXTRACTOR_CANDIDATES:
            raise KeyError(
                f"unknown candidate {candidate!r}; the declared set is "
                f"{sorted(EXTRACTOR_CANDIDATES)}"
            )
        config = EXTRACTOR_CANDIDATES[candidate]
        if config.get("withdrawn"):
            raise RuntimeError(
                f"candidate {candidate} was withdrawn before the bakeoff ran "
                f"({config['withdrawn']}). It stays in the declared table so the "
                "bakeoff's candidate set is readable after the fact; it is not "
                "selectable."
            )
    else:
        config = candidate
    return LlmExtractor(config, device=device, ledger=ledger)
