"""P5.2 — the rolling conversation summary (decision 3, gap G3).

**[EVIDENCE, qualified]** Mem0 (ECAI 2025) extracts with the previous *m* = 10
messages plus an asynchronously refreshed conversation summary.  The *content*
recipe is adopted; the *scheduling* is a declared adaptation, because
"asynchronous" in a one-process, no-services stack (architecture §0.4) would be
an invented scheduler with no way to test it.  The summary is refreshed
**synchronously every ``s = 10`` turns** by the same extractor model — one extra
LLM call per ten turns, metered like any other.

**A summary is derived state, not evidence.**  It is cached in the run directory
keyed by ``(conv_id, turn_ix)`` and **never written to the event log**: the log
stores what was said and what was extracted, and a summary is recomputable from
the log plus the frozen config.  Writing it would put model-generated prose into
the provenance chain, where `H`'s scope sub-check would eventually have to walk
it to a ``conv_id``.

Mem0 supports the recipe as *an implemented design, not an optimum* (plan §3.1's
own caveat), so ``m`` and ``s`` are §6 constants and not tuning knobs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from graft.ingest import prompts
from graft.ingest.pins import CONTEXT_TURNS, SUMMARY_EVERY, SUMMARY_MAX_TOKENS
from graft.schemas import Turn

__all__ = ["RollingSummary", "context_window"]

#: ``summarize(system, user) -> (text, tokens_in, tokens_out)``.  The same shape
#: ``LlmExtractor.complete`` returns, so the extractor model doubles as the
#: summariser without a second load — which is what makes "one extra LLM call per
#: ten turns" true rather than "one extra model".
Summarizer = Callable[[str, str], "tuple[str, int, int]"]


def context_window(turns: Sequence[Turn], ix: int, m: int = CONTEXT_TURNS) -> tuple[Turn, ...]:
    """The previous ``min(m, ix)`` turns, oldest first.

    Bounded at the *start of the conversation*, not of the session: a haystack's
    sessions are consecutive in time and a fact stated in session 3 can be
    referred to in session 4.  ``turns`` is therefore the conversation's turn
    stream and the caller does not window it by session first.
    """
    lo = max(0, ix - int(m))
    return tuple(turns[lo:ix])


class RollingSummary:
    """Synchronous rolling summary with an on-disk cache.

    The cache is keyed by ``(conv_id, turn_ix)`` and is a *cache*, not a record:
    deleting it changes throughput and nothing else, because the same summariser
    on the same prefix produces the same text under greedy decoding.
    """

    def __init__(
        self,
        summarizer: Summarizer | None = None,
        *,
        cache_dir: str | Path | None = None,
        every: int = SUMMARY_EVERY,
        max_tokens: int = SUMMARY_MAX_TOKENS,
    ) -> None:
        self.summarizer = summarizer
        self.every = int(every)
        self.max_tokens = int(max_tokens)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory: dict[tuple[str, int], str] = {}
        #: True when the on-disk cache could not be read and was skipped.  A
        #: reported flag rather than a silent recovery: the run is correct
        #: either way, but it spent LLM calls it expected to have cached.
        self.corrupt_cache = False
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    # -- cache -------------------------------------------------------------

    def _cache_path(self) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / "summaries.json"

    def _load_cache(self) -> None:
        """Read the cache, **tolerating a torn one**.

        The class docstring promises deleting this file changes throughput and
        nothing else; a corrupt file raising at construction would break that
        promise on exactly the crash path the write-through flush exists to
        serve — the run would die before touching a turn, and the fix would be
        to delete a file the documentation calls disposable.  ``EventLog`` sets
        the precedent: it scans and truncates a torn write on open rather than
        refusing to open.  A cache miss costs one LLM call; refusing to start
        costs the run.
        """
        path = self._cache_path()
        if path is None or not path.is_file():
            return
        try:
            raw: Mapping[str, str] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.corrupt_cache = True
            return
        for key, value in raw.items():
            conv_id, _, turn_ix = key.rpartition("#")
            try:
                self._memory[(conv_id, int(turn_ix))] = value
            except ValueError:  # a mangled key is a cache miss, not a crash
                self.corrupt_cache = True

    def flush(self) -> None:
        """Write the cache **atomically**: temp file, then ``os.replace``.

        ``write_text`` truncates at open, and since the write-through fix this
        runs once per refresh over multi-hour ingestions — so a crash mid-write
        is a realistic way to produce the torn file ``_load_cache`` now
        tolerates.  Tolerating it is the backstop; not creating it is the fix.
        """
        path = self._cache_path()
        if path is None:
            return
        payload = {f"{conv}#{ix}": text for (conv, ix), text in sorted(self._memory.items())}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(tmp, path)

    # -- the recipe --------------------------------------------------------

    def _refresh_point(self, ix: int) -> int:
        """The last turn index at which a refresh was due.

        Refreshes happen at multiples of ``s``, so every turn in ``[k·s, (k+1)·s)``
        shares one summary.  That is what makes the cost "one call per ten turns"
        rather than "one call per turn with a cheaper prompt".
        """
        return (ix // self.every) * self.every

    def summary_for(self, conv_id: str, turns: Sequence[Turn], ix: int) -> str:
        """The summary in force for turn ``ix`` of ``conv_id``.

        Empty for the first ``s`` turns: there is nothing established yet, and an
        LLM asked to summarise nothing produces filler that then enters every
        extraction prompt.
        """
        return self._summary_at(conv_id, turns, self._refresh_point(ix))

    def _summary_at(self, conv_id: str, turns: Sequence[Turn], point: int) -> str:
        """The summary as of refresh point ``point``, rebuilding the chain.

        **The chain is recomputed, not assumed** (found by the 13 Aug 2026
        audit): a resumed run skips already-ingested turns, so nothing walks the
        earlier refresh points — and reading the previous summary from the cache
        with a silent ``""`` default meant a resume with a cold cache built turn
        150's summary from turns 140–149 alone, changing the prompt, the
        extractions, and the digest on exactly the crash path G4 promises to
        repair.  Missing predecessors are now computed recursively (depth =
        ``point / s``, ~25 on the longest pilot question), and every fresh
        summary is **flushed to disk immediately** (write-through), so the cache
        survives the crash that makes the resume necessary.
        """
        if point <= 0:
            return ""
        key = (conv_id, point)
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        if self.summarizer is None:
            return ""

        previous = self._summary_at(conv_id, turns, point - self.every)
        recent = turns[max(0, point - self.every) : point]
        user = prompts.SUMMARY_USER.format(
            previous_block=(
                f"SUMMARY OF EVERYTHING BEFORE THESE TURNS:\n{previous}\n\n" if previous else ""
            ),
            n_turns=len(recent),
            turns="\n".join(f"[{t.speaker}] {t.text}" for t in recent),
        )
        text, n_in, n_out = self.summarizer(prompts.SUMMARY_SYSTEM, user)
        text = self._truncate(text)
        self.calls += 1
        self.tokens_in += int(n_in)
        self.tokens_out += int(n_out)
        self._memory[key] = text
        self.flush()
        return text

    def _truncate(self, text: str) -> str:
        """The word-count **backstop** behind decision 3's 512-token cap.

        The cap itself is enforced in real tokens at generation time — the
        pilot's summarizer passes ``max_new_tokens=SUMMARY_MAX_TOKENS`` through
        ``LlmExtractor.complete`` — because a word count alone is *not* a
        conservative bound: English words are one token **or more** on the
        pinned Qwen tokenizer (512 words measured at ~725 tokens), the exact
        opposite of what this docstring claimed before the 13 Aug 2026 audit.
        The word cap stays as a GPU-free backstop for summarizers that ignore
        the request: a ≤512-token generation always has ≤512 words, so on the
        live path this never binds.
        """
        words = (text or "").split()
        if len(words) <= self.max_tokens:
            return (text or "").strip()
        return " ".join(words[: self.max_tokens]).strip()

    def context_for(
        self, conv_id: str, turns: Sequence[Turn], ix: int, m: int = CONTEXT_TURNS
    ) -> tuple[str, tuple[Turn, ...]]:
        """``(summary, window)`` — the whole G3 recipe in one call (P5.2)."""
        return self.summary_for(conv_id, turns, ix), context_window(turns, ix, m)
