"""P10.2 — the frozen reader, and fix F7's model-loading discipline.

**This is the only module in ``graft.reader`` permitted to import torch**, on
the containment pattern Phase 8 set (torch reaches ``gate/model.py`` and nothing
else) and Phase 9 tightened.  ``pins``, ``serialize`` and ``parse`` stay cheap,
importable on a bare interpreter, and testable without weights — which is what
lets ``verify_handoff.py`` print the stage-E fingerprint on a machine that has
no GPU at all.

**Fix F7 is a resource constraint, and on this machine it is not theoretical.**
The architecture assumes *"1× RTX 5090 (32 GB)"*; `CLAUDE.md` §7 records that the
development machine is an **RTX 5050 Laptop GPU with 8 GB**.  Qwen2.5-3B at bf16
is ~6.2 GB of weights before activations, so the reader plus a resident embedder
plus a GNN does not fit.  F7's answer is stage-sequential execution — *"models
are context-managed (load → run stage → free)"* — and :class:`ModelSlot` is that
made enforceable rather than remembered: it refuses a second concurrent slot
rather than letting the run discover it as an OOM three stages in.

**The reader is frozen, and "frozen" is checked.**  No gradient is ever taken
here; :meth:`Reader.generate` runs under ``torch.no_grad()`` and the model is
placed in ``eval()``.  The supervisor's standing constraint is that learning
lives in the GNN/NN stack and the SLM only verbalises, so a trainable parameter
in this file would be a violation of the project's premise rather than a bug.

**Determinism is measured, not assumed** (G9).  Greedy decoding is deterministic
*given* identical inputs, dtype, kernels and batch composition — four conditions
this project has already been surprised by once, at the dtype: Stage A's
``ingestion_fingerprint`` hashes the extractor's dtype precisely because bf16 and
fp16 produce different digests from the same weights.  :meth:`Reader.determinism`
runs the same prompt three times and byte-compares, and the result is reported
either way.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from graft.reader.pins import DECODING, INSUFFICIENT, PROMPT_SHA, PROMPT_TEMPLATE, READER

__all__ = ["ModelSlot", "Reader", "build_prompt"]


class ModelSlot:
    """Fix F7, enforced: at most one LLM resident at a time.

    A process-wide, re-entrant-refusing guard.  It is deliberately **not** a
    counter that warns — the failure it prevents is an OOM on an 8 GB card
    partway through a run, and a warning that is ignored costs the whole run.

    ``threading.Lock`` rather than a bare flag because the ledger and the
    orchestrator may both hold references; the lock makes "already open" a fact
    rather than a race.  One process, one GPU is the architecture's own boring-
    stack rule, so nothing here needs to be cleverer than that.
    """

    _lock = threading.Lock()
    _holder: str | None = None

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "ModelSlot":
        if not ModelSlot._lock.acquire(blocking=False):
            raise RuntimeError(
                f"cannot open model slot {self.name!r}: slot "
                f"{ModelSlot._holder!r} is already resident. Fix F7 requires "
                "stage-sequential execution — load, run the stage, free — because "
                "the reader (~6.2 GB bf16) and a second model do not fit in 8 GB "
                "(CLAUDE.md §7). Free the other slot first."
            )
        ModelSlot._holder = self.name
        return self

    def __exit__(self, *exc: Any) -> None:
        ModelSlot._holder = None
        ModelSlot._lock.release()

    @classmethod
    def holder(cls) -> str | None:
        """Which slot is open, or ``None``.  For the orchestrator's report."""
        return cls._holder


def build_prompt(evidence: str, question: str) -> str:
    """The one template, filled.  No branch on corpus, system or evidence source.

    v1.2 §3.5 requires the same prompt for **every** compared system, and the
    architecture makes that a hash rather than a promise.  Phase 11's baselines
    call this same function, so a baseline that reworded its prompt would show up
    as a different ``PROMPT_SHA`` rather than as a better score.
    """
    return PROMPT_TEMPLATE.format(evidence=evidence, question=question)


class Reader:
    """Qwen2.5-3B-Instruct, frozen, greedy, context-managed.

    Weights are already local — the same checkpoint Phase 5 pinned as its
    extractor (`ingest/pins.py`), so no download is needed and the two stages
    cannot drift onto different revisions of the same name.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        ledger: Any = None,
    ) -> None:
        # **Metering lives inside the wrapper, counted where the tokens are.**
        # That is this project's standing convention (`ingest/extractor.py`
        # lines 505-508, `ingest/nli.py` line 168), and the reason is that a
        # caller-side count is a count someone forgets: the first version of this
        # class held no ledger at all, so `generate` structurally could not
        # meter, and the Phase-10 artefact reported zero tokens for a run that
        # invoked a 3-billion-parameter reader eighteen times. Found by
        # adversarial audit, 16 Aug 2026.
        self.ledger = ledger
        self.model_id = model_id or str(READER["model_id"])
        self.device = device or str(READER["device"])
        self.dtype = getattr(torch, str(READER["dtype"]))
        self.model: Any = None
        self.tokenizer: Any = None
        self._slot: ModelSlot | None = None

    # -- lifecycle (fix F7) ------------------------------------------------

    def __enter__(self) -> "Reader":
        self._slot = ModelSlot(f"reader:{self.model_id}")
        self._slot.__enter__()
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.free()
        if self._slot is not None:
            self._slot.__exit__(*exc)
            self._slot = None

    def load(self) -> "Reader":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=None
        ).to(self.device)
        # Frozen, and asserted rather than intended: the supervisor's constraint
        # is that the SLM only verbalises. `eval()` alone would still leave
        # gradients reachable.
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self

    def free(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- the read path -----------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """The reader's **own** tokenizer — what a reported ceiling 4 needs.

        ``serialize.approx_tokens`` is a declared heuristic for fixtures; a
        ceiling-4 number computed under it is marked approximate, and this is the
        counter that removes the mark.
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenizer not loaded; use `with Reader() as r:`")
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def generate(self, evidence: str, question: str) -> str:
        """One greedy generation from the frozen template.

        Returns only the **continuation**, not the prompt — the parser scores the
        answer, and a prompt echoed into the answer text would put the literal
        string ``INSUFFICIENT EVIDENCE`` (from rule 4) into every generation,
        making every query parse as an abstention.
        """
        if self.model is None:
            raise RuntimeError("model not loaded; use `with Reader() as r:`")
        prompt = build_prompt(evidence, question)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=bool(DECODING["do_sample"]),
            num_beams=int(DECODING["num_beams"]),
            max_new_tokens=int(DECODING["max_new_tokens"]),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        continuation = out[0][inputs["input_ids"].shape[1]:]

        if self.ledger is not None:
            # One generation = one call, one forward-pass unit, and its real
            # token counts. `CLAUDE.md` §9 makes latency and token cost a
            # *claimed* axis -- the project may not claim to beat full-context on
            # accuracy and must win here instead -- so an unmetered reader leaves
            # that claim with no numerator, and Phase 11's matched-budget
            # comparison with nothing to match against.
            self.ledger.count("llm_calls", 1)
            self.ledger.count("model_forwards", 1)
            self.ledger.count("llm_tokens_in", int(inputs["input_ids"].shape[1]))
            self.ledger.count("llm_tokens_out", int(continuation.shape[0]))

        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()

    # -- G9: determinism, measured -----------------------------------------

    def determinism(self, evidence: str, question: str, repeats: int = 3) -> dict[str, Any]:
        """Run the same prompt ``repeats`` times and byte-compare (run R1).

        Reported either way.  If it is **not** bit-reproducible that is a finding
        about the reader and a caveat on every downstream number — not a bug to
        chase, and not something to discover later when two runs disagree.
        """
        outputs = [self.generate(evidence, question) for _ in range(repeats)]
        identical = len(set(outputs)) == 1
        return {
            "repeats": repeats,
            "bit_identical": identical,
            "distinct_outputs": len(set(outputs)),
            "first": outputs[0],
            "note": (
                "greedy is deterministic given identical inputs, dtype, kernels "
                "and batch composition; this measures whether those hold here"
            ),
        }

    def report(self) -> dict[str, Any]:
        peak = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else None
        )
        return {
            "model_id": self.model_id,
            "dtype": str(READER["dtype"]),
            "device": self.device,
            "prompt_sha": PROMPT_SHA,
            "decoding": {k: v for k, v in sorted(DECODING.items())},
            "peak_vram_gb": round(peak, 3) if peak else None,
            "slot": ModelSlot.holder(),
            "abstain_string": INSUFFICIENT,
        }
