"""Everything Phase 5 freezes, in one place, importable without an ML library.

Two jobs, and the second is the reason this module exists separately from the
components that use these values.

**It is the §6 decision table, executable.**  Every constant here is a row of
`GRAFT_PHASE5_BUILD.md` §6: the extractor config (decision 2), the summary
cadence and context window (3, 4), the grounding ladder (5), the NLI pin (6),
the extraction schema's frozen ``kind`` vocabulary (9).  Values that belong to
the *config tree* are deliberately absent — ``tau_nli`` and ``support_policy``
live in ``graft.config`` and are read from there, because Phase 5 audits
``tau_nli`` and must not be able to retune it (G6).

**It is the ingestion fingerprint's payload (G11).**  A bf16 LLM forward pass is
not bit-stable across GPU architectures, so `README.md`'s cross-machine
byte-identity promise cannot cover Stage A.  What *can* be promised across
machines is **configuration identity**, and :func:`ingestion_fingerprint` is that
promise in checkable form — the same hash-equality discipline architecture §10.1
already mandates for the reader.  ``scripts/verify_handoff.py`` prints it, which
is why this module imports neither torch nor transformers: the handoff script
must keep running on a bare interpreter.
"""

from __future__ import annotations

from typing import Any

from graft.canonical import digest_of
from graft.ingest.prompts import REGISTRY_SHA
from graft.schemas import ASSERTION_KINDS, SCHEMA_VERSION

__all__ = [
    "EXTRACTOR_CANDIDATES",
    "RUNNABLE_CANDIDATES",
    "EXTRACTOR",
    "require_extractor",
    "NLI",
    "SUMMARY_EVERY",
    "SUMMARY_MAX_TOKENS",
    "CONTEXT_TURNS",
    "CONTEXT_CLIP_CHARS",
    "OBLIGATION_AUDIT_N",
    "FUZZY_ACCEPT",
    "GROUNDING_RUNGS",
    "MAX_REPAIRS",
    "PARSE_FAILURE_CEILING",
    "SPAN_PRECISION_FLOOR",
    "ASSERTION_KINDS",
    "ingestion_fingerprint",
    "frozen_values",
]


# --------------------------------------------------------------------------
# decision 2 — the extractor, and the three candidates the bakeoff ran
# --------------------------------------------------------------------------

#: The G2 bakeoff's candidates, declared **before** the run.  Each is a complete
#: configuration: a model, a precision, and a parse-failure strategy.  The pick
#: rule is in ``graft.ingest.bakeoff`` and was likewise declared first.
#:
#: ``revision`` is a commit sha on the Hugging Face repo, not a tag.  A tag can
#: be moved; a sha cannot, and "same model id" is not the same claim as "same
#: weights" once a repo has been updated.
#:
#: **Candidate C is withdrawn, by ruling rather than by measurement** (project
#: owner, 13 Aug 2026): the project's extractor is Qwen2.5-**3B**, extending the
#: Phase-2.5 amendment (`PHASE2_5_DECISIONS.md` §1) from the spike to Phase 5.
#: It stays in this table because the bakeoff's honesty depends on the candidate
#: set being the *declared* one — deleting a predeclared candidate after the run
#: would make the table unreadable — and it carries ``withdrawn`` so no code path
#: can select it.  The consequence is recorded rather than glossed: **whether the
#: architecture's 4-bit 7B extractor fits in 8 GB remains unmeasured**, which was
#: fix F7's original question, and the architecture's own extractor row is
#: superseded for this project by the 3B ruling.
#:
#: The bakeoff therefore decides between two *parse-failure strategies* on one
#: model, which is the question that actually matters here: the spike's 15.5%
#: silent turn loss is the defect, and A and B are the two ways to fix it.
EXTRACTOR_CANDIDATES: dict[str, dict[str, Any]] = {
    "A": {
        "candidate": "A",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "dtype": "bfloat16",
        "quantization": None,
        "repair": True,
        "constrained": False,
        "note": "the spike's measured stack plus a bounded repair-retry",
    },
    "B": {
        "candidate": "B",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "dtype": "bfloat16",
        "quantization": None,
        "repair": False,
        "constrained": True,
        # **Corrected 13 Aug 2026, by measurement.**  G2's table says candidate
        # B's parse failure is "0 by construction".  That is false under a finite
        # token budget, and the bakeoff measured it: a grammar guarantees the
        # output is a valid JSON *prefix*, not that the object *closes* within
        # `max_new_tokens`.  A truncated object has no closing brace and does not
        # parse.  So B's guarantee is "never malformed", not "never unparseable",
        # and the remaining failure mode is shared with A.
        "note": "grammar-constrained decoding; never malformed, but a truncated "
        "object still fails to parse — the guarantee is prefix validity, not "
        "completion within the token budget",
    },
    "C": {
        "candidate": "C",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "dtype": "bfloat16",
        "quantization": "nf4",
        "repair": True,
        "constrained": False,
        "withdrawn": "project owner, 13 Aug 2026: the extractor is Qwen2.5-3B",
        "note": "the architecture's own pick; withdrawn before the run, so whether "
        "a 4-bit 7B fits in 8 GB (fix F7's original question) stays unmeasured",
    },
}

#: The candidates the bakeoff may actually run.  Derived rather than hand-listed,
#: so a withdrawal is one edit and cannot leave a stale second list behind.
RUNNABLE_CANDIDATES: tuple[str, ...] = tuple(
    name for name, config in sorted(EXTRACTOR_CANDIDATES.items())
    if not config.get("withdrawn")
)

#: **Decision 2, frozen by the bakeoff** — transcribed here from
#: ``artefacts/phase5_bakeoff.json`` once the predeclared rule has picked a
#: winner, exactly as Phase 3's ``N`` and β are meant to be written into its §6
#: after the calibration gate.  ``None`` means the bakeoff has not run, and
#: every component needing an extractor raises rather than quietly defaulting to
#: the candidate that happens to be listed first.
#: **Frozen 14 Aug 2026 by the corrected bakeoff: candidate B.**
#: `artefacts/phase5_bakeoff.json` — B 1.7% parse failure (1 truncation, zero
#: malformed) against A's 23.3% (14 malformed), so stage 1's hard filter left B
#: as the sole survivor.  Transcribed by hand from the artefact, as the script
#: instructs and never writes.
EXTRACTOR: dict[str, Any] | None = dict(EXTRACTOR_CANDIDATES["B"])

#: Greedy, always.  Sampling would make two runs on one machine differ, which is
#: the *only* determinism Stage A can promise (G11) — spending it on decoding
#: temperature buys nothing extraction-shaped.
#:
#: ``max_new_tokens`` was **600 until 13 Aug 2026** and was raised by
#: measurement, before the bakeoff froze anything (`pins.EXTRACTOR` was still
#: ``None``, so this is an instrument correction, not a post-hoc change).  The
#: aborted first run's arithmetic: candidate A's 12,915 output tokens over 78
#: generations decompose as 15 × 600 + 63 × 62 — every one of its 15 parse
#: failures consumed exactly the cap while successful parses averaged 62 tokens.
#: 600 also sits exactly where a legitimately dense turn lands (ten assertions
#: at ~60 tokens each), so the old cap could not distinguish "the model rambled"
#: from "the turn was dense".  1024 clears legitimate density with headroom
#: while still bounding a repetition loop; ``truncated_at_token_cap`` in the
#: failure-cause table and ``truncated_generations`` measure the residual.
#:
#: **Raised again to 2048 on 14 Aug 2026, and the argument is the reason, not
#: the outcome.**  The first corrected bakeoff measured a residual: 2 of 60 turns
#: still truncated at 1024.  The principle this constant answers is *the cap is a
#: runaway guard, never a content limit* — it exists to stop a repetition loop,
#: and any value at which it binds on real extractions is silently deleting
#: evidence, which is the exact failure Phase 5 exists to remove (a turn that
#: truncates yields **nothing**, indistinguishable downstream from a turn with
#: nothing to say).  At 1024 it demonstrably binds on real content.  2048 is
#: ~10× the measured mean output (192–220 tokens across both candidates) and
#: covers ~20 assertions at the schema's ~70 tokens each, which is denser than
#: any turn in the pinned corpus; a repetition loop still terminates.
#:
#: **The expected effect was stated before the re-run**, because a budget raised
#: after seeing which candidate it rescues is not an instrument correction:
#: candidate B's failures are *all* truncations, so B was predicted to fall to
#: ~0%; candidate A's are 14 malformed against 1 truncated, so A was predicted to
#: stay near 25%.  The raise was expected to change the verdict, and saying so in
#: advance is what separates it from tuning until a candidate passes.
DECODING: dict[str, Any] = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 2048,
}

#: One reprompt on a parse failure, carrying the parse error back (candidate A
#: and C's repair policy).  Bounded because an unbounded retry loop turns a
#: throughput measurement into a wall-clock lottery.
MAX_REPAIRS = 1


# --------------------------------------------------------------------------
# decision 6 — the NLI verifier
# --------------------------------------------------------------------------

#: **[ANALYSIS]** the *pattern* is VeriCite's (SIGIR-AP 2025) — an NLI
#: cross-encoder was the cost-effective verifier against LLM judges (citation F1
#: 80.05 vs 73.01 for an 8B LLM) — but TRUE's own reference model is a T5-11B,
#: unusable in 8 GB, so the concrete pin is judgment made here.  DeBERTa-v3-base
#: NLI, ~184M parameters, fp32 on CPU or GPU in well under a gigabyte.
NLI: dict[str, Any] = {
    "model_id": "cross-encoder/nli-deberta-v3-base",
    "revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
    "entail_label": "entailment",
    "max_length": 512,
    "batch_size": 16,
}


# --------------------------------------------------------------------------
# decisions 3 and 4 — the extraction context
# --------------------------------------------------------------------------

#: **[EVIDENCE, qualified]** Mem0 (ECAI 2025) extracts with the previous
#: *m* = 10 messages plus an asynchronously refreshed conversation summary.  The
#: *content* recipe is kept; the *scheduling* is a declared adaptation, because
#: "asynchronous" in a one-process, no-services stack (architecture §0.4) would
#: be an invented scheduler.  Mem0 supports the recipe as an implemented design,
#: not an optimum, so these are §6 constants rather than tuning knobs.
CONTEXT_TURNS = 10
SUMMARY_EVERY = 10

#: The summary cap, **enforced in model tokens at generation time**: every
#: summary call passes ``max_new_tokens=SUMMARY_MAX_TOKENS`` through
#: ``LlmExtractor.complete``.  ``RollingSummary._truncate`` keeps a word-count
#: backstop for summarizers that ignore the request (and for the GPU-free test
#: path) — a backstop, not the enforcement, because for English words ≤ tokens,
#: so a 512-*word* cap alone admits ~725 tokens on the pinned Qwen tokenizer
#: (measured, 13 Aug 2026).
SUMMARY_MAX_TOKENS = 512

#: **Declared adaptation of decision 4, 13 Aug 2026 — before the bakeoff froze
#: anything.**  Each *context-window* turn is clipped to its first
#: ``CONTEXT_CLIP_CHARS`` characters when rendered into the extraction prompt;
#: the current turn — the extraction target — is never clipped.  [ANALYSIS]
#: Mem0's *m* = 10 recipe (ECAI 2025) is over short chat messages; LongMemEval's
#: assistant turns are essay-length, and applying m = 10 to them unclipped was
#: measured at **5,017 input tokens per turn**, which overflowed the 8 GB card
#: (9,515 MB peak) and collapsed throughput 12× (18.4 → 1.6 tok/s).  Clipping
#: long turns to their head keeps the recipe inside the regime its evidence
#: covers.  ~600 chars ≈ 150 tokens, so a full window costs ≈ 1.5k tokens.
#: Grounding is unaffected: quotes are resolved against the *full* turn text,
#: and a quote the model emits necessarily comes from the visible clipped part.
CONTEXT_CLIP_CHARS = 600

#: Decision 14's audit-set size (G7): ~50 questions, hand-labelled on all six
#: slots.  Its own constant rather than a reuse of ``SPAN_AUDIT_N`` — the two
#: audits belong to different decisions (1 vs 14) and different gaps (G1 vs G7),
#: and one constant with two meanings is how amending one silently resizes the
#: other.
OBLIGATION_AUDIT_N = 50


# --------------------------------------------------------------------------
# decision 5 — the grounding ladder
# --------------------------------------------------------------------------

#: The four rungs, in order.  Reported per span, because Phase 6 trains on these
#: offsets and "how was this offset obtained" is part of the label's quality.
GROUNDING_RUNGS: tuple[str, ...] = ("exact", "normalised", "fuzzy", "failed")

#: difflib acceptance for rung 3.  The spike's value, kept: it is the number the
#: one measured mis-bound span (`d1_0022`) was produced at, so moving it would
#: also move the baseline the G5 mis-bound rate is measured against.
FUZZY_ACCEPT = 0.85


# --------------------------------------------------------------------------
# decision 1 and the G2 rule's hard filter — the two declared thresholds
# --------------------------------------------------------------------------

#: G2's hard filter, declared before the bakeoff ran.  A production Stage A
#: cannot ship the spike's 15.5% silent turn loss: those turns yielded *nothing*,
#: which is evidence deleted before ceiling 1 could measure it.
PARSE_FAILURE_CEILING = 0.02

#: Decision 1 — the Gate-0 span-support threshold, on a 50-assertion audited
#: sample of pilot output.  **[ANALYSIS]** deliberately tighter than the spike's
#: 0.80 floor (`PHASE2_5_DECISIONS.md` A2): that floor guarded a *timing
#: measurement*, this one guards Phase 6's *training data*, and a 1-in-10
#: unsupported-assertion rate is the level the support gate exists to catch, not
#: to admit.
SPAN_PRECISION_FLOOR = 0.90
SPAN_AUDIT_N = 50


# --------------------------------------------------------------------------
# the fingerprint
# --------------------------------------------------------------------------


def require_extractor() -> dict[str, Any]:
    """The frozen extractor config, or a refusal that names what is missing.

    Fails closed for the same reason ``Assertion.eligibility`` defaults to
    ``quarantined``: an unrun bakeoff silently falling back to a candidate would
    put an *unmeasured* extractor behind every downstream artefact, and the
    predeclaration in G2 exists precisely so that cannot happen quietly.
    """
    if EXTRACTOR is None:
        raise RuntimeError(
            "no extractor is frozen: run scripts/phase5_bakeoff.py and transcribe "
            "the winner into pins.EXTRACTOR (decision 2). Until then only the "
            "ReplayExtractor is usable, which is what the write-path tests run on."
        )
    return dict(EXTRACTOR)


def frozen_values(extractor: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every Phase-5 constant that changes what an ingestion run produces.

    ``tau_nli`` and ``support_policy`` are **absent on purpose**: they are
    ``Config`` fields, they already move ``config_hash``, and duplicating them
    here would give one frozen value two homes — the failure mode `CLAUDE.md` §5
    catalogues.  A run's full identity is therefore the pair
    ``(config_hash, ingestion_fingerprint)``.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor": extractor if extractor is not None else EXTRACTOR,
        "decoding": DECODING,
        "max_repairs": MAX_REPAIRS,
        "nli": NLI,
        "context_turns": CONTEXT_TURNS,
        "context_clip_chars": CONTEXT_CLIP_CHARS,
        "summary_every": SUMMARY_EVERY,
        "summary_max_tokens": SUMMARY_MAX_TOKENS,
        "grounding_rungs": list(GROUNDING_RUNGS),
        "fuzzy_accept": FUZZY_ACCEPT,
        "assertion_kinds": list(ASSERTION_KINDS),
        "prompt_registry_sha": REGISTRY_SHA,
    }


def ingestion_fingerprint(
    extractor: dict[str, Any] | None = None, length: int | None = None
) -> str:
    """Hash of the *configuration*, never of the output (G11).

    Two machines running Stage A must agree on this and are **not** promised
    identical log digests.  Stating which of the two is guaranteed is the honest
    version of `README.md`'s cross-machine claim, and recording it here is what
    stops it being rediscovered as a mystery in Phase 11.
    """
    return digest_of(frozen_values(extractor), length)
