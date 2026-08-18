"""P10.0 — everything Phase 10 freezes, importable without an ML library.

Same shape and same reason as ``graft.ingest.pins``, ``graft.graphbuild.pins``,
``graft.retrieve.pins``, ``graft.gate.pins`` and ``graft.setgen.pins``: this
module is §6's **signed** decision table made executable, and it carries the
**stage-E fingerprint** that ``scripts/verify_handoff.py`` prints — so it must
stay importable on a bare interpreter.

**That last requirement is not decorative, and Phase 9 shipped a violation of
it.** `PHASE9_DECISIONS.md` §7.3 records a stage-D fingerprint whose *module*
imported clean while the *call* pulled torch in through a feature-name import —
the harder version of the bug to notice, because nothing fails until someone
runs ``verify_handoff.py`` on a machine without torch. Nothing here imports from
:mod:`graft.reader.read`, which is the only module in this package permitted to
touch torch.

**What the prompt is for, and why it is a pin rather than a string in a
function.** v1.2 §3.5 requires "same frozen SLM, same prompt template, same
decoding budget for **every** compared system", and the architecture makes that
enforceable rather than promised: *"The (prompt, decoding-config) SHA is stamped
into every ``OutputRecord``"*. Phase 11's baselines — full-context,
matched-budget RAG, Mem0 — must reuse :data:`PROMPT_TEMPLATE` byte-identically,
so a system that quietly reworded it would be visible as a different
fingerprint rather than as a better score.

**Values that belong elsewhere are absent here on purpose.**
``serialization_budget_tokens`` is the **config tree's** (it is a frozen
experimental condition that Phase 11's matched-budget baseline is *defined* by,
and it reaches identity through ``config_hash``); ``K`` and ``checker_budget``
likewise. Giving a frozen value two homes is the failure `CLAUDE.md` §5
catalogues. What this module freezes is what Phase 10 *adds*.
"""

from __future__ import annotations

from typing import Any

from graft.canonical import digest_of

__all__ = [
    "READER",
    "PROMPT_TEMPLATE",
    "PROMPT_SHA",
    "DECODING",
    "CLAIM_ID_FORMAT",
    "ORDERING",
    "BUDGET_LADDER",
    "ANSWER_EQUIVALENCE",
    "ANSWER_SCORING",
    "ABSTENTION_AGGREGATION",
    "CEILINGS",
    "CONTESTED",
    "POST_HOC_VERIFICATION",
    "frozen_values",
    "stage_e_fingerprint",
]

# --------------------------------------------------------------------------
# decision 2 — the frozen reader and its prompt (G2)
# --------------------------------------------------------------------------

#: The reader is **frozen** and is the same checkpoint Phase 5 used as its
#: extractor, so the weights are already local and no download is needed.
#:
#: **[EVIDENCE, provisional] Reader size is a declared experimental condition,
#: not a design law.**  arXiv 2607.00725 found careful evidence packing helped a
#: 3B reader (+0.022 F1, p<0.05), was null at 7B (−0.010, p=0.45) and
#: **reversed at 14B** (−0.029, p=0.013) — one setting, one dataset family, one
#: embedder.  `CLAUDE.md` §4.2 draws the consequence this project must live
#: with: *if you swap in a bigger reader, the minimality benefit may vanish or
#: invert*.  So the size is pinned and reported as a condition, never argued as
#: universally correct.
#:
#: ``dtype`` is part of the pin because it is part of the identity: Stage A's
#: ``ingestion_fingerprint`` hashes the extractor's dtype for exactly this
#: reason, bf16 and fp16 producing different digests on the same weights
#: (`DATASET_DECISION.md`).
READER: dict[str, Any] = {
    "model_id": "Qwen/Qwen2.5-3B-Instruct",
    # **Pinned by revision as well as by id** — added 16 Aug 2026 after an
    # adversarial audit noted that `graphbuild.pins.EMBEDDER` pins both and this
    # pinned only the name. A model id is a moving target: the same name can
    # serve different weights over time, and "same frozen reader for every
    # compared system" (v1.2 §3.5) is unenforceable if the fingerprint binds a
    # label rather than a checkpoint. `None` means "whatever is cached", which is
    # what the current local snapshot is; set it to the resolved commit before any
    # reported run so two machines cannot disagree silently.
    "revision": None,
    "dtype": "bfloat16",
    "device": "cuda",
    "trust_remote_code": False,
}

#: **The prompt, frozen.**  One template, no corpus branch, no system branch.
#:
#: Four properties are deliberate.  It asks for citations **inline by claim id**,
#: because ALCE (EMNLP 2023) scores citation and answer correctness separately
#: and a free-form bibliography cannot be resolved back to a span.  It states the
#: abstention wording explicitly, so "I don't know" is a *parseable* outcome
#: rather than something the parser has to infer from hedging.  It never mentions
#: the corpus, the system, or how the evidence was selected — a reader told
#: "these are GRAFT's minimal proofs" and one told "these are the top BM25 hits"
#: are not the same experimental condition, which is the whole point of Phase 11
#: reusing this byte-identically.
#:
#: And it asks for **the shortest answering phrase, not a sentence** — caught on
#: the first smoke run, 16 Aug 2026, before anything was frozen.  The first draft
#: said "Answer in one short sentence", and every benchmark this project scores
#: against (LongMemEval, 2Wiki, MuSiQue, LoCoMo) has a **short span** as gold.
#: A sentence answer scores ~0 exact-match against a span by construction, so
#: decision 10's local exact/F1 pair would have measured verbosity rather than
#: correctness — and the prompt SHA would have frozen that into the stage-E
#: fingerprint before a single run.  The citation requirement is what needs
#: brackets; the answer does not need prose.
PROMPT_TEMPLATE = (
    "You answer questions using only the numbered evidence provided.\n"
    "\n"
    "Rules:\n"
    "1. Use only the evidence below. Do not use outside knowledge.\n"
    "2. Answer with the shortest phrase that answers the question — a name, a "
    "date, or a few words. Do not write a full sentence.\n"
    "3. After the answer, cite the evidence you used with its bracketed id, "
    "like [c1] or [c2].\n"
    "4. If the evidence is not sufficient to answer, reply exactly: "
    "INSUFFICIENT EVIDENCE\n"
    "\n"
    "Evidence:\n"
    "{evidence}\n"
    "\n"
    "Question: {question}\n"
    "Answer:"
)

#: The abstention string the prompt names and the parser matches.  One constant,
#: two consumers — a parser looking for different words than the prompt asks for
#: is a silent abstention-detection failure.
INSUFFICIENT = "INSUFFICIENT EVIDENCE"

PROMPT_SHA = digest_of({"template": PROMPT_TEMPLATE, "insufficient": INSUFFICIENT})

#: Greedy, so the read path is a function of its inputs rather than of a seed.
#: ``max_new_tokens`` is small because the prompt asks for one sentence; a reader
#: allowed to ramble would make citation precision a function of verbosity.
#:
#: **Whether greedy is *bit*-reproducible here is measured, not assumed** (G9).
#: Greedy is deterministic given identical inputs, dtype, kernels and batch
#: composition — four conditions this project has already been surprised by once
#: at the dtype.
DECODING: dict[str, Any] = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 128,
    "temperature": None,
    "top_p": None,
}

# --------------------------------------------------------------------------
# decisions 3 and 4 — serialisation (G3, G4)
# --------------------------------------------------------------------------

#: Claim ids in **serialized order**, so ``[c1]`` is always the first evidence
#: block the reader sees.  The record carries the id → atom map, which is what
#: makes a citation resolvable to a character span in a real turn through the
#: chain ``H``'s support and scope sub-checks already walk.
CLAIM_ID_FORMAT = "c{index}"

#: **[EVIDENCE]** Lost in the Middle (TACL 2024): a U-shaped position curve —
#: models use evidence at the **edges** of a context far better than in the
#: middle.  So the strongest evidence goes first and last, and connective
#: evidence fills the middle.
#:
#: **The signals are inference-computable, and that is decision 3's whole
#: content.**  The architecture's own phrasing was *"anchor and **answer-binding**
#: evidence at the beginning and end"* — and answer-binding requires the answer.
#: This is the **third** appearance of that class of error in this project:
#: `PHASE9_DECISIONS.md` §1.3 caught it in the 2Wiki anchor rule (a gold
#: annotation reaching every arm's ``state_repr``), and Phase 9's G9 caught it in
#: fix F4's contested flag.  Each time it read as innocuous prose.
#:
#: ``head``/``tail`` take the highest-scoring atoms by the declared key;
#: ``middle`` takes the rest, edge atoms first, because an edge *is* the
#: connective evidence the citation is meant to describe.
ORDERING: dict[str, Any] = {
    "shape": "U — strongest at head and tail, connective in the middle",
    "signals": (
        "obligation_anchor_hit",   # the parser reads the question; no gold
        "fused_retrieval_score",   # Stage C's own output
        "atom_kind",               # node vs edge — structural, not semantic
    ),
    "forbidden_signals": (
        "gold_atom_ids",
        "answer_binding",
        "is_supporting",
    ),
    "gold_ordering": (
        "implemented as a DIAGNOSTIC only; the head-to-head gap against the "
        "honest ordering is the measurement of what honesty costs, and reporting "
        "either alone is the failure decision 3 exists to prevent"
    ),
}

#: Decision 1's reporting rule.  Ceiling 4 is reported at **every** rung, never
#: one — `CLAUDE.md` §8 records that quoting a packing result at a single
#: cherry-picked budget was one of this project's own caught errors, and 160 is
#: on the ladder so that exact comparison is reproducible in this project's
#: units.  ``Config.serialization_budget_tokens`` is the *live* value; this is
#: the set a ceiling-4 table must span.
BUDGET_LADDER: tuple[int, ...] = (160, 512, 1024)

# --------------------------------------------------------------------------
# decisions 9 and 10 — answer equivalence and scoring (G10)
# --------------------------------------------------------------------------

#: **[EVIDENCE]** the normalisation SQuAD (EMNLP 2016) introduced and which
#: HotpotQA, 2WikiMultiHopQA and MuSiQue all inherit — so a local number computed
#: this way is commensurable with published ones rather than being this
#: project's private dialect.
#:
#: **One rule, two consumers.**  The contested check (G8) and the local scorer
#: (decision 10) call the same function, because two equivalence rules in one
#: pipeline is how a system comes to disagree with itself about whether two
#: answers are the same.
ANSWER_EQUIVALENCE: dict[str, Any] = {
    # Order corrected 16 Aug 2026: the pin declared articles-before-punctuation
    # while `parse.normalise_answer` has always done punctuation first, which is
    # the official SQuAD order and the correct one -- stripping articles first
    # would leave "the-dog" intact. The pin was wrong, not the code, and the pin
    # is what the stage-E fingerprint hashes.
    "normalise": ("casefold", "delete_punctuation", "strip_articles", "collapse_whitespace"),
    "articles": ("a", "an", "the"),
    "then": "exact match, then alias-set match",
    "basis": "SQuAD (EMNLP 2016) normalisation, inherited by HotpotQA/2Wiki/MuSiQue",
}

#: Decision 10, fixed **before any end-to-end run** so the judge is not chosen
#: after seeing which one flatters the system.
#:
#: The primary is the benchmark's own script where one ships, so the number is
#: comparable to published ones; the local pair is the reproducible-without-API
#: floor.  If a benchmark ships no script, that is a **finding to record** and
#: the local pair becomes primary with the substitution stated — not a silent
#: fallback.
ANSWER_SCORING: dict[str, Any] = {
    "primary": "the benchmark's own metric, computed by its own script where one ships",
    "secondary": "local exact-match and token-F1 under ANSWER_EQUIVALENCE",
    "citation": "ALCE-style citation precision/recall against the CITED atom's provenance",
    "fixed_before": "any end-to-end run",
}

# --------------------------------------------------------------------------
# decision 7 — abstention aggregation (G7, inherited from Phase 9 §7.8)
# --------------------------------------------------------------------------

#: `PHASE9_DECISIONS.md` §7.8 left this to Phase 10's runner *"so the runner's
#: author inherits them as decisions rather than discovering them as bugs"*.
#:
#: **Excluded from utility means, counted separately, cause split reported.**
#: The two wrong answers are instructive: imputing 0 makes an abstaining system
#: look like a wrong-answering one, and dropping the query makes abstention free.
#: Both are reported instead — mean over *answered* queries, plus the abstention
#: rate with its cause split, which is the split Phase 8 reserved
#: ``ABSTAIN_CAUSES`` for and which must never be summed into one rate
#: (`PHASE5_DECISIONS.md` §1's quarantine-cause lesson, two phases on).
ABSTENTION_AGGREGATION: dict[str, str] = {
    "utility_mean": "over answered queries only; abstentions excluded, never imputed as 0",
    "abstention_rate": "reported beside it, split by cause (gate vs fallback)",
    "never": "sum the two causes into one rate",
}

# --------------------------------------------------------------------------
# decision 8 — the contested comparison (G8, transferred from Phase 9)
# --------------------------------------------------------------------------

#: Fix F4's inference-time half.  Phase 9 implemented the gold-bearing
#: diagnostic and transferred this here by name, because at deployment there is
#: no gold and the architecture's own words are *"costs one comparison"* — a
#: reader-level check.
#:
#: **It is costed.**  One extra reader call per contested-eligible query,
#: ledgered and reported as its own latency and token line.  Folding it into the
#: base cost would understate the read path against Phase 11's baselines on the
#: one axis `CLAUDE.md` §9 says this project must win.
CONTESTED: dict[str, Any] = {
    "eligible_when": "the portfolio returns >= 2 distinct valid sets",
    "compare": "one extra reader call on the runner-up set",
    "outcome": "contested when the two answers differ under ANSWER_EQUIVALENCE",
    "cost": "ledgered separately; never folded into the base per-query cost",
}

# --------------------------------------------------------------------------
# decision 11 — post-hoc verification: DECLINED (G11)
# --------------------------------------------------------------------------

#: **[EVIDENCE]** SynCheck (EMNLP 2024) detects unfaithful sentences at
#: **> 0.85 AUROC** across six long-form RAG tasks, and its FOD decoding improves
#: faithfulness by **> 10%**.  It is a real result and this is not a dismissal.
#:
#: **Declined for Phase 10 on three grounds, recorded so it can be reversed
#: knowingly.**  It needs a **trained monitor** — the only trainable parameter
#: this phase would contain, against the supervisor's constraint that the SLM
#: stays frozen and learning lives in the GNN/NN stack.  **FOD alters decoding**,
#: which breaks the byte-identical ``(prompt, decoding-config)`` hash that makes
#: v1.2 §3.5 enforceable rather than promised.  And it would have to be given to
#: **every Phase-11 baseline** too, or the comparison is unmatched.
#:
#: `CLAUDE.md` §5 already lists *"SynCheck described as free"* among this
#: project's caught errors.  This is that finding acted on rather than restated:
#: v1.2 §3.5's terms — "part of the inference algorithm, same compute budget,
#: latency reported" — are affordable to state and were not affordable to meet
#: here.
POST_HOC_VERIFICATION: dict[str, Any] = {
    "adopted": False,
    "candidate": "SynCheck / FOD (EMNLP 2024)",
    "grounds": (
        "needs a trained monitor (frozen-SLM constraint)",
        "FOD alters decoding (breaks the decoding-config hash)",
        "would have to be matched across every Phase-11 baseline",
    ),
    "reversible": "yes, but reversing re-opens the frozen-decoding hash and the matched budget",
}

# --------------------------------------------------------------------------
# decision 5 — the five ceilings and their tiers (G5)
# --------------------------------------------------------------------------

#: v1.2 §6.3's five, each a single function.  **Every one reports the tier it was
#: computed at**, because ceilings 1 and 2 need gold proof annotation on
#: conversation that does not exist (`CLAUDE.md` §7's binding constraint on
#: Contribution 1), so they run against the Tier-A/Tier-B definitions Phase 7
#: already froze rather than a new annotation.
#:
#: A ceiling computed at Tier A is a **weaker statement** than one at Tier B, and
#: a number that does not say which is uninterpretable — the discipline
#: `PHASE9_DECISIONS.md` applies to "H-valid" on Wikipedia pools, here.
#:
#: **[EVIDENCE, provisional]** ceiling 4's precedent is arXiv 2607.00725's
#: "answer-in-context", which predicted answer F1 far better than retrieval
#: recall (ΔR² = +0.17; F1 0.61 vs 0.20 conditional on it) **including among
#: questions where retrieval was perfect** — which is precisely why a recall
#: number alone cannot stand in for it.
CEILINGS: dict[int, dict[str, str]] = {
    1: {"name": "extraction", "asks": "are gold statements grounded assertions?", "tier": "declared per run"},
    2: {"name": "graph", "asks": "does the graph contain a sufficient proof?", "tier": "declared per run"},
    3: {"name": "candidate", "asks": "does Stage C retrieve every atom of some sufficient proof?",
        "tier": "delegates to retrieve.recall, inherits its saturation guard"},
    4: {"name": "packing", "asks": "does a sufficient proof survive the token budget?",
        "tier": "reported at every rung of BUDGET_LADDER"},
    5: {"name": "reader", "asks": "what does the frozen SLM achieve given the gold proof?",
        "tier": "n/a — needs no graph"},
}


def frozen_values() -> dict[str, Any]:
    """Everything that must agree across machines for two Stage-E numbers to compare.

    ``serialization_budget_tokens`` is **absent**: it is the config tree's and
    reaches identity through ``config_hash``.  The *ladder* is here because it is
    a reporting rule rather than a run parameter.
    """
    return {
        "reader": READER,
        "prompt_sha": PROMPT_SHA,
        "decoding": {k: v for k, v in sorted(DECODING.items())},
        "claim_id_format": CLAIM_ID_FORMAT,
        "ordering": {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(ORDERING.items())},
        "budget_ladder": list(BUDGET_LADDER),
        "answer_equivalence": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in sorted(ANSWER_EQUIVALENCE.items())
        },
        "answer_scoring": dict(sorted(ANSWER_SCORING.items())),
        "abstention_aggregation": dict(sorted(ABSTENTION_AGGREGATION.items())),
        "contested": dict(sorted(CONTESTED.items())),
        "post_hoc_verification": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in sorted(POST_HOC_VERIFICATION.items())
        },
        "ceilings": {str(k): dict(sorted(v.items())) for k, v in sorted(CEILINGS.items())},
    }


def stage_e_fingerprint(length: int | None = None) -> str:
    """Configuration identity for Stage E, printed by ``verify_handoff.py``.

    Binds the **config, not the output** — the same G11 distinction Phases 5–9
    drew.  Two machines will not produce byte-identical generations from a 3B
    model under every driver; they must produce them from an identical setup,
    and they must serialise, order, cite and score with identical arithmetic, or
    two end-to-end numbers are not comparable.
    """
    return digest_of(frozen_values(), length)
