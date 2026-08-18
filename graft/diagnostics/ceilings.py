"""P10.5 — the five ceilings, one function each (v1.2 §6.3).

**[ANALYSIS]** *"Without this decomposition, a bad end-to-end number is
uninterpretable — it could be extraction, linking, retrieval, packing or the
reader. With it, every result points at a specific stage. This is cheap to
compute and is a genuine methodological contribution."*  That is Contribution 4,
and this module is where it stops being a design and becomes runnable.

**Three of the five already existed and are delegated, not reimplemented.**
Phase 7 built ``retrieve.recall.ceilings()`` (1 and 2) and ``recall_of`` /
``saturation`` (3) because it needed them to report an honest recall.  A second
implementation here would be a second chance to disagree with the first, and
`CLAUDE.md` §5's standing pattern is that a correction lands in three places out
of four.  What Phase 10 adds is **ceiling 4** (packing) and **ceiling 5**
(reader).

**Ceiling 3 carries Phase 7's saturation guard with it, and that is
load-bearing.**  `PHASE7_DECISIONS.md` §3.1 measured Tier-A recall at 1.000 on
9 of 10 pilot questions **by arithmetic** — every conversation held 8–23
eligible candidates against ``pool_cap = 64``, so the pool *was* the
conversation and nothing was ranked.  A ceiling-3 number without its saturation
flag is that artefact reported as a result.

**Every ceiling reports the tier it was computed at.**  Ceilings 1 and 2 need
gold proof annotation on conversation, which `CLAUDE.md` §7 records as the
binding constraint on Contribution 1 and which does not exist; they therefore
run against the Tier-A / Tier-B definitions Phase 7 froze.  A ceiling computed
at Tier A is a **weaker statement** than one at Tier B, and a number that does
not say which is uninterpretable — the discipline `PHASE9_DECISIONS.md` applies
to "H-valid" on Wikipedia pools, here.

**[EVIDENCE, provisional] Ceiling 4's precedent.**  arXiv 2607.00725 introduced
"answer-in-context" — whether the gold answer survives into the packed reader
context — and found it predicts answer F1 far better than retrieval recall
(ΔR² = +0.17 over recall; F1 0.61 vs 0.20 conditional on it), **including among
questions where retrieval was perfect**.  That last clause is why ceiling 3
cannot stand in for ceiling 4: perfect retrieval and a failed pack look
identical from upstream.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

from graft.config import Config
from graft.reader.pins import BUDGET_LADDER
from graft.schemas import Obligations, ProofSet

__all__ = [
    "ceiling_1_extraction",
    "ceiling_2_graph",
    "ceiling_3_candidate",
    "ceiling_4_packing",
    "ceiling_5_reader",
    "all_ceilings",
]


def _tiered(payload: dict[str, Any], tier: str) -> dict[str, Any]:
    """Stamp the tier onto a ceiling result.

    Not optional and not a convenience: the whole point of §6 decision 5 is that
    a number without its tier cannot be read, so the stamp is applied at the one
    place every ceiling returns through.
    """
    payload["tier"] = tier
    return payload


def ceiling_1_extraction(snapshot: Any, conv_id: str | None = None, *, tier: str = "gate") -> dict[str, Any]:
    """Are gold statements represented as correctly grounded assertions?

    Delegates to ``retrieve.recall.ceilings``, which computes it from the
    snapshot rather than quoting the Phase-5 report — so the number describes the
    graph it was measured on rather than a run that may have been re-ingested
    since.

    The default tier is ``"gate"``: this is the *support-gate survival rate*, not
    a comparison against annotated gold statements, because that annotation does
    not exist on conversation. It is the honest reading of what the snapshot can
    answer, and it is weaker than v1.2 §6.3's question.
    """
    from graft.retrieve import recall

    base = recall.ceilings(snapshot, conv_id)
    if not base.get("available"):
        return _tiered({"available": False, "ceiling": None}, tier)
    return _tiered(
        {
            "available": True,
            "ceiling": base["ceiling_1_extraction"],
            "assertions_total": base["assertions_total"],
            "assertions_eligible": base["assertions_eligible"],
            "note": (
                "support-gate survival rate; NOT a comparison against annotated "
                "gold statements, which do not exist on conversation "
                "(CLAUDE.md §7). Weaker than v1.2 §6.3's question."
            ),
        },
        tier,
    )


def ceiling_2_graph(snapshot: Any, conv_id: str | None = None, *, tier: str = "gate") -> dict[str, Any]:
    """Does the constructed graph contain a sufficient proof?

    Delegated for the same reason as ceiling 1, and carrying the same caveat: as
    computed it is the fraction of *eligible* assertions that reached an
    assertion-backed node — graph-construction survival — rather than a search
    for a sufficient proof, which needs the gold proof annotation Gate 0 funds
    only partially.
    """
    from graft.retrieve import recall

    base = recall.ceilings(snapshot, conv_id)
    if not base.get("available"):
        return _tiered({"available": False, "ceiling": None}, tier)
    return _tiered(
        {
            "available": True,
            "ceiling": base["ceiling_2_graph"],
            "eligible_on_graph": base["eligible_on_graph"],
            "nodes_in_scope": base["nodes_in_scope"],
            "note": (
                "graph-construction survival of eligible assertions; NOT an "
                "oracle search for a sufficient proof."
            ),
        },
        tier,
    )


def ceiling_3_candidate(
    retrieved: Iterable[str],
    gold: Iterable[str],
    *,
    snapshot: Any = None,
    conv_id: str | None = None,
    config: Config | None = None,
    tier: str = "tier_a",
) -> dict[str, Any]:
    """Does Stage C retrieve every atom needed by some sufficient proof?

    Delegates to ``recall.recall_of`` **and carries ``recall.saturation``**
    whenever a snapshot is supplied.  The guard is not decoration: without it a
    recall of 1.000 on a pool that admitted the entire conversation reads as a
    retrieval result rather than as arithmetic (`PHASE7_DECISIONS.md` §3.1).
    """
    from graft.retrieve import recall

    cfg = config or Config()
    payload = dict(recall.recall_of(retrieved, gold))
    payload["ceiling"] = payload.pop("recall")
    # `recall_of` returns None for empty gold rather than 1.0 -- a question with
    # nothing to recall must not inflate a pooled average. Surface that as
    # `available` so the five-ceiling table reads uniformly.
    payload["available"] = payload["ceiling"] is not None
    if snapshot is not None:
        payload["saturation"] = recall.saturation(snapshot, conv_id, cfg.pool_cap)
    return _tiered(payload, tier)


def ceiling_4_packing(
    gold: Iterable[str],
    serializer: Any,
    obligations: Obligations,
    scores: Mapping[str, float] | None = None,
    *,
    ladder: Sequence[int] = BUDGET_LADDER,
    tier: str = "tier_a",
) -> dict[str, Any]:
    """Does a sufficient proof survive the token budget and serialization?

    **Reported at every rung of the ladder, never one.**  `CLAUDE.md` §8 records
    that quoting a packing result at a single cherry-picked budget was one of
    this project's own caught errors — the submodular packer wins at 160 tokens
    and does not at the other three budgets in the same table — and a
    single-budget ceiling 4 is that mistake reproduced in this project's own
    instrument.

    ``survives`` is the ceiling proper: the gold proof went in and came out
    whole.  ``dropped`` says how badly it failed when it did, which is what
    separates "one atom over" from "the budget is hopeless here".
    """
    gold_set = frozenset(gold)
    if not gold_set:
        return _tiered({"available": False, "ceiling": None, "by_budget": {}}, tier)

    proof = ProofSet(atoms=gold_set)
    by_budget: dict[str, Any] = {}
    for budget in ladder:
        out = serializer.serialise(proof, obligations, scores or {}, budget=int(budget))
        by_budget[str(budget)] = {
            "survives": out.complete,
            "included": len(out.included),
            "dropped": len(out.dropped),
            "tokens": out.tokens,
            "approximate_tokens": out.counter != "reader_tokenizer",
        }
    survived = [b for b, r in by_budget.items() if r["survives"]]
    return _tiered(
        {
            "available": True,
            # The ceiling at the *live* budget is the headline; the ladder is
            # what stops it being quoted without its condition.
            "ceiling": by_budget[str(int(ladder[len(ladder) // 2]))]["survives"]
            if len(ladder) else None,
            "gold_atoms": len(gold_set),
            "by_budget": by_budget,
            "survives_at": sorted(int(b) for b in survived),
        },
        tier,
    )


def ceiling_5_reader(
    gold: Iterable[str],
    serializer: Any,
    obligations: Obligations,
    question: str,
    read_fn: Callable[[str, str], str],
    *,
    gold_answer: str = "",
    aliases: Sequence[str] = (),
    scores: Mapping[str, float] | None = None,
    tier: str = "tier_a",
) -> dict[str, Any]:
    """What does the frozen SLM achieve when handed the gold proof?

    **The ceiling every other number is bounded by.**  If the reader cannot
    answer from a *perfect* evidence set, no amount of retrieval or set
    construction upstream can fix it — which is the whole reason v1.2 §6.3 puts
    it last and separate.

    ``read_fn`` is injected rather than imported so this module stays free of
    torch and testable without weights.  It takes ``(evidence_text, question)``
    and returns the raw generation — the same surface ``reader.read`` exposes.
    """
    from graft.reader.parse import answers_equivalent, parse_answer, token_f1

    gold_set = frozenset(gold)
    if not gold_set:
        return _tiered({"available": False, "ceiling": None}, tier)

    # **Serialised WITHOUT a binding budget** — corrected 16 Aug 2026 after an
    # adversarial audit. Ceiling 5 asks *"what does the frozen reader achieve when
    # handed a gold proof"*, and serialising at the live budget meant a gold proof
    # that did not fit was scored as a **reader** failure when it was a **packing**
    # failure. That is precisely the conflation the five-ceiling protocol exists to
    # prevent: v1.2 §6.3's whole argument is that one number is uninterpretable
    # because it could be any of five stages, and a ceiling that absorbs the one
    # below it re-creates the problem inside the instrument.
    #
    # Ceiling 4 is where the budget question belongs, and it reports at every rung.
    # The packing status is still carried here so the two remain relatable — but as
    # a *separate field*, never folded into the ceiling value.
    packed = serializer.serialise(
        ProofSet(atoms=gold_set), obligations, scores or {}, budget=10**9
    )
    budgeted = serializer.serialise(ProofSet(atoms=gold_set), obligations, scores or {})
    generation = read_fn(packed.text, question)
    parsed = parse_answer(generation)

    correct = (
        answers_equivalent(parsed.answer_text, gold_answer, aliases)
        if gold_answer
        else None
    )
    return _tiered(
        {
            "available": True,
            "ceiling": correct,
            "f1": token_f1(parsed.answer_text, gold_answer) if gold_answer else None,
            "abstained": parsed.abstained,
            "citations": len(parsed.citations),
            "packed": packed.report(),
            "answer_text": parsed.answer_text,
            # Reported beside, never inside: this says whether ceiling 4 would
            # ALSO have failed at the live budget, so the two ceilings stay
            # separable rather than one silently absorbing the other.
            "survives_live_budget": budgeted.complete,
            "live_budget": budgeted.budget,
            "note": (
                "the reader was handed the GOLD proof at an UNBOUNDED budget, so "
                "this measures the reader alone; `survives_live_budget` reports "
                "ceiling 4's separate question."
            ),
        },
        tier,
    )


def all_ceilings(
    *,
    snapshot: Any = None,
    conv_id: str | None = None,
    retrieved: Iterable[str] = (),
    gold: Iterable[str] = (),
    serializer: Any = None,
    obligations: Obligations | None = None,
    scores: Mapping[str, float] | None = None,
    question: str = "",
    read_fn: Callable[[str, str], str] | None = None,
    gold_answer: str = "",
    aliases: Sequence[str] = (),
    config: Config | None = None,
    tier: str = "tier_a",
) -> dict[str, Any]:
    """All five, with the ones that could not run saying so.

    A ceiling that could not be computed returns ``available: False`` rather than
    being omitted: a five-ceiling table with four rows invites the reader to
    assume the fifth was fine, and the decomposition's entire value is that it
    attributes failure.
    """
    out: dict[str, Any] = {}
    out["1_extraction"] = (
        ceiling_1_extraction(snapshot, conv_id) if snapshot is not None
        else {"available": False, "ceiling": None, "reason": "no snapshot"}
    )
    out["2_graph"] = (
        ceiling_2_graph(snapshot, conv_id) if snapshot is not None
        else {"available": False, "ceiling": None, "reason": "no snapshot"}
    )
    out["3_candidate"] = ceiling_3_candidate(
        retrieved, gold, snapshot=snapshot, conv_id=conv_id, config=config, tier=tier
    )
    out["4_packing"] = (
        ceiling_4_packing(gold, serializer, obligations, scores, tier=tier)
        if serializer is not None and obligations is not None
        else {"available": False, "ceiling": None, "reason": "no serializer"}
    )
    out["5_reader"] = (
        ceiling_5_reader(
            gold, serializer, obligations, question, read_fn,
            gold_answer=gold_answer, aliases=aliases, scores=scores, tier=tier,
        )
        if read_fn is not None and serializer is not None and obligations is not None
        else {"available": False, "ceiling": None, "reason": "no reader"}
    )
    return out
