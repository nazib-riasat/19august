"""P7.7 — the two-tier recall instrument and the ablation loop (G1, G7, G9).

**This is the only module in ``graft.retrieve`` that may read a gold field.**
``has_answer`` and ``answer_session_ids`` enter here, through the sidecar
``graft.ingest.corpus.question_meta`` already quarantines, and nowhere else.  A
channel that could see them would make ceiling 3 a fiction — it would retrieve
the answer because it was told which turn the answer is in — and the claim
"retrieval recall" would be measuring the instrument.  ``test_structure`` asserts
the quarantine structurally rather than trusting this docstring.

**Two tiers, both built now, because Gate 0 is unsigned.**

*Tier A — runs today.*  Gold is the pool-representable atom set of eligible
assertions grounded in ``has_answer`` turns, closed under refs.  No `H`, no
minimisation.  This is an **over-estimate** of what a proof actually requires, so
recall against it is **conservative**: a pool that covers the superset covers
every minimal proof inside it.  It is ceiling 3's measurable proxy and is
labelled as the proxy it is, everywhere it is reported.

*Tier B — one flag, post-Gate-0.*  ``GATE0_CONTRACT.md`` item 3's full
definition: the same set, minimised by removing any atom whose deletion keeps `H`
true.  :func:`tier_b_gold` **refuses by name** until the contract signs — exit
criterion 4 — rather than silently returning Tier A, which would put two
different meanings behind one number in two different documents.

The instrument takes the gold set as a **callable**, so Tier B is a gold-set swap
and not a rewrite.

**Gold is built through the same ``build_pool`` the channels feed.**  If gold were
assembled by a second, parallel mapping, the two could disagree about what is
"pool-representable" and recall would silently measure the disagreement.  One
mapping, used twice.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from graft.retrieve.pool import build_pool, eligible_nodes, node_atom_id
from graft.schemas import ASSERTION_BACKED_NTYPES, PAYLOAD_ASSERTION_ID

__all__ = [
    "HONESTY_STAMP",
    "has_answer_turns",
    "gold_nodes",
    "tier_a_gold",
    "tier_b_gold",
    "ceilings",
    "recall_of",
    "channel_table",
    "saturation",
]

#: G9's three bounds, carried in every artefact this instrument produces.
#: Written down **before** any recall number existed, so that none of them can be
#: discovered as an excuse after an unflattering one arrives.
HONESTY_STAMP: dict[str, str] = {
    "graph": (
        "the pilot graph is stand-in-constructed (PHASE6_DECISIONS.md 2.1): links "
        "are a deterministic rule, not a trained D1. These numbers exercise the "
        "machinery; they do not measure Stage C's quality on a real graph."
    ),
    "questions": (
        "the pilot has 10 questions; the decisive corpus is Gate-0 item 9's scope "
        "decision. Every number is per-question-listed and is never presented as a "
        "distribution over 10 points."
    ),
    "ceilings": (
        "ceiling 1 already took 55% on the pilot (333 extracted -> 151 eligible). "
        "Ceiling 3 is reported beside ceilings 1-2 or a retrieval failure will be "
        "misread into a stage that did not cause it."
    ),
    "tier": (
        "Tier A gold is the has_answer-derived closed superset, not the H-minimised "
        "proof set. Recall against it is conservative and is a proxy for ceiling 3, "
        "labelled as such. Tier B refuses until GATE0_CONTRACT.md signs."
    ),
}


def has_answer_turns(instance: Mapping[str, Any]) -> frozenset[str]:
    """Turn ids LongMemEval marks ``has_answer``, for one raw corpus instance.

    **The gold sidecar, and the reason this module is the quarantine boundary.**
    The corpus ships this per turn — which is worth stating, because
    ``GATE0_CONTRACT.md`` item 2 justifies Stage C's coarse *training* signal by
    saying turn-level relevance would have to be annotated, and it would not
    (`DATASET_DECISION.md` §7.4).  The real reason to train on the coarse signal
    stands and is different: ``has_answer`` **defines the evaluation target
    here**, so training on it would collapse Tier-A recall into a self-fulfilling
    number.

    Reads the raw instance rather than ``question_meta`` because the latter keeps
    only ``evidence_session_ids`` — session granularity, where this needs turn.
    """
    from graft.ingest.corpus import turn_id_for

    question_id = str(instance["question_id"])
    out: list[str] = []
    session_ids = list(instance["haystack_session_ids"])
    for session_id, session in zip(session_ids, instance["haystack_sessions"]):
        for turn_ix, turn in enumerate(session):
            if turn.get("has_answer"):
                out.append(turn_id_for(question_id, str(session_id), turn_ix))
    return frozenset(out)


def gold_nodes(snapshot: Any, gold_turns: Iterable[str], conv_id: str | None = None) -> tuple[str, ...]:
    """Eligible assertion-backed nodes with at least one span in a gold turn.

    "At least one", not "all": a claim assembled across turns records every
    supporting span (``Assertion``'s docstring), and requiring every span to land
    in a ``has_answer`` turn would drop exactly the multi-turn claims that make
    conversational memory hard.
    """
    wanted = frozenset(gold_turns)
    if not wanted:
        return ()
    out: list[str] = []
    for node_id in eligible_nodes(snapshot, conv_id):
        node = snapshot.node(node_id)
        if node is None or node.ntype not in ASSERTION_BACKED_NTYPES:
            continue
        assertion = snapshot.assertion(str(node.payload.get(PAYLOAD_ASSERTION_ID) or ""))
        if assertion is None:
            continue
        for span_id in assertion.spans:
            span = snapshot.span(span_id)
            if span is not None and span.turn_id in wanted:
                out.append(node_id)
                break
    return tuple(sorted(out))


def tier_a_gold(
    snapshot: Any,
    gold_turns: Iterable[str],
    conv_id: str | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Tier-A gold: ``(closed_atoms, reachable_node_atoms)``.

    Two sets, because two different questions get asked of them and conflating
    them would make a channel look worse than it is:

    * ``closed_atoms`` — the whole closed set, including the structural atoms
      that arrive by closure.  This is the denominator for **pool** recall.
    * ``reachable_node_atoms`` — only the assertion-backed node atoms.  This is
      the denominator for **per-channel** recall, because no channel emits an
      edge atom: they are pulled in by assembly.  Scoring a channel against the
      closed set would charge it for atoms it structurally cannot return.

    The cap is deliberately far above ``pool_cap``: gold is what the pool *should*
    contain, and truncating it to 64 would make recall unmeasurable above the cap
    by construction.
    """
    nodes = gold_nodes(snapshot, gold_turns, conv_id)
    if not nodes:
        return frozenset(), frozenset()
    pool, _, _ = build_pool(snapshot, {n: 1.0 for n in nodes}, cap=max(1, 64 * len(nodes) + 64))
    return frozenset(pool.ids()), frozenset(node_atom_id(n) for n in nodes)


def tier_b_gold(*_args: Any, **_kwargs: Any) -> frozenset[str]:
    """Tier-B gold: `H`-minimised sufficient-proof sets.  **Refuses until Gate 0 signs.**

    Exit criterion 4 asks for a refusal *by name*, not a fallback.  Returning
    Tier A here would put "sufficient-proof recall" and "``has_answer``-derived
    superset recall" behind one number, and the two would then be quoted
    interchangeably across documents — the failure `GATE0_CONTRACT.md` item 8's
    own decision row calls "recall numbers that mean different things in
    different documents".
    """
    raise NotImplementedError(
        "Tier-B (H-minimised sufficient-proof) gold requires a signed "
        "GATE0_CONTRACT.md: item 3 defines the minimisation and item 9's corpus "
        "scope is still open. Use tier_a_gold and label it the conservative "
        "has_answer-derived proxy it is."
    )


def ceilings(snapshot: Any, conv_id: str | None = None) -> dict[str, Any]:
    """Ceilings 1 and 2 from the snapshot, so ceiling 3 is never reported alone (G9).

    Ceiling 1 is extraction: what fraction of stored assertions survived the
    support gate.  Ceiling 2 is graph construction: what fraction of *eligible*
    assertions actually reached an assertion-backed node.  Both are computed from
    the snapshot rather than quoted from the Phase-5 report, so they describe the
    graph this recall number was measured on.
    """
    assertions = getattr(snapshot, "_assertions", None)
    if not isinstance(assertions, dict):
        return {"available": False}
    total = len(assertions)
    eligible = [a for a in assertions.values() if a.eligibility == "eligible"]
    node_backed = {
        str(n.payload.get(PAYLOAD_ASSERTION_ID))
        for n in getattr(snapshot, "_nodes", {}).values()
        if n.ntype in ASSERTION_BACKED_NTYPES and n.payload.get(PAYLOAD_ASSERTION_ID)
    }
    on_graph = sum(1 for a in eligible if a.assertion_id in node_backed)
    return {
        "available": True,
        "assertions_total": total,
        "assertions_eligible": len(eligible),
        "ceiling_1_extraction": (len(eligible) / total) if total else 0.0,
        "eligible_on_graph": on_graph,
        "ceiling_2_graph": (on_graph / len(eligible)) if eligible else 0.0,
        "nodes_in_scope": len(eligible_nodes(snapshot, conv_id)),
    }


def saturation(snapshot: Any, conv_id: str | None, cap: int) -> dict[str, Any]:
    """Is there anything for retrieval to *do* on this question?

    **Measured 15 Aug 2026 on the pilot graph, and the answer was no.**  Every
    conversation holds 8–23 eligible assertion-backed nodes against
    ``pool_cap = 64``, so the pool admits the entire conversation and Tier-A
    recall is 1.000 for every question — by arithmetic, not by retrieval.  Nothing
    is ranked, nothing is excluded, and a channel that returned its input
    unchanged would score identically.

    **This is Phase 4's G9 arriving one stage later** and it is recorded the same
    way: there, greedy on exact `U` was globally optimal on 30/30 lattice
    instances, so best-of-K was arithmetic and the gate's rule had to move before
    it narrowed a claim on an artefact.  Here, a recall of 1.000 says only that
    the candidate set is smaller than the cap.

    It is also the measured form of `DATASET_DECISION.md` §5's argument for scope
    b′: *evidence-only sessions make retrieval artificially easy — Stage C's
    recall is measured against a haystack that contains almost nothing but the
    answer, and ceiling 3 becomes uninformative.*  Distractor sessions are what
    make ``candidates > cap`` true, and until it is true this instrument is
    measuring its own plumbing.

    Returned per question so that no recall number can be quoted without the flag
    that says whether it meant anything.
    """
    candidates = len(eligible_nodes(snapshot, conv_id))
    return {
        "candidates_in_scope": candidates,
        "pool_cap": int(cap),
        "exercised": candidates > int(cap),
        "reading": (
            "candidates <= pool_cap: the pool is the whole conversation, so recall "
            "is 1.0 by construction and this question does not test retrieval"
            if candidates <= int(cap)
            else "candidates > pool_cap: the cap binds and retrieval selects"
        ),
    }


def recall_of(retrieved: Iterable[str], gold: Iterable[str]) -> dict[str, Any]:
    """``|retrieved ∩ gold| / |gold|``, with the counts that make it auditable.

    Empty gold returns ``None`` rather than 1.0.  A question whose gold set is
    empty has nothing to recall, and scoring it a perfect 1.0 would inflate every
    pooled average by the questions the instrument could not measure — which on a
    10-question pilot is the difference between a number and an artefact.
    """
    got, want = set(retrieved), set(gold)
    hit = got & want
    return {
        "recall": (len(hit) / len(want)) if want else None,
        "gold": len(want),
        "retrieved": len(got),
        "hit": len(hit),
        "missed": sorted(want - got),
    }


def channel_table(
    channels: Mapping[str, Mapping[str, float]],
    reachable_gold: Iterable[str],
    *,
    latency_ms: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """G7's per-channel row: recall at the channel's own k, plus its latency.

    Channels emit ``node_id`` (see the package docstring); they are mapped to
    node-atom ids here, once, so no channel has to know that atom ids exist.
    """
    want = set(reachable_gold)
    timings = dict(latency_ms or {})
    out: dict[str, Any] = {}
    for name in sorted(channels):
        atoms = {node_atom_id(n) for n in channels[name]}
        row = recall_of(atoms, want)
        row["latency_ms"] = round(float(timings[name]), 3) if name in timings else None
        out[name] = row
    return out
