"""P7.7 — the two-tier recall instrument and the ablation loop (G1, G7, G9).

**This is the only module in ``graft.retrieve`` that may read a gold field.**
``has_answer`` and ``answer_session_ids`` enter here, through the sidecar
``graft.ingest.corpus.question_meta`` already quarantines, and nowhere else.  A
channel that could see them would make ceiling 3 a fiction — it would retrieve
the answer because it was told which turn the answer is in — and the claim
"retrieval recall" would be measuring the instrument.  ``test_structure`` asserts
the quarantine structurally rather than trusting this docstring.

**Two tiers, both built and both live since the 15 Aug 2026 Gate-0 signature.**

*Tier A — runs today.*  Gold is the pool-representable atom set of eligible
assertions grounded in ``has_answer`` turns, closed under refs.  No `H`, no
minimisation.  This is an **over-estimate** of what a proof actually requires, so
recall against it is **conservative**: a pool that covers the superset covers
every minimal proof inside it.  It is ceiling 3's measurable proxy and is
labelled as the proxy it is, everywhere it is reported.

*Tier B — built 15 August 2026, once Gate 0 signed.*  ``GATE0_CONTRACT.md``
item 3.2's full definition: the same set, minimised by removing any atom whose
deletion keeps `H` true.  Until the signature :func:`tier_b_gold` **refused by
name** rather than falling back to Tier A — exit criterion 4 — because a
fallback would have put two different meanings behind one number in two
different documents.  It now returns a **3-tuple** where Tier A returns a
2-tuple, so the two cannot be swapped by accident even though the refusal is
gone.

**Tier B is not strictly better than Tier A, and both are kept.**  Tier A is an
over-estimate, so recall against it is *conservative* — a pool covering the
superset covers every minimal proof inside it.  Tier B is the plan's §2.4
primary, and it is **narrower in a way that can be wrong**: `H`-minimisation
finds a *locally* minimal set, one of possibly several, so an atom missing from
it is not necessarily an atom a correct proof could do without.  Reporting both
is what makes that visible rather than a matter of which number someone quoted.

The instrument takes the gold set as a **callable**, so switching tiers is a
gold-set swap and not a rewrite.

**Gold is built through the same ``build_pool`` the channels feed.**  If gold were
assembled by a second, parallel mapping, the two could disagree about what is
"pool-representable" and recall would silently measure the disagreement.  One
mapping, used twice.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

from graft.config.schema import Config
from graft.retrieve.pool import eligible_nodes, node_atom_id, uncapped_pool
from graft.schemas import ASSERTION_BACKED_NTYPES, PAYLOAD_ASSERTION_ID, Obligations

__all__ = [
    "HONESTY_STAMP",
    "has_answer_turns",
    "gold_nodes",
    "required_sets",
    "tier_a_gold",
    "tier_b_gold",
    "minimise",
    "ceilings",
    "recall_of",
    "channel_table",
    "saturation",
    "evidence_turns",
    "distant_labels",
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
        "the pilot corpus is small (this artefact's `questions` key is the count); "
        "the decisive corpus is Gate-0 item 9's scope-c decision, not yet ingested. "
        "Every number is per-question-listed and is never presented as a "
        "distribution."
    ),
    "ceilings": (
        "ceiling 1 takes its toll before retrieval begins -- this artefact's own "
        "`ceilings` block carries the measured values for the graph it ran on. "
        "Ceiling 3 is reported beside ceilings 1-2 or a retrieval failure will be "
        "misread into a stage that did not cause it."
    ),
    "tier": (
        "Tier A gold is the has_answer-derived closed superset, not the H-minimised "
        "proof set. Recall against it is conservative and is a proxy for ceiling 3, "
        "labelled as such. Tier B is the H-minimised set under contract item 3.2 as "
        "amended 15 Aug 2026 (structural-only removal, size-exempt gold); its "
        "degenerate/under_constrained flags travel with every Tier-B number."
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


def evidence_turns(instance: Mapping[str, Any]) -> frozenset[str]:
    """Every turn id in the question's **evidence sessions** — the coarse signal.

    Distinct from :func:`has_answer_turns`, and the distinction is the whole of
    ``GATE0_CONTRACT.md`` item 2: Stage C **trains** on this session-level signal
    and is **evaluated** on the turn-level one.  An evidence session contains
    turns that carry no part of the answer, so this is noisy by construction —
    stated as a limitation rather than repaired, because repairing it means
    turn-level relevance annotation and item 8's budget is spent on D1/D2.

    **The real reason not to train on ``has_answer``** (correcting item 2's
    stated reason, per `DATASET_DECISION.md` §7.4): the corpus *does* ship
    ``has_answer`` per turn, so no annotation is needed. But ``has_answer``
    defines the Tier-A **evaluation** target, so training on it would collapse
    recall into a self-fulfilling number.
    """
    from graft.ingest.corpus import turn_id_for

    question_id = str(instance["question_id"])
    wanted = {str(s) for s in instance.get("answer_session_ids", ())}
    out: list[str] = []
    for session_id, session in zip(instance["haystack_session_ids"], instance["haystack_sessions"]):
        if str(session_id) not in wanted:
            continue
        for turn_ix, _turn in enumerate(session):
            out.append(turn_id_for(question_id, str(session_id), turn_ix))
    return frozenset(out)


def distant_labels(
    snapshot: Any,
    pool: Any,
    instance: Mapping[str, Any],
    conv_id: str | None = None,
) -> list[float]:
    """Item-2 training labels for one pool: ``1.0`` iff the atom derives from an evidence session.

    **This function is why ``scorer.py`` contains no gold field.**  The distant
    signal needs ``answer_session_ids``, which is gold; the structural test
    quarantines gold to this module; so the derivation lives here and the scorer
    receives plain numbers it cannot trace back. Moving it into the scorer would
    break that test, correctly.

    Returned in ``pool.ids()`` order — the same order
    :func:`graft.retrieve.scorer.atom_features` builds its rows in — because a
    label list and a feature matrix that disagree about row *i* is an error
    nothing downstream could see.

    Structural atoms (entities, intervals, the edges among them) label ``0.0``:
    they are support pulled in by closure, not evidence the signal is about.
    """
    relevant = set(gold_nodes(snapshot, evidence_turns(instance), conv_id))
    return [1.0 if pool[aid].target in relevant else 0.0 for aid in pool.ids()]


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


def required_sets(
    snapshot: Any,
    gold_turns: Iterable[str],
    conv_id: str | None = None,
) -> dict[str, frozenset[str]]:
    """The gold atom sets, split by what a metric may be asked of them.

    * ``closed`` — the whole closed set, structural atoms included: the
      denominator for **pool** recall (Tier A).
    * ``reachable`` — assertion-backed node atoms only: the denominator for
      **per-channel** recall, because no channel emits an edge atom.
    * ``node_atoms`` / ``edge_atoms`` — the kind split of ``closed``: the
      denominators for **required-node Recall@k** and **required-edge Recall@k**,
      the two metrics plan §3.3 and §6.4 name separately and which had no
      implementation before the 15 Aug 2026 audit (required-edge recall was
      absent entirely; atom ids are opaque hashes, so the split needs the pool
      object and is made here, once).

    Built through :func:`uncapped_pool` rather than a guessed finite cap: the old
    ``64 * len(nodes) + 64`` bound was not derived from the closure it bounded,
    and closure grows with the edges among the gold nodes — a silent truncation
    of the gold set would have flattered every method scored against it.
    """
    nodes = gold_nodes(snapshot, gold_turns, conv_id)
    if not nodes:
        return {
            "closed": frozenset(),
            "reachable": frozenset(),
            "node_atoms": frozenset(),
            "edge_atoms": frozenset(),
        }
    pool, _, _ = uncapped_pool(snapshot, {n: 1.0 for n in nodes}, conv_id=conv_id)
    return {
        "closed": frozenset(pool.ids()),
        "reachable": frozenset(node_atom_id(n) for n in nodes),
        "node_atoms": frozenset(a.atom_id for a in pool if a.kind == "node"),
        "edge_atoms": frozenset(a.atom_id for a in pool if a.kind == "edge"),
    }


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

    Delegates to :func:`required_sets` — one gold build, split once.
    """
    sets = required_sets(snapshot, gold_turns, conv_id)
    return sets["closed"], sets["reachable"]


def minimise(
    atoms: Iterable[str],
    pool: Any,
    obligations: Obligations,
    snapshot: Any,
    config: Config,
) -> tuple[frozenset[str], dict[str, Any]]:
    """Greedy `H`-minimisation: drop any atom whose deletion keeps `H` true.

    Item 3.2's operation, isolated so it is testable without a corpus.

    **Passes to a fixpoint, not one sweep.** Removing an edge atom can make its
    endpoint node atom removable in a later pass — sub-check 8 requires an atom's
    refs to be *selected*, so while the edge is present the node cannot go. A
    single sweep would therefore stop early and leave structural atoms behind,
    reporting a set that is not actually irreducible.

    **The result is *a* minimal proof, not *the* minimal proof.** Greedy
    single-atom removal reaches a **locally** minimal (irreducible) set; a
    different removal order can reach a different one, and neither is
    necessarily of minimum cardinality. That is not a defect to fix here — it is
    the shape of the object. Plan §4.1's target `p*` is a *distribution over
    valid terminals* precisely because a question can have several complete
    proofs, and `DATASET_DECISION.md` §2 adopts FEVEROUS's convention
    (~1.6 alternative complete evidence sets per claim) for exactly this reason.
    Sorted-id order makes the choice **deterministic**, which is what recall
    needs; it does not make it canonical, and no downstream text may call it
    "the" minimal proof.

    **``ledger=None`` is deliberate.** This is offline gold construction, not a
    query. Metering it would spend the per-query ``checker_budget`` that Stage
    D's primary metric is *defined in units of* (`CLAUDE.md` §6), so a gold set
    would consume the budget of the run being scored against it.
    """
    from graft.core.checker import H

    def is_evidence(atom_id: str) -> bool:
        atom = pool[atom_id]
        return atom.kind == "node" and snapshot.ntype(atom.target) in ASSERTION_BACKED_NTYPES

    current = set(atoms)
    removed: list[str] = []
    checks = 0
    passes = 0

    # **Structural atoms only — assertion-backed atoms are never removed.**
    # This is item 3.2 as amended on 15 Aug 2026, and the amendment exists
    # because the unrestricted form is degenerate: `H` is formal validity only
    # (sufficiency is routed to `U`, plan v1.2 §4.4), so nothing in it can say
    # "this question needs *both* claims".  Measured before the amendment, on all
    # 9 pilot questions with valid gold: every one collapsed to a single atom, 5
    # of 9 lost evidence doing it, and a bare ``Entity`` node — which asserts
    # nothing — satisfied `H` on its own.
    #
    # Restricting removal to scaffolding keeps every gold claim and drops only
    # the entities, intervals and edges that are not needed to keep the set
    # valid.  It is still exactly "remove any atom whose deletion keeps `H`
    # true", over a stated subset.
    #
    # A fixpoint loop rather than one sweep, because a structural atom can be
    # temporarily **blocked** by an edge referencing it: removing an interval
    # while its ``valid_during`` edge is present breaks closure, and it only
    # becomes removable after that edge goes.
    while True:
        passes += 1
        progressed = False
        for atom_id in sorted(a for a in current if not is_evidence(a)):
            trial = current - {atom_id}
            checks += 1
            if H(trial, obligations, snapshot, pool, config, ledger=None).ok:
                current = trial
                removed.append(atom_id)
                progressed = True
        if not progressed:
            break

    evidence = sorted(a for a in current if is_evidence(a))
    offered = sorted(a for a in set(atoms) if is_evidence(a))
    return frozenset(current), {
        "removed": len(removed),
        "kept": len(current),
        "evidence_atoms": len(evidence),
        "evidence_offered": len(offered),
        "evidence_dropped": len(offered) - len(evidence),
        "passes": passes,
        "h_checks": checks,
        "minimality": "local (irreducible over structural atoms), not minimum cardinality",
        "scope": "structural atoms only; assertion-backed atoms are never removed",
        # Both guards stay, and both are now expected to read clean -- which is
        # the point of keeping them.  They were what *detected* the pre-amendment
        # degeneracy, so removing them once the amendment landed would delete the
        # instrument that would notice a regression.
        "degenerate": not evidence,
        "under_constrained": len(offered) - len(evidence) > 0,
    }


def tier_b_gold(
    snapshot: Any,
    gold_turns: Iterable[str],
    obligations: Obligations,
    conv_id: str | None = None,
    *,
    config: Config | None = None,
) -> tuple[frozenset[str], frozenset[str], dict[str, Any]]:
    """Tier-B gold: the `H`-minimised sufficient-proof set (contract item 3.2).

    **Unblocked 15 August 2026** by the Gate-0 signature. Until then this
    function refused *by name* rather than falling back to Tier A — because a
    fallback would have put "sufficient-proof recall" and "``has_answer``-derived
    superset recall" behind one number, which decision 8's cost column calls
    "recall numbers that mean different things in different documents". The
    refusal is gone; the distinction it protected is now carried by the return
    type, which is a 3-tuple where Tier A's is a 2-tuple, so a caller cannot swap
    one for the other by accident.

    Item 3.2, verbatim: *the set of eligible assertions whose spans lie in
    ``has_answer`` turns of its evidence sessions, closed under the structural
    refs rule (fix F10), minimised by removing any atom whose deletion keeps `H`
    true.* The first two clauses are exactly :func:`tier_a_gold`, which is why
    this builds on it rather than re-deriving — one definition of
    "pool-representable", used by both tiers.

    **Needs per-question ``Obligations``, and that is the whole reason it could
    not run earlier** (G1). `H` is defined against a question's requirements;
    without them there is nothing for a proof to be sufficient *for*.

    Returns ``(minimal_atoms, reachable_node_atoms, report)``. When the Tier-A
    superset is **not itself valid**, this returns empty sets and a report saying
    so rather than inventing a proof: `H` failing on the superset means no valid
    proof exists inside it, and a recall denominator of "the answer we could not
    find" would silently flatter every method scored against it.
    """
    cfg = config or Config()
    closed, _reachable_a = tier_a_gold(snapshot, gold_turns, conv_id)
    if not closed:
        return frozenset(), frozenset(), {"status": "empty_tier_a", "h_checks": 0}

    nodes = gold_nodes(snapshot, gold_turns, conv_id)
    pool, _, _ = uncapped_pool(snapshot, {n: 1.0 for n in nodes}, conv_id=conv_id)

    # **`max_atoms` does not bind a gold set** (item 3.2 as amended 15 Aug 2026).
    # `H`'s ``size`` sub-check rejects a set larger than ``max_atoms`` = 16, which
    # is a bound on what a *candidate* set may select — Stage D's action budget.
    # A gold set is not a candidate set: it is what the question's evidence
    # actually is, and Tier A is capped at ``pool_cap`` = 64 precisely so gold can
    # exceed what one pool holds. Left unexempted, every question with more than
    # 16 gold atoms would fail `H` on its own superset and have no Tier-B gold —
    # thinning coverage exactly on the questions carrying the most evidence.
    # Raised rather than removed, so every other sub-check still applies.
    gold_cfg = replace(cfg, max_atoms=max(cfg.max_atoms, len(closed)), pool_cap=max(cfg.pool_cap, len(closed)))

    from graft.core.checker import H

    verdict = H(closed, obligations, snapshot, pool, gold_cfg, ledger=None)
    if not verdict.ok:
        return (
            frozenset(),
            frozenset(),
            {
                "status": "tier_a_superset_invalid",
                "violations": [v.check for v in verdict.violations],
                "h_checks": 1,
                "reading": (
                    "the has_answer-derived closed set does not satisfy H, so no "
                    "valid proof exists inside it and there is no Tier-B gold for "
                    "this question. Reported, never repaired by relaxing H."
                ),
            },
        )

    minimal, report = minimise(closed, pool, obligations, snapshot, gold_cfg)
    reachable = frozenset(
        aid
        for aid in minimal
        if pool[aid].kind == "node" and snapshot.ntype(pool[aid].target) in ASSERTION_BACKED_NTYPES
    )
    report["status"] = "ok"
    report["h_checks"] = report.get("h_checks", 0) + 1
    report["tier_a_size"] = len(closed)
    # The kind split of the minimal set, for required-node / required-edge
    # Recall@k (plan §3.3, §6.4).  Atom ids are opaque hashes, so a caller
    # without this pool object cannot recover the split -- it is made here, once.
    report["required"] = {
        "node_atoms": sorted(a for a in minimal if pool[a].kind == "node"),
        "edge_atoms": sorted(a for a in minimal if pool[a].kind == "edge"),
    }
    return minimal, reachable, report


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

    **Measured in the cap's own units** (corrected 15 Aug 2026 audit).  The first
    version compared the count of eligible assertion-backed *nodes* against
    ``pool_cap`` — but G8 applies the cap to the **closed atom set**, structural
    nodes and edges included, and the pilot's own artefact shows 23 candidate
    nodes closing to 59 atoms.  A conversation of 30 candidates could close past
    64 atoms — the cap binding, retrieval genuinely selecting — while the
    node-count comparison reported "unexercised".  Wrong in exactly the regime
    the flag exists for, so the comparison now runs on the closed size.
    """
    nodes = eligible_nodes(snapshot, conv_id)
    closed_size = 0
    if nodes:
        _, _, report = uncapped_pool(
            snapshot, {n: 1.0 for n in nodes}, conv_id=conv_id
        )
        closed_size = int(report["pool_size"])
    return {
        "candidates_in_scope": len(nodes),
        "closed_atoms_in_scope": closed_size,
        "pool_cap": int(cap),
        "exercised": closed_size > int(cap),
        "reading": (
            "closed candidate set fits inside pool_cap: the pool is the whole "
            "conversation, so recall is 1.0 by construction and this question "
            "does not test retrieval"
            if closed_size <= int(cap)
            else "closed candidate set exceeds pool_cap: the cap binds and "
            "retrieval selects"
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
        # Per channel, the miss *list* is replaced by its count: a channel that
        # misses everything would otherwise serialise the full gold id list once
        # per channel per question -- at scope c that is pure artefact bloat.
        # The full list survives in the tier rows, which exist once per question
        # and are the audit trail.
        row["missed_n"] = len(row.pop("missed"))
        row["latency_ms"] = round(float(timings[name]), 3) if name in timings else None
        out[name] = row
    return out
