"""P8.2 — the one gate module allowed to read a gold field (G3).

**Everything in `graft/gate/` except this file is gold-quarantined**, and a
structural test asserts it over the source rather than trusting a docstring —
the same boundary `retrieve/recall.py` holds for Stage C, for the same reason: a
gate that could read ``has_answer`` or ``answer_session_ids`` would *be* the
label, and every abstention number would be measuring the instrument.

**What this module builds: evidence-deletion contrast pairs on conversation**
(G1). For a train-slice question whose evidence sessions are in the graph, the
pool is built twice — once normally (**answerable**), once with the evidence
sessions' assertions excluded from the candidate scope (**unanswerable by
construction**).

**[EVIDENCE-adjacent]** the recipe is FEVEROUS's documented
construct-negatives-by-deleting-evidence pattern, which `DATASET_DECISION.md` §2
names for exactly this purpose; applying it to conversation is **[ANALYSIS]**.
It is conversation-native, costs zero annotation, and — the point — the negative's
features come through the *real* Stage-C stack rather than a simulation of it.

**The adaptation loss, stated because it is real:** deletion-unanswerable is
*cleaner* than natural-unanswerable. A deployed unanswerable question is not one
whose evidence someone removed; it is one whose evidence was never there, and
which may resemble an answerable question far more closely. So a gate trained
here may find the natural sets harder than its dev curve suggests — which is why
LoCoMo's 446 adversarial questions are the **primary** evaluation (decision 1)
and are never trained on.

**Stage B, not Stage A.** Building these pairs needs an ingested conversational
corpus at scope c, and that ingestion has not run (Gate-0 item 9: decided, not
run). :func:`deletion_pairs` therefore works on whatever graph it is handed and
is exercised on fixtures today; the *decisive* conversational track is exit
criterion 13's deferred-by-name item.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from graft.retrieve.pool import eligible_nodes, node_atom_id
from graft.retrieve.recall import evidence_turns, gold_nodes, has_answer_turns

__all__ = [
    "ANSWERABLE",
    "UNANSWERABLE",
    "excluded_nodes",
    "deletion_pairs",
    "split_for_gate",
]

#: Label convention: **1.0 = answerable**, i.e. a sufficient proof exists in the
#: snapshot. Stated as constants because the polarity is easy to invert silently
#: and every downstream metric (AURC, abstention recall, false-abstention rate)
#: reads the opposite direction from the one a "risk" framing suggests.
ANSWERABLE = 1.0
UNANSWERABLE = 0.0


def excluded_nodes(
    snapshot: Any,
    instance: Mapping[str, Any],
    conv_id: str | None = None,
    *,
    unit: str = "session",
) -> frozenset[str]:
    """The assertion-backed nodes a deletion twin must not see.

    ``unit="session"`` (the default, and G1's wording — "the evidence sessions'
    assertions") removes everything derived from any turn of an evidence session.
    ``unit="turn"`` removes only ``has_answer`` turns' assertions, which is a
    *weaker* deletion: it leaves the surrounding session context in place.

    The session unit is the default because it is the one that makes the twin
    unanswerable **by construction** rather than by hope — a neighbouring turn in
    the same session frequently restates enough of the answer that turn-level
    deletion leaves a proof standing, and a "negative" that is actually
    answerable is a label error the gate would train on.
    """
    turns = evidence_turns(instance) if unit == "session" else has_answer_turns(instance)
    return frozenset(gold_nodes(snapshot, turns, conv_id))


def deletion_pairs(
    snapshot: Any,
    instances: Mapping[str, Mapping[str, Any]],
    build_channels: Any,
    assemble_pool: Any,
    question_ids: Sequence[str] | None = None,
    *,
    unit: str = "session",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One answerable/unanswerable pair per question, through the real stack.

    ``build_channels(question_id) -> {channel: {node_id: score}}`` and
    ``assemble_pool(question_id, channels) -> (pool, atom_scores, report)`` are
    injected rather than imported so this module never constructs a retrieval
    stack of its own — the features must come through *the* Stage-C path, not a
    copy of it (exit criterion 3), and injection is what makes that checkable.

    **The deletion is applied to the channels' outputs**, so the twin's pool is
    assembled from a candidate set that lacks every excluded node. That equals a
    true scope exclusion whenever no channel's own top-k truncated — on the pilot
    it never does (8–23 candidates against ``top_k`` = ``pool_cap`` = 64). Stage C
    is frozen for this phase (§2), so a scope-level exclusion would be a Stage-C
    change and is deliberately not made; instead each pair records
    ``candidates_in_scope`` and ``max_channel_hits`` so the condition is
    **checkable per question** rather than asserted once in prose.

    Returns ``(pairs, report)``. Each pair carries both sides' pool, scores,
    report and label, plus the excluded set, so exit criterion 4 — *the twin's
    pool lacks every gold atom* — is checkable from the returned object.
    """
    wanted = list(question_ids if question_ids is not None else sorted(instances))
    pairs: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"no_instance": 0, "no_evidence_nodes": 0, "empty_answerable": 0}

    for question_id in wanted:
        instance = instances.get(question_id)
        if instance is None:
            skipped["no_instance"] += 1
            continue
        excluded = excluded_nodes(snapshot, instance, question_id, unit=unit)
        if not excluded:
            # Nothing to delete means no contrast: the "twin" would be identical
            # to the original and would be labelled unanswerable while being the
            # answerable pool. Skipped and counted, never emitted.
            skipped["no_evidence_nodes"] += 1
            continue

        channels = build_channels(question_id)
        pool_a, scores_a, report_a = assemble_pool(question_id, channels)
        if len(pool_a) == 0:
            skipped["empty_answerable"] += 1
            continue

        in_scope = len(eligible_nodes(snapshot, question_id))
        max_hits = max((len(h) for h in channels.values()), default=0)
        kept_channels = {
            name: {nid: s for nid, s in hits.items() if nid not in excluded}
            for name, hits in channels.items()
        }
        pool_u, scores_u, report_u = assemble_pool(question_id, kept_channels)

        gold_atoms = {node_atom_id(n) for n in excluded}
        if gold_atoms & set(pool_u.ids()):
            # Should be impossible: the excluded nodes were removed before
            # assembly. Raised rather than counted because it would mean the
            # twin is not unanswerable-by-construction, which is the one property
            # the whole training signal rests on.
            raise ValueError(
                f"deletion twin for {question_id} still contains gold atoms; the "
                "negative is not unanswerable by construction"
            )

        pairs.append(
            {
                "question_id": question_id,
                "excluded_nodes": sorted(excluded),
                # Recorded so the caveat above is checkable per question rather
                # than asserted: post-hoc filtering equals scope exclusion when no
                # channel truncated, and `max_channel_hits` against
                # `candidates_in_scope` is what a reader needs to see that.
                "candidates_in_scope": in_scope,
                "max_channel_hits": max_hits,
                "answerable": {
                    "label": ANSWERABLE,
                    "pool": pool_a,
                    "scores": scores_a,
                    "report": report_a,
                },
                "unanswerable": {
                    "label": UNANSWERABLE,
                    "pool": pool_u,
                    "scores": scores_u,
                    "report": report_u,
                },
            }
        )

    return pairs, {
        "requested": len(wanted),
        "pairs": len(pairs),
        "skipped": skipped,
        "deletion_unit": unit,
        "exclusion_note": (
            "the deletion is applied to channel outputs, which equals excluding "
            "the nodes from the candidate scope whenever no channel truncated at "
            "its top_k. Per-pair `candidates_in_scope` and `max_channel_hits` are "
            "recorded so that condition is checkable rather than assumed; Stage C "
            "is frozen for Phase 8 (section 2), so a true scope-level exclusion "
            "would be a Stage-C change and is not made here."
        ),
        "balance": "1:1 by construction (G6) — a constructed balance, never a natural frequency",
        "adaptation_loss": (
            "deletion-unanswerable is cleaner than natural-unanswerable: deployed "
            "unanswerables are not made by removing evidence. LoCoMo adversarial "
            "is the primary evaluation for this reason (decision 1)."
        ),
    }


def split_for_gate(
    question_ids: Iterable[str],
    question_types: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Gate-0 item 5's splits, reused rather than reinvented.

    ``graphbuild.train.split_questions`` is user-level and stratified at seed
    ``20260813``; a ``question_id`` *is* one simulated user's whole haystack, so
    splitting on it means no conversation spans two splits — item 5's requirement
    met structurally. Stratification matters here for the same reason it does for
    D2: an unstratified draw moves the rare question types between splits, and a
    gate's dev curve over a different type mix than its test is not a dev curve.
    """
    from graft.graphbuild.train import split_questions

    return split_questions(sorted(set(question_ids)), question_types)
