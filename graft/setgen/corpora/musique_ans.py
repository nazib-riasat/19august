"""P9.2 — the MuSiQue-Ans adapter.

**[EVIDENCE]** *MuSiQue: Multihop Questions via Single-hop Question Composition*
(TACL 2022).  Adopted as Stage D's second Tier-1 corpus for one specific reason
(`DATASET_DECISION.md` §2): it is **the only adopted source that ships the
obligations element** of the abstract ``(pool, obligations, gold_proof)``
triple.  2Wiki's obligations must be synthesised from an evidence triple;
MuSiQue's are *read* from its ``question_decomposition`` DAG.  Licence CC BY 4.0.

**A different split from Phase 8's, deliberately.**  The gate trains on
``musique_full``'s answerable/unanswerable contrast pairs; Stage D needs
``musique_ans``'s gold supporting sets.  Same archive, separate pins — two
experiments must not share one file entry, and `PHASE8_DECISIONS.md` §3.1
already records that ``full`` *substitutes* distractors rather than deleting
them, which is a property of that split and not of this one.

Verify-at-wiring findings, measured on the shipped dev split:

20 paragraphs per question, with ``is_supporting`` flags
    Paragraph-level gold, natively — which is what settles the granularity
    question 2Wiki had to decide by measurement, and is why both adapters emit
    paragraph documents.

hop questions are ``"<subject> >> <relation>"``
    e.g. ``"Green >> performer"``.  Later hops reference earlier answers as
    ``"#1"``, e.g. ``"#1 >> spouse"``.  The ``entity_anchor`` synthesis rule
    parses the subject of **hop 1** — the one hop guaranteed not to be a
    placeholder.

``answer_aliases`` exists and is frequently empty
    Carried into ``meta`` because ``portfolio.binding_of`` consumes it for the
    contested diagnostic, and an empty alias list there means "no alias
    matching", not "no answer".

The closed pool runs to ~60 atoms against ``pool_cap = 64``, so the cap barely
binds — which is the same arithmetic `PHASE7_DECISIONS.md` §3.1 found one stage
earlier, where every conversation held 8–23 candidates against the same cap and
recall was 1.000 *by arithmetic*.  It is reported per split rather than
discovered later.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from graft.config import Config
from graft.schemas import Obligations
from graft.setgen.corpora.scoring import score_texts
from graft.setgen.corpora.wiki2 import wiring_report
from graft.setgen.proofs import ProofExample, SourceDoc, build_example, register_adapter

__all__ = ["load_examples", "build_one", "synthesise_obligations", "hop_subject", "AGGREGATE_CUES"]

#: ``pins.OBLIGATION_SYNTHESIS["musique_ans"]["aggregate"]`` reads "from question
#: form", and this is that form, written down.  **[ANALYSIS]** — a declared
#: keyword rule, not a parser and not paper-backed.  It is listed here rather
#: than inlined so the wiring report's fill rate is interpretable: a 0% aggregate
#: rate means these cues did not fire, which is a fact about the rule as much as
#: about the corpus.
AGGREGATE_CUES: tuple[str, ...] = ("how many", "how much", "total number", "count of")


def hop_subject(question: str) -> str | None:
    """The subject of a decomposition hop, or ``None`` when it is a placeholder.

    MuSiQue writes hops as ``"<subject> >> <relation>"`` and refers to an earlier
    hop's answer as ``"#k"``.  A ``#k`` subject names nothing in the paragraph
    text, so returning it as an ``entity_anchor`` would create a slot that
    ``coverage`` counts as active and no atom can ever satisfy — inflating the
    denominator and depressing ``U`` on exactly the multi-hop questions the
    corpus exists to supply.
    """
    head = question.split(">>")[0].strip()
    if not head or head.startswith("#"):
        return None
    return head


def _resolve_to_title(subject: str | None, row: Mapping[str, Any]) -> str | None:
    """The paragraph title a hop subject names, or ``None`` when it names none.

    **AMENDED 16 Aug 2026, by measurement** — the second amendment to signed
    decision 5, and the same defect §1.3 fixed on 2Wiki, arriving from the other
    direction. :func:`hop_subject` promises in its own docstring to avoid
    emitting *"a slot that ``coverage`` counts as active and no atom can ever
    satisfy"*, but it only guarded the ``#k`` placeholder case. It emitted the
    raw subject **string**, while ``resolve.matches_anchor`` requires equality
    with an ``Entity`` node's name — and those names are paragraph **titles**
    (``proofs.build_snapshot``).

    Measured on the pinned dev split (2,417 rows): the raw subject equals a
    title on only **30.3%**, so **69.7% of questions carried an anchor no subset
    of the pool could satisfy**. Every one of those had its ``coverage``
    denominator inflated and its maximum achievable ``U`` silently depressed —
    and unevenly, since the miss rate rises with hop count, so the bias fell
    hardest on exactly the multi-hop questions the corpus is here to supply.

    Resolving by containment (longest title first) and emitting the **title**
    makes the slot satisfiable by construction wherever it is emitted, and
    ``None`` where it is not: unsatisfiable anchors go to **0%**, with 28.5% of
    rows honestly reporting no anchor rather than an impossible one. Titles are
    the retrieval candidates, so nothing here reads an annotation the way 2Wiki's
    retired ``evidences`` rule did.
    """
    if not subject:
        return None
    target = " ".join(str(subject).casefold().split())
    if not target:
        return None
    titles = [str(p.get("title", "")) for p in row.get("paragraphs", [])]
    matches = [
        t
        for t in titles
        if t and (" ".join(t.casefold().split()) == target
                  or target in " ".join(t.casefold().split())
                  or " ".join(t.casefold().split()) in target)
    ]
    # Longest first, ties on the string: the more specific title wins, and the
    # choice is reproducible rather than dict-ordered — 2Wiki's rule verbatim.
    return sorted(matches, key=lambda t: (-len(t), t))[0] if matches else None


def synthesise_obligations(row: Mapping[str, Any]) -> Obligations:
    """``pins.OBLIGATION_SYNTHESIS["musique_ans"]``, made executable.

    ``entity_anchor`` comes from hop 1, **resolved to a paragraph title** —
    see :func:`_resolve_to_title` for the measurement that forced the
    resolution step. ``aggregate`` comes from the question form.
    ``time_constraint`` stays absent because the pin says it is *not expressible*
    on this corpus, and a slot invented here would be one ``H``'s temporal
    sub-check could never bind on anyway.
    """
    hops = row.get("question_decomposition") or []
    subject = hop_subject(str(hops[0].get("question", ""))) if hops else None
    question = str(row.get("question", "")).casefold()
    return Obligations(
        entity_anchor=_resolve_to_title(subject, row),
        aggregate=any(cue in question for cue in AGGREGATE_CUES),
        scope=(str(row["id"]),),
    )


def build_one(row: Mapping[str, Any], embedder: Any, *, config: Config | None = None) -> ProofExample:
    """One MuSiQue-Ans row → a :class:`ProofExample`.

    ``entities`` is the paragraph title, giving the pool its ``about_entity``
    edges and making ``H``'s closure sub-check bind — the same construction the
    2Wiki adapter uses, so the two corpora produce structurally identical pools
    and a Gate-3 row that pools them is comparing questions rather than schemas.
    """
    docs: list[SourceDoc] = []
    for paragraph in row["paragraphs"]:
        index = int(paragraph["idx"])
        docs.append(
            SourceDoc(
                doc_id=f"p{index}",
                text=str(paragraph["paragraph_text"]).strip(),
                entities=(str(paragraph["title"]),),
                is_gold=bool(paragraph["is_supporting"]),
                meta={"title": str(paragraph["title"]), "index": index},
            )
        )

    fused, per_channel, per_channel_raw = score_texts(
        str(row["question"]), [d.doc_id for d in docs], [d.text for d in docs], embedder
    )
    hops = row.get("question_decomposition") or []
    return build_example(
        str(row["id"]),
        docs,
        synthesise_obligations(row),
        fused,
        channel_scores={
            **{k: v for k, v in per_channel.items()},
            **{f"{k}_raw": v for k, v in per_channel_raw.items()},
        },
        embed=lambda texts: embedder.embed(list(texts)),
        config=config,
        meta={
            "corpus": "musique_ans",
            "hops": len(hops),
            "answer": str(row.get("answer", "")),
            # Consumed by `portfolio.binding_of`; the answer itself leads so the
            # diagnostic works when the alias list is empty, which it often is.
            "answer_aliases": [str(row.get("answer", ""))]
            + [str(a) for a in (row.get("answer_aliases") or [])],
            "decomposition": [str(h.get("question", "")) for h in hops],
        },
    )


def load_examples(
    split: str = "train",
    *,
    limit: int | None = None,
    embedder: Any = None,
    config: Config | None = None,
    root: Any = None,
    verify: bool = True,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[ProofExample], dict[str, Any]]:
    """Load a pinned split and build examples, with the wiring report."""
    from graft.graphbuild.loaders import load_split

    if rows is None:
        rows = load_split("musique_ans", split, root=root, verify=verify)
    # **Stratified, not a head slice** — decision 2's rule, and on MuSiQue-Ans a
    # head slice is all-2-hop because the file is sorted by hop count.
    from graft.setgen.corpora import stratified_sample
    from graft.setgen.pins import SUBSET

    selected = stratified_sample(
        list(rows), lambda row: f"{len(row.get('question_decomposition') or [])}hop", limit, int(SUBSET["seed"])
    )

    examples = [build_one(row, embedder, config=config) for row in selected]
    report = wiring_report("musique_ans", split, selected, examples)
    # The stratum this corpus is sampled on (`pins.SUBSET["stratify"]`), reported
    # so a subset that drifted off its declared hop mix is visible.
    from collections import Counter

    report["types"] = dict(
        Counter(f"{len(r.get('question_decomposition') or [])}hop" for r in selected)
    )
    report["aggregate_rate"] = (
        round(sum(1 for e in examples if e.obligations.aggregate) / len(examples), 4)
        if examples
        else 0.0
    )
    return examples, report


register_adapter("musique_ans", load_examples)
