"""P9.2 — the 2WikiMultiHopQA adapter.

**[EVIDENCE]** *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation
of Reasoning Steps* (COLING 2020) — Ho, Duong Nguyen, Sugawara, Aizawa.  Adopted
as Stage D's first Tier-1 corpus because it is the only source whose annotation
is simultaneously an evidence **set** and a typed relational path, and because
evidence-F1 is one of its official metrics with human performance at 78.81
against a 14.94 baseline — the headroom Stage D's claim needs
(`DATASET_DECISION.md` §2).  Licence Apache-2.0, re-verified at download
16 Aug 2026 as G10 requires.

Verify-at-wiring findings, all **measured on the 12,576-row dev split** rather
than assumed, and all carried in the wiring report:

context is always 10 paragraphs
    12,576 / 12,576.  Pre-split into sentences, which is what makes the
    granularity question below a real one rather than a formality.

``supporting_facts`` are ``[title, sentence_idx]`` — **sentence-level**
    Every supporting-fact title resolves into the context (0 failures), so gold
    is always locatable.

3.7% of rows carry **duplicate context titles**
    463 / 12,576.  Titles are therefore *not* unique keys within a question, and
    the first version of this adapter keyed documents by title and was refused by
    ``build_snapshot``'s duplicate guard.  Documents are keyed by paragraph
    **index**.  Crucially, **0 rows have a duplicated title among their
    supporting facts**, so ``[title, idx]`` is never ambiguous for gold — the
    corpus's ambiguity exists but never touches a label.

the first evidence subject is a context title on only 82.3% of rows
    10,346 / 12,576.  The synthesised ``entity_anchor`` is therefore an entity
    *string* rather than a guaranteed paragraph title, which is exactly how
    ``atomfeat`` consumes it (normalised substring against atom text).

**Granularity is a decision, taken by measurement** (recorded in
``pins.CORPORA["wiki2"]["gold"]`` and `PHASE9_DECISIONS.md` §2.1 — there is no
``pins.GRANULARITY``; this docstring named one that never existed, found by the
16 Aug 2026 audit).  The plan's G3 says "gold atoms = the atoms derived from
supporting sentences", which reads as sentence-level documents.  Measured over
40 dev rows at ``pool_cap = 64``:

=============  ==================  ==================
granularity    closed pool size    gold survives cap
=============  ==================  ==================
paragraph      29 – 30             **40 / 40**
sentence       34 – 64             28 / 40
=============  ==================  ==================

Sentence-level puts the closed pool **on the cap** and drops a gold atom on 30%
of questions.  An example whose gold is partial is not a harder instance — its
``sufficiency`` can never reach 1.0, so its maximum achievable reward is
silently below every other example's, and training on it biases the budget
toward whatever the pool happened to fit.  So documents are **paragraphs**, and
the plan's sentence-level intent is honoured as: *a paragraph is gold iff it
contains a supporting sentence*.  MuSiQue annotates paragraphs natively, so this
also makes the two Tier-1 corpora commensurable in one training run.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from graft.config import Config
from graft.schemas import Obligations
from graft.setgen.corpora.scoring import score_texts
from graft.setgen.proofs import (
    ProofExample,
    SourceDoc,
    atom_text,
    build_example,
    register_adapter,
)

__all__ = ["QUESTION_TYPES", "load_examples", "build_one", "synthesise_obligations"]

#: The corpus's own four types — the strata ``pins.SUBSET["stratify"]["wiki2"]``
#: names, verified against the dev split rather than taken from the paper:
#: compositional 5,236 · comparison 3,040 · bridge_comparison 2,751 ·
#: inference 1,549.
QUESTION_TYPES: tuple[str, ...] = (
    "comparison",
    "inference",
    "compositional",
    "bridge_comparison",
)


def _normalise(text: str) -> str:
    return " ".join(str(text).casefold().split())


def synthesise_obligations(row: Mapping[str, Any]) -> Obligations:
    """``pins.OBLIGATION_SYNTHESIS["wiki2"]`` — **amended 16 Aug 2026, by measurement.**

    The rule as pinned read *"subject of the first evidence triple"*, and an
    adversarial audit was right to call that a **gold leak**.  ``evidences`` is
    2WikiMultiHopQA's own annotation — one of the fields its evidence-F1 metric
    scores against — so it exists on train and dev and **nowhere at inference**,
    where the frozen obligation parser reads the *question*.  Measured on the
    full 12,576-row dev split, the first evidence subject:

    * matches a **gold** paragraph title on **92.5%** of rows,
    * matches a **distractor** title on only **9.3%**.

    So ``entity_anchor_hit`` was a near-exact gold-paragraph indicator, and it
    does not stop at one feature: the anchor drives ``d(s)``'s first deficit
    component, so it reaches ``state_repr`` for **every** arm, the ``Δd`` block
    for L7/L7b, and — through the pooled atom matrix — the distilled head, whose
    docstring guarantees no gold reaches an input.  Every Stage-A Gate-3 number
    would have been measured under supervision unavailable at deployment.

    **The replacement is question-derived and measured at 100% coverage:** the
    longest context title that appears in the question.  Both halves are
    available at inference — the question is the query, and candidate titles are
    what retrieval is ranking — so nothing here reads an annotation field.
    ``supporting_facts`` and ``evidences`` are untouched by this function.

    *Longest* rather than first: a question naming both "Green" and "Green
    (album)" should anchor on the more specific one, and length is the cheapest
    total order that does it.  Ties break on the title string so the choice is
    reproducible rather than dict-ordered.

    Every other slot stays absent, and that absence is the honest report: this
    corpus expresses no time constraint, no value type and no source
    requirement, so filling them would invent obligations the question does not
    impose and inflate ``U``'s ``coverage`` denominator.
    """
    question = _normalise(row.get("question", ""))
    titles = [str(entry[0]) for entry in row.get("context", [])]
    matched = sorted(
        (t for t in titles if _normalise(t) and _normalise(t) in question),
        key=lambda t: (-len(t), t),
    )
    return Obligations(
        entity_anchor=matched[0] if matched else None,
        scope=(str(row["_id"]),),
    )


def build_one(row: Mapping[str, Any], embedder: Any, *, config: Config | None = None) -> ProofExample:
    """One 2Wiki row → a :class:`ProofExample`.

    Documents are paragraphs keyed by **index** (see the module docstring on
    duplicate titles); ``entities`` is the paragraph title, which is what gives
    the pool its ``about_entity`` edges and therefore makes ``H``'s closure
    sub-check bind rather than sit vacuous.
    """
    supporting_titles = {str(t) for t, _ in row.get("supporting_facts", [])}
    docs: list[SourceDoc] = []
    for index, entry in enumerate(row["context"]):
        title, sentences = entry[0], entry[1]
        docs.append(
            SourceDoc(
                doc_id=f"p{index}",
                text=" ".join(str(s) for s in sentences).strip(),
                entities=(str(title),),
                is_gold=str(title) in supporting_titles,
                meta={"title": str(title), "index": index},
            )
        )

    fused, per_channel, per_channel_raw = score_texts(
        str(row["question"]), [d.doc_id for d in docs], [d.text for d in docs], embedder
    )
    return build_example(
        str(row["_id"]),
        docs,
        synthesise_obligations(row),
        fused,
        # Raw **and** normalised, both keyed channel -> {doc: score}. The raw
        # view is what `PHASE8_DECISIONS.md` §3.3 had to add after normalisation
        # flattened 10 of 13 features to constants.
        channel_scores={
            **{k: v for k, v in per_channel.items()},
            **{f"{k}_raw": v for k, v in per_channel_raw.items()},
        },
        embed=lambda texts: embedder.embed(list(texts)),
        config=config,
        meta={
            "corpus": "wiki2",
            "type": str(row.get("type", "")),
            "answer": str(row.get("answer", "")),
            # **Was hardcoded to `[]`** (found by adversarial audit, 16 Aug
            # 2026), which structurally disabled `portfolio.binding_of` and the
            # contested diagnostic on this corpus: an empty alias list means
            # "match nothing", so every 2Wiki set bound nothing and
            # `answer_agreement` (then named `contested_rate`) reported
            # `bound: 0` unconditionally. 2Wiki ships
            # no alias field, so the answer itself is the alias list — the same
            # convention the MuSiQue adapter uses for its frequently-empty one.
            "answer_aliases": [str(row.get("answer", ""))],
            "evidences": [list(map(str, e)) for e in (row.get("evidences") or [])],
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
    """Load a pinned split and build examples, with the wiring report.

    ``rows`` bypasses the loader for fixtures **only**; the SHA check is the
    project's one guarantee that a run read the file it says it did, and
    ``load_split(verify=False)`` exists for the same narrow purpose.
    """
    from graft.graphbuild.loaders import load_split

    if rows is None:
        rows = load_split("2wiki", split, root=root, verify=verify)
    # **Stratified, not a head slice** — decision 2's rule, and on MuSiQue-Ans a
    # head slice is all-2-hop because the file is sorted by hop count.
    from graft.setgen.corpora import stratified_sample
    from graft.setgen.pins import SUBSET

    selected = stratified_sample(
        list(rows), lambda row: str(row.get("type", "")), limit, int(SUBSET["seed"])
    )

    examples: list[ProofExample] = []
    for row in selected:
        examples.append(build_one(row, embedder, config=config))
    return examples, wiring_report("wiki2", split, selected, examples)


def wiring_report(
    corpus: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    examples: Sequence[ProofExample],
) -> dict[str, Any]:
    """Counts, obligation fill rates, collisions and gold survival.

    Exit criteria 7–9 in one object.  Every number here is *reported* rather than
    asserted, because each of them is a way an adapter can be quietly wrong: an
    obligation slot that is always empty makes ``coverage`` a constant, a
    content-key collision splits flow that belongs to one object (G11), and a
    dropped gold atom lowers an example's achievable reward without lowering
    anyone's expectations of it.
    """
    from collections import Counter

    slots = Counter()
    for ex in examples:
        for name in ex.obligations.active_slots():
            slots[name] += 1

    # **G11's instrument, corrected 16 Aug 2026 after an adversarial audit
    # showed the original could never read non-zero.**
    #
    # `content_key` is `(kind, target, label, *refs)`, and `target` is a node id
    # minted from the assertion id, which is minted from the text and spans. Two
    # atoms with one content key would therefore already share an `atom_id` and
    # `AtomPool` would have collapsed them. So "content-key collisions" is
    # structurally zero on any pool `proofs.build_snapshot` produces -- a proof,
    # not a measurement, and reporting it as the latter made "measured, not
    # assumed" hollow.
    #
    # The quantity that *can* vary is **distinct atoms carrying identical
    # normalised text**: two paragraphs with the same text get different ids
    # (their spans differ). It is measurable and non-zero on real corpora, so it
    # is reported as a **dataset diagnostic**.
    #
    # **It is NOT what Symmetry-Aware GFlowNets corrects, and the earlier version
    # of this comment said it was** (corrected 16 Aug 2026 after reading the
    # paper). Cor 5.1 scales a terminal by ``|Aut(G)|`` — the *graph automorphism
    # group*, under a node-by-node generation scheme, where orbit-equivalent
    # construction actions build the same graph. Here the state is a set of atom
    # **ids**: two identical-text paragraphs have different ids, different spans
    # and different provenance, so they are genuinely different terminals, not
    # one object reached twice. Provenance-preservation is Contribution 1's
    # claim; collapsing two provenances would contradict it.
    #
    # The duplicated class does carry ~2x the reward mass of a de-duplicated one
    # (measured by enumerating a real pool's valid singleton terminals). That is
    # arithmetic of the *corpus*, not a sampler bias with an automorphism group
    # to quotient — so no scalar is applied, and the number is recorded here so a
    # later reader cannot infer from silence that one should be.
    structural = duplicate_text = 0
    for ex in examples:
        keys = [ex.pool[a].content_key() for a in ex.pool.ids()]
        structural += len(keys) - len(set(keys))
        texts = [
            " ".join(atom_text(ex.snapshot, ex.pool[a]).casefold().split())
            for a in ex.pool.ids()
            if ex.pool[a].kind == "node"
        ]
        texts = [t for t in texts if t]
        duplicate_text += len(texts) - len(set(texts))

    sizes = sorted(len(ex.pool) for ex in examples)
    complete = sum(1 for ex in examples if ex.gold_complete)
    return {
        "corpus": corpus,
        "split": split,
        "rows_read": len(rows),
        "examples_built": len(examples),
        "pool_size": {
            "min": sizes[0] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "max": sizes[-1] if sizes else 0,
        },
        "gold_complete": complete,
        "gold_incomplete": len(examples) - complete,
        # G5: per-slot fill rates. A slot filled 0% of the time is a slot the
        # synthesis rule claims and the corpus does not supply.
        "obligation_fill": {
            name: round(slots[name] / len(examples), 4) if examples else 0.0
            for name in ("entity_anchor", "value_type", "time_constraint", "needs_source")
        },
        # G11. `content_key_collisions` is structurally zero by id derivation and
        # is reported so the *proof* is visible rather than implied.
        # `equivalent_evidence_atoms` is the quantity that can vary — a **dataset
        # diagnostic**, not a sampler bias with a correction attached; see the
        # comment above and `PHASE9_DECISIONS.md` §1.4 for why Cor 5.1 does not
        # apply to an id-defined MDP.
        "content_key_collisions": structural,
        "equivalent_evidence_atoms": duplicate_text,
        "types": dict(Counter(str(r.get("type", "")) for r in rows)) if rows else {},
    }


register_adapter("wiki2", load_examples)
