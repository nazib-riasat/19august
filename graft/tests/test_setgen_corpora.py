"""Phase 9 build step 2 — the corpus adapters.

Exit criteria 7–9. Fixtures are hand-built rows in each corpus's *real* shape
(verified against the downloaded splits on 16 Aug 2026), so this file runs with
no data on disk; the two tests that need the real files skip when they are
absent, because raw corpora are gitignored and fetched per machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graft.graphbuild.embed import StubEmbedder
from graft.graphbuild.loaders import PHASE9_ROOT
from graft.graphbuild.pins import DATASETS
from graft.setgen.corpora import musique_ans, scoring, wiki2
from graft.setgen.proofs import adapter_names, get_adapter

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def embedder():
    return StubEmbedder(dim=384)


def wiki_row(dup_title: bool = False, _id: str = "w1") -> dict:
    """One 2Wiki row in the real shape: 10 paragraphs, sentence-level facts."""
    titles = ["Alpha", "Beta", "Gamma", "Delta", "Eps", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
    if dup_title:
        titles[3] = "Alpha"  # 3.7% of dev rows do exactly this
    return {
        "_id": _id,
        "type": "compositional",
        "question": "Who directed the film Alpha?",
        "answer": "Beta Person",
        "context": [[t, [f"{t} sentence one.", f"{t} sentence two."]] for t in titles],
        "supporting_facts": [["Alpha", 0], ["Beta", 1]],
        "evidences": [["Alpha", "director", "Beta Person"], ["Beta Person", "born", "Gamma"]],
    }


def musique_row(hops: int = 2, _id: str = "m1", question: str = "Who is the spouse of the Green performer?") -> dict:
    return {
        "id": _id,
        "question": question,
        "answer": "Miquette Giraudy",
        "answer_aliases": ["Miquette"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": i,
                "title": f"Title {i}",
                "paragraph_text": f"Paragraph {i} discusses Green and other matters.",
                "is_supporting": i in (5, 10),
            }
            for i in range(20)
        ],
        "question_decomposition": (
            [{"id": "1", "question": "Green >> performer", "answer": "Steve Hillage",
              "paragraph_support_idx": "10"}]
            + [{"id": str(k), "question": f"#{k - 1} >> spouse", "answer": "Miquette Giraudy",
                "paragraph_support_idx": "5"} for k in range(2, hops + 1)]
        ),
    }


# --------------------------------------------------------------------------
# registration and the boundary
# --------------------------------------------------------------------------


def test_both_tier1_adapters_register_on_import():
    """Adapters register on import, not by scanning.

    An adapter that is never imported is never registered, so the import-graph
    boundary holds at runtime and not merely in the source.
    """
    assert set(adapter_names()) >= {"wiki2", "musique_ans"}
    assert get_adapter("wiki2") is wiki2.load_examples
    assert get_adapter("musique_ans") is musique_ans.load_examples


def test_an_unknown_adapter_names_the_import_that_would_fix_it():
    with pytest.raises(KeyError, match="graft.setgen.corpora"):
        get_adapter("hover")


# --------------------------------------------------------------------------
# 2Wiki
# --------------------------------------------------------------------------


def test_wiki2_gold_is_every_paragraph_holding_a_supporting_sentence(embedder):
    """G3's gold rule, at the granularity measurement settled on.

    ``supporting_facts`` are ``[title, sentence_idx]``; documents are paragraphs;
    so a paragraph is gold iff it contains a supporting sentence.
    """
    examples, _ = wiki2.load_examples("dev", rows=[wiki_row()], embedder=embedder)
    ex = examples[0]
    assert ex.gold_complete
    assert len(ex.gold_atom_ids) == 2, "Alpha and Beta each hold one supporting sentence"


def test_wiki2_survives_duplicate_context_titles(embedder):
    """Measured on dev: 463/12,576 rows (3.7%) repeat a context title.

    The first version of this adapter keyed documents by title and
    ``build_snapshot``'s duplicate guard refused it. Documents are keyed by
    paragraph **index** instead. The corpus's ambiguity is real, but 0 rows have
    a duplicated title among their *supporting facts*, so gold is never ambiguous.
    """
    examples, report = wiki2.load_examples("dev", rows=[wiki_row(dup_title=True)], embedder=embedder)
    ex = examples[0]
    assert report["examples_built"] == 1
    assert ex.gold_complete
    # The duplicated title makes BOTH "Alpha" paragraphs gold, which is the
    # honest consequence of a title-keyed label on an index-keyed document set.
    assert len(ex.gold_atom_ids) == 3


def test_wiki2_entity_anchor_is_question_derived_not_gold_derived(embedder):
    """The rule amended 16 Aug 2026 after an adversarial audit called the
    original a gold leak — and it was one.

    ``evidences`` is 2Wiki's own annotation, scored by its evidence-F1 metric and
    **absent at inference**. Measured on the full 12,576-row dev split, the first
    evidence subject matched a gold title on 92.5% of rows and a distractor on
    9.3%, making ``entity_anchor_hit`` a near-exact gold indicator that reached
    every arm's ``state_repr`` through ``d(s)``.

    The replacement reads only the **question** and the **candidate titles**,
    both of which a retriever has at inference.
    """
    examples, report = wiki2.load_examples("dev", rows=[wiki_row()], embedder=embedder)
    assert examples[0].obligations.entity_anchor == "Alpha"
    assert examples[0].obligations.scope == ("w1",)
    assert report["obligation_fill"]["entity_anchor"] == 1.0
    # Absent slots are absent, not invented: filling them would inflate U's
    # coverage denominator with obligations the question does not impose.
    for slot in ("value_type", "time_constraint", "needs_source"):
        assert report["obligation_fill"][slot] == 0.0


def test_wiki2_anchor_ignores_the_evidence_annotation_entirely(embedder):
    """Stripping ``evidences`` must not change the anchor.

    This is the regression that would catch the leak coming back: if any future
    edit reads the annotation again, removing it will move the anchor and this
    test goes red. Asserting the *absence* of a dependency is the only way to
    keep it absent.
    """
    with_ev = wiki_row()
    without = wiki_row()
    without["evidences"] = []
    a, _ = wiki2.load_examples("dev", rows=[with_ev], embedder=embedder)
    b, _ = wiki2.load_examples("dev", rows=[without], embedder=embedder)
    assert a[0].obligations.entity_anchor == b[0].obligations.entity_anchor == "Alpha"


def test_wiki2_anchor_prefers_the_longest_matching_title(embedder):
    """A question naming both "Alpha" and "Alpha Sessions" anchors on the specific one."""
    row = wiki_row()
    row["question"] = "Who directed the Alpha Sessions documentary about Alpha?"
    row["context"] = [["Alpha Sessions", ["s."]], ["Alpha", ["t."]]] + row["context"][2:]
    examples, _ = wiki2.load_examples("dev", rows=[row], embedder=embedder)
    assert examples[0].obligations.entity_anchor == "Alpha Sessions"


def test_wiki2_anchor_is_absent_when_no_title_appears_in_the_question(embedder):
    """Absent, not invented. Measured at 100% coverage on the real dev split, but
    the corpus does not guarantee it and a fabricated anchor would add an
    obligation slot no atom can satisfy."""
    row = wiki_row()
    row["question"] = "What colour is the sky on a clear day?"
    examples, _ = wiki2.load_examples("dev", rows=[row], embedder=embedder)
    assert examples[0].obligations.entity_anchor is None


# --------------------------------------------------------------------------
# MuSiQue-Ans
# --------------------------------------------------------------------------


def test_musique_reads_gold_from_is_supporting(embedder):
    examples, _ = musique_ans.load_examples("dev", rows=[musique_row()], embedder=embedder)
    assert examples[0].gold_complete
    assert len(examples[0].gold_atom_ids) == 2


def test_musique_hop_subject_refuses_a_placeholder():
    """``#1 >> spouse`` names nothing in the paragraph text.

    Returning it as an ``entity_anchor`` would create a slot ``coverage`` counts
    as active and no atom can satisfy — inflating the denominator and depressing
    ``U`` on exactly the multi-hop questions this corpus exists to supply.
    """
    assert musique_ans.hop_subject("Green >> performer") == "Green"
    assert musique_ans.hop_subject("#1 >> spouse") is None
    assert musique_ans.hop_subject("   ") is None


def test_musique_entity_anchor_resolves_to_a_title_or_is_absent(embedder):
    """**Amended 16 Aug 2026 by measurement** — the second amendment to signed
    decision 5, and the same defect §1.3 fixed on 2Wiki from the other side.

    ``hop_subject`` promises to avoid emitting "a slot that ``coverage`` counts
    as active and no atom can ever satisfy", but it emitted the raw subject
    **string** while ``resolve.matches_anchor`` compares against an ``Entity``
    node's name — which is a paragraph **title**. Measured on the pinned dev
    split: the raw subject equalled a title on only 30.3%, so **69.7% of
    questions carried an unsatisfiable anchor**, inflating their ``coverage``
    denominator and depressing their maximum achievable ``U`` — worst on the
    high-hop questions, i.e. exactly the ones this corpus is here to supply.

    The contract is now: emit the resolved **title**, or ``None``. Both branches
    are asserted, because emitting nothing is the honest half of the fix and a
    test that only checked the happy path would let it regress to a string.
    """
    # hop-1 subject "Green" matches no title -> absent, not unsatisfiable
    absent, _ = musique_ans.load_examples("dev", rows=[musique_row()], embedder=embedder)
    assert absent[0].obligations.entity_anchor is None

    # and when it does name a paragraph, the emitted anchor is that title, so
    # `matches_anchor`'s exact comparison can succeed
    row = musique_row()
    row["question_decomposition"][0]["question"] = "Title 10 >> performer"
    resolved, _ = musique_ans.load_examples("dev", rows=[row], embedder=embedder)
    assert resolved[0].obligations.entity_anchor == "Title 10"


def test_musique_aggregate_is_a_declared_keyword_rule(embedder):
    plain, _ = musique_ans.load_examples("dev", rows=[musique_row()], embedder=embedder)
    counted, _ = musique_ans.load_examples(
        "dev", rows=[musique_row(question="How many albums did Green release?")], embedder=embedder
    )
    assert plain[0].obligations.aggregate is False
    assert counted[0].obligations.aggregate is True


def test_musique_carries_the_answer_first_in_its_aliases(embedder):
    """``portfolio.binding_of`` consumes this, and ``answer_aliases`` is often empty."""
    examples, _ = musique_ans.load_examples("dev", rows=[musique_row()], embedder=embedder)
    aliases = examples[0].meta["answer_aliases"]
    assert aliases[0] == "Miquette Giraudy"
    assert "Miquette" in aliases


# --------------------------------------------------------------------------
# the wiring report — exit criteria 7-9
# --------------------------------------------------------------------------


def test_the_wiring_report_carries_what_the_exit_criteria_ask_for(embedder):
    examples, report = wiki2.load_examples(
        "dev", rows=[wiki_row(_id=f"w{i}") for i in range(4)], embedder=embedder
    )
    for key in (
        "rows_read", "examples_built", "pool_size", "gold_complete",
        "obligation_fill", "content_key_collisions", "types",
    ):
        assert key in report, f"the wiring report is missing {key}"
    # G11: expected zero on paragraph-derived pools, MEASURED not assumed. The
    # Symmetry-Aware terminal-reward correction applies only if it is non-zero.
    assert report["content_key_collisions"] == 0
    assert report["gold_complete"] == len(examples)


def test_both_corpora_produce_structurally_identical_pools(embedder):
    """A Gate-3 row that pools the two corpora must compare questions, not schemas."""
    w, _ = wiki2.load_examples("dev", rows=[wiki_row()], embedder=embedder)
    m, _ = musique_ans.load_examples("dev", rows=[musique_row()], embedder=embedder)
    for ex in (w[0], m[0]):
        kinds = {a.kind for a in ex.pool}
        assert kinds == {"node", "edge"}, f"{ex.meta['corpus']} pool has kinds {kinds}"
        assert any(a.label == "about_entity" for a in ex.pool if a.kind == "edge")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_scoring_agrees_with_the_phase8_implementation(embedder):
    """Two independent expressions of one **declared** arithmetic must agree.

    Not a self-referential test: Stage C declares per-channel min–max then ``max``
    across channels, and `PHASE8_DECISIONS.md` §2.5 requires both views ship. If
    these two drifted, channel-score features would mean different things in the
    gate and in Stage D while both claimed to be "the declared arithmetic".
    """
    from graft.gate.adapt_musique import paragraph_scores

    texts = ["Green was performed by Steve Hillage.", "Bananas are yellow.", "London is a city."]
    question = "Who performed Green?"
    ids = [f"p{i}" for i in range(len(texts))]

    mine_fused, mine_norm, mine_raw = scoring.score_texts(question, ids, texts, embedder)
    theirs_fused, theirs_norm, theirs_raw = paragraph_scores(
        question, [{"paragraph_text": t} for t in texts], embedder
    )
    assert mine_fused == pytest.approx(theirs_fused)

    # Theirs is keyed doc -> {channel: score}; mine is channel -> {doc: score}.
    # The transposition is the only difference, and this asserts exactly that.
    #
    # Compared leaf by leaf: `pytest.approx` does not recurse into a dict of
    # dicts, so `nested == approx(nested)` is False even for identical values --
    # which is how the first version of this test failed against code that
    # agreed exactly. A comparison that reports a difference where there is none
    # is worse than no comparison, because the next person deletes it.
    def flatten(by_channel):
        return {
            (did, channel): value
            for channel, per_doc in by_channel.items()
            for did, value in per_doc.items()
        }

    def flatten_by_doc(by_doc):
        return {
            (did, channel): value
            for did, per_channel in by_doc.items()
            for channel, value in per_channel.items()
        }

    assert flatten(mine_norm) == pytest.approx(flatten_by_doc(theirs_norm))
    assert flatten(mine_raw) == pytest.approx(flatten_by_doc(theirs_raw))


def test_a_flat_channel_maps_to_one_not_zero():
    """Mapping a flat channel to the floor would drag the ``max`` fusion down on
    every document it scored."""
    assert scoring.minmax({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
    assert scoring.minmax({}) == {}
    assert scoring.minmax({"a": 0.0, "b": 2.0}) == {"a": 0.0, "b": 1.0}


def test_scoring_survives_an_all_stopword_corpus(embedder):
    """bm25s leaves an empty vocabulary and raises; the channel goes quiet and
    the dense one carries the question."""
    fused, norm, raw = scoring.score_texts("the", ["p0", "p1"], ["the", "a"], embedder)
    assert "dense" in raw
    assert set(fused) == {"p0", "p1"}


# --------------------------------------------------------------------------
# the pins, and the real files when present
# --------------------------------------------------------------------------


def test_both_corpora_are_registered_with_licence_and_shas():
    """Exit criterion 7. Licences are re-verified at download against the primary
    source, because ``DATASET_DECISION.md`` §8 records that its own verification
    pass did not complete and every adoption there is single-sourced.
    """
    for name, licence in (("2wiki", "Apache-2.0"), ("musique_ans", "CC BY 4.0")):
        spec = DATASETS[name]
        assert spec["phase"] == 9
        assert licence in spec["licence"]
        assert {"train", "dev"} <= set(spec["files"])
        for _rel, sha in spec["files"].values():
            assert len(sha) == 64, "a SHA pin is not a sha256"


@pytest.mark.parametrize("corpus", ["2wiki", "musique_ans"])
def test_the_pinned_files_verify_when_present(corpus):
    """The SHA check is the one guarantee that a run read the file it says it did."""
    from graft.graphbuild.loaders import load_split

    rel, _sha = DATASETS[corpus]["files"]["dev"]
    if not (REPO / PHASE9_ROOT / rel).is_file():
        pytest.skip(f"{corpus} dev not fetched on this machine (raw corpora are gitignored)")
    rows = load_split(corpus, "dev", root=REPO / PHASE9_ROOT)
    assert len(rows) > 1000
