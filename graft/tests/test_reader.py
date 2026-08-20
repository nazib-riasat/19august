"""Phase 10 Stage A/B — serializer, parser, ceilings, and fix F7's slot guard.

Every test here runs on fixtures with no GPU and no weights: the reader itself is
exercised by ``scripts/phase10_read.py``'s runs (R1–R3), and what this file
guards is the *deterministic* surface those runs depend on.

Two of these are regressions for defects the first runs found — the edge-atom
duplicate rendering and the citation-marker scoring bug — and both are marked as
such, because the reasoning matters more than the assertion.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from graft.config import Config
from graft.diagnostics import ceilings as C
from graft.reader import pins
from graft.reader.parse import (
    CitationError,
    answers_equivalent,
    normalise_answer,
    parse_answer,
    resolve_citations,
    token_f1,
)
from graft.reader.serialize import ProofSerializer, approx_tokens
from graft.schemas import Obligations, ProofSet
from graft.setgen.proofs import SourceDoc, build_example

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def example():
    docs = [
        SourceDoc(f"p{i}", f"Document {i} states that London is a capital and fact {i} holds.",
                  ("London",), is_gold=(i < 2))
        for i in range(5)
    ]
    return build_example(
        "q1", docs, Obligations(entity_anchor="London", scope=("q1",)),
        {f"p{i}": 1.0 - 0.1 * i for i in range(5)},
    )


@pytest.fixture
def serializer(example):
    return ProofSerializer(example.snapshot, example.pool, config=Config())


def _all_atoms(example) -> ProofSet:
    """The whole pool — which is **closed**, and that matters.

    An earlier version sliced ``sorted(pool.ids())[:6]``, producing a set whose
    edge atoms referenced endpoints outside it. That is not a set Stage D can
    emit (fix F10's closure rule is enforced by the ``ADD`` masks), so testing
    the serializer against one was testing it against an input the live path
    cannot produce — and it masked the closure-repair defect the 16 Aug 2026
    audit found, because the repair's own bug and the fixture's malformedness
    cancelled out.
    """
    return ProofSet(atoms=frozenset(example.pool.ids()))


# --------------------------------------------------------------------------
# G3 — the ordering reads no gold, structurally
# --------------------------------------------------------------------------


def test_the_serializer_reads_no_forbidden_signal():
    """§6 decision 3, asserted on the AST rather than remembered.

    The architecture's own phrasing was "anchor and **answer-binding** evidence
    at the beginning and end", and answer-binding requires the answer. This is
    the third appearance of that class of error in this project —
    `PHASE9_DECISIONS.md` §1.3 caught a gold annotation reaching every arm's
    `state_repr`, and Phase 9's G9 caught it in fix F4's contested flag — so it
    is guarded rather than trusted.

    Names, not substrings: the module docstring legitimately *discusses* the
    forbidden signals in order to explain why they are forbidden, and a guard
    that fails on its own documentation is a guard someone deletes (the Phase-9
    gold-quarantine test made exactly that mistake first).
    """
    tree = ast.parse((REPO / "graft/reader/serialize.py").read_text(encoding="utf-8"))
    forbidden = set(pins.ORDERING["forbidden_signals"])
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            seen.add(node.attr)
        if isinstance(node, ast.Name) and node.id in forbidden:
            seen.add(node.id)
    assert not seen, f"the serializer reads {seen}, which is not available at inference"


def test_the_gold_ordering_is_reachable_only_through_an_explicit_argument(example, serializer):
    """The diagnostic ordering exists, and it announces itself.

    Its whole purpose is that the gap against the honest ordering measures what
    honesty costs; a caller that got it by accident would report a number the
    inference path cannot reproduce, so the ordering name travels with the result.
    """
    proof = _all_atoms(example)
    honest = serializer.serialise(proof, example.obligations, example.atom_scores)
    gold = serializer.serialise(
        proof, example.obligations, example.atom_scores, gold=example.gold_atom_ids
    )
    assert honest.ordering == "u_shaped_inference_computable"
    assert gold.ordering == "gold_diagnostic"


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def test_serialisation_is_deterministic(example, serializer):
    """Two runs must serialise the same set identically, or claim ids are unstable."""
    proof = _all_atoms(example)
    a = serializer.serialise(proof, example.obligations, example.atom_scores)
    b = serializer.serialise(proof, example.obligations, example.atom_scores)
    assert a.text == b.text
    assert a.claim_map == b.claim_map


def test_an_edge_atom_renders_as_a_relation_not_a_requoted_span(example, serializer):
    """Regression — found by the first smoke run, 16 Aug 2026.

    Edge atoms carry their endpoint's provenance by construction:
    `proofs.build_snapshot` sources an `about_entity` edge from the very span
    that sources its claim, because the paragraph is the warrant for both. Quoted
    the same way, they emitted the identical sentence under two different claim
    ids — so the reader saw duplicated evidence and a citation to the relation
    scored as a citation to the claim.
    """
    out = serializer.serialise(_all_atoms(example), example.obligations, example.atom_scores)
    lines = out.text.splitlines()
    assert len(lines) == len(set(lines)), "two evidence blocks render identically"
    edges = [a for a in out.included if example.pool[a].kind == "edge"]
    assert edges, "fixture has no edge atoms; the regression is untested"
    for line in lines:
        if "→" in line:
            # An edge names its source claim, so two edges into one entity from
            # different claims are distinguishable.
            assert line.count("[") >= 2, f"edge block does not name its source: {line}"


def test_claim_ids_are_stable_across_the_budget_ladder(example, serializer):
    """Ceiling 4 is reported at every rung (decision 1), so the rungs must be
    comparable atom-by-atom. Renumbering per budget would make the same atom
    `[c3]` at 512 and `[c2]` at 160."""
    proof = _all_atoms(example)
    maps = {}
    for budget in pins.BUDGET_LADDER:
        out = serializer.serialise(proof, example.obligations, example.atom_scores, budget=budget)
        maps[budget] = out.claim_map
    shared = set.intersection(*(set(m) for m in maps.values()))
    for claim_id in shared:
        atoms = {m[claim_id] for m in maps.values()}
        assert len(atoms) == 1, f"{claim_id} names different atoms at different budgets"


def test_truncation_preserves_closure(example, serializer):
    """The budget counts tokens and knows nothing about references, so it can
    drop a node while keeping an edge that points at it — leaving the reader a
    proof whose citation resolves to something not shown. Closure is what makes a
    partial set sound (fix F10) and it has to survive serialization."""
    out = serializer.serialise(
        _all_atoms(example), example.obligations, example.atom_scores, budget=40
    )
    kept = set(out.included)
    for atom_id in out.included:
        for ref in example.pool[atom_id].refs:
            assert ref in kept, f"{atom_id} survived but its reference {ref} did not"


def test_a_single_oversized_atom_is_still_emitted(example, serializer):
    """An empty evidence block would make the reader abstain for a reason that
    has nothing to do with the evidence. The overflow is visible instead."""
    out = serializer.serialise(
        _all_atoms(example), example.obligations, example.atom_scores, budget=1
    )
    assert len(out.included) == 1
    assert out.tokens > out.budget
    assert not out.complete


def test_the_token_counter_is_reported(example, serializer):
    """A ceiling-4 number under the heuristic is marked approximate — the same
    discipline Phase 6/7 apply to the stub embedder."""
    out = serializer.serialise(_all_atoms(example), example.obligations, example.atom_scores)
    assert out.report()["approximate_tokens"] is True
    assert out.counter == "approx_tokens"

    real = ProofSerializer(
        example.snapshot, example.pool, config=Config(),
        count_tokens=approx_tokens, counter_name="reader_tokenizer",
    )
    assert real.serialise(
        _all_atoms(example), example.obligations, example.atom_scores
    ).report()["approximate_tokens"] is False


# --------------------------------------------------------------------------
# parsing — including the R2 regression
# --------------------------------------------------------------------------


def test_citation_markers_do_not_count_as_answer_tokens():
    """**Regression for what run R2 found on 16 Aug 2026.**

    The reader answered ``"London\\n[c1]"`` against gold ``"London"`` — correct —
    and ceiling 5 scored it False. SQuAD normalisation strips punctuation, so
    ``[c1]`` became the bare token ``c1`` and the normalised answer was
    ``"london c1"``.

    The error was systematic and one-directional: it could only ever mark a
    correct answer *wrong*, and it would have done so on every cited answer —
    i.e. on every answer the prompt asks for.
    """
    assert normalise_answer("London\n[c1]") == "london"
    assert answers_equivalent("London\n[c1]", "London")
    assert token_f1("London\n[c1]", "London") == pytest.approx(1.0)
    assert normalise_answer("The Big Apple [c1][c3].") == "big apple"
    # and it still catches a genuinely wrong answer
    assert not answers_equivalent("Paris [c1]", "London")


def test_abstention_is_parsed_from_the_string_the_prompt_asks_for():
    """One constant, two consumers. A parser hunting for hedging would make the
    abstention rate a function of the reader's politeness rather than its
    decision, and plan §4.2 needs abstention to be a decision."""
    assert parse_answer(pins.INSUFFICIENT).abstained
    assert parse_answer("INSUFFICIENT EVIDENCE").citations == ()
    assert not parse_answer("London [c1]").abstained


def test_citations_are_deduplicated_and_case_insensitive():
    assert parse_answer("A [c1] B [c1] C [c2].").citations == ("c1", "c2")
    assert parse_answer("Yes [C2].").citations == ("c2",)


def test_an_unresolvable_citation_raises_rather_than_being_dropped():
    """A reader citing a claim it was never shown has hallucinated a citation,
    and that is a reader-ceiling finding. Discarding it would inflate citation
    precision by removing exactly the cases that should lower it."""
    parsed = parse_answer("London [c1] and [c9].")
    with pytest.raises(CitationError, match="c9"):
        resolve_citations(parsed, {"c1": "atom-a"})
    lax = resolve_citations(parsed, {"c1": "atom-a"}, strict=False)
    assert lax.unresolved == ("c9",)


def test_citations_resolve_to_real_spans(example, serializer):
    """Through the same provenance chain `H`'s support sub-check walks — not a
    second traversal, or citation precision measures a different object than
    formal validity does."""
    out = serializer.serialise(_all_atoms(example), example.obligations, example.atom_scores)
    first = sorted(out.claim_map)[0]
    parsed = resolve_citations(
        parse_answer(f"Something [{first}]."), out.claim_map,
        example.snapshot, example.pool,
    )
    assert parsed.spans[first], "a citation resolved to no span"


# --------------------------------------------------------------------------
# the five ceilings
# --------------------------------------------------------------------------


def test_all_five_ceilings_report_and_stamp_their_tier(example, serializer):
    """Decision 5. A ceiling without its tier is uninterpretable, and a table
    with four rows invites the reader to assume the fifth was fine."""
    out = C.all_ceilings(
        snapshot=example.snapshot, conv_id="q1",
        retrieved=sorted(example.pool.ids()), gold=example.gold_atom_ids,
        serializer=serializer, obligations=example.obligations,
        scores=example.atom_scores, question="What?",
        read_fn=lambda e, q: "London [c1]", gold_answer="London",
    )
    assert set(out) == {"1_extraction", "2_graph", "3_candidate", "4_packing", "5_reader"}
    for name, payload in out.items():
        assert "available" in payload, f"{name} does not say whether it ran"
        assert "ceiling" in payload, f"{name} has no ceiling value"
        if payload["available"]:
            assert payload.get("tier"), f"{name} ran without stamping its tier"


def test_ceiling_3_carries_the_saturation_guard(example):
    """`PHASE7_DECISIONS.md` §3.1: Tier-A recall was 1.000 on 9/10 pilot
    questions *by arithmetic*. A ceiling-3 number without the flag is that
    artefact reported as a result."""
    out = C.ceiling_3_candidate(
        sorted(example.pool.ids()), example.gold_atom_ids,
        snapshot=example.snapshot, conv_id="q1",
    )
    assert "saturation" in out


def test_ceiling_4_reports_every_rung_of_the_ladder(example, serializer):
    """`CLAUDE.md` §8 records that quoting a packing result at one cherry-picked
    budget was one of this project's own caught errors."""
    out = C.ceiling_4_packing(
        example.gold_atom_ids, serializer, example.obligations, example.atom_scores
    )
    assert set(out["by_budget"]) == {str(b) for b in pins.BUDGET_LADDER}
    for rung in out["by_budget"].values():
        assert "survives" in rung and "dropped" in rung


def test_a_ceiling_that_could_not_run_says_so(example):
    """Omission would let a reader assume it was fine."""
    out = C.all_ceilings(retrieved=(), gold=())
    for name in ("1_extraction", "2_graph", "4_packing", "5_reader"):
        assert out[name]["available"] is False
        assert out[name].get("reason")


# --------------------------------------------------------------------------
# pins and fix F7
# --------------------------------------------------------------------------


def test_pins_import_without_torch_and_bind_the_prompt():
    """`verify_handoff.py` calls the fingerprint on a bare interpreter.
    `PHASE9_DECISIONS.md` §7.3 records a stage-D fingerprint whose *module*
    imported clean while the *call* pulled torch in — the harder version to
    notice, because nothing fails until someone runs it without torch."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; from graft.reader.pins import stage_e_fingerprint; "
         "stage_e_fingerprint(); "
         "raise SystemExit(1 if 'torch' in sys.modules else 0)"],
        cwd=REPO, capture_output=True,
    )
    assert proc.returncode == 0, "computing the stage-E fingerprint imported torch"

    frozen = pins.frozen_values()
    assert frozen["prompt_sha"] == pins.PROMPT_SHA
    assert len(pins.stage_e_fingerprint()) == 64


def test_the_prompt_asks_for_a_phrase_not_a_sentence():
    """Caught on the first smoke run, before the SHA was frozen into anything.

    Every benchmark this project scores against has a **short span** as gold, so
    a sentence answer scores ~0 exact-match by construction — decision 10's local
    pair would have measured verbosity rather than correctness.
    """
    assert "shortest phrase" in pins.PROMPT_TEMPLATE
    assert "Never a full sentence" in pins.PROMPT_TEMPLATE
    assert pins.INSUFFICIENT in pins.PROMPT_TEMPLATE


def test_the_prompt_carries_the_three_format_alignments():
    """Amended 19 Aug 2026, before any decisive run existed -- the last §6b-clean
    moment. Format alignment, not optimisation: the reference systems' own
    prompts do the same, and a frozen reader that loses token-F1 to formatting
    is measuring prose style, not memory.

    Three properties, each protecting a scoring failure that is invisible in
    the answer's *correctness*: a date rule (an ISO date is right and scores ~0
    against a day-month-year gold), a multi-item rule (token recall punishes
    naming one of two cities), and format examples (a 3B model anchors on
    demonstrations, not rules)."""
    assert "8 May 2023" in pins.PROMPT_TEMPLATE, "date-format rule missing"
    assert "separated by commas" in pins.PROMPT_TEMPLATE, "multi-item rule missing"
    assert "these are not evidence" in pins.PROMPT_TEMPLATE, (
        "format examples must be labelled so they cannot be read as evidence"
    )
    # The examples' citation ids must never resolve against a real pool by
    # accident -- they are demonstrations, and the runner's resolve_citations
    # would count a copied [c2] as unresolved, which is the correct outcome.
    #
    # **The example VALUES moved on 20 Aug 2026** (run-3 fix 4). Run 2 measured
    # the reader emitting `Rome, Lisbon [c1][c4]` verbatim as an answer, so the
    # demonstrations are now values LoCoMo cannot contain, and rule 7 says not to
    # copy them. What this test pins is the *shape* -- one example carrying a
    # citation -- not the particular city, so a future re-pick does not need to
    # edit an assertion that was never about the value.
    assert "Tbilisi [c2]" in pins.PROMPT_TEMPLATE
    assert "Never copy an answer from the format examples" in pins.PROMPT_TEMPLATE


def test_a_leading_answer_echo_is_stripped_before_scoring():
    """The prompt ends with 'Answer:' and instruct models echo it. Left in, the
    token 'answer' survives normalisation (articles are stripped, this word is
    not) and costs F1 precision on every echoed reply -- a formatting artefact
    scored as a wrong word. Extraction hygiene; the scoring rule is untouched."""
    from graft.reader.parse import parse_answer

    echoed = parse_answer("Answer: London [c1]")
    plain = parse_answer("London [c1]")
    assert echoed.answer_text == plain.answer_text == "London [c1]"
    assert echoed.citations == ("c1",)
    # And it must not eat a legitimate answer that merely starts with the word.
    assert parse_answer("answering machine [c1]").answer_text == "answering machine [c1]"


def test_the_prompt_takes_no_corpus_or_system_branch():
    """v1.2 §3.5's matched comparison is enforced by hash equality, so Phase 11's
    baselines must reuse this byte-identically. A template that mentioned the
    system would make that impossible."""
    lowered = pins.PROMPT_TEMPLATE.lower()
    for word in ("graft", "musique", "2wiki", "longmemeval", "locomo", "bm25", "baseline"):
        assert word not in lowered, f"the prompt names {word!r}"


def test_fix_f7_refuses_a_second_model_slot():
    """The constraint is a resource fact on an 8 GB card, not a convention:
    the reader alone measured 6.317 GB peak on run R1."""
    from graft.reader.read import ModelSlot

    with ModelSlot("first"):
        assert ModelSlot.holder() == "first"
        with pytest.raises(RuntimeError, match="already resident"):
            with ModelSlot("second"):
                pass
    assert ModelSlot.holder() is None


def test_post_hoc_verification_is_declined_with_its_grounds():
    """G11. `CLAUDE.md` §5 lists "SynCheck described as free" among this
    project's caught errors; decision 11 is that finding acted on."""
    assert pins.POST_HOC_VERIFICATION["adopted"] is False
    assert len(pins.POST_HOC_VERIFICATION["grounds"]) == 3


# --------------------------------------------------------------------------
# regressions for the 16 Aug 2026 adversarial audit
# --------------------------------------------------------------------------


def test_budget_truncation_drops_a_tail_not_a_first_fit(example, serializer):
    """**Regression.** The loop used `continue`, making truncation *first-fit*:
    a cheaper atom further down the order slipped in after a more important one
    was dropped (measured kept indices [0,1,2,3,5], skipping 4).

    That is not cosmetic. The U-shape's premise (Lost in the Middle, TACL 2024)
    is that head and tail positions carry the most weight, so first-fit spends the
    budget on exactly the positions the evidence says matter least.

    Gaps from **closure repair** are legitimate and expected — an edge whose
    endpoint the budget dropped must go too — so the assertion is that every gap
    is an atom removed for closure, never a node skipped for cheapness.
    """
    proof = _all_atoms(example)
    full = serializer.serialise(proof, example.obligations, example.atom_scores)
    order = list(full.included)
    tight = serializer.serialise(
        proof, example.obligations, example.atom_scores, budget=45
    )
    kept = [order.index(a) for a in tight.included if a in order]
    assert kept, "fixture did not truncate; the regression is untested"
    for gap in (i for i in range(max(kept) + 1) if i not in kept):
        atom = example.pool[order[gap]]
        assert atom.refs, (
            f"index {gap} ({atom.kind}) was skipped but references nothing — "
            "that is first-fit, not closure repair"
        )
    # And closure still holds over what survived.
    survivors = set(tight.included)
    for atom_id in tight.included:
        for ref in example.pool[atom_id].refs:
            assert ref in survivors


def test_ceiling_5_does_not_inherit_ceiling_4s_failure(example, serializer):
    """**Regression.** Ceiling 5 serialised at the *live* budget, so a gold proof
    that did not fit was scored as a **reader** failure when it was a **packing**
    failure.

    That is the conflation the five-ceiling protocol exists to prevent: v1.2 §6.3
    argues one number is uninterpretable because it could be any of five stages,
    and a ceiling that absorbs the one below it re-creates the problem inside the
    instrument.
    """
    seen = {}

    def read_fn(evidence, question):
        seen["evidence"] = evidence
        return "London [c1]"

    out = C.ceiling_5_reader(
        example.gold_atom_ids, serializer, example.obligations,
        "Where?", read_fn, gold_answer="London", scores=example.atom_scores,
    )
    # The whole gold proof reached the reader, whatever the live budget is.
    assert out["packed"]["dropped"] == 0
    assert out["packed"]["complete"] is True
    # And ceiling 4's separate question is reported beside it, never folded in.
    assert "survives_live_budget" in out
    assert "live_budget" in out


def test_the_honest_versus_gold_ordering_gap_is_computable(example, serializer):
    """**Regression.** Decision 3 mandates that the gap be the measurement of what
    the honest ordering costs — and no code path could compute it. Both orderings
    existed; the comparison did not, which is the harder half to notice because
    the code looked complete."""
    from graft.reader.serialize import ordering_gap

    gap = ordering_gap(
        serializer, _all_atoms(example), example.obligations,
        example.atom_scores, example.gold_atom_ids,
    )
    assert gap["honest"]["ordering"] == "u_shaped_inference_computable"
    assert gap["gold_diagnostic"]["ordering"] == "gold_diagnostic"
    assert gap["shared_atoms"] >= 2
    assert gap["rank_correlation"] is not None
    assert -1.0 <= gap["rank_correlation"] <= 1.0


def test_the_reader_is_pinned_by_revision_not_only_by_name():
    """A model id is a moving target: the same name can serve different weights
    over time, so "same frozen reader for every compared system" (v1.2 §3.5) is
    unenforceable if the fingerprint binds a label rather than a checkpoint.
    `graphbuild.pins.EMBEDDER` already pins both."""
    assert "revision" in pins.READER
    assert "revision" in pins.frozen_values()["reader"]
