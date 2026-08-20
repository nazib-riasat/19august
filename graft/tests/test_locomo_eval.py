"""The LoCoMo end-to-end join — Stage C → Stage D → Stage E over a real snapshot.

`scripts/phase10_read.py` drives the read path over hand-built fixtures, which is
what made runs R1-R3 wiring tests.  ``scripts/locomo_eval.py`` drives it over a
real graph, and this file is what makes that join checkable **without a GPU and
without ingestion** -- because the alternative is discovering a wiring error after
spending reader time on 1,986 questions.

The snapshot fixture is `test_retrieve.py`'s, imported rather than rebuilt: one
graph fixture project-wide, the same rule Phase 7 applies to the pool mapping.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from graft.config import Config
from graft.graphbuild.embed import StubEmbedder
from graft.ledger import Ledger
from graft.reader.orchestrator import ReadPathStamp, answer, cost_report
from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer
from graft.setgen.distill import HeadScorer, build_head
from graft.setgen.realenv import RealEnvironment
from graft.setgen.policy import Policy
from graft.tests.test_retrieve import snap  # noqa: F401  — the shared graph fixture

REPO = Path(__file__).resolve().parents[2]


def _runner():
    """Load the runner as a module. It lives in ``scripts/``, which is not a package.

    Imported rather than duplicated: a test that reimplemented ``stage_c`` would
    pass while the runner was broken, which is the failure mode this file exists
    to prevent.
    """
    path = REPO / "scripts" / "locomo_eval.py"
    spec = importlib.util.spec_from_file_location("locomo_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["locomo_eval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _runner()


def test_the_runner_imports_without_a_gpu_or_a_corpus(runner):
    """Every import in the read path resolves on a bare interpreter. Cheap, and it
    is the failure that would otherwise surface only after model load."""
    assert callable(runner.stage_c)
    assert callable(runner.build_example)
    assert callable(runner.default_obligations)


def test_the_default_obligation_asserts_nothing_it_did_not_parse(runner):
    """The cheap path is the *empty* obligation, not a guess at the question's
    semantics. Phase-1 G5 scores an unbounded interval as temporal_correctness
    1.0 and the temporal filter fails open, so this weakens retrieval rather than
    corrupting validity."""
    obl = runner.default_obligations("c1")
    assert obl.entity_anchor is None
    assert obl.time_constraint is None
    assert obl.needs_source is False
    assert obl.scope == ("c1",)


def test_stage_c_builds_a_pool_and_meters_it_as_one_stage(runner, snap):  # noqa: F811
    """Retrieval cost has to land in the same per-query snapshot as generation.
    `assemble` is called with `stage=None` because the runner owns the wider
    boundary, and ledger stages do not nest."""
    config = Config()
    ledger = Ledger.from_config(config)
    with ledger.query_scope("q1"):
        pool, atom_scores, report, obligations, channels = runner.stage_c(
            snap, StubEmbedder(), "what is my weight", "c1", config, ledger
        )
        snapshot = ledger.snapshot()

    assert pool.ids(), "Stage C returned an empty pool on a populated graph"
    assert set(channels) == {"bm25", "dense", "entity", "expand"}
    assert "stage_c" in snapshot["stages"]
    # The dense channel's question encode is a per-query GPU cost and is counted.
    assert snapshot["stages"]["stage_c"]["model_forwards"] >= 1
    # Both score views survive to the example -- PHASE8_DECISIONS.md §3.3.
    assert "raw_channel_scores" in report


def test_the_conversational_example_is_a_drop_in_for_the_wikipedia_one(runner, snap):  # noqa: F811
    """`GRAFT_PHASE9_BUILD.md`'s promise: the learner side takes
    ``(pool, obligations, gold_proof)`` and knows nothing about the corpus. If this
    passes, no file in ``graft/setgen/`` had to move to support conversation."""
    config = Config()
    ledger = Ledger.from_config(config)
    with ledger.query_scope("q1"):
        pool, atom_scores, report, obligations, _ = runner.stage_c(
            snap, StubEmbedder(), "what is my weight", "c1", config, ledger
        )
    example = runner.build_example(
        "q1", snap, pool, atom_scores, report, obligations,
        frozenset(list(pool.ids())[:1]), StubEmbedder(),
    )
    assert example.meta["gold_tier"] == "A"
    assert example.meta["corpus"] == "locomo"
    # The learner constructs over it unchanged.
    env = RealEnvironment(example, config, range_samples=0)
    assert env.example is example


def test_the_whole_read_path_runs_over_a_real_graph(runner, snap):  # noqa: F811
    """The join Phase 10 deferred, end to end on a stub reader.

    This is the test that would have caught a wiring error before 1,986 questions
    of reader time rather than after.
    """
    config = Config()
    ledger = Ledger.from_config(config)
    stamp = ReadPathStamp(
        policy_trained=False, gate_source="none", scorer_source="distilled_head",
        token_counter="approx_tokens", ordering="u_shaped_inference_computable",
    )
    with ledger.query_scope("q1"):
        pool, atom_scores, report, obligations, _ = runner.stage_c(
            snap, StubEmbedder(), "what is my weight", "c1", config, ledger
        )
        example = runner.build_example(
            "q1", snap, pool, atom_scores, report, obligations,
            frozenset(list(pool.ids())[:1]), StubEmbedder(),
        )
        env = RealEnvironment(example, config, range_samples=0)
        featurizer = RealFeaturizer(
            example, Policy(*RealFeaturizer.dims(), hidden=16), config, delta_d=False
        )
        scorer = HeadScorer(build_head(ATOM_WIDTH, seed=config.seeds[0]), featurizer)
        result = answer(
            "what is my weight", env=env, featurizer=featurizer, scorer=scorer,
            read_fn=lambda e, q: "70kg [c1]", gate_decision=None,
            obligations=obligations, atom_scores=atom_scores,
            rng=np.random.default_rng(config.seeds[0]),
            ledger=ledger, config=config, stamp=stamp, query_id="q1",
            contested_check=False,
        )

    assert result.record.outcome in ("answer", "contested", "abstain")
    snapshot = result.record.ledger_snapshot
    # All three stages accounted, which is what makes the cost axis complete.
    for stage in ("stage_c", "stage_d", "stage_e"):
        assert stage in snapshot["stages"], f"{stage} unaccounted"


def test_gate_features_are_recorded_so_a_threshold_costs_no_gpu(runner, snap):  # noqa: F811
    """The runner's decision 2: the gate is a small model over Stage-C outputs and
    the reader is the whole cost, so features are written per question and a
    threshold is applied post hoc. Without this, an abstention analysis means a
    second full reader pass."""
    from graft.gate import features as gfeatures
    from graft.gate import pins as gpins

    config = Config()
    ledger = Ledger.from_config(config)
    with ledger.query_scope("q1"):
        pool, atom_scores, report, obligations, _ = runner.stage_c(
            snap, StubEmbedder(), "what is my weight", "c1", config, ledger
        )
    vec, names, flags = gfeatures.build_features(
        obligations=obligations, pool=pool, atom_scores=atom_scores,
        assembly_report=report, snapshot=snap,
    )
    assert len(vec) == len(names) and len(vec) > 0
    assert any(flags.values()), "no feature block was present on a real pool"
    # And the prevalence a post-hoc threshold must be reweighted to exists.
    assert gpins.EVAL_PREVALENCES["locomo"] == 0.2246


def test_the_cost_shim_reports_the_same_arithmetic_as_the_orchestrator(runner, snap):  # noqa: F811
    """The runner rebuilds ``ReadResult``-likes from its JSONL rows so a resumed
    run costs what a fresh one does. One implementation of the cost arithmetic,
    reached two ways -- if these diverged, a resumed run would report a different
    cost for identical work."""
    row = {
        "outcome": "answer",
        "abstain_cause": None,
        "ledger_snapshot": {
            "totals": {
                "llm_calls": 2, "llm_tokens_in": 500, "llm_tokens_out": 12,
                "model_forwards": 40, "wall_clock_ms": 900,
                "terminal_checks": 0, "incremental_ops": 0,
            },
            "stages": {},
        },
    }

    class _R:
        def __init__(self, r):
            self.record = type("rec", (), {
                "ledger_snapshot": r["ledger_snapshot"],
                "outcome": r["outcome"],
                "abstain_cause": r["abstain_cause"],
            })()

    cost = cost_report([_R(row)])
    assert cost["queries_metered"] == 1
    assert cost["llm_tokens_total_per_query"]["mean"] == pytest.approx(512.0)


def test_the_channel_cache_builds_once_per_conversation(runner, snap):  # noqa: F811
    """The single largest waste in the runner's first version: both channels index
    the conversation in their constructor -- BM25 tokenises it, dense *embeds* it --
    and the loop built both per question. LoCoMo has 96-260 questions per
    conversation, so the corpus was re-embedded up to 260 times to answer 260
    questions about it."""
    config = Config()
    cache = runner.ChannelCache(snap, StubEmbedder(), config)
    ledger = Ledger.from_config(config)
    with ledger.query_scope("q1"):
        for _ in range(5):
            cache.for_conv("c1", ledger)
    assert cache.builds == 1, "rebuilt within one conversation"
    with ledger.query_scope("q2"):
        cache.for_conv("c2", ledger)
    assert cache.builds == 2, "did not rebuild on a new conversation"


def test_a_cached_channel_gives_the_identical_pool_to_a_cold_one(runner, snap):  # noqa: F811
    """The correctness guarantee the cache needs. Both channels are pure functions
    of (snapshot, conv_id) and the snapshot is a pinned event-log offset, so warm
    and cold must agree exactly -- otherwise the cache would trade GPU time for
    silently different evidence."""
    config = Config()
    embedder = StubEmbedder()
    ledger = Ledger.from_config(config)

    with ledger.query_scope("cold"):
        cold_pool, cold_scores, _, _, cold_ch = runner.stage_c(
            snap, embedder, "what is my weight", "c1", config, ledger, cache=None
        )
    cache = runner.ChannelCache(snap, embedder, config)
    with ledger.query_scope("warm1"):
        runner.stage_c(snap, embedder, "an unrelated first question", "c1", config, ledger, cache=cache)
    with ledger.query_scope("warm2"):
        warm_pool, warm_scores, _, _, warm_ch = runner.stage_c(
            snap, embedder, "what is my weight", "c1", config, ledger, cache=cache
        )

    assert sorted(warm_pool.ids()) == sorted(cold_pool.ids())
    assert warm_scores == cold_scores
    assert warm_ch == cold_ch
    assert cache.builds == 1


def test_the_cached_dense_channel_still_meters_each_question_encode(runner, snap):  # noqa: F811
    """Caching removes the *corpus* encode, which is index construction. It must
    not remove the *question* encode, which is a real per-query GPU cost -- the
    ledger reference is re-pointed on every cache hit for exactly this reason."""
    config = Config()
    cache = runner.ChannelCache(snap, StubEmbedder(), config)
    for i in range(3):
        ledger = Ledger.from_config(config)
        with ledger.query_scope(f"q{i}"):
            runner.stage_c(snap, StubEmbedder(), "q", "c1", config, ledger, cache=cache)
            snapshot = ledger.snapshot()
        assert snapshot["stages"]["stage_c"]["model_forwards"] >= 1, (
            f"query {i} charged no question encode"
        )


def test_run_dir_prefers_stageb_only_when_default_and_present(runner, tmp_path):
    """The eval must never silently read a Stage-A-only log when a Stage-B one
    exists -- and must never override an explicit --run-dir."""
    assert runner.pick_run_dir("somewhere/else", tmp_path) == "somewhere/else"
    assert runner.pick_run_dir("artefacts/locomo", tmp_path) == "artefacts/locomo"
    d = tmp_path / "artefacts" / "locomo_stageb"
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("", encoding="utf-8")
    assert runner.pick_run_dir("artefacts/locomo", tmp_path) == "artefacts/locomo_stageb"


def test_a_nodeless_graph_is_refused_with_the_missing_stage_named(runner):
    """The 19 Aug 2026 integration gap: Stage A writes no nodes, Stage C
    retrieves over nodes, and the fixture-driven join tests could not see the
    missing stand-in stage. A Stage-A-only log must refuse BEFORE 35 minutes of
    reader time produces 1,986 abstentions that read as total system failure."""
    class _Snap:
        def counts(self):
            return {"nodes": 0, "assertions": 3367}

    with pytest.raises(SystemExit, match="locomo_stageb"):
        runner.require_nodes(_Snap())

    class _Ok:
        def counts(self):
            return {"nodes": 12, "assertions": 40}

    runner.require_nodes(_Ok())  # must not raise


# -- per-query cost, end to end on the real read path (19 Aug 2026) -----------


def _drive(runner, snap, config, ledger, qid, question):  # noqa: F811
    """One question through Stage C -> D -> E, exactly as the runner does it."""
    stamp = ReadPathStamp(
        policy_trained=False, gate_source="none", scorer_source="distilled_head",
        token_counter="approx_tokens", ordering="u_shaped_inference_computable",
    )
    with ledger.query_scope(qid):
        pool, atom_scores, report, obligations, _ = runner.stage_c(
            snap, StubEmbedder(), question, "c1", config, ledger
        )
        example = runner.build_example(
            qid, snap, pool, atom_scores, report, obligations,
            frozenset(list(pool.ids())[:1]), StubEmbedder(),
        )
        env = RealEnvironment(example, config, range_samples=0)
        featurizer = RealFeaturizer(
            example, Policy(*RealFeaturizer.dims(), hidden=16), config, delta_d=False
        )
        scorer = HeadScorer(build_head(ATOM_WIDTH, seed=config.seeds[0]), featurizer)
        return answer(
            question, env=env, featurizer=featurizer, scorer=scorer,
            read_fn=lambda e, q: "70kg [c1]", gate_decision=None,
            obligations=obligations, atom_scores=atom_scores,
            rng=np.random.default_rng(config.seeds[0]),
            ledger=ledger, config=config, stamp=stamp, query_id=qid,
            contested_check=False,
        )


def test_cost_per_query_is_flat_across_a_multi_question_run(runner, snap):  # noqa: F811
    """The blocker, end to end: three identical questions on **one** ledger.

    Identical work must cost the same whether it is the first question or the
    third.  Reading cumulative `totals` made the third report three questions'
    spend under a per-query label, and over 1,986 questions that inflates the
    published tokens-per-query figure by roughly a thousandfold -- on
    `CLAUDE.md` §9's own cost axis.
    """
    config = Config()
    ledger = Ledger.from_config(config)
    results = [
        _drive(runner, snap, config, ledger, f"q{i}", "what is my weight")
        for i in range(3)
    ]

    # The premise: the shared ledger really is accumulating underneath.
    running = [r.record.ledger_snapshot["totals"]["model_forwards"] for r in results]
    assert running == sorted(running) and running[-1] > running[0], (
        "the fixture must actually share a ledger, or this proves nothing"
    )

    cost = cost_report(results)
    assert cost["queries_metered"] == 3
    for meter in ("model_forwards_per_query", "llm_calls_per_query",
                  "llm_tokens_total_per_query"):
        stat = cost[meter]
        if stat is None:
            continue
        assert stat["mean"] == pytest.approx(stat["max"]), (
            f"{meter} varies across identical questions: {stat} -- per-query cost "
            "is accumulating across the run"
        )


def test_a_shared_ledger_and_a_per_question_ledger_agree_end_to_end(runner, snap):  # noqa: F811
    """The runner now builds a ledger per question.  That must be a wiring
    detail, not something the reported cost depends on."""
    config = Config()

    shared = Ledger.from_config(config)
    shared_results = [
        _drive(runner, snap, config, shared, f"q{i}", "what is my weight")
        for i in range(3)
    ]
    per_question = [
        _drive(runner, snap, config, Ledger.from_config(config), f"q{i}",
               "what is my weight")
        for i in range(3)
    ]

    a = cost_report(shared_results)
    b = cost_report(per_question)
    for key in ("llm_calls_per_query", "llm_tokens_in_per_query",
                "llm_tokens_out_per_query", "llm_tokens_total_per_query",
                "model_forwards_per_query"):
        assert a[key] == b[key], f"{key} depends on how the runner wired its ledger"


# -- ceilings must resolve gold that retrieval did not return (20 Aug 2026) ---


def test_ceiling_4_serialises_gold_that_is_outside_the_capped_pool(snap):  # noqa: F811
    """The crash that killed the first full run, at question 2.

    ``tier_a_gold`` builds from the snapshot; the retrieval pool is capped at
    ``pool_cap``.  Gold outside the pool is not an error -- it is precisely what
    ceiling 3 measures -- but ceilings 4 and 5 *serialise gold*, so a serializer
    built over the capped pool raised ``KeyError`` on the first question whose
    gold missed the cap.  Latent until Stage-B coverage tripled and pools
    started saturating.

    Scoring it against the capped pool instead would be worse than the crash:
    a retrieval shortfall would be recorded as a packing failure, which is
    `PHASE10_DECISIONS.md` §5 A3 exactly.
    """
    from graft.diagnostics import ceilings as C
    from graft.reader.serialize import ProofSerializer
    from graft.retrieve.pool import eligible_nodes, uncapped_pool
    from graft.schemas import Obligations

    config = Config()
    nodes = eligible_nodes(snap, "c1")
    full, _, _ = uncapped_pool(snap, {n: 1.0 for n in nodes}, config=config, conv_id="c1")

    # A pool that deliberately returns only one atom -- the capped case, forced.
    starved, _, _ = uncapped_pool(
        snap, {nodes[0]: 1.0}, config=config, conv_id="c1"
    )
    gold = sorted(full.ids())
    assert not set(gold) <= set(starved.ids()), "fixture must actually starve the pool"

    out = C.all_ceilings(
        snapshot=snap, conv_id="c1",
        retrieved=sorted(starved.ids()),          # ceiling 3 sees the real shortfall
        gold=gold,
        serializer=ProofSerializer(snap, full, config=config),
        obligations=Obligations(), scores={}, question="what is my weight",
        read_fn=None, gold_answer="70kg", aliases=(), config=config, tier="tier_a",
    )

    assert out["4_packing"]["available"] is True, "packing must be computable"
    assert out["3_candidate"]["ceiling"] < 1.0, (
        "ceiling 3 is where a retrieval shortfall belongs"
    )


def test_a_capped_pool_serializer_still_fails_on_missing_gold(snap):  # noqa: F811
    """The other half: this is a real constraint, not a defensive `try`.

    If ``ProofSerializer`` ever started tolerating unknown atom ids, the fix
    above would be silently unnecessary *and* a genuinely malformed proof would
    serialise -- so the strictness is pinned here.
    """
    from graft.reader.serialize import ProofSerializer
    from graft.retrieve.pool import eligible_nodes, uncapped_pool
    from graft.schemas import Obligations

    config = Config()
    nodes = eligible_nodes(snap, "c1")
    full, _, _ = uncapped_pool(snap, {n: 1.0 for n in nodes}, config=config, conv_id="c1")
    starved, _, _ = uncapped_pool(snap, {nodes[0]: 1.0}, config=config, conv_id="c1")

    with pytest.raises(KeyError):
        ProofSerializer(snap, starved, config=config).serialise(
            sorted(full.ids()), Obligations(), {}, budget=512
        )


# -- run-3 changes (20 Aug 2026) ----------------------------------------------


@pytest.mark.parametrize(
    "generation, abstained, junk, text",
    [
        ("[Audrey]", False, False, "Audrey"),
        ("[2023-05-31]", False, False, "2023-05-31"),
        ("[INSUFFICIENT EVIDENCE]", True, False, "INSUFFICIENT EVIDENCE"),
        ("[c12]", True, True, "c12"),
        ("[ci]", True, True, "ci"),
        ("[?]", True, True, "?"),
        ("[c8][c10]", True, True, "[c8][c10]"),
        ("Oslo, Nairobi [c1]", False, False, "Oslo, Nairobi [c1]"),
    ],
)
def test_clean_answer_handles_every_observed_shape(runner, generation, abstained, junk, text):
    """The eight shapes measured on run 2's rows.

    `[c12]` is the one that broke the first implementation: unwrapping before the
    junk test turned it into the respectable-looking token `c12`.
    """
    got_text, got_abstained, got_junk = runner.clean_answer(generation)
    assert (got_abstained, got_junk) == (abstained, junk)
    assert got_text == text


def test_clean_answer_never_rewrites_a_real_answer(runner):
    """Hygiene reclassifies; it must not touch a string that will be scored."""
    for good in ("London", "Oslo, Nairobi [c1]", "9 October 2022", "pottery, painting"):
        text, abstained, junk = runner.clean_answer(good)
        assert text == good and not abstained and not junk


def test_dates_render_as_day_month_year():
    """Prompt rule 4 says "day month year"; run 2 fed the reader ISO and it
    copied the format it was shown into answers scored against "9 October 2022"."""
    from graft.reader.serialize import format_date

    assert format_date("2022-10-09T00:00:00+00:00") == "9 October 2022"
    assert format_date("2023-05-20T02:21:00Z") == "20 May 2023"
    assert format_date("") == ""
    assert format_date("not-a-date") == "not-a-date"


def test_the_prompt_examples_cannot_collide_with_corpus_answers():
    """Run 2's reader parroted `Rome, Lisbon [c1][c4]` verbatim. The examples are
    now values LoCoMo does not contain, and rule 7 says not to copy them."""
    from graft.reader.pins import PROMPT_TEMPLATE

    for parroted in ("Toronto", "Rome, Lisbon", "12 March 2022"):
        assert parroted not in PROMPT_TEMPLATE
    assert "Never copy an answer from the format examples" in PROMPT_TEMPLATE


def test_raw_turns_are_deterministic_and_fit_the_room(runner, snap):  # noqa: F811
    """The raw tier: same question, same turns, and the block respects its cap."""
    cache = runner.ChannelCache(snap, StubEmbedder(), Config())
    a = runner.top_raw_turns(cache, "c1", "what is my weight", StubEmbedder(), k=3)
    b = runner.top_raw_turns(cache, "c1", "what is my weight", StubEmbedder(), k=3)
    assert [t.turn_id for t in a] == [t.turn_id for t in b], "selection must be stable"
    assert a, "the fixture conversation must yield turns"
    assert cache.turn_builds == 1, "the index must be built once per conversation"

    words = lambda t: len(t.split())
    block, kept = runner.raw_evidence_block(a, words, 10_000)
    assert kept == len(a) and words(block) <= 10_000
    tight, kept_tight = runner.raw_evidence_block(a, words, 25)
    assert kept_tight < len(a), "a small room must drop turns from the tail"
    assert not tight or words(tight) <= 25


def test_raw_turns_carry_no_citable_ids(runner, snap):  # noqa: F811
    """Raw dialogue is context, not evidence. If it carried `[c#]` ids the reader
    could cite text the checker never validated, and citation precision would be
    scored against something `H` never saw."""
    cache = runner.ChannelCache(snap, StubEmbedder(), Config())
    turns = runner.top_raw_turns(cache, "c1", "what is my weight", StubEmbedder(), k=3)
    block, _ = runner.raw_evidence_block(turns, lambda t: len(t.split()), 10_000)
    import re

    assert not re.search(r"\[c\d+\]", block)
    assert "cite only the [c#] claims above" in block


def test_the_evidence_suffix_reaches_the_reader(runner, snap):  # noqa: F811
    """`answer()` serialises internally, so the suffix has to be threaded through
    it rather than prepended by the caller -- otherwise there would be two prompt
    assembly paths and only one of them metered."""
    seen = {}

    def capture(evidence, question):
        seen["evidence"] = evidence
        return "70kg [c1]"

    config = Config()
    ledger = Ledger.from_config(config)
    stamp = ReadPathStamp(
        policy_trained=False, head_trained=True, gate_source="none",
        scorer_source="distilled_head", token_counter="approx_tokens",
        ordering="u_shaped_inference_computable",
        selection="training_free_relevance",
    )
    with ledger.query_scope("q1"):
        pool, atom_scores, report, obligations, _ = runner.stage_c(
            snap, StubEmbedder(), "what is my weight", "c1", config, ledger
        )
        example = runner.build_example(
            "q1", snap, pool, atom_scores, report, obligations,
            frozenset(list(pool.ids())[:1]), StubEmbedder(),
        )
        env = RealEnvironment(example, config, range_samples=0)
        featurizer = RealFeaturizer(
            example, Policy(*RealFeaturizer.dims(), hidden=16), config, delta_d=False
        )
        answer(
            "what is my weight", env=env, featurizer=featurizer,
            scorer=HeadScorer(build_head(ATOM_WIDTH, seed=13), featurizer),
            read_fn=capture, gate_decision=None, obligations=obligations,
            atom_scores=atom_scores, rng=np.random.default_rng(13),
            ledger=ledger, config=config, stamp=stamp, query_id="q1",
            contested_check=False,
            evidence_suffix="MARKER-SUFFIX",
        )
    assert seen["evidence"].endswith("MARKER-SUFFIX")

