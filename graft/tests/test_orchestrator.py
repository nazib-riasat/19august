"""Phase 10 Stage C — the read-path orchestrator (fix F7, gaps G7, G8, G12).

Runs entirely on fixtures with a stub reader: the frozen reader is exercised by
``scripts/phase10_read.py``'s run R3, and what this file guards is the wiring,
the accounting and the three conventions Phase 9 transferred here by name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graft.config import Config
from graft.ledger import METERS, Ledger
from graft.reader.orchestrator import (
    ReadPathStamp,
    UnmeteredError,
    UnrankableError,
    aggregate,
    answer,
    cost_report,
)
from graft.schemas import GateDecision, Obligations
from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer
from graft.setgen.distill import HeadScorer, build_head
from graft.setgen.policy import Policy
from graft.setgen.proofs import SourceDoc, build_example
from graft.setgen.realenv import RealEnvironment


def _example(n: int = 4, example_id: str = "q1"):
    docs = [
        SourceDoc(f"p{i}", f"Doc {i}: Ada Lovelace was born in London in 181{i}.", ("London",))
        for i in range(n)
    ]
    return build_example(
        example_id, docs,
        Obligations(entity_anchor="London", scope=(example_id,)),
        {f"p{i}": 1.0 - 0.1 * i for i in range(n)},
    )


@pytest.fixture
def wired():
    """A complete read path with a stub reader and an untrained policy."""
    cfg = Config()
    ex = _example()
    env = RealEnvironment(ex, cfg, range_samples=0)
    torch.manual_seed(13)
    feat = RealFeaturizer(ex, Policy(*RealFeaturizer.dims(), hidden=16), cfg, delta_d=False)
    scorer = HeadScorer(build_head(ATOM_WIDTH, seed=13), feat)
    stamp = ReadPathStamp(
        policy_trained=False, gate_source="musique_placeholder",
        scorer_source="distilled_head", token_counter="approx_tokens",
        ordering="u_shaped_inference_computable",
    )
    return cfg, env, feat, scorer, stamp


def _answer(wired, read_fn=None, gate=None, **kw):
    cfg, env, feat, scorer, stamp = wired
    return answer(
        "Where was Ada Lovelace born?",
        env=env, featurizer=feat, scorer=scorer,
        read_fn=read_fn or (lambda e, q: "London [c1]"),
        gate_decision=gate, rng=np.random.default_rng(3),
        config=cfg, stamp=stamp, **kw,
    )


# --------------------------------------------------------------------------
# the two abstention routes — G7, and never summed
# --------------------------------------------------------------------------


def test_the_gate_route_abstains_with_its_own_cause(wired):
    result = _answer(wired, gate=GateDecision(p_answerable=0.1, answerable=False, threshold=0.6))
    assert result.record.outcome == "abstain"
    assert result.record.abstain_cause == "gate"
    # The gate declined, so Stage D and the reader were never reached.
    assert "stage_d" not in result.stages
    assert result.record.proofset is None


def test_the_fallback_route_abstains_with_a_different_cause(wired):
    """Fix F3's licensed reading: "no valid proof found under this pool, policy,
    attempt count and budget" — never "no proof exists".

    This is the event Phase 8 reserved ``abstain_fallback`` for, and its counter
    has been zero by design until exactly here.
    """
    cfg, ex_env, feat, scorer, stamp = wired
    # `max_atoms = 0` makes every ADD illegal, and the empty set is never a legal
    # terminal (Phase-1 G1), so every rollout dead-ends at the root.
    starved = RealEnvironment(ex_env.example, Config(max_atoms=0), range_samples=0)
    result = answer(
        "q", env=starved, featurizer=feat, scorer=scorer,
        read_fn=lambda e, q: "unreachable", rng=np.random.default_rng(3),
        config=cfg, stamp=stamp,
        gate_decision=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
    )
    assert result.record.outcome == "abstain"
    assert result.record.abstain_cause == "fallback"
    assert result.portfolio.fallback is True


def test_the_two_causes_are_reported_split_and_never_summed(wired):
    """`PHASE5_DECISIONS.md` §1's quarantine-cause lesson, two phases on: two
    different reasons flattened into one rate inflate whichever rate is judged."""
    gate_no = _answer(wired, gate=GateDecision(p_answerable=0.1, answerable=False, threshold=0.6))
    answered = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    agg = aggregate([gate_no, answered])
    assert agg["abstain_by_cause"] == {"gate": 1}
    assert agg["abstained"] == 1
    assert "abstain_rate_total" not in agg, "a single summed rate reappeared"


# --------------------------------------------------------------------------
# G7's aggregation rule
# --------------------------------------------------------------------------


def test_abstentions_are_excluded_from_means_and_counted_separately(wired):
    """§6 decision 7, inherited from `PHASE9_DECISIONS.md` §7.8.

    Imputing 0 makes an abstaining system look like a wrong-answering one;
    dropping the query makes abstention free. The rule does neither.
    """
    gate_no = _answer(wired, gate=GateDecision(p_answerable=0.1, answerable=False, threshold=0.6))
    answered = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))

    only_answered = aggregate([answered])
    with_abstention = aggregate([gate_no, answered])

    # The mean is over answered queries, so adding an abstention must not move it.
    assert with_abstention["mean_proof_size"] == only_answered["mean_proof_size"]
    assert with_abstention["mean_citations"] == only_answered["mean_citations"]
    # But it must be visible.
    assert with_abstention["abstention_rate"] == 0.5
    assert with_abstention["answered"] == 1


def test_an_all_abstention_run_reports_none_rather_than_zero(wired):
    """A mean over zero answered queries is undefined, not 0.0 — reporting 0
    would read as "the system answered badly" rather than "it did not answer"."""
    gate_no = _answer(wired, gate=GateDecision(p_answerable=0.1, answerable=False, threshold=0.6))
    agg = aggregate([gate_no, gate_no])
    assert agg["mean_proof_size"] is None
    assert agg["abstention_rate"] == 1.0


# --------------------------------------------------------------------------
# the vacuous-utility guard — the finding that shaped this module
# --------------------------------------------------------------------------


def test_ranking_without_a_scorer_raises_rather_than_falling_back_to_utility(wired):
    """**Measured while building this module:** ``sufficiency(X, ∅) = 1.0``.

    With no gold, every set scores maximally sufficient, so ``env.utility`` ranks
    on four of its six terms while claiming six. Fix F1 splits train-time ``U``
    from the inference-time head precisely so that cannot happen silently, and a
    loud refusal is the only thing that keeps the split intact against a caller
    in a hurry.
    """
    cfg, env, feat, _scorer, stamp = wired
    with pytest.raises(UnrankableError, match="sufficiency"):
        answer(
            "q", env=env, featurizer=feat, scorer=None,
            read_fn=lambda e, q: "x", rng=np.random.default_rng(3),
            config=cfg, stamp=stamp,
        )


def test_the_vacuous_utility_claim_is_true(wired):
    """The premise of the guard above, asserted rather than asserted-about."""
    from graft.core.utility import sufficiency

    assert sufficiency(("a", "b"), frozenset()) == 1.0


# --------------------------------------------------------------------------
# G8 — the contested comparison, costed
# --------------------------------------------------------------------------


def test_the_contested_comparison_fires_and_is_costed_separately(wired):
    """Fix F4's inference half. The architecture's own words are "costs one
    comparison", and folding that cost into the base per-query number would
    understate the read path on the one axis `CLAUDE.md` §9 says the project must
    win — latency and token cost."""
    cfg, env, feat, scorer, stamp = wired
    ledger = Ledger.from_config(cfg)
    with ledger.query_scope("q1"):
        result = answer(
            "q", env=env, featurizer=feat, scorer=scorer,
            read_fn=lambda e, q: "London [c1]", rng=np.random.default_rng(3),
            ledger=ledger, config=cfg, stamp=stamp,
            gate_decision=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
        )
    assert result.portfolio.distinct_valid >= 2, "fixture returned <2 sets; G8 untested"
    assert result.contested["eligible"] is True
    assert result.contested["extra_reader_calls"] == 1
    # Its own ledger stage, not folded into stage_e.
    assert "contested" in result.record.ledger_snapshot["stages"]


def test_equivalent_answers_from_different_evidence_are_agreement_not_conflict(wired):
    """The comparison is over ANSWERS under decision 9's equivalence, not over
    atom sets. The same answer supported by different evidence is agreement."""
    result = _answer(
        wired, read_fn=lambda e, q: "London [c1]",
        gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
    )
    assert result.contested["contested"] is False
    assert result.record.outcome == "answer"


def test_disagreeing_answers_produce_the_contested_outcome(wired):
    """`OUTCOMES` has carried "contested" since Phase 0 and nothing wrote it
    until now."""
    calls = {"n": 0}

    def alternating(evidence, question):
        calls["n"] += 1
        return "London [c1]" if calls["n"] == 1 else "Paris [c1]"

    result = _answer(
        wired, read_fn=alternating,
        gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
    )
    assert result.contested["contested"] is True
    assert result.record.outcome == "contested"


def test_the_contested_check_can_be_disabled_and_then_costs_nothing(wired):
    result = _answer(
        wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
        contested_check=False,
    )
    assert result.contested["extra_reader_calls"] == 0
    assert result.contested["eligible"] is False


# --------------------------------------------------------------------------
# G12 — the per-query ledger is the only place the cost claim is auditable
# --------------------------------------------------------------------------


def test_every_record_carries_a_per_stage_ledger_snapshot(wired):
    """v1.2 §6.4 asks for latency, context tokens and call counts; `CLAUDE.md` §9
    makes cost a *claimed* axis. A record without a snapshot is a record whose
    cost claim cannot be audited."""
    cfg, env, feat, scorer, stamp = wired
    ledger = Ledger.from_config(cfg)
    with ledger.query_scope("q1"):
        result = answer(
            "q", env=env, featurizer=feat, scorer=scorer,
            read_fn=lambda e, q: "London [c1]", rng=np.random.default_rng(3),
            ledger=ledger, config=cfg, stamp=stamp,
            gate_decision=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
        )
    snapshot = result.record.ledger_snapshot
    assert "stages" in snapshot and "totals" in snapshot
    for stage in ("stage_d", "stage_e"):
        assert stage in snapshot["stages"], f"{stage} not accounted"
        assert "wall_clock_ms" in snapshot["stages"][stage]

    # **Values, not just keys** — the audit's own diagnosis of why the unmetered
    # read path shipped: `Ledger.stage()` initialises every meter to 0, so an
    # entirely unmetered run satisfies a presence check while reporting
    # `llm_tokens_in = 0` for a query that ran a 3-billion-parameter reader.
    # A false zero is worse than a missing value, because it reads as a
    # measurement.
    assert snapshot["stages"]["stage_d"]["model_forwards"] >= 1, (
        "Stage D ran a policy and metered no forward pass"
    )
    # `terminal_checks` stays 0 and MUST: the portfolio constructs through the
    # masks where `stop_allowed` IS `H`, so it owes none (`PHASE9_DECISIONS.md`
    # §1.1). This is a documented true zero, not an unmetered one.
    assert snapshot["stages"]["stage_d"]["terminal_checks"] == 0

    # And the snapshot must reach `report()`, which is the only thing the runner
    # persists — the ledger being wired is not the same as the artefact carrying
    # it, and the first version had the former without the latter.
    assert "ledger_snapshot" in result.report()


def test_a_metered_query_reports_cost_in_the_phase_11_unit(wired):
    """`GRAFT_PHASE11_BUILD.md` G7: the comparison unit is total LLM tokens and
    LLM calls **per query**, read from the ledger rather than from the configured
    budget. A budget is what the packer was allowed; the ledger is what was
    spent, and a proof that packs short makes the two differ."""
    cfg, env, feat, scorer, stamp = wired
    ledger = Ledger.from_config(cfg)
    with ledger.query_scope("q1"):
        result = answer(
            "q", env=env, featurizer=feat, scorer=scorer,
            read_fn=lambda e, q: "London [c1]", rng=np.random.default_rng(3),
            ledger=ledger, config=cfg, stamp=stamp,
            gate_decision=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
        )
    cost = cost_report([result])
    assert cost["queries_metered"] == 1
    assert cost["queries_unledgered"] == 0
    assert cost["llm_tokens_total_per_query"]["mean"] >= 0.0
    assert cost["model_forwards_per_query"]["mean"] >= 1.0
    assert cost["wall_clock_ms_per_query"] is not None
    # Ingestion is an offline per-turn cost and must never appear on this axis.
    assert not any("ingest" in k for k in cost), "ingestion folded into query cost"


def test_a_gate_abstention_is_allowed_to_spend_nothing(wired):
    """The carve-out that makes the guard survivable. :func:`answer` returns on
    the gate route before Stage D and before any reader call, so an all-zero
    snapshot there is *correct*. A guard reading "no all-zero snapshots" would
    fail on correct behaviour, get relaxed, and take the real check with it."""
    cfg, env, feat, scorer, stamp = wired
    ledger = Ledger.from_config(cfg)
    with ledger.query_scope("q1"):
        result = answer(
            "q", env=env, featurizer=feat, scorer=scorer,
            read_fn=lambda e, q: "London [c1]", rng=np.random.default_rng(3),
            ledger=ledger, config=cfg, stamp=stamp,
            gate_decision=GateDecision(p_answerable=0.1, answerable=False, threshold=0.6),
        )
    assert result.record.abstain_cause == "gate"
    cost = cost_report([result])
    assert cost["queries_free_by_gate"] == 1
    assert cost["queries_metered"] == 0


def test_a_wired_ledger_that_spent_nothing_is_refused(wired):
    """The A1 regression guard. `PHASE10_DECISIONS.md` §5 A1: the read path spent
    no meter while ``Ledger.stage()``'s zero-initialised counters made the
    absence read as a measurement of zero. A1 is fixed, but the failure is silent
    by construction -- a regression looks like a cheap query, not a broken
    instrument -- so it is asserted rather than trusted.

    Built by mutating a real record's snapshot to all-zero on a non-gate outcome,
    which is exactly the shape A1 produced.
    """
    result = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    assert result.record.outcome in ("answer", "contested")
    object.__setattr__(
        result.record,
        "ledger_snapshot",
        {"totals": {m: 0 for m in METERS}, "stages": {}},
    )
    with pytest.raises(UnmeteredError, match="A1"):
        cost_report([result])


def test_no_ledger_and_a_zero_ledger_are_different_states(wired):
    """An empty snapshot says *nobody was counting*; an all-zero one says
    *counting happened and came to nothing*. Collapsing them is how A1 stayed
    invisible, so they are reported as separate counts and only one of them is
    fatal here -- the runner owns the other."""
    unledgered = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    assert unledgered.record.ledger_snapshot == {}
    cost = cost_report([unledgered])
    assert cost["queries_unledgered"] == 1
    assert cost["queries_metered"] == 0
    # Not fatal at this layer, and that is deliberate: unit tests legitimately
    # run the read path without a ledger. `scripts/phase10_read.py` refuses it.
    assert cost["llm_calls_per_query"] is None


def test_aggregate_carries_the_cost_block(wired):
    """No summary may reach an artefact without the cost axis. `CLAUDE.md` §9
    claims latency and token cost; A1 was that claim having no numerator."""
    result = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    assert "cost" in aggregate([result])


def test_stage_d_spends_no_terminal_checks(wired):
    """`PHASE9_DECISIONS.md` §1.1: the portfolio constructs through the masks
    where ``stop_allowed`` *is* ``H``, so it is valid by construction and owes no
    terminal check. Charging it would collapse the 0-vs-1 check-family split
    Gate 3's budget row reports."""
    result = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    assert result.portfolio.terminal_checks == 0


# --------------------------------------------------------------------------
# the record's identity, and the honesty stamp
# --------------------------------------------------------------------------


def test_the_record_binds_the_prompt_and_stage_e_fingerprint(wired):
    """v1.2 §3.5's "same frozen reader, same prompt, same budget for every
    compared system" is enforced by hash equality rather than by promise, so the
    hash has to be *in the record* — per record, not per run."""
    from graft.reader.pins import PROMPT_SHA, stage_e_fingerprint

    a = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    b = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    assert a.record.config_hash == b.record.config_hash
    assert len(a.record.config_hash) == 64

    from graft.canonical import digest_of

    expected = digest_of({
        "config": Config().to_dict(),
        "stage_e": stage_e_fingerprint(),
        "prompt_sha": PROMPT_SHA,
    })
    assert a.record.config_hash == expected


def test_an_untrained_artefact_stamps_the_run_as_a_wiring_test(wired):
    """§6 decision 12. `PHASE7_DECISIONS.md` §7 records a `--smoke` artefact
    quoted as a measured result; this is the cheapest possible prevention."""
    result = _answer(wired, gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6))
    assert result.stamp["is_wiring_test"] is True
    assert "WIRING TEST" in result.stamp["warning"]
    assert aggregate([result])["is_wiring_test"] is True


def test_a_fully_trained_stamp_is_not_a_wiring_test():
    stamp = ReadPathStamp(
        policy_trained=True, gate_source="conversational",
        scorer_source="distilled_head", token_counter="reader_tokenizer",
        ordering="u_shaped_inference_computable",
    )
    assert stamp.is_wiring_test is False
    assert "all consumed artefacts" in stamp.to_dict()["warning"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_trained", False),
        ("gate_source", "musique_placeholder"),
        ("scorer_source", "none"),
        ("token_counter", "approx_tokens"),
    ],
)
def test_each_placeholder_alone_is_enough_to_stamp_a_wiring_test(field, value):
    """Any one of them invalidates the run, so the check is a disjunction — not a
    score to be traded off."""
    kw = dict(
        policy_trained=True, gate_source="conversational",
        scorer_source="distilled_head", token_counter="reader_tokenizer",
        ordering="u_shaped_inference_computable",
    )
    kw[field] = value
    assert ReadPathStamp(**kw).is_wiring_test is True


# --------------------------------------------------------------------------
# the reader declining a valid set
# --------------------------------------------------------------------------


def test_a_reader_that_declines_a_valid_set_is_recorded_without_a_new_cause(wired):
    """``ABSTAIN_CAUSES`` is a frozen two-member vocabulary and the orchestrator
    must not widen it from the inside. The reader's own refusal is recorded as a
    fallback with the distinction kept in ``stages``."""
    from graft.reader.pins import INSUFFICIENT

    result = _answer(
        wired, read_fn=lambda e, q: INSUFFICIENT,
        gate=GateDecision(p_answerable=0.9, answerable=True, threshold=0.6),
    )
    assert result.record.outcome == "abstain"
    assert result.record.abstain_cause == "fallback"
    assert result.stages.get("reader_declined") is True
    # The proof set is still recorded: Stage D found one, the reader declined it,
    # and losing the set would lose the evidence for that finding.
    assert result.record.proofset is not None


def test_the_reader_meters_its_own_tokens_and_calls():
    """`CLAUDE.md` §9 makes latency and token cost a *claimed* axis — the project
    may not claim to beat full-context on accuracy and must win here instead — so
    an unmetered reader leaves that claim with no numerator and Phase 11's
    matched-budget comparison with nothing to match against.

    Metering lives **inside the wrapper**, counted where the tokens are, which is
    this project's standing convention (`ingest/extractor.py` lines 505-508,
    `ingest/nli.py` line 168). A caller-side count is a count someone forgets —
    and the first version of `Reader` held no ledger reference at all, so
    `generate` structurally could not meter.
    """
    import inspect

    from graft.reader.read import Reader

    source = inspect.getsource(Reader)
    assert "self.ledger" in source, "Reader holds no ledger; generate cannot meter"
    for meter in ("llm_calls", "model_forwards", "llm_tokens_in", "llm_tokens_out"):
        assert f'count("{meter}"' in source, f"{meter} is never spent by the reader"
    # And the count must be inside generate, not at construction.
    generate_src = inspect.getsource(Reader.generate)
    assert 'count("llm_tokens_in"' in generate_src


# -- per-query cost under one long-lived ledger (19 Aug 2026) -----------------


class _Snapped:
    """A ``ReadResult`` stand-in carrying a real ``Ledger.snapshot()``."""

    def __init__(self, snapshot, outcome="answer", cause=None):
        self.record = type("rec", (), {
            "ledger_snapshot": snapshot, "outcome": outcome, "abstain_cause": cause,
        })()


def _three_queries_on_one_ledger():
    """Three identical queries, one ledger, snapshotted inside each scope --
    exactly what the orchestrator records and what the eval runner used to do."""
    from graft.config import load_config
    from graft.ledger import Ledger

    ledger = Ledger.from_config(load_config(), log=None)
    snapshots = []
    for i in range(3):
        with ledger.query_scope(f"q{i}"):
            ledger.count("llm_calls", 2)
            ledger.count("llm_tokens_in", 500)
            snapshots.append(ledger.snapshot())
    return snapshots


def test_per_query_cost_does_not_accumulate_across_queries():
    """The blocker: ``snapshot()["totals"]`` is cumulative over a ledger's whole
    life, so the n-th query reported the sum of the first n.

    Each query here spends exactly 2 calls and 500 tokens_in.  Reading totals
    gave mean 4.0 / 1000.0 and max 6 / 1500 -- and over LoCoMo's 1,986 questions
    that inflates the published tokens-per-query figure by ~1000x, on the axis
    `CLAUDE.md` §9 names as one this project can honestly win on.
    """
    snapshots = _three_queries_on_one_ledger()

    # The premise, asserted rather than assumed: totals really do accumulate.
    assert [s["totals"]["llm_calls"] for s in snapshots] == [2, 4, 6]
    assert [s["query"]["llm_calls"] for s in snapshots] == [2, 2, 2]

    cost = cost_report([_Snapped(s) for s in snapshots])
    assert cost["queries_metered"] == 3
    assert cost["llm_calls_per_query"]["mean"] == pytest.approx(2.0)
    assert cost["llm_calls_per_query"]["max"] == pytest.approx(2.0)
    assert cost["llm_tokens_in_per_query"]["mean"] == pytest.approx(500.0)
    assert cost["llm_tokens_in_per_query"]["max"] == pytest.approx(500.0)
    assert cost["llm_tokens_total_per_query"]["mean"] == pytest.approx(500.0)


def test_a_fresh_ledger_per_query_reports_the_same_cost():
    """The two conventions must agree, or "cost per query" would depend on how
    the runner happened to wire its ledger rather than on the work done."""
    from graft.config import load_config
    from graft.ledger import Ledger

    fresh = []
    for i in range(3):
        ledger = Ledger.from_config(load_config(), log=None)
        with ledger.query_scope(f"q{i}"):
            ledger.count("llm_calls", 2)
            ledger.count("llm_tokens_in", 500)
            fresh.append(ledger.snapshot())

    shared = _three_queries_on_one_ledger()
    a = cost_report([_Snapped(s) for s in fresh])
    b = cost_report([_Snapped(s) for s in shared])
    for key in ("llm_calls_per_query", "llm_tokens_in_per_query",
                "llm_tokens_total_per_query"):
        assert a[key] == b[key], f"{key} depends on ledger wiring, not on work"


def test_an_unscoped_snapshot_still_falls_back_to_totals():
    """``query`` is ``None`` outside a scope.  Records written without one --
    every unit test that meters directly -- must keep reporting what they did."""
    from graft.config import load_config
    from graft.ledger import Ledger

    ledger = Ledger.from_config(load_config(), log=None)
    ledger.count("llm_calls", 3)
    snapshot = ledger.snapshot()
    assert snapshot["query"] is None

    cost = cost_report([_Snapped(snapshot)])
    assert cost["llm_calls_per_query"]["mean"] == pytest.approx(3.0)


def test_the_gate_route_is_still_free_under_a_scoped_snapshot():
    """A gate abstention spends nothing.  Under the scoped read that must stay
    distinguishable from "nobody was counting", or `UnmeteredError` fires on a
    legitimate record."""
    from graft.config import load_config
    from graft.ledger import Ledger

    ledger = Ledger.from_config(load_config(), log=None)
    with ledger.query_scope("q0"):
        snapshot = ledger.snapshot()

    cost = cost_report([_Snapped(snapshot, outcome="abstain", cause="gate")])
    assert cost["queries_free_by_gate"] == 1
    assert cost["queries_metered"] == 0


def test_a_trained_head_does_not_stamp_the_policy_as_trained():
    """`policy_trained` means the Stage-D sampler; `head_trained` means the
    distilled utility head.  The runner derived the first from the second, so
    loading a trained head claimed the untrained policy was trained -- in the
    honesty stamp, the one field a reader trusts to be conservative."""
    stamp = ReadPathStamp(
        policy_trained=False, head_trained=True, gate_source="none",
        scorer_source="distilled_head", token_counter="reader_tokenizer",
        ordering="u_shaped_inference_computable",
    )
    out = stamp.to_dict()
    assert out["policy_trained"] is False
    assert out["head_trained"] is True
    assert out["is_wiring_test"] is True, "an untrained policy is still a wiring test"


def test_the_head_flag_cannot_clear_the_wiring_test_on_its_own():
    """A trained scorer beside an untrained sampler is exactly the state
    `CLAUDE.md` §7 records -- it must not read as a clean run."""
    trained_head_only = ReadPathStamp(
        policy_trained=False, head_trained=True, gate_source="conversational",
        scorer_source="distilled_head", token_counter="reader_tokenizer",
        ordering="u_shaped_inference_computable",
    )
    assert trained_head_only.is_wiring_test is True

