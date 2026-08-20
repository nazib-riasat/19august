"""Phase 8 — the answerability gate's exit criteria, as tests.

Numbering follows `GRAFT_PHASE8_BUILD.md` §5. Criterion 2 (the gold quarantine),
the torch confinement and criterion 12's fingerprint live in
``test_structure.py``, with the rest of the import-graph rules.

Everything runs on a bare interpreter except the two that train, which need the
ML environment — the Phase-3/6/7 pattern. That split is the point of G9's
staging: the feature contract, the adaptation and the whole selective-prediction
instrument are checkable without torch, and those are the parts the write-up
depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from graft.gate import pins as gpins
from graft.gate.decide import ARM_BLOCKS, arm_mask, decide
from graft.gate.features import BLOCK_FEATURES, block_mask, build_features, feature_names
from graft.gate.riskcov import (
    aurc,
    bootstrap_interval,
    brier,
    choose_threshold,
    contrast_pair_accuracy,
    evaluate,
    expected_calibration_error,
    reweight,
    risk_coverage,
    selective_metrics,
)
from graft.schemas import GateDecision, Interval, Obligations, OutputRecord

DIM = 8  # a small embedding, so the vectors in these tests stay readable


# -- criterion 3: the feature contract --------------------------------------


def test_the_vector_is_the_same_width_and_order_whatever_is_supplied():
    """Exit criterion 3, and the property the ablation rests on.

    The MuSiQue adapter supplies a strict subset of the blocks (G8). If the width
    or order moved with what was available, a column mask computed on one path
    would select different features on the other — and training and inference
    would be reading different objects.
    """
    empty, names_a, flags_a = build_features(embed_dim=DIM)
    full, names_b, flags_b = build_features(
        obligations=Obligations(entity_anchor="x"),
        saturation={"candidates_in_scope": 3, "closed_atoms_in_scope": 5, "pool_cap": 64},
        question_vector=[0.1] * DIM,
        embed_dim=DIM,
    )
    assert empty.shape == full.shape
    assert names_a == names_b
    assert set(flags_a) == set(flags_b) == set(gpins.FEATURE_BLOCKS)
    assert not any(flags_a.values())  # nothing supplied -> every flag cleared


def test_presence_flags_distinguish_absent_from_zero():
    """G8's degenerate-shortcut guard.

    Without flags the gate can learn "slot features are zero ⇒ MuSiQue ⇒ 50%
    unanswerable" — a **dataset** classifier wearing an answerability
    classifier's name. The flag makes absence an explicit input instead of a
    confound.
    """
    absent, names, flags_absent = build_features(embed_dim=DIM)
    present, _, flags_present = build_features(
        obligations=Obligations(), embed_dim=DIM  # a real parse that happens to be empty
    )
    assert flags_absent["slot_coverage"] is False
    assert flags_present["slot_coverage"] is True
    ix = names.index("present.slot_coverage")
    assert absent[ix] == 0.0 and present[ix] == 1.0
    # the *values* are identical; only the flag separates the two cases
    span = len(BLOCK_FEATURES["slot_coverage"])
    assert np.array_equal(absent[:span], present[:span])


def test_a_non_finite_feature_is_refused():
    """A NaN would train to a dead arm and surface as "the model didn't learn"
    rather than as a bad input — the failure hardest to attribute later."""
    with pytest.raises(ValueError, match="non-finite"):
        build_features(question_vector=[float("nan")] * DIM, embed_dim=DIM)


def test_a_mis_sized_question_vector_is_refused():
    with pytest.raises(ValueError, match="frozen so an ablation stays a column mask"):
        build_features(question_vector=[0.0] * (DIM + 1), embed_dim=DIM)


def test_the_arm_mask_is_a_column_mask_over_one_featurisation():
    """Exit criterion 9's "identical budgets", in the strongest available sense:
    the two arms see *the same numbers* in every shared column."""
    names = feature_names(DIM)
    pool_only = arm_mask(names, "pool_only")
    with_question = arm_mask(names, "with_question")
    assert with_question.all()
    assert not any(n.startswith("q_emb_") for n, keep in zip(names, pool_only) if keep)
    assert "present.question_embedding" not in [
        n for n, keep in zip(names, pool_only) if keep
    ]
    # every non-embedding column is shared
    shared = [n for n, keep in zip(names, pool_only) if keep]
    assert all(n in shared for n in names if not n.startswith(("q_emb_", "present.question")))


def test_block_mask_handles_flags_and_embedding_columns():
    names = feature_names(DIM)
    only_shape = block_mask(names, ["pool_shape"])
    selected = [n for n, keep in zip(names, only_shape) if keep]
    assert selected == [f"pool_shape.{f}" for f in BLOCK_FEATURES["pool_shape"]] + [
        "present.pool_shape"
    ]


def test_anchor_coverage_requires_the_entity_to_be_IN_the_pool():
    """**Regression: the feature checked the graph, not the pool.**

    The signed contract is "obligation-slot coverage **by the pool**". As first
    written this called ``match_entities(snapshot, ...)`` and never checked the
    result was retrieved, so an entity the pool had *excluded* still scored
    ``anchor_matched_entity = 1`` — inverting the feature's meaning in exactly the
    case the gate exists for: the graph has it, the pool does not, so the question
    is unanswerable *from this pool*.
    """
    from graft.graphstore import DictGraphSnapshot
    from graft.retrieve.pool import build_pool

    from graft.schemas import Assertion, AssertionFlags, Edge, Node, SourceSpan, Turn

    flags = AssertionFlags(asserted_by="user", entailed_by_span=True, entailed_score=0.9)
    snap = DictGraphSnapshot(
        1,
        [
            Node("n_c", "Claim", {"assertion_id": "a1"}),
            Node("n_e", "Entity", {"name": "my weight", "aliases": [], "conv_id": "c1"}),
        ],
        [Edge("e1", "about_entity", "n_c", "n_e", "2023-05-01T10:00:00Z", ("sp1",))],
        [Assertion("a1", "claim", "my weight is 70kg", ("sp1",), flags, "2023-05-01T10:00:00Z", "eligible")],
        [Turn("t1", "c1", "s1", "user", "2023-05-01T10:00:00Z", "my weight is 70kg")],
        [SourceSpan("sp1", "t1", 0, 17)],
    )
    obligations = Obligations(entity_anchor="my weight", scope=("c1",))
    names = feature_names(DIM)
    ix = names.index("slot_coverage.anchor_matched_entity")

    pool, _, _ = build_pool(snap, {"n_c": 1.0})
    assert any(a.target == "n_e" for a in pool)  # the entity rode along as support
    with_entity, _, _ = build_features(obligations, pool, snapshot=snap, embed_dim=DIM)
    assert with_entity[ix] == 1.0

    # the entity exists in the graph but is absent from an empty pool
    empty, _, _ = build_features(obligations, (), snapshot=snap, embed_dim=DIM)
    assert empty[ix] == 0.0


def test_slot_coverage_reads_the_obligation_and_the_pool():
    """The block is *about* the question's requirements, so an obligation with
    more active slots must move it."""
    _, names, _ = build_features(embed_dim=DIM)
    ix = names.index("slot_coverage.active_slot_count")
    thin, _, _ = build_features(obligations=Obligations(entity_anchor="a"), embed_dim=DIM)
    thick, _, _ = build_features(
        obligations=Obligations(
            entity_anchor="a", value_type="mass", needs_source=True,
            time_constraint=Interval(0.0, 10.0),
        ),
        embed_dim=DIM,
    )
    assert thin[ix] == 1.0 and thick[ix] == 4.0


# -- criteria 5, 6: selective prediction ------------------------------------


def test_aurc_matches_a_hand_computed_toy():
    """Criterion 5's arithmetic, checked against numbers derived by hand.

    Four questions, perfectly ranked: risk is 0, 0, 1/3, 1/2 at coverages
    .25, .5, .75, 1. Every literal here is re-derivable without running the code,
    which is `CLAUDE.md` §6's rule for a test literal.

    **AURC is the arithmetic mean of those four risks** — the standard empirical
    estimator (**[EVIDENCE]** Franc et al., JMLR 2023, Eq. 27) — giving
    ``(0 + 0 + 1/3 + 1/2) / 4 = 0.208333``. This test previously asserted a
    renormalised trapezoid (0.194444), which is not the published definition and
    would have made every AURC here incomparable with every published one.
    """
    curve = risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert [round(v, 3) for v in curve["coverage"]] == [0.25, 0.5, 0.75, 1.0]
    assert [round(v, 3) for v in curve["risk"]] == [0.0, 0.0, 0.333, 0.5]
    assert aurc(curve) == pytest.approx((0.0 + 0.0 + 1 / 3 + 0.5) / 4)
    assert aurc(curve) == pytest.approx(0.208333, abs=1e-6)


def test_the_curve_never_orders_by_the_gold_label():
    """**Regression for a metric that peeked at the answer it was scoring.**

    The first version broke score ties with ``np.lexsort((y, -s))`` — the gold
    label ordering the curve — and, separately, put label 0 first, the opposite of
    what its docstring claimed. Both members of a tied group now carry the group's
    expected error rate, so relabelling a tie cannot move the number.
    """
    a = aurc(risk_coverage([0.5, 0.5], [1, 0]))
    b = aurc(risk_coverage([0.5, 0.5], [0, 1]))
    assert a == b == pytest.approx(0.5)


def test_tied_scores_are_marked_unrealizable():
    """A threshold accepts a tie group whole or not at all, so an intermediate
    coverage point inside one is not something any operating point can produce."""
    curve = risk_coverage([0.5, 0.5], [1, 0])
    assert curve["realizable"] == [False, True]
    # and with no ties every prefix is realizable
    assert risk_coverage([0.9, 0.1], [1, 0])["realizable"] == [True, True]


def test_a_worse_ranker_scores_a_worse_aurc():
    """Lower is better, and the direction is asserted rather than assumed — an
    inverted comparison would make every later claim read backwards."""
    good = aurc(risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]))
    bad = aurc(risk_coverage([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]))
    assert bad > good


def test_reweighting_moves_the_prevalence_and_nothing_else():
    """G4's mechanism. Reweighting rather than resampling means the two curves
    are computed over *identical* model outputs and differ only by the
    prevalence assumption."""
    y = [1, 1, 0, 0]
    w = reweight(y, 0.06)
    assert float(w[2:].sum() / w.sum()) == pytest.approx(0.06)
    assert float(w[:2].sum() / w.sum()) == pytest.approx(0.94)
    # a degenerate split cannot be reweighted and must not divide by zero
    assert np.array_equal(reweight([1, 1, 1], 0.5), np.ones(3))


def test_both_prevalence_curves_ship(monkeypatch):
    """Criterion 6: both curves in the artefact, threshold from the natural one."""
    out = evaluate([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], interval=False)
    assert set(out["curves"]) == set(gpins.PREVALENCES)
    assert set(out["aurc"]) == set(gpins.PREVALENCES)
    assert out["aurc_primary"] == out["aurc"]["natural"]
    assert out["threshold"]["prevalence"] == gpins.PREVALENCES["natural"]


def test_the_threshold_rule_is_the_declared_one():
    """Decision 5: lowest threshold meeting the risk budget at natural prevalence."""
    chosen = choose_threshold([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], target_risk=0.1)
    assert chosen["met"] is True
    assert chosen["weighted_risk"] <= 0.1
    assert chosen["prevalence"] == gpins.PREVALENCES["natural"]


def test_a_target_risk_at_or_above_the_base_rate_is_flagged_vacuous():
    """**Regression for the defect that made every reported operating point
    meaningless.**

    At natural prevalence the trivial gate — answer everything — has weighted risk
    exactly 0.06. An unpinned ``target_risk=0.10`` default was therefore satisfied
    without abstaining on anything, and the first full run selected weighted
    coverage 1.0 on every arm and seed. The pinned target is now below the base
    rate, and a target above it is reported as vacuous rather than silently
    honoured.
    """
    rng = np.random.default_rng(0)
    scores, labels = rng.random(200), (rng.random(200) > 0.5).astype(float)

    assert gpins.TARGET_RISK < gpins.PREVALENCES["natural"]
    assert choose_threshold(scores, labels)["vacuous"] is False

    lax = choose_threshold(scores, labels, target_risk=0.10)
    assert lax["vacuous"] is True
    assert "answering everything already meets the budget" in lax["vacuity_reading"]


def test_the_target_risk_is_pinned_not_a_bare_default():
    """Geifman & El-Yaniv require a *declared* desired risk. A default that lives
    only in a signature is not declared, and is not in the fingerprint either."""
    import inspect

    assert inspect.signature(choose_threshold).parameters["target_risk"].default is None
    assert choose_threshold([0.9, 0.1], [1, 0])["target_risk"] == gpins.TARGET_RISK


def test_an_unmeetable_risk_budget_is_reported_not_silently_met():
    """Returning a threshold that misses the budget is how a risk target stops
    being one."""
    chosen = choose_threshold([0.9, 0.9], [0, 0], target_risk=0.0)
    assert chosen["met"] is False and "reason" in chosen


def test_the_safety_secondary_catches_an_over_abstaining_gate():
    """AURC rewards ranking, and a gate that abstains on everything has no risk
    at zero coverage. The false-abstention rate is what makes that visible —
    which is why decision 6 names it."""
    metrics = selective_metrics([0.1, 0.1, 0.1, 0.1], [1, 1, 0, 0], threshold=0.9)
    assert metrics["coverage"] == 0.0
    assert metrics["false_abstention_rate"] == 1.0  # refused both answerable ones
    assert metrics["abstention_recall"] == 1.0      # and both unanswerable ones


def test_contrast_pair_accuracy_is_the_leakage_immune_metric():
    """**The measurement that corrected this build's own assumption.**

    The runner first asserted that an AURC gap between the arms would indicate a
    masking bug. It would not: AURC is a *global* ranking, so the question
    embedding can improve cross-question calibration without ever separating a
    twin. Within a pair the question is byte-identical, so this metric can only
    be won on pool-side features.
    """
    ids = ["q1", "q1", "q2", "q2"]
    perfect = contrast_pair_accuracy([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0], ids)
    assert perfect["accuracy"] == 1.0 and perfect["pairs_scored"] == 2
    inverted = contrast_pair_accuracy([0.1, 0.9, 0.2, 0.8], [1, 0, 1, 0], ids)
    assert inverted["accuracy"] == 0.0
    # ties count as failures: a gate that cannot order the pair has not decided
    tied = contrast_pair_accuracy([0.5, 0.5], [1, 0], ["q1", "q1"])
    assert tied["accuracy"] == 0.0 and tied["ties"] == 1


def test_malformed_contrast_groups_are_skipped_and_counted():
    out = contrast_pair_accuracy([0.9, 0.8, 0.7], [1, 1, 0], ["q1", "q1", "q2"])
    assert out["pairs_skipped"] == 2 and out["pairs_scored"] == 0


def test_calibration_numbers_exist_and_behave():
    """Decision 9 reports Brier and ECE and adds no calibrator — these two are
    the instrument that would justify one later."""
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.0, 1.0], [1, 0]) == 1.0
    assert expected_calibration_error([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert expected_calibration_error([0.5, 0.5], [1, 0]) == pytest.approx(0.0, abs=1e-9)


def test_the_interval_is_reported_for_small_n():
    """Criterion 7: LongMemEval's 30 must carry an interval, never a point
    estimate — at n = 30 one flipped decision moves a rate by 3.3 points."""
    rng = np.random.default_rng(0)
    scores = rng.random(30)
    labels = (rng.random(30) > 0.5).astype(float)
    out = bootstrap_interval(scores, labels, resamples=200, seed=13)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["resamples"] > 0
    assert "not Dror" in out["method"]  # the [ANALYSIS] label is part of the result


def test_the_interval_describes_the_reported_primary_metric():
    """**Regression: the interval used to bracket a different number entirely.**

    ``evaluate`` declares natural-prevalence AURC as primary while the bootstrap
    defaulted to the *unweighted* one — measured, a primary of 0.036 against an
    interval of [0.362, 0.399], which describes nothing on the page.
    """
    rng = np.random.default_rng(1)
    scores, labels = rng.random(120), (rng.random(120) > 0.5).astype(float)
    out = evaluate(scores, labels, interval=True, seed=13)
    interval = out["aurc_interval"]
    assert interval["point"] == pytest.approx(out["aurc_primary"])
    assert interval["lo"] <= out["aurc_primary"] <= interval["hi"]
    assert "natural prevalence" in interval["statistic"]


def test_the_bootstrap_resamples_contrast_pairs_not_rows():
    """Dror et al. (ACL 2018): choose the test to match the experimental
    structure. The evaluation set is made of contrast **pairs** — a question and
    its twin are not independent observations — so resampling rows splits twins
    across draws and understates the variance."""
    rng = np.random.default_rng(2)
    scores, labels = rng.random(80), np.tile([1.0, 0.0], 40)
    groups = [f"q{i // 2}" for i in range(80)]

    paired = bootstrap_interval(scores, labels, group_ids=groups, resamples=200, seed=13)
    rowwise = bootstrap_interval(scores, labels, resamples=200, seed=13)
    assert paired["unit"] == "contrast pair"
    assert rowwise["unit"] == "row"
    assert paired["point"] == pytest.approx(rowwise["point"])  # same statistic, same data

    with pytest.raises(ValueError, match="group ids"):
        bootstrap_interval(scores, labels, group_ids=groups[:10])


# -- criteria 1, 8: the decision and its cause ------------------------------


def test_decide_is_pure_and_returns_the_frozen_shape():
    """Criterion 1. No model loading, no I/O — Phase 10 wires this unchanged."""
    calls: list[np.ndarray] = []

    def fake_predict(matrix: np.ndarray) -> np.ndarray:
        calls.append(matrix)
        return np.asarray([0.7])

    out = decide(fake_predict, threshold=0.5, arm="pool_only", embed_dim=DIM)
    assert isinstance(out, GateDecision)
    assert out.p_answerable == pytest.approx(0.7)
    assert out.answerable is True
    assert out.threshold == 0.5
    assert out.arm == "pool_only"
    assert out.feature_names and not any(n.startswith("q_emb_") for n in out.feature_names)
    assert calls[0].shape == (1, len(out.feature_names))


def test_a_probability_exactly_at_the_threshold_answers():
    """Declared: the comparison is ``>=``. An undeclared ``>`` would make an
    operating point behave differently at inference than when it was chosen."""
    assert decide(lambda m: [0.5], threshold=0.5, embed_dim=DIM).answerable is True
    assert decide(lambda m: [0.499], threshold=0.5, embed_dim=DIM).answerable is False


def test_the_gate_can_only_produce_the_gate_cause():
    """G5: the two abstain routes stay distinct. ``fallback`` is Stage D's and is
    never set from here."""
    assert decide(lambda m: [0.1], threshold=0.5, embed_dim=DIM).abstain_cause == "gate"
    assert decide(lambda m: [0.9], threshold=0.5, embed_dim=DIM).abstain_cause is None


def test_the_abstain_cause_vocabulary_is_reserved_and_guarded():
    """Criterion 8. The counter exists and stays zero until Phase 9 wires it."""
    assert gpins.ABSTAIN_CAUSES == ("gate", "fallback")
    assert OutputRecord(outcome="abstain", abstain_cause="fallback").abstain_cause == "fallback"
    with pytest.raises(ValueError, match="only an abstention has a cause"):
        OutputRecord(outcome="answer", abstain_cause="gate")
    with pytest.raises(ValueError, match="must be one of"):
        OutputRecord(outcome="abstain", abstain_cause="timeout")
    # records written before Phase 8 read as None rather than failing
    assert OutputRecord.from_dict({"outcome": "answer"}).abstain_cause is None


def test_gate_decision_refuses_a_non_probability():
    with pytest.raises(ValueError, match="must be a probability"):
        GateDecision(p_answerable=1.5, answerable=True, threshold=0.5)


def test_an_unknown_arm_is_refused():
    with pytest.raises(KeyError):
        arm_mask(feature_names(DIM), "whatever")


# -- criterion 12: the frozen surface ---------------------------------------


def test_the_fingerprint_binds_the_actual_feature_contract():
    """**Regression: two different gates used to share one fingerprint.**

    ``frozen_values`` bound only the five block *names*, which the 15 Aug
    decision-3 amendment did not change — so the normalised and raw feature sets
    hashed identically while training different models. It must bind the feature
    names and order, the top-*k* constant and the target risk, because each of
    them changes the numbers a run reports.
    """
    import json

    frozen = gpins.frozen_values()
    blob = json.dumps(frozen)
    assert "bm25_raw_top3" in blob and "dense_raw_max" in blob
    assert frozen["top_k"] == 3
    assert frozen["target_risk"] == gpins.TARGET_RISK
    # the order within a block is part of the contract: a mask is positional
    assert frozen["feature_names"]["channel_scores"][0] == "bm25_raw_max"

    from graft.gate.features import BLOCK_FEATURES

    assert frozen["feature_names"] == {k: list(v) for k, v in sorted(BLOCK_FEATURES.items())}


def test_the_stage_g_fingerprint_is_stable_and_covers_the_decisions():
    first = gpins.stage_g_fingerprint()
    assert first == gpins.stage_g_fingerprint()
    frozen = gpins.frozen_values()
    assert frozen["primary_metric"] == "aurc"
    assert frozen["safety_secondary"] == "false_abstention_rate_on_answerable"
    assert frozen["threshold_rule"] == "dev_risk_coverage_at_natural_prevalence"
    assert frozen["class_handling"] == "class_weights_not_resampling"
    assert frozen["prevalences"] == {"constructed": 0.5, "natural": 0.06}
    # pool_cap belongs to the config tree and must not be duplicated here
    assert "pool_cap" not in frozen


def test_the_arms_are_the_declared_two_and_differ_only_by_the_embedding():
    assert gpins.ARMS == ("pool_only", "with_question")
    assert set(ARM_BLOCKS["with_question"]) - set(ARM_BLOCKS["pool_only"]) == {
        "question_embedding"
    }


# -- criteria 4, 10, 11: the training path (needs torch) --------------------

torch = pytest.importorskip("torch", reason="the gate's models are the only ML piece")


def _toy(n: int = 64, dim: int = 6, seed: int = 0):
    """A separable toy: the label is a threshold on the first column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    y = (x[:, 0] > 0).astype(float)
    return x, y


def test_both_arms_train_and_stay_under_the_parameter_cap():
    """Criterion 2's model half, and decision 2's cap. LR is the control that
    says whether the MLP's capacity bought anything."""
    from graft.gate.model import build_gate, parameter_count, train_gate

    x, y = _toy()
    budget = {**gpins.TRAINING_GATE, "epochs": 6, "early_stop_patience": 3}
    for arm in ("lr", "mlp"):
        model = build_gate(arm, x.shape[1], seed=13)
        assert parameter_count(model) <= gpins.MODELS["max_params"]
        out = train_gate(model, (x, y), (x, y), seed=13, budget=budget)
        assert out["epochs_run"] >= 1
        assert np.isfinite(out["best_dev_loss"])
        assert out["class_balance"]["handling"].startswith("class weights")


def test_an_over_capacity_configuration_is_refused():
    from graft.gate.model import build_gate

    with pytest.raises(ValueError, match="over decision 2's"):
        build_gate("mlp", 100_000, seed=13)


def test_the_seed_reaches_initialisation():
    """Guard one, inherited from P6.11. Seeding only inside the loop leaves every
    seed sharing one random init, so three seeds would estimate variance over
    batch order alone — the defect Phase 6 shipped and its determinism test
    caught."""
    from graft.gate.model import build_gate, predict

    x, _ = _toy()
    a = predict(build_gate("mlp", x.shape[1], seed=13), x)
    b = predict(build_gate("mlp", x.shape[1], seed=13), x)
    c = predict(build_gate("mlp", x.shape[1], seed=42), x)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_training_refuses_when_no_dev_row_is_scorable():
    """Guard three. A gate that never saw a scorable dev row is a *random* gate,
    and reporting it as "early stopped" would put noise into every abstention
    number."""
    from graft.gate.model import build_gate, train_gate

    x, y = _toy()
    with pytest.raises(ValueError, match="no scorable dev item"):
        train_gate(
            build_gate("lr", x.shape[1], seed=13),
            (x, y),
            (np.zeros((0, x.shape[1])), np.zeros(0)),
            seed=13,
            budget={**gpins.TRAINING_GATE, "epochs": 2},
        )


def test_training_restores_the_best_state_rather_than_merely_stopping():
    """Guard two."""
    from graft.gate.model import build_gate, predict, train_gate

    x, y = _toy()
    model = build_gate("lr", x.shape[1], seed=13)
    out = train_gate(
        model, (x, y), (x, y), seed=13,
        budget={**gpins.TRAINING_GATE, "epochs": 8, "early_stop_patience": 2},
    )
    # the restored model must reproduce the reported best dev loss
    scores = predict(model, x)
    assert scores.shape == (x.shape[0],)
    assert 0.0 <= scores.min() and scores.max() <= 1.0
    assert np.isfinite(out["best_dev_loss"])


def test_class_weights_are_reported_and_are_weights_not_resampling():
    """Criterion 10 / G6: the weights are reported, the data is not resampled,
    and no natural-frequency claim may be read off a constructed balance."""
    from graft.gate.model import class_weights

    weight, report = class_weights([1.0, 1.0, 0.0, 0.0])
    assert weight == 1.0  # 1:1 contrast pairs
    assert report["answerable"] == 2 and report["unanswerable"] == 2
    assert "not resampling" in report["handling"]
    assert "natural-frequency" in report["reading"]

    skewed, report = class_weights([1.0] * 9 + [0.0])
    assert skewed == pytest.approx(1 / 9)


def test_the_locomo_prevalence_is_available_but_outside_the_fingerprint():
    """Added 19 Aug 2026. A new evaluation target's base rate changes neither the
    trained model nor any number the MuSiQue Stage-A run reported, so folding it
    into `frozen_values` would move the stage-G fingerprint and retrospectively
    mark `PHASE8_DECISIONS.md` §2's numbers as a different gate's -- churn with no
    measurement behind it.

    It still has to exist: a threshold picked at a 0.06 base rate and applied
    where the rate is 0.2246 is the mistake PREVALENCES exists to prevent, one
    dataset on.
    """
    assert gpins.EVAL_PREVALENCES["locomo"] == 0.2246
    assert "locomo" not in gpins.frozen_values()["prevalences"]
    assert gpins.frozen_values()["prevalences"] == {"constructed": 0.5, "natural": 0.06}
