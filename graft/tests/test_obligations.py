"""The obligation deficit vector ``d(s)``, coverage, and interval measure.

Discharges Phase-1 exit criterion 16 (fixed dimension 6, every component in
[0, 1], ``DEFICIT_COMPONENTS`` exported so Phase 3 cannot hard-code an index).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from graft.config import load_config
from graft.core.obligations import (
    DEFICIT_COMPONENTS,
    SLOT_COMPONENTS,
    coverage,
    covered_fraction,
    deficit,
    delta_deficit,
    parse,
    slot_level_scores,
    slot_status,
)
from graft.schemas import Interval, Obligations
from graft.tests.fixtures import build_instance, closed_subsets, instances, random_subsets

CFG = load_config()


@pytest.fixture
def inst():
    return build_instance(random.Random(13))


# -- criterion 16 -----------------------------------------------------------


def test_the_vector_is_six_components_in_a_named_order():
    assert DEFICIT_COMPONENTS == ("anchor", "value", "time", "source", "binding", "closure")
    assert len(DEFICIT_COMPONENTS) == 6


def test_every_component_is_in_range_on_ten_thousand_states():
    rng = random.Random(4)
    checked = 0
    for instance in instances(seed=61, count=8):
        subsets = list(random_subsets(rng, instance.pool, 700, CFG.max_atoms))
        subsets += list(closed_subsets(rng, instance.pool, 600, CFG.max_atoms))
        for subset in subsets:
            d = deficit(subset, instance.pool, instance.obligations, instance.graph)
            assert d.shape == (6,)
            assert np.all((d >= 0.0) & (d <= 1.0)), d
            checked += 1
    assert checked >= 10_000


def test_slot_components_cover_exactly_the_four_obligation_slots():
    """Components 5 and 6 are structural, not slots."""
    assert set(SLOT_COMPONENTS) == {
        "entity_anchor",
        "value_type",
        "time_constraint",
        "needs_source",
    }
    assert set(SLOT_COMPONENTS.values()) == set(DEFICIT_COMPONENTS[:4])


# -- semantics --------------------------------------------------------------


def test_the_empty_set_owes_everything(inst):
    d = deficit([], inst.pool, inst.obligations, inst.graph)
    assert list(d) == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # closure is vacuously satisfied


def test_gold_discharges_everything(inst):
    d = deficit(inst.gold.atoms, inst.pool, inst.obligations, inst.graph)
    assert list(d) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_d_source_is_one_when_nothing_is_bound():
    """Phase-1 gap G4, the fix that repairs G1's arithmetic.

    The obvious phrasing — 'fraction of bindings with no span, 0 when there are
    no bindings' — would score ``needs_source`` as fully covered by binding
    nothing at all.
    """
    instance = build_instance(random.Random(13))
    q = instance.obligations
    assert q.needs_source
    nodes_only = [a.atom_id for a in instance.pool if a.kind == "node"][:4]
    status = slot_status(nodes_only, instance.pool, q, instance.graph)
    assert status["source"] == 1.0


def test_d_source_is_zero_when_the_requirement_is_absent(inst):
    unsourced = Obligations(entity_anchor=inst.obligations.entity_anchor, needs_source=False)
    status = slot_status([], inst.pool, unsourced, inst.graph)
    assert status["source"] == 0.0


def test_d_closure_counts_atoms_waiting_on_refs(inst):
    edge = next(a for a in inst.pool if a.kind == "edge")
    alone = deficit([edge.atom_id], inst.pool, inst.obligations, inst.graph)
    assert alone[DEFICIT_COMPONENTS.index("closure")] == 1.0
    with_refs = deficit(
        {edge.atom_id, *edge.refs}, inst.pool, inst.obligations, inst.graph
    )
    assert with_refs[DEFICIT_COMPONENTS.index("closure")] == 0.0


def test_delta_is_positive_where_an_obligation_was_discharged(inst):
    before = deficit([], inst.pool, inst.obligations, inst.graph)
    after = deficit(inst.gold.atoms, inst.pool, inst.obligations, inst.graph)
    delta = delta_deficit(before, after)
    assert np.all(delta >= 0.0)
    assert delta.sum() > 0


def test_deficit_is_order_invariant(inst):
    forward = deficit(sorted(inst.gold.atoms), inst.pool, inst.obligations, inst.graph)
    backward = deficit(
        sorted(inst.gold.atoms, reverse=True), inst.pool, inst.obligations, inst.graph
    )
    assert np.array_equal(forward, backward)


# -- coverage ---------------------------------------------------------------


def test_coverage_is_binary_per_slot_not_one_minus_the_graded_deficit():
    """Taken graded, ``temporal_correctness`` would be ``1 - d_time`` exactly, so
    ``d_time`` would enter ``U`` twice and time would carry 2-5x the weight of any
    other slot.  Coverage asks *is it addressed*; temporal asks *how precisely*."""
    q = Obligations(entity_anchor="E", time_constraint=Interval(0.0, 100.0))
    partial = {"anchor": 0.0, "value": 0.0, "time": 0.4, "source": 0.0}
    # Graded would give 1 - (0 + 0.4)/2 = 0.8; binary gives 2/2 = 1.0.
    assert coverage(partial, q) == 1.0
    unaddressed = {"anchor": 0.0, "value": 0.0, "time": 1.0, "source": 0.0}
    assert coverage(unaddressed, q) == 0.5


def test_coverage_of_a_question_with_no_active_slots_is_one():
    assert coverage({"anchor": 0.0, "value": 0.0, "time": 0.0, "source": 0.0}, Obligations()) == 1.0


def test_coverage_matches_the_slot_status_of_the_empty_set(inst):
    assert coverage(slot_status([], inst.pool, inst.obligations, inst.graph), inst.obligations) == 0.0


# -- interval measure -------------------------------------------------------


def test_covered_fraction_is_a_proper_ratio():
    c = Interval(0.0, 100.0)
    assert covered_fraction(c, []) == 0.0
    assert covered_fraction(c, [Interval(0.0, 100.0)]) == 1.0
    assert covered_fraction(c, [Interval(0.0, 50.0)]) == pytest.approx(0.5)
    assert covered_fraction(c, [Interval(-50.0, 150.0)]) == 1.0


def test_overlapping_evidence_is_not_double_counted():
    c = Interval(0.0, 100.0)
    assert covered_fraction(c, [Interval(0.0, 60.0), Interval(40.0, 100.0)]) == pytest.approx(1.0)
    assert covered_fraction(c, [Interval(0.0, 30.0), Interval(10.0, 20.0)]) == pytest.approx(0.3)


def test_disjoint_evidence_sums():
    c = Interval(0.0, 100.0)
    assert covered_fraction(c, [Interval(0.0, 20.0), Interval(80.0, 100.0)]) == pytest.approx(0.4)


def test_evidence_outside_the_constraint_contributes_nothing():
    assert covered_fraction(Interval(0.0, 100.0), [Interval(200.0, 300.0)]) == 0.0


def test_an_unbounded_constraint_scores_one_and_is_flagged():
    """Phase-1 gap G5: the measure is infinite, so no ratio is meaningful.  The
    concession is visible via u_terms' ``_temporal_unbounded`` diagnostic and
    Phase 5 must report how often the parser emits these."""
    assert covered_fraction(Interval(start=0.0), [Interval(0.0, 1.0)]) == 1.0
    assert covered_fraction(Interval(end=100.0), []) == 1.0
    assert covered_fraction(Interval(), []) == 1.0


def test_an_empty_time_constraint_is_refused_at_construction():
    """So the ratio never meets a zero denominator."""
    with pytest.raises(ValueError, match="empty interval"):
        Obligations(time_constraint=Interval(start=7.0, end=7.0))


# -- parsing ----------------------------------------------------------------


def test_exact_mode_is_a_lookup(inst):
    assert parse(inst.obligations, mode="exact") is inst.obligations
    assert parse(inst, mode="exact") is inst.obligations


def test_exact_mode_refuses_something_that_carries_no_obligations():
    with pytest.raises(TypeError, match="exact mode needs an Obligations"):
        parse("who did Alice work for in 2019?", mode="exact")


def test_learned_mode_raises_until_phase_five():
    with pytest.raises(NotImplementedError, match="Phase-5 extractor"):
        parse("anything", mode="learned")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="must be 'exact' or 'learned'"):
        parse("anything", mode="guess")


def test_slot_scores_count_wrong_values_on_both_sides():
    gold = [Obligations(entity_anchor="A", value_type="job"), Obligations(entity_anchor="B")]
    predicted = [Obligations(entity_anchor="A", value_type="job"), Obligations(entity_anchor="X")]
    scores = slot_level_scores(predicted, gold)
    assert scores["entity_anchor.precision"] == pytest.approx(0.5)
    assert scores["entity_anchor.recall"] == pytest.approx(0.5)
    assert scores["value_type.precision"] == pytest.approx(1.0)


def test_hallucinated_slots_are_punished():
    """The defect this metric replaced: scoring only where gold was active let a
    parser that invents an anchor on every blank question report 1.0."""
    gold = [Obligations(), Obligations(), Obligations(entity_anchor="A")]
    predicted = [
        Obligations(entity_anchor="GHOST"),
        Obligations(entity_anchor="GHOST"),
        Obligations(entity_anchor="A"),
    ]
    scores = slot_level_scores(predicted, gold)
    assert scores["entity_anchor.precision"] == pytest.approx(1 / 3)
    assert scores["entity_anchor.recall"] == pytest.approx(1.0)


def test_missed_slots_are_punished_by_recall_not_precision():
    gold = [Obligations(entity_anchor="A"), Obligations(entity_anchor="B")]
    predicted = [Obligations(entity_anchor="A"), Obligations()]
    scores = slot_level_scores(predicted, gold)
    assert scores["entity_anchor.precision"] == pytest.approx(1.0)
    assert scores["entity_anchor.recall"] == pytest.approx(0.5)
    assert scores["entity_anchor.f1"] == pytest.approx(2 / 3)


def test_a_slot_nobody_uses_is_nan_not_a_flattering_one():
    scores = slot_level_scores([Obligations()], [Obligations()])
    assert math.isnan(scores["scope.precision"])
    assert math.isnan(scores["scope.recall"])


def test_an_all_wrong_parser_scores_f1_zero_not_nan():
    """An all-wrong parser had F1 = NaN — indistinguishable from an unexercised
    slot, the same metric ambiguity §5.5 was fixed for.  There WAS something to
    score here: precision and recall are both defined and both 0, so F1 is 0.
    (Found by the 13 Aug 2026 audit.)"""
    gold = [Obligations(entity_anchor="A"), Obligations(entity_anchor="B")]
    predicted = [Obligations(entity_anchor="X"), Obligations(entity_anchor="Y")]
    scores = slot_level_scores(predicted, gold)
    assert scores["entity_anchor.precision"] == 0.0
    assert scores["entity_anchor.recall"] == 0.0
    assert scores["entity_anchor.f1"] == 0.0


def test_boolean_slots_are_scored_by_accuracy_including_aggregate():
    """``False`` is a prediction, not an absence, so precision does not fit —
    and ``aggregate`` was omitted from the metric entirely."""
    gold = [Obligations(needs_source=True, aggregate=False), Obligations(aggregate=True)]
    predicted = [Obligations(needs_source=True, aggregate=False), Obligations(aggregate=False)]
    scores = slot_level_scores(predicted, gold)
    assert scores["needs_source.accuracy"] == pytest.approx(1.0)
    assert scores["aggregate.accuracy"] == pytest.approx(0.5)


def test_every_slot_of_obligations_is_scored():
    """A slot added to ``Obligations`` and not to the metric would go unaudited."""
    from graft.core.obligations import BOOLEAN_SLOTS, OPTIONAL_SLOTS

    scored = set(OPTIONAL_SLOTS) | set(BOOLEAN_SLOTS)
    fields = set(Obligations().to_dict())
    assert fields <= scored, f"unaudited obligation slots: {sorted(fields - scored)}"


def test_slot_scores_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="predictions for"):
        slot_level_scores([Obligations()], [])
