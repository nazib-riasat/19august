"""The ProofLattice generator and the three frozen suites.

Discharges Phase-2 exit criteria 9 (the G1 bands), 10 (the pool is built to
spec, asserted directly), 11's suite-generation budget, 12's independent
enumeration, and 17 (the planted structure exists, per instance).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from graft.config import load_config
from graft.core.checker import CHECK_BINDING, CHECK_RETIRED, CHECK_SUPPORT, CHECK_TEMPORAL, H
from graft.core.incremental import IncrementalChecker
from graft.schemas import AtomPool
from graft.synth.audits import POOL_MAX, POOL_MIN, band_report
from graft.synth.enumerate import BandViolation, reachable_states
from graft.synth.lattice import (
    MAIN_SEED,
    PROBE_SEED,
    SUITE_SIZES,
    TUNING_SEED,
    LatticeSpec,
    benchmark_suite,
    generate,
)


# -- the spec ---------------------------------------------------------------


def test_the_spec_carries_the_synthetic_config(main_suite):
    """P2.1.  A generator taking only an ``rng`` would have to reach for a global
    or default the caps, and defaulting to the *real* profile (64/16) instead of
    the synthetic one (32/8) would silently produce an environment nobody can
    enumerate."""
    for instance in main_suite:
        assert instance.cfg.profile == "synthetic"
        assert instance.cfg.pool_cap == 32
        assert instance.cfg.max_atoms == 8


def test_the_real_profile_is_refused():
    with pytest.raises(ValueError, match="synthetic profile"):
        LatticeSpec(load_config(preset="default"))


# -- criterion 10 -----------------------------------------------------------


def test_the_pool_is_built_to_spec(main_suite, probe, tuning):
    """Exit criterion 10.  The state-count band does **not** subsume this: a
    generator that passed the wrong ``cap`` could still happen to produce a small
    graph, and the failure would only surface in Phase 7 against real pools.

    ``AtomPool`` accepts ``cap=None`` by design (``PHASE1_DECISIONS.md`` §5.6), so
    nothing but this assertion stops an uncapped lattice pool.
    """
    for instance in (*main_suite, *probe, *tuning):
        assert instance.pool.cap == instance.cfg.pool_cap == 32
        assert POOL_MIN <= len(instance.pool) <= POOL_MAX


# -- criterion 9 ------------------------------------------------------------


def test_every_benchmark_instance_is_inside_the_G1_bands(main_suite):
    """Exit criterion 9."""
    for instance in main_suite:
        spec = instance.spec
        counts = instance.meta["bands"]["counts"]
        assert spec.min_terminals <= counts["terminals"] <= spec.max_terminals
        assert counts["states"] <= spec.max_states
        assert counts["edges"] <= spec.max_edges


def test_rejection_is_recorded_rather_than_silent(main_suite):
    """G1: instances are generated, enumerated and rejected if out of band, with
    the seed and the rejection count recorded."""
    for instance in main_suite:
        assert instance.meta["attempts"] >= 1
        assert len(instance.meta["rejections"]) == instance.meta["attempts"] - 1
        for rejection in instance.meta["rejections"]:
            assert rejection["band"]
    assert any(i.meta["attempts"] > 1 for i in main_suite), (
        "no instance was ever rejected, so the bands are not constraining the "
        "generator and would not notice if it drifted"
    )


def test_the_generator_raises_rather_than_widening_a_band():
    """G1.  After the attempt limit it raises: a generator that cannot control
    its counts is a defect to fix, not a limit to raise."""
    spec = LatticeSpec(min_terminals=10**9, max_attempts=2)
    with pytest.raises(BandViolation) as exc:
        generate(np.random.default_rng(0), spec)
    assert exc.value.band == "attempts"


def test_the_resident_suite_fits_in_two_gigabytes(main_suite):
    """Exit criterion 11's memory bound — "comfortable under the ``uint64`` state
    representation of G3, and not under a ``frozenset`` one".

    A ``frozenset`` of short strings costs roughly 700 bytes against 8, which is
    the difference between ~0.4 MB and ~70 MB *per instance*.  The dict from mask
    to index is the other real term, so it is counted rather than waved at.
    """
    total = 0
    for instance in main_suite:
        graph = reachable_states(instance, instance.cfg)
        assert graph.mask.dtype == np.uint64
        arrays = (
            graph.mask,
            graph.size,
            graph.edge_parent,
            graph.edge_action,
            graph.edge_child,
            graph.stop_allowed,
            graph.dead_end,
            graph.indegree,
        )
        total += sum(a.nbytes for a in arrays)
        # CPython's dict overhead, generously: ~100 B per int -> int entry.
        total += 100 * graph.n_states
    assert total <= 2 * 1024**3, f"{total / 1024**2:.1f} MB resident for the main suite"


def test_suite_generation_stays_inside_its_budget():
    """Exit criterion 11: total suite generation including rejections <= 30 min.

    Exceeding it is a generator that is not controlling its counts, which is a
    defect to fix rather than a wait to endure.
    """
    start = time.perf_counter()
    benchmark_suite()
    assert (time.perf_counter() - start) <= 1_800.0


# -- the three suites -------------------------------------------------------


def test_the_suites_have_the_declared_sizes_and_seeds(main_suite, probe, tuning):
    """Decision 12.  The generator seeds are *not* one of the training seeds
    ``{13, 42, 7}`` — reusing one would tie the environment to a run's
    randomness."""
    assert (MAIN_SEED, PROBE_SEED, TUNING_SEED) == (20260808, 20260809, 20260811)
    assert not {MAIN_SEED, PROBE_SEED, TUNING_SEED} & set(main_suite[0].cfg.seeds)
    for suite, name in ((main_suite, "main"), (probe, "probe"), (tuning, "tuning")):
        assert len(suite) == SUITE_SIZES[name]
        assert all(i.meta["suite"] == name for i in suite)


def test_the_probe_suite_is_distractor_heavy_and_exempt_from_the_delta_d_band(probe, main_suite):
    """G9.  Its purpose is to check whether a Gate-2 result survives where the
    ``Δd`` signal is sparse, so applying the band there would require it to
    satisfy the condition it exists to violate."""
    for instance in probe:
        assert instance.spec.distractor_heavy
        assert not instance.spec.enforce_delta_d
        assert not instance.spec.enforce_neither_mass
    probe_zero = np.mean(
        [i.meta["bands"]["delta_d"]["zero_delta_d_structural"] for i in probe]
    )
    main_zero = np.mean(
        [i.meta["bands"]["delta_d"]["zero_delta_d_structural"] for i in main_suite]
    )
    assert probe_zero > main_zero, (
        "the probe suite exists to test a sparser Δd signal; if it is not sparser "
        "it is not a robustness check on anything"
    )


def test_the_tuning_suite_matches_the_main_one_except_for_its_seed(main_suite, tuning):
    """β is swept on the tuning suite and applied to the main one, so the two
    must be the same environment family — otherwise that transfer is an unstated
    assumption on top of an already-declared one."""
    a, b = main_suite[0].spec.to_dict(), tuning[0].spec.to_dict()
    a.pop("label"), b.pop("label")
    assert a == b


def test_the_main_and_tuning_suites_are_disjoint_environments(main_suite, tuning):
    """v1.2 §4.1 requires β to be chosen without touching test data."""
    from graft.synth.enumerate import environment_fingerprint

    main_prints = {
        environment_fingerprint(i, reachable_states(i, i.cfg)) for i in main_suite
    }
    tuning_prints = {
        environment_fingerprint(i, reachable_states(i, i.cfg)) for i in tuning
    }
    assert not (main_prints & tuning_prints)


# -- criterion 17 -----------------------------------------------------------


def test_the_planted_structure_exists_on_every_instance(main_suite, probe, tuning):
    """Exit criterion 17, decision 24.

    Nothing else in the criteria list asserts this, and they can all pass on an
    instance whose deliberate mechanisms are broken: criterion 14 needs only
    *one* reachable dead end, which a cap-induced one supplies even if both
    planted failure routes are dead, and a malformed ``P_B`` produces zero
    alternative-mode mass — which G10 treats as a *diagnostic*, so a broken
    instance would be written up as "effectively unimodal" instead of rejected.
    """
    for instance in (*main_suite, *probe, *tuning):
        s = instance.meta["bands"]["structure"]
        assert s["template_A_valid"] and s["template_A_is_terminal"]
        assert s["template_B_valid"] and s["template_B_is_terminal"]
        assert s["templates_differ"]
        assert s["template_overlap"] <= 0.5
        for name, check in (
            ("duplicate_slot_mechanism", CHECK_BINDING),
            ("temporal_disjoint_mechanism", CHECK_TEMPORAL),
        ):
            mech = s[name]
            assert mech["reachable"] and mech["invalid"] and mech["fires_expected_check"]
        assert s["distinct_source_tiers"] >= 2
        assert s["distinct_feature_vectors"] >= 2 and s["featureless_atoms"] == 0
        assert s["constraint_bounded"] and s["partially_overlapping_intervals"] >= 1
        assert not s["atoms_with_unresolved_target"]
        assert s["invalidated_edges_in_snapshot"] >= 1
        assert s["quarantined_assertions_in_snapshot"] >= 1
        assert s["negative_case_atoms"] and not s["negative_case_atoms_reachable"]


def test_the_two_planted_mechanisms_fail_for_different_reasons(main_suite):
    """"Each independently", not "at least one dead end exists"."""
    instance = main_suite[0]
    pool, q, G, cfg = instance.pool, instance.obligations, instance.graph, instance.cfg

    dup = set(instance.meta["duplicate_slot_pair"])
    for aid in tuple(dup):
        dup.update(pool[aid].refs)
    dup_result = H(sorted(dup), q, G, pool, cfg, ledger=None)
    assert not dup_result.ok
    assert CHECK_BINDING in dup_result.categories()
    assert CHECK_TEMPORAL not in dup_result.categories()

    bx = instance.meta["disjoint_binding"]
    disjoint = sorted({bx, *pool[bx].refs})
    dis_result = H(disjoint, q, G, pool, cfg, ledger=None)
    assert not dis_result.ok
    assert CHECK_TEMPORAL in dis_result.categories()
    assert CHECK_BINDING not in dis_result.categories()


def test_the_negative_cases_fail_their_own_sub_checks(main_suite):
    """The retired edge and the quarantined claim are pruned by the masks, so
    they are negative cases for sub-checks 4 and 7 — not selectable evidence."""
    instance = main_suite[0]
    pool, q, G, cfg = instance.pool, instance.obligations, instance.graph, instance.cfg

    retired = instance.meta["retired_edge_atom"]
    result = H([retired], q, G, pool, cfg, ledger=None)
    assert CHECK_RETIRED in result.categories()

    quarantined = instance.meta["quarantined_atoms"][0]
    result = H([quarantined], q, G, pool, cfg, ledger=None)
    assert CHECK_SUPPORT in result.categories()

    chk = IncrementalChecker(pool, q, G, cfg)
    assert not chk.atom_is_admissible(retired)
    assert not chk.atom_is_admissible(quarantined)


# -- tiny_instance ----------------------------------------------------------


def test_the_tiny_instance_is_the_fixture_it_claims_to_be(tiny, tiny_graph):
    assert len(tiny.pool) == 6
    assert tiny.cfg.max_atoms == 3 and tiny.cfg.pool_cap == 6
    assert tiny_graph.n_states == 17
    assert tiny_graph.n_terminals == 15
    assert tiny_graph.n_dead_ends == 1
    # Decision 11: the MC cross-check needs k <= 100 for TV < 0.02 to be a real
    # assertion at N = 200,000.
    assert tiny_graph.n_terminals <= 100


def test_the_tiny_instance_is_exempt_from_the_suite_bands(tiny):
    """It is a fixture for the evaluator, not a Gate-2 environment."""
    assert tiny.spec.label == "tiny"
    assert not tiny.spec.enforce_delta_d


def test_the_tiny_pool_is_capped_like_a_lattice_pool(tiny):
    assert tiny.pool.cap == tiny.cfg.pool_cap


# -- the band machinery itself ----------------------------------------------


def test_band_report_rejects_an_uncapped_pool(main_suite):
    """The `AtomPool(atoms, cap=None)` gotcha, as a rejection rather than a
    comment."""
    from graft.synth.lattice import LatticeInstance

    instance = main_suite[0]
    uncapped = LatticeInstance(
        pool=AtomPool(list(instance.pool), cap=None),
        obligations=instance.obligations,
        graph=instance.graph,
        gold=instance.gold,
        template_a=instance.template_a,
        template_b=instance.template_b,
        spec=instance.spec,
    )
    with pytest.raises(BandViolation) as exc:
        band_report(uncapped)
    assert exc.value.band == "pool_cap"


def test_band_report_rejects_overlapping_templates(main_suite):
    """Decision 23: ``P_A != P_B`` permits a one-atom difference, which is not
    materially different evidence."""
    from graft.synth.lattice import LatticeInstance

    instance = main_suite[0]
    nearly_identical = LatticeInstance(
        pool=instance.pool,
        obligations=instance.obligations,
        graph=instance.graph,
        gold=instance.gold,
        template_a=instance.template_a,
        template_b=instance.template_a | {sorted(instance.pool.ids())[0]},
        spec=instance.spec,
    )
    with pytest.raises(BandViolation) as exc:
        band_report(nearly_identical)
    assert exc.value.band == "template_overlap"
