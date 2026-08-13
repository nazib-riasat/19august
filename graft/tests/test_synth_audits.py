"""The audits Gate 2 consumes.

Discharges Phase-2 exit criteria 12 (unconstructible-valid-terminal rate is 0),
13 (collisions), 14 (``FAIL`` reachability and dead-end absorption), 15 (``d``
informativeness), 16 (target mass by bucket) and 21 (``graft.synth`` imports no
ML library).
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import math
import pkgutil
import sys
from pathlib import Path

import numpy as np
import pytest

import graft.synth
from graft.synth.audits import (
    closure_audit,
    collision_audit,
    d_informativeness,
    dead_end_absorption,
    fail_reachability,
    jaccard_spread,
    run_audits,
)
from graft.synth.enumerate import reachable_states
from graft.synth.exact import target_distribution


# -- criterion 12 -----------------------------------------------------------


def test_no_valid_terminal_is_unconstructible(main_suite, probe, tuning):
    """Exit criterion 12 — fix F10 as a regression test, on **every** instance.

    The closure rule was chosen over the alternatives partly because it **proves**
    the unconstructible-valid-terminal rate to 0: nodes-first construction always
    works, which turns a required measurement into a regression test.

    The audit enumerates closed subsets from the pool's ``refs`` directly, in
    topological order, and calls batch ``H`` — it never touches ``legal_adds``,
    so it is not agreeing with the thing it audits.

    Criterion 12 says "on every instance"; until 13 Aug 2026 this ran on the
    main suite only, leaving the probe and tuning suites closure-unaudited
    anywhere in the build.  All 30 instances now run.
    """
    for instance in (*main_suite, *probe, *tuning):
        graph = reachable_states(instance, instance.cfg)
        report = closure_audit(instance, graph, instance.cfg)
        assert report["unconstructible_valid_terminals"] == 0, report["examples"]
        assert report["unreachable_closed_subsets"] == 0
        assert report["closed_subsets"] == graph.n_states
        assert not report["blocked_atoms_passing_H"], (
            "an inadmissible atom passed H on its own, so the audit's "
            "'only admissible atoms need enumerating' step does not hold"
        )
        assert report["blocked_atoms"] >= 2


# -- criterion 13 -----------------------------------------------------------


def test_the_collision_rate_is_zero_on_every_instance(main_suite, probe, tuning):
    """Exit criterion 13.  Zero by G3's theorem, with the SA-GFN correction *not*
    applied — the correction is over a state space quotiented by graph
    isomorphism, and ours is over labelled sets and is not quotiented."""
    for instance in (*main_suite, *probe, *tuning):
        graph = reachable_states(instance, instance.cfg)
        report = collision_audit(graph)
        assert report["equivalent_action_collisions"] == 0
        assert report["state_fingerprint_collisions"] == 0


def test_the_proof_is_in_the_module_docstring():
    """Decision 5 requires the reason recorded next to the audit, so a reader
    does not reach for a correction that does not apply."""
    doc = collision_audit.__doc__ or ""
    assert "labelled" in doc and "does not apply" in doc


# -- criterion 14 -----------------------------------------------------------


def test_fail_is_reachable_on_every_instance(main_suite, probe, tuning):
    """Exit criterion 14.  If ``FAIL`` were unreachable, ``STOP``-masking would
    be doing nothing, the ``FAIL`` terminal would be decorative, and fix F3's
    whole construction would be untested."""
    for instance in (*main_suite, *probe, *tuning):
        graph = reachable_states(instance, instance.cfg)
        assert fail_reachability(graph)["reachable_dead_ends"] >= 1


def test_dead_end_absorption_is_late_and_conditional(main_suite):
    """Exit criterion 14, decision 16.

    The band is the **conditional** early share
    ``Σ_{|d| < max_atoms−1} f(d) / Σ_D f(d) <= 0.05``, not the absolute mass: an
    instance whose *every* dead end is early but whose total dead-end mass is
    1e-3 passes the absolute form and fails the conditional one — and the
    conditional one answers the question the audit exists for, which is *where*
    failures happen, not how many.

    Early dead ends mean the ``ADD`` masks are too tight, not that the budget is
    small.  That is the distinction Phase 1 asked Phase 2 to make.
    """
    for instance in main_suite:
        graph = reachable_states(instance, instance.cfg)
        report = dead_end_absorption(instance, graph, instance.cfg)
        assert report["total_dead_end_mass"] > 0.0
        assert report["early_share"] <= instance.spec.max_early_dead_end_share
        profile = report["absorption_by_size"]
        assert len(profile) == instance.cfg.max_atoms + 1
        assert profile[instance.cfg.max_atoms] > 0.0


def test_absorption_is_measured_under_forced_continuation(main_suite):
    """``UniformPolicy`` includes ``STOP`` in its support whenever it is allowed,
    so it stops early and would report a clean profile by construction."""
    from graft.synth.exact import forward_mass
    from graft.synth.policies import ForcedContinuationPolicy, UniformPolicy

    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    forced, _, _ = forward_mass(ForcedContinuationPolicy(), graph)
    uniform, _, _ = forward_mass(UniformPolicy(), graph)
    assert float(forced[graph.dead_ix].sum()) > float(uniform[graph.dead_ix].sum())

    report = dead_end_absorption(instance, graph, instance.cfg)
    assert report["total_dead_end_mass"] == pytest.approx(
        float(forced[graph.dead_ix].sum()), rel=1e-12
    )


# -- criterion 15 -----------------------------------------------------------


def test_d_is_informative_on_every_main_suite_instance(main_suite):
    """Exit criterion 15 — **the most important audit in the phase** (G5).

    Gate 2's decision rule is "L7 beats capacity-matched L6 on exact TV", and L7
    *is* L6 plus ``Δd`` as input features and nothing else.  A null result that
    cannot distinguish "the hypothesis is false" from "the instrument could not
    resolve it" is the worst outcome available, because it looks like an answer.
    """
    for instance in main_suite:
        spec = instance.spec
        graph = reachable_states(instance, instance.cfg)
        report = d_informativeness(instance, graph, instance.cfg)
        assert report["sizes_with_varying_d"] >= spec.min_d_varying_sizes
        assert report["distinct_d_values"] >= spec.min_distinct_d
        assert report["zero_delta_d_structural"] <= spec.max_zero_delta_d
        assert report["zero_delta_d_visitation"] <= spec.max_zero_delta_d


def test_both_delta_d_densities_are_reported_not_just_the_flattering_one(main_suite):
    """G5.  The structural one describes the environment; the visitation-weighted
    one describes the transitions a learner actually samples, which is what
    Gate 2's comparison is made of.  They can differ substantially, and reporting
    only one would be a choice made after seeing them."""
    for instance in main_suite:
        report = instance.meta["bands"]["delta_d"]
        assert math.isfinite(report["zero_delta_d_structural"])
        assert math.isfinite(report["zero_delta_d_visitation"])
    structural = np.array(
        [i.meta["bands"]["delta_d"]["zero_delta_d_structural"] for i in main_suite]
    )
    visitation = np.array(
        [i.meta["bands"]["delta_d"]["zero_delta_d_visitation"] for i in main_suite]
    )
    assert np.any(np.abs(structural - visitation) > 1e-6), (
        "if the two densities were identical there would be no reason to report "
        "both, and the visitation weighting would be doing nothing"
    )


def test_stop_transitions_are_excluded_from_the_delta_d_densities(main_suite):
    """``STOP`` does not change the selected set, so ``Δd = 0`` for every one of
    them by construction; counting them would inflate the zero fraction
    mechanically.  The densities are over ``ADD`` edges only."""
    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    report = d_informativeness(instance, graph, instance.cfg)
    assert report["add_transitions"] == graph.n_edges


def test_d_is_not_a_reparameterised_step_counter(main_suite):
    """If ``d`` were determined by ``|s|``, conditioning on ``Δd`` would be
    conditioning on a step counter and L7 would differ from L6 by nothing of
    substance."""
    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    by_size = d_informativeness(instance, graph, instance.cfg)["distinct_d_by_size"]
    assert sum(1 for v in by_size.values() if v > 1) >= 3


# -- criterion 16 -----------------------------------------------------------


def test_target_mass_is_reported_per_bucket(main_suite):
    """Exit criterion 16.  Mass on the ``neither`` bucket is a **hard band**; the
    alternative mode's >= 1% share is a **diagnostic that changes the claim, not
    an acceptance test that blocks the build** (G10)."""
    alt_shares = []
    for instance in main_suite:
        profile = instance.meta["bands"]["target_mass"]
        mass = profile["mode_mass"]
        assert set(mass) == {"A", "B", "mixed", "neither"}
        assert sum(mass.values()) == pytest.approx(
            1.0 - profile["target_p_fail"], abs=1e-9
        )
        assert mass["neither"] <= instance.spec.max_neither_mass
        assert profile["effective_support"] > 1.0
        assert 0.0 <= profile["top10_mass"] <= 1.0
        assert profile["zero_sufficiency_mass"] >= 0.0
        alt_shares.append(mass["B"])
    # The >= 1% figure is a **diagnostic** (G10, decision 14): on a regenerated
    # suite that falls below it, the build must not block — the write-up narrows
    # to "effectively unimodal" instead.  What IS asserted here is a regression
    # pin on the *frozen* instrument: the frozen main suite measured mode-B mass
    # 0.079–0.101 (PHASE2_DECISIONS.md §4.3), so a value below 1% in this test
    # means the environment changed underneath its fingerprint.  On a deliberate
    # §6b regeneration this pin moves with the instrument.  (Corrected 13 Aug
    # 2026 — the previous message read as the diagnostic acting as a gate,
    # which is exactly the conflation decision 14 forbids.)
    assert min(alt_shares) >= 0.01, (
        "mode-B mass fell below the frozen suite's measured 0.079-0.101: the "
        "instrument changed — regenerate deliberately under §6b and move this "
        "pin. As a *diagnostic*, sub-1% narrows the write-up to 'effectively "
        "unimodal'; it never blocks a build by itself"
    )


def test_modes_are_bucketed_by_completion_not_membership(main_suite):
    """Decision 13.  An earlier draft bucketed on "contains >= 1 atom of ``A*``",
    which would count a single chain-head node plus seven distractors as mode-A
    mass — inflating exactly the number the audit exists to measure, with
    terminals that prove nothing.  A mode is a *finished* proof or it is not a
    mode."""
    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    target = target_distribution(instance, instance.cfg, graph=graph)
    for label, state in zip(target.mode_labels, graph.terminal_ix.tolist()):
        atoms = set(graph.atoms_of(state))
        a, b = instance.template_a <= atoms, instance.template_b <= atoms
        expected = "mixed" if (a and b) else "A" if a else "B" if b else "neither"
        assert label == expected
    # A partial chain is `neither`, however many of its atoms are present.
    partial = sorted(instance.template_a)[:-1]
    state = graph.state_of(partial)
    if state is not None and graph.stop_allowed[state]:
        position = graph.terminal_ix.tolist().index(state)
        assert target.mode_labels[position] == "neither"


def test_source_tiers_are_counted_only_where_the_source_resolves(main_suite):
    """Criterion 17 says ">= 2 distinct source tiers occur **among atoms whose
    ``Source`` resolves**".  A post-build review found the audit counting
    distinct ``source_tier`` *scores* over the whole pool instead.

    Those differ, because ``source_tier`` returns ``default_tier`` when nothing
    resolves and ``unknown`` is itself a tier carrying that same score.  So one
    resolved tier plus a pile of defaulted atoms satisfied a criterion that asks
    for two resolved ones.  Constructed here: exactly one ``Source`` left live.
    """
    from graft.core.resolve import source_tier
    from graft.graphstore import DictGraphSnapshot
    from graft.schemas import PAYLOAD_TIER, Edge
    from graft.synth.audits import _assert_structure, structural_assertions
    from graft.synth.enumerate import BandViolation
    from graft.synth.lattice import LatticeInstance

    instance = main_suite[0]
    cfg = instance.cfg
    default = cfg.source_tiers[cfg.default_tier]
    nodes = [instance.graph.node(n) for n in sorted(instance.graph._nodes)]
    edges = [instance.graph.edge(e) for e in sorted(instance.graph._edges)]

    # Keep the highest-scoring Source live, so its score differs from
    # ``default_tier`` and the *old* value-counting logic would still see two.
    keep = max(
        (n for n in nodes if n.ntype == "Source"),
        key=lambda n: cfg.source_tiers[n.payload[PAYLOAD_TIER]],
    )
    assert cfg.source_tiers[keep.payload[PAYLOAD_TIER]] != default

    doctored_edges = [
        Edge(
            edge_id=e.edge_id,
            etype=e.etype,
            src=e.src,
            dst=e.dst,
            t_created=e.t_created,
            provenance=e.provenance,
            t_invalid="2026-08-08T00:00:00+00:00"
            if (e.etype == "asserted_by" and e.dst != keep.node_id)
            else e.t_invalid,
            superseded_by=e.superseded_by,
        )
        for e in edges
    ]
    snapshot = DictGraphSnapshot(
        snapshot_id=instance.graph.snapshot_id,
        nodes=nodes,
        edges=doctored_edges,
        assertions=[instance.graph.assertion(a) for a in sorted(instance.graph._assertions)],
        turns=[instance.graph.turn(t) for t in sorted(instance.graph._turns)],
        spans=[instance.graph.span(s) for s in sorted(instance.graph._spans)],
    )
    doctored = LatticeInstance(
        pool=instance.pool,
        obligations=instance.obligations,
        graph=snapshot,
        gold=instance.gold,
        template_a=instance.template_a,
        template_b=instance.template_b,
        spec=instance.spec,
        meta=dict(instance.meta),
    )
    graph = reachable_states(doctored, cfg)
    report = structural_assertions(doctored, graph, cfg)

    # What the retired logic saw, and why it passed:
    assert len({source_tier(atom, snapshot, cfg) for atom in doctored.pool}) >= 2
    # What the criterion actually asks:
    assert report["distinct_source_tiers"] == 1
    assert report["atoms_without_resolved_source"] > 0
    with pytest.raises(BandViolation) as exc:
        _assert_structure(report)
    assert exc.value.band == "source_tiers"


def test_the_resolved_and_defaulted_populations_are_both_non_empty(main_suite):
    """The distinction above is live on the real suites, not academic: some atoms
    (Entity, TimeInterval, bindings) have no ``asserted_by`` path at all."""
    for instance in main_suite:
        report = instance.meta["bands"]["structure"]
        assert report["distinct_source_tiers"] >= 2
        assert report["atoms_without_resolved_source"] > 0


def test_the_jaccard_spread_is_descriptive_only(main_suite):
    instance = main_suite[0]
    graph = reachable_states(instance, instance.cfg)
    spread = jaccard_spread(graph)
    assert 0.0 <= spread["jaccard_mean"] <= 1.0
    assert spread["pairs"] > 0


# -- the whole suite --------------------------------------------------------


def test_run_audits_reports_per_instance(main_suite):
    """Gotcha from P2.5: these are consumed by a gate, so they are reported *per
    instance and aggregated*, never a single pooled number that one bad instance
    can hide inside."""
    report = run_audits(main_suite[0])
    for key in (
        "counts",
        "pool_size",
        "environment_fingerprint",
        "closure",
        "collisions",
        "fail",
        "absorption",
        "delta_d",
        "target_mass",
        "jaccard",
        "structure",
    ):
        assert key in report
    assert report["suite"] == "main"


# -- criterion 21 -----------------------------------------------------------

ML_LIBRARIES = (
    "torch",
    "torch_geometric",
    "transformers",
    "sentence_transformers",
    "bitsandbytes",
    "pcst_fast",
    "bm25s",
)


def test_graft_synth_imports_no_ml_library():
    """Exit criterion 21.  Phase 2 builds the environment and the ruler; the
    evaluator is numpy and dictionaries.  If a ``graft/synth/`` file imports
    ``torch``, something has gone wrong.

    **Run in a subprocess since Phase 3 arrived.**  This test originally read
    ``sys.modules`` in-process, which was sound while nothing in the suite
    imported torch.  Phase 3's ``graft.setgen`` does, so an in-process check now
    reports whatever another test happened to import first — it would have gone
    red for a reason that has nothing to do with ``graft.synth``, and worse,
    could have gone *green* while ``graft.synth`` genuinely pulled torch in.  A
    clean interpreter is the only honest way to ask the question.
    """
    root = Path(graft.synth.__file__).parent
    names = [i.name for i in pkgutil.walk_packages([str(root)], prefix="graft.synth.")]
    program = "\n".join(
        [
            "import importlib, sys, json",
            f"for n in {names!r}:",
            "    importlib.import_module(n)",
            f"print(json.dumps(sorted("
            f"l for l in {list(ML_LIBRARIES)!r} if l in sys.modules)))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        cwd=str(Path(graft.__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert not leaked, f"graft.synth imported ML libraries: {leaked}"

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] not in ML_LIBRARIES, f"{path.name}: {module}"


def test_the_phase3_handoff_imports_resolve():
    """§8 of the build plan, verbatim.  It works on the day Phase 2 exits, or the
    handoff is a wish."""
    from graft.synth.audits import run_audits  # noqa: F401
    from graft.synth.enumerate import (  # noqa: F401
        StateGraph,
        reachable_states,
        valid_terminals,
    )
    from graft.synth.exact import (  # noqa: F401
        Target,
        fcs,
        js,
        kl,
        policy_distribution,
        target_distribution,
        tv,
    )
    from graft.synth.lattice import (  # noqa: F401
        LatticeInstance,
        LatticeSpec,
        benchmark_suite,
        generate,
        probe_suite,
        tiny_instance,
        tuning_suite,
    )
    from graft.synth.policies import (  # noqa: F401
        ActionPolicy,
        FlowOraclePolicy,
        UniformPolicy,
    )
