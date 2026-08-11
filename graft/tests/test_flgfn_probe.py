"""The FL-GFN discharge (Phase-3 P3.55, decision 24).

The load-bearing test here is **not** that the residual is large on the real
suite — that is the measurement. It is that the fitter can reach **zero** when
the target genuinely lies in the family. A fitter that always returned a large
residual would "prove" the family fails no matter what, which is the shape of a
result that looks like evidence and is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from graft.core import obligations as ob
from graft.setgen.flgfn_probe import (
    POTENTIAL_TERMS,
    design_matrix,
    fit_potential,
    probe_instance,
    probe_suite,
)
from graft.synth.enumerate import reachable_states


# -- the fitter is trustworthy before its verdict is -----------------------


def test_the_fit_reaches_zero_when_the_target_is_in_the_family():
    """Construct a target that *is* a member: `y = features · θ_true`.

    If the solver cannot recover it, every residual it reports elsewhere is
    uninterpretable.
    """
    rng = np.random.default_rng(0)
    theta_true = rng.normal(size=len(POTENTIAL_TERMS))
    blocks = []
    for _ in range(3):
        features = rng.normal(size=(40, len(POTENTIAL_TERMS)))
        blocks.append((features, features @ theta_true))

    fit = fit_potential(blocks)
    assert fit["residual_max_abs"] < 1e-9
    assert fit["terminals_off_tolerance"] == 0
    recovered = np.array([fit["coefficients"][t] for t in POTENTIAL_TERMS])
    assert np.allclose(recovered, theta_true, atol=1e-8)


def test_the_per_instance_constant_is_genuinely_free():
    """An energy is defined up to an additive constant *per instance*, so shifting
    one block's target must not move the residual. Without this freedom the fit
    would charge the family for a constant that changes no distribution."""
    rng = np.random.default_rng(1)
    theta_true = rng.normal(size=len(POTENTIAL_TERMS))
    blocks, shifted = [], []
    for i in range(3):
        features = rng.normal(size=(30, len(POTENTIAL_TERMS)))
        y = features @ theta_true
        blocks.append((features, y))
        shifted.append((features, y + 7.5 * (i + 1)))

    base, moved = fit_potential(blocks), fit_potential(shifted)
    assert moved["residual_max_abs"] < 1e-9
    assert moved["residual_rms"] == pytest.approx(base["residual_rms"], abs=1e-9)
    assert not np.allclose(moved["instance_constants"], base["instance_constants"])


def test_a_target_outside_the_family_leaves_a_residual():
    """The complement of the first test: the fitter must not absorb everything."""
    rng = np.random.default_rng(2)
    features = rng.normal(size=(60, len(POTENTIAL_TERMS)))
    y = (features[:, 0] ** 2) * 3.0 + rng.normal(scale=0.5, size=60)
    fit = fit_potential([(features, y)])
    assert fit["residual_rms"] > 0.1


# -- the design matrix says what it claims ---------------------------------


def test_the_design_matrix_matches_the_plan_formula(bench, bench_graph):
    """`Φ(s) = −Σ ω_j d_j(s) − λ_v|V| − λ_e|E| − λ_b|B|`: the columns are the
    *negated* deficit components and size counts, so a fitted `θ` is read
    directly as the plan's coefficients."""
    features, beta_u = design_matrix(bench, bench_graph, bench.cfg)
    assert features.shape == (bench_graph.n_terminals, len(POTENTIAL_TERMS))
    assert beta_u.shape == (bench_graph.n_terminals,)

    d_cols = len(ob.DEFICIT_COMPONENTS)
    for row, state in enumerate(bench_graph.terminal_ix.tolist()[:25]):
        atoms = bench_graph.atoms_of(state)
        expected = -ob.deficit(atoms, bench.pool, bench.obligations, bench.graph)
        assert np.allclose(features[row, :d_cols], expected)
        kinds = [bench.pool[a].kind for a in atoms]
        assert features[row, d_cols + 0] == -kinds.count("node")
        assert features[row, d_cols + 1] == -kinds.count("edge")
        assert features[row, d_cols + 2] == -kinds.count("binding")


def test_the_binding_term_is_granted_deliberately():
    """Plan §4.5.2's formula predates the three-kind atom schema and names only
    `|V|` and `|E|`. The extra `λ_b` is generosity: another free parameter can
    only *lower* the residual, so a failure with it in place is stronger."""
    assert "lambda_bindings" in POTENTIAL_TERMS
    assert len(POTENTIAL_TERMS) == len(ob.DEFICIT_COMPONENTS) + 3


# -- the measurement -------------------------------------------------------


def test_the_deficit_potential_fails_terminal_identity_on_the_suite(main_suite):
    """Decision 24, the reported discharge.

    FL-GFN requires a scalar energy whose terminal value equals `−log R`. Here
    the identity is `Φ(X) = β·U(X)`, and it fails for **every** member of the
    family — with all coefficients and a free per-instance constant fitted.
    """
    report = probe_suite(main_suite, betas=[4.0, 8.0])
    assert report["n_instances"] == len(main_suite)

    for beta in (4.0, 8.0):
        fit = report["by_beta"][beta]
        assert fit["residual_rms"] > 1e-3, "a near-zero residual would mean the family works"
        assert fit["terminals_off_tolerance"] > 0
        assert fit["n_terminals"] == sum(
            i.meta["bands"]["counts"]["terminals"] for i in main_suite
        )


def test_the_closure_component_fits_to_zero_as_phase_2_predicted(main_suite):
    """A cross-check that the pipeline is wired to the right quantity, not a
    result: `d_closure` is identically 0 on every mask-respecting state (the
    masks enforce refs-before-atom), so its coefficient is unidentifiable and
    least squares returns 0 for it."""
    fit = probe_suite(main_suite[:5])["by_beta"][main_suite[0].cfg.beta]
    assert fit["coefficients"]["omega_closure"] == pytest.approx(0.0, abs=1e-9)


def test_the_residual_scales_with_beta_and_r_squared_does_not(main_suite):
    """`Φ` is fitted against `β·U`, so doubling β doubles the target and the
    residual with it while leaving the *explained fraction* untouched. A
    conclusion that held at one β and not another would be an artefact."""
    report = probe_suite(main_suite[:5], betas=[4.0, 8.0])
    a, b = report["by_beta"][4.0], report["by_beta"][8.0]
    assert b["residual_rms"] == pytest.approx(2.0 * a["residual_rms"], rel=1e-6)
    assert b["r_squared"] == pytest.approx(a["r_squared"], abs=1e-9)


def test_a_single_instance_probe_agrees_with_the_suite_machinery(bench):
    fit = probe_instance(bench)
    assert fit["n_terminals"] == bench.meta["bands"]["counts"]["terminals"]
    assert set(fit["coefficients"]) == set(POTENTIAL_TERMS)


# -- the claim travels with the numbers ------------------------------------


def test_the_named_special_cases_are_reported_and_never_beat_the_best_fit(main_suite):
    """Decision 24 promises named baselines beside the best fit.

    A constrained fit can never beat an unconstrained one over the same design,
    so this is also a check on the pinning machinery: if a pinned fit came out
    *lower*, the columns are not being held where they claim.
    """
    fit = probe_suite(main_suite[:6])["by_beta"][main_suite[0].cfg.beta]
    cases = fit["special_cases"]
    assert set(cases) == {"uniform_omega", "deficits_only"}

    for name, case in cases.items():
        assert fit["residual_rms"] <= case["residual_rms"] + 1e-9, name
        for term, value in case["pinned"].items():
            assert case["coefficients"][term] == pytest.approx(value, abs=1e-12)

    assert set(cases["uniform_omega"]["pinned"]) == set(POTENTIAL_TERMS)
    assert all(t.startswith("lambda_") for t in cases["deficits_only"]["pinned"])


def test_pinning_an_unknown_coefficient_is_refused():
    rng = np.random.default_rng(9)
    block = [(rng.normal(size=(20, len(POTENTIAL_TERMS))), rng.normal(size=20))]
    with pytest.raises(ValueError, match="unknown coefficient"):
        fit_potential(block, {"omega_nonexistent": 1.0})


def test_omega_equals_u_weights_is_not_a_well_defined_choice():
    """The plan originally named `ω = u_weights` as the natural special case.

    It is not definable: the deficit components and the utility weights are
    different vocabularies of the same length, with no correspondence —
    `sufficiency` and `redundancy` have no deficit component, `binding` and
    `closure` have no utility term. Reporting it would have meant inventing a
    mapping and then reporting the number that mapping produced.
    """
    from graft.config import UWeights

    assert len(ob.DEFICIT_COMPONENTS) == len(UWeights().to_dict())
    assert set(ob.DEFICIT_COMPONENTS).isdisjoint(UWeights().to_dict())


def test_the_reported_claim_does_not_overreach(main_suite):
    """Phase-3 G1. Plan §4.5.2 already retired the stronger reading once; the
    string that ships with the numbers must disprove *the potential family* and
    say the complementary limitation is argued, not measured."""
    claim = probe_suite(main_suite[:3])["claim"]
    assert "does NOT" in claim and "inapplicable" in claim
    assert "argued, not measured" in claim
    assert "deficit-potential family" in claim


def test_the_probe_needs_no_ml_library():
    """It depends on nothing Phase 3 builds, which is why it runs at build step 2
    and can be reported whatever happens at Gate 2."""
    import ast
    from pathlib import Path

    import graft.setgen.flgfn_probe as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert "torch" not in names
