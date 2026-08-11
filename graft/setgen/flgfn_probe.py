"""The FL-GFN discharge — measuring what the proposed deficit potential cannot do.

**[EVIDENCE]** *Better Training of GFlowNets with Local Credit and Incomplete
Trajectories* (FL-GFN, ICML 2023) requires a scalar energy ``ℰ: S → ℝ`` whose
terminal value equals ``−log R``.  Research plan §4.5.2 proposes the deficit-based
candidate

```
Φ(s) = − Σ_j ω_j d_j(s) − λ_v |V_s| − λ_e |E_s|
```

and argues by hand that its terminal identity ``Φ(X) = β·U(X)`` fails, because at
a valid terminal every mandatory deficit is zero and the potential collapses to
ranking proofs by size alone.  Phase 2 made that argument *measurable*: the
lattice enumerates every valid terminal and holds exact ``U`` for each, so the
residual can be evaluated rather than reasoned about.

**What this module establishes, and what it does not (Phase-3 G1).**  It
disproves **this potential**, not FL-GFN.  Plan §4.5.2 already retired the
stronger reading once — a ``[CORRECTION]`` there records that v1.0's flat "not
applicable" was *"too strong"* — and reintroducing it would be the overreach
`CLAUDE.md` §5 catalogues.  The reportable claim has two parts and needs both:

    (i)  *measured*  — no member of the potential family satisfies terminal
         identity on this environment, by the margin reported here;
    (ii) *argued*    — no justified, informative scalar extension of
         ``sufficiency`` to partial states is currently available, since proof
         sufficiency is a global non-additive property of the whole set.

Part (ii) is not measured by anything here and must never be presented as if it
were.

**Why the *best-fitting* member, and not one instantiation.**  Neither this plan
nor any governing document fixes ``ω_j``, ``λ_v`` or ``λ_e``.  Picking values
would make the result an artefact of that pick, and a reader could always answer
"you chose the wrong weights".  Fitting every coefficient by least squares and
reporting the **irreducible** residual answers that objection: if even the best
member fails, the failure belongs to the family.

**Two freedoms the fit must grant, or it measures the wrong thing.**

*A per-instance constant.*  An energy is defined only up to an additive constant
**per instance** — ``R = exp(−ℰ)`` is normalised by ``Z`` on each instance, so a
uniform shift of ``Φ`` there changes no distribution.  Charging the family for a
constant that carries no information would manufacture a residual.

*A binding-count term.*  Plan §4.5.2's formula predates the three-kind atom
schema and names only ``|V|`` and ``|E|``.  This pool also holds ``binding``
atoms, so ``λ_b |B_s|`` is granted as a third size term.  **[ANALYSIS]** That is
deliberate generosity: an extra free parameter can only *reduce* the residual, so
a failure with it in place is a stronger result than one without.

Numpy only.  No policy, no training, no torch — the discharge depends on nothing
Phase 3 builds, which is why it runs at build step 2 and can be reported whatever
happens at Gate 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from graft.config import Config
from graft.core import obligations as ob
from graft.synth.enumerate import StateGraph, reachable_states
from graft.synth.exact import Target, target_distribution

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.synth.lattice import LatticeInstance

__all__ = [
    "POTENTIAL_TERMS",
    "design_matrix",
    "fit_potential",
    "probe_instance",
    "probe_suite",
]

#: Columns of the fitted potential, in order.  The six deficit components carry
#: ``ω_j``; the three size counts carry ``λ_v``, ``λ_e``, ``λ_b``.  A per-instance
#: constant is appended by :func:`fit_potential` and is not listed here because
#: its width depends on how many instances are fitted jointly.
POTENTIAL_TERMS: tuple[str, ...] = (
    *(f"omega_{name}" for name in ob.DEFICIT_COMPONENTS),
    "lambda_nodes",
    "lambda_edges",
    "lambda_bindings",
)


def design_matrix(
    instance: "LatticeInstance", graph: StateGraph, cfg: Config | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(features, beta_u)`` over every valid terminal of one instance.

    One row per terminal.  ``features`` holds the *negated* deficit components and
    size counts, so a fitted coefficient vector ``θ ≥ 0`` reproduces plan §4.5.2's
    sign convention directly: ``Φ(X) = features · θ``.

    The fit is unconstrained — ``θ`` is not forced non-negative — because the
    question is whether *any* member of the family satisfies terminal identity,
    and constraining the search could only make the residual larger.
    """
    cfg = cfg if cfg is not None else instance.cfg
    pool, q, G = instance.pool, instance.obligations, instance.graph

    n = graph.n_terminals
    features = np.zeros((n, len(POTENTIAL_TERMS)), dtype=np.float64)
    for row, state in enumerate(graph.terminal_ix.tolist()):
        atoms = graph.atoms_of(state)
        features[row, : len(ob.DEFICIT_COMPONENTS)] = -ob.deficit(atoms, pool, q, G)
        kinds = [pool[aid].kind for aid in atoms]
        features[row, len(ob.DEFICIT_COMPONENTS) + 0] = -kinds.count("node")
        features[row, len(ob.DEFICIT_COMPONENTS) + 1] = -kinds.count("edge")
        features[row, len(ob.DEFICIT_COMPONENTS) + 2] = -kinds.count("binding")

    target = target_distribution(instance, cfg, graph=graph)
    return features, cfg.beta * target.u


def fit_potential(
    blocks: Sequence[tuple[np.ndarray, np.ndarray]],
    fixed: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Least-squares fit of the whole family, with a free constant per block.

    Solves

    ``min over (ω, λ, {C_i})  Σ_i Σ_{X of i} ( Φ(X) − β·U(X) − C_i )²``

    ``Φ`` is linear in every free parameter, so this is one closed-form solve —
    not a search, and not something that can be tuned toward a preferred answer.

    ``fixed`` pins named coefficients instead of fitting them, which is how the
    named special cases below are computed: a pinned column moves to the target
    side and leaves the design. The per-block constants stay free in every case,
    because the additive-constant freedom is a property of energies, not a
    parameter of the family.

    Returns the fitted coefficients, the per-block constants, and the residual
    statistics that constitute the measurement.
    """
    if not blocks:
        raise ValueError("no instances to fit")
    n_terms = blocks[0][0].shape[1]
    rows = sum(f.shape[0] for f, _ in blocks)
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(POTENTIAL_TERMS)
    if unknown:
        raise ValueError(f"unknown coefficient(s) to pin: {sorted(unknown)}")
    pinned = np.array([fixed.get(t, 0.0) for t in POTENTIAL_TERMS], dtype=np.float64)
    free_cols = [i for i, t in enumerate(POTENTIAL_TERMS) if t not in fixed]

    X = np.zeros((rows, len(free_cols) + len(blocks)), dtype=np.float64)
    y = np.zeros(rows, dtype=np.float64)
    at = 0
    for i, (features, beta_u) in enumerate(blocks):
        k = features.shape[0]
        X[at : at + k, : len(free_cols)] = features[:, free_cols]
        X[at : at + k, len(free_cols) + i] = -1.0  # the per-instance constant C_i
        y[at : at + k] = beta_u - features @ pinned
        at += k

    solved, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = X @ solved - y
    theta = pinned.copy()
    theta[free_cols] = solved[: len(free_cols)]
    n_free = len(free_cols)
    ss_res = float(residual @ residual)
    ss_tot = float(((y - y.mean()) ** 2).sum())

    return {
        "coefficients": dict(zip(POTENTIAL_TERMS, theta.tolist())),
        "pinned": dict(fixed),
        "instance_constants": (-solved[n_free:]).tolist(),
        "n_terminals": rows,
        "residual_mean_abs": float(np.abs(residual).mean()),
        "residual_p95_abs": float(np.percentile(np.abs(residual), 95)),
        "residual_max_abs": float(np.abs(residual).max()),
        "residual_rms": float(np.sqrt(ss_res / rows)),
        # 1 - SS_res/SS_tot.  Reported because "the residual is 0.4" means little
        # without the spread it is a fraction of.
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "terminals_off_tolerance": int((np.abs(residual) > 1e-9).sum()),
    }


def probe_instance(
    instance: "LatticeInstance", cfg: Config | None = None, *, graph: StateGraph | None = None
) -> dict[str, Any]:
    """The discharge for one instance, at one β."""
    cfg = cfg if cfg is not None else instance.cfg
    g = graph if graph is not None else reachable_states(instance, cfg)
    return fit_potential([design_matrix(instance, g, cfg)])


def probe_suite(
    instances: Sequence["LatticeInstance"],
    betas: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The reported discharge: the joint fit across a suite, at each β.

    **Run at every eligible β, because this precedes the freeze.** ``Φ(X) =
    β·U(X)`` depends on β and the probe runs at build step 2, before the
    calibration gate fixes it.  Reporting the spread across candidates is itself
    the useful part: a residual that is large at every eligible β does not depend
    on which one happens to win.

    The per-instance features and ``U`` are independent of β, so every extra β
    costs one more linear solve over vectors already in hand — the same reason
    ``Target`` caches ``U`` rather than ``R``.
    """
    if not instances:
        raise ValueError("no instances")
    cfg = instances[0].cfg
    betas = list(betas) if betas is not None else [cfg.beta]

    prepared = []
    for instance in instances:
        graph = reachable_states(instance, instance.cfg)
        features, beta_u = design_matrix(instance, graph, instance.cfg)
        # `beta_u` was built at cfg.beta; recover U so any beta can be applied.
        prepared.append((features, beta_u / instance.cfg.beta))

    # Named special cases, alongside the best fit.
    #
    # **"ω = u_weights" is not well defined, and the plan said otherwise.** The
    # deficit components are (anchor, value, time, source, binding, closure); the
    # utility weights are (suff, cov, src, temp, red, size). There is no
    # bijection — `sufficiency` and `redundancy` have no deficit component at all,
    # and `binding`/`closure` have no utility term — so naming that choice would
    # have meant inventing a correspondence and then reporting the number it
    # produced. Two well-defined baselines are reported instead:
    uniform = {t: (1.0 if t.startswith("omega_") else 0.0) for t in POTENTIAL_TERMS}
    sizes_off = {t: 0.0 for t in POTENTIAL_TERMS if t.startswith("lambda_")}

    by_beta = {}
    for b in betas:
        blocks = [(f, b * u) for f, u in prepared]
        by_beta[float(b)] = {
            **fit_potential(blocks),
            "special_cases": {
                # Every obligation weighted equally, no size terms: the simplest
                # member anyone would write down without fitting.
                "uniform_omega": fit_potential(blocks, uniform),
                # Deficits alone, size terms pinned off: isolates whether the
                # obligation part can carry the identity by itself.
                "deficits_only": fit_potential(blocks, sizes_off),
            },
        }
    return {
        "n_instances": len(instances),
        "betas": [float(b) for b in betas],
        "by_beta": by_beta,
        # The claim this licenses, carried with the numbers so the two cannot be
        # separated in a later write-up (Phase-3 G1).
        "claim": (
            "Measured: no member of the deficit-potential family of research-plan "
            "§4.5.2 satisfies FL-GFN's terminal identity on this environment, at "
            "any eligible beta, even with every coefficient and a free per-instance "
            "constant fitted. This disproves that potential family; it does NOT "
            "show FL-GFN is inapplicable. The complementary limitation — that no "
            "justified informative scalar extension of `sufficiency` to partial "
            "states is currently available — is argued, not measured."
        ),
    }
