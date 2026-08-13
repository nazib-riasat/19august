"""Phase 4 — the sixteen exit criteria of ``GRAFT_PHASE4_BUILD.md`` §5.

Organised by the criterion each test discharges, so a reader can check coverage
against the plan rather than against the code.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from graft.core.checker import H
from graft.core.masks import legal_add_ids, stop_allowed
from graft.core.utility import U
from graft.ledger import Ledger, LedgerError
from graft.setgen.search import (
    DIRECT_BUILDERS,
    MASK_DRIVEN,
    SEARCH_METHODS,
    BeamSearch,
    GreedySearch,
    PCSTSearch,
    PortfolioSearch,
    SearchModule,
    SubmodularGreedy,
    build_checker,
    close_under_refs,
    dedup_sets,
    relevance_vector,
)
from graft.setgen.search.base import admissible_atoms, h_filter
from graft.setgen.search.gate3 import (
    best_of_k_ceiling,
    exact_scorer,
    higher_is_better_bootstrap,
    jaccard_diversity,
    run_stage_a,
)
from graft.setgen.search.relevance import RELEVANCE_VARIANTS
from graft.setgen.search.s3_submodular import PAPER_WEIGHTS, SATURATION, SubmodularObjective
from graft.setgen.search.s4_pcst import (
    EDGE_COST_GRID,
    MAX_COMPONENT_ATOMS,
    _prizes_top_k,
    calibrate_edge_cost,
    reference_components,
    solve_pcst_forest,
)
from graft.setgen.trainer import Environment
from graft.synth.lattice import tuning_suite


@pytest.fixture(scope="module")
def tuning_envs():
    return [Environment(i) for i in tuning_suite()]


@pytest.fixture(scope="module")
def env(tuning_envs):
    return tuning_envs[0]


def _methods(env, k=None):
    rel = relevance_vector(env, "obligation")
    k = k or env.instance.cfg.K
    return [
        GreedySearch(rel, k),
        BeamSearch(k=k),
        SubmodularGreedy(rel, k),
        PCSTSearch(rel, 2.0, k),
    ]


def _run(method, env, ledger=None):
    inst = env.instance
    return method.run(env, inst.obligations, exact_scorer(env), ledger)


# -- criterion 1: every returned set passes H -------------------------------


def test_every_returned_set_passes_h(tuning_envs):
    """Criterion 1 — the filter is **asserted**, not assumed, per method per
    instance.

    **The assertion runs with ``ledger=None``**, deliberately: ``checker.H``
    increments ``terminal_checks`` on every call that receives a ledger, so
    verifying through the live ledger would charge the mask-driven arms a check
    each and erase the 0-versus-1 family difference criterion 6 exists to
    expose. Out-of-band verification is what Phase 2's exhaustive enumeration
    already does, for the same reason.
    """
    for env in tuning_envs:
        inst = env.instance
        for method in _methods(env):
            result = _run(method, env)
            for proof in result.sets:
                assert H(
                    proof.atoms, inst.obligations, inst.graph, inst.pool, inst.cfg
                ).ok, f"{method.name} returned an invalid set on {inst.meta.get('seed')}"


# -- criterion 2: closure, by construction and after completion -------------


def test_mask_driven_output_is_closed_by_construction(tuning_envs):
    """Criterion 2, first half: S1 and S2 use the masks, so closure holds without
    a completion step."""
    for env in tuning_envs:
        pool = env.instance.pool
        for method in (_methods(env)[0], _methods(env)[1]):
            for proof in _run(method, env).sets:
                assert set(proof.atoms) == set(close_under_refs(proof.atoms, pool))


def test_direct_builder_output_is_closed_after_completion(tuning_envs):
    """Criterion 2, second half."""
    for env in tuning_envs:
        pool = env.instance.pool
        for method in _methods(env)[2:]:
            for proof in _run(method, env).sets:
                assert set(proof.atoms) == set(close_under_refs(proof.atoms, pool))


def test_a_connected_but_unclosed_pcst_output_is_completed_and_passes_h(env):
    """Criterion 2's named case — **G2's example, as a regression test**.

    A PCST subtree may contain an edge atom and one endpoint: connected under
    Mapping A, *not* closed, because the other endpoint is referenced and
    absent. This is why fix F10's "no conversion logic" is false under the
    mapping Phase 4 adopts, and why completion exists.
    """
    inst = env.instance
    pool = inst.pool
    edge_atom = next(a for a in pool.ids() if len(pool[a].refs) == 2)
    partial = (edge_atom, pool[edge_atom].refs[0])

    # connected under Mapping A ...
    assert len(reference_components(partial, pool)) == 1
    # ... and not closed.
    assert set(partial) != set(close_under_refs(partial, pool))
    # Completion adds exactly the missing referent.
    completed = close_under_refs(partial, pool)
    assert set(completed) == set(partial) | {pool[edge_atom].refs[1]}
    assert H(completed, inst.obligations, inst.graph, pool, inst.cfg).ok


def test_a_gold_set_spanning_two_components_is_reachable(tuning_envs):
    """Criterion 12d's premise: the forest ruling exists because a single tree
    could not reach a gold set that spans components (decision 2, G2).

    **Measured on the graph S4 is actually handed** — the admissible-atom refs
    graph — not on the components *induced on gold alone*. The induced measure
    is 100% on every instance (gold atoms are not mutually reference-linked), so
    a test written against it asserts something that cannot become false and
    could not detect a regression in the ruled premise. Decision 2 states the
    8/20 main-suite figure on the pool and admissible graphs, and those are the
    populations checked here.
    """
    spans = 0
    for env in tuning_envs:
        inst = env.instance
        ground = admissible_atoms(env, inst.obligations)
        components = reference_components(ground, inst.pool)
        holding = {
            i for i, comp in enumerate(components) if set(comp) & set(inst.gold.atoms)
        }
        if len(holding) > 1:
            spans += 1
    assert 1 <= spans < len(tuning_envs), (
        f"gold spans >1 admissible component on {spans}/{len(tuning_envs)} tuning "
        "instances; decision 2's forest ruling rests on this being neither 0 "
        "(nothing to fix) nor universal (a measure that cannot discriminate)"
    )


# -- criterion 3: S3 reproduces the paper's objective -----------------------


def test_s3_uses_the_papers_five_constants_unmodified():
    """Criterion 3 and decision 7. **Five**, not four — the saturation was
    omitted by earlier drafts of the plan and by the architecture."""
    assert PAPER_WEIGHTS == {"rel": 1.0, "qry": 0.5, "cov": 0.4, "div": 0.3}
    assert SATURATION == 0.3


def test_s3_objective_is_monotone_submodular_and_normalised(env):
    """Criterion 3. The paper states each term is *"monotone and submodular,
    normalized to [0, 1]"*; ``F(∅) = 0`` and ``F(V) = Σ weights`` follow, and
    diminishing returns is checked directly on random nested pairs."""
    inst = env.instance
    rel = relevance_vector(env, "obligation")
    obj = SubmodularObjective(inst.pool, inst.obligations, inst.graph, rel)
    ids = list(obj.ids)

    assert obj.F([]) == pytest.approx(0.0)
    assert obj.F(ids) == pytest.approx(sum(PAPER_WEIGHTS.values()))

    rng = np.random.default_rng(4)
    for _ in range(200):
        pick = rng.permutation(len(ids))
        s = sorted(pick[:3].tolist())
        t = sorted(set(s) | set(pick[3:6].tolist()))
        extra = next((int(i) for i in pick[6:] if int(i) not in t), None)
        if extra is None:
            continue
        gain_s = obj.F([ids[i] for i in s + [extra]]) - obj.F([ids[i] for i in s])
        gain_t = obj.F([ids[i] for i in t + [extra]]) - obj.F([ids[i] for i in t])
        assert gain_s >= gain_t - 1e-12, "marginal gain grew as the set grew"


def _toy_objective(rel_values, feats, alpha=SATURATION):
    """A three-atom pool with hand-chosen features, for closed-form arithmetic.

    Built here rather than taken from a lattice instance so the expected values
    below are **literals derived on paper**, not the implementation's own
    internals rearranged. A test that recomputes ``obj.sim``/``obj._repr_total``
    and asserts equality proves only that one expression equals itself — three
    of the four terms of the previous version were in fact `0 == 0`, and
    deleting ``alpha`` entirely passed it.
    """
    from graft.graphstore import DictGraphSnapshot
    from graft.schemas import AtomPool, CandidateAtom, Obligations

    pool = AtomPool(
        [
            CandidateAtom(
                atom_id=f"t{i}", kind="node", target="", feat=np.asarray(f, dtype=np.float32)
            )
            for i, f in enumerate(feats)
        ]
    )
    rel = {f"t{i}": v for i, v in enumerate(rel_values)}
    return SubmodularObjective(
        pool, Obligations(), DictGraphSnapshot(), rel, alpha=alpha
    )


def test_s3_objective_on_a_hand_computable_case():
    """Criterion 3 — every term checked against arithmetic done on paper.

    Three orthogonal unit features, so ``sim`` is the identity: ``deg_i = 1`` for
    every ground element, the saturation cap is ``α = 0.3``, and ``Repr({t0}) =
    min(1, 0.3) = 0.3`` against a maximum of ``3 × 0.3 = 0.9`` — so the
    normalised term is exactly ``1/3``.

    With relevance ``(2, 1, 1)``: ``Rel({t0}) = 2/4 = 0.5``. No obligation slots
    are active, so ``QueryCov ≡ 1`` by the same convention ``coverage`` uses.
    No atom resolves a ``Source``, so all three share one bucket and
    ``Div({t0}) = √2 / √4 = 0.7071``.
    """
    obj = _toy_objective([2.0, 1.0, 1.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert obj.rel_term([0]) == pytest.approx(0.5)
    assert obj.qry_term([0]) == pytest.approx(1.0)
    assert obj.repr_term([0]) == pytest.approx(1 / 3)
    assert obj.div_term([0]) == pytest.approx(np.sqrt(2.0) / 2.0)

    expected = 1.0 * 0.5 + 0.5 * 1.0 + 0.4 * (1 / 3) + 0.3 * (np.sqrt(2.0) / 2.0)
    assert obj.F(["t0"]) == pytest.approx(expected)
    assert obj.F([]) == pytest.approx(0.0)
    assert obj.F(["t0", "t1", "t2"]) == pytest.approx(sum(PAPER_WEIGHTS.values()))


def test_querycov_covers_exactly_the_slots_an_atom_addresses():
    """The post-fix audit's surviving MAJOR: no test exercised ``QueryCov``'s
    covers predicate — ``_toy_objective`` uses ``Obligations()`` (no active
    slots), so inverting ``< 1.0`` to ``>= 1.0`` in ``covers`` passed 42/42
    while silently gutting S3 on real instances (distinct 2 → 0 on tuning[0]).

    This fixture has one active slot and two atoms with known ground truth: an
    Entity atom whose target's name **is** the anchor (covers the slot), and a
    bare atom with no target (covers nothing). The precondition block validates
    the fixture through ``slot_status`` — the project's single implementation of
    "addresses a slot" — and the assertions are then literals, so an inverted
    predicate flips both and dies.
    """
    from graft.graphstore import DictGraphSnapshot
    from graft.schemas import AtomPool, CandidateAtom, Node, Obligations

    G = DictGraphSnapshot(
        nodes=[Node(node_id="n1", ntype="Entity", payload={"name": "alice"})]
    )
    pool = AtomPool(
        [
            CandidateAtom(
                atom_id="hit", kind="node", target="n1",
                feat=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            CandidateAtom(
                atom_id="miss", kind="node", target="",
                feat=np.asarray([0.0, 1.0], dtype=np.float32),
            ),
        ]
    )
    q = Obligations(entity_anchor="alice")

    # Fixture validation, through the core's own slot machinery — if these
    # fail, the fixture is wrong, not the objective.
    from graft.core.obligations import slot_status

    assert slot_status(["hit"], pool, q, G)["anchor"] == 0.0
    assert slot_status(["miss"], pool, q, G)["anchor"] == 1.0

    obj = SubmodularObjective(pool, q, G, {"hit": 1.0, "miss": 1.0})
    assert obj.qry_term([obj.index["hit"]]) == pytest.approx(1.0)
    assert obj.qry_term([obj.index["miss"]]) == pytest.approx(0.0)
    assert obj.qry_term([]) == pytest.approx(0.0)
    assert obj.qry_term([obj.index["miss"], obj.index["hit"]]) == pytest.approx(1.0)


def test_stage_a_diversity_is_computed_over_the_multiplicity_portfolio(tuning_envs):
    """The post-fix audit's second MAJOR: reverting ``run_stage_a`` to
    ``jaccard_diversity(result.sets)`` — the review round's headline F1 defect —
    passed 42/42, because the pinned-convention test was unit-level and nothing
    guarded the **call site**.

    S1 is the discriminating arm: its portfolio carries duplicates (openers
    funnel to ~2.45 distinct sets out of ~7 successful runs), so the portfolio
    and deduplicated diversities genuinely differ, and the summary must carry
    the portfolio one.
    """
    env = tuning_envs[0]
    result = _run(_methods(env)[0], env)
    assert len(result.portfolio) > result.distinct_valid, (
        "S1's portfolio no longer carries multiplicity, so this guard tests nothing"
    )
    over_portfolio = jaccard_diversity(result.portfolio)
    over_dedup = jaccard_diversity(result.sets)
    assert over_portfolio < over_dedup, "duplicates must lower the score"

    summary = run_stage_a([env], variant="obligation", edge_cost=2.0).summary()
    s1 = summary["methods"]["s1_greedy"]
    assert s1["diversity_mean"] == pytest.approx(over_portfolio)
    assert s1["diversity_mean"] != pytest.approx(over_dedup)
    assert s1["portfolio_size_mean"] == pytest.approx(len(result.portfolio))


def test_the_saturation_parameter_is_load_bearing():
    """Decision 7's named risk — *"α is silently defaulted"* — as a test.

    With orthogonal features, ``Repr({t0}) = min(1, α)`` normalised by ``3α``.
    At ``α = 0.3`` that is ``1/3``; at ``α = 1.0`` the cap stops binding and the
    term is still ``1/3`` for the *singleton*, but the **pair** separates them:
    ``min(1, α)`` saturates two elements at ``α`` and the third at 0, so
    ``Repr({t0,t1}) = 2/3`` either way — the discriminating case is a
    *duplicate*, which is what saturation exists to refuse.
    """
    dup = [[1, 0], [1, 0], [0, 1]]  # t0 and t1 identical
    tight = _toy_objective([1.0, 1.0, 1.0], dup, alpha=0.3)
    loose = _toy_objective([1.0, 1.0, 1.0], dup, alpha=1.0)

    # One of a near-duplicate pair, versus both: under saturation the second
    # copy buys strictly less than it does without.
    tight_gain = tight.repr_term([0, 1]) - tight.repr_term([0])
    loose_gain = loose.repr_term([0, 1]) - loose.repr_term([0])
    assert tight_gain < loose_gain, (
        "saturation is not restraining near-duplicates, so alpha is inert — "
        "which is decision 7's named risk"
    )
    assert tight.F(["t0"]) != pytest.approx(loose.F(["t0"]))


def test_the_singleton_fallback_is_inert_at_this_cap_but_not_by_proof(tuning_envs):
    """Criterion 3 / build step 4 — **the constructed unit case it demands**.

    The Lin–Bilmes fallback fires only when the best feasible singleton
    **strictly** outscores the greedy set. It is inert at ``max_atoms = 8``, and
    an earlier draft called that *provable*: the chain's first pick would be the
    argmax singleton, so monotonicity would bound ``F(greedy)`` below.

    **The premise is false.** ``_chain`` starts from a *forced opener*, so the
    argmax coincidence holds on a minority of chains — and at a smaller budget
    the fallback demonstrably fires. Both halves are asserted here, so the test
    can tell "inert" from "absent": deleting the fallback block passes the first
    assertion and fails the second.
    """
    fired_at_cap = 0
    for env in tuning_envs:
        fired_at_cap += _run(_methods(env)[2], env).extra["singleton_fallbacks"]
    assert fired_at_cap == 0, "inert at the frozen cap, as measured"

    # The constructed case: the same shipped `_chain`, at a smaller budget.
    env = tuning_envs[0]
    inst = env.instance
    rel = relevance_vector(env, "obligation")
    ground = admissible_atoms(env, inst.obligations)
    obj = SubmodularObjective(inst.pool, inst.obligations, inst.graph, rel, ground=ground)
    method = SubmodularGreedy(rel)
    fired_small = sum(
        int(method._chain(obj, 3, opener)[1])
        for opener in sorted(obj.ids, key=lambda a: (-float(obj.rel[obj.index[a]]), a))[
            : inst.cfg.K
        ]
    )
    assert fired_small > 0, (
        "the fallback never fires at budget 3 either, so this test cannot "
        "distinguish an inert fallback from a deleted one — re-derive the case"
    )


# -- criterion 4: S5 loads without the trainer ------------------------------


def test_s5_imports_no_trainer():
    """Criterion 4 and Phase-3 §8 requirement 1, as an import-graph assertion."""
    import ast
    import pathlib

    source = pathlib.Path("graft/setgen/search/s5_portfolio.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "graft.setgen.trainer" not in imported
    assert "graft.setgen.policy" in imported


def test_s5_is_declared_stochastic_and_the_rest_deterministic():
    """Decision 4: S1–S4 are deterministic under decision 3's generation rules
    and reported once; only S5's seed varies anything."""
    kinds = {cls.name: cls.deterministic for cls in SEARCH_METHODS}
    assert kinds == {
        "s1_greedy": True,
        "s2_beam": True,
        "s3_submodular": True,
        "s4_pcst": True,
        "s5_portfolio": False,
    }


def test_every_method_satisfies_the_protocol(env):
    """P4.1: every method carries the protocol's surface.

    Checked on an **instance**, not with ``issubclass``: ``SearchModule`` has
    non-method members (``name``, ``deterministic``), and ``issubclass`` against
    such a Protocol raises by design in Python 3.11. ``isinstance`` on a
    ``runtime_checkable`` Protocol is the supported form and is what makes
    registry membership a test rather than a promise.
    """
    rel = relevance_vector(env, "obligation")
    built = [
        GreedySearch(rel),
        BeamSearch(),
        SubmodularGreedy(rel),
        PCSTSearch(rel, 2.0),
        PortfolioSearch("nonexistent.pt"),
    ]
    assert [type(m) for m in built] == list(SEARCH_METHODS)
    for method in built:
        assert isinstance(method, SearchModule)
        assert isinstance(method.name, str)
        assert isinstance(method.deterministic, bool)


# -- criteria 5, 6: the budget is the same constant, enforced, and split ----


def test_k_and_checker_budget_are_read_from_config_not_restated(env):
    """Criterion 5, fix F5: one constant, everywhere."""
    cfg = env.instance.cfg
    assert cfg.K == 8 and cfg.checker_budget == 32
    result = _run(GreedySearch(relevance_vector(env, "obligation")), env)
    assert result.attempted <= cfg.K


def test_the_spend_differs_by_family_and_is_zero_for_mask_driven(tuning_envs):
    """Criterion 6 and G6 — **the finding, not an accounting detail**.

    ``masks.stop_allowed`` *is* ``H``, so a terminal reached through the masks is
    valid by construction and has nothing left to validate: S1/S2 spend 0. S3
    and S4 bypass the masks and pay 1 per distinct candidate. A table charging
    both the same is the fiction G6 retires.
    """
    for env in tuning_envs:
        for method in _methods(env):
            ledger = Ledger.from_config(env.instance.cfg)
            with ledger.query_scope("q"):
                result = _run(method, env, ledger)
                metered = ledger.snapshot()["query"]["terminal_checks"]
            assert result.terminal_checks == metered
            if method.name in MASK_DRIVEN:
                assert metered == 0, f"{method.name} spent {metered}"
            else:
                assert method.name in DIRECT_BUILDERS
                assert 0 < metered <= env.instance.cfg.K


def test_stop_allowed_is_h_which_is_why_mask_driven_arms_spend_nothing(env):
    """The premise under criterion 6, asserted directly rather than assumed."""
    inst = env.instance
    state = build_checker(env, inst.obligations)
    for atom_id in legal_add_ids(state)[:3]:
        state.add(atom_id)
    assert stop_allowed(state) == H(
        state.selected(), inst.obligations, inst.graph, inst.pool, inst.cfg
    ).ok


def test_the_budget_is_enforced_before_it_is_spent(env):
    """Criterion 6 / decision 6: ``would_exceed()`` **before** the spend, so a
    method stops cleanly at the cap instead of overrunning."""
    inst = env.instance
    pool_ids = list(inst.pool.ids())
    many = [tuple(sorted(c)) for c in itertools.combinations(pool_ids, 2)][:40]
    ledger = Ledger.from_config(inst.cfg)
    with ledger.query_scope("q"):
        valid, spent, exhausted, _, portfolio = h_filter(
            many, inst.obligations, inst.graph, inst.pool, inst.cfg, ledger
        )
        metered = ledger.snapshot()["query"]["terminal_checks"]
    assert metered == spent <= inst.cfg.checker_budget
    assert exhausted is (len(dedup_sets(many)) > inst.cfg.checker_budget)


def test_a_capped_spend_outside_a_query_scope_raises(env):
    """The ledger's own rule (Phase-0 §2.9): the metric is per query, and a
    global counter cannot reconstruct it."""
    inst = env.instance
    ledger = Ledger.from_config(inst.cfg)
    with pytest.raises(LedgerError):
        h_filter(
            [tuple(sorted(inst.gold.atoms))],
            inst.obligations,
            inst.graph,
            inst.pool,
            inst.cfg,
            ledger,
        )


# -- criteria 7, 8: seeds and the relevance variant -------------------------


def test_deterministic_methods_repeat_exactly(tuning_envs):
    """Criterion 7: a deterministic method is reported once, so it had better be
    deterministic — including its tie-breaks."""
    env = tuning_envs[0]
    for method_a, method_b in zip(_methods(env), _methods(env)):
        first, second = _run(method_a, env), _run(method_b, env)
        assert [p.atoms for p in first.sets] == [p.atoms for p in second.sets]
        assert first.scores == second.scores


def test_both_relevance_variants_exist_and_differ(env):
    """Criterion 8 and decision 1: both are reported in every table, and they
    are genuinely different signals — if they coincided, the informed variant
    would close no escape hatch."""
    assert RELEVANCE_VARIANTS == ("obligation", "informed")
    obligation = relevance_vector(env, "obligation")
    informed = relevance_vector(env, "informed")
    assert set(obligation) == set(informed)
    assert any(
        abs(obligation[a] - informed[a]) > 1e-9 for a in obligation
    ), "the two relevance variants are identical, so decision 1 buys nothing"


def test_an_unknown_relevance_variant_raises(env):
    with pytest.raises(ValueError, match="unknown relevance variant"):
        relevance_vector(env, "whatever")


# -- criterion 9: exact U, with its caveat ----------------------------------


def test_all_methods_score_with_exact_u(env):
    """Criterion 9, fix F13. The scorer *is* ``U`` — which is what removes
    Robust Scheduling's proxy/true-evaluator gap and why Phase 4 cannot host
    Gate 3's comparison (G9)."""
    inst = env.instance
    scorer = exact_scorer(env)
    atoms = sorted(inst.gold.atoms)
    assert scorer(atoms) == pytest.approx(
        U(atoms, inst.obligations, inst.graph, inst.pool, inst.gold, inst.cfg)
    )


# -- criteria 10, 11b: the ceiling row, and greedy's optimality -------------


def test_the_ceiling_is_the_closed_form_and_below_greedy(tuning_envs):
    """Criteria 10 and 11b, and G9's measurement.

    ``E[best-of-K | p*]`` is what a *flawless* sampler attains. Greedy on exact
    ``U`` sits **above** it — the gap is an artefact of fix F13's perfect
    scorer, not a result about any method, which is exactly why decision 5
    forbids reading a best-of-K comparison as a Gate-3 verdict.
    """
    for env in tuning_envs:
        k = env.instance.cfg.K
        ceiling = best_of_k_ceiling(env.target, k)
        assert ceiling <= float(np.max(env.target.u)) + 1e-12
        assert best_of_k_ceiling(env.target, 1) <= ceiling
        assert ceiling <= best_of_k_ceiling(env.target, 256) + 1e-12


def test_ceiling_matches_a_monte_carlo_estimate(env):
    """The closed form, checked against sampling — the same instrument-verifies-
    itself discipline Phase 2 applies to its DP."""
    k = env.instance.cfg.K
    rng = np.random.default_rng(3)
    u, p = np.asarray(env.target.u), np.asarray(env.target.p_star[:-1])
    p = p / p.sum()
    draws = rng.choice(len(u), size=(20_000, k), p=p)
    empirical = float(u[draws].max(axis=1).mean())
    assert best_of_k_ceiling(env.target, k) == pytest.approx(empirical, abs=0.01)


def test_greedy_reaches_the_global_optimum(tuning_envs):
    """G9, reproduced: greedy on exact ``U`` attains the global argmax.

    This is the measurement that moved Gate 3's decision to Phase 9, so it is
    pinned here rather than left in a document.
    """
    for env in tuning_envs:
        result = _run(_methods(env)[0], env)
        assert result.best_utility == pytest.approx(float(np.max(env.target.u)))


# -- criterion 12c: the distinct count is reported, never assumed -----------


def test_the_distinct_count_is_reported_and_greedy_does_not_reach_k(tuning_envs):
    """Criterion 12c and decision 3.

    Every method *attempts* ``K``; greedy genuinely returns ~2.45 because G9's
    funnelling sends it to the same optimum whatever the opener, and the table
    says so. Engineering eight forks out of greedy would be engineering the
    baseline.
    """
    counts = []
    for env in tuning_envs:
        inst = env.instance
        result = _run(_methods(env)[0], env)
        # ``attempted`` is the number of forced openers actually available: the
        # k highest-``rel`` atoms **legal at the root**.  An instance with fewer
        # than K legal openers attempts fewer, which is the honest count rather
        # than a padded one — and is itself why the distinct count is reported.
        legal_at_root = len(legal_add_ids(build_checker(env, inst.obligations)))
        assert result.attempted == min(inst.cfg.K, legal_at_root)
        assert result.extra["forced_openers"] == result.attempted
        assert result.distinct_valid < inst.cfg.K
        counts.append(result.distinct_valid)
    assert 2 <= min(counts) and max(counts) <= 3


def test_beam_saturates_the_distinct_cap(tuning_envs):
    """Decision 5's arithmetic: beam-8 returns 8 distinct sets, which is why a
    "S5 returns strictly more" rule has ``P(pass) = 0``."""
    for env in tuning_envs:
        assert _run(_methods(env)[1], env).distinct_valid == env.instance.cfg.K


def test_s3_and_s4_report_their_own_distinct_counts(tuning_envs):
    """Criterion 12c's correction: S1's 2.45 was measured on masked greedy over
    exact ``U`` and does **not** cover S3, which is unmasked greedy over ``F``
    with a stand-in ``rel``."""
    for env in tuning_envs:
        for method in _methods(env)[2:]:
            result = _run(method, env)
            assert result.distinct_valid == len(
                {frozenset(p.atoms) for p in result.sets}
            )


# -- criterion 12b/12d: C_e calibration and the forest ----------------------


def test_edge_cost_grid_is_the_predeclared_one():
    """Decision 8: anchored on G-Retriever's own two values (1.0 SceneGraphs,
    0.5 WebQSP) rather than invented around them."""
    assert EDGE_COST_GRID == (0.25, 0.5, 1.0, 2.0)
    assert 0.5 in EDGE_COST_GRID and 1.0 in EDGE_COST_GRID


def test_calibration_selects_on_the_breach_rate_not_the_median(tuning_envs):
    """Decision 8 as **amended by measurement**.

    "Median closest to ``max_atoms`` without exceeding it" admits a median that
    *equals* the cap, which puts half the outputs over it: measured on the
    tuning suite at ``C_e = 0.5`` under obligation relevance the median is
    exactly 8.0 and the breach rate is 0.500. Selection is therefore on the
    breach rate, which is the property the criterion was always for.
    """
    out = calibrate_edge_cost(
        tuning_envs, lambda e: relevance_vector(e, "obligation")
    )
    assert out["chosen"] in EDGE_COST_GRID
    rates = {row["edge_cost"]: row["breach_rate"] for row in out["grid"]}
    assert rates[out["chosen"]] == min(rates.values())
    straddle = next(r for r in out["grid"] if r["edge_cost"] == 0.5)
    assert straddle["median_size"] == pytest.approx(8.0)
    assert straddle["breach_rate"] > 0.4, (
        "the straddle this amendment exists for is no longer measurable; "
        "re-derive the criterion rather than deleting the test"
    )


def test_pcst_solves_per_component_and_unions(env):
    """Decision 2's forest ruling, exercised directly."""
    inst = env.instance
    ground = admissible_atoms(env, inst.obligations)
    rel = relevance_vector(env, "informed")
    prize = _prizes_top_k(ground, rel, inst.cfg.K)
    chosen, meta = solve_pcst_forest(ground, inst.pool, prize, 0.25)
    assert meta["components"] >= 1
    assert meta["components_used"] <= meta["components"]
    assert meta["largest_component"] <= MAX_COMPONENT_ATOMS
    # every chosen atom belongs to a component that was used
    assert set(chosen) <= set(ground)


def test_pcst_solver_is_exact_on_a_hand_built_case(env):
    """The exact solver, on a case whose optimum is computable by hand.

    Prizes ``[3, 3]`` on two atoms joined by one reference link at
    ``C_e = 1``: taking both scores ``3 + 3 − 1 = 5`` against ``3`` for either
    alone, so both must be selected. ``pcst_fast`` 1.0.10's Windows wheel
    returns ``[0, 0]`` here — the bug that overturned decision 8's solver.
    """
    inst = env.instance
    pool = inst.pool
    edge_atom = next(a for a in pool.ids() if len(pool[a].refs) == 2)
    endpoint = pool[edge_atom].refs[0]
    ground = (edge_atom, endpoint)
    prize = {edge_atom: 3.0, endpoint: 3.0}
    chosen, _ = solve_pcst_forest(ground, pool, prize, 1.0)
    assert set(chosen) == {edge_atom, endpoint}
    # At a cost that outweighs the second prize, only one survives.
    chosen_dear, _ = solve_pcst_forest(ground, pool, prize, 7.0)
    assert len(chosen_dear) == 1


def test_pcst_fast_is_wrong_on_this_platform_which_is_why_the_solver_is_exact():
    """The measurement that overturned ruled decision 8 (`PHASE4_DECISIONS.md` §1.1).

    Decision 8 pinned ``pcst_fast`` because a prebuilt ``cp311-win_amd64`` wheel
    was verified to **exist**. It exists and is **wrong**: on a case whose
    optimum is unambiguous — two vertices, prizes 3 and 3, one edge of cost 1, so
    taking both scores 5 against 3 for either alone — the wheel returns a vertex
    array of the right *length* whose every element equals its first.

    Skipped where the library is absent: it is not a project dependency, it
    survives only as the evidence for this ruling. If the upstream wheel is ever
    fixed this test goes red, which is the right way to find out.
    """
    pcst_fast = pytest.importorskip("pcst_fast")

    edges = np.array([[0, 1]], dtype=np.int64)
    prizes = np.array([3.0, 3.0], dtype=np.float64)
    costs = np.array([1.0], dtype=np.float64)
    vertices, chosen_edges = pcst_fast.pcst_fast(edges, prizes, costs, -1, 1, "gw", 0)

    assert chosen_edges.tolist() == [0], "the edge choice is right; the bug is the vertex array"
    assert len(vertices) == 2, "the length is right, which is what makes the bug silent"
    assert vertices.tolist() != [0, 1], (
        "pcst_fast now returns the correct vertex set — the wheel was fixed. "
        "Re-open decision 8: the exact solver stays correct either way, but the "
        "reason recorded in PHASE4_DECISIONS.md 1.1 no longer holds."
    )
    assert vertices.tolist() == [0, 0], (
        "the failure signature changed; re-measure against a brute-force "
        "reference before trusting the library"
    )


def test_top_k_prizes_follow_g_retriever_equation_6(env):
    """G-Retriever Eq. 6: *"the top k nodes are assigned descending prize values
    from k down to 1, with the rest assigned zero"*."""
    ground = tuple(sorted(env.instance.pool.ids()))[:6]
    rel = {a: float(i) for i, a in enumerate(ground)}  # last is most relevant
    prize = _prizes_top_k(ground, rel, 3)
    assert sorted(prize.values(), reverse=True) == [3.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    assert prize[ground[-1]] == 3.0


# -- criterion 11: the bootstrap direction and shape ------------------------


def test_the_bootstrap_is_negated_for_a_higher_is_better_metric():
    """Criterion 11 and G5's first defect.

    ``gate2.paired_bootstrap``'s ``wins = upper < 0`` is written for exact TV,
    where lower is better. Passing a higher-better metric unnegated would report
    the winner as the loser — so ``gate3`` negates, and **records that it did**.
    """
    better = np.array([[0.9, 0.8, 0.85]] * 3)
    worse = np.array([[0.5, 0.4, 0.45]] * 3)
    out = higher_is_better_bootstrap(better, worse)
    assert out["negated_for_higher_is_better"] is True
    assert out["wins"] is True
    assert out["mean_difference"] > 0
    flipped = higher_is_better_bootstrap(worse, better)
    assert flipped["wins"] is False
    assert flipped["mean_difference"] < 0

    # The published interval must sit on the same side of the mean as the
    # reader's frame: an earlier version re-negated `mean_difference` and left
    # `upper_95` on the negated scale, so a strictly better `a` reported
    # mean +0.4 against an "upper" bound of -0.4.
    assert out["lower_95_on_a_minus_b"] <= out["mean_difference"]
    assert out["lower_95_on_a_minus_b"] > 0  # the whole interval favours `a`
    assert "upper_95" not in out, "the ambiguous key must not survive"
    assert out["upper_95_on_negated_scale"] == pytest.approx(
        -out["lower_95_on_a_minus_b"]
    )


def test_a_deterministic_arm_broadcasts_to_the_seed_count():
    """Criterion 11 and G4's resolution: a deterministic arm's seed variance
    really is zero, and broadcasting is what lets the bootstrap see zero rather
    than raise on a shape mismatch."""
    from graft.setgen.search.gate3 import _broadcast

    rows = _broadcast([0.5, 0.7], 3)
    assert rows.shape == (3, 2)
    assert np.all(rows[0] == rows[1]) and np.all(rows[1] == rows[2])
    out = higher_is_better_bootstrap(rows, _broadcast([0.4, 0.6], 3))
    assert out["n_seeds"] == 3 and out["n_instances"] == 2


# -- diversity, the one live condition --------------------------------------


def test_diversity_is_pinned_to_the_declared_convention():
    """Decision 5's amended cell. This is the **one live gate**, so the estimator
    may not float: mean pairwise Jaccard **distance** over unordered pairs, with
    duplicate pairs **included**, and 0 for a portfolio of fewer than two."""
    from graft.schemas import ProofSet

    a = ProofSet(atoms=frozenset({"x", "y"}))
    b = ProofSet(atoms=frozenset({"x", "z"}))
    assert jaccard_diversity([a]) == 0.0
    assert jaccard_diversity([]) == 0.0
    assert jaccard_diversity([a, a]) == 0.0  # a collapsed portfolio scores low
    assert jaccard_diversity([a, b]) == pytest.approx(1 - 1 / 3)


# -- criteria 13, 14, 15: audits, fingerprints, the two-stage exit ----------


def test_stage_a_carries_its_audits_fingerprints_and_caveats(tuning_envs):
    """Criteria 13, 14 and 15 together.

    The audits come from ``gate2.audit_block`` — the same Phase-2 source Gate 2
    uses, at zero marginal cost — and the table is labelled **Stage A**, which
    is what stops a table without S5 being read as Gate 3 (G7).
    """
    report = run_stage_a(tuning_envs, variant="obligation", edge_cost=2.0)
    summary = report.summary()

    assert summary["stage"] == "A"
    assert "s5_portfolio" not in summary["methods"]
    assert summary["relevance_variant"] == "obligation"

    audits = summary["audits"]
    assert audits["equivalent_action_collisions"] == 0
    assert audits["unconstructible_valid_terminals"] == 0
    assert "neither_mass" in audits and "beta" in audits
    assert "zero_delta_d_structural" in audits

    for env in tuning_envs:
        prints = env.fingerprints()
        assert set(prints) == {"environment", "target"}

    caveats = summary["caveats"]
    assert "optimistic" in caveats["scorer"]
    assert "diagnostic only" in caveats["gate"]
    assert "INERT" in caveats["checker_budget"]
    assert "Stage A is not Gate 3" in caveats["stage"]


def test_the_frozen_checker_budget_is_inert_on_this_environment(tuning_envs):
    """Criterion 10's second half, stated rather than implied.

    ``checker_budget = 32`` is the unit the Stage-D primary is *defined* in
    (`CLAUDE.md` §6), but the most any method can spend here is ``K = 8``, so the
    budget curve's levels are ``{1, 2, 4, 8}`` and the frozen 32 constrains
    nobody. It becomes binding at Phase 9.
    """
    for env in tuning_envs:
        for method in _methods(env):
            ledger = Ledger.from_config(env.instance.cfg)
            with ledger.query_scope("q"):
                result = _run(method, env, ledger)
            assert result.terminal_checks <= env.instance.cfg.K
            assert not result.budget_exhausted


def test_stage_a_reports_the_ceiling_beside_every_best_of_k(tuning_envs):
    """Criterion 11b: best-of-K is reported **against the ceiling**, never
    against a rival alone."""
    summary = run_stage_a(tuning_envs, variant="informed", edge_cost=2.0).summary()
    assert np.isfinite(summary["ceiling_mean"])
    for row in summary["methods"].values():
        assert "gap_to_ceiling_mean" in row


def test_the_structural_numbers_survive_into_the_summary(tuning_envs):
    """Criteria 11c, 12 and 12d, guarded rather than merely computed.

    The review round (PHASE4_DECISIONS §1.4 F7) found `completion_rate`,
    `max_atoms_breaches`, `h_rejections` and `E[U]` of returned sets reaching
    the per-instance rows and being **dropped by the summary** — computed but
    unreported, with nothing failing if they regressed. This asserts the
    aggregate actually carries them, and that the two direct builders' rows
    carry the fields their criteria name.
    """
    summary = run_stage_a(tuning_envs, variant="informed", edge_cost=2.0).summary()
    for method, row in summary["methods"].items():
        for key in (
            "mean_utility_of_returned",
            "portfolio_size_mean",
            "h_rejections_total",
            "completion_rate_mean",
            "max_atoms_breaches_total",
            "best_utility_scored_on",
            "best_utility_mean_failures_as_zero",
        ):
            assert key in row, f"{method} summary dropped {key}"
        if method != "s4_pcst":
            # None, not 0.0: a method that never reports completion must not
            # publish a number that reads as "never needed completion".
            assert row["completion_rate_mean"] is None
    # Criterion 12/12d: S4's completion and forest numbers are real, not zero-
    # filled — completion fires on this suite, and the component count is >= 1.
    s4 = summary["methods"]["s4_pcst"]
    assert s4["completion_rate_mean"] > 0.0
    # Criterion 14: both digests, per instance, in the report itself — the
    # ceiling row is p*-derived, so a u_weights move must not pass unseen.
    assert len(summary["fingerprints"]) == len(tuning_envs)
    assert all(set(f) == {"environment", "target"} for f in summary["fingerprints"])


def test_the_budget_curve_is_computed_over_the_declared_levels(tuning_envs):
    """Criterion 10's first half — the half the review round found unimplemented.

    v1.2 §5.2 requires performance plotted **across budget levels**; G3 fixes the
    levels at ``{1, 2, 4, 8}`` because the frozen 32 binds on nobody here. S1's
    curve is non-decreasing by a real theorem — its runs at ``k`` are a subset of
    its runs at ``k' > k`` — and the ceiling's monotonicity is order-statistic
    arithmetic. The other methods' shapes are reported, not asserted: beam's
    frontier pruning gives no such guarantee, and asserting one would be the
    "asserting math without checking it" pattern again.
    """
    from graft.setgen.search.gate3 import budget_curve

    curve = budget_curve(tuning_envs, variant="informed", edge_cost=2.0, levels=(1, 2, 4, 8))
    assert curve["levels"] == [1, 2, 4, 8]
    assert set(curve["methods"]) == {"s1_greedy", "s2_beam", "s3_submodular", "s4_pcst"}
    for method, rows in curve["methods"].items():
        assert len(rows["best_of_k"]) == 4 and len(rows["checks"]) == 4
    s1 = curve["methods"]["s1_greedy"]["best_of_k"]
    assert all(a <= b + 1e-12 for a, b in zip(s1, s1[1:])), (
        "S1's best-of-K decreased as k grew, which contradicts run-set nesting"
    )
    ceiling = curve["ceiling"]
    assert all(a <= b + 1e-12 for a, b in zip(ceiling, ceiling[1:]))
    # The failures-as-zero reading is monotone for every nested-candidate method
    # (S1/S3 openers and S4's top-k sweep are prefixes of the next level); the
    # conditional mean is allowed to dip when new instances enter the population.
    for method in ("s1_greedy", "s3_submodular", "s4_pcst"):
        zero = curve["methods"][method]["best_of_k_zero"]
        assert all(a <= b + 1e-9 for a, b in zip(zero, zero[1:])), (
            f"{method}'s failures-as-zero best-of-K decreased as k grew"
        )
    # The direct builders' spend rises with k; the mask-driven arms stay at 0.
    assert curve["methods"]["s1_greedy"]["checks"] == [0.0, 0.0, 0.0, 0.0]
    assert curve["methods"]["s3_submodular"]["checks"][-1] > 0.0
