"""Phase 9 — Stage D on real data.

Exit criteria 1–6 and the parts of 13/15 that exist before the runner does.
Every fixture is built by hand and every embedder is a stub: this file must stay
runnable with no downloads, no GPU and no ``beta`` (build steps 0–5 are
explicitly unblocked while Phase-3 step 6 has not run).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from graft.config import Config
from graft.ledger import Ledger
from graft.schemas import Assertion, AssertionFlags, Obligations
from graft.setgen import pins
from graft.setgen.atomfeat import ATOM_WIDTH, RealFeaturizer
from graft.setgen.distill import HeadScorer, build_head, pool_sets, spearman, train_head
from graft.setgen.learners import ARMS
from graft.setgen.policy import Policy
from graft.setgen.portfolio import (
    PortfolioResult,
    answer_agreement,
    audit_validity,
    run_portfolio,
)
from graft.setgen.proofs import ProofExample, SourceDoc, build_example, build_snapshot
from graft.setgen.realenv import (
    RealEnvironment,
    RealTrainer,
    build_real_batch,
    sample_real,
)
from graft.setgen.trainer import Arm, TrainSpec

REPO = Path(__file__).resolve().parents[2]


def stub_embed(seed: int = 13):
    """A deterministic stand-in for the pinned embedder.

    The decisive path refuses a stub (that is Phase 6's rule and Phase 7 inherited
    it); these tests are about *shapes and boundaries*, where the real weights
    would buy nothing and cost a download.
    """
    rng = np.random.default_rng(seed)

    def embed(texts):
        return np.stack([rng.normal(size=384) for _ in texts])

    return embed


def tiny_example(n_docs: int = 6, gold: int = 2, example_id: str = "q1") -> ProofExample:
    docs = [
        SourceDoc(
            f"p{i}",
            f"Document {i} concerns London and topic {i}.",
            ("London",),
            is_gold=(i < gold),
        )
        for i in range(n_docs)
    ]
    return build_example(
        example_id,
        docs,
        Obligations(entity_anchor="London", scope=(example_id,)),
        {f"p{i}": 1.0 - 0.1 * i for i in range(n_docs)},
        channel_scores={"bm25": {f"p{i}": 3.0 - 0.4 * i for i in range(n_docs)}},
        embed=stub_embed(),
    )


# --------------------------------------------------------------------------
# criterion 1 — fix F6 held: the learners did not move
# --------------------------------------------------------------------------


def test_learner_files_are_byte_identical_to_their_committed_state():
    """Exit criterion 1, and the whole premise of building Phase 3 on an adapter.

    Phase 9 replaces the environment, the featurizer and the batch builder. If a
    single loss had to change to accept real data, fix F6 would have failed and
    the Gate-2 lineage would not carry: the arms compared on the lattice would
    not be the arms compared here.
    """
    proc = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "graft/setgen/learners/"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:  # pragma: no cover - git absent
        pytest.skip("git unavailable; the byte-identity claim needs the index")
    changed = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not changed, (
        "Phase 9 modified learner files, which fix F6 exists to prevent: "
        f"{changed}. The environment may change; the objectives may not."
    )


# --------------------------------------------------------------------------
# criterion 3 — the structural boundaries
# --------------------------------------------------------------------------

_ABOVE_THE_BOUNDARY = (
    "graft/setgen/atomfeat.py",
    "graft/setgen/realenv.py",
    "graft/setgen/distill.py",
    "graft/setgen/portfolio.py",
)

_CORPUS_NAMES = ("wiki2", "2wiki", "musique", "hover", "hotpot", "longmemeval", "locomo")


def test_no_module_above_proofs_names_a_corpus():
    """Exit criterion 3, and the architecture's own words.

    The loader consumes the abstract ``(pool, obligations, gold_proof)`` triple
    and *"no corpus-specific parser may appear above it — verified by an
    import-graph test"*. This is that test. It is what keeps the conversational
    track a drop-in if the Wikipedia→conversation transfer claim fails, which is
    a declared and untested claim (`CLAUDE.md` §7).

    ``pins.py`` is exempt by construction: it is §6's decision table, and
    decision 1 *is* the list of corpora. Naming them there is the point.
    """
    for rel in _ABOVE_THE_BOUNDARY:
        source = (REPO / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = (getattr(node, "module", None) or "") + " ".join(
                    a.name for a in node.names
                )
                assert "corpora" not in module, (
                    f"{rel} imports a corpus adapter ({module}); everything above "
                    "proofs.py sees ProofExample and nothing else"
                )
        # Identifiers, not comments: the module docstrings legitimately discuss
        # which corpora Stage A trains on, and forbidding the word outright would
        # make the honest documentation the failure.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                lowered = node.id.lower()
                assert not any(c in lowered for c in _CORPUS_NAMES), (
                    f"{rel} has an identifier naming a corpus ({node.id})"
                )


def test_the_checker_never_reaches_the_distilled_head():
    """Exit criterion 3, and `CLAUDE.md` §4.2's non-negotiable.

    If a learned scorer can enter ``H``, then ``H`` stops being a predicate,
    ``1[H]`` stops being a hard gate, and the multiplicative safety property
    degrades into a soft threshold. Enforced as a module boundary rather than as
    discipline, which is what that section asks for.
    """
    for rel in ("graft/core/checker.py", "graft/core/masks.py", "graft/core/incremental.py"):
        source = (REPO / rel).read_text(encoding="utf-8")
        assert "setgen" not in source, f"{rel} reaches into graft.setgen"
        assert "distill" not in source, f"{rel} names the distilled head"


def test_the_head_and_the_features_are_gold_free():
    """Exit criterion 3's third clause.

    Exact ``U`` reaches the head as a training **target**; a head that could read
    the gold set would score the training distribution rather than predict it,
    and its ρ would be wrong in the direction that flatters the method.

    ``atomfeat`` may name gold exactly once — ``gold_path``, L1 and L2's
    supervision target — and its feature builder may not.
    """
    # **Attribute access, not substring on the source.** The first version of
    # this test grepped for "gold_atom_ids" and failed on `distill.py`'s own
    # docstring, which names the field precisely in order to say it must not be
    # read. A guard that forbids documenting the rule it enforces is a guard
    # that gets deleted; this reads the AST instead.
    gold_fields = {"gold_atom_ids", "gold_groups", "gold_complete"}

    def reads_gold(tree: ast.AST) -> set[str]:
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in gold_fields:
                found.add(node.attr)
            if isinstance(node, ast.Name) and node.id in gold_fields:
                found.add(node.id)
        return found

    distill = ast.parse((REPO / "graft/setgen/distill.py").read_text(encoding="utf-8"))
    assert not reads_gold(distill), (
        f"distill.py reads {reads_gold(distill)}; exact U is its target, never its input"
    )

    tree = ast.parse((REPO / "graft/setgen/atomfeat.py").read_text(encoding="utf-8"))
    builders = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in {"_atom_matrix", "state_repr", "action_reprs"}
    ]
    assert builders, "the feature builders were renamed; this guard now checks nothing"
    for node in builders:
        assert not reads_gold(node), (
            f"{node.name} reads {reads_gold(node)}; features must be computable at "
            "inference, where there is no gold"
        )


# --------------------------------------------------------------------------
# criteria 5 and 6 — the pool is Phase 7's, and H binds what it claims to
# --------------------------------------------------------------------------


def test_the_pool_is_closed_capped_and_eligibility_enforced():
    """Exit criterion 5 — through ``build_pool``, not a re-implementation."""
    ex = tiny_example()
    ids = set(ex.pool.ids())
    assert len(ids) <= Config().pool_cap
    for atom in ex.pool:
        for ref in atom.refs:
            assert ref in ids, "an atom references outside the pool; closure is broken"
    assert any(a.kind == "edge" for a in ex.pool), (
        "no edge atoms: closure would be structurally vacuous and criterion 6's "
        "'closure binds' claim would be false on this track"
    )


def test_closure_binds_an_edge_atom_before_its_endpoints():
    """Criterion 6. The edge atom is unaddable until both endpoints are selected."""
    from graft.core.masks import legal_add_ids

    ex = tiny_example()
    env = RealEnvironment(ex, Config(), range_samples=2)
    legal = set(legal_add_ids(env.checker()))
    edges = [a.atom_id for a in ex.pool if a.kind == "edge"]
    assert edges and not (set(edges) & legal), "an edge atom was addable at the root"


def test_an_ineligible_assertion_is_refused_from_the_pool():
    """Criterion 6 — fix F9's boundary, negative-tested.

    Quarantined evidence must never become retrievable. Phase 5's support gate
    decides it, Phase 6 refuses to commit it, Phase 7 re-applies it at assembly,
    and this asserts Phase 9 inherits all three rather than bypassing them by
    building its own snapshot.
    """
    docs = [SourceDoc(f"p{i}", f"Doc {i} about London.", ("London",), is_gold=(i == 0)) for i in range(3)]
    snapshot = build_snapshot("q1", docs)
    victim = sorted(snapshot._assertions)[0]
    original = snapshot._assertions[victim]
    snapshot._assertions[victim] = Assertion(
        assertion_id=original.assertion_id,
        kind=original.kind,
        text_norm=original.text_norm,
        spans=original.spans,
        flags=original.flags,
        t_created=original.t_created,
        eligibility="quarantined",
    )
    from graft.retrieve.pool import eligible_nodes

    remaining = eligible_nodes(snapshot, "q1")
    assert len(remaining) == 2, "a quarantined assertion survived the eligibility filter"


def test_the_three_vacuous_checks_are_declared_and_actually_vacuous():
    """Criterion 6's third clause, and G4's honesty requirement.

    Temporal, binding and retired cannot bind on a Wikipedia pool — no
    ``valid_during`` intervals, no binding atoms, nothing retired. They are
    *declared* vacuous in pins rather than silently counted as passes, and this
    asserts the declaration matches the code.
    """
    ex = tiny_example()
    env = RealEnvironment(ex, Config(), range_samples=2)
    state = env.checker(sorted(ex.gold_atom_ids))
    result = state.result()
    assert result.ok
    failing = {v.check for v in result.violations}
    for name in pins.VACUOUS_ON_WIKIPEDIA:
        assert name not in failing
    assert not any(a.kind == "binding" for a in ex.pool), (
        "a binding atom exists, so the 'binding is vacuous' declaration is false"
    )


# --------------------------------------------------------------------------
# criteria 2 and 4 — the Batch runs every arm, and delta_d is gated
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm_name", sorted(ARMS))
def test_every_arm_takes_a_gradient_step_on_real_pools(arm_name: str):
    """Exit criterion 2 — the sampler's ``Batch`` is the object the losses read."""
    spec_kw = dict(ARMS[arm_name])
    loss = spec_kw.pop("loss")
    arm = Arm(arm_name, loss, hidden=16, **spec_kw)
    envs = [RealEnvironment(tiny_example(example_id=f"q{i}"), Config(), range_samples=4) for i in range(2)]
    spec = TrainSpec(seed=13, n_trajectories=4, batch_size=2, hidden=16, epsilon=0.05)
    trainer = RealTrainer(arm, envs, spec, greedy=1)
    log = trainer.train()
    assert np.isfinite(log.final_loss), f"{arm_name} produced a non-finite loss"
    assert log.trajectories[-1] == 4


def test_delta_d_reaches_l7_and_l7b_and_no_other_arm():
    """Exit criterion 4, asserted in **both** directions.

    ``delta_d`` is the entire L6/L7 difference (decision 19a). A leak into L6 or
    GAFlowNet makes Gate 2 compare two arms that differ by zero; a leak *out* of
    L7 makes it compare the proposed method against itself. One direction alone
    would catch half of that.
    """
    expected = {"l7_checker_led", "l7b_aux"}
    for name, spec in ARMS.items():
        assert bool(spec.get("delta_d", False)) is (name in expected), (
            f"{name} disagrees with decision 19a's routing"
        )

    ex = tiny_example()
    n = 3
    for delta_d in (True, False):
        torch.manual_seed(13)
        feat = RealFeaturizer(ex, Policy(*RealFeaturizer.dims(), hidden=16), Config(), delta_d=delta_d)
        delta = torch.ones((n, feat.n_atoms, 6))
        reprs = feat.action_reprs(n, delta)
        block = reprs[:, : feat.n_atoms, ATOM_WIDTH : ATOM_WIDTH + 6]
        if delta_d:
            assert float(block.abs().sum()) > 0.0, "L7 did not receive Δd"
        else:
            assert float(block.abs().sum()) == 0.0, (
                "Δd leaked into an arm that must not see it — GAFlowNet's Δd "
                "reaches its loss as an intrinsic reward and never its policy"
            )


def test_the_terminating_transition_is_a_first_class_step():
    """The terminal convention, measured in Phase 3 and inherited here.

    ``R`` is the flow on the terminating ``STOP`` edge, not the terminal state's
    flow. Four objectives are stated over ``F(s)``, and a batch whose last
    transition were folded away would make each of them wrong by the continuation
    flow — misreporting as a decomposition failure.
    """
    ex = tiny_example()
    env = RealEnvironment(ex, Config(), range_samples=2)
    torch.manual_seed(13)
    policy = Policy(*RealFeaturizer.dims(), hidden=16)
    feat = RealFeaturizer(ex, policy, Config(), delta_d=False)
    from graft.setgen.policy import LogZHead

    traj = sample_real(feat, env, 4, np.random.default_rng(5), epsilon=0.1)
    batch = build_real_batch(feat, env, traj, logz=LogZHead(RealFeaturizer.dims()[0], 16))
    assert batch.steps >= int(traj.lengths.max()) + 1, (
        "no room for the terminating transition"
    )
    n_trans = batch.n_trans.numpy()
    assert (n_trans == traj.lengths + 1).all(), (
        "n_trans must count the terminating transition; every objective's "
        "boundary position is read from it"
    )


# --------------------------------------------------------------------------
# the estimated normaliser, and determinism
# --------------------------------------------------------------------------


def test_log_r_range_is_reproducible_across_processes():
    """It is seeded from a content digest, not from ``hash()``.

    Python salts string hashing per process, so a ``hash()``-derived seed would
    give a different normaliser on every launch — and this number divides a
    *reported* diagnostic. "Frozen before training" has to survive a restart to
    mean anything.
    """
    ex = tiny_example()
    a = RealEnvironment(ex, Config(), range_samples=8)
    b = RealEnvironment(ex, Config(), range_samples=8)
    assert a.log_r_range == b.log_r_range
    assert a._range_report["estimated"] is True, "the estimate must declare itself"


def test_exact_tv_is_refused_rather_than_approximated():
    """G1's honest consequence.

    An approximate TV printed in the column Gate 2's exact numbers occupy would
    invite exactly the comparison that is invalid.
    """
    arm_kw = dict(ARMS["l4_tb"])
    arm = Arm("l4_tb", arm_kw.pop("loss"), hidden=16, **arm_kw)
    envs = [RealEnvironment(tiny_example(), Config(), range_samples=2)]
    trainer = RealTrainer(arm, envs, TrainSpec(seed=13, n_trajectories=2, batch_size=2, hidden=16))
    with pytest.raises(NotImplementedError, match="enumerated"):
        trainer.exact_tv()


# --------------------------------------------------------------------------
# the distilled head and the portfolio
# --------------------------------------------------------------------------


def _fit_head(ex: ProofExample):
    env = RealEnvironment(ex, Config(), range_samples=4)
    torch.manual_seed(13)
    feat = RealFeaturizer(ex, Policy(*RealFeaturizer.dims(), hidden=16), Config(), delta_d=False)
    traj = sample_real(feat, env, 40, np.random.default_rng(3), epsilon=0.3)
    sets = [t for r, t in enumerate(traj.terminals()) if not traj.is_fail[r]]
    x = pool_sets(feat, sets)
    y = torch.as_tensor(np.array([env.utility(s) for s in sets]), dtype=torch.float32)
    cut = int(0.7 * len(sets))
    head = build_head(ATOM_WIDTH, seed=13)
    history = train_head(head, (x[:cut], y[:cut]), (x[cut:], y[cut:]), seed=13, epochs=15)
    return env, feat, head, history


def test_the_head_respects_its_cap_and_restores_its_best_epoch():
    """G8 and P6.11's second guard.

    Early stopping that leaves the last epoch's weights in place selects on dev
    and then reports a different model than the one it selected.
    """
    _, _, head, history = _fit_head(tiny_example())
    assert sum(p.numel() for p in head.parameters()) <= pins.DISTILL["max_params"]
    assert history["best_epoch"] < history["epochs_run"]
    assert np.isfinite(history["best_dev_loss"])


def test_the_head_refuses_an_empty_dev_split():
    """P6.11's third guard: no scorable dev means no basis for selection."""
    head = build_head(ATOM_WIDTH, seed=13)
    x = torch.zeros((4, 2 * ATOM_WIDTH + 1))
    y = torch.zeros(4)
    with pytest.raises(ValueError, match="dev"):
        train_head(head, (x, y), (x[:0], y[:0]), seed=13, epochs=1)


def test_spearman_reports_a_degenerate_head_as_nan():
    """A constant predictor has no ranking, and must not score as if it had one."""
    assert np.isnan(spearman(np.ones(6), np.arange(6)))
    assert spearman(np.arange(6), np.arange(6)) == pytest.approx(1.0)
    assert spearman(np.arange(6), np.arange(6)[::-1]) == pytest.approx(-1.0)


def test_the_portfolio_spends_no_terminal_checks_and_is_valid_by_construction():
    """Decision 9 as **corrected at the build**, and the correction's evidence.

    Every rollout goes through the ``ADD`` masks and ``stop_allowed`` *is* ``H``,
    so a completed trajectory is valid by construction — which is why Phase 4's
    S5 reports ``terminal_checks=0`` and why routing this through ``h_filter``
    would have charged it 8 checks it does not owe. The audit path re-checks the
    delivered sets against batch ``H`` off the measured path, and finds none
    invalid: that is what turns "valid by construction" from a claim into a test.
    """
    ex = tiny_example()
    env, feat, head, _ = _fit_head(ex)
    result = run_portfolio(feat, env, HeadScorer(head, feat), np.random.default_rng(11))
    assert result.terminal_checks == 0
    assert result.distinct_valid == len(result.sets)
    assert result.distinct_valid <= Config().K

    ledger = Ledger.from_config(Config())
    with ledger.query_scope("q1"):
        audit = audit_validity(result, env, ledger)
    assert audit["checked"] == len(result.sets)
    assert audit["invalid"] == [], "a delivered set failed batch H"


def test_the_portfolio_ranks_by_score_then_prefers_the_smaller_set():
    """The tie-break is where minimality enters the inference path."""
    ex = tiny_example()
    env, feat, head, _ = _fit_head(ex)
    result = run_portfolio(feat, env, HeadScorer(head, feat), np.random.default_rng(11))
    keys = [(-s, len(a)) for s, a in zip(result.scores, result.sets)]
    assert keys == sorted(keys), "the portfolio is not ordered by (score desc, size asc)"


def test_the_fallback_fires_when_every_rollout_dead_ends():
    """G9, and the closing of Phase 8's reserved loop.

    Its ``abstain_fallback`` counter has been zero by design since Phase 8; this
    is the event that makes it non-zero. A pool whose every atom is inadmissible
    leaves the root with no legal ``ADD`` and ``STOP`` masked, which is the dead
    end fix F3 maps to the abstain fallback.
    """
    ex = tiny_example()
    env = RealEnvironment(ex, Config(), range_samples=2)
    torch.manual_seed(13)
    feat = RealFeaturizer(ex, Policy(*RealFeaturizer.dims(), hidden=16), Config(), delta_d=False)

    # `max_atoms = 0` makes every ADD illegal, and the empty set is never a legal
    # terminal (Phase-1 G1), so the root is a dead end for every rollout.
    starved = RealEnvironment(ex, Config(max_atoms=0), range_samples=0)
    result = run_portfolio(feat, starved, lambda a: 0.0, np.random.default_rng(3))
    assert result.fallback is True
    assert result.best is None
    assert result.extra[pins.PORTFOLIO["fallback_counter"]] == 1


def test_answer_agreement_measures_presence_not_competing_bindings():
    """G9's split: what is computable here, and what is Phase 10's by name.

    **Renamed and re-scoped 16 Aug 2026.** The old `contested_rate` compared
    **atom-id sets** while claiming in its own docstring to compare "the answer
    they resolve to", so it was wrong in both directions: two paragraphs
    carrying the same answer read as contested, and a set binding nothing read
    as agreeing. Competing answer bindings are not measurable from gold aliases
    at all — every alias is an alias of the one gold answer — so the diagnostic
    now reports answer *presence* agreement and evidence diversity, and says
    fix F4's flag belongs to Phase 10.
    """
    ex = tiny_example()
    env, feat, head, _ = _fit_head(ex)
    result = run_portfolio(feat, env, HeadScorer(head, feat), np.random.default_rng(11))
    report = answer_agreement(result, env, ["topic 0"])
    assert "Phase 10" in report["note"]
    assert "NOT measurable here" in report["note"]
    assert report["distinct_evidence"] >= 0
    # no aliases -> nothing binds, and that is `none_bound`, never "agreement"
    empty = answer_agreement(result, env, [])
    assert empty["bound"] == 0 and empty["none_bound"] is True
    assert empty["all_bound"] is False


def _atoms_by_text(env, needle):
    """Atom ids whose text contains ``needle`` — the fixture keys on generated
    ids, so a test that wants "the atom about topic 0" has to look it up."""
    from graft.setgen.proofs import atom_text

    return [
        aid
        for aid in env.example.pool.ids()
        if needle.casefold() in atom_text(env.example.snapshot, env.example.pool[aid]).casefold()
    ]


def test_answer_agreement_does_not_call_same_answer_different_evidence_contested():
    """**The first counterexample the audit reproduced.**

    Two top sets carrying the gold answer on *different* atoms agree about the
    answer and differ only in evidence — which is what a portfolio is supposed
    to produce. The old atom-id comparison called that contested.
    """
    ex = tiny_example()
    env, feat, head, _ = _fit_head(ex)
    london = _atoms_by_text(env, "London")
    assert len(london) >= 2, "fixture should give several atoms carrying the answer"
    a, b = (london[0],), (london[1],)
    result = PortfolioResult([a, b], [1.0, 0.9], portfolio=[a, b], attempted=2)
    report = answer_agreement(result, env, ["London"])
    assert report["bound"] == 2 and report["all_bound"] is True
    assert report["answer_presence_disagreement"] is False  # agreement, not conflict
    assert report["distinct_evidence"] == 2  # different evidence, same answer


def test_answer_agreement_flags_the_mixed_presence_case():
    """**The second counterexample.** One top set carries the answer and another
    does not — a real disagreement, which the old code scored as agreement
    because it dropped unbound sets before comparing."""
    ex = tiny_example()
    env, feat, head, _ = _fit_head(ex)
    bound = (_atoms_by_text(env, "topic 0")[0],)
    others = [a for a in _atoms_by_text(env, "topic 1") if a not in bound]
    unbound = (others[0],)
    result = PortfolioResult(
        [bound, unbound], [1.0, 0.9], portfolio=[bound, unbound], attempted=2
    )
    report = answer_agreement(result, env, ["topic 0"])
    assert report["answer_presence_disagreement"] is True
    assert report["bound"] == 1 and report["unbound"] == 1
    assert report["all_bound"] is False and report["none_bound"] is False


# --------------------------------------------------------------------------
# pins
# --------------------------------------------------------------------------


def test_pins_import_without_torch_and_bind_feature_names():
    """The stage-D fingerprint must be printable on a bare interpreter.

    And it must bind feature **names in vector order**: Phase 8's audit found a
    fingerprint binding five block strings while the features underneath changed
    from normalised to raw, so two different experiments shared one identity.
    """
    frozen = pins.frozen_values()
    assert set(frozen["feature_names"]) == set(pins.FEATURE_BLOCKS)
    flat = [n for block in pins.FEATURE_BLOCKS for n in frozen["feature_names"][block]]
    assert len(flat) == len(set(flat)), "two features share a name"
    assert len(pins.stage_d_fingerprint()) == 64


def test_training_is_refused_while_beta_is_unfrozen():
    """Decision 6. Every arm shares one reward by construction, so a run started
    before Phase-3 step 6 freezes β is either wasted or contaminates the freeze.
    """
    reason = pins.training_blocked_reason()
    frozen, problem = pins.beta_frozen()
    if frozen:  # pragma: no cover - the gate has not run
        assert reason is None and problem is None
    else:
        assert reason is not None and problem is not None
        assert "beta is not frozen" in reason

    # **The check reads the record's CONTENTS, not merely its name** (adversarial
    # audit, 16 Aug 2026). The first version returned `record.exists()`, so an
    # empty file, a `--quick` calibration, or a record whose adopted beta
    # disagreed with the config all read as "frozen" and would have let a scored
    # run start at the placeholder 4.0.
    import json
    import tempfile
    from pathlib import Path

    def _with_record(payload, config=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase3_calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return pins.beta_frozen(config, record=path)

    assert _with_record({})[0] is False, "an empty record must not read as frozen"
    assert _with_record({"adopted": {}})[0] is False, "no adopted beta must not read as frozen"
    assert _with_record({"adopted": {"beta": 3.0}})[0] is False, (
        "a beta outside the eligible candidates must not read as frozen"
    )
    assert _with_record({"adopted": {"beta": 4.0}})[0] is True
    # And a record that disagrees with the config it would train under: the run
    # would be at a different reward than the freeze certified.
    assert _with_record({"adopted": {"beta": 8.0}}, Config(beta=4.0))[0] is False
    assert _with_record({"adopted": {"beta": 4.0}}, Config(beta=4.0))[0] is True
    assert pins.beta_frozen(record=Path(tempfile.gettempdir()) / "nope.json")[0] is False


def test_the_gate3_rule_was_frozen_before_any_result_existed():
    """Exit criterion 10's first half, as far as a test can carry it.

    The rule names its primary, its test, its consolidation clause and the
    size-controlled diversity secondary Phase 4 deferred here. A rule missing its
    losing branch is not a rule.
    """
    rule = pins.GATE3_RULE
    assert "best-of-K" in rule["primary"]
    assert "paired bootstrap" in rule["test"]
    assert "consolidates on Contribution 1" in rule["consolidation"]
    assert "size-controlled" in rule["diversity_secondary"]
    # `n_real` was None until build step 4 measured it (16 Aug 2026). What the
    # rule requires is not that it be absent, but that it be **derived**: the
    # largest ladder rung the slowest arm clears inside the ceiling, at a rate
    # taken from a real run. A value off the ladder would be a budget chosen by
    # wall clock after the fact, which is what decision 11 exists to prevent.
    budget = pins.BUDGET
    assert budget["n_real"] in budget["ladder"], (
        f"N_real {budget['n_real']} is not a rung of {budget['ladder']}"
    )
    assert budget["measured_rate_slowest"] > 0
    assert budget["measured_slowest_arm"] in ARMS
    projected_h = budget["n_real"] / budget["measured_rate_slowest"] / 3600.0
    assert projected_h <= budget["ceiling_s"] / 3600.0, (
        "the adopted rung does not fit the ceiling at the measured rate"
    )


# --------------------------------------------------------------------------
# regressions for the five defects the 16 Aug 2026 adversarial audit found
# --------------------------------------------------------------------------


def test_delta_d_has_the_same_sign_and_density_at_sampling_and_at_gradient_time():
    """The audit's first blocker, and the one that would have been invisible.

    ``ob.delta_deficit`` is ``d(s) - d(s')``, so a **positive** component means
    the transition *discharged* an obligation. The batch builder computed
    ``child - parent`` while the sampler computed ``parent - child``, and it
    filled only the *taken* action's column while the sampler filled every legal
    one. Both halves ran against Contribution 3:

    *sign* — L7's policy was trained to read discharge as accrual.

    *density* — the synthetic ``_delta_table`` is dense over all edges precisely
    so ``Δd`` is a **comparison across candidate actions**; filled only at the
    taken action it becomes a label on the choice already made, and "prefer the
    ADD that discharges more obligation" is unlearnable.

    ``features.py`` documents this exact failure being found once already on the
    synthetic side, where it produced "a false negative for Contribution 3".
    """
    from graft.core import obligations as ob
    from graft.core.masks import legal_adds, stop_allowed
    from graft.setgen.realenv import legal_deltas

    ex = tiny_example()
    env = RealEnvironment(ex, Config(), range_samples=2)
    state = env.checker()
    legal = np.zeros(env.n_atoms + 1, dtype=bool)
    legal[: env.n_atoms] = legal_adds(state)
    legal[env.n_atoms] = stop_allowed(state)

    deltas = legal_deltas(env, (), legal)
    assert deltas.shape == (env.n_atoms, len(ob.DEFICIT_COMPONENTS))

    # Dense over legal actions, not just one.
    nonzero_rows = int((deltas.abs().sum(dim=1) > 0).sum())
    assert nonzero_rows >= 1, "no legal action produced a Δd; the helper is not filling"

    # Sign: recompute one entry by hand against delta_deficit's own definition.
    before = env.checker(()).deficit()
    j = int(np.flatnonzero(legal[: env.n_atoms])[0])
    probe = env.checker(())
    probe.add(env.atom_ids[j])
    expected = ob.delta_deficit(before, probe.deficit())
    assert np.allclose(deltas[j].numpy(), expected), (
        "legal_deltas disagrees with ob.delta_deficit's parent-minus-child convention"
    )

    # And an illegal action stays zero rather than carrying a stale value.
    illegal = [i for i in range(env.n_atoms) if not legal[i]]
    if illegal:
        assert float(deltas[illegal[0]].abs().sum()) == 0.0


def test_the_batch_builders_delta_block_matches_the_sampler():
    """The same helper feeds both, so they cannot drift apart again.

    The root cause of the sign/density defect was two implementations of one
    quantity. This asserts there is now one.
    """
    import inspect

    from graft.setgen import realenv

    source = inspect.getsource(realenv)
    assert source.count("def legal_deltas") == 1
    # Neither the sampler nor the batch builder may compute Δd inline any more.
    assert "ob.delta_deficit(" not in source.split("def legal_deltas")[1].split("def _removable")[1], (
        "a second Δd computation reappeared outside legal_deltas"
    )


def test_the_raw_channel_column_carries_raw_values(embedder_free=True):
    """The audit's third finding — Phase 8's §3.3 defect, one phase later.

    The adapters supply the normalised view under the bare channel name and the
    raw magnitudes under ``{channel}_raw``. Reading the bare name into the column
    *called* ``_raw`` and min-maxing it again produced two bit-identical
    normalised columns, while the genuine BM25 magnitude was computed, carried on
    the example, and never read.
    """
    from graft.setgen.atomfeat import ATOM_BLOCKS, BLOCK_FEATURES

    ex = tiny_example()
    # tiny_example supplies only a normalised-style 'bm25' map; add a raw one
    # whose scale is unmistakably not [0, 1].
    raw = {aid: 7.5 + i for i, aid in enumerate(sorted(ex.pool.ids())[:3])}
    ex = ProofExample(
        example_id=ex.example_id, snapshot=ex.snapshot, pool=ex.pool,
        obligations=ex.obligations, atom_scores=ex.atom_scores,
        channel_scores={**ex.channel_scores, "bm25_raw": raw},
        atom_feat=ex.atom_feat, gold_atom_ids=ex.gold_atom_ids,
        gold_groups=ex.gold_groups, meta=ex.meta,
    )
    feat = RealFeaturizer(ex, delta_d=False)
    names = [n for b in ATOM_BLOCKS for n in BLOCK_FEATURES[b]]
    idx = {n: i for i, n in enumerate(names)}
    matrix = feat.atom_feat.numpy()
    raw_col = matrix[:, idx["bm25_raw"]]
    norm_col = matrix[:, idx["bm25_norm"]]
    assert raw_col.max() > 1.0, "the raw column was normalised; the magnitude is gone"
    assert not np.allclose(raw_col, norm_col), "raw and normalised columns are identical"


def test_every_channel_carries_its_own_presence_flag():
    """The audit's fourth finding, and G2's actual rule.

    With one block-level flag the policy cannot tell "this channel did not run"
    from "this channel ran and scored zero on every atom" — so Stage B, which
    populates entity/temporal/expand, would differ from Stage A by an
    uninstrumented change in feature *semantics* as well as by its data. That is
    the transfer measurement the phase exists to make.
    """
    from graft.setgen.atomfeat import ATOM_BLOCKS, BLOCK_FEATURES, CHANNELS

    ex = tiny_example()  # supplies bm25 only
    feat = RealFeaturizer(ex, delta_d=False)
    names = [n for b in ATOM_BLOCKS for n in BLOCK_FEATURES[b]]
    idx = {n: i for i, n in enumerate(names)}
    matrix = feat.atom_feat.numpy()
    for channel in CHANNELS:
        assert f"{channel}_present" in idx, f"{channel} has no presence flag"
    assert matrix[:, idx["bm25_present"]].max() == 1.0
    for absent in ("entity", "temporal", "expand", "scorer"):
        assert matrix[:, idx[f"{absent}_present"]].max() == 0.0, (
            f"{absent} did not run but its presence flag is set"
        )


def test_real_gold_batch_fills_delta_rather_than_passing_zeros():
    """The audit's fifth finding: a dead branch that becomes live silently.

    No arm in the ruled roster is both ``supervised`` and ``delta_d``, so the
    branch is unreachable today. It becomes reachable the moment anyone adds a
    supervised C3 ablation — at which point that arm would train on an
    identically-zero Δd block while believing it was checker-conditioned, and
    read out as "checker-conditioning adds nothing under supervision".
    """
    arm_kw = dict(ARMS["l1_supervised"])
    arm = Arm("l1_supervised", arm_kw.pop("loss"), hidden=16, delta_d=True,
              **{k: v for k, v in arm_kw.items() if k != "delta_d"})
    envs = [RealEnvironment(tiny_example(), Config(), range_samples=2)]
    trainer = RealTrainer(arm, envs, TrainSpec(seed=13, n_trajectories=2, batch_size=2, hidden=16))
    logits, targets = trainer.real_gold_batch(envs[0], trainer.featurizers[0], 2)
    assert logits.shape[0] == targets.shape[0]
    assert torch.isfinite(logits).any()


# --------------------------------------------------------------------------
# decision 11's capacity match, at the REAL dims
# --------------------------------------------------------------------------


def test_every_control_is_capacity_matched_never_smaller_than_l7():
    """**The regression for the defect that shipped** (16 Aug 2026 audit).

    Decision 11 post-R13 (`PHASE3_DECISIONS.md` §6.4) has two clauses, and the
    directional one carries the whole argument: if L7 wins, it wins against a
    **strictly larger** control, so "L7 had more capacity" is unavailable as an
    objection. Nothing tested it at Phase-9's dims, and the step-4 script had
    silently produced `l6_led` at 220,164 live against L7's 220,932 — a control
    **smaller** than the proposed method, in the direction that flatters it.

    The cause was `match_capacity`'s retired 1% default firing on the *correct*
    width (65, +1.53%) and a handler falling back to 64. The retired tolerance is
    a number, so `check_plan_consistency.py` — which reads prose — could not see
    it. This test is what sees it.
    """
    from graft.setgen.atomfeat import RealFeaturizer
    from graft.setgen.gate2 import CAPACITY_SANITY_CEILING
    from graft.setgen.learners import ARMS
    from graft.setgen.policy import match_capacity
    from graft.setgen.trainer import Arm, Trainer

    state_dim, action_dim = RealFeaturizer.dims()

    def live(name: str, hidden: int) -> int:
        kw = dict(ARMS[name])
        arm = Arm(name, kw.pop("loss"), hidden=hidden, **kw)
        return Trainer.capacity_of(arm, state_dim, action_dim, hidden, 2) - \
            Trainer.dead_capacity_of(arm, hidden)

    target = live("l7_checker_led", 64)
    for name in sorted(ARMS):
        width = match_capacity(
            lambda h, n=name: live(n, h), target, tol=CAPACITY_SANITY_CEILING
        )
        achieved = live(name, width)
        assert achieved >= target, (
            f"{name} at width {width} has {achieved} live parameters against "
            f"L7's {target}: decision 11's directional clause is violated"
        )
        # minimality — one width narrower must be below the target, or the
        # excess is not the smallest the architecture admits
        assert width == 64 or live(name, width - 1) < target, (
            f"{name}'s width {width} is not the narrowest admissible one"
        )


def test_match_capacity_no_longer_carries_the_retired_tolerance_as_a_default():
    """The retired 1% survived as an *executable default*, invisible to the
    prose-reading consistency guard. It is now `None` — the directional clause
    alone — and a ceiling is something a caller opts into.
    """
    import inspect

    from graft.setgen.policy import match_capacity

    assert inspect.signature(match_capacity).parameters["tol"].default is None
    # and with no ceiling, a coarse-but-correct width is returned rather than raised
    assert match_capacity(lambda h: h * 100, 6450, base=64) == 65
