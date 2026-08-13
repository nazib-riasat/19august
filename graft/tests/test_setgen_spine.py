"""The Phase-3 correctness spine: adapter, policy, heads and sampler.

Discharges exit criteria 1, 2, 3, 5, 6 and 7 — everything that must hold *before*
any learner is trained, because a fault here produces plausible loss curves and
meaningless Gate-2 numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graft.setgen.features import SyntheticFeaturizer
from graft.setgen.policy import LogZHead, Policy, PotentialHead, StateFlowHead, capacity
from graft.setgen.rollout import action_table, empirical_distribution, sample_trajectories
from graft.synth.exact import policy_distribution, tv
from graft.synth.policies import UniformPolicy

torch.manual_seed(0)


class _Constant(torch.nn.Module):
    """Emits the same logit for every action, so masking alone decides the
    distribution — which must then be exactly uniform over legal actions."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.hidden = hidden

    def action_logits(self, state_repr, action_reprs):
        return torch.zeros(action_reprs.shape[:2], dtype=action_reprs.dtype)


def _featurizer(instance, graph, policy, *, delta_d=False, dtype=torch.float64):
    return SyntheticFeaturizer(
        instance, graph, policy, instance.cfg, delta_d=delta_d, dtype=dtype
    )


def _real_policy(instance, graph, hidden=8):
    """Sized from the environment, not guessed — see `SyntheticFeaturizer.dims`."""
    state_dim, action_dim = SyntheticFeaturizer.dims(instance, graph)
    return Policy(state_dim, action_dim, hidden=hidden)


# -- criterion 1: the adapter introduces no distortion ---------------------


def test_a_constant_policy_reproduces_the_uniform_distribution(tiny, tiny_graph):
    """Exit criterion 1. If the adapter distorted the distribution, every TV
    downstream would be measuring the adapter."""
    got = policy_distribution(_featurizer(tiny, tiny_graph, _Constant()), tiny_graph)
    want = policy_distribution(UniformPolicy(), tiny_graph)
    assert np.allclose(got, want, atol=1e-12)


def test_the_adapter_matches_uniform_on_a_full_lattice(bench, bench_graph):
    """Exit criterion 1, on a full lattice and in float64.

    The tolerance is a claim about the **adapter**, so it is measured where the
    adapter is the only source of error.
    """
    got = policy_distribution(_featurizer(bench, bench_graph, _Constant()), bench_graph)
    want = policy_distribution(UniformPolicy(), bench_graph)
    assert np.abs(got - want).max() < 1e-12
    assert tv(got, want) < 1e-12


def test_float32_agrees_to_float32_precision_and_that_is_a_dtype_fact(bench, bench_graph):
    """Training runs in float32, which carries ~1e-7 of relative precision — so
    the 1e-12 above is unreachable there and it would be dishonest to assert it.

    Separated deliberately: criterion 1 asks whether the *adapter* distorts the
    distribution, and the answer must not be entangled with the dtype the
    learners happen to train in. Gate-2 TV differences are orders of magnitude
    larger than this floor, so float32 is not a threat to the comparison — but
    that is an argument, and it belongs next to the number.
    """
    got = policy_distribution(
        _featurizer(bench, bench_graph, _Constant(), dtype=torch.float32), bench_graph
    )
    want = policy_distribution(UniformPolicy(), bench_graph)
    assert np.abs(got - want).max() < 1e-6
    assert tv(got, want) < 1e-6


# -- criterion 2: masking happens before the softmax -----------------------


def test_illegal_actions_get_exactly_zero_probability(tiny, tiny_graph):
    """Exit criterion 2. Masking *after* a softmax leaves a small positive mass
    on illegal actions, which would leak into the DP as reachable flow."""
    torch.manual_seed(1)
    feat = _featurizer(tiny, tiny_graph, _real_policy(tiny, tiny_graph).double())
    query = np.flatnonzero(~tiny_graph.dead_end)
    log_add, log_stop = feat.action_log_probs(query, tiny_graph)

    legal = np.zeros((query.size, tiny_graph.n_atoms), dtype=bool)
    rows = {int(s): i for i, s in enumerate(query)}
    for e in range(tiny_graph.n_edges):
        legal[rows[int(tiny_graph.edge_parent[e])], int(tiny_graph.edge_action[e])] = True

    assert np.all(np.exp(log_add)[~legal] == 0.0)
    assert np.all(np.isfinite(log_add) == legal)
    assert np.allclose(np.exp(log_add).sum(axis=1) + np.exp(log_stop), 1.0, atol=1e-6)


def test_a_dead_end_is_refused(bench, bench_graph):
    feat = _featurizer(bench, bench_graph, _Constant())
    with pytest.raises(ValueError, match="dead-end"):
        feat.action_log_probs(bench_graph.dead_ix[:1], bench_graph)


def test_the_featurizer_is_bound_to_its_graph(bench, bench_graph):
    from graft.synth.enumerate import reachable_states

    feat = _featurizer(bench, bench_graph, _Constant())
    with pytest.raises(ValueError, match="different StateGraph"):
        feat.action_log_probs(np.array([0]), reachable_states(bench, bench.cfg))


# -- criterion 5: the Δd gate, in both directions --------------------------


def test_the_delta_d_gate_holds_in_both_directions(bench, bench_graph):
    """Exit criterion 5.

    One direction alone proves nothing: without the first check `Δd` may be
    leaking into a control, without the second L7 may be ignoring it — and either
    way Gate 2 compares two arms that differ by zero.
    """
    torch.manual_seed(2)
    policy = _real_policy(bench, bench_graph).double()

    off = SyntheticFeaturizer(bench, bench_graph, policy, bench.cfg, delta_d=False)
    on = SyntheticFeaturizer(bench, bench_graph, policy, bench.cfg, delta_d=True)
    idx = torch.arange(6, dtype=torch.long)

    a_off, a_on = off.action_reprs(idx), on.action_reprs(idx)
    width = off.atom_feat.shape[1]
    block = slice(width, width + 6)

    assert torch.all(a_off[:, :, block] == 0.0), "delta_d=False must zero the Δd block"
    assert torch.any(a_on[:, :, block] != 0.0), "delta_d=True must populate it"
    # Everything outside the Δd block is identical, so the flag changes that and
    # nothing else.
    assert torch.equal(a_off[:, :, :width], a_on[:, :, :width])
    assert torch.equal(a_off[:, :, block.stop :], a_on[:, :, block.stop :])


def test_repeated_states_in_one_query_each_get_their_delta_d(bench, bench_graph):
    """A regression test for a bug that would have produced a *false negative*
    for Contribution 3.

    The first implementation scattered per-edge `Δd` through a state→row map, so
    a query containing the same state twice kept only the last occurrence and
    zeroed the rest. Unique queries — the exact DP, the sampler — never hit it.
    **Training batches hit it constantly**, because a batch of trajectories
    revisits states by design. L7 would have trained on mostly-zeroed features
    and looked like "Δd does not help".
    """
    feat = _featurizer(bench, bench_graph, _Constant(), delta_d=True)
    state = int(bench_graph.edge_parent[0])
    width = feat.atom_feat.shape[1]

    once = feat.action_reprs(torch.tensor([state]))
    thrice = feat.action_reprs(torch.tensor([state, state, state]))
    assert bool((once[0, :, width : width + 6] != 0).any()), "fixture state has no Δd"
    for row in range(3):
        assert torch.equal(thrice[row], once[0]), f"row {row} differs from the single query"

    # And interleaved with other states, which is what a real batch looks like.
    other = int(bench_graph.edge_parent[-1])
    mixed = feat.action_reprs(torch.tensor([state, other, state]))
    assert torch.equal(mixed[0], mixed[2])
    assert torch.equal(mixed[0], once[0])


def test_the_delta_d_gate_holds_on_the_forward_output(bench, bench_graph):
    """Exit criterion 5, on **logits** rather than on the feature block.

    Checking that the featurizer zeroes a block is necessary and is not the
    criterion: the criterion is that an L6 forward pass is unchanged and an L7
    pass changes. Perturbing the underlying `Δd` and re-running the *model* is
    what tests that — a policy that ignored the block, or a featurizer that
    leaked it, would both survive the block-level check alone.
    """
    torch.manual_seed(7)
    policy = _real_policy(bench, bench_graph).double()
    idx = torch.arange(24, dtype=torch.long)

    l6 = _featurizer(bench, bench_graph, policy, delta_d=False)
    l7 = _featurizer(bench, bench_graph, policy, delta_d=True)
    before_l6, before_l7 = l6.logits(idx).clone(), l7.logits(idx).clone()

    # L7 must actually use it: same weights, same states, different output.
    assert not torch.allclose(before_l6, before_l7)

    # Perturb the environment's Δd and re-run both.
    l7._delta_table = l7._delta_table + 0.37
    assert l6._delta_table is None, "an L6 featurizer must not even hold the table"
    after_l6, after_l7 = l6.logits(idx), l7.logits(idx)

    assert torch.equal(before_l6, after_l6), "L6's output moved with Δd — it is leaking"
    assert not torch.allclose(before_l7, after_l7), "L7's output ignored Δd"


def test_stop_carries_no_delta_d(bench, bench_graph):
    """`STOP` does not change the selected set, so its `Δd` is zero by
    construction — the same exclusion Phase-2 G5 applies to the density."""
    feat = _featurizer(bench, bench_graph, _Constant(), delta_d=True)
    reprs = feat.action_reprs(torch.arange(8, dtype=torch.long))
    width = feat.atom_feat.shape[1]
    assert torch.all(reprs[:, bench_graph.n_atoms, width : width + 6] == 0.0)
    assert torch.all(reprs[:, bench_graph.n_atoms, width + 6] == 1.0)


# -- criterion 3: the sampler agrees with the exact DP ---------------------


def test_sampled_terminals_match_the_exact_dp(tiny, tiny_graph):
    """Exit criterion 3 — the hinge.

    A sampler that disagreed with the DP would still produce plausible loss
    curves; nothing else in the phase would notice.
    """
    feat = _featurizer(tiny, tiny_graph, _Constant())
    exact = policy_distribution(feat, tiny_graph)
    rng = np.random.default_rng(20260810)
    empirical = empirical_distribution(
        tiny_graph, sample_trajectories(feat, tiny_graph, 200_000, rng)
    )
    assert tv(exact, empirical) < 0.02

    se = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / 200_000)
    assert float(np.max(np.abs(empirical - exact) / se)) < 5.0


def test_trajectories_are_well_formed(bench, bench_graph):
    feat = _featurizer(bench, bench_graph, _Constant())
    traj = sample_trajectories(feat, bench_graph, 500, np.random.default_rng(3))
    assert len(traj) == 500
    assert traj.lengths.max() <= bench_graph.max_atoms

    table = action_table(bench_graph)
    rows, parents, acts, children = traj.transitions()
    assert np.all(table[parents, acts] == children)
    assert np.all(bench_graph.size[children] == bench_graph.size[parents] + 1)

    stopped = ~traj.is_fail
    assert np.all(bench_graph.stop_allowed[traj.terminal[stopped]])
    assert np.all(traj.terminal[traj.is_fail] == -1)


def test_fail_trajectories_are_counted_against_the_budget(bench, bench_graph):
    """Decision 3: a rollout counts whether it ends at `STOP` or at `FAIL`, or a
    learner that dead-ends often buys extra gradient steps for free."""
    from graft.synth.policies import ForcedContinuationPolicy

    forced = policy_distribution(ForcedContinuationPolicy(), bench_graph)
    assert forced[-1] > 0.0
    feat = _featurizer(bench, bench_graph, _Constant())
    traj = sample_trajectories(feat, bench_graph, 2_000, np.random.default_rng(4))
    assert len(traj) == 2_000, "every rollout counts, including FAIL"


# -- criteria 6 and 7: the layering and the ML boundary --------------------


#: The modules permitted to see a ``StateGraph``, a ``LatticeInstance`` or an
#: atom id. It is a **closed list**, not a skip list: a new module in `setgen/`
#: that reaches for the environment fails this test until someone adds it here
#: deliberately, which is the point at which the F6 boundary gets re-argued
#: rather than eroded.
#:
#: P3.1 names `features.py` alone, and that was already inaccurate when written —
#: P3.3's own surface is `sample_trajectories(policy, graph, n, rng)`, so
#: `rollout.py` takes a graph by specification. `trainer.py` enumerates and holds
#: the environments. The **binding** rule, and the one exit criterion 6 states,
#: is about `learners/`.
#:
#: **Phase 4's `search/` package is on this side, added deliberately** (P4.1).
#: Search modules are **not** learners: S3 and S4 *must* see the pool and its
#: `refs` to build sets directly, and fix F10 makes that legal precisely because
#: `H` re-checks closure afterwards — which is also why they pay one terminal
#: check per candidate while the mask-driven arms pay none. Exit criterion 6
#: binds `learners/`, and every module under it is still scanned.
_ADAPTER_LAYER = {
    "graft/setgen/__init__.py",
    "graft/setgen/features.py",
    "graft/setgen/flgfn_probe.py",
    "graft/setgen/rollout.py",
    "graft/setgen/trainer.py",
    "graft/setgen/gate2.py",
}

#: Phase 4's package, listed separately so the top-level closed list stays a
#: statement about *top-level* modules and the search package's membership is
#: its own recorded decision rather than an entry lost in a set.
_SEARCH_LAYER = {
    "graft/setgen/search/__init__.py",
    "graft/setgen/search/base.py",
    "graft/setgen/search/relevance.py",
    "graft/setgen/search/s1_greedy.py",
    "graft/setgen/search/s2_beam.py",
    "graft/setgen/search/s3_submodular.py",
    "graft/setgen/search/s4_pcst.py",
    "graft/setgen/search/s5_portfolio.py",
    "graft/setgen/search/gate3.py",
}

#: Every top-level `setgen/` module. `policy.py` is deliberately **not** an
#: adapter — it is the F6 interface itself and imports nothing but torch, so the
#: scan above covers it like a learner does.
_TOPLEVEL = _ADAPTER_LAYER | {"graft/setgen/policy.py"}


def test_no_learner_module_touches_the_environment():
    """Exit criterion 6, fix F6.

    No module under `graft/setgen/learners/` may import a `StateGraph`, a
    `LatticeInstance` or an atom id. Every objective reads a `Batch` of padded
    tensors instead, so at Phase 9 the Stage-B graph encoder replaces the
    featurizer and nothing in `learners/` changes.
    """
    import ast
    from pathlib import Path

    import graft.setgen

    root = Path(graft.setgen.__file__).parent
    forbidden = {"StateGraph", "LatticeInstance", "reachable_states", "AtomPool"}
    checked = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parents[1]).as_posix()
        if rel in _ADAPTER_LAYER or rel in _SEARCH_LAYER:
            continue
        checked += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {a.name.rsplit(".", 1)[-1] for a in node.names}
                assert not (names & forbidden), f"{rel} imports {names & forbidden}"
                module = getattr(node, "module", None) or ""
                assert not module.startswith("graft.synth"), (
                    f"{rel} imports from graft.synth ({module}); the environment "
                    "reaches learners only as tensors"
                )
    assert checked >= 10, "the learner package was not actually scanned"


def test_the_adapter_layer_is_a_closed_list():
    """The guard above is only as good as its exemption list.

    If a new `setgen/` module appears, it is in `learners/` and is checked, or it
    is an adapter and someone had to say so here. Silently growing the exemption
    set is how an F6 boundary erodes without any test going red.
    """
    from pathlib import Path

    import graft.setgen

    root = Path(graft.setgen.__file__).parent
    present = {
        p.relative_to(root.parents[1]).as_posix()
        for p in root.glob("*.py")
    }
    assert present == _TOPLEVEL, (
        "top-level setgen modules changed; each one is either an adapter that "
        "may see the environment or a module the scan above covers, and the "
        f"answer has to be recorded. Added: {present - _TOPLEVEL}. "
        f"Removed: {_TOPLEVEL - present}"
    )
    assert "graft/setgen/policy.py" not in _ADAPTER_LAYER, (
        "policy.py is the F6 interface; exempting it would exempt the boundary"
    )

    # Phase 4's package is a closed list for the same reason, and for a sharper
    # one: a *learner* smuggled in under `search/` would be exempt from exit
    # criterion 6 by filing location alone.
    search_root = root / "search"
    if search_root.is_dir():
        present_search = {
            p.relative_to(root.parents[1]).as_posix()
            for p in search_root.glob("*.py")
        }
        assert present_search == _SEARCH_LAYER, (
            "the Phase-4 search package changed; each module either sees the "
            "environment deliberately (P4.1) or belongs elsewhere. "
            f"Added: {present_search - _SEARCH_LAYER}. "
            f"Removed: {_SEARCH_LAYER - present_search}"
        )


def test_the_portfolio_is_one_greedy_plus_seven_sampled(bench, bench_graph):
    """Fix F5 defines `K = 8` **and its composition**: 1 greedy + 7 sampled. An
    earlier build sampled all eight, which measures a different portfolio from
    the one Phase 9 ships and Phase 4's S5 is compared against — the greedy
    candidate is the one a reward-maximiser would return.

    The greedy row is deterministic: it takes the argmax legal action at every
    step, so it does not move with the seed."""
    from graft.setgen.learners import build_arm
    from graft.setgen.trainer import Environment, Trainer, TrainSpec

    env = Environment(bench, bench_graph)
    trainer = Trainer(build_arm("l4_tb"), [env], TrainSpec(n_trajectories=256))
    trainer.train()
    feat = trainer.featurizers[0]

    firsts = {
        int(sample_trajectories(feat, bench_graph, 8, np.random.default_rng(s), 0.0,
                                greedy=1).terminal[0])
        for s in (1, 2, 3, 4)
    }
    assert len(firsts) == 1, "the greedy candidate moved with the seed"

    # ...and it is genuinely the argmax walk, not just some fixed rollout
    traj = sample_trajectories(feat, bench_graph, 8, np.random.default_rng(1), 0.0,
                               greedy=1)
    log_add, log_stop = feat.action_log_probs(np.asarray([0], dtype=np.int64), bench_graph)
    best = int(np.argmax(np.concatenate([log_add[0], log_stop])))
    assert int(traj.actions[0, 0]) == best or best == bench_graph.n_atoms

    # the sampled rows still vary with the seed
    tails = {
        tuple(sample_trajectories(feat, bench_graph, 8, np.random.default_rng(s), 0.0,
                                  greedy=1).terminal[1:].tolist())
        for s in (1, 2, 3)
    }
    assert len(tails) > 1, "the sampled candidates are not sampling"

    with pytest.raises(ValueError, match="greedy"):
        sample_trajectories(feat, bench_graph, 4, np.random.default_rng(0), 0.0, greedy=9)


def test_a_checkpoint_round_trips_and_loads_without_the_trainer(tmp_path, bench, bench_graph):
    """Build step 4's done-when ("checkpointing round-trips") and Phase-3 §8
    requirement 1 ("loadable without the trainer"). Neither held: the trainer's
    "checkpoints" are exact-TV evaluations, no `state_dict` was ever written, and
    `run_matrix` deleted each trainer as it went — so a Gate-2 run discarded
    every model it trained and Phase 4's S5 had nothing to consume.

    The load path is exercised through `graft.setgen.policy` alone, which is the
    requirement: a module importing only torch."""
    import torch

    from graft.setgen.learners import build_arm
    from graft.setgen.policy import CHECKPOINT_FORMAT, load_policy
    from graft.setgen.trainer import Environment, Trainer, TrainSpec

    env = Environment(bench, bench_graph)
    trainer = Trainer(build_arm("l7_checker_led"), [env], TrainSpec(n_trajectories=64))
    trainer.train()
    path = trainer.save_checkpoint(tmp_path / "l7.seed13.pt")

    policy, blob = load_policy(path)
    assert blob["format"] == CHECKPOINT_FORMAT
    assert blob["arm"] == "l7_checker_led"
    # the caller needs delta_d: an L7 policy behind an L6 featurisation reads a
    # zeroed Δd block, which is a silently wrong policy rather than an error
    assert blob["delta_d"] is True
    assert blob["fingerprints"] == [env.fingerprints()]
    assert blob["live_capacity"] == trainer.live_capacity

    # ...and the weights are the trained ones, not a fresh initialisation
    for before, after in zip(
        trainer.policy.state_dict().values(), policy.state_dict().values()
    ):
        assert torch.equal(before, after)

    with pytest.raises(ValueError, match="format"):
        blob["format"] = 999
        torch.save(blob, tmp_path / "stale.pt")
        load_policy(tmp_path / "stale.pt")


def test_uniform_backward_is_re_exported_not_reimplemented():
    """Decision 18 and the P3.3 surface. Phase 2's version reads the enumerated
    in-edges, which *are* the removable atoms; a second implementation here would
    be a second chance to get "uniform over selected" wrong."""
    from graft.setgen import rollout
    from graft.synth import policies

    assert rollout.uniform_backward is policies.uniform_backward
    assert "uniform_backward" in rollout.__all__


def test_the_heads_every_objective_needs_exist():
    """G12. LED-DB is defined over `log F̃(s)` and GAFlowNet's augmented TB
    carries `r/F(s′)`; without a state-flow head neither is implementable."""
    torch.manual_seed(5)
    policy = Policy(10, 10, hidden=16)
    flow, logz = StateFlowHead(16), LogZHead(10, hidden=16)
    phi = PotentialHead(10, 10, hidden=16)

    h = policy.encode(torch.randn(4, 10))
    assert flow(h).shape == (4,)
    assert logz(torch.randn(4, 10)).shape == (4,)
    assert phi(torch.randn(4, 10), torch.randn(4, 10)).shape == (4,)


def test_the_potential_is_its_own_network_not_a_head_on_the_policy():
    """Decision 23a. LED Appendix C: "the neural network architecture of the
    potential function is identical to that of the GFlowNet policy" — *identical
    architecture*, which is a separate network of the same shape.

    It has to be separate: the potential trains under its own optimiser at lr
    0.001, and a shared trunk would let that optimiser drag the policy with it,
    at a learning rate three times the shared protocol's.
    """
    torch.manual_seed(7)
    policy = Policy(10, 11, hidden=16)
    phi = PotentialHead(10, 11, hidden=16)

    shared = {id(p) for p in policy.parameters()} & {id(p) for p in phi.parameters()}
    assert not shared, "the potential shares weights with the policy"
    assert capacity(policy) == capacity(phi), (
        "identical architecture means identical parameter count; they differ, so "
        "the two are not the same shape"
    )


def test_capacity_counts_every_head(bench, bench_graph):
    """G4, decision 27. A match computed over the trunk alone is wrong by the
    size of the heads — in the direction that flatters L7."""
    torch.manual_seed(6)
    policy = Policy(10, 10, hidden=16)
    flow = StateFlowHead(16)
    trunk_only = capacity(policy)
    with_heads = capacity(policy, flow)
    assert with_heads > trunk_only
    assert with_heads - trunk_only == sum(p.numel() for p in flow.parameters())
    # idempotent under repetition, so a shared head is not double-counted
    assert capacity(policy, flow, flow) == with_heads
