"""The numbers Gate 2 reports, one function each.

============================================  =====================  ===========
audit                                          expected               if violated
============================================  =====================  ===========
unconstructible valid terminals                **0**, by fix F10      the closure rule or the masks are wrong
equivalent-action collisions                   **0**, by G3           the pool is malformed
state-fingerprint collisions                   **0**                  ``canon_set_hash`` collided; identity is still exact, the *fingerprint* is unusable
``FAIL`` reachability                          >= 1 reachable dead end  ``STOP``-masking is doing nothing (G4)
dead-end absorption mass by ``|X|``            early share <= 0.05     the ``ADD`` masks are too tight, not that the budget is small
``d`` informativeness                          not ``|s|``-determined; both zero-``Δd`` <= 0.6  **Gate 2 cannot resolve L7 from L6** (G5)
target mass by mode bucket                     reported; alt. mode >= 1% is a *diagnostic*  the target is effectively unimodal (G10)
target mass on ``neither``                     <= 0.5                  the distractor tail dominates what TV measures (G10)
============================================  =====================  ===========

**Reported per instance and aggregated, never a single pooled number** — these
feed a gate, and pooling lets one bad instance hide inside nineteen good ones.

**Every audit that can be computed exactly is.**  The state graph is enumerated,
so dead-end absorption mass comes from the same forward DP rather than from
rollouts, and the ``Δd`` densities are sums over the edge list rather than
estimates.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np

from graft.config import Config
from graft.core import obligations as ob
from graft.core.checker import CHECK_BINDING, CHECK_TEMPORAL, H
from graft.core.incremental import IncrementalChecker
from graft.core.resolve import atom_intervals, source_node
from graft.schemas import PAYLOAD_TIER, AtomPool
from graft.synth.enumerate import (
    BandViolation,
    StateGraph,
    environment_fingerprint,
    reachable_states,
    state_fingerprints,
)
from graft.synth.exact import Target, forward_mass, target_distribution
from graft.synth.policies import ForcedContinuationPolicy, UniformPolicy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.synth.lattice import LatticeInstance

__all__ = [
    "run_audits",
    "band_report",
    "closure_audit",
    "collision_audit",
    "fail_reachability",
    "dead_end_absorption",
    "d_informativeness",
    "structural_assertions",
    "jaccard_spread",
    "POOL_MIN",
    "POOL_MAX",
]

#: The architecture's universe size (exit criterion 10).  Asserted directly,
#: because the state-count band does **not** subsume it: a generator that passed
#: the wrong ``cap`` could still happen to produce a small graph, and the failure
#: would only surface in Phase 7 against real pools.
POOL_MIN, POOL_MAX = 20, 30

_JACCARD_PAIRS = 2_000
_JACCARD_SEED = 20260813


# --------------------------------------------------------------------------
# closure / fix F10
# --------------------------------------------------------------------------


def _topological_atoms(pool: AtomPool, keep: Iterable[str]) -> tuple[str, ...]:
    """Admissible atoms ordered refs-before-referrer.

    Needed because the independent enumeration below extends subsets in index
    order: an atom's references must have smaller index or a closed subset would
    be skipped.  ``AtomPool`` already refuses reference cycles, so this
    terminates.
    """
    keep = set(keep)
    order: list[str] = []
    seen: set[str] = set()

    def visit(aid: str) -> None:
        if aid in seen or aid not in keep:
            return
        seen.add(aid)
        for ref in pool[aid].refs:
            visit(ref)
        order.append(aid)

    for aid in sorted(keep):
        visit(aid)
    return tuple(order)


def closure_audit(
    instance: "LatticeInstance", graph: StateGraph, cfg: Config | None = None
) -> dict[str, Any]:
    """Unconstructible-valid-terminal rate, expected **0** (fix F10, criterion 12).

    Enumerated **independently of the masks**: this walks the pool's ``refs``
    directly, in topological order, and then calls batch ``H`` on each closed
    subset.  Re-using ``legal_adds`` would make the audit agree with the thing it
    is auditing.

    Only admissible atoms are enumerated, and that is a proof rather than a
    shortcut: a per-atom violation makes ``H`` fail for *any* set containing that
    atom, so no such set is a valid terminal.  The claim is checked on a sample
    rather than assumed.
    """
    cfg = cfg if cfg is not None else instance.cfg
    pool, q, G = instance.pool, instance.obligations, instance.graph
    chk = IncrementalChecker(pool, q, G, cfg, ledger=None)
    admissible = [aid for aid in pool.ids() if chk.atom_is_admissible(aid)]
    order = _topological_atoms(pool, admissible)
    position = {aid: i for i, aid in enumerate(order)}

    closed: list[frozenset[str]] = []

    def extend(start: int, current: set[str]) -> None:
        closed.append(frozenset(current))
        if len(current) >= cfg.max_atoms:
            return
        for i in range(start, len(order)):
            aid = order[i]
            if all(r in current for r in pool[aid].refs):
                current.add(aid)
                extend(i + 1, current)
                current.discard(aid)

    extend(0, set())

    unconstructible: list[tuple[str, ...]] = []
    unreachable: list[tuple[str, ...]] = []
    reachable_terminals = {int(t) for t in graph.terminal_ix.tolist()}
    for subset in closed:
        state = graph.state_of(subset)
        valid = bool(H(subset, q, G, pool, cfg, ledger=None))
        if state is None:
            unreachable.append(tuple(sorted(subset)))
            if valid:
                unconstructible.append(tuple(sorted(subset)))
            continue
        if valid != (state in reachable_terminals):
            unconstructible.append(tuple(sorted(subset)))

    # The "inadmissible atoms cannot appear in a valid set" step, sampled rather
    # than asserted: an inadmissible atom on its own must already fail `H`.
    blocked = [aid for aid in pool.ids() if aid not in set(admissible)]
    blocked_valid = [aid for aid in blocked if H([aid], q, G, pool, cfg, ledger=None)]

    return {
        "closed_subsets": len(closed),
        "unconstructible_valid_terminals": len(unconstructible),
        "unreachable_closed_subsets": len(unreachable),
        "admissible_atoms": len(admissible),
        "blocked_atoms": len(blocked),
        "blocked_atoms_passing_H": blocked_valid,
        "examples": unconstructible[:3],
    }


# --------------------------------------------------------------------------
# collisions / G3
# --------------------------------------------------------------------------


def collision_audit(graph: StateGraph) -> dict[str, Any]:
    """Equivalent-action collisions, expected **0** — a theorem, not a measurement.

        For a state ``S`` and distinct legal actions ``a != b``, the children are
        the **sets** ``S ∪ {a}`` and ``S ∪ {b}``.  Since ``a ∉ S`` and ``b ∉ S``,
        these sets differ.  Two distinct legal actions therefore never produce the
        same child state.  ∎

    The statement is about **set equality**, not about hashes: ``canon_set_hash``
    is a 64-bit truncation of SHA-256 and is injective only with overwhelming
    probability, so the proof must not lean on it.  This audit is defined over
    exact child-set equality and a non-zero result means the pool is malformed —
    not that a correction is needed.

    **[EVIDENCE]** Symmetry-Aware GFlowNets (ICML 2025) measured a systematic,
    paradigm-dependent sampling bias — its fragment-based setting *over*-produced
    a highly **symmetric** fragment, 5,220 cyclohexanes per 5,000 sampled
    molecules uncorrected against 1,042 corrected, while node-by-node generation
    biases the opposite way, toward fewer symmetries — on a state space quotiented
    by graph **isomorphism**, with its Theorem 4.6 stated over a graph-level
    policy and ratios of automorphism-group sizes.  Ours is over *labelled* sets
    and is not quotiented, so **the correction does not apply and is not
    applied** (decision 5).  The honest sentence is "our action space admits no
    equivalent actions by construction; we verify this".

    The separate fingerprint check is milder in consequence: state identity is
    the exact bitmask, so a ``canon_set_hash`` collision would not corrupt the
    evaluator — it would make the fingerprint unusable for cross-machine
    comparison.
    """
    collisions = 0
    for s in range(graph.n_states):
        actions, children = graph.children_of(s)
        if actions.shape[0] < 2:
            continue
        masks = graph.mask[children]
        if np.unique(masks).shape[0] != masks.shape[0]:
            collisions += 1

    prints = state_fingerprints(graph)
    fingerprint_collisions = len(prints) - len(set(prints))
    return {
        "equivalent_action_collisions": collisions,
        "state_fingerprint_collisions": fingerprint_collisions,
    }


# --------------------------------------------------------------------------
# FAIL / G4
# --------------------------------------------------------------------------


def fail_reachability(graph: StateGraph) -> dict[str, Any]:
    """Dead ends, by ``|X|``.

    ``FAIL`` is reached exactly when **construction can neither legally continue
    nor legally stop**; budget exhaustion is the common case, not the meaning.
    If it were unreachable, ``STOP``-masking would be doing nothing, the ``FAIL``
    terminal would be decorative, and fix F3's whole construction would be
    untested.
    """
    sizes = graph.size[graph.dead_ix] if graph.n_dead_ends else np.zeros(0, dtype=np.int16)
    return {
        "reachable_dead_ends": graph.n_dead_ends,
        "dead_end_sizes": np.bincount(sizes, minlength=graph.max_atoms + 1).tolist(),
    }


def dead_end_absorption(
    instance: "LatticeInstance", graph: StateGraph, cfg: Config | None = None
) -> dict[str, Any]:
    """Exact absorption mass by ``|X|``, and the **conditional** early share.

    ::

        early_share = Σ_{d ∈ D, |d| < max_atoms − 1} f(d)  /  Σ_{d ∈ D} f(d)   <= 0.05

    Not the absolute ``Σ_early f(d) <= 0.05``.  The two can give opposite
    verdicts: an instance whose *every* dead end is early but whose total
    dead-end mass is 1e-3 passes the absolute form and fails the conditional one
    — and the conditional one answers the question the audit exists for, which is
    *where* failures happen, not how many.  This is the Phase-1 handoff asking
    Phase 2 to distinguish "budget too small" from "masks too tight".

    **Measured under :class:`ForcedContinuationPolicy`, not the uniform one**
    (decision 16).  ``UniformPolicy`` includes ``STOP`` in its support whenever it
    is allowed, so it stops early, barely reaches dead ends at all, and would
    report a clean profile by construction.
    """
    cfg = cfg if cfg is not None else instance.cfg
    f, _, _ = forward_mass(ForcedContinuationPolicy(), graph)
    dead = graph.dead_ix
    mass = f[dead] if dead.shape[0] else np.zeros(0)
    total = float(mass.sum())
    by_size = np.zeros(cfg.max_atoms + 1, dtype=np.float64)
    if dead.shape[0]:
        np.add.at(by_size, graph.size[dead], mass)
    early = float(by_size[: max(0, cfg.max_atoms - 1)].sum())
    return {
        "absorption_by_size": by_size.tolist(),
        "total_dead_end_mass": total,
        # The denominator is non-zero because criterion 14 requires at least one
        # reachable dead end; `nan` here means that criterion already failed.
        "early_share": (early / total) if total > 0 else math.nan,
    }


# --------------------------------------------------------------------------
# d informativeness / G5
# --------------------------------------------------------------------------


def _deficits(instance: "LatticeInstance", graph: StateGraph) -> np.ndarray:
    pool, q, G = instance.pool, instance.obligations, instance.graph
    d = np.empty((graph.n_states, len(ob.DEFICIT_COMPONENTS)), dtype=np.float64)
    for i in range(graph.n_states):
        d[i] = ob.deficit(graph.atoms_of(i), pool, q, G)
    # Quantised before comparison: `d_time` is a ratio of interval measures, so
    # two states that cover the same window can differ in the last bits.
    return np.round(d, 9)


def d_informativeness(
    instance: "LatticeInstance", graph: StateGraph, cfg: Config | None = None
) -> dict[str, Any]:
    """**The most important audit in the phase** (G5).

    Gate 2's decision rule is "L7 beats capacity-matched L6 on exact TV", and L7
    *is* L6 plus ``Δd`` as input features and nothing else.  So the entire
    discriminating power of Gate 2 rests on ``Δd`` carrying information on this
    environment.  A null result that cannot distinguish "the hypothesis is false"
    from "the instrument could not resolve it" is the worst outcome available,
    because it looks like an answer.

    Two degeneracies are ruled out.  **Is ``d`` determined by ``|s|``?**  If it
    were, conditioning on ``Δd`` would be conditioning on a re-parameterised step
    counter.  **How often is ``Δd`` zero?**  That is a ceiling on how much L7's
    extra input can help, because its signal is silent on those transitions.

    **Both densities, because they answer different questions**, and each needs a
    formula because "fraction of transitions" has several readings that disagree.
    Let ``E`` be the set of ``ADD`` edges ``(s, a, s')``::

        structural          = |{e ∈ E : Δd(e) = 0}| / |E|
        visitation-weighted = Σ_{e ∈ E, Δd(e)=0} f(s)·P_F(a|s) / Σ_{e ∈ E} f(s)·P_F(a|s)

    The structural one describes the environment; the visitation-weighted one
    describes the transitions a learner actually samples, which is what Gate 2's
    comparison is made of.  Reporting only the flattering one would be a choice
    made after seeing them.

    Three exclusions, each of which would otherwise move the number:

    * **``STOP`` transitions are excluded** — ``STOP`` does not change the
      selected set, so ``Δd = 0`` for every one of them by construction, and
      counting them would inflate the zero fraction mechanically;
    * **dead-end absorption is excluded** — it is not a transition;
    * **weights are visitation mass** ``f(s)·P_F(a|s)``, normalised over ``ADD``
      edges — global conditioning on "an ``ADD`` was taken".  The rejected
      alternative is per-state renormalisation, which upweights states where
      *stopping* is likely and answers "what does a typical visited state look
      like" rather than "what does a typical sampled transition look like".  L7
      conditions on transitions, so the second is the question.
    """
    cfg = cfg if cfg is not None else instance.cfg
    d = _deficits(instance, graph)
    delta = d[graph.edge_parent] - d[graph.edge_child]
    is_zero = np.all(delta == 0, axis=1)

    f, edge_prob, _ = forward_mass(UniformPolicy(), graph)
    weights = f[graph.edge_parent] * edge_prob
    total_w = float(weights.sum())

    per_size: dict[int, int] = {}
    for s in range(cfg.max_atoms + 1):
        lo, hi = graph.state_slice(s)
        if hi > lo:
            per_size[s] = len({tuple(row) for row in d[lo:hi]})

    return {
        "zero_delta_d_structural": float(is_zero.mean()) if is_zero.size else math.nan,
        "zero_delta_d_visitation": (
            float(weights[is_zero].sum() / total_w) if total_w > 0 else math.nan
        ),
        "distinct_d_values": len({tuple(row) for row in d}),
        "distinct_d_by_size": per_size,
        "sizes_with_varying_d": sum(1 for v in per_size.values() if v > 1),
        "add_transitions": int(graph.n_edges),
    }


# --------------------------------------------------------------------------
# target mass / G10
# --------------------------------------------------------------------------


def jaccard_spread(graph: StateGraph, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Pairwise Jaccard over valid terminals — **descriptive, never banded**.

    Retained from the clustering definition that decision 13 rejected, because
    the *measurement* is informative even though the definition was not: that
    definition connected terminals whose similarity was **<= 0.5** and called the
    components modes, which is the *dissimilarity* relation, so its components
    grouped unlike proofs together.

    Sampled rather than exhaustive: at 5,000 terminals the full matrix is 12.5M
    pairs for a number that carries no band.
    """
    n = graph.n_terminals
    if n < 2:
        return {"jaccard_mean": math.nan, "jaccard_std": math.nan, "pairs": 0}
    rng = rng if rng is not None else np.random.default_rng(_JACCARD_SEED)
    pairs = min(_JACCARD_PAIRS, n * (n - 1) // 2)
    masks = graph.mask[graph.terminal_ix].astype(object)
    i = rng.integers(0, n, size=pairs)
    j = (i + 1 + rng.integers(0, n - 1, size=pairs)) % n
    scores = np.empty(pairs, dtype=np.float64)
    for k in range(pairs):
        a, b = int(masks[i[k]]), int(masks[j[k]])
        inter = bin(a & b).count("1")
        union = bin(a | b).count("1")
        scores[k] = inter / union if union else 0.0
    return {
        "jaccard_mean": float(scores.mean()),
        "jaccard_std": float(scores.std(ddof=1)) if pairs > 1 else 0.0,
        "pairs": int(pairs),
    }


# --------------------------------------------------------------------------
# the planted structure / criterion 17
# --------------------------------------------------------------------------


def structural_assertions(
    instance: "LatticeInstance", graph: StateGraph, cfg: Config | None = None
) -> dict[str, Any]:
    """**The planted structure is asserted to exist** (criterion 17, decision 24).

    Nothing else in the audit list does, and every other criterion can pass on an
    instance whose deliberate mechanisms are broken.  Criterion 14 needs only
    *one* reachable dead end, which a cap-induced one supplies even if both
    planted failure routes are dead.  Worst of all, a malformed ``P_B`` produces
    zero alternative-mode mass, which G10 treats as a **diagnostic** — so a broken
    instance would be written up as "effectively unimodal" instead of rejected.
    """
    cfg = cfg if cfg is not None else instance.cfg
    pool, q, G, meta = instance.pool, instance.obligations, instance.graph, instance.meta
    out: dict[str, Any] = {}

    # -- both templates are valid, closed, reachable terminals -------------
    for name, template in (("A", instance.template_a), ("B", instance.template_b)):
        state = graph.state_of(template)
        out[f"template_{name}_reachable"] = state is not None
        out[f"template_{name}_valid"] = bool(H(template, q, G, pool, cfg, ledger=None))
        out[f"template_{name}_is_terminal"] = bool(
            state is not None and graph.stop_allowed[state]
        )
    out["templates_differ"] = instance.template_a != instance.template_b
    out["template_overlap"] = instance.template_overlap()

    # -- each planted failure mechanism, **separately** --------------------
    def mechanism(atoms: Iterable[str], category: str) -> dict[str, Any]:
        atoms = tuple(atoms)
        state = graph.state_of(atoms)
        result = H(atoms, q, G, pool, cfg, ledger=None)
        return {
            "atoms": sorted(atoms),
            "reachable": state is not None,
            "invalid": not result.ok,
            "fires_expected_check": category in result.categories(),
        }

    dup_pair = tuple(meta.get("duplicate_slot_pair", ()))
    dup_atoms = set(dup_pair)
    for aid in dup_pair:
        dup_atoms.update(pool[aid].refs)
    out["duplicate_slot_mechanism"] = mechanism(dup_atoms, CHECK_BINDING)

    disjoint = meta.get("disjoint_binding")
    disjoint_atoms = {disjoint, *pool[disjoint].refs} if disjoint else set()
    out["temporal_disjoint_mechanism"] = mechanism(disjoint_atoms, CHECK_TEMPORAL)

    # -- tiers, features, intervals, targets -------------------------------
    # **Counted over atoms whose ``Source`` actually resolves, and by tier
    # *key* rather than by score.**  ``source_tier`` returns ``default_tier`` for
    # an atom with no source at all, and ``unknown`` is itself a tier carrying
    # that same score — so counting distinct scores over the whole pool would let
    # one resolved tier plus a pile of defaulted atoms satisfy criterion 17's
    # ">= 2 distinct source tiers occur **among atoms whose Source resolves**".
    # That is a weaker guard than the criterion asks for, and it would have gone
    # on passing if the generator ever stopped varying tiers.
    resolved = [source_node(atom, G) for atom in pool]
    out["distinct_source_tiers"] = len(
        {node.payload.get(PAYLOAD_TIER) for node in resolved if node is not None}
    )
    out["atoms_without_resolved_source"] = sum(1 for node in resolved if node is None)
    feats = {tuple(np.round(atom.feat, 6).tolist()) for atom in pool}
    out["distinct_feature_vectors"] = len(feats)
    out["featureless_atoms"] = sum(1 for atom in pool if not np.any(atom.feat))

    constraint = q.time_constraint
    out["constraint_bounded"] = bool(
        constraint is not None and constraint.start is not None and constraint.end is not None
    )
    partial = 0
    if constraint is not None:
        for atom in pool:
            for iv in atom_intervals(atom, G):
                if not iv.overlaps(constraint):
                    continue
                contains = (
                    (iv.start is None or iv.start <= constraint.start)
                    and (iv.end is None or iv.end >= constraint.end)
                )
                if not contains:
                    partial += 1
    out["partially_overlapping_intervals"] = partial

    from graft.core.resolve import target_resolves

    out["atoms_with_unresolved_target"] = [
        atom.atom_id for atom in pool if not target_resolves(atom, G)
    ]

    # -- the negative cases are present **and unreachable** ----------------
    # An earlier draft asked for them to be "reachable", which Phase 1's masks
    # make impossible by design: a per-atom violation is permanent, so such atoms
    # are pruned.  They are negative cases for sub-checks 4 and 7, not selectable
    # evidence (decision 24).
    counts = G.counts()
    out["invalidated_edges_in_snapshot"] = counts["edges"] - counts["live_edges"]
    out["quarantined_assertions_in_snapshot"] = (
        counts["assertions"] - counts["eligible_assertions"]
    )
    negatives = [meta.get("retired_edge_atom"), *meta.get("quarantined_atoms", ())]
    reachable_atoms: set[str] = set()
    if graph.n_edges:
        reachable_atoms = {graph.atom_ids[a] for a in np.unique(graph.edge_action).tolist()}
    out["negative_case_atoms"] = [a for a in negatives if a]
    out["negative_case_atoms_reachable"] = [
        a for a in negatives if a and a in reachable_atoms
    ]
    return out


# --------------------------------------------------------------------------
# the whole suite
# --------------------------------------------------------------------------


def run_audits(
    instance: "LatticeInstance",
    graph: StateGraph | None = None,
    target: Target | None = None,
    *,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Every audit, for one instance.  Aggregation is the caller's job."""
    cfg = cfg if cfg is not None else instance.cfg
    g = graph if graph is not None else reachable_states(instance, cfg)
    t = target if target is not None else target_distribution(instance, cfg, graph=g)
    return {
        "suite": instance.meta.get("suite", instance.spec.label),
        "counts": g.counts(),
        "pool_size": len(instance.pool),
        "pool_cap": instance.pool.cap,
        "environment_fingerprint": environment_fingerprint(instance, g),
        "closure": closure_audit(instance, g, cfg),
        "collisions": collision_audit(g),
        "fail": fail_reachability(g),
        "absorption": dead_end_absorption(instance, g, cfg),
        "delta_d": d_informativeness(instance, g, cfg),
        "target_mass": t.mass_profile(),
        "jaccard": jaccard_spread(g),
        "structure": structural_assertions(instance, g, cfg),
    }


# --------------------------------------------------------------------------
# generation-time bands
# --------------------------------------------------------------------------


def band_report(instance: "LatticeInstance") -> dict[str, Any]:
    """Every band a generated instance must satisfy, or :class:`BandViolation`.

    Called once per generation attempt.  Ordered cheapest-first so a rejection
    costs as little as possible: the enumerator aborts on the state and edge caps
    before anything downstream runs (G1).
    """
    spec = instance.spec
    cfg = spec.cfg

    if not POOL_MIN <= len(instance.pool) <= POOL_MAX:
        raise BandViolation(
            "pool_size",
            len(instance.pool),
            f"{POOL_MIN} <= |pool| <= {POOL_MAX}, the architecture's universe size",
        )
    if instance.pool.cap != cfg.pool_cap:
        raise BandViolation(
            "pool_cap",
            instance.pool.cap,
            f"pool built as AtomPool(atoms, cap=cfg.pool_cap={cfg.pool_cap}); "
            "AtomPool accepts cap=None by design, so nothing else catches this",
        )
    if instance.template_overlap() > 0.5:
        raise BandViolation(
            "template_overlap",
            round(instance.template_overlap(), 3),
            "|P_A ∩ P_B| / |P_A ∪ P_B| <= 0.5 (decision 23); P_A != P_B alone "
            "permits a one-atom difference, which is not materially different evidence",
        )

    graph = reachable_states(instance, cfg)  # raises on the state and edge caps
    if not spec.min_terminals <= graph.n_terminals <= spec.max_terminals:
        raise BandViolation(
            "valid_terminals",
            graph.n_terminals,
            f"{spec.min_terminals} <= k <= {spec.max_terminals} (G1)",
        )
    if graph.n_dead_ends < 1:
        raise BandViolation(
            "fail_reachability",
            0,
            ">= 1 reachable dead end, or STOP-masking is doing nothing (G4)",
        )

    absorption = dead_end_absorption(instance, graph, cfg)
    if absorption["early_share"] > spec.max_early_dead_end_share:
        raise BandViolation(
            "early_dead_end_share",
            round(absorption["early_share"], 4),
            f"<= {spec.max_early_dead_end_share} of dead-end mass below "
            f"|X| = max_atoms - 1; more means the ADD masks are too tight, "
            "not that the budget is small",
        )

    delta = d_informativeness(instance, graph, cfg)
    if spec.enforce_delta_d:
        if delta["sizes_with_varying_d"] < spec.min_d_varying_sizes:
            raise BandViolation(
                "d_determined_by_size",
                delta["sizes_with_varying_d"],
                f">= {spec.min_d_varying_sizes} set sizes at which d is not "
                "determined by |s| (G5)",
            )
        if delta["distinct_d_values"] < spec.min_distinct_d:
            raise BandViolation(
                "distinct_d_values", delta["distinct_d_values"], f">= {spec.min_distinct_d} (G5)"
            )
        for key in ("zero_delta_d_structural", "zero_delta_d_visitation"):
            if delta[key] > spec.max_zero_delta_d:
                raise BandViolation(
                    key,
                    round(delta[key], 4),
                    f"<= {spec.max_zero_delta_d}; above it Gate 2 cannot resolve "
                    "L7 from L6 (G5)",
                )

    target = target_distribution(instance, cfg, graph=graph)
    mass = target.mass_profile()
    if spec.enforce_neither_mass and mass["mode_mass"]["neither"] > spec.max_neither_mass:
        raise BandViolation(
            "neither_mass",
            round(mass["mode_mass"]["neither"], 4),
            f"<= {spec.max_neither_mass} of p* on terminals completing no designed "
            f"proof, at beta={target.beta} (G10)",
        )

    structure = structural_assertions(instance, graph, cfg)
    _assert_structure(structure)

    return {
        "counts": graph.counts(),
        "pool_size": len(instance.pool),
        "absorption": absorption,
        "delta_d": delta,
        "target_mass": mass,
        "structure": structure,
    }


def _assert_structure(structure: Mapping[str, Any]) -> None:
    """Turn the criterion-17 report into a rejection."""
    required_true = (
        "template_A_reachable",
        "template_A_valid",
        "template_A_is_terminal",
        "template_B_reachable",
        "template_B_valid",
        "template_B_is_terminal",
        "templates_differ",
        "constraint_bounded",
    )
    for key in required_true:
        if not structure[key]:
            raise BandViolation("structure", key, "true (criterion 17)")
    for name in ("duplicate_slot_mechanism", "temporal_disjoint_mechanism"):
        mech = structure[name]
        if not (mech["reachable"] and mech["invalid"] and mech["fires_expected_check"]):
            raise BandViolation("structure", {name: mech}, "a reachable invalid state "
                                "firing its own sub-check, independently of the other")
    if structure["distinct_source_tiers"] < 2:
        raise BandViolation("source_tiers", structure["distinct_source_tiers"], ">= 2")
    if structure["distinct_feature_vectors"] < 2 or structure["featureless_atoms"]:
        raise BandViolation(
            "features",
            (structure["distinct_feature_vectors"], structure["featureless_atoms"]),
            ">= 2 distinct non-zero feat vectors and no featureless atom",
        )
    if structure["partially_overlapping_intervals"] < 1:
        raise BandViolation(
            "partial_intervals",
            0,
            ">= 1 claim interval that *partially* overlaps the constraint — neither "
            "disjoint nor containing, or temporal_correctness reverts to a presence flag",
        )
    if structure["atoms_with_unresolved_target"]:
        raise BandViolation(
            "targets",
            structure["atoms_with_unresolved_target"][:3],
            "every atom's target resolves in the backing snapshot (Phase-1 gap G2)",
        )
    if structure["invalidated_edges_in_snapshot"] < 1:
        raise BandViolation("invalidated_edges", 0, ">= 1 (sub-check 4 negative case)")
    if structure["quarantined_assertions_in_snapshot"] < 1:
        raise BandViolation("quarantined_assertions", 0, ">= 1 (sub-check 7 negative case)")
    if structure["negative_case_atoms_reachable"]:
        raise BandViolation(
            "negative_cases",
            structure["negative_case_atoms_reachable"],
            "the retired-edge and quarantined-claim atoms are present in the snapshot "
            "and **unreachable** — the masks prune them, so they are negative cases "
            "rather than selectable evidence (decision 24)",
        )
