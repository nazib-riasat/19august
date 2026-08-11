"""The ProofLattice generator — an environment whose properties are *declared*.

**This is a measuring instrument, not a toy dataset** (Phase-2 plan §0).  Gate 2
is the cheapest place in the project to learn that Contribution 3 does not hold,
and an instrument that cannot resolve the thing it is pointed at is worse than
none, because it still produces a number.  Every structural property Gate 2
depends on is therefore a **band that is verified by construction and rejected
when missed**, never an emergent property nobody measured (gaps G1, G5, G10).

**What the generator plants, and which requirement each discharges**

============================================  ==================================
feature                                        serves
============================================  ==================================
required entity anchor, requested value type   ``d_anchor``, ``d_value``, coverage
dependency structure via ``refs``              the ``ADD`` mask and the lattice
two substitutable claim chains                 multiple disjoint valid modes
duplicate-slot binding pairs                   sub-check 9; ``FAIL`` reachability
temporally disjoint claims                     sub-check 3; ``FAIL`` reachability
bounded, partially overlapping intervals       graded ``temporal_correctness``
mixed source tiers                             ``source_quality`` non-degenerate
clustered ``feat`` vectors                     ``redundancy`` non-degenerate
>= 1 invalidated edge, >= 1 quarantined         sub-checks 4 and 7, as *negative*
  assertion                                      cases: the masks prune them
distractor atoms                               makes ``U`` discriminate
backing ``DictGraphSnapshot``                  ``target`` must resolve (P1 gap G2)
============================================  ==================================

**Two atom populations, and the second one is deliberate.**  A pool holds 20-30
atoms (exit criterion 10) but only some of them are *admissible* — the rest carry
a permanent per-atom violation (a retired edge, a quarantined assertion) and the
``ADD`` mask prunes them at every state.  That is what a real Stage-C pool looks
like, and it is also the lever that keeps the enumeration inside its band: pool
size and state count are decoupled, so the universe can be the size the
architecture asks for without the lattice becoming unenumerable.

**``LatticeSpec`` carries the ``Config``.**  Pool construction must use
``cfg.pool_cap`` and enumeration must use ``cfg.max_atoms``.  A generator taking
only an ``rng`` would have to reach for a global or default them, and defaulting
to the *real* profile (64/16) instead of the synthetic one (32/8) would silently
produce an environment nobody can enumerate.

Plain classes rather than dataclasses: ``graft/tests/test_structure.py``
reserves ``@dataclass`` for ``schemas.py``, and these types cross module
boundaries.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from graft import ids
from graft.config import Config, load_config
from graft.graphstore import DictGraphSnapshot
from graft.synth.enumerate import BandViolation
from graft.schemas import (
    PAYLOAD_ALIASES,
    PAYLOAD_ASSERTION_ID,
    PAYLOAD_NAME,
    PAYLOAD_TIER,
    PAYLOAD_VALUE_TYPE,
    Assertion,
    AssertionFlags,
    AtomPool,
    CandidateAtom,
    Edge,
    Interval,
    Node,
    Obligations,
    ProofSet,
    SourceSpan,
    Turn,
)

__all__ = [
    "LatticeSpec",
    "LatticeInstance",
    "BandViolation",
    "generate",
    "tiny_instance",
    "benchmark_suite",
    "probe_suite",
    "tuning_suite",
    "MAIN_SEED",
    "PROBE_SEED",
    "TUNING_SEED",
    "SUITE_SIZES",
    "ENVIRONMENT_CONFIG_FIELDS",
]

#: Suite seeds (decision 12).  Deliberately **not** one of the training seeds
#: ``{13, 42, 7}`` — reusing one would tie the environment to a run's randomness.
MAIN_SEED = 20260808
PROBE_SEED = 20260809
TUNING_SEED = 20260811

SUITE_SIZES: Mapping[str, int] = {"main": 20, "probe": 5, "tuning": 5}

TS = "2026-08-08T00:00:00+00:00"
FEAT_DIM = 12
ANCHOR_NAME = "Anchor Entity"
VALUE_TYPE = "occupation"
CONV_ID = "conv-lattice"

#: The question's window.  Bounded on both sides (exit criterion 17): an
#: unbounded constraint makes ``covered_fraction`` return 1.0 by convention and
#: ``temporal_correctness`` stops doing any work at all (Phase-1 gap G5).
CONSTRAINT_START = 0.0
CONSTRAINT_END = 100.0

#: **[ANALYSIS]** The ``Config`` fields that change what is *built and
#: enumerated*, and therefore the only ones ``environment_fingerprint`` may bind
#: (decision 21).
#:
#: ``pool_cap`` sizes the pool, ``max_atoms`` sets both the enumeration depth and
#: ``H``'s size check, and ``profile`` names which environment family this is.
#: Everything else in the config is either reward (``beta``, ``u_weights``,
#: ``r_fail``, ``source_tiers``), budget (``K``, ``checker_budget``), ingestion
#: (``tau_nli``, ``support_policy``) or runtime — none of which changes a single
#: state, edge or flag.
#:
#: **Excluding the reward fields is the whole point.**  Putting the resolved
#: ``Config`` in wholesale made the environment digest move when β moved, so
#: freezing β after the Phase-3 sweep would have changed the identity of suites
#: that were frozen at Gate 0 — exactly the failure decision 21 exists to
#: prevent, and one that would also let a real structural divergence between two
#: machines be waved away as "β differs".
#:
#: The reward fields are not left unbound: ``target_fingerprint`` carries
#: ``(beta, u_weights, r_fail)`` explicitly and hashes the **computed** ``p*``,
#: which is what catches ``source_tiers`` too.  ``r_fail_margin`` binds nothing
#: numerically — it constrains which ``(beta, r_fail)`` pairs the loader accepts,
#: not what any number is — so it is deliberately in neither digest.
ENVIRONMENT_CONFIG_FIELDS: tuple[str, ...] = ("profile", "pool_cap", "max_atoms")


# ``BandViolation`` is defined in ``enumerate`` rather than here, because the
# earliest and cheapest rejection is the enumerator aborting the moment the state
# or edge count passes its cap (G1).  Importing it the other way round would make
# lattice -> enumerate -> lattice a cycle.


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------


class LatticeSpec:
    """The banded knobs, the structural knobs, and the ``Config``.

    Every field here is part of ``environment_fingerprint``, so two machines
    that disagree about a band cannot silently compare results (decision 21).
    """

    __slots__ = (
        "cfg",
        "min_terminals",
        "max_terminals",
        "max_states",
        "max_edges",
        "max_zero_delta_d",
        "min_distinct_d",
        "min_d_varying_sizes",
        "max_neither_mass",
        "max_early_dead_end_share",
        "enforce_delta_d",
        "enforce_neither_mass",
        "distractor_heavy",
        "n_distractor_nodes",
        "n_distractor_edges",
        "n_chain_edges",
        "pool_size",
        "max_attempts",
        "suite_budget_seconds",
        "label",
    )

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        # -- G1: what controls the size of the enumeration -------------------
        min_terminals: int = 200,
        max_terminals: int = 5_000,
        max_states: int = 100_000,
        max_edges: int = 2_000_000,
        # -- G5: what makes Gate 2 able to resolve L7 from L6 ----------------
        max_zero_delta_d: float = 0.6,
        min_distinct_d: int = 10,
        min_d_varying_sizes: int = 3,
        # -- G10 / dead ends -------------------------------------------------
        max_neither_mass: float = 0.5,
        max_early_dead_end_share: float = 0.05,
        # -- which bands this suite is scored on -----------------------------
        enforce_delta_d: bool = True,
        enforce_neither_mass: bool = True,
        distractor_heavy: bool = False,
        # -- structure -------------------------------------------------------
        n_distractor_nodes: tuple[int, int] = (1, 1),
        n_distractor_edges: tuple[int, int] = (0, 1),
        n_chain_edges: tuple[int, int] = (2, 3),
        pool_size: tuple[int, int] = (20, 26),
        max_attempts: int = 200,
        suite_budget_seconds: float = 1_800.0,
        label: str = "custom",
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config(preset="synthetic")
        if self.cfg.profile != "synthetic":
            raise ValueError(
                f"LatticeSpec needs the synthetic profile (pool_cap 32, max_atoms 8); "
                f"got profile {self.cfg.profile!r} with pool_cap={self.cfg.pool_cap}, "
                f"max_atoms={self.cfg.max_atoms}. The real profile's 64/16 produces an "
                "environment nobody can enumerate."
            )
        self.min_terminals = int(min_terminals)
        self.max_terminals = int(max_terminals)
        self.max_states = int(max_states)
        self.max_edges = int(max_edges)
        self.max_zero_delta_d = float(max_zero_delta_d)
        self.min_distinct_d = int(min_distinct_d)
        self.min_d_varying_sizes = int(min_d_varying_sizes)
        self.max_neither_mass = float(max_neither_mass)
        self.max_early_dead_end_share = float(max_early_dead_end_share)
        self.enforce_delta_d = bool(enforce_delta_d)
        self.enforce_neither_mass = bool(enforce_neither_mass)
        self.distractor_heavy = bool(distractor_heavy)
        self.n_distractor_nodes = (int(n_distractor_nodes[0]), int(n_distractor_nodes[1]))
        self.n_distractor_edges = (int(n_distractor_edges[0]), int(n_distractor_edges[1]))
        self.n_chain_edges = (int(n_chain_edges[0]), int(n_chain_edges[1]))
        self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        self.max_attempts = int(max_attempts)
        self.suite_budget_seconds = float(suite_budget_seconds)
        self.label = str(label)

    def to_dict(self) -> dict[str, Any]:
        """Everything ``environment_fingerprint`` binds.

        Only :data:`ENVIRONMENT_CONFIG_FIELDS` of the ``Config`` go in — the
        digest must be **independent of β** (decision 21), and the resolved
        config carries β, ``u_weights`` and ``r_fail`` alongside the caps.
        """
        cfg = self.cfg.to_dict()
        return {
            "cfg": {name: cfg[name] for name in ENVIRONMENT_CONFIG_FIELDS},
            "min_terminals": self.min_terminals,
            "max_terminals": self.max_terminals,
            "max_states": self.max_states,
            "max_edges": self.max_edges,
            "max_zero_delta_d": self.max_zero_delta_d,
            "min_distinct_d": self.min_distinct_d,
            "min_d_varying_sizes": self.min_d_varying_sizes,
            "max_neither_mass": self.max_neither_mass,
            "max_early_dead_end_share": self.max_early_dead_end_share,
            "enforce_delta_d": self.enforce_delta_d,
            "enforce_neither_mass": self.enforce_neither_mass,
            "distractor_heavy": self.distractor_heavy,
            "n_distractor_nodes": list(self.n_distractor_nodes),
            "n_distractor_edges": list(self.n_distractor_edges),
            "n_chain_edges": list(self.n_chain_edges),
            "pool_size": list(self.pool_size),
            "label": self.label,
        }

    def replace(self, **kwargs: Any) -> "LatticeSpec":
        current = {name: getattr(self, name) for name in self.__slots__}
        current.update(kwargs)
        cfg = current.pop("cfg")
        return LatticeSpec(cfg, **current)


# --------------------------------------------------------------------------
# instance
# --------------------------------------------------------------------------


class LatticeInstance:
    """One coherent (pool, obligations, snapshot, gold, templates) environment.

    ``template_a`` and ``template_b`` are **complete minimal proof templates**,
    each the full atom set its proof needs *including atoms shared with the
    other* (decision 13).  Mode buckets are then ``P_A ⊆ X`` — a predicate with
    no ambiguity — rather than "contains an atom unique to chain A", which would
    bucket a chain-head plus seven distractors as a completed proof.
    """

    __slots__ = ("pool", "obligations", "graph", "gold", "template_a", "template_b", "spec", "meta")

    def __init__(
        self,
        pool: AtomPool,
        obligations: Obligations,
        graph: DictGraphSnapshot,
        gold: ProofSet,
        template_a: frozenset[str],
        template_b: frozenset[str],
        spec: LatticeSpec,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.pool = pool
        self.obligations = obligations
        self.graph = graph
        self.gold = gold
        self.template_a = frozenset(template_a)
        self.template_b = frozenset(template_b)
        self.spec = spec
        self.meta: dict[str, Any] = dict(meta or {})

    @property
    def cfg(self) -> Config:
        return self.spec.cfg

    def identity_payload(self) -> dict[str, Any]:
        """What ``environment_fingerprint`` hashes, minus the enumerated graph.

        Pool atoms go in whole — including ``feat`` — because a changed feature
        vector changes ``redundancy`` and therefore every reward on the lattice.
        """
        return {
            "spec": self.spec.to_dict(),
            "pool_cap": self.pool.cap,
            "pool": [atom.to_dict() for atom in self.pool],
            "obligations": self.obligations.to_dict(),
            "snapshot": self.graph.state_digest(),
            "gold": sorted(self.gold.atoms),
            "template_a": sorted(self.template_a),
            "template_b": sorted(self.template_b),
        }

    def template_overlap(self) -> float:
        """``|P_A ∩ P_B| / |P_A ∪ P_B|`` (decision 23).

        ``P_A != P_B`` permits a one-atom difference, which is not "materially
        different evidence" — the property v1.2 §9 wants demonstrated.
        """
        union = self.template_a | self.template_b
        if not union:
            return 0.0
        return len(self.template_a & self.template_b) / len(union)


# --------------------------------------------------------------------------
# graph construction helpers
# --------------------------------------------------------------------------


def _feat(rng: np.random.Generator, cluster: int) -> np.ndarray:
    """A feature vector near one of a few cluster centres.

    Clustered rather than uniform so ``redundancy`` is non-degenerate: two atoms
    from one cluster overlap, atoms from different clusters do not, and the term
    needs both to exist.

    **Clusters are assigned, not sequential.**  The atoms of a designed proof get
    distinct clusters and the distractors reuse them, so a proof of complementary
    evidence scores low ``redundancy`` and a pile of interchangeable distractors
    scores high.  A sequential assignment made ``redundancy`` a function of pool
    position, which is a property of the generator rather than of the evidence.
    """
    base = np.zeros(FEAT_DIM, dtype=np.float64)
    base[cluster % FEAT_DIM] = 1.0
    noise = rng.normal(0.0, 0.10, size=FEAT_DIM)
    return np.asarray(base + noise, dtype=np.float32)


class _Builder:
    """Accumulates the snapshot and the pool together, so every atom's ``target``
    resolves by construction rather than by review."""

    def __init__(self, rng: np.random.Generator, cfg: Config) -> None:
        self.rng = rng
        self.cfg = cfg
        self.turns: list[Turn] = []
        self.spans: list[SourceSpan] = []
        self.assertions: list[Assertion] = []
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.atoms: list[CandidateAtom] = []
        self.node_atom: dict[str, str] = {}
        self.edge_atom: dict[str, str] = {}
        self._cluster = 0
        self._turn = 0

    # -- graph -----------------------------------------------------------

    def span_for(self, tag: str) -> SourceSpan:
        turn_id = ids.node_id("Turn", f"{tag}-turn-{self._turn}")
        self.turns.append(
            Turn(
                turn_id=turn_id,
                conv_id=CONV_ID,
                session_id="s0",
                speaker="user" if self._turn % 2 == 0 else "assistant",
                ts=TS,
                text=f"utterance {tag}",
            )
        )
        self._turn += 1
        span = SourceSpan(span_id=ids.span_id(turn_id, 0, 12), turn_id=turn_id, start=0, end=12)
        self.spans.append(span)
        return span

    def assertion_for(self, tag: str, span: SourceSpan, *, eligible: bool) -> str:
        aid = ids.assertion_id("claim", tag, [span.span_id])
        self.assertions.append(
            Assertion(
                assertion_id=aid,
                kind="claim",
                text_norm=tag,
                spans=(span.span_id,),
                flags=AssertionFlags(
                    asserted_by="user", entailed_by_span=True, entailed_score=0.93
                ),
                t_created=TS,
                eligibility="eligible" if eligible else "quarantined",
            )
        )
        return aid

    def node(self, ntype: str, key: str, payload: Mapping[str, Any] | None = None) -> str:
        nid = ids.node_id(ntype, key)
        self.nodes.append(Node(node_id=nid, ntype=ntype, payload=dict(payload or {})))
        return nid

    def edge(
        self,
        etype: str,
        src: str,
        dst: str,
        span: SourceSpan,
        *,
        label: str = "",
        invalid: bool = False,
    ) -> str:
        eid = ids.edge_id(etype, src, dst, label)
        self.edges.append(
            Edge(
                edge_id=eid,
                etype=etype,
                src=src,
                dst=dst,
                t_created=TS,
                provenance=(span.span_id,),
                t_invalid=TS if invalid else None,
            )
        )
        return eid

    # -- atoms -----------------------------------------------------------

    def _next_cluster(self, cluster: int | None) -> int:
        if cluster is not None:
            return cluster
        self._cluster += 1
        return self._cluster

    def node_atom_for(self, node_id: str, label: str, cluster: int | None = None) -> str:
        aid = ids.atom_id("node", (), node_id)
        self.atoms.append(
            CandidateAtom(
                atom_id=aid,
                kind="node",
                target=node_id,
                label=label,
                feat=_feat(self.rng, self._next_cluster(cluster)),
            )
        )
        self.node_atom[node_id] = aid
        return aid

    def edge_atom_for(
        self, edge_id: str, src: str, dst: str, label: str, cluster: int | None = None
    ) -> str:
        refs = (self.node_atom[src], self.node_atom[dst])
        aid = ids.atom_id("edge", refs, edge_id, label)
        self.atoms.append(
            CandidateAtom(
                atom_id=aid,
                kind="edge",
                refs=refs,
                target=edge_id,
                label=label,
                feat=_feat(self.rng, self._next_cluster(cluster)),
            )
        )
        self.edge_atom[edge_id] = aid
        return aid

    def binding_atom(
        self, refs: Sequence[str], slot: str = "answer", cluster: int | None = None
    ) -> str:
        aid = ids.atom_id("binding", tuple(refs), "", slot)
        self.atoms.append(
            CandidateAtom(
                atom_id=aid,
                kind="binding",
                refs=tuple(refs),
                label=slot,
                feat=_feat(self.rng, self._next_cluster(cluster)),
            )
        )
        return aid

    def snapshot(self) -> DictGraphSnapshot:
        return DictGraphSnapshot(
            snapshot_id=1,
            nodes=self.nodes,
            edges=self.edges,
            assertions=self.assertions,
            turns=self.turns,
            spans=self.spans,
        )


def _interval_node(b: _Builder, key: str, start: float, end: float) -> str:
    return b.node("TimeInterval", key, {"start": float(start), "end": float(end)})


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


def _attempt(rng: np.random.Generator, spec: LatticeSpec) -> LatticeInstance:
    """Build one candidate instance.  Bands are checked by :func:`generate`.

    The shape is fixed and the *quantities* are drawn, because that is what makes
    rejection work: a suite of twenty structurally identical lattices would
    either all pass a band or all fail it, and the band would be testing the
    design rather than the instance.
    """
    cfg = spec.cfg
    b = _Builder(rng, cfg)
    tiers = sorted(cfg.source_tiers)

    # -- sources, one node per tier so source_quality has a real spread ----
    source_of: dict[str, str] = {}
    for tier in tiers:
        source_of[tier] = b.node("Source", tier, {PAYLOAD_TIER: tier})

    # -- the anchor entity and one distractor entity -----------------------
    e_anchor = b.node(
        "Entity", "anchor", {PAYLOAD_NAME: ANCHOR_NAME, PAYLOAD_ALIASES: ["AE"]}
    )
    e_other = b.node("Entity", "other", {PAYLOAD_NAME: "Other Entity", PAYLOAD_ALIASES: []})

    # -- intervals.  Bounded, and *partially* overlapping the constraint: with
    # every interval either exact or disjoint, ``temporal_correctness`` collapses
    # to a presence flag and ``d_time`` carries one bit (PHASE1_DECISIONS.md §4).
    # Jittered per instance so the suite is not twenty copies of one lattice.
    #
    # ``iv_c`` is the **complement** of the gold window rather than a second copy
    # of it.  That is what lets extra temporal evidence *add* coverage — so an
    # ``ADD`` moves ``d_time`` and the zero-``Δd`` fraction stays inside its band
    # (G5) — while leaving the gold interval expensive to drop, because nothing
    # else covers the part of the constraint it covers.
    span_a = float(rng.integers(55, 72))
    iv_a = (0.0, span_a)                                    # the gold window
    iv_c = (span_a, CONSTRAINT_END)                         # its complement
    iv_p = (float(rng.integers(30, 50)), 150.0)             # overlaps both, runs past
    iv_x = (200.0, 300.0)                                   # disjoint — the contradiction

    n_ia = _interval_node(b, "iv-a", *iv_a)
    n_ic = _interval_node(b, "iv-c", *iv_c)
    n_ip = _interval_node(b, "iv-p", *iv_p)
    n_ix = _interval_node(b, "iv-x", *iv_x)

    # -- the two substitutable claim chains --------------------------------
    def chain(
        tag: str, interval_node: str, tier: str, *, value_type: str, value: str | None = None
    ) -> dict[str, str]:
        span = b.span_for(tag)
        aid = b.assertion_for(f"claim-{tag}", span, eligible=True)
        claim = b.node("Claim", f"claim-{tag}", {PAYLOAD_ASSERTION_ID: aid})
        owns_value = value is None
        if owns_value:
            value = b.node(
                "Value", f"value-{tag}", {PAYLOAD_ASSERTION_ID: aid, PAYLOAD_VALUE_TYPE: value_type}
            )
        b.edge("about_entity", claim, e_anchor, span)
        b.edge("has_value", claim, value, span)
        b.edge("valid_during", claim, interval_node, span)
        b.edge("asserted_by", claim, source_of[tier], span)
        if owns_value:
            # One `asserted_by` per Value: `source_tier` returns the first live
            # one it finds, so a second would make the term depend on edge order.
            b.edge("asserted_by", value, source_of[tier], span)
        return {"span": span, "claim": claim, "value": value, "interval": interval_node}

    tier_a, tier_b = (tiers[i] for i in rng.permutation(len(tiers))[:2])
    ch_a = chain("a", n_ia, tier_a, value_type=VALUE_TYPE)
    # **The two chains reach the same answer**, so they share the Value node and
    # the validity window.  That is what makes them substitutable rather than
    # merely different, and it leaves the requested value type and the requested
    # period with *unique* providers: dropping either from a proof costs the
    # obligation slot outright instead of being repaired by the other chain's
    # copy.  Interchangeable copies made every near-miss of the gold proof cheap,
    # which is what put target mass in the `neither` bucket (G10).
    ch_b = chain("b", n_ia, tier_b, value_type=VALUE_TYPE, value=ch_a["value"])
    b.edge("valid_during", ch_b["claim"], n_ic, b.span_for("b-alt"))

    # -- the temporally disjoint referent (planted mechanism 2, G4) ---------
    # A distractor Value the graph says was only valid outside the window.  Hung
    # on a Value rather than on a Claim of its own so the lattice does not pay a
    # whole extra free node atom for it: every free node atom multiplies the
    # state count and adds one more way to build a near-miss.
    span_d = b.span_for("d")
    aid_d = b.assertion_for("claim-d", span_d, eligible=True)
    v_other = b.node(
        "Value", "value-other", {PAYLOAD_ASSERTION_ID: aid_d, PAYLOAD_VALUE_TYPE: "location"}
    )
    b.edge("asserted_by", v_other, source_of[tiers[0]], span_d)
    b.edge("valid_during", v_other, n_ix, span_d)

    # -- live distractor edges ---------------------------------------------
    b.edge("has_value", ch_b["claim"], v_other, span_d, label="alt")
    b.edge("about_entity", ch_a["claim"], e_other, span_d, label="alt")

    # -- the invalidated edge (negative case for sub-check 4) --------------
    # Deliberately a twin of the useful `has_value` edge: an atom that looks
    # exactly like admissible evidence and is pruned only because the snapshot
    # says the edge was retired.
    e_retired = b.edge(
        "has_value", ch_a["claim"], ch_a["value"], ch_a["span"], label="stale", invalid=True
    )

    # -- quarantined claims (negative case for sub-check 7) ----------------
    # These pad the pool to the architecture's 20-30 atom universe *without*
    # enlarging the lattice: each carries a permanent per-atom violation, so the
    # `ADD` mask prunes it at every state.  Pool size and state count are
    # therefore independent knobs, which is what makes exit criterion 10 and the
    # G1 bands satisfiable at the same time.
    pool_target = int(rng.integers(spec.pool_size[0], spec.pool_size[1] + 1))
    blocked_claims: list[str] = []

    def add_blocked_claim(i: int) -> str:
        span_q = b.span_for(f"q{i}")
        aid_q = b.assertion_for(f"claim-q{i}", span_q, eligible=False)
        c_q = b.node("Claim", f"claim-q{i}", {PAYLOAD_ASSERTION_ID: aid_q})
        b.edge("has_value", c_q, ch_a["value"], span_q)
        b.edge("asserted_by", c_q, source_of[tiers[-1]], span_q)
        blocked_claims.append(c_q)
        return c_q

    for i in range(pool_target):  # an upper bound; only some become atoms
        add_blocked_claim(i)

    graph = b.snapshot()

    # ---------------------------------------------------------------------
    # atoms.  Nodes first, then edges (which reference node atoms), then
    # bindings — the ordering that makes closure satisfiable by construction.
    #
    # Feature clusters are *assigned*, not sequential: the atoms of each designed
    # proof get distinct clusters, and distractors and edges reuse them.  A proof
    # of complementary evidence then scores `redundancy` near 0 while a pile of
    # interchangeable distractors scores high — which is what the term is for.
    # ---------------------------------------------------------------------
    C_ANCHOR, C_CA, C_VAL, C_TA, C_CB, C_VO = 0, 1, 2, 3, 4, 5
    C_BA, C_BB, C_BX, C_BLOCK = 6, 7, 8, 9

    a_anchor = b.node_atom_for(e_anchor, "Entity", C_ANCHOR)
    a_ca = b.node_atom_for(ch_a["claim"], "Claim", C_CA)
    a_va = b.node_atom_for(ch_a["value"], "Value", C_VAL)
    a_ta = b.node_atom_for(n_ia, "TimeInterval", C_TA)
    a_cb = b.node_atom_for(ch_b["claim"], "Claim", C_CB)
    a_vo = b.node_atom_for(v_other, "Value", C_VO)

    # admissible distractor node atoms, deliberately reusing template clusters
    n_dnodes = int(rng.integers(spec.n_distractor_nodes[0], spec.n_distractor_nodes[1] + 1))
    if spec.distractor_heavy:
        n_dnodes += 2
    distractor_nodes: list[str] = []
    for node_id, label, cluster in (
        (n_ic, "TimeInterval", C_TA),
        (n_ip, "TimeInterval", C_VO),
        (e_other, "Entity", C_ANCHOR),
    )[:n_dnodes]:
        distractor_nodes.append(b.node_atom_for(node_id, label, cluster))

    def edge_atom(etype: str, src: str, dst: str, cluster: int, label: str = "") -> str:
        return b.edge_atom_for(ids.edge_id(etype, src, dst, label), src, dst, etype, cluster)

    # **How many chain edge atoms is a knob, and it moves two bands at once.**
    # An edge atom addresses no obligation slot, so every `ADD` of one has
    # `Δd = 0`; it is also one more way to build a near-miss of the gold proof.
    # Cutting them lowers the zero-`Δd` fraction (G5) *and* the `neither` mass
    # (G10), which is why the two bands are satisfiable together at all.
    n_chain_edges = int(rng.integers(spec.n_chain_edges[0], spec.n_chain_edges[1] + 1))
    if spec.distractor_heavy:
        n_chain_edges += 2
    admissible_edges = [
        edge_atom(*parts)
        for parts in (
            ("has_value", ch_a["claim"], ch_a["value"], C_VAL),
            ("about_entity", ch_a["claim"], e_anchor, C_ANCHOR),
            ("valid_during", ch_a["claim"], n_ia, C_TA),
            ("has_value", ch_b["claim"], ch_b["value"], C_VAL),
            ("about_entity", ch_b["claim"], e_anchor, C_ANCHOR),
            ("valid_during", ch_b["claim"], n_ia, C_CB),
        )[:n_chain_edges]
    ]
    # Distractor edges are only offered where *both* endpoints already have node
    # atoms: an edge atom whose refs can never all be selected is permanently
    # unaddable dead weight, which `AtomPool` accepts and the lattice would carry.
    n_dedges = int(rng.integers(spec.n_distractor_edges[0], spec.n_distractor_edges[1] + 1))
    if spec.distractor_heavy:
        n_dedges += 2
    n_distractor_edges = 0
    for etype, src, dst, label, cluster in (
        ("has_value", ch_b["claim"], v_other, "alt", C_VO),
        ("valid_during", ch_b["claim"], n_ic, "", C_TA),
        ("about_entity", ch_a["claim"], e_other, "alt", C_ANCHOR),
    ):
        if n_distractor_edges >= n_dedges:
            break
        if src not in b.node_atom or dst not in b.node_atom:
            continue
        admissible_edges.append(edge_atom(etype, src, dst, cluster, label))
        n_distractor_edges += 1

    # bindings.  All three claim the *same* slot, so any two of them together are
    # a sub-check 9 violation (planted mechanism 1, G4).  `a_bx` rests on the
    # temporally disjoint value, so it is the sub-check 3 violation on its own.
    a_ba = b.binding_atom((a_ca, a_va), cluster=C_BA)
    a_bb = b.binding_atom((a_cb, a_va), cluster=C_BB)
    a_bx = b.binding_atom((a_vo,), cluster=C_BX)

    # blocked atoms — permanently inadmissible, so they set the pool size without
    # touching the state count.  The retired edge and the quarantined-endpoint
    # edge come first: exit criterion 17 requires both to be present and
    # *unreachable*.
    blocked_atoms = [
        b.edge_atom_for(e_retired, ch_a["claim"], ch_a["value"], "has_value", C_BLOCK)
    ]
    for i, c_q in enumerate(blocked_claims):
        if len(b.atoms) >= pool_target and i >= 1:
            break
        blocked_atoms.append(b.node_atom_for(c_q, "Claim", C_BLOCK))
        if i == 0:
            blocked_atoms.append(
                b.edge_atom_for(
                    ids.edge_id("has_value", c_q, ch_a["value"]),
                    c_q,
                    ch_a["value"],
                    "has_value",
                    C_BLOCK,
                )
            )

    pool = AtomPool(b.atoms, cap=cfg.pool_cap)

    obligations = Obligations(
        entity_anchor=ANCHOR_NAME,
        value_type=VALUE_TYPE,
        time_constraint=Interval(start=CONSTRAINT_START, end=CONSTRAINT_END),
        needs_source=True,
        aggregate=False,
        scope=(),
    )

    # -- the two complete minimal proof templates --------------------------
    # Every atom in a template is load-bearing: drop one and either an obligation
    # slot goes unaddressed or the binding loses a referent it needs for closure.
    # Padding a template with atoms that contribute nothing to `U` is what makes
    # near-misses cheap and drives target mass into the `neither` bucket (G10).
    template_a = frozenset({a_anchor, a_ca, a_va, a_ta, a_ba})
    template_b = frozenset({a_anchor, a_cb, a_va, a_ta, a_bb})
    gold = ProofSet(atoms=template_a, bindings=pool.derive_bindings(template_a))

    meta = {
        "interval_gold": iv_a,
        "interval_complement": iv_c,
        "interval_partial": iv_p,
        "interval_disjoint": iv_x,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "pool_target": pool_target,
        "n_distractor_nodes": len(distractor_nodes),
        "n_distractor_edges": n_distractor_edges,
        "n_chain_edges": len(admissible_edges) - n_distractor_edges,
        "n_blocked_atoms": len(blocked_atoms),
        "binding_atoms": sorted((a_ba, a_bb, a_bx)),
        "disjoint_binding": a_bx,
        "duplicate_slot_pair": sorted((a_ba, a_bb)),
        "retired_edge_atom": b.edge_atom[e_retired],
        "quarantined_atoms": sorted(
            b.node_atom[c] for c in blocked_claims if c in b.node_atom
        ),
    }
    return LatticeInstance(
        pool=pool,
        obligations=obligations,
        graph=graph,
        gold=gold,
        template_a=template_a,
        template_b=template_b,
        spec=spec,
        meta=meta,
    )


def generate(rng: np.random.Generator, spec: LatticeSpec) -> LatticeInstance:
    """One instance inside every declared band, or a raise (G1).

    Rejection is not free.  Three things bound it, all from G1:

    * **early abort on the upper bounds** — enumeration stops the moment the
      state or edge count passes its cap, so an oversized instance is rejected
      in the time it takes to exceed the bound.  Undersized-terminal rejections
      still pay full enumeration and are *not* necessarily cheap;
    * **at most ``spec.max_attempts`` attempts**, after which this raises rather
      than silently widening the band;
    * the suite-level budget in :func:`_suite`.

    If acceptance needs many attempts the fix is to make the generator target the
    counts directly — distractor and blocked-atom counts are the levers — not to
    raise the attempt limit.
    """
    from graft.synth.audits import band_report

    rejections: list[dict[str, Any]] = []
    for attempt in range(spec.max_attempts):
        instance = _attempt(rng, spec)
        try:
            report = band_report(instance)
        except BandViolation as exc:
            rejections.append({"attempt": attempt, "band": exc.band, "measured": exc.measured})
            continue
        instance.meta["attempts"] = attempt + 1
        instance.meta["rejections"] = rejections
        instance.meta["bands"] = report
        return instance
    raise BandViolation(
        "attempts",
        rejections[-1] if rejections else None,
        f"an instance inside every band within {spec.max_attempts} attempts. "
        "The generator is not controlling its counts; fix the levers "
        "(distractor and blocked-atom counts), do not raise the limit.",
    )


# --------------------------------------------------------------------------
# suites (G9, decision 12)
# --------------------------------------------------------------------------


def _suite(spec: LatticeSpec, seed: int, count: int) -> tuple[LatticeInstance, ...]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    out: list[LatticeInstance] = []
    for _ in range(count):
        out.append(generate(rng, spec))
    elapsed = time.perf_counter() - started
    if elapsed > spec.suite_budget_seconds:
        raise BandViolation(
            "suite_budget_seconds",
            round(elapsed, 1),
            f"<= {spec.suite_budget_seconds} s of total suite generation (G1). "
            "Exceeding it is a generator that is not controlling its counts.",
        )
    for i, instance in enumerate(out):
        instance.meta["suite"] = spec.label
        instance.meta["suite_index"] = i
        instance.meta["suite_seed"] = seed
    return tuple(out)


def _spec_for(scope: str, spec: LatticeSpec | None) -> LatticeSpec:
    """The generator settings each suite is built with.

    **Main and tuning use identical settings and differ only in seed.**  β is
    swept on the tuning suite and applied to the main one, so if the two were
    built from different environment families that transfer would be an
    unstated assumption on top of an already-declared one.  What is main-only is
    the *β-dependent re-validation* — ``Target.validate_bands("main")``
    (decision 10) — not the generator.

    The probe suite is the deliberate exception: distractor-heavy and built
    **without** the G5 ``Δd`` band, because applying it there would require the
    probe to satisfy the condition it exists to violate (G9).  Its target-mass
    profile is reported, never gated, for the same reason.
    """
    base = spec if spec is not None else LatticeSpec()
    if scope in ("main", "tuning"):
        return base.replace(
            enforce_delta_d=True,
            enforce_neither_mass=True,
            distractor_heavy=False,
            label=scope,
        )
    if scope == "probe":
        return base.replace(
            enforce_delta_d=False,
            enforce_neither_mass=False,
            distractor_heavy=True,
            label="probe",
        )
    raise ValueError(f"scope must be one of main/probe/tuning, got {scope!r}")


def benchmark_suite(spec: LatticeSpec | None = None) -> tuple[LatticeInstance, ...]:
    """The 20-instance main suite at seed ``20260808`` — what Gate 2 is scored on."""
    return _suite(_spec_for("main", spec), MAIN_SEED, SUITE_SIZES["main"])


def probe_suite(spec: LatticeSpec | None = None) -> tuple[LatticeInstance, ...]:
    """5 distractor-heavy instances at seed ``20260809``, **without** the ``Δd`` band.

    Its purpose is to check whether a Gate-2 result survives where the ``Δd``
    signal is sparse.  Run once, at the end — a robustness check on the
    conclusion, not part of the primary comparison.
    """
    return _suite(_spec_for("probe", spec), PROBE_SEED, SUITE_SIZES["probe"])


def tuning_suite(spec: LatticeSpec | None = None) -> tuple[LatticeInstance, ...]:
    """5 instances at seed ``20260811`` — where Phase 3's β sweep runs.

    Separate from the main suite because β is a reward parameter shared by all
    seven learners: sweeping it on the instances the learners are scored on would
    violate v1.2 §4.1's requirement that β be chosen without touching test data.
    """
    return _suite(_spec_for("tuning", spec), TUNING_SEED, SUITE_SIZES["tuning"])


# --------------------------------------------------------------------------
# the hand-checkable instance
# --------------------------------------------------------------------------

TINY_CONSTRAINT = Interval(start=0.0, end=100.0)


def tiny_instance() -> LatticeInstance:
    """Six atoms, ``max_atoms = 3`` — small enough that ``p*`` is a written table.

    Seventeen closed states, fifteen valid terminals, **one reachable dead end**.
    The dead end is not incidental: without it the ``FAIL`` branch of the flow
    oracle is never executed by any test, and a construction that routes no mass
    to ``FAIL`` would still pass an aggregate ``TV < 1e-9`` check because
    ``p*(FAIL) ≈ 2.5e-12`` cannot move it (G6, exit criterion 3).

    ===========  ==========================================================
    atom          role
    ===========  ==========================================================
    ``nE``        Entity — satisfies the anchor slot
    ``nC``        Claim, valid during ``[0, 100)`` — overlaps the constraint
    ``nV``        Value of the requested type
    ``nX``        Claim, valid during ``[200, 300)`` — **disjoint**
    ``bOK``       binding "answer" over (``nC``, ``nV``)
    ``bBAD``      binding "answer" over (``nX``, ``nV``) — sub-check 3 fires
    ===========  ==========================================================

    ``{nX, nV, bBAD}`` is therefore reachable, formally invalid, and at
    ``|X| = max_atoms`` — no legal ``ADD`` and a masked ``STOP``.  That is the
    dead end.  ``{nC, nV, bOK}`` is its valid twin, which is what makes the pair
    a test of the checker rather than of the generator.

    Deliberately **not** produced by :func:`generate`: it is a fixture for the
    evaluator, not a Gate-2 environment, and it is exempt from the bands.
    """
    cfg = load_config(preset="synthetic", overrides={"pool_cap": 6, "max_atoms": 3})
    rng = np.random.default_rng(20260807)
    b = _Builder(rng, cfg)

    src = b.node("Source", "first_party", {PAYLOAD_TIER: "first_party"})
    entity = b.node("Entity", "tiny", {PAYLOAD_NAME: "Tiny Anchor", PAYLOAD_ALIASES: ["TA"]})
    iv_ok = _interval_node(b, "tiny-ok", 0.0, 100.0)
    iv_bad = _interval_node(b, "tiny-bad", 200.0, 300.0)

    span_c = b.span_for("tiny-c")
    aid_c = b.assertion_for("tiny-claim", span_c, eligible=True)
    claim = b.node("Claim", "tiny-claim", {PAYLOAD_ASSERTION_ID: aid_c})
    value = b.node(
        "Value", "tiny-value", {PAYLOAD_ASSERTION_ID: aid_c, PAYLOAD_VALUE_TYPE: VALUE_TYPE}
    )
    b.edge("valid_during", claim, iv_ok, span_c)
    b.edge("asserted_by", claim, src, span_c)
    b.edge("asserted_by", value, src, span_c)

    span_x = b.span_for("tiny-x")
    aid_x = b.assertion_for("tiny-stale", span_x, eligible=True)
    stale = b.node("Claim", "tiny-stale", {PAYLOAD_ASSERTION_ID: aid_x})
    b.edge("valid_during", stale, iv_bad, span_x)
    b.edge("asserted_by", stale, src, span_x)

    graph = b.snapshot()

    a_e = b.node_atom_for(entity, "Entity")
    a_c = b.node_atom_for(claim, "Claim")
    a_v = b.node_atom_for(value, "Value")
    a_x = b.node_atom_for(stale, "Claim")
    a_ok = b.binding_atom((a_c, a_v))
    a_bad = b.binding_atom((a_x, a_v))

    pool = AtomPool(b.atoms, cap=cfg.pool_cap)
    obligations = Obligations(
        entity_anchor="Tiny Anchor",
        value_type=VALUE_TYPE,
        time_constraint=TINY_CONSTRAINT,
        needs_source=True,
    )
    template_a = frozenset({a_e, a_c, a_v})
    template_b = frozenset({a_c, a_v, a_ok})
    gold = ProofSet(atoms=template_b, bindings=pool.derive_bindings(template_b))

    spec = LatticeSpec(
        cfg,
        min_terminals=1,
        max_terminals=100,
        max_states=1_000,
        max_edges=10_000,
        enforce_delta_d=False,
        label="tiny",
    )
    return LatticeInstance(
        pool=pool,
        obligations=obligations,
        graph=graph,
        gold=gold,
        template_a=template_a,
        template_b=template_b,
        spec=spec,
        meta={
            "atoms": {
                "nE": a_e,
                "nC": a_c,
                "nV": a_v,
                "nX": a_x,
                "bOK": a_ok,
                "bBAD": a_bad,
            },
            "dead_end": sorted((a_x, a_v, a_bad)),
        },
    )


