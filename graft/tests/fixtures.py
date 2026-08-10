"""Minimal coherent pools, with a backing snapshot.

Two things Phase 0's ``generators.py`` cannot do, and both matter:

* its ``candidate_atom`` gives edges and bindings ``refs`` of random tokens, so
  every pool built from it fails :meth:`AtomPool.validate`;
* nothing it produces has a ``target`` that resolves, so sub-checks 3, 4, 5 and 7
  and ``U``'s ``source_quality`` would be exercised only in-pool — which is
  precisely the Phase-9 surprise Phase-1 gap G2 exists to prevent.

Every instance here therefore emits a ``DictGraphSnapshot`` alongside its pool,
containing at least one invalidated edge and one quarantined assertion so that
checks 4 and 7 have negative cases available.

**This is not the ProofLattice.** No conflicting pairs, no dependency chains
beyond ``refs``, no enumerability, no exhaustive terminal set. It exists to
exercise ``H``, ``U`` and the masks. Phase 2 builds the thing Gate 2 runs on, and
keeping the two apart matters — a fixture generator that grows into an early
lattice is the failure mode Phase 2.5 was written to avoid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from graft import ids
from graft.graphstore import DictGraphSnapshot
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

TS = "2026-08-08T00:00:00+00:00"
FEAT_DIM = 6
SOURCE_TIERS = ("first_party", "corroborated", "reported", "unknown")

#: Validity windows assigned to successive claims, against a question constraint
#: of ``[0, 100)``.  Deliberately a mix of exact, **partial** and disjoint.
#:
#: The partial ones are the point.  With every interval either matching the
#: constraint exactly or missing it entirely, ``temporal_correctness`` collapses
#: to a presence flag taking only {0, 1} — which still passes a "at least two
#: distinct values" check while testing nothing about the graded behaviour that
#: Phase-1 gap G5 split it out to provide.  Partial overlaps are what make the
#: difference between ``H``'s contradiction test and ``U``'s precision grade
#: observable.
CLAIM_INTERVALS = (
    (0.0, 100.0),     # exact — the gold claim
    (50.0, 150.0),    # partial: covers half the constraint
    (200.0, 300.0),   # disjoint — the temporal-contradiction case
    (-40.0, 30.0),    # partial: covers 30%
    (25.0, 60.0),     # partial: covers 35%, wholly inside
    (300.0, 400.0),   # disjoint
)
CONSTRAINT = Interval(start=0.0, end=100.0)


@dataclass(frozen=True)
class Instance:
    """One coherent (pool, question, snapshot, gold) tuple."""

    pool: AtomPool
    obligations: Obligations
    graph: DictGraphSnapshot
    gold: ProofSet
    conv_id: str


def _feat(rng: random.Random, cluster: int) -> np.ndarray:
    """A feature vector near one of a few cluster centres.

    Clustered rather than uniform so that ``redundancy`` is non-degenerate: two
    atoms from the same cluster overlap, atoms from different clusters do not,
    and exit criterion 10 needs both to exist.
    """
    base = np.zeros(FEAT_DIM, dtype=np.float32)
    base[cluster % FEAT_DIM] = 1.0
    noise = np.asarray([rng.gauss(0.0, 0.15) for _ in range(FEAT_DIM)], dtype=np.float32)
    return (base + noise).astype(np.float32)


def build_instance(
    rng: random.Random,
    *,
    n_entities: int = 3,
    n_claims: int = 4,
    # The real-data cap, not the synthetic one.  These fixtures are not the
    # lattice and are not bound by its state-space budget; they need to be
    # comfortably larger than ``max_atoms`` so that the size limit in sub-check 6
    # actually bites during construction.
    pool_cap: int | None = 64,
    quarantine: bool = True,
    invalidate: bool = True,
    bounded_constraint: bool = True,
    scoped: bool = False,
) -> Instance:
    """Build one instance whose pool validates and whose targets resolve.

    Atoms are emitted nodes-first, then edges referencing those node atoms, then
    bindings — the same ordering that makes closure satisfiable, used here to
    guarantee the pool validates by construction rather than by luck.
    """
    conv_id = "conv-A"
    other_conv = "conv-B"

    turns: list[Turn] = []
    spans: list[SourceSpan] = []
    assertions: list[Assertion] = []
    nodes: list[Node] = []
    edges: list[Edge] = []

    # -- sources, one per tier so source_quality has a spread ---------------
    source_ids: dict[str, str] = {}
    for tier in SOURCE_TIERS:
        nid = ids.node_id("Source", tier)
        source_ids[tier] = nid
        nodes.append(Node(node_id=nid, ntype="Source", payload={PAYLOAD_TIER: tier}))

    # -- entities -----------------------------------------------------------
    entity_ids: list[str] = []
    for i in range(n_entities):
        nid = ids.node_id("Entity", f"entity-{i}")
        entity_ids.append(nid)
        nodes.append(
            Node(
                node_id=nid,
                ntype="Entity",
                payload={PAYLOAD_NAME: f"Entity {i}", PAYLOAD_ALIASES: [f"E{i}"]},
            )
        )
    anchor_name = "Entity 0"

    # -- claims, each with a turn, a span, an assertion and a value ---------
    claim_ids: list[str] = []
    value_ids: list[str] = []
    interval_ids: list[str] = []
    quarantined_claim: str | None = None

    for i in range(n_claims):
        # A minority of claims come from another conversation, so scope has a
        # negative case available.
        turn_conv = other_conv if (scoped and i == n_claims - 1) else conv_id
        turn_id = ids.node_id("Turn", f"turn-{i}")
        turns.append(
            Turn(
                turn_id=turn_id,
                conv_id=turn_conv,
                session_id="s0",
                speaker="user",
                ts=TS,
                text=f"utterance {i}",
            )
        )
        span = SourceSpan(span_id=ids.span_id(turn_id, 0, 10), turn_id=turn_id, start=0, end=10)
        spans.append(span)

        aid = ids.assertion_id("claim", f"claim-{i}", [span.span_id])
        eligible = "eligible"
        if quarantine and i == 1:
            eligible = "quarantined"
            quarantined_claim = aid
        assertions.append(
            Assertion(
                assertion_id=aid,
                kind="claim",
                text_norm=f"claim-{i}",
                spans=(span.span_id,),
                flags=AssertionFlags(
                    asserted_by="user", entailed_by_span=True, entailed_score=0.95
                ),
                t_created=TS,
                eligibility=eligible,
            )
        )

        cid = ids.node_id("Claim", f"claim-{i}")
        claim_ids.append(cid)
        nodes.append(Node(node_id=cid, ntype="Claim", payload={PAYLOAD_ASSERTION_ID: aid}))

        vid = ids.node_id("Value", f"value-{i}")
        value_ids.append(vid)
        nodes.append(
            Node(
                node_id=vid,
                ntype="Value",
                payload={PAYLOAD_ASSERTION_ID: aid, PAYLOAD_VALUE_TYPE: "occupation"},
            )
        )

        start, end = CLAIM_INTERVALS[i % len(CLAIM_INTERVALS)]
        tid = ids.node_id("TimeInterval", f"iv-{i}")
        interval_ids.append(tid)
        nodes.append(
            Node(node_id=tid, ntype="TimeInterval", payload={"start": start, "end": end})
        )

        anchor = entity_ids[i % n_entities]
        tier = SOURCE_TIERS[i % len(SOURCE_TIERS)]
        edges.extend(
            [
                Edge(
                    edge_id=ids.edge_id("about_entity", cid, anchor),
                    etype="about_entity",
                    src=cid,
                    dst=anchor,
                    t_created=TS,
                    provenance=(span.span_id,),
                ),
                Edge(
                    edge_id=ids.edge_id("has_value", cid, vid),
                    etype="has_value",
                    src=cid,
                    dst=vid,
                    t_created=TS,
                    provenance=(span.span_id,),
                ),
                Edge(
                    edge_id=ids.edge_id("valid_during", cid, tid),
                    etype="valid_during",
                    src=cid,
                    dst=tid,
                    t_created=TS,
                    provenance=(span.span_id,),
                ),
                Edge(
                    edge_id=ids.edge_id("asserted_by", cid, source_ids[tier]),
                    etype="asserted_by",
                    src=cid,
                    dst=source_ids[tier],
                    t_created=TS,
                    provenance=(span.span_id,),
                ),
            ]
        )

    # One retired edge, so check 4 has a negative case.
    if invalidate and n_claims >= 3:
        stale = ids.edge_id("about_entity", claim_ids[2], entity_ids[2 % n_entities])
        edges = [
            Edge(
                edge_id=e.edge_id,
                etype=e.etype,
                src=e.src,
                dst=e.dst,
                t_created=e.t_created,
                provenance=e.provenance,
                t_invalid=TS if e.edge_id == stale else e.t_invalid,
                superseded_by=None,
            )
            for e in edges
        ]

    graph = DictGraphSnapshot(
        snapshot_id=1, nodes=nodes, edges=edges, assertions=assertions, turns=turns, spans=spans
    )

    # -- atoms: nodes first, then edges, then bindings ----------------------
    atoms: list[CandidateAtom] = []
    atom_of_node: dict[str, str] = {}
    cluster = 0
    for node in nodes:
        aid_atom = ids.atom_id("node", (), node.node_id)
        atom_of_node[node.node_id] = aid_atom
        atoms.append(
            CandidateAtom(
                atom_id=aid_atom,
                kind="node",
                target=node.node_id,
                label=node.ntype,
                feat=_feat(rng, cluster),
            )
        )
        cluster += 1

    atom_of_edge: dict[str, str] = {}
    for edge in edges:
        refs = (atom_of_node[edge.src], atom_of_node[edge.dst])
        aid_atom = ids.atom_id("edge", refs, edge.edge_id, edge.etype)
        atom_of_edge[edge.edge_id] = aid_atom
        atoms.append(
            CandidateAtom(
                atom_id=aid_atom,
                kind="edge",
                refs=refs,
                target=edge.edge_id,
                label=edge.etype,
                feat=_feat(rng, cluster),
            )
        )
        cluster += 1

    # One binding per claim, referring to the claim node atom and its value.
    binding_of_claim: dict[str, str] = {}
    for i, cid in enumerate(claim_ids):
        refs = (atom_of_node[cid], atom_of_node[value_ids[i]])
        aid_atom = ids.atom_id("binding", refs, "", "answer")
        binding_of_claim[cid] = aid_atom
        atoms.append(
            CandidateAtom(
                atom_id=aid_atom,
                kind="binding",
                refs=refs,
                label="answer",
                feat=_feat(rng, i),
            )
        )

    pool = AtomPool(atoms, cap=pool_cap)

    # -- the question -------------------------------------------------------
    constraint = CONSTRAINT if bounded_constraint else Interval(start=0.0)
    obligations = Obligations(
        entity_anchor=anchor_name,
        value_type="occupation",
        time_constraint=constraint,
        needs_source=True,
        aggregate=False,
        scope=(conv_id,) if scoped else (),
    )

    # -- gold: claim 0's node, its entity, value, interval, the edges that
    # connect them, and the binding.  A genuine, minimal, valid proof.
    gold_claim = claim_ids[0]
    gold_atoms = {
        atom_of_node[gold_claim],
        atom_of_node[entity_ids[0]],
        atom_of_node[value_ids[0]],
        atom_of_node[interval_ids[0]],
        atom_of_edge[ids.edge_id("about_entity", gold_claim, entity_ids[0])],
        atom_of_edge[ids.edge_id("has_value", gold_claim, value_ids[0])],
        atom_of_edge[ids.edge_id("valid_during", gold_claim, interval_ids[0])],
        binding_of_claim[gold_claim],
    }
    gold = ProofSet(atoms=frozenset(gold_atoms), bindings=pool.derive_bindings(gold_atoms))

    assert quarantined_claim is None or not graph.is_eligible(quarantined_claim)
    return Instance(pool=pool, obligations=obligations, graph=graph, gold=gold, conv_id=conv_id)


def coherent_pool(rng: random.Random, **kwargs) -> AtomPool:
    """Just the pool, for tests that do not need the rest."""
    return build_instance(rng, **kwargs).pool


def instances(seed: int = 13, count: int = 12, **kwargs) -> list[Instance]:
    """A deterministic batch, for the property and non-degeneracy tests."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        out.append(
            build_instance(
                rng,
                n_entities=2 + (i % 3),
                n_claims=3 + (i % 3),
                **kwargs,
            )
        )
    return out


def random_subsets(
    rng: random.Random, pool: AtomPool, count: int, max_size: int
) -> Iterable[frozenset[str]]:
    """Arbitrary subsets — most invalid, which is what the checker tests need."""
    all_ids = pool.ids()
    for _ in range(count):
        size = rng.randint(0, min(max_size, len(all_ids)))
        yield frozenset(rng.sample(all_ids, size))


def closed_subsets(
    rng: random.Random, pool: AtomPool, count: int, max_size: int
) -> Iterable[frozenset[str]]:
    """Subsets built the way the ADD masks build them: refs before the atom."""
    all_ids = pool.ids()
    for _ in range(count):
        selected: set[str] = set()
        target = rng.randint(1, min(max_size, len(all_ids)))
        attempts = 0
        while len(selected) < target and attempts < 200:
            attempts += 1
            candidate = rng.choice(all_ids)
            if candidate in selected:
                continue
            if all(r in selected for r in pool[candidate].refs):
                selected.add(candidate)
        yield frozenset(selected)
