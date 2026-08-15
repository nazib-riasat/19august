"""P7.0 — the graph→pool mapping, with closure enforced at assembly (G8).

**This module is the untrusted site named in ``CandidateAtom``'s docstring.**
Recorded there on 13 Aug 2026: nothing in the project verifies that an edge
atom's ``refs`` denote *its target edge's endpoints*.  ``AtomPool.validate()``
checks that refs resolve in-pool and checker sub-check 8 checks that they are
selected, but neither compares them against ``G.edge(target).src/.dst`` — so
"closure" would stop meaning "endpoints present" with no check anywhere to
notice.  Project-built pools (Phase-2's lattice, the fixtures) wire them
correctly by construction; **this builder reads a real graph and is where the
correspondence has to be enforced**, which :func:`validate_edge_refs` does and
:func:`build_pool` calls before returning.

**The mapping, verbatim from G8:**

* **node atoms** for assertion-backed nodes (``Claim``/``Value``/``Event``)
  whose assertion is **eligible** — fix F9's boundary holds at retrieval too,
  so a quarantined assertion cannot become retrievable evidence — and for the
  ``Entity``/``TimeInterval`` nodes their edges reference;
* **edge atoms** for live edges among selected nodes, ``refs`` = their endpoint
  atoms;
* retrieval scores attach to assertion-backed node atoms; referenced structural
  atoms ride along at **score 0** — they are *support*, not hits;
* **closure is enforced at assembly**: selecting an atom pulls its refs in, and
  the cap applies to the **closed** set, so ``pool_cap`` counts what Stage D
  actually receives.

**Why the cap is applied to the closed set rather than to the hits.**  A cap
counted on hits alone would let a pool of 64 claims drag in 90 structural atoms
and hand Stage D a 154-atom environment — and the generalization bound the cap
exists to buy (**[EVIDENCE]** Generalization and Distributed Learning of
GFlowNets, ICLR 2025: data-dependent bounds that degrade with state-space size)
is stated over the state space, which is the closed set.  So admission is
greedy over the closed size, and a candidate that does not fit is **skipped
rather than terminating the loop**: a later, cheaper candidate may still fit,
and the count of skips is reported so a cap that binds hard is visible instead
of inferred.

``pool_cap`` is **read from the config tree, never redefined here**.  It is
frozen at 64 (`CLAUDE.md` §6) and giving a frozen value two homes is the
failure mode §5 of that file catalogues; ``graft.retrieve.pins`` carries only
what Phase 7 itself freezes.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from graft import ids
from graft.config.schema import Config
from graft.schemas import ASSERTION_BACKED_NTYPES, PAYLOAD_ASSERTION_ID, AtomPool, CandidateAtom

__all__ = [
    "STRUCTURAL_NTYPES",
    "COMPANION_ETYPES",
    "eligible_nodes",
    "node_atom_id",
    "node_text",
    "build_pool",
    "validate_edge_refs",
]

#: The two node types that ride along as support.  Deliberately **not** ``Source``
#: or ``SourceSpan``: provenance is reachable from the snapshot through the
#: assertion, and putting it in the pool would spend the cap on atoms no proof
#: ever needs to *select*.  G8 names exactly these two.
STRUCTURAL_NTYPES = ("Entity", "TimeInterval")

#: The edge types that pull a structural companion in.  ``about_entity`` gives
#: the entity channel and the expansion walk something to stand on;
#: ``valid_during`` gives the temporal filter the interval it tests.  Both are
#: assertion-backed → structural in :data:`graft.schemas.ENDPOINT_TABLE`, which is
#: why only the outgoing direction is followed.
COMPANION_ETYPES = ("about_entity", "valid_during")


def node_atom_id(node_id: str) -> str:
    """The atom id denoting ``node_id``.

    One function rather than four call sites, because the entity channel, the
    expansion walk and the recall instrument all need to turn a graph node into
    the atom that stands for it, and three copies of ``ids.atom_id("node", (),
    nid)`` is three chances to pass the arguments in a different order.
    """
    return ids.atom_id("node", (), node_id)


def node_text(snapshot: Any, node_id: str) -> str:
    """The searchable text of an assertion-backed node: its assertion's ``text_norm``.

    Both text channels come through here so that BM25 and the dense encoder index
    **the same string**.  If they diverged, a per-channel recall difference would
    be partly an artefact of what each one was shown, and the G7 table — the
    instrument that justifies every fusion decision — would be measuring two
    things at once.

    ``text_norm`` rather than the raw turn: it is what the extractor asserted and
    what the support gate checked, so it is the unit the pool selects.  An entity
    or interval node has no assertion and returns ``""``; they are support atoms
    and are never text-retrieved.
    """
    node = snapshot.node(node_id)
    if node is None or node.ntype not in ASSERTION_BACKED_NTYPES:
        return ""
    assertion_id = node.payload.get(PAYLOAD_ASSERTION_ID)
    if not assertion_id:
        return ""
    assertion = snapshot.assertion(str(assertion_id))
    return "" if assertion is None else str(assertion.text_norm)


def eligible_nodes(snapshot: Any, conv_id: str | None = None) -> tuple[str, ...]:
    """Assertion-backed node ids whose backing assertion passed the support gate.

    **Fix F9's boundary, applied at retrieval.**  Phase 5's support gate decides
    eligibility and Phase 6 refuses to commit anything else, but a retrieval
    channel that scanned nodes directly could still surface a quarantined claim —
    so the filter is re-applied here rather than assumed upstream.  Exit
    criterion 2 negative-tests it.

    ``GraphSnapshot`` is deliberately minimal (Phase-0 gap G2), so there is no
    ``nodes_of_type``; the concrete store's index is used when present and an
    empty tuple otherwise — the same accommodation ``graphbuild.candidates``
    makes, and for the same reason: forcing a protocol change would tax the
    Phase-2 lattice for a convenience Stage C alone wants.
    """
    nodes = getattr(snapshot, "_nodes", None)
    if not isinstance(nodes, dict):
        return ()
    out: list[str] = []
    for node in nodes.values():
        if node.ntype not in ASSERTION_BACKED_NTYPES:
            continue
        assertion_id = node.payload.get(PAYLOAD_ASSERTION_ID)
        if not assertion_id or not snapshot.is_eligible(str(assertion_id)):
            continue
        if conv_id is not None and _conv_of(snapshot, str(assertion_id)) != conv_id:
            continue
        out.append(node.node_id)
    return tuple(sorted(out))


def _conv_of(snapshot: Any, assertion_id: str) -> str | None:
    """The conversation an assertion belongs to, via its first span's turn.

    Provenance is the only route: an ``Assertion`` carries spans, a ``SourceSpan``
    carries a turn, and a ``Turn`` carries the ``conv_id``.  The same walk `H`'s
    scope sub-check makes, which is why it is not cached into a payload —
    duplicating it would give provenance two sources of truth.
    """
    assertion = snapshot.assertion(assertion_id)
    if assertion is None:
        return None
    for span_id in assertion.spans:
        span = snapshot.span(span_id)
        if span is None:
            continue
        turn = snapshot.turn(span.turn_id)
        if turn is not None:
            return turn.conv_id
    return None


def _companions(snapshot: Any, node_id: str) -> tuple[str, ...]:
    """The ``Entity``/``TimeInterval`` nodes this node's live edges reference."""
    out: set[str] = set()
    for edge in snapshot.edges_of(node_id):
        if edge.src != node_id or edge.etype not in COMPANION_ETYPES:
            continue
        if not snapshot.is_live(edge.edge_id):
            continue
        if snapshot.ntype(edge.dst) in STRUCTURAL_NTYPES:
            out.add(edge.dst)
    return tuple(sorted(out))


def _materialise(snapshot: Any, chosen: frozenset[str]) -> list[CandidateAtom]:
    """The closed atom set for a chosen node set — the whole mapping, in one place.

    Sorted at every step.  Set iteration order is randomised per process, so an
    unsorted walk here would produce a different *atom order* on every launch,
    and exit criterion 3 (two runs, identical pools and artefacts) would fail for
    a reason that has nothing to do with retrieval.
    """
    atoms: list[CandidateAtom] = []
    atom_of: dict[str, str] = {}
    for node_id in sorted(chosen):
        aid = node_atom_id(node_id)
        atom_of[node_id] = aid
        atoms.append(
            CandidateAtom(
                atom_id=aid,
                kind="node",
                target=node_id,
                label=snapshot.ntype(node_id) or "",
            )
        )

    seen: set[str] = set()
    for node_id in sorted(chosen):
        for edge in snapshot.edges_of(node_id):
            if edge.edge_id in seen:
                continue
            if edge.src not in chosen or edge.dst not in chosen:
                continue
            if not snapshot.is_live(edge.edge_id):
                continue
            seen.add(edge.edge_id)
            refs = (atom_of[edge.src], atom_of[edge.dst])
            atoms.append(
                CandidateAtom(
                    atom_id=ids.atom_id("edge", refs, edge.edge_id, edge.etype),
                    kind="edge",
                    refs=refs,
                    target=edge.edge_id,
                    label=edge.etype,
                )
            )
    return sorted(atoms, key=lambda a: a.atom_id)


def validate_edge_refs(pool: AtomPool, snapshot: Any) -> None:
    """Every edge atom's ``refs`` are exactly its target edge's endpoint atoms.

    The check ``CandidateAtom``'s docstring asks Phase 7 for by name.  Without
    it, an edge atom could satisfy closure — its refs resolve, they are selected —
    while denoting endpoints that are not the edge's own, and `H` would certify a
    proof whose structure does not match the graph it claims to read.

    Order matters and is checked: ``refs`` is ``(src, dst)``, and
    ``ids.atom_id`` hashes references in order, so a transposed pair is a
    different atom denoting a different claim about the graph.
    """
    for atom in pool:
        if atom.kind != "edge":
            continue
        edge = snapshot.edge(atom.target)
        if edge is None:
            raise ValueError(
                f"edge atom {atom.atom_id} targets {atom.target!r}, which the "
                "snapshot does not contain; the atom denotes nothing"
            )
        expected = (node_atom_id(edge.src), node_atom_id(edge.dst))
        if atom.refs != expected:
            raise ValueError(
                f"edge atom {atom.atom_id} (target {atom.target}) has refs "
                f"{atom.refs}, but its edge runs {edge.src} -> {edge.dst}, whose "
                f"endpoint atoms are {expected}; closure would hold while meaning "
                "something other than 'both endpoints present'"
            )


def build_pool(
    snapshot: Any,
    scored_nodes: Mapping[str, float] | Iterable[tuple[str, float]],
    *,
    cap: int | None = None,
    config: Config | None = None,
) -> tuple[AtomPool, dict[str, float], dict[str, Any]]:
    """Scored assertion-backed nodes → a closed, capped ``AtomPool``.

    Returns ``(pool, scores, report)`` as a plain tuple: ``graft.retrieve``
    defines no dataclass (criterion 12 — ``schemas.py`` is the single home of the
    data model), and a three-field result object here would be the first crack in
    that rule for no gain.

    ``scores`` maps **every** atom id in the pool to its fused score, structural
    companions included at 0.0, so a caller never has to ask which atoms were
    hits — Phase 9's featurizer reads this map directly (fix F6's interface).

    ``report`` carries what the artefact needs to be honest about the cap:
    ``cap``, ``hits_offered``, ``hits_admitted``, ``cap_skipped`` and the closed
    size.  A pool that admitted everything and one that was truncated look
    identical from the outside otherwise.  ``hits_refused_ineligible`` is
    reported separately from ``cap_skipped`` for the reason `PHASE5_DECISIONS.md`
    §1 gives about the quarantine causes: two different reasons an atom did not
    make it, flattened into one count, inflate whichever rate is being judged.
    """
    cfg = config or Config()
    limit = int(cfg.pool_cap if cap is None else cap)
    if limit <= 0:
        raise ValueError(f"pool cap must be positive, got {limit}")

    offered = (
        sorted(scored_nodes.items()) if isinstance(scored_nodes, Mapping) else sorted(scored_nodes)
    )
    # **Fix F9 is enforced here, not only in the channels.**  Every channel filters
    # through ``eligible_nodes`` already, so this is redundant on the built-in
    # paths -- and it is exactly the redundancy the project keeps needing: a guard
    # that lives only in the callers is one new caller away from being absent.
    # A negative test passes a quarantined node in directly; before this filter it
    # was admitted, because assembly trusted whoever scored it.
    eligible = set(eligible_nodes(snapshot))
    pairs = [(node_id, score) for node_id, score in offered if node_id in eligible]
    refused = len(offered) - len(pairs)
    # Highest score first, ties by node id.  Both keys matter: the score is the
    # ranking Stage C produces, and the id makes the order total, so two runs
    # admit the same atoms rather than whichever the dict happened to yield.
    ranked = sorted(pairs, key=lambda kv: (-float(kv[1]), kv[0]))

    chosen: frozenset[str] = frozenset()
    admitted: list[str] = []
    skipped = 0
    for node_id, _score in ranked:
        if node_id in chosen:
            continue
        trial = chosen | {node_id} | set(_companions(snapshot, node_id))
        if len(_materialise(snapshot, trial)) <= limit:
            chosen = trial
            admitted.append(node_id)
        else:
            skipped += 1

    atoms = _materialise(snapshot, chosen)
    pool = AtomPool(atoms, cap=limit)
    validate_edge_refs(pool, snapshot)

    hit_score = {str(k): float(v) for k, v in pairs}
    scores = {
        atom.atom_id: (
            hit_score.get(atom.target, 0.0)
            if atom.kind == "node" and atom.target in hit_score
            else 0.0
        )
        for atom in pool
    }
    # Membership, not ``score == 0.0``.  Min–max normalisation puts the
    # *lowest-scoring* retrieved atom at exactly 0.0, so counting zeros would
    # report a genuine hit as support and would do it most often on the pools
    # where one channel dominates — the case the G7 table is read to detect.
    hit_atoms = {a.atom_id for a in pool if a.kind == "node" and a.target in hit_score}
    report = {
        "cap": limit,
        "hits_offered": len(offered),
        "hits_eligible": len(ranked),
        "hits_refused_ineligible": refused,
        "hits_admitted": len(admitted),
        "cap_skipped": skipped,
        "pool_size": len(pool),
        "node_atoms": sum(1 for a in pool if a.kind == "node"),
        "edge_atoms": sum(1 for a in pool if a.kind == "edge"),
        "support_atoms": len(pool) - len(hit_atoms),
    }
    return pool, scores, report
