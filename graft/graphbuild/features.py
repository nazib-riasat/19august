"""P6.4's other half — graph state → ``GraphFeatures``, in two declared variants.

**This module is what makes "E3 beats E2" a statement about features.**  Until
it existed (built 14 Aug 2026, after the audit found ``GraftEncoder`` was an
empty subclass with byte-identical parameters to E2), the proposed encoder *was*
the baseline: nothing anywhere constructed the GRAFT feature set the
architecture names — "bge-small text embeddings, provenance flags, time deltas,
degree" — so the one-mechanism discipline E3 inherits from Phase 3's L7 had no
mechanism to discipline.

Two variants, declared here and frozen as §6 decision 15:

* ``base`` — what **E1 and E2** consume: a bias term, log-degree, and a
  **relative temporal encoding** (sinusoidal Δt to the graph's reference time at
  day/week/month/year periods).  Time is in the *base* set deliberately: HGT's
  architecture row names relative temporal encoding as load-bearing (WWW 2020),
  and GraphMixer's published strength is time-encoded link features (ICLR 2023)
  — a time-blind E1/E2 would be the weakened-control failure Phase 3's R13
  audit caught three times, in exactly this direction.
* ``graft`` — what **E3** consumes: ``base`` **plus** the four provenance flags
  (eligibility, entailed score, externally-verified,
  current-under-update-policy; zeros for nodes that carry no assertion) and the
  pinned text embedding (entity canonical name, or the backing assertion's
  ``text_norm``).  **[HYPOTHESIS]** — if this does not beat ``base`` under both
  backbones' better one, the encoder story is "HGT suffices" and the plan says
  to write that down.

**The RTE is a declared adaptation, not the paper's mechanism.**  Hu et al.
inject Δt *per edge* inside attention; PyG's ``HGTConv`` has no edge-time input,
so the encoding here is node-level Δt against a single reference time — the same
signal at coarser granularity, stated rather than passed off as the original.

**Reverse relations are an encoder convention, not schema edges.**
``encoder_metadata`` emits every endpoint-table relation *and* a ``rev_``
counterpart, because message passing over a directed heterogeneous graph
otherwise starves node types that only ever appear as sources (PyG drops
node types receiving no messages).  The graph itself stays directed; ``rev_*``
never appears in a log or a snapshot, and the convention lives here so tests and
trainers import it instead of inventing it.
"""

from __future__ import annotations

from datetime import datetime
from math import cos, log1p, sin
from typing import Any, Callable, Mapping, Sequence

import torch

from graft.schemas import (
    ASSERTION_BACKED_NTYPES,
    ENDPOINT_TABLE,
    NODE_TYPES,
    PAYLOAD_ASSERTION_ID,
    PAYLOAD_NAME,
)

__all__ = [
    "BASE_DIM",
    "GRAFT_DIM",
    "RTE_PERIODS_DAYS",
    "encoder_metadata",
    "build_features",
]

#: Sinusoidal periods for the relative temporal encoding, in days.
RTE_PERIODS_DAYS: tuple[float, ...] = (1.0, 7.0, 30.0, 365.0)

#: bias + log-degree + (sin, cos) per period.
BASE_DIM = 2 + 2 * len(RTE_PERIODS_DAYS)

#: base + four provenance flags + the pinned embedder's dimension.
GRAFT_DIM = BASE_DIM + 4 + 384


def encoder_metadata() -> tuple[list[str], list[tuple[str, str, str]]]:
    """The typed-graph metadata every HGT-family encoder is built against.

    Derived from the frozen endpoint table (G3) — the schema enumerates every
    relation that can ever appear, so the parameterisation is complete by
    construction — expanded to concrete ``(src, rel, dst)`` triples and doubled
    with ``rev_`` counterparts (see the module docstring).
    """
    relations: list[tuple[str, str, str]] = []
    for etype, (src_types, dst_types) in sorted(ENDPOINT_TABLE.items()):
        for src in src_types:
            for dst in dst_types:
                relations.append((src, etype, dst))
                relations.append((dst, f"rev_{etype}", src))
    # Deduplicate while keeping order (a symmetric relation like same_as yields
    # identical forward and reverse triples).
    seen: set[tuple[str, str, str]] = set()
    unique = [r for r in relations if not (r in seen or seen.add(r))]
    return list(NODE_TYPES), unique


def _epoch(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except ValueError:
        return None


def _rte(delta_days: float) -> list[float]:
    out: list[float] = []
    for period in RTE_PERIODS_DAYS:
        out.append(sin(delta_days / period))
        out.append(cos(delta_days / period))
    return out


def build_features(
    snapshot: Any,
    variant: str = "base",
    embed: Callable[[Sequence[str]], Any] | None = None,
    ref_ts: str | None = None,
) -> "GraphFeatures":
    """A committed graph → per-type feature tensors and live-edge indices.

    ``variant`` selects the declared feature set (module docstring); ``embed``
    is the pinned embedder's callable (or a stub), used only by ``graft``;
    ``ref_ts`` anchors the temporal encoding and defaults to the latest edge
    creation time — "now", as the graph knows it.

    Only **live** edges enter the message-passing structure: an invalidated
    edge is history, and an encoder aggregating over retired facts would
    reintroduce exactly what supersession removed.
    """
    from graft.graphbuild.encoders import GraphFeatures

    if variant not in ("base", "graft"):
        raise KeyError(f"unknown feature variant {variant!r}; declared: base, graft")

    nodes: dict[str, Any] = dict(getattr(snapshot, "_nodes", {}) or {})
    edges = {
        eid: e
        for eid, e in (getattr(snapshot, "_edges", {}) or {}).items()
        if e.t_invalid is None
    }

    # Reference time: the latest edge creation, else 0.
    ref = _epoch(ref_ts)
    if ref is None:
        stamps = [t for t in (_epoch(e.t_created) for e in edges.values()) if t is not None]
        ref = max(stamps) if stamps else 0.0

    # Node timestamps: an assertion-backed node inherits its assertion's
    # ``t_created``; an entity takes its earliest incident edge; a TimeInterval
    # takes its own start.  Anything else sits at the reference (Δt = 0).
    degree: dict[str, int] = {}
    earliest_edge: dict[str, float] = {}
    for e in edges.values():
        for endpoint in (e.src, e.dst):
            degree[endpoint] = degree.get(endpoint, 0) + 1
            t = _epoch(e.t_created)
            if t is not None:
                earliest_edge[endpoint] = min(earliest_edge.get(endpoint, t), t)

    def node_time(node: Any) -> float:
        if node.ntype in ASSERTION_BACKED_NTYPES:
            aid = node.payload.get(PAYLOAD_ASSERTION_ID)
            assertion = snapshot.assertion(str(aid)) if aid else None
            t = _epoch(getattr(assertion, "t_created", None))
            if t is not None:
                return t
        if node.ntype == "TimeInterval":
            start = node.payload.get("start")
            if isinstance(start, (int, float)):
                return float(start)
        return earliest_edge.get(node.node_id, ref)

    def node_text(node: Any) -> str:
        if node.ntype == "Entity":
            return str(node.payload.get(PAYLOAD_NAME, ""))
        if node.ntype in ASSERTION_BACKED_NTYPES:
            aid = node.payload.get(PAYLOAD_ASSERTION_ID)
            assertion = snapshot.assertion(str(aid)) if aid else None
            return str(getattr(assertion, "text_norm", ""))
        return ""

    def provenance_flags(node: Any) -> list[float]:
        if node.ntype not in ASSERTION_BACKED_NTYPES:
            return [0.0, 0.0, 0.0, 0.0]
        aid = node.payload.get(PAYLOAD_ASSERTION_ID)
        assertion = snapshot.assertion(str(aid)) if aid else None
        if assertion is None:
            return [0.0, 0.0, 0.0, 0.0]
        flags = assertion.flags
        return [
            1.0 if assertion.eligibility == "eligible" else 0.0,
            float(flags.entailed_score),
            1.0 if flags.externally_verified else 0.0,
            1.0 if flags.current_under_update_policy else 0.0,
        ]

    by_type: dict[str, list[Any]] = {}
    for node in nodes.values():
        by_type.setdefault(node.ntype, []).append(node)
    for ntype in by_type:
        by_type[ntype].sort(key=lambda n: n.node_id)  # deterministic rows

    # Text embeddings in one batch (the F7 discipline: one pass, then free).
    texts: list[str] = []
    slots: list[tuple[str, int]] = []
    if variant == "graft":
        for ntype, members in sorted(by_type.items()):
            for ix, node in enumerate(members):
                text = node_text(node)
                if text:
                    texts.append(text)
                    slots.append((ntype, ix))
    vectors = None
    if texts and embed is not None:
        import numpy as np

        vectors = np.asarray(embed(texts), dtype="float32")

    x: dict[str, torch.Tensor] = {}
    node_ids: dict[str, list[str]] = {}
    dim = BASE_DIM if variant == "base" else GRAFT_DIM
    for ntype, members in sorted(by_type.items()):
        rows = torch.zeros((len(members), dim))
        for ix, node in enumerate(members):
            delta_days = (ref - node_time(node)) / 86400.0
            base = [1.0, log1p(float(degree.get(node.node_id, 0)))] + _rte(delta_days)
            rows[ix, :BASE_DIM] = torch.tensor(base)
            if variant == "graft":
                rows[ix, BASE_DIM : BASE_DIM + 4] = torch.tensor(provenance_flags(node))
        x[ntype] = rows
        node_ids[ntype] = [n.node_id for n in members]

    if variant == "graft" and vectors is not None:
        for (ntype, ix), vec in zip(slots, vectors):
            x[ntype][ix, BASE_DIM + 4 :] = torch.from_numpy(vec)

    # Live edges → per-relation index tensors, forward and reverse.
    positions = {
        ntype: {nid: ix for ix, nid in enumerate(ids)} for ntype, ids in node_ids.items()
    }
    edge_index: dict[tuple[str, str, str], list[list[int]]] = {}
    for e in edges.values():
        src_node, dst_node = nodes.get(e.src), nodes.get(e.dst)
        if src_node is None or dst_node is None:
            continue
        fwd = (src_node.ntype, e.etype, dst_node.ntype)
        rev = (dst_node.ntype, f"rev_{e.etype}", src_node.ntype)
        s_ix = positions[src_node.ntype][e.src]
        d_ix = positions[dst_node.ntype][e.dst]
        edge_index.setdefault(fwd, [[], []])
        edge_index[fwd][0].append(s_ix)
        edge_index[fwd][1].append(d_ix)
        edge_index.setdefault(rev, [[], []])
        edge_index[rev][0].append(d_ix)
        edge_index[rev][1].append(s_ix)

    tensors = {
        key: torch.tensor(pair, dtype=torch.long) for key, pair in edge_index.items()
    }
    return GraphFeatures(x=x, edge_index=tensors, node_ids=node_ids)
