"""P7.5 — the bounded graph walk (G10, decision 6).

**A hop bound is not a bound.**  "≤ 2-hop from matched entities" bounds depth and
says nothing about width, and in a memory graph the user is a hub that links to
everything — so 2-hop expansion reaches the whole graph and ``pool_cap`` then
decides recall by *truncation order*, an invisible determinant of the headline
number.  Hence a per-node, per-hop **fan-out cap** (default 32,
``pins.EXPANSION``) that keeps the **most recent** edges by ``t_created`` when it
binds.

Recency as the tie-break is **[ANALYSIS]**, declared rather than inherited: a
memory graph's hubs skew old, so keeping the oldest edges of a hub would return
the least current view of an entity — the opposite of what a memory system is
for.  **The number of times the cap binds is reported**, so a cap doing real work
to the recall number is visible rather than inferred.  A cap that binds often is
the signal to revisit it, and revisiting it is a declared amendment.

**Live edges only.**  An invalidated edge is a fact the graph has retired;
walking it would let a superseded claim re-enter the pool through the back door,
which is precisely the property Contribution 1 claims to have.

**Hop scores are ``1 / hop``** — 1.0 at one hop, 0.5 at two.  **[ANALYSIS]**, and
deliberately crude: this channel's job is *reach*, and a graph-distance decay is
the least assuming shape that still prefers the near.  It is stated here because
under G5's ``max`` fusion a channel's internal scale is only visible in the G7
ablation table, and an undocumented curve would be untraceable there.

**[EVIDENCE, provisional]** the ≤ 2 bound follows Beyond Static Retrieval (arXiv
2509.25530): two iterations was the cost–benefit optimum across four graph-RAG
systems, and iteration *hurt* simple questions through over-retrieval.  Flagged
provisional per `CLAUDE.md` §3 and carrying no claim on its own.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from graft.retrieve.pins import EXPANSION
from graft.schemas import ASSERTION_BACKED_NTYPES

__all__ = ["expand_channel"]


def _live_edges(snapshot: Any, node_id: str) -> list[Any]:
    return [e for e in snapshot.edges_of(node_id) if snapshot.is_live(e.edge_id)]


def _keep_recent(edges: Sequence[Any], fan_out: int) -> tuple[list[Any], bool]:
    """The ``fan_out`` most recent edges, and whether the cap actually bound.

    ``t_created`` is ISO-8601, which sorts chronologically as a string *given a
    consistent format* — which Stage A guarantees, since it writes every
    timestamp through one converter.  The secondary key is ``edge_id``: without
    it, two edges sharing a timestamp would be ordered by whatever
    ``edges_of`` yielded, and the walk would stop being reproducible on exactly
    the graphs where the cap matters most.
    """
    if len(edges) <= fan_out:
        return list(edges), False
    ordered = sorted(edges, key=lambda e: (e.t_created, e.edge_id), reverse=True)
    return ordered[:fan_out], True


def expand_channel(
    snapshot: Any,
    seeds: Iterable[str],
    *,
    max_hops: int | None = None,
    fan_out: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Breadth-first walk from ``seeds`` over live edges, bounded in depth and width.

    Returns ``(hits, report)`` where ``hits`` maps assertion-backed ``node_id`` to
    its ``1 / hop`` score — the seeds themselves are entities and are never hits,
    since a pool selects evidence rather than the thing the evidence is about.

    ``report`` carries ``fan_out_binds``, which exit criterion 10 requires in the
    artefact.
    """
    hops = int(EXPANSION["max_hops"] if max_hops is None else max_hops)
    width = int(EXPANSION["fan_out"] if fan_out is None else fan_out)
    if width <= 0:
        raise ValueError(f"fan_out must be positive, got {width}")

    frontier = sorted(set(seeds))
    visited = set(frontier)
    hits: dict[str, float] = {}
    binds = 0
    per_hop: list[int] = []

    for hop in range(1, hops + 1):
        nxt: set[str] = set()
        for node_id in frontier:
            edges, bound = _keep_recent(_live_edges(snapshot, node_id), width)
            binds += int(bound)
            for edge in edges:
                other = edge.dst if edge.src == node_id else edge.src
                if other == node_id or other in visited:
                    continue
                visited.add(other)
                nxt.add(other)
                if snapshot.ntype(other) in ASSERTION_BACKED_NTYPES:
                    # ``max`` rather than assignment: a node first reached at hop
                    # 2 down one path and hop 1 down another keeps the nearer
                    # score.  Frontier order would otherwise decide it.
                    hits[other] = max(hits.get(other, 0.0), 1.0 / hop)
        per_hop.append(len(nxt))
        frontier = sorted(nxt)
        if not frontier:
            break

    return dict(sorted(hits.items())), {
        "seeds": len(set(seeds)),
        "max_hops": hops,
        "fan_out": width,
        "fan_out_binds": binds,
        "visited": len(visited),
        "frontier_per_hop": per_hop,
        "hits": len(hits),
    }
