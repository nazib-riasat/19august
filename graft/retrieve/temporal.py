"""P7.4 — the temporal filter (G4, decision 5).  It removes; it never adds.

**Fail open in both directions, because the alternative empties the graph.**
The architecture says "drop atoms whose intervals contradict the obligation's
time constraint".  Read literally against the current corpus that is a pool
emptier: **no ``valid_during`` edges exist on real data yet** — D4's
conversational application is Gate-1-time work (`PHASE6_DECISIONS.md` §2.2) — so
a filter that dropped atoms *lacking* an interval would drop everything, and the
recall number would read 0.0 for a reason that has nothing to do with retrieval.

So the rule is **provable contradiction only**:

* the question carries no ``time_constraint`` → the filter never runs;
* the node carries no live ``valid_during`` interval → it passes;
* the node carries intervals and **none** overlaps the constraint → dropped.

``Interval.overlaps`` is Phase 0's and is frozen; the half-open convention is
its, not re-derived here.

**Why "none overlaps" rather than "any fails".**  A claim may carry several
validity intervals.  One of them overlapping the asked-about window is enough to
make the claim relevant, so the drop condition is the *conjunction* of failures.
Written the other way round — drop if any interval misses — a claim valid across
two separate periods would be discarded for the period the question did not ask
about, which is the opposite of what the constraint means.

**This is the same principle as the support gate, pointed the other way.**  The
gate fails *closed* because admitting an unsupported assertion is unrecoverable
downstream; this filter fails *open* because a too-large pool costs budget and a
wrongly emptied one costs the answer.  Each errs toward the side whose mistake is
recoverable.

**Applied before pool assembly, not to the assembled pool.**  P7.4's row says
"applied to the fused pool", and this module deliberately takes the fused *node
scores* instead — recorded as a departure in `PHASE7_DECISIONS.md`.  Filtering an
already-closed ``AtomPool`` would strand any edge atom whose endpoint had just
been removed, breaking the invariant ``AtomPool.validate()`` exists to hold and
turning a temporal drop into a malformed pool.  Filtering first and closing
afterwards produces a pool that is closed by construction, which is what G8
promises Stage D.
"""

from __future__ import annotations

from typing import Any, Mapping

from graft.schemas import Interval

__all__ = ["intervals_of", "temporal_filter"]


def intervals_of(snapshot: Any, node_id: str) -> tuple[Interval, ...]:
    """Validity intervals reachable from ``node_id`` by live ``valid_during`` edges.

    The payload keys are ``start``/``end`` in epoch seconds, written by
    ``graphbuild.commit.commit_temporal``.  They are read as literals here
    because Phase 7 may not change Stage A or Stage B (§2), and adding a
    ``PAYLOAD_*`` constant for them is a ``schemas.py`` edit — worth doing, and
    worth doing as its own deliberate change rather than as a side effect of
    building a filter.
    """
    out: list[Interval] = []
    for edge in snapshot.edges_of(node_id, "valid_during"):
        if edge.src != node_id or not snapshot.is_live(edge.edge_id):
            continue
        node = snapshot.node(edge.dst)
        if node is None or node.ntype != "TimeInterval":
            continue
        start, end = node.payload.get("start"), node.payload.get("end")
        start = float(start) if isinstance(start, (int, float)) else None
        end = float(end) if isinstance(end, (int, float)) else None
        if start is None and end is None:
            continue
        try:
            out.append(Interval(start=start, end=end))
        except ValueError:
            # A stored interval whose start is after its end is a Stage-B defect,
            # not something a retrieval filter should decide about.  Skipping it
            # means the node keeps its other intervals and, if it has none left,
            # passes -- fail-open, consistent with the rest of this module.
            continue
    return tuple(out)


def temporal_filter(
    snapshot: Any,
    scored: Mapping[str, float],
    constraint: Interval | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Drop only what provably contradicts ``constraint``.

    Returns ``(kept, report)``.  The report carries ``applied`` — whether the
    filter ran at all — beside the counts, because "0 dropped" and "never ran"
    are different facts about a question and the artefact has to be able to tell
    them apart.
    """
    if constraint is None:
        return dict(sorted(scored.items())), {
            "applied": False,
            "examined": 0,
            "with_interval": 0,
            "dropped": 0,
        }

    kept: dict[str, float] = {}
    with_interval = 0
    dropped = 0
    for node_id, score in sorted(scored.items()):
        intervals = intervals_of(snapshot, node_id)
        if not intervals:
            kept[node_id] = score
            continue
        with_interval += 1
        if any(iv.overlaps(constraint) for iv in intervals):
            kept[node_id] = score
        else:
            dropped += 1
    return kept, {
        "applied": True,
        "examined": len(scored),
        "with_interval": with_interval,
        "dropped": dropped,
    }
