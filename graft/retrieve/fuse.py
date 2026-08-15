"""P7.6 — the declared fusion arithmetic, and the assembly it feeds (G5, decision 3).

"Score-normalized weighted union → dedup → cap at 64" (architecture) hid three
knobs nobody had declared: the normalisation, the weights and the tie-break.  All
three are now constants in ``graft.retrieve.pins``, and this module is the only
place they are applied.

**min–max per channel, per question.**  BM25 scores and cosines share no scale,
so a union over raw values would be a comparison between a term-frequency
statistic and an inner product.  min–max is a rank-shape transform — it preserves
each channel's ordering and puts every channel on [0, 1] so ``max`` across them
means something.

*The degenerate case is declared, not discovered.*  A channel returning one
result, or several with identical scores, has ``max == min`` and would divide by
zero.  Those score **1.0**: the channel retrieved them and has no basis to
separate them, and 1.0 is the reading that preserves recall.  It is also what
makes the entity channel work at all — it scores every hit 1.0 by design, so it
is *always* in the degenerate case, and mapping that to 0.0 would delete the one
channel that knows about atoms no text query could reach.

**max, not sum or mean.**  Union semantics: one strong channel suffices.
Averaging would punish an atom only one channel could possibly know about, which
for a *recall* stage is exactly backwards — and the plan says twice (§3.3) that
recall, not final selection, is this stage's purpose.

**Weights are applied even though all five are 1.0.**  A weight that exists only
in a docstring is not a knob anyone can move without editing arithmetic; applying
it makes the declared constant real, and the G7 ablation table is what would ever
justify changing one.

**Order of operations: fuse → filter → close → cap.**  The temporal filter runs
on fused *node* scores rather than on the assembled pool, for the reason its own
module records — removing a node atom from a closed pool would strand the edge
atoms that reference it.  Closing after filtering yields a pool that is closed by
construction, which is what G8 promises Stage D.
"""

from __future__ import annotations

from typing import Any, Mapping

from graft.config.schema import Config
from graft.retrieve.pins import CHANNEL_WEIGHTS
from graft.retrieve.pool import build_pool
from graft.retrieve.temporal import temporal_filter
from graft.schemas import Interval

__all__ = ["minmax", "fuse", "assemble"]


def minmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalise one channel's results to [0, 1]; a flat channel maps to all-1.0."""
    if not scores:
        return {}
    values = [float(v) for v in scores.values()]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in sorted(scores)}
    span = hi - lo
    return {k: (float(scores[k]) - lo) / span for k in sorted(scores)}


def fuse(
    channels: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-channel min–max, weighted, combined by ``max``.

    Returns ``(fused, report)`` where ``fused`` maps ``node_id`` to its fused
    score and ``report`` records each channel's contribution.

    **Three counts per channel, because one of them lies on its own.**
    ``returned`` is how many results the channel offered.  ``wins`` is how many
    atoms it was the strict argmax for — and *ties go to the alphabetically first
    channel*, which makes ``wins`` alone actively misleading: a channel that
    reached exactly the same atoms at exactly the same normalised score as an
    earlier one reads as ``wins: 0``, and a G7 table showing that would invite
    deleting a channel that was pulling its weight.  The entity channel is
    permanently in this position, since it scores every hit 1.0 by design.
    ``unique`` is therefore reported beside it: atoms **no other channel
    returned at all**, which is the count that actually answers "what would we
    lose by dropping this channel".
    """
    w = dict(CHANNEL_WEIGHTS if weights is None else weights)
    fused: dict[str, float] = {}
    winner: dict[str, str] = {}
    per_channel: dict[str, dict[str, Any]] = {}

    for name in sorted(channels):
        normalised = minmax(channels[name])
        weight = float(w.get(name, 1.0))
        for node_id in sorted(normalised):
            value = normalised[node_id] * weight
            current = fused.get(node_id)
            # Strictly greater, so an earlier channel keeps a tie.  Channels are
            # walked in sorted-name order, making the tie-break total and stable
            # rather than dependent on dict insertion.
            if current is None or value > current:
                fused[node_id] = value
                winner[node_id] = name
        per_channel[name] = {"returned": len(channels[name]), "weight": weight, "wins": 0}

    for name in winner.values():
        per_channel[name]["wins"] += 1

    for name in sorted(channels):
        others = set()
        for other in sorted(channels):
            if other != name:
                others |= set(channels[other])
        per_channel[name]["unique"] = len(set(channels[name]) - others)

    return dict(sorted(fused.items())), {
        "channels": per_channel,
        "union_size": len(fused),
    }


def assemble(
    snapshot: Any,
    channels: Mapping[str, Mapping[str, float]],
    *,
    constraint: Interval | None = None,
    weights: Mapping[str, float] | None = None,
    cap: int | None = None,
    config: Config | None = None,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """The whole read path: fuse → temporal filter → close → cap.

    Returns ``(pool, atom_scores, report)``.  ``report`` nests the fusion,
    temporal and pool reports under their own keys rather than flattening them —
    `PHASE5_DECISIONS.md` §1 records what flattening a report costs: the
    quarantine causes were nearly collapsed into one rate, which would have
    inflated the number the gate is judged on.  Three stages that can each drop
    an atom deserve three separately readable accounts of what they dropped.
    """
    fused, fusion_report = fuse(channels, weights)
    kept, temporal_report = temporal_filter(snapshot, fused, constraint)
    pool, atom_scores, pool_report = build_pool(snapshot, kept, cap=cap, config=config)
    return pool, atom_scores, {
        "fusion": fusion_report,
        "temporal": temporal_report,
        "pool": pool_report,
        "union_pre_cap": len(fused),
        "after_temporal": len(kept),
    }
