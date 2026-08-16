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

import time
from typing import Any, Mapping

from graft.config.schema import Config
from graft.retrieve.pins import CHANNEL_WEIGHTS
from graft.retrieve.pool import build_pool
from graft.retrieve.temporal import temporal_filter
from graft.schemas import Interval

__all__ = ["minmax", "weighted_scores", "fuse", "assemble"]


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


def weighted_scores(
    channels: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Each channel's min–max-normalised, weighted node scores — the fusion input.

    Factored out (15 Aug 2026 audit) because two consumers need exactly this
    object and only one existed: ``fuse`` takes the ``max`` over it, and
    ``assemble`` must hand the *per-channel* values to Phases 8 and 9 — the
    architecture's Phase-8 features are "max/mean channel scores" and its Phase-9
    ``AtomFeaturizer`` row says "retrieval channel scores", plural, per atom
    (§9.1).  Before this helper the values were computed and immediately
    collapsed to one scalar, so the downstream contract was unimplementable from
    the handoff.
    """
    w = dict(CHANNEL_WEIGHTS if weights is None else weights)
    out: dict[str, dict[str, float]] = {}
    for name in sorted(channels):
        weight = float(w.get(name, 1.0))
        normalised = minmax(channels[name])
        out[name] = {node_id: value * weight for node_id, value in normalised.items()}
    return out


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
    weighted = weighted_scores(channels, weights)
    fused: dict[str, float] = {}
    winner: dict[str, str] = {}
    per_channel: dict[str, dict[str, Any]] = {}

    for name in sorted(channels):
        for node_id in sorted(weighted[name]):
            value = weighted[name][node_id]
            current = fused.get(node_id)
            # Strictly greater, so an earlier channel keeps a tie.  Channels are
            # walked in sorted-name order, making the tie-break total and stable
            # rather than dependent on dict insertion.
            if current is None or value > current:
                fused[node_id] = value
                winner[node_id] = name
        per_channel[name] = {
            "returned": len(channels[name]),
            "weight": float(w.get(name, 1.0)),
            "wins": 0,
        }

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
    conv_id: str | None = None,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """The whole read path: fuse → temporal filter → close → cap.

    Returns ``(pool, atom_scores, report)``.  ``report`` nests the fusion,
    temporal and pool reports under their own keys rather than flattening them —
    `PHASE5_DECISIONS.md` §1 records what flattening a report costs: the
    quarantine causes were nearly collapsed into one rate, which would have
    inflated the number the gate is judged on.  Three stages that can each drop
    an atom deserve three separately readable accounts of what they dropped.

    Two report keys added by the 15 Aug 2026 audit:

    * ``channel_scores`` — per admitted atom, the min–max-normalised weighted
      score **per channel** that returned it.  This is the Phase-8 feature input
      ("max/mean channel scores") and the Phase-9 ``AtomFeaturizer`` row
      ("retrieval channel scores", architecture §9.1); before it existed, those
      contracts were unimplementable from the handoff because fusion collapsed
      every channel to one scalar.  A channel absent from an atom's map scored
      it 0.0 — absence is the sparse encoding, not missing data.  Structural
      and edge atoms have empty maps: no channel can return them.
    * ``timings_ms`` — wall-clock per stage, because exit criterion 14 asks for
      per-channel latency and the temporal filter (one of the five training-free
      channels) runs only in here, so it had no timing row anywhere.
    """
    mark = time.perf_counter()
    fused, fusion_report = fuse(channels, weights)
    fuse_ms = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    kept, temporal_report = temporal_filter(snapshot, fused, constraint)
    temporal_ms = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    pool, atom_scores, pool_report = build_pool(
        snapshot, kept, cap=cap, config=config, conv_id=conv_id
    )
    pool_ms = (time.perf_counter() - mark) * 1000.0

    weighted = weighted_scores(channels, weights)
    channel_scores: dict[str, dict[str, float]] = {}
    raw_channel_scores: dict[str, dict[str, float]] = {}
    for atom in pool:
        if atom.kind != "node":
            channel_scores[atom.atom_id] = {}
            raw_channel_scores[atom.atom_id] = {}
            continue
        channel_scores[atom.atom_id] = dict(
            sorted(
                (name, weighted[name][atom.target])
                for name in weighted
                if atom.target in weighted[name]
            )
        )
        raw_channel_scores[atom.atom_id] = dict(
            sorted(
                (name, float(channels[name][atom.target]))
                for name in channels
                if atom.target in channels[name]
            )
        )

    return pool, atom_scores, {
        "fusion": fusion_report,
        "temporal": temporal_report,
        "pool": pool_report,
        "union_pre_cap": len(fused),
        "after_temporal": len(kept),
        "channel_scores": channel_scores,
        # **The un-normalised scores, added 15 Aug 2026 alongside — not instead
        # of — the normalised ones.**  Min–max is the *fusion* input and is
        # correct there (decision 3): taking a `max` across channels needs a
        # shared scale.  But it is scale-invariant *per question*, so the
        # normalised maximum is identically 1.0 and carries no information
        # across questions — measured on 200 MuSiQue pairs, where
        # `bm25_max`/`dense_max`/`fused_max` were constant at 1.0 for every row
        # while the same statistic on raw scores separated a contrast pair
        # 0.665/0.495 of the time and a raw top-3 sum 0.740/0.795.
        #
        # Phase 8's gate is the consumer that needs absolute evidence strength
        # ("is there a strongly-matching atom at all?"), which survives only
        # before normalisation.  `channel_scores` is left exactly as it was so
        # nothing downstream of the fusion contract moves.
        "raw_channel_scores": raw_channel_scores,
        "timings_ms": {
            "fuse": round(fuse_ms, 3),
            "temporal": round(temporal_ms, 3),
            "pool": round(pool_ms, 3),
        },
    }
