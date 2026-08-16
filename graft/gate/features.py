"""P8.1 — the gate's feature vector (G2, decision 3).

**The features are the claim.** Contribution 2's gate is "a small trained
classifier that says whether a sufficient proof exists in the current snapshot",
and what makes that a *decoupled* design rather than a re-implementation of the
reader is precisely *which* inputs it may see. So G2's table is implemented here
verbatim, the block order is frozen in ``pins.FEATURE_BLOCKS``, and a silent
change to either is a different gate wearing the same name.

**Every block ships a presence flag, and that is not bookkeeping** (G8). The
MuSiQue adapter cannot supply slot coverage — Wikipedia paragraph sets have no
obligation parse — and a conversational question whose parser returned nothing is
in the same position. Encoding "absent" as a bare 0.0 would let the gate learn
*"slot features are zero ⇒ this is MuSiQue ⇒ 50% unanswerable"*, a **dataset
classifier** wearing an answerability classifier's name. The flag makes absence
an explicit input instead of a confound, and it is the same "degrade visibly,
never silently" rule Phase 7's runner applies to a question with no slots.

**Names ship with the vector.** An ablation is a *column mask* over
``names``, never a re-featurisation: recomputing features per arm invites the two
arms to differ by more than the arm.

**No gold field reaches this module** (G3). ``has_answer`` and
``answer_session_ids`` live in ``gate/labels.py`` alone, mirroring
``retrieve/recall.py``'s boundary, and a structural test asserts it over the
source rather than trusting this paragraph.

**Nothing here imports torch.** Featurisation is pure numpy over Stage-C's
reports, which keeps the feature contract testable on a bare interpreter — and
the contract is the part the write-up depends on.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from graft.gate.pins import FEATURE_BLOCKS
from graft.schemas import Obligations

__all__ = [
    "BLOCK_FEATURES",
    "feature_names",
    "build_features",
    "block_mask",
]

#: The scalar features each block contributes, in order. Written out rather than
#: generated so that the vector's shape is readable in one place and a rename is
#: a visible diff rather than a silent reindex.
#:
#: ``question_embedding`` is the shared pinned bge-small's dimension and is
#: expanded lazily in :func:`feature_names`, since listing 384 names here would
#: bury the rest.
BLOCK_FEATURES: dict[str, tuple[str, ...]] = {
    "slot_coverage": (
        "anchor_present",
        "anchor_matched_entity",
        "value_type_present",
        "value_type_satisfiable",
        "time_constraint_present",
        "time_constraint_satisfiable",
        "needs_source",
        "active_slot_count",
    ),
    # **Amended 15 Aug 2026 — a recorded decision-3 amendment, by measurement.**
    #
    # As first built this block was ``{channel}_max`` and ``{channel}_mean`` over
    # the *normalised* scores, following the architecture's "max/mean channel
    # scores" wording.  Measured on 200 real MuSiQue dev pairs, **10 of the 13
    # features were constant**: min–max normalisation puts the top-scoring item
    # at exactly 1.0 for every question, so ``bm25_max``, ``dense_max`` and
    # ``fused_max`` carried no information at all, and the three absent channels
    # contributed six more zeros.  The gate was left with three varying numbers
    # and scored 0.52 within-pair accuracy against a 0.50 chance line.
    #
    # The same statistics on **raw** scores separate a contrast pair far better —
    # measured over the same 200 pairs, win-rate against 0.5 = no signal:
    # ``bm25_max`` 0.665, ``bm25_top3`` 0.740, ``dense_top3`` **0.795**.
    # ``dense_max`` alone is 0.495, which is why the top-*k* sum is carried and
    # not just the maximum: one strongly-matching paragraph is weak evidence, and
    # *several* is what distinguishes an answerable multi-hop question.
    #
    # **[ANALYSIS]**, and deliberately labelled so.  The architecture's feature
    # row says "max/mean channel scores"; departing from it is this build's
    # judgement, justified by the measurement above and by nothing in the
    # literature.  The normalised mean is *kept* — it is the one normalised
    # statistic that is not constant, and it carries the score *distribution's
    # shape*, which the raw statistics do not.
    "channel_scores": (
        "bm25_raw_max", "bm25_raw_top3", "bm25_raw_mean", "bm25_norm_mean",
        "dense_raw_max", "dense_raw_top3", "dense_raw_mean", "dense_norm_mean",
        "entity_raw_max", "entity_raw_top3", "entity_raw_mean", "entity_norm_mean",
        "expand_raw_max", "expand_raw_top3", "expand_raw_mean", "expand_norm_mean",
        "scorer_raw_max", "scorer_raw_top3", "scorer_raw_mean", "scorer_norm_mean",
        # The **fused** score carries no raw/normalised distinction: it *is* the
        # max over the normalised channels (decision 3's arithmetic), so a "raw
        # fused" score does not exist.  Its maximum is consequently 1.0 for every
        # question with any channel hit — measured constant over 400 dev pairs —
        # and a "raw mean" would duplicate the normalised one exactly (both 0.453
        # on the same measurement).  Two features rather than four, with the two
        # provably-redundant ones dropped rather than shipped as padding.
        "fused_top3", "fused_mean",
        "channels_present",
    ),
    "pool_shape": (
        "pool_size",
        "node_atoms",
        "edge_atoms",
        "support_atoms",
        "hits_offered",
        "hits_admitted",
        "cap_skipped",
        "hits_refused_ineligible",
        "union_pre_cap",
        "after_temporal",
        "temporal_dropped",
    ),
    "saturation": (
        "candidates_in_scope",
        "closed_atoms_in_scope",
        "cap_headroom",
        "exercised",
    ),
}

#: The channels the ``channel_scores`` block reads, in order. Four training-free
#: channels plus the optional learned scorer plus the fused score. **A missing
#: channel key is 0.0 with the block's presence flag still set** — G2's "absence
#: of a channel key = 0.0" — because a channel that returned nothing is a
#: different fact from a channel that was never run, and only the second should
#: look like absence.
_CHANNELS: tuple[str, ...] = ("bm25", "dense", "entity", "expand", "scorer")


def feature_names(embed_dim: int) -> tuple[str, ...]:
    """Every feature name, in vector order, including the presence flags.

    The presence flags are appended **after** all value blocks rather than
    interleaved, so a value-block mask and a flag mask are contiguous slices —
    which is what makes ``block_mask`` a slice computation instead of a lookup.
    """
    names: list[str] = []
    for block in FEATURE_BLOCKS:
        if block == "question_embedding":
            names.extend(f"q_emb_{i}" for i in range(embed_dim))
        else:
            names.extend(f"{block}.{f}" for f in BLOCK_FEATURES[block])
    names.extend(f"present.{block}" for block in FEATURE_BLOCKS)
    return tuple(names)


def _slot_coverage(
    obligations: Obligations | None,
    pool: Any,
    snapshot: Any,
) -> tuple[list[float], bool]:
    """G2's first row: what the question asks for, and whether the pool has it.

    Present only when an obligation parse exists. Fix F2's rule travels with this
    block wherever it is reported: the parser's measured slot quality
    (`PHASE5_DECISIONS.md` §5a — ``entity_anchor`` exact-match **0.167**) bounds
    what any coverage number here can mean, and the runner quotes it beside them.
    """
    if obligations is None:
        return [0.0] * len(BLOCK_FEATURES["slot_coverage"]), False

    from graft.retrieve.entity import match_entities
    from graft.retrieve.temporal import intervals_of

    anchor = obligations.entity_anchor
    conv_id = obligations.scope[0] if obligations.scope else None
    # **Matched *in the pool*, not merely in the snapshot** (corrected 15 Aug
    # 2026).  The signed feature contract is "obligation-slot coverage **by the
    # pool**"; as first written this asked only whether the graph contains a
    # matching entity, so an entity that retrieval had *excluded* still scored
    # ``anchor_matched_entity = 1``.  That inverts the feature's meaning — the
    # gate is being asked what the pool can support, and "the graph has it but the
    # pool does not" is precisely the unanswerable case.
    matched = False
    if anchor:
        candidates = set(match_entities(snapshot, anchor, conv_id))
        if candidates:
            pooled = {atom.target for atom in pool if atom.kind == "node"}
            matched = bool(candidates & pooled)

    # A `value_type`-compatible atom: a Value node whose payload declares the
    # asked-for type. Absent a parsed type there is nothing to satisfy, which is
    # recorded as "not present" rather than as a failure to satisfy.
    wanted = (obligations.value_type or "").strip().lower()
    satisfiable_value = False
    if wanted:
        for atom in pool:
            node = snapshot.node(atom.target) if atom.kind == "node" else None
            if node is not None and str(node.payload.get("value_type", "")).lower() == wanted:
                satisfiable_value = True
                break

    # A time constraint is satisfiable when some pooled atom carries an interval
    # that overlaps it. Fail-open's mirror image: the temporal *filter* keeps
    # atoms with no interval, so "no atom has any interval" reads here as
    # unsatisfiable rather than as satisfied-by-default.
    constraint = obligations.time_constraint
    satisfiable_time = False
    if constraint is not None:
        for atom in pool:
            if atom.kind != "node":
                continue
            if any(iv.overlaps(constraint) for iv in intervals_of(snapshot, atom.target)):
                satisfiable_time = True
                break

    return (
        [
            float(bool(anchor)),
            float(matched),
            float(bool(obligations.value_type)),
            float(satisfiable_value),
            float(constraint is not None),
            float(satisfiable_time),
            float(bool(obligations.needs_source)),
            float(len(obligations.active_slots())),
        ],
        True,
    )


#: How many of the top scores the ``raw_top3`` statistic sums.
#:
#: **[ANALYSIS]**, and 3 rather than 1 for a measured reason: MuSiQue's
#: answerable questions carry **2** supporting paragraphs and its usual negative
#: removes only one of them, so "is there *a* strong match" (the maximum) is far
#: weaker evidence than "are there *several*" — measured, ``dense_max`` separates
#: a pair 0.495 of the time against ``dense_top3``'s 0.795.  A multi-hop question
#: needs multiple pieces of evidence by definition, which is the same reason the
#: proof sets this project selects are *sets*.
TOP_K = 3


def _stats(values: list[float]) -> list[float]:
    """``[max, top-k sum, mean]`` — the three raw statistics, in block order."""
    if not values:
        return [0.0, 0.0, 0.0]
    ordered = sorted(values, reverse=True)
    return [ordered[0], float(sum(ordered[:TOP_K])), float(sum(values) / len(values))]


def _channel_scores(
    atom_scores: Mapping[str, float] | None,
    channel_scores: Mapping[str, Mapping[str, float]] | None,
    raw_channel_scores: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[list[float], bool]:
    """G2's second row, over Phase 7 fix B1's restored per-channel contract.

    Reads **both** score views from ``assemble()``'s report: ``raw_channel_scores``
    for absolute evidence strength and ``channel_scores`` (min–max normalised)
    for the distribution's shape.  The block's docstring above records why the
    normalised maximum was dropped — it is identically 1.0 by construction.

    When only the normalised view is supplied the raw statistics fall back to it,
    so a caller that predates the raw contract still produces a well-formed
    vector rather than an exception; the values are then weaker, which is
    visible in the block's own numbers rather than hidden.
    """
    present = channel_scores is not None or raw_channel_scores is not None
    raw_view = raw_channel_scores if raw_channel_scores is not None else channel_scores

    def collect(view: Mapping[str, Mapping[str, float]] | None) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {name: [] for name in _CHANNELS}
        for scores in (view or {}).values():
            for name in _CHANNELS:
                if name in scores:
                    out[name].append(float(scores[name]))
        return out

    raw = collect(raw_view)
    norm = collect(channel_scores)

    values: list[float] = []
    for name in _CHANNELS:
        got = norm[name]
        values.extend(_stats(raw[name]))
        values.append((sum(got) / len(got)) if got else 0.0)

    fused = [float(v) for v in (atom_scores or {}).values()]
    _max, top_k, mean = _stats(fused)
    values.extend([top_k, mean])  # the maximum is constant at 1.0 by construction
    values.append(float(sum(1 for name in _CHANNELS if raw[name] or norm[name])))
    return values, present


def _pool_shape(report: Mapping[str, Any] | None) -> tuple[list[float], bool]:
    """G2's third row.

    ``cap_skipped > 0`` is a **real answerability signal**, not diagnostics:
    evidence may exist beyond the cap, so a question whose pool was truncated is
    one where "no proof in the pool" is weaker evidence of "no proof". That is
    the plan's own note on this row and the reason the field is a feature rather
    than a log line.
    """
    if report is None:
        return [0.0] * len(BLOCK_FEATURES["pool_shape"]), False
    pool_report = dict(report.get("pool") or {})
    temporal = dict(report.get("temporal") or {})
    return (
        [
            float(pool_report.get("pool_size", 0)),
            float(pool_report.get("node_atoms", 0)),
            float(pool_report.get("edge_atoms", 0)),
            float(pool_report.get("support_atoms", 0)),
            float(pool_report.get("hits_offered", 0)),
            float(pool_report.get("hits_admitted", 0)),
            float(pool_report.get("cap_skipped", 0)),
            float(pool_report.get("hits_refused_ineligible", 0)),
            float(report.get("union_pre_cap", 0)),
            float(report.get("after_temporal", 0)),
            float(temporal.get("dropped", 0)),
        ],
        True,
    )


def _saturation(saturation: Mapping[str, Any] | None) -> tuple[list[float], bool]:
    """G2's fourth row — whether retrieval selected at all.

    In **closed-atom units** (`PHASE7_DECISIONS.md` §7.2): the first version of
    Stage C's saturation flag compared *node* counts against a closed-atom cap
    and was corrected. Reading ``closed_atoms_in_scope`` here rather than
    recomputing keeps the two in the same units by construction.
    """
    if saturation is None:
        return [0.0] * len(BLOCK_FEATURES["saturation"]), False
    closed = float(saturation.get("closed_atoms_in_scope", 0))
    cap = float(saturation.get("pool_cap", 0))
    return (
        [
            float(saturation.get("candidates_in_scope", 0)),
            closed,
            cap - closed,  # signed: negative means the cap binds
            float(bool(saturation.get("exercised", False))),
        ],
        True,
    )


def build_features(
    obligations: Obligations | None = None,
    pool: Any = None,
    atom_scores: Mapping[str, float] | None = None,
    assembly_report: Mapping[str, Any] | None = None,
    *,
    snapshot: Any = None,
    saturation: Mapping[str, Any] | None = None,
    question_vector: Sequence[float] | None = None,
    embed_dim: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, bool]]:
    """Stage-C outputs → ``(vector, names, presence_flags)``.

    Every argument is optional and every block degrades to zeros **with its flag
    cleared**, because the MuSiQue adapter supplies a strict subset (G8) and the
    vector must stay the same width and the same order across both paths — exit
    criterion 3. A caller that supplies nothing gets a well-formed all-zero
    vector with every flag false, which is a legal (and uninformative) input
    rather than an error.

    ``question_vector`` is the ``with_question`` arm's only extra input. Passing
    ``None`` yields the ``pool_only`` arm at identical width, so the two arms
    differ by a **column mask** and nothing else — which is what makes the
    ablation an ablation.
    """
    dim = int(embed_dim if embed_dim is not None else (len(question_vector) if question_vector is not None else 384))

    slot_values, slot_present = _slot_coverage(obligations, pool or (), snapshot)
    channel_values, channel_present = _channel_scores(
        atom_scores,
        (assembly_report or {}).get("channel_scores"),
        (assembly_report or {}).get("raw_channel_scores"),
    )
    shape_values, shape_present = _pool_shape(assembly_report)
    sat_values, sat_present = _saturation(saturation)

    if question_vector is None:
        embed_values = [0.0] * dim
        embed_present = False
    else:
        embed_values = [float(v) for v in question_vector]
        embed_present = True
        if len(embed_values) != dim:
            raise ValueError(
                f"question_vector has {len(embed_values)} dims, expected {dim}; the "
                "vector width is frozen so an ablation stays a column mask"
            )

    flags = {
        "slot_coverage": slot_present,
        "channel_scores": channel_present,
        "pool_shape": shape_present,
        "saturation": sat_present,
        "question_embedding": embed_present,
    }
    by_block = {
        "slot_coverage": slot_values,
        "channel_scores": channel_values,
        "pool_shape": shape_values,
        "saturation": sat_values,
        "question_embedding": embed_values,
    }

    values: list[float] = []
    for block in FEATURE_BLOCKS:
        values.extend(by_block[block])
    values.extend(float(flags[block]) for block in FEATURE_BLOCKS)

    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        # A non-finite feature would train to NaN and surface as a dead arm
        # rather than as a bad input, which is the failure mode hardest to
        # attribute later.
        raise ValueError("feature vector holds a non-finite value")
    names = feature_names(dim)
    if len(names) != vector.shape[0]:
        raise ValueError(f"{len(names)} names for {vector.shape[0]} features")
    return vector, names, flags


def block_mask(names: Sequence[str], blocks: Sequence[str]) -> np.ndarray:
    """Boolean column mask selecting ``blocks`` — how an arm is applied.

    Masking rather than re-featurising is deliberate: it guarantees the two arms
    saw *the same numbers* for every shared column, so a difference between them
    is the arm and not a recomputation.
    """
    wanted = set(blocks)
    out = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        if name.startswith("present."):
            out[i] = name.split(".", 1)[1] in wanted
        elif name.startswith("q_emb_"):
            out[i] = "question_embedding" in wanted
        else:
            out[i] = name.split(".", 1)[0] in wanted
    return out
