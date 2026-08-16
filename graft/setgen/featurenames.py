"""P9.3a — the feature vocabulary, importable without torch.

**Split out of ``atomfeat`` on 16 August 2026, after an adversarial audit.**
``pins.frozen_values()`` needs the feature names to build the stage-D
fingerprint, and ``scripts/verify_handoff.py`` calls that on a bare interpreter.
Importing them from ``atomfeat`` pulled torch in — the *module* imported clean
and the *call* did not, which is the harder version of that bug to notice.

There is still exactly one definition: ``atomfeat`` imports from here.  A second
copy would be the "one value, two homes" failure `CLAUDE.md` §5 catalogues, and
this file exists to avoid it rather than to create it.
"""

from __future__ import annotations

from graft.core import obligations as ob
from graft.graphbuild.pins import EMBEDDER, TRAINING

__all__ = [
    "BLOCK_FEATURES",
    "ATOM_BLOCKS",
    "ATOM_WIDTH",
    "CHANNELS",
    "EMBED_DIM",
    "STAGE_B_DIM",
    "STATE_EXTRA_DIMS",
    "ACTION_EXTRA_DIMS",
]

#: The pinned embedder's width, and the Stage-B encoder's.  Read from their
#: owning pins rather than restated, so a change there cannot leave this module
#: silently disagreeing about a vector width.
EMBED_DIM = int(EMBEDDER["dim"])
STAGE_B_DIM = int(TRAINING["hidden"])

#: Channels whose scores become features, in **vector order**.  Stage A supplies
#: bm25 and dense; the conversational track adds entity, temporal, expand and
#: optionally the trained scorer, and they occupy the same slots by name.  Each
#: carries its own ``{channel}_present`` flag, so "did not run" and "ran and
#: scored zero" are distinguishable inputs (G2).
CHANNELS: tuple[str, ...] = ("bm25", "dense", "entity", "temporal", "expand", "scorer")

#: ``d(s)`` (6) + ``|s|/max_atoms`` + ``budget_left``.
STATE_EXTRA_DIMS = len(ob.DEFICIT_COMPONENTS) + 2

#: ``Δd`` (6) + the ``STOP``-slot flag.
ACTION_EXTRA_DIMS = len(ob.DEFICIT_COMPONENTS) + 1


def _emb_names(prefix: str, n: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{i:03d}" for i in range(n))


#: **Every feature name, in vector order, per block** — the contract
#: ``pins.frozen_values`` hashes.
#:
#: Phase 8's audit found a fingerprint binding five *block* strings while the
#: features underneath them changed from normalised to raw, so two different
#: experiments produced the same identity.  Names-and-order is the fix, adopted
#: here from the start rather than after the same failure.
BLOCK_FEATURES: dict[str, tuple[str, ...]] = {
    "text_embedding": _emb_names("emb", EMBED_DIM) + ("text_embedding_present",),
    # Three columns per channel, and the third is the correction.
    #
    # **``present`` is PER CHANNEL, not per block** (found by adversarial audit,
    # 16 Aug 2026). The block previously carried a single ``channel_scores_present``
    # set from "any channel map is non-empty" — so on the Wikipedia track, where
    # only bm25 and dense run, the entity/temporal/expand/scorer columns were
    # constant 0.0 while the flag read 1.0. G2's rule is that **absence must be an
    # input**; with one block flag the policy has no way to tell "this channel did
    # not run" from "this channel ran and scored zero everywhere", and Stage B —
    # which populates those four — would then differ from Stage A by an
    # uninstrumented change in feature *semantics* rather than by its data alone.
    # That is precisely the transfer measurement the phase exists to make.
    "channel_scores": tuple(
        f"{c}_{view}" for c in CHANNELS for view in ("raw", "norm", "present")
    ) + ("channel_scores_present",),
    "stage_b_embedding": _emb_names("sb", STAGE_B_DIM) + ("stage_b_embedding_present",),
    "obligation_match": (
        "entity_anchor_hit",
        "value_type_hit",
        "needs_source_hit",
        "scope_ok",
        "is_node",
        "is_edge",
        "obligation_match_present",
    ),
    "deficit": tuple(f"d_{c}" for c in ob.DEFICIT_COMPONENTS)
    + ("size_frac", "budget_left")
    + tuple(f"delta_{c}" for c in ob.DEFICIT_COMPONENTS)
    + ("stop_flag",),
}

#: Per-atom blocks, in vector order.  ``deficit`` is excluded: it is per-state
#: and per-transition, and it occupies the ``*_EXTRA_DIMS`` tail rather than the
#: atom block.
ATOM_BLOCKS: tuple[str, ...] = (
    "text_embedding",
    "channel_scores",
    "stage_b_embedding",
    "obligation_match",
)

#: Width of the per-atom feature vector — the sum of the atom blocks.
ATOM_WIDTH = sum(len(BLOCK_FEATURES[b]) for b in ATOM_BLOCKS)



