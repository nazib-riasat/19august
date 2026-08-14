"""P6.11 — the multi-seed trainer, and the arms Gate 1 compares (decision 10).

**Why this module exists at all, recorded because its absence was a plan
defect.**  `GRAFT_PHASE6_BUILD.md` §3 lists ten modules and none of them is a
trainer; build-order step 5 asked only for "one smoke epoch per arm", which the
learning tests satisfy.  So encoders forwarded, decoders backwarded and
calibration fitted — and nothing drove them through `TRAINING`'s budget across
arms and seeds, which is what a *comparison* needs.  Built 14 Aug 2026 after the
audit; the plan gains P6.11 rather than pretending the gap was an open item.

**The budget is one dict, and that is the whole point** (decision 10, and Phase
3's lesson one stage later).  Every learned arm gets the same optimizer, the
same epoch ceiling, the same early-stop rule and the same seeds; parameter
counts are reported per arm because capacity differs by architecture and a
comparison that hides that is uninterpretable in whichever direction it lands.

**Three properties this module is responsible for, each easy to lose:**

*Splits are user-level and chronological* (`GATE0_CONTRACT.md` item 5).
User-level is structural here: a split boundary is a ``conv_id`` boundary, and
a LongMemEval ``question_id`` *is* one simulated user's whole haystack, so no
conversation can straddle two splits.  Chronology inside a conversation is
handled upstream and better: every item was featurized against the construction
snapshot it was decided at (G12's ``stage_b_seq``), so a later update cannot
leak into an earlier example even within one split.

*Unreachable gold is counted, never dropped quietly.*  When gold says
`LINK_EXISTING(e)` and `e` is not in that item's candidate list, **no model can
emit it** — the candidate generator's recall is the ceiling (G4).  Such items
are excluded from the loss (there is no correct logit to raise) and scored as
**automatic failures at test**, with the count reported.  Dropping them from the
test set would let every arm's score float up on the generator's misses.

*Class weights, never resampling* (decision 6, contract item 6) — resampling
would make the quoted class balance unreproducible, and D2's rare classes are
the contribution.

**The similarity arm has no parameters, and that is not an exemption.**  It gets
the same dev split to choose its one threshold on, so "tuned on dev, scored on
test" holds for every arm; it simply has one knob instead of a gradient.
"""

from __future__ import annotations

import copy
import math
import random
import re
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from graft.graphbuild.decoders import (
    NON_LINK_ACTIONS,
    D1Decoder,
    D2Decoder,
    TemperatureScaler,
    brier_score,
    class_weights,
    expected_calibration_error,
)
from graft.graphbuild.encoders import build_encoder, parameter_count
from graft.graphbuild.features import BASE_DIM, GRAFT_DIM, build_features, encoder_metadata
from graft.graphbuild.items import D2_LABELS
from graft.graphbuild.pins import EMBEDDER, TRAINING
from graft.graphstore import ReplayGraphStore

__all__ = [
    "SPLIT_SEED",
    "ARM_VARIANTS",
    "split_questions",
    "parse_d1_gold",
    "read_labels",
    "FeatureCache",
    "D1Arm",
    "SimilarityArm",
    "build_arm",
    "d1_action_weights",
    "d1_loss",
    "train_d1_arm",
    "predict_d1",
    "train_d2",
    "calibrate",
]

#: `GATE0_CONTRACT.md` item 5's seed — the project's date-seed convention.
SPLIT_SEED = 20260813

#: Which feature variant each arm consumes (decision 15).  E3's *only*
#: difference from E2 is this row, which is what makes "E3 beats E2" a statement
#: about the feature set rather than about depth or width.
ARM_VARIANTS: Mapping[str, str] = {
    "similarity": "base",
    "E1": "base",
    "E2": "base",
    "E3": "graft",
}

_LINK = re.compile(r"^LINK_EXISTING\((.*)\)$")


# --------------------------------------------------------------------------
# splits and labels
# --------------------------------------------------------------------------


def split_questions(
    question_ids: Sequence[str],
    question_types: Mapping[str, str] | None = None,
    proportions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = SPLIT_SEED,
) -> dict[str, tuple[str, ...]]:
    """Question ids → ``{train, dev, test}``, user-level and stratified.

    **User-level by construction**: a LongMemEval ``question_id`` is one
    simulated user's entire haystack and is the ``conv_id`` of every turn in it,
    so splitting on question ids means no conversation appears in two splits —
    item 5's requirement, met structurally rather than by a check.

    Stratified by ``question_type`` when supplied: knowledge-update questions
    are 78 of 500 and an unstratified draw moves D2's rare classes between
    splits (item 5's own reasoning).  Sorted before shuffling so the draw is a
    function of the seed and not of dict order.
    """
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError(f"proportions must sum to 1, got {proportions}")

    by_type: dict[str, list[str]] = {}
    for qid in sorted(set(question_ids)):
        by_type.setdefault((question_types or {}).get(qid, ""), []).append(qid)

    out: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    rng = random.Random(seed)
    for _, members in sorted(by_type.items()):
        pool = list(members)
        rng.shuffle(pool)
        n = len(pool)
        n_train = int(round(n * proportions[0]))
        n_dev = int(round(n * proportions[1]))
        # A stratum too small to fill every split gives what it can, train
        # first: an empty dev stratum is a reported thinness, not a crash.
        out["train"].extend(pool[:n_train])
        out["dev"].extend(pool[n_train : n_train + n_dev])
        out["test"].extend(pool[n_train + n_dev :])
    return {k: tuple(sorted(v)) for k, v in out.items()}


def parse_d1_gold(label: str) -> tuple[str, str | None]:
    """``"LINK_EXISTING(e7)"`` → ``("LINK_EXISTING", "e7")``; a bare action → ``(action, None)``.

    The annotate CLI writes the entity id *inside* the action string (Phase 2.5),
    and the Gate-1 primary needs the two apart — action and id are scored as a
    conjunction (plan §6.4), so a parser that dropped the id would silently
    weaken the metric to action-only.
    """
    match = _LINK.match(label.strip())
    if match:
        return "LINK_EXISTING", match.group(1)
    action = label.strip()
    if action not in NON_LINK_ACTIONS:
        raise ValueError(
            f"unknown D1 label {label!r}; expected LINK_EXISTING(<id>) or one of "
            f"{list(NON_LINK_ACTIONS)}"
        )
    return action, None


def read_labels(rows: Sequence[Mapping[str, Any]], pass_no: int = 1) -> dict[str, str]:
    """``item_id -> label`` from annotate-CLI rows, one pass only.

    Pass 2 is the κ re-annotation of a *subset* (Phase 2.5 amendment A1); mixing
    it into training would put two labels of the same item in one map, with the
    later write silently winning.
    """
    return {
        str(r["item_id"]): str(r["label"])
        for r in rows
        if int(r.get("pass", 1)) == pass_no
    }


# --------------------------------------------------------------------------
# features, once
# --------------------------------------------------------------------------


class FeatureCache:
    """``stage_b_seq -> GraphFeatures``, built once and reused across every
    epoch, seed and arm of one variant.

    G12 requires each example to be featurized against the graph *as it stood
    when that decision was made* — ``at(stage_b_seq)`` on the construction log.
    Doing that per example per epoch would replay the log tens of thousands of
    times; doing it once per distinct sequence is the same computation with the
    redundancy removed, and the cache key *is* the correctness condition.
    """

    def __init__(
        self,
        log: Any,
        variant: str,
        embed: Callable[[Sequence[str]], Any] | None = None,
    ) -> None:
        self.store = ReplayGraphStore(log)
        self.variant = variant
        self.embed = embed
        self._cache: dict[tuple[int, str], Any] = {}

    def at(self, seq: int | None, ref_ts: str | None = None) -> Any:
        """Features at construction position ``seq``, timed against ``ref_ts``.

        **``ref_ts`` is the decision's own "now", and passing it is not
        optional in a comparison.**  Left to default, ``build_features`` anchors
        the relative temporal encoding at the latest edge in the *whole*
        snapshot — which, on a construction log holding many conversations, is
        some other user's calendar: measured on the pilot's Stage-B log, one
        conversation's newest node carried Δ = 203.9 days instead of 0, and the
        RTE is 8 of the 10 ``base`` dimensions E1 and E2 consume.  Anchoring at
        the item's own turn makes Δt mean "how old is this node from where the
        decision stands", which is what a *relative* encoding is for, and
        removes the cross-conversation dependence entirely.  (Found by the
        14 Aug review.)
        """
        key = (-1 if seq is None else int(seq), ref_ts or "")
        if key not in self._cache:
            snapshot = self.store.at(None if key[0] < 0 else key[0])
            self._cache[key] = build_features(
                snapshot, self.variant, embed=self.embed, ref_ts=ref_ts
            )
        return self._cache[key]

    def __len__(self) -> int:
        return len(self._cache)


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------


class D1Arm(nn.Module):
    """One learned mention-resolution arm: encoder + mention projector + D1 head.

    The **mention projector** is the one piece not named in the architecture, and
    it is here because mentions are deliberately not graph nodes
    (`PHASE6_DECISIONS.md` §4): a mention's representation has to come from its
    text, and the D1 head compares it against *encoder* outputs, so the two must
    live in one space.  It is a single `Linear(embed_dim → hidden)`, **identical
    across every arm**, so it cannot advantage one of them — the same reasoning
    that made the capacity match Phase 3's concern.
    """

    def __init__(
        self,
        arm: str,
        hidden: int | None = None,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(hidden or TRAINING["hidden"])
        embed_dim = int(embed_dim or EMBEDDER["dim"])
        in_dim = GRAFT_DIM if ARM_VARIANTS[arm] == "graft" else BASE_DIM
        self.arm = arm
        self.variant = ARM_VARIANTS[arm]
        self.hidden = hidden
        self.encoder = build_encoder(arm, in_dim, encoder_metadata(), hidden)
        self.mention_projector = nn.Linear(embed_dim, hidden)
        self.d1 = D1Decoder(hidden, hidden)

    def forward(
        self, features: Any, mention_vec: "torch.Tensor", candidate_ids: Sequence[str]
    ) -> "torch.Tensor":
        """``(len(candidates) + 3,)`` logits for one item."""
        node_reprs = self.encoder(features)
        mention = self.mention_projector(mention_vec)
        rows = _gather_entities(features, node_reprs, candidate_ids, self.hidden)
        return self.d1(mention, rows)


def _gather_entities(
    features: Any, node_reprs: Mapping[str, "torch.Tensor"], ids: Sequence[str], hidden: int
) -> "torch.Tensor":
    """Candidate ids → their encoder rows, in the order the item lists them.

    Order is load-bearing: the head's logit *i* is candidate *i*, and the gold
    target is an index into that same list.  A candidate the graph does not
    contain (possible only if a caller mixes snapshots) contributes a zero row
    rather than shifting every later index.
    """
    entity_rows = node_reprs.get("Entity")
    positions = {nid: ix for ix, nid in enumerate(features.node_ids.get("Entity", []))}
    if not ids:
        return torch.zeros((0, hidden))
    out = torch.zeros((len(ids), hidden))
    for slot, entity_id in enumerate(ids):
        ix = positions.get(entity_id)
        if ix is not None and entity_rows is not None and ix < entity_rows.shape[0]:
            out[slot] = entity_rows[ix]
    return out


class SimilarityArm:
    """The no-learning control: link to the best candidate above a threshold.

    **[EVIDENCE-adjacent]** this is the string/embedding-similarity linker the
    plan's Tier-1 ladder names as the arm that must be beaten before any learned
    claim is worth making — the Stage-B analogue of Gate 3's training-free
    baselines, and the cheapest thing that could embarrass the project.

    Its single knob (the threshold) is chosen on **dev**, like every other arm's
    parameters, so "tuned on dev, scored on test" is uniform across the
    comparison rather than an exemption granted to the simple method.
    """

    name = "similarity"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def fit(self, items: Sequence[Mapping[str, Any]], gold: Mapping[str, str]) -> float:
        """Sweep the threshold on dev and keep the best end-to-end score."""
        grid = [i / 20 for i in range(21)]
        best, best_score = self.threshold, -1.0
        for candidate in grid:
            self.threshold = candidate
            score = sum(
                1
                for item in items
                if item["item_id"] in gold
                and _correct(self.predict_one(item), parse_d1_gold(gold[item["item_id"]]))
            )
            if score > best_score:
                best, best_score = candidate, score
        self.threshold = best
        return best

    def predict_one(self, item: Mapping[str, Any]) -> dict[str, Any]:
        candidates = list(item.get("candidates", ()))
        if not candidates:
            return {"action": "CREATE_NEW_ENTITY", "entity_id": None, "confidence": 1.0}
        best = max(candidates, key=lambda c: float(c.get("score", 0.0)))
        if float(best.get("score", 0.0)) >= self.threshold:
            return {
                "action": "LINK_EXISTING",
                "entity_id": str(best["entity_id"]),
                "confidence": float(best.get("score", 0.0)),
            }
        return {"action": "CREATE_NEW_ENTITY", "entity_id": None, "confidence": 1.0}

    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_one(i) for i in items]

    @staticmethod
    def parameter_count() -> int:
        return 0


def build_arm(arm: str, hidden: int | None = None, seed: int | None = None) -> Any:
    """Construct one arm, **seeding initialisation** when a seed is given.

    The seed belongs here, not only in the training loop: seeding after
    construction would leave every seed sharing one random initialisation, so
    decision 10's three seeds would estimate variance over dropout and batch
    order alone — a far narrower interval than the one a reader will assume,
    and irreproducible besides.  (Caught by the determinism test, 14 Aug 2026.)
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    if arm == "similarity":
        return SimilarityArm()
    if arm in ("E1", "E2", "E3"):
        return D1Arm(arm, hidden)
    raise KeyError(f"unknown arm {arm!r}; the Tier-1 ladder is similarity, E1, E2, E3")


def _correct(prediction: Mapping[str, Any], gold: tuple[str, str | None]) -> bool:
    action, entity = gold
    if prediction.get("action") != action:
        return False
    return action != "LINK_EXISTING" or prediction.get("entity_id") == entity


# --------------------------------------------------------------------------
# the identical-budget loop
# --------------------------------------------------------------------------


def _target_index(item: Mapping[str, Any], gold: tuple[str, str | None]) -> int | None:
    """The logit index gold names, or ``None`` when the head cannot express it.

    ``None`` is the **unreachable gold** case: gold links to an entity the
    candidate generator did not propose, so no logit corresponds to the right
    answer.  Reported, excluded from the loss, and scored as a failure at test
    (see the module docstring).
    """
    action, entity = gold
    ids = [str(c["entity_id"]) for c in item.get("candidates", ())]
    if action == "LINK_EXISTING":
        return ids.index(entity) if entity in ids else None
    return len(ids) + NON_LINK_ACTIONS.index(action)


def _forward_item(
    arm: D1Arm, cache: FeatureCache, item: Mapping[str, Any], mention_vec: "torch.Tensor"
) -> "torch.Tensor":
    features = cache.at(item.get("stage_b_seq"), item.get("turn_ts"))
    ids = [str(c["entity_id"]) for c in item.get("candidates", ())]
    return arm(features, mention_vec, ids)


def d1_action_weights(
    items: Sequence[Mapping[str, Any]], gold: Mapping[str, str]
) -> "torch.Tensor":
    """Inverse-frequency weights over the four actions — never resampling.

    Module-level rather than inline so a caller can *recompute the same loss*
    the trainer minimised: a dev loss computed under different weights is a
    different quantity, and comparing the two silently is how an early-stopping
    check turns into a class-balance artefact.
    """
    action_index = {a: i for i, a in enumerate(("LINK_EXISTING",) + NON_LINK_ACTIONS)}
    labels = [
        action_index[parse_d1_gold(gold[i["item_id"]])[0]]
        for i in items
        if i["item_id"] in gold
    ]
    if not labels:
        return torch.ones(len(action_index))
    return class_weights(labels, len(action_index))


def d1_loss(
    arm: D1Arm,
    cache: FeatureCache,
    items: Sequence[Mapping[str, Any]],
    gold: Mapping[str, str],
    mention_vectors: Mapping[str, "torch.Tensor"],
    weights: "torch.Tensor",
) -> tuple["torch.Tensor", int, int]:
    """``(mean weighted cross-entropy, scored, unreachable)`` over ``items``.

    One definition, used by training, by the dev check and by any caller that
    wants to re-derive a reported number — the same "one home" rule the rest of
    the project applies to shared quantities.
    """
    action_index = {a: i for i, a in enumerate(("LINK_EXISTING",) + NON_LINK_ACTIONS)}
    total = torch.zeros(())
    scored = unreachable = 0
    for item in items:
        label = gold.get(item["item_id"])
        if label is None:
            continue
        parsed = parse_d1_gold(label)
        target = _target_index(item, parsed)
        if target is None:
            unreachable += 1
            continue
        logits = _forward_item(arm, cache, item, mention_vectors[item["item_id"]])
        total = total + weights[action_index[parsed[0]]] * nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([target])
        )
        scored += 1
    return total / max(scored, 1), scored, unreachable


def train_d1_arm(
    arm: D1Arm,
    cache: FeatureCache,
    items_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    gold: Mapping[str, str],
    mention_vectors: Mapping[str, "torch.Tensor"],
    seed: int,
    budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one arm under the shared budget, early-stopping on dev loss.

    Returns the history and the restored-best state's dev loss.  **The budget is
    read from ``pins.TRAINING`` and never from an argument default**, so a caller
    cannot quietly give one arm more epochs; ``budget`` exists to let tests run a
    two-epoch version and is reported in the result when it differs.
    """
    budget = dict(budget or TRAINING)
    torch.manual_seed(seed)

    optimiser = torch.optim.Adam(
        arm.parameters(), lr=float(budget["lr"]), weight_decay=float(budget["weight_decay"])
    )
    train_items = list(items_by_split.get("train", ()))
    dev_items = list(items_by_split.get("dev", ()))

    # Class weights over the *action* distribution, never resampling (decision 6),
    # derived from TRAIN only — dev-derived weights would leak the dev balance.
    weights = d1_action_weights(train_items, gold)

    def _loss(items: Sequence[Mapping[str, Any]]) -> tuple["torch.Tensor", int, int]:
        return d1_loss(arm, cache, items, gold, mention_vectors, weights)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(arm.state_dict())
    best_dev = math.inf
    patience = int(budget["early_stop_patience"])
    since_best = 0
    n_train = unreachable = 0  # bound before the loop: epochs=0 is a legal budget

    for epoch in range(int(budget["epochs"])):
        arm.train()
        optimiser.zero_grad()
        train_loss, n_train, unreachable = _loss(train_items)
        if n_train:
            train_loss.backward()
            optimiser.step()

        arm.eval()
        with torch.no_grad():
            dev_loss, n_dev, _ = _loss(dev_items)
        dev_value = float(dev_loss.item()) if n_dev else math.inf
        history.append(
            {"epoch": epoch, "train_loss": float(train_loss.item()), "dev_loss": dev_value}
        )

        if dev_value < best_dev - 1e-6:
            best_dev, since_best = dev_value, 0
            best_state = copy.deepcopy(arm.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break

    if not math.isfinite(best_dev):
        # **A model that never saw a scorable dev item is a random model, and
        # reporting it as "early stopped" would be the worst kind of quiet
        # failure** — every arm returns its initialisation, `predict_d1` scores
        # noise, and the McNemar table reads as four arms agreeing.  Reachable
        # on small splits: ``n_dev = round(n*0.2)`` is 0 for n ≤ 2, and a dev
        # draw whose gold is entirely unreachable scores nothing either.
        # (Found by the 14 Aug review.)
        raise ValueError(
            f"arm {arm.arm} has no scorable dev item ({len(dev_items)} dev items, "
            f"{n_train} train items scored, {unreachable} unreachable): early "
            "stopping cannot select, so training would return the random "
            "initialisation. Enlarge the split or the labelled set."
        )
    arm.load_state_dict(best_state)  # early stopping restores, never merely stops
    arm.eval()
    return {
        "arm": arm.arm,
        "seed": seed,
        "epochs_run": len(history),
        "best_dev_loss": best_dev,
        "history": history,
        "parameters": parameter_count(arm),
        "train_items_scored": n_train,
        "train_items_unreachable": unreachable,
        "class_weights": [float(w) for w in weights],
        "budget": {k: v for k, v in budget.items() if k != "seeds"},
    }


def predict_d1(
    arm: Any,
    cache: FeatureCache | None,
    items: Sequence[Mapping[str, Any]],
    mention_vectors: Mapping[str, "torch.Tensor"] | None = None,
    gold: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Predictions in item order, one per item — the shape `gate1` scores.

    An item whose gold is **unreachable** (the generator never proposed the right
    entity) is emitted as the arm's actual prediction, which will be wrong by
    construction; it is not skipped, because skipping would shrink the
    denominator and inflate every arm equally.
    """
    if isinstance(arm, SimilarityArm):
        return arm.predict(items)

    out: list[dict[str, Any]] = []
    assert cache is not None and mention_vectors is not None
    arm.eval()
    with torch.no_grad():
        for item in items:
            ids = [str(c["entity_id"]) for c in item.get("candidates", ())]
            logits = _forward_item(arm, cache, item, mention_vectors[item["item_id"]])
            action, entity, confidence = D1Decoder.decode(logits, ids)
            out.append(
                {"action": action, "entity_id": entity, "confidence": confidence}
            )
    return out


# --------------------------------------------------------------------------
# D2 (a Gate-1 secondary) and calibration
# --------------------------------------------------------------------------


def train_d2(
    decoder: D2Decoder,
    cache: FeatureCache,
    items_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    gold: Mapping[str, str],
    claim_vectors: Callable[[Mapping[str, Any], Any], tuple["torch.Tensor", "torch.Tensor"]],
    seed: int,
    budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The same loop for D2's four-way pair decision (a Gate-1 secondary).

    Separate from :func:`train_d1_arm` rather than generalised, because the two
    differ in exactly the place a generalisation would hide: D1's output width
    varies per item and its gold may be unreachable; D2's is a fixed four-way
    softmax over a pair.  One function pretending to both would need a branch at
    every step of the loop.
    """
    budget = dict(budget or TRAINING)
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(
        decoder.parameters(), lr=float(budget["lr"]), weight_decay=float(budget["weight_decay"])
    )
    index = {label: i for i, label in enumerate(D2_LABELS)}
    labels = [
        index[gold[i["item_id"]]]
        for i in items_by_split.get("train", ())
        if i["item_id"] in gold
    ]
    weights = class_weights(labels, len(D2_LABELS)) if labels else torch.ones(len(D2_LABELS))

    def _loss(items: Sequence[Mapping[str, Any]]) -> tuple["torch.Tensor", int]:
        total = torch.zeros(())
        n = 0
        for item in items:
            label = gold.get(item["item_id"])
            if label is None:
                continue
            features = cache.at(item.get("stage_b_seq"), item.get("turn_ts"))
            a, b = claim_vectors(item, features)
            logits = decoder(a, b)
            total = total + weights[index[label]] * nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([index[label]])
            )
            n += 1
        return total / max(n, 1), n

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(decoder.state_dict())
    best_dev, since_best = math.inf, 0
    for epoch in range(int(budget["epochs"])):
        decoder.train()
        optimiser.zero_grad()
        train_loss, n_train = _loss(items_by_split.get("train", ()))
        if n_train:
            train_loss.backward()
            optimiser.step()
        decoder.eval()
        with torch.no_grad():
            dev_loss, n_dev = _loss(items_by_split.get("dev", ()))
        dev_value = float(dev_loss.item()) if n_dev else math.inf
        history.append({"epoch": epoch, "train_loss": float(train_loss.item()), "dev_loss": dev_value})
        if dev_value < best_dev - 1e-6:
            best_dev, since_best = dev_value, 0
            best_state = copy.deepcopy(decoder.state_dict())
        else:
            since_best += 1
            if since_best >= int(budget["early_stop_patience"]):
                break
    if not math.isfinite(best_dev):
        # Same guard as `train_d1_arm`, for the same reason: a decoder that
        # never saw a scorable dev item is its random initialisation, and
        # returning it as "trained" would poison the D2 secondary silently.
        raise ValueError(
            f"D2 has no scorable dev item ({len(items_by_split.get('dev', ()))} "
            "dev items): early stopping cannot select, so training would return "
            "the random initialisation. Enlarge the split or the labelled set."
        )
    decoder.load_state_dict(best_state)
    decoder.eval()
    return {
        "decoder": "D2",
        "seed": seed,
        "epochs_run": len(history),
        "best_dev_loss": best_dev,
        "history": history,
        "parameters": parameter_count(decoder),
        "class_weights": [float(w) for w in weights],
    }


def calibrate(
    logits: "torch.Tensor", targets: "torch.Tensor"
) -> dict[str, Any]:
    """Fit a temperature on **dev** and report Brier/ECE before and after.

    Decision 8, and decision 9's prerequisite: the D3/D4 commit floor is a
    confidence threshold, and an uncalibrated threshold is a number with no
    meaning.  Both metrics are reported because they fail differently — ECE is
    blind to a model that is confidently wrong in a balanced way across bins.
    """
    before = torch.softmax(logits, dim=-1)
    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, targets)
    after = torch.softmax(scaler(logits), dim=-1)
    return {
        "temperature": temperature,
        "brier_before": brier_score(before, targets),
        "brier_after": brier_score(after, targets),
        "ece_before": expected_calibration_error(before, targets),
        "ece_after": expected_calibration_error(after, targets),
        "fitted_on": "dev",
    }
