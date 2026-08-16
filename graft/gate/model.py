"""P8.4 — the two gate arms: logistic regression and a 2-layer MLP (decision 2).

**Deliberately small, and that is the architecture's own word.** Phase 8's row
reads "deliberately small; the gate's value is the *decision protocol*, not
capacity". LR is the baseline arm and the MLP the capacity arm; both are
reported, and the **dev-selected** one is headlined — selecting on dev rather
than after seeing the test is the whole of Gate-0 item 10's discipline applied
one phase later.

**Why two arms at all.** A single MLP that beat nothing would leave "the gate
works" resting on an unmeasured comparison. LR is the control that says whether
the capacity bought anything, and it is capacity-matched in the only sense that
matters here: both see the identical feature matrix, under the identical budget
and seeds. That is Phase 3's lesson — a comparison at unmatched budget is
uninterpretable — applied where it is cheap.

**The three P6.11 guards, verbatim and for the reasons Phase 6 learned them:**

1. **the seed reaches initialisation** (:func:`build_gate`, the ``build_arm``
   pattern) — seeding only inside the loop leaves every seed sharing one random
   init, so three seeds would estimate variance over batch order alone;
2. **early stopping restores** the argmin-dev state rather than merely stopping;
3. **a loop with no scorable dev item refuses** rather than returning its
   initialisation — an unrefused version reports a *random* gate as "early
   stopped", and every abstention number downstream would be noise wearing a
   trained model's name.

**Class weights, never resampling** (G6, Gate-0 item 6). The training balance is
1:1 by construction on contrast pairs, so the weights are usually 1.0 — they
exist for the conversational track, where skipped questions can unbalance it, and
they are **reported** either way so no natural-frequency claim can be read off a
constructed balance.

Torch is imported at module scope here and **nowhere else in ``graft.gate``**:
the containment guard in ``test_structure`` asserts that every other gate module
stays importable on a bare interpreter, so the feature contract, the label
recipe and the risk–coverage instrument can all be checked without the ML stack.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from graft.gate.pins import MODELS, TRAINING_GATE

__all__ = ["GateNet", "build_gate", "class_weights", "train_gate", "predict"]


class GateNet(nn.Module):
    """One arm. ``hidden = 0`` is logistic regression; otherwise a 2-layer MLP.

    One class rather than two so the arms cannot drift apart in anything except
    the layer they differ by — the same reasoning that made Phase 6's mention
    projector identical across its arms.
    """

    def __init__(self, in_dim: int, hidden: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        if self.hidden <= 0:
            self.net: nn.Module = nn.Linear(self.in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(n,)`` logits — *not* probabilities.

        Logits, because the loss is ``binary_cross_entropy_with_logits`` (numerically
        stabler than sigmoid-then-BCE) and because the threshold in
        :mod:`graft.gate.decide` is defined on the probability, applied once, in
        one place.
        """
        return self.net(x).reshape(-1)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_gate(arm: str, in_dim: int, seed: int | None = None) -> GateNet:
    """Construct one arm, **seeding initialisation** when a seed is given.

    Refuses a configuration over ``pins.MODELS['max_params']``: the gate is
    specified as small, and one that quietly grew past the cap would turn the
    LR-vs-MLP comparison into a capacity confound rather than a protocol
    comparison.
    """
    if arm not in ("lr", "mlp"):
        raise KeyError(f"unknown arm {arm!r}; decision 2's arms are 'lr' and 'mlp'")
    if seed is not None:
        torch.manual_seed(int(seed))
    spec = MODELS[arm]
    model = GateNet(in_dim, hidden=int(spec["hidden"]), dropout=float(spec.get("dropout", 0.0)))
    count = parameter_count(model)
    if count > int(MODELS["max_params"]):
        raise ValueError(
            f"gate arm {arm!r} has {count:,} parameters, over decision 2's "
            f"{int(MODELS['max_params']):,}; the gate is specified as small and the "
            "LR comparison would become a capacity confound"
        )
    return model


def class_weights(labels: Sequence[float]) -> tuple[float, dict[str, Any]]:
    """``pos_weight`` for BCE, plus the balance report exit criterion 10 requires.

    Weights, **not** resampling: resampling changes the dataset and hides the
    imbalance; a weight leaves the data alone and puts the correction somewhere a
    reader can see it. On 1:1 contrast pairs this returns 1.0, which is the
    honest answer rather than a no-op to be optimised away.
    """
    positives = float(sum(1 for y in labels if y >= 0.5))
    negatives = float(len(labels) - positives)
    weight = (negatives / positives) if positives else 1.0
    return weight, {
        "answerable": int(positives),
        "unanswerable": int(negatives),
        "pos_weight": weight,
        "handling": "class weights, not resampling (Gate-0 item 6)",
        "reading": (
            "a constructed 1:1 training balance; no natural-frequency claim may be "
            "made from it (G6). Evaluation keeps natural prevalence."
        ),
    }


def train_gate(
    model: GateNet,
    train: tuple[np.ndarray, np.ndarray],
    dev: tuple[np.ndarray, np.ndarray],
    seed: int,
    budget: Mapping[str, Any] | None = None,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train one arm under the shared budget, early-stopping on dev loss.

    ``mask`` is the arm's column mask (``features.block_mask``). Masking rather
    than re-featurising guarantees the two arms saw *the same numbers* in every
    shared column, so a difference between them is the arm and not a
    recomputation — exit criterion 9's "identical budgets" in the strongest
    available sense.

    The budget is read from ``pins.TRAINING_GATE`` and never from an argument
    default, so a caller cannot quietly give one arm more epochs; ``budget``
    exists for tests and is reported when it differs.
    """
    spec = dict(budget or TRAINING_GATE)
    torch.manual_seed(int(seed))

    x_tr, y_tr = _as_tensors(train, mask)
    x_dev, y_dev = _as_tensors(dev, mask)
    weight, balance = class_weights(y_tr.tolist())

    optimiser = torch.optim.Adam(
        model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"])
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_dev = math.inf
    patience = int(spec["early_stop_patience"])
    since_best = 0
    batch = max(1, int(spec["batch_size"]))

    for epoch in range(int(spec["epochs"])):
        model.train()
        # Deterministic batch order per epoch: a torch.randperm here would make
        # the run depend on global RNG consumed elsewhere, which is exactly what
        # made Phase 6's seeds estimate the wrong variance.
        order = torch.arange(x_tr.shape[0])
        if x_tr.shape[0] > 1:
            generator = torch.Generator().manual_seed(int(seed) + epoch)
            order = torch.randperm(x_tr.shape[0], generator=generator)
        total = 0.0
        for start in range(0, x_tr.shape[0], batch):
            idx = order[start : start + batch]
            optimiser.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(idx)
        train_loss = total / max(1, x_tr.shape[0])

        model.eval()
        with torch.no_grad():
            dev_value = (
                float(loss_fn(model(x_dev), y_dev).item()) if x_dev.shape[0] else math.inf
            )
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_value})

        if dev_value < best_dev - 1e-6:
            best_dev, since_best = dev_value, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break

    if not math.isfinite(best_dev):
        raise ValueError(
            f"gate arm has no scorable dev item ({x_dev.shape[0]} dev rows): early "
            "stopping cannot select, so training would return the random "
            "initialisation and every abstention number would be noise. Enlarge "
            "the split or the labelled set."
        )
    model.load_state_dict(best_state)  # restores, never merely stops
    model.eval()
    return {
        "arm": "lr" if model.hidden <= 0 else "mlp",
        "seed": int(seed),
        "epochs_run": len(history),
        "best_dev_loss": best_dev,
        "history": history,
        "parameters": parameter_count(model),
        "features_used": int(x_tr.shape[1]),
        "class_balance": balance,
        "budget": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
    }


def predict(model: GateNet, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``p_answerable`` per row — the probability, not the logit."""
    matrix = x[:, mask] if mask is not None else x
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(np.asarray(matrix), dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def _as_tensors(
    data: tuple[np.ndarray, np.ndarray], mask: np.ndarray | None
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = data
    x = np.asarray(x, dtype=np.float32)
    if mask is not None:
        x = x[:, mask]
    return (
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32),
    )
