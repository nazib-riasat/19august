"""The frozen learner-facing interface, and the heads every objective needs.

**Fix F6 in one line:** ``policy(state_repr, action_reprs) → logits``. A learner
sees tensors and never learns where they came from — an MLP featurizer on the
synthetic lattice, the Stage-B graph encoder at Phase 9. Nothing in this module
imports a ``StateGraph``, a ``LatticeInstance`` or an atom id, and a test asserts
it.

**Four heads, and three of them are not optional** (G12):

``action_logits``
    ADD-per-atom ∪ STOP. Masking happens in the featurizer, *before* the softmax.

``LogZHead``
    ``log Z_θ(q)`` over the pooled instance embedding — a **conditional**
    GFlowNet, **[EVIDENCE]** per *GFlowNet Foundations* (JMLR 2023). One model
    per (arm, seed) is trained across all 20 main-suite instances and conditioned
    on which one it is (G6); without this head that is not expressible.

``StateFlowHead``
    ``log F(s)``. **[EVIDENCE]** LED-GFN's objective is DB-style over
    ``log F̃(s)`` and its Algorithm 1 initialises "forward and backward policy
    ``P_F``, ``P_B``, state flow ``F̃``, and model ``φ_θ``"; GAFlowNet's augmented
    trajectory balance carries ``r(s→s′)/F(s′)``. Plain TB needs no state flow —
    **augmented TB does**, and so does SubTB. Required by L5, L6, L7, L7b and
    GAFlowNet.

``PotentialHead``
    ``φ_θ(s→s′)``, the learnable energy decomposition. **[EVIDENCE]** LED-GFN
    Appendix C: "the neural network architecture of potential function is
    identical to that of GFlowNet policy" — *identical architecture*, which is a
    **separate network of the same shape**, not a head hanging off the policy's
    trunk. It has to be separate: decision 23a gives the potential its own
    learning rate (0.001) and its own inner loop, and a shared trunk would let
    that optimiser drag the policy along with it.

``DeficitHead``
    L7b only (decision 26): predicts ``d(X(τ))``, the deficit vector of the
    terminal the rollout actually reached. **Forward-looking by construction** —
    fix F11 retired the head that predicted ``d(s)``, which is already an encoder
    input and therefore a copy task. This one sits on the policy trunk because it
    is an auxiliary loss on the policy's representation, which is the only way it
    can shape what the policy learns.

**Capacity counts every head** (G4, decision 27). Contribution 3's control is
capacity-matched LED-GFN, and a match computed over the policy trunk alone would
be wrong by the size of the heads — in the direction that flatters L7, since L7
is the arm with the extra input features.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

__all__ = [
    "mlp",
    "Policy",
    "LogZHead",
    "StateFlowHead",
    "PotentialHead",
    "DeficitHead",
    "capacity",
    "match_capacity",
]


def mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    """The one MLP factory. LED requires the potential network to match the
    policy's architecture, so both come from here."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    layers: list[nn.Module] = []
    width = in_dim
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class Policy(nn.Module):
    """``action_logits(state_repr, action_reprs) → logits`` and nothing else.

    Scoring is **per action against the state**, not a fixed-width output layer:
    the action set is the pool, which differs between instances and between the
    synthetic and real environments. A fixed head would have hard-coded
    ``n_atoms`` and broken fix F6 at Phase 9.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.trunk = mlp(state_dim, hidden, hidden, depth)
        self.scorer = mlp(hidden + action_dim, hidden, 1, depth)
        self.hidden = hidden

    def encode(self, state_repr: torch.Tensor) -> torch.Tensor:
        """``h_s`` — shared by the logits, the flow head and the potential."""
        return self.trunk(state_repr)

    def action_logits(
        self, state_repr: torch.Tensor, action_reprs: torch.Tensor
    ) -> torch.Tensor:
        h = self.encode(state_repr)
        n, n_actions, _ = action_reprs.shape
        pair = torch.cat([h.unsqueeze(1).expand(n, n_actions, self.hidden), action_reprs], dim=2)
        return self.scorer(pair).squeeze(-1)


class LogZHead(nn.Module):
    """``log Z_θ(q)`` from a pooled instance embedding (G6)."""

    def __init__(self, instance_dim: int, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.net = mlp(instance_dim, hidden, 1, depth)

    def forward(self, instance_repr: torch.Tensor) -> torch.Tensor:
        return self.net(instance_repr).squeeze(-1)


class StateFlowHead(nn.Module):
    """``log F(s)`` (G12). Required by SubTB, LED-DB and augmented TB."""

    def __init__(self, hidden: int, depth: int = 2) -> None:
        super().__init__()
        self.net = mlp(hidden, hidden, 1, depth)

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        return self.net(h_s).squeeze(-1)


class PotentialHead(nn.Module):
    """``φ_θ(s→s′)`` — LED-GFN's learnable energy decomposition.

    Defined on the **transition**, which is what makes it a decomposition rather
    than a state value: LED's Eq. 3 sums ``φ_θ`` along a trajectory to recover the
    terminal energy, and its Eq. 5 trains that sum with dropout.

    Its shape is ``Policy``'s shape — trunk over the state, then a scorer over
    ``(h, action)`` — per Appendix C, but its weights are its own.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden: int = 64, depth: int = 2
    ) -> None:
        super().__init__()
        self.trunk = mlp(state_dim, hidden, hidden, depth)
        self.net = mlp(hidden + action_dim, hidden, 1, depth)

    def forward(self, state_repr: torch.Tensor, action_repr: torch.Tensor) -> torch.Tensor:
        h = self.trunk(state_repr)
        return self.net(torch.cat([h, action_repr], dim=-1)).squeeze(-1)


class DeficitHead(nn.Module):
    """L7b's auxiliary target: ``d(X(τ)) ∈ [0,1]^6`` from ``h_s`` (decision 26).

    Sigmoid because every deficit component is normalised to [0, 1] by
    ``graft.core.obligations``; an unbounded head would spend its capacity
    learning that range instead of the ordering inside it.
    """

    def __init__(self, hidden: int, out_dim: int, depth: int = 2) -> None:
        super().__init__()
        self.net = mlp(hidden, hidden, out_dim, depth)

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h_s))


def capacity(*modules: nn.Module) -> int:
    """Trainable parameters across every module given (G4, decision 11).

    Buffers and frozen parameters are excluded; **every head is included**. The
    matched-pair assertion reads this, so a head left out here is a capacity
    match that is wrong by exactly that head.
    """
    seen: set[int] = set()
    total = 0
    for module in modules:
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def match_capacity(
    count: "Callable[[int], int]",
    target: int,
    *,
    base: int = 64,
    tol: float = 0.01,
    limit: int = 4096,
) -> int:
    """Smallest hidden width whose capacity is ``>= target`` and within ``tol``.

    **Decision 11 has two clauses and only one of them is symmetric.** "Within
    1%" is a tolerance; "the control is never smaller" is a direction. A control
    with fewer parameters than the proposed method turns any win into "L7 had
    more capacity", which is the objection capacity matching exists to remove —
    so the search rounds *up* and then checks the tolerance, rather than picking
    whichever width is nearest.

    Needed because arms differ by whole networks, not by a scale factor: L7 and
    L6 match exactly (identical shapes; only the ``Δd`` block's *contents*
    differ), but GAFlowNet carries no ``PotentialHead`` and is ~37% smaller at
    equal width. Widening its trunk is the only knob that closes that without
    adding parameters no gradient ever reaches — which would be a match on paper
    and dead capacity in fact.
    """
    if count(base) >= target:
        width = base
    else:
        width = base
        while width < limit and count(width) < target:
            width += 1
    achieved = count(width)
    if achieved < target:
        raise ValueError(
            f"no hidden width <= {limit} reaches {target} parameters "
            f"(best {achieved} at width {width})"
        )
    if achieved > target * (1.0 + tol):
        raise ValueError(
            f"capacity granularity too coarse: width {width} gives {achieved} "
            f"against a target of {target}, which is "
            f"{100.0 * (achieved / target - 1.0):.2f}% over a {100 * tol:.0f}% "
            "tolerance. Decision 11 is not satisfiable by width alone here."
        )
    return width
