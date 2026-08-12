"""L6 — LED-GFN, in its **LED-DB** form (decision 23b).

**[EVIDENCE]** *Learning Energy Decomposition for Partial Inference in
GFlowNets*. Two objectives, trained together, and it is worth being precise
about which is which because R4 got it wrong once already:

**1. The decomposition** (its Eq. 3 and Eq. 5). A learned potential
``φ_θ(s→s′)`` is required to sum along a trajectory to that trajectory's terminal
energy::

    Σ_{t} φ_θ(s_t → s_{t+1})  =  ℰ(x)  =  −log R(x)

trained by one squared loss with **dropout** as the variance regulariser. There
are **no** "two regulariser coefficients" — that was an assumed parameterisation
the paper does not use. Appendix C's values (decision 23a): potential learning
rate 0.001, dropout 0.10, ``N = 8`` decomposition iterations per round,
``B2 = B1``.

**No replay buffer, and that is the paper's own choice for this task shape.**
Algorithm 1 draws the potential's mini-batches from a buffer, but Appendix C
states per task which experiments use one: bag and RNA generation reuse a buffer
their base implementation already had, while **molecule generation uses the
round's own samples**, and **set generation inherits molecule generation's
decomposition settings**. Set generation is the closest published task to this
one, so the trainer's reuse of the current batch is Appendix C's configuration
rather than a departure from it (``trainer._train_potential``).

**2. The GFlowNet** (its Eq. 4), DB-style over ``log F̃(s)``, with the
decomposed potential supplying per-transition credit::

    ( log F̃(s) + log P_F(s′|s) − log F̃(s′) − log P_B(s|s′) + φ_θ(s→s′) )²

**Where the reward went.** Telescoping that residual over a whole trajectory
gives ``log F̃(s_0) + Σ log P_F − log F̃(x) − Σ log P_B + ℰ(x) = 0``, and matching
it against trajectory balance forces ``log F̃(x) = 0`` at **every** terminal —
valid stop or dead end alike. So LED's boundary is ``0``, not ``log R(x)``: the
reward is carried entirely by ``Σ φ``. That is the single line where LED differs
structurally from L4 and L5, and it is why ``Batch.g`` takes its terminal
boundary as an argument.

**Which of the paper's two variants this is** (decision 23b). Appendix B.1
describes a correction term added to the terminal flow, names that variant
``LED-GFN*``, and then states that the implementation behind every reported
experiment instead follows Ren et al. (2022) and **uniformly redistributes the
decomposition error over a trajectory's transitions** — that is what "LED-GFN"
denotes in its Figures 4, 5 and 11, and Figure 11 plots the two against each
other. The paper's choice is therefore between *redistribution* and
*correction term*; a form with neither is a third thing it never runs. An
earlier revision of decision 23b read the alternatives as "plain versus
correction term" and picked plain, which would have made the mandatory baseline
of plan §5.1 a variant with no published behaviour. ``redistribute`` below is
the paper's form; the correction-term variant stays out, as 23b intends.

**The terminating transition is decomposed too.** ``STOP`` is an action in this
MDP, so ``P_F(STOP | x)`` appears in the balance conditions and needs its own
``φ`` slot, or the identity above does not close. A dead end terminates with
probability 1 and gets the same slot with ``log P_F = 0``.

**Terminal consistency is measurable here and is a hard constraint** (G10,
decision 13). ``ℰ(x)`` is known exactly for every terminal of the lattice, so
``|Σ φ − ℰ|`` is not a proxy — it is the quantity itself. Plan §4.5.4 requires
that L7's TV win not be bought by relaxing this regulariser, which is why
``consistency_error`` lives here rather than in L7's file: the control and the
proposed method are held to one implementation of one definition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.trainer import Batch, Trainer

__all__ = ["decomposition_loss", "led_db_loss", "redistribute", "consistency_error"]


def decomposition_loss(
    potential: torch.Tensor,
    batch: "Batch",
    dropout: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """LED's **Eq. 5**, verbatim — the potential's own loss.

    .. code-block:: text

        ℓ_LS(τ) = E_{z~Bern(λ)} [ ( ℰ(x)/T − (Σ_t z_t·φ_θ(s_t→s_{t+1})) / C )² ]
        C = Σ_t z_t,   T = the trajectory's transition count

    **Both normalisers are the paper's, and neither is cosmetic.** An earlier
    build wrote ``(Σ z·φ/(1−p) − ℰ)²`` — dropout rescaled by the *expected* keep
    rate against the *unnormalised* energy. That differs from Eq. 5 twice over:

    ``1/C`` versus ``1/(1−p)``
        the paper divides by the **realised** kept count, so each draw is
        self-normalising; a constant factor is only right in expectation, and
        ``1/C`` is not a linear function of ``C``.

    ``ℰ(x)/T``
        scales each trajectory's residual by ``1/T``, so the loss weights a
        short trajectory more heavily than a long one. Trajectories here run 2
        to 9 transitions, a spread of roughly 20× in that weight — the paper's
        own set-generation benchmark is fixed-length and never sees it, which is
        precisely why the difference had to be read off the equation rather than
        inferred from the task being similar.

    ``rng`` is the trainer's, not a fresh one: the run's seed has to reach every
    stochastic component or two "identical" runs differ by their dropout masks.
    """
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    valid = batch.valid.to(potential.dtype)
    length = valid.sum(dim=1)                                 # T
    z = valid
    if dropout > 0.0:
        keep = torch.as_tensor(
            rng.random(tuple(potential.shape)) >= dropout,
            dtype=potential.dtype,
            device=potential.device,
        )
        z = valid * keep
    kept = z.sum(dim=1)                                       # C

    scored = kept > 0
    if not bool(scored.any()):  # pragma: no cover - probability p^T per trajectory
        # Every transition of every trajectory dropped. 1/C is undefined and a
        # NaN here would propagate into the potential's weights on the next step.
        return (potential * 0.0).sum()

    residual = (
        batch.energy() / length.clamp(min=1.0)
        - (potential * z).sum(dim=1) / kept.clamp(min=1.0)
    )
    return (residual[scored] ** 2).mean()


def redistribute(potential: torch.Tensor, batch: "Batch") -> torch.Tensor:
    """Spread ``ℰ(x) − Σφ`` uniformly over a trajectory's transitions (App. B.1).

    .. code-block:: text

        φ̃(s_t→s_{t+1}) = φ_θ(s_t→s_{t+1}) + ( ℰ(x) − Σ_k φ_θ(s_k→s_{k+1}) ) / T

    This is the LED-GFN of the paper's experiments, not an addition to it: B.1
    presents exactly two ways of keeping the optimal policy under an inaccurate
    ``φ_θ`` — a correction term on the terminal flow (``LED-GFN*``) or this
    uniform redistribution — and states the reported results use the latter.

    **What it buys, in this module's own terms.** The telescoping argument in the
    header needs ``Σ φ = ℰ(x)``; the raw potential satisfies that only as well as
    Eq. 5 has trained it, whereas ``Σ φ̃ = ℰ(x)`` holds **exactly, per
    trajectory, by construction**. So LED's boundary of ``0`` is now an identity
    rather than an approximation, which is the property B.1 is about.

    Applied inside :func:`led_db_loss` and nowhere else — deliberately. The
    potential arrives detached (the trainer detaches it so the GFlowNet loss
    cannot reshape its own reward), and the adjustment is a per-trajectory
    constant, so this adds no gradient path. Padding is excluded by ``valid``,
    so a padded slot keeps whatever it held and stays masked out downstream.
    """
    valid = batch.valid.to(potential.dtype)
    length = valid.sum(dim=1).clamp(min=1.0)
    error = batch.energy() - (potential * valid).sum(dim=1)
    return potential + (error / length).unsqueeze(1) * valid


def led_db_loss(batch: "Batch", trainer: "Trainer", spent: int) -> torch.Tensor:
    """LED's Eq. 4, per transition, with ``log F̃`` pinned to ``0`` at terminals."""
    if batch.potential is None:  # pragma: no cover - defensive
        raise ValueError("LED-DB needs a potential; the arm was built without one")

    terminal = torch.zeros_like(batch.log_reward)
    g = batch.g(terminal)                                # [n, L+1]
    phi = redistribute(batch.potential, batch)           # Appendix B.1
    residual = g[:, :-1] - g[:, 1:] + phi                # [n, L]

    mask = batch.valid.to(residual.dtype)
    # Mean over **real** transitions, not over the padded rectangle: padding
    # varies with the batch's longest rollout, so a rectangle mean would make the
    # loss depend on which trajectories happened to be sampled together.
    return (residual**2 * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def consistency_error(
    potential: torch.Tensor, batch: "Batch"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(normalised |Σφ − ℰ| per trajectory, is_fail)`` — decision 13's metric.

    **Measured on the raw ``φ_θ``, never on the redistributed ``φ̃``**, and the
    distinction is what keeps decision 13 meaningful. ``redistribute`` forces
    ``Σ φ̃ = ℰ(x)`` by construction, so this quantity computed on ``φ̃`` would be
    exactly 0 for every arm on every trajectory — a band that always passes and
    measures nothing. What plan §4.5.4 actually requires is that L7's TV win not
    be bought by relaxing LED's terminal-consistency regulariser, and the
    regulariser in question is **Eq. 5**, which trains the raw potential. So the
    thing to measure is how well the raw potential decomposes the energy, which
    is what callers pass in.

    Normalised by the **per-instance valid-terminal ``log R`` range**, which is
    the correction R3 forced: the per-terminal ``|log R|`` that preceded it
    permitted 0.0018 of absolute error at one terminal of an instance and 0.3927
    at another — a 216× swing inside a single environment, from a rule that reads
    as a single tolerance.

    The ``FAIL`` flag comes back with the errors rather than being folded in,
    because ``ℰ(FAIL) = −log r_fail = 13.8`` sits far outside the valid range and
    pooling it would let one trajectory dominate a p95 over thousands.
    """
    predicted = (potential * batch.valid.to(potential.dtype)).sum(dim=1)
    error = (predicted - batch.energy()).abs()
    if batch.log_r_range < 0.1:
        # Decision 13's exclusion guard: dividing by a near-zero range turns a
        # tolerance into noise amplification, and an instance whose valid
        # terminals all carry the same reward has no scale to normalise by.
        return torch.full_like(error, float("nan")), batch.is_fail
    return error / batch.log_r_range, batch.is_fail
