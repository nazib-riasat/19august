"""P9.5 — the distilled utility head (G8, fix F13 inverted).

**Why this module is the experimental condition, not a convenience.**

Phase 4 set ``scorer = exact U`` and then measured what that costs: greedy
attains the global optimum on **30/30** lattice instances, and a *flawless*
sampler reaches 1.8865 against greedy's 1.9245 at ``K = 8``.  With a perfect
scorer, best-of-K is arithmetic rather than learning, so the synthetic stage
answered Gate 3 by construction and the decision moved here
(`PHASE4_DECISIONS.md`, `CLAUDE.md` §8).

**[EVIDENCE]** Robust Scheduling with GFlowNets (ICLR 2023) — which `CLAUDE.md`
§4.2 calls "the single best argument for using a flow method at all" — requires
a **cheap, imperfect proxy** with an **expensive true evaluator**.  Fix F13
deleted that precondition on the lattice.  This head restores it: at inference
there is no gold, so ``sufficiency`` is unavailable and ranking runs on a
regression of ``U`` instead.  Gate 3 is asked under *this* head's noise, which is
why :func:`spearman` ships beside every Gate-3 row — **a row without its ρ cannot
be read**, because "the learned sampler won" and "the scorer was nearly perfect"
are different findings and the number is what separates them.

**Two structural guarantees, asserted rather than intended.**

``H`` never sees this head
    ``graft.core.checker`` and ``graft.core.masks`` import nothing from
    ``graft.setgen``.  If a learned score could enter ``H``, then ``H`` stops
    being a predicate, ``1[H]`` stops being a hard gate, and the multiplicative
    safety property degrades into a soft threshold — `CLAUDE.md` §4.2's
    non-negotiable.  ``test_structure.py`` asserts the import graph.

no gold reaches an input
    ``U`` values are the **target**; the features are the same pooled atom
    vectors the policy sees.  A head that could read ``gold_atom_ids`` would
    score the training distribution rather than predict it, and its ρ would be
    meaningless in exactly the direction that flatters the method.

**Mean ⊕ max pooling** because a set has no order (plan §3.4's canonical-state
rule).  A sequence model here would impose one and would score the same set
differently depending on how it was built — which is the property the whole
canonical-state design exists to prevent.

**This component may early-stop.**  It is not one of the compared arms: fix
F12's fixed-budget discipline binds the nine learners, whose comparison would
break if one stopped early, and says nothing about a regression head fitted once
and reported with its fidelity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from graft.setgen.pins import DISTILL

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.atomfeat import RealFeaturizer

__all__ = ["UtilityHead", "build_head", "train_head", "spearman", "HeadScorer"]


class UtilityHead(nn.Module):
    """``[n_atoms features] → Û(X)``, permutation-invariant by construction.

    Pooling is mean **and** max concatenated: the mean carries what the set is
    on average, the max carries whether it contains a strong atom at all, and a
    proof's utility depends on both — ``coverage`` is a mean-like property and
    ``sufficiency`` is closer to a max-like one.
    """

    def __init__(self, atom_dim: int, hidden: int | None = None, dropout: float | None = None) -> None:
        super().__init__()
        hidden = int(DISTILL["hidden"] if hidden is None else hidden)
        drop = float(DISTILL["dropout"] if dropout is None else dropout)
        self.net = nn.Sequential(
            nn.Linear(2 * atom_dim + 1, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


def pool_sets(
    feat: "RealFeaturizer", sets: Sequence[Sequence[str]], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``[m, 2·atom_dim + 1]`` — mean ⊕ max over selected atoms, plus size.

    The size scalar is supplied explicitly for the same reason the policy's
    ``state_repr`` supplies it: ``U``'s own ``size`` term is ``|X| / max_atoms``,
    so a head that had to infer set size from a mean-pooled vector would be
    fitting something it was handed for free.

    An empty set pools to zeros rather than raising: the head is asked to score
    candidate sets, and a caller comparing against the empty set should get a
    number rather than an exception.
    """
    atom_feat = feat.atom_feat
    width = int(atom_feat.shape[1])
    out = torch.zeros((len(sets), 2 * width + 1), dtype=dtype)
    for i, atoms in enumerate(sets):
        atoms = list(atoms)
        rows = [feat.index[a] for a in atoms if a in feat.index]
        # **An empty set legitimately pools to zeros; a non-empty set none of
        # whose atoms this featurizer knows is a wiring error** (found by
        # adversarial audit, 16 Aug 2026). Both used to `continue`, so a
        # `HeadScorer` built on example A and applied to example B scored every
        # candidate identically — collapsing best-of-K's ranking to the
        # tie-break alone (size ascending, then hash) with nothing raising and
        # every artefact reading as a completed portfolio run. `run_portfolio`
        # takes `feat` and `scorer` as independent arguments and cannot check
        # they agree, so the check belongs here.
        if atoms and not rows:
            raise KeyError(
                f"none of this set's {len(atoms)} atoms are in the featurizer's "
                "pool: the scorer was built on a different example, and every "
                "candidate would score identically"
            )
        if not rows:
            continue
        block = atom_feat[rows]
        out[i, :width] = block.mean(dim=0)
        out[i, width : 2 * width] = block.max(dim=0).values
        out[i, 2 * width] = float(len(rows)) / float(feat.cfg.max_atoms)
    return out


def build_head(atom_dim: int, *, seed: int, hidden: int | None = None) -> UtilityHead:
    """A head with its initialisation **seeded before construction**.

    P6.11's first guard: seeding after the modules exist leaves the
    initialisation drawn from whatever state the process happened to be in, and
    two "identical" runs then differ by their weights alone — reproducible in
    the log and not in the artefact.
    """
    torch.manual_seed(int(seed))
    head = UtilityHead(atom_dim, hidden=hidden)
    params = sum(p.numel() for p in head.parameters())
    cap = int(DISTILL["max_params"])
    if params > cap:
        raise ValueError(
            f"utility head has {params} parameters against the pinned cap {cap}. "
            "The head is a cheap proxy by design (G8); one that grew past the cap "
            "would stop being the noisy scorer Gate 3 is asked under."
        )
    return head


def train_head(
    head: UtilityHead,
    train: tuple[torch.Tensor, torch.Tensor],
    dev: tuple[torch.Tensor, torch.Tensor],
    *,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 5,
) -> dict[str, Any]:
    """Fit ``Û`` to exact ``U``, with P6.11's three guards.

    Returns the history plus the **restored** best epoch, so the caller cannot
    accidentally report a final-epoch head as the selected one.
    """
    x_tr, y_tr = train
    x_dev, y_dev = dev
    if x_dev.shape[0] == 0:
        # P6.11's third guard: a head with no scorable dev set has no basis for
        # selection, and silently keeping the last epoch would report an
        # unselected model as a selected one.
        raise ValueError(
            "no dev rows: the utility head is selected on dev and reported with a "
            "held-out rho, so an empty dev split makes both meaningless"
        )
    torch.manual_seed(int(seed))
    generator = torch.Generator().manual_seed(int(seed))
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    best = {"epoch": -1, "dev_loss": float("inf")}
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(int(epochs)):
        head.train()
        order = torch.randperm(x_tr.shape[0], generator=generator)
        total = 0.0
        for lo in range(0, x_tr.shape[0], batch_size):
            idx = order[lo : lo + batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(head(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        head.eval()
        with torch.no_grad():
            dev_loss = float(loss_fn(head(x_dev), y_dev))
        history.append(
            {"epoch": epoch, "train_loss": total / max(1, x_tr.shape[0]), "dev_loss": dev_loss}
        )
        if dev_loss < best["dev_loss"]:
            best = {"epoch": epoch, "dev_loss": dev_loss}
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        elif epoch - int(best["epoch"]) >= patience:
            break

    # P6.11's **third** guard, which this function was missing (found by
    # adversarial audit, 16 Aug 2026). If every epoch's dev loss is non-finite —
    # a diverged head — `best_state` is never set, the restore below is skipped,
    # and the function returns the **diverged weights** as though they had been
    # selected. Probed: `lr=1e6` returned `best_dev_loss=inf`, `best_epoch=-1`
    # and a plausible-looking `dev_spearman`, with non-finite parameters. That is
    # exactly Phase 6's "no scorable dev item returns the initialisation" defect
    # one phase later, and Stage D would have ranked best-of-K with it.
    if best_state is None:
        raise ValueError(
            f"no epoch produced a finite dev loss over {len(history)} epochs: the "
            "utility head diverged, so there is nothing to select and the returned "
            "weights would be the diverged ones. Lower the learning rate."
        )
    # P6.11's second guard: **restore** the argmin-dev weights. Early stopping
    # that leaves the last epoch's weights in place selects on dev and then
    # reports a different model than the one it selected.
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        rho = spearman(head(x_dev).numpy(), y_dev.numpy())
    return {
        "epochs_run": len(history),
        "best_epoch": int(best["epoch"]),
        "best_dev_loss": float(best["dev_loss"]),
        "dev_spearman": rho,
        "train_rows": int(x_tr.shape[0]),
        "dev_rows": int(x_dev.shape[0]),
        "history": history,
    }


def spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation between ``Û`` and exact ``U``.

    Spearman rather than Pearson because the head's job is to **rank** candidate
    sets: the portfolio takes an argmax, so a monotone distortion of the scale
    costs nothing and a rank inversion costs everything.

    Ties are averaged, which is the standard definition and matters here — a
    head that collapses to a constant would otherwise score well against an
    arbitrary tie-break rather than reading as the degenerate fit it is.  A
    constant predictor has zero rank variance and returns ``nan``, not 1.0.

    **Non-finite inputs return ``nan``, and that guard is not defensive
    boilerplate** (found by adversarial audit, 16 Aug 2026).  Without it
    ``spearman`` returned **1.0** for an all-``NaN`` prediction vector: ``NaN``
    compares unequal to itself, so the tie-detection below never fires, every
    ``NaN`` receives a distinct rank in ``argsort``'s stable order, and the
    result correlates perfectly with any monotone target.  A **diverged head
    would have reported perfect fidelity** — and ρ is the number that licenses
    reading a Gate-3 row at all (`pins.DISTILL["report_rho"]`: *"a row without
    its ρ cannot be read"*), so the one instrument that detects a broken scorer
    was the one guaranteed to look healthiest when it broke.
    """
    p = np.asarray(predicted, dtype=np.float64).ravel()
    a = np.asarray(actual, dtype=np.float64).ravel()
    if p.size != a.size:
        raise ValueError(f"{p.size} predictions against {a.size} targets")
    if p.size < 2:
        return float("nan")
    if not (np.isfinite(p).all() and np.isfinite(a).all()):
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        out = np.empty_like(v)
        sorted_v = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            out[order[i : j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return out

    rp, ra = ranks(p), ranks(a)
    sp, sa = rp.std(), ra.std()
    if sp == 0.0 or sa == 0.0:
        return float("nan")
    return float(((rp - rp.mean()) * (ra - ra.mean())).mean() / (sp * sa))


class HeadScorer:
    """``Callable[[Iterable[str]], float]`` — the shape Phase 4's search modules take.

    Wrapping rather than exposing the module directly, so that S1–S5 and the
    Stage-D portfolio consume **one** scorer object and a run cannot accidentally
    score two methods with two different things.  That is the parity Gate 3's
    primary depends on: the arms differ by their search, not by their scorer.
    """

    def __init__(self, head: UtilityHead, feat: "RealFeaturizer") -> None:
        self.head = head
        self.feat = feat
        self.head.eval()
        self.calls = 0

    def __call__(self, atoms: Iterable[str]) -> float:
        self.calls += 1
        with torch.no_grad():
            return float(self.head(pool_sets(self.feat, [tuple(atoms)]))[0])

    def score_many(self, sets: Sequence[Sequence[str]]) -> np.ndarray:
        self.calls += len(sets)
        with torch.no_grad():
            return self.head(pool_sets(self.feat, sets)).numpy()
