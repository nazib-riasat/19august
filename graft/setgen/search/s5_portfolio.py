"""S5 — the GFlowNet portfolio: sample ``K``, ``H``-filter, rank by the scorer.

**[EVIDENCE]** the sample-then-filter pattern is Robust Scheduling's (ICLR
2023) result: diverse candidates sampled under a **cheap proxy** beat
proxy-optimization when the **true evaluator is expensive**. `CLAUDE.md` §4.2
calls it *"the single best argument for using a flow method at all"*.

**And this environment removes that argument's precondition, which is why
Phase 4 cannot decide Gate 3** (G9, decision 5). Fix F13 sets
``scorer = exact U = the evaluation metric``, so there is no proxy/true-evaluator
gap at all: greedy on the metric attains the global optimum on 30/30 instances
and a *flawless* sampler is 0.038 short at ``K = 8``. Phase 4 keeps the pattern
and deletes its precondition; Phase 9, whose scorer is the noisy distilled head,
is where the comparison becomes live.

**Portfolio composition is fix F5's: 1 greedy + 7 sampled** — one constant used
in Phase 3's ``best_of_k`` and here, so S5 is compared against the same object
Phase 9 ships.

**Spend: 0 terminal checks.** Every rollout goes through the ``ADD`` masks, and
``stop_allowed`` *is* ``H``, so a completed trajectory is valid by construction
and a dead-ended one is recognisable without a checker call (G6).

**No trainer import** — Phase-3 §8 requirement 1 makes the checkpoint format
"loadable without the trainer", and this module is what proves it (exit
criterion 4): it reaches for ``graft.setgen.policy.load_policy`` alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from graft.ledger import Ledger
from graft.schemas import Obligations, ProofSet
from graft.setgen.search.base import SearchResult

__all__ = ["PortfolioSearch"]

#: One frozen stream, so every seed's portfolio is read under the same draw and
#: differences between arms are differences of policy — the same reason Phase 3
#: freezes ``BEST_OF_K_SEED``.
PORTFOLIO_SEED = 20260818


class PortfolioSearch:
    """S5.  **Not** deterministic: its seed is the trained model *and* the draw.

    Decision 4: S1–S4 are deterministic and reported once; S5 runs the frozen
    ``{13, 42, 7}``, and the paired test broadcasts the deterministic arms to
    match its shape.
    """

    name = "s5_portfolio"
    deterministic = False

    def __init__(
        self,
        checkpoint: str | Path,
        k: int | None = None,
        seed: int = PORTFOLIO_SEED,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.k = k
        self.seed = int(seed)
        self._policy = None
        self._blob: dict[str, Any] = {}

    def _load(self) -> tuple[Any, dict[str, Any]]:
        if self._policy is None:
            from graft.setgen.policy import load_policy  # no trainer import

            self._policy, self._blob = load_policy(self.checkpoint)
        return self._policy, self._blob

    def run(
        self,
        env: Any,
        obligations: Obligations,
        scorer: Callable[[Iterable[str]], float],
        ledger: Ledger | None = None,
    ) -> SearchResult:
        from graft.setgen.features import SyntheticFeaturizer
        from graft.setgen.rollout import sample_trajectories

        policy, blob = self._load()
        inst = env.instance
        cfg = inst.cfg
        k = int(self.k if self.k is not None else cfg.K)

        # ``delta_d`` must come from the checkpoint, never be assumed: an L7
        # policy behind an L6 featurisation reads a zeroed Δd block, which is a
        # silently wrong policy rather than an error (Phase-3 §5).
        featurizer = SyntheticFeaturizer(
            inst, env.graph, policy, cfg, delta_d=bool(blob.get("delta_d", False))
        )
        traj = sample_trajectories(
            featurizer, env.graph, k, np.random.default_rng(self.seed), 0.0, greedy=1
        )

        pool = inst.pool
        sets: dict[tuple[str, ...], ProofSet] = {}
        drawn: list[ProofSet] = []
        fails = 0
        for terminal, failed in zip(traj.terminal.tolist(), traj.is_fail.tolist()):
            if failed:
                fails += 1
                continue
            atoms = env.graph.atoms_of(int(terminal))
            if atoms not in sets:
                sets[atoms] = ProofSet(
                    atoms=frozenset(atoms), bindings=pool.derive_bindings(atoms)
                )
            drawn.append(sets[atoms])
        scored = sorted(((scorer(a), a) for a in sets), key=lambda t: (-t[0], t[1]))
        return SearchResult(
            self.name,
            [sets[a] for _, a in scored],
            [s for s, _ in scored],
            attempted=k,
            distinct_attempted=len(sets),
            terminal_checks=0,
            # **The draw, with multiplicity** — this is the arm decision 5's one
            # live condition is about, and a mode-collapsed sampler that draws
            # one terminal eight times must score 0, not 1.
            portfolio=drawn,
            extra={
                "arm": blob.get("arm"),
                "delta_d": bool(blob.get("delta_d", False)),
                "dead_ended": fails,
                "checkpoint": str(self.checkpoint),
                "seed": self.seed,
            },
        )
