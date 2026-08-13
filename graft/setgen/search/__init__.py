"""Phase 4 — the five Tier-1 search algorithms (`GRAFT_PHASE4_BUILD.md`).

Gate 3's synthetic stage, run early on purpose: `CLAUDE.md` §8 names submodular
greedy and PCST as the two baselines most likely to embarrass the project and
says *run them early*.

**What this phase may and may not conclude** (decision 5, G5/G9). Under fix
F13's perfect scorer this environment cannot host the architecture's Gate-3
comparison: greedy on exact ``U`` is globally optimal on 30/30 instances while a
*flawless* sampler reaches only 1.8865 of greedy's 1.9245 at ``K = 8``. So
best-of-K here is arithmetic, reported **against the ``p*`` ceiling** and never
against a rival as a verdict — in **either** direction. The gate's decision
moves to Phase 9, where the distilled head is noisy and Robust Scheduling's
precondition actually holds. One necessary condition stays live here: if S5's
portfolio diversity does not exceed the training-free arms', the flow method is
not producing a portfolio at all.

**Search modules are not learners.** Phase-3's fix-F6 boundary keeps
``setgen/learners/`` away from ``StateGraph``, ``LatticeInstance`` and atom ids;
S3 and S4 *must* see the pool and its ``refs`` to build sets directly, and fix
F10 makes that legal precisely because ``H`` re-checks closure afterwards. This
package therefore sits on the adapter side of Phase-3's ``_ADAPTER_LAYER``, and
that list is closed, so being here is a recorded act rather than a drift.
"""

from __future__ import annotations

from graft.setgen.search.base import (
    SearchModule,
    SearchResult,
    admissible_atoms,
    build_checker,
    close_under_refs,
    dedup_sets,
    h_filter,
)
from graft.setgen.search.relevance import (
    RELEVANCE_VARIANTS,
    informed_relevance,
    obligation_relevance,
    relevance_vector,
)
from graft.setgen.search.s1_greedy import GreedySearch
from graft.setgen.search.s2_beam import BeamSearch
from graft.setgen.search.s3_submodular import SubmodularGreedy
from graft.setgen.search.s4_pcst import PCSTSearch
from graft.setgen.search.s5_portfolio import PortfolioSearch

#: Every Tier-1 search method, in the architecture's order.  ``SearchModule`` is
#: ``runtime_checkable``, so registry membership is a test rather than a promise.
SEARCH_METHODS = (
    GreedySearch,
    BeamSearch,
    SubmodularGreedy,
    PCSTSearch,
    PortfolioSearch,
)

#: The two families the budget separates (G6): mask-driven methods reach
#: ``stop_allowed`` — which *is* ``H`` — so they are valid by construction and
#: spend **0**; direct builders bypass the masks and pay **1 per distinct
#: candidate**.  A table charging both the same is the accounting fiction G6
#: retires.
MASK_DRIVEN = ("s1_greedy", "s2_beam", "s5_portfolio")
DIRECT_BUILDERS = ("s3_submodular", "s4_pcst")

__all__ = [
    "SearchModule",
    "SearchResult",
    "admissible_atoms",
    "build_checker",
    "close_under_refs",
    "dedup_sets",
    "h_filter",
    "RELEVANCE_VARIANTS",
    "obligation_relevance",
    "informed_relevance",
    "relevance_vector",
    "GreedySearch",
    "BeamSearch",
    "SubmodularGreedy",
    "PCSTSearch",
    "PortfolioSearch",
    "SEARCH_METHODS",
    "MASK_DRIVEN",
    "DIRECT_BUILDERS",
]
