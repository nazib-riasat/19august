"""Terminal reward.

```
R(X | q, G) = 1[H(X, q, G, pool)] * exp(beta * U(X, q, G, pool, gold))
R(FAIL)     = r_fail
```

**The indicator is multiplicative, and that is the whole point.** An earlier
draft set ``U = 0`` on hard failure, which yields ``exp(0) = 1`` — a positive and
in a low-utility regime a *competitive* reward for a set that failed a hard
check. That algebra error inverted the intended safety property, and it is kept
here as a named regression test (v1.2 §1.1).

**``FAIL`` is a member of the target's support**, so
``Z = sum_valid R(X) + r_fail`` and ``p*`` is over ``{valid terminals} u {FAIL}``.
It is reached when construction can neither legally continue nor legally stop
(budget exhaustion is the common case, not the definition), never by an action —
there is no ``ABSTAIN`` action (v1.2 §3.4/§4.2). It licenses one claim — *no valid proof was found under this pool, policy,
attempt count and budget* — and not the stronger one that none exists, since
another action sequence from the same root may reach a valid terminal. That is
exactly the inference-time abstain fallback, so the write path and the read path
agree on what it means (architecture fix F3).

The safety property is therefore stated precisely as **no formally invalid proof
set is ever returned** — guaranteed by ``STOP``-masking — rather than as "invalid
outcomes have zero probability", which ``FAIL`` carrying positive mass would
contradict.
"""

from __future__ import annotations

import math
from typing import Iterable

from graft.config import Config
from graft.core.checker import H
from graft.core.utility import U
from graft.graphstore import GraphSnapshot
from graft.ledger import Ledger
from graft.schemas import AtomPool, CheckResult, Obligations, ProofSet

__all__ = ["reward", "log_reward", "fail_reward", "log_fail_reward", "reward_from_parts"]


def reward_from_parts(valid: bool, utility: float, cfg: Config) -> float:
    """``1[H] * exp(beta * U)`` given a decided validity and utility.

    Split out so the multiplicative form can be tested directly, without a pool
    or a snapshot, at any ``beta`` and any ``U`` — including ``U = 0``, which is
    where the original error hid.
    """
    if not valid:
        return 0.0
    return math.exp(cfg.beta * utility)


def log_reward_from_parts(valid: bool, utility: float, cfg: Config) -> float:
    """``beta * U`` when valid, ``-inf`` otherwise."""
    if not valid:
        return -math.inf
    return cfg.beta * utility


def reward(
    X: ProofSet | Iterable[str],
    q: Obligations,
    G: GraphSnapshot,
    pool: AtomPool,
    gold: ProofSet | Iterable[str] | None,
    cfg: Config,
    *,
    ledger: Ledger | None = None,
    check: CheckResult | None = None,
) -> float:
    """``R(X | q, G)``.

    ``check`` lets a caller that has already validated the set reuse the result
    rather than spend a second ``terminal_check`` on it — the portfolio path
    filters by ``H`` and then ranks, and paying twice would halve the effective
    budget.
    """
    result = check if check is not None else H(X, q, G, pool, cfg, ledger=ledger)
    if not result.ok:
        return 0.0
    return reward_from_parts(True, U(X, q, G, pool, gold, cfg), cfg)


def log_reward(
    X: ProofSet | Iterable[str],
    q: Obligations,
    G: GraphSnapshot,
    pool: AtomPool,
    gold: ProofSet | Iterable[str] | None,
    cfg: Config,
    *,
    ledger: Ledger | None = None,
    check: CheckResult | None = None,
) -> float:
    """``log R(X | q, G)`` — ``beta * U`` when valid, ``-inf`` otherwise.

    Every Phase-3 balance loss works in log space, and none of them may compute
    ``log(0)``.  Invalid sets are not reachable terminals (``STOP`` is masked),
    so ``-inf`` marks a set that should never have been scored; ``FAIL`` carries
    ``log(r_fail)`` explicitly instead.
    """
    result = check if check is not None else H(X, q, G, pool, cfg, ledger=ledger)
    if not result.ok:
        return -math.inf
    return log_reward_from_parts(True, U(X, q, G, pool, gold, cfg), cfg)


def fail_reward(cfg: Config) -> float:
    """``R(FAIL) = r_fail``."""
    return cfg.r_fail


def log_fail_reward(cfg: Config) -> float:
    """``log r_fail`` — finite by construction, since the config refuses ``r_fail <= 0``."""
    return math.log(cfg.r_fail)
