"""Budget metering and enforcement.

The Stage-D primary metric and the entire search comparison are *defined* in
this module's units, so it cannot be an afterthought.  Two behaviours matter
more than the counting itself:

**It enforces, it does not merely observe.**  ``count`` raises when a capped
meter would go over, and ``would_exceed`` lets a search module stop cleanly
first.  A ledger that only counts lets methods drift over budget by accident,
and the comparison becomes silently unfair without anyone noticing.

**Capped meters require an active query scope.**  The metric is *per query*; a
global counter cannot produce it.  Spending a terminal check outside a query
scope is a bug and raises rather than being absorbed into a total.

Two meters, not one (Phase-0 gap G1):

``terminal_checks``
    Full validation of a *completed* candidate set.  The primary budget axis.
    This is the "expensive true evaluator" in the Robust Scheduling framing, and
    the budget has to meter that evaluator specifically.
``incremental_ops``
    Cached validity updates during construction, maintained incrementally and
    therefore free.  Logged and reported, never capped — charging these would
    make constructive methods artificially expensive against one-shot methods
    like PCST, and wall-clock and FLOPs are reported separately anyway.

``terminal_checks`` must be incremented **inside** the checker (Phase 1), never
by its callers.  Caller-side counting always drifts.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Mapping, Protocol

__all__ = ["Ledger", "BudgetExceeded", "LedgerError", "METERS", "CHECKPOINT_OP"]

METERS: tuple[str, ...] = (
    "terminal_checks",
    "incremental_ops",
    "model_forwards",
    "llm_calls",
    "llm_tokens_in",
    "llm_tokens_out",
    "wall_clock_ms",
)

CHECKPOINT_OP = "ledger.checkpoint"


class LedgerError(RuntimeError):
    """Misuse of the ledger — an unknown meter, or a capped spend out of scope."""


class BudgetExceeded(RuntimeError):
    """A capped meter would have gone over budget."""


class _Appendable(Protocol):
    def append(self, op: str, payload: Mapping[str, Any]) -> int: ...


class Ledger:
    """Meters and enforces per-query budgets, single-threaded.

    One process, one GPU (architecture §0.4), so there is no locking here.
    """

    def __init__(
        self,
        caps: Mapping[str, int] | None = None,
        log: _Appendable | None = None,
    ) -> None:
        unknown = sorted(set(caps or {}) - set(METERS))
        if unknown:
            raise LedgerError(f"unknown meters in caps: {', '.join(unknown)}")
        self._caps: dict[str, int] = dict(caps or {})
        self._log = log
        self._totals: dict[str, int] = {m: 0 for m in METERS}
        self._stages: dict[str, dict[str, int]] = {}
        self._query: dict[str, int] | None = None
        self._query_id: str | None = None
        self._stage: str | None = None
        self._queries_seen: int = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Any, log: _Appendable | None = None) -> "Ledger":
        """Build a ledger whose caps come from the frozen config."""
        return cls(caps={"terminal_checks": cfg.checker_budget}, log=log)

    # -- spending ----------------------------------------------------------

    def count(self, meter: str, n: int = 1) -> None:
        """Record ``n`` units on ``meter``, refusing to exceed a cap."""
        if meter not in self._totals:
            raise LedgerError(f"unknown meter {meter!r}; known: {', '.join(METERS)}")
        if n < 0:
            raise LedgerError(f"cannot spend a negative amount ({n}) on {meter}")

        if meter in self._caps:
            if self._query is None:
                raise LedgerError(
                    f"{meter} is capped per query and was spent outside a "
                    "query_scope(); the per-query metric cannot be reconstructed "
                    "from a global counter"
                )
            used = self._query[meter]
            cap = self._caps[meter]
            if used + n > cap:
                raise BudgetExceeded(
                    f"{meter} budget exhausted: {used} used, {n} requested, cap {cap}. "
                    "Call would_exceed() before spending so the method can stop cleanly."
                )

        self._totals[meter] += n
        if self._query is not None:
            self._query[meter] += n
        if self._stage is not None:
            self._stages[self._stage][meter] += n

    def would_exceed(self, meter: str, n: int = 1) -> bool:
        """True when spending ``n`` more would break a cap.  Never raises on caps."""
        if meter not in self._totals:
            raise LedgerError(f"unknown meter {meter!r}")
        if meter not in self._caps:
            return False
        if self._query is None:
            raise LedgerError(f"{meter} is capped per query; open a query_scope() first")
        return self._query[meter] + n > self._caps[meter]

    def remaining(self, meter: str) -> int | None:
        """Units left on a capped meter; ``None`` when the meter is uncapped."""
        if meter not in self._totals:
            raise LedgerError(f"unknown meter {meter!r}")
        if meter not in self._caps:
            return None
        if self._query is None:
            return self._caps[meter]
        return self._caps[meter] - self._query[meter]

    # -- scoping -----------------------------------------------------------

    @contextmanager
    def query_scope(self, query_id: str | None = None) -> Iterator[dict[str, int]]:
        """Reset per-query counters for the duration of one query."""
        if self._query is not None:
            raise LedgerError("query scopes do not nest")
        self._query = {m: 0 for m in METERS}
        self._query_id = query_id
        try:
            yield self._query
        finally:
            self._queries_seen += 1
            self._query = None
            self._query_id = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Attribute everything spent inside to a named stage, and time it."""
        if self._stage is not None:
            raise LedgerError(f"stage {self._stage!r} is already open; stages do not nest")
        self._stages.setdefault(name, {m: 0 for m in METERS})
        self._stage = name
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int(round((time.perf_counter() - started) * 1000))
            self._stage = None
            # Recorded directly rather than through count(), because the stage
            # is already closed and wall clock is never capped.
            self._totals["wall_clock_ms"] += elapsed_ms
            self._stages[name]["wall_clock_ms"] += elapsed_ms
            if self._query is not None:
                self._query["wall_clock_ms"] += elapsed_ms

    # -- reporting and durability -----------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A plain-dict view of everything counted so far."""
        return {
            "totals": dict(self._totals),
            "stages": {k: dict(v) for k, v in sorted(self._stages.items())},
            "caps": dict(self._caps),
            "queries_seen": self._queries_seen,
            "query": None if self._query is None else dict(self._query),
            "query_id": self._query_id,
        }

    def checkpoint(self) -> int:
        """Append cumulative totals to the event log.

        Durability comes from the log rather than from a second file: counters
        survive a crash because they are in the same append-only stream as
        everything else, and ``restore`` replays it.
        """
        if self._log is None:
            raise LedgerError("no event log attached; construct with Ledger(log=...)")
        return self._log.append(
            CHECKPOINT_OP,
            {
                "totals": dict(self._totals),
                "stages": {k: dict(v) for k, v in sorted(self._stages.items())},
                "queries_seen": self._queries_seen,
            },
        )

    @classmethod
    def restore(
        cls,
        log: Any,
        caps: Mapping[str, int] | None = None,
        upto: int | None = None,
    ) -> "Ledger":
        """Rebuild a ledger from the last checkpoint in ``log``."""
        ledger = cls(caps=caps, log=log)
        last: Mapping[str, Any] | None = None
        for event in log.replay(upto=upto):
            if event.op == CHECKPOINT_OP:
                last = event.payload
        if last is not None:
            ledger._totals.update({m: int(last["totals"].get(m, 0)) for m in METERS})
            ledger._stages = {
                name: {m: int(counts.get(m, 0)) for m in METERS}
                for name, counts in last.get("stages", {}).items()
            }
            ledger._queries_seen = int(last.get("queries_seen", 0))
        return ledger
