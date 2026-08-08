"""Budget metering and enforcement.

Discharges Phase-0 exit criteria 5 (enforcement halts at exactly the cap),
6 (per-query scoping) and 7 (counters survive a crash-resume).
"""

from __future__ import annotations

import pytest

from graft.config import load_config
from graft.eventlog import EventLog
from graft.ledger import METERS, BudgetExceeded, Ledger, LedgerError


def test_enforcement_halts_at_exactly_the_budget():
    """Criterion 5: a loop that asks first stops at 32, never at 33."""
    cfg = load_config()
    ledger = Ledger.from_config(cfg)
    spent = 0
    with ledger.query_scope("q"):
        while not ledger.would_exceed("terminal_checks"):
            ledger.count("terminal_checks")
            spent += 1
        assert ledger.remaining("terminal_checks") == 0
    assert spent == cfg.checker_budget == 32


def test_spending_past_the_cap_raises_rather_than_drifting():
    ledger = Ledger(caps={"terminal_checks": 2})
    with ledger.query_scope():
        ledger.count("terminal_checks", 2)
        with pytest.raises(BudgetExceeded, match="budget exhausted"):
            ledger.count("terminal_checks")


def test_a_bulk_spend_that_would_overshoot_is_refused_entirely():
    ledger = Ledger(caps={"terminal_checks": 32})
    with ledger.query_scope():
        ledger.count("terminal_checks", 30)
        with pytest.raises(BudgetExceeded):
            ledger.count("terminal_checks", 3)
        assert ledger.remaining("terminal_checks") == 2


def test_per_query_counters_are_independent():
    """Criterion 6: the metric is per query, so a global counter cannot produce it."""
    ledger = Ledger.from_config(load_config())
    with ledger.query_scope("q1"):
        ledger.count("terminal_checks", 5)
        assert ledger.remaining("terminal_checks") == 27
    with ledger.query_scope("q2") as scope:
        assert scope["terminal_checks"] == 0
        ledger.count("terminal_checks", 1)
        assert ledger.remaining("terminal_checks") == 31
    assert ledger.snapshot()["totals"]["terminal_checks"] == 6
    assert ledger.snapshot()["queries_seen"] == 2


def test_capped_meters_cannot_be_spent_outside_a_query_scope():
    ledger = Ledger.from_config(load_config())
    with pytest.raises(LedgerError, match="outside a query_scope"):
        ledger.count("terminal_checks")


def test_uncapped_meters_need_no_scope():
    """Construction-time validity is maintained incrementally and is free; charging
    it would make constructive methods artificially expensive against PCST."""
    ledger = Ledger.from_config(load_config())
    ledger.count("incremental_ops", 500)
    assert ledger.remaining("incremental_ops") is None
    assert ledger.would_exceed("incremental_ops", 10_000) is False


def test_query_scopes_do_not_nest():
    ledger = Ledger()
    with ledger.query_scope():
        with pytest.raises(LedgerError, match="do not nest"):
            with ledger.query_scope():
                pass


def test_unknown_meters_are_rejected_everywhere():
    with pytest.raises(LedgerError, match="unknown meters in caps"):
        Ledger(caps={"made_up": 1})
    ledger = Ledger()
    with pytest.raises(LedgerError, match="unknown meter"):
        ledger.count("made_up")
    with pytest.raises(LedgerError, match="unknown meter"):
        ledger.would_exceed("made_up")


def test_negative_spends_are_rejected():
    ledger = Ledger()
    with pytest.raises(LedgerError, match="negative"):
        ledger.count("model_forwards", -1)


def test_stages_attribute_spend_and_record_wall_clock():
    ledger = Ledger()
    with ledger.stage("retrieve"):
        ledger.count("model_forwards", 3)
    with ledger.stage("read"):
        ledger.count("model_forwards", 2)
    snapshot = ledger.snapshot()
    assert snapshot["stages"]["retrieve"]["model_forwards"] == 3
    assert snapshot["stages"]["read"]["model_forwards"] == 2
    assert snapshot["totals"]["model_forwards"] == 5
    assert snapshot["totals"]["wall_clock_ms"] >= 0


def test_stages_do_not_nest():
    ledger = Ledger()
    with ledger.stage("a"):
        with pytest.raises(LedgerError, match="do not nest"):
            with ledger.stage("b"):
                pass


def test_snapshot_reports_every_declared_meter():
    snapshot = Ledger().snapshot()
    assert set(snapshot["totals"]) == set(METERS)


def test_counters_survive_a_crash_resume(tmp_path):
    """Criterion 7.  Durability comes from the append-only log rather than a
    second file: the counters are in the same stream as everything else."""
    path = tmp_path / "events.jsonl"

    log = EventLog.open(path)
    ledger = Ledger.from_config(load_config(), log=log)
    with ledger.query_scope("q1"):
        ledger.count("terminal_checks", 7)
    with ledger.stage("stage-a"):
        ledger.count("llm_calls", 2)
    ledger.checkpoint()
    before = ledger.snapshot()
    log.close()  # stand-in for the crash

    reopened = EventLog.open(path)
    restored = Ledger.restore(reopened, caps={"terminal_checks": 32})
    after = restored.snapshot()
    reopened.close()

    assert after["totals"] == before["totals"]
    assert after["stages"] == before["stages"]
    assert after["queries_seen"] == before["queries_seen"] == 1


def test_restore_without_a_checkpoint_starts_clean(tmp_path):
    log = EventLog.open(tmp_path / "events.jsonl")
    log.append("node.add", {"node_id": "n", "ntype": "Entity", "payload": {}})
    ledger = Ledger.restore(log, caps={"terminal_checks": 32})
    log.close()
    assert ledger.snapshot()["totals"]["terminal_checks"] == 0


def test_checkpoint_requires_a_log():
    with pytest.raises(LedgerError, match="no event log attached"):
        Ledger().checkpoint()
