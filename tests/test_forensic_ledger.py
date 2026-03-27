"""Tests for the Forensic Ledger."""

import os
import tempfile
import pytest

from mnemos.audit.forensic_ledger import ForensicLedger


@pytest.fixture
def ledger(tmp_path):
    db_path = str(tmp_path / "test_audit.db")
    return ForensicLedger(db_path=db_path)


class TestForensicLedger:
    def test_log_transaction(self, ledger):
        tx_id = ledger.log_transaction(
            component="test",
            action="index",
            content="Indexed 5 documents",
            status="success",
            latency=0.05,
        )
        assert tx_id > 0

    def test_log_event(self, ledger):
        ev_id = ledger.log_event(
            event_type="startup",
            description="MNEMOS started",
            source="test",
        )
        assert ev_id > 0

    def test_get_recent_events(self, ledger):
        ledger.log_event("a", "event a")
        ledger.log_event("b", "event b")
        events = ledger.get_recent_events(limit=5)
        assert len(events) == 2
        assert events[0]["type"] == "b"  # most recent first

    def test_search_traces(self, ledger):
        ledger.log_transaction("retrieval", "search", "quantum physics papers")
        ledger.log_transaction("retrieval", "search", "machine learning papers")
        results = ledger.search_traces("quantum")
        assert len(results) >= 1
        assert "quantum" in results[0]["content"]

    def test_performance_summary(self, ledger):
        ledger.log_transaction("a", "op1", "x", latency=0.1)
        ledger.log_transaction("a", "op2", "y", latency=0.2)
        summary = ledger.get_performance_summary()
        assert summary["total_transactions"] == 2
        assert summary["success_rate"] == 1.0

    def test_get_stats(self, ledger):
        ledger.log_transaction("a", "op1", "x")
        stats = ledger.get_stats()
        assert stats["transaction_count"] == 1
        assert stats["db_size_mb"] >= 0

    def test_critical_callback(self, tmp_path):
        calls = []
        def on_critical(component, action, content, status, metadata):
            calls.append({"component": component, "status": status})

        ledger = ForensicLedger(
            db_path=str(tmp_path / "cb_test.db"),
            on_critical_event=on_critical,
        )
        ledger.log_transaction("indexer", "crash", "disk full", status="failure")
        assert len(calls) == 1
        assert calls[0]["status"] == "failure"

    def test_recent_transactions_filter(self, ledger):
        ledger.log_transaction("tier_a", "index", "doc 1")
        ledger.log_transaction("tier_b", "search", "query")
        results = ledger.get_recent_transactions(component="tier_a")
        assert len(results) == 1
        assert results[0]["component"] == "tier_a"
