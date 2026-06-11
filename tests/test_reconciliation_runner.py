"""Tests for Phase 10 knowledge reconciliation scaffolding."""

from __future__ import annotations

import datetime

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.reconciliation_runner import ReconciliationRunner
from mnemos.governance.models.memory_state import GovernanceMeta


class _CapturingIndexer:
    def __init__(self):
        self.indexed = []

    def index(self, engrams):
        self.indexed.extend(engrams)
        return len(engrams)


def _iso_days_ago(days: float) -> str:
    now = datetime.datetime(2026, 6, 11, tzinfo=datetime.timezone.utc)
    ref = now - datetime.timedelta(days=days)
    return ref.isoformat()


def _engram(eid: str, value: str, trust: float = 0.8) -> Engram:
    return Engram(
        id=eid,
        content=f"Project X status is {value}.",
        source=f"fixture://{eid}",
        created_at=_iso_days_ago(0),
        governance=GovernanceMeta(
            entity_key="project:x",
            attribute_key="status",
            normalized_value=value,
            trust_score=trust,
            utility_score=0.8,
        ),
    )


def test_dry_run_synthesizes_resolution_engram_without_writing(monkeypatch):
    monkeypatch.delenv("SMC_LLM_BASE_URL", raising=False)
    old = _engram("old", "cancelled_2024", trust=0.6)
    new = _engram("new", "extended_2026", trust=0.9)

    report = ReconciliationRunner().run([old, new], dry_run=True)

    assert report.contradictions_found == 1
    assert report.resolution_engram_writes == 0
    assert len(report.records) == 1
    resolution = report.records[0].resolution_engram
    assert resolution.metadata["is_resolution_engram"] is True
    assert resolution.source == "derived://reconciliation/project_x"
    assert sorted(resolution.edges) == ["new", "old"]
    assert resolution.governance.derived_from == ["new", "old"]
    assert resolution.governance.entity_key == "project:x"
    assert resolution.governance.attribute_key == "status"


def test_apply_requires_indexer():
    old = _engram("old", "cancelled_2024")
    new = _engram("new", "extended_2026")

    with pytest.raises(ValueError, match="indexer"):
        ReconciliationRunner().run([old, new], dry_run=False)


def test_apply_writes_resolution_engram(monkeypatch):
    monkeypatch.delenv("SMC_LLM_BASE_URL", raising=False)
    old = _engram("old", "cancelled_2024")
    new = _engram("new", "extended_2026")
    indexer = _CapturingIndexer()

    report = ReconciliationRunner().run([old, new], dry_run=False, indexer=indexer)

    assert report.resolution_engram_writes == 1
    assert len(indexer.indexed) == 1
    assert indexer.indexed[0].metadata["parent_ids"] == ["new", "old"]
