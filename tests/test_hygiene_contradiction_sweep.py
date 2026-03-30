"""
tests/test_hygiene_contradiction_sweep.py — Wave 4 ContradictionSweepRunner tests.

Verifies:
  - Two engrams with same (entity_key, attribute_key), different normalized_value
    → contradiction detected and conflict state written to GovernanceMeta.
  - Winner selection follows the same 5-level chain as ContradictionPolicy.
  - Engrams with empty entity_key or attribute_key are skipped.
  - Clusters of one member produce no contradictions.
  - Members sharing the same normalized_value produce no contradiction.
  - conflict_group_id set on both winner and loser GovernanceMeta.
  - conflict_status set to "winner" / "suppressed" correctly.
  - superseded_by set on loser to the winner's ID.
  - dry_run: no mutations, accurate report.
  - Multiple clusters handled independently.
  - ContradictionSweepReport counts are correct.
  - Engrams without governance metadata are skipped.
"""

from __future__ import annotations

import datetime

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.hygiene.contradiction_sweep import (
    ContradictionSweepRunner,
    ContradictionSweepReport,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso_days_ago(days: float) -> str:
    now = datetime.datetime(2026, 3, 30, tzinfo=datetime.timezone.utc)
    ref = now - datetime.timedelta(days=days)
    return ref.isoformat()


def _make_engram(
    eid: str,
    entity_key: str = "",
    attribute_key: str = "",
    normalized_value: str = "",
    trust: float = 0.8,
    utility: float = 0.8,
    created_at_days_ago: float = 0.0,
    has_governance: bool = True,
) -> Engram:
    e = Engram(
        id=eid,
        content=f"content for {eid}",
        created_at=_iso_days_ago(created_at_days_ago),
    )
    if has_governance:
        e.governance = GovernanceMeta(
            entity_key=entity_key,
            attribute_key=attribute_key,
            normalized_value=normalized_value,
            trust_score=trust,
            utility_score=utility,
        )
    return e


def _runner() -> ContradictionSweepRunner:
    return ContradictionSweepRunner()


# ── Tests: basic detection ────────────────────────────────────────────────────

class TestBasicDetection:
    def test_two_conflicting_engrams_detected(self):
        e1 = _make_engram("e1", entity_key="user:alice", attribute_key="city",
                          normalized_value="auckland")
        e2 = _make_engram("e2", entity_key="user:alice", attribute_key="city",
                          normalized_value="wellington")
        report = _runner().run([e1, e2])
        assert report.contradictions_found == 1
        assert report.clusters_scanned == 1

    def test_conflict_group_id_set_on_both_engrams(self):
        e1 = _make_engram("e1", entity_key="user:alice", attribute_key="city",
                          normalized_value="auckland")
        e2 = _make_engram("e2", entity_key="user:alice", attribute_key="city",
                          normalized_value="wellington")
        _runner().run([e1, e2])
        assert e1.governance.conflict_group_id == "conflict:user:alice:city"
        assert e2.governance.conflict_group_id == "conflict:user:alice:city"

    def test_conflict_status_set_correctly(self):
        # e1 has higher trust → wins
        e1 = _make_engram("e1", entity_key="user:alice", attribute_key="city",
                          normalized_value="auckland", trust=0.9)
        e2 = _make_engram("e2", entity_key="user:alice", attribute_key="city",
                          normalized_value="wellington", trust=0.5)
        _runner().run([e1, e2])
        assert e1.governance.conflict_status == "winner"
        assert e2.governance.conflict_status == "suppressed"

    def test_superseded_by_set_on_loser(self):
        e1 = _make_engram("e1", entity_key="user:alice", attribute_key="city",
                          normalized_value="auckland", trust=0.9)
        e2 = _make_engram("e2", entity_key="user:alice", attribute_key="city",
                          normalized_value="wellington", trust=0.5)
        _runner().run([e1, e2])
        assert e2.governance.superseded_by == "e1"
        assert e1.governance.superseded_by is None


# ── Tests: winner selection parity ───────────────────────────────────────────

class TestWinnerSelection:
    def test_higher_trust_wins(self):
        e_high = _make_engram("high", entity_key="proj", attribute_key="status",
                              normalized_value="active", trust=0.9)
        e_low = _make_engram("low", entity_key="proj", attribute_key="status",
                             normalized_value="inactive", trust=0.3)
        _runner().run([e_high, e_low])
        assert e_high.governance.conflict_status == "winner"

    def test_newer_timestamp_wins_on_trust_tie(self):
        e_old = _make_engram("old", entity_key="proj", attribute_key="status",
                             normalized_value="active", trust=0.8,
                             created_at_days_ago=10)
        e_new = _make_engram("new", entity_key="proj", attribute_key="status",
                             normalized_value="inactive", trust=0.8,
                             created_at_days_ago=1)
        _runner().run([e_old, e_new])
        assert e_new.governance.conflict_status == "winner"

    def test_higher_utility_wins_on_trust_recency_tie(self):
        ts = _iso_days_ago(5)
        e_low_util = _make_engram("lu", entity_key="proj", attribute_key="owner",
                                  normalized_value="alice", trust=0.8, utility=0.4)
        e_high_util = _make_engram("hu", entity_key="proj", attribute_key="owner",
                                   normalized_value="bob", trust=0.8, utility=0.9)
        # Same created_at by default (both use _iso_days_ago(0) = same second).
        # Trust tied; recency tied (same created_at approx); utility decides.
        e_low_util.created_at = ts
        e_high_util.created_at = ts
        _runner().run([e_low_util, e_high_util])
        assert e_high_util.governance.conflict_status == "winner"


# ── Tests: no-contradiction cases ────────────────────────────────────────────

class TestNoContradiction:
    def test_same_normalized_value_not_a_contradiction(self):
        e1 = _make_engram("e1", entity_key="u", attribute_key="city",
                          normalized_value="auckland")
        e2 = _make_engram("e2", entity_key="u", attribute_key="city",
                          normalized_value="auckland")
        report = _runner().run([e1, e2])
        assert report.contradictions_found == 0
        assert e1.governance.conflict_status == "none"
        assert e2.governance.conflict_status == "none"

    def test_single_cluster_member_no_contradiction(self):
        e = _make_engram("e1", entity_key="u", attribute_key="city",
                         normalized_value="auckland")
        report = _runner().run([e])
        assert report.contradictions_found == 0

    def test_empty_entity_key_skipped(self):
        e = _make_engram("e1", entity_key="", attribute_key="city",
                         normalized_value="auckland")
        report = _runner().run([e])
        assert report.skipped == 1
        assert report.clusters_scanned == 0

    def test_empty_attribute_key_skipped(self):
        e = _make_engram("e1", entity_key="u", attribute_key="",
                         normalized_value="auckland")
        report = _runner().run([e])
        assert report.skipped == 1

    def test_no_governance_skipped(self):
        e = _make_engram("e1", has_governance=False)
        report = _runner().run([e])
        assert report.skipped == 1


# ── Tests: dry-run ────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_does_not_mutate_conflict_status(self):
        e1 = _make_engram("e1", entity_key="u", attribute_key="city",
                          normalized_value="a", trust=0.9)
        e2 = _make_engram("e2", entity_key="u", attribute_key="city",
                          normalized_value="b", trust=0.5)
        report = _runner().run([e1, e2], dry_run=True)
        assert e1.governance.conflict_status == "none"
        assert e2.governance.conflict_status == "none"
        assert report.contradictions_found == 1  # detected but not applied

    def test_dry_run_report_counts_accurate(self):
        e1 = _make_engram("e1", entity_key="u", attribute_key="city",
                          normalized_value="a")
        e2 = _make_engram("e2", entity_key="u", attribute_key="city",
                          normalized_value="b")
        report = _runner().run([e1, e2], dry_run=True)
        assert report.winners_set == 1
        assert report.losers_set == 1


# ── Tests: multiple clusters ──────────────────────────────────────────────────

class TestMultipleClusters:
    def test_two_independent_clusters_both_resolved(self):
        # Cluster 1: user:alice / city
        e1 = _make_engram("e1", entity_key="user:alice", attribute_key="city",
                          normalized_value="auckland")
        e2 = _make_engram("e2", entity_key="user:alice", attribute_key="city",
                          normalized_value="wellington")
        # Cluster 2: project:mnemos / status
        e3 = _make_engram("e3", entity_key="project:mnemos", attribute_key="status",
                          normalized_value="active")
        e4 = _make_engram("e4", entity_key="project:mnemos", attribute_key="status",
                          normalized_value="inactive")
        report = _runner().run([e1, e2, e3, e4])
        assert report.contradictions_found == 2
        assert report.clusters_scanned == 2
        assert report.winners_set == 2
        assert report.losers_set == 2


# ── Tests: report structure ───────────────────────────────────────────────────

class TestReportStructure:
    def test_sweep_record_fields(self):
        e1 = _make_engram("winner", entity_key="u", attribute_key="x",
                          normalized_value="a", trust=0.9)
        e2 = _make_engram("loser", entity_key="u", attribute_key="x",
                          normalized_value="b", trust=0.3)
        report = _runner().run([e1, e2])
        assert len(report.records) == 1
        rec = report.records[0]
        assert rec.cluster_key == "u:x"
        assert rec.winner_id == "winner"
        assert rec.loser_ids == ["loser"]
        assert rec.conflict_group_id == "conflict:u:x"
        assert rec.resolution_reason  # non-empty
