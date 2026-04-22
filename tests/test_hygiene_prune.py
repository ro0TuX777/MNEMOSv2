"""
tests/test_hygiene_prune.py — Wave 4 PrunePromoter unit tests.

Verifies:
  - Low composite score → promoted to prune_candidate.
  - High composite score → not promoted.
  - Already prune_candidate or archived → skipped.
  - soft_deleted / tombstone deletion_state → skipped.
  - Contradiction loser (conflict_status=suppressed) → reduced composite.
  - respect_stale_state: stale lifecycle_state → always promoted.
  - dry_run: no mutations, accurate report.
  - PruneReport counts are correct.
"""

from __future__ import annotations

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.hygiene.prune_promoter import PruneConfig, PruneReport, PrunePromoter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_engram(
    eid: str = "e1",
    utility: float = 0.8,
    trust: float = 0.8,
    lifecycle_state: str = "active",
    deletion_state: str = "active",
    conflict_status: str = "none",
    has_governance: bool = True,
) -> Engram:
    e = Engram(id=eid, content=f"content for {eid}")
    if has_governance:
        e.governance = GovernanceMeta(
            utility_score=utility,
            trust_score=trust,
            lifecycle_state=lifecycle_state,
            deletion_state=deletion_state,
            conflict_status=conflict_status,
        )
    return e


def _promoter(composite_floor: float = 0.05, respect_stale: bool = True) -> PrunePromoter:
    return PrunePromoter(PruneConfig(composite_floor=composite_floor,
                                    respect_stale_state=respect_stale))


# ── Tests: composite score floor ─────────────────────────────────────────────

class TestCompositeFloor:
    def test_low_composite_promotes_to_prune_candidate(self):
        # composite = 0.02 * 0.9 * 1.0 = 0.018 < 0.05
        e = _make_engram(utility=0.02, trust=0.9)
        _promoter().run([e])
        assert e.governance.lifecycle_state == "prune_candidate"

    def test_high_composite_not_promoted(self):
        # composite = 0.8 * 0.8 * 1.0 = 0.64 > 0.05
        e = _make_engram(utility=0.8, trust=0.8)
        _promoter().run([e])
        assert e.governance.lifecycle_state == "active"

    def test_boundary_composite_at_floor_not_promoted(self):
        # composite exactly at floor = 0.05 is NOT below floor → not promoted
        # utility=0.25, trust=0.2 → composite = 0.05
        e = _make_engram(utility=0.25, trust=0.2)
        _promoter(composite_floor=0.05).run([e])
        assert e.governance.lifecycle_state == "active"

    def test_composite_just_below_floor_promoted(self):
        # composite = 0.24 * 0.2 = 0.048 < 0.05
        e = _make_engram(utility=0.24, trust=0.2)
        _promoter(composite_floor=0.05).run([e])
        assert e.governance.lifecycle_state == "prune_candidate"


# ── Tests: contradiction loser penalty ───────────────────────────────────────

class TestContradictionLoser:
    def test_loser_composite_reduced_by_factor(self):
        # conflict_status="suppressed" applies 0.25 factor.
        # utility=0.6, trust=0.6 → composite = 0.6 * 0.6 * 0.25 = 0.09 > 0.05 → not promoted
        e = _make_engram(utility=0.6, trust=0.6, conflict_status="suppressed")
        _promoter(composite_floor=0.05).run([e])
        assert e.governance.lifecycle_state == "active"

    def test_loser_with_low_scores_promoted(self):
        # utility=0.4, trust=0.5 → composite = 0.4 * 0.5 * 0.25 = 0.05 → not below
        # utility=0.3, trust=0.5 → composite = 0.3 * 0.5 * 0.25 = 0.0375 < 0.05 → promoted
        e = _make_engram(utility=0.3, trust=0.5, conflict_status="suppressed")
        _promoter(composite_floor=0.05).run([e])
        assert e.governance.lifecycle_state == "prune_candidate"

    def test_winner_conflict_status_uses_full_composite(self):
        # conflict_status="winner" → factor = 1.0 (no penalty)
        e = _make_engram(utility=0.3, trust=0.5, conflict_status="winner")
        # composite = 0.3 * 0.5 * 1.0 = 0.15 > 0.05 → not promoted
        _promoter(composite_floor=0.05).run([e])
        assert e.governance.lifecycle_state == "active"


# ── Tests: skip conditions ────────────────────────────────────────────────────

class TestSkipConditions:
    def test_already_prune_candidate_skipped(self):
        e = _make_engram(utility=0.01, lifecycle_state="prune_candidate")
        report = _promoter().run([e])
        assert e.governance.lifecycle_state == "prune_candidate"
        assert report.skipped == 1
        assert report.promoted == 0

    def test_archived_skipped(self):
        e = _make_engram(utility=0.01, lifecycle_state="archived")
        report = _promoter().run([e])
        assert report.skipped == 1

    def test_soft_deleted_skipped(self):
        e = _make_engram(deletion_state="soft_deleted")
        report = _promoter().run([e])
        assert report.skipped == 1

    def test_tombstone_skipped(self):
        e = _make_engram(deletion_state="tombstone")
        report = _promoter().run([e])
        assert report.skipped == 1

    def test_no_governance_skipped(self):
        e = _make_engram(has_governance=False)
        report = _promoter().run([e])
        assert report.skipped == 1
        assert report.scanned == 0


# ── Tests: stale state respect ────────────────────────────────────────────────

class TestStaleStateRespect:
    def test_stale_always_promoted_when_flag_true(self):
        # High composite but already stale → promoted.
        e = _make_engram(utility=0.9, trust=0.9, lifecycle_state="stale")
        _promoter(respect_stale=True).run([e])
        assert e.governance.lifecycle_state == "prune_candidate"

    def test_stale_not_specially_promoted_when_flag_false(self):
        # respect_stale=False → evaluated by composite only.
        # utility=0.9, trust=0.9, composite=0.81 > 0.05 → not promoted.
        e = _make_engram(utility=0.9, trust=0.9, lifecycle_state="stale")
        _promoter(respect_stale=False).run([e])
        assert e.governance.lifecycle_state == "stale"


# ── Tests: dry-run ────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_does_not_mutate(self):
        e = _make_engram(utility=0.01, trust=0.5)
        report = _promoter().run([e], dry_run=True)
        assert e.governance.lifecycle_state == "active"
        assert report.promoted == 1  # would have been promoted

    def test_dry_run_stale_not_mutated(self):
        e = _make_engram(lifecycle_state="stale")
        report = _promoter(respect_stale=True).run([e], dry_run=True)
        assert e.governance.lifecycle_state == "stale"
        assert report.promoted == 1


# ── Tests: report counts ──────────────────────────────────────────────────────

class TestReportCounts:
    def test_mixed_batch_counts(self):
        e_promote = _make_engram("e1", utility=0.01)
        e_keep = _make_engram("e2", utility=0.9)
        e_skip = _make_engram("e3", has_governance=False)
        report = _promoter().run([e_promote, e_keep, e_skip])
        assert report.scanned == 2
        assert report.promoted == 1
        assert report.skipped == 1
