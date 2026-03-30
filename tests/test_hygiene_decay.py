"""
tests/test_hygiene_decay.py — Wave 4 DecayRunner unit tests.

Verifies:
  - Memories within the inactivity horizon are untouched.
  - Decay formula is applied correctly past the horizon.
  - Stale promotion fires when utility drops below the threshold.
  - utility_score never goes below the floor.
  - dry_run produces accurate report without any mutations.
  - Engrams without governance metadata are skipped.
  - DecayReport counts are correct.
"""

from __future__ import annotations

import datetime
from typing import Optional

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.hygiene.decay_runner import DecayConfig, DecayReport, DecayRunner


# ── Helpers ───────────────────────────────────────────────────────────────────

_BASE_NOW = "2026-03-30T00:00:00Z"


def _iso_days_ago(days: float, base: str = _BASE_NOW) -> str:
    """Return an ISO timestamp that is `days` days before `base`."""
    now = datetime.datetime.fromisoformat(base.replace("Z", "+00:00"))
    ref = now - datetime.timedelta(days=days)
    return ref.isoformat()


def _make_engram(
    eid: str = "e1",
    utility: float = 0.8,
    trust: float = 1.0,
    last_used_at: Optional[str] = None,
    created_at_days_ago: float = 0.0,
    lifecycle_state: str = "active",
    has_governance: bool = True,
) -> Engram:
    created_at = _iso_days_ago(created_at_days_ago)
    e = Engram(id=eid, content=f"content for {eid}", created_at=created_at)
    if has_governance:
        e.governance = GovernanceMeta(
            utility_score=utility,
            trust_score=trust,
            last_used_at=last_used_at,
            lifecycle_state=lifecycle_state,
        )
    return e


def _runner(horizon_days: int = 60, decay_per_day: float = 0.005,
            stale_threshold: float = 0.20, min_utility: float = 0.0) -> DecayRunner:
    return DecayRunner(DecayConfig(
        horizon_days=horizon_days,
        decay_per_day=decay_per_day,
        stale_threshold=stale_threshold,
        min_utility=min_utility,
    ))


# ── Tests: within-horizon behaviour ──────────────────────────────────────────

class TestWithinHorizon:
    def test_memory_used_yesterday_is_untouched(self):
        e = _make_engram(last_used_at=_iso_days_ago(1))
        runner = _runner()
        report = runner.run([e], now_iso=_BASE_NOW)
        assert e.governance.utility_score == 0.8
        assert report.decayed == 0
        assert report.scanned == 1

    def test_memory_at_exact_horizon_is_untouched(self):
        # elapsed == horizon_days is ≤, not <, so boundary should NOT decay.
        e = _make_engram(last_used_at=_iso_days_ago(60))
        runner = _runner(horizon_days=60)
        report = runner.run([e], now_iso=_BASE_NOW)
        assert e.governance.utility_score == 0.8
        assert report.decayed == 0

    def test_memory_one_second_inside_horizon_is_untouched(self):
        e = _make_engram(last_used_at=_iso_days_ago(59.999))
        runner = _runner(horizon_days=60)
        report = runner.run([e], now_iso=_BASE_NOW)
        assert e.governance.utility_score == 0.8


# ── Tests: decay formula ──────────────────────────────────────────────────────

class TestDecayFormula:
    def test_decay_30_days_past_horizon(self):
        # elapsed=90, horizon=60 -> 30 past -> delta = 30 * 0.005 = 0.15
        # new_utility = 0.8 - 0.15 = 0.65
        e = _make_engram(last_used_at=_iso_days_ago(90))
        runner = _runner(horizon_days=60, decay_per_day=0.005)
        runner.run([e], now_iso=_BASE_NOW)
        assert abs(e.governance.utility_score - 0.65) < 1e-9

    def test_decay_uses_last_used_at_over_created_at(self):
        # last_used_at is 90 days ago; created_at is 200 days ago.
        # Decay should be calculated from last_used_at (90 days).
        e = _make_engram(
            last_used_at=_iso_days_ago(90),
            created_at_days_ago=200,
        )
        runner = _runner(horizon_days=60, decay_per_day=0.005)
        runner.run([e], now_iso=_BASE_NOW)
        expected = 0.8 - (30 * 0.005)
        assert abs(e.governance.utility_score - expected) < 1e-9

    def test_decay_falls_back_to_created_at_when_no_last_used(self):
        # No last_used_at; created_at is 90 days ago.
        e = _make_engram(created_at_days_ago=90)
        assert e.governance.last_used_at is None
        runner = _runner(horizon_days=60, decay_per_day=0.005)
        runner.run([e], now_iso=_BASE_NOW)
        expected = 0.8 - (30 * 0.005)
        assert abs(e.governance.utility_score - expected) < 1e-9

    def test_report_decayed_count(self):
        e1 = _make_engram("e1", last_used_at=_iso_days_ago(90))
        e2 = _make_engram("e2", last_used_at=_iso_days_ago(30))  # within horizon
        runner = _runner()
        report = runner.run([e1, e2], now_iso=_BASE_NOW)
        assert report.decayed == 1
        assert report.scanned == 2


# ── Tests: stale promotion ────────────────────────────────────────────────────

class TestStalePromotion:
    def test_utility_below_threshold_sets_stale(self):
        # utility=0.18 already below stale_threshold=0.20 but still > 0.
        # After decay of 0.01 it's 0.17 → should become stale.
        e = _make_engram(utility=0.18, last_used_at=_iso_days_ago(62))
        runner = _runner(horizon_days=60, decay_per_day=0.005, stale_threshold=0.20)
        runner.run([e], now_iso=_BASE_NOW)
        assert e.governance.lifecycle_state == "stale"

    def test_utility_above_threshold_stays_active(self):
        e = _make_engram(utility=0.8, last_used_at=_iso_days_ago(62))
        runner = _runner(stale_threshold=0.20)
        runner.run([e], now_iso=_BASE_NOW)
        assert e.governance.lifecycle_state == "active"

    def test_already_stale_lifecycle_state_not_double_counted(self):
        e = _make_engram(utility=0.18, last_used_at=_iso_days_ago(62),
                         lifecycle_state="stale")
        runner = _runner(stale_threshold=0.20)
        report = runner.run([e], now_iso=_BASE_NOW)
        # Still decayed (utility reduced), but stale_promoted should be 0
        # because it was already stale.
        assert report.stale_promoted == 0

    def test_stale_promoted_count_in_report(self):
        e1 = _make_engram("e1", utility=0.18, last_used_at=_iso_days_ago(62))
        e2 = _make_engram("e2", utility=0.8, last_used_at=_iso_days_ago(62))
        runner = _runner(stale_threshold=0.20)
        report = runner.run([e1, e2], now_iso=_BASE_NOW)
        assert report.stale_promoted == 1


# ── Tests: floor and bounds ───────────────────────────────────────────────────

class TestFloorAndBounds:
    def test_utility_never_goes_below_floor(self):
        # Start at 0.1, decay 200 days past horizon → would go negative.
        e = _make_engram(utility=0.1, last_used_at=_iso_days_ago(260))
        runner = _runner(horizon_days=60, decay_per_day=0.005, min_utility=0.0)
        runner.run([e], now_iso=_BASE_NOW)
        assert e.governance.utility_score >= 0.0

    def test_custom_min_utility_floor(self):
        e = _make_engram(utility=0.15, last_used_at=_iso_days_ago(200))
        runner = _runner(min_utility=0.10)
        runner.run([e], now_iso=_BASE_NOW)
        assert e.governance.utility_score >= 0.10


# ── Tests: dry-run ────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_does_not_mutate_utility(self):
        e = _make_engram(utility=0.8, last_used_at=_iso_days_ago(90))
        runner = _runner()
        report = runner.run([e], now_iso=_BASE_NOW, dry_run=True)
        assert e.governance.utility_score == 0.8
        assert report.decayed == 1  # would have been decayed

    def test_dry_run_does_not_set_stale(self):
        e = _make_engram(utility=0.18, last_used_at=_iso_days_ago(62))
        runner = _runner()
        report = runner.run([e], now_iso=_BASE_NOW, dry_run=True)
        assert e.governance.lifecycle_state == "active"
        assert report.stale_promoted == 1  # would have been promoted


# ── Tests: no governance metadata ────────────────────────────────────────────

class TestNoGovernance:
    def test_engram_without_governance_is_skipped(self):
        e = _make_engram(has_governance=False)
        runner = _runner()
        report = runner.run([e], now_iso=_BASE_NOW)
        assert report.skipped == 1
        assert report.scanned == 0

    def test_mixed_list_counts_correctly(self):
        e_gov = _make_engram("e1", last_used_at=_iso_days_ago(90))
        e_no_gov = _make_engram("e2", has_governance=False)
        runner = _runner()
        report = runner.run([e_gov, e_no_gov], now_iso=_BASE_NOW)
        assert report.scanned == 1
        assert report.skipped == 1
        assert report.decayed == 1
