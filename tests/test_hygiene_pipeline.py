"""
tests/test_hygiene_pipeline.py — Wave 4 HygienePipeline + Governor.run_hygiene() tests.

Verifies:
  - Pipeline chains decay -> prune -> contradiction sweep in order.
  - Stale state set by decay is respected by prune on the same pass.
  - dry_run propagates to all three runners.
  - Empty engram list: all sub-reports are zeroed.
  - HygienePipelineReport.total_mutations aggregates correctly.
  - Governor.run_hygiene() returns a HygienePipelineReport.
  - Governor.stats() includes hygiene counters after run_hygiene().
  - Custom DecayConfig / PruneConfig override works through Governor.
"""

from __future__ import annotations

import datetime

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.hygiene import (
    DecayConfig,
    HygienePipeline,
    HygienePipelineReport,
    PruneConfig,
)
from mnemos.governance.governor import Governor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso_days_ago(days: float) -> str:
    now = datetime.datetime(2026, 3, 30, tzinfo=datetime.timezone.utc)
    return (now - datetime.timedelta(days=days)).isoformat()


_NOW = "2026-03-30T00:00:00Z"


def _make_engram(
    eid: str,
    utility: float = 0.8,
    trust: float = 0.8,
    last_used_at_days_ago: float | None = None,
    lifecycle_state: str = "active",
    entity_key: str = "",
    attribute_key: str = "",
    normalized_value: str = "",
) -> Engram:
    e = Engram(id=eid, content=f"content for {eid}")
    e.governance = GovernanceMeta(
        utility_score=utility,
        trust_score=trust,
        last_used_at=_iso_days_ago(last_used_at_days_ago) if last_used_at_days_ago else None,
        lifecycle_state=lifecycle_state,
        entity_key=entity_key,
        attribute_key=attribute_key,
        normalized_value=normalized_value,
    )
    return e


# ── Tests: pipeline chain ─────────────────────────────────────────────────────

class TestPipelineChain:
    def test_decay_then_prune_promotion_on_same_pass(self):
        """
        Decay sets lifecycle_state="stale"; the prune pass in the same pipeline
        run then promotes that same engram to prune_candidate.
        """
        # utility=0.18 + 90d past horizon → decays below stale_threshold → "stale"
        # Then PrunePromoter respects stale → "prune_candidate"
        e = _make_engram("e1", utility=0.18, last_used_at_days_ago=90)
        pipeline = HygienePipeline(
            decay_config=DecayConfig(horizon_days=60, stale_threshold=0.20),
            prune_config=PruneConfig(respect_stale_state=True),
        )
        report = pipeline.run([e], now_iso=_NOW)
        assert e.governance.lifecycle_state == "prune_candidate"
        assert report.decay.stale_promoted == 1
        assert report.prune.promoted == 1

    def test_pipeline_returns_all_three_sub_reports(self):
        e = _make_engram("e1", last_used_at_days_ago=90)
        pipeline = HygienePipeline()
        report = pipeline.run([e], now_iso=_NOW)
        assert isinstance(report.decay, type(report.decay))
        assert isinstance(report.prune, type(report.prune))
        assert isinstance(report.sweep, type(report.sweep))

    def test_contradiction_sweep_runs_independently(self):
        e1 = _make_engram("w", entity_key="u", attribute_key="x",
                          normalized_value="a", trust=0.9)
        e2 = _make_engram("l", entity_key="u", attribute_key="x",
                          normalized_value="b", trust=0.3)
        pipeline = HygienePipeline()
        report = pipeline.run([e1, e2], now_iso=_NOW)
        assert report.sweep.contradictions_found == 1


# ── Tests: dry-run propagation ────────────────────────────────────────────────

class TestDryRunPropagation:
    def test_dry_run_prevents_all_mutations(self):
        # Decay-eligible
        e_decay = _make_engram("decay", utility=0.18, last_used_at_days_ago=90)
        # Prune-eligible
        e_prune = _make_engram("prune", utility=0.01, trust=0.5)
        # Contradiction pair
        e1 = _make_engram("w", entity_key="u", attribute_key="x",
                          normalized_value="a", trust=0.9)
        e2 = _make_engram("l", entity_key="u", attribute_key="x",
                          normalized_value="b", trust=0.3)

        pipeline = HygienePipeline(
            decay_config=DecayConfig(horizon_days=60, stale_threshold=0.20),
        )
        report = pipeline.run([e_decay, e_prune, e1, e2],
                              now_iso=_NOW, dry_run=True)

        # Nothing mutated
        assert e_decay.governance.utility_score == 0.18
        assert e_decay.governance.lifecycle_state == "active"
        assert e_prune.governance.lifecycle_state == "active"
        assert e1.governance.conflict_status == "none"
        assert e2.governance.conflict_status == "none"

        # But reports show what would have changed
        assert report.decay.stale_promoted == 1
        assert report.prune.promoted >= 1  # stale not set, uses composite
        assert report.sweep.contradictions_found == 1


# ── Tests: edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_engram_list_returns_zeroed_reports(self):
        pipeline = HygienePipeline()
        report = pipeline.run([], now_iso=_NOW)
        assert report.decay.scanned == 0
        assert report.prune.scanned == 0
        assert report.sweep.clusters_scanned == 0
        assert report.total_mutations == 0

    def test_total_mutations_property(self):
        e_decay = _make_engram("d", utility=0.8, last_used_at_days_ago=90)
        e_prune = _make_engram("p", utility=0.01, trust=0.5)
        e1 = _make_engram("w", entity_key="u", attribute_key="x",
                          normalized_value="a", trust=0.9)
        e2 = _make_engram("l", entity_key="u", attribute_key="x",
                          normalized_value="b", trust=0.3)
        pipeline = HygienePipeline()
        report = pipeline.run([e_decay, e_prune, e1, e2], now_iso=_NOW)
        # decay.decayed + prune.promoted + sweep.winners_set + sweep.losers_set
        expected = (
            report.decay.decayed
            + report.prune.promoted
            + report.sweep.winners_set
            + report.sweep.losers_set
        )
        assert report.total_mutations == expected


# ── Tests: Governor.run_hygiene() ─────────────────────────────────────────────

class TestGovernorRunHygiene:
    def test_governor_run_hygiene_returns_pipeline_report(self):
        gov = Governor()
        e = _make_engram("e1", last_used_at_days_ago=90)
        report = gov.run_hygiene([e], now_iso=_NOW)
        assert isinstance(report, HygienePipelineReport)

    def test_governor_stats_include_hygiene_counters_after_run(self):
        gov = Governor()
        e1 = _make_engram("w", entity_key="u", attribute_key="x",
                          normalized_value="a", trust=0.9)
        e2 = _make_engram("l", entity_key="u", attribute_key="x",
                          normalized_value="b", trust=0.3)
        gov.run_hygiene([e1, e2], now_iso=_NOW)
        stats = gov.stats()
        assert "total_decay_runs" in stats
        assert "total_stale_promoted" in stats
        assert "total_prune_candidates" in stats
        assert "total_contradiction_sweep_clusters" in stats
        assert stats["total_decay_runs"] == 1

    def test_governor_stats_accumulate_across_runs(self):
        gov = Governor()
        e1 = _make_engram("w", entity_key="u", attribute_key="x",
                          normalized_value="a", trust=0.9)
        e2 = _make_engram("l", entity_key="u", attribute_key="x",
                          normalized_value="b", trust=0.3)
        gov.run_hygiene([e1, e2], now_iso=_NOW)
        gov.run_hygiene([e1, e2], now_iso=_NOW)
        stats = gov.stats()
        assert stats["total_decay_runs"] == 2

    def test_governor_run_hygiene_dry_run(self):
        gov = Governor()
        e = _make_engram("e1", utility=0.18, last_used_at_days_ago=90)
        gov.run_hygiene([e], now_iso=_NOW, dry_run=True)
        assert e.governance.lifecycle_state == "active"
        # Stats still record the run
        assert gov.stats()["total_decay_runs"] == 1

    def test_governor_run_hygiene_custom_config(self):
        gov = Governor()
        e = _make_engram("e1", utility=0.8, last_used_at_days_ago=10)
        # Very short horizon → triggers decay even at 10 days
        report = gov.run_hygiene(
            [e],
            now_iso=_NOW,
            decay_config=DecayConfig(horizon_days=5, decay_per_day=0.1),
        )
        assert report.decay.decayed == 1
