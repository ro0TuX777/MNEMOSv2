"""Tests for Memory Over Maps Phase 1 benchmark runner."""

from benchmarks.runners.memory_over_maps_runner import (
    run_phase1_lineage_track,
    run_phase2_candidate_envelope_track,
    run_phase3_derived_views_track,
    run_phase4_cache_invalidation_track,
    run_phase5_reflect_bounded_track,
)


def test_phase1_lineage_runner_meets_baseline_targets():
    result = run_phase1_lineage_track(sample_size=8).to_dict()
    assert result["lineage_completeness_rate"] == 1.0
    assert result["responses_with_source_artifact_coverage_rate"] == 1.0
    assert result["derived_view_input_completeness_rate"] == 1.0
    assert result["orphan_derived_views"] == 0
    assert result["audit_log_derived_view_events"] >= 1


def test_phase2_candidate_envelope_runner_meets_baseline_targets():
    result = run_phase2_candidate_envelope_track().to_dict()
    assert result["initial_candidate_count"] > result["final_candidate_count"]
    assert result["compression_ratio"] < 1.0
    assert result["answer_support_retention_rate"] >= 0.75
    assert result["deterministic_replay_match"] is True


def test_phase3_derived_views_runner_meets_baseline_targets():
    result = run_phase3_derived_views_track().to_dict()
    assert result["reproducibility_success_rate"] == 1.0
    assert result["regeneration_mismatch_count"] == 0
    assert result["input_completeness_rate"] == 1.0


def test_phase4_cache_invalidation_runner_meets_baseline_targets():
    result = run_phase4_cache_invalidation_track().to_dict()
    assert result["invalidation_trigger_coverage_rate"] >= 1.0
    assert result["stale_cache_survival_rate"] == 0.0
    assert result["dry_run_real_run_parity"] is True


def test_phase5_reflect_runner_meets_baseline_targets():
    result = run_phase5_reflect_bounded_track().to_dict()
    assert result["bounded_candidate_adherence_rate"] == 1.0
    assert result["proper_noun_sensitivity_rate"] >= 0.9
    assert result["generic_short_memory_false_positive_rate"] <= 0.0
    assert result["trust_recovery_delta"] >= 0.05
    assert result["enforced_mode_drift_rate"] <= 0.0
    assert result["concurrent_reflect_success_rate"] >= 0.95
