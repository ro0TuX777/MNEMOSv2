import json

from tools.run_coala_cycle_validation import (
    REPRESENTATIVE_QUERY_SET,
    run_validation,
)


def test_validation_harness_passes_all_gates():
    report = run_validation()
    assert report["status"] == "PASS"
    assert report["representative_query_count"] == len(REPRESENTATIVE_QUERY_SET)
    assert all(gate["passed"] for gate in report["gates"].values())


def test_representative_query_set_covers_approved_paths():
    report = run_validation()
    cases = set(report["representative_query_set"])
    assert {
        "CLASS_A_DIRECT_LOOKUP",
        "CLASS_B_MULTI_HOP",
        "CLASS_C_GLOBAL_SYNTHESIS",
        "CONTRADICTION_RECONCILIATION",
        "HIGH_VOLATILITY_GOVERNANCE",
        "FORECAST_TRIGGERED_PULSE",
        "PRE_COGNITIVE_SHADOW_SEARCH",
        "DERIVED_VIEW_EVIDENCE_BUNDLE",
    }.issubset(cases)


def test_attention_is_evidence_derived_not_explanation_generated():
    report = run_validation()
    for cycle in report["cycles"]:
        for decision in cycle["attention_decisions"]:
            assert decision.get("policy_source")
            assert "llm" not in decision["policy_source"].lower()
            assert "generated_explanation" not in decision["policy_source"].lower()


def test_records_are_bounded_and_redacted():
    report = run_validation()
    max_bytes = report["gates"]["bounded_record"]["max_record_bytes"]
    sensitive = {"api_key", "authorization", "password", "secret", "token"}
    for cycle in report["cycles"]:
        assert len(json.dumps(cycle).encode("utf-8")) <= max_bytes
        assert len(cycle["query_or_event"]) <= 240
        assert sensitive.isdisjoint({key.lower() for key in cycle.keys()})


def test_forecast_cycle_has_resolved_forecast_lifecycle():
    report = run_validation()
    forecast_cycles = [
        c for c in report["cycles"]
        if c["validation_case_id"] == "FORECAST_TRIGGERED_PULSE"
    ]
    assert len(forecast_cycles) == 1
    assert forecast_cycles[0]["outcome_observation_plan"] == "forecast_outcome_record"
    assert report["forecast_outcomes"]
    for forecast in report["forecast_outcomes"]:
        assert forecast["actual_outcome"] is not None
        assert forecast["resolved_at"] is not None


def test_pattern_candidates_are_advisory_and_include_required_fields():
    report = run_validation()
    assert report["pattern_engram_candidates"]
    for candidate in report["pattern_engram_candidates"]:
        assert candidate["proposed_learning_class"]
        assert candidate["risk_if_wrong"]
        assert candidate["promotion_status"] != "approved"
        assert candidate["authoritative_for_retrieval"] is False
        assert candidate["affects_ranking"] is False
        assert candidate["mutates_policy"] is False
