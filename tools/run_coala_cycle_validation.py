"""
CoALA cycle operational validation harness for MNEMOS.

This harness validates the cognitive-cycle contract against deterministic
representative records.  It proves that the cycle record is evidence-derived,
bounded, redacted, SAM-compatible, and linked to forecast lifecycle state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnemos.cognitive.assembler import CycleAssembler
from mnemos.cognitive.attention import build_attention_decisions
from mnemos.cognitive.forecast_outcome import ForecastOutcomeRecord
from mnemos.cognitive.learning_boundary import (
    AUDIT_WRITE,
    BLOCKED_PROCEDURAL_MUTATION,
    CACHE_WRITE,
    EPISODIC_WRITE,
    FORECAST_OUTCOME_WRITE,
    GOVERNANCE_SCORE_UPDATE,
    PROCEDURAL_CHANGE_CANDIDATE,
    SEMANTIC_CANDIDATE_WRITE,
    SEMANTIC_WRITE,
    classify_learning_write,
)
from mnemos.cognitive.pattern_engram import PatternEngramCandidate


SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
    "raw_engram_content",
    "raw_prompt",
    "private_reasoning",
}

MAX_RECORD_BYTES = 16000
REQUIRED_SAM_KEYS = {
    "cycle_id",
    "trigger_type",
    "query_or_event",
    "working_memory_snapshot",
    "attention_decisions",
    "retrieval_actions",
    "reasoning_actions",
    "forecast_actions",
    "governance_evaluations",
    "execution_actions",
    "learning_writes",
    "forensic_ledger_refs",
    "selected_route",
    "final_status",
}


REPRESENTATIVE_QUERY_SET: Tuple[Tuple[str, str, str], ...] = (
    ("CLASS_A_DIRECT_LOOKUP", "What is the capital of France?", "factoid"),
    ("CLASS_B_MULTI_HOP", "Which retained policy controls affect audit deletion?", "multi_hop"),
    ("CLASS_C_GLOBAL_SYNTHESIS", "Summarise the corpus-wide governance posture.", "global_synthesis"),
    ("CONTRADICTION_RECONCILIATION", "Reconcile conflicting CEO-name memories.", "reconciliation"),
    ("HIGH_VOLATILITY_GOVERNANCE", "Who is the current release owner?", "high_volatility"),
    ("FORECAST_TRIGGERED_PULSE", "Predicted p95 breach pulse event.", "forecast"),
    ("PRE_COGNITIVE_SHADOW_SEARCH", "Likely follow-up from intent trajectory.", "shadow"),
    ("DERIVED_VIEW_EVIDENCE_BUNDLE", "Return evidence bundle for retention rules.", "evidence_bundle"),
)


def _mode_meta(kind: str) -> Dict[str, Any]:
    envelope = {
        "enabled": True,
        "initial_candidate_count": 16,
        "final_candidate_count": 8,
        "suppression_summary": {
            "duplicate_similarity": 3,
            "source_cap_exceeded": 2,
            "bounded_limit_exceeded": 3,
        },
    }
    meta: Dict[str, Any] = {
        "retrieval_mode": "hybrid",
        "fusion_policy": "balanced",
        "fusion_engine": "python_rrf",
        "candidate_envelope": envelope,
        "lexical_available": True,
    }
    if kind in {"forecast", "high_volatility"}:
        meta["forecast_advisory"] = {
            "suggested_plan": "conservative",
            "forecast_pressure": 0.82,
            "forecast_reason": "predicted p95 or contradiction pressure",
            "confidence_score": 0.84,
        }
    return meta


def build_representative_cycles() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cycles: List[Dict[str, Any]] = []
    forecasts: List[Dict[str, Any]] = []
    patterns: List[Dict[str, Any]] = []

    for index, (case_id, query, family) in enumerate(REPRESENTATIVE_QUERY_SET):
        kind = "forecast" if family == "forecast" else family
        if family == "high_volatility":
            kind = "high_volatility"
        meta = _mode_meta(kind)
        pre_cognitive_hit = family == "shadow"
        derive_views = ["evidence_bundle"] if family == "evidence_bundle" else None
        shadow_present = family == "shadow"
        intent_forecast_present = family == "shadow"

        asm = CycleAssembler(
            trigger_type="pulse" if family == "forecast" else "search",
            trigger_source="autonomous" if family == "forecast" else "api",
            query_or_event=query,
            active_profile="core_appliance",
            governance_profile="default",
        )
        asm.set_working_memory(
            query=query,
            query_class=family,
            query_class_confidence=0.91,
            candidate_count_pre_governance=16,
            candidate_count_post_governance=8,
            suppression_count=2 if family in {"reconciliation", "high_volatility"} else 0,
            contradiction_count=1 if family == "reconciliation" else 0,
            active_retrieval_mode="pre_cognitive_cache" if pre_cognitive_hit else "hybrid",
            active_fusion_policy="balanced",
            active_governance_mode="enforced" if family in {"reconciliation", "high_volatility"} else "advisory",
            active_policy_profile="default",
            summary_eligible=family == "global_synthesis",
            derived_fact_eligible=derive_views is not None,
            forecast_advisory_present="forecast_advisory" in meta,
            pre_cognitive_cache_hit=pre_cognitive_hit,
        )
        asm.add_attention_decisions(build_attention_decisions(
            mode_meta=meta,
            governance_mode="enforced" if family in {"reconciliation", "high_volatility"} else "advisory",
            governance_profile="default",
            phase1_enabled=True,
            phase2_enabled=True,
            phase3_enabled=True,
            derive_views_requested=derive_views,
            pre_cognitive_hit=pre_cognitive_hit,
            query_family=family,
            query_family_confidence=0.91,
            rerank_applied=family in {"multi_hop", "global_synthesis"},
            shadow_search_present=shadow_present,
            intent_forecast_present=intent_forecast_present,
        ))
        if not pre_cognitive_hit:
            asm.add_retrieval_action(
                "python_hybrid_rrf",
                inputs={"query_class": family, "top_k": 8},
                outputs={"candidates_returned": 8},
                latency_ms=20.0 + index,
            )
        else:
            asm.add_retrieval_action(
                "pre_cognitive_cache",
                inputs={"cache_key": "intent_cluster_42"},
                outputs={"cache_hit": True},
                latency_ms=1.2,
            )
        if family in {"reconciliation", "evidence_bundle", "global_synthesis"}:
            asm.add_reasoning_action(
                "contradiction_detection" if family == "reconciliation" else "evidence_bundle_view",
                inputs={"candidate_count": 8},
                outputs={"groups_detected": 1 if family == "reconciliation" else 0},
            )
        if family == "forecast":
            record = ForecastOutcomeRecord.from_pulse_advisory(meta["forecast_advisory"])
            record.resolve(
                actual_outcome="latency_breach_not_observed",
                forecast_error_delta=0.04,
                operator_feedback="accurate",
                learning_recommendation="keep advisory threshold unchanged",
            )
            forecasts.append(record.to_dict())
            asm.add_forecast_action(
                "pulse_forecast",
                inputs={"source_signal": record.source_signal},
                outputs={"forecast_id": record.forecast_id, "resolved": record.resolved_at is not None},
            )
            asm.add_learning_write(
                target_memory_type="semantic",
                operation="write",
                write_class=FORECAST_OUTCOME_WRITE,
                triggered_by="forecast_resolution",
            )
            asm.add_ledger_ref(f"forecast-ledger-{index}")
        elif family == "shadow":
            asm.add_execution_action(
                "cache_write",
                inputs={"source": "intent_trajectory"},
                outputs={"cache_scope": "pre_cognitive"},
            )
            asm.add_learning_write(
                target_memory_type="cache",
                operation="write",
                write_class=CACHE_WRITE,
                triggered_by="intent_shadow_search",
            )
        elif family == "reconciliation":
            asm.add_learning_write(
                target_memory_type="semantic",
                operation="update",
                write_class=GOVERNANCE_SCORE_UPDATE,
                triggered_by="contradiction_reconciliation",
            )
        elif family == "evidence_bundle":
            asm.add_learning_write(
                target_memory_type="audit",
                operation="write",
                write_class=AUDIT_WRITE,
                triggered_by="evidence_bundle_view",
            )
            asm.add_ledger_ref(f"audit-ledger-{index}")
        else:
            asm.add_learning_write(
                target_memory_type="episodic" if family == "factoid" else "semantic",
                operation="write",
                write_class=EPISODIC_WRITE if family == "factoid" else SEMANTIC_WRITE,
                triggered_by="cycle_observation",
            )

        cycle = asm.build(
            selected_route="pre_warm" if family == "forecast" else "return",
            outcome_observation_plan="forecast_outcome_record" if family == "forecast" else None,
        ).to_dict()
        cycle["validation_case_id"] = case_id
        cycles.append(cycle)

    patterns.append(PatternEngramCandidate(
        pattern_summary="Repeated evidence-bundle requests benefit from bounded candidate envelopes.",
        supporting_cycle_ids=[cycles[-1]["cycle_id"], cycles[1]["cycle_id"]],
        supporting_engram_ids=["eng-policy-1", "eng-policy-2"],
        confidence_score=0.72,
        support_score=0.8,
        contradiction_score=0.05,
        pattern_type="descriptive",
        recommended_scope="evidence_bundle_queries",
        applies_when="Query requests a derived evidence bundle over governed policy memory.",
        does_not_apply_when="Query requires default factoid retrieval only.",
        risk_if_wrong="MNEMOS could over-use derived views for simple lookup queries.",
        proposed_learning_class=SEMANTIC_CANDIDATE_WRITE,
    ).to_dict())
    operational = PatternEngramCandidate(
        pattern_summary="Forecast spikes may justify conservative routing recommendations.",
        supporting_cycle_ids=[c["cycle_id"] for c in cycles if c["validation_case_id"] == "FORECAST_TRIGGERED_PULSE"],
        confidence_score=0.94,
        support_score=0.9,
        contradiction_score=0.02,
        pattern_type="operational_recommendation",
        recommended_scope="pulse_governance",
        applies_when="Pulse pressure exceeds the advisory threshold with high confidence.",
        does_not_apply_when="Forecast remains unresolved or low confidence.",
        risk_if_wrong="MNEMOS could route conservatively during normal load.",
        proposed_learning_class=PROCEDURAL_CHANGE_CANDIDATE,
    )
    operational.recommend_promotion(gate_id="coala_validation_gate")
    patterns.append(operational.to_dict())

    blocked = PatternEngramCandidate(
        pattern_summary="Do not mutate routing policy from pattern candidates without review.",
        supporting_cycle_ids=[cycles[4]["cycle_id"], cycles[5]["cycle_id"]],
        confidence_score=0.99,
        support_score=0.95,
        contradiction_score=0.0,
        pattern_type="routing_mutation",
        recommended_scope="router_policy",
        applies_when="A candidate proposes future behavior change.",
        does_not_apply_when="Candidate is descriptive only.",
        risk_if_wrong="MNEMOS could amplify its own interpretation into policy.",
        proposed_learning_class=BLOCKED_PROCEDURAL_MUTATION,
    )
    blocked.recommend_promotion(gate_id="coala_validation_gate")
    patterns.append(blocked.to_dict())

    return cycles, forecasts, patterns


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _gate_attention_faithfulness(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    for cycle in cycles:
        decisions = cycle["attention_decisions"]
        if not decisions:
            failures.append(f"{cycle['validation_case_id']}: no attention decisions")
        for decision in decisions:
            source = decision.get("policy_source", "")
            if not source:
                failures.append(f"{cycle['validation_case_id']}: missing policy_source")
            if "llm" in source.lower() or "generated_explanation" in source.lower():
                failures.append(f"{cycle['validation_case_id']}: explanation-generated policy_source")
    return {"passed": not failures, "failures": failures}


def _gate_bounded_record(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    for cycle in cycles:
        size = len(json.dumps(cycle, sort_keys=True).encode("utf-8"))
        if size > MAX_RECORD_BYTES:
            failures.append(f"{cycle['validation_case_id']}: {size} bytes")
        if len(cycle["query_or_event"]) > 240:
            failures.append(f"{cycle['validation_case_id']}: query_or_event exceeds 240 chars")
    return {"passed": not failures, "failures": failures, "max_record_bytes": MAX_RECORD_BYTES}


def _gate_redaction(cycles: List[Dict[str, Any]], forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    for record in list(cycles) + list(forecasts):
        leaked = SENSITIVE_KEYS.intersection(set(_walk_keys(record)))
        if leaked:
            failures.append(f"{record.get('validation_case_id', record.get('forecast_id'))}: {sorted(leaked)}")
    return {"passed": not failures, "failures": failures}


def _gate_sam_compatibility(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    for cycle in cycles:
        missing = REQUIRED_SAM_KEYS.difference(cycle)
        if missing:
            failures.append(f"{cycle['validation_case_id']}: missing {sorted(missing)}")
        for action_group in ("retrieval_actions", "reasoning_actions", "forecast_actions", "execution_actions"):
            for action in cycle[action_group]:
                if not action.get("operation_types"):
                    failures.append(f"{cycle['validation_case_id']}: action missing operation_types")
    return {"passed": not failures, "failures": failures}


def _gate_forecast_resolution(cycles: List[Dict[str, Any]], forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
    forecast_cycles = [c for c in cycles if c["validation_case_id"] == "FORECAST_TRIGGERED_PULSE"]
    failures = []
    if not forecast_cycles:
        failures.append("missing forecast-triggered cycle")
    for forecast in forecasts:
        if forecast.get("actual_outcome") is None or forecast.get("resolved_at") is None:
            failures.append(f"{forecast['forecast_id']}: unresolved forecast")
    return {"passed": not failures, "failures": failures}


def _gate_learning_boundary(cycles: List[Dict[str, Any]], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    for cycle in cycles:
        for write in cycle["learning_writes"]:
            write_class = write.get("write_class")
            if not write_class:
                failures.append(f"{cycle['validation_case_id']}: missing write_class")
                continue
            decision = classify_learning_write(write_class)
            if write_class == AUDIT_WRITE and decision.authoritative_for_retrieval:
                failures.append(f"{cycle['validation_case_id']}: audit_write is retrieval-authoritative")
    for pattern in patterns:
        if "risk_if_wrong" not in pattern or not pattern["risk_if_wrong"]:
            failures.append(f"{pattern['candidate_id']}: missing risk_if_wrong")
        if "proposed_learning_class" not in pattern:
            failures.append(f"{pattern['candidate_id']}: missing proposed_learning_class")
        if pattern["promotion_status"] == "approved":
            failures.append(f"{pattern['candidate_id']}: automatic approval not allowed")
        if pattern["authoritative_for_retrieval"] or pattern["affects_ranking"] or pattern["mutates_policy"]:
            failures.append(f"{pattern['candidate_id']}: candidate is not advisory-only")
    return {"passed": not failures, "failures": failures}


def run_validation() -> Dict[str, Any]:
    cycles, forecasts, patterns = build_representative_cycles()
    gates = {
        "attention_faithfulness": _gate_attention_faithfulness(cycles),
        "bounded_record": _gate_bounded_record(cycles),
        "redaction": _gate_redaction(cycles, forecasts),
        "sam_compatibility": _gate_sam_compatibility(cycles),
        "forecast_resolution": _gate_forecast_resolution(cycles, forecasts),
        "learning_boundary": _gate_learning_boundary(cycles, patterns),
    }
    return {
        "status": "PASS" if all(gate["passed"] for gate in gates.values()) else "FAIL",
        "representative_query_count": len(cycles),
        "representative_query_set": [case_id for case_id, _, _ in REPRESENTATIVE_QUERY_SET],
        "gates": gates,
        "cycles": cycles,
        "forecast_outcomes": forecasts,
        "pattern_engram_candidates": patterns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MNEMOS CoALA cycle coverage.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--summary-only", action="store_true", help="Print only status and gates.")
    args = parser.parse_args()

    report = run_validation()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    printable: Dict[str, Any]
    if args.summary_only:
        printable = {
            "status": report["status"],
            "representative_query_count": report["representative_query_count"],
            "gates": report["gates"],
        }
    else:
        printable = report
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
