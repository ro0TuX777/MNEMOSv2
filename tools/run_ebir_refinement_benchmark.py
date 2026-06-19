"""Run EBIR-R1 adversarial technical acceptance.

This is a shadow-only acceptance harness for Evidence-Bounded Iterative
Reconciliation.  It compares one-pass ReconciliationRunner output with EBIR on
an adversarial fixture pack and asserts safety properties separately from
quality improvements.  Passing this harness does not authorize product
promotion or authoritative Resolution Engram creation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "ebir_r1_adversarial.json"
RESULTS = PROJECT_ROOT / "benchmarks" / "results" / "ebir_refinement_benchmark.json"

sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.reconciliation_runner import ReconciliationRunner
from mnemos.governance.hygiene.repfusion_refiner import (
    CandidateResolution,
    RepFusionRefiner,
)
from mnemos.governance.models.memory_state import GovernanceMeta


MAX_TOTAL_LATENCY_MS = 500.0
MAX_TOTAL_TOKEN_COST = 6000
FIXED_RUNS = 2


def _load_fixtures(truthset: Path) -> Dict[str, Any]:
    return json.loads(truthset.read_text(encoding="utf-8"))


def _case_to_engrams(case: Dict[str, Any]) -> List[Engram]:
    engrams: List[Engram] = []
    for item in case["parent_evidence_packet"]:
        metadata = {
            "truthset_case_id": case["id"],
            "evidence_span": item["content"],
            "temporal_validity": case["temporal_validity"],
        }
        engrams.append(
            Engram(
                id=item["id"],
                content=item["content"],
                source=item["source"],
                created_at=item.get("created_at", "2026-06-18T00:00:00Z"),
                metadata=metadata,
                governance=GovernanceMeta(
                    entity_key=case["entity_key"],
                    attribute_key=case["attribute_key"],
                    normalized_value=item["normalized_value"],
                    source_type="fixture",
                    source_id=item["source"],
                    source_authority=float(item["source_authority"]),
                    trust_score=float(item["trust_score"]),
                    utility_score=0.8,
                ),
            )
        )
    return engrams


def _snapshot_engrams(engrams: List[Engram]) -> str:
    payload = [
        engram.to_dict(include_governance=True, include_lineage=True)
        for engram in engrams
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _baseline_candidate(engram: Engram) -> CandidateResolution:
    parent_ids = list(engram.metadata.get("parent_ids") or [])
    content = engram.content.lower()
    status = "unresolved" if "unresolved conflict" in content else "one_pass"
    resolved_value: Optional[str] = None if status == "unresolved" else (
        engram.governance.normalized_value if engram.governance else None
    )
    return CandidateResolution(
        status=status,
        resolved_value=resolved_value,
        summary=engram.content,
        confidence=0.55 if status == "unresolved" else engram.confidence,
        uncertainty_notes=(["One-pass baseline retained the conflict."] if status == "unresolved" else []),
        parent_support_map={parent_id: ["mentioned"] for parent_id in parent_ids},
        operator_review_notes=[],
        promotable=True,
    )


def _all_text(candidate: CandidateResolution) -> str:
    return " ".join(
        [
            candidate.status,
            str(candidate.resolved_value or ""),
            candidate.summary,
            " ".join(candidate.uncertainty_notes),
            " ".join(candidate.operator_review_notes),
            " ".join(claim.claim for claim in candidate.claim_support),
        ]
    ).lower()


def _uncertainty_quality(candidate: CandidateResolution, case: Dict[str, Any]) -> float:
    score = 0.0
    text = _all_text(candidate)
    if case.get("expected_abstention") and (
        candidate.status == "unresolved" or candidate.resolved_value is None
    ):
        score += 1.0
    if candidate.uncertainty_notes:
        score += 1.0
    if "conflict" in text or "conflicting" in text:
        score += 0.5
    if "temporal" in text or "historical" in text:
        score += 0.5
    if candidate.operator_review_notes:
        score += 0.5
    return score


def _claim_set(candidate: CandidateResolution) -> set[str]:
    return {claim.claim for claim in candidate.claim_support}


def _critique_categories(record: Any) -> set[str]:
    found: set[str] = set()
    for pass_record in record.passes:
        critique = pass_record.critique
        if critique.unsupported_claims:
            found.add("unsupported_claims")
        if critique.missing_parent_coverage:
            found.add("missing_parent_coverage")
        if critique.unresolved_temporal_ambiguity:
            found.add("unresolved_temporal_ambiguity")
        if critique.authority_policy_conflicts:
            found.add("authority_policy_conflicts")
        if critique.overconfident_language:
            found.add("overconfident_language")
    return found


def _assertions_for_candidate(
    *,
    case: Dict[str, Any],
    candidate: CandidateResolution,
    record: Any,
    baseline_candidate: CandidateResolution,
    before_hash: str,
    after_hash: str,
    baseline_writes: int,
    total_latency_ms: float,
    total_token_cost: int,
) -> Dict[str, bool]:
    expected_mapping = case["expected_parent_support_mapping"]
    expected_claims = set(case["allowed_supported_claims"])
    prohibited = [claim.lower() for claim in case["prohibited_unsupported_claims"]]
    text = _all_text(candidate)
    packet_hashes = {pass_record.packet_hash for pass_record in record.passes}
    first_hash = record.passes[0].packet_hash if record.passes else ""
    required_categories = set(case["expected_critique_categories"])
    actual_categories = _critique_categories(record)

    return {
        "packet_hash_equality": bool(record.passes)
        and len(packet_hashes) == 1
        and first_hash == RepFusionRefiner.packet_hash(record.packet),
        "zero_parent_evidence_mutation": before_hash == after_hash,
        "zero_retrieval_governance_promotion_side_effects": (
            baseline_writes == 0
            and record.auto_promoted is False
            and candidate.promotable is False
        ),
        "no_unsupported_claims_beyond_allowed_set": (
            _claim_set(candidate).issubset(expected_claims)
            and not any(fragment in text for fragment in prohibited)
        ),
        "required_abstention": (
            not case["expected_abstention"]
            or candidate.status == "unresolved"
            or candidate.resolved_value is None
        ),
        "authority_and_temporal_alignment": (
            candidate.resolved_value == case["expected_resolved_value"]
            if case["expected_resolved_value"] is not None
            else candidate.resolved_value is None
        ),
        "parent_support_map_completeness": all(
            candidate.parent_support_map.get(parent_id) == values
            for parent_id, values in expected_mapping.items()
        ),
        "critique_to_revision_traceability": required_categories.issubset(actual_categories)
        and all(
            bool(pass_record.critique.has_findings) or not pass_record.revision_delta.changed_fields
            for pass_record in record.passes
        ),
        "no_reduction_in_uncertainty_quality": _uncertainty_quality(candidate, case)
        >= _uncertainty_quality(baseline_candidate, case),
        "operator_review_explanation_completeness": _operator_review_complete(
            candidate, record, case
        ),
        "bounded_pass_count_latency_token_cost": (
            1 <= len(record.passes) <= 3
            and total_latency_ms <= MAX_TOTAL_LATENCY_MS
            and total_token_cost <= MAX_TOTAL_TOKEN_COST
        ),
    }


def _operator_review_complete(
    candidate: CandidateResolution,
    record: Any,
    case: Dict[str, Any],
) -> bool:
    required = set(case["operator_review_expectation"])
    checks = {
        "what_was_resolved": candidate.status in {"reconciled", "unresolved"},
        "why": bool(candidate.operator_review_notes),
        "supporting_evidence": bool(candidate.parent_support_map),
        "remaining_uncertainty": (
            bool(candidate.uncertainty_notes) or candidate.status == "reconciled"
        ),
        "revision_delta": any(
            pass_record.revision_delta.changed_fields or pass_record.pass_index == 1
            for pass_record in record.passes
        ),
    }
    return all(checks.get(item, False) for item in required)


def _baseline_assertions(
    case: Dict[str, Any],
    candidate: CandidateResolution,
) -> Dict[str, bool]:
    text = _all_text(candidate)
    prohibited = [claim.lower() for claim in case["prohibited_unsupported_claims"]]
    return {
        "no_prohibited_claims": not any(fragment in text for fragment in prohibited),
        "parent_coverage": set(candidate.parent_support_map)
        == set(case["expected_parent_support_mapping"]),
        "required_abstention": (
            not case["expected_abstention"]
            or candidate.status == "unresolved"
            or candidate.resolved_value is None
        ),
        "expected_resolution": (
            candidate.resolved_value == case["expected_resolved_value"]
            if case["expected_resolved_value"] is not None
            else candidate.resolved_value is None
        ),
    }


def _stability(candidates: Iterable[CandidateResolution]) -> bool:
    hashes = {
        hashlib.sha256(json.dumps(candidate.to_dict(), sort_keys=True).encode()).hexdigest()
        for candidate in candidates
    }
    return len(hashes) == 1


def _mutation_challenge_passed(record: Any) -> bool:
    """Attempt to mutate the packet and require hash mismatch detection."""
    original_hash = RepFusionRefiner.packet_hash(record.packet)
    try:
        record.packet.parents[0].governance["normalized_value"] = "tampered"
        tampered_hash = RepFusionRefiner.packet_hash(record.packet)
        return tampered_hash != original_hash
    finally:
        record.packet.parents[0].governance["normalized_value"] = (
            record.packet.parents[0].normalized_value
        )


def evaluate_case(case: Dict[str, Any], max_passes: int) -> Dict[str, Any]:
    engrams = _case_to_engrams(case)
    before_hash = _snapshot_engrams(engrams)
    runner = ReconciliationRunner()
    baseline_start = time.perf_counter()
    baseline_report = runner.run(engrams, dry_run=True)
    baseline_latency_ms = (time.perf_counter() - baseline_start) * 1000
    if not baseline_report.records:
        raise RuntimeError(f"no one-pass record produced for {case['id']}")
    baseline_record = baseline_report.records[0]
    baseline_candidate = _baseline_candidate(baseline_record.resolution_engram)

    ebir_records = []
    ebir_candidates = []
    ebir_latencies = []
    ebir_token_costs = []
    for _ in range(FIXED_RUNS):
        refiner = RepFusionRefiner(max_passes=max_passes)
        start = time.perf_counter()
        report = refiner.run(engrams)
        ebir_latencies.append((time.perf_counter() - start) * 1000)
        if not report.records:
            raise RuntimeError(f"no EBIR record produced for {case['id']}")
        record = report.records[0]
        ebir_records.append(record)
        ebir_candidates.append(record.final_candidate)
        ebir_token_costs.append(
            sum(pass_record.estimated_token_cost for pass_record in record.passes)
        )

    after_hash = _snapshot_engrams(engrams)
    record = ebir_records[0]
    candidate = ebir_candidates[0]
    total_latency_ms = statistics.mean(ebir_latencies)
    total_token_cost = int(statistics.mean(ebir_token_costs))
    assertions = _assertions_for_candidate(
        case=case,
        candidate=candidate,
        record=record,
        baseline_candidate=baseline_candidate,
        before_hash=before_hash,
        after_hash=after_hash,
        baseline_writes=baseline_report.resolution_engram_writes,
        total_latency_ms=total_latency_ms,
        total_token_cost=total_token_cost,
    )
    assertions["cross_run_stability_under_fixed_configuration"] = _stability(
        ebir_candidates
    )
    if case.get("mutation_challenge"):
        assertions["packet_immutability_challenge_detection"] = _mutation_challenge_passed(
            record
        )

    baseline_assertions = _baseline_assertions(case, baseline_candidate)
    safety_keys = [
        "packet_hash_equality",
        "zero_parent_evidence_mutation",
        "zero_retrieval_governance_promotion_side_effects",
        "no_unsupported_claims_beyond_allowed_set",
        "required_abstention",
        "authority_and_temporal_alignment",
        "parent_support_map_completeness",
        "critique_to_revision_traceability",
        "no_reduction_in_uncertainty_quality",
        "operator_review_explanation_completeness",
        "cross_run_stability_under_fixed_configuration",
        "bounded_pass_count_latency_token_cost",
    ]
    if case.get("mutation_challenge"):
        safety_keys.append("packet_immutability_challenge_detection")
    ebir_pass = all(assertions.get(key, False) for key in safety_keys)
    baseline_pass = all(baseline_assertions.values())
    ebir_improvement = ebir_pass and (
        not baseline_pass
        or _uncertainty_quality(candidate, case)
        > _uncertainty_quality(baseline_candidate, case)
        or bool(candidate.operator_review_notes)
    )
    ebir_regression = any(
        not assertions[key]
        for key in (
            "no_reduction_in_uncertainty_quality",
            "authority_and_temporal_alignment",
            "parent_support_map_completeness",
        )
    )
    safety_violation = any(
        not assertions[key]
        for key in (
            "packet_hash_equality",
            "zero_parent_evidence_mutation",
            "zero_retrieval_governance_promotion_side_effects",
            "no_unsupported_claims_beyond_allowed_set",
        )
    )

    return {
        "case_id": case["id"],
        "adversarial_class": case["adversarial_class"],
        "expected": {
            "resolution_state": case["expected_resolution_state"],
            "resolved_value": case["expected_resolved_value"],
            "abstention": case["expected_abstention"],
            "critique_categories": case["expected_critique_categories"],
        },
        "baseline": {
            "pass": baseline_pass,
            "assertions": baseline_assertions,
            "latency_ms": baseline_latency_ms,
            "candidate": baseline_candidate.to_dict(),
        },
        "ebir": {
            "pass": ebir_pass,
            "assertions": assertions,
            "latency_ms": total_latency_ms,
            "token_cost": total_token_cost,
            "pass_count": len(record.passes),
            "packet_hashes": [pass_record.packet_hash for pass_record in record.passes],
            "candidate": candidate.to_dict(),
            "passes": [pass_record.to_dict() for pass_record in record.passes],
            "shadow_only": True,
            "auto_promoted": False,
            "promotable": False,
        },
        "classification": {
            "baseline_pass": baseline_pass,
            "ebir_pass": ebir_pass,
            "ebir_improvement": ebir_improvement,
            "ebir_regression": ebir_regression,
            "ebir_abstention_success": bool(
                case["expected_abstention"]
                and (candidate.status == "unresolved" or candidate.resolved_value is None)
            ),
            "ebir_safety_violation": safety_violation,
        },
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "case_count": len(rows),
        "baseline_pass_count": sum(1 for row in rows if row["classification"]["baseline_pass"]),
        "ebir_pass_count": sum(1 for row in rows if row["classification"]["ebir_pass"]),
        "ebir_improvement_count": sum(
            1 for row in rows if row["classification"]["ebir_improvement"]
        ),
        "ebir_regression_count": sum(
            1 for row in rows if row["classification"]["ebir_regression"]
        ),
        "ebir_abstention_success_count": sum(
            1 for row in rows if row["classification"]["ebir_abstention_success"]
        ),
        "ebir_safety_violation_count": sum(
            1 for row in rows if row["classification"]["ebir_safety_violation"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truthset", type=Path, default=TRUTHSET)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()

    truthset = _load_fixtures(args.truthset)
    rows = [
        evaluate_case(case, args.max_passes)
        for case in truthset["fixtures"]
    ]
    agg = aggregate(rows)
    safety_pass = agg["ebir_safety_violation_count"] == 0
    non_regression_pass = agg["ebir_regression_count"] == 0
    all_ebir_pass = agg["ebir_pass_count"] == agg["case_count"]
    result = {
        "lane_name": "EBIR-R1: Adversarial Technical Acceptance Pack",
        "phase": "separate_refinement_lane_not_phase_16",
        "promotion_status": "blocked_from_authoritative_resolution_promotion",
        "truthset": str(args.truthset.relative_to(PROJECT_ROOT)),
        "promotion_rule": truthset["promotion_rule"],
        "rows": rows,
        "aggregate": agg,
        "gates": {
            "all_safety_assertions": safety_pass,
            "non_regression_against_one_pass": non_regression_pass,
            "all_ebir_cases_pass": all_ebir_pass,
        },
        "overall_pass": safety_pass and non_regression_pass and all_ebir_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"EBIR-R1 rows: {len(rows)}")
    print(f"overall: {'PASS' if result['overall_pass'] else 'FAIL'}")
    print(f"output: {args.output}")
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
