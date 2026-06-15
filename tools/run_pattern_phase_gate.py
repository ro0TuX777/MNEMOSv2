"""
PatternEngramCandidate Phase Gate — Phase 21 validation harness for MNEMOS.

Validates the end-to-end PatternEngramCandidate extraction pipeline across
8 representative scenarios and 5 cross-cutting gate assertions.

Mirrors the structure of tools/run_coala_cycle_validation.py.

Usage
-----
    python tools/run_pattern_phase_gate.py
    python tools/run_pattern_phase_gate.py --output benchmarks/results/pattern_phase_gate.json
    python tools/run_pattern_phase_gate.py --fail-on-gate   # non-zero exit on any failure
    python tools/run_pattern_phase_gate.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnemos.cognitive.assembler import CycleAssembler
from mnemos.cognitive.attention import build_attention_decisions
from mnemos.cognitive.cycle import CognitiveCycleRecord
from mnemos.cognitive.cycle_evaluator import (
    CycleEvaluator,
    QUALITY_GOOD,
    QUALITY_BAD,
    KLOW,
    KHIGH,
)
from mnemos.cognitive.learning_boundary import (
    BLOCKED_PROCEDURAL_MUTATION,
    PROCEDURAL_CHANGE_CANDIDATE,
    SEMANTIC_CANDIDATE_WRITE,
    SEMANTIC_WRITE,
)
from mnemos.cognitive.pattern_engram import PatternEngramCandidate
from mnemos.cognitive.pattern_learner import (
    PatternLearner,
    PatternConsolidator,
    SituationAbstractor,
)
from mnemos.cognitive.pattern_store import PatternCandidateStore
from mnemos.cognitive.promoted_pattern import PatternEngram


# ── Cycle builders ─────────────────────────────────────────────────────────────

def _build_good_class_a() -> CognitiveCycleRecord:
    """
    High-quality CLASS_A (factoid) search cycle.

    Designed to score ≥ KHIGH on the rubric:
    - routing_precision 3: factoid, conf=0.93, completed
    - candidate_efficiency 3: ratio=1.0 (8→8), no suppressions
    - governance_appropriate 3: off + no contradictions
    - forecast_utilisation 2: no forecast signals (neutral)
    - attention_coverage 3: 11 dimensions
    - write_integrity 3: all writes have valid write_class
    """
    meta = {
        "retrieval_mode": "hybrid",
        "fusion_policy": "balanced",
        "fusion_engine": "python_rrf",
        "candidate_envelope": {
            "enabled": True,
            "initial_candidate_count": 8,
            "final_candidate_count": 8,
            "suppression_summary": {
                "duplicate_similarity": 0,
                "source_cap_exceeded": 0,
                "bounded_limit_exceeded": 0,
            },
        },
        "lexical_available": True,
    }
    asm = CycleAssembler(
        trigger_type="search",
        trigger_source="api",
        query_or_event="What is the retention policy for audit logs?",
        active_profile="core_appliance",
        governance_profile="default",
    )
    asm.set_working_memory(
        query="What is the retention policy for audit logs?",
        query_class="factoid",
        query_class_confidence=0.93,
        candidate_count_pre_governance=8,
        candidate_count_post_governance=8,
        suppression_count=0,
        contradiction_count=0,
        active_retrieval_mode="hybrid",
        active_fusion_policy="balanced",
        active_governance_mode="off",
        active_policy_profile="default",
        summary_eligible=False,
        derived_fact_eligible=False,
        forecast_advisory_present=False,
        pre_cognitive_cache_hit=False,
    )
    asm.add_attention_decisions(build_attention_decisions(
        mode_meta=meta,
        governance_mode="off",
        governance_profile="default",
        phase1_enabled=True,
        phase2_enabled=True,
        phase3_enabled=True,
        derive_views_requested=None,
        pre_cognitive_hit=False,
        query_family="factoid",
        query_family_confidence=0.93,
        rerank_applied=False,
        shadow_search_present=False,
        intent_forecast_present=False,
    ))
    asm.add_retrieval_action(
        "python_hybrid_rrf",
        inputs={"query_class": "factoid", "top_k": 8},
        outputs={"candidates_returned": 8},
        latency_ms=14.0,
    )
    from mnemos.cognitive.learning_boundary import EPISODIC_WRITE
    asm.add_learning_write(
        target_memory_type="episodic",
        operation="write",
        write_class=EPISODIC_WRITE,
        triggered_by="cycle_observation",
    )
    return asm.build(selected_route="return")


def _build_bad_class_b() -> CognitiveCycleRecord:
    """
    Low-quality CLASS_B (multi_hop) search cycle.

    Designed to score ≤ KLOW on the rubric:
    - routing_precision 1: multi_hop conf=0.35 (low)
    - candidate_efficiency 0: ratio=30 (oversized pool)
    - governance_appropriate 1: off + 2 contradictions detected
    - forecast_utilisation 2: no forecast (neutral)
    - attention_coverage 0: only 3 dimensions populated
    - write_integrity 3: no writes (vacuous pass)
    Combined expected: ~7 (border or below KLOW)
    """
    meta = {
        "retrieval_mode": "hybrid",
        "fusion_policy": "balanced",
        "fusion_engine": "python_rrf",
        "candidate_envelope": {
            "enabled": True,
            "initial_candidate_count": 30,
            "final_candidate_count": 3,
            "suppression_summary": {
                "duplicate_similarity": 15,
                "source_cap_exceeded": 8,
                "bounded_limit_exceeded": 4,
            },
        },
        "lexical_available": True,
    }
    asm = CycleAssembler(
        trigger_type="search",
        trigger_source="api",
        query_or_event="Multi-hop query requiring synthesis across policy domains.",
        active_profile="core_appliance",
        governance_profile="default",
    )
    asm.set_working_memory(
        query="Multi-hop query requiring synthesis across policy domains.",
        query_class="multi_hop",
        query_class_confidence=0.35,
        candidate_count_pre_governance=30,
        candidate_count_post_governance=3,
        suppression_count=27,
        contradiction_count=2,
        active_retrieval_mode="hybrid",
        active_fusion_policy="balanced",
        active_governance_mode="off",
        active_policy_profile="default",
        summary_eligible=False,
        derived_fact_eligible=False,
        forecast_advisory_present=False,
        pre_cognitive_cache_hit=False,
    )
    # Only populate 3 attention dimensions (sparse coverage → score 0)
    from mnemos.cognitive.cycle import AttentionDecision
    asm.add_attention_decisions([
        AttentionDecision(
            dimension="retrieval_mode",
            decision="hybrid",
            reason="default retrieval mode",
            policy_source="config",
        ),
        AttentionDecision(
            dimension="candidate_envelope",
            decision="enabled",
            reason="candidate envelope active",
            policy_source="config",
        ),
        AttentionDecision(
            dimension="governance_gate",
            decision="off",
            reason="governance disabled for this query",
            policy_source="policy:default",
        ),
    ])
    asm.add_retrieval_action(
        "python_hybrid_rrf",
        inputs={"query_class": "multi_hop", "top_k": 30},
        outputs={"candidates_returned": 3},
        latency_ms=65.0,
    )
    return asm.build(selected_route="return", final_status="completed")


def _build_good_class_a_variant() -> CognitiveCycleRecord:
    """
    Second good CLASS_A cycle with nearly identical situation to the first.
    Used in consolidator_merge scenario.
    """
    meta = {
        "retrieval_mode": "hybrid",
        "fusion_policy": "balanced",
        "fusion_engine": "python_rrf",
        "candidate_envelope": {
            "enabled": True,
            "initial_candidate_count": 8,
            "final_candidate_count": 8,
            "suppression_summary": {
                "duplicate_similarity": 0,
                "source_cap_exceeded": 0,
                "bounded_limit_exceeded": 0,
            },
        },
        "lexical_available": True,
    }
    asm = CycleAssembler(
        trigger_type="search",
        trigger_source="api",
        query_or_event="What is the data classification level for customer PII?",
        active_profile="core_appliance",
        governance_profile="default",
    )
    asm.set_working_memory(
        query="What is the data classification level for customer PII?",
        query_class="factoid",
        query_class_confidence=0.88,
        candidate_count_pre_governance=8,
        candidate_count_post_governance=8,
        suppression_count=0,
        contradiction_count=0,
        active_retrieval_mode="hybrid",
        active_fusion_policy="balanced",
        active_governance_mode="off",
        active_policy_profile="default",
        summary_eligible=False,
        derived_fact_eligible=False,
        forecast_advisory_present=False,
        pre_cognitive_cache_hit=False,
    )
    asm.add_attention_decisions(build_attention_decisions(
        mode_meta=meta,
        governance_mode="off",
        governance_profile="default",
        phase1_enabled=True,
        phase2_enabled=True,
        phase3_enabled=True,
        derive_views_requested=None,
        pre_cognitive_hit=False,
        query_family="factoid",
        query_family_confidence=0.88,
        rerank_applied=False,
        shadow_search_present=False,
        intent_forecast_present=False,
    ))
    asm.add_retrieval_action(
        "python_hybrid_rrf",
        inputs={"query_class": "factoid", "top_k": 8},
        outputs={"candidates_returned": 8},
        latency_ms=12.0,
    )
    from mnemos.cognitive.learning_boundary import EPISODIC_WRITE
    asm.add_learning_write(
        target_memory_type="episodic",
        operation="write",
        write_class=EPISODIC_WRITE,
        triggered_by="cycle_observation",
    )
    return asm.build(selected_route="return")


# ── Scenario runners ───────────────────────────────────────────────────────────

def _scenario_eval_good_class_a() -> Dict[str, Any]:
    """Good CLASS_A cycle → CycleEvaluator → quality_label='good'."""
    cycle = _build_good_class_a()
    evaluator = CycleEvaluator()
    result = evaluator.evaluate(cycle)

    passed = result.quality_label == QUALITY_GOOD
    return {
        "scenario": "eval_good_class_a",
        "passed": passed,
        "quality_label": result.quality_label,
        "aggregate_score": result.aggregate_score,
        "rubric_scores": dict(result.rubric_scores),
        "failure": None if passed else (
            f"Expected quality_label='good' (score≥{KHIGH}), "
            f"got '{result.quality_label}' (score={result.aggregate_score})"
        ),
    }


def _scenario_eval_bad_class_b() -> Dict[str, Any]:
    """Bad CLASS_B cycle with weak routing → quality_label='bad'."""
    cycle = _build_bad_class_b()
    evaluator = CycleEvaluator()
    result = evaluator.evaluate(cycle)

    passed = result.quality_label == QUALITY_BAD
    return {
        "scenario": "eval_bad_class_b",
        "passed": passed,
        "quality_label": result.quality_label,
        "aggregate_score": result.aggregate_score,
        "rubric_scores": dict(result.rubric_scores),
        "failure": None if passed else (
            f"Expected quality_label='bad' (score≤{KLOW}), "
            f"got '{result.quality_label}' (score={result.aggregate_score})"
        ),
    }


def _scenario_learner_descriptive() -> Dict[str, Any]:
    """Good cycle → PatternLearner → PatternEngramCandidate with pattern_type='descriptive'."""
    cycle = _build_good_class_a()
    evaluator = CycleEvaluator()
    evaluation = evaluator.evaluate(cycle)

    abstractor = SituationAbstractor()
    situation = abstractor.abstract(cycle, evaluation)

    learner = PatternLearner()
    candidate = learner.extract(cycle, evaluation, situation)

    passed = (
        candidate.pattern_type == "descriptive"
        and not candidate.authoritative_for_retrieval
        and not candidate.affects_ranking
        and not candidate.mutates_policy
        and candidate.proposed_learning_class == SEMANTIC_CANDIDATE_WRITE
    )
    return {
        "scenario": "learner_descriptive",
        "passed": passed,
        "pattern_type": candidate.pattern_type,
        "proposed_learning_class": candidate.proposed_learning_class,
        "authoritative_for_retrieval": candidate.authoritative_for_retrieval,
        "pattern_summary": candidate.pattern_summary[:120],
        "failure": None if passed else (
            f"Expected pattern_type='descriptive' with SEMANTIC_CANDIDATE_WRITE, "
            f"got type='{candidate.pattern_type}' class='{candidate.proposed_learning_class}'"
        ),
    }


def _scenario_learner_operational() -> Dict[str, Any]:
    """Bad cycle with weak routing/governance → pattern_type='operational_recommendation'."""
    cycle = _build_bad_class_b()
    evaluator = CycleEvaluator()
    evaluation = evaluator.evaluate(cycle)

    abstractor = SituationAbstractor()
    situation = abstractor.abstract(cycle, evaluation)

    learner = PatternLearner()
    candidate = learner.extract(cycle, evaluation, situation)

    passed = (
        candidate.pattern_type == "operational_recommendation"
        and candidate.proposed_learning_class == PROCEDURAL_CHANGE_CANDIDATE
        and not candidate.authoritative_for_retrieval
    )
    return {
        "scenario": "learner_operational",
        "passed": passed,
        "pattern_type": candidate.pattern_type,
        "proposed_learning_class": candidate.proposed_learning_class,
        "weak_dimensions": situation.weak_dimensions,
        "pattern_summary": candidate.pattern_summary[:120],
        "failure": None if passed else (
            f"Expected pattern_type='operational_recommendation', "
            f"got '{candidate.pattern_type}' (weak dims: {situation.weak_dimensions})"
        ),
    }


def _scenario_consolidator_merge() -> Dict[str, Any]:
    """Two near-duplicate situations → merged candidate with ≥ 2 supporting_cycle_ids."""
    evaluator = CycleEvaluator()
    abstractor = SituationAbstractor()
    learner = PatternLearner()
    consolidator = PatternConsolidator()

    cycle_a = _build_good_class_a()
    eval_a = evaluator.evaluate(cycle_a)
    sit_a = abstractor.abstract(cycle_a, eval_a)
    cand_a = learner.extract(cycle_a, eval_a, sit_a)

    cycle_b = _build_good_class_a_variant()
    eval_b = evaluator.evaluate(cycle_b)
    sit_b = abstractor.abstract(cycle_b, eval_b)
    cand_b = learner.extract(cycle_b, eval_b, sit_b)

    merged, action = consolidator.consolidate(
        cand_b,
        [cand_a],
        new_situation=sit_b,
        existing_situations=[sit_a],
    )

    passed = (
        action == "merged"
        and len(merged.supporting_cycle_ids) >= 2
    )
    return {
        "scenario": "consolidator_merge",
        "passed": passed,
        "consolidation_action": action,
        "supporting_cycle_ids_count": len(merged.supporting_cycle_ids),
        "failure": None if passed else (
            f"Expected action='merged' with ≥2 cycle IDs, "
            f"got action='{action}' ids={len(merged.supporting_cycle_ids)}"
        ),
    }


def _scenario_consolidator_contradict() -> Dict[str, Any]:
    """Same situation text but opposite quality → contradicting_engram_ids populated."""
    evaluator = CycleEvaluator()
    abstractor = SituationAbstractor()
    learner = PatternLearner()
    consolidator = PatternConsolidator(merge_threshold=0.3)  # lower threshold to force match

    cycle_good = _build_good_class_a()
    eval_good = evaluator.evaluate(cycle_good)
    sit_good = abstractor.abstract(cycle_good, eval_good)
    cand_good = learner.extract(cycle_good, eval_good, sit_good)

    cycle_bad = _build_bad_class_b()
    eval_bad = evaluator.evaluate(cycle_bad)

    # Build a bad-quality candidate but with a situation text that overlaps
    # with the good one enough to trigger contradiction at the low threshold.
    # We reuse sit_good's text for the bad situation by constructing a SituationSummary
    # with inverted quality.
    from mnemos.cognitive.pattern_learner import SituationSummary
    sit_bad_overlap = SituationSummary(
        trigger_type=sit_good.trigger_type,
        route_class=sit_good.route_class,
        governance_mode=sit_good.governance_mode,
        forecast_active=sit_good.forecast_active,
        candidate_pressure=sit_good.candidate_pressure,
        weak_dimensions=["routing_precision", "governance_appropriate"],
        strong_dimensions=[],
        quality_label=QUALITY_BAD,
        situation_text=sit_good.situation_text,  # identical text → high Jaccard
    )
    cand_bad = PatternEngramCandidate(
        pattern_summary="IF bad version of same situation THEN address routing and governance.",
        supporting_cycle_ids=[cycle_bad.cycle_id],
        confidence_score=0.65,
        support_score=0.0,
        contradiction_score=0.65,
        pattern_type="operational_recommendation",
        recommended_scope="local",
        applies_when=sit_good.situation_text,
        does_not_apply_when="N/A",
        risk_if_wrong="May apply bad recommendations to good situations.",
        proposed_learning_class=PROCEDURAL_CHANGE_CANDIDATE,
    )

    contradicted, action = consolidator.consolidate(
        cand_bad,
        [cand_good],
        new_situation=sit_bad_overlap,
        existing_situations=[sit_good],
    )

    passed = (
        action == "contradicted"
        and len(contradicted.contradicting_engram_ids) > 0
    )
    return {
        "scenario": "consolidator_contradict",
        "passed": passed,
        "consolidation_action": action,
        "contradicting_engram_ids": list(contradicted.contradicting_engram_ids),
        "contradiction_score": contradicted.contradiction_score,
        "failure": None if passed else (
            f"Expected action='contradicted' with populated contradicting_engram_ids, "
            f"got action='{action}' ids={contradicted.contradicting_engram_ids}"
        ),
    }


def _scenario_recall_advisory() -> Dict[str, Any]:
    """Store populated → find_relevant() → top-3 advisory candidates returned."""
    store = PatternCandidateStore()
    evaluator = CycleEvaluator()
    abstractor = SituationAbstractor()
    learner = PatternLearner()

    cycle = _build_good_class_a()
    evaluation = evaluator.evaluate(cycle)
    situation = abstractor.abstract(cycle, evaluation)

    # Populate store with 5 candidates
    for i in range(5):
        cand = PatternEngramCandidate(
            pattern_summary=f"Advisory pattern {i}: efficient factoid retrieval applies_when class a.",
            supporting_cycle_ids=[cycle.cycle_id],
            confidence_score=0.7 + i * 0.04,
            support_score=0.7 + i * 0.04,
            contradiction_score=0.0,
            pattern_type="descriptive",
            recommended_scope="local",
            applies_when=f"A CLASS_A search cycle with off governance pattern {i}.",
            does_not_apply_when="Not CLASS_A.",
            risk_if_wrong="Misapply pattern to wrong class.",
            proposed_learning_class=SEMANTIC_CANDIDATE_WRITE,
        )
        store.add(cand)

    results = store.find_relevant(situation.situation_text, top_k=3)

    passed = (
        len(results) == 3
        and all(not r.authoritative_for_retrieval for r in results)
        and all(not r.affects_ranking for r in results)
    )
    return {
        "scenario": "recall_advisory",
        "passed": passed,
        "candidates_returned": len(results),
        "all_advisory_only": all(not r.authoritative_for_retrieval for r in results),
        "store_total": store.count(),
        "failure": None if passed else (
            f"Expected 3 advisory candidates from store of 5, "
            f"got {len(results)} candidates"
        ),
    }


def _scenario_promotion_gate() -> Dict[str, Any]:
    """
    Full lifecycle: candidate → recommend → approve → PatternEngram with
    governance_review_id set.
    """
    store = PatternCandidateStore()
    cycle = _build_good_class_a()
    evaluator = CycleEvaluator()
    evaluation = evaluator.evaluate(cycle)
    abstractor = SituationAbstractor()
    situation = abstractor.abstract(cycle, evaluation)
    learner = PatternLearner()
    candidate = learner.extract(cycle, evaluation, situation)

    candidate_id = store.add(candidate)

    # Step 1: recommend
    recommended = store.recommend(
        candidate_id,
        gate_id="pattern_phase_gate_21",
        confidence_threshold=0.0,
    )

    # Step 2: promote (approve)
    engram = store.promote(
        candidate_id,
        governance_review_id="gov-review-phase21-001",
        ledger_ref="ledger-phase21-001",
    )

    passed = (
        isinstance(engram, PatternEngram)
        and engram.governance_review_id == "gov-review-phase21-001"
        and not engram.authoritative_for_retrieval
        and not engram.affects_ranking
        and not engram.mutates_policy
        and engram.write_class == SEMANTIC_WRITE
        and len(store.list_promoted()) == 1
    )
    return {
        "scenario": "promotion_gate",
        "passed": passed,
        "pattern_id": engram.pattern_id,
        "governance_review_id": engram.governance_review_id,
        "write_class": engram.write_class,
        "authoritative_for_retrieval": engram.authoritative_for_retrieval,
        "promoted_count": len(store.list_promoted()),
        "failure": None if passed else (
            f"Promotion lifecycle failed: engram={type(engram).__name__} "
            f"gov_id='{engram.governance_review_id}' "
            f"auth={engram.authoritative_for_retrieval}"
        ),
    }


# ── Gate assertions ────────────────────────────────────────────────────────────

def _gate_evaluator_determinism() -> Dict[str, Any]:
    """Same cycle → same score on repeated calls."""
    cycle = _build_good_class_a()
    evaluator = CycleEvaluator()
    result_1 = evaluator.evaluate(cycle)
    result_2 = evaluator.evaluate(cycle)
    result_3 = evaluator.evaluate(cycle)

    passed = (
        result_1.aggregate_score == result_2.aggregate_score == result_3.aggregate_score
        and result_1.quality_label == result_2.quality_label == result_3.quality_label
        and result_1.rubric_scores == result_2.rubric_scores == result_3.rubric_scores
    )
    return {
        "gate": "evaluator_determinism",
        "passed": passed,
        "scores": [result_1.aggregate_score, result_2.aggregate_score, result_3.aggregate_score],
        "labels": [result_1.quality_label, result_2.quality_label, result_3.quality_label],
        "failure": None if passed else "CycleEvaluator produced different scores on repeated calls",
    }


def _gate_safety_invariant() -> Dict[str, Any]:
    """No candidate in any state has authoritative_for_retrieval=True."""
    failures: List[str] = []

    # Create candidates in all lifecycle states
    cycle = _build_good_class_a()
    store = PatternCandidateStore()
    evaluator = CycleEvaluator()
    abstractor = SituationAbstractor()
    learner = PatternLearner()

    evaluation = evaluator.evaluate(cycle)
    situation = abstractor.abstract(cycle, evaluation)

    # pending state
    cand_pending = learner.extract(cycle, evaluation, situation)
    if cand_pending.authoritative_for_retrieval:
        failures.append(f"pending candidate {cand_pending.candidate_id} is authoritative")

    # recommended state
    cid = store.add(cand_pending)
    recommended = store.recommend(cid, gate_id="safety_gate", confidence_threshold=0.0)
    if recommended.authoritative_for_retrieval:
        failures.append(f"recommended candidate {cid} is authoritative")

    # approved / PatternEngram
    engram = store.promote(cid, governance_review_id="gov-safety-001")
    if engram.authoritative_for_retrieval:
        failures.append(f"promoted PatternEngram {engram.pattern_id} is authoritative")
    if engram.affects_ranking:
        failures.append(f"promoted PatternEngram {engram.pattern_id} affects_ranking=True")
    if engram.mutates_policy:
        failures.append(f"promoted PatternEngram {engram.pattern_id} mutates_policy=True")

    passed = not failures
    return {
        "gate": "safety_invariant",
        "passed": passed,
        "failures": failures,
        "failure": None if passed else "; ".join(failures),
    }


def _gate_promotion_boundary() -> Dict[str, Any]:
    """from_approved_candidate() raises PermissionError on non-approved input."""
    failures: List[str] = []

    cycle = _build_good_class_a()
    evaluator = CycleEvaluator()
    abstractor = SituationAbstractor()
    learner = PatternLearner()
    evaluation = evaluator.evaluate(cycle)
    situation = abstractor.abstract(cycle, evaluation)
    candidate = learner.extract(cycle, evaluation, situation)

    # Pending state must raise
    try:
        PatternEngram.from_approved_candidate(candidate, governance_review_id="test")
        failures.append("from_approved_candidate() did not raise on pending candidate")
    except PermissionError:
        pass  # expected

    # Recommended state must raise
    candidate.recommend_promotion(gate_id="boundary_gate", confidence_threshold=0.0)
    try:
        PatternEngram.from_approved_candidate(candidate, governance_review_id="test")
        failures.append("from_approved_candidate() did not raise on recommended candidate")
    except PermissionError:
        pass  # expected

    # Missing governance_review_id must raise
    candidate.approve_promotion(governance_review_id="boundary-001", explicit_approval=True)
    try:
        PatternEngram.from_approved_candidate(candidate, governance_review_id="")
        failures.append("from_approved_candidate() did not raise on empty governance_review_id")
    except ValueError:
        pass  # expected

    passed = not failures
    return {
        "gate": "promotion_boundary",
        "passed": passed,
        "failures": failures,
        "failure": None if passed else "; ".join(failures),
    }


def _gate_blocked_types_never_promoted() -> Dict[str, Any]:
    """
    PatternLearner never produces *_mutation pattern types from good or bad cycles.
    Verifies that policy_mutation, routing_mutation, template_mutation cannot
    reach PatternEngram via the standard extraction pipeline.
    """
    evaluator = CycleEvaluator()
    abstractor = SituationAbstractor()
    learner = PatternLearner()

    mutation_types = {"policy_mutation", "routing_mutation", "template_mutation"}
    violations: List[str] = []

    for build_fn in (_build_good_class_a, _build_bad_class_b, _build_good_class_a_variant):
        cycle = build_fn()
        evaluation = evaluator.evaluate(cycle)
        if evaluation.quality_label == "neutral":
            continue
        situation = abstractor.abstract(cycle, evaluation)
        candidate = learner.extract(cycle, evaluation, situation)
        if candidate.pattern_type in mutation_types:
            violations.append(
                f"{build_fn.__name__}: PatternLearner produced blocked type "
                f"'{candidate.pattern_type}'"
            )

    # Also verify that even manually created *_mutation candidates are marked
    # advisory-only (safety properties are enforced at the class level).
    manual = PatternEngramCandidate(
        pattern_summary="Manually crafted routing mutation candidate.",
        supporting_cycle_ids=["test-cycle-1"],
        confidence_score=0.99,
        support_score=0.99,
        contradiction_score=0.0,
        pattern_type="routing_mutation",
        recommended_scope="local",
        applies_when="Any routing change attempt.",
        does_not_apply_when="N/A",
        risk_if_wrong="Corrupts routing policy if promoted.",
        proposed_learning_class=BLOCKED_PROCEDURAL_MUTATION,
    )
    if manual.authoritative_for_retrieval:
        violations.append("routing_mutation candidate has authoritative_for_retrieval=True")
    if manual.affects_ranking:
        violations.append("routing_mutation candidate has affects_ranking=True")
    if manual.mutates_policy:
        violations.append("routing_mutation candidate has mutates_policy=True")

    passed = not violations
    return {
        "gate": "blocked_types_never_promoted",
        "passed": passed,
        "violations": violations,
        "failure": None if passed else "; ".join(violations),
    }


def _gate_ledger_traceability() -> Dict[str, Any]:
    """Every promoted PatternEngram has a non-empty governance_review_id."""
    store = PatternCandidateStore()
    evaluator = CycleEvaluator()
    abstractor = SituationAbstractor()
    learner = PatternLearner()

    violations: List[str] = []

    for i, build_fn in enumerate((_build_good_class_a, _build_good_class_a_variant)):
        cycle = build_fn()
        evaluation = evaluator.evaluate(cycle)
        situation = abstractor.abstract(cycle, evaluation)
        candidate = learner.extract(cycle, evaluation, situation)
        cid = store.add(candidate)
        store.recommend(cid, gate_id="ledger_gate", confidence_threshold=0.0)
        engram = store.promote(
            cid,
            governance_review_id=f"gov-ledger-{i:03d}",
            ledger_ref=f"ledger-txn-{i:03d}",
        )
        if not engram.governance_review_id:
            violations.append(f"PatternEngram {engram.pattern_id} has empty governance_review_id")

    # Verify that all promoted engrams have review IDs
    for engram in store.list_promoted():
        if not engram.governance_review_id:
            violations.append(f"PatternEngram {engram.pattern_id} missing governance_review_id")

    passed = not violations
    return {
        "gate": "ledger_traceability",
        "passed": passed,
        "promoted_count": len(store.list_promoted()),
        "violations": violations,
        "failure": None if passed else "; ".join(violations),
    }


# ── Orchestrator ───────────────────────────────────────────────────────────────

SCENARIO_RUNNERS = [
    _scenario_eval_good_class_a,
    _scenario_eval_bad_class_b,
    _scenario_learner_descriptive,
    _scenario_learner_operational,
    _scenario_consolidator_merge,
    _scenario_consolidator_contradict,
    _scenario_recall_advisory,
    _scenario_promotion_gate,
]

GATE_RUNNERS = [
    _gate_evaluator_determinism,
    _gate_safety_invariant,
    _gate_promotion_boundary,
    _gate_blocked_types_never_promoted,
    _gate_ledger_traceability,
]


def run_gate() -> Dict[str, Any]:
    scenarios: List[Dict[str, Any]] = []
    for runner in SCENARIO_RUNNERS:
        try:
            result = runner()
        except Exception as exc:  # noqa: BLE001
            result = {
                "scenario": runner.__name__.lstrip("_scenario_"),
                "passed": False,
                "failure": f"Unhandled exception: {type(exc).__name__}: {exc}",
            }
        scenarios.append(result)

    gates: List[Dict[str, Any]] = []
    for runner in GATE_RUNNERS:
        try:
            result = runner()
        except Exception as exc:  # noqa: BLE001
            result = {
                "gate": runner.__name__.lstrip("_gate_"),
                "passed": False,
                "failure": f"Unhandled exception: {type(exc).__name__}: {exc}",
            }
        gates.append(result)

    all_passed = all(s["passed"] for s in scenarios) and all(g["passed"] for g in gates)
    return {
        "status": "PASS" if all_passed else "FAIL",
        "scenario_count": len(scenarios),
        "gate_count": len(gates),
        "scenarios_passed": sum(1 for s in scenarios if s["passed"]),
        "gates_passed": sum(1 for g in gates if g["passed"]),
        "scenarios": scenarios,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 21 — PatternEngramCandidate extraction harness gate."
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit non-zero if any scenario or gate fails.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Print summary only.")
    args = parser.parse_args()

    report = run_gate()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.summary_only:
        printable: Dict[str, Any] = {
            "status": report["status"],
            "scenarios_passed": f"{report['scenarios_passed']}/{report['scenario_count']}",
            "gates_passed": f"{report['gates_passed']}/{report['gate_count']}",
            "scenario_results": [
                {"scenario": s["scenario"], "passed": s["passed"], "failure": s.get("failure")}
                for s in report["scenarios"]
            ],
            "gate_results": [
                {"gate": g["gate"], "passed": g["passed"], "failure": g.get("failure")}
                for g in report["gates"]
            ],
        }
    else:
        printable = report

    print(json.dumps(printable, indent=2, sort_keys=True))
    if args.fail_on_gate and report["status"] != "PASS":
        return 1
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
