import pytest

from mnemos.cognitive.learning_boundary import (
    AUDIT_WRITE,
    BLOCKED_PROCEDURAL_MUTATION,
    PROCEDURAL_CHANGE_CANDIDATE,
    SEMANTIC_CANDIDATE_WRITE,
    classify_learning_write,
    classify_pattern_candidate,
)
from mnemos.cognitive.pattern_engram import PatternEngramCandidate


def test_audit_write_is_append_only_and_non_authoritative():
    decision = classify_learning_write(AUDIT_WRITE)
    assert decision.mutation_allowed is True
    assert decision.authoritative_for_retrieval is False
    assert decision.advisory_only is False


def test_pattern_type_mapping_splits_semantic_and_procedural_candidates():
    descriptive = classify_pattern_candidate("descriptive")
    operational = classify_pattern_candidate("operational_recommendation")
    mutation = classify_pattern_candidate("routing_mutation")

    assert descriptive.write_class == SEMANTIC_CANDIDATE_WRITE
    assert operational.write_class == PROCEDURAL_CHANGE_CANDIDATE
    assert mutation.write_class == BLOCKED_PROCEDURAL_MUTATION
    assert mutation.mutation_allowed is False


def test_pattern_candidate_requires_risk_and_learning_class_alignment():
    with pytest.raises(ValueError, match="risk_if_wrong"):
        PatternEngramCandidate(
            pattern_summary="missing risk",
            supporting_cycle_ids=["cycle-1"],
            proposed_learning_class=SEMANTIC_CANDIDATE_WRITE,
        )

    with pytest.raises(ValueError, match="proposed_learning_class"):
        PatternEngramCandidate(
            pattern_summary="wrong class",
            supporting_cycle_ids=["cycle-1"],
            pattern_type="operational_recommendation",
            risk_if_wrong="could change behavior incorrectly",
            proposed_learning_class=SEMANTIC_CANDIDATE_WRITE,
        )


def test_pattern_candidate_is_advisory_non_ranking_non_mutating():
    candidate = PatternEngramCandidate(
        pattern_summary="bounded envelopes help evidence bundles",
        supporting_cycle_ids=["cycle-1", "cycle-2"],
        confidence_score=0.95,
        pattern_type="operational_recommendation",
        risk_if_wrong="could over-route simple lookups",
        proposed_learning_class=PROCEDURAL_CHANGE_CANDIDATE,
    )

    candidate.recommend_promotion(gate_id="test_gate", confidence_threshold=0.9)
    d = candidate.to_dict()

    assert d["promotion_status"] == "promotion_recommended"
    assert d["governance_review_required"] is True
    assert d["authoritative_for_retrieval"] is False
    assert d["affects_ranking"] is False
    assert d["mutates_policy"] is False


def test_explicit_governance_required_for_promotion():
    candidate = PatternEngramCandidate(
        pattern_summary="descriptive pattern",
        supporting_cycle_ids=["cycle-1"],
        risk_if_wrong="could create an unsupported abstraction",
        proposed_learning_class=SEMANTIC_CANDIDATE_WRITE,
    )
    with pytest.raises(PermissionError):
        candidate.approve_promotion(governance_review_id="review-1", explicit_approval=False)

    candidate.approve_promotion(governance_review_id="review-1", explicit_approval=True)
    assert candidate.to_dict()["promotion_status"] == "approved"
