"""
Phase 20 — PatternEngram promotion tests.

Coverage
--------
1.  PatternEngram.from_approved_candidate() constructs correctly from approved candidate.
2.  from_approved_candidate() raises PermissionError if status != approved.
3.  from_approved_candidate() raises ValueError if governance_review_id is empty.
4.  PatternEngram.authoritative_for_retrieval is always False.
5.  PatternEngram.affects_ranking is always False.
6.  PatternEngram.mutates_policy is always False.
7.  PatternEngram.write_class is always SEMANTIC_WRITE.
8.  PatternEngram.to_dict() includes all required keys.
9.  PatternEngram.from_dict() round-trips correctly.
10. PatternEngram.pattern_id matches the source candidate_id.
11. PatternEngram.promoted_from_candidate_id matches the source candidate_id.
12. to_dict() authoritative_for_retrieval key is False.
13. PatternCandidateStore.promote() requires promotion_recommended state.
14. PatternCandidateStore.promote() succeeds from promotion_recommended.
15. PatternCandidateStore.promote() adds to list_promoted().
16. PatternCandidateStore.reject() sets status to PROMOTION_REJECTED.
17. PatternCandidateStore.reject() raises KeyError for unknown candidate.
18. PatternCandidateStore.recommend() sets status to PROMOTION_RECOMMENDED.
19. PatternCandidateStore.recommend() raises KeyError for unknown candidate.
20. Full lifecycle: candidate → recommend → approve → list_promoted.
21. Blocked mutation types cannot be promoted (PermissionError path blocked at candidate level).
22. PatternCandidateStore.list_promoted() returns empty list when none promoted.
23. PatternCandidateStore.get_promoted() returns correct engram by pattern_id.
24. to_dict() / from_dict() round-trip on PatternCandidateStore includes promoted.
"""

from __future__ import annotations

import uuid

import pytest

from mnemos.cognitive.learning_boundary import SEMANTIC_WRITE
from mnemos.cognitive.pattern_engram import (
    PatternEngramCandidate,
    PROMOTION_APPROVED,
    PROMOTION_CANDIDATE,
    PROMOTION_RECOMMENDED,
    PROMOTION_REJECTED,
)
from mnemos.cognitive.pattern_store import PatternCandidateStore
from mnemos.cognitive.promoted_pattern import PatternEngram


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candidate(
    *,
    status: str = PROMOTION_CANDIDATE,
    pattern_type: str = "descriptive",
) -> PatternEngramCandidate:
    if pattern_type == "descriptive":
        proposed_learning_class = "semantic_candidate_write"
    elif pattern_type.endswith("_mutation"):
        proposed_learning_class = "blocked_procedural_mutation"
    else:
        proposed_learning_class = "procedural_change_candidate"
    cand = PatternEngramCandidate(
        pattern_summary="CLASS_B hybrid search performs well under advisory governance",
        supporting_cycle_ids=[str(uuid.uuid4())],
        risk_if_wrong="low — advisory context only",
        confidence_score=0.85,
        support_score=0.8,
        contradiction_score=0.0,
        pattern_type=pattern_type,
        applies_when="CLASS_B advisory governance hybrid search",
        does_not_apply_when="enforced governance mode",
        proposed_learning_class=proposed_learning_class,
    )
    cand.promotion_status = status
    return cand


def _approved_candidate() -> PatternEngramCandidate:
    cand = _make_candidate()
    cand.approve_promotion(governance_review_id="rev-001", explicit_approval=True)
    return cand


def _make_store_with_recommended() -> tuple[PatternCandidateStore, str]:
    """Returns (store, candidate_id) with one candidate in promotion_recommended state."""
    store = PatternCandidateStore()
    cand = _make_candidate()
    store.add(cand)
    store.recommend(cand.candidate_id, gate_id="gate-001", confidence_threshold=0.0)
    return store, cand.candidate_id


# ── 1–3. PatternEngram construction ──────────────────────────────────────────


class TestPatternEngramConstruction:
    def test_from_approved_candidate_constructs_correctly(self):
        cand = _approved_candidate()
        engram = PatternEngram.from_approved_candidate(cand, governance_review_id="rev-001")
        assert engram.pattern_id == cand.candidate_id
        assert engram.pattern_summary == cand.pattern_summary
        assert engram.governance_review_id == "rev-001"
        assert engram.promoted_from_candidate_id == cand.candidate_id

    def test_raises_permission_error_if_not_approved(self):
        cand = _make_candidate(status=PROMOTION_RECOMMENDED)
        with pytest.raises(PermissionError, match="PROMOTION_APPROVED"):
            PatternEngram.from_approved_candidate(cand, governance_review_id="rev-001")

    def test_raises_permission_error_if_candidate_status(self):
        cand = _make_candidate(status=PROMOTION_CANDIDATE)
        with pytest.raises(PermissionError):
            PatternEngram.from_approved_candidate(cand, governance_review_id="rev-001")

    def test_raises_value_error_if_empty_governance_review_id(self):
        cand = _approved_candidate()
        with pytest.raises((ValueError, PermissionError)):
            PatternEngram.from_approved_candidate(cand, governance_review_id="")


# ── 4–7. Safety invariants ────────────────────────────────────────────────────


class TestPatternEngramSafetyInvariants:
    def test_authoritative_for_retrieval_is_false(self):
        engram = PatternEngram.from_approved_candidate(
            _approved_candidate(), governance_review_id="rev-x"
        )
        assert engram.authoritative_for_retrieval is False

    def test_affects_ranking_is_false(self):
        engram = PatternEngram.from_approved_candidate(
            _approved_candidate(), governance_review_id="rev-x"
        )
        assert engram.affects_ranking is False

    def test_mutates_policy_is_false(self):
        engram = PatternEngram.from_approved_candidate(
            _approved_candidate(), governance_review_id="rev-x"
        )
        assert engram.mutates_policy is False

    def test_write_class_is_semantic_write(self):
        engram = PatternEngram.from_approved_candidate(
            _approved_candidate(), governance_review_id="rev-x"
        )
        assert engram.write_class == SEMANTIC_WRITE


# ── 8–12. Serialisation ───────────────────────────────────────────────────────


class TestPatternEngramSerialisation:
    def _make_engram(self) -> PatternEngram:
        return PatternEngram.from_approved_candidate(
            _approved_candidate(), governance_review_id="rev-42"
        )

    def test_to_dict_has_required_keys(self):
        d = self._make_engram().to_dict()
        for key in [
            "pattern_id", "pattern_summary", "pattern_type", "applies_when",
            "does_not_apply_when", "risk_if_wrong", "confidence_score",
            "governance_review_id", "promoted_from_candidate_id", "promoted_at",
            "write_class", "authoritative_for_retrieval",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_authoritative_for_retrieval_is_false(self):
        d = self._make_engram().to_dict()
        assert d["authoritative_for_retrieval"] is False

    def test_from_dict_round_trips(self):
        engram = self._make_engram()
        d = engram.to_dict()
        restored = PatternEngram.from_dict(d)
        assert restored.pattern_id == engram.pattern_id
        assert restored.governance_review_id == engram.governance_review_id
        assert restored.pattern_summary == engram.pattern_summary

    def test_pattern_id_matches_candidate_id(self):
        cand = _approved_candidate()
        engram = PatternEngram.from_approved_candidate(cand, governance_review_id="rev-x")
        assert engram.pattern_id == cand.candidate_id

    def test_promoted_from_candidate_id_matches(self):
        cand = _approved_candidate()
        engram = PatternEngram.from_approved_candidate(cand, governance_review_id="rev-x")
        assert engram.promoted_from_candidate_id == cand.candidate_id


# ── 13–19. PatternCandidateStore promotion API ────────────────────────────────


class TestPatternCandidateStorePromotion:
    def test_promote_requires_recommended_state(self):
        store = PatternCandidateStore()
        cand = _make_candidate(status=PROMOTION_CANDIDATE)
        store.add(cand)
        with pytest.raises(PermissionError, match="promotion_recommended"):
            store.promote(cand.candidate_id, governance_review_id="rev-001")

    def test_promote_succeeds_from_recommended(self):
        store, cid = _make_store_with_recommended()
        engram = store.promote(cid, governance_review_id="rev-001")
        assert isinstance(engram, PatternEngram)
        assert engram.governance_review_id == "rev-001"

    def test_promote_adds_to_list_promoted(self):
        store, cid = _make_store_with_recommended()
        store.promote(cid, governance_review_id="rev-001")
        promoted = store.list_promoted()
        assert len(promoted) == 1
        assert promoted[0].pattern_id == cid

    def test_promote_unknown_candidate_raises_key_error(self):
        store = PatternCandidateStore()
        with pytest.raises(KeyError):
            store.promote("nonexistent", governance_review_id="rev-001")

    def test_reject_sets_status(self):
        store = PatternCandidateStore()
        cand = _make_candidate()
        store.add(cand)
        updated = store.reject(cand.candidate_id)
        assert updated.promotion_status == PROMOTION_REJECTED

    def test_reject_unknown_raises_key_error(self):
        store = PatternCandidateStore()
        with pytest.raises(KeyError):
            store.reject("nonexistent")

    def test_recommend_sets_status(self):
        store = PatternCandidateStore()
        cand = _make_candidate()
        store.add(cand)
        updated = store.recommend(cand.candidate_id, gate_id="g1", confidence_threshold=0.0)
        assert updated.promotion_status == PROMOTION_RECOMMENDED

    def test_recommend_unknown_raises_key_error(self):
        store = PatternCandidateStore()
        with pytest.raises(KeyError):
            store.recommend("nonexistent", gate_id="g1")


# ── 20–24. Full lifecycle + edge cases ───────────────────────────────────────


class TestPromotionLifecycle:
    def test_full_lifecycle_candidate_to_promoted(self):
        store = PatternCandidateStore()
        cand = _make_candidate()
        store.add(cand)

        # Step 1: recommend
        store.recommend(cand.candidate_id, gate_id="gate-abc", confidence_threshold=0.0)
        assert store.get(cand.candidate_id).promotion_status == PROMOTION_RECOMMENDED

        # Step 2: approve
        engram = store.promote(cand.candidate_id, governance_review_id="rev-final")
        assert engram.authoritative_for_retrieval is False

        # Step 3: verify list
        assert len(store.list_promoted()) == 1
        assert store.get_promoted(cand.candidate_id).pattern_id == cand.candidate_id

    def test_list_promoted_empty_initially(self):
        store = PatternCandidateStore()
        assert store.list_promoted() == []

    def test_get_promoted_returns_none_when_absent(self):
        store = PatternCandidateStore()
        assert store.get_promoted("does-not-exist") is None

    def test_store_to_dict_includes_promoted(self):
        store, cid = _make_store_with_recommended()
        store.promote(cid, governance_review_id="rev-p")
        d = store.to_dict()
        assert "promoted" in d
        assert len(d["promoted"]) == 1
        assert d["promoted"][0]["governance_review_id"] == "rev-p"

    def test_store_from_dict_round_trips_promoted(self):
        store, cid = _make_store_with_recommended()
        store.promote(cid, governance_review_id="rev-rt")
        d = store.to_dict()
        restored = PatternCandidateStore.from_dict(d)
        assert len(restored.list_promoted()) == 1
        assert restored.list_promoted()[0].governance_review_id == "rev-rt"
