"""
Phase 19 — Advisory Pattern Recall Integration tests.

Coverage
--------
1. CognitiveCycleRecord.advisory_patterns field exists and defaults to [].
2. CognitiveCycleRecord.to_dict() includes advisory_patterns.
3. CognitiveCycleRecord.from_dict() round-trips advisory_patterns.
4. CycleAssembler.add_advisory_patterns() stores candidate summaries.
5. CycleAssembler.add_advisory_patterns() strips authoritative_for_retrieval=True.
6. CycleAssembler.add_advisory_patterns() accepts plain dicts.
7. CycleAssembler.add_advisory_patterns() accumulates across multiple calls.
8. CycleAssembler.build() propagates advisory_patterns into CognitiveCycleRecord.
9. add_advisory_patterns with no candidates leaves list empty.
10. Advisory pattern summary dict has expected keys.
11. build_attention_decisions pattern_advisory=disabled when store not set.
12. build_attention_decisions pattern_advisory=recalled when candidates found.
13. build_attention_decisions pattern_advisory=none when store enabled but 0 candidates.
14. PatternCandidateStore.find_relevant returns only advisory-status candidates.
15. PatternCandidateStore.find_relevant with empty store returns [].
16. Full round-trip: store → find_relevant → add_advisory_patterns → build → to_dict → from_dict.
17. advisory_patterns list in cycle is copies, not live references.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

import pytest

from mnemos.cognitive.assembler import CycleAssembler
from mnemos.cognitive.attention import build_attention_decisions
from mnemos.cognitive.cycle import CognitiveCycleRecord, WorkingMemorySnapshot
from mnemos.cognitive.pattern_engram import (
    PatternEngramCandidate,
    PROMOTION_CANDIDATE,
    PROMOTION_RECOMMENDED,
    PROMOTION_APPROVED,
    PROMOTION_REJECTED,
)
from mnemos.cognitive.pattern_store import PatternCandidateStore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candidate(
    *,
    pattern_summary: str = "CLASS_B hybrid search performs well with advisory governance",
    pattern_type: str = "descriptive",
    confidence_score: float = 0.8,
    applies_when: str = "CLASS_B hybrid advisory governance search cycle",
    status: str = PROMOTION_CANDIDATE,
) -> PatternEngramCandidate:
    # Derive the correct proposed_learning_class from pattern_type
    if pattern_type == "descriptive":
        proposed_learning_class = "semantic_candidate_write"
    elif pattern_type.endswith("_mutation"):
        proposed_learning_class = "blocked_procedural_mutation"
    else:
        proposed_learning_class = "procedural_change_candidate"
    return PatternEngramCandidate(
        pattern_summary=pattern_summary,
        supporting_cycle_ids=[str(uuid.uuid4())],
        supporting_engram_ids=[],
        contradicting_engram_ids=[],
        confidence_score=confidence_score,
        support_score=0.7,
        contradiction_score=0.0,
        pattern_type=pattern_type,
        recommended_scope="local",
        applies_when=applies_when,
        does_not_apply_when="enforced governance mode",
        risk_if_wrong="low",
        proposed_learning_class=proposed_learning_class,
        promotion_status=status,
    )


def _make_assembler() -> CycleAssembler:
    return CycleAssembler(
        trigger_type="search",
        trigger_source="api",
        query_or_event="test query for pattern recall",
        active_profile="default",
        governance_profile="default",
    )


def _base_mode_meta() -> Dict[str, Any]:
    return {
        "retrieval_mode": "semantic",
        "fusion_policy": None,
        "lexical_available": False,
        "candidate_envelope": {"enabled": False},
        "forecast_advisory": {},
    }


# ── 1–3. CognitiveCycleRecord advisory_patterns field ────────────────────────


class TestCognitiveCycleRecordAdvisoryPatterns:
    def test_advisory_patterns_defaults_to_empty(self):
        record = CognitiveCycleRecord(
            cycle_id=str(uuid.uuid4()),
            trigger_type="search",
            trigger_source="api",
            query_or_event="q",
            active_profile="default",
            governance_profile="default",
            working_memory_snapshot=WorkingMemorySnapshot(),
        )
        assert record.advisory_patterns == []

    def test_to_dict_includes_advisory_patterns(self):
        record = CognitiveCycleRecord(
            cycle_id=str(uuid.uuid4()),
            trigger_type="search",
            trigger_source="api",
            query_or_event="q",
            active_profile="default",
            governance_profile="default",
            working_memory_snapshot=WorkingMemorySnapshot(),
            advisory_patterns=[{"candidate_id": "abc", "pattern_summary": "test"}],
        )
        d = record.to_dict()
        assert "advisory_patterns" in d
        assert len(d["advisory_patterns"]) == 1
        assert d["advisory_patterns"][0]["candidate_id"] == "abc"

    def test_from_dict_round_trips_advisory_patterns(self):
        patterns = [
            {
                "candidate_id": "abc123",
                "pattern_summary": "adaptive pattern",
                "pattern_type": "predictive",
                "confidence_score": 0.75,
                "applies_when": "advisory governance",
                "promotion_status": PROMOTION_CANDIDATE,
            }
        ]
        record = CognitiveCycleRecord(
            cycle_id=str(uuid.uuid4()),
            trigger_type="search",
            trigger_source="api",
            query_or_event="q",
            active_profile="default",
            governance_profile="default",
            working_memory_snapshot=WorkingMemorySnapshot(),
            advisory_patterns=patterns,
        )
        d = record.to_dict()
        restored = CognitiveCycleRecord.from_dict(d)
        assert restored.advisory_patterns == patterns


# ── 4–9. CycleAssembler.add_advisory_patterns() ──────────────────────────────


class TestCycleAssemblerAdvisoryPatterns:
    def test_add_candidate_objects(self):
        assembler = _make_assembler()
        cand = _make_candidate()
        assembler.add_advisory_patterns([cand])
        cycle = assembler.build()
        assert len(cycle.advisory_patterns) == 1

    def test_strips_authoritative_for_retrieval(self):
        """add_advisory_patterns must never carry a candidate with authoritative=True."""
        assembler = _make_assembler()
        # Manually construct a dict that has authoritative_for_retrieval=True
        bad_dict = {
            "candidate_id": "bad",
            "pattern_summary": "should be skipped",
            "pattern_type": "descriptive",
            "confidence_score": 0.9,
            "applies_when": "anything",
            "promotion_status": PROMOTION_CANDIDATE,
            "authoritative_for_retrieval": True,
        }
        assembler.add_advisory_patterns([bad_dict])
        cycle = assembler.build()
        assert cycle.advisory_patterns == []

    def test_accepts_plain_dicts(self):
        assembler = _make_assembler()
        d = {
            "candidate_id": "x1",
            "pattern_summary": "plain dict pattern",
            "pattern_type": "predictive",
            "confidence_score": 0.6,
            "applies_when": "semantic mode",
            "promotion_status": PROMOTION_RECOMMENDED,
        }
        assembler.add_advisory_patterns([d])
        cycle = assembler.build()
        assert len(cycle.advisory_patterns) == 1
        assert cycle.advisory_patterns[0]["candidate_id"] == "x1"

    def test_accumulates_across_multiple_calls(self):
        assembler = _make_assembler()
        assembler.add_advisory_patterns([_make_candidate(applies_when="situation a")])
        assembler.add_advisory_patterns([_make_candidate(applies_when="situation b")])
        cycle = assembler.build()
        assert len(cycle.advisory_patterns) == 2

    def test_build_propagates_advisory_patterns(self):
        assembler = _make_assembler()
        cand = _make_candidate(confidence_score=0.88)
        assembler.add_advisory_patterns([cand])
        cycle = assembler.build()
        assert cycle.advisory_patterns[0]["confidence_score"] == pytest.approx(0.88)

    def test_empty_candidates_leaves_empty(self):
        assembler = _make_assembler()
        assembler.add_advisory_patterns([])
        cycle = assembler.build()
        assert cycle.advisory_patterns == []

    def test_summary_dict_has_expected_keys(self):
        assembler = _make_assembler()
        cand = _make_candidate()
        assembler.add_advisory_patterns([cand])
        cycle = assembler.build()
        expected_keys = {
            "candidate_id", "pattern_summary", "pattern_type",
            "confidence_score", "applies_when", "promotion_status",
        }
        assert set(cycle.advisory_patterns[0].keys()) == expected_keys


# ── 10. Advisory pattern isolation ───────────────────────────────────────────


class TestAdvisoryPatternIsolation:
    def test_advisory_patterns_list_is_a_copy(self):
        """Mutations to the cycle's advisory_patterns list must not affect assembler state."""
        assembler = _make_assembler()
        cand = _make_candidate()
        assembler.add_advisory_patterns([cand])
        cycle = assembler.build()
        cycle.advisory_patterns.clear()
        # Build a second cycle — should still have 1 pattern
        cycle2 = assembler.build()
        assert len(cycle2.advisory_patterns) == 1


# ── 11–13. build_attention_decisions pattern_advisory dimension ───────────────


class TestPatternAdvisoryAttentionDimension:
    def _dims(self, decisions: list) -> Dict[str, str]:
        return {d.dimension: d.decision for d in decisions}

    def test_disabled_when_store_not_configured(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            pattern_store_enabled=False,
            pattern_advisory_count=0,
        )
        d = self._dims(decisions)
        assert d["pattern_advisory"] == "disabled"

    def test_recalled_when_candidates_found(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            pattern_store_enabled=True,
            pattern_advisory_count=3,
        )
        d = self._dims(decisions)
        assert d["pattern_advisory"] == "recalled"

    def test_none_when_store_enabled_but_empty(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            pattern_store_enabled=True,
            pattern_advisory_count=0,
        )
        d = self._dims(decisions)
        assert d["pattern_advisory"] == "none"


# ── 14–15. PatternCandidateStore recall filtering ────────────────────────────


class TestPatternStoreRecallFiltering:
    def _populated_store(self) -> PatternCandidateStore:
        store = PatternCandidateStore()
        store.add(_make_candidate(
            applies_when="CLASS_B hybrid advisory governance search",
            status=PROMOTION_CANDIDATE,
        ))
        store.add(_make_candidate(
            applies_when="CLASS_A factoid semantic governance",
            status=PROMOTION_RECOMMENDED,
        ))
        store.add(_make_candidate(
            applies_when="advisory governance enforced",
            status=PROMOTION_APPROVED,
        ))
        store.add(_make_candidate(
            applies_when="rejected pattern advisory",
            status=PROMOTION_REJECTED,
        ))
        return store

    def test_find_relevant_excludes_approved_rejected(self):
        """Only PROMOTION_CANDIDATE and PROMOTION_RECOMMENDED are advisory-eligible."""
        store = self._populated_store()
        # Query that is broad enough to match all 4 texts
        results = store.find_relevant("CLASS_B advisory governance search hybrid", top_k=10)
        statuses = {c.promotion_status for c in results}
        assert PROMOTION_APPROVED not in statuses
        assert PROMOTION_REJECTED not in statuses

    def test_find_relevant_empty_store_returns_empty(self):
        store = PatternCandidateStore()
        results = store.find_relevant("some situation text", top_k=5)
        assert results == []


# ── 16. Full round-trip integration ──────────────────────────────────────────


class TestFullRoundTrip:
    def test_store_recall_to_cycle_to_dict_to_from_dict(self):
        # Build a store with one candidate
        store = PatternCandidateStore()
        cand = _make_candidate(
            applies_when="CLASS_B hybrid advisory governance search cycle",
            confidence_score=0.82,
        )
        store.add(cand)

        # Recall from store
        recalled = store.find_relevant("CLASS_B advisory hybrid search", top_k=3)
        assert len(recalled) >= 1

        # Wire into assembler
        assembler = _make_assembler()
        assembler.add_advisory_patterns(recalled)
        cycle = assembler.build()

        # Verify cycle
        assert len(cycle.advisory_patterns) >= 1
        assert cycle.advisory_patterns[0]["confidence_score"] == pytest.approx(0.82)

        # Round-trip through dict
        d = cycle.to_dict()
        assert "advisory_patterns" in d
        restored = CognitiveCycleRecord.from_dict(d)
        assert restored.advisory_patterns == cycle.advisory_patterns
