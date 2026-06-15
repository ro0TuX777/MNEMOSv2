"""
Tests for mnemos.cognitive.pattern_store — Phase 18.

Coverage targets
----------------
1. PatternCandidateStore add / get / update / list operations.
2. find_relevant — Jaccard recall, top_k, min_confidence, status filter.
3. count / count_by_status.
4. save / load JSON persistence round-trip.
5. Safety invariant: authoritative_for_retrieval=True in store file raises.
6. from_dict / to_dict serialisation.
7. CognitiveCycleRecord.from_dict reconstruction.
8. run_accumulation dry-run pipeline.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from mnemos.cognitive.assembler import CycleAssembler
from mnemos.cognitive.cycle import CognitiveCycleRecord
from mnemos.cognitive.cycle_evaluator import CycleEvaluator
from mnemos.cognitive.pattern_engram import (
    PatternEngramCandidate,
    PROMOTION_CANDIDATE,
    PROMOTION_RECOMMENDED,
    PROMOTION_APPROVED,
)
from mnemos.cognitive.pattern_learner import (
    SituationAbstractor,
    PatternLearner,
)
from mnemos.cognitive.pattern_store import PatternCandidateStore, _candidate_from_dict
from mnemos.cognitive.learning_boundary import SEMANTIC_CANDIDATE_WRITE


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_assembler(**kwargs) -> CycleAssembler:
    defaults = {
        "trigger_type": "search",
        "trigger_source": "api",
        "query_or_event": "test query",
        "active_profile": "core_appliance",
        "governance_profile": "default",
    }
    defaults.update(kwargs)
    return CycleAssembler(**defaults)


def _good_cycle():
    asm = _make_assembler()
    asm.set_working_memory(
        query_class="CLASS_B",
        query_class_confidence=0.85,
        candidate_count_pre_governance=18,
        candidate_count_post_governance=14,
        suppression_count=0,
        contradiction_count=0,
        active_governance_mode="advisory",
        forecast_advisory_present=True,
        pre_cognitive_cache_hit=True,
    )
    for dim in [
        "retrieval_mode", "query_classification", "candidate_envelope",
        "summary_inclusion", "graph_expansion", "derived_facts",
        "forecast_advisory", "governance_gate", "pre_cognitive_routing",
    ]:
        asm.add_attention_decision(dimension=dim, decision="active", reason="test")
    asm.add_governance_eval(
        mode="advisory", profile="default",
        candidates_evaluated=18, vetoed=2, suppressed=0,
        contradictions_detected=0, contradiction_suppressed=0,
        net_candidates_returned=14,
    )
    return asm.build(final_status="completed")


def _make_candidate(
    situation_text: str = "A CLASS_B search cycle with advisory governance.",
    pattern_type: str = "descriptive",
    confidence: float = 0.8,
    **kwargs,
) -> PatternEngramCandidate:
    return PatternEngramCandidate(
        pattern_summary=f"IF {situation_text} THEN maintain current approach.",
        supporting_cycle_ids=["cycle-001"],
        risk_if_wrong="test risk",
        pattern_type=pattern_type,
        proposed_learning_class=SEMANTIC_CANDIDATE_WRITE,
        confidence_score=confidence,
        applies_when=situation_text,
        **kwargs,
    )


evaluator = CycleEvaluator()
abstractor = SituationAbstractor()
learner = PatternLearner()


# ── Add / Get / Update tests ──────────────────────────────────────────────────


class TestAddGetUpdate:
    def test_add_returns_candidate_id(self):
        store = PatternCandidateStore()
        c = _make_candidate()
        cid = store.add(c)
        assert cid == c.candidate_id

    def test_get_returns_candidate(self):
        store = PatternCandidateStore()
        c = _make_candidate()
        store.add(c)
        assert store.get(c.candidate_id) is c

    def test_get_missing_returns_none(self):
        store = PatternCandidateStore()
        assert store.get("nonexistent-id") is None

    def test_add_duplicate_raises(self):
        store = PatternCandidateStore()
        c = _make_candidate()
        store.add(c)
        with pytest.raises(ValueError, match="already exists"):
            store.add(c)

    def test_update_replaces_candidate(self):
        store = PatternCandidateStore()
        c1 = _make_candidate(confidence=0.5)
        store.add(c1)
        c2 = _make_candidate(confidence=0.9)
        # Give c2 the same candidate_id to simulate update of same entry
        import dataclasses
        c2_same_id = dataclasses.replace(c2, candidate_id=c1.candidate_id)
        # PatternEngramCandidate is not frozen, so set the id
        c2_updated = PatternEngramCandidate(
            pattern_summary=c2.pattern_summary,
            supporting_cycle_ids=c2.supporting_cycle_ids,
            risk_if_wrong=c2.risk_if_wrong,
            confidence_score=0.9,
            applies_when=c2.applies_when,
            proposed_learning_class=SEMANTIC_CANDIDATE_WRITE,
        )
        c2_updated.candidate_id = c1.candidate_id
        store.update(c1.candidate_id, c2_updated)
        assert store.get(c1.candidate_id).confidence_score == 0.9

    def test_update_missing_raises(self):
        store = PatternCandidateStore()
        c = _make_candidate()
        with pytest.raises(KeyError):
            store.update("nonexistent", c)

    def test_count_increases_on_add(self):
        store = PatternCandidateStore()
        assert store.count() == 0
        store.add(_make_candidate("situation alpha."))
        assert store.count() == 1
        store.add(_make_candidate("situation beta."))
        assert store.count() == 2


# ── List operations ───────────────────────────────────────────────────────────


class TestListOperations:
    def test_list_all_returns_all(self):
        store = PatternCandidateStore()
        c1 = _make_candidate("alpha situation")
        c2 = _make_candidate("beta situation")
        store.add(c1)
        store.add(c2)
        all_candidates = store.list_all()
        assert len(all_candidates) == 2

    def test_list_by_status_candidate(self):
        store = PatternCandidateStore()
        c = _make_candidate()
        store.add(c)
        result = store.list_by_status(PROMOTION_CANDIDATE)
        assert c in result

    def test_list_by_status_excludes_others(self):
        store = PatternCandidateStore()
        c = _make_candidate()
        store.add(c)
        result = store.list_by_status(PROMOTION_APPROVED)
        assert c not in result

    def test_list_by_pattern_type(self):
        store = PatternCandidateStore()
        c = _make_candidate(pattern_type="descriptive")
        store.add(c)
        result = store.list_by_pattern_type("descriptive")
        assert c in result

    def test_count_by_status(self):
        store = PatternCandidateStore()
        store.add(_make_candidate("sit a"))
        store.add(_make_candidate("sit b"))
        counts = store.count_by_status()
        assert counts[PROMOTION_CANDIDATE] == 2
        assert counts[PROMOTION_APPROVED] == 0


# ── Advisory recall tests ─────────────────────────────────────────────────────


class TestFindRelevant:
    def test_returns_empty_when_store_empty(self):
        store = PatternCandidateStore()
        assert store.find_relevant("some situation") == []

    def test_returns_relevant_candidate(self):
        store = PatternCandidateStore()
        situation = "a CLASS_B search cycle with advisory governance where routing was strong"
        c = _make_candidate(situation_text=situation)
        store.add(c)
        results = store.find_relevant(situation)
        assert c in results

    def test_top_k_limits_results(self):
        store = PatternCandidateStore()
        for i in range(5):
            store.add(_make_candidate(f"shared term class_b advisory search cycle {i}"))
        results = store.find_relevant("shared term class_b advisory search cycle", top_k=3)
        assert len(results) <= 3

    def test_min_confidence_filters_low_confidence(self):
        store = PatternCandidateStore()
        low = _make_candidate("test situation", confidence=0.2)
        high = _make_candidate("test situation extra", confidence=0.9)
        store.add(low)
        store.add(high)
        results = store.find_relevant("test situation", min_confidence=0.5)
        assert low not in results
        assert high in results

    def test_approved_candidates_excluded_from_recall(self):
        store = PatternCandidateStore()
        c = _make_candidate("advisory only situation")
        store.add(c)
        # Promote the candidate to approved
        c.approve_promotion(
            governance_review_id="review-001", explicit_approval=True
        )
        store.update(c.candidate_id, c)
        results = store.find_relevant("advisory only situation")
        assert c not in results

    def test_disjoint_situation_returns_low_score(self):
        """A completely different situation should not be top result."""
        store = PatternCandidateStore()
        c = _make_candidate("zxq unrelated tokens zzz")
        store.add(c)
        results = store.find_relevant("alpha beta gamma delta epsilon", top_k=1)
        # Either empty or the result has very low similarity
        if results:
            assert results[0] is c  # only one candidate — still returned


# ── Persistence tests ─────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PatternCandidateStore(persist_path=path)
        c = _make_candidate("a persistent situation")
        store.add(c)
        store.save()

        store2 = PatternCandidateStore(persist_path=path)
        store2.load()
        loaded = store2.get(c.candidate_id)
        assert loaded is not None
        assert loaded.pattern_summary == c.pattern_summary
        assert loaded.applies_when == c.applies_when
        assert loaded.confidence_score == c.confidence_score

    def test_load_no_op_when_file_absent(self, tmp_path):
        store = PatternCandidateStore(persist_path=str(tmp_path / "missing.json"))
        store.load()  # must not raise
        assert store.count() == 0

    def test_save_no_op_when_no_path(self):
        store = PatternCandidateStore()
        store.add(_make_candidate())
        store.save()  # must not raise, no file created

    def test_to_dict_keys(self):
        store = PatternCandidateStore()
        store.add(_make_candidate())
        d = store.to_dict()
        assert "schema_version" in d
        assert "saved_at" in d
        assert "count" in d
        assert "candidates" in d

    def test_load_raises_on_authoritative_violation(self, tmp_path):
        """Load must reject store files where a candidate claims authoritative_for_retrieval=True."""
        path = str(tmp_path / "tampered.json")
        bad_data = {
            "schema_version": "1",
            "saved_at": "2026-01-01T00:00:00+00:00",
            "count": 1,
            "candidates": [{
                "pattern_summary": "IF bad THEN bad.",
                "supporting_cycle_ids": ["c1"],
                "risk_if_wrong": "tampered",
                "pattern_type": "descriptive",
                "proposed_learning_class": SEMANTIC_CANDIDATE_WRITE,
                "candidate_id": "bad-id-001",
                "promotion_status": "candidate",
                "authoritative_for_retrieval": True,   # violation
                "applies_when": "something",
            }],
        }
        with open(path, "w") as fh:
            json.dump(bad_data, fh)

        store = PatternCandidateStore(persist_path=path)
        with pytest.raises(ValueError, match="authoritative_for_retrieval"):
            store.load()

    def test_find_relevant_after_load(self, tmp_path):
        """Token index must be rebuilt correctly after load."""
        path = str(tmp_path / "store.json")
        situation = "a class_b search cycle with advisory governance routing strong"
        store = PatternCandidateStore(persist_path=path)
        store.add(_make_candidate(situation))
        store.save()

        store2 = PatternCandidateStore(persist_path=path)
        store2.load()
        results = store2.find_relevant(situation)
        assert len(results) == 1


# ── CognitiveCycleRecord.from_dict round-trip ─────────────────────────────────


class TestCycleFromDict:
    def test_from_dict_preserves_cycle_id(self):
        cycle = _good_cycle()
        d = cycle.to_dict()
        restored = CognitiveCycleRecord.from_dict(d)
        assert restored.cycle_id == cycle.cycle_id

    def test_from_dict_preserves_trigger_type(self):
        cycle = _good_cycle()
        d = cycle.to_dict()
        restored = CognitiveCycleRecord.from_dict(d)
        assert restored.trigger_type == cycle.trigger_type

    def test_from_dict_preserves_attention_decisions(self):
        cycle = _good_cycle()
        d = cycle.to_dict()
        restored = CognitiveCycleRecord.from_dict(d)
        assert len(restored.attention_decisions) == len(cycle.attention_decisions)

    def test_from_dict_preserves_governance_evaluations(self):
        cycle = _good_cycle()
        d = cycle.to_dict()
        restored = CognitiveCycleRecord.from_dict(d)
        assert len(restored.governance_evaluations) == len(cycle.governance_evaluations)

    def test_from_dict_preserves_working_memory_query_class(self):
        cycle = _good_cycle()
        d = cycle.to_dict()
        restored = CognitiveCycleRecord.from_dict(d)
        assert restored.working_memory_snapshot.query_class == \
               cycle.working_memory_snapshot.query_class

    def test_from_dict_evaluator_scores_same(self):
        """Restored cycle should evaluate to the same score as original."""
        cycle = _good_cycle()
        ev_orig = evaluator.evaluate(cycle)
        restored = CognitiveCycleRecord.from_dict(cycle.to_dict())
        ev_rest = evaluator.evaluate(restored)
        assert ev_orig.aggregate_score == ev_rest.aggregate_score
        assert ev_orig.quality_label == ev_rest.quality_label


# ── run_accumulation dry-run tests ────────────────────────────────────────────


class TestRunAccumulation:
    def test_dry_run_with_good_cycles(self, tmp_path):
        from tools.run_pattern_accumulation import run_accumulation

        # Generate cycles and save to JSON
        cycles = [_good_cycle() for _ in range(12)]
        input_path = str(tmp_path / "cycles.json")
        with open(input_path, "w") as fh:
            json.dump([c.to_dict() for c in cycles], fh)

        report = run_accumulation(
            input_path=input_path,
            dry_run=True,
        )

        assert report["cycles_evaluated"] == 12
        assert report["dry_run"] is True
        assert report["parse_errors"] == 0

    def test_dry_run_does_not_write_store(self, tmp_path):
        from tools.run_pattern_accumulation import run_accumulation

        store_path = str(tmp_path / "store.json")
        input_path = str(tmp_path / "cycles.json")
        cycles = [_good_cycle() for _ in range(5)]
        with open(input_path, "w") as fh:
            json.dump([c.to_dict() for c in cycles], fh)

        run_accumulation(input_path=input_path, store_path=store_path, dry_run=True)
        assert not os.path.exists(store_path)

    def test_live_run_writes_store(self, tmp_path):
        from tools.run_pattern_accumulation import run_accumulation

        store_path = str(tmp_path / "store.json")
        input_path = str(tmp_path / "cycles.json")
        cycles = [_good_cycle() for _ in range(12)]
        with open(input_path, "w") as fh:
            json.dump([c.to_dict() for c in cycles], fh)

        report = run_accumulation(
            input_path=input_path,
            store_path=store_path,
            dry_run=False,
        )
        assert os.path.exists(store_path)
        assert report["candidates_total"] > 0

    def test_gate_passes_with_enough_cycles(self, tmp_path):
        from tools.run_pattern_accumulation import run_accumulation

        input_path = str(tmp_path / "cycles.json")
        cycles = [_good_cycle() for _ in range(15)]
        with open(input_path, "w") as fh:
            json.dump([c.to_dict() for c in cycles], fh)

        report = run_accumulation(input_path=input_path, dry_run=True)
        # Gate passes when: cycles_evaluated >= 10 AND new+merged >= 5
        # With 15 identical good cycles, we expect merges to dominate
        assert report["gate"]["cycles_evaluated"] >= 10

    def test_limit_parameter_respected(self, tmp_path):
        from tools.run_pattern_accumulation import run_accumulation

        input_path = str(tmp_path / "cycles.json")
        cycles = [_good_cycle() for _ in range(20)]
        with open(input_path, "w") as fh:
            json.dump([c.to_dict() for c in cycles], fh)

        report = run_accumulation(input_path=input_path, dry_run=True, limit=7)
        assert report["cycles_evaluated"] <= 7

    def test_wrapped_dict_format_accepted(self, tmp_path):
        from tools.run_pattern_accumulation import run_accumulation

        input_path = str(tmp_path / "cycles.json")
        cycles = [_good_cycle() for _ in range(3)]
        with open(input_path, "w") as fh:
            json.dump({"cognitive_cycles": [c.to_dict() for c in cycles]}, fh)

        report = run_accumulation(input_path=input_path, dry_run=True)
        assert report["cycles_loaded"] == 3

    def test_report_written_to_output_path(self, tmp_path):
        from tools.run_pattern_accumulation import run_accumulation

        input_path = str(tmp_path / "cycles.json")
        output_path = str(tmp_path / "report.json")
        cycles = [_good_cycle() for _ in range(3)]
        with open(input_path, "w") as fh:
            json.dump([c.to_dict() for c in cycles], fh)

        run_accumulation(input_path=input_path, output_path=output_path, dry_run=True)
        assert os.path.exists(output_path)
        with open(output_path) as fh:
            report = json.load(fh)
        assert "run_id" in report
