"""
Tests for mnemos.cognitive.cycle_evaluator — Phase 16.

Coverage targets
----------------
1. CycleEvaluationRecord schema and serialisation.
2. CycleEvaluator threshold validation.
3. routing_precision dimension — all four score levels.
4. candidate_efficiency dimension — all score levels incl. cache-hit shortcut.
5. governance_appropriate dimension — off/advisory/enforced modes.
6. forecast_utilisation dimension — with/without forecast signals.
7. attention_coverage dimension — dimension count thresholds.
8. write_integrity dimension — valid/None/unknown write classes.
9. Quality label assignment — good / bad / neutral from aggregate score.
10. evaluate_batch — preserves order, one record per cycle.
11. filter_by_quality — correct filtering and bad-label guard.
12. Determinism — same cycle always produces the same score.
"""

from __future__ import annotations

import pytest

from mnemos.cognitive.assembler import CycleAssembler
from mnemos.cognitive.cycle_evaluator import (
    KLOW,
    KHIGH,
    QUALITY_BAD,
    QUALITY_GOOD,
    QUALITY_NEUTRAL,
    CycleEvaluator,
)
from mnemos.cognitive.learning_boundary import (
    SEMANTIC_CANDIDATE_WRITE,
    AUDIT_WRITE,
)


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
    """Build a cycle that should score 'good' across all dimensions."""
    asm = _make_assembler()
    asm.set_working_memory(
        query_class="CLASS_B",
        query_class_confidence=0.85,
        candidate_count_pre_governance=20,
        candidate_count_post_governance=15,
        suppression_count=0,
        contradiction_count=0,
        active_governance_mode="advisory",
        forecast_advisory_present=False,
    )
    # 9 attention dimensions — full coverage
    for dim in [
        "retrieval_mode", "query_classification", "candidate_envelope",
        "summary_inclusion", "graph_expansion", "derived_facts",
        "forecast_advisory", "governance_gate", "pre_cognitive_routing",
    ]:
        asm.add_attention_decision(
            dimension=dim, decision="active", reason="test"
        )
    asm.add_governance_eval(
        mode="advisory", profile="default",
        candidates_evaluated=20, vetoed=0, suppressed=0,
        contradictions_detected=0, contradiction_suppressed=0,
        net_candidates_returned=15,
    )
    asm.add_learning_write(
        target_memory_type="semantic",
        operation="write",
        write_class=SEMANTIC_CANDIDATE_WRITE,
    )
    return asm.build(final_status="completed")


def _bad_cycle():
    """Build a cycle that should score 'bad' across multiple dimensions."""
    asm = _make_assembler()
    asm.set_working_memory(
        query_class=None,                        # routing_precision → 0
        candidate_count_pre_governance=50,
        candidate_count_post_governance=0,       # candidate_efficiency → 0
        suppression_count=50,
        active_governance_mode="enforced",       # enforced + zero net → 0
        contradiction_count=0,
        forecast_advisory_present=False,
    )
    # 2 attention dimensions — insufficient coverage → 0
    asm.add_attention_decision(dimension="retrieval_mode", decision="semantic", reason="test")
    asm.add_attention_decision(dimension="governance_gate", decision="off", reason="test")
    asm.add_governance_eval(
        mode="enforced", profile="default",
        candidates_evaluated=50, vetoed=50, suppressed=50,
        contradictions_detected=0, contradiction_suppressed=0,
        net_candidates_returned=0,
    )
    # write with None write_class → write_integrity → 0
    asm.add_learning_write(
        target_memory_type="semantic",
        operation="write",
        write_class=None,
    )
    return asm.build(final_status="blocked")


# ── CycleEvaluationRecord tests ───────────────────────────────────────────────


class TestCycleEvaluationRecord:
    def test_to_dict_keys(self):
        cycle = _good_cycle()
        ev = CycleEvaluator().evaluate(cycle)
        d = ev.to_dict()
        for key in (
            "cycle_id", "trigger_type", "rubric_scores", "aggregate_score",
            "quality_label", "reasons", "evaluator_version", "evaluated_at",
        ):
            assert key in d, f"missing key: {key}"

    def test_rubric_scores_all_dimensions_present(self):
        cycle = _good_cycle()
        ev = CycleEvaluator().evaluate(cycle)
        expected_dims = {
            "routing_precision", "candidate_efficiency", "governance_appropriate",
            "forecast_utilisation", "attention_coverage", "write_integrity",
        }
        assert set(ev.rubric_scores.keys()) == expected_dims

    def test_aggregate_equals_sum_of_scores(self):
        cycle = _good_cycle()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.aggregate_score == sum(ev.rubric_scores.values())

    def test_all_scores_in_range(self):
        for cycle in (_good_cycle(), _bad_cycle()):
            ev = CycleEvaluator().evaluate(cycle)
            for dim, score in ev.rubric_scores.items():
                assert 0 <= score <= 3, f"{dim} score {score} out of range"

    def test_reasons_all_dimensions_present(self):
        cycle = _good_cycle()
        ev = CycleEvaluator().evaluate(cycle)
        assert set(ev.reasons.keys()) == set(ev.rubric_scores.keys())

    def test_frozen_record_immutable(self):
        cycle = _good_cycle()
        ev = CycleEvaluator().evaluate(cycle)
        with pytest.raises((AttributeError, TypeError)):
            ev.aggregate_score = 99  # type: ignore[misc]


# ── CycleEvaluator construction tests ────────────────────────────────────────


class TestCycleEvaluatorInit:
    def test_default_thresholds(self):
        ev = CycleEvaluator()
        assert ev.klow == KLOW
        assert ev.khigh == KHIGH

    def test_custom_thresholds_accepted(self):
        ev = CycleEvaluator(klow=4, khigh=14)
        assert ev.klow == 4
        assert ev.khigh == 14

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError, match="Invalid thresholds"):
            CycleEvaluator(klow=10, khigh=5)

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError, match="Invalid thresholds"):
            CycleEvaluator(klow=9, khigh=9)


# ── routing_precision dimension ───────────────────────────────────────────────


class TestRoutingPrecision:
    def test_score_3_high_confidence_completed(self):
        asm = _make_assembler()
        asm.set_working_memory(query_class="CLASS_A", query_class_confidence=0.90)
        cycle = asm.build(final_status="completed")
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["routing_precision"] == 3

    def test_score_2_moderate_confidence(self):
        asm = _make_assembler()
        asm.set_working_memory(query_class="CLASS_B", query_class_confidence=0.60)
        cycle = asm.build(final_status="completed")
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["routing_precision"] == 2

    def test_score_1_low_confidence(self):
        asm = _make_assembler()
        asm.set_working_memory(query_class="CLASS_C", query_class_confidence=0.30)
        cycle = asm.build(final_status="completed")
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["routing_precision"] == 1

    def test_score_0_no_query_class(self):
        asm = _make_assembler()
        asm.set_working_memory(query_class=None)
        cycle = asm.build(final_status="completed")
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["routing_precision"] == 0

    def test_score_0_blocked_status(self):
        asm = _make_assembler()
        asm.set_working_memory(query_class="CLASS_A", query_class_confidence=0.95)
        cycle = asm.build(final_status="blocked")
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["routing_precision"] == 0


# ── candidate_efficiency dimension ────────────────────────────────────────────


class TestCandidateEfficiency:
    def test_score_3_cache_hit(self):
        asm = _make_assembler()
        asm.set_working_memory(
            pre_cognitive_cache_hit=True,
            candidate_count_pre_governance=0,
            candidate_count_post_governance=0,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["candidate_efficiency"] == 3

    def test_score_3_tight_pool_no_suppression(self):
        asm = _make_assembler()
        asm.set_working_memory(
            candidate_count_pre_governance=10,
            candidate_count_post_governance=8,
            suppression_count=0,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["candidate_efficiency"] == 3

    def test_score_0_all_candidates_suppressed(self):
        asm = _make_assembler()
        asm.set_working_memory(
            candidate_count_pre_governance=30,
            candidate_count_post_governance=0,
            suppression_count=30,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["candidate_efficiency"] == 0

    def test_score_1_wide_pool(self):
        asm = _make_assembler()
        asm.set_working_memory(
            candidate_count_pre_governance=20,
            candidate_count_post_governance=5,
            suppression_count=0,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["candidate_efficiency"] == 1


# ── governance_appropriate dimension ─────────────────────────────────────────


class TestGovernanceAppropriate:
    def test_score_3_off_no_contradictions(self):
        asm = _make_assembler()
        asm.set_working_memory(
            active_governance_mode="off",
            contradiction_count=0,
            suppression_count=0,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["governance_appropriate"] == 3

    def test_score_1_off_with_contradictions(self):
        asm = _make_assembler()
        asm.set_working_memory(
            active_governance_mode="off",
            contradiction_count=3,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["governance_appropriate"] == 1

    def test_score_3_enforced_with_suppressions(self):
        asm = _make_assembler()
        asm.set_working_memory(
            active_governance_mode="enforced",
            suppression_count=5,
        )
        asm.add_governance_eval(
            mode="enforced", profile="default",
            candidates_evaluated=15, vetoed=5, suppressed=5,
            contradictions_detected=0, contradiction_suppressed=0,
            net_candidates_returned=10,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["governance_appropriate"] == 3

    def test_score_0_enforced_eliminates_all(self):
        asm = _make_assembler()
        asm.set_working_memory(active_governance_mode="enforced", suppression_count=10)
        asm.add_governance_eval(
            mode="enforced", profile="default",
            candidates_evaluated=10, vetoed=10, suppressed=10,
            contradictions_detected=0, contradiction_suppressed=0,
            net_candidates_returned=0,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["governance_appropriate"] == 0

    def test_score_3_advisory_with_vetoes(self):
        asm = _make_assembler()
        asm.set_working_memory(active_governance_mode="advisory")
        asm.add_governance_eval(
            mode="advisory", profile="default",
            candidates_evaluated=10, vetoed=2, suppressed=0,
            contradictions_detected=0, contradiction_suppressed=0,
            net_candidates_returned=10,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["governance_appropriate"] == 3


# ── attention_coverage dimension ─────────────────────────────────────────────


class TestAttentionCoverage:
    def test_score_3_nine_or_more_dimensions(self):
        asm = _make_assembler()
        for i in range(9):
            asm.add_attention_decision(dimension=f"dim_{i}", decision="active", reason="test")
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["attention_coverage"] == 3

    def test_score_2_seven_dimensions(self):
        asm = _make_assembler()
        for i in range(7):
            asm.add_attention_decision(dimension=f"dim_{i}", decision="active", reason="test")
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["attention_coverage"] == 2

    def test_score_0_fewer_than_four_dimensions(self):
        asm = _make_assembler()
        asm.add_attention_decision(dimension="retrieval_mode", decision="semantic", reason="test")
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["attention_coverage"] == 0

    def test_duplicate_dimensions_counted_once(self):
        asm = _make_assembler()
        for _ in range(10):
            asm.add_attention_decision(dimension="retrieval_mode", decision="semantic", reason="dup")
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        # 10 records but only 1 unique dimension → score 0
        assert ev.rubric_scores["attention_coverage"] == 0


# ── write_integrity dimension ─────────────────────────────────────────────────


class TestWriteIntegrity:
    def test_score_3_no_writes(self):
        asm = _make_assembler()
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["write_integrity"] == 3

    def test_score_3_all_valid_write_classes(self):
        asm = _make_assembler()
        asm.add_learning_write(
            target_memory_type="semantic",
            operation="write",
            write_class=SEMANTIC_CANDIDATE_WRITE,
        )
        asm.add_learning_write(
            target_memory_type="episodic",
            operation="write",
            write_class=AUDIT_WRITE,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["write_integrity"] == 3

    def test_score_0_write_class_none(self):
        asm = _make_assembler()
        asm.add_learning_write(
            target_memory_type="semantic",
            operation="write",
            write_class=None,
        )
        cycle = asm.build()
        ev = CycleEvaluator().evaluate(cycle)
        assert ev.rubric_scores["write_integrity"] == 0


# ── Quality label and batch tests ─────────────────────────────────────────────


class TestQualityLabels:
    def test_good_cycle_labelled_good(self):
        ev = CycleEvaluator().evaluate(_good_cycle())
        assert ev.quality_label == QUALITY_GOOD

    def test_bad_cycle_labelled_bad(self):
        ev = CycleEvaluator().evaluate(_bad_cycle())
        assert ev.quality_label == QUALITY_BAD

    def test_klow_boundary_is_bad(self):
        ev = CycleEvaluator()
        assert ev._label(KLOW) == QUALITY_BAD

    def test_khigh_boundary_is_good(self):
        ev = CycleEvaluator()
        assert ev._label(KHIGH) == QUALITY_GOOD

    def test_midpoint_is_neutral(self):
        ev = CycleEvaluator()
        mid = (KLOW + KHIGH) // 2
        assert ev._label(mid) == QUALITY_NEUTRAL


class TestEvaluateBatch:
    def test_batch_returns_one_per_cycle(self):
        cycles = [_good_cycle(), _bad_cycle(), _good_cycle()]
        records = CycleEvaluator().evaluate_batch(cycles)
        assert len(records) == 3

    def test_batch_preserves_cycle_ids(self):
        cycles = [_good_cycle(), _bad_cycle()]
        records = CycleEvaluator().evaluate_batch(cycles)
        assert records[0].cycle_id == cycles[0].cycle_id
        assert records[1].cycle_id == cycles[1].cycle_id

    def test_batch_empty_returns_empty(self):
        assert CycleEvaluator().evaluate_batch([]) == []


class TestFilterByQuality:
    def test_filter_good(self):
        records = CycleEvaluator().evaluate_batch([_good_cycle(), _bad_cycle()])
        good = CycleEvaluator().filter_by_quality(records, QUALITY_GOOD)
        assert all(r.quality_label == QUALITY_GOOD for r in good)

    def test_filter_bad(self):
        records = CycleEvaluator().evaluate_batch([_good_cycle(), _bad_cycle()])
        bad = CycleEvaluator().filter_by_quality(records, QUALITY_BAD)
        assert all(r.quality_label == QUALITY_BAD for r in bad)

    def test_filter_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unknown quality label"):
            CycleEvaluator().filter_by_quality([], "excellent")


class TestDeterminism:
    def test_same_cycle_same_score_repeated(self):
        cycle = _good_cycle()
        ev = CycleEvaluator()
        scores = [ev.evaluate(cycle).aggregate_score for _ in range(5)]
        assert len(set(scores)) == 1, "evaluate() is not deterministic"

    def test_same_cycle_same_quality_label(self):
        cycle = _bad_cycle()
        ev = CycleEvaluator()
        labels = [ev.evaluate(cycle).quality_label for _ in range(5)]
        assert len(set(labels)) == 1
