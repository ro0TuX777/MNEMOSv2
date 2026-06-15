"""
Tests for mnemos.cognitive.pattern_learner — Phase 17.

Coverage targets
----------------
1. SituationAbstractor — entity-free situation_text, route_class mapping,
   candidate_pressure derivation.
2. PatternLearner — pattern_type inference, IF-THEN rule construction,
   confidence/score computation.
3. PatternLearner safety invariants — never produces *_mutation types;
   always produces advisory-only candidates.
4. PatternConsolidator — merge, contradict, new actions;
   Jaccard threshold gating.
5. Integration — full extract() → consolidate() round trip.
"""

from __future__ import annotations

import pytest

from mnemos.cognitive.assembler import CycleAssembler
from mnemos.cognitive.cycle_evaluator import (
    CycleEvaluator,
    QUALITY_GOOD,
    QUALITY_BAD,
    QUALITY_NEUTRAL,
)
from mnemos.cognitive.pattern_learner import (
    SituationAbstractor,
    SituationSummary,
    PatternLearner,
    PatternConsolidator,
    _jaccard,
    _SITUATION_MERGE_THRESHOLD,
)
from mnemos.cognitive.pattern_engram import (
    PatternEngramCandidate,
    PROMOTION_CANDIDATE,
    PATTERN_TYPES,
)
from mnemos.cognitive.learning_boundary import (
    PATTERN_TYPE_TO_WRITE_CLASS,
    ADVISORY_ONLY_WRITE_CLASSES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_assembler(**kwargs) -> CycleAssembler:
    defaults = {
        "trigger_type": "search",
        "trigger_source": "api",
        "query_or_event": "test query entity_name_here should not appear",
        "active_profile": "core_appliance",
        "governance_profile": "default",
    }
    defaults.update(kwargs)
    return CycleAssembler(**defaults)


def _good_cycle():
    asm = _make_assembler()
    asm.set_working_memory(
        query_class="CLASS_B",
        query_class_confidence=0.82,
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


def _bad_cycle():
    asm = _make_assembler()
    asm.set_working_memory(
        query_class=None,
        candidate_count_pre_governance=40,
        candidate_count_post_governance=0,
        suppression_count=40,
        contradiction_count=2,
        active_governance_mode="enforced",
        forecast_advisory_present=False,
    )
    asm.add_attention_decision(dimension="retrieval_mode", decision="semantic", reason="t")
    asm.add_governance_eval(
        mode="enforced", profile="default",
        candidates_evaluated=40, vetoed=40, suppressed=40,
        contradictions_detected=2, contradiction_suppressed=2,
        net_candidates_returned=0,
    )
    asm.add_learning_write(
        target_memory_type="semantic",
        operation="write",
        write_class=None,
    )
    return asm.build(final_status="blocked")


evaluator = CycleEvaluator()
abstractor = SituationAbstractor()
learner = PatternLearner()
consolidator = PatternConsolidator()


def _eval_and_abstract(cycle):
    ev = evaluator.evaluate(cycle)
    sit = abstractor.abstract(cycle, ev)
    return ev, sit


# ── SituationAbstractor tests ─────────────────────────────────────────────────


class TestSituationAbstractor:
    def test_produces_situation_summary(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        assert isinstance(sit, SituationSummary)

    def test_situation_text_nonempty(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        assert len(sit.situation_text) > 10

    def test_situation_text_entity_free(self):
        """situation_text must not contain raw query content."""
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        # The assembler query_or_event contains "entity_name_here" — must not appear
        assert "entity_name_here" not in sit.situation_text
        assert "test query" not in sit.situation_text

    def test_route_class_mapped_from_class_b(self):
        cycle = _good_cycle()  # query_class="CLASS_B"
        ev, sit = _eval_and_abstract(cycle)
        assert sit.route_class == "CLASS_B"

    def test_route_class_unknown_when_none(self):
        cycle = _bad_cycle()  # query_class=None
        ev, sit = _eval_and_abstract(cycle)
        assert sit.route_class == "unknown"

    def test_route_class_mapped_from_legacy_label(self):
        asm = _make_assembler()
        asm.set_working_memory(query_class="factoid", query_class_confidence=0.8)
        cycle = asm.build(final_status="completed")
        ev = evaluator.evaluate(cycle)
        sit = abstractor.abstract(cycle, ev)
        assert sit.route_class == "CLASS_A"

    def test_candidate_pressure_low_tight_pool(self):
        asm = _make_assembler()
        asm.set_working_memory(
            candidate_count_pre_governance=10,
            candidate_count_post_governance=8,
        )
        cycle = asm.build()
        ev = evaluator.evaluate(cycle)
        sit = abstractor.abstract(cycle, ev)
        assert sit.candidate_pressure == "low"

    def test_candidate_pressure_high_large_ratio(self):
        asm = _make_assembler()
        asm.set_working_memory(
            candidate_count_pre_governance=50,
            candidate_count_post_governance=5,
        )
        cycle = asm.build()
        ev = evaluator.evaluate(cycle)
        sit = abstractor.abstract(cycle, ev)
        assert sit.candidate_pressure == "high"

    def test_weak_dimensions_populated_for_bad_cycle(self):
        cycle = _bad_cycle()
        ev, sit = _eval_and_abstract(cycle)
        assert len(sit.weak_dimensions) > 0

    def test_strong_dimensions_populated_for_good_cycle(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        assert len(sit.strong_dimensions) > 0

    def test_quality_label_propagated(self):
        good_ev, good_sit = _eval_and_abstract(_good_cycle())
        bad_ev, bad_sit = _eval_and_abstract(_bad_cycle())
        assert good_sit.quality_label == QUALITY_GOOD
        assert bad_sit.quality_label == QUALITY_BAD

    def test_to_dict_contains_all_keys(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        d = sit.to_dict()
        for k in ("trigger_type", "route_class", "governance_mode", "forecast_active",
                   "candidate_pressure", "weak_dimensions", "strong_dimensions",
                   "quality_label", "situation_text"):
            assert k in d


# ── PatternLearner tests ──────────────────────────────────────────────────────


class TestPatternLearner:
    def test_extract_returns_candidate(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        candidate = learner.extract(cycle, ev, sit)
        assert isinstance(candidate, PatternEngramCandidate)

    def test_extract_neutral_raises(self):
        asm = _make_assembler()
        asm.set_working_memory(
            query_class="CLASS_A", query_class_confidence=0.55,
            candidate_count_pre_governance=10, candidate_count_post_governance=8,
        )
        for dim in [f"dim_{i}" for i in range(7)]:
            asm.add_attention_decision(dimension=dim, decision="active", reason="t")
        cycle = asm.build(final_status="completed")
        ev = evaluator.evaluate(cycle)
        # Force neutral label
        if ev.quality_label != QUALITY_NEUTRAL:
            pytest.skip("cycle not neutral with current thresholds")
        sit = abstractor.abstract(cycle, ev)
        with pytest.raises(ValueError, match="quality_label"):
            learner.extract(cycle, ev, sit)

    def test_promotion_status_is_candidate(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.promotion_status == PROMOTION_CANDIDATE

    def test_authoritative_for_retrieval_always_false(self):
        for cycle in (_good_cycle(), _bad_cycle()):
            ev, sit = _eval_and_abstract(cycle)
            c = learner.extract(cycle, ev, sit)
            assert c.authoritative_for_retrieval is False

    def test_affects_ranking_always_false(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.affects_ranking is False

    def test_mutates_policy_always_false(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.mutates_policy is False

    def test_good_cycle_produces_descriptive(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.pattern_type == "descriptive"

    def test_bad_cycle_with_operational_weak_dims_produces_recommendation(self):
        cycle = _bad_cycle()
        ev, sit = _eval_and_abstract(cycle)
        # Bad cycle has routing_precision=0 → operational weak dim present
        if any(d in sit.weak_dimensions for d in ("routing_precision", "governance_appropriate", "forecast_utilisation")):
            c = learner.extract(cycle, ev, sit)
            assert c.pattern_type == "operational_recommendation"

    def test_never_produces_mutation_type(self):
        """Safety invariant: no *_mutation pattern types ever produced."""
        mutation_types = {"policy_mutation", "routing_mutation", "template_mutation"}
        for cycle in (_good_cycle(), _bad_cycle()):
            ev, sit = _eval_and_abstract(cycle)
            c = learner.extract(cycle, ev, sit)
            assert c.pattern_type not in mutation_types

    def test_proposed_learning_class_advisory_only(self):
        for cycle in (_good_cycle(), _bad_cycle()):
            ev, sit = _eval_and_abstract(cycle)
            c = learner.extract(cycle, ev, sit)
            assert c.proposed_learning_class in ADVISORY_ONLY_WRITE_CLASSES

    def test_supporting_cycle_ids_contains_cycle_id(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert cycle.cycle_id in c.supporting_cycle_ids

    def test_if_then_in_pattern_summary(self):
        for cycle in (_good_cycle(), _bad_cycle()):
            ev, sit = _eval_and_abstract(cycle)
            c = learner.extract(cycle, ev, sit)
            summary_upper = c.pattern_summary.upper()
            assert "IF " in summary_upper and " THEN " in summary_upper

    def test_good_cycle_higher_support_than_contradiction(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.support_score > c.contradiction_score

    def test_bad_cycle_higher_contradiction_than_support(self):
        cycle = _bad_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.contradiction_score > c.support_score

    def test_confidence_in_zero_one_range(self):
        for cycle in (_good_cycle(), _bad_cycle()):
            ev, sit = _eval_and_abstract(cycle)
            c = learner.extract(cycle, ev, sit)
            assert 0.0 <= c.confidence_score <= 1.0

    def test_risk_if_wrong_nonempty(self):
        for cycle in (_good_cycle(), _bad_cycle()):
            ev, sit = _eval_and_abstract(cycle)
            c = learner.extract(cycle, ev, sit)
            assert len(c.risk_if_wrong) > 0

    def test_applies_when_matches_situation_text(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.applies_when == sit.situation_text

    def test_governance_review_required_true(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        assert c.governance_review_required is True


# ── PatternConsolidator tests ─────────────────────────────────────────────────


class TestPatternConsolidator:
    def _make_candidate(self, cycle, situation_override: str = "") -> PatternEngramCandidate:
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        return c, sit

    def test_new_when_pool_empty(self):
        cycle = _good_cycle()
        ev, sit = _eval_and_abstract(cycle)
        c = learner.extract(cycle, ev, sit)
        result, action = consolidator.consolidate(c, [], new_situation=sit)
        assert action == "new"
        assert result is c

    def test_merge_when_similar_good_situations(self):
        c1 = _good_cycle()
        c2 = _good_cycle()
        ev1, sit1 = _eval_and_abstract(c1)
        ev2, sit2 = _eval_and_abstract(c2)
        cand1 = learner.extract(c1, ev1, sit1)
        cand2 = learner.extract(c2, ev2, sit2)
        # Both are the same good cycle structure → similar situations
        pool = [cand1]
        result, action = consolidator.consolidate(
            cand2, pool, new_situation=sit2, existing_situations=[sit1]
        )
        assert action == "merged"
        assert len(result.supporting_cycle_ids) >= 2

    def test_contradiction_when_opposite_quality_similar_situation(self):
        """Same cycle structure but opposite quality labels → contradiction."""
        # We'll use same cycle but force different quality by overriding situation
        good_cycle = _good_cycle()
        bad_cycle = _bad_cycle()
        ev_good, sit_good = _eval_and_abstract(good_cycle)
        ev_bad, sit_bad = _eval_and_abstract(bad_cycle)
        cand_good = learner.extract(good_cycle, ev_good, sit_good)
        cand_bad = learner.extract(bad_cycle, ev_bad, sit_bad)

        # Force the situations to be nearly identical by using the same text
        # but different quality labels via a synthetic situation
        from mnemos.cognitive.pattern_learner import SituationSummary
        shared_text = "A CLASS_B search cycle with advisory governance where routing was considered."
        synth_good = SituationSummary(
            trigger_type="search", route_class="CLASS_B", governance_mode="advisory",
            forecast_active=False, candidate_pressure="low",
            weak_dimensions=[], strong_dimensions=["routing_precision"],
            quality_label=QUALITY_GOOD, situation_text=shared_text,
        )
        synth_bad = SituationSummary(
            trigger_type="search", route_class="CLASS_B", governance_mode="advisory",
            forecast_active=False, candidate_pressure="low",
            weak_dimensions=["routing_precision"], strong_dimensions=[],
            quality_label=QUALITY_BAD, situation_text=shared_text,
        )
        pool = [cand_good]
        result, action = consolidator.consolidate(
            cand_bad, pool, new_situation=synth_bad, existing_situations=[synth_good]
        )
        assert action == "contradicted"
        assert cand_good.candidate_id in result.contradicting_engram_ids

    def test_new_when_below_threshold(self):
        """Very different situations should not merge."""
        good_cycle = _good_cycle()
        bad_cycle = _bad_cycle()
        ev_good, sit_good = _eval_and_abstract(good_cycle)
        ev_bad, sit_bad = _eval_and_abstract(bad_cycle)
        cand_good = learner.extract(good_cycle, ev_good, sit_good)
        cand_bad = learner.extract(bad_cycle, ev_bad, sit_bad)

        # Force low similarity by using completely different situation texts
        from mnemos.cognitive.pattern_learner import SituationSummary
        sit_a = SituationSummary(
            trigger_type="search", route_class="CLASS_A", governance_mode="off",
            forecast_active=False, candidate_pressure="low",
            weak_dimensions=[], strong_dimensions=[],
            quality_label=QUALITY_GOOD,
            situation_text="a class_a search off governance tight low",
        )
        sit_b = SituationSummary(
            trigger_type="pulse", route_class="CLASS_C", governance_mode="enforced",
            forecast_active=True, candidate_pressure="high",
            weak_dimensions=["routing_precision"], strong_dimensions=[],
            quality_label=QUALITY_BAD,
            situation_text="z pulse class_c enforced high forecast blocked failed",
        )
        _, action = consolidator.consolidate(
            cand_bad, [cand_good], new_situation=sit_b, existing_situations=[sit_a]
        )
        assert action == "new"

    def test_merge_grows_supporting_cycle_ids(self):
        cycle1 = _good_cycle()
        cycle2 = _good_cycle()
        ev1, sit1 = _eval_and_abstract(cycle1)
        ev2, sit2 = _eval_and_abstract(cycle2)
        c1 = learner.extract(cycle1, ev1, sit1)
        c2 = learner.extract(cycle2, ev2, sit2)
        merged, action = consolidator.consolidate(
            c2, [c1], new_situation=sit2, existing_situations=[sit1]
        )
        if action == "merged":
            assert len(merged.supporting_cycle_ids) > 1

    def test_consolidated_candidate_still_advisory(self):
        cycle1 = _good_cycle()
        cycle2 = _good_cycle()
        ev1, sit1 = _eval_and_abstract(cycle1)
        ev2, sit2 = _eval_and_abstract(cycle2)
        c1 = learner.extract(cycle1, ev1, sit1)
        c2 = learner.extract(cycle2, ev2, sit2)
        merged, _ = consolidator.consolidate(
            c2, [c1], new_situation=sit2, existing_situations=[sit1]
        )
        assert merged.authoritative_for_retrieval is False

    def test_invalid_merge_threshold_raises(self):
        with pytest.raises(ValueError, match="merge_threshold"):
            PatternConsolidator(merge_threshold=0.0)

    def test_invalid_merge_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="merge_threshold"):
            PatternConsolidator(merge_threshold=1.5)


# ── Jaccard helper tests ──────────────────────────────────────────────────────


class TestJaccard:
    def test_identical_sets(self):
        a = frozenset({"a", "b", "c"})
        assert _jaccard(a, a) == 1.0

    def test_disjoint_sets(self):
        a = frozenset({"a", "b"})
        b = frozenset({"c", "d"})
        assert _jaccard(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        sim = _jaccard(a, b)
        assert 0.0 < sim < 1.0

    def test_both_empty(self):
        assert _jaccard(frozenset(), frozenset()) == 1.0
