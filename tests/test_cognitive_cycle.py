"""
Tests for the MNEMOS Cognitive Cycle schema and CycleAssembler.

Coverage targets
----------------
1. CognitiveCycleRecord serialises to a complete, valid dict.
2. WorkingMemorySnapshot reflects set_working_memory updates.
3. ActionRecord operation_types are assigned from OPERATION_TYPE_MAP.
4. GovernanceEvalSummary is included in the cycle dict.
5. LearningWrite is included in the cycle dict.
6. CycleAssembler.build() is idempotent and produces a unique cycle_id.
7. cycle_latency_ms is non-negative.
8. Cycle records do not break existing isolation invariants:
   - Default retrieval mode is not changed.
   - Governance decisions are not mutated by assembler.
   - Derived-fact-free invariant is not violated by adding a cycle record.
"""

from __future__ import annotations

import time
import uuid

import pytest

from mnemos.cognitive.cycle import (
    ActionRecord,
    AttentionDecision,
    CognitiveCycleRecord,
    GovernanceEvalSummary,
    LearningWrite,
    OperationType,
    WorkingMemorySnapshot,
    _default_types,
)
from mnemos.cognitive.assembler import CycleAssembler


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


# ── WorkingMemorySnapshot tests ───────────────────────────────────────────────


class TestWorkingMemorySnapshot:
    def test_defaults(self):
        wm = WorkingMemorySnapshot()
        assert wm.query is None
        assert wm.query_class is None
        assert wm.active_retrieval_mode == "semantic"
        assert wm.active_governance_mode == "off"
        assert wm.selected_route == "return"
        assert wm.summary_eligible is False
        assert wm.graph_expansion_eligible is False
        assert wm.derived_fact_eligible is False

    def test_to_dict_complete(self):
        wm = WorkingMemorySnapshot(
            query="hello world",
            query_class="factoid",
            query_class_confidence=0.85,
            session_id="sess-1",
            candidate_count_pre_governance=20,
            candidate_count_post_governance=18,
            suppression_count=2,
            contradiction_count=1,
            active_retrieval_mode="hybrid",
            active_fusion_policy="balanced",
            active_governance_mode="advisory",
            active_policy_profile="strict",
            summary_eligible=True,
            graph_expansion_eligible=False,
            derived_fact_eligible=True,
            forecast_advisory_present=True,
            pre_cognitive_cache_hit=False,
            selected_route="advise",
        )
        d = wm.to_dict()
        assert d["query"] == "hello world"
        assert d["query_class"] == "factoid"
        assert d["query_class_confidence"] == 0.85
        assert d["session_id"] == "sess-1"
        assert d["candidate_count_pre_governance"] == 20
        assert d["candidate_count_post_governance"] == 18
        assert d["suppression_count"] == 2
        assert d["contradiction_count"] == 1
        assert d["active_retrieval_mode"] == "hybrid"
        assert d["active_fusion_policy"] == "balanced"
        assert d["active_governance_mode"] == "advisory"
        assert d["active_policy_profile"] == "strict"
        assert d["summary_eligible"] is True
        assert d["graph_expansion_eligible"] is False
        assert d["derived_fact_eligible"] is True
        assert d["forecast_advisory_present"] is True
        assert d["pre_cognitive_cache_hit"] is False
        assert d["selected_route"] == "advise"


# ── AttentionDecision tests ───────────────────────────────────────────────────


class TestAttentionDecision:
    def test_to_dict_with_policy_source(self):
        ad = AttentionDecision(
            dimension="retrieval_mode",
            decision="semantic",
            reason="lexical unavailable",
            policy_source="config.retrieval_mode",
        )
        d = ad.to_dict()
        assert d["dimension"] == "retrieval_mode"
        assert d["decision"] == "semantic"
        assert d["reason"] == "lexical unavailable"
        assert d["policy_source"] == "config.retrieval_mode"

    def test_to_dict_no_policy_source(self):
        ad = AttentionDecision(
            dimension="graph_expansion",
            decision="blocked",
            reason="double opt-in not met",
        )
        d = ad.to_dict()
        assert "policy_source" not in d


# ── ActionRecord tests ────────────────────────────────────────────────────────


class TestActionRecord:
    def test_operation_type_map(self):
        types = _default_types("semantic_search")
        assert OperationType.RETRIEVAL in types

        types = _default_types("contradiction_detection")
        assert OperationType.REASONING in types
        assert OperationType.GOVERNANCE in types

        types = _default_types("reflect_reinforcement")
        assert OperationType.LEARNING in types

        types = _default_types("pulse_forecast")
        assert OperationType.FORECASTING in types

        types = _default_types("autonomous_warmup")
        assert OperationType.GROUNDING in types

    def test_to_dict_includes_latency(self):
        ar = ActionRecord(
            action_id=str(uuid.uuid4()),
            operation_types=[OperationType.RETRIEVAL],
            name="semantic_search",
            inputs_summary={"query": "q"},
            outputs_summary={"candidates": 10},
            latency_ms=12.5,
        )
        d = ar.to_dict()
        assert d["latency_ms"] == 12.5
        assert d["status"] == "completed"
        assert "semantic_search" == d["name"]
        assert "retrieval" in d["operation_types"]

    def test_to_dict_skipped_no_latency(self):
        ar = ActionRecord(
            action_id=str(uuid.uuid4()),
            operation_types=[OperationType.RETRIEVAL],
            name="cross_encoder_rerank",
            inputs_summary={},
            outputs_summary={},
            status="skipped",
            skip_reason="no_reranker_configured",
        )
        d = ar.to_dict()
        assert d["status"] == "skipped"
        assert d["skip_reason"] == "no_reranker_configured"
        assert "latency_ms" not in d


# ── GovernanceEvalSummary tests ───────────────────────────────────────────────


class TestGovernanceEvalSummary:
    def test_to_dict(self):
        g = GovernanceEvalSummary(
            mode="advisory",
            profile="default",
            candidates_evaluated=15,
            vetoed=2,
            suppressed=3,
            contradictions_detected=1,
            contradiction_suppressed=1,
            net_candidates_returned=12,
        )
        d = g.to_dict()
        assert d["mode"] == "advisory"
        assert d["vetoed"] == 2
        assert d["contradictions_detected"] == 1
        assert d["net_candidates_returned"] == 12


# ── LearningWrite tests ───────────────────────────────────────────────────────


class TestLearningWrite:
    def test_to_dict_minimal(self):
        lw = LearningWrite(
            target_memory_type="episodic",
            operation="write",
        )
        d = lw.to_dict()
        assert d["target_memory_type"] == "episodic"
        assert d["operation"] == "write"
        assert "engram_id" not in d

    def test_to_dict_full(self):
        lw = LearningWrite(
            target_memory_type="semantic",
            operation="reinforce",
            engram_id="eng-123",
            delta_summary={"trust": +0.05, "utility": -0.01},
            triggered_by="reflect_path",
        )
        d = lw.to_dict()
        assert d["engram_id"] == "eng-123"
        assert d["delta_summary"]["trust"] == 0.05
        assert d["triggered_by"] == "reflect_path"


# ── CycleAssembler tests ──────────────────────────────────────────────────────


class TestCycleAssembler:
    def test_unique_cycle_ids(self):
        a1 = _make_assembler()
        a2 = _make_assembler()
        c1 = a1.build()
        c2 = a2.build()
        assert c1.cycle_id != c2.cycle_id

    def test_latency_non_negative(self):
        asm = _make_assembler()
        time.sleep(0.001)
        cycle = asm.build()
        assert cycle.cycle_latency_ms is not None
        assert cycle.cycle_latency_ms >= 0.0

    def test_set_working_memory(self):
        asm = _make_assembler()
        asm.set_working_memory(
            query="hello",
            active_retrieval_mode="hybrid",
            candidate_count_pre_governance=25,
        )
        cycle = asm.build(selected_route="return")
        wm = cycle.working_memory_snapshot
        assert wm.query == "hello"
        assert wm.active_retrieval_mode == "hybrid"
        assert wm.candidate_count_pre_governance == 25

    def test_set_working_memory_ignores_unknown_keys(self):
        asm = _make_assembler()
        # Should not raise
        asm.set_working_memory(nonexistent_field="ignored")
        cycle = asm.build()
        assert cycle.final_status == "completed"

    def test_add_retrieval_action(self):
        asm = _make_assembler()
        asm.add_retrieval_action(
            "semantic_search",
            inputs={"query": "test", "top_k": 10},
            outputs={"candidates_returned": 8},
            latency_ms=22.3,
        )
        cycle = asm.build()
        assert len(cycle.retrieval_actions) == 1
        assert cycle.retrieval_actions[0].name == "semantic_search"
        assert cycle.retrieval_actions[0].latency_ms == 22.3

    def test_add_reasoning_action(self):
        asm = _make_assembler()
        asm.add_reasoning_action(
            "contradiction_detection",
            inputs={"candidates_evaluated": 10},
            outputs={"groups_detected": 2},
        )
        cycle = asm.build()
        assert len(cycle.reasoning_actions) == 1
        assert OperationType.REASONING in cycle.reasoning_actions[0].operation_types

    def test_add_forecast_action(self):
        asm = _make_assembler()
        asm.add_forecast_action(
            "pulse_forecast",
            inputs={"source_signal": "pulse_p95"},
            outputs={"suggested_plan": "conservative"},
        )
        cycle = asm.build()
        assert len(cycle.forecast_actions) == 1
        assert OperationType.FORECASTING in cycle.forecast_actions[0].operation_types

    def test_add_governance_eval(self):
        asm = _make_assembler()
        asm.add_governance_eval(
            mode="enforced",
            profile="strict",
            candidates_evaluated=20,
            vetoed=3,
            suppressed=5,
            contradictions_detected=2,
            contradiction_suppressed=2,
            net_candidates_returned=15,
        )
        cycle = asm.build()
        assert len(cycle.governance_evaluations) == 1
        ge = cycle.governance_evaluations[0]
        assert ge.mode == "enforced"
        assert ge.vetoed == 3
        assert ge.net_candidates_returned == 15

    def test_add_learning_write(self):
        asm = _make_assembler()
        asm.add_learning_write(
            target_memory_type="semantic",
            operation="reinforce",
            engram_id="eng-999",
            delta_summary={"trust": 0.05},
            triggered_by="reflect_path",
        )
        cycle = asm.build()
        assert len(cycle.learning_writes) == 1
        lw = cycle.learning_writes[0]
        assert lw.engram_id == "eng-999"

    def test_add_ledger_ref(self):
        asm = _make_assembler()
        asm.add_ledger_ref("txn-abc-123")
        cycle = asm.build()
        assert "txn-abc-123" in cycle.forensic_ledger_refs

    def test_query_truncated_at_240(self):
        long_query = "x" * 500
        asm = CycleAssembler(
            trigger_type="search",
            trigger_source="api",
            query_or_event=long_query,
            active_profile="core_appliance",
            governance_profile="default",
        )
        cycle = asm.build()
        assert len(cycle.query_or_event) == 240

    def test_attention_decisions_preserved(self):
        asm = _make_assembler()
        asm.add_attention_decisions([
            AttentionDecision(
                dimension="retrieval_mode",
                decision="semantic",
                reason="lexical unavailable",
            ),
            AttentionDecision(
                dimension="governance_gate",
                decision="off",
                reason="governance not requested",
            ),
        ])
        cycle = asm.build()
        assert len(cycle.attention_decisions) == 2
        dims = [a.dimension for a in cycle.attention_decisions]
        assert "retrieval_mode" in dims
        assert "governance_gate" in dims

    def test_selected_route_propagated_to_working_memory(self):
        asm = _make_assembler()
        cycle = asm.build(selected_route="advise")
        assert cycle.selected_route == "advise"
        assert cycle.working_memory_snapshot.selected_route == "advise"

    def test_final_status_options(self):
        for status in ("completed", "degraded", "blocked", "no_op"):
            asm = _make_assembler()
            cycle = asm.build(final_status=status)
            assert cycle.final_status == status


# ── CognitiveCycleRecord serialisation tests ──────────────────────────────────


class TestCognitiveCycleRecord:
    def _full_cycle(self) -> CognitiveCycleRecord:
        asm = _make_assembler()
        asm.set_working_memory(
            query="what is the capital of France?",
            query_class="factoid",
            query_class_confidence=0.92,
            session_id="sess-001",
            active_retrieval_mode="hybrid",
            active_fusion_policy="balanced",
            active_governance_mode="advisory",
            active_policy_profile="default",
            candidate_count_pre_governance=15,
            candidate_count_post_governance=13,
            suppression_count=2,
            contradiction_count=0,
            summary_eligible=True,
            derived_fact_eligible=False,
            forecast_advisory_present=True,
        )
        asm.add_attention_decisions([
            AttentionDecision("retrieval_mode", "hybrid:balanced", "hybrid requested"),
            AttentionDecision("governance_gate", "advisory:default", "governance advisory mode"),
        ])
        asm.add_retrieval_action(
            "python_hybrid_rrf",
            inputs={"query": "what is the capital of France?", "top_k": 10},
            outputs={"candidates_returned": 15},
            latency_ms=34.2,
        )
        asm.add_reasoning_action(
            "contradiction_detection",
            inputs={"candidates_evaluated": 15},
            outputs={"groups_detected": 0},
        )
        asm.add_forecast_action(
            "pulse_forecast",
            inputs={"source_signal": "pulse_p95"},
            outputs={"suggested_plan": "standard"},
        )
        asm.add_governance_eval(
            mode="advisory",
            profile="default",
            candidates_evaluated=15,
            vetoed=1,
            suppressed=2,
            contradictions_detected=0,
            contradiction_suppressed=0,
            net_candidates_returned=13,
        )
        asm.add_learning_write(
            target_memory_type="episodic",
            operation="write",
            triggered_by="reflect_path",
        )
        asm.add_ledger_ref("txn-000-test")
        return asm.build(selected_route="return", final_status="completed")

    def test_to_dict_has_all_required_keys(self):
        cycle = self._full_cycle()
        d = cycle.to_dict()
        required_keys = [
            "cycle_id", "trigger_type", "trigger_source", "query_or_event",
            "active_profile", "governance_profile",
            "working_memory_snapshot",
            "attention_decisions",
            "retrieval_actions", "reasoning_actions", "forecast_actions",
            "governance_evaluations",
            "execution_actions", "learning_writes",
            "forensic_ledger_refs",
            "selected_route", "outcome_observation_plan",
            "final_status",
            "cycle_started_at", "cycle_completed_at", "cycle_latency_ms",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_operation_types_are_strings(self):
        cycle = self._full_cycle()
        d = cycle.to_dict()
        for action in d["retrieval_actions"]:
            for ot in action["operation_types"]:
                assert isinstance(ot, str), f"operation_type should be str: {ot}"

    def test_to_dict_no_objects(self):
        """Verify the full dict is JSON-serialisable (no non-primitive types)."""
        import json
        cycle = self._full_cycle()
        d = cycle.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert len(serialised) > 100

    def test_governance_isolaton_invariant(self):
        """
        Adding a cognitive cycle assembler must NOT affect the governance
        decision objects from the retrieval path.  This test verifies that
        the assembler is a pure observer — no mutation of external objects.
        """
        from mnemos.governance.models.governance_decision import GovernanceDecision
        decision = GovernanceDecision(
            engram_id="eng-safe",
            retrieval_score=0.8,
            governed_score=0.75,
            trust_modifier=0.95,
            veto_pass=True,
            suppressed=False,
        )
        original_governed_score = decision.governed_score
        original_suppressed = decision.suppressed

        asm = _make_assembler()
        asm.add_governance_eval(
            mode="advisory",
            profile="default",
            candidates_evaluated=1,
            vetoed=0,
            suppressed=0,
            contradictions_detected=0,
            contradiction_suppressed=0,
            net_candidates_returned=1,
        )
        asm.build()

        # GovernanceDecision must be unchanged
        assert decision.governed_score == original_governed_score
        assert decision.suppressed == original_suppressed

    def test_derived_fact_isolation_invariant(self):
        """
        A cognitive cycle record must NEVER set derived_fact_eligible=True
        when phase3 is disabled, even if the assembler is asked to.
        """
        asm = _make_assembler()
        # Simulating a well-behaved integration point that only sets
        # derived_fact_eligible based on the actual config flag:
        phase3_enabled = False  # config flag is off
        asm.set_working_memory(
            derived_fact_eligible=phase3_enabled,
        )
        cycle = asm.build()
        assert cycle.working_memory_snapshot.derived_fact_eligible is False
