"""
CycleAssembler — builder for CognitiveCycleRecord.

The assembler is a lightweight builder object that integration points in
service/app.py instantiate at the start of a cycle, incrementally populate
during execution, and finalise with build() at the end.

Design principles
-----------------
- Zero performance impact when cognitive_cycle is not requested.
- No dependencies outside the cognitive module (no retrieval, governance
  or ledger imports at module level).
- Each record method is a no-op if called with empty/invalid arguments.
- The assembler never mutates existing MNEMOS objects.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


class CycleAssembler:
    """
    Incremental builder for a CognitiveCycleRecord.

    Usage
    -----
    ::

        assembler = CycleAssembler(
            trigger_type="search",
            trigger_source="api",
            query_or_event=query[:240],
            active_profile=config.profile,
            governance_profile=selected_profile or "default",
        )

        assembler.set_working_memory(
            query=query,
            active_retrieval_mode="hybrid",
            ...
        )
        assembler.add_attention_decisions(decisions)
        assembler.add_retrieval_action("semantic_search", inputs={...}, outputs={...})
        assembler.add_governance_eval(mode="advisory", ...)

        cycle = assembler.build(selected_route="return", final_status="completed")
        payload["cognitive_cycle"] = cycle.to_dict()
    """

    def __init__(
        self,
        *,
        trigger_type: str,
        trigger_source: str,
        query_or_event: str,
        active_profile: str,
        governance_profile: str,
    ) -> None:
        self._cycle_id: str = str(uuid.uuid4())
        self._trigger_type: str = trigger_type
        self._trigger_source: str = trigger_source
        self._query_or_event: str = query_or_event[:240]
        self._active_profile: str = active_profile
        self._governance_profile: str = governance_profile

        self._wm: WorkingMemorySnapshot = WorkingMemorySnapshot()
        self._attention_decisions: List[AttentionDecision] = []
        self._retrieval_actions: List[ActionRecord] = []
        self._reasoning_actions: List[ActionRecord] = []
        self._forecast_actions: List[ActionRecord] = []
        self._governance_evaluations: List[GovernanceEvalSummary] = []
        self._execution_actions: List[ActionRecord] = []
        self._learning_writes: List[LearningWrite] = []
        self._forensic_ledger_refs: List[str] = []

        self._started_at: str = datetime.now(timezone.utc).isoformat()
        self._start_monotonic: float = time.monotonic()

    # ── Working memory ────────────────────────────────────────────────────────

    def set_working_memory(self, **kwargs: Any) -> None:
        """
        Update fields on the WorkingMemorySnapshot.

        Accepts any keyword argument that matches a field name on
        WorkingMemorySnapshot.  Unknown keys are silently ignored.
        """
        for k, v in kwargs.items():
            if hasattr(self._wm, k):
                setattr(self._wm, k, v)

    # ── Attention decisions ───────────────────────────────────────────────────

    def add_attention_decisions(self, decisions: List[AttentionDecision]) -> None:
        """Append a list of AttentionDecision objects to the cycle."""
        self._attention_decisions.extend(decisions)

    def add_attention_decision(
        self,
        *,
        dimension: str,
        decision: str,
        reason: str,
        policy_source: Optional[str] = None,
    ) -> None:
        """Add a single AttentionDecision."""
        self._attention_decisions.append(
            AttentionDecision(
                dimension=dimension,
                decision=decision,
                reason=reason,
                policy_source=policy_source,
            )
        )

    # ── Action records ────────────────────────────────────────────────────────

    def _make_action(
        self,
        name: str,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: Optional[float] = None,
        status: str = "completed",
        skip_reason: Optional[str] = None,
        operation_types: Optional[List[OperationType]] = None,
    ) -> ActionRecord:
        return ActionRecord(
            action_id=str(uuid.uuid4()),
            operation_types=operation_types if operation_types is not None else _default_types(name),
            name=name,
            inputs_summary=inputs,
            outputs_summary=outputs,
            latency_ms=latency_ms,
            status=status,
            skip_reason=skip_reason,
        )

    def add_retrieval_action(
        self,
        name: str,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: Optional[float] = None,
        status: str = "completed",
        skip_reason: Optional[str] = None,
    ) -> None:
        """Record a retrieval-typed action."""
        self._retrieval_actions.append(
            self._make_action(
                name,
                inputs=inputs,
                outputs=outputs,
                latency_ms=latency_ms,
                status=status,
                skip_reason=skip_reason,
            )
        )

    def add_reasoning_action(
        self,
        name: str,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: Optional[float] = None,
        status: str = "completed",
        skip_reason: Optional[str] = None,
    ) -> None:
        """Record a reasoning-typed action."""
        self._reasoning_actions.append(
            self._make_action(
                name,
                inputs=inputs,
                outputs=outputs,
                latency_ms=latency_ms,
                status=status,
                skip_reason=skip_reason,
            )
        )

    def add_forecast_action(
        self,
        name: str,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: Optional[float] = None,
        status: str = "completed",
    ) -> None:
        """Record a forecasting-typed action."""
        self._forecast_actions.append(
            self._make_action(
                name,
                inputs=inputs,
                outputs=outputs,
                latency_ms=latency_ms,
                status=status,
                operation_types=[OperationType.FORECASTING],
            )
        )

    def add_execution_action(
        self,
        name: str,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: Optional[float] = None,
        status: str = "completed",
    ) -> None:
        """Record an execution/grounding action (e.g., warmup, cache write)."""
        self._execution_actions.append(
            self._make_action(
                name,
                inputs=inputs,
                outputs=outputs,
                latency_ms=latency_ms,
                status=status,
            )
        )

    # ── Governance ────────────────────────────────────────────────────────────

    def add_governance_eval(
        self,
        *,
        mode: str,
        profile: str,
        candidates_evaluated: int,
        vetoed: int,
        suppressed: int,
        contradictions_detected: int,
        contradiction_suppressed: int,
        net_candidates_returned: int,
    ) -> None:
        """Record the aggregate governance read-path outcome for this cycle."""
        self._governance_evaluations.append(
            GovernanceEvalSummary(
                mode=mode,
                profile=profile,
                candidates_evaluated=candidates_evaluated,
                vetoed=vetoed,
                suppressed=suppressed,
                contradictions_detected=contradictions_detected,
                contradiction_suppressed=contradiction_suppressed,
                net_candidates_returned=net_candidates_returned,
            )
        )

    # ── Learning writes ───────────────────────────────────────────────────────

    def add_learning_write(
        self,
        *,
        target_memory_type: str,
        operation: str,
        engram_id: Optional[str] = None,
        delta_summary: Optional[Dict[str, Any]] = None,
        triggered_by: Optional[str] = None,
    ) -> None:
        """Record a long-term memory write performed during this cycle."""
        self._learning_writes.append(
            LearningWrite(
                target_memory_type=target_memory_type,
                operation=operation,
                engram_id=engram_id,
                delta_summary=delta_summary,
                triggered_by=triggered_by,
            )
        )

    # ── Audit references ──────────────────────────────────────────────────────

    def add_ledger_ref(self, ref: str) -> None:
        """Add a forensic ledger transaction ID to the cycle's audit trail."""
        if ref:
            self._forensic_ledger_refs.append(ref)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        *,
        selected_route: str = "return",
        final_status: str = "completed",
        outcome_observation_plan: Optional[str] = None,
    ) -> CognitiveCycleRecord:
        """
        Finalise and return the CognitiveCycleRecord.

        Should be called exactly once, at the end of the cycle, after all
        actions, decisions, and evaluations have been recorded.
        """
        completed_at = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.monotonic() - self._start_monotonic) * 1000.0
        self._wm.selected_route = selected_route

        return CognitiveCycleRecord(
            cycle_id=self._cycle_id,
            trigger_type=self._trigger_type,
            trigger_source=self._trigger_source,
            query_or_event=self._query_or_event,
            active_profile=self._active_profile,
            governance_profile=self._governance_profile,
            working_memory_snapshot=self._wm,
            attention_decisions=list(self._attention_decisions),
            retrieval_actions=list(self._retrieval_actions),
            reasoning_actions=list(self._reasoning_actions),
            forecast_actions=list(self._forecast_actions),
            governance_evaluations=list(self._governance_evaluations),
            execution_actions=list(self._execution_actions),
            learning_writes=list(self._learning_writes),
            forensic_ledger_refs=list(self._forensic_ledger_refs),
            selected_route=selected_route,
            outcome_observation_plan=outcome_observation_plan,
            final_status=final_status,
            cycle_started_at=self._started_at,
            cycle_completed_at=completed_at,
            cycle_latency_ms=latency_ms,
        )
