"""
CognitiveCycleRecord — CoALA-aligned durable cognitive cycle schema for MNEMOS.

This module defines the first-class data structures for the MNEMOS cognitive
cycle contract.  Every major MNEMOS operation (search, reflect, pulse,
autonomous warmup, hygiene, shadow reconciliation) can emit a
CognitiveCycleRecord that captures the full path from trigger to outcome.

CoALA concept mapping
---------------------
Working memory      → WorkingMemorySnapshot (transient per-cycle state)
Attention           → List[AttentionDecision] (why MNEMOS focused where it did)
Retrieval actions   → List[ActionRecord] (semantic/hybrid search, envelope, rerank)
Reasoning actions   → List[ActionRecord] (contradiction detection, derived views)
Learning writes     → List[LearningWrite]  (reflect reinforcement, hygiene deltas)
Governance          → List[GovernanceEvalSummary] (read-path outcomes)
Forecasting         → List[ActionRecord]   (pulse advisory, intent trajectory)
Execution           → List[ActionRecord]   (warmup, cache write, ledger write)
Audit               → List[str] forensic_ledger_refs

All types are stdlib-only and produce JSON-serialisable dicts via to_dict().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


# ── CoALA operation taxonomy ─────────────────────────────────────────────────


class OperationType(str, Enum):
    """CoALA-aligned categories for MNEMOS operations."""

    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    LEARNING = "learning"
    GROUNDING = "grounding"
    GOVERNANCE = "governance"
    FORECASTING = "forecasting"
    AUDIT = "audit"


# Operation-type labels for named MNEMOS operations.
# Used by CycleAssembler when constructing ActionRecords.
OPERATION_TYPE_MAP: Dict[str, List[OperationType]] = {
    # Retrieval
    "semantic_search":          [OperationType.RETRIEVAL],
    "python_hybrid_rrf":        [OperationType.RETRIEVAL],
    "qdrant_rrf":               [OperationType.RETRIEVAL],
    "candidate_envelope":       [OperationType.RETRIEVAL],
    "cross_encoder_rerank":     [OperationType.RETRIEVAL, OperationType.REASONING],
    "pre_cognitive_cache":      [OperationType.RETRIEVAL],
    # Reasoning
    "contradiction_detection":          [OperationType.REASONING, OperationType.GOVERNANCE],
    "resolution_engram_synthesis":      [OperationType.REASONING, OperationType.LEARNING],
    "hierarchical_summary_view":        [OperationType.RETRIEVAL, OperationType.REASONING],
    "evidence_bundle_view":             [OperationType.RETRIEVAL, OperationType.REASONING],
    "contradiction_bundle_view":        [OperationType.REASONING, OperationType.GOVERNANCE],
    "derived_facts_shadow_eval":        [OperationType.REASONING],
    "shadow_search_intent":             [OperationType.RETRIEVAL, OperationType.REASONING],
    # Learning
    "reflect_reinforcement":    [OperationType.LEARNING],
    "episodic_write":           [OperationType.LEARNING],
    "semantic_write":           [OperationType.LEARNING],
    "semantic_candidate_write": [OperationType.LEARNING],
    "governance_score_update":  [OperationType.LEARNING, OperationType.GOVERNANCE],
    "forecast_outcome_write":   [OperationType.LEARNING, OperationType.FORECASTING],
    "cache_write":              [OperationType.LEARNING],
    "audit_write":              [OperationType.AUDIT],
    "procedural_change_candidate": [OperationType.LEARNING, OperationType.GOVERNANCE],
    "blocked_procedural_mutation": [OperationType.GOVERNANCE],
    # Governance & Hygiene
    "governance_read_path":     [OperationType.GOVERNANCE],
    "volatility_hygiene":       [OperationType.LEARNING, OperationType.GOVERNANCE],
    "freshness_decay":          [OperationType.LEARNING, OperationType.GOVERNANCE],
    "prune_promotion":          [OperationType.LEARNING, OperationType.GOVERNANCE],
    "contradiction_sweep":      [OperationType.REASONING, OperationType.GOVERNANCE],
    # Forecasting
    "pulse_forecast":           [OperationType.FORECASTING],
    "intent_trajectory":        [OperationType.FORECASTING],
    "volatility_forecast":      [OperationType.FORECASTING],
    # Grounding (autonomous external actions)
    "autonomous_warmup":        [OperationType.GROUNDING],
    "proactive_reconciliation": [OperationType.GROUNDING, OperationType.GOVERNANCE],
    # Audit
    "forensic_ledger_write":    [OperationType.AUDIT],
    "cognitive_cycle_emit":     [OperationType.AUDIT],
}


def _default_types(name: str) -> List[OperationType]:
    return OPERATION_TYPE_MAP.get(name, [OperationType.RETRIEVAL])


# ── Working memory ────────────────────────────────────────────────────────────


@dataclass
class WorkingMemorySnapshot:
    """
    Lightweight snapshot of the active MNEMOS working-memory state during a
    single cognitive cycle.

    This is NOT persisted by default; it is captured inside the
    CognitiveCycleRecord when cognitive_cycle=true is requested.

    Fields reflect the decisions and facts that were live at the moment the
    MNEMOS cycle executed its attention, retrieval, governance, and learning
    stages.
    """

    query: Optional[str] = None
    """Raw query or event description that triggered this cycle."""

    query_class: Optional[str] = None
    """Classified query family: factoid | multi_hop | global_synthesis | unknown."""

    query_class_confidence: float = 0.0
    """Classifier confidence score for query_class (0.0 – 1.0)."""

    session_id: Optional[str] = None
    """Session or tenant identifier extracted from filters."""

    # Candidate envelope
    candidate_count_pre_governance: int = 0
    """Raw candidate count returned by retrieval, before governance filtering."""

    candidate_count_post_governance: int = 0
    """Candidate count after governance read-path (advisory or enforced)."""

    suppression_count: int = 0
    """Number of candidates suppressed by governance during this cycle."""

    contradiction_count: int = 0
    """Number of contradiction groups detected during this cycle."""

    # Active routing state
    active_retrieval_mode: str = "semantic"
    """The retrieval mode that was executed: semantic | hybrid | pre_cognitive_cache."""

    active_fusion_policy: Optional[str] = None
    """The fusion policy applied in hybrid mode, or None for semantic-only."""

    active_governance_mode: str = "off"
    """The governance mode active for this cycle: off | advisory | enforced."""

    active_policy_profile: str = "default"
    """The policy profile ID applied during governance."""

    # Feature eligibility (attention gates)
    summary_eligible: bool = False
    """Whether hierarchical summary nodes were eligible for this query."""

    graph_expansion_eligible: bool = False
    """Whether graph-hybrid expansion was enabled (requires double opt-in)."""

    derived_fact_eligible: bool = False
    """Whether derived-fact lane was eligible (requires phase3+)."""

    forecast_advisory_present: bool = False
    """Whether a pulse forecast advisory was present during this cycle."""

    pre_cognitive_cache_hit: bool = False
    """Whether the query was served from the pre-cognitive intent cache."""

    # Decision
    selected_route: str = "return"
    """
    Final route selected by MNEMOS for this cycle:
    return | suppress | advise | forecast | pre_warm | reconcile | no_op | escalate
    """

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_class": self.query_class,
            "query_class_confidence": round(self.query_class_confidence, 4),
            "session_id": self.session_id,
            "candidate_count_pre_governance": self.candidate_count_pre_governance,
            "candidate_count_post_governance": self.candidate_count_post_governance,
            "suppression_count": self.suppression_count,
            "contradiction_count": self.contradiction_count,
            "active_retrieval_mode": self.active_retrieval_mode,
            "active_fusion_policy": self.active_fusion_policy,
            "active_governance_mode": self.active_governance_mode,
            "active_policy_profile": self.active_policy_profile,
            "summary_eligible": self.summary_eligible,
            "graph_expansion_eligible": self.graph_expansion_eligible,
            "derived_fact_eligible": self.derived_fact_eligible,
            "forecast_advisory_present": self.forecast_advisory_present,
            "pre_cognitive_cache_hit": self.pre_cognitive_cache_hit,
            "selected_route": self.selected_route,
        }


# ── Attention contract ────────────────────────────────────────────────────────


@dataclass
class AttentionDecision:
    """
    One focus or routing decision made by MNEMOS during the attention stage.

    External adapters and third-party tools can read the list of
    AttentionDecisions in a CognitiveCycleRecord to understand *why*
    MNEMOS paid attention to certain evidence and ignored other evidence,
    without needing to understand MNEMOS internal routing code.
    """

    dimension: str
    """
    What aspect of attention this decision covers:
      retrieval_mode | query_classification | candidate_envelope |
      summary_inclusion | graph_expansion | derived_facts |
      forecast_advisory | governance_gate | pre_cognitive_routing
    """

    decision: str
    """The value chosen or the gate result (e.g., 'hybrid', 'eligible', 'blocked')."""

    reason: str
    """Human-readable explanation of why this decision was made."""

    policy_source: Optional[str] = None
    """Config key, policy name, or feature flag that drove this decision."""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dimension": self.dimension,
            "decision": self.decision,
            "reason": self.reason,
        }
        if self.policy_source:
            d["policy_source"] = self.policy_source
        return d


# ── Action records ────────────────────────────────────────────────────────────


@dataclass
class ActionRecord:
    """
    A single typed operation performed during a cognitive cycle.

    Each record carries CoALA operation_types labels so that consuming systems
    and policy engines can understand the procedural-risk class of each action.
    """

    action_id: str
    """Unique ID for this action within the cycle."""

    operation_types: List[OperationType]
    """CoALA operation categories for this action (may be multi-typed)."""

    name: str
    """Operation name — must match a key in OPERATION_TYPE_MAP or be a freeform name."""

    inputs_summary: Dict[str, Any] = field(default_factory=dict)
    """Brief summary of action inputs (not full payload — keep small)."""

    outputs_summary: Dict[str, Any] = field(default_factory=dict)
    """Brief summary of action outputs or result metrics."""

    latency_ms: Optional[float] = None
    """Measured latency for this action in milliseconds, if available."""

    status: str = "completed"
    """completed | skipped | failed"""

    skip_reason: Optional[str] = None
    """Why this action was skipped, when status == 'skipped'."""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "action_id": self.action_id,
            "operation_types": [t.value for t in self.operation_types],
            "name": self.name,
            "inputs_summary": self.inputs_summary,
            "outputs_summary": self.outputs_summary,
            "status": self.status,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 2)
        if self.skip_reason:
            d["skip_reason"] = self.skip_reason
        return d


# ── Governance summary ────────────────────────────────────────────────────────


@dataclass
class GovernanceEvalSummary:
    """
    Aggregate outcome of the governance read-path for one cycle.

    Preserves the key counts that consuming systems need to understand what
    MNEMOS suppressed, vetoed, or flagged, without exposing per-candidate
    GovernanceDecision objects directly in the cycle record.
    """

    mode: str
    """Governance mode active during this evaluation: advisory | enforced."""

    profile: str
    """Policy profile ID applied."""

    candidates_evaluated: int
    vetoed: int
    suppressed: int
    contradictions_detected: int
    contradiction_suppressed: int
    net_candidates_returned: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "candidates_evaluated": self.candidates_evaluated,
            "vetoed": self.vetoed,
            "suppressed": self.suppressed,
            "contradictions_detected": self.contradictions_detected,
            "contradiction_suppressed": self.contradiction_suppressed,
            "net_candidates_returned": self.net_candidates_returned,
        }


# ── Learning writes ───────────────────────────────────────────────────────────


@dataclass
class LearningWrite:
    """
    A single write or update to long-term memory during or after a cycle.

    Maps to CoALA learning actions: writing experience to episodic memory,
    reflecting new facts to semantic memory, or updating procedural code
    (skills, prompts).

    MNEMOS never writes to procedural memory automatically — any procedural
    candidate should appear as a future_policy_candidate in a
    ForecastOutcomeRecord, not as a LearningWrite.
    """

    target_memory_type: str
    """episodic | semantic | procedural_candidate (never procedural directly)."""

    operation: str
    """write | update | reinforce | decay | prune | archive"""

    write_class: Optional[str] = None
    """
    Explicit learning/write boundary class, such as episodic_write,
    semantic_candidate_write, audit_write, or procedural_change_candidate.
    """

    engram_id: Optional[str] = None
    """Engram affected by this write, if applicable."""

    delta_summary: Optional[Dict[str, Any]] = None
    """Numeric or categorical delta applied (e.g., trust +0.05, utility -0.1)."""

    triggered_by: Optional[str] = None
    """What triggered this write: reflect_path | volatility_hygiene | decay_runner | etc."""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "target_memory_type": self.target_memory_type,
            "operation": self.operation,
        }
        if self.write_class:
            d["write_class"] = self.write_class
        if self.engram_id:
            d["engram_id"] = self.engram_id
        if self.delta_summary:
            d["delta_summary"] = self.delta_summary
        if self.triggered_by:
            d["triggered_by"] = self.triggered_by
        return d


# ── Cognitive cycle record ────────────────────────────────────────────────────


@dataclass
class CognitiveCycleRecord:
    """
    First-class cognitive cycle record for MNEMOS.

    Captures the complete path from trigger to outcome for any major MNEMOS
    operation.  This record is the core artifact of the CoALA-aligned cognitive
    cycle contract.

    External adapters and third-party tools can use this record to answer:
      - What did MNEMOS observe?
      - What did MNEMOS ignore?
      - What evidence entered the candidate envelope?
      - What route did MNEMOS choose?
      - What governance policy affected the result?
      - Was anything suppressed?  Forecasted?
      - Was any action autonomous, advisory, or blocked?
      - What did MNEMOS learn afterward?
      - Can this cycle be replayed or audited?

    Safety invariants
    -----------------
    This record describes MNEMOS memory operations, NOT the consuming
    application's private reasoning.  It does NOT expose derived-fact
    isolation internals, engram raw content, or governance scoring details
    beyond what is already surfaced by explain_governance=true.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    cycle_id: str
    """Unique identifier for this cognitive cycle."""

    trigger_type: str
    """
    What triggered this cycle:
      search | reflect | pulse | warmup | hygiene | shadow | reconciliation
    """

    trigger_source: str
    """Where the trigger originated: api | autonomous | scheduled"""

    query_or_event: str
    """
    The query text (for search/reflect) or event description (for pulse/hygiene).
    Truncated to 240 chars.
    """

    # ── Active configuration ──────────────────────────────────────────────────
    active_profile: str
    """MNEMOS deployment profile (e.g., 'governance_native', 'core_appliance')."""

    governance_profile: str
    """Policy profile ID active during governance evaluation."""

    # ── CoALA memory + action components ─────────────────────────────────────
    working_memory_snapshot: WorkingMemorySnapshot
    """Snapshot of the transient per-cycle working memory state."""

    attention_decisions: List[AttentionDecision] = field(default_factory=list)
    """Why MNEMOS focused on specific evidence, routes, or layers."""

    retrieval_actions: List[ActionRecord] = field(default_factory=list)
    """Retrieval-typed operations: semantic search, hybrid RRF, envelope, rerank."""

    reasoning_actions: List[ActionRecord] = field(default_factory=list)
    """Reasoning-typed operations: contradiction detection, derived views, shadow eval."""

    forecast_actions: List[ActionRecord] = field(default_factory=list)
    """Forecasting-typed operations: pulse advisory, intent trajectory, volatility forecast."""

    governance_evaluations: List[GovernanceEvalSummary] = field(default_factory=list)
    """Summary of governance read-path outcomes for this cycle."""

    execution_actions: List[ActionRecord] = field(default_factory=list)
    """Grounding/execution actions: warmup, proactive reconciliation, cache write."""

    learning_writes: List[LearningWrite] = field(default_factory=list)
    """Learning actions: reflect reinforcement, hygiene deltas, episodic writes."""

    # ── Audit ─────────────────────────────────────────────────────────────────
    forensic_ledger_refs: List[str] = field(default_factory=list)
    """References to forensic ledger transaction IDs emitted during this cycle."""

    # ── Advisory pattern recall (Phase 19) ───────────────────────────────────
    advisory_patterns: List[Dict[str, Any]] = field(default_factory=list)
    """
    Advisory PatternEngramCandidate summaries recalled from the pattern store
    at the start of this cycle.  These are informational only — candidates are
    advisory-only and never authoritative for retrieval.  Each entry is a dict
    with keys: candidate_id, pattern_summary, pattern_type, confidence_score,
    applies_when, promotion_status.
    """

    # ── Decision + outcome ────────────────────────────────────────────────────
    selected_route: str = "return"
    """
    The route MNEMOS selected for this cycle:
      return | suppress | advise | forecast | pre_warm | reconcile | shadow_eval |
      cache | escalate | no_op
    """

    outcome_observation_plan: Optional[str] = None
    """
    Description of how the outcome of this cycle will be observed or verified,
    if applicable (e.g., 'forecast_outcome_record::{forecast_id}').
    """

    final_status: str = "completed"
    """completed | degraded | blocked | no_op"""

    # ── Timing ────────────────────────────────────────────────────────────────
    cycle_started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    cycle_completed_at: Optional[str] = None
    cycle_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "trigger_type": self.trigger_type,
            "trigger_source": self.trigger_source,
            "query_or_event": self.query_or_event,
            "active_profile": self.active_profile,
            "governance_profile": self.governance_profile,
            "working_memory_snapshot": self.working_memory_snapshot.to_dict(),
            "attention_decisions": [a.to_dict() for a in self.attention_decisions],
            "retrieval_actions": [a.to_dict() for a in self.retrieval_actions],
            "reasoning_actions": [a.to_dict() for a in self.reasoning_actions],
            "forecast_actions": [a.to_dict() for a in self.forecast_actions],
            "governance_evaluations": [g.to_dict() for g in self.governance_evaluations],
            "execution_actions": [a.to_dict() for a in self.execution_actions],
            "learning_writes": [lw.to_dict() for lw in self.learning_writes],
            "forensic_ledger_refs": self.forensic_ledger_refs,
            "advisory_patterns": self.advisory_patterns,
            "selected_route": self.selected_route,
            "outcome_observation_plan": self.outcome_observation_plan,
            "final_status": self.final_status,
            "cycle_started_at": self.cycle_started_at,
            "cycle_completed_at": self.cycle_completed_at,
            "cycle_latency_ms": (
                round(self.cycle_latency_ms, 2)
                if self.cycle_latency_ms is not None
                else None
            ),
        }

    @classmethod
    def make_id(cls) -> str:
        return str(uuid.uuid4())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CognitiveCycleRecord":
        """Reconstruct a CognitiveCycleRecord from a to_dict() output."""
        wm_raw = d.get("working_memory_snapshot") or {}
        wm = WorkingMemorySnapshot(
            query=wm_raw.get("query"),
            query_class=wm_raw.get("query_class"),
            query_class_confidence=float(wm_raw.get("query_class_confidence", 0.0)),
            session_id=wm_raw.get("session_id"),
            candidate_count_pre_governance=int(wm_raw.get("candidate_count_pre_governance", 0)),
            candidate_count_post_governance=int(wm_raw.get("candidate_count_post_governance", 0)),
            suppression_count=int(wm_raw.get("suppression_count", 0)),
            contradiction_count=int(wm_raw.get("contradiction_count", 0)),
            active_retrieval_mode=wm_raw.get("active_retrieval_mode", "semantic"),
            active_fusion_policy=wm_raw.get("active_fusion_policy"),
            active_governance_mode=wm_raw.get("active_governance_mode", "off"),
            active_policy_profile=wm_raw.get("active_policy_profile", "default"),
            summary_eligible=bool(wm_raw.get("summary_eligible", False)),
            graph_expansion_eligible=bool(wm_raw.get("graph_expansion_eligible", False)),
            derived_fact_eligible=bool(wm_raw.get("derived_fact_eligible", False)),
            forecast_advisory_present=bool(wm_raw.get("forecast_advisory_present", False)),
            pre_cognitive_cache_hit=bool(wm_raw.get("pre_cognitive_cache_hit", False)),
            selected_route=wm_raw.get("selected_route", "return"),
        )

        def _action(a: Dict[str, Any]) -> ActionRecord:
            return ActionRecord(
                action_id=a.get("action_id", str(uuid.uuid4())),
                operation_types=[
                    OperationType(v) for v in a.get("operation_types", [])
                    if v in {e.value for e in OperationType}
                ],
                name=a.get("name", ""),
                inputs_summary=a.get("inputs_summary", {}),
                outputs_summary=a.get("outputs_summary", {}),
                latency_ms=a.get("latency_ms"),
                status=a.get("status", "completed"),
                skip_reason=a.get("skip_reason"),
            )

        attention = [
            AttentionDecision(
                dimension=a.get("dimension", ""),
                decision=a.get("decision", ""),
                reason=a.get("reason", ""),
                policy_source=a.get("policy_source"),
            )
            for a in d.get("attention_decisions", [])
        ]
        governance = [
            GovernanceEvalSummary(
                mode=g.get("mode", "off"),
                profile=g.get("profile", "default"),
                candidates_evaluated=int(g.get("candidates_evaluated", 0)),
                vetoed=int(g.get("vetoed", 0)),
                suppressed=int(g.get("suppressed", 0)),
                contradictions_detected=int(g.get("contradictions_detected", 0)),
                contradiction_suppressed=int(g.get("contradiction_suppressed", 0)),
                net_candidates_returned=int(g.get("net_candidates_returned", 0)),
            )
            for g in d.get("governance_evaluations", [])
        ]
        learning = [
            LearningWrite(
                target_memory_type=lw.get("target_memory_type", "semantic"),
                operation=lw.get("operation", "write"),
                write_class=lw.get("write_class"),
                engram_id=lw.get("engram_id"),
                delta_summary=lw.get("delta_summary"),
                triggered_by=lw.get("triggered_by"),
            )
            for lw in d.get("learning_writes", [])
        ]

        return cls(
            cycle_id=d.get("cycle_id", str(uuid.uuid4())),
            trigger_type=d.get("trigger_type", "search"),
            trigger_source=d.get("trigger_source", "api"),
            query_or_event=d.get("query_or_event", "")[:240],
            active_profile=d.get("active_profile", ""),
            governance_profile=d.get("governance_profile", "default"),
            working_memory_snapshot=wm,
            attention_decisions=attention,
            retrieval_actions=[_action(a) for a in d.get("retrieval_actions", [])],
            reasoning_actions=[_action(a) for a in d.get("reasoning_actions", [])],
            forecast_actions=[_action(a) for a in d.get("forecast_actions", [])],
            governance_evaluations=governance,
            execution_actions=[_action(a) for a in d.get("execution_actions", [])],
            learning_writes=learning,
            forensic_ledger_refs=d.get("forensic_ledger_refs", []),
            advisory_patterns=d.get("advisory_patterns", []),
            selected_route=d.get("selected_route", "return"),
            outcome_observation_plan=d.get("outcome_observation_plan"),
            final_status=d.get("final_status", "completed"),
            cycle_started_at=d.get("cycle_started_at", ""),
            cycle_completed_at=d.get("cycle_completed_at"),
            cycle_latency_ms=d.get("cycle_latency_ms"),
        )
