"""
Advisory PatternEngramCandidate schema.

Pattern candidates are higher-order abstractions over observed cognitive
cycles.  In this first implementation they are explicitly non-authoritative:
they cannot affect retrieval ranking, routing, or policy unless an operator
records explicit governance approval in a later review step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from mnemos.cognitive.learning_boundary import (
    PROCEDURAL_CHANGE_CANDIDATE,
    SEMANTIC_CANDIDATE_WRITE,
    classify_learning_write,
)


PROMOTION_CANDIDATE = "candidate"
PROMOTION_RECOMMENDED = "promotion_recommended"
PROMOTION_APPROVED = "approved"
PROMOTION_REJECTED = "rejected"

PATTERN_TYPES = frozenset({
    "descriptive",
    "operational_recommendation",
    "policy_mutation",
    "routing_mutation",
    "template_mutation",
})


@dataclass
class PatternEngramCandidate:
    """Governance-reviewable, advisory pattern abstraction."""

    pattern_summary: str
    supporting_cycle_ids: List[str]
    supporting_engram_ids: List[str] = field(default_factory=list)
    contradicting_engram_ids: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    support_score: float = 0.0
    contradiction_score: float = 0.0
    pattern_type: str = "descriptive"
    recommended_scope: str = "local"
    applies_when: str = ""
    does_not_apply_when: str = ""
    risk_if_wrong: str = ""
    proposed_learning_class: str = SEMANTIC_CANDIDATE_WRITE
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    promotion_status: str = PROMOTION_CANDIDATE
    governance_review_required: bool = True
    source_cycle_count: int = 0
    first_seen_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    governance_review_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pattern_type not in PATTERN_TYPES:
            raise ValueError(f"Unknown pattern_type: {self.pattern_type}")
        if not self.supporting_cycle_ids:
            raise ValueError("PatternEngramCandidate requires supporting_cycle_ids")
        if not self.risk_if_wrong:
            raise ValueError("PatternEngramCandidate requires risk_if_wrong")
        if self.source_cycle_count <= 0:
            self.source_cycle_count = len(set(self.supporting_cycle_ids))
        self._validate_learning_class()

    def _validate_learning_class(self) -> None:
        decision = classify_learning_write(self.proposed_learning_class)
        if self.pattern_type == "descriptive":
            expected = SEMANTIC_CANDIDATE_WRITE
        else:
            expected = PROCEDURAL_CHANGE_CANDIDATE
        if self.pattern_type.endswith("_mutation"):
            expected = "blocked_procedural_mutation"
        if self.proposed_learning_class != expected:
            raise ValueError(
                "Pattern type and proposed_learning_class disagree: "
                f"{self.pattern_type} requires {expected}"
            )
        if decision.authoritative_for_retrieval:
            raise ValueError("PatternEngramCandidate cannot be authoritative for retrieval")

    @property
    def authoritative_for_retrieval(self) -> bool:
        return False

    @property
    def affects_ranking(self) -> bool:
        return False

    @property
    def mutates_policy(self) -> bool:
        return False

    def recommend_promotion(self, *, gate_id: str, confidence_threshold: float = 0.9) -> None:
        """
        Mark the candidate as recommended, never promoted.

        High-confidence gates may only recommend promotion in this phase.
        Authoritative promotion requires explicit governance approval.
        """
        if self.confidence_score >= confidence_threshold:
            self.promotion_status = PROMOTION_RECOMMENDED
            self.governance_review_id = gate_id

    def approve_promotion(self, *, governance_review_id: str, explicit_approval: bool) -> None:
        """Approve only when a governance review explicitly says yes."""
        if not explicit_approval:
            raise PermissionError("Explicit governance approval is required for promotion")
        self.promotion_status = PROMOTION_APPROVED
        self.governance_review_id = governance_review_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "pattern_summary": self.pattern_summary,
            "supporting_cycle_ids": self.supporting_cycle_ids,
            "supporting_engram_ids": self.supporting_engram_ids,
            "contradicting_engram_ids": self.contradicting_engram_ids,
            "confidence_score": round(self.confidence_score, 4),
            "support_score": round(self.support_score, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "promotion_status": self.promotion_status,
            "governance_review_required": self.governance_review_required,
            "pattern_type": self.pattern_type,
            "recommended_scope": self.recommended_scope,
            "applies_when": self.applies_when,
            "does_not_apply_when": self.does_not_apply_when,
            "risk_if_wrong": self.risk_if_wrong,
            "source_cycle_count": self.source_cycle_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "proposed_learning_class": self.proposed_learning_class,
            "authoritative_for_retrieval": self.authoritative_for_retrieval,
            "affects_ranking": self.affects_ranking,
            "mutates_policy": self.mutates_policy,
            "governance_review_id": self.governance_review_id,
        }
