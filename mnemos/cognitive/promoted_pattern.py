"""
PatternEngram — Phase 20 authoritative promoted pattern.

A PatternEngram is the terminal promoted form of a PatternEngramCandidate
after explicit governance approval.  It is advisory (authoritative_for_retrieval
remains False) and is stored in the PatternCandidateStore promoted pool — NOT in
the Qdrant/pgvector engram index.

The ONLY valid constructor is ``PatternEngram.from_approved_candidate()``.

Safety invariants
-----------------
- ``authoritative_for_retrieval`` is always ``False`` — PatternEngrams are
  surfaced via dedicated endpoints, not injected into the retrieval path.
- ``from_approved_candidate()`` raises ``PermissionError`` unless the candidate
  has ``promotion_status == PROMOTION_APPROVED``.
- ``write_class`` is always ``SEMANTIC_WRITE`` — an approved advisory pattern
  is a semantic fact, not a procedural mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from mnemos.cognitive.learning_boundary import SEMANTIC_WRITE
from mnemos.cognitive.pattern_engram import (
    PatternEngramCandidate,
    PROMOTION_APPROVED,
)


@dataclass
class PatternEngram:
    """
    Authoritative advisory pattern, promoted from a governance-approved candidate.

    Construct only via ``PatternEngram.from_approved_candidate()``.
    """

    pattern_id: str
    """Carries over the candidate_id from the approved candidate."""

    pattern_summary: str
    pattern_type: str
    applies_when: str
    does_not_apply_when: str
    risk_if_wrong: str
    confidence_score: float
    supporting_cycle_ids: List[str]
    supporting_engram_ids: List[str]
    contradicting_engram_ids: List[str]
    governance_review_id: str
    """The governance review ID recorded at approval time — mandatory."""

    promoted_from_candidate_id: str
    promoted_at: str

    write_class: str = field(default=SEMANTIC_WRITE)
    """Always SEMANTIC_WRITE — approved patterns are semantic facts, not mutations."""

    # ── Safety invariants (read-only, hardcoded) ──────────────────────────────

    @property
    def authoritative_for_retrieval(self) -> bool:
        """PatternEngrams are advisory only — never inserted into the retrieval index."""
        return False

    @property
    def affects_ranking(self) -> bool:
        return False

    @property
    def mutates_policy(self) -> bool:
        return False

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_approved_candidate(
        cls,
        candidate: PatternEngramCandidate,
        *,
        governance_review_id: str,
    ) -> "PatternEngram":
        """
        Promote an approved candidate to a PatternEngram.

        Raises
        ------
        PermissionError
            If ``candidate.promotion_status`` is not ``PROMOTION_APPROVED``.
        ValueError
            If ``governance_review_id`` is empty.
        """
        if candidate.promotion_status != PROMOTION_APPROVED:
            raise PermissionError(
                f"Candidate {candidate.candidate_id!r} must be in PROMOTION_APPROVED state "
                f"(current: {candidate.promotion_status!r})"
            )
        if not governance_review_id:
            raise ValueError("governance_review_id is required for promotion")

        return cls(
            pattern_id=candidate.candidate_id,
            pattern_summary=candidate.pattern_summary,
            pattern_type=candidate.pattern_type,
            applies_when=candidate.applies_when,
            does_not_apply_when=candidate.does_not_apply_when,
            risk_if_wrong=candidate.risk_if_wrong,
            confidence_score=candidate.confidence_score,
            supporting_cycle_ids=list(candidate.supporting_cycle_ids),
            supporting_engram_ids=list(candidate.supporting_engram_ids),
            contradicting_engram_ids=list(candidate.contradicting_engram_ids),
            governance_review_id=governance_review_id,
            promoted_from_candidate_id=candidate.candidate_id,
            promoted_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_summary": self.pattern_summary,
            "pattern_type": self.pattern_type,
            "applies_when": self.applies_when,
            "does_not_apply_when": self.does_not_apply_when,
            "risk_if_wrong": self.risk_if_wrong,
            "confidence_score": round(self.confidence_score, 4),
            "supporting_cycle_ids": self.supporting_cycle_ids,
            "supporting_engram_ids": self.supporting_engram_ids,
            "contradicting_engram_ids": self.contradicting_engram_ids,
            "governance_review_id": self.governance_review_id,
            "promoted_from_candidate_id": self.promoted_from_candidate_id,
            "promoted_at": self.promoted_at,
            "write_class": self.write_class,
            "authoritative_for_retrieval": self.authoritative_for_retrieval,
            "affects_ranking": self.affects_ranking,
            "mutates_policy": self.mutates_policy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatternEngram":
        return cls(
            pattern_id=d["pattern_id"],
            pattern_summary=d.get("pattern_summary", ""),
            pattern_type=d.get("pattern_type", "descriptive"),
            applies_when=d.get("applies_when", ""),
            does_not_apply_when=d.get("does_not_apply_when", ""),
            risk_if_wrong=d.get("risk_if_wrong", ""),
            confidence_score=float(d.get("confidence_score", 0.0)),
            supporting_cycle_ids=d.get("supporting_cycle_ids", []),
            supporting_engram_ids=d.get("supporting_engram_ids", []),
            contradicting_engram_ids=d.get("contradicting_engram_ids", []),
            governance_review_id=d.get("governance_review_id", ""),
            promoted_from_candidate_id=d.get("promoted_from_candidate_id", ""),
            promoted_at=d.get("promoted_at", ""),
        )
