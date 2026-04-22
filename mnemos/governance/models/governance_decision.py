"""
GovernanceDecision — query-time outcome of governance evaluation for one candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GovernanceDecision:
    """
    Records the governance outcome for a single candidate during a search.

    ``retrieval_score``  — raw score from the retrieval tier (unchanged)
    ``governed_score``   — final score after all governance modifiers
    ``suppressed``       — True if this candidate should be withheld from context
    """

    engram_id: str
    retrieval_score: float
    governed_score: float

    # ── Modifier components (explain mode) ────────────────────────────────
    trust_modifier: float = 1.0
    utility_modifier: float = 1.0
    freshness_modifier: float = 1.0
    contradiction_modifier: float = 1.0
    veto_modifier: float = 1.0          # 0.0 = vetoed, 1.0 = passed

    # ── Veto result ────────────────────────────────────────────────────────
    veto_pass: bool = True
    veto_reason: Optional[str] = None

    # ── Conflict state (Wave 2) ────────────────────────────────────────────
    conflict_status: str = "none"
    # none | winner | suppressed | pending

    conflict_group_id: Optional[str] = None
    # ID of the contradiction group this candidate belongs to (if any)

    contradiction_winner: Optional[str] = None
    # memory_id of the winning candidate when this candidate is a loser

    contradiction_reason: Optional[str] = None
    # human-readable explanation of why the winner was selected

    # ── Final suppression state ────────────────────────────────────────────
    suppressed: bool = False
    suppressed_reason: Optional[str] = None

    suppressed_by_contradiction: bool = False
    # True when suppressed specifically due to losing a contradiction resolution

    would_be_suppressed_in_enforced_mode: bool = False
    # True when this candidate would be removed in enforced mode
    # (veto or contradiction loser); useful in advisory explain output

    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Compact payload — used when explain_governance=False."""
        d: Dict[str, Any] = {
            "retrieval_score": round(self.retrieval_score, 4),
            "governed_score": round(self.governed_score, 4),
            "veto_pass": self.veto_pass,
            "conflict_status": self.conflict_status,
            "suppressed": self.suppressed,
            "would_be_suppressed_in_enforced_mode": self.would_be_suppressed_in_enforced_mode,
        }
        if self.veto_reason:
            d["veto_reason"] = self.veto_reason
        if self.suppressed_reason:
            d["suppressed_reason"] = self.suppressed_reason
        if self.conflict_group_id:
            d["conflict_group_id"] = self.conflict_group_id
        if self.suppressed_by_contradiction:
            d["suppressed_by_contradiction"] = True
            d["contradiction_winner"] = self.contradiction_winner
        return d

    def to_dict_full(self) -> Dict[str, Any]:
        """Full explain payload including all modifier values."""
        return {
            "retrieval_score": round(self.retrieval_score, 4),
            "governed_score": round(self.governed_score, 4),
            "modifiers": {
                "trust": round(self.trust_modifier, 4),
                "utility": round(self.utility_modifier, 4),
                "freshness": round(self.freshness_modifier, 4),
                "contradiction": round(self.contradiction_modifier, 4),
                "veto": round(self.veto_modifier, 4),
            },
            "veto_pass": self.veto_pass,
            "veto_reason": self.veto_reason,
            "conflict_status": self.conflict_status,
            "conflict_group_id": self.conflict_group_id,
            "contradiction_winner": self.contradiction_winner,
            "contradiction_reason": self.contradiction_reason,
            "suppressed": self.suppressed,
            "suppressed_reason": self.suppressed_reason,
            "suppressed_by_contradiction": self.suppressed_by_contradiction,
            "would_be_suppressed_in_enforced_mode": self.would_be_suppressed_in_enforced_mode,
        }
