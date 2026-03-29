"""
ContradictionRecord — a fully resolved contradiction cluster.

Wave 2: populated by ContradictionPolicy.detect_and_resolve().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ContradictionRecord:
    """
    Represents a group of Engrams that make mutually inconsistent claims
    about the same entity and attribute slot.

    One ContradictionRecord per detected (entity_key, attribute_key) conflict.
    """

    conflict_group_id: str

    # ── Conflict identity ──────────────────────────────────────────────────
    entity_key: str = ""
    attribute_key: str = ""

    # ── Candidates ─────────────────────────────────────────────────────────
    candidate_memory_ids: List[str] = field(default_factory=list)

    candidate_values: Dict[str, str] = field(default_factory=dict)
    # {memory_id: normalized_value} for each member of the group

    # ── Resolution ─────────────────────────────────────────────────────────
    winner_memory_id: Optional[str] = None
    loser_memory_ids: List[str] = field(default_factory=list)

    resolution_reason: str = "unresolved"
    # e.g. "higher trust_score (0.900 vs 0.700)"
    #      "newer timestamp"
    #      "deterministic tie-breaker (memory_id)"

    status: str = "detected"
    # detected | resolved | ambiguous

    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_group_id": self.conflict_group_id,
            "entity_key": self.entity_key,
            "attribute_key": self.attribute_key,
            "candidate_memory_ids": self.candidate_memory_ids,
            "candidate_values": self.candidate_values,
            "winner_memory_id": self.winner_memory_id,
            "loser_memory_ids": self.loser_memory_ids,
            "resolution_reason": self.resolution_reason,
            "status": self.status,
            "created_at": self.created_at,
        }
