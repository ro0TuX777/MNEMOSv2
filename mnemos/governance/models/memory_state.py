"""
GovernanceMeta — stored governance state for a single Engram.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GovernanceMeta:
    """
    Governance state attached to an Engram.

    Wave 1: Most scores default to neutral (1.0).
    The reflect path (Wave 2) and background hygiene (Wave 3) will
    differentiate these over time based on actual memory usage.

    Serialised as ``_governance`` inside Engram.to_dict() when
    include_governance=True is requested.
    """

    # ── Memory classification ──────────────────────────────────────────────
    memory_type: str = "episodic"
    # episodic | semantic | derived | summary

    source_type: str = ""
    # document | api | user_input | derived

    source_id: str = ""
    # ID of the originating source artifact (for grounding and delete cascade)

    derived_from: List[str] = field(default_factory=list)
    # Parent Engram IDs — populated when this memory is derived from others.
    # Required for delete cascade correctness.

    # ── Contradiction identity (Wave 2) ───────────────────────────────────
    entity_key: str = ""
    # The entity this memory is about (e.g. "user:alice", "project:mnemos").
    # Required for contradiction detection — empty means "not a slot claim".

    attribute_key: str = ""
    # The attribute being claimed (e.g. "city", "employer", "status").
    # Combined with entity_key to form the contradiction group key.

    normalized_value: str = ""
    # Normalised string representation of the claimed value (e.g. "auckland").
    # Used for contradiction comparison.  Empty = no value claim.

    source_authority: float = 0.5
    # [0, 1] — how authoritative is this source.  Used as a tiebreaker in
    # contradiction winner selection.  0.5 = unknown/neutral.

    # ── Lifecycle ──────────────────────────────────────────────────────────
    lifecycle_state: str = "active"
    # active | fading | prune_candidate | archived

    # ── Scores (updated by reflect path / hygiene jobs) ─────────────────────
    trust_score: float = 1.0
    # [0, 1] — how reliable is the source and content

    utility_score: float = 1.0
    # [0, 1] — how often has this memory been useful

    stability_score: float = 1.0
    # [0, 1] — how consistent / non-contradicted this memory is

    retrievability_score: float = 1.0
    # [0, 1] — composite recency + usage signal; updated by hygiene

    # ── Access tracking ────────────────────────────────────────────────────
    last_accessed_at: Optional[str] = None
    last_used_at: Optional[str] = None

    # ── Conflict state ─────────────────────────────────────────────────────
    conflict_group_id: Optional[str] = None
    conflict_status: str = "none"
    # none | winner | suppressed | pending

    superseded_by: Optional[str] = None
    # ID of the memory that replaced this one

    # ── Deletion state ─────────────────────────────────────────────────────
    deletion_state: str = "active"
    # active | soft_deleted | tombstone | rederive_needed

    # ── Policy flags ───────────────────────────────────────────────────────
    policy_flags: List[str] = field(default_factory=list)
    # Recognised flags: toxic | stale | contradictory | protected | experimental

    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_type": self.memory_type,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "derived_from": self.derived_from,
            "entity_key": self.entity_key,
            "attribute_key": self.attribute_key,
            "normalized_value": self.normalized_value,
            "source_authority": self.source_authority,
            "lifecycle_state": self.lifecycle_state,
            "trust_score": self.trust_score,
            "utility_score": self.utility_score,
            "stability_score": self.stability_score,
            "retrievability_score": self.retrievability_score,
            "last_accessed_at": self.last_accessed_at,
            "last_used_at": self.last_used_at,
            "conflict_group_id": self.conflict_group_id,
            "conflict_status": self.conflict_status,
            "superseded_by": self.superseded_by,
            "deletion_state": self.deletion_state,
            "policy_flags": self.policy_flags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceMeta":
        return cls(
            memory_type=data.get("memory_type", "episodic"),
            source_type=data.get("source_type", ""),
            source_id=data.get("source_id", ""),
            derived_from=data.get("derived_from", []),
            entity_key=data.get("entity_key", ""),
            attribute_key=data.get("attribute_key", ""),
            normalized_value=data.get("normalized_value", ""),
            source_authority=float(data.get("source_authority", 0.5)),
            lifecycle_state=data.get("lifecycle_state", "active"),
            trust_score=float(data.get("trust_score", 1.0)),
            utility_score=float(data.get("utility_score", 1.0)),
            stability_score=float(data.get("stability_score", 1.0)),
            retrievability_score=float(data.get("retrievability_score", 1.0)),
            last_accessed_at=data.get("last_accessed_at"),
            last_used_at=data.get("last_used_at"),
            conflict_group_id=data.get("conflict_group_id"),
            conflict_status=data.get("conflict_status", "none"),
            superseded_by=data.get("superseded_by"),
            deletion_state=data.get("deletion_state", "active"),
            policy_flags=data.get("policy_flags", []),
        )
