"""
Associative Routing View E0 — data contracts.

Cue, Tag, and routing-response shapes exactly as specified by the E0
authorization. These are retrieval metadata only — see module docstring in
``router.py`` for the non-authority invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Fields a Cue or Tag record is permitted to carry. Anything outside this
#: set is rejected by the registry loader as a potential authority field
#: (trust/utility/promotion/governance-mutating data has no place here).
ALLOWED_CUE_FIELDS = {
    "cue_id",
    "canonical_value",
    "cue_type",
    "normalized_value",
    "aliases",
    "source_record_ids",
    "projection_snapshot",
    "status",
}

ALLOWED_TAG_FIELDS = {
    "tag_id",
    "tag_type",
    "from_cue_id",
    "to_content_id",
    "source_record_ids",
    "evidence_locator",
    "derivation_method",
    "schema_version",
    "projection_snapshot",
    "status",
    "tag_owner",
    "created_at",
}

#: Tag types allowed under the E0 relationship-creation policy.
ALLOWED_TAG_TYPES = {
    "blocked_by",
    "depends_on",
    "evidence_for",
    "supersedes",
    "superseded_by",
    "contradicts",
    "unresolved_with",
    "frozen_baseline_of",
    "handoff_for",
    "current_state_of",
    "preceded_by",
    "replaced_by",
}

#: Derivation methods allowed in E0 (no LLM extraction).
ALLOWED_DERIVATION_METHODS = {
    "controlled_metadata",
    "manually_curated_fixture",
}


@dataclass(frozen=True)
class Cue:
    cue_id: str
    canonical_value: str
    cue_type: str
    normalized_value: str
    source_record_ids: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    projection_snapshot: Optional[str] = None
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "canonical_value": self.canonical_value,
            "cue_type": self.cue_type,
            "normalized_value": self.normalized_value,
            "source_record_ids": list(self.source_record_ids),
            "aliases": list(self.aliases),
            "projection_snapshot": self.projection_snapshot,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cue":
        return cls(
            cue_id=data["cue_id"],
            canonical_value=data["canonical_value"],
            cue_type=data["cue_type"],
            normalized_value=data["normalized_value"],
            source_record_ids=list(data.get("source_record_ids", [])),
            aliases=list(data.get("aliases", [])),
            projection_snapshot=data.get("projection_snapshot"),
            status=data.get("status", "active"),
        )


@dataclass(frozen=True)
class Tag:
    tag_id: str
    tag_type: str
    from_cue_id: str
    to_content_id: str
    source_record_ids: List[str]
    evidence_locator: Dict[str, Any]
    derivation_method: str
    schema_version: str = "1"
    tag_owner: Optional[str] = None
    created_at: Optional[str] = None
    projection_snapshot: Optional[str] = None
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_id": self.tag_id,
            "tag_type": self.tag_type,
            "from_cue_id": self.from_cue_id,
            "to_content_id": self.to_content_id,
            "source_record_ids": list(self.source_record_ids),
            "evidence_locator": dict(self.evidence_locator),
            "derivation_method": self.derivation_method,
            "schema_version": self.schema_version,
            "tag_owner": self.tag_owner,
            "created_at": self.created_at,
            "projection_snapshot": self.projection_snapshot,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tag":
        return cls(
            tag_id=data["tag_id"],
            tag_type=data["tag_type"],
            from_cue_id=data["from_cue_id"],
            to_content_id=data["to_content_id"],
            source_record_ids=list(data.get("source_record_ids", [])),
            evidence_locator=dict(data.get("evidence_locator", {})),
            derivation_method=data["derivation_method"],
            schema_version=data.get("schema_version", "1"),
            tag_owner=data.get("tag_owner"),
            created_at=data.get("created_at"),
            projection_snapshot=data.get("projection_snapshot"),
            status=data.get("status", "active"),
        )


@dataclass(frozen=True)
class ContentRef:
    """A reference to an existing source-grounded MNEMOS record.

    E0 never stores or duplicates content — only a stable id, a pointer to
    the real source artifact, and a human label for explanation text.
    """

    content_id: str
    source_uri: str
    title: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "source_uri": self.source_uri,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentRef":
        return cls(
            content_id=data["content_id"],
            source_uri=data["source_uri"],
            title=data["title"],
        )


@dataclass(frozen=True)
class RoutingPath:
    path_id: str
    cue_ids: List[str]
    tag_ids: List[str]
    content_ids: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "cue_ids": list(self.cue_ids),
            "tag_ids": list(self.tag_ids),
            "content_ids": list(self.content_ids),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class Abstention:
    reason_code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"reason_code": self.reason_code, "message": self.message}


@dataclass(frozen=True)
class RoutingResponse:
    query: str
    projection_snapshot: str
    routing_result: str  # "resolved" | "abstained"
    matched_cues: List[str]
    routing_paths: List[RoutingPath]
    candidate_content_ids: List[str]
    abstention: Optional[Abstention]
    source_lineage_complete: bool = True
    non_authoritative_projection: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "projection_snapshot": self.projection_snapshot,
            "routing_result": self.routing_result,
            "matched_cues": list(self.matched_cues),
            "routing_paths": [p.to_dict() for p in self.routing_paths],
            "candidate_content_ids": list(self.candidate_content_ids),
            "abstention": self.abstention.to_dict() if self.abstention else None,
            "integrity": {
                "source_lineage_complete": self.source_lineage_complete,
                "non_authoritative_projection": self.non_authoritative_projection,
            },
        }
