"""Phase 1 Memory Over Maps schemas."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _require_non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass
class SourceArtifact:
    artifact_id: str
    artifact_type: str = "document"
    source_uri: str = ""
    source_authority: str = "unknown"
    content_hash: str = ""
    version_id: str = "v1"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    lifecycle_state: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty("artifact_id", self.artifact_id)
        _require_non_empty("artifact_type", self.artifact_type)
        _require_non_empty("version_id", self.version_id)
        _require_non_empty("lifecycle_state", self.lifecycle_state)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "source_uri": self.source_uri,
            "source_authority": self.source_authority,
            "content_hash": self.content_hash,
            "version_id": self.version_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "lifecycle_state": self.lifecycle_state,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceArtifact":
        artifact = cls(
            artifact_id=data.get("artifact_id", ""),
            artifact_type=data.get("artifact_type", "document"),
            source_uri=data.get("source_uri", ""),
            source_authority=data.get("source_authority", "unknown"),
            content_hash=data.get("content_hash", ""),
            version_id=data.get("version_id", "v1"),
            created_at=data.get("created_at", _utc_now()),
            updated_at=data.get("updated_at", _utc_now()),
            lifecycle_state=data.get("lifecycle_state", "active"),
            metadata=data.get("metadata", {}),
        )
        artifact.validate()
        return artifact


@dataclass
class Chunk:
    chunk_id: str
    artifact_id: str
    version_id: str = "v1"
    chunk_index: int = 0
    chunk_hash: str = ""
    text_span_ref: Optional[str] = None
    embedding_ref: Optional[str] = None
    lexical_ref: Optional[str] = None
    governance_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty("chunk_id", self.chunk_id)
        _require_non_empty("artifact_id", self.artifact_id)
        _require_non_empty("version_id", self.version_id)
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "chunk_id": self.chunk_id,
            "artifact_id": self.artifact_id,
            "version_id": self.version_id,
            "chunk_index": self.chunk_index,
            "chunk_hash": self.chunk_hash,
            "text_span_ref": self.text_span_ref,
            "embedding_ref": self.embedding_ref,
            "lexical_ref": self.lexical_ref,
            "governance_ref": self.governance_ref,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        chunk = cls(
            chunk_id=data.get("chunk_id", ""),
            artifact_id=data.get("artifact_id", ""),
            version_id=data.get("version_id", "v1"),
            chunk_index=int(data.get("chunk_index", 0)),
            chunk_hash=data.get("chunk_hash", ""),
            text_span_ref=data.get("text_span_ref"),
            embedding_ref=data.get("embedding_ref"),
            lexical_ref=data.get("lexical_ref"),
            governance_ref=data.get("governance_ref"),
            metadata=data.get("metadata", {}),
        )
        chunk.validate()
        return chunk


@dataclass
class DerivedView:
    view_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    view_type: str = "evidence_bundle"
    inputs: Dict[str, List[str]] = field(default_factory=dict)
    query_fingerprint: str = ""
    governance_state_hash: str = ""
    synthesis_policy: str = "default"
    created_at: str = field(default_factory=_utc_now)
    invalidated_at: Optional[str] = None
    cacheable: bool = False
    reproducible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty("view_id", self.view_id)
        _require_non_empty("view_type", self.view_type)
        if not isinstance(self.inputs, dict):
            raise ValueError("inputs must be a dictionary")
        if not self.inputs:
            raise ValueError("inputs must declare at least one lineage input")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "view_id": self.view_id,
            "view_type": self.view_type,
            "inputs": self.inputs,
            "query_fingerprint": self.query_fingerprint,
            "governance_state_hash": self.governance_state_hash,
            "synthesis_policy": self.synthesis_policy,
            "created_at": self.created_at,
            "invalidated_at": self.invalidated_at,
            "cacheable": self.cacheable,
            "reproducible": self.reproducible,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DerivedView":
        view = cls(
            view_id=data.get("view_id", str(uuid.uuid4())),
            view_type=data.get("view_type", ""),
            inputs=data.get("inputs", {}),
            query_fingerprint=data.get("query_fingerprint", ""),
            governance_state_hash=data.get("governance_state_hash", ""),
            synthesis_policy=data.get("synthesis_policy", "default"),
            created_at=data.get("created_at", _utc_now()),
            invalidated_at=data.get("invalidated_at"),
            cacheable=bool(data.get("cacheable", False)),
            reproducible=bool(data.get("reproducible", True)),
            metadata=data.get("metadata", {}),
        )
        view.validate()
        return view


@dataclass
class EvidenceBundle(DerivedView):
    supporting_artifact_ids: List[str] = field(default_factory=list)
    supporting_chunk_ids: List[str] = field(default_factory=list)
    support_roles: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "supporting_artifact_ids": self.supporting_artifact_ids,
                "supporting_chunk_ids": self.supporting_chunk_ids,
                "support_roles": self.support_roles,
                "exclusions": self.exclusions,
            }
        )
        return base


@dataclass
class ContradictionBundle(DerivedView):
    contradiction_cluster_id: str = ""
    winner_ids: List[str] = field(default_factory=list)
    loser_ids: List[str] = field(default_factory=list)
    comparison_factors: List[str] = field(default_factory=list)
    resolution_trace: str = ""

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "contradiction_cluster_id": self.contradiction_cluster_id,
                "winner_ids": self.winner_ids,
                "loser_ids": self.loser_ids,
                "comparison_factors": self.comparison_factors,
                "resolution_trace": self.resolution_trace,
            }
        )
        return base


@dataclass
class PreferenceSnapshot(DerivedView):
    subject_id: str = "default"
    preferred_memory_ids: List[str] = field(default_factory=list)
    suppressed_memory_ids: List[str] = field(default_factory=list)
    rationale_trace: str = ""

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "subject_id": self.subject_id,
                "preferred_memory_ids": self.preferred_memory_ids,
                "suppressed_memory_ids": self.suppressed_memory_ids,
                "rationale_trace": self.rationale_trace,
            }
        )
        return base


@dataclass
class TimelineSummary(DerivedView):
    timeline_subject: str = "query"
    ordered_event_refs: List[str] = field(default_factory=list)
    source_artifact_ids: List[str] = field(default_factory=list)
    temporal_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "timeline_subject": self.timeline_subject,
                "ordered_event_refs": self.ordered_event_refs,
                "source_artifact_ids": self.source_artifact_ids,
                "temporal_confidence": self.temporal_confidence,
            }
        )
        return base
