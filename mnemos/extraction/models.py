"""
Models for offline extraction proofs (SMC-1B).
Defines ExtractionReceipt, PassageNode, and ExtractionBatchManifest.

Hash Scopes:
- source_hash: Full source Engram text hash.
- passage_text_hash: Extracted passage text hash.
- output_hash: Canonicalized PassageNode JSON hash.
"""

import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from mnemos.governance.models.memory_state import GovernanceMeta


@dataclass
class ExtractionReceipt:
    receipt_id: str
    batch_id: str
    source_engram_id: str
    source_uri: str
    artifact_id: str
    chunk_id: str
    provenance_span: Tuple[int, int]
    source_hash: str
    passage_text_hash: str
    extractor_version: str
    prompt_hash: str
    model_name_version: str
    timestamp: str
    extraction_mode: str
    governance_snapshot: Dict[str, Any]
    output_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "batch_id": self.batch_id,
            "source_engram_id": self.source_engram_id,
            "source_uri": self.source_uri,
            "artifact_id": self.artifact_id,
            "chunk_id": self.chunk_id,
            "provenance_span": self.provenance_span,
            "source_hash": self.source_hash,
            "passage_text_hash": self.passage_text_hash,
            "extractor_version": self.extractor_version,
            "prompt_hash": self.prompt_hash,
            "model_name_version": self.model_name_version,
            "timestamp": self.timestamp,
            "extraction_mode": self.extraction_mode,
            "governance_snapshot": self.governance_snapshot,
            "output_hash": self.output_hash,
        }


@dataclass
class PassageNode:
    passage_id: str
    text: str
    source_engram_id: str
    provenance_span: Tuple[int, int]
    extraction_receipt_id: str
    inherited_governance: Dict[str, Any]
    status: str = "CANDIDATE"
    node_type: str = "passage"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "status": self.status,
            "node_type": self.node_type,
            "text": self.text,
            "source_engram_id": self.source_engram_id,
            "provenance_span": self.provenance_span,
            "extraction_receipt_id": self.extraction_receipt_id,
            "inherited_governance": self.inherited_governance,
            "created_at": self.created_at,
        }


@dataclass
class ExtractionBatchManifest:
    batch_id: str
    timestamp: str
    processed_count: int
    success_count: int
    error_count: int
    failures: List[str]
    input_fixture_path: str = ""
    input_fixture_hash: str = ""
    output_dir: str = ""
    extractor_version: str = ""
    validation_status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "failures": self.failures,
            "input_fixture_path": self.input_fixture_path,
            "input_fixture_hash": self.input_fixture_hash,
            "output_dir": self.output_dir,
            "extractor_version": self.extractor_version,
            "validation_status": self.validation_status,
        }

@dataclass
class FactExtractionReceipt:
    receipt_id: str
    batch_id: str
    source_engram_id: str
    passage_node_id: str
    source_uri: str
    artifact_id: str
    chunk_id: str
    passage_span: Tuple[int, int]
    evidence_text_hash: str
    parent_passage_text_hash: str
    extractor_version: str
    prompt_hash: str
    model_name_version: str
    timestamp: str
    extraction_mode: str
    inherited_governance_snapshot: Dict[str, Any]
    output_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "batch_id": self.batch_id,
            "source_engram_id": self.source_engram_id,
            "passage_node_id": self.passage_node_id,
            "source_uri": self.source_uri,
            "artifact_id": self.artifact_id,
            "chunk_id": self.chunk_id,
            "passage_span": self.passage_span,
            "evidence_text_hash": self.evidence_text_hash,
            "parent_passage_text_hash": self.parent_passage_text_hash,
            "extractor_version": self.extractor_version,
            "prompt_hash": self.prompt_hash,
            "model_name_version": self.model_name_version,
            "timestamp": self.timestamp,
            "extraction_mode": self.extraction_mode,
            "inherited_governance_snapshot": self.inherited_governance_snapshot,
            "output_hash": self.output_hash,
        }

@dataclass
class FactNode:
    fact_id: str
    statement: str
    evidence_text: str
    passage_span: Tuple[int, int]
    passage_node_id: str
    source_engram_id: str
    fact_receipt_id: str
    parent_passage_receipt_id: str
    source_uri: str
    artifact_id: str
    chunk_id: str
    evidence_hash: str
    passage_text_hash: str
    confidence_score: float
    inherited_governance: Dict[str, Any]
    validation_status: str
    rejection_reason: str = ""
    structured_claim: Optional[Dict[str, str]] = None
    status: str = "CANDIDATE"
    node_type: str = "fact"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "status": self.status,
            "node_type": self.node_type,
            "statement": self.statement,
            "evidence_text": self.evidence_text,
            "passage_span": self.passage_span,
            "passage_node_id": self.passage_node_id,
            "source_engram_id": self.source_engram_id,
            "fact_receipt_id": self.fact_receipt_id,
            "parent_passage_receipt_id": self.parent_passage_receipt_id,
            "source_uri": self.source_uri,
            "artifact_id": self.artifact_id,
            "chunk_id": self.chunk_id,
            "evidence_hash": self.evidence_hash,
            "passage_text_hash": self.passage_text_hash,
            "confidence_score": self.confidence_score,
            "inherited_governance": self.inherited_governance,
            "validation_status": self.validation_status,
            "rejection_reason": self.rejection_reason,
            "structured_claim": self.structured_claim,
            "created_at": self.created_at,
        }

@dataclass
class FactExtractionBatchManifest:
    batch_id: str
    timestamp: str
    input_passage_count: int
    generated_facts_count: int
    unsupported_facts_count: int
    rejected_facts_count: int
    failures: List[str]
    input_dir: str = ""
    output_dir: str = ""
    extractor_version: str = ""
    validation_status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "input_passage_count": self.input_passage_count,
            "generated_facts_count": self.generated_facts_count,
            "unsupported_facts_count": self.unsupported_facts_count,
            "rejected_facts_count": self.rejected_facts_count,
            "failures": self.failures,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "extractor_version": self.extractor_version,
            "validation_status": self.validation_status,
        }

@dataclass
class FactReviewLabel:
    fact_id: str
    review_label: str
    review_reason: str
    reviewer_type: str
    source_file: str
    passage_node_id: str
    source_engram_id: str
    receipt_id: str
    traceability_verified: bool
    governance_verified: bool
    atomicity_verified: bool
    faithfulness_verified: bool
    recommended_action: str
    duplicate_group_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "review_label": self.review_label,
            "review_reason": self.review_reason,
            "reviewer_type": self.reviewer_type,
            "source_file": self.source_file,
            "passage_node_id": self.passage_node_id,
            "source_engram_id": self.source_engram_id,
            "receipt_id": self.receipt_id,
            "traceability_verified": self.traceability_verified,
            "governance_verified": self.governance_verified,
            "atomicity_verified": self.atomicity_verified,
            "faithfulness_verified": self.faithfulness_verified,
            "recommended_action": self.recommended_action,
            "duplicate_group_id": self.duplicate_group_id,
        }

@dataclass
class FactPromotionReceipt:
    receipt_id: str
    promoted_fact_id: str
    human_review_label_id: str
    operator_id: str
    timestamp: str
    source_governance_snapshot: Dict[str, Any]
    conflict_sweep_hash: str
    promotion_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "promoted_fact_id": self.promoted_fact_id,
            "human_review_label_id": self.human_review_label_id,
            "operator_id": self.operator_id,
            "timestamp": self.timestamp,
            "source_governance_snapshot": self.source_governance_snapshot,
            "conflict_sweep_hash": self.conflict_sweep_hash,
            "promotion_status": self.promotion_status,
        }

@dataclass
class FactLifecycleEvent:
    event_id: str
    fact_id: str
    event_type: str  # e.g., PROMOTION_APPROVED, DOWNGRADED, REJECTED
    timestamp: str
    operator_id: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "fact_id": self.fact_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "operator_id": self.operator_id,
            "reason": self.reason,
            "metadata": self.metadata,
        }
