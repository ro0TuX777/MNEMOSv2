"""Content-free diagnostics for the GateMem G2 offline adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from prototype.gatemem_g1.models import ShadowObservation


@dataclass(frozen=True)
class G2Diagnostic:
    checkpoint_id: str
    projection_digest: str
    retrieved_record_ids: Tuple[str, ...]
    disclosed_record_ids: Tuple[str, ...]
    denied_record_ids: Tuple[str, ...]
    cross_principal_candidate_count: int
    blocked_cross_principal_count: int
    retrieval_candidate_count: int
    redaction_applied: bool
    denial_applied: bool
    deletion_evaluation_status: str
    normalized_action: str
    provenance_integrity: bool
    adapter_version: str = "gatemem-g2-offline-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "checkpoint_id": self.checkpoint_id,
            "projection_digest": self.projection_digest,
            "retrieved_record_ids": list(self.retrieved_record_ids),
            "disclosed_record_ids": list(self.disclosed_record_ids),
            "denied_record_ids": list(self.denied_record_ids),
            "cross_principal_candidate_count": self.cross_principal_candidate_count,
            "blocked_cross_principal_count": self.blocked_cross_principal_count,
            "retrieval_candidate_count": self.retrieval_candidate_count,
            "redaction_applied": self.redaction_applied,
            "denial_applied": self.denial_applied,
            "deletion_evaluation_status": self.deletion_evaluation_status,
            "normalized_action": self.normalized_action,
            "provenance_integrity": self.provenance_integrity,
            "offline_only": True,
            "deletion_capability_claim": False,
        }


@dataclass(frozen=True)
class G2Result:
    observation: ShadowObservation
    prediction: dict[str, Any]
    diagnostic: G2Diagnostic

