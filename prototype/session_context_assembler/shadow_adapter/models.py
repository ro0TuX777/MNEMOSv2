"""Plain immutable inputs for the isolated local shadow adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class LocalTransportContext:
    peer_consumer_id: str
    channel_id: str
    authenticated: bool = True
    confidentiality_protected: bool = True
    integrity_protected: bool = True


@dataclass(frozen=True)
class LocalAssemblyInputs:
    session_id: str
    task_id: str
    conversation_history: Tuple[Dict, ...]
    snapshot_reference: str
    artifact_classes_by_turn: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySnapshot:
    consumer_id: str
    adapter_id: str
    authorization_reference: str
    authorization_grant_fingerprint: str
    permitted_purpose: str
    tenant_scope: str
    session_scope: str
    allowed_artifact_classes: FrozenSet[str]
    allowed_source_ids: FrozenSet[str]
    allowed_engram_ids: FrozenSet[str]
    denied_turn_ids: FrozenSet[str]
    redacted_content_by_turn_id: Dict[str, str]
    eligibility_policy_id: str
    disclosure_policy_id: str
    redaction_policy_id: str
    budget_policy_id: str
    max_token_budget: int
    assembler_policy_version: str
    adapter_contract_version: str
    authorization_expires_at: datetime
    package_ttl_seconds: int = 300
    replay_grace_seconds: int = 30
