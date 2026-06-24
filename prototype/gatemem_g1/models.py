"""Immutable data contracts for the offline-only GateMem G1 prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class CleanTurn:
    turn_id: str
    speaker_principal_id: str
    speaker_role: str
    text: str
    timestamp: str | None = None
    turn_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "turn_id": self.turn_id,
            "speaker": {
                "principal_id": self.speaker_principal_id,
                "role": self.speaker_role,
            },
            "text": self.text,
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.turn_kind is not None:
            result["turn_kind"] = self.turn_kind
        return result


@dataclass(frozen=True)
class CleanRequester:
    principal_id: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"principal_id": self.principal_id, "role": self.role}


@dataclass(frozen=True)
class CleanInputProjection:
    checkpoint_id: str
    episode_id: str
    as_of_turn_id: str
    requester: CleanRequester
    query_text: str
    ordered_turns: Tuple[CleanTurn, ...]
    permitted_metadata: Mapping[str, Any]
    projection_digest: str
    contract_version: str = "gatemem-g1-clean-input-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "checkpoint_id": self.checkpoint_id,
            "episode_id": self.episode_id,
            "as_of_turn_id": self.as_of_turn_id,
            "requester": self.requester.to_dict(),
            "query_text": self.query_text,
            "ordered_turns": [turn.to_dict() for turn in self.ordered_turns],
            "permitted_metadata": dict(self.permitted_metadata),
            "projection_digest": self.projection_digest,
        }


@dataclass(frozen=True)
class RetrievedArtifact:
    """Ephemeral result supplied by an injected offline retrieval probe."""

    record_id: str
    text: str
    source_ids: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DisclosureResult:
    """Result supplied by an injected offline disclosure/redaction probe."""

    outcome: str  # allowed | redacted | denied
    response_text: str = ""
    disclosed_record_ids: Tuple[str, ...] = field(default_factory=tuple)
    policy_reference: str = "offline-shadow-policy"


@dataclass(frozen=True)
class ShadowObservation:
    checkpoint_id: str
    projection_digest: str
    retrieval_outcome: str
    disclosure_outcome: str
    response_text: str
    used_record_ids: Tuple[str, ...]
    retrieved_count: int
    disclosed_count: int
    deletion_evaluation_status: str
    policy_reference: str
    shadow_only: bool = True
    observation_contract_version: str = "gatemem-g1-shadow-observation-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_contract_version": self.observation_contract_version,
            "checkpoint_id": self.checkpoint_id,
            "projection_digest": self.projection_digest,
            "retrieval_outcome": self.retrieval_outcome,
            "disclosure_outcome": self.disclosure_outcome,
            "response_text": self.response_text,
            "used_record_ids": list(self.used_record_ids),
            "retrieved_count": self.retrieved_count,
            "disclosed_count": self.disclosed_count,
            "deletion_evaluation_status": self.deletion_evaluation_status,
            "policy_reference": self.policy_reference,
            "shadow_only": self.shadow_only,
        }


_OBSERVATION_FIELDS = frozenset(
    {
        "observation_contract_version",
        "checkpoint_id",
        "projection_digest",
        "retrieval_outcome",
        "disclosure_outcome",
        "response_text",
        "used_record_ids",
        "retrieved_count",
        "disclosed_count",
        "deletion_evaluation_status",
        "policy_reference",
        "shadow_only",
    }
)


def shadow_observation_from_dict(value: Mapping[str, Any]) -> ShadowObservation:
    """Load only the frozen G1 observation schema; reject every extra field."""

    if set(value) != _OBSERVATION_FIELDS:
        extra = sorted(set(value) - _OBSERVATION_FIELDS)
        missing = sorted(_OBSERVATION_FIELDS - set(value))
        raise ValueError(f"Observation schema mismatch: extra={extra}, missing={missing}")
    if value["observation_contract_version"] != "gatemem-g1-shadow-observation-v1":
        raise ValueError("Unsupported observation contract version.")
    used_ids = value["used_record_ids"]
    if not isinstance(used_ids, list) or not all(isinstance(item, str) for item in used_ids):
        raise ValueError("used_record_ids must be a list of strings.")
    if value["shadow_only"] is not True:
        raise ValueError("G1 observations must be shadow-only.")
    return ShadowObservation(
        checkpoint_id=str(value["checkpoint_id"]),
        projection_digest=str(value["projection_digest"]),
        retrieval_outcome=str(value["retrieval_outcome"]),
        disclosure_outcome=str(value["disclosure_outcome"]),
        response_text=str(value["response_text"]),
        used_record_ids=tuple(used_ids),
        retrieved_count=int(value["retrieved_count"]),
        disclosed_count=int(value["disclosed_count"]),
        deletion_evaluation_status=str(value["deletion_evaluation_status"]),
        policy_reference=str(value["policy_reference"]),
        shadow_only=True,
        observation_contract_version=str(value["observation_contract_version"]),
    )
