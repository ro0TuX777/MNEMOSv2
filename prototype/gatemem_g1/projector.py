"""Allowlist-only GateMem-visible input projection.

The raw checkpoint may contain evaluator annotations, but this module never
reads or enumerates them. Only explicitly visible fields are accessed.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import sha256_digest
from .models import CleanInputProjection, CleanRequester, CleanTurn

PROHIBITED_SCORING_FIELDS = frozenset(
    {
        "query_type",
        "attack_type",
        "expected_action",
        "judge_spec",
        "leak_targets",
        "gold_answer_structured",
        "gold_refusal_category",
    }
)


class ProjectionError(ValueError):
    pass


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if value is None or not str(value).strip():
        raise ProjectionError(f"Missing required visible field: {key}")
    return str(value)


def _copy_permitted(value: Any, *, path: str = "permitted_metadata") -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in PROHIBITED_SCORING_FIELDS:
                raise ProjectionError(f"Scoring-only field is prohibited at {path}.{key}")
            result[key] = _copy_permitted(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _copy_permitted(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ProjectionError(f"Unsupported metadata type at {path}: {type(value).__name__}")


def project_clean_input(
    episode: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    permitted_metadata: Mapping[str, Any] | None = None,
) -> CleanInputProjection:
    """Project one checkpoint without touching evaluator-only annotations."""

    episode_id = _required_text(episode, "episode_id")
    checkpoint_id = _required_text(checkpoint, "checkpoint_id")
    checkpoint_episode_id = _required_text(checkpoint, "episode_id")
    if checkpoint_episode_id != episode_id:
        raise ProjectionError("Checkpoint and episode identity do not match.")

    as_of_turn_id = _required_text(checkpoint, "as_of_turn_id")
    query_text = _required_text(checkpoint, "query_text")
    asker = checkpoint.get("asker")
    if not isinstance(asker, Mapping):
        raise ProjectionError("Visible requester identity is missing.")
    requester = CleanRequester(
        principal_id=_required_text(asker, "principal_id"),
        role=_required_text(asker, "role"),
    )

    raw_turns = episode.get("turns")
    if not isinstance(raw_turns, Sequence) or isinstance(raw_turns, (str, bytes)):
        raise ProjectionError("Episode turns must be an ordered sequence.")

    projected_turns: list[CleanTurn] = []
    seen_ids: set[str] = set()
    found_boundary = False
    for raw_turn in raw_turns:
        if not isinstance(raw_turn, Mapping):
            raise ProjectionError("Every visible turn must be an object.")
        turn_id = _required_text(raw_turn, "turn_id")
        if turn_id in seen_ids:
            raise ProjectionError(f"Duplicate turn_id: {turn_id}")
        seen_ids.add(turn_id)
        speaker = raw_turn.get("speaker")
        if not isinstance(speaker, Mapping):
            raise ProjectionError(f"Turn {turn_id} has no visible speaker identity.")
        timestamp = raw_turn.get("timestamp")
        turn_kind = raw_turn.get("turn_kind")
        projected_turns.append(
            CleanTurn(
                turn_id=turn_id,
                speaker_principal_id=_required_text(speaker, "principal_id"),
                speaker_role=_required_text(speaker, "role"),
                text=str(raw_turn.get("text") or ""),
                timestamp=str(timestamp) if timestamp is not None else None,
                turn_kind=str(turn_kind) if turn_kind is not None else None,
            )
        )
        if turn_id == as_of_turn_id:
            found_boundary = True
            break
    if not found_boundary:
        raise ProjectionError("as_of_turn_id is not present in the episode.")

    clean_metadata = _copy_permitted(permitted_metadata or {})
    digest_input = {
        "contract_version": "gatemem-g1-clean-input-v1",
        "checkpoint_id": checkpoint_id,
        "episode_id": episode_id,
        "as_of_turn_id": as_of_turn_id,
        "requester": requester.to_dict(),
        "query_text": query_text,
        "ordered_turns": [turn.to_dict() for turn in projected_turns],
        "permitted_metadata": clean_metadata,
    }
    return CleanInputProjection(
        checkpoint_id=checkpoint_id,
        episode_id=episode_id,
        as_of_turn_id=as_of_turn_id,
        requester=requester,
        query_text=query_text,
        ordered_turns=tuple(projected_turns),
        permitted_metadata=copy.deepcopy(clean_metadata),
        projection_digest=sha256_digest(digest_input),
    )


def clean_projection_from_dict(value: Mapping[str, Any]) -> CleanInputProjection:
    """Load and verify exactly the frozen G1 clean-projection schema."""

    required = {
        "contract_version",
        "checkpoint_id",
        "episode_id",
        "as_of_turn_id",
        "requester",
        "query_text",
        "ordered_turns",
        "permitted_metadata",
        "projection_digest",
    }
    if set(value) != required:
        raise ProjectionError(
            f"Projection schema mismatch: extra={sorted(set(value) - required)}, "
            f"missing={sorted(required - set(value))}"
        )
    if value["contract_version"] != "gatemem-g1-clean-input-v1":
        raise ProjectionError("Unsupported clean-projection contract version.")
    requester_raw = value["requester"]
    if not isinstance(requester_raw, Mapping) or set(requester_raw) != {
        "principal_id",
        "role",
    }:
        raise ProjectionError("Requester schema mismatch.")
    requester = CleanRequester(
        principal_id=_required_text(requester_raw, "principal_id"),
        role=_required_text(requester_raw, "role"),
    )
    turns_raw = value["ordered_turns"]
    if not isinstance(turns_raw, list) or not turns_raw:
        raise ProjectionError("ordered_turns must be a non-empty list.")
    turns: list[CleanTurn] = []
    seen: set[str] = set()
    for item in turns_raw:
        if not isinstance(item, Mapping):
            raise ProjectionError("Every projected turn must be an object.")
        allowed_turn_fields = {"turn_id", "speaker", "text", "timestamp", "turn_kind"}
        if not {"turn_id", "speaker", "text"}.issubset(item) or not set(item).issubset(
            allowed_turn_fields
        ):
            raise ProjectionError("Projected turn schema mismatch.")
        turn_id = _required_text(item, "turn_id")
        if turn_id in seen:
            raise ProjectionError(f"Duplicate turn_id: {turn_id}")
        seen.add(turn_id)
        speaker = item["speaker"]
        if not isinstance(speaker, Mapping) or set(speaker) != {"principal_id", "role"}:
            raise ProjectionError("Projected speaker schema mismatch.")
        turns.append(
            CleanTurn(
                turn_id=turn_id,
                speaker_principal_id=_required_text(speaker, "principal_id"),
                speaker_role=_required_text(speaker, "role"),
                text=str(item.get("text") or ""),
                timestamp=(str(item["timestamp"]) if "timestamp" in item else None),
                turn_kind=(str(item["turn_kind"]) if "turn_kind" in item else None),
            )
        )
    as_of_turn_id = _required_text(value, "as_of_turn_id")
    if turns[-1].turn_id != as_of_turn_id:
        raise ProjectionError("Projection must end exactly at as_of_turn_id.")
    metadata_raw = value["permitted_metadata"]
    if not isinstance(metadata_raw, Mapping):
        raise ProjectionError("permitted_metadata must be an object.")
    clean_metadata = _copy_permitted(metadata_raw)
    digest_input = {
        "contract_version": value["contract_version"],
        "checkpoint_id": _required_text(value, "checkpoint_id"),
        "episode_id": _required_text(value, "episode_id"),
        "as_of_turn_id": as_of_turn_id,
        "requester": requester.to_dict(),
        "query_text": _required_text(value, "query_text"),
        "ordered_turns": [turn.to_dict() for turn in turns],
        "permitted_metadata": clean_metadata,
    }
    expected_digest = sha256_digest(digest_input)
    supplied_digest = _required_text(value, "projection_digest")
    if supplied_digest != expected_digest:
        raise ProjectionError("Clean-projection digest verification failed.")
    return CleanInputProjection(
        checkpoint_id=digest_input["checkpoint_id"],
        episode_id=digest_input["episode_id"],
        as_of_turn_id=as_of_turn_id,
        requester=requester,
        query_text=digest_input["query_text"],
        ordered_turns=tuple(turns),
        permitted_metadata=copy.deepcopy(clean_metadata),
        projection_digest=supplied_digest,
    )
