"""Artifact-local response construction and canonical package digesting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .canonical import response_digest, sha256_digest, verify_response_digest
from .errors import ShadowAdapterError
from .models import LocalAssemblyInputs, PolicySnapshot
from .policy_and_disclosure_boundary import EffectivePolicy
from prototype.session_context_assembler.extractor import extract_ids_from_turn
from prototype.session_context_assembler.models import turn_from_dict


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _rationale_codes(rationale: list[str]) -> list[str]:
    codes = []
    for item in rationale:
        if "semantic fill blocked" in item:
            code = "SEMANTIC_FILL_WITHHELD_AFTER_ABSTENTION"
        elif " skipped:" in item:
            code = "CANDIDATE_SKIPPED_BUDGET"
        elif "tier=1" in item:
            code = "MANDATORY_DECISION_SELECTED"
        elif "tier=2" in item:
            code = "MANDATORY_CONTRADICTION_SELECTED"
        elif "tier=3" in item:
            code = "MANDATORY_SOURCE_SELECTED"
        else:
            code = "SEMANTIC_CONTEXT_SELECTED"
        if code not in codes:
            codes.append(code)
    return codes


def build_response(
    request: dict,
    inputs: LocalAssemblyInputs,
    policy: PolicySnapshot,
    effective: EffectivePolicy,
    package: dict,
    now: datetime,
) -> dict:
    turns = {turn["turn_id"]: turn for turn in effective.filtered_history}
    package_id = "PKG-" + sha256_digest(
        {
            "request_id": request["request_id"],
            "consumer_id": policy.consumer_id,
            "policy_fingerprint": effective.policy_fingerprint,
            "issued_at": _iso(now),
        }
    )[:24]
    artifacts = []
    label_decisions = set()
    label_sources = set()
    labels_index = []
    for index, label in enumerate(package["synthetic_context_labels"], 1):
        artifact_id = f"{package_id}-ART-{index:03d}"
        parent_turn_ids = list(label["parent_turn_ids"])
        parent_decisions = list(label["parent_engram_ids"])
        parent_sources = list(label["parent_source_ids"])
        label_decisions.update(parent_decisions)
        label_sources.update(parent_sources)
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": "selected_session_segment",
                "content": [
                    _content_turn(turns[turn_id])
                    for turn_id in parent_turn_ids
                ],
                "synthetic_context": True,
                "non_authoritative": True,
                "non_promotable": True,
                "parent_engram_ids": parent_decisions,
                "parent_source_ids": parent_sources,
                "lineage_complete": True,
            }
        )
        labels_index.append({"artifact_id": artifact_id, "label": "synthetic_context"})
    if set(package["selected_parent_engram_ids"]) != label_decisions:
        raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Decision lineage is incomplete.")
    if set(package["selected_source_ids"]) != label_sources:
        raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Source lineage is incomplete.")
    if not artifacts:
        raise ShadowAdapterError("SCOPE_EMPTY", "No context package could be assembled.")

    policy_expiry = policy.authorization_expires_at
    if policy_expiry.tzinfo is None:
        policy_expiry = policy_expiry.replace(tzinfo=timezone.utc)
    expires_at = min(policy_expiry, now + timedelta(seconds=policy.package_ttl_seconds))
    rationale_codes = _rationale_codes(package["selection_rationale"])
    if package["context_budget_insufficient"]:
        rationale_codes.append("MANDATORY_CANDIDATE_OMITTED_BUDGET")
    response = {
        "request_id": request["request_id"],
        "package_id": package_id,
        "consumer_id": policy.consumer_id,
        "adapter_contract_version": policy.adapter_contract_version,
        "issued_at": _iso(now),
        "expires_at": _iso(expires_at),
        "context_package": {
            "selected_session_artifacts": artifacts,
            "selection_metadata": {
                "selected_episode_ids": list(package["selected_episode_ids"]),
                "selection_rationale_codes": rationale_codes,
            },
        },
        "synthetic_context_labels": labels_index,
        "provenance_metadata": {
            "corpus_or_snapshot_reference": inputs.snapshot_reference,
            "package_lineage_complete": True,
            "eligibility_policy_id": policy.eligibility_policy_id,
            "disclosure_policy_id": policy.disclosure_policy_id,
            "redaction_policy_id": policy.redaction_policy_id,
        },
        "abstention_state": {
            "context_budget_insufficient": package["context_budget_insufficient"],
            "omitted_required_artifact_types": list(
                package["omitted_required_artifact_types"]
            ),
            "selection_abstention_reason": (
                "Mandatory eligible artifact types exceeded the bounded context budget."
                if package["context_budget_insufficient"]
                else None
            ),
        },
        "token_estimate": package["token_estimate"],
        "policy_identifiers": {
            "assembler_policy_version": policy.assembler_policy_version,
            "adapter_contract_version": policy.adapter_contract_version,
            "budget_policy_id": policy.budget_policy_id,
        },
    }
    response["package_digest"] = {
        "algorithm": "sha256",
        "canonicalization": "canonical-json-v1",
        "value": response_digest(response),
    }
    validate_response_contract(response)
    return response


def _content_turn(turn: dict) -> dict:
    result = {
        "turn_id": turn["turn_id"],
        "speaker": turn["speaker"],
        "content": turn["content"],
    }
    if turn.get("linked_source_ids"):
        result["source_links"] = sorted(turn["linked_source_ids"])
    return result


def validate_response_contract(response: dict) -> None:
    if not verify_response_digest(response):
        raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Package integrity failed.")
    artifacts = response["context_package"]["selected_session_artifacts"]
    indexed = {
        item["artifact_id"] for item in response["synthetic_context_labels"]
        if item.get("label") == "synthetic_context"
    }
    artifact_ids = {artifact["artifact_id"] for artifact in artifacts}
    if indexed != artifact_ids:
        raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Synthetic labels are incomplete.")
    for artifact in artifacts:
        if not (
            artifact.get("synthetic_context") is True
            and artifact.get("non_authoritative") is True
            and artifact.get("non_promotable") is True
            and artifact.get("lineage_complete") is True
        ):
            raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Artifact labeling is incomplete.")
        decisions = set()
        sources = set()
        for item in artifact["content"]:
            turn = turn_from_dict(
                {
                    "turn_id": item["turn_id"],
                    "speaker": item["speaker"],
                    "content": item["content"],
                    "linked_source_ids": item.get("source_links", []),
                }
            )
            turn_decisions, turn_sources = extract_ids_from_turn(turn)
            decisions.update(turn_decisions)
            sources.update(turn_sources)
        if decisions != set(artifact["parent_engram_ids"]):
            raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Decision lineage does not match.")
        if sources != set(artifact["parent_source_ids"]):
            raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Source lineage does not match.")
    metadata = response["context_package"]["selection_metadata"]
    mandatory_omitted = "MANDATORY_CANDIDATE_OMITTED_BUDGET" in metadata[
        "selection_rationale_codes"
    ]
    abstention = response["abstention_state"]
    if mandatory_omitted != bool(abstention["context_budget_insufficient"]):
        raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Abstention state is inconsistent.")
    if mandatory_omitted and not (
        abstention["omitted_required_artifact_types"]
        and abstention["selection_abstention_reason"]
    ):
        raise ShadowAdapterError("LINEAGE_INCOMPLETE", "Abstention disclosure is incomplete.")
