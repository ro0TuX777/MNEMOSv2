"""Deny-by-default local policy, scope, disclosure, and redaction boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple

from prototype.session_context_assembler.extractor import extract_ids_from_turn
from prototype.session_context_assembler.models import turn_from_dict

from .canonical import sha256_digest
from .errors import ShadowAdapterError
from .models import LocalAssemblyInputs, PolicySnapshot


@dataclass(frozen=True)
class EffectivePolicy:
    filtered_history: Tuple[Dict, ...]
    effective_budget: int
    policy_fingerprint: str


def _allows(allowed: frozenset[str], values: set[str]) -> bool:
    return "*" in allowed or values <= set(allowed)


def evaluate_policy(
    request: dict,
    inputs: LocalAssemblyInputs,
    policy: PolicySnapshot,
    now: datetime,
) -> EffectivePolicy:
    if policy.authorization_expires_at.tzinfo is None:
        policy_expiry = policy.authorization_expires_at.replace(tzinfo=timezone.utc)
    else:
        policy_expiry = policy.authorization_expires_at
    identity = request["consumer_identity"]
    auth = request["authorization_context"]
    scope = request["eligible_context_scope"]
    requested_classes = set(scope.get("allowed_artifact_classes", []))

    if now >= policy_expiry:
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Authorization has expired.")
    if (
        identity.get("consumer_id") != policy.consumer_id
        or identity.get("adapter_id") != policy.adapter_id
        or identity.get("purpose") != policy.permitted_purpose
        or auth.get("authorization_reference") != policy.authorization_reference
    ):
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Authorization does not match policy.")
    if (
        scope.get("tenant_scope") != policy.tenant_scope
        or scope.get("session_scope") != policy.session_scope
        or request["consumer_session_reference"] != policy.session_scope
        or inputs.session_id != policy.session_scope
    ):
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Requested scope is not permitted.")
    if not requested_classes or not requested_classes <= set(policy.allowed_artifact_classes):
        raise ShadowAdapterError("DISCLOSURE_DENIED", "Artifact class is not permitted.")
    if request["adapter_contract_version"] != policy.adapter_contract_version:
        raise ShadowAdapterError(
            "CONTRACT_VERSION_UNSUPPORTED", "Adapter contract version is unsupported."
        )
    if request["requested_budget"].get("budget_policy_id") != policy.budget_policy_id:
        raise ShadowAdapterError(
            "POLICY_VERSION_INCOMPATIBLE", "Budget policy version does not match."
        )
    if scope.get("eligibility_policy_id") != policy.eligibility_policy_id:
        raise ShadowAdapterError(
            "POLICY_VERSION_INCOMPATIBLE", "Eligibility policy version does not match."
        )

    filtered = []
    for original in inputs.conversation_history:
        turn_id = original["turn_id"]
        if turn_id in policy.denied_turn_ids or not original.get("eligible", True):
            continue
        artifact_class = inputs.artifact_classes_by_turn.get(turn_id, "session_turn")
        if artifact_class not in requested_classes:
            continue
        turn = dict(original)
        if turn_id in policy.redacted_content_by_turn_id:
            turn["content"] = policy.redacted_content_by_turn_id[turn_id]
        parsed = turn_from_dict(turn)
        decisions, sources = extract_ids_from_turn(parsed)
        if not _allows(policy.allowed_source_ids, set(sources)):
            raise ShadowAdapterError("DISCLOSURE_DENIED", "Source disclosure is not permitted.")
        if not _allows(policy.allowed_engram_ids, set(decisions)):
            raise ShadowAdapterError("DISCLOSURE_DENIED", "Artifact disclosure is not permitted.")
        filtered.append(turn)
    if not filtered:
        raise ShadowAdapterError("SCOPE_EMPTY", "No eligible context is available.")

    effective_budget = min(
        request["requested_budget"]["token_limit"], policy.max_token_budget
    )
    if effective_budget <= 0:
        raise ShadowAdapterError("BUDGET_INSUFFICIENT", "No context budget is available.")
    request_fingerprint = {
        "request": request,
        "consumer_id": policy.consumer_id,
        "adapter_id": policy.adapter_id,
        "authorization_grant_fingerprint": policy.authorization_grant_fingerprint,
        "tenant_scope": policy.tenant_scope,
        "session_scope": policy.session_scope,
        "allowed_artifact_classes": sorted(policy.allowed_artifact_classes),
        "allowed_source_ids": sorted(policy.allowed_source_ids),
        "allowed_engram_ids": sorted(policy.allowed_engram_ids),
        "denied_turn_ids": sorted(policy.denied_turn_ids),
        "redacted_turn_ids": sorted(policy.redacted_content_by_turn_id),
        "redacted_content_digest": sha256_digest(policy.redacted_content_by_turn_id),
        "filtered_history_digest": sha256_digest(filtered),
        "snapshot_reference": inputs.snapshot_reference,
        "eligibility_policy_id": policy.eligibility_policy_id,
        "disclosure_policy_id": policy.disclosure_policy_id,
        "redaction_policy_id": policy.redaction_policy_id,
        "budget_policy_id": policy.budget_policy_id,
        "effective_budget": effective_budget,
        "assembler_policy_version": policy.assembler_policy_version,
        "adapter_contract_version": policy.adapter_contract_version,
    }
    return EffectivePolicy(
        filtered_history=tuple(filtered),
        effective_budget=effective_budget,
        policy_fingerprint=sha256_digest(request_fingerprint),
    )
