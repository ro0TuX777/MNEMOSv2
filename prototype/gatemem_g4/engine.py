"""Deterministic G3-contract reference engine for synthetic G4 cases."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .audit import ContentFreeAuditSink
from .canonical import digest
from .identity import (
    FixtureIdentityAuthority,
    IdentityValidationError,
    mutate_envelope,
)


CASE_FIELDS = {
    "case_id",
    "now",
    "principal_claims",
    "envelope_mutation",
    "request",
    "session",
    "entitlement",
    "artifact",
    "policy",
    "replay_token",
}

CLASSIFICATION = {"public": 0, "internal": 1, "restricted": 2, "secret": 3}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    outcome: str
    reason_code: str
    redacted: bool
    disclosed_count: int
    provenance_integrity: bool
    package: dict[str, Any] | None
    audit_event: dict[str, Any]

    def content_free_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "redacted": self.redacted,
            "disclosed_count": self.disclosed_count,
            "provenance_integrity": self.provenance_integrity,
            "decision_fingerprint": self.audit_event["decision_fingerprint"],
            "package_digest": self.audit_event["package_digest"],
        }


def validate_case_schema(case: dict[str, Any]) -> None:
    if set(case) != CASE_FIELDS:
        raise ValueError("CASE_SCHEMA_INVALID")
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("CASE_SCHEMA_INVALID")
    for field in ("principal_claims", "request", "session", "entitlement", "artifact", "policy"):
        if not isinstance(case[field], dict):
            raise ValueError("CASE_SCHEMA_INVALID")
    if case["replay_token"] is not None and not isinstance(case["replay_token"], dict):
        raise ValueError("CASE_SCHEMA_INVALID")


def _active(item: dict[str, Any], now: int) -> bool:
    return (
        item.get("status", "active") == "active"
        and int(item.get("valid_from", 0)) <= now
        and (item.get("valid_until") is None or int(item["valid_until"]) >= now)
    )


def _redact(
    content: dict[str, Any], obligations: list[dict[str, Any]], forbidden: list[str]
) -> tuple[dict[str, Any], bool]:
    transformed = deepcopy(content)
    applied = False
    for obligation in obligations:
        if set(obligation) != {"type", "value"}:
            raise ValueError("REDACTION_OBLIGATION_INVALID")
        kind = obligation["type"]
        value = str(obligation["value"])
        if kind == "remove_field":
            if value not in transformed:
                raise ValueError("REDACTION_FIELD_MISSING")
            del transformed[value]
            applied = True
        elif kind == "redact_literal":
            found = False
            for key, item in tuple(transformed.items()):
                if isinstance(item, str) and value in item:
                    transformed[key] = item.replace(value, "[REDACTED]")
                    found = True
            if not found:
                raise ValueError("REDACTION_SPAN_MISSING")
            applied = True
        elif kind == "non_authoritative_label":
            transformed["authority_label"] = value
            applied = True
        else:
            raise ValueError("REDACTION_OBLIGATION_UNKNOWN")
    serialized = str(transformed)
    if any(value and value in serialized for value in forbidden):
        raise ValueError("REDACTION_VERIFICATION_FAILED")
    return transformed, applied


def _snapshot(case: dict[str, Any], principal_id: str) -> dict[str, str]:
    return {
        "identity": digest(case["principal_claims"]),
        "principal": digest(principal_id),
        "entitlement": digest(case["entitlement"]),
        "policy": digest(case["policy"]),
        "descriptor": digest(case["artifact"]["descriptor"]),
        "redaction": digest(case["policy"].get("redaction_version")),
    }


def replay_token_for(case: dict[str, Any]) -> dict[str, str]:
    return _snapshot(case, str(case["principal_claims"]["principal_id"]))


class OfflineAuthorizationEngine:
    def __init__(self, authority: FixtureIdentityAuthority):
        self._authority = authority

    def evaluate(self, case: dict[str, Any], audit: ContentFreeAuditSink) -> CaseResult:
        case_id = str(case.get("case_id", "schema-rejected"))
        try:
            validate_case_schema(case)
        except ValueError:
            return self._deny_schema(case_id, audit)

        now = int(case["now"])
        claims = case["principal_claims"]
        envelope = mutate_envelope(
            self._authority.issue(claims), str(case["envelope_mutation"])
        )
        try:
            principal = self._authority.validate(envelope, now=now)
        except IdentityValidationError as exc:
            return self._finish(case, audit, "DENIED", str(exc), None, False, "")

        request = case["request"]
        session = case["session"]
        entitlement = case["entitlement"]
        descriptor = case["artifact"]["descriptor"]

        membership = [
            item
            for item in principal.tenant_memberships
            if item.get("tenant_id") == session.get("tenant_id") and _active(item, now)
        ]
        if len(membership) != 1:
            return self._finish(case, audit, "DENIED", "TENANT_MEMBERSHIP_DENIED", None, False, principal.identity_snapshot_version)
        if request.get("tenant_id") != session.get("tenant_id"):
            return self._finish(case, audit, "DENIED", "CALLER_SCOPE_WIDENING", None, False, principal.identity_snapshot_version)
        if request.get("session_id") != session.get("session_id"):
            return self._finish(case, audit, "DENIED", "SESSION_SCOPE_INVALID", None, False, principal.identity_snapshot_version)
        if not _active(session, now):
            return self._finish(case, audit, "DENIED", "SESSION_INACTIVE", None, False, principal.identity_snapshot_version)
        relationship_principal = principal.principal_id
        if principal.principal_id not in session.get("allowed_principal_ids", []):
            delegation = principal.delegation
            delegation_valid = bool(
                delegation
                and delegation.get("delegating_principal_id")
                in session.get("allowed_principal_ids", [])
                and request.get("operation")
                in delegation.get("permitted_operations", [])
                and request.get("purpose") in delegation.get("permitted_purposes", [])
                and int(delegation.get("expires_at", -1)) >= now
            )
            if not delegation_valid:
                return self._finish(case, audit, "DENIED", "SESSION_ACCESS_DENIED", None, False, principal.identity_snapshot_version)
            relationship_principal = str(delegation["delegating_principal_id"])

        roles = [
            item
            for item in principal.scoped_roles
            if item.get("tenant_id") == session.get("tenant_id")
            and item.get("resource_scope") == session.get("session_id")
            and _active(item, now)
        ]
        if not roles:
            return self._finish(case, audit, "DENIED", "ROLE_ASSIGNMENT_DENIED", None, False, principal.identity_snapshot_version)
        role_ids = {str(item.get("role_id")) for item in roles}

        for denial in entitlement.get("denials", []):
            if denial.get("matching_scope") in {"all", request.get("operation")}:
                return self._finish(case, audit, "DENIED", "OVERRIDING_DENIAL", None, False, principal.identity_snapshot_version)

        grant = None
        failure = "ENTITLEMENT_MISSING"
        for candidate in entitlement.get("grants", []):
            if not _active(candidate, now):
                failure = "ENTITLEMENT_EXPIRED"
                continue
            checks = [
                (bool(role_ids & set(candidate.get("role_ids", []))), "ROLE_NOT_ENTITLED"),
                (request.get("operation") in candidate.get("operations", []), "OPERATION_DENIED"),
                (request.get("purpose") in candidate.get("purposes", []), "PURPOSE_DENIED"),
                (session.get("session_id") in candidate.get("resource_scopes", []), "RESOURCE_SCOPE_DENIED"),
                (relationship_principal in candidate.get("subject_principal_ids", []), "SUBJECT_RELATIONSHIP_DENIED"),
                (descriptor.get("artifact_class") in candidate.get("artifact_classes", []), "ARTIFACT_CLASS_DENIED"),
                (descriptor.get("source_class") in candidate.get("source_classes", []), "SOURCE_CLASS_DENIED"),
                (
                    CLASSIFICATION.get(str(descriptor.get("classification")), 99)
                    <= CLASSIFICATION.get(str(candidate.get("classification_ceiling")), -1),
                    "CLASSIFICATION_EXCEEDED",
                ),
            ]
            failed = next((reason for ok, reason in checks if not ok), None)
            if failed is None:
                grant = candidate
                break
            failure = failed
        if grant is None:
            return self._finish(case, audit, "DENIED", failure, None, False, principal.identity_snapshot_version)
        if descriptor.get("tenant_id") != session.get("tenant_id") or descriptor.get("session_id") != session.get("session_id"):
            return self._finish(case, audit, "DENIED", "DESCRIPTOR_SCOPE_MISMATCH", None, False, principal.identity_snapshot_version)
        if descriptor.get("lineage_complete") is not True:
            return self._finish(case, audit, "DENIED", "LINEAGE_INCOMPLETE", None, False, principal.identity_snapshot_version)

        replay = case["replay_token"]
        if replay is not None and replay != _snapshot(case, principal.principal_id):
            return self._finish(case, audit, "DENIED", "REPLAY_CONFLICT", None, False, principal.identity_snapshot_version)

        try:
            transformed, redacted = _redact(
                case["artifact"]["content"],
                list(grant.get("obligations", [])),
                list(case["artifact"].get("forbidden_after_redaction", [])),
            )
        except ValueError as exc:
            return self._finish(case, audit, "DENIED", str(exc), None, False, principal.identity_snapshot_version)

        if int(request.get("max_records", 0)) < 1 or len(str(transformed)) > int(
            request.get("max_characters", 0)
        ):
            return self._finish(case, audit, "DENIED", "PACKAGE_BUDGET_EXCEEDED", None, False, principal.identity_snapshot_version)

        package = {
            "outcome": "DISCLOSED_WITH_REDACTION" if redacted else "DISCLOSED",
            "content": transformed,
            "artifact_id": descriptor["artifact_id"],
            "parent_source_ids": list(descriptor.get("parent_source_ids", [])),
            "policy_version": case["policy"]["version"],
        }
        return self._finish(
            case,
            audit,
            "DISCLOSED_WITH_REDACTION" if redacted else "DISCLOSED",
            "AUTHORIZED",
            package,
            redacted,
            principal.identity_snapshot_version,
        )

    def _deny_schema(self, case_id: str, audit: ContentFreeAuditSink) -> CaseResult:
        stub = {
            "case_id": case_id,
            "now": 0,
            "request": {"tenant_id": "", "session_id": "", "operation": "", "purpose": "", "query_text": ""},
            "session": {"tenant_id": "", "session_id": ""},
            "entitlement": {"policy_version": "unknown"},
            "policy": {"version": "unknown"},
        }
        return self._finish(stub, audit, "DENIED", "CASE_SCHEMA_INVALID", None, False, "unknown")

    def _finish(
        self,
        case: dict[str, Any],
        audit: ContentFreeAuditSink,
        outcome: str,
        reason: str,
        package: dict[str, Any] | None,
        redacted: bool,
        identity_version: str,
    ) -> CaseResult:
        request = case.get("request", {})
        session = case.get("session", {})
        entitlement = case.get("entitlement", {})
        policy = case.get("policy", {})
        principal_id = str(case.get("principal_claims", {}).get("principal_id", "unknown"))
        decision = {
            "case_id": case.get("case_id", "schema-rejected"),
            "outcome": outcome,
            "reason": reason,
            "policy_version": policy.get("version", "unknown"),
            "entitlement_fingerprint": digest(entitlement),
        }
        event = {
            "schema_version": "gatemem-g4-audit-v1",
            "event_id": digest({"decision": decision, "event": 1}),
            "request_digest": digest(request),
            "principal_digest": digest(principal_id),
            "tenant_digest": digest(session.get("tenant_id", "")),
            "session_digest": digest(session.get("session_id", "")),
            "consumer_id": "gatemem-g4-offline-harness",
            "operation": str(request.get("operation", "unknown")),
            "purpose": str(request.get("purpose", "unknown")),
            "outcome": outcome,
            "reason_code": reason,
            "policy_version": str(policy.get("version", "unknown")),
            "identity_snapshot_version": identity_version,
            "entitlement_fingerprint": digest(entitlement),
            "decision_fingerprint": digest(decision),
            "package_digest": digest(package) if package is not None else None,
            "disclosed_count": 1 if package is not None else 0,
            "redacted_count": 1 if redacted else 0,
            "retention_class": "g4_case_audit",
            "retention_days": 30,
            "event_time": int(case.get("now", 0)),
        }
        audit.emit(event)
        provenance = bool(
            package is None
            or (
                package.get("artifact_id")
                == case.get("artifact", {}).get("descriptor", {}).get("artifact_id")
                and package.get("parent_source_ids")
                == case.get("artifact", {}).get("descriptor", {}).get("parent_source_ids", [])
            )
        )
        return CaseResult(
            case_id=str(case.get("case_id", "schema-rejected")),
            outcome=outcome,
            reason_code=reason,
            redacted=redacted,
            disclosed_count=1 if package is not None else 0,
            provenance_integrity=provenance,
            package=package,
            audit_event=event,
        )
