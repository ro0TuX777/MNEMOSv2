"""Strict content-free audit sink for the G4 offline reference lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import write_jsonl


AUDIT_FIELDS = {
    "schema_version",
    "event_id",
    "request_digest",
    "principal_digest",
    "tenant_digest",
    "session_digest",
    "consumer_id",
    "operation",
    "purpose",
    "outcome",
    "reason_code",
    "policy_version",
    "identity_snapshot_version",
    "entitlement_fingerprint",
    "decision_fingerprint",
    "package_digest",
    "disclosed_count",
    "redacted_count",
    "retention_class",
    "retention_days",
    "event_time",
}


class ContentFreeAuditSink:
    def __init__(self, *, prohibited_canaries: Iterable[str] = ()):
        self._events: list[dict[str, Any]] = []
        self._canaries = tuple(value for value in prohibited_canaries if value)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def emit(self, event: dict[str, Any]) -> None:
        unknown = set(event) - AUDIT_FIELDS
        missing = AUDIT_FIELDS - set(event)
        if unknown or missing:
            raise ValueError(
                f"audit schema violation: unknown={sorted(unknown)} missing={sorted(missing)}"
            )
        serialized = json.dumps(event, ensure_ascii=False, sort_keys=True)
        leaked = [canary for canary in self._canaries if canary in serialized]
        if leaked:
            raise ValueError("audit content canary detected")
        self._events.append(dict(event))

    def write(self, path: str | Path) -> None:
        write_jsonl(path, self._events)
