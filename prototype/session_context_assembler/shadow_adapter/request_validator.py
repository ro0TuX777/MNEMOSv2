"""Strict request and authenticated local-transport validation."""

from __future__ import annotations

from datetime import datetime, timezone

from .errors import ShadowAdapterError
from .models import LocalTransportContext

REQUEST_FIELDS = {
    "request_id",
    "current_task",
    "consumer_session_reference",
    "eligible_context_scope",
    "requested_budget",
    "consumer_identity",
    "authorization_context",
    "adapter_contract_version",
}


def validate_request(
    request: dict, transport: LocalTransportContext, now: datetime
) -> None:
    if set(request) != REQUEST_FIELDS:
        raise ShadowAdapterError("CONTRACT_VERSION_UNSUPPORTED", "Invalid request contract.")
    if not all(
        isinstance(request.get(field), str) and request[field].strip()
        for field in ("request_id", "current_task", "consumer_session_reference")
    ):
        raise ShadowAdapterError("SCOPE_EMPTY", "Required request scope is empty.")
    if not (
        transport.authenticated
        and transport.confidentiality_protected
        and transport.integrity_protected
    ):
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Authenticated transport required.")
    identity = request["consumer_identity"]
    if identity.get("consumer_id") != transport.peer_consumer_id:
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Consumer identity mismatch.")
    if not transport.channel_id:
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Authenticated channel required.")
    budget = request["requested_budget"].get("token_limit")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ShadowAdapterError("BUDGET_INSUFFICIENT", "Invalid context budget.")
    auth = request["authorization_context"]
    if auth.get("permitted_operation") != "read_context_package":
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Operation is not permitted.")
    expiry = auth.get("expiry")
    try:
        parsed = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Authorization expiry is invalid.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if now >= parsed:
        raise ShadowAdapterError("AUTHORIZATION_DENIED", "Authorization has expired.")
