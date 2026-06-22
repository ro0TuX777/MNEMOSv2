"""Local content-free technical shadow telemetry sink."""

from __future__ import annotations

from copy import deepcopy

from .errors import ShadowAdapterError
from .response_builder_and_digest import validate_response_contract

ALLOWED_EVENT_FIELDS = {
    "request_id",
    "package_id",
    "consumer_id",
    "adapter_id",
    "package_digest",
    "adapter_contract_version",
    "assembler_policy_version",
    "budget_policy_id",
    "token_estimate",
    "context_budget_insufficient",
    "lineage_complete",
    "outcome_code",
    "channel_id",
    "event_time",
    "replayed",
}


class ContentFreeShadowSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    @staticmethod
    def validate_event(event: dict) -> None:
        if set(event) != ALLOWED_EVENT_FIELDS:
            raise ShadowAdapterError("DISCLOSURE_DENIED", "Telemetry field is not permitted.")

    def record_success(
        self,
        response: dict,
        adapter_id: str,
        channel_id: str,
        event_time: str,
        replayed: bool,
    ) -> None:
        validate_response_contract(response)
        event = {
            "request_id": response["request_id"],
            "package_id": response["package_id"],
            "consumer_id": response["consumer_id"],
            "adapter_id": adapter_id,
            "package_digest": response["package_digest"]["value"],
            "adapter_contract_version": response["adapter_contract_version"],
            "assembler_policy_version": response["policy_identifiers"][
                "assembler_policy_version"
            ],
            "budget_policy_id": response["policy_identifiers"]["budget_policy_id"],
            "token_estimate": response["token_estimate"],
            "context_budget_insufficient": response["abstention_state"][
                "context_budget_insufficient"
            ],
            "lineage_complete": response["provenance_metadata"][
                "package_lineage_complete"
            ],
            "outcome_code": "SHADOW_PACKAGE_ASSEMBLED",
            "channel_id": channel_id,
            "event_time": event_time,
            "replayed": replayed,
        }
        self.validate_event(event)
        self.events.append(deepcopy(event))

    def record_error(
        self,
        request_id: str | None,
        consumer_id: str | None,
        adapter_id: str | None,
        contract_version: str,
        channel_id: str,
        event_time: str,
        error_code: str,
    ) -> None:
        event = {
            "request_id": request_id,
            "package_id": None,
            "consumer_id": consumer_id,
            "adapter_id": adapter_id,
            "package_digest": None,
            "adapter_contract_version": contract_version,
            "assembler_policy_version": None,
            "budget_policy_id": None,
            "token_estimate": None,
            "context_budget_insufficient": error_code == "BUDGET_INSUFFICIENT",
            "lineage_complete": False,
            "outcome_code": error_code,
            "channel_id": channel_id,
            "event_time": event_time,
            "replayed": False,
        }
        self.validate_event(event)
        self.events.append(event)
