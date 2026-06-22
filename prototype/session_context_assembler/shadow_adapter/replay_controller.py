"""Bounded in-memory replay cache pinned to request and policy fingerprints."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

from .errors import ShadowAdapterError


@dataclass(frozen=True)
class ReplayEntry:
    request_digest: str
    policy_fingerprint: str
    expires_at: datetime
    response: dict


class ReplayController:
    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str, str, str], ReplayEntry] = {}
        self.write_count = 0

    @staticmethod
    def key(request: dict) -> Tuple[str, str, str, str]:
        identity = request["consumer_identity"]
        major = request["adapter_contract_version"].split(".", 1)[0]
        return (
            identity["consumer_id"],
            identity["adapter_id"],
            request["request_id"],
            major,
        )

    def lookup(
        self,
        request: dict,
        request_digest: str,
        policy_fingerprint: str,
        now: datetime,
    ) -> dict | None:
        entry = self._entries.get(self.key(request))
        if entry is None:
            return None
        if now >= entry.expires_at:
            raise ShadowAdapterError("PACKAGE_EXPIRED", "Cached package has expired.")
        if entry.request_digest != request_digest:
            raise ShadowAdapterError(
                "REQUEST_REPLAY_CONFLICT", "Request ID was reused with different input."
            )
        if entry.policy_fingerprint != policy_fingerprint:
            raise ShadowAdapterError(
                "REPLAY_POLICY_MISMATCH", "Issuance policy fingerprint changed."
            )
        return copy.deepcopy(entry.response)

    def put(
        self,
        request: dict,
        request_digest: str,
        policy_fingerprint: str,
        expires_at: datetime,
        response: dict,
    ) -> None:
        self._entries[self.key(request)] = ReplayEntry(
            request_digest=request_digest,
            policy_fingerprint=policy_fingerprint,
            expires_at=expires_at,
            response=copy.deepcopy(response),
        )
        self.write_count += 1

    def invalidate_all(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
