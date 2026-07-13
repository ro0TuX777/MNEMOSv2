"""Test-only fixture identity authority for G4 synthetic development cases."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes, digest


class IdentityValidationError(ValueError):
    """Raised when a synthetic principal envelope fails closed."""


@dataclass(frozen=True)
class ValidatedPrincipal:
    principal_id: str
    tenant_memberships: tuple[dict[str, Any], ...]
    scoped_roles: tuple[dict[str, Any], ...]
    delegation: dict[str, Any] | None
    identity_snapshot_version: str
    expires_at: int
    credential_fingerprint: str


class FixtureIdentityAuthority:
    """Owns an in-memory HMAC key that is never exposed through public state."""

    __slots__ = ("__key", "issuer")

    def __init__(
        self,
        *,
        key: bytes,
        _capability: object,
        issuer: str = "mnemos-g4-fixture",
    ):
        if _capability is not _HARNESS_CAPABILITY:
            raise PermissionError("fixture identity authority is harness-owned")
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("fixture HMAC key must contain at least 32 bytes")
        self.__key = bytes(key)
        self.issuer = issuer

    def issue(self, claims: dict[str, Any]) -> dict[str, Any]:
        body = {"issuer": self.issuer, "claims": claims}
        signature = hmac.new(
            self.__key, canonical_bytes(body), hashlib.sha256
        ).hexdigest()
        return {**body, "signature": signature}

    def validate(self, envelope: dict[str, Any], *, now: int) -> ValidatedPrincipal:
        if set(envelope) != {"issuer", "claims", "signature"}:
            raise IdentityValidationError("IDENTITY_ENVELOPE_INVALID")
        if envelope["issuer"] != self.issuer:
            raise IdentityValidationError("IDENTITY_ISSUER_UNKNOWN")
        claims = envelope["claims"]
        if not isinstance(claims, dict):
            raise IdentityValidationError("IDENTITY_CLAIMS_INVALID")
        expected = hmac.new(
            self.__key,
            canonical_bytes({"issuer": envelope["issuer"], "claims": claims}),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(str(envelope["signature"]), expected):
            raise IdentityValidationError("IDENTITY_SIGNATURE_INVALID")
        required = {
            "principal_id",
            "tenant_memberships",
            "scoped_roles",
            "delegation",
            "identity_snapshot_version",
            "expires_at",
        }
        if set(claims) != required:
            raise IdentityValidationError("IDENTITY_CLAIMS_INVALID")
        if int(claims["expires_at"]) < now:
            raise IdentityValidationError("IDENTITY_EXPIRED")
        if not isinstance(claims["tenant_memberships"], list) or not isinstance(
            claims["scoped_roles"], list
        ):
            raise IdentityValidationError("IDENTITY_CLAIMS_INVALID")
        return ValidatedPrincipal(
            principal_id=str(claims["principal_id"]),
            tenant_memberships=tuple(claims["tenant_memberships"]),
            scoped_roles=tuple(claims["scoped_roles"]),
            delegation=(
                dict(claims["delegation"])
                if isinstance(claims["delegation"], dict)
                else None
            ),
            identity_snapshot_version=str(claims["identity_snapshot_version"]),
            expires_at=int(claims["expires_at"]),
            credential_fingerprint=digest(
                {"issuer": envelope["issuer"], "signature": envelope["signature"]}
            ),
        )


def mutate_envelope(envelope: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Apply an adversarial mutation after signing; never accepts key material."""

    changed = {
        "issuer": envelope["issuer"],
        "claims": dict(envelope["claims"]),
        "signature": envelope["signature"],
    }
    if mutation == "none":
        return changed
    if mutation == "forged_signature":
        changed["signature"] = "0" * 64
    elif mutation == "unknown_issuer":
        changed["issuer"] = "untrusted-fixture-issuer"
    elif mutation == "tampered_claims":
        changed["claims"]["principal_id"] = "principal-attacker"
    else:
        raise IdentityValidationError("IDENTITY_MUTATION_UNKNOWN")
    return changed


# Private construction capability imported only by the fixture harness.
_HARNESS_CAPABILITY = object()
