"""Canonical JSON and SHA-256 integrity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def response_digest(response: dict) -> str:
    payload = dict(response)
    payload.pop("package_digest", None)
    return sha256_digest(payload)


def verify_response_digest(response: dict) -> bool:
    digest = response.get("package_digest", {})
    return (
        digest.get("algorithm") == "sha256"
        and digest.get("canonicalization") == "canonical-json-v1"
        and digest.get("value") == response_digest(response)
    )
