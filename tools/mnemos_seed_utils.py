"""Utilities for deterministic MNEMOS repo seeding."""

from __future__ import annotations

import hashlib
import uuid
from typing import Iterable


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def normalized_content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def build_seed_identity(
    *,
    canonical_source_uri: str,
    seed_kind: str,
    schema_version: str,
) -> str:
    return f"{canonical_source_uri}::{seed_kind}::{schema_version}"


def build_seed_engram_id(seed_identity: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mnemos-seed:{seed_identity}"))


def build_seed_snapshot_id(components: Iterable[str]) -> str:
    joined = "||".join(str(component) for component in components)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
