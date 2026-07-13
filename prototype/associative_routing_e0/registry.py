"""
Load and validate the controlled metadata registries (Cue, Tag, Content)
that back the Associative Routing View E0 projection.

This module enforces the E0 relationship-creation policy: every record must
be present in the allowlisted field set, every Tag must cite resolvable
source records, and no authority-bearing field may appear.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .models import (
    ALLOWED_CUE_FIELDS,
    ALLOWED_DERIVATION_METHODS,
    ALLOWED_TAG_FIELDS,
    ALLOWED_TAG_TYPES,
    ContentRef,
    Cue,
    Tag,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class RegistryValidationError(ValueError):
    """Raised when a fixture record violates the E0 relationship policy."""


@dataclass(frozen=True)
class Corpus:
    """A loaded, validated set of Content/Cue/Tag registries plus their raw
    bytes (needed by the projection builder to hash source inputs)."""

    content_index: Dict[str, ContentRef]
    cues: Dict[str, Cue]
    tags: Dict[str, Tag]
    raw_source_index_bytes: bytes
    raw_cue_registry_bytes: bytes
    raw_tag_registry_bytes: bytes


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def _check_allowed_fields(record: dict, allowed: set, kind: str, record_id: str) -> None:
    extra = set(record.keys()) - allowed
    if extra:
        raise RegistryValidationError(
            f"{kind} {record_id!r} contains disallowed field(s) {sorted(extra)}; "
            "only retrieval metadata fields are permitted in E0."
        )


def load_corpus(fixtures_dir: Path = FIXTURES_DIR) -> Corpus:
    """Load and validate the Content, Cue, and Tag registries from disk."""

    source_index_raw, source_index_bytes = _read_json(fixtures_dir / "source_index.json")
    cue_registry_raw, cue_registry_bytes = _read_json(fixtures_dir / "cue_registry.json")
    tag_registry_raw, tag_registry_bytes = _read_json(fixtures_dir / "tag_registry.json")

    content_index: Dict[str, ContentRef] = {}
    for entry in source_index_raw.get("content", []):
        ref = ContentRef.from_dict(entry)
        if ref.content_id in content_index:
            raise RegistryValidationError(f"Duplicate content_id {ref.content_id!r}")
        content_index[ref.content_id] = ref

    cues: Dict[str, Cue] = {}
    for entry in cue_registry_raw.get("cues", []):
        _check_allowed_fields(entry, ALLOWED_CUE_FIELDS, "Cue", entry.get("cue_id", "<unknown>"))
        cue = Cue.from_dict(entry)
        if cue.cue_id in cues:
            raise RegistryValidationError(f"Duplicate cue_id {cue.cue_id!r}")
        if not cue.source_record_ids:
            raise RegistryValidationError(f"Cue {cue.cue_id!r} has no source_record_ids")
        for sid in cue.source_record_ids:
            if sid not in content_index:
                raise RegistryValidationError(
                    f"Cue {cue.cue_id!r} cites unresolvable source_record_id {sid!r}"
                )
        cues[cue.cue_id] = cue

    tags: Dict[str, Tag] = {}
    for entry in tag_registry_raw.get("tags", []):
        _check_allowed_fields(entry, ALLOWED_TAG_FIELDS, "Tag", entry.get("tag_id", "<unknown>"))
        tag = Tag.from_dict(entry)
        if tag.tag_id in tags:
            raise RegistryValidationError(f"Duplicate tag_id {tag.tag_id!r}")
        if tag.tag_type not in ALLOWED_TAG_TYPES:
            raise RegistryValidationError(f"Tag {tag.tag_id!r} has disallowed tag_type {tag.tag_type!r}")
        if tag.derivation_method not in ALLOWED_DERIVATION_METHODS:
            raise RegistryValidationError(
                f"Tag {tag.tag_id!r} has disallowed derivation_method {tag.derivation_method!r}"
            )
        if not tag.source_record_ids:
            raise RegistryValidationError(f"Tag {tag.tag_id!r} has no source_record_ids (rejected)")
        for sid in tag.source_record_ids:
            if sid not in content_index:
                raise RegistryValidationError(
                    f"Tag {tag.tag_id!r} cites unresolvable source_record_id {sid!r} (rejected)"
                )
        if tag.from_cue_id not in cues:
            raise RegistryValidationError(
                f"Tag {tag.tag_id!r} references nonexistent from_cue_id {tag.from_cue_id!r}"
            )
        if tag.to_content_id not in content_index:
            raise RegistryValidationError(
                f"Tag {tag.tag_id!r} references nonexistent to_content_id {tag.to_content_id!r} (rejected)"
            )
        if not tag.evidence_locator or not tag.evidence_locator.get("source_uri"):
            raise RegistryValidationError(f"Tag {tag.tag_id!r} is missing an evidence_locator.source_uri")
        tags[tag.tag_id] = tag

    orphan_cues: List[str] = [
        cue_id for cue_id in cues if not any(t.from_cue_id == cue_id for t in tags.values())
    ]
    if orphan_cues:
        raise RegistryValidationError(f"Orphan cue(s) with no outgoing tag: {sorted(orphan_cues)}")

    return Corpus(
        content_index=content_index,
        cues=cues,
        tags=tags,
        raw_source_index_bytes=source_index_bytes,
        raw_cue_registry_bytes=cue_registry_bytes,
        raw_tag_registry_bytes=tag_registry_bytes,
    )
