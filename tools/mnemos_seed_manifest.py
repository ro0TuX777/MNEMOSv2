"""Seed manifest helpers for reproducible MNEMOS repo seeding."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.mnemos_seed_utils import build_seed_snapshot_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "data" / "seed_manifests" / "repo_seed_manifest.json"
MANIFEST_SCHEMA_VERSION = "repo_seed_manifest_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_seed_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "sections": {},
            "seed_snapshot_id": "unknown",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("seed manifest must be a JSON object")
    payload.setdefault("manifest_schema_version", MANIFEST_SCHEMA_VERSION)
    payload.setdefault("sections", {})
    payload.setdefault("seed_snapshot_id", "unknown")
    return payload


def _compute_manifest_snapshot_id(sections: dict[str, Any]) -> str:
    components = [MANIFEST_SCHEMA_VERSION]
    for name in sorted(sections):
        section = sections[name] or {}
        components.append(str(name))
        components.append(str(section.get("seed_snapshot_id", "unknown")))
        components.append(str(section.get("seed_schema_version", "unknown")))
        for identity in section.get("seed_identities", []):
            components.append(str(identity))
    return build_seed_snapshot_id(components)


def update_manifest_section(
    *,
    section_name: str,
    section_payload: dict[str, Any],
    path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = load_seed_manifest(path)
    sections = dict(manifest.get("sections") or {})
    sections[section_name] = section_payload
    manifest["sections"] = sections
    manifest["seed_snapshot_id"] = _compute_manifest_snapshot_id(sections)
    manifest["generated_at"] = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
