"""Upload manifest for the research intake UI.

Tracks what has been stored and indexed so that a later upload can be
recognised as an exact duplicate, a new document, or a new *version* of an
existing document. This is the foundation for version supersession: because
each record carries the engram ids that version put in the index, retiring an
old version is a lookup-and-delete rather than a fragile id reconstruction.

Design: docs/superpowers/specs/2026-07-22-research-upload-versioning-design.md

Identity rule: content sha256 decides identity; a project-scoped normalized
filename is the *document identity key* used to relate versions; size and date
are never identity — size only pre-filters and date only orders/informs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = ".manifest.json"

# Actions a stored upload can resolve to.
REUSED_EXACT = "reused_exact"
NEW_DOCUMENT = "new_document"
NEW_VERSION = "new_version"


@dataclass
class StoreDecision:
    """Outcome of storing one uploaded file."""

    filename: str  # original (secure) upload filename
    stored_path: Path  # where the bytes now live
    sha256: str
    size: int
    identity_key: str
    action: str  # REUSED_EXACT | NEW_DOCUMENT | NEW_VERSION
    client_mtime: float | None = None
    superseded: dict[str, Any] | None = None  # prior record to retire post-index


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_identity_key(project: str, filename: str) -> str:
    """Project-scoped, case-insensitive document identity.

    Scoping by project keeps two unrelated ``report.pdf`` files in different
    projects from ever colliding.
    """
    proj = (project or "").strip().lower()
    name = Path(filename or "").name.strip().lower()
    return f"{proj}::{name}"


def load_manifest(upload_dir: Path) -> dict[str, Any]:
    path = Path(upload_dir) / MANIFEST_FILENAME
    if not path.exists():
        return {"version": 1, "records": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "records": []}
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        return {"version": 1, "records": []}
    return data


def save_manifest(upload_dir: Path, manifest: dict[str, Any]) -> None:
    path = Path(upload_dir) / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def active_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in manifest.get("records", []) if not r.get("superseded_by")]


def find_active_by_identity(manifest: dict[str, Any], identity_key: str) -> dict[str, Any] | None:
    for record in active_records(manifest):
        if record.get("identity_key") == identity_key:
            return record
    return None


def find_active_by_sha(manifest: dict[str, Any], sha256: str) -> dict[str, Any] | None:
    for record in active_records(manifest):
        if record.get("sha256") == sha256:
            return record
    return None


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_copy_on_disk(upload_dir: Path, data: bytes, sha256: str) -> Path | None:
    """Find an already-stored file with identical content (size-filtered).

    Covers files present before the manifest existed, so exact-dedup keeps
    working during the transition without a backfill.
    """
    size = len(data)
    for candidate in sorted(upload_dir.iterdir()):
        if candidate.name == MANIFEST_FILENAME or not candidate.is_file():
            continue
        if candidate.stat().st_size != size:
            continue
        if _file_digest(candidate) == sha256:
            return candidate
    return None


def plan_and_store(
    upload_dir: Path,
    manifest: dict[str, Any],
    *,
    filename: str,
    data: bytes,
    project: str,
    client_mtime: float | None = None,
) -> StoreDecision:
    """Store one upload, deciding exact-reuse / new-document / new-version.

    The bytes are written here (except for exact reuse), but the manifest is
    NOT mutated — call :func:`record_decision` after indexing, once the engram
    ids are known.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    sha256 = _digest_bytes(data)
    size = len(data)
    identity_key = normalize_identity_key(project, filename)

    # 1. Exact duplicate — identical bytes already stored (manifest or disk).
    record = find_active_by_sha(manifest, sha256)
    if record is not None and (upload_dir / record["stored_path"]).exists():
        stored_path = upload_dir / record["stored_path"]
    else:
        stored_path = _existing_copy_on_disk(upload_dir, data, sha256)
    if stored_path is not None:
        return StoreDecision(
            filename=filename,
            stored_path=stored_path,
            sha256=sha256,
            size=size,
            identity_key=identity_key,
            action=REUSED_EXACT,
            client_mtime=client_mtime,
        )

    # 2. New version — same document identity, different bytes. Overwrite the
    #    canonical stored path so the identity key stays stable across versions.
    prior = find_active_by_identity(manifest, identity_key)
    if prior is not None:
        target = upload_dir / prior["stored_path"]
        target.write_bytes(data)
        return StoreDecision(
            filename=filename,
            stored_path=target,
            sha256=sha256,
            size=size,
            identity_key=identity_key,
            action=NEW_VERSION,
            client_mtime=client_mtime,
            superseded=prior,
        )

    # 3. New document. Number only against an unrelated file already holding
    #    the name (distinct content that is not a tracked version).
    target = upload_dir / filename
    counter = 1
    while target.exists():
        target = upload_dir / f"{Path(filename).stem}-{counter}{Path(filename).suffix}"
        counter += 1
    target.write_bytes(data)
    return StoreDecision(
        filename=filename,
        stored_path=target,
        sha256=sha256,
        size=size,
        identity_key=identity_key,
        action=NEW_DOCUMENT,
        client_mtime=client_mtime,
    )


def record_decision(
    manifest: dict[str, Any],
    decision: StoreDecision,
    *,
    engram_ids: list[str],
    uploaded_at: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Fold a stored upload into the manifest once its engram ids are known.

    Returns the resulting record and the list of stale engram ids that should
    be deleted from the index (non-empty only for a superseding new version).
    """
    uploaded_at = uploaded_at or utc_now()
    engram_ids = list(engram_ids or [])
    stale: list[str] = []

    if decision.action == REUSED_EXACT:
        existing = find_active_by_sha(manifest, decision.sha256)
        if existing is not None:
            existing["last_seen_at"] = uploaded_at
            if engram_ids and not existing.get("engram_ids"):
                existing["engram_ids"] = engram_ids
            return existing, stale
        # No manifest record yet (legacy on-disk file) — fall through to create.

    prior = find_active_by_identity(manifest, decision.identity_key)
    version = 1
    if prior is not None and prior.get("sha256") != decision.sha256:
        prior["superseded_by"] = decision.sha256
        prior["superseded_at"] = uploaded_at
        version = int(prior.get("version", 1)) + 1
        old_ids = set(prior.get("engram_ids") or [])
        stale = [eid for eid in old_ids if eid not in set(engram_ids)]

    record = {
        "sha256": decision.sha256,
        "stored_path": Path(decision.stored_path).name,
        "identity_key": decision.identity_key,
        "size": decision.size,
        "client_mtime": decision.client_mtime,
        "engram_ids": engram_ids,
        "version": version,
        "uploaded_at": uploaded_at,
        "last_seen_at": uploaded_at,
        "superseded_by": None,
    }
    manifest.setdefault("records", []).append(record)
    return record, stale
