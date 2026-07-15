from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import artifact_id, canonical_json_bytes, sha256_bytes, source_uri
from .errors import ErrorCode, ProjectMemoryError
from .markdown_extractor import extract_markdown
from .models import (
    ProjectArtifact,
    ProjectPacket,
    ScopeSpec,
    SnapshotFile,
    SnapshotManifest,
    SourceSpan,
)
from .python_extractor import extract_python
from .snapshot import build_snapshot


PACKET_SCHEMA_VERSION = "local_project_memory_packet_r0"
BEGIN_SENTINEL = "<!-- MNEMOS_PROJECT_PACKET_JSON_BEGIN:v1 -->"
END_SENTINEL = "<!-- MNEMOS_PROJECT_PACKET_JSON_END:v1 -->"
AUTHORITY_BOUNDARY = (
    "MNEMOS project memory is read-only source-backed evidence, not authority. "
    "Human approval is required before lint execution and separately before code mutation. "
    "Any admitted-file mutation makes this packet stale and requires a rebuild."
)
APPROVAL_CHECKPOINTS = (
    "Operator verified repo ID, snapshot ID, admitted scope, and exclusions.",
    "Operator approved the exact read-only command before lint execution; no autofix, formatting, write, or generated-file mode is allowed.",
    "Operator separately approved the proposed changes before code mutation.",
    "After mutation, rebuild the packet before trusting further project-memory retrieval.",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _payload(packet: ProjectPacket, *, include_hash: bool) -> dict[str, Any]:
    value = {
        "packet_schema_version": packet.packet_schema_version,
        "snapshot": packet.snapshot.to_dict(),
        "artifacts": [item.to_dict() for item in packet.artifacts],
        "usable": packet.usable,
        "failures": list(packet.failures),
        "authority_boundary": packet.authority_boundary,
        "approval_checkpoints": list(packet.approval_checkpoints),
        "created_at": packet.created_at,
    }
    if include_hash:
        value["packet_sha256"] = packet.packet_sha256
    return value


def _packet_hash(packet: ProjectPacket) -> str:
    return sha256_bytes(canonical_json_bytes(_payload(packet, include_hash=False)))


def _validate_artifact(artifact: ProjectArtifact, manifest: SnapshotManifest) -> None:
    files = {item.path: item for item in manifest.files}
    snapshot_file = files.get(artifact.file_path)
    if snapshot_file is None:
        raise ProjectMemoryError(
            ErrorCode.CROSS_SCOPE_EVIDENCE,
            "artifact path is outside the snapshot manifest",
        )
    if (
        artifact.repo_id != manifest.repo_id
        or artifact.snapshot_id != manifest.snapshot_id
        or artifact.file_hash != snapshot_file.file_hash
        or artifact.language != snapshot_file.language
    ):
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "artifact lineage does not match the snapshot",
        )
    if sha256_bytes(artifact.content.encode("utf-8")) != artifact.content_hash:
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "artifact content hash is invalid",
        )
    expected_id = artifact_id(
        artifact.repo_id,
        artifact.snapshot_id,
        artifact.file_path,
        artifact.artifact_type,
        artifact.qualified_name,
        artifact.span,
        artifact.content_hash,
    )
    if expected_id != artifact.artifact_id:
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "artifact identity is invalid",
        )
    if artifact.source_uri != source_uri(
        artifact.repo_id,
        artifact.file_path,
        artifact.snapshot_id,
    ):
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "artifact source URI is invalid",
        )


def validate_packet(packet: ProjectPacket) -> None:
    if packet.packet_schema_version != PACKET_SCHEMA_VERSION:
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "unsupported packet schema",
        )
    if packet.usable and packet.failures:
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "usable packet cannot contain extraction failures",
        )
    for artifact in packet.artifacts:
        _validate_artifact(artifact, packet.snapshot)
    if not packet.packet_sha256 or packet.packet_sha256 != _packet_hash(packet):
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "packet digest is invalid",
        )


def build_packet(
    project_root: Path,
    repo_id: str,
    scope: ScopeSpec,
) -> ProjectPacket:
    manifest = build_snapshot(project_root, repo_id, scope)
    artifacts: list[ProjectArtifact] = []
    failures: list[dict[str, Any]] = []
    extractors = {"python": extract_python, "markdown": extract_markdown}
    for snapshot_file in manifest.files:
        try:
            artifacts.extend(extractors[snapshot_file.language](project_root, manifest, snapshot_file))
        except ProjectMemoryError as exc:
            if exc.code is not ErrorCode.STRUCTURED_PARSE_INCOMPLETE:
                raise
            failures.append(
                {
                    "file_path": snapshot_file.path,
                    "reason_code": exc.code.value,
                    "category": exc.details.get("category", "structured_parse_failure"),
                    "line": exc.details.get("line"),
                }
            )
    packet = ProjectPacket(
        packet_schema_version=PACKET_SCHEMA_VERSION,
        snapshot=manifest,
        artifacts=tuple(
            sorted(
                artifacts,
                key=lambda item: (
                    item.file_path,
                    item.span.start_line,
                    item.span.end_line,
                    item.artifact_id,
                ),
            )
        ),
        usable=not failures,
        failures=tuple(failures),
        authority_boundary=AUTHORITY_BOUNDARY,
        approval_checkpoints=APPROVAL_CHECKPOINTS,
        created_at=_utc_now(),
    )
    packet = replace(packet, packet_sha256=_packet_hash(packet))
    validate_packet(packet)
    return packet


def _human_report(packet: ProjectPacket) -> str:
    lines = [
        "# MNEMOS Local Project Memory Packet R0",
        "",
        f"- Repository ID: `{packet.snapshot.repo_id}`",
        f"- Snapshot ID: `{packet.snapshot.snapshot_id}`",
        f"- Packet SHA-256: `{packet.packet_sha256}`",
        f"- Branch: `{packet.snapshot.branch}`",
        f"- Base commit: `{packet.snapshot.commit_hash}`",
        f"- Working tree: `{packet.snapshot.working_tree_state}`",
        f"- Status: `{'COMPLETE' if packet.usable else 'INCOMPLETE_ABSTAINED'}`",
        f"- Admitted files: `{len(packet.snapshot.files)}`",
        f"- Source-backed artifacts: `{len(packet.artifacts)}`",
        "- Storage mode: `packet_only`",
        "- MNEMOS collection: `null`",
        "",
        "## Boundaries",
        "",
        f"- Scope roots: `{', '.join(packet.snapshot.scope.roots) or 'none'}`",
        f"- Scope files: `{', '.join(packet.snapshot.scope.files) or 'none'}`",
        f"- Operator exclusions: `{', '.join(packet.snapshot.scope.excludes) or 'none'}`",
        f"- Authority: {packet.authority_boundary}",
        "",
        "## Admitted files",
        "",
    ]
    for item in packet.snapshot.files:
        lines.append(f"- `{item.path}` — {item.language}, {item.byte_size} bytes, `{item.file_hash}`")
    lines.extend(["", "## Exclusions", ""])
    if packet.snapshot.exclusions:
        for item in packet.snapshot.exclusions:
            lines.append(f"- `{item['path']}` — `{item['reason']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Extraction failures", ""])
    if packet.failures:
        for item in packet.failures:
            lines.append(f"- `{item['file_path']}` — `{item['reason_code']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Human approval checkpoints", ""])
    for checkpoint in packet.approval_checkpoints:
        lines.append(f"- [ ] {checkpoint}")
    return "\n".join(lines) + "\n"


def write_packet(path: Path, packet: ProjectPacket) -> None:
    validate_packet(packet)
    output = Path(path)
    if output.exists():
        raise ProjectMemoryError(
            ErrorCode.OUTPUT_ALREADY_EXISTS,
            "output already exists",
        )
    if not output.parent.is_dir():
        raise ProjectMemoryError(
            ErrorCode.PROJECT_ROOT_REQUIRED,
            "output parent directory does not exist",
        )
    json_payload = json.dumps(
        _payload(packet, include_hash=True),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    body = (
        f"{BEGIN_SENTINEL}\n"
        "```json\n"
        f"{json_payload}\n"
        "```\n"
        f"{END_SENTINEL}\n\n"
        f"{_human_report(packet)}"
    )
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
    except FileExistsError as exc:
        raise ProjectMemoryError(
            ErrorCode.OUTPUT_ALREADY_EXISTS,
            "output already exists",
        ) from exc


def _scope_from_dict(value: dict[str, Any]) -> ScopeSpec:
    return ScopeSpec(
        roots=tuple(value["roots"]),
        files=tuple(value["files"]),
        excludes=tuple(value["excludes"]),
        max_files=int(value["max_files"]),
        max_total_bytes=int(value["max_total_bytes"]),
        max_file_bytes=int(value["max_file_bytes"]),
    )


def _manifest_from_dict(value: dict[str, Any]) -> SnapshotManifest:
    return SnapshotManifest(
        schema_version=value["schema_version"],
        repo_id=value["repo_id"],
        snapshot_id=value["snapshot_id"],
        branch=value["branch"],
        commit_hash=value["commit_hash"],
        working_tree_state=value["working_tree_state"],
        scope=_scope_from_dict(value["scope"]),
        files=tuple(SnapshotFile(**item) for item in value["files"]),
        dirty_paths=tuple(value.get("dirty_paths", ())),
        exclusions=tuple(value.get("exclusions", ())),
        excluded_counts=dict(value.get("excluded_counts", {})),
        created_at=value.get("created_at", ""),
    )


def _artifact_from_dict(value: dict[str, Any]) -> ProjectArtifact:
    span = value["provenance_span"]
    return ProjectArtifact(
        artifact_id=value["artifact_id"],
        repo_id=value["repo_id"],
        snapshot_id=value["snapshot_id"],
        file_path=value["file_path"],
        file_hash=value["file_hash"],
        language=value["language"],
        artifact_type=value["artifact_type"],
        qualified_name=value["qualified_name"],
        span=SourceSpan(int(span["start_line"]), int(span["end_line"])),
        content=value["content"],
        content_hash=value["content_hash"],
        source_uri=value["source_uri"],
        parser=value["parser"],
        parser_version=value["parser_version"],
        parse_status=value.get("parse_status", "parsed"),
        metadata=dict(value.get("metadata", {})),
    )


def load_packet(path: Path) -> ProjectPacket:
    try:
        text = Path(path).read_text(encoding="utf-8")
        start = text.index(BEGIN_SENTINEL) + len(BEGIN_SENTINEL)
        end = text.index(END_SENTINEL, start)
        fenced = text[start:end].strip()
        if not fenced.startswith("```json\n") or not fenced.endswith("```"):
            raise ValueError("invalid JSON fence")
        value = json.loads(fenced[len("```json\n") : -len("```")].strip())
        packet = ProjectPacket(
            packet_schema_version=value["packet_schema_version"],
            snapshot=_manifest_from_dict(value["snapshot"]),
            artifacts=tuple(_artifact_from_dict(item) for item in value["artifacts"]),
            usable=bool(value["usable"]),
            failures=tuple(value.get("failures", ())),
            authority_boundary=value["authority_boundary"],
            approval_checkpoints=tuple(value["approval_checkpoints"]),
            created_at=value["created_at"],
            packet_sha256=value["packet_sha256"],
        )
    except ProjectMemoryError:
        raise
    except Exception as exc:
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "packet container is invalid",
        ) from exc
    validate_packet(packet)
    return packet
