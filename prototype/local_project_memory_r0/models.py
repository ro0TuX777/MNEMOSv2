from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import normalize_relative_path
from .errors import ErrorCode, ProjectMemoryError


def _unique_paths(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalize_relative_path(item) for item in values))


@dataclass(frozen=True)
class ScopeSpec:
    roots: tuple[str, ...]
    files: tuple[str, ...]
    excludes: tuple[str, ...]
    max_files: int = 500
    max_total_bytes: int = 10_485_760
    max_file_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _unique_paths(self.roots))
        object.__setattr__(self, "files", _unique_paths(self.files))
        object.__setattr__(self, "excludes", _unique_paths(self.excludes))
        if not self.roots and not self.files:
            raise ProjectMemoryError(ErrorCode.SCOPE_REQUIRED, "explicit scope is required")
        if min(self.max_files, self.max_total_bytes, self.max_file_bytes) <= 0:
            raise ProjectMemoryError(
                ErrorCode.RESOURCE_LIMIT_EXCEEDED,
                "resource ceilings must be positive",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": list(self.roots),
            "files": list(self.files),
            "excludes": list(self.excludes),
            "max_files": self.max_files,
            "max_total_bytes": self.max_total_bytes,
            "max_file_bytes": self.max_file_bytes,
        }


@dataclass(frozen=True)
class SourceSpan:
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ProjectMemoryError(
                ErrorCode.PACKET_INTEGRITY_INVALID,
                "source spans must be one-based and inclusive",
            )

    def to_dict(self) -> dict[str, int]:
        return {"start_line": self.start_line, "end_line": self.end_line}


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    language: str
    byte_size: int
    file_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "byte_size": self.byte_size,
            "file_hash": self.file_hash,
        }


@dataclass(frozen=True)
class SnapshotManifest:
    schema_version: str
    repo_id: str
    snapshot_id: str
    branch: str
    commit_hash: str
    working_tree_state: str
    scope: ScopeSpec
    files: tuple[SnapshotFile, ...]
    dirty_paths: tuple[str, ...] = ()
    exclusions: tuple[dict[str, Any], ...] = ()
    excluded_counts: dict[str, int] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "snapshot_id": self.snapshot_id,
            "branch": self.branch,
            "commit_hash": self.commit_hash,
            "working_tree_state": self.working_tree_state,
            "scope": self.scope.to_dict(),
            "files": [item.to_dict() for item in self.files],
            "dirty_paths": list(self.dirty_paths),
            "exclusions": list(self.exclusions),
            "excluded_counts": dict(self.excluded_counts),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ProjectArtifact:
    artifact_id: str
    repo_id: str
    snapshot_id: str
    file_path: str
    file_hash: str
    language: str
    artifact_type: str
    qualified_name: str
    span: SourceSpan
    content: str
    content_hash: str
    source_uri: str
    parser: str
    parser_version: str
    parse_status: str = "parsed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "repo_id": self.repo_id,
            "snapshot_id": self.snapshot_id,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "language": self.language,
            "artifact_type": self.artifact_type,
            "qualified_name": self.qualified_name,
            "provenance_span": self.span.to_dict(),
            "content": self.content,
            "content_hash": self.content_hash,
            "source_uri": self.source_uri,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "parse_status": self.parse_status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProjectPacket:
    packet_schema_version: str
    snapshot: SnapshotManifest
    artifacts: tuple[ProjectArtifact, ...]
    usable: bool
    failures: tuple[dict[str, Any], ...]
    authority_boundary: str
    approval_checkpoints: tuple[str, ...]
    created_at: str
    packet_sha256: str = ""


@dataclass(frozen=True)
class SearchHit:
    artifact: ProjectArtifact
    score: int
    score_components: dict[str, int]
    match_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SnapshotVerification:
    fresh: bool
    reason_code: str | None
    mismatches: tuple[dict[str, Any], ...] = ()
