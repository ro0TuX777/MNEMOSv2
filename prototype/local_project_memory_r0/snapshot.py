from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .canonical import canonical_json_bytes, normalize_relative_path, sha256_bytes
from .errors import ErrorCode, ProjectMemoryError
from .models import ScopeSpec, SnapshotFile, SnapshotManifest, SnapshotVerification


SNAPSHOT_SCHEMA_VERSION = "local_project_snapshot_r0"
EXCLUSION_POLICY_VERSION = "local_project_exclusions_r0"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LANGUAGES = {".py": "python", ".md": "markdown"}
_HARD_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "vendor",
    "vendored",
    "third_party",
}
_SECRET_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "secrets",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


class GitReader:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.project_root,
                check=check,
                capture_output=True,
                shell=False,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ProjectMemoryError(
                ErrorCode.NOT_GIT_WORKTREE,
                "read-only Git identity command failed",
            ) from exc

    def text(self, *args: str, check: bool = True) -> str:
        result = self._run(list(args), check=check)
        return result.stdout.decode("utf-8", errors="strict").strip()

    def nul_paths(self, *args: str) -> tuple[str, ...]:
        output = self._run(list(args)).stdout
        return tuple(
            normalize_relative_path(item.decode("utf-8", errors="strict"))
            for item in output.split(b"\0")
            if item
        )

    def root(self) -> Path:
        return Path(self.text("rev-parse", "--show-toplevel")).resolve()

    def commit_hash(self) -> str:
        return self.text("rev-parse", "HEAD")

    def branch(self) -> str:
        result = self._run(
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
            check=False,
        )
        if result.returncode:
            return "DETACHED_HEAD"
        return result.stdout.decode("utf-8", errors="strict").strip()

    def tracked(self, scope: ScopeSpec) -> tuple[str, ...]:
        pathspecs = [*scope.roots, *scope.files]
        return self.nul_paths("ls-files", "--cached", "-z", "--", *pathspecs)

    def status(self) -> tuple[dict[str, str], int]:
        raw = self._run(
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
        ).stdout
        entries = raw.split(b"\0")
        states: dict[str, str] = {}
        untracked_count = 0
        index = 0
        while index < len(entries):
            entry = entries[index]
            index += 1
            if not entry:
                continue
            status = entry[:2].decode("ascii", errors="replace")
            path_bytes = entry[3:]
            if status == "??":
                untracked_count += 1
                continue
            if status[0] in {"R", "C"} and index < len(entries):
                path_bytes = entries[index]
                index += 1
            path = normalize_relative_path(path_bytes.decode("utf-8", errors="strict"))
            states[path] = status
        return states, untracked_count


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_scope_paths(root: Path, scope: ScopeSpec) -> None:
    for relative in (*scope.roots, *scope.files, *scope.excludes):
        resolved = (root / relative).resolve(strict=False)
        if not _inside(root, resolved):
            raise ProjectMemoryError(
                ErrorCode.SCOPE_OUTSIDE_PROJECT,
                "scope path resolves outside the project root",
            )
    for relative in scope.roots:
        if not (root / relative).is_dir():
            raise ProjectMemoryError(
                ErrorCode.SCOPE_FILE_NOT_ADMITTED,
                "explicit scope root does not exist",
            )


def _matches(relative: str, prefixes: Iterable[str]) -> bool:
    return any(relative == item or relative.startswith(item + "/") for item in prefixes)


def _secret_path(relative: str) -> bool:
    path = Path(relative)
    lowered_parts = [part.lower() for part in path.parts]
    if any(part in _HARD_DIRS for part in lowered_parts):
        return True
    name = path.name.lower()
    return (
        name in _SECRET_NAMES
        or name.startswith(".env")
        or path.suffix.lower() in _SECRET_SUFFIXES
        or "credential" in name
        or "secret" in name
    )


def _redacted_path(relative: str) -> str:
    return "redacted:" + sha256_bytes(relative.encode("utf-8"))[7:19]


def _snapshot_preimage(
    *,
    repo_id: str,
    branch: str,
    commit_hash: str,
    working_tree_state: str,
    scope: ScopeSpec,
    files: tuple[SnapshotFile, ...],
    dirty_paths: tuple[str, ...],
) -> dict:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "exclusion_policy_version": EXCLUSION_POLICY_VERSION,
        "repo_id": repo_id,
        "branch": branch,
        "commit_hash": commit_hash,
        "working_tree_state": working_tree_state,
        "scope": scope.to_dict(),
        "files": [item.to_dict() for item in files],
        "dirty_paths": list(dirty_paths),
    }


def build_snapshot(
    project_root: Path,
    repo_id: str,
    scope: ScopeSpec,
) -> SnapshotManifest:
    if not str(project_root).strip():
        raise ProjectMemoryError(ErrorCode.PROJECT_ROOT_REQUIRED, "project root is required")
    if not _REPO_ID.fullmatch(str(repo_id).strip()):
        raise ProjectMemoryError(ErrorCode.REPO_ID_REQUIRED, "repo ID is invalid")
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ProjectMemoryError(ErrorCode.PROJECT_ROOT_REQUIRED, "project root is not a directory")
    git = GitReader(root)
    if git.root() != root:
        raise ProjectMemoryError(
            ErrorCode.NOT_GIT_WORKTREE,
            "project root must be the Git worktree root",
        )
    _validate_scope_paths(root, scope)

    tracked = tuple(sorted(git.tracked(scope)))
    tracked_set = set(tracked)
    for explicit in scope.files:
        if explicit not in tracked_set:
            raise ProjectMemoryError(
                ErrorCode.SCOPE_FILE_NOT_ADMITTED,
                "explicit scope file is not tracked",
            )
        if _secret_path(explicit):
            raise ProjectMemoryError(
                ErrorCode.SECRET_PATH_REJECTED,
                "explicit scope file is denied by the secret policy",
            )
        if Path(explicit).suffix.lower() not in _LANGUAGES:
            raise ProjectMemoryError(
                ErrorCode.SCOPE_FILE_NOT_ADMITTED,
                "explicit scope file has an unsupported language",
            )

    status_by_path, untracked_count = git.status()
    files: list[SnapshotFile] = []
    exclusions: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    total_bytes = 0

    for relative in tracked:
        if _matches(relative, scope.excludes):
            exclusions.append({"path": relative, "reason": "operator_exclusion"})
            counts["operator_exclusion"] += 1
            continue
        if _secret_path(relative):
            exclusions.append({"path": _redacted_path(relative), "reason": "secret_path"})
            counts["secret_path"] += 1
            continue
        suffix = Path(relative).suffix.lower()
        if suffix not in _LANGUAGES:
            exclusions.append({"path": relative, "reason": "unsupported_language"})
            counts["unsupported_language"] += 1
            continue
        path = root / relative
        if not path.exists():
            if relative in scope.files:
                raise ProjectMemoryError(
                    ErrorCode.SCOPE_FILE_NOT_ADMITTED,
                    "explicit tracked file is deleted",
                )
            exclusions.append({"path": relative, "reason": "tracked_deleted"})
            counts["tracked_deleted"] += 1
            continue
        resolved = path.resolve()
        if not _inside(root, resolved):
            raise ProjectMemoryError(
                ErrorCode.SCOPE_OUTSIDE_PROJECT,
                "tracked path resolves outside the project root",
            )
        if not path.is_file():
            exclusions.append({"path": relative, "reason": "not_regular_file"})
            counts["not_regular_file"] += 1
            continue
        raw = path.read_bytes()
        if len(raw) > scope.max_file_bytes:
            raise ProjectMemoryError(
                ErrorCode.RESOURCE_LIMIT_EXCEEDED,
                "an admitted file exceeds max_file_bytes",
            )
        total_bytes += len(raw)
        if total_bytes > scope.max_total_bytes:
            raise ProjectMemoryError(
                ErrorCode.RESOURCE_LIMIT_EXCEEDED,
                "admitted files exceed max_total_bytes",
            )
        files.append(
            SnapshotFile(
                path=relative,
                language=_LANGUAGES[suffix],
                byte_size=len(raw),
                file_hash=sha256_bytes(raw),
            )
        )
        if len(files) > scope.max_files:
            raise ProjectMemoryError(
                ErrorCode.RESOURCE_LIMIT_EXCEEDED,
                "admitted files exceed max_files",
            )

    if not files:
        raise ProjectMemoryError(
            ErrorCode.EMPTY_ADMITTED_SCOPE,
            "explicit scope admitted no Python or Markdown files",
        )

    snapshot_files = tuple(files)
    admitted_paths = {item.path for item in snapshot_files}
    dirty_paths = tuple(sorted(path for path in status_by_path if path in admitted_paths))
    working_tree_state = "dirty" if dirty_paths else "clean"
    branch = git.branch()
    commit_hash = git.commit_hash()
    preimage = _snapshot_preimage(
        repo_id=repo_id,
        branch=branch,
        commit_hash=commit_hash,
        working_tree_state=working_tree_state,
        scope=scope,
        files=snapshot_files,
        dirty_paths=dirty_paths,
    )
    counts["untracked_excluded"] = untracked_count
    return SnapshotManifest(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        repo_id=repo_id,
        snapshot_id=sha256_bytes(canonical_json_bytes(preimage)),
        branch=branch,
        commit_hash=commit_hash,
        working_tree_state=working_tree_state,
        scope=scope,
        files=snapshot_files,
        dirty_paths=dirty_paths,
        exclusions=tuple(exclusions),
        excluded_counts=dict(sorted(counts.items())),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def verify_snapshot(project_root: Path, manifest: SnapshotManifest) -> SnapshotVerification:
    try:
        current = build_snapshot(project_root, manifest.repo_id, manifest.scope)
    except ProjectMemoryError as exc:
        return SnapshotVerification(
            fresh=False,
            reason_code=ErrorCode.SNAPSHOT_MISMATCH.value,
            mismatches=({"reason": exc.code.value},),
        )
    expected = {item.path: item.file_hash for item in manifest.files}
    observed = {item.path: item.file_hash for item in current.files}
    mismatches = []
    for path in sorted(set(expected) | set(observed)):
        if expected.get(path) != observed.get(path):
            mismatches.append(
                {
                    "path": path,
                    "expected_hash": expected.get(path),
                    "observed_hash": observed.get(path),
                }
            )
    if current.snapshot_id != manifest.snapshot_id and not mismatches:
        mismatches.append(
            {
                "reason": "snapshot_identity_changed",
                "expected_snapshot_id": manifest.snapshot_id,
                "observed_snapshot_id": current.snapshot_id,
            }
        )
    return SnapshotVerification(
        fresh=not mismatches,
        reason_code=None if not mismatches else ErrorCode.SNAPSHOT_MISMATCH.value,
        mismatches=tuple(mismatches),
    )
