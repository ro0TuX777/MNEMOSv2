from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    PROJECT_ROOT_REQUIRED = "PROJECT_ROOT_REQUIRED"
    NOT_GIT_WORKTREE = "NOT_GIT_WORKTREE"
    REPO_ID_REQUIRED = "REPO_ID_REQUIRED"
    SCOPE_REQUIRED = "SCOPE_REQUIRED"
    SCOPE_OUTSIDE_PROJECT = "SCOPE_OUTSIDE_PROJECT"
    SCOPE_FILE_NOT_ADMITTED = "SCOPE_FILE_NOT_ADMITTED"
    EMPTY_ADMITTED_SCOPE = "EMPTY_ADMITTED_SCOPE"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    SECRET_PATH_REJECTED = "SECRET_PATH_REJECTED"
    DIRTY_STATE_UNRESOLVED = "DIRTY_STATE_UNRESOLVED"
    STRUCTURED_PARSE_INCOMPLETE = "STRUCTURED_PARSE_INCOMPLETE"
    OUTPUT_INSIDE_PROJECT = "OUTPUT_INSIDE_PROJECT"
    OUTPUT_ALREADY_EXISTS = "OUTPUT_ALREADY_EXISTS"
    PACKET_INTEGRITY_INVALID = "PACKET_INTEGRITY_INVALID"
    REPO_ID_MISMATCH = "REPO_ID_MISMATCH"
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"
    CROSS_SCOPE_EVIDENCE = "CROSS_SCOPE_EVIDENCE"


class ProjectMemoryError(RuntimeError):
    """Fail-closed error with a stable, non-sensitive reason code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.details = dict(details or {})
