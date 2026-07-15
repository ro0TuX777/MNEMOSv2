from __future__ import annotations

from dataclasses import replace

import pytest

from prototype.local_project_memory_r0.canonical import (
    artifact_id,
    canonical_json_bytes,
    normalize_relative_path,
    sha256_bytes,
    source_uri,
)
from prototype.local_project_memory_r0.errors import ErrorCode, ProjectMemoryError
from prototype.local_project_memory_r0.models import ScopeSpec, SourceSpan


def test_canonical_json_and_hash_are_stable() -> None:
    left = canonical_json_bytes({"b": 2, "a": [1]})
    right = canonical_json_bytes({"a": [1], "b": 2})
    assert left == right == b'{"a":[1],"b":2}'
    assert sha256_bytes(left).startswith("sha256:")


@pytest.mark.parametrize(
    "value",
    ["../secret.py", "/tmp/a.py", "C:/a.py", "a/../../b.py", ""],
)
def test_relative_path_escape_fails_closed(value: str) -> None:
    with pytest.raises(ProjectMemoryError) as exc:
        normalize_relative_path(value)
    assert exc.value.code is ErrorCode.SCOPE_OUTSIDE_PROJECT


def test_relative_path_normalizes_windows_separators() -> None:
    assert normalize_relative_path(r"mnemos\retrieval\router.py") == "mnemos/retrieval/router.py"


def test_artifact_identity_binds_snapshot_span_and_content() -> None:
    snapshot = "sha256:" + "a" * 64
    span = SourceSpan(start_line=10, end_line=12)
    content_hash = "sha256:" + "b" * 64
    first = artifact_id(
        "mnemos",
        snapshot,
        "mnemos/config.py",
        "python_symbol",
        "Settings",
        span,
        content_hash,
    )
    second = artifact_id(
        "mnemos",
        snapshot,
        "mnemos/config.py",
        "python_symbol",
        "Settings",
        replace(span, end_line=13),
        content_hash,
    )
    assert first != second
    assert first.startswith("sha256:")
    assert source_uri("mnemos", "mnemos/config.py", snapshot) == (
        f"repo://mnemos/mnemos/config.py?snapshot={snapshot}"
    )


def test_empty_scope_contract_is_rejected() -> None:
    with pytest.raises(ProjectMemoryError) as exc:
        ScopeSpec(roots=(), files=(), excludes=())
    assert exc.value.code is ErrorCode.SCOPE_REQUIRED


def test_scope_normalizes_and_deduplicates_paths() -> None:
    scope = ScopeSpec(
        roots=(r"mnemos\retrieval", "mnemos/retrieval"),
        files=("README.md", "README.md"),
        excludes=(r"mnemos\retrieval\generated",),
    )
    assert scope.roots == ("mnemos/retrieval",)
    assert scope.files == ("README.md",)
    assert scope.excludes == ("mnemos/retrieval/generated",)


def test_source_span_requires_one_based_inclusive_lines() -> None:
    with pytest.raises(ProjectMemoryError):
        SourceSpan(start_line=0, end_line=1)
    with pytest.raises(ProjectMemoryError):
        SourceSpan(start_line=4, end_line=3)
