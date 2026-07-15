from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

from .errors import ErrorCode, ProjectMemoryError


HASH_PREFIX = "sha256:"
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return HASH_PREFIX + hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", str(value))


def normalize_relative_path(value: str) -> str:
    normalized = normalize_text(value).replace("\\", "/").strip()
    if (
        not normalized
        or normalized.startswith("/")
        or _WINDOWS_DRIVE.match(normalized)
    ):
        raise ProjectMemoryError(
            ErrorCode.SCOPE_OUTSIDE_PROJECT,
            "path must be repository-relative",
        )
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ProjectMemoryError(
            ErrorCode.SCOPE_OUTSIDE_PROJECT,
            "path traversal or empty segments are not allowed",
        )
    return path.as_posix()


def source_uri(repo_id: str, path: str, snapshot_id: str) -> str:
    normalized_path = normalize_relative_path(path)
    return (
        f"repo://{quote(normalize_text(repo_id), safe='-._~')}/"
        f"{quote(normalized_path, safe='/-._~')}?snapshot={snapshot_id}"
    )


def _length_prefixed(parts: Iterable[str]) -> bytes:
    output = bytearray()
    for part in parts:
        encoded = normalize_text(part).encode("utf-8")
        output.extend(str(len(encoded)).encode("ascii"))
        output.extend(b":")
        output.extend(encoded)
    return bytes(output)


def artifact_id(
    repo_id: str,
    snapshot_id: str,
    path: str,
    artifact_type: str,
    qualified_name: str,
    span: Any,
    content_hash: str,
) -> str:
    return sha256_bytes(
        _length_prefixed(
            (
                repo_id,
                snapshot_id,
                normalize_relative_path(path),
                artifact_type,
                qualified_name,
                str(span.start_line),
                str(span.end_line),
                content_hash,
            )
        )
    )
