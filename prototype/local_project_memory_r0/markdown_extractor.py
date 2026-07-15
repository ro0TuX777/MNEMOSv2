from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .canonical import artifact_id, sha256_bytes, source_uri
from .errors import ErrorCode, ProjectMemoryError
from .models import ProjectArtifact, SnapshotFile, SnapshotManifest, SourceSpan


PARSER_NAME = "markdown_structural_r0"
PARSER_VERSION = "1"
_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT = re.compile(r"^\s*(=+|-+)\s*$")
_FIELD = re.compile(r"^\s*(status|date|decision date|supersedes|superseded by)\s*:\s*(.*?)\s*$", re.I)
_STATUS = {
    "proposed", "accepted", "deprecated", "superseded", "complete",
    "incomplete", "blocked", "pass", "fail",
}


def _slice(lines: list[str], span: SourceSpan) -> str:
    return "".join(lines[span.start_line - 1 : span.end_line])


def _explicit_fields(content: str) -> dict[str, Any]:
    values: dict[str, list[str]] = {}
    for line in content.splitlines():
        match = _FIELD.match(line)
        if match:
            values.setdefault(match.group(1).lower(), []).append(match.group(2).strip())
    raw_status = (values.get("status") or [None])[0]
    status = None
    if raw_status is not None:
        normalized = raw_status.strip().lower().replace(" ", "_")
        status = {
            "raw": raw_status,
            "normalized": normalized if normalized in _STATUS else "unknown_explicit",
        }
    return {
        "status": status,
        "decision_date": ((values.get("decision date") or values.get("date") or [None])[0]),
        "supersedes": values.get("supersedes", []),
        "superseded_by": values.get("superseded by", []),
    }


def _document_type(path: str, title: str) -> tuple[str, str]:
    lowered = f"{path} {title}".lower()
    name = Path(path).name.upper()
    if "/adr/" in "/" + path.lower() or re.search(r"\badr[-_ ]?\d+", lowered):
        return "markdown_adr", "path_or_title_adr"
    if "decision" in lowered:
        return "markdown_decision", "path_or_title_decision"
    if "handoff" in lowered:
        return "markdown_handoff", "path_or_title_handoff"
    if any(term in lowered for term in ("evaluation", "benchmark", "closeout", "results")):
        return "markdown_evaluation_closeout", "path_or_title_evaluation"
    if name in {"AGENTS.MD", "CLAUDE.MD"}:
        return "markdown_agent_instruction", "allowlisted_filename"
    return "markdown_document", "markdown_file"


def _make_artifact(
    *,
    manifest: SnapshotManifest,
    snapshot_file: SnapshotFile,
    artifact_type: str,
    qualified_name: str,
    span: SourceSpan,
    content: str,
    metadata: dict[str, Any],
) -> ProjectArtifact:
    content_hash = sha256_bytes(content.encode("utf-8"))
    identity = artifact_id(
        manifest.repo_id,
        manifest.snapshot_id,
        snapshot_file.path,
        artifact_type,
        qualified_name,
        span,
        content_hash,
    )
    return ProjectArtifact(
        artifact_id=identity,
        repo_id=manifest.repo_id,
        snapshot_id=manifest.snapshot_id,
        file_path=snapshot_file.path,
        file_hash=snapshot_file.file_hash,
        language="markdown",
        artifact_type=artifact_type,
        qualified_name=qualified_name,
        span=span,
        content=content,
        content_hash=content_hash,
        source_uri=source_uri(manifest.repo_id, snapshot_file.path, manifest.snapshot_id),
        parser=PARSER_NAME,
        parser_version=PARSER_VERSION,
        metadata=metadata,
    )


def _headings(lines: list[str]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        plain = line.rstrip("\r\n")
        atx = _ATX.match(plain)
        if atx:
            headings.append({"start": index + 1, "level": len(atx.group(1)), "title": atx.group(2).strip()})
            continue
        if index > 0:
            setext = _SETEXT.match(plain)
            previous = lines[index - 1].rstrip("\r\n").strip()
            if setext and previous:
                headings.append({
                    "start": index,
                    "level": 1 if setext.group(1).startswith("=") else 2,
                    "title": previous,
                    "underline": index + 1,
                })
    return headings


def extract_markdown(
    project_root: Path,
    manifest: SnapshotManifest,
    snapshot_file: SnapshotFile,
) -> tuple[ProjectArtifact, ...]:
    if snapshot_file.language != "markdown":
        raise ProjectMemoryError(ErrorCode.PACKET_INTEGRITY_INVALID, "snapshot file is not Markdown")
    path = Path(project_root) / snapshot_file.path
    raw = path.read_bytes()
    if sha256_bytes(raw) != snapshot_file.file_hash:
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "Markdown source no longer matches the snapshot",
        )
    try:
        source = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProjectMemoryError(
            ErrorCode.STRUCTURED_PARSE_INCOMPLETE,
            "Markdown UTF-8 decoding failed",
        ) from exc
    lines = source.splitlines(keepends=True) or [""]
    headings = _headings(lines)
    title = headings[0]["title"] if headings else ""
    document_type, detection_basis = _document_type(snapshot_file.path, title)
    document_span = SourceSpan(1, max(1, len(lines)))
    artifacts = [
        _make_artifact(
            manifest=manifest,
            snapshot_file=snapshot_file,
            artifact_type=document_type,
            qualified_name=snapshot_file.path,
            span=document_span,
            content=source,
            metadata={
                "heading_path": [],
                "heading_level": 0,
                "document_artifact_type": document_type,
                "detection_basis": detection_basis,
                **_explicit_fields(source),
            },
        )
    ]
    stack: list[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        level = int(heading["level"])
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, str(heading["title"])))
        end = len(lines)
        for later in headings[index + 1 :]:
            if int(later["level"]) <= level:
                end = int(later["start"]) - 1
                break
        span = SourceSpan(int(heading["start"]), max(int(heading["start"]), end))
        content = _slice(lines, span)
        heading_path = [item[1] for item in stack]
        artifacts.append(
            _make_artifact(
                manifest=manifest,
                snapshot_file=snapshot_file,
                artifact_type="markdown_section",
                qualified_name=f"{snapshot_file.path}#{'/'.join(heading_path)}",
                span=span,
                content=content,
                metadata={
                    "heading_path": heading_path,
                    "heading_level": level,
                    "document_artifact_type": document_type,
                    "detection_basis": "source_heading",
                    **_explicit_fields(content),
                },
            )
        )
    return tuple(sorted(artifacts, key=lambda item: (item.span.start_line, item.span.end_line, item.artifact_id)))
