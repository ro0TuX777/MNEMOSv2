from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .canonical import normalize_relative_path
from .errors import ErrorCode, ProjectMemoryError
from .models import ProjectArtifact, ProjectPacket, SearchHit
from .packet import validate_packet


_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_STRUCTURED_TYPES = {
    "python_symbol",
    "python_config_constant",
    "python_import",
    "markdown_section",
    "markdown_adr",
    "markdown_decision",
    "markdown_handoff",
    "markdown_evaluation_closeout",
    "markdown_agent_instruction",
}


def _tokens(value: str) -> set[str]:
    return {item.lower() for item in _TOKEN.findall(value) if item}


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)


class ProjectMemoryIndex:
    """Deterministic structured lexical index over one verified packet."""

    def __init__(self, packet: ProjectPacket) -> None:
        if not packet.usable:
            raise ProjectMemoryError(
                ErrorCode.STRUCTURED_PARSE_INCOMPLETE,
                "incomplete packets cannot be indexed",
            )
        validate_packet(packet)
        admitted = {item.path for item in packet.snapshot.files}
        for artifact in packet.artifacts:
            if artifact.file_path not in admitted:
                raise ProjectMemoryError(
                    ErrorCode.CROSS_SCOPE_EVIDENCE,
                    "packet contains an artifact outside its manifest",
                )
        self.packet = packet
        self._by_id = {item.artifact_id: item for item in packet.artifacts}
        if len(self._by_id) != len(packet.artifacts):
            raise ProjectMemoryError(
                ErrorCode.PACKET_INTEGRITY_INVALID,
                "packet contains duplicate artifact IDs",
            )

    def get(self, artifact_id: str) -> ProjectArtifact:
        return self._by_id[artifact_id]

    def search(
        self,
        query: str,
        top_k: int = 8,
        path_prefix: str | None = None,
        artifact_types: tuple[str, ...] = (),
    ) -> tuple[SearchHit, ...]:
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        normalized_query = str(query).strip().lower().replace("\\", "/")
        if not normalized_query:
            return ()
        query_tokens = _tokens(normalized_query)
        prefix = normalize_relative_path(path_prefix.rstrip("/")) if path_prefix else None
        allowed_types = set(artifact_types)
        hits: list[SearchHit] = []
        for artifact in self.packet.artifacts:
            if prefix and not (
                artifact.file_path == prefix
                or artifact.file_path.startswith(prefix + "/")
            ):
                continue
            if allowed_types and artifact.artifact_type not in allowed_types:
                continue
            components: dict[str, int] = {}
            reasons: list[str] = []
            qualified = artifact.qualified_name.lower()
            path = artifact.file_path.lower()
            if normalized_query == qualified:
                components["exact_qualified_name"] = 100
                reasons.append("exact_qualified_name")
            if normalized_query == path:
                components["exact_path"] = 90
                reasons.append("exact_path")

            literal_match = False
            for value in _strings(artifact.metadata):
                lowered = value.strip().lower()
                if len(lowered) >= 2 and normalized_query == lowered:
                    literal_match = True
                    break
            if literal_match:
                components["exact_literal"] = 80
                reasons.append("exact_literal")

            headings = artifact.metadata.get("heading_path") or []
            if any(
                normalized_query == str(heading).strip().lower()
                or str(heading).strip().lower() in normalized_query
                for heading in headings
                if str(heading).strip()
            ):
                components["exact_heading"] = 60
                reasons.append("exact_heading")

            identity_matches = query_tokens & _tokens(
                f"{artifact.qualified_name} {artifact.file_path}"
            )
            if identity_matches:
                components["identity_tokens"] = 20 * len(identity_matches)
                reasons.append("identity_tokens")
            content_matches = query_tokens & _tokens(artifact.content)
            if content_matches:
                components["content_tokens"] = 10 * len(content_matches)
                reasons.append("content_tokens")
            if artifact.artifact_type in _STRUCTURED_TYPES:
                components["structural_prior"] = 5
            score = sum(components.values())
            if score:
                hits.append(
                    SearchHit(
                        artifact=artifact,
                        score=score,
                        score_components=components,
                        match_reasons=tuple(reasons),
                    )
                )
        hits.sort(
            key=lambda item: (
                -item.score,
                -item.score_components.get("exact_qualified_name", 0),
                -item.score_components.get("exact_path", 0),
                -item.score_components.get("exact_literal", 0),
                -item.score_components.get("exact_heading", 0),
                item.artifact.artifact_id,
            )
        )
        return tuple(hits[:top_k])
