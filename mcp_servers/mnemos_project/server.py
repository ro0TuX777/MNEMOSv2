#!/usr/bin/env python3
"""Serve one snapshot-bound local project-memory packet over read-only MCP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from prototype.local_project_memory_r0.errors import (  # noqa: E402
    ErrorCode,
    ProjectMemoryError,
)
from prototype.local_project_memory_r0.models import ProjectArtifact  # noqa: E402
from prototype.local_project_memory_r0.packet import (  # noqa: E402
    AUTHORITY_BOUNDARY,
    load_packet,
)
from prototype.local_project_memory_r0.retrieval import ProjectMemoryIndex  # noqa: E402
from prototype.local_project_memory_r0.snapshot import verify_snapshot  # noqa: E402


READ_ONLY_TOOL_NAMES = {
    "project_memory_health",
    "get_project_identity",
    "search_project_memory",
    "get_project_artifact",
    "verify_project_snapshot",
}

mcp = FastMCP("mnemos-project-memory-r0")


@dataclass(frozen=True)
class _ServerState:
    packet_path: Path
    project_root: Path
    repo_id: str
    index: ProjectMemoryIndex


_STATE: _ServerState | None = None


def _unavailable(code: ErrorCode, message: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason_code": code.value,
        "message": message,
        "authority_boundary": AUTHORITY_BOUNDARY,
    }


def _load_state() -> _ServerState:
    global _STATE
    if _STATE is not None:
        return _STATE
    packet_value = os.getenv("MNEMOS_PROJECT_PACKET", "").strip()
    root_value = os.getenv("MNEMOS_PROJECT_ROOT", "").strip()
    repo_id = os.getenv("MNEMOS_PROJECT_REPO_ID", "").strip()
    if not packet_value or not root_value or not repo_id:
        raise ProjectMemoryError(
            ErrorCode.PROJECT_ROOT_REQUIRED,
            "project packet, root, and repo ID configuration are required",
        )
    packet = load_packet(Path(packet_value))
    if packet.snapshot.repo_id != repo_id:
        raise ProjectMemoryError(
            ErrorCode.REPO_ID_MISMATCH,
            "configured repo ID does not match the packet",
        )
    _STATE = _ServerState(
        packet_path=Path(packet_value).resolve(),
        project_root=Path(root_value).resolve(),
        repo_id=repo_id,
        index=ProjectMemoryIndex(packet),
    )
    return _STATE


def _verification(state: _ServerState):
    return verify_snapshot(state.project_root, state.index.packet.snapshot)


def _artifact_value(artifact: ProjectArtifact) -> dict[str, Any]:
    return artifact.to_dict()


def _hit_value(hit) -> dict[str, Any]:
    artifact = hit.artifact
    return {
        **_artifact_value(artifact),
        "score": hit.score,
        "score_components": dict(hit.score_components),
        "match_reasons": list(hit.match_reasons),
    }


def _abstained(state: _ServerState, verification) -> dict[str, Any]:
    return {
        "status": "abstained",
        "reason_code": verification.reason_code or ErrorCode.SNAPSHOT_MISMATCH.value,
        "repo_id": state.repo_id,
        "snapshot_id": state.index.packet.snapshot.snapshot_id,
        "mismatches": list(verification.mismatches),
        "results": [],
        "authority_boundary": AUTHORITY_BOUNDARY,
    }


@mcp.tool()
def project_memory_health() -> dict[str, Any]:
    """Check packet integrity, configured repo identity, and live snapshot freshness."""
    try:
        state = _load_state()
        verification = _verification(state)
        if not verification.fresh:
            return _abstained(state, verification)
        return {
            "status": "ok",
            "repo_id": state.repo_id,
            "snapshot_id": state.index.packet.snapshot.snapshot_id,
            "packet_sha256": state.index.packet.packet_sha256,
            "artifact_count": len(state.index.packet.artifacts),
            "fresh": True,
            "authority_boundary": AUTHORITY_BOUNDARY,
        }
    except ProjectMemoryError as exc:
        return _unavailable(exc.code, "project memory is not available")
    except Exception:
        return _unavailable(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "project memory initialization failed",
        )


@mcp.tool()
def get_project_identity() -> dict[str, Any]:
    """Return packet identity, explicit scope, exclusions, and freshness without source content."""
    try:
        state = _load_state()
        verification = _verification(state)
        packet = state.index.packet
        return {
            "status": "ok" if verification.fresh else "abstained",
            "reason_code": verification.reason_code,
            "repo_id": packet.snapshot.repo_id,
            "snapshot_id": packet.snapshot.snapshot_id,
            "packet_sha256": packet.packet_sha256,
            "branch": packet.snapshot.branch,
            "commit_hash": packet.snapshot.commit_hash,
            "working_tree_state": packet.snapshot.working_tree_state,
            "scope": packet.snapshot.scope.to_dict(),
            "exclusions": list(packet.snapshot.exclusions),
            "admitted_file_count": len(packet.snapshot.files),
            "artifact_count": len(packet.artifacts),
            "fresh": verification.fresh,
            "authority_boundary": AUTHORITY_BOUNDARY,
        }
    except ProjectMemoryError as exc:
        return _unavailable(exc.code, "project identity is not available")
    except Exception:
        return _unavailable(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "project identity initialization failed",
        )


@mcp.tool()
def search_project_memory(
    query: str,
    top_k: int = 8,
    path_prefix: str = "",
    artifact_types_json: str = "[]",
) -> dict[str, Any]:
    """Search fresh, source-backed Python and Markdown evidence in the configured packet."""
    try:
        state = _load_state()
        verification = _verification(state)
        if not verification.fresh:
            return _abstained(state, verification)
        parsed_types = json.loads(artifact_types_json or "[]")
        if not isinstance(parsed_types, list) or not all(isinstance(item, str) for item in parsed_types):
            raise ValueError("artifact_types_json must be a JSON string list")
        hits = state.index.search(
            query,
            top_k=top_k,
            path_prefix=path_prefix or None,
            artifact_types=tuple(parsed_types),
        )
        return {
            "status": "ok",
            "repo_id": state.repo_id,
            "snapshot_id": state.index.packet.snapshot.snapshot_id,
            "retrieval_mode": "structured_lexical_r0",
            "results": [_hit_value(hit) for hit in hits],
            "authority_boundary": AUTHORITY_BOUNDARY,
        }
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid_request",
            "reason_code": "INVALID_SEARCH_REQUEST",
            "message": str(exc),
            "results": [],
            "authority_boundary": AUTHORITY_BOUNDARY,
        }
    except ProjectMemoryError as exc:
        return _unavailable(exc.code, "project search is not available")
    except Exception:
        return _unavailable(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "project search initialization failed",
        )


@mcp.tool()
def get_project_artifact(artifact_id: str) -> dict[str, Any]:
    """Return one exact artifact only when the configured live snapshot is fresh."""
    try:
        state = _load_state()
        verification = _verification(state)
        if not verification.fresh:
            return _abstained(state, verification)
        try:
            artifact = state.index.get(artifact_id)
        except KeyError:
            return {
                "status": "not_found",
                "reason_code": "ARTIFACT_NOT_FOUND",
                "authority_boundary": AUTHORITY_BOUNDARY,
            }
        return {
            "status": "ok",
            "repo_id": state.repo_id,
            "snapshot_id": state.index.packet.snapshot.snapshot_id,
            "artifact": _artifact_value(artifact),
            "authority_boundary": AUTHORITY_BOUNDARY,
        }
    except ProjectMemoryError as exc:
        return _unavailable(exc.code, "project artifact is not available")
    except Exception:
        return _unavailable(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "project artifact initialization failed",
        )


@mcp.tool()
def verify_project_snapshot() -> dict[str, Any]:
    """Rehash admitted live files and compare them with the configured packet snapshot."""
    try:
        state = _load_state()
        verification = _verification(state)
        return {
            "status": "ok" if verification.fresh else "abstained",
            "reason_code": verification.reason_code,
            "repo_id": state.repo_id,
            "snapshot_id": state.index.packet.snapshot.snapshot_id,
            "fresh": verification.fresh,
            "mismatches": list(verification.mismatches),
            "authority_boundary": AUTHORITY_BOUNDARY,
        }
    except ProjectMemoryError as exc:
        return _unavailable(exc.code, "project snapshot verification is not available")
    except Exception:
        return _unavailable(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "project snapshot verification failed",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "9710")))
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        os.environ["PORT"] = str(args.port)
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
