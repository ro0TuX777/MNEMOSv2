#!/usr/bin/env python3
"""MCP bridge for MNEMOS agent memory.

This server follows the MFS bridge shape: a small MCP facade wraps the
registered MNEMOS REST service without changing MNEMOS runtime behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP


SERVICE_NAME = "mnemos"
DEFAULT_BASE_URL = "http://localhost:8700"
DEFAULT_TIMEOUT_S = 10.0

mcp = FastMCP("mnemos")


def _base_url() -> str:
    return os.getenv("MNEMOS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _timeout_s() -> float:
    try:
        return max(0.1, float(os.getenv("MNEMOS_TIMEOUT_S", DEFAULT_TIMEOUT_S)))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = os.getenv("MNEMOS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(method: str, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{_base_url()}{path}"
    headers = _headers()
    if payload is not None:
        headers["Content-Type"] = "application/json"
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            json=payload,
            timeout=_timeout_s(),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("MNEMOS response must be a JSON object")
        return data
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "mnemos-mcp",
            "service_url": _base_url(),
            "method": method,
            "path": path,
            "error": str(exc),
        }


def _memory_document(
    *,
    content: str,
    memory_type: str,
    source: str,
    confidence: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_metadata = {
        "memory_type": memory_type,
        "source_uri": source,
        "ingestion_source": "mnemos-mcp",
        **(metadata or {}),
    }
    return {
        "content": content,
        "source": source,
        "neuro_tags": [memory_type, "agent_memory", "mcp"],
        "confidence": confidence,
        "metadata": merged_metadata,
    }


@mcp.tool()
def health_check() -> dict[str, Any]:
    """Check MNEMOS service health before using memory."""
    return _request("GET", "/health")


@mcp.tool()
def get_capabilities() -> dict[str, Any]:
    """Inspect MNEMOS capabilities, governance modes, and feature support."""
    return _request("GET", "/v1/mnemos/capabilities")


@mcp.tool()
def search_memory(
    query: str,
    top_k: int = 5,
    filters_json: str = "{}",
    retrieval_mode: str = "hybrid",
    explain: bool = True,
) -> dict[str, Any]:
    """Retrieve project memory with optional evidence and explanation fields."""
    filters = json.loads(filters_json) if filters_json else {}
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "explain": explain,
    }
    if filters:
        payload["filters"] = filters
    return _request("POST", "/v1/mnemos/search", payload=payload)


@mcp.tool()
def write_observation(
    content: str,
    source: str = "agent-session",
    confidence: float = 0.7,
    metadata_json: str = "{}",
) -> dict[str, Any]:
    """Store an agent observation as provenance-bearing memory."""
    metadata = json.loads(metadata_json) if metadata_json else {}
    document = _memory_document(
        content=content,
        memory_type="observation",
        source=source,
        confidence=confidence,
        metadata=metadata,
    )
    return _request("POST", "/v1/mnemos/index", payload={"documents": [document]})


@mcp.tool()
def record_decision(
    decision: str,
    rationale: str,
    source: str = "agent-session",
    confidence: float = 0.85,
    metadata_json: str = "{}",
) -> dict[str, Any]:
    """Store an architecture or workflow decision for future agent use."""
    metadata = json.loads(metadata_json) if metadata_json else {}
    content = f"Decision: {decision}\nRationale: {rationale}"
    document = _memory_document(
        content=content,
        memory_type="decision",
        source=source,
        confidence=confidence,
        metadata=metadata,
    )
    return _request("POST", "/v1/mnemos/index", payload={"documents": [document]})


@mcp.tool()
def find_related_context(
    current_task: str,
    top_k: int = 8,
    filters_json: str = "{}",
) -> dict[str, Any]:
    """Find prior decisions, files, fixes, or constraints related to a task."""
    return search_memory(
        query=current_task,
        top_k=top_k,
        filters_json=filters_json,
        retrieval_mode="hybrid",
        explain=True,
    )


@mcp.tool()
def detect_contradictions(
    claim: str,
    top_k: int = 8,
) -> dict[str, Any]:
    """Search for potentially conflicting or stale memories for a claim."""
    query = f"contradiction stale superseded conflicting evidence: {claim}"
    return search_memory(query=query, top_k=top_k, retrieval_mode="hybrid", explain=True)


@mcp.tool()
def summarize_session_handoff(
    summary: str,
    source: str = "agent-session-handoff",
    metadata_json: str = "{}",
) -> dict[str, Any]:
    """Store a concise session handoff summary for future continuation."""
    metadata = json.loads(metadata_json) if metadata_json else {}
    document = _memory_document(
        content=summary,
        memory_type="session_handoff",
        source=source,
        confidence=0.8,
        metadata=metadata,
    )
    return _request("POST", "/v1/mnemos/index", payload={"documents": [document]})


@mcp.tool()
def explain_memory_provenance(engram_id: str) -> dict[str, Any]:
    """Inspect an engram and its metadata/provenance."""
    return _request("GET", f"/v1/mnemos/engrams/{engram_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "9700")))
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        os.environ["PORT"] = str(args.port)
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
