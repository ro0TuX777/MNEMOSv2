"""Seed a focused set of MNEMOS repo documents into a running MNEMOS service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mnemos_seed_manifest import DEFAULT_MANIFEST_PATH, update_manifest_section
from tools.mnemos_seed_utils import (
    build_seed_engram_id,
    build_seed_identity,
    build_seed_snapshot_id,
    normalized_content_hash,
)


DEFAULT_BASE_URL = "http://localhost:8700"
SEED_SCHEMA_VERSION = "repo_context_seed_v1"
DEFAULT_FILES = [
    "docs/benchmarks/gatemem_program_status.md",
    "docs/benchmarks/gatemem_g4_offline_reference_implementation.md",
    "docs/adr/0013-gatemem-g4-offline-reference-implementation-proposal.md",
    "benchmarks/results/gatemem_g4_frozen_reference_manifest.md",
]


def _document_for_path(
    relative_path: str,
    tags: list[str],
    topic: str,
    *,
    seed_snapshot_id: str | None = None,
) -> dict[str, Any]:
    path = ROOT / relative_path
    content = path.read_text(encoding="utf-8")
    seed_identity = build_seed_identity(
        canonical_source_uri=relative_path,
        seed_kind="repo_context_document",
        schema_version=SEED_SCHEMA_VERSION,
    )
    return {
        "id": build_seed_engram_id(seed_identity),
        "content": content,
        "source": relative_path,
        "neuro_tags": tags,
        "confidence": 0.95,
        "metadata": {
            "source_uri": relative_path,
            "filename": path.name,
            "ingestion_source": "seed_mnemos_repo_context",
            "topic": topic,
            "canonical_source_uri": relative_path,
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_identity": seed_identity,
            "normalized_content_hash": normalized_content_hash(content),
            "seed_snapshot_id": seed_snapshot_id or _context_seed_snapshot_id(DEFAULT_FILES),
            "source_linked": True,
            "is_superseded": False,
        },
    }


def _context_seed_snapshot_id(files: list[str]) -> str:
    return build_seed_snapshot_id([SEED_SCHEMA_VERSION, *files])


def seed_documents(
    *,
    base_url: str,
    files: list[str],
    tags: list[str],
    topic: str,
    timeout_s: float,
) -> dict[str, Any]:
    snapshot_id = _context_seed_snapshot_id(files)
    documents = [
        _document_for_path(path, tags, topic, seed_snapshot_id=snapshot_id)
        for path in files
    ]
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/mnemos/index",
        json={"documents": documents},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("index response must be a JSON object")
    return payload


def search_seed(
    *,
    base_url: str,
    query: str,
    timeout_s: float,
    top_k: int,
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/mnemos/search",
        json={
            "query": query,
            "top_k": top_k,
            "retrieval_mode": "semantic",
            "explain": True,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("search response must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--topic", default="gatemem_reference")
    parser.add_argument("--query", default="GateMem G4 frozen regression baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    files = args.files or list(DEFAULT_FILES)
    tags = args.tags or ["repo_context", "gatemem", "markdown"]

    seed_result = seed_documents(
        base_url=args.base_url,
        files=files,
        tags=tags,
        topic=args.topic,
        timeout_s=args.timeout_s,
    )
    context_snapshot_id = _context_seed_snapshot_id(files)
    context_documents = [
        _document_for_path(path, tags, args.topic, seed_snapshot_id=context_snapshot_id)
        for path in files
    ]
    manifest = update_manifest_section(
        section_name="repo_context",
        section_payload={
            "seed_schema_version": SEED_SCHEMA_VERSION,
            "seed_snapshot_id": context_snapshot_id,
            "document_count": len(context_documents),
            "seed_identities": [
                item["metadata"]["seed_identity"] for item in context_documents
            ],
            "normalized_content_hashes": [
                item["metadata"]["normalized_content_hash"] for item in context_documents
            ],
            "canonical_source_uris": files,
        },
        path=Path(args.manifest_path),
    )
    search_result = search_seed(
        base_url=args.base_url,
        query=args.query,
        timeout_s=args.timeout_s,
        top_k=args.top_k,
    )

    print(f"Indexed: {seed_result.get('result', {}).get('indexed')}")
    print(f"Engram IDs: {len(seed_result.get('result', {}).get('engram_ids', []))}")
    print(f"Seed manifest: {args.manifest_path}")
    print(f"Composite seed snapshot: {manifest.get('seed_snapshot_id')}")
    print(f"Search status: {search_result.get('status')}")
    print(f"Search result count: {len(search_result.get('results', []))}")
    if search_result.get("results"):
        top = search_result["results"][0]
        engram = top.get("engram", {})
        print(f"Top source: {engram.get('source')}")
        print(f"Top score: {top.get('score')}")
        print(f"Top content: {str(engram.get('content', ''))[:180]}")


if __name__ == "__main__":
    main()
