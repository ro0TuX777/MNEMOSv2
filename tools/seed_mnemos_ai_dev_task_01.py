"""Seed the E1 task_01 starter docs into the active MNEMOS service collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mnemos_seed_utils import (
    build_seed_engram_id,
    build_seed_identity,
    build_seed_snapshot_id,
    normalized_content_hash,
)


DEFAULT_BASE_URL = "http://localhost:8700"
SEED_SCHEMA_VERSION = "ai_dev_task_01_seed_v1"
STARTER_ROOT = ROOT / "benchmarks" / "evaluation" / "ai_dev_memory_quality_e1_task_01_starter_repo"
DEFAULT_MANIFEST_PATH = STARTER_ROOT / "task_control_manifest.json"
DEFAULT_DOCS = [
    "docs/product_scope.md",
    "docs/architecture_decisions.md",
    "docs/data_contract.md",
    "docs/known_issues.md",
    "docs/release_constraints.md",
]
DEFAULT_QUERIES = [
    "Saved views are a local-only productivity feature. No account sync, backend, sharing, or cloud export is in scope.",
    "priority_desc tie-break by updatedAt descending, then title ascending, then id ascending",
    "Missing priority defaults to low",
    "The starter implementation's priority_desc sorting is not compliant with the sorting contract.",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _task_seed_snapshot_id(files: list[str]) -> str:
    components = [SEED_SCHEMA_VERSION, *files]
    return build_seed_snapshot_id(components)


def _build_document(relative_path: str, *, snapshot_id: str) -> dict[str, Any]:
    path = STARTER_ROOT / relative_path
    content = path.read_text(encoding="utf-8")
    seed_identity = build_seed_identity(
        canonical_source_uri=relative_path,
        seed_kind="ai_dev_task_doc",
        schema_version=SEED_SCHEMA_VERSION,
    )
    return {
        "id": build_seed_engram_id(seed_identity),
        "content": content,
        "source": relative_path,
        "neuro_tags": ["ai_dev_task", "task_01", "starter_doc", "source_grounded"],
        "confidence": 0.99,
        "metadata": {
            "source_uri": relative_path,
            "filename": path.name,
            "ingestion_source": "seed_mnemos_ai_dev_task_01",
            "topic": "ai_dev_memory_quality_e1_task_01",
            "canonical_source_uri": relative_path,
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_identity": seed_identity,
            "normalized_content_hash": normalized_content_hash(content),
            "seed_snapshot_id": snapshot_id,
            "retrieval_only": True,
            "source_linked": True,
            "is_superseded": False,
        },
    }


def _get_json(url: str, timeout_s: float) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return data


def _active_collection_name(base_url: str, timeout_s: float) -> str:
    stats = _get_json(f"{base_url.rstrip('/')}/v1/mnemos/stats", timeout_s)
    retrieval = (((stats.get("stats") or {}).get("retrieval")) or {})
    qdrant = ((retrieval.get("tiers") or {}).get("qdrant")) or {}
    collection = qdrant.get("collection")
    return str(collection or "unknown")


def _collection_snapshot(base_url: str, timeout_s: float) -> str:
    stats = _get_json(f"{base_url.rstrip('/')}/v1/mnemos/stats", timeout_s)
    retrieval = (((stats.get("stats") or {}).get("retrieval")) or {})
    qdrant = ((retrieval.get("tiers") or {}).get("qdrant")) or {}
    collection = str(qdrant.get("collection") or "unknown")
    document_count = qdrant.get("document_count", "unknown")
    return f"{collection}:{document_count}"


def seed_task_docs(*, base_url: str, timeout_s: float, files: list[str]) -> dict[str, Any]:
    snapshot_id = _task_seed_snapshot_id(files)
    documents = [_build_document(path, snapshot_id=snapshot_id) for path in files]
    return _post_json(
        f"{base_url.rstrip('/')}/v1/mnemos/index",
        {"documents": documents},
        timeout_s,
    )


def search_query(*, base_url: str, timeout_s: float, query: str, top_k: int) -> dict[str, Any]:
    return _post_json(
        f"{base_url.rstrip('/')}/v1/mnemos/search",
        {
            "query": query,
            "top_k": top_k,
            "retrieval_mode": "semantic",
            "explain": True,
        },
        timeout_s,
    )


def update_task_manifest(
    *,
    manifest_path: Path,
    base_url: str,
    files: list[str],
    timeout_s: float,
    search_results: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    manifest["mnemos_seed_collection_name"] = _active_collection_name(base_url, timeout_s)
    manifest["mnemos_seed_snapshot_id"] = _task_seed_snapshot_id(files)
    manifest["mnemos_seed_collection_snapshot"] = _collection_snapshot(base_url, timeout_s)
    manifest["mnemos_seed_schema_version"] = SEED_SCHEMA_VERSION
    manifest["mnemos_seed_ingestion_source"] = "seed_mnemos_ai_dev_task_01"
    manifest["mnemos_seed_scope"] = files
    manifest["mnemos_seed_query_smoke_checks"] = search_results
    _write_json(manifest_path, manifest)
    return manifest


def _summarize_search_result(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    top_sources = []
    for row in results[:3]:
        engram = row.get("engram", {})
        top_sources.append(
            {
                "source": engram.get("source"),
                "source_uri": (engram.get("metadata") or {}).get("source_uri"),
                "score": row.get("score"),
            }
        )
    return {
        "query": query,
        "result_count": len(results),
        "top_sources": top_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument("--query", action="append", dest="queries")
    args = parser.parse_args()

    files = args.files or list(DEFAULT_DOCS)
    queries = args.queries or list(DEFAULT_QUERIES)

    index_payload = seed_task_docs(
        base_url=args.base_url,
        timeout_s=args.timeout_s,
        files=files,
    )

    search_summaries = []
    for query in queries:
        payload = search_query(
            base_url=args.base_url,
            timeout_s=args.timeout_s,
            query=query,
            top_k=args.top_k,
        )
        search_summaries.append(_summarize_search_result(query, payload))

    manifest = update_task_manifest(
        manifest_path=args.manifest_path,
        base_url=args.base_url,
        files=files,
        timeout_s=args.timeout_s,
        search_results=search_summaries,
    )

    print(f"Indexed: {index_payload.get('result', {}).get('indexed')}")
    print(f"Engram IDs: {len(index_payload.get('result', {}).get('engram_ids', []))}")
    print(f"Active collection: {manifest.get('mnemos_seed_collection_name')}")
    print(f"Seed snapshot: {manifest.get('mnemos_seed_snapshot_id')}")
    print(f"Collection snapshot: {manifest.get('mnemos_seed_collection_snapshot')}")
    for row in search_summaries:
        print(f"Query: {row['query']}")
        print(f"Result count: {row['result_count']}")
        if row["top_sources"]:
            top = row["top_sources"][0]
            print(f"Top source: {top.get('source_uri') or top.get('source')} score={top.get('score')}")
        print()


if __name__ == "__main__":
    main()
