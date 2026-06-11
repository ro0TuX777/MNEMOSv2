"""Emit a Phase 9 RAPTOR-lite hierarchy dry-run report from Qdrant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.clustering_runner import HierarchicalClusteringRunner

DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "outputs" / "raw" / "hierarchy_report.json"


def _extract_vector(point: Any) -> np.ndarray | None:
    vector = getattr(point, "vector", None)
    if isinstance(vector, dict):
        vector = vector.get("dense_768") or vector.get("dense")
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32)


def _load_qdrant_collection(
    *,
    url: str,
    collection: str,
    limit: int,
) -> Tuple[List[Engram], np.ndarray]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, timeout=30)
    engrams: List[Engram] = []
    vectors: List[np.ndarray] = []
    offset = None
    while len(engrams) < limit:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=min(256, limit - len(engrams)),
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        for point in points:
            payload: Dict[str, Any] = getattr(point, "payload", None) or {}
            vector = _extract_vector(point)
            if vector is None:
                continue
            app_meta = {
                key.removeprefix("app_"): value
                for key, value in payload.items()
                if key.startswith("app_")
            }
            engram_id = str(payload.get("_mnemos_id") or payload.get("id") or payload.get("engram_id") or getattr(point, "id"))
            engrams.append(
                Engram(
                    id=engram_id,
                    content=str(payload.get("content") or ""),
                    source=str(payload.get("source") or ""),
                    metadata=app_meta,
                )
            )
            vectors.append(vector)
        if next_offset is None:
            break
        offset = next_offset
    if not vectors:
        return engrams, np.empty((0, 0), dtype=np.float32)
    return engrams, np.vstack(vectors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 9 hierarchy dry-run from Qdrant")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="mnemos_engrams_nomic_mrl")
    parser.add_argument("--limit", type=int, default=2121)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Index generated summary engrams into Qdrant")
    parser.add_argument("--embedding-model", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.apply:
        # Replace, don't accumulate: drop prior summary engrams before the
        # leaf scan so re-runs never orphan stale summaries or skew the scan
        # window, then ensure the summary filter has a payload index.
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
            PayloadSchemaType,
        )

        cleanup_client = QdrantClient(url=args.qdrant_url, timeout=30)
        cleanup_client.delete(
            collection_name=args.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="app_is_summary_engram", match=MatchValue(value=True))]
                )
            ),
        )
        try:
            cleanup_client.create_payload_index(
                collection_name=args.collection,
                field_name="app_is_summary_engram",
                field_schema=PayloadSchemaType.BOOL,
            )
        except Exception:
            pass  # index already exists

    engrams, vectors = _load_qdrant_collection(
        url=args.qdrant_url,
        collection=args.collection,
        limit=args.limit,
    )
    indexer = None
    if args.apply:
        from mnemos.retrieval.qdrant_tier import QdrantTier

        indexer = QdrantTier(
            url=args.qdrant_url,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            embedding_dim=vectors.shape[1] if vectors.size else 768,
            gpu_device=args.gpu_device,
        )

    report = HierarchicalClusteringRunner(
        n_clusters=args.clusters,
        model_name=args.embedding_model,
    ).run(
        engrams,
        vectors=vectors if len(engrams) else None,
        dry_run=not args.apply,
        indexer=indexer,
        output_path=args.output,
    )
    print(f"collection: {args.collection}")
    print(f"engrams scanned: {report.engrams_scanned}")
    print(f"clusters: {report.cluster_count}")
    print(f"summary writes: {report.summary_engram_writes}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
