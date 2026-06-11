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
            engram_id = str(payload.get("id") or payload.get("engram_id") or getattr(point, "id"))
            engrams.append(
                Engram(
                    id=engram_id,
                    content=str(payload.get("content") or ""),
                    source=str(payload.get("source") or ""),
                    metadata=payload if isinstance(payload, dict) else {},
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    engrams, vectors = _load_qdrant_collection(
        url=args.qdrant_url,
        collection=args.collection,
        limit=args.limit,
    )
    report = HierarchicalClusteringRunner(n_clusters=args.clusters).run(
        engrams,
        vectors=vectors if len(engrams) else None,
        output_path=args.output,
    )
    print(f"collection: {args.collection}")
    print(f"engrams scanned: {report.engrams_scanned}")
    print(f"clusters: {report.cluster_count}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
