"""Disposable live probe for retrieval duplicate suppression."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mnemos.engram.model import Engram
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.qdrant_tier import QdrantTier
from mnemos.retrieval.retrieval_router import RetrievalRouter
from tools.mnemos_seed_utils import normalized_content_hash


DEFAULT_QDRANT_URL = os.getenv("MNEMOS_QDRANT_URL", "http://localhost:6333")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "MNEMOS_EMBEDDING_MODEL",
    "nomic-ai/nomic-embed-text-v1.5",
)


def run_probe(*, qdrant_url: str, embedding_model: str) -> dict[str, Any]:
    suffix = uuid.uuid4().hex[:8]
    collection_name = f"mnemos_dup_probe_{suffix}"
    qdrant_client = QdrantClient(url=qdrant_url, timeout=30)

    try:
        qdrant_tier = QdrantTier(
            url=qdrant_url,
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_dim=768,
            gpu_device=os.getenv("MNEMOS_GPU_DEVICE", "cuda"),
        )
        router = RetrievalRouter(
            semantic_fusion=TierFusion([qdrant_tier]),
            lexical_tier=None,
        )

        canonical_source_uri = "docs/benchmarks/gatemem_program_status.md"
        duplicate_content = (
            "Further GateMem policy and implementation work is paused. "
            "The next meaningful milestone requires an independent sealed-evaluation custodian."
        )
        duplicate_hash = normalized_content_hash(duplicate_content)

        engrams = [
            Engram(
                id=f"dup-{i}",
                content=duplicate_content,
                source=canonical_source_uri,
                confidence=0.99,
                neuro_tags=["repo_summary", "gatemem", "pause"],
                metadata={
                    "canonical_source_uri": canonical_source_uri,
                    "source_uri": canonical_source_uri,
                    "normalized_content_hash": duplicate_hash,
                    "is_seeded_summary": True,
                    "seed_identity": f"dup::{i}",
                    "schema_version": "probe_v1",
                },
            )
            for i in range(3)
        ]
        engrams.append(
            Engram(
                id="supporting-doc",
                content=(
                    "GateMem continuation is blocked pending sealed evaluation governance and external custodian setup."
                ),
                source="docs/benchmarks/gatemem_g4_offline_reference_implementation.md",
                confidence=0.95,
                metadata={
                    "canonical_source_uri": "docs/benchmarks/gatemem_g4_offline_reference_implementation.md",
                    "source_uri": "docs/benchmarks/gatemem_g4_offline_reference_implementation.md",
                    "normalized_content_hash": normalized_content_hash(
                        "GateMem continuation is blocked pending sealed evaluation governance and external custodian setup."
                    ),
                },
            )
        )

        qdrant_tier.index(engrams)
        results, meta = router.search(
            query="Further GateMem policy and implementation work is paused",
            top_k=5,
            retrieval_mode="semantic",
            fusion_policy="semantic_dominant",
            explain=True,
        )

        top_results = []
        for row in results:
            top_results.append(
                {
                    "engram_id": row.engram.id,
                    "source": row.engram.source,
                    "score": round(float(row.score), 4),
                    "duplicate_suppression": row.metadata.get("duplicate_suppression"),
                }
            )

        return {
            "status": "PASS",
            "collection_name": collection_name,
            "retrieval_mode": "semantic",
            "result_count": len(results),
            "top_results": top_results,
            "meta_duplicate_suppression": meta.get("duplicate_suppression"),
            "suppression_applied": bool(
                (meta.get("duplicate_suppression") or {}).get("applied")
            ),
        }
    finally:
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    result = run_probe(
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
