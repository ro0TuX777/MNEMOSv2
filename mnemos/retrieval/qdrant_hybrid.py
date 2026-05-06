"""
Qdrant-Native Hybrid Fusion (Server-Side RRF)
===============================================

Performs hybrid lexical + semantic search in a single Qdrant round-trip
using the ``prefetch`` parameter and ``Fusion.RRF``.

Requires:
    - Qdrant server >= v1.17.0
    - qdrant-client >= 1.13
    - A full-text payload index on the ``content`` field
      (created automatically by ``QdrantTier._ensure_text_index()``)

Falls back to Python-side ``HybridFusion`` when the text index is
unavailable or the query fails.
"""

from __future__ import annotations

import logging
import time
import uuid as _uuid
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import SearchResult
from mnemos.engram.model import Engram

logger = logging.getLogger(__name__)


class QdrantHybridFusion:
    """Server-side hybrid RRF fusion via Qdrant prefetch.

    Combines dense vector search with full-text payload search in a
    single ``query_points()`` call, delegating rank fusion to Qdrant's
    built-in Reciprocal Rank Fusion.
    """

    def __init__(self, qdrant_tier):
        """
        Args:
            qdrant_tier: An initialised ``QdrantTier`` instance.
        """
        self._tier = qdrant_tier

    @property
    def available(self) -> bool:
        """True if both the Qdrant client and text index are ready."""
        return (
            self._tier._client is not None
            and getattr(self._tier, "_text_index_ready", False)
        )

    @staticmethod
    def _make_trace_id() -> str:
        return str(_uuid.uuid4())

    def fuse(
        self,
        *,
        query: str,
        query_vector: List[float],
        top_k: int = 25,
        filters: Optional[Dict[str, Any]] = None,
        semantic_limit: int = 50,
        lexical_limit: int = 50,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """Execute a single-round-trip hybrid search with server-side RRF.

        Args:
            query: Raw query text for the full-text arm.
            query_vector: Pre-computed embedding for the dense arm.
            top_k: Maximum fused results to return.
            filters: Optional metadata filters (applied to both arms).
            semantic_limit: Dense arm prefetch depth.
            lexical_limit: Lexical arm prefetch depth.

        Returns:
            Tuple of (ranked results, telemetry dict).

        Raises:
            RuntimeError: If the Qdrant client or text index is not ready.
        """
        if not self.available:
            raise RuntimeError(
                "QdrantHybridFusion is not available "
                "(client or text index not ready)"
            )

        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchText,
            Prefetch,
            FusionQuery,
            Fusion,
        )

        trace_id = self._make_trace_id()
        query_filter = self._tier._build_filter(filters) if filters else None

        t0 = time.perf_counter()

        try:
            # Build prefetch arms
            semantic_prefetch = Prefetch(
                query=query_vector,
                limit=semantic_limit,
                filter=query_filter,
            )

            lexical_prefetch = Prefetch(
                query=query_vector,       # Still needs a vector for scoring
                limit=lexical_limit,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="content",
                            match=MatchText(text=query),
                        ),
                    ]
                    + (query_filter.must if query_filter and query_filter.must else []),
                ),
            )

            # Single round-trip: Qdrant fuses both arms via RRF
            response = self._tier._client.query_points(
                collection_name=self._tier._collection_name,
                prefetch=[semantic_prefetch, lexical_prefetch],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000

            results_raw = getattr(response, "points", response)
            results = [self._tier._hit_to_result(hit) for hit in results_raw]

            # Tag each result with the fusion engine
            for r in results:
                r.metadata["fusion_engine"] = "qdrant_rrf"
                r.tier = "hybrid"

            telemetry = {
                "fusion_engine": "qdrant_rrf",
                "trace_id": trace_id,
                "semantic_prefetch_limit": float(semantic_limit),
                "lexical_prefetch_limit": float(lexical_limit),
                "fused_result_count": float(len(results)),
                "elapsed_ms": round(elapsed_ms, 2),
                "query_filter_applied": query_filter is not None,
                # Qdrant RRF doesn't break down per-arm counts; approximate
                "lexical_candidates": float(lexical_limit),
                "semantic_candidates": float(semantic_limit),
                "union_candidates": float(len(results)),
                "overlap_candidates": 0.0,  # not available server-side
                "lexical_only_candidates": 0.0,
                "semantic_only_candidates": 0.0,
            }

            logger.debug(
                f"QdrantHybridFusion: {len(results)} results in {elapsed_ms:.1f}ms "
                f"(trace={trace_id})"
            )

            return results, telemetry

        except Exception as e:
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            logger.error(f"QdrantHybridFusion failed (trace={trace_id}): {e}")
            raise RuntimeError(f"Qdrant hybrid fusion failed: {e}") from e
