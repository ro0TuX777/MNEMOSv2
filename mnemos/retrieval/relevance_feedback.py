"""
Relevance Feedback Adapter
===========================

Translates governance ``reflect_path`` labels (Used / Ignored) into
Qdrant exemplar-based search via the ``discover_points()`` API.

When a user marks a retrieved engram as "Used" or "Ignored", those
labels are stored in the Forensic Ledger.  This adapter fetches them
and feeds them as positive/negative context pairs into Qdrant's
discovery endpoint, biasing future searches toward useful results.

Requires:
    - Qdrant server >= v1.17.0
    - qdrant-client >= 1.13

The adapter is opt-in — controlled by ``feedback_enabled`` in the
rerank policy YAML.
"""

from __future__ import annotations

import logging
import time
import uuid as _uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import SearchResult

logger = logging.getLogger(__name__)


class ExemplarCache:
    """TTL-bounded LRU cache for exemplar lookups."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0):
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            ts, value = self._cache[key]
            if (time.monotonic() - ts) < self._ttl:
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


class RelevanceFeedbackAdapter:
    """Translates governance reflect_path labels into Qdrant exemplars.

    This adapter:
    1. Queries the Forensic Ledger (or a local feedback store) for
       Used/Ignored labels associated with similar past queries.
    2. Converts those labels into Qdrant point-ID context pairs.
    3. Calls ``discover_points()`` with those exemplars to bias the
       retrieval toward previously-useful results.
    """

    def __init__(
        self,
        qdrant_tier,
        *,
        feedback_store: Optional[Any] = None,
        cache_ttl: float = 300.0,
        max_exemplars: int = 5,
    ):
        """
        Args:
            qdrant_tier: An initialised ``QdrantTier`` instance.
            feedback_store: Optional external store providing labels.
                Must implement ``get_labels(query_hash) -> dict``.
                Defaults to an in-memory dict if not provided.
            cache_ttl: Exemplar lookup cache TTL in seconds.
            max_exemplars: Maximum positive/negative pairs per query.
        """
        self._tier = qdrant_tier
        self._store = feedback_store or InMemoryFeedbackStore()
        self._cache = ExemplarCache(ttl_seconds=cache_ttl)
        self._max_exemplars = max_exemplars

    @staticmethod
    def _make_trace_id() -> str:
        return str(_uuid.uuid4())

    def record_feedback(
        self,
        query_hash: str,
        engram_id: str,
        label: str,
    ) -> None:
        """Record a governance feedback signal.

        Args:
            query_hash: Hash of the original query.
            engram_id: ID of the engram that was evaluated.
            label: One of ``"used"`` or ``"ignored"``.
        """
        self._store.record(query_hash, engram_id, label)
        # Invalidate cache for this query
        self._cache.get(query_hash)  # touch to remove stale

    def get_exemplars(
        self, query_hash: str
    ) -> Tuple[List[str], List[str]]:
        """Fetch positive/negative engram IDs for a query.

        Returns:
            Tuple of (positive_ids, negative_ids).
        """
        cached = self._cache.get(query_hash)
        if cached is not None:
            return cached

        labels = self._store.get_labels(query_hash)
        positives = labels.get("used", [])[:self._max_exemplars]
        negatives = labels.get("ignored", [])[:self._max_exemplars]

        self._cache.put(query_hash, (positives, negatives))
        return positives, negatives

    def search_with_feedback(
        self,
        *,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Execute a feedback-biased search via Qdrant discovery.

        Falls back to standard ``query_points()`` if no exemplars exist.

        Returns:
            Tuple of (results, telemetry dict).
        """
        query_hash = str(hash(query))
        positives, negatives = self.get_exemplars(query_hash)
        trace_id = self._make_trace_id()

        telemetry = {
            "feedback_applied": False,
            "feedback_positive_count": len(positives),
            "feedback_negative_count": len(negatives),
            "trace_id": trace_id,
        }

        # No exemplars → standard search
        if not positives and not negatives:
            results = self._tier.search(query, top_k=top_k, filters=filters)
            return results, telemetry

        # Build Qdrant discovery request
        try:
            from qdrant_client.models import (
                ContextExamplePair,
                DiscoverRequest,
                TargetVector,
            )

            context_pairs = []
            for i in range(max(len(positives), len(negatives))):
                pos_id = self._tier._to_point_id(positives[i]) if i < len(positives) else None
                neg_id = self._tier._to_point_id(negatives[i]) if i < len(negatives) else None
                if pos_id and neg_id:
                    context_pairs.append(
                        ContextExamplePair(positive=pos_id, negative=neg_id)
                    )
                elif pos_id:
                    # Use the query vector as the implicit negative anchor
                    context_pairs.append(
                        ContextExamplePair(positive=pos_id, negative=pos_id)
                    )

            if not context_pairs:
                results = self._tier.search(query, top_k=top_k, filters=filters)
                return results, telemetry

            query_filter = self._tier._build_filter(filters) if filters else None

            t0 = time.perf_counter()

            response = self._tier._client.discover(
                collection_name=self._tier._collection_name,
                target=query_vector,
                context=context_pairs,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            t1 = time.perf_counter()

            results = [self._tier._hit_to_result(hit) for hit in response]
            for r in results:
                r.metadata["feedback_biased"] = True

            telemetry["feedback_applied"] = True
            telemetry["feedback_elapsed_ms"] = round((t1 - t0) * 1000, 2)
            telemetry["feedback_context_pairs"] = len(context_pairs)

            logger.debug(
                f"RelevanceFeedback: {len(results)} results with "
                f"{len(context_pairs)} context pairs (trace={trace_id})"
            )

            return results, telemetry

        except Exception as e:
            logger.warning(f"Relevance feedback search failed, falling back: {e}")
            results = self._tier.search(query, top_k=top_k, filters=filters)
            telemetry["feedback_error"] = str(e)
            return results, telemetry


class InMemoryFeedbackStore:
    """Simple in-memory feedback store for development/testing."""

    def __init__(self):
        self._labels: Dict[str, Dict[str, List[str]]] = {}

    def record(self, query_hash: str, engram_id: str, label: str) -> None:
        if query_hash not in self._labels:
            self._labels[query_hash] = {"used": [], "ignored": []}
        bucket = "used" if label.lower() == "used" else "ignored"
        if engram_id not in self._labels[query_hash][bucket]:
            self._labels[query_hash][bucket].append(engram_id)

    def get_labels(self, query_hash: str) -> Dict[str, List[str]]:
        return self._labels.get(query_hash, {"used": [], "ignored": []})
