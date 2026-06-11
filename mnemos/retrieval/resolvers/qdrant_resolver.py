import logging
import time
from typing import Dict, List, Optional

from mnemos.engram.model import Engram
from mnemos.retrieval.graph_tier import EngramResolver
from mnemos.retrieval.qdrant_tier import QdrantTier
from mnemos.retrieval.telemetry import get_telemetry_sink

logger = logging.getLogger(__name__)

class QdrantEngramResolver(EngramResolver):
    """
    Read-only live graph resolver using Qdrant payload edges.
    Enforces batching to avoid N+1 queries.
    """

    def __init__(self, qdrant_tier: QdrantTier):
        self._tier = qdrant_tier
        self._cache: Dict[str, Engram] = {}
        self._telemetry_sink = get_telemetry_sink({})

    def get_by_id(self, engram_id: str) -> Optional[Engram]:
        # Return from cache. If not in cache, we assume missing or not prefetched.
        return self._cache.get(engram_id)

    def get_degree(self, engram_id: str) -> int:
        eng = self.get_by_id(engram_id)
        return len(eng.edges) if eng else 0

    def get_edge_type(self, source_id: str, target_id: str) -> str:
        # Currently edge types are not typed in the payload natively beyond the default structural link
        return "structural"

    def prefetch(self, engram_ids: List[str]) -> None:
        """
        Batch retrieve missing engrams from Qdrant to avoid N+1 queries.
        """
        # Deduplicate and filter out already cached IDs
        missing_ids = list({eid for eid in engram_ids if eid not in self._cache})
        
        if not missing_ids or not self._tier._client:
            return

        try:
            point_ids = [self._tier._to_point_id(eid) for eid in missing_ids]
            
            start_t = time.perf_counter()
            results = self._tier._client.retrieve(
                collection_name=self._tier._collection_name,
                ids=point_ids,
                with_payload=True
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000

            # Log telemetry for the batch retrieval
            self._telemetry_sink.emit({
                "action": "qdrant_resolver_prefetch",
                "requested_ids": len(missing_ids),
                "retrieved_ids": len(results),
                "latency_ms": elapsed_ms
            })

            for hit in results:
                # Use QdrantTier's internal hit_to_result reconstruction logic
                search_res = self._tier._hit_to_result(hit)
                eng = search_res.engram
                self._cache[eng.id] = eng

        except Exception as e:
            logger.error(f"Qdrant resolver batch retrieve failed: {e}")
            self._telemetry_sink.emit({
                "warning": "QDRANT_RESOLVER_FAILURE",
                "action": "qdrant_resolver_prefetch",
                "error": str(e)
            })
            # Fail gracefully, leaving cache empty for missing IDs

    # ──────────────────────────────────────────────────────────
    # Mutating Methods (Blocked)
    # ──────────────────────────────────────────────────────────

    def add_edge(self, source_id: str, target_id: str) -> None:
        raise NotImplementedError("Read-only resolver")

    def update_engram(self, engram: Engram) -> None:
        raise NotImplementedError("Read-only resolver")

    def delete_edge(self, source_id: str, target_id: str) -> None:
        raise NotImplementedError("Read-only resolver")

    def save(self) -> None:
        raise NotImplementedError("Read-only resolver")
