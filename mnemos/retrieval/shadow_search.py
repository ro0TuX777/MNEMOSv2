"""Low-priority shadow search for pre-cognitive retrieval."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

from mnemos.memory_over_maps.view_cache import DerivedViewCache


def shadow_cache_key(session_id: str, cluster_id: int) -> str:
    raw = f"pre_cognitive::{session_id}::{int(cluster_id)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ShadowSearchRunner:
    """Runs forecasted intent searches and injects results into cache."""

    def __init__(
        self,
        *,
        search_callable: Callable[[str], List[Dict[str, Any]]],
        cache: DerivedViewCache,
        top_k: int = 5,
        context_provider: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    ) -> None:
        self._search = search_callable
        self._cache = cache
        self._top_k = int(top_k)
        self._context_provider = context_provider
        self.last_run: Optional[Dict[str, Any]] = None

    def run(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        cluster_id = int(forecast["predicted_cluster_id"])
        session_id = str(forecast.get("session_id", "default"))
        query = str(forecast.get("centroid_query", f"cluster {cluster_id}"))
        started = time.perf_counter()
        results = self._search(query)[: self._top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        key = shadow_cache_key(session_id, cluster_id)
        view = {
            "view_type": "pre_cognitive_shadow_search",
            "query": query,
            "predicted_cluster_id": cluster_id,
            "results": results,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "latency_ms": round(elapsed_ms, 3),
        }
        self._cache.set_pre_cognitive(
            key=key,
            query=query,
            cluster_id=cluster_id,
            view=view,
            dependency_refs={"session_id": session_id},
            cache_context=(
                self._context_provider(session_id, query)
                if self._context_provider is not None
                else None
            ),
        )
        self.last_run = {
            "triggered": True,
            "session_id": session_id,
            "predicted_cluster_id": cluster_id,
            "query": query,
            "cache_key": key,
            "latency_ms": round(elapsed_ms, 3),
            "result_count": len(results),
        }
        return dict(self.last_run)


__all__ = ["ShadowSearchRunner", "shadow_cache_key"]
