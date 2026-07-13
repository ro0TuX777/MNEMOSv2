"""Associative Routing E2 — opt-in, bounded candidate-expansion engine.

Adds at most a few source-linked candidates to an already-computed normal
retrieval result list. This module never calls the retrieval router's
default path on its own initiative — it is only invoked by the caller
(service/app.py) after normal retrieval, governance, and the low-relevance
abstention decision have already completed on the normal-origin results, and
only when the caller has explicitly opted in. See
docs/associative_routing_e2_design_note.md and the E2 authorization header
for the full invariant list:

* fail-closed on the kill switch or any internal error;
* bounded by MAX_PATHS / MAX_ADDED_CANDIDATES / MAX_EXPANSION_LATENCY_MS;
* never blends or renormalizes scores to look like normal retrieval;
* never durable-writes (only ever calls retrieval_router.search, never an
  index/write path);
* candidates are appended after existing results, never interleaved by
  score, so they cannot outrank normal candidates.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prototype.associative_routing_e0 import AssociativeRouter, build_projection
from mnemos.retrieval.base import SearchResult

from .config import (
    CANDIDATE_EXPANSION_ENABLE_ENV,
    E2_FIXTURES_DIR,
    MAX_ADDED_CANDIDATES,
    MAX_EXPANSION_LATENCY_MS,
    MAX_PATHS,
)

logger = logging.getLogger(__name__)


def _block(
    *,
    status: str,
    latency_ms: float,
    projection_snapshot: Optional[str] = None,
    candidate_paths_considered: int = 0,
    candidate_paths_used: int = 0,
    candidates_added: int = 0,
    candidates_deduplicated: int = 0,
    candidates_rejected_by_policy: int = 0,
    reason_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "projection_snapshot": projection_snapshot,
        "candidate_paths_considered": candidate_paths_considered,
        "candidate_paths_used": candidate_paths_used,
        "candidates_added": candidates_added,
        "candidates_deduplicated": candidates_deduplicated,
        "candidates_rejected_by_policy": candidates_rejected_by_policy,
        "latency_ms": round(latency_ms, 3),
        "reason_codes": reason_codes or [],
        "non_authoritative": True,
    }


def _extract_source_uri(result: SearchResult) -> Optional[str]:
    engram = getattr(result, "engram", None)
    engram_meta = getattr(engram, "metadata", None) or {}
    own_meta = getattr(result, "metadata", None) or {}
    return (
        engram_meta.get("source_uri")
        or engram_meta.get("canonical_source_uri")
        or own_meta.get("source_uri")
        or (getattr(engram, "source", None) if engram else None)
    )


class CandidateExpansionEngine:
    """Read-only, opt-in, bounded candidate-expansion lane over the E2 projection."""

    def __init__(self, fixtures_dir: Path = E2_FIXTURES_DIR) -> None:
        self._fixtures_dir = fixtures_dir
        self._router: Optional[AssociativeRouter] = None
        self._source_uri_by_content_id: Dict[str, str] = {}

    def _ensure_built(self) -> None:
        if self._router is not None:
            return
        projection = build_projection(self._fixtures_dir)
        self._router = AssociativeRouter(projection=projection)
        self._source_uri_by_content_id = {
            content_id: ref.source_uri for content_id, ref in projection.content_index.items()
        }

    def expand(
        self,
        query: str,
        *,
        existing_results: List[SearchResult],
        retrieval_router: Any,
        filters: Optional[Dict[str, Any]],
        retrieval_mode: str,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Return (injected_candidates, response_block). Never raises."""
        t0 = time.perf_counter()

        if os.environ.get(CANDIDATE_EXPANSION_ENABLE_ENV, "false").lower() != "true":
            return [], _block(
                status="disabled",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                reason_codes=["kill_switch_disabled"],
            )

        try:
            self._ensure_built()
            router = self._router
            assert router is not None
            routing_response = router.route(query)
        except Exception:
            logger.exception("Associative expansion engine failed; reporting unavailable.")
            return [], _block(
                status="unavailable",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                reason_codes=["adapter_error"],
            )

        if routing_response.routing_result == "abstained":
            reason = routing_response.abstention.reason_code if routing_response.abstention else None
            return [], _block(
                status="abstained",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                projection_snapshot=routing_response.projection_snapshot,
                reason_codes=[reason] if reason else [],
            )

        existing_source_uris = {
            uri for uri in (_extract_source_uri(r) for r in existing_results) if uri
        }

        candidate_content_ids = routing_response.candidate_content_ids[:MAX_PATHS]
        paths_considered = len(routing_response.routing_paths)
        paths_used = 0
        candidates_added = 0
        candidates_deduplicated = 0
        candidates_rejected_by_policy = 0
        reason_codes: List[str] = []
        injected: List[SearchResult] = []

        for content_id in candidate_content_ids:
            if candidates_added >= MAX_ADDED_CANDIDATES:
                break

            source_uri = self._source_uri_by_content_id.get(content_id)
            if not source_uri:
                candidates_rejected_by_policy += 1
                continue
            if source_uri in existing_source_uris:
                candidates_deduplicated += 1
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms >= MAX_EXPANSION_LATENCY_MS:
                if "latency_budget_exceeded" not in reason_codes:
                    reason_codes.append("latency_budget_exceeded")
                break

            resolved = self._resolve_candidate(
                retrieval_router=retrieval_router,
                query=query,
                source_uri=source_uri,
                filters=filters,
                retrieval_mode=retrieval_mode,
            )
            if resolved is None:
                candidates_rejected_by_policy += 1
                continue

            path = next(
                (p for p in routing_response.routing_paths if content_id in p.content_ids),
                None,
            )
            resolved.metadata["candidate_origin"] = "associative_expansion"
            resolved.metadata["associative_routing_path"] = {
                "cue_ids": list(path.cue_ids) if path else [],
                "tag_ids": list(path.tag_ids) if path else [],
                "explanation": path.explanation if path else "",
            }
            resolved.metadata["associative_projection_snapshot"] = routing_response.projection_snapshot
            resolved.metadata["non_authoritative"] = True

            injected.append(resolved)
            existing_source_uris.add(source_uri)
            candidates_added += 1
            paths_used += 1

        status = "expanded" if injected else "resolved_no_new_candidates"
        return injected, _block(
            status=status,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            projection_snapshot=routing_response.projection_snapshot,
            candidate_paths_considered=paths_considered,
            candidate_paths_used=paths_used,
            candidates_added=candidates_added,
            candidates_deduplicated=candidates_deduplicated,
            candidates_rejected_by_policy=candidates_rejected_by_policy,
            reason_codes=reason_codes,
        )

    def _resolve_candidate(
        self,
        *,
        retrieval_router: Any,
        query: str,
        source_uri: str,
        filters: Optional[Dict[str, Any]],
        retrieval_mode: str,
    ) -> Optional[SearchResult]:
        """Resolve a candidate source_uri to a real, already-indexed engram.

        No "fetch by source_uri" primitive exists on the router or tiers
        (only fetch-by-engram-id); this reuses the router's existing
        metadata-filtered search, narrowed to top_k=1, rather than adding a
        new tier method. See docs/associative_routing_e2_design_note.md for
        why this is the chosen approach and its latency tradeoff.
        """
        try:
            lookup_filters = dict(filters or {})
            lookup_filters["metadata.source_uri"] = source_uri
            results, _meta = retrieval_router.search(
                query=query,
                top_k=1,
                filters=lookup_filters,
                retrieval_mode=retrieval_mode or "semantic",
            )
        except Exception:
            logger.exception("Associative expansion candidate resolution failed for %s", source_uri)
            return None

        for result in results:
            if _extract_source_uri(result) == source_uri:
                return result
        return None


#: Process-wide singleton reused by the REST search handler; the projection is
#: built once (lazily, on first use) and is immutable thereafter.
default_engine = CandidateExpansionEngine()
