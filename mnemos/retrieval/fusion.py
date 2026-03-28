"""
Multi-Tier Fusion
==================

Merges results from multiple retrieval tiers with configurable
weighting and deduplication.
"""

import logging
from typing import Any, Dict, List, Optional

from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class TierFusion:
    """
    Fuses results from multiple retrieval tiers.

    Supports:
    - Weighted score combination
    - Deduplication across tiers
    - Per-tier metadata in results
    """

    def __init__(self, tiers: List[BaseRetriever],
                 weights: Optional[Dict[str, float]] = None):
        """
        Args:
            tiers: List of active retrieval tiers.
            weights: Optional tier weights (e.g. {"qdrant": 1.0, "colbert": 1.5}).
                     If not specified, all tiers weighted equally.
        """
        self._tiers = tiers
        self._weights = weights or {t.tier_name: 1.0 for t in tiers}

    @property
    def tier_names(self) -> List[str]:
        return [t.tier_name for t in self._tiers]

    def search(self, query: str, top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None,
               tiers: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Search across all active tiers and fuse results.

        Args:
            query: Search query text.
            top_k: Maximum total results (after fusion).
            filters: Optional metadata filters (passed to each tier).
            tiers: Optional list of tier names to query (subset of active tiers).

        Returns:
            Fused, deduplicated, ranked list of SearchResult.
        """
        active_tiers = self._tiers
        if tiers:
            active_tiers = [t for t in self._tiers if t.tier_name in tiers]

        # Collect results from each tier
        all_results: Dict[str, SearchResult] = {}  # deduplicate by engram ID
        tier_contributions: Dict[str, int] = {}

        for tier in active_tiers:
            try:
                results = tier.search(query, top_k=top_k, filters=filters)
                weight = self._weights.get(tier.tier_name, 1.0)
                tier_contributions[tier.tier_name] = len(results)

                for r in results:
                    weighted_score = r.score * weight

                    if r.engram.id in all_results:
                        # Combine scores from multiple tiers
                        existing = all_results[r.engram.id]
                        existing.score += weighted_score
                        existing.metadata["tiers"] = existing.metadata.get("tiers", []) + [tier.tier_name]
                    else:
                        r.score = weighted_score
                        r.metadata["tiers"] = [tier.tier_name]
                        all_results[r.engram.id] = r

            except Exception as e:
                logger.error(f"Tier {tier.tier_name} search failed: {e}")
                tier_contributions[tier.tier_name] = 0

        # Sort by fused score
        fused = sorted(all_results.values(), key=lambda r: r.score, reverse=True)

        logger.debug(
            f"Fusion: {sum(tier_contributions.values())} raw results "
            f"→ {len(fused)} unique → returning top {top_k} "
            f"(contributions: {tier_contributions})"
        )

        return fused[:top_k]

    def index(self, engrams, tiers: Optional[List[str]] = None) -> Dict[str, int]:
        """Index engrams across all (or specified) tiers."""
        active_tiers = self._tiers
        if tiers:
            active_tiers = [t for t in self._tiers if t.tier_name in tiers]

        counts = {}
        for tier in active_tiers:
            try:
                counts[tier.tier_name] = tier.index(engrams)
            except Exception as e:
                logger.error(f"Tier {tier.tier_name} indexing failed: {e}")
                counts[tier.tier_name] = 0
        return counts

    def delete(self, engram_ids: List[str]) -> Dict[str, int]:
        """Delete engrams from all tiers."""
        counts = {}
        for tier in self._tiers:
            try:
                counts[tier.tier_name] = tier.delete(engram_ids)
            except Exception as e:
                logger.error(f"Tier {tier.tier_name} delete failed: {e}")
                counts[tier.tier_name] = 0
        return counts

    def stats(self) -> Dict[str, Any]:
        """Get statistics from all tiers."""
        return {
            "active_tiers": self.tier_names,
            "weights": self._weights,
            "tiers": {
                tier.tier_name: tier.stats() for tier in self._tiers
            },
        }
