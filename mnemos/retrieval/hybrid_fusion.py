"""
Hybrid lexical + semantic fusion engine.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.policies.fusion_policies import (
    DEFAULT_FUSION_POLICY,
    FUSION_POLICIES,
)


def _normalize_rank(results: List[SearchResult]) -> Dict[str, float]:
    """Rank-based normalization to [0, 1]."""
    n = len(results)
    if n <= 0:
        return {}
    if n == 1:
        return {results[0].engram.id: 1.0}

    normalized: Dict[str, float] = {}
    for idx, r in enumerate(results):
        normalized[r.engram.id] = (n - idx) / n
    return normalized


class HybridFusion:
    """Combine lexical and semantic candidates into fused ranked results."""

    def fuse(
        self,
        *,
        lexical_results: List[SearchResult],
        semantic_results: List[SearchResult],
        top_k: int,
        fusion_policy: str = DEFAULT_FUSION_POLICY,
        filters: Optional[Dict] = None,
        explain: bool = False,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        policy = fusion_policy if fusion_policy in FUSION_POLICIES else DEFAULT_FUSION_POLICY
        weights = FUSION_POLICIES[policy]

        lexical_norm = _normalize_rank(lexical_results)
        semantic_norm = _normalize_rank(semantic_results)

        by_id: Dict[str, SearchResult] = {}

        for r in lexical_results:
            by_id[r.engram.id] = SearchResult(
                engram=r.engram,
                score=0.0,
                tier="hybrid",
                metadata={
                    "component_scores": {
                        "lexical": 0.0,
                        "semantic": 0.0,
                        "fused": 0.0,
                    },
                    "retrieval_sources": [],
                },
            )

        for r in semantic_results:
            if r.engram.id not in by_id:
                by_id[r.engram.id] = SearchResult(
                    engram=r.engram,
                    score=0.0,
                    tier="hybrid",
                    metadata={
                        "component_scores": {
                            "lexical": 0.0,
                            "semantic": 0.0,
                            "fused": 0.0,
                        },
                        "retrieval_sources": [],
                    },
                )

        for engram_id, hit in by_id.items():
            lex = lexical_norm.get(engram_id, 0.0)
            sem = semantic_norm.get(engram_id, 0.0)
            fused = (weights["lexical"] * lex) + (weights["semantic"] * sem)

            hit.score = fused
            hit.metadata["component_scores"] = {
                "lexical": round(lex, 6),
                "semantic": round(sem, 6),
                "fused": round(fused, 6),
            }

            sources: List[str] = []
            if lex > 0:
                sources.append("lexical")
            if sem > 0:
                sources.append("semantic")
            hit.metadata["retrieval_sources"] = sources
            hit.metadata["fusion_policy"] = policy
            if filters:
                hit.metadata["filters_applied"] = filters

            if not explain:
                # Keep payload minimal when explain is disabled.
                hit.metadata.pop("component_scores", None)
                hit.metadata.pop("filters_applied", None)
                hit.metadata.pop("fusion_policy", None)

        ranked = sorted(
            by_id.values(),
            key=lambda r: (r.score, r.engram.id),
            reverse=True,
        )[:top_k]

        overlap_ids = set(lexical_norm.keys()).intersection(semantic_norm.keys())
        telemetry = {
            "lexical_candidates": float(len(lexical_results)),
            "semantic_candidates": float(len(semantic_results)),
            "union_candidates": float(len(by_id)),
            "overlap_candidates": float(len(overlap_ids)),
            "lexical_only_candidates": float(len(set(lexical_norm.keys()) - set(semantic_norm.keys()))),
            "semantic_only_candidates": float(len(set(semantic_norm.keys()) - set(lexical_norm.keys()))),
        }

        return ranked, telemetry
