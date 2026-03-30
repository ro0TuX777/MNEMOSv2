"""
Retrieval mode router for semantic-only and hybrid search.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.retrieval.candidate_envelope import (
    CandidateEnvelopeConfig,
    apply_candidate_envelope,
)
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.hybrid_fusion import HybridFusion
from mnemos.retrieval.policies.fusion_policies import DEFAULT_FUSION_POLICY, FUSION_POLICIES


class RetrievalRouter:
    """Routes search requests between semantic and hybrid retrieval modes."""

    def __init__(
        self,
        *,
        semantic_fusion: TierFusion,
        lexical_tier: Optional[BaseRetriever] = None,
    ):
        self._semantic_fusion = semantic_fusion
        self._lexical_tier = lexical_tier
        self._hybrid_fusion = HybridFusion()
        self._stats = {
            "hybrid_query_count": 0,
            "semantic_query_count": 0,
            "retrieval_mode_counters": {"semantic": 0, "hybrid": 0},
            "fusion_policy_counters": {
                "semantic_dominant": 0,
                "balanced": 0,
                "lexical_dominant": 0,
            },
            "hybrid_lexical_only_contribution_rate": 0.0,
            "hybrid_semantic_only_contribution_rate": 0.0,
            "hybrid_dual_hit_overlap_rate": 0.0,
            "hybrid_avg_fused_candidate_pool_size": 0.0,
            "hybrid_latency_p50_ms": 0.0,
            "hybrid_latency_p95_ms": 0.0,
            "hybrid_last_policy": DEFAULT_FUSION_POLICY,
            "hybrid_last_telemetry": {},
            "hybrid_available": bool(lexical_tier),
            "candidate_pool_raw_count": 0,
            "candidate_pool_narrowed_count": 0,
            "candidate_duplicate_suppressed_count": 0,
            "candidate_source_cap_applied_count": 0,
        }
        self._hybrid_latencies_ms: List[float] = []

    def _record_candidate_envelope_stats(self, envelope_meta: Dict[str, Any]) -> None:
        self._stats["candidate_pool_raw_count"] += int(
            envelope_meta.get("initial_candidate_count", 0)
        )
        self._stats["candidate_pool_narrowed_count"] += int(
            envelope_meta.get("final_candidate_count", 0)
        )
        summary = envelope_meta.get("suppression_summary", {}) or {}
        self._stats["candidate_duplicate_suppressed_count"] += int(
            summary.get("duplicate_similarity", 0)
        )
        self._stats["candidate_source_cap_applied_count"] += int(
            summary.get("source_cap_exceeded", 0)
        )

    @property
    def semantic_tiers(self) -> List[str]:
        return self._semantic_fusion.tier_names

    @property
    def lexical_available(self) -> bool:
        return self._lexical_tier is not None

    def _record_hybrid_stats(self, telemetry: Dict[str, float], elapsed_ms: float, policy: str):
        self._stats["hybrid_query_count"] += 1
        self._stats["retrieval_mode_counters"]["hybrid"] += 1
        self._stats["fusion_policy_counters"][policy] += 1
        self._stats["hybrid_last_policy"] = policy
        self._stats["hybrid_last_telemetry"] = {
            "lexical_candidates": telemetry.get("lexical_candidates", 0.0),
            "semantic_candidates": telemetry.get("semantic_candidates", 0.0),
            "overlap_candidates": telemetry.get("overlap_candidates", 0.0),
            "union_candidates": telemetry.get("union_candidates", 0.0),
            "lexical_only_candidates": telemetry.get("lexical_only_candidates", 0.0),
            "semantic_only_candidates": telemetry.get("semantic_only_candidates", 0.0),
        }

        union = telemetry.get("union_candidates", 0.0)
        lexical_only = telemetry.get("lexical_only_candidates", 0.0)
        semantic_only = telemetry.get("semantic_only_candidates", 0.0)
        overlap = telemetry.get("overlap_candidates", 0.0)

        if union > 0:
            self._stats["hybrid_lexical_only_contribution_rate"] = round(lexical_only / union, 4)
            self._stats["hybrid_semantic_only_contribution_rate"] = round(semantic_only / union, 4)
            self._stats["hybrid_dual_hit_overlap_rate"] = round(overlap / union, 4)
        self._stats["hybrid_avg_fused_candidate_pool_size"] = round(union, 2)

        self._hybrid_latencies_ms.append(elapsed_ms)
        self._hybrid_latencies_ms = self._hybrid_latencies_ms[-200:]

        sorted_lat = sorted(self._hybrid_latencies_ms)
        if sorted_lat:
            mid = len(sorted_lat) // 2
            p50 = sorted_lat[mid]
            p95 = sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))]
            self._stats["hybrid_latency_p50_ms"] = round(p50, 2)
            self._stats["hybrid_latency_p95_ms"] = round(p95, 2)

    def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        tiers: Optional[List[str]] = None,
        retrieval_mode: str = "semantic",
        fusion_policy: str = DEFAULT_FUSION_POLICY,
        explain: bool = False,
        lexical_top_k: int = 25,
        semantic_top_k: int = 25,
        bounded_envelope: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        mode = retrieval_mode if retrieval_mode in {"semantic", "hybrid"} else "semantic"
        policy = fusion_policy if fusion_policy in FUSION_POLICIES else DEFAULT_FUSION_POLICY
        envelope_cfg = CandidateEnvelopeConfig.from_request(bounded_envelope)
        desired_pool = max(top_k, envelope_cfg.candidate_pool_limit) if envelope_cfg.enabled else top_k

        if mode == "semantic" or not self._lexical_tier:
            self._stats["semantic_query_count"] += 1
            self._stats["retrieval_mode_counters"]["semantic"] += 1
            hits = self._semantic_fusion.search(query, top_k=desired_pool, filters=filters, tiers=tiers)
            narrowed, envelope_meta = apply_candidate_envelope(hits, envelope_cfg)
            self._record_candidate_envelope_stats(envelope_meta)
            return narrowed[:top_k], {
                "retrieval_mode": "semantic",
                "fusion_policy": None,
                "lexical_available": self.lexical_available,
                "candidate_envelope": envelope_meta,
            }

        start = time.perf_counter()

        lexical_results = self._lexical_tier.search(query, top_k=lexical_top_k, filters=filters)
        semantic_results = self._semantic_fusion.search(
            query,
            top_k=semantic_top_k,
            filters=filters,
            tiers=tiers,
        )

        fused, telemetry = self._hybrid_fusion.fuse(
            lexical_results=lexical_results,
            semantic_results=semantic_results,
            top_k=desired_pool,
            fusion_policy=policy,
            filters=filters,
            explain=explain,
        )
        narrowed, envelope_meta = apply_candidate_envelope(fused, envelope_cfg)
        self._record_candidate_envelope_stats(envelope_meta)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._record_hybrid_stats(telemetry, elapsed_ms, policy)

        meta = {
            "retrieval_mode": "hybrid",
            "fusion_policy": policy,
            "lexical_available": self.lexical_available,
            "telemetry": telemetry,
            "candidate_envelope": envelope_meta,
        }
        return narrowed[:top_k], meta

    def stats(self) -> Dict[str, Any]:
        out = dict(self._stats)
        out["supported_retrieval_modes"] = ["semantic", "hybrid"]
        out["supported_fusion_policies"] = sorted(FUSION_POLICIES.keys())
        return out
