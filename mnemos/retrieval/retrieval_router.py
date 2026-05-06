"""
Retrieval mode router for semantic-only and hybrid search.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.retrieval.candidate_envelope import (
    CandidateEnvelopeConfig,
    apply_candidate_envelope,
)
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.hybrid_fusion import HybridFusion
from mnemos.retrieval.qdrant_hybrid import QdrantHybridFusion
from mnemos.retrieval.policies.fusion_policies import DEFAULT_FUSION_POLICY, FUSION_POLICIES
from mnemos.retrieval.policies.rerank_policy import RerankPolicy
from mnemos.retrieval.policies.query_classifier import get_classifier
from mnemos.retrieval.telemetry import get_telemetry_sink
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RetrievalRouter:
    """Routes search requests between semantic and hybrid retrieval modes."""

    def __init__(
        self,
        *,
        semantic_fusion: TierFusion,
        lexical_tier: Optional[BaseRetriever] = None,
        reranker: Optional[Any] = None,
    ):
        self._semantic_fusion = semantic_fusion
        self._lexical_tier = lexical_tier
        self._reranker = reranker
        self._rerank_policy = RerankPolicy()
        self._classifier = get_classifier(self._rerank_policy.config.get("query_family_classifier", {}))
        self._telemetry_sink = get_telemetry_sink(self._rerank_policy.config.get("telemetry", {}))
        self._hybrid_fusion = HybridFusion()

        # Qdrant-native hybrid RRF — initialised lazily from the first
        # QdrantTier found in the semantic fusion tiers.
        self._qdrant_hybrid: Optional[QdrantHybridFusion] = None
        self._init_qdrant_hybrid(semantic_fusion)
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

    def _init_qdrant_hybrid(self, semantic_fusion: TierFusion) -> None:
        """Try to find a QdrantTier with text_index_ready and wrap it."""
        try:
            from mnemos.retrieval.qdrant_tier import QdrantTier
            for tier in semantic_fusion._tiers:
                if isinstance(tier, QdrantTier) and getattr(tier, "_text_index_ready", False):
                    self._qdrant_hybrid = QdrantHybridFusion(tier)
                    logger.info("QdrantHybridFusion enabled (qdrant_rrf fusion policy available)")
                    return
        except Exception:
            pass
        logger.debug("QdrantHybridFusion not available — will use Python-side hybrid only")

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

    def _apply_conditional_rerank(self, query: str, candidates: List[SearchResult], dense_latency_ms: float) -> Tuple[List[SearchResult], Dict[str, Any]]:
        telemetry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": str(uuid.uuid4()),
            "query_hash": str(hash(query)),
            "query_family": "unknown",
            "query_family_confidence": 0.0,
            "rerank_eligible": False,
            "rerank_applied": False,
            "shadow_evaluated": False,
            "rerank_model": self._reranker.model_name if getattr(self._reranker, "model_name", None) else "none",
            "rerank_depth": 0,
            "rerank_skip_reason": None,
            "dense_latency_ms": dense_latency_ms,
            "rerank_latency_ms": 0.0,
            "total_latency_ms": dense_latency_ms,
            "circuit_breaker_state": self._rerank_policy.circuit_breaker.state,
            "dense_top_k": len(candidates),
            "dense_top1_doc_id": candidates[0].engram.id if candidates else None,
            "final_top1_doc_id": candidates[0].engram.id if candidates else None,
            "timeout_occurred": False,
            "error_occurred": False,
            "service_healthy": False
        }
        
        family, confidence = self._classifier.classify(query)
        telemetry["query_family"] = family
        telemetry["query_family_confidence"] = confidence
        
        # Fast exit if no reranker
        if self._reranker is None:
            telemetry["rerank_skip_reason"] = "no_reranker_configured"
            if self._rerank_policy.config.get("telemetry", {}).get("persist_enabled", True):
                self._telemetry_sink.emit(telemetry)
            return candidates, telemetry
            
        health = self._reranker.health() if hasattr(self._reranker, "health") else {"healthy": True}
        service_healthy = bool(health.get("healthy"))
        telemetry["service_healthy"] = service_healthy
        telemetry["rerank_service_health"] = health
        
        eligibility = self._rerank_policy.is_eligible(family, len(candidates), service_healthy)
        telemetry["rerank_eligible"] = eligibility["eligible"]
        
        if not eligibility["eligible"]:
            if not self._rerank_policy.should_shadow_execute(eligibility["skip_reason"]):
                telemetry["rerank_skip_reason"] = eligibility["skip_reason"]
                if self._rerank_policy.config.get("telemetry", {}).get("persist_enabled", True):
                    self._telemetry_sink.emit(telemetry)
                return candidates, telemetry
            
        depth = self._rerank_policy.get_depth(family)
        telemetry["rerank_depth"] = depth
        
        candidates_to_rerank = candidates[:depth]
        telemetry["rerank_skip_reason"] = eligibility["skip_reason"]  # May be set by shadow mode
        
        t0 = time.perf_counter()
        
        try:
            # Re-initialize or check actual health
            if getattr(self._reranker, "_initialize", None):
                 self._reranker._initialize()
                 if self._reranker._model is None:
                      raise RuntimeError("Model unavailable")
                      
            reranked = self._reranker.rerank(query, candidates_to_rerank)
            t1 = time.perf_counter()
            rerank_lat = (t1 - t0) * 1000
            
            telemetry["rerank_latency_ms"] = rerank_lat
            telemetry["total_latency_ms"] = dense_latency_ms + rerank_lat
            self._rerank_policy.record_latency(family, rerank_lat)
            self._rerank_policy.circuit_breaker.record_success()
            
            # Form final array
            final_results = reranked + candidates[depth:]
            
            # Shadow mode check
            if not eligibility["eligible"] and self._rerank_policy.should_shadow_execute(eligibility["skip_reason"]):
                telemetry["shadow_evaluated"] = True
                telemetry["top3_changed"] = bool({r.engram.id for r in candidates[:3]} != {r.engram.id for r in final_results[:3]})
                telemetry["top10_changed"] = bool({r.engram.id for r in candidates[:10]} != {r.engram.id for r in final_results[:10]})
                
                if self._rerank_policy.config.get("telemetry", {}).get("persist_enabled", True):
                    self._telemetry_sink.emit(telemetry)
                # Discard ordering, return original
                return candidates, telemetry
                
            telemetry["rerank_applied"] = True
            telemetry["final_top1_doc_id"] = final_results[0].engram.id if final_results else None
            
            top3_original = {r.engram.id for r in candidates[:3]}
            top3_new = {r.engram.id for r in final_results[:3]}
            telemetry["top3_changed"] = bool(top3_original != top3_new)
            
            top10_orig = {r.engram.id for r in candidates[:10]}
            top10_new = {r.engram.id for r in final_results[:10]}
            telemetry["top10_changed"] = bool(top10_orig != top10_new)

            if self._rerank_policy.config.get("telemetry", {}).get("persist_enabled", True):
                self._telemetry_sink.emit(telemetry)

            return final_results, telemetry
            
        except Exception as e:
            err_msg = str(e).lower()
            if "timeout" in err_msg:
                 telemetry["timeout_occurred"] = True
                 telemetry["rerank_skip_reason"] = "timeout"
                 self._rerank_policy.circuit_breaker.record_timeout()
            else:
                 telemetry["error_occurred"] = True
                 telemetry["rerank_skip_reason"] = "error"
                 self._rerank_policy.circuit_breaker.record_error()
                 
            telemetry["rerank_applied"] = False
            telemetry["shadow_evaluated"] = False
            
            if self._rerank_policy.config.get("telemetry", {}).get("persist_enabled", True):
                self._telemetry_sink.emit(telemetry)
                
            return candidates, telemetry

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
            t_s = time.perf_counter()
            self._stats["semantic_query_count"] += 1
            self._stats["retrieval_mode_counters"]["semantic"] += 1
            hits = self._semantic_fusion.search(query, top_k=desired_pool, filters=filters, tiers=tiers)
            t_e = (time.perf_counter() - t_s) * 1000
            
            hits, rr_telemetry = self._apply_conditional_rerank(query, hits, t_e)

            narrowed, envelope_meta = apply_candidate_envelope(hits, envelope_cfg)
            self._record_candidate_envelope_stats(envelope_meta)
            return narrowed[:top_k], {
                "retrieval_mode": "semantic",
                "fusion_policy": None,
                "lexical_available": self.lexical_available,
                "candidate_envelope": envelope_meta,
                "reranker_used": self._reranker is not None,
                "rerank_telemetry": rr_telemetry,
            }

        start = time.perf_counter()

        # ── Qdrant-native RRF path ──────────────────────────────
        if policy == "qdrant_rrf" and self._qdrant_hybrid and self._qdrant_hybrid.available:
            try:
                # Compute embedding once for the RRF call
                query_vec = self._qdrant_hybrid._tier._embed([query])[0]
                fused, telemetry = self._qdrant_hybrid.fuse(
                    query=query,
                    query_vector=query_vec,
                    top_k=desired_pool,
                    filters=filters,
                    semantic_limit=semantic_top_k,
                    lexical_limit=lexical_top_k,
                )
            except Exception as e:
                logger.warning(f"QdrantHybridFusion failed, falling back to Python hybrid: {e}")
                # Fall through to Python-side fusion below
                fused = None
                telemetry = None

            if fused is not None:
                fused, rr_telemetry = self._apply_conditional_rerank(
                    query, fused, (time.perf_counter() - start) * 1000
                )

                narrowed, envelope_meta = apply_candidate_envelope(fused, envelope_cfg)
                self._record_candidate_envelope_stats(envelope_meta)

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._record_hybrid_stats(telemetry, elapsed_ms, policy)

                meta = {
                    "retrieval_mode": "hybrid",
                    "fusion_policy": policy,
                    "fusion_engine": "qdrant_rrf",
                    "lexical_available": self.lexical_available,
                    "telemetry": telemetry,
                    "candidate_envelope": envelope_meta,
                    "reranker_used": self._reranker is not None,
                    "rerank_telemetry": rr_telemetry,
                }
                return narrowed[:top_k], meta

        # ── Python-side hybrid fusion (original path) ──────────
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

        fused, rr_telemetry = self._apply_conditional_rerank(query, fused, (time.perf_counter() - start) * 1000)

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
            "reranker_used": self._reranker is not None,
            "rerank_telemetry": rr_telemetry,
        }
        return narrowed[:top_k], meta

    def stats(self) -> Dict[str, Any]:
        out = dict(self._stats)
        raw_total = max(1, int(out.get("candidate_pool_raw_count", 0)))
        narrowed_total = int(out.get("candidate_pool_narrowed_count", 0))
        out["candidate_envelope_avg_compression_ratio"] = round(
            narrowed_total / raw_total, 4
        )
        out["candidate_envelope_total_reduction"] = max(0, raw_total - narrowed_total)
        out["supported_retrieval_modes"] = ["semantic", "hybrid"]
        out["supported_fusion_policies"] = sorted(FUSION_POLICIES.keys())
        return out
