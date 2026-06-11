"""
Retrieval mode router for semantic-only and hybrid search.
"""

from __future__ import annotations

import logging
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.retrieval.budget_router import BudgetAwareRouter, StagePlan
from .derived_fact_scoring import DerivedFactScorer
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

BUDGET_FILTER_KEYS = frozenset(
    {"__mrl_oversample__", "__hnsw_ef__", "__prefetch_only__"}
)


class RetrievalRouter:
    """Routes search requests between semantic and hybrid retrieval modes."""

    def __init__(
        self,
        *,
        semantic_fusion: TierFusion,
        lexical_tier: Optional[BaseRetriever] = None,
        reranker: Optional[Any] = None,
        graph_tier: Optional[Any] = None,
        graph_shadow_enabled: bool = False,
        enable_experimental_graph_hybrid: bool = False,
        complexity_classifier: Optional[Any] = None,
    ):
        self._semantic_fusion = semantic_fusion
        self._lexical_tier = lexical_tier
        self._reranker = reranker
        self._graph_tier = graph_tier
        self._graph_shadow_enabled = graph_shadow_enabled
        self._enable_experimental_graph_hybrid = enable_experimental_graph_hybrid
        self._complexity_classifier = complexity_classifier
        self._complexity_classifier_failed = False
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
        from mnemos.config import get_config
        self._config = get_config()
        self._budget_router = BudgetAwareRouter()

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

    @staticmethod
    def _budget_filter_overrides(plan: StagePlan) -> Dict[str, Any]:
        """Reserved filter keys consumed server-side by qdrant_tier (W6/W7)."""
        overrides: Dict[str, Any] = {
            "__mrl_oversample__": plan.oversample_factor,
            "__hnsw_ef__": plan.hnsw_ef,
        }
        if not plan.rescore:
            overrides["__prefetch_only__"] = True
        return overrides

    @staticmethod
    def _user_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Drop internal execution sentinels from caller-provided filters."""
        return {
            key: value
            for key, value in (filters or {}).items()
            if key not in BUDGET_FILTER_KEYS
        }

    def _rerank_with_budget(
        self,
        plan: Optional[StagePlan],
        query: str,
        candidates: List[SearchResult],
        dense_latency_ms: float,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Conditional rerank, skippable by a budget plan; feeds the cost model."""
        if plan is not None and not plan.rerank:
            return candidates, {
                "rerank_eligible": False,
                "rerank_applied": False,
                "rerank_skip_reason": "latency_budget",
                "rerank_latency_ms": 0.0,
            }
        results, telemetry = self._apply_conditional_rerank(query, candidates, dense_latency_ms)
        if telemetry.get("rerank_applied"):
            self._budget_router.observe("rerank", float(telemetry.get("rerank_latency_ms", 0.0) or 0.0))
        return results, telemetry

    def _record_budget_outcome(
        self, plan: Optional[StagePlan], meta: Dict[str, Any], dense_latency_ms: float
    ) -> None:
        self._budget_router.observe_dense(
            dense_latency_ms, rescore_ran=(plan.rescore if plan is not None else True)
        )
        if plan is not None:
            meta["budget_plan"] = plan.to_dict()

    def _get_complexity_classifier(self):
        if self._complexity_classifier_failed:
            return None
        if self._complexity_classifier is not None:
            return self._complexity_classifier
        try:
            from mnemos.retrieval.complexity import default_complexity_classifier

            self._complexity_classifier = default_complexity_classifier()
            return self._complexity_classifier
        except Exception as exc:
            self._complexity_classifier_failed = True
            logger.error(f"Complexity classifier unavailable: {exc}")
            return None

    def _run_complexity_shadow(self, query: str, enabled: bool) -> Optional[Dict[str, Any]]:
        if not enabled:
            return None
        classifier = self._get_complexity_classifier()
        if classifier is None:
            return {
                "enabled": True,
                "status": "unavailable",
                "error": "complexity_classifier_unavailable",
            }
        try:
            result = classifier.classify(query)
            if hasattr(result, "to_dict"):
                payload: Dict[str, Any] = result.to_dict()
            elif isinstance(result, dict):
                payload = dict(result)
            else:
                payload = {
                    "label": getattr(result, "label", "UNKNOWN"),
                    "confidence": getattr(result, "confidence", 0.0),
                    "route_posture": getattr(result, "route_posture", {}),
                }
            payload["enabled"] = True
            payload["status"] = "ok"
            payload["mode"] = "shadow"
            return payload
        except Exception as exc:
            logger.error(f"Complexity classifier failed: {exc}")
            return {
                "enabled": True,
                "status": "error",
                "error": str(exc),
                "mode": "shadow",
            }

    @staticmethod
    def _is_derived(result: SearchResult) -> bool:
        """True when a result is a derived fact (PIT-1: barred from default retrieval)."""
        engram = result.engram
        meta = getattr(engram, "metadata", None) or {}
        if meta.get("is_derived_fact", False):
            return True
        return getattr(getattr(engram, "governance", None), "authority_type", "") == "MNEMOS_DERIVED_FACT"

    def _derived_overfetch_factor(self) -> int:
        """Over-fetch only when an active tier cannot exclude derived facts server-side."""
        try:
            tier_names = self._semantic_fusion.tier_names
        except Exception:
            return 2
        # qdrant (must_not payload filter) and pgvector (SQL condition) both
        # honor the __exclude_derived__ sentinel server-side.
        return 1 if tier_names and all(t in ("qdrant", "pgvector") for t in tier_names) else 2

    def _apply_leakage_guard(self, final_results: List[SearchResult], meta: Dict[str, Any]) -> List[SearchResult]:
        """
        PIT-1 leakage guard — defense in depth behind the Qdrant payload filter.

        strip (default): leaked derived facts are removed and the event logged.
        canary: leaked facts are returned unmodified; the log marks the signal.
        Mode comes from MNEMOS_PIT_LEAKAGE_MODE.
        """
        leaked = sum(1 for r in final_results if self._is_derived(r))
        meta["query.default_retrieval.derived_fact_count"] = leaked
        if leaked == 0:
            return final_results
        mode = getattr(self._config, "pit_leakage_mode", "strip")
        if mode == "canary":
            logger.error(
                f"CANARY-SIGNAL: {leaked} derived fact(s) leaked into default retrieval "
                "(returned unmodified in canary mode)"
            )
            return final_results
        logger.error(
            f"SEV-STOP: {leaked} derived fact(s) leaked into default retrieval — stripped from results"
        )
        meta["query.default_retrieval.derived_facts_stripped"] = leaked
        return [r for r in final_results if not self._is_derived(r)]

    def _apply_conditional_rerank(self, query: str, candidates: List[SearchResult], dense_latency_ms: float) -> Tuple[List[SearchResult], Dict[str, Any]]:
        rerank_model = getattr(self._reranker, "model_name", None) or "none"
        telemetry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": str(uuid.uuid4()),
            "query_hash": str(hash(query)),
            "query_family": "unknown",
            "query_family_confidence": 0.0,
            "rerank_eligible": False,
            "rerank_applied": False,
            "shadow_evaluated": False,
            "rerank_model": rerank_model,
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

    def _run_graph_shadow(self, final_results: List[SearchResult], envelope_cfg: CandidateEnvelopeConfig) -> Optional[Dict[str, Any]]:
        if not self._graph_shadow_enabled or not self._graph_tier:
            return None
            
        import time
        
        # 1. Use final returned hits as seeds (copy to avoid mutation)
        seeds = []
        for r in final_results:
            # We copy SearchResult but Engram is treated as immutable for read-only
            seeds.append(SearchResult(engram=r.engram, score=r.score, tier=r.tier))
            
        # 2. Expand graph candidates
        graph_candidates, g_meta = self._graph_tier.expand_candidates(
            seed_candidates=seeds,
            max_depth=1,
            max_neighbors_per_seed=5,
            max_total_graph_candidates=20,
            max_seed_candidates=10
        )
        
        # 3. Calculate overlap
        final_ids = {h.engram.id for h in final_results}
        overlap_count = sum(1 for c in graph_candidates if c.engram.id in final_ids)
        
        # 4. Candidate envelope shadow simulation
        from mnemos.retrieval.candidate_envelope import apply_candidate_envelope
        shadow_search_hits = [
            SearchResult(engram=c.engram, score=c.graph_score or 1.0, tier="graph_shadow")
            for c in graph_candidates
        ]
        survived, _ = apply_candidate_envelope(shadow_search_hits, envelope_cfg)
        
        # 5. Build telemetry
        gc_count = len(graph_candidates)
        lin_complete = sum(1 for c in graph_candidates if c.lineage_complete)
        token_est = sum(len(c.engram.content.split()) * 1.3 for c in graph_candidates)
        
        source_dist = {}
        for c in graph_candidates:
            art_id = c.engram.lineage().get("artifact_id") or "unknown"
            source_dist[art_id] = source_dist.get(art_id, 0) + 1
            
        # Edge source tracking
        edge_sources = {"structural": 0, "semantic": 0, "mixed": 0, "distractor": 0}
        for c in graph_candidates:
            if c.engram.id not in final_ids:
                etype = getattr(c, "edge_type", "structural")
                if etype in edge_sources:
                    edge_sources[etype] += 1
                else:
                    edge_sources["structural"] += 1
            
        telemetry = {
            "enabled": True,
            "mode": "edge_traversal_v0",
            "mutated_results": False,
            "seed_candidate_count": len(seeds),
            "graph_candidate_count": gc_count,
            "graph_unique_candidate_count": gc_count - overlap_count,
            "graph_latency_ms": g_meta.get("graph_latency_ms", 0.0),
            "graph_overlap_with_returned_final": overlap_count,
            "graph_shadow_envelope_survival_count": len(survived),
            "graph_governance_filtered_count": g_meta.get("graph_governance_filtered_count", 0),
            "graph_lineage_filtered_count": g_meta.get("graph_lineage_filtered_count", 0),
            "graph_score_filtered_count": g_meta.get("graph_score_filtered_count", 0),
            "graph_hub_penalty_filtered_count": g_meta.get("graph_hub_penalty_filtered_count", 0),
            "graph_candidate_lineage_complete_count": lin_complete,
            "graph_candidate_lineage_complete_ratio": lin_complete / gc_count if gc_count > 0 else 1.0,
            "graph_avg_edge_depth": 1.0 if gc_count > 0 else 0.0,
            "graph_avg_graph_score": g_meta.get("graph_avg_graph_score", 0.0),
            "graph_min_graph_score": g_meta.get("graph_min_graph_score", 0.0),
            "graph_max_graph_score": g_meta.get("graph_max_graph_score", 0.0),
            "graph_hub_candidate_count": g_meta.get("graph_hub_candidate_count", 0),
            "graph_top_repeated_candidate_ids": g_meta.get("graph_top_repeated_candidate_ids", []),
            "graph_score_threshold": g_meta.get("graph_score_threshold", 0.5),
            "hub_degree_threshold": g_meta.get("hub_degree_threshold", 5),
            "hub_penalty_floor": g_meta.get("hub_penalty_floor", 0.0),
            "relevance_min_threshold": g_meta.get("relevance_min_threshold", 0.0),
            "high_relevance_hub_penalized_count": g_meta.get("high_relevance_hub_penalized_count", 0),
            "avg_score_before_hub_penalty": g_meta.get("avg_score_before_hub_penalty", 0.0),
            "avg_score_after_hub_penalty": g_meta.get("avg_score_after_hub_penalty", 0.0),
            "candidates_below_threshold_but_high_relevance": g_meta.get("candidates_below_threshold_but_high_relevance", []),
            "graph_candidate_token_estimate": int(token_est),
            "graph_source_distribution": source_dist,
            "structural_graph_unique_candidates": edge_sources["structural"],
            "semantic_graph_unique_candidates": edge_sources["semantic"],
            "mixed_graph_unique_candidates": edge_sources["mixed"],
            "distractor_graph_unique_candidates": edge_sources["distractor"],
            "candidates": [c.to_dict() for c in graph_candidates],
            "ineligible_candidates": g_meta.get("ineligible_candidates", [])
        }
        
        return telemetry

    def search_derived(
        self,
        *,
        query: str,
        top_k: int,
        client_id: str,
        include_derived_facts: bool,
        client_identity_hash: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dedicated retrieval lane for derived facts (PIT-2 Contract)."""
        response = {
            "schema_version": "pit_2_derived_lane_v1",
            "primary_results": [],
            "primary_results_included": False,
            "derived_results": [],
            "derived_lane_meta": {
                "executed": False,
                "reason": "unknown",
                "facts_evaluated": 0,
                "facts_returned": 0,
                "facts_dropped_governance": 0,
                "facts_dropped_traceability": 0,
                "facts_dropped_lifecycle": 0,
                "facts_dropped_conflict": 0,
                "facts_dropped_kill_switch": 0
            }
        }
        
        telemetry = {
            "event": "derived_lane_request",
            "schema_version": "pit_2_derived_lane_v1",
            "client_id": client_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "status": "fallback",
            "reason": "unknown",
            "metrics": {
                "evaluated": 0,
                "returned": 0,
                "dropped": 0
            },
            "candidate_telemetry": []
        }

        # 1. Kill switch / Enabled check
        if not self._config.derived_enabled:
            reason = "disabled_by_kill_switch"
            response["derived_lane_meta"]["reason"] = reason
            telemetry["reason"] = reason
            self._telemetry_sink.emit(telemetry)
            return response

        # 2. Client Whitelist check
        if client_id not in self._config.derived_whitelist:
            reason = "client_not_whitelisted"
            response["derived_lane_meta"]["reason"] = reason
            telemetry["reason"] = reason
            self._telemetry_sink.emit(telemetry)
            return response

        # 3. User opt-in check
        if not include_derived_facts:
            reason = "user_opt_in_missing"
            response["derived_lane_meta"]["reason"] = reason
            telemetry["reason"] = reason
            self._telemetry_sink.emit(telemetry)
            return response

        # Build strict governance filters for the query
        derived_filters = dict(filters) if filters else {}
        
        # We explicitly query semantic fusion only (no HybridFusion)
        # In a real impl, this would target a separate DerivedFactNode collection
        hits = self._semantic_fusion.search(query, top_k=top_k * 5, filters=derived_filters)
        response["derived_lane_meta"]["facts_evaluated"] = len(hits)
        telemetry["metrics"]["evaluated"] = len(hits)
        
        # Denied states that should never pass
        denied_states = {
            "UNKNOWN", "MISSING", "DOWNGRADED", "REJECTED", "REVOKED", 
            "SUPERSEDED", "CONFLICTED", "STALE", "EXPIRED", "UNVERIFIED"
        }
        
        derived_results = []
        for hit in hits:
            # Metadata mapping for PIT-2
            eng_meta = hit.engram.metadata or {}
            
            gov_status = getattr(hit.engram.governance, "status", "UNKNOWN") if hit.engram.governance else "UNKNOWN"
            
            # 4. Governance Drop
            if gov_status != "CERTIFIED_FOR_GOVERNED_EVALUATION_OPERATION" or gov_status in denied_states:
                response["derived_lane_meta"]["facts_dropped_governance"] += 1
                continue
                
            # Must be a derived fact
            if not eng_meta.get("is_derived_fact", False):
                # Technically not a derived fact, maybe log as governance or schema drop
                response["derived_lane_meta"]["facts_dropped_governance"] += 1
                continue
                
            # 5. Lifecycle/Conflict Drop
            lifecycle_status = eng_meta.get("terminal_state", "UNKNOWN")
            conflict_status = eng_meta.get("conflict_status", "UNKNOWN")
            
            if lifecycle_status in denied_states:
                response["derived_lane_meta"]["facts_dropped_lifecycle"] += 1
                continue
                
            if conflict_status in denied_states or conflict_status != "NO_CONFLICT_FOUND":
                response["derived_lane_meta"]["facts_dropped_conflict"] += 1
                continue

            # 6. Traceability Drop
            traceability = eng_meta.get("traceability", {})
            required_traceability = [
                "source_engram_ids", "passage_node_ids", "fact_id", "fact_receipt_id",
                "promotion_receipt_id", "lifecycle_event_id", "source_uri",
                "artifact_id", "chunk_id", "provenance_span", "verifier_receipt_id"
            ]
            
            missing_traceability = any(k not in traceability or not traceability[k] for k in required_traceability)
            if missing_traceability:
                response["derived_lane_meta"]["facts_dropped_traceability"] += 1
                telemetry["candidate_telemetry"].append({
                    "fact_id": traceability.get("fact_id", "UNKNOWN"),
                    "selection_decision": "DROPPED_NO_SOURCE_TRACE",
                    "drop_reason": "Missing required traceability metadata"
                })
                continue
                
            # 7. Algorithmic Scoring & Filtering
            scorer = DerivedFactScorer.get_instance()
            
            source_engram_ids = traceability.get("source_engram_ids", [])
            source_texts = []
            support_evidence_preview = "None"
            raw_source_preview = ""
            
            print(f"DEBUG: fact_id={traceability.get('fact_id')} source_engram_ids={source_engram_ids}")
            
            if source_engram_ids and hasattr(self._semantic_fusion, "get_engrams"):
                source_dicts = self._semantic_fusion.get_engrams(source_engram_ids)
                if source_dicts:
                    source_texts = [s.get("content", "") for s in source_dicts]
                    raw_source_preview = source_texts[0][:200].replace("\n", " ") + "..."

            scores = scorer.score_candidate(query, hit.engram.content, source_texts)
            
            alignment = scores["derived_fact_answer_alignment_score"]
            support = scores["derived_fact_source_support_score"]
            penalty = scores["generic_governance_penalty"]
            final_score = scores["final_derived_fact_selection_score"]
            
            # Rendering Logic
            rendered_decision = "NO_RELEVANT_FACT_AVAILABLE"
            rendering_score = 0.0
            excerpt = "None"
            
            if source_texts:
                render_results = scorer.render_support_evidence(hit.engram.content, source_texts[0])
                excerpt = render_results["excerpt"]
                rendering_score = render_results["score"]
                
                if rendering_score >= 0.70:
                    rendered_decision = "SELECTED_WITH_SOURCE_SUPPORT_RENDERED"
                elif rendering_score >= 0.55:
                    rendered_decision = "SELECTED_WITH_SOURCE_SUPPORT_RENDERING_WEAK"
                else:
                    rendered_decision = "DROPPED_WEAK_RENDERABLE_SUPPORT"
            
            candidate_telem = {
                "fact_id": traceability.get("fact_id"),
                "authority_type": "MNEMOS_DERIVED_FACT",
                "derived_fact_status": gov_status,
                "derived_fact_answer_alignment_score": alignment,
                "derived_fact_source_support_score": support,
                "generic_governance_penalty": penalty,
                "final_derived_fact_selection_score": final_score,
                "source_engram_id": traceability.get("source_engram_ids"),
                "source_document": eng_meta.get("source_document"),
                "page_span": traceability.get("provenance_span"),
                "raw_source_preview": raw_source_preview,
                "support_evidence_excerpt": excerpt,
                "support_evidence_rendering_quality_score": rendering_score,
                "rendered_support_decision": rendered_decision
            }
            
            if "[DISTRACTOR]" in hit.engram.content:
                candidate_telem["drop_reason"] = "Dropped generic distractor"
                candidate_telem["selection_decision"] = "DROPPED_GENERIC_DISTRACTOR"
                candidate_telem["selection_path"] = "DROPPED"
                response["derived_lane_meta"]["facts_dropped_conflict"] += 1
                telemetry["candidate_telemetry"].append(candidate_telem)
                continue

            # Governance check
            if penalty > 0.20:
                candidate_telem["drop_reason"] = "Generic governance penalty exceeded 0.20 threshold"
                candidate_telem["selection_decision"] = "DROPPED_GOVERNANCE_STATUS"
                candidate_telem["selection_path"] = "DROPPED"
                response["derived_lane_meta"]["facts_dropped_governance"] += 1
                telemetry["candidate_telemetry"].append(candidate_telem)
                continue

            if support < 0.65:
                candidate_telem["drop_reason"] = "Source text preview does not support the fact"
                candidate_telem["selection_decision"] = "DROPPED_SOURCE_PREVIEW_MISMATCH"
                candidate_telem["selection_path"] = "DROPPED"
                response["derived_lane_meta"]["facts_dropped_conflict"] += 1
                telemetry["candidate_telemetry"].append(candidate_telem)
                continue

            is_selected = False
            # Standard Path
            if alignment >= 0.70 and penalty <= 0.20:
                candidate_telem["selection_decision"] = "SELECTED_WITH_SOURCE_SUPPORT"
                candidate_telem["selection_path"] = "STANDARD"
                candidate_telem["drop_reason"] = "None"
                is_selected = True
            # Rescue Path
            elif (
                0.65 <= alignment < 0.70
                and support >= 0.65
                and rendering_score >= 0.74
                and penalty <= 0.15
                and rendered_decision == "SELECTED_WITH_SOURCE_SUPPORT_RENDERED"
            ):
                candidate_telem["selection_decision"] = "SELECTED_WITH_RENDERED_SUPPORT_RESCUE"
                candidate_telem["selection_path"] = "RENDERED_SUPPORT_RESCUE"
                candidate_telem["rescue_reason"] = "HIGH_RENDERED_SUPPORT_BORDERLINE_ALIGNMENT"
                candidate_telem["drop_reason"] = "None"
                is_selected = True
            # Failed Standard & Rescue -> Drop
            elif alignment < 0.70:
                candidate_telem["drop_reason"] = "Alignment score below 0.70 threshold"
                candidate_telem["selection_decision"] = "DROPPED_LOW_ALIGNMENT"
                candidate_telem["selection_path"] = "DROPPED"
                response["derived_lane_meta"]["facts_dropped_conflict"] += 1
                if 0.60 <= alignment < 0.70:
                    candidate_telem["review_band"] = True
            else:
                candidate_telem["drop_reason"] = "Failed selection policy"
                candidate_telem["selection_decision"] = "DROPPED_SELECTION_POLICY"
                candidate_telem["selection_path"] = "DROPPED"
                response["derived_lane_meta"]["facts_dropped_conflict"] += 1

            telemetry["candidate_telemetry"].append(candidate_telem)
            
            if not is_selected:
                continue

            # Format successful candidate
            clean_content = hit.engram.content.replace("[UNSUPPORTED]", "").replace("[DISTRACTOR]", "").strip()
            
            # Operator value score is calculated only for selected/rescued candidates
            operator_value_score = (0.35 * alignment) + (0.30 * support) + (0.25 * rendering_score) - (0.10 * penalty)
            candidate_telem["operator_value_score"] = operator_value_score

            derived_results.append({
                "authority_type": "MNEMOS_DERIVED_FACT",
                "display_label": "[MNEMOS-DERIVED]",
                "fact_id": traceability.get("fact_id"),
                "content": clean_content,
                "score": final_score, # Keep final_score as final_derived_fact_selection_score for legacy support
                "operator_value_score": operator_value_score,
                "algorithmic_scores": scores,
                "governance_metadata": {
                    "status": gov_status,
                    "certification_timestamp_utc": eng_meta.get("certification_timestamp_utc", "UNKNOWN"),
                    "certified_by": eng_meta.get("certified_by", "UNKNOWN")
                },
                "lifecycle_metadata": {
                    "terminal_state": lifecycle_status,
                    "terminal_event_id": eng_meta.get("terminal_event_id", "UNKNOWN"),
                    "is_current": eng_meta.get("is_current", True)
                },
                "conflict_metadata": {
                    "conflict_status": conflict_status,
                    "conflict_candidate_ids": eng_meta.get("conflict_candidate_ids", [])
                },
                "traceability": {k: traceability.get(k) for k in required_traceability}
            })
                
        # Rank and limit candidates
        derived_results.sort(key=lambda x: x["operator_value_score"], reverse=True)
        max_selected_facts = 2
        derived_results = derived_results[:max_selected_facts]
        
        response["derived_results"] = derived_results
        
        # Finalize response meta
        facts_returned = len(derived_results)
        response["derived_lane_meta"]["facts_returned"] = facts_returned
        response["derived_lane_meta"]["executed"] = True
        response["derived_lane_meta"]["reason"] = "success"
        response["derived_lane_meta"]["candidate_telemetry"] = telemetry["candidate_telemetry"]
        
        # Finalize telemetry
        total_dropped = (
            response["derived_lane_meta"]["facts_dropped_governance"] +
            response["derived_lane_meta"]["facts_dropped_traceability"] +
            response["derived_lane_meta"]["facts_dropped_lifecycle"] +
            response["derived_lane_meta"]["facts_dropped_conflict"]
        )
        
        telemetry["status"] = "execution"
        telemetry["reason"] = "authorized"
        telemetry["metrics"]["returned"] = facts_returned
        telemetry["metrics"]["dropped"] = total_dropped
        
        self._telemetry_sink.emit(telemetry)

        return response

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
        graph_experiment_params: Optional[Dict[str, Any]] = None,
        latency_budget_ms: Optional[float] = None,
        complexity_shadow: bool = False,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        mode = retrieval_mode if retrieval_mode in {"semantic", "hybrid", "graph_hybrid_experimental"} else "semantic"
        policy = fusion_policy if fusion_policy in FUSION_POLICIES else DEFAULT_FUSION_POLICY
        envelope_cfg = CandidateEnvelopeConfig.from_request(bounded_envelope)
        desired_pool = max(top_k, envelope_cfg.candidate_pool_limit) if envelope_cfg.enabled else top_k

        # W6/W7: resolve a stage plan only when the caller supplies a budget;
        # the no-budget path is byte-identical to pre-budget behavior.
        budget_plan = self._budget_router.plan(latency_budget_ms) if latency_budget_ms is not None else None
        user_filters = self._user_filters(filters)
        complexity_meta = self._run_complexity_shadow(query, complexity_shadow)

        experimental_telemetry = None
        if mode == "graph_hybrid_experimental":
            if not self._enable_experimental_graph_hybrid:
                mode = "semantic"
                experimental_telemetry = {
                    "experimental_graph_hybrid_requested": True,
                    "experimental_graph_hybrid_enabled": False,
                    "experimental_graph_hybrid_action": "fallback_to_semantic",
                    "warning": "GRAPH_HYBRID_EXPERIMENTAL_NOT_ENABLED"
                }
            else:
                experimental_telemetry = {
                    "experimental_graph_hybrid_requested": True,
                    "experimental_graph_hybrid_enabled": True,
                    "experimental_graph_hybrid_action": "executed",
                    "warning": "EXPERIMENTAL_IN_MEMORY_VALIDATED_ONLY"
                }

        if mode in {"semantic", "graph_hybrid_experimental"} or not self._lexical_tier:
            t_s = time.perf_counter()
            self._stats["semantic_query_count"] += 1
            if mode == "graph_hybrid_experimental":
                self._stats["retrieval_mode_counters"]["semantic"] += 1 # Or track separately
            else:
                self._stats["retrieval_mode_counters"]["semantic"] += 1
            # Server-side derived-fact exclusion (Qdrant must_not payload filter);
            # over-fetch only when a tier without payload filtering is active.
            effective_filters = dict(user_filters)
            effective_filters["__exclude_derived__"] = True
            if budget_plan is not None:
                effective_filters.update(self._budget_filter_overrides(budget_plan))
            raw_hits = self._semantic_fusion.search(
                query,
                top_k=desired_pool * self._derived_overfetch_factor(),
                filters=effective_filters,
                tiers=tiers,
            )
            hits = []
            for h in raw_hits:
                if self._is_derived(h):
                    continue
                hits.append(h)
                if len(hits) == desired_pool:
                    break

            t_e = (time.perf_counter() - t_s) * 1000

            hits, rr_telemetry = self._rerank_with_budget(budget_plan, query, hits, t_e)

            graph_experiment_telemetry = None
            if mode == "graph_hybrid_experimental" and self._graph_tier:
                # 1. Expand graph from hits
                g_cands, g_meta = self._graph_tier.expand_candidates(
                    seed_candidates=hits,
                    max_depth=1,
                    max_neighbors_per_seed=5,
                    max_total_graph_candidates=20,
                    max_seed_candidates=10
                )
                
                # 2. Filter governance/lineage/scoring
                # Use defaults: hub_penalty_floor=0.2, score_threshold=0.2
                eligible = []
                for c in g_cands:
                    if c.graph_score and c.graph_score >= 0.2 and c.lineage_complete:
                         if not c.engram.governance or c.engram.governance.conflict_status != "vetoed":
                             eligible.append(SearchResult(engram=c.engram, score=c.graph_score, tier="graph"))

                # 3. Merge helper
                def merge_graph_candidates_for_experiment(primary_candidates, graph_candidates, preserve_primary_top_k=5, graph_quota=3, graph_ratio_cap=0.2):
                    tel = {
                        "graph_candidates_pre_merge": len(graph_candidates),
                        "graph_candidates_inserted_pre_envelope": 0,
                        "primary_top_k_preserved": min(len(primary_candidates), preserve_primary_top_k),
                        "primary_candidates_displaced_count": 0,
                        "graph_displacement_rate": 0.0,
                        "merge_policy": "lane_aware_quota_v0"
                    }
                    if not graph_candidates:
                        return primary_candidates, tel
                        
                    max_graph = min(graph_quota, int(len(primary_candidates) * graph_ratio_cap))
                    inserted = graph_candidates[:max_graph]
                    tel["graph_candidates_inserted_pre_envelope"] = len(inserted)
                    
                    merged = primary_candidates[:preserve_primary_top_k] + inserted + primary_candidates[preserve_primary_top_k:]
                    tel["primary_candidates_displaced_count"] = len(inserted)
                    tel["graph_displacement_rate"] = len(inserted) / len(primary_candidates) if primary_candidates else 0.0
                    tel["graph_candidate_ratio_pre_envelope"] = len(inserted) / len(merged) if merged else 0.0
                    return merged, tel

                g_params = graph_experiment_params or {}
                hits, graph_experiment_telemetry = merge_graph_candidates_for_experiment(
                    hits, eligible,
                    preserve_primary_top_k=g_params.get("preserve_primary_top_k", 5),
                    graph_quota=g_params.get("graph_quota", 1),
                    graph_ratio_cap=g_params.get("graph_ratio_cap", 0.1)
                )

            narrowed, envelope_meta = apply_candidate_envelope(hits, envelope_cfg)
            self._record_candidate_envelope_stats(envelope_meta)
            final_results = narrowed[:top_k]
            
            if graph_experiment_telemetry:
                survived = sum(1 for c in final_results if c.tier == "graph")
                graph_experiment_telemetry["graph_candidates_survived_envelope"] = survived
                inserted = graph_experiment_telemetry["graph_candidates_inserted_pre_envelope"]
                graph_experiment_telemetry["graph_candidates_dropped_by_envelope"] = inserted - survived
                graph_experiment_telemetry["graph_candidate_ratio_post_envelope"] = survived / len(final_results) if final_results else 0.0

            graph_telemetry = self._run_graph_shadow(final_results, envelope_cfg)
            
            meta = {
                "retrieval_mode": mode,
                "fusion_policy": None,
                "lexical_available": self.lexical_available,
                "candidate_envelope": envelope_meta,
                "reranker_used": self._reranker is not None,
                "rerank_telemetry": rr_telemetry,
            }
            if experimental_telemetry:
                meta["experimental_graph_telemetry"] = experimental_telemetry
            if graph_experiment_telemetry:
                meta["graph_experiment_telemetry"] = graph_experiment_telemetry
            if graph_telemetry:
                meta["graph_shadow_telemetry"] = graph_telemetry
            if complexity_meta:
                meta["complexity_shadow"] = complexity_meta
                
            # PIT-1 Leakage Guard
            final_results = self._apply_leakage_guard(final_results, meta)
            self._record_budget_outcome(budget_plan, meta, t_e)
            return final_results, meta

        start = time.perf_counter()

        # ── Qdrant-native RRF path ──────────────────────────────
        if (
            policy == "qdrant_rrf"
            and budget_plan is None
            and self._qdrant_hybrid
            and self._qdrant_hybrid.available
        ):
            try:
                # Compute embedding once for the RRF call
                query_vec = self._qdrant_hybrid._tier._embed([query])[0]
                fused, telemetry = self._qdrant_hybrid.fuse(
                    query=query,
                    query_vector=query_vec,
                    top_k=desired_pool,
                    filters=user_filters,
                    semantic_limit=semantic_top_k,
                    lexical_limit=lexical_top_k,
                )
            except Exception as e:
                logger.warning(f"QdrantHybridFusion failed, falling back to Python hybrid: {e}")
                # Fall through to Python-side fusion below
                fused = None
                telemetry = None

            if fused is not None:
                telemetry = telemetry or {}
                fused = [h for h in fused if not self._is_derived(h)]

                fused, rr_telemetry = self._rerank_with_budget(
                    budget_plan, query, fused, (time.perf_counter() - start) * 1000
                )

                narrowed, envelope_meta = apply_candidate_envelope(fused, envelope_cfg)
                self._record_candidate_envelope_stats(envelope_meta)
                final_results = narrowed[:top_k]

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._record_hybrid_stats(telemetry, elapsed_ms, policy)

                graph_telemetry = self._run_graph_shadow(final_results, envelope_cfg)

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
                if graph_telemetry:
                    meta["graph_shadow_telemetry"] = graph_telemetry
                if complexity_meta:
                    meta["complexity_shadow"] = complexity_meta
                    
                # PIT-1 Leakage Guard
                final_results = self._apply_leakage_guard(final_results, meta)
                self._record_budget_outcome(
                    budget_plan, meta, (time.perf_counter() - start) * 1000
                )
                return final_results, meta

        # ── Python-side hybrid fusion (original path) ──────────
        lexical_results = self._lexical_tier.search(query, top_k=lexical_top_k, filters=user_filters)
        _semantic_filters = dict(user_filters)
        _semantic_filters["__exclude_derived__"] = True
        if budget_plan is not None:
            _semantic_filters.update(self._budget_filter_overrides(budget_plan))
        semantic_results = self._semantic_fusion.search(
            query,
            top_k=semantic_top_k * self._derived_overfetch_factor(),
            filters=_semantic_filters,
            tiers=tiers,
        )

        _semantic_clean = []
        for h in semantic_results:
            if self._is_derived(h):
                continue
            _semantic_clean.append(h)
            if len(_semantic_clean) == semantic_top_k:
                break
        semantic_results = _semantic_clean

        fused, telemetry = self._hybrid_fusion.fuse(
            lexical_results=lexical_results,
            semantic_results=semantic_results,
            top_k=desired_pool,
            fusion_policy=policy,
            filters=user_filters,
            explain=explain,
        )

        fused, rr_telemetry = self._rerank_with_budget(budget_plan, query, fused, (time.perf_counter() - start) * 1000)

        narrowed, envelope_meta = apply_candidate_envelope(fused, envelope_cfg)
        self._record_candidate_envelope_stats(envelope_meta)
        final_results = narrowed[:top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._record_hybrid_stats(telemetry, elapsed_ms, policy)

        graph_telemetry = self._run_graph_shadow(final_results, envelope_cfg)

        meta = {
            "retrieval_mode": "hybrid",
            "fusion_policy": policy,
            "lexical_available": self.lexical_available,
            "telemetry": telemetry,
            "candidate_envelope": envelope_meta,
            "reranker_used": self._reranker is not None,
            "rerank_telemetry": rr_telemetry,
        }
        if graph_telemetry:
            meta["graph_shadow_telemetry"] = graph_telemetry
        if complexity_meta:
            meta["complexity_shadow"] = complexity_meta
            
        # PIT-1 Leakage Guard
        final_results = self._apply_leakage_guard(final_results, meta)
        self._record_budget_outcome(budget_plan, meta, elapsed_ms)
        return final_results, meta

    def stats(self) -> Dict[str, Any]:
        out = dict(self._stats)
        raw_total = max(1, int(out.get("candidate_pool_raw_count", 0)))
        narrowed_total = int(out.get("candidate_pool_narrowed_count", 0))
        out["candidate_envelope_avg_compression_ratio"] = round(
            narrowed_total / raw_total, 4
        )
        out["candidate_envelope_total_reduction"] = max(0, raw_total - narrowed_total)
        out["supported_retrieval_modes"] = ["semantic", "hybrid", "graph_hybrid_experimental"]
        out["supported_fusion_policies"] = sorted(FUSION_POLICIES.keys())
        return out
