"""
MNEMOS REST API Service
========================

Flask-based REST API with MFS contract compliance.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.config import get_config, MnemosConfig
from mnemos.engram.model import Engram, EngramBatch
from mnemos.memory_over_maps.view_builder import (
    SUPPORTED_DERIVED_VIEWS,
    build_requested_views,
)
from mnemos.memory_over_maps.view_cache import (
    DerivedViewCache,
    build_cache_key,
    governance_state_hash,
    lineage_inputs,
    query_fingerprint,
)
from mnemos.retrieval.policies.fusion_policies import FUSION_POLICIES
from mnemos.retrieval.intent import IntentEngine
from mnemos.retrieval.pulse import (
    PulseEngine,
    PulseHarvester,
    TimesFMProvider,
    build_pulse_payload,
)
from mnemos.retrieval.shadow_search import ShadowSearchRunner
from mnemos.governance.counterfactuals import compute_counterfactuals
from mnemos.governance.governor import Governor
from mnemos.governance.hygiene.volatility import VolatilityEngine, family_key_from_engram
from mnemos.governance.policy_profiles import load_policy_profiles
from mnemos.governance.read_path import GOVERNANCE_MODES

logger = logging.getLogger("mnemos.service")

CONTRACT_VERSION = "v1"
SUPPORTED_RETRIEVAL_MODES = {"semantic", "hybrid"}
RESERVED_FILTER_KEYS = {"__mrl_oversample__", "__hnsw_ef__", "__prefetch_only__"}

app = Flask(__name__)


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _authorized() -> bool:
    config = get_config()
    if not config.token:
        return True
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {config.token}"


# ──────────────────── Runtime ────────────────────


class MnemosRuntime:
    """Holds the MNEMOS backend: retrieval tiers, fusion engine, and audit ledger."""

    def __init__(self):
        self._initialized = False
        self._config: Optional[MnemosConfig] = None
        self._semantic_fusion = None
        self._router = None
        self._lexical_tier = None
        self._ledger = None
        self._governor: Optional[Governor] = None
        self._volatility_engine = VolatilityEngine()
        self._view_cache: Optional[DerivedViewCache] = None
        self._pulse_engine = PulseEngine(harvester=PulseHarvester())
        self._intent_engine = IntentEngine()
        self._shadow_runner: Optional[ShadowSearchRunner] = None
        self._status = "healthy"
        self._error: Optional[str] = None
        self._mom_stats: Dict[str, int] = {
            "derived_view_generated_count": 0,
            "derived_view_cache_hit_count": 0,
            "derived_view_cache_miss_count": 0,
            "derived_view_invalidated_count": 0,
            "derived_view_invalidation_events": 0,
            "derived_view_invalidation_fanout_total": 0,
            "governed_evidence_bundle_total": 0,
            "governed_contradiction_bundle_total": 0,
            "governed_source_trace_complete_total": 0,
            "economics_query_count": 0,
            "economics_cost_units_total": 0,
            "economics_envelope_initial_total": 0,
            "economics_envelope_final_total": 0,
            # PIT-7 Telemetry
            "query.default_retrieval.derived_fact_count": 0,
            "echoframe.production_prompt.derived_count": 0,
            "derived_lane.execution_count": 0,
            "derived_lane.denied_count": 0,
            "derived_lane.kill_switch_count": 0,
            "evaluate_derived_shadow.request_count": 0,
            "evaluate_derived_shadow.denied_count": 0,
            "evaluate_derived_shadow.rendered_derived_fact_count": 0,
        }

    def initialize(self):
        if self._initialized:
            return

        try:
            self._config = get_config()
            self._pulse_engine = PulseEngine(
                harvester=PulseHarvester(),
                provider=TimesFMProvider(
                    base_url=self._config.timesfm_sidecar_url,
                    timeout_s=self._config.timesfm_timeout_s,
                    enabled=self._config.timesfm_enabled,
                ),
                horizon_minutes=self._config.pulse_horizon_minutes,
                actions_mode=self._config.pulse_actions,
                warmup_callback=self.predictive_warmup,
                audit_callback=self._audit_autonomous_warmup,
                cooldown_seconds=self._config.pulse_warmup_cooldown_s,
            )
            self._volatility_engine = VolatilityEngine(
                reconciliation_callback=self._proactive_reconciliation,
                audit_callback=self._audit_proactive_reconciliation,
            )

            # Set up logging
            logging.basicConfig(
                level=getattr(logging, self._config.log_level, logging.INFO),
                format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            )

            # Build retrieval tiers
            tiers = []

            if self._config.has_qdrant:
                from mnemos.retrieval.qdrant_tier import QdrantTier
                tiers.append(QdrantTier(
                    url=self._config.qdrant_url,
                    collection_name=self._config.qdrant_collection,
                    embedding_model=self._config.embedding_model,
                    gpu_device=self._config.gpu_device,
                ))

            if self._config.has_pgvector:
                from mnemos.retrieval.pgvector_tier import PgvectorTier
                tiers.append(PgvectorTier(
                    dsn=self._config.postgres_dsn,
                    table_name=self._config.pgvector_table,
                    embedding_model=self._config.embedding_model,
                    gpu_device=self._config.gpu_device,
                ))

            if self._config.has_lancedb:
                from mnemos.retrieval.lancedb_tier import LanceDBTier
                tiers.append(LanceDBTier(db_dir=f"{self._config.data_dir}/lance"))

            if not tiers:
                raise RuntimeError("No retrieval tiers configured. Set MNEMOS_TIERS.")

            from mnemos.retrieval.fusion import TierFusion
            from mnemos.retrieval.retrieval_router import RetrievalRouter

            self._semantic_fusion = TierFusion(tiers)

            # Lexical lane is backed by Postgres FTS and enabled when Postgres is configured.
            self._lexical_tier = None
            if self._config.has_postgres:
                from mnemos.retrieval.lexical_tier import LexicalTier

                self._lexical_tier = LexicalTier(
                    dsn=self._config.postgres_dsn,
                    table_name=self._config.lexical_table,
                )

            if getattr(self._config, "use_reranker", False):
                from mnemos.retrieval.cross_encoder import CrossEncoderReranker
                reranker = CrossEncoderReranker(model_name=self._config.reranker_model)
            else:
                reranker = None

            self._router = RetrievalRouter(
                semantic_fusion=self._semantic_fusion,
                lexical_tier=self._lexical_tier,
                reranker=reranker,
                adaptive_routing_enabled=bool(getattr(self._config, "adaptive_routing", True)),
                pulse_engine=self._pulse_engine,
                pulse_p95_budget_ms=self._config.pulse_p95_budget_ms,
            )

            # Set up audit ledger
            if self._config.audit_enabled:
                if self._config.has_postgres:
                    from mnemos.audit.postgres_ledger import PostgresLedger
                    self._ledger = PostgresLedger(dsn=self._config.postgres_dsn)
                else:
                    from mnemos.audit.forensic_ledger import ForensicLedger
                    Path(self._config.audit_db_path).parent.mkdir(parents=True, exist_ok=True)
                    self._ledger = ForensicLedger(db_path=self._config.audit_db_path)

            # Governance layer (always initialised; off by default)
            policy_profiles = load_policy_profiles(
                raw_json=os.getenv("MNEMOS_GOVERNANCE_POLICY_PROFILES_JSON", ""),
                base_min_score_threshold=self._config.governance_min_score,
                base_freshness_half_life_days=self._config.governance_freshness_half_life,
            )
            self._governor = Governor(
                min_score_threshold=self._config.governance_min_score,
                freshness_half_life_days=self._config.governance_freshness_half_life,
                policy_profiles=policy_profiles,
                volatility_engine=self._volatility_engine,
                volatility_bias_enabled=self._config.governance_volatility_bias,
            )
            self._view_cache = DerivedViewCache(ttl_seconds=3600)
            self._shadow_runner = ShadowSearchRunner(
                search_callable=self._shadow_search_payload,
                cache=self._view_cache,
            )
            self._intent_engine = IntentEngine(
                harvester=None,
                horizon_steps=3,
                shadow_callback=self._shadow_runner.run,
            )

            self._initialized = True
            logger.info(
                f"🚀 MNEMOS runtime initialized: semantic_tiers={self._semantic_fusion.tier_names}, "
                f"lexical_available={bool(self._lexical_tier)}, "
                f"governance_mode={self._config.governance_mode}"
            )

        except Exception as e:
            self._status = "unavailable"
            self._error = str(e)
            # Reset all components so a retry starts from a clean slate.
            self._config = None
            self._semantic_fusion = None
            self._router = None
            self._lexical_tier = None
            self._ledger = None
            self._governor = None
            self._view_cache = None
            logger.exception("MNEMOS runtime initialization failed")
            raise

    def _base_payload(self) -> Dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "status": self._status,
            "source": "mnemos-service",
            "generated_at": _utc_now(),
            "error": self._error,
        }

    @staticmethod
    def _build_governance_trace(
        *,
        decision: Any,
        raw_rank: Optional[int],
        final_rank: Optional[int],
    ) -> Dict[str, Any]:
        modifiers = {
            "trust": float(decision.trust_modifier),
            "utility": float(decision.utility_modifier),
            "freshness": float(decision.freshness_modifier),
            "contradiction": float(decision.contradiction_modifier),
            "veto": float(decision.veto_modifier),
        }
        top_factors = sorted(
            [
                {"name": k, "value": round(v, 4), "impact": round(abs(v - 1.0), 4)}
                for k, v in modifiers.items()
                if abs(v - 1.0) > 1e-9
            ],
            key=lambda row: row["impact"],
            reverse=True,
        )[:3]

        if not decision.veto_pass:
            outcome = "vetoed"
            reason = decision.veto_reason or "vetoed by policy"
        elif decision.suppressed_by_contradiction:
            outcome = "contradiction_loser"
            reason = decision.contradiction_reason or decision.suppressed_reason or "lost contradiction adjudication"
        elif decision.conflict_status == "winner":
            outcome = "contradiction_winner"
            reason = decision.contradiction_reason or "won contradiction adjudication"
        elif decision.would_be_suppressed_in_enforced_mode:
            outcome = "would_be_suppressed"
            reason = decision.suppressed_reason or "would be suppressed in enforced mode"
        else:
            outcome = "retained"
            reason = "retained after governance scoring"

        trace: Dict[str, Any] = {
            "outcome": outcome,
            "reason": reason,
            "score_delta": round(float(decision.governed_score) - float(decision.retrieval_score), 4),
            "top_factors": top_factors,
        }
        if raw_rank is not None:
            trace["raw_rank"] = int(raw_rank)
        if final_rank is not None:
            trace["final_rank"] = int(final_rank)
        if raw_rank is not None and final_rank is not None:
            trace["rank_shift"] = int(raw_rank - final_rank)
        if decision.conflict_group_id:
            trace["conflict_group_id"] = decision.conflict_group_id
        if decision.contradiction_winner:
            trace["contradiction_winner"] = decision.contradiction_winner
        return trace

    def _audit(self, action: str, content: str, status: str = "success",
               metadata: Optional[Dict] = None, latency: float = 0.0):
        if self._ledger:
            self._ledger.log_transaction(
                component="mnemos-api",
                action=action,
                content=content,
                status=status,
                latency=latency,
                metadata=metadata,
            )

    def _audit_autonomous_warmup(self, action: Dict[str, Any]) -> None:
        reason = str(action.get("reason", "predicted need"))
        confidence = float(action.get("confidence_score", 0.0) or 0.0)
        self._audit(
            "autonomous_prewarm",
            f"[ACTION] Autonomous Pre-warm triggered - Reason: {reason} (Confidence: {confidence:.2f})",
            metadata={
                "forecast_reason": reason,
                "confidence_score": confidence,
                "query_spike_pct": action.get("query_spike_pct"),
                "p95_rise_ms": action.get("p95_rise_ms"),
                "target": action.get("target", {}),
                "warmup_result": action.get("warmup_result", {}),
            },
            latency=0.0,
        )

    def _audit_proactive_reconciliation(self, action: Dict[str, Any]) -> None:
        reason = str(action.get("reason", "predicted contradiction spike"))
        confidence = float(action.get("confidence_score", 0.0) or 0.0)
        self._audit(
            "proactive_reconciliation",
            f"[ACTION] Proactive Reconciliation triggered - Reason: {reason} (Confidence: {confidence:.2f})",
            metadata={
                "family_key": action.get("family_key"),
                "entity_key": action.get("entity_key"),
                "forecast_reason": reason,
                "confidence_score": confidence,
                "predicted_obsolescence": action.get("predicted_obsolescence"),
                "reconciliation_result": action.get("reconciliation_result", {}),
            },
            latency=0.0,
        )

    def _proactive_reconciliation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "scheduled",
            "family_key": action.get("family_key"),
            "entity_key": action.get("entity_key"),
        }

    def _shadow_search_payload(self, query: str) -> List[Dict[str, Any]]:
        if self._router is None:
            return []
        hits, _ = self._router.search(
            query=query,
            top_k=5,
            filters=None,
            tiers=None,
            retrieval_mode=self._config.retrieval_mode if self._config else "semantic",
            fusion_policy=self._config.fusion_policy if self._config else "balanced",
            explain=False,
            lexical_top_k=self._config.lexical_top_k if self._config else 25,
            semantic_top_k=self._config.semantic_top_k if self._config else 25,
        )
        return [
            {
                "engram": hit.engram.to_dict(),
                "score": round(hit.score, 4),
                "tier": hit.tier,
                "tiers": hit.metadata.get("tiers", [hit.tier]),
            }
            for hit in hits
        ]

    def _audit_derived_view_generation(
        self,
        *,
        view_type: str,
        view_id: str,
        inputs: Dict[str, Any],
        query_fingerprint: str,
        governance_state_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._ledger:
            return
        if hasattr(self._ledger, "log_derived_view_generation"):
            self._ledger.log_derived_view_generation(
                view_type=view_type,
                view_id=view_id,
                inputs=inputs,
                query_fingerprint=query_fingerprint,
                governance_state_hash=governance_state_hash,
                metadata=metadata,
            )

    def capabilities(self) -> Dict[str, Any]:
        payload = self._base_payload()
        retrieval_stats = self._router.stats() if self._router else {}
        payload.update({
            "feature": "mnemos_memory",
            "profile": self._config.profile if self._config else "unknown",
            "supports": ["index", "search", "warmup", "engrams", "audit", "stats", "pulse"],
            "pulse": {
                "enabled": True,
                "timesfm_enabled": bool(getattr(self._config, "timesfm_enabled", True)),
                "actions_mode": getattr(self._config, "pulse_actions", "advisory"),
                "horizon_minutes": int(getattr(self._config, "pulse_horizon_minutes", 15)),
                "warmup_cooldown_s": int(getattr(self._config, "pulse_warmup_cooldown_s", 900)),
                "provider": "timesfm_sidecar_with_linear_fallback",
            },
            "tiers": self._semantic_fusion.tier_names if self._semantic_fusion else [],
            "retrieval_modes": retrieval_stats.get("supported_retrieval_modes", ["semantic"]),
            "fusion_policies": retrieval_stats.get("supported_fusion_policies", []),
            "retrieval_mode_default": self._config.retrieval_mode if self._config else "semantic",
            "fusion_policy_default": self._config.fusion_policy if self._config else "balanced",
            "lexical_lane_available": bool(self._lexical_tier),
            "explain_support": True,
            "compression": {
                "enabled": self._config.has_compression if self._config else False,
                "bits": self._config.quant_bits if self._config else 0,
            },
            "gpu_device": self._config.gpu_device if self._config else "unknown",
            "governance": {
                "supported_modes": sorted(GOVERNANCE_MODES),
                "default_mode": self._config.governance_mode if self._config else "off",
                "policy_profiles": self._governor.policy_profile_ids() if self._governor else ["default"],
            },
            "memory_over_maps": {
                "phase1_enabled": bool(
                    getattr(self._config, "memory_over_maps_phase1", False)
                ),
                "phase2_enabled": bool(
                    getattr(self._config, "memory_over_maps_phase2", False)
                ),
                "phase3_enabled": bool(
                    getattr(self._config, "memory_over_maps_phase3", False)
                ),
                "phase4_enabled": bool(
                    getattr(self._config, "memory_over_maps_phase4", False)
                ),
                "phase5_enabled": bool(
                    getattr(self._config, "memory_over_maps_phase5", False)
                ),
                "supported_derived_views": sorted(SUPPORTED_DERIVED_VIEWS),
            },
        })
        return payload

    def index_documents(self, documents: List[Dict], options: Dict) -> Dict[str, Any]:
        """Ingest documents into engrams and index across tiers."""
        import time
        t0 = time.time()

        engrams = []
        for doc in documents:
            engram = Engram(
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                neuro_tags=doc.get("neuro_tags", []),
                confidence=doc.get("confidence", 1.0),
                metadata=doc.get("metadata", {}),
            )
            engrams.append(engram)
            if self._volatility_engine is not None:
                self._volatility_engine.harvester.record_index_update(engram)

        tiers = options.get("tiers")
        counts = self._semantic_fusion.index(engrams, tiers=tiers)
        index_lexical = options.get("index_lexical", True)
        if index_lexical and self._lexical_tier:
            counts["lexical"] = self._lexical_tier.index(engrams)

        elapsed = time.time() - t0
        self._audit("index", f"Indexed {len(engrams)} documents", metadata={
            "count": len(engrams), "tiers": counts,
        }, latency=elapsed)

        payload = self._base_payload()
        payload["result"] = {
            "indexed": len(engrams),
            "tiers": counts,
            "engram_ids": [e.id for e in engrams],
            "latency_s": round(elapsed, 3),
        }
        return payload

    def warmup(self, query: str = "mnemos warmup readiness probe") -> Dict[str, Any]:
        """Load the promoted retrieval path before admitting live traffic."""
        import time

        t0 = time.time()
        results, mode_meta = self._router.search(
            query=query,
            top_k=1,
            filters=None,
            tiers=None,
            retrieval_mode="semantic",
            fusion_policy=self._config.fusion_policy,
            explain=False,
            lexical_top_k=self._config.lexical_top_k,
            semantic_top_k=1,
            bounded_envelope=None,
        )
        payload = self._base_payload()
        payload["warmup"] = {
            "query": query,
            "result_count": len(results),
            "retrieval_mode": mode_meta.get("retrieval_mode", "semantic"),
            "latency_s": round(time.time() - t0, 3),
        }
        return payload

    def predictive_warmup(self, action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Warm runtime components before a predicted spike arrives."""
        action = action or {}
        target = action.get("target", {}) if isinstance(action.get("target"), dict) else {}
        warmed: List[str] = []
        errors: List[str] = []

        if self._router is not None:
            warmed.append("retrieval_router")

        if self._semantic_fusion is not None:
            try:
                _ = self._semantic_fusion.stats()
                warmed.append("semantic_fusion")
            except Exception as exc:
                errors.append(f"semantic_fusion: {exc}")

        reranker = getattr(self._router, "_reranker", None) if self._router is not None else None
        if reranker is not None:
            try:
                if getattr(reranker, "_initialize", None):
                    reranker._initialize()
                if hasattr(reranker, "health"):
                    _ = reranker.health()
                warmed.append("reranker")
            except Exception as exc:
                errors.append(f"reranker: {exc}")

        if target.get("layer") == "hierarchical_summary":
            if self._view_cache is None:
                self._view_cache = DerivedViewCache(ttl_seconds=3600)
            warmed.append("hierarchical_summary")
            warmed.append("derived_view_cache")

        return {
            "status": "success" if not errors else "degraded",
            "warmed_components": warmed,
            "errors": errors,
            "target": target,
        }

    def search_documents(
        self,
        query: str,
        top_k: int,
        tiers: Optional[List[str]],
        filters: Optional[Dict],
        retrieval_mode: Optional[str],
        fusion_policy: Optional[str],
        explain: Optional[bool],
        governance: Optional[str] = None,
        explain_governance: Optional[bool] = None,
        governance_profile: Optional[str] = None,
        bounded_envelope: Optional[Dict[str, Any]] = None,
        derive_views: Optional[List[str]] = None,
        latency_budget_ms: Optional[float] = None,
        complexity_shadow: bool = False,
    ) -> Dict[str, Any]:
        """Search across tiers and return fused results."""
        import time
        t0 = time.time()
        session_id = "default"
        if isinstance(filters, dict):
            session_id = str(filters.get("session_id") or filters.get("tenant_id") or "default")

        if self._view_cache is not None and self._intent_engine is not None:
            try:
                cluster_id = self._intent_engine.harvester.map_query(query)
                cached = self._view_cache.fuzzy_pre_cognitive_get(
                    query=query,
                    cluster_id=cluster_id,
                )
                if cached is not None:
                    self._audit(
                        "shadow_hit",
                        f"Pre-cognitive cache hit for query: '{query[:80]}'",
                        metadata={
                            "forecast_reason": "intent trajectory matched pre-cognitive cache",
                            "pre_cognitive": True,
                            "cache": cached.get("_cache", {}),
                            "cluster_id": cluster_id,
                            "session_id": session_id,
                        },
                        latency=0.0,
                    )
                    payload = self._base_payload()
                    payload["results"] = cached.get("results", [])
                    payload["meta"] = {
                        "query": query,
                        "top_k": top_k,
                        "result_count": len(payload["results"]),
                        "latency_s": 0.002,
                        "retrieval_mode": "pre_cognitive_cache",
                        "fusion_policy": None,
                        "pre_cognitive": True,
                        "cache": cached.get("_cache", {}),
                    }
                    self._intent_engine.harvester.record_query(
                        session_id=session_id,
                        query=query,
                    )
                    return payload
            except Exception:
                pass

        selected_mode = retrieval_mode or self._config.retrieval_mode
        selected_policy = fusion_policy or self._config.fusion_policy
        selected_explain = self._config.explain_default if explain is None else bool(explain)
        selected_governance = governance or getattr(self._config, "governance_mode", "off")
        selected_explain_gov = bool(explain_governance) if explain_governance is not None else False
        selected_profile = governance_profile or ""
        if not selected_profile and isinstance(filters, dict):
            selected_profile = str(filters.get("tenant_policy") or filters.get("tenant_id") or "").strip()
        if selected_profile and self._governor and not self._governor.has_policy_profile(selected_profile):
            selected_profile = ""

        results, mode_meta = self._router.search(
            query=query,
            top_k=top_k,
            filters=filters,
            tiers=tiers,
            retrieval_mode=selected_mode,
            fusion_policy=selected_policy,
            explain=selected_explain,
            lexical_top_k=self._config.lexical_top_k,
            semantic_top_k=self._config.semantic_top_k,
            bounded_envelope=bounded_envelope if getattr(self._config, "memory_over_maps_phase2", False) else None,
            latency_budget_ms=latency_budget_ms,
            complexity_shadow=complexity_shadow,
            adaptive_routing=bool(getattr(self._config, "adaptive_routing", True)),
        )
        raw_rank_by_id = {r.engram.id: idx + 1 for idx, r in enumerate(results)}

        # ── Governance ────────────────────────────────────────────────────
        decisions = []
        contradiction_records = []
        if selected_governance != "off" and self._governor:
            results, decisions, contradiction_records = self._governor.govern(
                results=results,
                query=query,
                governance_mode=selected_governance,
                top_k=top_k,
                governance_profile=selected_profile or None,
            )
            for decision in decisions:
                trace = getattr(decision, "policy_trace", {}) or {}
                if "volatility_bias" in trace:
                    self._audit(
                        "volatility_decay",
                        f"Volatility-driven freshness decay applied to engram {decision.engram_id}",
                        metadata={
                            "forecast_reason": "predicted semantic volatility shortened freshness half-life",
                            "engram_id": decision.engram_id,
                            "volatility_bias": trace["volatility_bias"],
                            "freshness_modifier": decision.freshness_modifier,
                        },
                        latency=0.0,
                    )

        derived_views_payload: List[Dict[str, Any]] = []
        query_cache_hits = 0
        query_cache_misses = 0
        if (
            bool(getattr(self._config, "memory_over_maps_phase3", False))
            and derive_views
        ):
            phase4_cache_enabled = bool(getattr(self._config, "memory_over_maps_phase4", False)) and self._view_cache is not None
            li = lineage_inputs(results)
            qfp = query_fingerprint(query)
            ghash = governance_state_hash(decisions)

            for view_name in derive_views:
                view_payload: Optional[Dict[str, Any]] = None
                cache_key = build_cache_key(
                    view_type=view_name,
                    query_fingerprint_value=qfp,
                    artifact_ids=li.get("artifact_ids", []),
                    chunk_ids=li.get("chunk_ids", []),
                    governance_state_hash_value=ghash,
                    synthesis_policy_version="default",
                    embedding_model_name=self._config.embedding_model,
                )
                if phase4_cache_enabled:
                    cached = self._view_cache.get(cache_key)
                    if cached is not None:
                        query_cache_hits += 1
                        self._mom_stats["derived_view_cache_hit_count"] += 1
                        cached["_cache"] = {"hit": True, "key": cache_key}
                        view_payload = cached
                    else:
                        query_cache_misses += 1
                        self._mom_stats["derived_view_cache_miss_count"] += 1

                if view_payload is None:
                    built = build_requested_views(
                        requested=[view_name],
                        query=query,
                        results=results,
                        decisions=decisions,
                        contradiction_records=contradiction_records,
                        subject_id=(filters or {}).get("subject_id") if isinstance(filters, dict) else None,
                    )
                    if not built:
                        continue
                    view_payload = built[0]
                    self._mom_stats["derived_view_generated_count"] += 1
                    if phase4_cache_enabled:
                        dependency_refs = {
                            "artifact_ids": view_payload.get("inputs", {}).get("artifact_ids", []),
                            "chunk_ids": view_payload.get("inputs", {}).get("chunk_ids", []),
                            "governance_state_hash": view_payload.get("governance_state_hash"),
                            "synthesis_policy_version": view_payload.get("synthesis_policy", "default"),
                            "contradiction_cluster_id": view_payload.get("contradiction_cluster_id"),
                            "lifecycle_states": [],
                        }
                        self._view_cache.set(
                            key=cache_key,
                            view=view_payload,
                            dependency_refs=dependency_refs,
                        )
                        view_payload = dict(view_payload)
                        view_payload["_cache"] = {"hit": False, "key": cache_key}

                derived_views_payload.append(view_payload)

            for view in derived_views_payload:
                if view.get("view_type") == "evidence_bundle":
                    self._mom_stats["governed_evidence_bundle_total"] += 1
                if view.get("view_type") == "contradiction_bundle":
                    self._mom_stats["governed_contradiction_bundle_total"] += 1
                inputs = view.get("inputs", {})
                if inputs.get("artifact_ids") and inputs.get("chunk_ids"):
                    self._mom_stats["governed_source_trace_complete_total"] += 1
                self._audit_derived_view_generation(
                    view_type=str(view.get("view_type", "unknown")),
                    view_id=str(view.get("view_id", "")),
                    inputs=inputs,
                    query_fingerprint=str(view.get("query_fingerprint", "")),
                    governance_state_hash=str(view.get("governance_state_hash", "")),
                    metadata={"phase": "phase4" if phase4_cache_enabled else "phase3"},
                )

        elapsed = time.time() - t0
        if self._volatility_engine is not None:
            for r in results:
                self._volatility_engine.harvester.record_usage(r.engram)
            by_id = {r.engram.id: r.engram for r in results}
            for record in contradiction_records:
                family_key = "family:unknown"
                for eid in getattr(record, "candidate_memory_ids", []) or []:
                    if eid in by_id:
                        family_key = family_key_from_engram(by_id[eid])
                        break
                self._volatility_engine.harvester.record_contradiction(
                    family_key,
                    entity_key=getattr(record, "entity_key", ""),
                )
        envelope_meta = mode_meta.get("candidate_envelope") or {}
        env_initial = int(envelope_meta.get("initial_candidate_count", 0) or 0)
        env_final = int(envelope_meta.get("final_candidate_count", 0) or 0)
        self._audit("search", f"Search: '{query[:80]}' → {len(results)} results",
                     metadata={
                         "query": query,
                         "top_k": top_k,
                         "result_count": len(results),
                         "retrieval_mode": mode_meta.get("retrieval_mode", "semantic"),
                         "fusion_policy": mode_meta.get("fusion_policy"),
                         "governance_mode": selected_governance,
                     },
                     latency=elapsed)
        forecast_advisory = mode_meta.get("forecast_advisory") or {}
        if (
            getattr(self._config, "pulse_actions", "advisory") == "advisory"
            and forecast_advisory
            and forecast_advisory.get("suggested_plan") == "conservative"
        ):
            self._audit(
                "forecast_advisory",
                "Forecast predicts p95 breach; suggested shift to Conservative plan.",
                metadata={
                    "suggested_routing_plan": "conservative",
                    "forecast_reason": forecast_advisory.get("forecast_reason"),
                    "confidence_score": forecast_advisory.get("confidence_score"),
                    "forecast_pressure": forecast_advisory.get("forecast_pressure"),
                    "max_forecast_p95_latency_ms": forecast_advisory.get("max_forecast_p95_latency_ms"),
                    "max_forecast_degrade_count": forecast_advisory.get("max_forecast_degrade_count"),
                },
                latency=0.0,
            )
        self._pulse_engine.record_query(
            latency_ms=elapsed * 1000.0,
            cache_hits=query_cache_hits,
            cache_misses=query_cache_misses,
            degraded=self._status != "healthy",
            candidate_envelope=env_initial or env_final or len(results),
        )

        # ── Build per-result payload ───────────────────────────────────────
        decision_map = {d.engram_id: d for d in decisions}
        result_list = []
        include_lineage = bool(getattr(self._config, "memory_over_maps_phase1", False)) and selected_explain
        for idx, r in enumerate(results):
            entry: Dict = {
                "engram": r.engram.to_dict(include_lineage=include_lineage),
                "score": round(r.score, 4),
                "tier": r.tier,
                "tiers": r.metadata.get("tiers", [r.tier]),
            }
            if selected_explain and mode_meta.get("retrieval_mode") == "hybrid":
                entry.update({
                    "component_scores": r.metadata.get("component_scores"),
                    "retrieval_sources": r.metadata.get("retrieval_sources", []),
                    "filters_applied": r.metadata.get("filters_applied", filters or {}),
                    "fusion_policy": r.metadata.get("fusion_policy", selected_policy),
                })
            dec = decision_map.get(r.engram.id)
            if dec is not None and selected_governance != "off":
                entry["governed_score"] = round(dec.governed_score, 4)
                if selected_explain_gov:
                    entry["governance"] = dec.to_dict_full()
                    entry["governance_trace"] = self._build_governance_trace(
                        decision=dec,
                        raw_rank=raw_rank_by_id.get(r.engram.id),
                        final_rank=idx + 1,
                    )
            result_list.append(entry)

        payload = self._base_payload()
        payload["results"] = result_list
        payload["meta"] = {
            "query": query,
            "top_k": top_k,
            "result_count": len(results),
            "latency_s": round(elapsed, 3),
            "retrieval_mode": mode_meta.get("retrieval_mode", "semantic"),
            "fusion_policy": mode_meta.get("fusion_policy"),
            "lexical_lane_available": mode_meta.get("lexical_available", False),
            "explain": selected_explain,
        }
        envelope_ratio = round((env_final / env_initial), 4) if env_initial else 0.0
        cost_units = (
            env_initial
            + int(mode_meta.get("telemetry", {}).get("lexical_candidates", 0) or 0)
            + int(mode_meta.get("telemetry", {}).get("semantic_candidates", 0) or 0)
            + len(derived_views_payload) * 5
        )
        self._mom_stats["economics_query_count"] += 1
        self._mom_stats["economics_cost_units_total"] += int(cost_units)
        self._mom_stats["economics_envelope_initial_total"] += env_initial
        self._mom_stats["economics_envelope_final_total"] += env_final
        payload["meta"]["economics"] = {
            "candidate_envelope_initial": env_initial,
            "candidate_envelope_final": env_final,
            "candidate_envelope_compression_ratio": envelope_ratio,
            "derived_view_cache_hits": query_cache_hits,
            "derived_view_cache_misses": query_cache_misses,
            "estimated_cost_units": int(cost_units),
        }
        if mode_meta.get("telemetry"):
            payload["meta"]["hybrid_telemetry"] = mode_meta["telemetry"]
        if mode_meta.get("candidate_envelope"):
            payload["meta"]["candidate_envelope"] = mode_meta["candidate_envelope"]
        if mode_meta.get("forecast_advisory"):
            payload["meta"]["forecast_advisory"] = mode_meta["forecast_advisory"]
        if mode_meta.get("complexity_classification"):
            payload["meta"]["complexity_classification"] = mode_meta["complexity_classification"]
        if mode_meta.get("routing_posture"):
            payload["meta"]["routing_posture"] = mode_meta["routing_posture"]
        if mode_meta.get("complexity_shadow"):
            payload["meta"]["complexity_shadow"] = mode_meta["complexity_shadow"]
        if selected_governance != "off":
            payload["meta"]["governance_mode"] = selected_governance
            payload["meta"]["governance_profile"] = selected_profile or "default"
            payload["meta"]["governance_summary"] = {
                "candidates_evaluated": len(decisions),
                "vetoed": sum(1 for d in decisions if not d.veto_pass),
                "suppressed": sum(1 for d in decisions if d.suppressed),
                "contradictions_detected": len(contradiction_records),
                "contradiction_suppressed": sum(
                    1 for d in decisions if d.suppressed_by_contradiction
                ),
            }
            if selected_explain_gov:
                payload["meta"]["governance_explain"] = {
                    "suppressed_candidates": [
                        {
                            "engram_id": d.engram_id,
                            "reason": d.suppressed_reason or d.veto_reason or d.contradiction_reason,
                            "vetoed": not d.veto_pass,
                            "suppressed_by_contradiction": bool(d.suppressed_by_contradiction),
                            "contradiction_winner": d.contradiction_winner,
                            "governed_score": round(d.governed_score, 4),
                        }
                        for d in decisions
                        if d.would_be_suppressed_in_enforced_mode
                    ],
                    "counterfactuals": compute_counterfactuals(
                        decisions,
                        top_n=3,
                        min_score_threshold=self._governor.effective_min_score(
                            selected_profile or None
                        ) if self._governor else 0.0,
                        created_at_by_id={
                            r.engram.id: r.engram.created_at for r in results
                        },
                        freshness_half_life_days=self._governor.effective_freshness_half_life(
                            selected_profile or None
                        ) if self._governor else 180.0,
                    ),
                }
        if derived_views_payload:
            payload["derived_views"] = derived_views_payload
        if self._intent_engine is not None:
            forecast = self._intent_engine.record_and_forecast(
                session_id=session_id,
                query=query,
            )
            if forecast:
                payload["meta"]["intent_forecast"] = forecast
            if self._intent_engine.last_shadow_result:
                payload["meta"]["shadow_search"] = self._intent_engine.last_shadow_result
        return payload

    def get_pulse(self, observed_limit: int = 60) -> Dict[str, Any]:
        """Return observed MNEMOS pulse and advisory forecast if enabled."""
        horizon = int(getattr(self._config, "pulse_horizon_minutes", 15)) if self._config else 15
        actions_mode = getattr(self._config, "pulse_actions", "advisory") if self._config else "advisory"
        forecast = self._pulse_engine.refresh_forecast() if self._pulse_engine.provider is not None else None
        autonomous_action = self._pulse_engine.evaluate_and_trigger(forecast)
        payload = build_pulse_payload(
            contract_version=CONTRACT_VERSION,
            status=self._status,
            source="mnemos-service",
            generated_at=_utc_now(),
            harvester=self._pulse_engine.harvester,
            provider=None,
            observed_limit=observed_limit,
            horizon_minutes=horizon,
            actions_mode=actions_mode,
            error=self._error,
        )
        payload["forecast"] = forecast
        payload["autonomous_action"] = autonomous_action
        return payload

    def search_derived_trial(self, query: str, top_k: int, client_id: str) -> Dict[str, Any]:
        """Expose feature-flagged derived facts safely via api.py."""
        from mnemos.retrieval.api import search_derived_trial as execute_search_derived_trial
        from mnemos.retrieval.auditor import EvaluationAuditor
        auditor = EvaluationAuditor()
        return execute_search_derived_trial(self._router, query, top_k, client_id, auditor)

    def get_engram(self, engram_id: str) -> Dict[str, Any]:
        """Retrieve a specific engram by ID."""
        payload = self._base_payload()
        for tier in (self._semantic_fusion._tiers if self._semantic_fusion else []):
            engram = tier.get(engram_id)
            if engram:
                payload["engram"] = engram.to_dict()
                return payload
        if self._lexical_tier:
            engram = self._lexical_tier.get(engram_id)
            if engram:
                payload["engram"] = engram.to_dict()
                return payload
        payload["error"] = f"Engram {engram_id} not found"
        payload["status"] = "degraded"
        return payload

    def delete_engram(self, engram_id: str) -> Dict[str, Any]:
        """Delete an engram from all tiers."""
        counts = self._semantic_fusion.delete([engram_id])
        if self._lexical_tier:
            counts["lexical"] = self._lexical_tier.delete([engram_id])
        self._audit("delete", f"Deleted engram {engram_id}", metadata={"tiers": counts})

        payload = self._base_payload()
        payload["result"] = {"deleted_id": engram_id, "tiers": counts}
        return payload

    def get_audit(self, limit: int, query: Optional[str]) -> Dict[str, Any]:
        """Query the forensic ledger."""
        payload = self._base_payload()
        if not self._ledger:
            payload["error"] = "Audit ledger is disabled"
            return payload

        if query:
            payload["traces"] = self._ledger.search_traces(query, limit=limit)
        else:
            payload["transactions"] = self._ledger.get_recent_transactions(limit=limit)
        payload["performance"] = self._ledger.get_performance_summary()
        return payload

    def get_governance_stats(self) -> Dict[str, Any]:
        """Return aggregate governance statistics."""
        payload = self._base_payload()
        if self._governor is None:
            payload["error"] = "Governance layer not initialized"
            return payload
        payload["governance"] = self._governor.stats()
        return payload

    def governance_reflect(
        self,
        query: str,
        answer: str,
        candidates: List[Dict],
        cited_ids: Optional[List[str]] = None,
        governance_mode: str = "advisory",
        governance_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the Wave 3 reflect loop for a completed query/answer pair.

        ``candidates`` must be a list of engram dicts as returned by
        /search (each entry needs at least ``id`` and ``content``; include
        ``_governance`` for full reinforcement accuracy).

        Governance metadata on the in-memory Engram objects is updated.
        Callers that want durability should re-index the updated engrams.
        """
        from mnemos.engram.model import Engram
        from mnemos.retrieval.base import SearchResult
        from mnemos.governance.models.governance_decision import GovernanceDecision

        payload = self._base_payload()
        if self._governor is None:
            payload["error"] = "Governance layer not initialized"
            return payload

        # Reconstruct lightweight SearchResult objects from the payload
        results: List[SearchResult] = []
        for c in candidates:
            engram_dict = dict(c)
            score = float(engram_dict.pop("score", 0.5))
            tier = engram_dict.pop("tier", "reflect")
            engram = Engram.from_dict(engram_dict)
            results.append(SearchResult(engram=engram, score=score, tier=tier))

        # Build minimal GovernanceDecisions from GovernanceMeta state
        decisions: List[GovernanceDecision] = []
        for r in results:
            gov = r.engram.governance
            veto_pass = True
            suppressed_by_contradiction = False
            if gov is not None:
                veto_pass = (
                    gov.deletion_state not in ("soft_deleted", "tombstone")
                    and "toxic" not in gov.policy_flags
                )
                suppressed_by_contradiction = gov.conflict_status == "suppressed"
            decisions.append(
                GovernanceDecision(
                    engram_id=r.engram.id,
                    retrieval_score=r.score,
                    governed_score=r.score,
                    veto_pass=veto_pass,
                    suppressed=(not veto_pass) or suppressed_by_contradiction,
                    suppressed_by_contradiction=suppressed_by_contradiction,
                    conflict_status=gov.conflict_status if gov else "none",
                )
            )

        reflect_result = self._governor.reflect(
            query=query,
            answer=answer,
            results=results,
            decisions=decisions,
            cited_ids=cited_ids,
            governance_mode=governance_mode,
            governance_profile=governance_profile,
        )

        payload["reflect"] = reflect_result.to_dict()
        return payload

    def get_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        payload = self._base_payload()
        retrieval_stats = self._semantic_fusion.stats() if self._semantic_fusion else {}
        router_stats = self._router.stats() if self._router else {}
        lexical_stats = self._lexical_tier.stats() if self._lexical_tier else {"available": False}

        retrieval_stats["hybrid"] = router_stats
        retrieval_stats["lexical_lane"] = lexical_stats
        payload["stats"] = {
            "retrieval": retrieval_stats,
            "compression": {
                "enabled": self._config.has_compression if self._config else False,
                "bits": self._config.quant_bits if self._config else 0,
                "algorithm": "TurboQuant (arXiv:2504.19874)",
            },
            "audit": self._ledger.get_stats() if self._ledger else {"enabled": False},
            "memory_over_maps": dict(self._mom_stats),
            "derived_lane": {
                "query.default_retrieval.derived_fact_count": self._mom_stats.get("query.default_retrieval.derived_fact_count", 0),
                "echoframe.production_prompt.derived_count": self._mom_stats.get("echoframe.production_prompt.derived_count", 0),
                "derived_lane.execution_count": self._mom_stats.get("derived_lane.execution_count", 0),
                "derived_lane.denied_count": self._mom_stats.get("derived_lane.denied_count", 0),
                "derived_lane.kill_switch_count": self._mom_stats.get("derived_lane.kill_switch_count", 0),
                "evaluate_derived_shadow.request_count": self._mom_stats.get("evaluate_derived_shadow.request_count", 0),
                "evaluate_derived_shadow.denied_count": self._mom_stats.get("evaluate_derived_shadow.denied_count", 0),
                "evaluate_derived_shadow.rendered_derived_fact_count": self._mom_stats.get("evaluate_derived_shadow.rendered_derived_fact_count", 0),
            }
        }
        if self._view_cache is not None:
            payload["stats"]["memory_over_maps"]["derived_view_cache"] = self._view_cache.stats()
        query_count = max(1, self._mom_stats.get("economics_query_count", 0))
        env_initial_total = self._mom_stats.get("economics_envelope_initial_total", 0)
        env_final_total = self._mom_stats.get("economics_envelope_final_total", 0)
        payload["stats"]["economics"] = {
            "query_count": self._mom_stats.get("economics_query_count", 0),
            "avg_estimated_cost_units_per_query": round(
                self._mom_stats.get("economics_cost_units_total", 0) / query_count, 4
            ),
            "envelope_initial_total": env_initial_total,
            "envelope_final_total": env_final_total,
            "envelope_compression_ratio": round(
                (env_final_total / env_initial_total), 4
            ) if env_initial_total else 0.0,
            "cache_hit_total": self._mom_stats.get("derived_view_cache_hit_count", 0),
            "cache_miss_total": self._mom_stats.get("derived_view_cache_miss_count", 0),
            "cache_hit_ratio": round(
                self._mom_stats.get("derived_view_cache_hit_count", 0)
                / max(1, self._mom_stats.get("derived_view_cache_hit_count", 0) + self._mom_stats.get("derived_view_cache_miss_count", 0)),
                4,
            ),
            "invalidation_event_count": self._mom_stats.get("derived_view_invalidation_events", 0),
            "invalidation_fanout_total": self._mom_stats.get("derived_view_invalidation_fanout_total", 0),
            "avg_invalidation_fanout": round(
                self._mom_stats.get("derived_view_invalidation_fanout_total", 0)
                / max(1, self._mom_stats.get("derived_view_invalidation_events", 0)),
                4,
            ),
        }
        return payload

    def has_governance_profile(self, profile_id: str) -> bool:
        if self._governor is None:
            return False
        return self._governor.has_policy_profile(profile_id)

    def governance_profiles(self) -> List[str]:
        if self._governor is None:
            return ["default"]
        return self._governor.policy_profile_ids()

    def invalidate_derived_view_cache(
        self,
        *,
        event_type: str,
        refs: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Invalidate derived-view cache entries by dependency event."""
        if self._view_cache is None:
            return {"error": "derived view cache unavailable"}
        trace = self._view_cache.invalidate(event_type=event_type, refs=refs, dry_run=dry_run)
        self._mom_stats["derived_view_invalidation_events"] += 1
        self._mom_stats["derived_view_invalidation_fanout_total"] += len(trace.get("impacted_keys", []))
        if not dry_run:
            self._mom_stats["derived_view_invalidated_count"] += len(trace.get("impacted_keys", []))
        return trace


# ──────────────────── Singleton ────────────────────

_runtime = MnemosRuntime()


def _ensure_runtime():
    try:
        _runtime.initialize()
        return None
    except Exception as e:
        return {
            "contract_version": CONTRACT_VERSION,
            "status": "unavailable",
            "source": "mnemos-service",
            "generated_at": _utc_now(),
            "error": str(e),
        }


# ──────────────────── Routes ────────────────────


@app.get("/health")
def health():
    err = _ensure_runtime()
    if err:
        return jsonify({"status": "degraded", "service": "mnemos-service"}), 200
    return jsonify({"status": "ok", "service": "mnemos-service", "contract_version": CONTRACT_VERSION}), 200


@app.get("/")
def root():
    return jsonify({
        "service": "mnemos-service",
        "status": "ok",
        "contract_version": CONTRACT_VERSION,
            "routes": {
                "health": "/health",
                "capabilities": "/v1/mnemos/capabilities",
                "warmup": "/v1/mnemos/warmup",
                "pulse": "/v1/mnemos/pulse",
                "index": "/v1/mnemos/index",
            "search": "/v1/mnemos/search",
            "engrams": "/v1/mnemos/engrams/{id}",
            "audit": "/v1/mnemos/audit",
            "stats": "/v1/mnemos/stats",
            "governance_stats": "/v1/mnemos/governance/stats",
            "governance_reflect": "/v1/mnemos/governance/reflect",
        },
    }), 200


@app.get("/v1/mnemos/capabilities")
def capabilities():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200
    return jsonify(_runtime.capabilities()), 200


@app.post("/v1/mnemos/warmup")
def warmup():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200

    body = request.get_json(silent=True) or {}
    query = body.get("query", "mnemos warmup readiness probe")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "query must be a non-empty string"}), 400
    return jsonify(_runtime.warmup(query=query.strip())), 200


@app.post("/v1/mnemos/index")
def index():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200

    body = request.get_json(silent=True) or {}
    documents = body.get("documents", [])
    options = body.get("options", {})

    if not documents:
        return jsonify({"error": "No documents provided"}), 400

    return jsonify(_runtime.index_documents(documents, options)), 200


@app.post("/api/v1/query")
@app.post("/v1/mnemos/search")
def search():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200

    body = request.get_json(silent=True) or {}
    
    if body.get("evaluation_mode") is True:
        _runtime._mom_stats["derived_lane.denied_count"] += 1
        return jsonify({"error": "evaluation_mode=true not supported on production routes"}), 400

    query = body.get("query", "")
    top_k = body.get("top_k", 10)
    tiers = body.get("tiers")
    filters = body.get("filters")
    retrieval_mode = body.get("retrieval_mode")
    fusion_policy = body.get("fusion_policy")
    explain = body.get("explain")
    governance = body.get("governance")
    explain_governance = body.get("explain_governance")
    governance_profile = body.get("governance_profile")
    bounded_envelope = body.get("bounded_envelope")
    derive_views = body.get("derive_views")
    latency_budget_ms = body.get("latency_budget_ms")
    complexity_shadow = body.get("complexity_shadow", False)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if retrieval_mode is not None and retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        return jsonify({
            "error": "Invalid retrieval_mode",
            "supported_retrieval_modes": sorted(SUPPORTED_RETRIEVAL_MODES),
        }), 400

    if fusion_policy is not None and fusion_policy not in FUSION_POLICIES:
        return jsonify({
            "error": "Invalid fusion_policy",
            "supported_fusion_policies": sorted(FUSION_POLICIES.keys()),
        }), 400

    if explain is not None and not isinstance(explain, bool):
        return jsonify({"error": "explain must be a boolean"}), 400

    if not isinstance(complexity_shadow, bool):
        return jsonify({"error": "complexity_shadow must be a boolean"}), 400

    if governance is not None and governance not in GOVERNANCE_MODES:
        return jsonify({
            "error": "Invalid governance",
            "supported_governance_modes": sorted(GOVERNANCE_MODES),
        }), 400

    if explain_governance is not None and not isinstance(explain_governance, bool):
        return jsonify({"error": "explain_governance must be a boolean"}), 400

    if governance_profile is not None:
        if not isinstance(governance_profile, str) or not governance_profile.strip():
            return jsonify({"error": "governance_profile must be a non-empty string"}), 400
        if not _runtime.has_governance_profile(governance_profile.strip()):
            return jsonify({
                "error": "Invalid governance_profile",
                "supported_governance_profiles": _runtime.governance_profiles(),
            }), 400

    if bounded_envelope is not None and not isinstance(bounded_envelope, dict):
        return jsonify({"error": "bounded_envelope must be an object"}), 400

    if filters is not None:
        if not isinstance(filters, dict):
            return jsonify({"error": "filters must be an object"}), 400
        reserved = sorted(k for k in filters if k in RESERVED_FILTER_KEYS)
        if reserved:
            return jsonify({
                "error": "Reserved filter key",
                "reserved_filter_keys": reserved,
            }), 400

    if latency_budget_ms is not None:
        if isinstance(latency_budget_ms, bool) or not isinstance(latency_budget_ms, (int, float)) or latency_budget_ms <= 0:
            return jsonify({"error": "latency_budget_ms must be a positive number"}), 400
        latency_budget_ms = float(latency_budget_ms)

    if derive_views is not None:
        if not isinstance(derive_views, list) or any(not isinstance(v, str) for v in derive_views):
            return jsonify({"error": "derive_views must be a list of strings"}), 400
        invalid = [v for v in derive_views if v not in SUPPORTED_DERIVED_VIEWS]
        if invalid:
            return jsonify(
                {
                    "error": "Invalid derive_views entries",
                    "supported_derived_views": sorted(SUPPORTED_DERIVED_VIEWS),
                    "invalid": invalid,
                }
            ), 400

    if body.get("enable_derived_facts") is True:
        config = get_config()
        if not config.derived_enabled:
            _runtime._mom_stats["derived_lane.kill_switch_count"] += 1
            return jsonify({"error": "derived_lane_disabled"}), 503
            
        client_id = request.headers.get("X-Client-Id", "unknown")
        if client_id not in config.derived_whitelist:
            _runtime._mom_stats["derived_lane.denied_count"] += 1
            return jsonify({"error": "client_not_authorized"}), 403
            
        _runtime._mom_stats["derived_lane.execution_count"] += 1
        res = _runtime.search_derived_trial(query=query, top_k=top_k, client_id=client_id)
        return jsonify(res), 200

    res = _runtime.search_documents(
            query,
            top_k,
            tiers,
            filters,
            retrieval_mode,
            fusion_policy,
            explain,
            governance,
            explain_governance,
            governance_profile,
            bounded_envelope,
            derive_views,
            latency_budget_ms,
            complexity_shadow,
        )

    derived_cnt = len(res.get("derived_results", []))
    if derived_cnt > 0:
        _runtime._mom_stats["query.default_retrieval.derived_fact_count"] += derived_cnt
        raise RuntimeError("SEV-STOP: query.default_retrieval.derived_fact_count > 0")
    else:
        # Implicitly 0, as per standing invariant
        pass

    return jsonify(res), 200


@app.post("/api/v1/evaluate_derived_shadow")
def evaluate_derived_shadow():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200
    t_start = time.perf_counter()

    _runtime._mom_stats["evaluate_derived_shadow.request_count"] += 1

    config = get_config()

    # 1. Kill-Switch
    if not config.derived_enabled:
        _runtime._mom_stats["derived_lane.kill_switch_count"] += 1
        return jsonify({"error": "derived_lane_disabled"}), 503

    # 2. Client Whitelist Check
    t_auth_start = time.perf_counter()
    client_id = request.headers.get("X-Client-Id", "unknown")
    if client_id not in config.derived_whitelist:
        _runtime._mom_stats["derived_lane.denied_count"] += 1
        _runtime._mom_stats["evaluate_derived_shadow.denied_count"] += 1
        return jsonify({"error": "derived_fact_client_not_authorized"}), 403
    t_auth = (time.perf_counter() - t_auth_start) * 1000

    body = request.get_json(silent=True) or {}

    # 3. Double Opt-In Flags
    if not body.get("evaluation_mode") or not body.get("include_derived_facts"):
        _runtime._mom_stats["derived_lane.denied_count"] += 1
        _runtime._mom_stats["evaluate_derived_shadow.denied_count"] += 1
        return jsonify({"error": "missing_required_eval_flags"}), 400

    # Passed gates!
    _runtime._mom_stats["derived_lane.execution_count"] += 1

    query = body.get("query", "")
    top_k = body.get("top_k", 10)

    # Call default search, assert 0 derived
    t_default_search_start = time.perf_counter()
    res = _runtime.search_documents(query, top_k, None, None, None, None, None, None, None, None, None, None)
    if "derived_results" in res and len(res["derived_results"]) > 0:
        raise RuntimeError("SEV-STOP: Default search leaked derived facts!")
    t_default_search = (time.perf_counter() - t_default_search_start) * 1000

    # Perform Shadow Fetch
    t_search_derived_start = time.perf_counter()
    pit2_response = _runtime._router.search_derived(
        query=query, 
        top_k=top_k,
        client_id=client_id,
        include_derived_facts=body.get("include_derived_facts", True)
    )
    t_search_derived = (time.perf_counter() - t_search_derived_start) * 1000
    
    # Governance Ledger Check (currently bundled in search, but we log 0 for now or small overhead)
    t_gov = 0.0

    t_serializer_start = time.perf_counter()
    from mnemos.evaluation.derived_shadow_packet import DerivedShadowPacketSerializer
    serializer = DerivedShadowPacketSerializer()
    shadow_packet = serializer.serialize(pit2_response)
    t_serializer = (time.perf_counter() - t_serializer_start) * 1000

    # Render it
    t_renderer_start = time.perf_counter()
    from mnemos.evaluation.derived_evaluation_renderer import render_derived_evaluation_context
    rendered_block = render_derived_evaluation_context(shadow_packet)
    t_renderer = (time.perf_counter() - t_renderer_start) * 1000

    t_telemetry_start = time.perf_counter()
    rendered_cnt = shadow_packet.get("derived_fact_count", 0)
    _runtime._mom_stats["evaluate_derived_shadow.rendered_derived_fact_count"] += rendered_cnt
    t_telemetry = (time.perf_counter() - t_telemetry_start) * 1000

    t_total = (time.perf_counter() - t_start) * 1000

    stage_latencies_ms = {
        "auth_whitelist_check_ms": round(t_auth, 2),
        "default_search_ms": round(t_default_search, 2),
        "search_derived_ms": round(t_search_derived, 2),
        "governance_ledger_check_ms": round(t_gov, 2),
        "shadow_serializer_ms": round(t_serializer, 2),
        "evaluation_renderer_ms": round(t_renderer, 2),
        "telemetry_stats_update_ms": round(t_telemetry, 2),
        "response_serialization_ms": 0.0, # Will be set implicitly or ignored as Flask handles jsonification
        "total_request_ms": round(t_total, 2)
    }

    res["shadow_evaluation"] = {
        "rendered_block": rendered_block,
        "shadow_packet": shadow_packet,
        "stage_latencies_ms": stage_latencies_ms,
        "candidate_telemetry": pit2_response.get("derived_lane_meta", {}).get("candidate_telemetry", [])
    }

    return jsonify(res), 200

@app.get("/v1/mnemos/engrams/<engram_id>")
def get_engram(engram_id):
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200
    return jsonify(_runtime.get_engram(engram_id)), 200


@app.delete("/v1/mnemos/engrams/<engram_id>")
def delete_engram(engram_id):
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200
    return jsonify(_runtime.delete_engram(engram_id)), 200


@app.get("/v1/mnemos/audit")
def audit():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200

    limit = int(request.args.get("limit", "50"))
    query = request.args.get("q")
    return jsonify(_runtime.get_audit(limit, query)), 200


@app.get("/v1/mnemos/pulse")
def pulse():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200

    try:
        limit = int(request.args.get("limit", "60"))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    if limit < 1 or limit > 1440:
        return jsonify({"error": "limit must be between 1 and 1440"}), 400
    return jsonify(_runtime.get_pulse(observed_limit=limit)), 200


@app.get("/v1/mnemos/stats")
def stats():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200
    return jsonify(_runtime.get_stats()), 200


@app.get("/v1/mnemos/governance/stats")
def governance_stats():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200
    return jsonify(_runtime.get_governance_stats()), 200


@app.post("/v1/mnemos/governance/reflect")
def governance_reflect():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200

    body = request.get_json(silent=True) or {}

    query = body.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "query is required and must be a non-empty string"}), 400

    answer = body.get("answer", "")
    if not isinstance(answer, str):
        return jsonify({"error": "answer must be a string"}), 400

    candidates = body.get("candidates")
    if not isinstance(candidates, list):
        return jsonify({"error": "candidates must be a list of engram objects"}), 400

    cited_ids = body.get("cited_ids")
    if cited_ids is not None and not isinstance(cited_ids, list):
        return jsonify({"error": "cited_ids must be a list of strings"}), 400

    governance_mode = body.get("governance_mode", "advisory")
    if governance_mode not in ("off", "advisory", "enforced"):
        return jsonify({"error": "Invalid governance_mode"}), 400
    governance_profile = body.get("governance_profile")
    if governance_profile is not None:
        if not isinstance(governance_profile, str) or not governance_profile.strip():
            return jsonify({"error": "governance_profile must be a non-empty string"}), 400
        if not _runtime.has_governance_profile(governance_profile.strip()):
            return jsonify({
                "error": "Invalid governance_profile",
                "supported_governance_profiles": _runtime.governance_profiles(),
            }), 400

    return jsonify(
        _runtime.governance_reflect(
            query=query,
            answer=answer,
            candidates=candidates,
            cited_ids=cited_ids,
            governance_mode=governance_mode,
            governance_profile=governance_profile.strip() if isinstance(governance_profile, str) else None,
        )
    ), 200


# ──────────────────── Entry point ────────────────────

if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info(f"🚀 Starting MNEMOS service on port {config.port}")
    app.run(host="0.0.0.0", port=config.port, debug=False)
