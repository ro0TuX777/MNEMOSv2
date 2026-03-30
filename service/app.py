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
from mnemos.governance.governor import Governor
from mnemos.governance.read_path import GOVERNANCE_MODES

logger = logging.getLogger("mnemos.service")

CONTRACT_VERSION = "v1"
SUPPORTED_RETRIEVAL_MODES = {"semantic", "hybrid"}

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
        self._view_cache: Optional[DerivedViewCache] = None
        self._status = "healthy"
        self._error: Optional[str] = None
        self._mom_stats: Dict[str, int] = {
            "derived_view_generated_count": 0,
            "derived_view_cache_hit_count": 0,
            "derived_view_cache_miss_count": 0,
            "derived_view_invalidated_count": 0,
            "governed_evidence_bundle_total": 0,
            "governed_contradiction_bundle_total": 0,
            "governed_source_trace_complete_total": 0,
        }

    def initialize(self):
        if self._initialized:
            return

        try:
            self._config = get_config()

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
                tiers.append(LanceDBTier(db_dir=self._config.lance_dir))

            if self._config.has_colbert:
                from mnemos.retrieval.colbert_tier import ColBERTTier, ColBERTConfig
                tiers.append(ColBERTTier(ColBERTConfig(
                    model_name=self._config.colbert_model,
                    quantize_bits=self._config.quant_bits,
                    index_dir=self._config.colbert_index_dir,
                )))

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

            self._router = RetrievalRouter(
                semantic_fusion=self._semantic_fusion,
                lexical_tier=self._lexical_tier,
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
            self._governor = Governor(
                min_score_threshold=self._config.governance_min_score,
                freshness_half_life_days=self._config.governance_freshness_half_life,
            )
            self._view_cache = DerivedViewCache(ttl_seconds=3600)

            self._initialized = True
            logger.info(
                f"🚀 MNEMOS runtime initialized: semantic_tiers={self._semantic_fusion.tier_names}, "
                f"lexical_available={bool(self._lexical_tier)}, "
                f"governance_mode={self._config.governance_mode}"
            )

        except Exception as e:
            self._status = "unavailable"
            self._error = str(e)
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
            "supports": ["index", "search", "engrams", "audit", "stats"],
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
        bounded_envelope: Optional[Dict[str, Any]] = None,
        derive_views: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search across tiers and return fused results."""
        import time
        t0 = time.time()

        selected_mode = retrieval_mode or self._config.retrieval_mode
        selected_policy = fusion_policy or self._config.fusion_policy
        selected_explain = self._config.explain_default if explain is None else bool(explain)
        selected_governance = governance or getattr(self._config, "governance_mode", "off")
        selected_explain_gov = bool(explain_governance) if explain_governance is not None else False

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
        )

        # ── Governance ────────────────────────────────────────────────────
        decisions = []
        contradiction_records = []
        if selected_governance != "off" and self._governor:
            results, decisions, contradiction_records = self._governor.govern(
                results=results,
                query=query,
                governance_mode=selected_governance,
                top_k=top_k,
            )

        derived_views_payload: List[Dict[str, Any]] = []
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
                )
                if phase4_cache_enabled:
                    cached = self._view_cache.get(cache_key)
                    if cached is not None:
                        self._mom_stats["derived_view_cache_hit_count"] += 1
                        cached["_cache"] = {"hit": True, "key": cache_key}
                        view_payload = cached
                    else:
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

        # ── Build per-result payload ───────────────────────────────────────
        decision_map = {d.engram_id: d for d in decisions}
        result_list = []
        include_lineage = bool(getattr(self._config, "memory_over_maps_phase1", False)) and selected_explain
        for r in results:
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
        if mode_meta.get("telemetry"):
            payload["meta"]["hybrid_telemetry"] = mode_meta["telemetry"]
        if mode_meta.get("candidate_envelope"):
            payload["meta"]["candidate_envelope"] = mode_meta["candidate_envelope"]
        if selected_governance != "off":
            payload["meta"]["governance_mode"] = selected_governance
            payload["meta"]["governance_summary"] = {
                "candidates_evaluated": len(decisions),
                "vetoed": sum(1 for d in decisions if not d.veto_pass),
                "suppressed": sum(1 for d in decisions if d.suppressed),
                "contradictions_detected": len(contradiction_records),
                "contradiction_suppressed": sum(
                    1 for d in decisions if d.suppressed_by_contradiction
                ),
            }
        if derived_views_payload:
            payload["derived_views"] = derived_views_payload
        return payload

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
        }
        if self._view_cache is not None:
            payload["stats"]["memory_over_maps"]["derived_view_cache"] = self._view_cache.stats()
        return payload

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


@app.post("/v1/mnemos/search")
def search():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    err = _ensure_runtime()
    if err:
        return jsonify(err), 200

    body = request.get_json(silent=True) or {}
    query = body.get("query", "")
    top_k = body.get("top_k", 10)
    tiers = body.get("tiers")
    filters = body.get("filters")
    retrieval_mode = body.get("retrieval_mode")
    fusion_policy = body.get("fusion_policy")
    explain = body.get("explain")
    governance = body.get("governance")
    explain_governance = body.get("explain_governance")
    bounded_envelope = body.get("bounded_envelope")
    derive_views = body.get("derive_views")

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

    if governance is not None and governance not in GOVERNANCE_MODES:
        return jsonify({
            "error": "Invalid governance",
            "supported_governance_modes": sorted(GOVERNANCE_MODES),
        }), 400

    if explain_governance is not None and not isinstance(explain_governance, bool):
        return jsonify({"error": "explain_governance must be a boolean"}), 400

    if bounded_envelope is not None and not isinstance(bounded_envelope, dict):
        return jsonify({"error": "bounded_envelope must be an object"}), 400

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

    return jsonify(
        _runtime.search_documents(
            query,
            top_k,
            tiers,
            filters,
            retrieval_mode,
            fusion_policy,
            explain,
            governance,
            explain_governance,
            bounded_envelope,
            derive_views,
        )
    ), 200


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

    return jsonify(
        _runtime.governance_reflect(
            query=query,
            answer=answer,
            candidates=candidates,
            cited_ids=cited_ids,
            governance_mode=governance_mode,
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
