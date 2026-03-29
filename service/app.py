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
        self._status = "healthy"
        self._error: Optional[str] = None

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
        for r in results:
            entry: Dict = {
                "engram": r.engram.to_dict(),
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
        }
        return payload


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


# ──────────────────── Entry point ────────────────────

if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info(f"🚀 Starting MNEMOS service on port {config.port}")
    app.run(host="0.0.0.0", port=config.port, debug=False)
