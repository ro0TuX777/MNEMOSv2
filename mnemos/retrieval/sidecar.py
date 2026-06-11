import os
from typing import Dict, Any, List, Tuple, Optional
from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.extraction.promotion_engine import PromotionEngine
from mnemos.retrieval.shadow_packet import ShadowPacketBuilder

class ShadowModeDisabledError(Exception):
    pass

class FactAwareEvaluationSidecar:
    """Orchestrator for Fact-Aware Evaluation Mode. Operates explicitly outside core paths."""
    
    def __init__(self, baseline_retriever: RetrievalRouter, promotion_engine: PromotionEngine):
        self.baseline_retriever = baseline_retriever
        self.promotion_engine = promotion_engine
        
    def execute_fact_aware_query(
        self,
        query: str,
        top_k: int,
        operator_override: bool = False,
        enable_fact_awareness: bool = False,
        governance_overrides: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        # 1. Kill Switch
        if os.environ.get("VFR_DISABLE_SHADOW_MODE", "false").lower() == "true":
            raise ShadowModeDisabledError("Shadow mode explicitly disabled via kill switch.")
            
        # 2. Baseline fetch
        b_res, b_meta = self.baseline_retriever.search(query=query, top_k=top_k, **kwargs)
        
        # 3. Double Opt-In enforcement
        if not operator_override or not enable_fact_awareness:
            telemetry = {
                "sidecar_active": False,
                "derived_fact_count": 0,
                "double_opt_in_satisfied": False,
                "masked_fact_count": 0,
                "baseline_retrieval_mode": b_meta.get("retrieval_mode")
            }
            packet = ShadowPacketBuilder.build(query, b_res, [])
            return packet, telemetry
            
        # 4. Fetch and mask Derived Facts
        validated_facts = self.promotion_engine.fetch_validated_facts()
        eligible = []
        masked_count = 0
        
        for chain in validated_facts:
            src_id = chain["candidate_fact"]["source_engram_id"]
            if governance_overrides and src_id in governance_overrides and governance_overrides[src_id] in ("suppressed", "deleted", "expired", "vetoed", "tombstoned"):
                masked_count += 1
                continue
            eligible.append(chain)
            
        # 5. Build isolated packet
        packet = ShadowPacketBuilder.build(query, b_res, eligible)
        
        telemetry = {
            "sidecar_active": True,
            "derived_fact_count": len(eligible),
            "double_opt_in_satisfied": True,
            "masked_fact_count": masked_count,
            "baseline_retrieval_mode": b_meta.get("retrieval_mode")
        }
        
        return packet, telemetry
