import os
import json
from typing import Dict, Any, List, Tuple, Optional

from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.extraction.promotion_engine import PromotionEngine

class ShadowModeDisabledError(Exception):
    pass

class ShadowPacketRenderer:
    """Offline renderer for formatting EchoFrame-compatible evaluation packets."""
    
    @staticmethod
    def render_packet(
        query: str,
        primary_results: List[SearchResult],
        derived_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Renders a packet strictly separating Primary Engrams from Derived FactNodes.
        """
        packet = {
            "query": query,
            "context": []
        }
        
        # Primary Engrams section
        packet["context"].append("<Primary_Engrams>")
        for res in primary_results:
            packet["context"].append(f"[Engram_ID: {res.engram.id}]\n{res.engram.content}")
        packet["context"].append("</Primary_Engrams>")
        
        # Derived FactNodes section
        if derived_facts:
            packet["context"].append("\n<Derived_FactNodes>")
            for chain in derived_facts:
                fact = chain["candidate_fact"]
                receipt = chain["promotion_receipt"]
                packet["context"].append(
                    f"[Derived FactNode] (Derived from Engram: {fact['source_engram_id']})\n"
                    f"Statement: {fact['statement']}\n"
                    f"Lineage: source_engram_id={fact['source_engram_id']} | "
                    f"passage_node_id={fact['passage_node_id']} | "
                    f"promotion_receipt_id={receipt['receipt_id']}"
                )
            packet["context"].append("</Derived_FactNodes>")
            
        return packet


class ValidatedFactShadowRetriever:
    """Offline wrapper mapping baseline retrieval against validated candidate facts."""
    
    def __init__(self, baseline_retriever: RetrievalRouter, promotion_engine: PromotionEngine):
        self.baseline_retriever = baseline_retriever
        self.promotion_engine = promotion_engine
        
    def _apply_readonly_governance_override(self, facts: List[Dict[str, Any]], overrides: Dict[str, str]) -> List[Dict[str, Any]]:
        eligible = []
        for chain in facts:
            src_id = chain["source_engram_id"]
            if src_id in overrides and overrides[src_id] in ("suppressed", "deleted", "expired", "vetoed", "tombstoned"):
                continue
            eligible.append(chain)
        return eligible

    def search_shadow_mode(
        self,
        query: str,
        top_k: int,
        runtime_flag: bool = False,
        eval_config_flag: bool = False,
        governance_overrides: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[List[SearchResult], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes a shadow run enforcing the Double Opt-In and Kill Switch.
        """
        # Kill Switch Enforcement
        if os.environ.get("VFR_DISABLE_SHADOW_MODE", "false").lower() == "true":
            raise ShadowModeDisabledError("Shadow mode is globally disabled via kill switch.")
            
        # Double Opt-In Enforcement
        if not (runtime_flag and eval_config_flag):
            # Fail closed: run standard baseline only, return empty derived facts.
            primary_results, meta = self.baseline_retriever.search(query=query, top_k=top_k, **kwargs)
            meta["shadow_mode_active"] = False
            meta["derived_fact_count"] = 0
            return primary_results, [], meta
            
        # 1. Baseline Search
        primary_results, meta = self.baseline_retriever.search(query=query, top_k=top_k, **kwargs)
        
        # 2. Validated Fact Shadow Join
        validated_facts = self.promotion_engine.fetch_validated_facts()
        
        if governance_overrides:
            validated_facts = self._apply_readonly_governance_override(validated_facts, governance_overrides)
            
        # Optional: Narrow facts by semantic distance to query? For this wrapper, we just return the eligible.
        # In a real system, you'd embed the fact node. For VFR-1, we pass the active validated facts.
        
        meta["shadow_mode_active"] = True
        meta["derived_fact_count"] = len(validated_facts)
        
        return primary_results, validated_facts, meta
