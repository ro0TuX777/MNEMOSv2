import os
from typing import Dict, Any, List
from mnemos.retrieval.base import SearchResult

class ShadowPacketBuilder:
    """Disjoint packet builder for EchoFrame-compatible structural isolation."""
    
    @staticmethod
    def build(query: str, primary_results: List[SearchResult], derived_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        context = []
        
        # Primary Engrams
        context.append("<Primary_Engrams>")
        for res in primary_results:
            context.append(f"[Engram_ID: {res.engram.id}]\n{res.engram.content}")
        context.append("</Primary_Engrams>")
        
        # Derived Facts
        if derived_facts:
            context.append("\n<Derived_FactNodes>")
            context.append("WARNING: THIS PACKET CONTAINS DERIVED FACT NODES. THESE ARE SECONDARY TO PRIMARY ENGRAMS.")
            for chain in derived_facts:
                fact = chain["candidate_fact"]
                receipt = chain["promotion_receipt"]
                context.append(
                    f"[Derived FactNode]\n"
                    f"Statement: {fact['statement']}\n"
                    f"Lineage: source_engram_id={fact['source_engram_id']} | "
                    f"source_span={fact['passage_span']} | "
                    f"promotion_receipt_id={receipt['receipt_id']}"
                )
            context.append("</Derived_FactNodes>")
            
        return {
            "query": query,
            "context": context
        }
