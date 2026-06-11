"""
Derived Shadow Packet Serializer
================================

Standalone module for serializing derived facts into evaluation-only shadow packets.
This module is physically separated from production EchoFrame rendering to enforce
the PIT-3 boundary constraint: derived facts must never enter the live LLM prompt.
"""

from typing import Dict, Any, List
import logging
from mnemos.config import get_config

logger = logging.getLogger(__name__)

class DerivedShadowPacketSerializer:
    def __init__(self):
        self._config = get_config()

    def _estimate_tokens(self, text: str) -> int:
        """Rough approximation of tokens for budget enforcement."""
        # A simple approximation: 1 token ~= 4 chars
        return len(text) // 4

    def serialize(self, pit2_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize a PIT-2 derived lane response into a PIT-3 Shadow Packet.
        
        Args:
            pit2_response: The dict returned by `retrieval_router.search_derived()`.
            
        Returns:
            A strictly structured Shadow Packet dict.
        """
        derived_results = pit2_response.get("derived_results", [])
        
        packet = {
            "schema_version": "pit_3_derived_shadow_packet_v1",
            "shadow_only": True,
            "production_prompt_allowed": False,
            "is_production_prompted": False,
            "derived_fact_count": 0,
            "packet_token_delta": 0,
            "primary_results_included": False,
            "derived_evaluation_payload": []
        }
        
        max_facts = self._config.pit3_max_derived_facts_per_shadow_packet
        max_tokens = self._config.pit3_max_derived_fact_tokens
        
        total_tokens = 0
        added_facts = 0
        
        for fact in derived_results:
            if added_facts >= max_facts:
                logger.warning("Shadow packet limits reached (max facts). Dropping remainder.")
                break
                
            # 1. Enforce authority label
            if fact.get("authority_type") != "MNEMOS_DERIVED_FACT" or fact.get("display_label") != "[MNEMOS-DERIVED]":
                logger.error(f"Missing authority label on fact {fact.get('fact_id')}. Dropping.")
                continue
                
            # 2. Enforce traceability
            trace = fact.get("traceability", {})
            required_traceability = [
                "source_engram_ids", "passage_node_ids", "fact_id", "fact_receipt_id",
                "promotion_receipt_id", "lifecycle_event_id", "source_uri",
                "artifact_id", "chunk_id", "provenance_span", "verifier_receipt_id"
            ]
            if any(k not in trace or not trace[k] for k in required_traceability):
                logger.error(f"Missing traceability fields on fact {fact.get('fact_id')}. Dropping.")
                continue
                
            # 3. Enforce lifecycle and conflict states
            gov = fact.get("governance_metadata", {})
            lif = fact.get("lifecycle_metadata", {})
            con = fact.get("conflict_metadata", {})
            
            # The PIT-2 router already drops these, but we re-verify at serialization boundary
            denied_states = {
                "UNKNOWN", "MISSING", "DOWNGRADED", "REJECTED", "REVOKED", 
                "SUPERSEDED", "CONFLICTED", "STALE", "EXPIRED", "UNVERIFIED"
            }
            
            if gov.get("status") != "CERTIFIED_FOR_GOVERNED_EVALUATION_OPERATION":
                continue
            if lif.get("terminal_state") in denied_states:
                continue
            if con.get("conflict_status") != "NO_CONFLICT_FOUND":
                continue
                
            # 4. Token limit enforcement
            fact_text = fact.get("content", "")
            est_tokens = self._estimate_tokens(fact_text)
            if total_tokens + est_tokens > max_tokens:
                logger.warning("Shadow packet token limit reached. Dropping remainder.")
                break
                
            # Build Authority Matrix
            gaps = []
            if "REDACTED" in fact_text.upper():
                gaps.append("REDACTED_CONTEXT: Source may contain redacted content.")
            # Mocking other gap extraction if it existed in conflict/lifecycle metadata
            
            authority_matrix = {
                "authority_type": "MNEMOS_DERIVED_FACT",
                "display_label": "[MNEMOS-DERIVED]",
                "confidence_level": gov.get("status"),
                "source_diversity_count": len(trace.get("source_engram_ids", [])),
                "source_authority_summary": "inherits from source Engrams",
                "evidence_gaps": gaps,
                "known_limitations": [],
                "governance_warnings": [],
                "conflict_status": con.get("conflict_status"),
                "lifecycle_state": lif.get("terminal_state")
            }
            
            shadow_fact = {
                "authority_type": "MNEMOS_DERIVED_FACT",
                "display_label": "[MNEMOS-DERIVED]",
                "string_prefix": "[AUTHORITY: MNEMOS_DERIVED_FACT] [MNEMOS-DERIVED]",
                "content": fact_text,
                "traceability": trace,
                "lifecycle_metadata": lif,
                "conflict_metadata": con,
                "authority_matrix": authority_matrix
            }
            
            packet["derived_evaluation_payload"].append(shadow_fact)
            total_tokens += est_tokens
            added_facts += 1
            
        packet["derived_fact_count"] = added_facts
        packet["packet_token_delta"] = total_tokens
        
        return packet
