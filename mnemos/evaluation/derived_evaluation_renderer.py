"""
Standalone Derived Evaluation Renderer
======================================

Strictly isolated from production EchoFrame generation paths.
Renders the Evaluation Mode context blocks.
"""

from typing import Dict, Any

def render_derived_evaluation_context(shadow_packet: Dict[str, Any]) -> str:
    """
    Renders the structured Shadow Packet into an explicit text block.
    This renderer must never be invoked by the production prompt builder.
    """
    if not shadow_packet.get("shadow_only"):
        raise ValueError("SEV-STOP: Cannot render a non-shadow packet in evaluation mode.")

    payload = shadow_packet.get("derived_evaluation_payload", [])
    if not payload:
        return ""

    lines = []
    lines.append("=== [MNEMOS-DERIVED EVALUATION CONTEXT] ===")
    
    for fact in payload:
        prefix = fact.get("string_prefix", "[AUTHORITY: MNEMOS_DERIVED_FACT] [MNEMOS-DERIVED]")
        content = fact.get("content", "")
        
        lines.append(prefix)
        lines.append(content)
        
        gaps = fact.get("authority_matrix", {}).get("evidence_gaps", [])
        if gaps:
            lines.append(f"Evidence Gaps: {', '.join(gaps)}")
            
        lines.append("")

    lines.append("===========================================")
    return "\n".join(lines)
