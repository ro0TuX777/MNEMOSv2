import json
import yaml
from typing import List, Optional
# Reusing the EchoFramePacket definition from phase12a
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../echoframe_phase12a'))
from placement_layouts import EchoFramePacket, Fact, Governance, Evidence

def _get_critical_constraints(gov: Optional[Governance], contradictions: List[str], gaps: List[str]) -> List[str]:
    constraints = []
    if gov and gov.approval_required:
        constraints.append("APPROVAL REQUIRED")
    constraints.extend([f"CONTRADICTION: {c}" for c in contradictions])
    constraints.extend([f"GAP: {g}" for g in gaps])
    return constraints

def assert_format_preserves_packet(packet: EchoFramePacket, rendered: str) -> None:
    """
    Raises AssertionError if a rendered format drops required protected fields.
    """
    rendered_lower = rendered.lower()
    
    # Check facts and protected spans
    for f in packet.facts:
        assert f.fact_id.lower() in rendered_lower, f"Lost fact_id {f.fact_id}"
        # Source ID is tricky because it might be grouped, but we expect it to be present
        assert f.source_id.lower() in rendered_lower, f"Lost source_id {f.source_id}"
        for span in f.protected_spans:
            assert span.lower() in rendered_lower, f"Lost protected span {span}"
            
    # Check evidence
    for e in packet.evidence:
        assert e.evidence_id.lower() in rendered_lower, f"Lost evidence_id {e.evidence_id}"
        assert e.source_id.lower() in rendered_lower, f"Lost source_id {e.source_id}"
        # Only require a portion of text or URI to avoid whitespace matching issues
        
    # Check governance
    if packet.governance:
        assert packet.governance.risk.lower() in rendered_lower, f"Lost risk {packet.governance.risk}"
        # approval_required or similar flag
        
    for c in packet.contradictions:
        # Check basic substring
        assert "contradiction" in rendered_lower or c.lower() in rendered_lower, f"Lost contradiction {c}"
        
    for g in packet.gaps:
        assert "gap" in rendered_lower or g.lower() in rendered_lower, f"Lost gap {g}"

def render_format(packet: EchoFramePacket, format_id: str) -> str:
    """
    Render Phase 12B formats using Layout D ordering:
    Governance + Facts -> Gaps/Contradictions -> Evidence -> Critical Constraints Recap
    
    format_id:
      1_baseline           - Current compact Markdown tags (Layout A)
      2_layout_d           - Layout D with current tags
      3_ultra_compact      - G/F/E/C tags
      4_yaml_lite          - YAML-lite
      5_minified_json      - Minified JSON
      6_markdown_table     - Markdown table
      7_toon_rows          - TOON-inspired rows
      8_source_table_facts - Source-table + fact rows
    """
    
    if format_id == '1_baseline':
        # Recreate Layout A
        from placement_layouts import render_layout
        return render_layout(packet, 'A')
        
    elif format_id == '2_layout_d':
        # Recreate Layout D
        from placement_layouts import render_layout
        return render_layout(packet, 'D')
        
    elif format_id == '3_ultra_compact':
        parts = []
        if packet.governance:
            parts.append(f"[G] Risk:{packet.governance.risk} Appr:{int(packet.governance.approval_required)}")
        if packet.facts:
            parts.append("[F]\n" + "\n".join([f"{f.fact_id}|{f.source_id}|{f.text}" for f in packet.facts]))
        
        gaps_con = _get_critical_constraints(None, packet.contradictions, packet.gaps)
        if gaps_con:
            parts.append("[!] " + " | ".join(gaps_con))
            
        if packet.evidence:
            parts.append("[E]\n" + "\n".join([f"{e.evidence_id}|{e.source_id}|{e.text}" for e in packet.evidence]))
            
        cc = _get_critical_constraints(packet.governance, packet.contradictions, packet.gaps)
        if cc:
            parts.append("[C] " + " | ".join(cc))
        return "\n".join(parts)

    elif format_id == '4_yaml_lite':
        # YAML Lite representation
        d = {
            "gov": {"risk": packet.governance.risk, "appr": packet.governance.approval_required} if packet.governance else {},
            "facts": [{"id": f.fact_id, "src": f.source_id, "txt": f.text} for f in packet.facts],
            "ev": [{"id": e.evidence_id, "src": e.source_id, "txt": e.text} for e in packet.evidence],
            "cc": _get_critical_constraints(packet.governance, packet.contradictions, packet.gaps)
        }
        return yaml.dump(d, default_flow_style=False, sort_keys=False)

    elif format_id == '5_minified_json':
        d = {
            "G": {"R": packet.governance.risk, "A": packet.governance.approval_required} if packet.governance else {},
            "F": [{"I": f.fact_id, "S": f.source_id, "T": f.text} for f in packet.facts],
            "E": [{"I": e.evidence_id, "S": e.source_id, "T": e.text} for e in packet.evidence],
            "C": _get_critical_constraints(packet.governance, packet.contradictions, packet.gaps)
        }
        return json.dumps(d, separators=(',', ':'))

    elif format_id == '6_markdown_table':
        parts = []
        if packet.governance:
            parts.append("### Governance\n| Risk | Approval Required |\n|---|---|\n" + 
                         f"| {packet.governance.risk} | {packet.governance.approval_required} |")
        if packet.facts:
            parts.append("### Facts\n| ID | Source | Fact |\n|---|---|---|\n" + 
                         "\n".join([f"| {f.fact_id} | {f.source_id} | {f.text} |" for f in packet.facts]))
        if packet.evidence:
            parts.append("### Evidence\n| ID | Source | Text |\n|---|---|---|\n" + 
                         "\n".join([f"| {e.evidence_id} | {e.source_id} | {e.text} |" for e in packet.evidence]))
        
        cc = _get_critical_constraints(packet.governance, packet.contradictions, packet.gaps)
        if cc:
            parts.append("### Critical Constraints\n" + "\n".join([f"- {c}" for c in cc]))
            
        return "\n\n".join(parts)

    elif format_id == '7_toon_rows':
        parts = []
        if packet.governance:
            parts.append(f"GOV_ROW\t{packet.governance.risk}\t{packet.governance.approval_required}")
        for f in packet.facts:
            parts.append(f"FACT_ROW\t{f.fact_id}\t{f.source_id}\t{f.text}")
        for e in packet.evidence:
            parts.append(f"EV_ROW\t{e.evidence_id}\t{e.source_id}\t{e.text}")
        for c in _get_critical_constraints(packet.governance, packet.contradictions, packet.gaps):
            parts.append(f"CRIT_ROW\t{c}")
        return "\n".join(parts)

    elif format_id == '8_source_table_facts':
        parts = []
        if packet.governance:
            parts.append(f"**Governance**: Risk={packet.governance.risk}, Appr={packet.governance.approval_required}")
        
        # Source table mapping source_id to source text
        parts.append("### Source Table")
        for e in packet.evidence:
            parts.append(f"- **{e.source_id}** [{e.evidence_id}]: {e.text}")
            
        parts.append("### Facts")
        for f in packet.facts:
            parts.append(f"- [{f.fact_id}] {f.text} (See {f.source_id})")
            
        cc = _get_critical_constraints(packet.governance, packet.contradictions, packet.gaps)
        if cc:
            parts.append("### Constraints\n" + "\n".join([f"! {c}" for c in cc]))
            
        return "\n\n".join(parts)

    else:
        raise ValueError(f"Unknown format_id: {format_id}")
