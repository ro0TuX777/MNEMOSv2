from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Fact:
    fact_id: str
    text: str
    protected_spans: List[str]
    source_id: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

@dataclass
class Governance:
    risk: str
    approval_required: bool
    flags: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    source: str
    text: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

@dataclass
class EchoFramePacket:
    facts: List[Fact] = field(default_factory=list)
    governance: Optional[Governance] = None
    evidence: List[Evidence] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            facts=[Fact.from_dict(f) for f in d.get('facts', [])],
            governance=Governance.from_dict(d['governance']) if d.get('governance') else None,
            evidence=[Evidence.from_dict(e) for e in d.get('evidence', [])],
            gaps=d.get('gaps', []),
            contradictions=d.get('contradictions', [])
        )

def _render_facts(facts: List[Fact]) -> str:
    if not facts:
        return ""
    lines = ["[FACTS]"]
    for f in facts:
        lines.append(f"- [{f.fact_id}] {f.text} (Source: {f.source_id})")
    return "\n".join(lines)

def _render_governance(gov: Optional[Governance]) -> str:
    if not gov:
        return ""
    lines = ["[GOVERNANCE]"]
    lines.append(f"Risk Level: {gov.risk}")
    if gov.approval_required:
        lines.append("WARNING: Manager Approval Required")
    if gov.flags:
        lines.append(f"Flags: {', '.join(gov.flags)}")
    return "\n".join(lines)

def _render_evidence(evidence: List[Evidence]) -> str:
    if not evidence:
        return ""
    lines = ["[EVIDENCE]"]
    for e in evidence:
        lines.append(f"[{e.evidence_id}] Source: {e.source_id} ({e.source})")
        lines.append(f"Text: {e.text}")
    return "\n".join(lines)

def _render_critical_constraints(gov: Optional[Governance], contradictions: List[str], gaps: List[str]) -> str:
    lines = ["[CRITICAL_CONSTRAINTS]"]
    if gov and gov.approval_required:
        lines.append("- APPROVAL REQUIRED")
    for c in contradictions:
        lines.append(f"- CONTRADICTION: {c}")
    for g in gaps:
        lines.append(f"- GAP: {g}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)

def _render_gaps_contradictions(packet: EchoFramePacket) -> str:
    lines = []
    if packet.contradictions:
        lines.append("[CONTRADICTIONS]")
        for c in packet.contradictions:
            lines.append(f"- {c}")
    if packet.gaps:
        lines.append("[EVIDENCE GAPS]")
        for g in packet.gaps:
            lines.append(f"- {g}")
    return "\n".join(lines)

def render_layout(packet: EchoFramePacket, layout_id: str) -> str:
    """
    Render an EchoFrame packet using one of the Phase 12A placement layouts.

    layout_id:
      A = facts_governance_evidence
      B = governance_facts_evidence
      C = facts_evidence_governance_recap
      D = top_bottom_critical_constraints
      E = source_local
    """
    parts = []
    
    if layout_id == 'A':
        parts.append(_render_facts(packet.facts))
        parts.append(_render_governance(packet.governance))
        parts.append(_render_gaps_contradictions(packet))
        parts.append(_render_evidence(packet.evidence))
        
    elif layout_id == 'B':
        parts.append(_render_governance(packet.governance))
        parts.append(_render_facts(packet.facts))
        parts.append(_render_gaps_contradictions(packet))
        parts.append(_render_evidence(packet.evidence))
        
    elif layout_id == 'C':
        parts.append(_render_facts(packet.facts))
        parts.append(_render_gaps_contradictions(packet))
        parts.append(_render_evidence(packet.evidence))
        parts.append("[GOVERNANCE_RECAP]")
        parts.append(_render_governance(packet.governance))
        
    elif layout_id == 'D':
        parts.append(_render_governance(packet.governance))
        parts.append(_render_facts(packet.facts))
        parts.append(_render_gaps_contradictions(packet))
        parts.append(_render_evidence(packet.evidence))
        parts.append(_render_critical_constraints(packet.governance, packet.contradictions, packet.gaps))
        
    elif layout_id == 'E':
        # Source-local: Fact -> Source -> Governance repeated per evidence unit
        # Map facts and evidence by source_id
        source_map = {}
        for ev in packet.evidence:
            source_map[ev.source_id] = {'evidence': ev, 'facts': []}
        for f in packet.facts:
            if f.source_id in source_map:
                source_map[f.source_id]['facts'].append(f)
            else:
                source_map[f.source_id] = {'evidence': None, 'facts': [f]}
                
        lines = ["[SOURCE-LOCAL PACKET]"]
        for sid, content in source_map.items():
            lines.append(f"--- SOURCE: {sid} ---")
            if content['evidence']:
                ev = content['evidence']
                lines.append(f"URI: {ev.source}")
                lines.append(f"Content: {ev.text}")
            if content['facts']:
                lines.append("Derived Facts:")
                for f in content['facts']:
                    lines.append(f" - [{f.fact_id}] {f.text}")
            if packet.governance:
                lines.append("Governance Restrictions for this Source:")
                lines.append(f" - Risk: {packet.governance.risk}")
                if packet.governance.approval_required:
                    lines.append(" - WARNING: Manager Approval Required")
        
        parts.append("\n".join(lines))
        parts.append(_render_gaps_contradictions(packet))
        
    else:
        raise ValueError(f"Unknown layout_id: {layout_id}")

    return "\n\n".join(filter(bool, parts))
