"""Shadow-only evidence-bounded iterative reconciliation.

This module is a RepFusion-inspired refinement lane for Phase 10-style
Resolution Engram candidates.  It is not diffusion and it does not alter the
governance, retrieval, ranking, or promotion paths.  The refiner repeatedly
challenges a synthetic candidate against an immutable evidence packet and only
allows revisions that respond to structured critique categories.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mnemos.audit.forensic_ledger import ForensicLedger
from mnemos.engram.model import Engram
from mnemos.governance.hygiene.contradiction_sweep import (
    ContradictionSweepRecord,
    ContradictionSweepReport,
)
from mnemos.governance.hygiene.reconciliation_runner import (
    ReconciliationRunner,
)


_OVERCONFIDENT_TERMS = ("definitive", "certain", "guaranteed", "must be treated as")
_TEMPORAL_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d+\s*(?:hours?|days?|months?|years?)\b", re.I)


@dataclass(frozen=True)
class ParentEvidence:
    """Immutable parent evidence included in a reconciliation packet."""

    engram_id: str
    entity_key: str
    attribute_key: str
    normalized_value: str
    source: str
    source_type: str
    source_id: str
    source_authority: float
    trust_score: float
    created_at: str
    evidence_span: str
    lineage: Dict[str, Any]
    governance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engram_id": self.engram_id,
            "entity_key": self.entity_key,
            "attribute_key": self.attribute_key,
            "normalized_value": self.normalized_value,
            "source": self.source,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_authority": self.source_authority,
            "trust_score": self.trust_score,
            "created_at": self.created_at,
            "evidence_span": self.evidence_span,
            "lineage": dict(self.lineage),
            "governance": dict(self.governance),
        }


@dataclass(frozen=True)
class ReconciliationPacket:
    """Immutable packet used by the shadow refinement loop."""

    packet_id: str
    cluster_key: str
    conflict_group_id: str
    entity_key: str
    attribute_key: str
    winner_id: str
    loser_ids: Tuple[str, ...]
    parents: Tuple[ParentEvidence, ...]
    governance_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "cluster_key": self.cluster_key,
            "conflict_group_id": self.conflict_group_id,
            "entity_key": self.entity_key,
            "attribute_key": self.attribute_key,
            "winner_id": self.winner_id,
            "loser_ids": list(self.loser_ids),
            "parents": [parent.to_dict() for parent in self.parents],
            "governance_metadata": dict(self.governance_metadata),
        }


@dataclass
class ClaimSupport:
    """Structured claim support without hidden reasoning traces."""

    claim: str
    supporting_parent_ids: List[str] = field(default_factory=list)
    unsupported: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "supporting_parent_ids": list(self.supporting_parent_ids),
            "unsupported": self.unsupported,
        }


@dataclass
class CandidateResolution:
    """Constrained shadow candidate schema for iterative reconciliation."""

    status: str
    resolved_value: Optional[str]
    summary: str
    confidence: float
    uncertainty_notes: List[str] = field(default_factory=list)
    parent_support_map: Dict[str, List[str]] = field(default_factory=dict)
    claim_support: List[ClaimSupport] = field(default_factory=list)
    operator_review_notes: List[str] = field(default_factory=list)
    promotable: bool = False

    def to_content(self) -> str:
        value = self.resolved_value or "unresolved"
        notes = " ".join(self.uncertainty_notes)
        return f"{self.summary} Resolution value: {value}. {notes}".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "resolved_value": self.resolved_value,
            "summary": self.summary,
            "confidence": self.confidence,
            "uncertainty_notes": list(self.uncertainty_notes),
            "parent_support_map": {
                key: list(value) for key, value in self.parent_support_map.items()
            },
            "claim_support": [claim.to_dict() for claim in self.claim_support],
            "operator_review_notes": list(self.operator_review_notes),
            "promotable": self.promotable,
        }


@dataclass
class EvidenceChallenge:
    """Structured critique categories for one pass."""

    unsupported_claims: List[str] = field(default_factory=list)
    missing_parent_coverage: List[str] = field(default_factory=list)
    unresolved_temporal_ambiguity: List[str] = field(default_factory=list)
    authority_policy_conflicts: List[str] = field(default_factory=list)
    overconfident_language: List[str] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return any(
            (
                self.unsupported_claims,
                self.missing_parent_coverage,
                self.unresolved_temporal_ambiguity,
                self.authority_policy_conflicts,
                self.overconfident_language,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unsupported_claims": list(self.unsupported_claims),
            "missing_parent_coverage": list(self.missing_parent_coverage),
            "unresolved_temporal_ambiguity": list(self.unresolved_temporal_ambiguity),
            "authority_policy_conflicts": list(self.authority_policy_conflicts),
            "overconfident_language": list(self.overconfident_language),
        }


@dataclass
class RevisionDelta:
    """Allowed revision delta for one refinement pass."""

    changed_fields: List[str] = field(default_factory=list)
    added_parent_support: List[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    notes_added: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed_fields": list(self.changed_fields),
            "added_parent_support": list(self.added_parent_support),
            "confidence_delta": self.confidence_delta,
            "notes_added": list(self.notes_added),
        }


@dataclass
class RefinementPassRecord:
    """Forensic-safe record for one pass."""

    pass_index: int
    packet_id: str
    packet_hash: str
    source_references: List[Dict[str, Any]]
    candidate_before: CandidateResolution
    critique: EvidenceChallenge
    revision_delta: RevisionDelta
    candidate_after: CandidateResolution
    latency_ms: float
    estimated_token_cost: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "packet_id": self.packet_id,
            "packet_hash": self.packet_hash,
            "source_references": list(self.source_references),
            "candidate_before": self.candidate_before.to_dict(),
            "critique": self.critique.to_dict(),
            "revision_delta": self.revision_delta.to_dict(),
            "candidate_after": self.candidate_after.to_dict(),
            "latency_ms": self.latency_ms,
            "estimated_token_cost": self.estimated_token_cost,
        }


@dataclass
class RepFusionRefinementRecord:
    """Final record for one shadow refinement lane candidate."""

    packet: ReconciliationPacket
    baseline_resolution_engram: Engram
    final_candidate: CandidateResolution
    passes: List[RefinementPassRecord]
    final_confidence: float
    shadow_only: bool = True
    auto_promoted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet": self.packet.to_dict(),
            "baseline_resolution_engram": self.baseline_resolution_engram.to_dict(
                include_governance=True,
                include_lineage=True,
            ),
            "final_candidate": self.final_candidate.to_dict(),
            "passes": [record.to_dict() for record in self.passes],
            "final_confidence": self.final_confidence,
            "shadow_only": self.shadow_only,
            "auto_promoted": self.auto_promoted,
        }


@dataclass
class RepFusionRefinementReport:
    """Summary for the separate EBIR refinement lane."""

    lane_name: str = "EBIR: Evidence-Bounded Iterative Reconciliation"
    shadow_only: bool = True
    max_passes: int = 3
    records: List[RepFusionRefinementRecord] = field(default_factory=list)
    skipped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_name": self.lane_name,
            "shadow_only": self.shadow_only,
            "max_passes": self.max_passes,
            "records": [record.to_dict() for record in self.records],
            "skipped": self.skipped,
        }


class RepFusionRefiner:
    """Shadow-only bounded iterative refiner around ReconciliationRunner."""

    def __init__(
        self,
        *,
        reconciliation_runner: Optional[ReconciliationRunner] = None,
        ledger: Optional[ForensicLedger] = None,
        max_passes: int = 3,
    ) -> None:
        if max_passes < 1 or max_passes > 3:
            raise ValueError("max_passes must be between 1 and 3")
        self._runner = reconciliation_runner or ReconciliationRunner()
        self._ledger = ledger
        self.max_passes = max_passes

    def run(
        self,
        engrams: List[Engram],
        *,
        sweep_report: Optional[ContradictionSweepReport] = None,
    ) -> RepFusionRefinementReport:
        """Run the shadow refinement lane without writes or promotion."""
        baseline = self._runner.run(
            engrams,
            sweep_report=sweep_report,
            dry_run=True,
            indexer=None,
        )
        report = RepFusionRefinementReport(max_passes=self.max_passes)
        engram_by_id = {engram.id: engram for engram in engrams}

        for baseline_record in baseline.records:
            parents = [
                engram_by_id[parent_id]
                for parent_id in baseline_record.parent_ids
                if parent_id in engram_by_id
            ]
            if len(parents) < 2:
                report.skipped += 1
                continue

            packet = self.build_packet(
                baseline_record.cluster_key,
                baseline_record.conflict_group_id,
                baseline_record.parent_ids,
                baseline_record.parent_ids[0],
                parents,
            )
            packet_hash = self.packet_hash(packet)
            candidate = self.generate_candidate(packet)
            pass_records: List[RefinementPassRecord] = []

            for pass_index in range(1, self.max_passes + 1):
                start = time.perf_counter()
                if self.packet_hash(packet) != packet_hash:
                    raise RuntimeError(
                        f"reconciliation packet mutated before pass {pass_index}"
                    )
                before = _copy_candidate(candidate)
                critique = self.challenge(packet, candidate)
                candidate, delta = self.revise(packet, candidate, critique)
                if self.packet_hash(packet) != packet_hash:
                    raise RuntimeError(
                        f"reconciliation packet mutated during pass {pass_index}"
                    )
                latency_ms = (time.perf_counter() - start) * 1000
                pass_record = RefinementPassRecord(
                    pass_index=pass_index,
                    packet_id=packet.packet_id,
                    packet_hash=packet_hash,
                    source_references=self._source_references(packet),
                    candidate_before=before,
                    critique=critique,
                    revision_delta=delta,
                    candidate_after=_copy_candidate(candidate),
                    latency_ms=latency_ms,
                    estimated_token_cost=self._estimate_token_cost(packet, before, critique),
                )
                pass_records.append(pass_record)
                self._log_pass(pass_record)
                if not critique.has_findings:
                    break

            candidate.promotable = False
            record = RepFusionRefinementRecord(
                packet=packet,
                baseline_resolution_engram=baseline_record.resolution_engram,
                final_candidate=candidate,
                passes=pass_records,
                final_confidence=candidate.confidence,
            )
            report.records.append(record)
            self._log_final(record)

        return report

    def build_packet(
        self,
        cluster_key: str,
        conflict_group_id: str,
        parent_ids: List[str],
        winner_id: str,
        parents: List[Engram],
    ) -> ReconciliationPacket:
        entity_key, attribute_key = ReconciliationRunner._split_cluster_key(cluster_key)
        parent_evidence = [self._parent_evidence(parent) for parent in parents]
        packet_basis = "|".join([cluster_key, conflict_group_id, *parent_ids])
        packet_id = "ebir_" + hashlib.sha256(packet_basis.encode("utf-8")).hexdigest()[:16]
        losers = [parent_id for parent_id in parent_ids if parent_id != winner_id]
        return ReconciliationPacket(
            packet_id=packet_id,
            cluster_key=cluster_key,
            conflict_group_id=conflict_group_id,
            entity_key=entity_key,
            attribute_key=attribute_key,
            winner_id=winner_id,
            loser_ids=tuple(losers),
            parents=tuple(parent_evidence),
            governance_metadata={
                "mode": "shadow_only",
                "lane": "EBIR",
                "max_passes": self.max_passes,
                "parent_count": len(parent_evidence),
            },
        )

    def generate_candidate(self, packet: ReconciliationPacket) -> CandidateResolution:
        best = max(
            packet.parents,
            key=lambda parent: (parent.source_authority, parent.trust_score),
        )
        values = sorted({parent.normalized_value for parent in packet.parents})
        status = "unresolved"
        resolved_value: Optional[str] = None
        if self._authority_margin(packet.parents) >= 0.2 and not self._security_sensitive(packet):
            status = "reconciled"
            resolved_value = best.normalized_value
        summary = (
            f"{packet.entity_key}:{packet.attribute_key} has conflicting parent "
            f"evidence across {len(values)} values: {', '.join(values)}."
        )
        if resolved_value:
            summary += f" Current shadow preference is {resolved_value}."
        candidate = CandidateResolution(
            status=status,
            resolved_value=resolved_value,
            summary=summary,
            confidence=0.55 if status == "unresolved" else 0.68,
            uncertainty_notes=[
                "Shadow candidate only; parent engrams remain authoritative inputs."
            ],
            parent_support_map={parent.engram_id: [] for parent in packet.parents},
            operator_review_notes=["Review all parent spans before promotion decisions."],
        )
        if self._security_sensitive(packet):
            candidate.operator_review_notes.append(
                "Security-sensitive contradiction requires operator review before resolution."
            )
        for parent in packet.parents:
            claim = f"{parent.engram_id} supports value {parent.normalized_value}"
            candidate.claim_support.append(
                ClaimSupport(claim=claim, supporting_parent_ids=[parent.engram_id])
            )
            candidate.parent_support_map[parent.engram_id].append(parent.normalized_value)
        return candidate

    def challenge(
        self,
        packet: ReconciliationPacket,
        candidate: CandidateResolution,
    ) -> EvidenceChallenge:
        challenge = EvidenceChallenge()
        parent_ids = {parent.engram_id for parent in packet.parents}
        covered_ids = {
            parent_id
            for parent_id, supports in candidate.parent_support_map.items()
            if supports
        }
        challenge.missing_parent_coverage.extend(sorted(parent_ids - covered_ids))

        evidence_text = " ".join(
            f"{parent.normalized_value} {parent.evidence_span}"
            for parent in packet.parents
        ).lower()
        for claim in candidate.claim_support:
            if claim.supporting_parent_ids:
                continue
            if claim.claim.lower() not in evidence_text:
                challenge.unsupported_claims.append(claim.claim)

        if candidate.resolved_value and candidate.resolved_value not in {
            parent.normalized_value for parent in packet.parents
        }:
            challenge.unsupported_claims.append(
                f"resolved_value {candidate.resolved_value} not present in parent values"
            )

        temporal_markers = {
            marker
            for parent in packet.parents
            for marker in _TEMPORAL_RE.findall(
                f"{parent.normalized_value} {parent.evidence_span}"
            )
        }
        if len(temporal_markers) > 1 and not any(
            "temporal" in note.lower() for note in candidate.uncertainty_notes
        ):
            challenge.unresolved_temporal_ambiguity.append(
                "multiple temporal markers require explicit uncertainty"
            )

        if self._security_sensitive(packet) and candidate.status == "reconciled":
            challenge.authority_policy_conflicts.append(
                "security-sensitive contradiction should remain unresolved in shadow"
            )
        elif 0.0 <= self._authority_margin(packet.parents) < 0.2 and candidate.status == "reconciled":
            challenge.authority_policy_conflicts.append(
                "source authority margin too narrow for resolved candidate"
            )

        combined = " ".join(
            [candidate.summary, *candidate.uncertainty_notes, *candidate.operator_review_notes]
        ).lower()
        challenge.overconfident_language.extend(
            term for term in _OVERCONFIDENT_TERMS if term in combined
        )
        return challenge

    def revise(
        self,
        packet: ReconciliationPacket,
        candidate: CandidateResolution,
        critique: EvidenceChallenge,
    ) -> tuple[CandidateResolution, RevisionDelta]:
        revised = _copy_candidate(candidate)
        delta = RevisionDelta()

        for parent_id in critique.missing_parent_coverage:
            parent = next(parent for parent in packet.parents if parent.engram_id == parent_id)
            revised.parent_support_map.setdefault(parent_id, []).append(
                parent.normalized_value
            )
            revised.claim_support.append(
                ClaimSupport(
                    claim=f"{parent_id} supports value {parent.normalized_value}",
                    supporting_parent_ids=[parent_id],
                )
            )
            delta.added_parent_support.append(parent_id)
        if critique.missing_parent_coverage:
            delta.changed_fields.extend(["parent_support_map", "claim_support"])

        if critique.unsupported_claims:
            supported_claims = [
                claim for claim in revised.claim_support if not claim.unsupported
            ]
            revised.claim_support = supported_claims
            revised.confidence = max(0.0, revised.confidence - 0.1)
            delta.changed_fields.extend(["claim_support", "confidence"])
            delta.confidence_delta -= 0.1

        if critique.unresolved_temporal_ambiguity:
            note = "Temporal ambiguity remains unresolved across parent evidence."
            if note not in revised.uncertainty_notes:
                revised.uncertainty_notes.append(note)
                delta.notes_added.append(note)
            if self._has_supersession_signal(packet, revised.resolved_value):
                revised.status = "reconciled"
                revised.confidence = min(revised.confidence, 0.66)
                delta.changed_fields.extend(["status", "uncertainty_notes", "confidence"])
            else:
                revised.status = "unresolved"
                revised.resolved_value = None
                revised.confidence = min(revised.confidence, 0.6)
                delta.changed_fields.extend(["status", "resolved_value", "uncertainty_notes"])

        if critique.authority_policy_conflicts:
            note = "Authority or policy conflict requires operator review before resolution."
            if note not in revised.operator_review_notes:
                revised.operator_review_notes.append(note)
                delta.notes_added.append(note)
            revised.status = "unresolved"
            revised.resolved_value = None
            revised.confidence = min(revised.confidence, 0.5)
            delta.changed_fields.extend(
                ["status", "resolved_value", "operator_review_notes", "confidence"]
            )

        if critique.overconfident_language:
            for term in critique.overconfident_language:
                revised.summary = re.sub(term, "candidate", revised.summary, flags=re.I)
            revised.confidence = min(revised.confidence, 0.6)
            delta.changed_fields.extend(["summary", "confidence"])

        delta.changed_fields = sorted(set(delta.changed_fields))
        return revised, delta

    def _parent_evidence(self, parent: Engram) -> ParentEvidence:
        gov = parent.governance
        governance = gov.to_dict() if gov else {}
        span = (
            parent.metadata.get("evidence_span")
            or parent.metadata.get("provenance_span")
            or parent.content
        )
        return ParentEvidence(
            engram_id=parent.id,
            entity_key=gov.entity_key if gov else "",
            attribute_key=gov.attribute_key if gov else "",
            normalized_value=gov.normalized_value if gov else "",
            source=parent.source,
            source_type=gov.source_type if gov else "",
            source_id=gov.source_id if gov else "",
            source_authority=float(gov.source_authority if gov else 0.5),
            trust_score=float(gov.trust_score if gov else parent.confidence),
            created_at=parent.created_at,
            evidence_span=str(span),
            lineage=parent.lineage(),
            governance=governance,
        )

    def _source_references(self, packet: ReconciliationPacket) -> List[Dict[str, Any]]:
        return [
            {
                "engram_id": parent.engram_id,
                "source": parent.source,
                "lineage": parent.lineage,
                "evidence_span": parent.evidence_span,
            }
            for parent in packet.parents
        ]

    @staticmethod
    def packet_hash(packet: ReconciliationPacket) -> str:
        """Hash the full structured packet for pass-to-pass immutability checks."""
        canonical = json.dumps(packet.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _log_pass(self, record: RefinementPassRecord) -> None:
        if self._ledger is None:
            return
        self._ledger.log_transaction(
            component="governance.repfusion_refiner",
            action="shadow_refinement_pass",
            content=f"EBIR pass {record.pass_index} for {record.packet_id}",
            status="success",
            latency=record.latency_ms / 1000,
            metadata={
                "tags": "governance,reconciliation,ebir,shadow",
                "packet_id": record.packet_id,
                "pass_index": record.pass_index,
                "source_references": record.source_references,
                "critique": record.critique.to_dict(),
                "revision_delta": record.revision_delta.to_dict(),
                "estimated_token_cost": record.estimated_token_cost,
                "final_confidence": record.candidate_after.confidence,
            },
        )

    def _log_final(self, record: RepFusionRefinementRecord) -> None:
        if self._ledger is None:
            return
        self._ledger.log_transaction(
            component="governance.repfusion_refiner",
            action="shadow_refinement_final",
            content=f"EBIR final candidate for {record.packet.packet_id}",
            status="success",
            metadata={
                "tags": "governance,reconciliation,ebir,shadow",
                "packet_id": record.packet.packet_id,
                "final_candidate": record.final_candidate.to_dict(),
                "final_confidence": record.final_confidence,
                "shadow_only": True,
                "auto_promoted": False,
            },
        )

    @staticmethod
    def _estimate_token_cost(
        packet: ReconciliationPacket,
        candidate: CandidateResolution,
        critique: EvidenceChallenge,
    ) -> int:
        text = " ".join(
            [
                str(packet.to_dict()),
                str(candidate.to_dict()),
                str(critique.to_dict()),
            ]
        )
        return max(1, len(text.split()))

    @staticmethod
    def _has_supersession_signal(
        packet: ReconciliationPacket,
        resolved_value: Optional[str],
    ) -> bool:
        if not resolved_value:
            return False
        selected = [
            parent for parent in packet.parents if parent.normalized_value == resolved_value
        ]
        if not selected:
            return False
        selected_parent = max(
            selected,
            key=lambda parent: (parent.source_authority, parent.trust_score),
        )
        selected_text = selected_parent.evidence_span.lower()
        has_update_language = any(
            marker in selected_text
            for marker in ("supersedes", "superseded", "replaces", "updated", "current")
        )
        if not has_update_language:
            return False
        other_authorities = [
            parent.source_authority
            for parent in packet.parents
            if parent.engram_id != selected_parent.engram_id
        ]
        return not other_authorities or selected_parent.source_authority >= max(other_authorities)

    @staticmethod
    def _authority_margin(parents: Iterable[ParentEvidence]) -> float:
        scores = sorted(
            (parent.source_authority, parent.trust_score) for parent in parents
        )
        if len(scores) < 2:
            return 0.0
        best = scores[-1][0] + scores[-1][1] * 0.25
        second = scores[-2][0] + scores[-2][1] * 0.25
        return best - second

    @staticmethod
    def _security_sensitive(packet: ReconciliationPacket) -> bool:
        haystack = " ".join(
            [
                packet.entity_key,
                packet.attribute_key,
                " ".join(parent.evidence_span for parent in packet.parents),
                " ".join(parent.normalized_value for parent in packet.parents),
            ]
        ).lower()
        return any(
            marker in haystack
            for marker in ("secret", "classification", "enclave", "security-sensitive")
        )


def _copy_candidate(candidate: CandidateResolution) -> CandidateResolution:
    return CandidateResolution(
        status=candidate.status,
        resolved_value=candidate.resolved_value,
        summary=candidate.summary,
        confidence=candidate.confidence,
        uncertainty_notes=list(candidate.uncertainty_notes),
        parent_support_map={
            key: list(value) for key, value in candidate.parent_support_map.items()
        },
        claim_support=[
            ClaimSupport(
                claim=claim.claim,
                supporting_parent_ids=list(claim.supporting_parent_ids),
                unsupported=claim.unsupported,
            )
            for claim in candidate.claim_support
        ],
        operator_review_notes=list(candidate.operator_review_notes),
        promotable=candidate.promotable,
    )
