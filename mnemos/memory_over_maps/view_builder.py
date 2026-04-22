"""On-demand derived view builders (Phase 3)."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

from mnemos.governance.models.contradiction_record import ContradictionRecord
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.memory_over_maps.models import (
    ContradictionBundle,
    EvidenceBundle,
    PreferenceSnapshot,
    TimelineSummary,
)
from mnemos.retrieval.base import SearchResult
from mnemos.memory_over_maps.view_cache import governance_state_hash

SUPPORTED_DERIVED_VIEWS = {
    "evidence_bundle",
    "contradiction_bundle",
    "preference_snapshot",
    "timeline_summary",
}


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_fingerprint(query: str) -> str:
    return _stable_hash(query.strip().lower())[:16]


def _lineage_inputs(results: Sequence[SearchResult]) -> Dict[str, List[str]]:
    artifact_ids: List[str] = []
    chunk_ids: List[str] = []
    for r in results:
        lineage = r.engram.lineage()
        artifact_id = str(lineage.get("artifact_id", ""))
        chunk_id = str(lineage.get("chunk_id", ""))
        if artifact_id and artifact_id not in artifact_ids:
            artifact_ids.append(artifact_id)
        if chunk_id and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    return {"artifact_ids": artifact_ids, "chunk_ids": chunk_ids}


def build_evidence_bundle(
    *,
    query: str,
    results: Sequence[SearchResult],
    decisions: Sequence[GovernanceDecision],
) -> EvidenceBundle:
    inputs = _lineage_inputs(results)
    return EvidenceBundle(
        view_type="evidence_bundle",
        inputs=inputs,
        query_fingerprint=_query_fingerprint(query),
        governance_state_hash=governance_state_hash(decisions),
        cacheable=False,
        reproducible=True,
        supporting_artifact_ids=inputs["artifact_ids"],
        supporting_chunk_ids=inputs["chunk_ids"],
        support_roles=["governed_support"] * len(inputs["chunk_ids"]),
        exclusions=[],
    )


def build_contradiction_bundle(
    *,
    query: str,
    results: Sequence[SearchResult],
    decisions: Sequence[GovernanceDecision],
    contradiction_records: Sequence[ContradictionRecord],
) -> ContradictionBundle:
    inputs = _lineage_inputs(results)
    winner_ids: List[str] = []
    loser_ids: List[str] = []
    comparison_factors: List[str] = []
    cluster_id = "none"
    trace = "no contradiction records available"
    if contradiction_records:
        first = contradiction_records[0]
        cluster_id = first.conflict_group_id
        if first.winner_memory_id:
            winner_ids.append(first.winner_memory_id)
        loser_ids.extend(first.loser_memory_ids)
        comparison_factors.append(first.resolution_reason)
        trace = first.resolution_reason
    return ContradictionBundle(
        view_type="contradiction_bundle",
        inputs=inputs,
        query_fingerprint=_query_fingerprint(query),
        governance_state_hash=governance_state_hash(decisions),
        cacheable=False,
        reproducible=True,
        contradiction_cluster_id=cluster_id,
        winner_ids=winner_ids,
        loser_ids=loser_ids,
        comparison_factors=comparison_factors,
        resolution_trace=trace,
    )


def build_preference_snapshot(
    *,
    query: str,
    results: Sequence[SearchResult],
    decisions: Sequence[GovernanceDecision],
    subject_id: Optional[str] = None,
) -> PreferenceSnapshot:
    inputs = _lineage_inputs(results)
    by_id = {d.engram_id: d for d in decisions}
    preferred = [r.engram.id for r in results if r.engram.id in by_id and not by_id[r.engram.id].suppressed]
    suppressed = [r.engram.id for r in results if r.engram.id in by_id and by_id[r.engram.id].suppressed]
    sid = subject_id or "query_subject"
    return PreferenceSnapshot(
        view_type="preference_snapshot",
        inputs=inputs,
        query_fingerprint=_query_fingerprint(query),
        governance_state_hash=governance_state_hash(decisions),
        cacheable=False,
        reproducible=True,
        subject_id=sid,
        preferred_memory_ids=preferred,
        suppressed_memory_ids=suppressed,
        rationale_trace="preferred memories are governed non-suppressed survivors",
    )


def build_timeline_summary(
    *,
    query: str,
    results: Sequence[SearchResult],
    decisions: Sequence[GovernanceDecision],
) -> TimelineSummary:
    inputs = _lineage_inputs(results)
    ordered = sorted(
        results,
        key=lambda r: (
            r.engram.created_at or "",
            r.engram.id,
        ),
    )
    event_refs = [r.engram.lineage().get("chunk_id", r.engram.id) for r in ordered]
    source_artifact_ids = [r.engram.lineage().get("artifact_id", f"artifact:{r.engram.id}") for r in ordered]
    return TimelineSummary(
        view_type="timeline_summary",
        inputs=inputs,
        query_fingerprint=_query_fingerprint(query),
        governance_state_hash=governance_state_hash(decisions),
        cacheable=False,
        reproducible=True,
        timeline_subject="query_timeline",
        ordered_event_refs=event_refs,
        source_artifact_ids=source_artifact_ids,
        temporal_confidence=1.0 if len(event_refs) > 1 else 0.5,
    )


def build_requested_views(
    *,
    requested: Sequence[str],
    query: str,
    results: Sequence[SearchResult],
    decisions: Sequence[GovernanceDecision],
    contradiction_records: Sequence[ContradictionRecord],
    subject_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    for view_name in requested:
        if view_name == "evidence_bundle":
            views.append(build_evidence_bundle(query=query, results=results, decisions=decisions).to_dict())
        elif view_name == "contradiction_bundle":
            views.append(
                build_contradiction_bundle(
                    query=query,
                    results=results,
                    decisions=decisions,
                    contradiction_records=contradiction_records,
                ).to_dict()
            )
        elif view_name == "preference_snapshot":
            views.append(
                build_preference_snapshot(
                    query=query,
                    results=results,
                    decisions=decisions,
                    subject_id=subject_id,
                ).to_dict()
            )
        elif view_name == "timeline_summary":
            views.append(build_timeline_summary(query=query, results=results, decisions=decisions).to_dict())
    return views
