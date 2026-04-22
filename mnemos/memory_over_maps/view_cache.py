"""Derived-view cache and invalidation engine (Phase 4)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.retrieval.base import SearchResult


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def query_fingerprint(query: str) -> str:
    return _stable_hash(query.strip().lower())[:16]


def governance_state_hash(decisions: Sequence[GovernanceDecision]) -> str:
    rows = [
        f"{d.engram_id}|{d.conflict_status}|{int(d.suppressed)}|{d.governed_score:.6f}"
        for d in sorted(decisions, key=lambda x: x.engram_id)
    ]
    return _stable_hash("||".join(rows))[:24]


def lineage_inputs(results: Sequence[SearchResult]) -> Dict[str, List[str]]:
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


def source_artifact_set_hash(artifact_ids: Sequence[str]) -> str:
    return _stable_hash("|".join(sorted(artifact_ids)))[:16]


def chunk_set_hash(chunk_ids: Sequence[str]) -> str:
    return _stable_hash("|".join(sorted(chunk_ids)))[:16]


def build_cache_key(
    *,
    view_type: str,
    query_fingerprint_value: str,
    artifact_ids: Sequence[str],
    chunk_ids: Sequence[str],
    governance_state_hash_value: str,
    synthesis_policy_version: str = "default",
) -> str:
    parts = [
        view_type,
        query_fingerprint_value,
        source_artifact_set_hash(artifact_ids),
        chunk_set_hash(chunk_ids),
        governance_state_hash_value,
        synthesis_policy_version,
    ]
    return _stable_hash("||".join(parts))


@dataclass
class CacheEntry:
    key: str
    view: Dict[str, Any]
    created_at: float
    last_verified_at: float
    invalidated: bool = False
    invalidated_reason: Optional[str] = None
    dependency_refs: Dict[str, Any] = field(default_factory=dict)


class DerivedViewCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._entries: Dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._invalidation_event_count = 0
        self._invalidated_key_total = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._entries.get(key)
        if not entry:
            self._miss_count += 1
            return None
        if entry.invalidated:
            self._miss_count += 1
            return None
        now = time.time()
        if (now - entry.created_at) > self._ttl_seconds:
            entry.invalidated = True
            entry.invalidated_reason = "ttl_expired"
            self._miss_count += 1
            return None
        entry.last_verified_at = now
        self._hit_count += 1
        return dict(entry.view)

    def set(self, *, key: str, view: Dict[str, Any], dependency_refs: Dict[str, Any]) -> None:
        now = time.time()
        self._entries[key] = CacheEntry(
            key=key,
            view=dict(view),
            created_at=now,
            last_verified_at=now,
            dependency_refs=dict(dependency_refs),
        )

    def invalidate(
        self,
        *,
        event_type: str,
        refs: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        refs = refs or {}
        impacted: List[str] = []
        reasons: Dict[str, str] = {}
        for key, entry in list(self._entries.items()):
            reason = self._match_invalidation_reason(event_type=event_type, refs=refs, entry=entry)
            if reason:
                impacted.append(key)
                reasons[key] = reason
                if not dry_run:
                    entry.invalidated = True
                    entry.invalidated_reason = reason
        self._invalidation_event_count += 1
        if not dry_run:
            self._invalidated_key_total += len(impacted)
        return {
            "event_type": event_type,
            "dry_run": dry_run,
            "checked_entries": len(self._entries),
            "impacted_keys": impacted,
            "reasons": reasons,
        }

    def _match_invalidation_reason(
        self,
        *,
        event_type: str,
        refs: Dict[str, Any],
        entry: CacheEntry,
    ) -> Optional[str]:
        deps = entry.dependency_refs or {}
        dep_artifacts = set(deps.get("artifact_ids", []))
        dep_chunks = set(deps.get("chunk_ids", []))
        dep_cluster = deps.get("contradiction_cluster_id")
        dep_governance_hash = deps.get("governance_state_hash")
        dep_synthesis_policy = deps.get("synthesis_policy_version")
        dep_lifecycle = set(deps.get("lifecycle_states", []))

        if event_type == "source_artifact_updated":
            if refs.get("artifact_id") in dep_artifacts:
                return "source_artifact_updated"
        if event_type == "source_artifact_deleted":
            if refs.get("artifact_id") in dep_artifacts:
                return "source_artifact_deleted"
        if event_type == "chunk_set_changed":
            if refs.get("chunk_id") in dep_chunks:
                return "chunk_set_changed"
        if event_type == "contradiction_cluster_changed":
            if refs.get("contradiction_cluster_id") and refs.get("contradiction_cluster_id") == dep_cluster:
                return "contradiction_cluster_changed"
        if event_type == "governance_state_changed":
            if refs.get("governance_state_hash") and refs.get("governance_state_hash") != dep_governance_hash:
                return "governance_state_changed"
        if event_type == "lifecycle_state_changed":
            if refs.get("lifecycle_state") in dep_lifecycle:
                return "lifecycle_state_changed"
        if event_type == "synthesis_config_changed":
            if refs.get("synthesis_policy_version") and refs.get("synthesis_policy_version") != dep_synthesis_policy:
                return "synthesis_config_changed"
        return None

    def stats(self) -> Dict[str, Any]:
        invalidated = sum(1 for e in self._entries.values() if e.invalidated)
        total_lookups = self._hit_count + self._miss_count
        avg_fanout = (
            self._invalidated_key_total / self._invalidation_event_count
            if self._invalidation_event_count
            else 0.0
        )
        return {
            "entry_count": len(self._entries),
            "invalidated_count": invalidated,
            "ttl_seconds": self._ttl_seconds,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_ratio": round(self._hit_count / total_lookups, 4) if total_lookups else 0.0,
            "invalidation_event_count": self._invalidation_event_count,
            "invalidated_key_total": self._invalidated_key_total,
            "avg_invalidation_fanout": round(avg_fanout, 4),
        }
