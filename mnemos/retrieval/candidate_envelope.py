"""Bounded candidate envelope for Phase 2 (Memory Over Maps)."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import SearchResult


@dataclass(frozen=True)
class CandidateEnvelopeConfig:
    enabled: bool = False
    candidate_pool_limit: int = 40
    dedupe_similarity_threshold: float = 0.90
    max_per_source_artifact: int = 3
    diversity_policy: str = "off"
    bounded_adjudication_enabled: bool = True

    @classmethod
    def from_request(cls, request_data: Optional[Dict[str, Any]]) -> "CandidateEnvelopeConfig":
        if not isinstance(request_data, dict):
            return cls(enabled=False)
        return cls(
            enabled=bool(request_data.get("enabled", False)),
            candidate_pool_limit=max(1, int(request_data.get("candidate_pool_limit", 40))),
            dedupe_similarity_threshold=float(request_data.get("dedupe_similarity_threshold", 0.90)),
            max_per_source_artifact=max(1, int(request_data.get("max_per_source_artifact", 3))),
            diversity_policy=str(request_data.get("diversity_policy", "off")),
            bounded_adjudication_enabled=bool(request_data.get("bounded_adjudication_enabled", True)),
        )


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a.lower().strip(), b=b.lower().strip()).ratio()


def _average_pairwise_similarity(results: List[SearchResult]) -> float:
    if len(results) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            pairs += 1
            total += _text_similarity(results[i].engram.content, results[j].engram.content)
    return total / pairs if pairs else 0.0


def apply_candidate_envelope(
    candidates: List[SearchResult],
    config: CandidateEnvelopeConfig,
) -> Tuple[List[SearchResult], Dict[str, Any]]:
    """
    Narrow candidates deterministically before governance and synthesis.
    """
    initial_count = len(candidates)
    suppression_summary = {
        "duplicate_similarity": 0,
        "source_cap_exceeded": 0,
        "low_rank_after_diversity": 0,
        "bounded_limit_exceeded": 0,
        "policy_excluded": 0,
    }

    if not config.enabled or not config.bounded_adjudication_enabled:
        return candidates, {
            "enabled": False,
            "initial_candidate_count": initial_count,
            "post_dedupe_count": initial_count,
            "post_source_cap_count": initial_count,
            "post_diversity_count": initial_count,
            "final_candidate_count": initial_count,
            "suppression_summary": suppression_summary,
            "source_distribution": {},
            "source_concentration_ratio": 0.0,
            "average_pairwise_similarity": round(_average_pairwise_similarity(candidates), 4),
            "diversity_policy_applied": "off",
            "config_snapshot": config.__dict__,
        }

    # 1) Dedupe by near-identical content.
    deduped: List[SearchResult] = []
    for hit in candidates:
        is_dup = False
        for kept in deduped:
            sim = _text_similarity(hit.engram.content, kept.engram.content)
            if sim >= config.dedupe_similarity_threshold:
                is_dup = True
                break
        if is_dup:
            suppression_summary["duplicate_similarity"] += 1
        else:
            deduped.append(hit)

    # 2) Source balancing.
    per_source: Dict[str, int] = {}
    source_balanced: List[SearchResult] = []
    for hit in deduped:
        source_artifact = hit.engram.lineage().get("artifact_id") or f"artifact:{hit.engram.id}"
        count = per_source.get(source_artifact, 0)
        if count >= config.max_per_source_artifact:
            suppression_summary["source_cap_exceeded"] += 1
            continue
        per_source[source_artifact] = count + 1
        source_balanced.append(hit)

    # 3) Diversity policy (Phase 2 default: off/no-op).
    post_diversity = source_balanced
    diversity_applied = "off"
    if config.diversity_policy != "off":
        diversity_applied = config.diversity_policy

    # 4) Hard limit.
    final = post_diversity[: config.candidate_pool_limit]
    suppression_summary["bounded_limit_exceeded"] = max(0, len(post_diversity) - len(final))

    source_distribution: Dict[str, int] = {}
    for hit in final:
        source_artifact = hit.engram.lineage().get("artifact_id") or f"artifact:{hit.engram.id}"
        source_distribution[source_artifact] = source_distribution.get(source_artifact, 0) + 1

    source_concentration_ratio = 0.0
    if final:
        source_concentration_ratio = max(source_distribution.values()) / len(final)

    meta = {
        "enabled": True,
        "initial_candidate_count": initial_count,
        "post_dedupe_count": len(deduped),
        "post_source_cap_count": len(source_balanced),
        "post_diversity_count": len(post_diversity),
        "final_candidate_count": len(final),
        "suppression_summary": suppression_summary,
        "source_distribution": source_distribution,
        "source_concentration_ratio": round(source_concentration_ratio, 4),
        "average_pairwise_similarity": round(_average_pairwise_similarity(final), 4),
        "diversity_policy_applied": diversity_applied,
        "config_snapshot": config.__dict__,
    }
    return final, meta

