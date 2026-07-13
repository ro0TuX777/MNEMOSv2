"""Data model for Evidence Admission and Budgeting R0.

All types here are plain, frozen dataclasses with no behavior beyond
``to_dict()`` — pure data, importable without a running service and without
pulling in any retrieval/governance/service module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Admission routes — a vocabulary that is deliberately separate from
#: ``RetrievalRouter``'s ``retrieval_mode``/``fusion_policy`` fields (see
#: design note §4). Never written into ``retrieval_mode``, ``fusion_policy``,
#: or ``routing_posture``.
ADMISSION_ROUTES = (
    "NO_RETRIEVAL",
    "CACHE_ONLY",
    "CUE_ONLY_LOOKUP",
    "SEMANTIC_RETRIEVAL",
    "HYBRID_RETRIEVAL",
    "ASSOCIATIVE_EXPANSION_ELIGIBLE",
    "ABSTAIN_OR_REQUEST_SCOPE",
)

#: Post-retrieval sufficiency labels.
SUFFICIENCY_LABELS = (
    "SUFFICIENT",
    "INSUFFICIENT_MORE_EVIDENCE_NEEDED",
    "AMBIGUOUS",
    "OUT_OF_SCOPE",
)

ADMISSION_STATUSES = ("recommended", "abstained", "unavailable")


@dataclass(frozen=True)
class CacheLookupResult:
    """Read-only cache freshness/scope signal handed in by the caller.

    R0 never queries ``DerivedViewCache`` itself; the caller (the service
    layer) performs the read-only peek and passes the boolean result in, so
    this module stays import-safe and has no dependency on a running cache.
    """

    cache_name: str  # "pre_cognitive" | "derived_view" | "none"
    fresh_and_scope_matched: bool


@dataclass(frozen=True)
class AdmissionRequestContext:
    """Deterministic, pre-retrieval-available inputs only.

    Every field here must be obtainable without executing
    ``RetrievalRouter.search()`` and without any new model/embedding call.
    See design note §2 for the source of each field.
    """

    top_k: int
    latency_budget_ms: Optional[float] = None
    known_collection_scope: Optional[str] = None
    cache_lookup: Optional[CacheLookupResult] = None
    complexity_label: Optional[str] = None
    complexity_confidence: Optional[float] = None
    cue_terms_matched: List[str] = field(default_factory=list)
    tag_terms_matched: List[str] = field(default_factory=list)
    explicit_artifact_ids: List[str] = field(default_factory=list)
    associative_expansion_globally_enabled: bool = False
    prior_verified_context_supplied: bool = False


@dataclass(frozen=True)
class AdmissionRecommendation:
    """Output of ``policy.recommend_admission`` — non-authoritative."""

    status: str
    recommended_route: Optional[str]
    candidate_budget: int
    context_token_budget: int
    expansion_budget: int
    latency_budget_ms: Optional[float]
    stop_condition: Optional[str]
    reason_codes: List[str]
    input_snapshot: Optional[str]
    latency_ms: float
    underlying_route: Optional[str] = None
    sufficiency: Optional[str] = None
    sufficiency_reason_codes: List[str] = field(default_factory=list)
    non_authoritative: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "recommended_route": self.recommended_route,
            "candidate_budget": self.candidate_budget,
            "context_token_budget": self.context_token_budget,
            "expansion_budget": self.expansion_budget,
            "latency_budget_ms": self.latency_budget_ms,
            "stop_condition": self.stop_condition,
            "reason_codes": list(self.reason_codes),
            "sufficiency": self.sufficiency,
            "sufficiency_reason_codes": list(self.sufficiency_reason_codes),
            "input_snapshot": self.input_snapshot,
            "latency_ms": round(self.latency_ms, 3),
            "non_authoritative": self.non_authoritative,
        }


@dataclass(frozen=True)
class SufficiencyAssessment:
    """Output of ``sufficiency.assess_sufficiency`` — post-retrieval only."""

    sufficiency: str
    reason_codes: List[str]
    latency_ms: float
    non_authoritative: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sufficiency": self.sufficiency,
            "sufficiency_reason_codes": list(self.reason_codes),
            "sufficiency_latency_ms": round(self.latency_ms, 3),
            "non_authoritative": self.non_authoritative,
        }
