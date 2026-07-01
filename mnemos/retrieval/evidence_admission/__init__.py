"""Evidence Admission and Budgeting R0 — read-only, shadow-only.

Public surface:

* ``recommend_admission(query, request_context)`` — pre-retrieval only.
  Deterministic; never raises (degrades to ``status="unavailable"``); uses
  no inputs that depend on having executed normal retrieval.
* ``assess_sufficiency(results, mode_meta, low_relevance_abstention=...)`` —
  post-retrieval only. Consumes real retrieval outputs; never feeds back
  into ``recommend_admission``'s inputs or reason codes.

Neither function calls ``RetrievalRouter.search()``, governance, or any
durable-write path. See
``docs/evidence_admission_and_budgeting_r0_design_note.md`` for the full
design rationale and ``docs/evidence_admission_and_budgeting_r0_checkpoint_001.md``
for the first implementation checkpoint report.
"""

from __future__ import annotations

import time

from .budget import resolve_budget
from .config import (
    EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV,
    EVIDENCE_ADMISSION_SHADOW_FLAG,
)
from .feature_extraction import extract_features, input_snapshot_hash
from .models import (
    ADMISSION_ROUTES,
    ADMISSION_STATUSES,
    SUFFICIENCY_LABELS,
    AdmissionRecommendation,
    AdmissionRequestContext,
    CacheLookupResult,
    SufficiencyAssessment,
)
from .policy import classify_route
from .r1_enforcement import (
    ALLOWED_ENFORCED_ROUTE_LABELS,
    FORBIDDEN_ENFORCED_ROUTE_LABELS,
    R1_ENFORCEMENT_ENABLE_ENV,
    R1_ENFORCEMENT_REQUEST_FLAG,
    EnforcementDecision,
    bounded_retrieval_overrides,
    decide_enforcement,
    fallback_required,
)
from .sufficiency import assess_sufficiency
from .telemetry import redact_for_telemetry

__all__ = [
    "recommend_admission",
    "assess_sufficiency",
    "AdmissionRecommendation",
    "AdmissionRequestContext",
    "CacheLookupResult",
    "SufficiencyAssessment",
    "ADMISSION_ROUTES",
    "ADMISSION_STATUSES",
    "SUFFICIENCY_LABELS",
    "EVIDENCE_ADMISSION_SHADOW_FLAG",
    "EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV",
    "redact_for_telemetry",
    # R1 — bounded, opt-in enforcement (additive to the R0 names above).
    "ALLOWED_ENFORCED_ROUTE_LABELS",
    "FORBIDDEN_ENFORCED_ROUTE_LABELS",
    "R1_ENFORCEMENT_ENABLE_ENV",
    "R1_ENFORCEMENT_REQUEST_FLAG",
    "EnforcementDecision",
    "bounded_retrieval_overrides",
    "decide_enforcement",
    "fallback_required",
]


def recommend_admission(
    query: str,
    request_context: AdmissionRequestContext,
) -> AdmissionRecommendation:
    """Pure, deterministic, pre-retrieval-only admission recommendation.

    Never raises into the caller — internal errors degrade to
    ``status="unavailable"`` so a bug here can never break a real search
    request, consistent with the E1/E2 shadow-adapter precedent.
    """
    t0 = time.perf_counter()
    try:
        features = extract_features(query, request_context)
        route, underlying_route, reason_codes = classify_route(features)
        budget = resolve_budget(
            route,
            underlying_route=underlying_route,
            latency_budget_ms=request_context.latency_budget_ms,
        )
        snapshot = input_snapshot_hash(features)
        status = "abstained" if route == "ABSTAIN_OR_REQUEST_SCOPE" else "recommended"
        return AdmissionRecommendation(
            status=status,
            recommended_route=route,
            candidate_budget=budget["candidate_budget"],
            context_token_budget=budget["context_token_budget"],
            expansion_budget=budget["expansion_budget"],
            latency_budget_ms=budget["latency_budget_ms"],
            stop_condition=budget["stop_condition"],
            reason_codes=reason_codes,
            input_snapshot=snapshot,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            underlying_route=underlying_route,
        )
    except Exception as exc:  # noqa: BLE001 - shadow path must never raise
        return AdmissionRecommendation(
            status="unavailable",
            recommended_route=None,
            candidate_budget=0,
            context_token_budget=0,
            expansion_budget=0,
            latency_budget_ms=request_context.latency_budget_ms if request_context else None,
            stop_condition=None,
            reason_codes=[f"ADMISSION_INTERNAL_ERROR:{type(exc).__name__}"],
            input_snapshot=None,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
