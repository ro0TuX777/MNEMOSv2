"""Post-retrieval sufficiency assessment for Evidence Admission R0.

Strictly post-retrieval: every signal here is derived from the results
``RetrievalRouter.search()`` (plus governance/abstention) actually
produced. Never consumes ``AdmissionRequestContext`` or the pre-retrieval
feature snapshot, and its reason codes are namespaced separately from
admission reason codes (design note §5 / R0 authorization point 2) so a
post-retrieval signal can never be presented as having informed the
pre-retrieval recommendation.

Thresholds below are R0 development defaults — conservative starting
points to be exercised by the development query pack, not tuned against
the frozen or fresh verification packs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .config import SUFFICIENCY_REASON_PREFIX
from .models import SufficiencyAssessment

#: Flat-score-spread threshold below which top results are considered
#: indistinguishable from each other (ambiguous ranking signal).
SCORE_SPREAD_FLAT_THRESHOLD = 0.02

_LINEAGE_KEYS = ("canonical_source_uri", "source_uri")


def _reason(code: str) -> str:
    return f"{SUFFICIENCY_REASON_PREFIX}{code}"


def _result_score(result: Any) -> float:
    return float(getattr(result, "score", 0.0) or 0.0)


def _result_has_lineage(result: Any) -> bool:
    engram = getattr(result, "engram", None)
    metadata = getattr(engram, "metadata", None) or {}
    return any(metadata.get(key) for key in _LINEAGE_KEYS) or bool(
        getattr(engram, "source", None)
    )


def _result_is_superseded(result: Any) -> bool:
    engram = getattr(result, "engram", None)
    metadata = getattr(engram, "metadata", None) or {}
    return bool(metadata.get("is_superseded", False))


def assess_sufficiency(
    results: List[Any],
    mode_meta: Optional[Dict[str, Any]] = None,
    *,
    low_relevance_abstention: Optional[Dict[str, Any]] = None,
) -> SufficiencyAssessment:
    t0 = time.perf_counter()
    mode_meta = mode_meta or {}

    if low_relevance_abstention:
        return SufficiencyAssessment(
            sufficiency="OUT_OF_SCOPE",
            reason_codes=[_reason("NORMAL_RETRIEVAL_ABSTAINED")],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    if not results:
        return SufficiencyAssessment(
            sufficiency="INSUFFICIENT_MORE_EVIDENCE_NEEDED",
            reason_codes=[_reason("NO_CANDIDATES_RETURNED")],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    top_results = results[:3]
    scores = [_result_score(r) for r in top_results]
    score_spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0

    lineage_flags = [_result_has_lineage(r) for r in top_results]
    lineage_complete_ratio = sum(lineage_flags) / len(lineage_flags) if lineage_flags else 0.0

    current_state_present = any(not _result_is_superseded(r) for r in top_results)

    envelope_meta = mode_meta.get("candidate_envelope") or {}
    source_concentration_ratio = envelope_meta.get("source_concentration_ratio")
    if source_concentration_ratio is None:
        sources = {
            (getattr(getattr(r, "engram", None), "id", None) or idx)
            for idx, r in enumerate(top_results)
        }
        source_concentration_ratio = 1.0 - (len(sources) / len(top_results) if top_results else 0.0)

    duplicate_meta = mode_meta.get("duplicate_suppression") or {}

    reason_codes: List[str] = []

    if lineage_complete_ratio < 1.0:
        reason_codes.append(_reason("SOURCE_LINEAGE_INCOMPLETE"))
    if not current_state_present:
        reason_codes.append(_reason("ONLY_SUPERSEDED_SOURCES_PRESENT"))
    if duplicate_meta.get("applied"):
        reason_codes.append(_reason("DUPLICATE_SUPPRESSION_APPLIED"))

    if score_spread < SCORE_SPREAD_FLAT_THRESHOLD and len(scores) > 1:
        reason_codes.append(_reason("SCORE_SPREAD_FLAT_LOW_DISCRIMINATION"))
        sufficiency = "AMBIGUOUS"
    elif lineage_complete_ratio < 1.0 or not current_state_present:
        sufficiency = "INSUFFICIENT_MORE_EVIDENCE_NEEDED"
    else:
        reason_codes.append(_reason("LINEAGE_COMPLETE_CURRENT_STATE_PRESENT"))
        sufficiency = "SUFFICIENT"

    return SufficiencyAssessment(
        sufficiency=sufficiency,
        reason_codes=reason_codes,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )
