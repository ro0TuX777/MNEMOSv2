"""Deterministic admission rule table for Evidence Admission R0.

Pure function of a feature snapshot (see ``feature_extraction.py``) to an
admission route plus reason codes. This is the "Initial Deterministic
Admission Rules" table from the R0 authorization — conservative defaults
for the development pack to exercise, not yet tuned against any frozen or
fresh verification pack.

Rules are evaluated in order; the first match wins.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .config import ADMISSION_REASON_PREFIX
from .models import ADMISSION_ROUTES
from .feature_extraction import MIN_TOKEN_COUNT_FOR_CONFIDENT_SCOPE


def _reason(code: str) -> str:
    return f"{ADMISSION_REASON_PREFIX}{code}"


def classify_route(features: Dict[str, Any]) -> Tuple[str, Optional[str], List[str]]:
    """
    Return ``(route, underlying_route, reason_codes)``.

    ``underlying_route`` is only set for ``ASSOCIATIVE_EXPANSION_ELIGIBLE``:
    it names the route whose budget defaults the expansion-eligible
    recommendation inherits (spec: "expansion_budget = normal route
    budget"), so ``budget.py`` does not need to re-derive it.
    """
    explicit_artifact_ids = features["explicit_artifact_ids"]
    cache_fresh = features["cache_fresh_and_scope_matched"]
    known_scope = features["known_collection_scope"]
    complexity_label = features["complexity_label"]
    cue_terms = features["cue_terms_matched"]
    tag_terms = features["tag_terms_matched"]
    e2_globally_enabled = features["associative_expansion_globally_enabled"]
    token_count = features["query_token_count"]

    # 1. Explicit known artifact/source ID resolves locally.
    if explicit_artifact_ids:
        return "CUE_ONLY_LOOKUP", None, [_reason("EXPLICIT_ARTIFACT_ID_RESOLVED_LOCALLY")]

    # 2. Fresh exact cache entry exists and scope matches.
    if cache_fresh:
        return "CACHE_ONLY", None, [_reason("FRESH_CACHE_SCOPE_MATCH")]

    # 3. Query is unrelated to active corpus or fails scope checks.
    if not known_scope:
        return "ABSTAIN_OR_REQUEST_SCOPE", None, [_reason("UNKNOWN_OR_OUT_OF_SCOPE_TARGET")]

    # 4. Typed-relation cue/tag match with E2 explicitly enabled — eligible
    #    only, never auto-invoked.
    if tag_terms and e2_globally_enabled:
        underlying = "HYBRID_RETRIEVAL" if complexity_label == "CLASS_C" else "SEMANTIC_RETRIEVAL"
        return (
            "ASSOCIATIVE_EXPANSION_ELIGIBLE",
            underlying,
            [_reason("TYPED_RELATION_CUE_MATCH_AND_E2_GLOBALLY_ENABLED")],
        )

    # 5. Direct status/factoid query with high-confidence exact cue match.
    if complexity_label == "CLASS_A":
        if cue_terms:
            return "CUE_ONLY_LOOKUP", None, [_reason("DIRECT_FACTOID_EXACT_CUE_MATCH")]
        return "SEMANTIC_RETRIEVAL", None, [_reason("DIRECT_FACTOID_NO_CUE_MATCH")]

    # 6. Multi-source comparison, contradiction, timeline, or dependency
    #    query (CLASS_C global/synthesis, CLASS_B multi-hop/relationship).
    if complexity_label == "CLASS_C":
        return "HYBRID_RETRIEVAL", None, [_reason("MULTI_SOURCE_SYNTHESIS_CLASS")]
    if complexity_label == "CLASS_B":
        return "HYBRID_RETRIEVAL", None, [_reason("MULTI_HOP_RELATIONSHIP_CLASS")]

    # 7. Weak, underspecified, or ambiguous query with no useful evidence
    #    path: no complexity signal, no cue/tag match, very short query.
    if complexity_label is None and not cue_terms and not tag_terms and token_count < MIN_TOKEN_COUNT_FOR_CONFIDENT_SCOPE:
        return "ABSTAIN_OR_REQUEST_SCOPE", None, [_reason("WEAK_UNDERSPECIFIED_QUERY")]

    # 8. Standard project/context lookup (default).
    return "SEMANTIC_RETRIEVAL", None, [_reason("STANDARD_LOOKUP_DEFAULT")]


assert set(ADMISSION_ROUTES) >= {
    "CUE_ONLY_LOOKUP",
    "CACHE_ONLY",
    "ABSTAIN_OR_REQUEST_SCOPE",
    "ASSOCIATIVE_EXPANSION_ELIGIBLE",
    "SEMANTIC_RETRIEVAL",
    "HYBRID_RETRIEVAL",
}
