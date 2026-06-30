"""Non-authoritative budget defaults for Evidence Admission R0.

Maps an admission route to the bounded ``candidate_budget`` /
``context_token_budget`` / ``expansion_budget`` defaults from the R0
authorization. Pure lookup — these are R0 evaluation defaults, not
production settings, and nothing here is enforced anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Spec-literal "Initial bounded defaults" table.
ROUTE_BUDGET_DEFAULTS: Dict[str, Dict[str, int]] = {
    "NO_RETRIEVAL": {"candidate_budget": 0, "context_token_budget": 0, "expansion_budget": 0},
    "CACHE_ONLY": {"candidate_budget": 1, "context_token_budget": 400, "expansion_budget": 0},
    "CUE_ONLY_LOOKUP": {"candidate_budget": 2, "context_token_budget": 600, "expansion_budget": 0},
    "SEMANTIC_RETRIEVAL": {"candidate_budget": 8, "context_token_budget": 1200, "expansion_budget": 0},
    "HYBRID_RETRIEVAL": {"candidate_budget": 12, "context_token_budget": 1800, "expansion_budget": 0},
    "ABSTAIN_OR_REQUEST_SCOPE": {"candidate_budget": 0, "context_token_budget": 0, "expansion_budget": 0},
}

#: expansion_budget added on top of the underlying route's budget for
#: ASSOCIATIVE_EXPANSION_ELIGIBLE. context_token_budget sees "no increase in
#: R0" per spec — only the underlying route's value is used.
ASSOCIATIVE_EXPANSION_BUDGET_ADD = 3

STOP_CONDITIONS: Dict[str, str] = {
    "NO_RETRIEVAL": "no_retrieval_required",
    "ABSTAIN_OR_REQUEST_SCOPE": "scope_clarification_required",
}
DEFAULT_STOP_CONDITION = "minimum_evidence_satisfied"


def resolve_budget(
    route: str,
    *,
    underlying_route: Optional[str] = None,
    latency_budget_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve the bounded budget object for ``route``."""
    if route == "ASSOCIATIVE_EXPANSION_ELIGIBLE":
        base_route = underlying_route or "SEMANTIC_RETRIEVAL"
        base = ROUTE_BUDGET_DEFAULTS.get(base_route, ROUTE_BUDGET_DEFAULTS["SEMANTIC_RETRIEVAL"])
        candidate_budget = base["candidate_budget"]
        context_token_budget = base["context_token_budget"]
        expansion_budget = ASSOCIATIVE_EXPANSION_BUDGET_ADD
    else:
        base = ROUTE_BUDGET_DEFAULTS.get(route, ROUTE_BUDGET_DEFAULTS["SEMANTIC_RETRIEVAL"])
        candidate_budget = base["candidate_budget"]
        context_token_budget = base["context_token_budget"]
        expansion_budget = base["expansion_budget"]

    stop_condition = STOP_CONDITIONS.get(route, DEFAULT_STOP_CONDITION)

    return {
        "candidate_budget": candidate_budget,
        "context_token_budget": context_token_budget,
        "expansion_budget": expansion_budget,
        "latency_budget_ms": latency_budget_ms,
        "stop_condition": stop_condition,
    }
