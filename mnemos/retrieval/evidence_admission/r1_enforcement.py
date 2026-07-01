"""Evidence Admission and Budgeting R1 — bounded, opt-in enforcement.

R1 is additive to R0. R0's ``recommend_admission`` / ``assess_sufficiency``
(``policy.py`` / ``sufficiency.py`` / ``budget.py``) are untouched by this
module; R1 only decides *whether and how* to act on an R0 recommendation,
behind an explicit two-part opt-in (a global operational kill switch plus a
per-request flag), and only ever produces one of a small, fixed set of
enforced route labels.

See ``docs/evidence_admission_and_budgeting_r1_design_note.md`` and
``docs/evidence_admission_and_budgeting_r1_preregistration.md`` for the
governing constraints this module implements:

* ``NO_DEFAULT_ENFORCEMENT`` / ``OPT_IN_KILL_SWITCH_ONLY`` — enforcement
  never activates unless both the global env-var gate and the per-request
  flag are true; any other state (absent, false, malformed, unsupported)
  must leave retrieval behaviorally identical to R1 not existing at all.
  The caller (``service/app.py``) is responsible for gating; this module
  assumes it is only invoked once both gates are confirmed on.
* ``NO_HYBRID_ROUTE_ENFORCEMENT_IN_R1`` — ``HYBRID_RETRIEVAL`` and
  ``ASSOCIATIVE_EXPANSION_ELIGIBLE`` (R0's vocabulary) are never mapped to
  an enforced action; they always resolve to ``NORMAL_RETRIEVAL_FALLBACK``.
* ``NORMAL_RETRIEVAL_FALLBACK`` — mandatory whenever a bounded route is not
  safely enforceable pre-retrieval, or (post-retrieval, via
  ``fallback_required``) whenever the bounded attempt's own results are not
  judged ``SUFFICIENT`` by R0's unmodified ``assess_sufficiency``.
* ``READ_ONLY_POLICY_INPUTS`` — this module reads an ``AdmissionRecommendation``
  and a ``SufficiencyAssessment.sufficiency`` label; it writes nothing back
  into either.

Conservative-by-design note on ``ABSTAIN_OR_REQUEST_SCOPE``: R0's rule table
(``policy.classify_route``) reaches this route for two different reasons —
"the collection scope itself is unknown/unconfigured" (a service-level
condition, safe to skip retrieval for) and "the query is short and matched
no cue/tag" (a query-content heuristic that is currently over-broad because
the cue/tag registries feeding ``AdmissionRequestContext`` are not yet
populated in production — see ``service/app.py``'s
``_build_admission_request_context``, which always passes empty lists).
R1 only ever enforces a pre-retrieval abstain for the first (service-scope)
reason; the second maps to ``NORMAL_RETRIEVAL_FALLBACK`` so a real query is
never silently skipped on an under-populated heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import AdmissionRecommendation

#: Request body field name callers must set to ``true`` to ask for R1
#: enforcement (as opposed to R0's read-only ``evidence_admission_shadow``).
#: Absent or false => R1 code path is not entered at all.
R1_ENFORCEMENT_REQUEST_FLAG = "evidence_admission_enforce"

#: Global operational kill switch. Defaults to disabled. Even when a caller
#: sets the request flag above, R1 only ever changes retrieval behavior when
#: this is explicitly the string "true" (case-insensitive) — any other value,
#: including absent, empty, malformed, or unsupported, keeps enforcement off.
R1_ENFORCEMENT_ENABLE_ENV = "MNEMOS_EVIDENCE_ADMISSION_R1_ENFORCEMENT_ENABLED"

#: The only route labels R1 is authorized to enforce. Matches the formal
#: pack template's ``allowed_enforced_route_labels`` exactly.
ALLOWED_ENFORCED_ROUTE_LABELS = (
    "CUE_ONLY_LOOKUP",
    "CACHE_ONLY",
    "BOUNDED_SEMANTIC_RETRIEVAL",
    "ABSTAIN_OR_REQUEST_SCOPE",
    "NORMAL_RETRIEVAL_FALLBACK",
)

#: Route/mode labels R1 must never enforce. Matches the formal pack
#: template's ``forbidden_enforced_route_labels`` exactly. Included here so
#: a single assertion at import time catches any accidental overlap with
#: ``ALLOWED_ENFORCED_ROUTE_LABELS`` above.
FORBIDDEN_ENFORCED_ROUTE_LABELS = (
    "HYBRID_RETRIEVAL",
    "ASSOCIATIVE_EXPANSION_ELIGIBLE",
    "graph_hybrid_experimental",
    "derived_facts",
    "summary_inclusion",
    "governance_override",
)

#: R0 admission reason codes that are safe to enforce a pre-retrieval
#: abstain for (service-scope-level, not query-content-level; see module
#: docstring). Any other reason behind an ``ABSTAIN_OR_REQUEST_SCOPE``
#: recommendation resolves to ``NORMAL_RETRIEVAL_FALLBACK`` instead.
_SAFE_PRERETRIEVAL_ABSTAIN_REASONS = {"ADMISSION_UNKNOWN_OR_OUT_OF_SCOPE_TARGET"}

#: R0 recommended routes R1 bounds rather than skips or passes through, and
#: the enforced label each maps to.
_BOUNDED_ROUTE_MAP = {
    "CUE_ONLY_LOOKUP": "CUE_ONLY_LOOKUP",
    "CACHE_ONLY": "CACHE_ONLY",
    "SEMANTIC_RETRIEVAL": "BOUNDED_SEMANTIC_RETRIEVAL",
}

#: Enforced routes for which a bounded retrieval call actually changes the
#: retrieval request (top_k / mode / adaptive_routing). ``CACHE_ONLY`` is
#: deliberately excluded: a real fresh cache hit already short-circuits
#: ``search_documents`` before R1 ever runs (see ``service/app.py``), so by
#: construction this module never needs to change retrieval parameters for
#: it — reaching R1 with ``CACHE_ONLY`` recommended means the cache turned
#: out not to be servable, so the caller should just proceed with the
#: request's normal (unbounded) parameters.
ROUTES_REQUIRING_BOUNDED_RETRIEVAL_CALL = ("CUE_ONLY_LOOKUP", "BOUNDED_SEMANTIC_RETRIEVAL")


@dataclass(frozen=True)
class EnforcementDecision:
    """Pre-retrieval R1 enforcement decision, derived from one R0
    ``AdmissionRecommendation``. Non-authoritative fields (``recommended_route``,
    ``pre_reason_codes``) are carried through unchanged from R0 for audit;
    ``enforced_route`` / ``enforcement_reason_codes`` are R1's own, kept in
    a separate namespace so a post-hoc reader can never confuse "why R0
    recommended this" with "why R1 did or didn't act on it"."""

    enforced: bool
    enforced_route: str
    recommended_route: Optional[str]
    pre_reason_codes: List[str] = field(default_factory=list)
    enforcement_reason_codes: List[str] = field(default_factory=list)
    candidate_budget: Optional[int] = None
    context_token_budget: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enforced": self.enforced,
            "enforced_route": self.enforced_route,
            "recommended_route": self.recommended_route,
            "pre_reason_codes": list(self.pre_reason_codes),
            "enforcement_reason_codes": list(self.enforcement_reason_codes),
            "candidate_budget": self.candidate_budget,
            "context_token_budget": self.context_token_budget,
        }


def decide_enforcement(recommendation: AdmissionRecommendation) -> EnforcementDecision:
    """Map one R0 ``AdmissionRecommendation`` onto R1's restricted,
    allowed enforcement vocabulary.

    Never returns a route from ``FORBIDDEN_ENFORCED_ROUTE_LABELS``. Defaults
    to ``NORMAL_RETRIEVAL_FALLBACK`` (``enforced=False``) for anything R1 is
    not authorized or not confident enough to enforce pre-retrieval — this
    is the "decline to enforce" path, not an error path.
    """
    route = recommendation.recommended_route
    reasons = list(recommendation.reason_codes)

    if recommendation.status == "unavailable" or route is None:
        return EnforcementDecision(
            enforced=False,
            enforced_route="NORMAL_RETRIEVAL_FALLBACK",
            recommended_route=route,
            pre_reason_codes=reasons,
            enforcement_reason_codes=["ENFORCEMENT_DECLINED_RECOMMENDATION_UNAVAILABLE"],
        )

    if route in _BOUNDED_ROUTE_MAP:
        enforced_route = _BOUNDED_ROUTE_MAP[route]
        return EnforcementDecision(
            enforced=True,
            enforced_route=enforced_route,
            recommended_route=route,
            pre_reason_codes=reasons,
            enforcement_reason_codes=[f"ENFORCEMENT_APPLIED_{enforced_route}"],
            candidate_budget=recommendation.candidate_budget,
            context_token_budget=recommendation.context_token_budget,
        )

    if route == "ABSTAIN_OR_REQUEST_SCOPE":
        if any(r in _SAFE_PRERETRIEVAL_ABSTAIN_REASONS for r in reasons):
            return EnforcementDecision(
                enforced=True,
                enforced_route="ABSTAIN_OR_REQUEST_SCOPE",
                recommended_route=route,
                pre_reason_codes=reasons,
                enforcement_reason_codes=["ENFORCEMENT_APPLIED_ABSTAIN_UNKNOWN_SCOPE"],
            )
        return EnforcementDecision(
            enforced=False,
            enforced_route="NORMAL_RETRIEVAL_FALLBACK",
            recommended_route=route,
            pre_reason_codes=reasons,
            enforcement_reason_codes=["ENFORCEMENT_DECLINED_ABSTAIN_REASON_NOT_PRERETRIEVAL_SAFE"],
        )

    # HYBRID_RETRIEVAL, ASSOCIATIVE_EXPANSION_ELIGIBLE, NO_RETRIEVAL, or any
    # route outside R0's current vocabulary: never enforced in R1.
    return EnforcementDecision(
        enforced=False,
        enforced_route="NORMAL_RETRIEVAL_FALLBACK",
        recommended_route=route,
        pre_reason_codes=reasons,
        enforcement_reason_codes=[f"ENFORCEMENT_DECLINED_ROUTE_NOT_R1_ENFORCEABLE:{route}"],
    )


def fallback_required(enforced_route: str, sufficiency: Optional[str]) -> bool:
    """Whether ``NORMAL_RETRIEVAL_FALLBACK`` must fire after the fact.

    Only bounded retrieval routes can trigger a post-hoc fallback — R1
    changed the retrieval request for these, so it is responsible for
    checking the result was good enough. ``ABSTAIN_OR_REQUEST_SCOPE``
    skipped retrieval entirely (nothing to assess); ``CACHE_ONLY`` and
    ``NORMAL_RETRIEVAL_FALLBACK`` never changed the retrieval request in
    the first place.
    """
    if enforced_route not in ROUTES_REQUIRING_BOUNDED_RETRIEVAL_CALL:
        return False
    return sufficiency != "SUFFICIENT"


def bounded_retrieval_overrides(
    decision: EnforcementDecision,
    *,
    requested_top_k: int,
    configured_semantic_top_k: int,
) -> Dict[str, Any]:
    """Retrieval-call keyword overrides for a bounded enforced route.

    Returns an empty dict for any route that should use the request's
    normal (unbounded) parameters unchanged — including ``CACHE_ONLY``,
    ``ABSTAIN_OR_REQUEST_SCOPE``, and ``NORMAL_RETRIEVAL_FALLBACK`` — so
    callers can always do ``**bounded_retrieval_overrides(...)`` safely.
    """
    if decision.enforced_route not in ROUTES_REQUIRING_BOUNDED_RETRIEVAL_CALL:
        return {}
    budget_cap = decision.candidate_budget or requested_top_k
    bounded_top_k = max(1, min(int(requested_top_k), int(budget_cap)))
    bounded_semantic_top_k = max(1, min(int(configured_semantic_top_k), int(budget_cap)))
    return {
        "top_k": bounded_top_k,
        "semantic_top_k": bounded_semantic_top_k,
        # "semantic" is always a supported RetrievalRouter.search() mode and
        # is never in FORBIDDEN_ENFORCED_ROUTE_LABELS; forcing it (rather
        # than leaving the caller's requested/configured mode in place) is
        # what makes CUE_ONLY_LOOKUP / BOUNDED_SEMANTIC_RETRIEVAL bounded
        # instead of an unmodified hybrid/graph call.
        "retrieval_mode": "semantic",
        # Adaptive routing can independently escalate "semantic" requests
        # to hybrid based on its own complexity classification; disabling
        # it here is what makes the "semantic" override above actually
        # hold, per NO_HYBRID_ROUTE_ENFORCEMENT_IN_R1.
        "adaptive_routing": False,
    }


assert not (set(ALLOWED_ENFORCED_ROUTE_LABELS) & set(FORBIDDEN_ENFORCED_ROUTE_LABELS))
assert set(_BOUNDED_ROUTE_MAP.values()) <= set(ALLOWED_ENFORCED_ROUTE_LABELS)
