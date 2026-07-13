"""Tests for Evidence Admission and Budgeting R1 — bounded, opt-in enforcement.

R1 is additive to R0 (untouched here — see
``tests/test_evidence_admission_and_budgeting_r0.py``). Covers:

* the pure ``r1_enforcement`` decision/override functions in isolation
  (no service, no I/O, no router);
* the two-part opt-in (env kill switch + request flag) at the service layer,
  including every "off" permutation leaving behavior byte-identical to R0;
* that a bounded route (``CUE_ONLY_LOOKUP`` / ``BOUNDED_SEMANTIC_RETRIEVAL``)
  actually changes the retrieval call, and that an insufficient bounded
  attempt triggers a mandatory, unbounded ``NORMAL_RETRIEVAL_FALLBACK``
  re-run before governance/response assembly ever see it;
* that R1 never enforces a forbidden route label.

See docs/evidence_admission_and_budgeting_r1_design_note.md and
docs/evidence_admission_and_budgeting_r1_preregistration.md for the
governing constraints.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.evidence_admission import (
    ADMISSION_ROUTES,
    ALLOWED_ENFORCED_ROUTE_LABELS,
    FORBIDDEN_ENFORCED_ROUTE_LABELS,
    R1_ENFORCEMENT_ENABLE_ENV,
    AdmissionRecommendation,
    bounded_retrieval_overrides,
    decide_enforcement,
    fallback_required,
)

_QUERY = "What is the current status of GateMem G4?"


def _recommendation(
    route: Optional[str],
    *,
    status: str = "recommended",
    reason_codes: Optional[List[str]] = None,
    candidate_budget: int = 8,
    context_token_budget: int = 1200,
) -> AdmissionRecommendation:
    return AdmissionRecommendation(
        status=status,
        recommended_route=route,
        candidate_budget=candidate_budget,
        context_token_budget=context_token_budget,
        expansion_budget=0,
        latency_budget_ms=None,
        stop_condition="minimum_evidence_satisfied",
        reason_codes=reason_codes or [],
        input_snapshot="sha256:x",
        latency_ms=0.1,
    )


# ---------------------------------------------------------------------------
# Pure decision-module tests (no service, no I/O, no router)
# ---------------------------------------------------------------------------


class TestDecideEnforcement:
    def test_cue_only_lookup_is_enforced_unchanged(self) -> None:
        rec = _recommendation("CUE_ONLY_LOOKUP", reason_codes=["ADMISSION_EXPLICIT_ARTIFACT_ID_RESOLVED_LOCALLY"])
        d = decide_enforcement(rec)
        assert d.enforced is True
        assert d.enforced_route == "CUE_ONLY_LOOKUP"
        assert d.recommended_route == "CUE_ONLY_LOOKUP"

    def test_cache_only_is_enforced_unchanged(self) -> None:
        rec = _recommendation("CACHE_ONLY", reason_codes=["ADMISSION_FRESH_CACHE_SCOPE_MATCH"])
        d = decide_enforcement(rec)
        assert d.enforced is True
        assert d.enforced_route == "CACHE_ONLY"

    def test_semantic_retrieval_maps_to_bounded_semantic_retrieval(self) -> None:
        rec = _recommendation("SEMANTIC_RETRIEVAL", reason_codes=["ADMISSION_STANDARD_LOOKUP_DEFAULT"])
        d = decide_enforcement(rec)
        assert d.enforced is True
        assert d.enforced_route == "BOUNDED_SEMANTIC_RETRIEVAL"
        assert d.recommended_route == "SEMANTIC_RETRIEVAL"

    def test_abstain_with_unknown_scope_reason_is_enforced(self) -> None:
        rec = _recommendation(
            "ABSTAIN_OR_REQUEST_SCOPE",
            status="abstained",
            reason_codes=["ADMISSION_UNKNOWN_OR_OUT_OF_SCOPE_TARGET"],
        )
        d = decide_enforcement(rec)
        assert d.enforced is True
        assert d.enforced_route == "ABSTAIN_OR_REQUEST_SCOPE"

    def test_abstain_with_weak_query_reason_declines_to_normal_fallback(self) -> None:
        """The weak/underspecified-query abstain rule is currently over-broad
        in production (cue/tag registries are empty — see r1_enforcement.py
        module docstring), so R1 must not skip real retrieval for it."""
        rec = _recommendation(
            "ABSTAIN_OR_REQUEST_SCOPE",
            status="abstained",
            reason_codes=["ADMISSION_WEAK_UNDERSPECIFIED_QUERY"],
        )
        d = decide_enforcement(rec)
        assert d.enforced is False
        assert d.enforced_route == "NORMAL_RETRIEVAL_FALLBACK"

    def test_hybrid_retrieval_is_never_enforced(self) -> None:
        rec = _recommendation("HYBRID_RETRIEVAL", reason_codes=["ADMISSION_MULTI_SOURCE_SYNTHESIS_CLASS"])
        d = decide_enforcement(rec)
        assert d.enforced is False
        assert d.enforced_route == "NORMAL_RETRIEVAL_FALLBACK"

    def test_associative_expansion_eligible_is_never_enforced(self) -> None:
        rec = _recommendation(
            "ASSOCIATIVE_EXPANSION_ELIGIBLE",
            reason_codes=["ADMISSION_TYPED_RELATION_CUE_MATCH_AND_E2_GLOBALLY_ENABLED"],
        )
        d = decide_enforcement(rec)
        assert d.enforced is False
        assert d.enforced_route == "NORMAL_RETRIEVAL_FALLBACK"

    def test_unavailable_recommendation_declines_to_normal_fallback(self) -> None:
        rec = _recommendation(None, status="unavailable", reason_codes=["ADMISSION_INTERNAL_ERROR:RuntimeError"])
        d = decide_enforcement(rec)
        assert d.enforced is False
        assert d.enforced_route == "NORMAL_RETRIEVAL_FALLBACK"

    def test_decision_never_returns_a_forbidden_route_label(self) -> None:
        for route in list(ADMISSION_ROUTES) + [None, "NO_RETRIEVAL"]:
            for reason_codes in ([], ["ADMISSION_UNKNOWN_OR_OUT_OF_SCOPE_TARGET"], ["ADMISSION_WEAK_UNDERSPECIFIED_QUERY"]):
                rec = _recommendation(route, reason_codes=reason_codes)
                d = decide_enforcement(rec)
                assert d.enforced_route in ALLOWED_ENFORCED_ROUTE_LABELS
                assert d.enforced_route not in FORBIDDEN_ENFORCED_ROUTE_LABELS

    def test_pre_and_enforcement_reason_codes_stay_in_separate_namespaces(self) -> None:
        rec = _recommendation("CUE_ONLY_LOOKUP", reason_codes=["ADMISSION_EXPLICIT_ARTIFACT_ID_RESOLVED_LOCALLY"])
        d = decide_enforcement(rec)
        assert all(code.startswith("ADMISSION_") for code in d.pre_reason_codes)
        assert all(not code.startswith("ADMISSION_") for code in d.enforcement_reason_codes)


class TestFallbackRequired:
    @pytest.mark.parametrize("route", ["CUE_ONLY_LOOKUP", "BOUNDED_SEMANTIC_RETRIEVAL"])
    def test_bounded_routes_require_fallback_when_not_sufficient(self, route: str) -> None:
        assert fallback_required(route, "SUFFICIENT") is False
        assert fallback_required(route, "INSUFFICIENT_MORE_EVIDENCE_NEEDED") is True
        assert fallback_required(route, "AMBIGUOUS") is True
        assert fallback_required(route, "OUT_OF_SCOPE") is True
        assert fallback_required(route, None) is True

    @pytest.mark.parametrize("route", ["CACHE_ONLY", "ABSTAIN_OR_REQUEST_SCOPE", "NORMAL_RETRIEVAL_FALLBACK"])
    def test_non_bounded_routes_never_require_fallback(self, route: str) -> None:
        for sufficiency in ("SUFFICIENT", "INSUFFICIENT_MORE_EVIDENCE_NEEDED", "AMBIGUOUS", "OUT_OF_SCOPE", None):
            assert fallback_required(route, sufficiency) is False


class TestBoundedRetrievalOverrides:
    def test_non_bounded_routes_return_no_overrides(self) -> None:
        for route in ("CACHE_ONLY", "ABSTAIN_OR_REQUEST_SCOPE", "NORMAL_RETRIEVAL_FALLBACK"):
            d = decide_enforcement(_recommendation("CACHE_ONLY"))
            d = type(d)(**{**d.to_dict_kwargs()}) if hasattr(d, "to_dict_kwargs") else d
            # Build directly rather than relying on decide_enforcement's own
            # route mapping, to isolate bounded_retrieval_overrides itself.
            from mnemos.retrieval.evidence_admission.r1_enforcement import EnforcementDecision

            manual = EnforcementDecision(enforced=False, enforced_route=route, recommended_route=None)
            overrides = bounded_retrieval_overrides(manual, requested_top_k=10, configured_semantic_top_k=25)
            assert overrides == {}

    def test_bounded_route_caps_top_k_to_budget_and_forces_semantic_mode(self) -> None:
        from mnemos.retrieval.evidence_admission.r1_enforcement import EnforcementDecision

        d = EnforcementDecision(
            enforced=True, enforced_route="BOUNDED_SEMANTIC_RETRIEVAL", recommended_route="SEMANTIC_RETRIEVAL",
            candidate_budget=8, context_token_budget=1200,
        )
        overrides = bounded_retrieval_overrides(d, requested_top_k=25, configured_semantic_top_k=25)
        assert overrides["top_k"] == 8
        assert overrides["semantic_top_k"] == 8
        assert overrides["retrieval_mode"] == "semantic"
        assert overrides["adaptive_routing"] is False
        assert overrides["retrieval_mode"] not in FORBIDDEN_ENFORCED_ROUTE_LABELS

    def test_bounded_route_never_exceeds_requesters_own_top_k(self) -> None:
        from mnemos.retrieval.evidence_admission.r1_enforcement import EnforcementDecision

        d = EnforcementDecision(
            enforced=True, enforced_route="CUE_ONLY_LOOKUP", recommended_route="CUE_ONLY_LOOKUP",
            candidate_budget=2, context_token_budget=600,
        )
        overrides = bounded_retrieval_overrides(d, requested_top_k=1, configured_semantic_top_k=25)
        assert overrides["top_k"] == 1  # min(requested=1, budget=2)


assert not (set(ALLOWED_ENFORCED_ROUTE_LABELS) & set(FORBIDDEN_ENFORCED_ROUTE_LABELS))


# ---------------------------------------------------------------------------
# Service-layer wiring (real MnemosRuntime.search_documents + a stub router)
# ---------------------------------------------------------------------------

pytest.importorskip("flask")

import service.app as app_mod  # noqa: E402


class _RecordingStubRouter:
    """Records every ``search()`` call's kwargs and returns one canned
    result set per call, in order (repeats the last one if exhausted)."""

    def __init__(self, hit_sequence: List[List[SearchResult]]) -> None:
        self._hit_sequence = hit_sequence
        self.calls: List[Dict[str, Any]] = []

    def search(self, *, query, top_k, filters=None, retrieval_mode="semantic", **kwargs):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters, "retrieval_mode": retrieval_mode, **kwargs})
        idx = min(len(self.calls) - 1, len(self._hit_sequence) - 1)
        hits = self._hit_sequence[idx]
        meta = {"retrieval_mode": retrieval_mode, "fusion_policy": kwargs.get("fusion_policy"), "lexical_available": True}
        return list(hits), meta

    def index(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Evidence Admission R1 must never call a write/index path")

    def stats(self) -> Dict[str, Any]:
        return {}


def _hit(score: float, *, source_uri: Optional[str] = "docs/x.md") -> SearchResult:
    metadata: Dict[str, Any] = {}
    if source_uri:
        metadata["source_uri"] = source_uri
    return SearchResult(engram=Engram(id=f"e:{score}:{source_uri}", content="x", metadata=metadata), score=score, tier="qdrant")


_SUFFICIENT_HITS = [_hit(0.95), _hit(0.40), _hit(0.10)]
_INSUFFICIENT_HITS = [_hit(0.9, source_uri=None), _hit(0.4, source_uri=None)]


def _runtime_with_router(router: _RecordingStubRouter) -> Any:
    from service.app import MnemosRuntime

    rt = MnemosRuntime()
    rt._config = SimpleNamespace(
        retrieval_mode="semantic", fusion_policy="balanced", explain_default=False,
        lexical_top_k=25, semantic_top_k=25, has_compression=False, quant_bits=4,
        memory_over_maps_phase2=False, memory_over_maps_phase3=False, memory_over_maps_phase4=False,
        embedding_model="BAAI/bge-base-en-v1.5", adaptive_routing=False,
        qdrant_collection="test_collection",
    )
    rt._router = router
    rt._semantic_fusion = None
    rt._lexical_tier = None
    rt._ledger = None
    rt._view_cache = None
    rt._status = "healthy"
    rt._error = None
    return rt


class TestR1TwoPartOptIn:
    def test_neither_flag_nor_env_set_adds_no_meta_key(self, monkeypatch) -> None:
        monkeypatch.delenv(R1_ENFORCEMENT_ENABLE_ENV, raising=False)
        router = _RecordingStubRouter([_SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        out = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        )
        assert "evidence_admission_r1_enforcement" not in out["meta"]

    def test_request_flag_set_but_env_off_reports_globally_disabled_and_changes_nothing(self, monkeypatch) -> None:
        monkeypatch.delenv(R1_ENFORCEMENT_ENABLE_ENV, raising=False)
        router = _RecordingStubRouter([_SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        baseline_router = _RecordingStubRouter([_SUFFICIENT_HITS])
        baseline_rt = _runtime_with_router(baseline_router)
        baseline = baseline_rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        )
        out = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_enforce=True,
        )
        block = out["meta"]["evidence_admission_r1_enforcement"]
        assert block == {
            "requested": True,
            "globally_enabled": False,
            "decision": None,
            "fallback_triggered": False,
            "final_route_served": None,
        }
        assert [r["engram"]["id"] for r in out["results"]] == [r["engram"]["id"] for r in baseline["results"]]
        assert len(router.calls) == 1
        assert router.calls[0]["top_k"] == 5
        assert router.calls[0]["retrieval_mode"] == "semantic"

    def test_env_on_but_request_flag_absent_adds_no_meta_key(self, monkeypatch) -> None:
        monkeypatch.setenv(R1_ENFORCEMENT_ENABLE_ENV, "true")
        router = _RecordingStubRouter([_SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        out = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        )
        assert "evidence_admission_r1_enforcement" not in out["meta"]
        assert len(router.calls) == 1
        assert router.calls[0]["top_k"] == 5

    def test_malformed_env_value_is_treated_as_disabled(self, monkeypatch) -> None:
        monkeypatch.setenv(R1_ENFORCEMENT_ENABLE_ENV, "TRUE_ISH_BUT_NOT_EXACT")
        router = _RecordingStubRouter([_SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        out = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_enforce=True,
        )
        assert out["meta"]["evidence_admission_r1_enforcement"]["globally_enabled"] is False


class TestR1BoundedEnforcement:
    def test_explicit_artifact_id_enforces_bounded_cue_only_lookup(self, monkeypatch) -> None:
        monkeypatch.setenv(R1_ENFORCEMENT_ENABLE_ENV, "true")
        router = _RecordingStubRouter([_SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        out = rt.search_documents(
            query=_QUERY, top_k=25, tiers=None,
            filters={"artifact_id": "doc:gatemem"},
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_enforce=True,
        )
        block = out["meta"]["evidence_admission_r1_enforcement"]
        assert block["requested"] is True
        assert block["globally_enabled"] is True
        assert block["decision"]["enforced"] is True
        assert block["decision"]["enforced_route"] == "CUE_ONLY_LOOKUP"
        assert block["fallback_triggered"] is False
        assert block["final_route_served"] == "CUE_ONLY_LOOKUP"
        # Exactly one bounded call — no fallback re-run needed since the hit
        # set is SUFFICIENT.
        assert len(router.calls) == 1
        assert router.calls[0]["top_k"] == 2  # R0's CUE_ONLY_LOOKUP candidate_budget
        assert router.calls[0]["retrieval_mode"] == "semantic"
        assert router.calls[0]["adaptive_routing"] is False

    def test_default_route_enforces_bounded_semantic_retrieval(self, monkeypatch) -> None:
        monkeypatch.setenv(R1_ENFORCEMENT_ENABLE_ENV, "true")
        router = _RecordingStubRouter([_SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        out = rt.search_documents(
            query="tell me about the project architecture overview", top_k=25, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_enforce=True,
        )
        block = out["meta"]["evidence_admission_r1_enforcement"]
        assert block["decision"]["enforced_route"] == "BOUNDED_SEMANTIC_RETRIEVAL"
        assert block["final_route_served"] == "BOUNDED_SEMANTIC_RETRIEVAL"
        assert len(router.calls) == 1
        assert router.calls[0]["top_k"] == 8  # R0's SEMANTIC_RETRIEVAL candidate_budget

    def test_insufficient_bounded_attempt_triggers_mandatory_fallback(self, monkeypatch) -> None:
        monkeypatch.setenv(R1_ENFORCEMENT_ENABLE_ENV, "true")
        router = _RecordingStubRouter([_INSUFFICIENT_HITS, _SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        out = rt.search_documents(
            query="tell me about the project architecture overview", top_k=25, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_enforce=True,
        )
        block = out["meta"]["evidence_admission_r1_enforcement"]
        assert block["fallback_triggered"] is True
        assert block["final_route_served"] == "NORMAL_RETRIEVAL_FALLBACK"
        # Bounded attempt, then one unbounded fallback re-run.
        assert len(router.calls) == 2
        assert router.calls[0]["top_k"] == 8
        assert router.calls[1]["top_k"] == 25  # caller's original, unbounded request
        # Final results served are the fallback run's (SUFFICIENT) hits, not
        # the bounded attempt's insufficient ones.
        served_ids = [r["engram"]["id"] for r in out["results"]]
        assert served_ids == [f"e:{h.score}:{h.engram.metadata.get('source_uri')}" for h in _SUFFICIENT_HITS]

    def test_forbidden_route_labels_never_appear_as_retrieval_mode(self, monkeypatch) -> None:
        monkeypatch.setenv(R1_ENFORCEMENT_ENABLE_ENV, "true")
        router = _RecordingStubRouter([_SUFFICIENT_HITS, _SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        rt.search_documents(
            query="tell me about the project architecture overview", top_k=25, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_enforce=True,
        )
        for call in router.calls:
            assert call["retrieval_mode"] not in FORBIDDEN_ENFORCED_ROUTE_LABELS

    def test_r1_never_calls_index(self, monkeypatch) -> None:
        """The stub router raises AssertionError if .index() is ever called;
        a passing run is itself the no-durable-write assertion."""
        monkeypatch.setenv(R1_ENFORCEMENT_ENABLE_ENV, "true")
        router = _RecordingStubRouter([_SUFFICIENT_HITS])
        rt = _runtime_with_router(router)
        rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters={"artifact_id": "doc:gatemem"},
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_enforce=True,
        )
