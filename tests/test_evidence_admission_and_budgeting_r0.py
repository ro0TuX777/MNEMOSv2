"""Tests for Evidence Admission and Budgeting R0.

Covers the R0 authorization invariants: flag-off response identity,
deterministic admission recommendation, the two-phase pre-/post-retrieval
split (distinct inputs, outputs, and reason-code namespaces), the global
operational gate's three-way behavior, bounded candidate/context budgets,
no durable writes, no governance mutation, no normal-result suppression or
reranking, and telemetry redaction.

See docs/evidence_admission_and_budgeting_r0_design_note.md and
docs/evidence_admission_and_budgeting_r0_checkpoint_001.md for the full
design rationale.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from mnemos.engram.model import Engram
from mnemos.memory_over_maps.view_cache import DerivedViewCache, build_retrieval_cache_context
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.evidence_admission import (
    ADMISSION_ROUTES,
    EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV,
    AdmissionRequestContext,
    CacheLookupResult,
    assess_sufficiency,
    recommend_admission,
)
from mnemos.retrieval.evidence_admission.feature_extraction import extract_features
from mnemos.retrieval.evidence_admission.policy import classify_route
from mnemos.retrieval.evidence_admission.telemetry import redact_for_telemetry
from mnemos.retrieval.associative_expansion import CANDIDATE_EXPANSION_ENABLE_ENV

_QUERY = "What is the current status of GateMem G4?"


# ---------------------------------------------------------------------------
# Pure policy/budget unit tests (no service, no I/O)
# ---------------------------------------------------------------------------


class TestAdmissionPolicy:
    def test_explicit_artifact_id_routes_cue_only_lookup(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default", explicit_artifact_ids=["doc:gatemem"])
        rec = recommend_admission(_QUERY, ctx)
        assert rec.status == "recommended"
        assert rec.recommended_route == "CUE_ONLY_LOOKUP"
        assert any(code.startswith("ADMISSION_") for code in rec.reason_codes)

    def test_fresh_cache_routes_cache_only(self) -> None:
        ctx = AdmissionRequestContext(
            top_k=5,
            known_collection_scope="mnemos_default",
            cache_lookup=CacheLookupResult(cache_name="pre_cognitive", fresh_and_scope_matched=True),
        )
        rec = recommend_admission(_QUERY, ctx)
        assert rec.recommended_route == "CACHE_ONLY"
        assert rec.candidate_budget == 1
        assert rec.context_token_budget == 400

    def test_unknown_scope_routes_abstain(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope=None)
        rec = recommend_admission(_QUERY, ctx)
        assert rec.recommended_route == "ABSTAIN_OR_REQUEST_SCOPE"
        assert rec.status == "abstained"
        assert rec.candidate_budget == 0
        assert rec.context_token_budget == 0

    def test_class_a_with_cue_routes_cue_only_lookup(self) -> None:
        ctx = AdmissionRequestContext(
            top_k=5, known_collection_scope="mnemos_default",
            complexity_label="CLASS_A", cue_terms_matched=["gatemem-g4"],
        )
        rec = recommend_admission(_QUERY, ctx)
        assert rec.recommended_route == "CUE_ONLY_LOOKUP"

    def test_class_a_without_cue_routes_semantic(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default", complexity_label="CLASS_A")
        rec = recommend_admission(_QUERY, ctx)
        assert rec.recommended_route == "SEMANTIC_RETRIEVAL"

    def test_class_c_routes_hybrid(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default", complexity_label="CLASS_C")
        rec = recommend_admission(_QUERY, ctx)
        assert rec.recommended_route == "HYBRID_RETRIEVAL"
        assert rec.candidate_budget == 12
        assert rec.context_token_budget == 1800

    def test_weak_underspecified_query_routes_abstain(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default")
        rec = recommend_admission("g4", ctx)
        assert rec.recommended_route == "ABSTAIN_OR_REQUEST_SCOPE"

    def test_default_standard_lookup_routes_semantic(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default")
        rec = recommend_admission("tell me about the project architecture overview", ctx)
        assert rec.recommended_route == "SEMANTIC_RETRIEVAL"

    def test_associative_eligible_requires_both_tag_match_and_global_enable(self) -> None:
        ctx_enabled = AdmissionRequestContext(
            top_k=5, known_collection_scope="mnemos_default",
            tag_terms_matched=["blocked_by"], associative_expansion_globally_enabled=True,
        )
        rec_enabled = recommend_admission(_QUERY, ctx_enabled)
        assert rec_enabled.recommended_route == "ASSOCIATIVE_EXPANSION_ELIGIBLE"
        assert rec_enabled.expansion_budget == 3

        ctx_disabled = AdmissionRequestContext(
            top_k=5, known_collection_scope="mnemos_default",
            tag_terms_matched=["blocked_by"], associative_expansion_globally_enabled=False,
        )
        rec_disabled = recommend_admission(_QUERY, ctx_disabled)
        assert rec_disabled.recommended_route != "ASSOCIATIVE_EXPANSION_ELIGIBLE"

    def test_associative_eligible_never_invokes_e2_engine(self, monkeypatch) -> None:
        """R0 may only *label* a request eligible — it must never call the
        E0/E1/E2 routers/engines itself."""
        import mnemos.retrieval.evidence_admission.policy as policy_mod

        def _fail_if_called(*args, **kwargs):  # pragma: no cover
            raise AssertionError("Evidence Admission R0 must never invoke E0/E1/E2 engines")

        # Nothing in policy.py imports AssociativeRouter/CandidateExpansionEngine;
        # assert that remains true rather than monkeypatching a nonexistent hook.
        assert "AssociativeRouter" not in dir(policy_mod)
        assert "CandidateExpansionEngine" not in dir(policy_mod)

    def test_all_route_budgets_are_bounded_non_negative(self) -> None:
        for label, kwargs in [
            ("no_scope", {"known_collection_scope": None}),
            ("cache", {"known_collection_scope": "x", "cache_lookup": CacheLookupResult("pre_cognitive", True)}),
            ("artifact", {"known_collection_scope": "x", "explicit_artifact_ids": ["a"]}),
            ("class_a", {"known_collection_scope": "x", "complexity_label": "CLASS_A"}),
            ("class_b", {"known_collection_scope": "x", "complexity_label": "CLASS_B"}),
            ("class_c", {"known_collection_scope": "x", "complexity_label": "CLASS_C"}),
            ("default", {"known_collection_scope": "x", "complexity_label": "CLASS_A", "cue_terms_matched": []}),
        ]:
            ctx = AdmissionRequestContext(top_k=5, **kwargs)
            rec = recommend_admission("a reasonably long standard lookup query", ctx)
            assert rec.recommended_route in ADMISSION_ROUTES, label
            assert rec.candidate_budget >= 0, label
            assert rec.context_token_budget >= 0, label
            assert rec.expansion_budget >= 0, label

    def test_deterministic_recommendation_for_same_input_snapshot(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default", complexity_label="CLASS_C")
        rec_a = recommend_admission(_QUERY, ctx)
        rec_b = recommend_admission(_QUERY, ctx)
        assert rec_a.input_snapshot == rec_b.input_snapshot
        assert rec_a.recommended_route == rec_b.recommended_route
        assert rec_a.reason_codes == rec_b.reason_codes
        assert rec_a.candidate_budget == rec_b.candidate_budget

    def test_input_snapshot_changes_with_query(self) -> None:
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default")
        rec_a = recommend_admission("query one", ctx)
        rec_b = recommend_admission("query two", ctx)
        assert rec_a.input_snapshot != rec_b.input_snapshot

    def test_recommend_admission_never_raises_degrades_to_unavailable(self, monkeypatch) -> None:
        import mnemos.retrieval.evidence_admission as ea_mod

        def _boom(features):
            raise RuntimeError("simulated internal failure")

        monkeypatch.setattr(ea_mod, "classify_route", _boom)
        ctx = AdmissionRequestContext(top_k=5, known_collection_scope="mnemos_default")
        rec = recommend_admission(_QUERY, ctx)
        assert rec.status == "unavailable"
        assert rec.recommended_route is None
        assert rec.candidate_budget == 0
        assert any("ADMISSION_INTERNAL_ERROR" in code for code in rec.reason_codes)


# ---------------------------------------------------------------------------
# Pure sufficiency unit tests — strictly post-retrieval inputs
# ---------------------------------------------------------------------------


def _hit(score: float, *, source_uri: Optional[str] = "docs/x.md", superseded: bool = False) -> SearchResult:
    metadata: Dict[str, Any] = {}
    if source_uri:
        metadata["source_uri"] = source_uri
    if superseded:
        metadata["is_superseded"] = True
    return SearchResult(engram=Engram(id=f"e:{score}", content="x", metadata=metadata), score=score, tier="qdrant")


class TestSufficiency:
    def test_abstention_yields_out_of_scope(self) -> None:
        out = assess_sufficiency([], {}, low_relevance_abstention={"applied": True})
        assert out.sufficiency == "OUT_OF_SCOPE"
        assert all(code.startswith("SUFFICIENCY_") for code in out.reason_codes)

    def test_no_results_yields_insufficient(self) -> None:
        out = assess_sufficiency([], {})
        assert out.sufficiency == "INSUFFICIENT_MORE_EVIDENCE_NEEDED"

    def test_flat_score_spread_yields_ambiguous(self) -> None:
        results = [_hit(0.51), _hit(0.50), _hit(0.50)]
        out = assess_sufficiency(results, {})
        assert out.sufficiency == "AMBIGUOUS"
        assert any("SCORE_SPREAD_FLAT" in code for code in out.reason_codes)

    def test_incomplete_lineage_yields_insufficient(self) -> None:
        results = [_hit(0.9, source_uri=None), _hit(0.4, source_uri=None)]
        out = assess_sufficiency(results, {})
        assert out.sufficiency in {"INSUFFICIENT_MORE_EVIDENCE_NEEDED", "AMBIGUOUS"}

    def test_complete_lineage_and_separation_yields_sufficient(self) -> None:
        results = [_hit(0.95), _hit(0.40), _hit(0.10)]
        out = assess_sufficiency(results, {})
        assert out.sufficiency == "SUFFICIENT"

    def test_sufficiency_reason_codes_never_use_admission_prefix(self) -> None:
        results = [_hit(0.95), _hit(0.40)]
        out = assess_sufficiency(results, {})
        assert all(not code.startswith("ADMISSION_") for code in out.reason_codes)


# ---------------------------------------------------------------------------
# Telemetry redaction
# ---------------------------------------------------------------------------


class TestTelemetryRedaction:
    def test_raw_query_is_redacted(self) -> None:
        record = {"query": "What is the status of GateMem G4?", "route": "SEMANTIC_RETRIEVAL"}
        redacted = redact_for_telemetry(record)
        assert redacted["route"] == "SEMANTIC_RETRIEVAL"
        assert redacted["query"]["redacted"] is True
        assert "GateMem" not in str(redacted["query"])


# ---------------------------------------------------------------------------
# Cache peek side-effect-free check
# ---------------------------------------------------------------------------


class TestCachePeekSideEffectFree:
    def test_peek_does_not_mutate_hit_miss_counters(self) -> None:
        cache = DerivedViewCache(ttl_seconds=3600)
        ctx = build_retrieval_cache_context(
            query="hello world",
            authorized_scope="default",
            collection_snapshot="col:seed1",
            retrieval_profile="semantic|none|lexical_top_k=25|semantic_top_k=25",
            embedding_model_name="model-x",
            seed_snapshot="seed1",
        )
        cache.set_pre_cognitive(key="k1", query="hello world", cluster_id=1, view={"results": []}, cache_context=ctx)
        stats_before = cache.stats()

        fresh = cache.peek_pre_cognitive(query="hello world", cluster_id=1, cache_context=ctx)
        assert fresh is True
        assert cache.stats()["hit_count"] == stats_before["hit_count"]
        assert cache.stats()["miss_count"] == stats_before["miss_count"]

        miss = cache.peek_pre_cognitive(query="unrelated", cluster_id=999, cache_context=ctx)
        assert miss is False
        assert cache.stats()["hit_count"] == stats_before["hit_count"]
        assert cache.stats()["miss_count"] == stats_before["miss_count"]


# ---------------------------------------------------------------------------
# Service-layer wiring (Flask test client + real MnemosRuntime.search_documents)
# ---------------------------------------------------------------------------

pytest.importorskip("flask")

import service.app as app_mod  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_mod, "_ensure_runtime", lambda: None)
    app_mod.app.config["TESTING"] = True
    with app_mod.app.test_client() as c:
        yield c


def _fake_search_documents_factory(captured: Dict[str, Any]):
    def fake_search_documents(
        query, top_k, tiers, filters, retrieval_mode, fusion_policy, explain,
        governance=None, explain_governance=None, governance_profile=None,
        bounded_envelope=None, derive_views=None, latency_budget_ms=None,
        complexity_shadow=False, cognitive_cycle=None,
        associative_routing_shadow=False, associative_candidate_expansion=False,
        evidence_admission_shadow=False, evidence_admission_enforce=False,
    ):
        captured["evidence_admission_shadow"] = evidence_admission_shadow
        captured["evidence_admission_enforce"] = evidence_admission_enforce
        return {
            "status": "healthy",
            "results": [],
            "meta": {"retrieval_mode": retrieval_mode or "semantic", "fusion_policy": fusion_policy},
        }

    return fake_search_documents


class TestRequestLayerWiring:
    def test_flag_defaults_false_when_absent(self, client, monkeypatch) -> None:
        captured: Dict[str, Any] = {}
        monkeypatch.setattr(app_mod._runtime, "search_documents", _fake_search_documents_factory(captured))
        resp = client.post("/v1/mnemos/search", json={"query": _QUERY})
        assert resp.status_code == 200
        assert captured["evidence_admission_shadow"] is False

    def test_flag_true_is_forwarded(self, client, monkeypatch) -> None:
        captured: Dict[str, Any] = {}
        monkeypatch.setattr(app_mod._runtime, "search_documents", _fake_search_documents_factory(captured))
        resp = client.post("/v1/mnemos/search", json={"query": _QUERY, "evidence_admission_shadow": True})
        assert resp.status_code == 200
        assert captured["evidence_admission_shadow"] is True

    def test_non_bool_flag_is_rejected(self, client) -> None:
        resp = client.post("/v1/mnemos/search", json={"query": _QUERY, "evidence_admission_shadow": "yes"})
        assert resp.status_code == 400
        assert "evidence_admission_shadow" in resp.get_json()["error"]


class _StubRouter:
    def __init__(self, hits: List[SearchResult]) -> None:
        self._hits = hits

    def search(self, *, query, top_k, filters=None, retrieval_mode="semantic", **kwargs):
        meta = {"retrieval_mode": retrieval_mode, "fusion_policy": kwargs.get("fusion_policy"), "lexical_available": True}
        return list(self._hits), meta

    def index(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Evidence Admission R0 must never call a write/index path")

    def stats(self) -> Dict[str, Any]:
        return {}


def _runtime_with_stub_router(hits: Optional[List[SearchResult]] = None) -> Any:
    from service.app import MnemosRuntime

    hits = hits if hits is not None else [
        SearchResult(engram=Engram(id="doc1", content="GateMem G4 status", metadata={"source_uri": "docs/g4.md"}), score=0.95, tier="qdrant"),
        SearchResult(engram=Engram(id="doc2", content="related background", metadata={"source_uri": "docs/bg.md"}), score=0.40, tier="qdrant"),
    ]
    rt = MnemosRuntime()
    rt._config = SimpleNamespace(
        retrieval_mode="semantic", fusion_policy="balanced", explain_default=False,
        lexical_top_k=25, semantic_top_k=25, has_compression=False, quant_bits=4,
        memory_over_maps_phase2=False, memory_over_maps_phase3=False, memory_over_maps_phase4=False,
        embedding_model="BAAI/bge-base-en-v1.5", adaptive_routing=False,
        qdrant_collection="test_collection",
    )
    rt._router = _StubRouter(hits)
    rt._semantic_fusion = None
    rt._lexical_tier = None
    rt._ledger = None
    rt._view_cache = None
    rt._status = "healthy"
    rt._error = None
    return rt


class _RejectAllGovernor:
    def govern(self, *, results, query, governance_mode, top_k, governance_profile=None):
        return [], [], []


class TestRuntimeWiring:
    def test_flag_absent_response_byte_identical_to_no_r0(self, monkeypatch) -> None:
        monkeypatch.delenv(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, raising=False)
        rt = _runtime_with_stub_router()
        baseline = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        )
        with_flag_false = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_shadow=False,
        )
        assert "evidence_admission_shadow" not in baseline["meta"]
        assert "evidence_admission_shadow" not in with_flag_false["meta"]
        baseline_meta = dict(baseline["meta"])
        with_flag_false_meta = dict(with_flag_false["meta"])
        baseline_meta.pop("latency_s", None)
        with_flag_false_meta.pop("latency_s", None)
        assert baseline_meta == with_flag_false_meta
        assert baseline["results"] == with_flag_false["results"]

    def test_flag_present_gate_off_returns_bounded_status_only(self, monkeypatch) -> None:
        monkeypatch.delenv(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, raising=False)
        rt = _runtime_with_stub_router()
        out = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_shadow=True,
        )
        block = out["meta"]["evidence_admission_shadow"]
        assert block["status"] == "unavailable"
        assert block["recommended_route"] is None
        assert block["candidate_budget"] == 0
        assert block["sufficiency"] is None
        assert "ADMISSION_SHADOW_GLOBALLY_DISABLED" in block["reason_codes"]

    def test_flag_present_gate_on_returns_recommendation_and_sufficiency(self, monkeypatch) -> None:
        monkeypatch.setenv(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        out = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_shadow=True,
        )
        block = out["meta"]["evidence_admission_shadow"]
        assert block["status"] in {"recommended", "abstained"}
        assert block["recommended_route"] in ADMISSION_ROUTES
        assert block["sufficiency"] is not None
        assert block["non_authoritative"] is True

    def test_flag_on_never_suppresses_or_reorders_normal_results(self, monkeypatch) -> None:
        monkeypatch.setenv(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        baseline = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        )
        shadowed = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_shadow=True,
        )
        assert [r["engram"]["id"] for r in baseline["results"]] == [r["engram"]["id"] for r in shadowed["results"]]
        assert [r["score"] for r in baseline["results"]] == [r["score"] for r in shadowed["results"]]

    def test_flag_on_does_not_mutate_governance_outcome(self, monkeypatch) -> None:
        monkeypatch.setenv(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        rt._governor = _RejectAllGovernor()
        without_shadow = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            governance="enforced",
        )
        with_shadow = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            governance="enforced", evidence_admission_shadow=True,
        )
        assert without_shadow["meta"]["governance_summary"] == with_shadow["meta"]["governance_summary"]
        assert without_shadow["results"] == with_shadow["results"]

    def test_flag_on_with_no_router_does_not_call_index(self, monkeypatch) -> None:
        """The stub router raises AssertionError if .index() is ever called;
        a passing run is itself the no-durable-write assertion."""
        monkeypatch.setenv(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_shadow=True,
        )

    def test_low_relevance_abstention_flows_into_out_of_scope_sufficiency(self, monkeypatch) -> None:
        monkeypatch.setenv(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router(hits=[
            SearchResult(engram=Engram(id="weak1", content="x", metadata={}), score=0.001, tier="qdrant"),
        ])
        out = rt.search_documents(
            query=_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            evidence_admission_shadow=True,
        )
        assert out["meta"].get("abstention_guard") is not None
        block = out["meta"]["evidence_admission_shadow"]
        assert block["sufficiency"] == "OUT_OF_SCOPE"
        assert any("NORMAL_RETRIEVAL_ABSTAINED" in code for code in block["sufficiency_reason_codes"])
