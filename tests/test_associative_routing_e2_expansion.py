"""Isolation tests for the Associative Routing E2 opt-in candidate-expansion lane.

Covers the E2 authorization invariants: expansion is double opt-in and
fail-closed, bounded (count + latency), never blends scores or labels
candidates as normal-origin, never suppresses normal results, passes through
the same governance checks normal candidates do, and is fully absent from
the response when the request does not opt in.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from mnemos.engram.model import Engram
from mnemos.retrieval.associative_expansion import (
    CANDIDATE_EXPANSION_ENABLE_ENV,
    E2_FIXTURES_DIR,
    MAX_ADDED_CANDIDATES,
    MAX_TRAVERSAL_DEPTH,
    CandidateExpansionEngine,
)
from mnemos.retrieval.base import SearchResult
from prototype.associative_routing_e0 import AssociativeRouter, build_projection, verify_projection

_FORBIDDEN_AUTHORITY_KEYS = {
    "trust_score",
    "promotion_status",
    "governance_state",
    "authority",
    "disclosure",
    "disclosure_decision",
    "access_decision",
    "retention_decision",
    "deletion_decision",
}

_GATEMEM_PAUSE_QUERY = "Why is GateMem work paused?"


class _ResolverRouter:
    """Stub `retrieval_router` for engine.expand(): resolves any
    `metadata.source_uri` filter to a real SearchResult, unless that
    source_uri is in `missing`, mimicking an inactive/unresolvable target."""

    def __init__(self, missing: Optional[set] = None, sleep_s: float = 0.0) -> None:
        self._missing = missing or set()
        self._sleep_s = sleep_s
        self.index_called = False

    def search(self, *, query, top_k, filters=None, retrieval_mode="semantic", **kwargs):
        if self._sleep_s:
            time.sleep(self._sleep_s)
        filters = filters or {}
        uri = filters.get("metadata.source_uri")
        if not uri or uri in self._missing:
            return [], {}
        engram = Engram(id=f"resolved:{uri}", content="resolved content", metadata={"source_uri": uri})
        return [SearchResult(engram=engram, score=0.42, tier="qdrant")], {}

    def index(self, *args, **kwargs):  # pragma: no cover - must never be called
        self.index_called = True
        raise AssertionError("CandidateExpansionEngine must never call a write/index path")


# ---------------------------------------------------------------------------
# Engine-level isolation (no service/router involved)
# ---------------------------------------------------------------------------


class TestEngineIsolation:
    def test_e2_projection_verifies_clean(self) -> None:
        result = verify_projection(E2_FIXTURES_DIR)
        assert result["status"] == "pass"
        assert all(result["checks"].values())

    def test_kill_switch_blocks_expansion_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv(CANDIDATE_EXPANSION_ENABLE_ENV, raising=False)
        engine = CandidateExpansionEngine()
        injected, block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=_ResolverRouter(),
            filters={}, retrieval_mode="semantic",
        )
        assert injected == []
        assert block["status"] == "disabled"

    def test_flag_on_resolves_bounded_candidates(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        injected, block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=_ResolverRouter(),
            filters={}, retrieval_mode="semantic",
        )
        assert block["status"] == "expanded"
        assert 1 <= len(injected) <= MAX_ADDED_CANDIDATES
        assert block["candidates_added"] == len(injected)
        for r in injected:
            assert r.metadata["candidate_origin"] == "associative_expansion"
            assert r.metadata["non_authoritative"] is True

    def test_abstains_on_unrelated_query(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        injected, block = engine.expand(
            "What is the capital of France?", existing_results=[], retrieval_router=_ResolverRouter(),
            filters={}, retrieval_mode="semantic",
        )
        assert injected == []
        assert block["status"] == "abstained"

    def test_adapter_resilience_on_internal_error(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()

        def _boom() -> None:
            raise RuntimeError("simulated projection failure")

        monkeypatch.setattr(engine, "_ensure_built", _boom)
        injected, block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=_ResolverRouter(),
            filters={}, retrieval_mode="semantic",
        )
        assert injected == []
        assert block["status"] == "unavailable"

    def test_dedup_against_existing_envelope(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        already_present = "docs/benchmarks/gatemem_g5/README.md"
        existing = [
            SearchResult(
                engram=Engram(id="x", content="...", metadata={"source_uri": already_present}),
                score=0.9,
                tier="qdrant",
            )
        ]
        injected, block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=existing, retrieval_router=_ResolverRouter(),
            filters={}, retrieval_mode="semantic",
        )
        assert all(r.engram.metadata.get("source_uri") != already_present for r in injected)
        assert block["candidates_deduplicated"] >= 1

    def test_inactive_or_unresolvable_target_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        router = _ResolverRouter(
            missing={"docs/benchmarks/gatemem_g5/README.md", "docs/benchmarks/gatemem_program_status.md"}
        )
        injected, block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=router,
            filters={}, retrieval_mode="semantic",
        )
        assert injected == []
        assert block["candidates_rejected_by_policy"] >= 1
        assert block["status"] == "resolved_no_new_candidates"

    def test_latency_budget_fallback_stops_early_not_mid_call(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        engine._ensure_built()  # warm so the one-time projection build cost is excluded
        router = _ResolverRouter(sleep_s=0.02)  # 20ms/call, over the 10ms budget
        injected, block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=router,
            filters={}, retrieval_mode="semantic",
        )
        # GateMem-pause resolves 2 routing paths; the first lookup starts
        # under budget and completes, the second is skipped pre-emptively
        # rather than started and aborted mid-call.
        assert len(injected) == 1
        assert "latency_budget_exceeded" in block["reason_codes"]

    def test_no_authority_field_injection(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        injected, block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=_ResolverRouter(),
            filters={}, retrieval_mode="semantic",
        )
        assert _FORBIDDEN_AUTHORITY_KEYS.isdisjoint(block.keys())
        for r in injected:
            assert _FORBIDDEN_AUTHORITY_KEYS.isdisjoint(r.metadata.keys())

    def test_no_durable_write_side_effect(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        router = _ResolverRouter()
        engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=router,
            filters={}, retrieval_mode="semantic",
        )
        assert router.index_called is False

    def test_source_lineage_completeness(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        engine = CandidateExpansionEngine()
        injected, _block = engine.expand(
            _GATEMEM_PAUSE_QUERY, existing_results=[], retrieval_router=_ResolverRouter(),
            filters={}, retrieval_mode="semantic",
        )
        assert injected
        for r in injected:
            path = r.metadata["associative_routing_path"]
            assert path["cue_ids"] and path["tag_ids"] and path["explanation"]
            assert r.metadata["associative_projection_snapshot"].startswith("sha256:")

    def test_traversal_depth_satisfied_by_construction(self) -> None:
        """E0's router is structurally a single cue->tag->content hop; this
        confirms every routing path used by E2 stays within
        MAX_TRAVERSAL_DEPTH=2 (cue-tag, tag-content) without needing runtime
        enforcement."""
        assert MAX_TRAVERSAL_DEPTH == 2
        projection = build_projection(E2_FIXTURES_DIR)
        router = AssociativeRouter(projection=projection)
        response = router.route(_GATEMEM_PAUSE_QUERY)
        for path in response.routing_paths:
            assert len(path.cue_ids) == 1
            assert len(path.tag_ids) == 1


# ---------------------------------------------------------------------------
# Request-layer wiring (Flask route parses/forwards the flag; search_documents
# itself is monkeypatched here, mirroring tests/test_associative_routing_e1_shadow.py)
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
        query,
        top_k,
        tiers,
        filters,
        retrieval_mode,
        fusion_policy,
        explain,
        governance=None,
        explain_governance=None,
        governance_profile=None,
        bounded_envelope=None,
        derive_views=None,
        latency_budget_ms=None,
        complexity_shadow=False,
        cognitive_cycle=None,
        associative_routing_shadow=False,
        associative_candidate_expansion=False,
    ):
        captured["associative_candidate_expansion"] = associative_candidate_expansion
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
        resp = client.post("/v1/mnemos/search", json={"query": _GATEMEM_PAUSE_QUERY})
        assert resp.status_code == 200
        assert captured["associative_candidate_expansion"] is False

    def test_flag_true_is_forwarded(self, client, monkeypatch) -> None:
        captured: Dict[str, Any] = {}
        monkeypatch.setattr(app_mod._runtime, "search_documents", _fake_search_documents_factory(captured))
        resp = client.post(
            "/v1/mnemos/search",
            json={"query": _GATEMEM_PAUSE_QUERY, "associative_candidate_expansion": True},
        )
        assert resp.status_code == 200
        assert captured["associative_candidate_expansion"] is True

    def test_non_bool_flag_is_rejected(self, client) -> None:
        resp = client.post(
            "/v1/mnemos/search",
            json={"query": _GATEMEM_PAUSE_QUERY, "associative_candidate_expansion": "yes"},
        )
        assert resp.status_code == 400
        assert "associative_candidate_expansion" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# Runtime-layer wiring: exercise the real MnemosRuntime.search_documents body
# against a stub router that can resolve source_uri-filtered lookups.
# ---------------------------------------------------------------------------


class _StubRouter:
    """Combines normal multi-result search with the source_uri-filtered
    resolution lookup the expansion engine performs via the same router."""

    def __init__(self, hits: List[SearchResult]) -> None:
        self._hits = hits

    def search(self, *, query, top_k, filters=None, retrieval_mode="semantic", **kwargs):
        filters = filters or {}
        uri = filters.get("metadata.source_uri")
        if uri:
            engram = Engram(id=f"resolved:{uri}", content="resolved content", metadata={"source_uri": uri})
            return [SearchResult(engram=engram, score=0.5, tier="qdrant")], {}
        meta = {"retrieval_mode": retrieval_mode, "fusion_policy": kwargs.get("fusion_policy"), "lexical_available": True}
        return list(self._hits), meta

    def stats(self) -> Dict[str, Any]:
        return {}


def _runtime_with_stub_router() -> Any:
    from service.app import MnemosRuntime

    hit = SearchResult(
        engram=Engram(id="doc1", content="GateMem program status"),
        score=0.9,
        tier="qdrant",
        metadata={"retrieval_sources": ["semantic"]},
    )
    rt = MnemosRuntime()
    rt._config = SimpleNamespace(
        retrieval_mode="semantic",
        fusion_policy="balanced",
        explain_default=False,
        lexical_top_k=25,
        semantic_top_k=25,
        has_compression=False,
        quant_bits=4,
        memory_over_maps_phase2=False,
        memory_over_maps_phase3=False,
        memory_over_maps_phase4=False,
        embedding_model="BAAI/bge-base-en-v1.5",
        adaptive_routing=False,
        qdrant_collection="test_collection",
    )
    rt._router = _StubRouter([hit])
    rt._semantic_fusion = None
    rt._lexical_tier = None
    rt._ledger = None
    rt._status = "healthy"
    rt._error = None
    return rt


class _RejectAllGovernor:
    """Vetoes every candidate it is asked to govern."""

    def govern(self, *, results, query, governance_mode, top_k, governance_profile=None):
        return [], [], []


class TestRuntimeWiring:
    def test_flag_off_response_shape_unchanged(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        out = rt.search_documents(
            query=_GATEMEM_PAUSE_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        )
        assert "associative_candidate_expansion" not in out["meta"]
        assert len(out["results"]) == 1
        assert "candidate_origin" not in out["results"][0]

    def test_kill_switch_disabled_by_default_even_with_flag(self, monkeypatch) -> None:
        monkeypatch.delenv(CANDIDATE_EXPANSION_ENABLE_ENV, raising=False)
        rt = _runtime_with_stub_router()
        out = rt.search_documents(
            query=_GATEMEM_PAUSE_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            associative_candidate_expansion=True,
        )
        assert out["meta"]["associative_candidate_expansion"]["status"] == "disabled"
        assert len(out["results"]) == 1

    def test_flag_on_cannot_suppress_normal_results(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        baseline = rt.search_documents(
            query=_GATEMEM_PAUSE_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        )
        expanded = rt.search_documents(
            query=_GATEMEM_PAUSE_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            associative_candidate_expansion=True,
        )
        assert len(expanded["results"]) > len(baseline["results"])
        # the normal candidate must still be present, first, and unchanged.
        assert expanded["results"][0]["engram"]["id"] == baseline["results"][0]["engram"]["id"]
        assert expanded["results"][0]["score"] == baseline["results"][0]["score"]

    def test_candidate_origin_labeling_present_on_all_entries(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        out = rt.search_documents(
            query=_GATEMEM_PAUSE_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            associative_candidate_expansion=True,
        )
        origins = {entry["candidate_origin"] for entry in out["results"]}
        assert "normal_retrieval_candidate" in origins
        assert "associative_expansion" in origins
        for entry in out["results"]:
            if entry["candidate_origin"] == "associative_expansion":
                assert entry["non_authoritative"] is True
                assert entry["associative_routing_path"]

    def test_governance_rejection_preserved_for_expansion_candidates(self, monkeypatch) -> None:
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")
        rt = _runtime_with_stub_router()
        rt._governor = _RejectAllGovernor()
        out = rt.search_documents(
            query=_GATEMEM_PAUSE_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            governance="enforced",
            associative_candidate_expansion=True,
        )
        block = out["meta"]["associative_candidate_expansion"]
        assert block["candidates_added"] == 0
        assert block["candidates_rejected_by_policy"] >= 1
        assert block["status"] == "resolved_no_new_candidates"
        assert all(entry["candidate_origin"] != "associative_expansion" for entry in out["results"])

    def test_normal_retrieval_abstention_blocks_expansion(self, monkeypatch) -> None:
        """If normal retrieval abstains (empty, low-relevance), expansion
        must not manufacture a non-empty response out of an intentional
        abstention."""
        monkeypatch.setenv(CANDIDATE_EXPANSION_ENABLE_ENV, "true")

        class _LowScoreRouter(_StubRouter):
            def search(self, *, query, top_k, filters=None, retrieval_mode="semantic", **kwargs):
                filters = filters or {}
                if filters.get("metadata.source_uri"):
                    return super().search(query=query, top_k=top_k, filters=filters, retrieval_mode=retrieval_mode)
                low = SearchResult(engram=Engram(id="low", content="..."), score=0.0001, tier="qdrant")
                return [low], {"retrieval_mode": retrieval_mode}

        rt = _runtime_with_stub_router()
        rt._router = _LowScoreRouter([])
        out = rt.search_documents(
            query=_GATEMEM_PAUSE_QUERY, top_k=5, tiers=None, filters=None,
            retrieval_mode="semantic", fusion_policy="balanced", explain=False,
            associative_candidate_expansion=True,
        )
        assert out["results"] == []
        block = out["meta"]["associative_candidate_expansion"]
        assert block["candidates_added"] == 0
        assert "normal_retrieval_abstained" in block["reason_codes"]
