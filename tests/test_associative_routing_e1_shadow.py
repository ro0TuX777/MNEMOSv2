"""Isolation tests for the Associative Routing E1 opt-in shadow lane.

Covers the E1 authorization invariants: the shadow path is fail-closed, never
leaks authority/governance fields, never alters normal retrieval results, and
is fully absent from the response when the request does not opt in.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from mnemos.engram.model import Engram
from mnemos.retrieval.associative_shadow import (
    ASSOCIATIVE_SHADOW_DISABLE_ENV,
    E1_FIXTURES_DIR,
    AssociativeShadowAdapter,
)
from mnemos.retrieval.base import SearchResult
from prototype.associative_routing_e0 import verify_projection

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


# ---------------------------------------------------------------------------
# Adapter-level isolation (no service/router involved)
# ---------------------------------------------------------------------------


class TestAdapterIsolation:
    def test_e1_projection_verifies_clean(self) -> None:
        result = verify_projection(E1_FIXTURES_DIR)
        assert result["status"] == "pass"
        assert all(result["checks"].values())

    def test_kill_switch_returns_unavailable(self, monkeypatch) -> None:
        monkeypatch.setenv(ASSOCIATIVE_SHADOW_DISABLE_ENV, "true")
        adapter = AssociativeShadowAdapter()
        block = adapter.run("Why is GateMem work paused?")
        assert block["status"] == "unavailable"
        assert block["abstention_reason"] == "kill_switch_enabled"
        assert block["candidate_source_ids"] == []
        assert block["non_authoritative"] is True

    def test_adapter_resilience_on_internal_error(self, monkeypatch) -> None:
        adapter = AssociativeShadowAdapter()

        def _boom() -> None:
            raise RuntimeError("simulated projection failure")

        monkeypatch.setattr(adapter, "_ensure_built", _boom)
        block = adapter.run("Why is GateMem work paused?")
        assert block["status"] == "unavailable"
        assert block["abstention_reason"] == "adapter_error"

    def test_resolved_response_has_expected_shape(self) -> None:
        adapter = AssociativeShadowAdapter()
        block = adapter.run("Why is GateMem work paused?")
        assert block["status"] == "resolved"
        assert block["projection_snapshot"].startswith("sha256:")
        assert block["candidate_count"] == len(block["candidate_source_ids"])
        assert block["candidate_source_ids"]
        assert all(isinstance(s, str) and "/" in s for s in block["candidate_source_ids"])
        assert block["latency_ms"] >= 0.0
        assert block["non_authoritative"] is True

    def test_abstains_on_unrelated_query(self) -> None:
        adapter = AssociativeShadowAdapter()
        block = adapter.run("What is the capital of France?")
        assert block["status"] == "abstained"
        assert block["candidate_count"] == 0
        assert block["abstention_reason"] == "NO_SUPPORTED_ASSOCIATIVE_PATH"

    def test_no_authority_fields_in_shadow_payload(self) -> None:
        adapter = AssociativeShadowAdapter()
        block = adapter.run("Why is GateMem work paused?")
        assert _FORBIDDEN_AUTHORITY_KEYS.isdisjoint(block.keys())
        for path in block["routing_paths"]:
            assert _FORBIDDEN_AUTHORITY_KEYS.isdisjoint(path.keys())


# ---------------------------------------------------------------------------
# Request-layer wiring (Flask route parses/forwards the flag; search_documents
# itself is monkeypatched here, mirroring tests/test_service_hybrid_api.py)
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
        captured["associative_routing_shadow"] = associative_routing_shadow
        return {
            "status": "healthy",
            "results": [],
            "meta": {"retrieval_mode": retrieval_mode or "semantic", "fusion_policy": fusion_policy},
        }

    return fake_search_documents


class TestRequestLayerWiring:
    def test_flag_defaults_false_when_absent(self, client, monkeypatch) -> None:
        captured: Dict[str, Any] = {}
        monkeypatch.setattr(
            app_mod._runtime, "search_documents", _fake_search_documents_factory(captured)
        )
        resp = client.post("/v1/mnemos/search", json={"query": "why is this work paused"})
        assert resp.status_code == 200
        assert captured["associative_routing_shadow"] is False
        assert "associative_routing_shadow" not in (resp.get_json().get("meta") or {})

    def test_flag_true_is_forwarded(self, client, monkeypatch) -> None:
        captured: Dict[str, Any] = {}
        monkeypatch.setattr(
            app_mod._runtime, "search_documents", _fake_search_documents_factory(captured)
        )
        resp = client.post(
            "/v1/mnemos/search",
            json={"query": "why is this work paused", "associative_routing_shadow": True},
        )
        assert resp.status_code == 200
        assert captured["associative_routing_shadow"] is True

    def test_non_bool_flag_is_rejected(self, client) -> None:
        resp = client.post(
            "/v1/mnemos/search",
            json={"query": "why is this work paused", "associative_routing_shadow": "yes"},
        )
        assert resp.status_code == 400
        assert "associative_routing_shadow" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# Runtime-layer wiring: exercise the real MnemosRuntime.search_documents body
# (not monkeypatched away) against a stub router, to prove the shadow block
# is additive-only and absent when the flag is off.
# ---------------------------------------------------------------------------


class _StubRouter:
    def __init__(self, hits):
        self._hits = hits

    def search(self, **kwargs):
        meta = {
            "retrieval_mode": kwargs.get("retrieval_mode", "semantic"),
            "fusion_policy": kwargs.get("fusion_policy"),
            "lexical_available": True,
            "telemetry": {
                "lexical_candidates": 1.0,
                "semantic_candidates": 1.0,
                "union_candidates": 1.0,
                "overlap_candidates": 1.0,
            },
        }
        return self._hits, meta

    def stats(self) -> Dict[str, Any]:
        return {}


def _runtime_with_stub_router() -> Any:
    from service.app import MnemosRuntime

    hit = SearchResult(
        engram=Engram(id="doc1", content="GateMem program status"),
        score=0.9,
        tier="hybrid",
        metadata={
            "component_scores": {"lexical": 0.8, "semantic": 0.9, "fused": 0.85},
            "retrieval_sources": ["lexical", "semantic"],
            "fusion_policy": "balanced",
        },
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


class TestRuntimeMetaMerge:
    def test_flag_off_omits_shadow_key_and_leaves_results_untouched(self) -> None:
        rt = _runtime_with_stub_router()
        out = rt.search_documents(
            query="Why is GateMem work paused?",
            top_k=5,
            tiers=None,
            filters=None,
            retrieval_mode="semantic",
            fusion_policy="balanced",
            explain=False,
        )
        assert "associative_routing_shadow" not in out["meta"]
        assert out["results"][0]["engram"]["id"] == "doc1"

    def test_flag_on_attaches_shadow_block_additively(self, monkeypatch) -> None:
        canned_block = {
            "status": "resolved",
            "projection_snapshot": "sha256:deadbeef",
            "matched_cues": ["cue:gatemem"],
            "routing_paths": [],
            "candidate_source_ids": ["docs/benchmarks/gatemem_program_status.md"],
            "candidate_count": 1,
            "abstention_reason": None,
            "latency_ms": 0.5,
            "non_authoritative": True,
        }
        monkeypatch.setattr(app_mod._associative_shadow_adapter, "run", lambda query: canned_block)

        rt = _runtime_with_stub_router()
        baseline = rt.search_documents(
            query="Why is GateMem work paused?",
            top_k=5,
            tiers=None,
            filters=None,
            retrieval_mode="semantic",
            fusion_policy="balanced",
            explain=False,
        )
        shadowed = rt.search_documents(
            query="Why is GateMem work paused?",
            top_k=5,
            tiers=None,
            filters=None,
            retrieval_mode="semantic",
            fusion_policy="balanced",
            explain=False,
            associative_routing_shadow=True,
        )

        assert shadowed["meta"]["associative_routing_shadow"] == canned_block
        assert shadowed["results"] == baseline["results"]
        assert _FORBIDDEN_AUTHORITY_KEYS.isdisjoint(shadowed["meta"]["associative_routing_shadow"].keys())
