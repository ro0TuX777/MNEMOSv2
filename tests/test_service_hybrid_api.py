"""Integration-style API tests for hybrid /search behavior."""

from typing import Any, Dict

import pytest

pytest.importorskip("flask")

import service.app as app_mod


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_mod, "_ensure_runtime", lambda: None)
    app_mod.app.config["TESTING"] = True
    with app_mod.app.test_client() as c:
        yield c


def test_search_hybrid_valid_request_forwards_params(client, monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_search_documents(query, top_k, tiers, filters, retrieval_mode, fusion_policy, explain,
                              _governance=None, _explain_governance=None):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["retrieval_mode"] = retrieval_mode
        captured["fusion_policy"] = fusion_policy
        captured["explain"] = explain
        return {
            "status": "healthy",
            "results": [
                {
                    "engram": {"id": "d1", "content": "SOC2 control mapping"},
                    "score": 0.91,
                    "tier": "hybrid",
                    "tiers": ["lexical", "qdrant"],
                    "component_scores": {
                        "lexical": 0.9,
                        "semantic": 0.8,
                        "fused": 0.85,
                    },
                    "retrieval_sources": ["lexical", "semantic"],
                    "fusion_policy": "balanced",
                }
            ],
            "meta": {
                "retrieval_mode": "hybrid",
                "fusion_policy": "balanced",
                "explain": True,
            },
        }

    monkeypatch.setattr(app_mod._runtime, "search_documents", fake_search_documents)

    resp = client.post(
        "/v1/mnemos/search",
        json={
            "query": "SOC2 control objective",
            "top_k": 5,
            "retrieval_mode": "hybrid",
            "fusion_policy": "balanced",
            "explain": True,
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["results"][0]["component_scores"]["fused"] == 0.85
    assert set(data["results"][0]["retrieval_sources"]) == {"lexical", "semantic"}
    assert data["results"][0]["fusion_policy"] == "balanced"
    assert captured["retrieval_mode"] == "hybrid"
    assert captured["fusion_policy"] == "balanced"
    assert captured["explain"] is True


def test_search_hybrid_filter_payload_forwarding(client, monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_search_documents(query, top_k, tiers, filters, retrieval_mode, fusion_policy, explain,
                              _governance=None, _explain_governance=None):
        captured["filters"] = filters
        return {
            "status": "healthy",
            "results": [
                {
                    "engram": {"id": "d2", "content": "legal control"},
                    "score": 0.72,
                    "tier": "hybrid",
                    "tiers": ["lexical", "qdrant"],
                    "component_scores": {"lexical": 0.6, "semantic": 0.8, "fused": 0.7},
                    "retrieval_sources": ["semantic"],
                    "filters_applied": {"source": "dept-legal"},
                    "fusion_policy": "semantic_dominant",
                }
            ],
            "meta": {"retrieval_mode": "hybrid", "fusion_policy": "semantic_dominant", "explain": True},
        }

    monkeypatch.setattr(app_mod._runtime, "search_documents", fake_search_documents)

    resp = client.post(
        "/v1/mnemos/search",
        json={
            "query": "control mapping",
            "retrieval_mode": "hybrid",
            "fusion_policy": "semantic_dominant",
            "filters": {"source": "dept-legal"},
            "explain": True,
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert captured["filters"] == {"source": "dept-legal"}
    assert body["results"][0]["filters_applied"] == {"source": "dept-legal"}
    assert set(body["results"][0]["retrieval_sources"]).issubset({"lexical", "semantic"})


def test_search_invalid_retrieval_mode_rejected(client):
    resp = client.post(
        "/v1/mnemos/search",
        json={
            "query": "test",
            "retrieval_mode": "hybrid_plus",
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "Invalid retrieval_mode"


def test_search_invalid_fusion_policy_rejected(client):
    resp = client.post(
        "/v1/mnemos/search",
        json={
            "query": "test",
            "retrieval_mode": "hybrid",
            "fusion_policy": "aggressive",
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "Invalid fusion_policy"


def test_search_invalid_explain_type_rejected(client):
    resp = client.post(
        "/v1/mnemos/search",
        json={
            "query": "test",
            "retrieval_mode": "hybrid",
            "fusion_policy": "balanced",
            "explain": "yes",
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "explain must be a boolean"
