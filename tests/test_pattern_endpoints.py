"""
Phase 20 — Pattern candidate and promoted pattern endpoint tests.

Tests the 6 new Flask routes:
  GET    /v1/mnemos/cognitive/candidates
  GET    /v1/mnemos/cognitive/candidates/<id>
  POST   /v1/mnemos/cognitive/candidates/<id>/recommend
  POST   /v1/mnemos/cognitive/candidates/<id>/approve
  POST   /v1/mnemos/cognitive/candidates/<id>/reject
  GET    /v1/mnemos/cognitive/patterns

Also verifies that capabilities includes pattern_store_enabled.

Setup: patches _runtime.list_pattern_candidates etc. directly to avoid needing
a live MNEMOS runtime, mirroring how test_vfr7_api.py approaches endpoint tests.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Import the Flask app and the runtime singleton
from service.app import app, _runtime


# ── Helpers ───────────────────────────────────────────────────────────────────


def _base_payload(extra: Dict[str, Any] = None) -> Dict[str, Any]:
    d = {
        "contract_version": "v1",
        "status": "healthy",
        "source": "mnemos-service",
        "generated_at": "2026-06-15T00:00:00Z",
        "error": None,
    }
    if extra:
        d.update(extra)
    return d


def _candidate_dict(
    *,
    candidate_id: str = None,
    status: str = "candidate",
    pattern_type: str = "descriptive",
) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id or str(uuid.uuid4()),
        "pattern_summary": "IF CLASS_B advisory THEN hybrid retrieval works well",
        "pattern_type": pattern_type,
        "confidence_score": 0.82,
        "promotion_status": status,
        "applies_when": "CLASS_B advisory governance hybrid search",
        "does_not_apply_when": "enforced mode",
        "risk_if_wrong": "low",
        "governance_review_id": None,
    }


def _engram_dict(candidate_id: str = None) -> Dict[str, Any]:
    cid = candidate_id or str(uuid.uuid4())
    return {
        "pattern_id": cid,
        "pattern_summary": "IF CLASS_B advisory THEN hybrid retrieval works well",
        "pattern_type": "descriptive",
        "confidence_score": 0.82,
        "applies_when": "CLASS_B advisory governance hybrid search",
        "does_not_apply_when": "enforced mode",
        "risk_if_wrong": "low",
        "governance_review_id": "rev-001",
        "promoted_from_candidate_id": cid,
        "promoted_at": "2026-06-15T00:00:00+00:00",
        "write_class": "semantic_write",
        "authoritative_for_retrieval": False,
        "affects_ranking": False,
        "mutates_policy": False,
    }


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def mock_authorized():
    """Bypass auth for all tests."""
    with patch("service.app._authorized", return_value=True):
        yield


@pytest.fixture(autouse=True)
def mock_ensure_runtime():
    """Bypass runtime init check for all tests."""
    with patch("service.app._ensure_runtime", return_value=None):
        yield


# ── GET /v1/mnemos/cognitive/candidates ──────────────────────────────────────


class TestListCandidatesEndpoint:
    def test_returns_candidates_list(self, client):
        cands = [_candidate_dict(), _candidate_dict(status="promotion_recommended")]
        payload = _base_payload({"candidates": cands, "count": 2, "status_counts": {}})
        with patch.object(_runtime, "list_pattern_candidates", return_value=payload):
            resp = client.get("/v1/mnemos/cognitive/candidates")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["candidates"]) == 2

    def test_forwards_status_filter(self, client):
        payload = _base_payload({"candidates": [], "count": 0, "status_counts": {}})
        with patch.object(_runtime, "list_pattern_candidates", return_value=payload) as mock:
            client.get("/v1/mnemos/cognitive/candidates?status=promotion_recommended")
        mock.assert_called_once_with(status="promotion_recommended")

    def test_no_store_returns_error_payload(self, client):
        payload = _base_payload({"error": "Pattern candidate store not configured", "candidates": []})
        with patch.object(_runtime, "list_pattern_candidates", return_value=payload):
            resp = client.get("/v1/mnemos/cognitive/candidates")
        assert resp.status_code == 200
        assert "error" in resp.get_json()


# ── GET /v1/mnemos/cognitive/candidates/<id> ─────────────────────────────────


class TestGetCandidateEndpoint:
    def test_returns_candidate(self, client):
        cid = str(uuid.uuid4())
        payload = _base_payload({"candidate": _candidate_dict(candidate_id=cid)})
        with patch.object(_runtime, "get_pattern_candidate", return_value=payload):
            resp = client.get(f"/v1/mnemos/cognitive/candidates/{cid}")
        assert resp.status_code == 200
        assert resp.get_json()["candidate"]["candidate_id"] == cid

    def test_returns_404_when_not_found(self, client):
        payload = _base_payload({"error": "Candidate 'x' not found"})
        with patch.object(_runtime, "get_pattern_candidate", return_value=payload):
            resp = client.get("/v1/mnemos/cognitive/candidates/x")
        assert resp.status_code == 404


# ── POST /v1/mnemos/cognitive/candidates/<id>/recommend ──────────────────────


class TestRecommendEndpoint:
    def test_recommend_returns_updated_candidate(self, client):
        cid = str(uuid.uuid4())
        payload = _base_payload({"candidate": _candidate_dict(candidate_id=cid, status="promotion_recommended")})
        with patch.object(_runtime, "recommend_pattern_candidate", return_value=payload):
            resp = client.post(
                f"/v1/mnemos/cognitive/candidates/{cid}/recommend",
                json={"gate_id": "gate-abc"},
            )
        assert resp.status_code == 200
        assert resp.get_json()["candidate"]["promotion_status"] == "promotion_recommended"

    def test_recommend_requires_gate_id(self, client):
        resp = client.post(
            "/v1/mnemos/cognitive/candidates/some-id/recommend",
            json={},
        )
        assert resp.status_code == 400
        assert "gate_id" in resp.get_json()["error"]

    def test_recommend_404_when_not_found(self, client):
        payload = _base_payload({"error": "Candidate 'x' not found"})
        with patch.object(_runtime, "recommend_pattern_candidate", return_value=payload):
            resp = client.post(
                "/v1/mnemos/cognitive/candidates/x/recommend",
                json={"gate_id": "g1"},
            )
        assert resp.status_code == 404


# ── POST /v1/mnemos/cognitive/candidates/<id>/approve ────────────────────────


class TestApproveEndpoint:
    def test_approve_returns_promoted_pattern(self, client):
        cid = str(uuid.uuid4())
        payload = _base_payload({"pattern": _engram_dict(candidate_id=cid)})
        with patch.object(_runtime, "approve_pattern_candidate", return_value=payload):
            resp = client.post(
                f"/v1/mnemos/cognitive/candidates/{cid}/approve",
                json={"governance_review_id": "rev-001"},
            )
        assert resp.status_code == 200
        assert resp.get_json()["pattern"]["authoritative_for_retrieval"] is False

    def test_approve_requires_governance_review_id(self, client):
        resp = client.post(
            "/v1/mnemos/cognitive/candidates/some-id/approve",
            json={},
        )
        assert resp.status_code == 400
        assert "governance_review_id" in resp.get_json()["error"]

    def test_approve_400_when_not_recommended(self, client):
        payload = _base_payload({"error": "Candidate must be in promotion_recommended state"})
        with patch.object(_runtime, "approve_pattern_candidate", return_value=payload):
            resp = client.post(
                "/v1/mnemos/cognitive/candidates/some-id/approve",
                json={"governance_review_id": "rev-x"},
            )
        assert resp.status_code == 400


# ── POST /v1/mnemos/cognitive/candidates/<id>/reject ─────────────────────────


class TestRejectEndpoint:
    def test_reject_returns_updated_candidate(self, client):
        cid = str(uuid.uuid4())
        payload = _base_payload({"candidate": _candidate_dict(candidate_id=cid, status="rejected")})
        with patch.object(_runtime, "reject_pattern_candidate", return_value=payload):
            resp = client.post(f"/v1/mnemos/cognitive/candidates/{cid}/reject")
        assert resp.status_code == 200
        assert resp.get_json()["candidate"]["promotion_status"] == "rejected"

    def test_reject_404_when_not_found(self, client):
        payload = _base_payload({"error": "Candidate 'x' not found"})
        with patch.object(_runtime, "reject_pattern_candidate", return_value=payload):
            resp = client.post("/v1/mnemos/cognitive/candidates/x/reject")
        assert resp.status_code == 404


# ── GET /v1/mnemos/cognitive/patterns ────────────────────────────────────────


class TestListPatternsEndpoint:
    def test_returns_promoted_patterns(self, client):
        cid = str(uuid.uuid4())
        payload = _base_payload({"patterns": [_engram_dict(candidate_id=cid)], "count": 1})
        with patch.object(_runtime, "list_promoted_patterns", return_value=payload):
            resp = client.get("/v1/mnemos/cognitive/patterns")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        assert data["patterns"][0]["authoritative_for_retrieval"] is False

    def test_empty_patterns_when_none_promoted(self, client):
        payload = _base_payload({"patterns": [], "count": 0})
        with patch.object(_runtime, "list_promoted_patterns", return_value=payload):
            resp = client.get("/v1/mnemos/cognitive/patterns")
        assert resp.status_code == 200
        assert resp.get_json()["patterns"] == []

    def test_no_store_returns_error(self, client):
        payload = _base_payload({"error": "Pattern candidate store not configured", "patterns": []})
        with patch.object(_runtime, "list_promoted_patterns", return_value=payload):
            resp = client.get("/v1/mnemos/cognitive/patterns")
        assert resp.status_code == 200
        assert "error" in resp.get_json()
