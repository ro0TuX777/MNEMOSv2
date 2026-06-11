import pytest
import json
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    from service.app import app, _runtime
    with patch('mnemos.retrieval.qdrant_tier.QdrantTier._initialize'):
        _runtime.initialize()
    app.config['TESTING'] = True
    # Reset stats
    for k in _runtime._mom_stats.keys():
        if isinstance(_runtime._mom_stats[k], int):
            _runtime._mom_stats[k] = 0
    with app.test_client() as client:
        with patch.object(_runtime, 'search_documents', return_value={}):
            yield client

def test_production_route_ignores_and_returns_zero_derived(client):
    """/api/v1/query returns 0 derived facts under all configs"""
    resp = client.post("/api/v1/query", json={"query": "test"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "derived_results" not in data or len(data["derived_results"]) == 0

def test_production_route_rejects_eval_mode(client):
    """/api/v1/query rejects evaluation_mode=true"""
    resp = client.post("/api/v1/query", json={"query": "test", "evaluation_mode": True})
    assert resp.status_code == 400
    assert "not supported on production routes" in json.loads(resp.data)["error"]

@patch('service.app.get_config')
def test_eval_route_missing_flags(mock_get_config, client):
    """/api/v1/evaluate_derived_shadow requires both flags"""
    from service.app import _runtime
    mock_conf = MagicMock()
    mock_conf.token = ""
    mock_conf.derived_enabled = True
    mock_conf.derived_whitelist = ["eval_dashboard"]
    mock_get_config.return_value = mock_conf
    headers = {"X-Client-Id": "eval_dashboard"}
    # Missing include_derived_facts
    resp = client.post("/api/v1/evaluate_derived_shadow", headers=headers, json={
        "evaluation_mode": True
    })
    assert resp.status_code == 400
    assert "missing_required_eval_flags" in json.loads(resp.data)["error"]

@patch('service.app.get_config')
def test_eval_route_not_whitelisted(mock_get_config, client):
    """non-whitelisted clients receive 403"""
    mock_conf = MagicMock()
    mock_conf.token = ""
    mock_conf.derived_enabled = True
    mock_conf.derived_whitelist = ["eval_dashboard"]
    mock_get_config.return_value = mock_conf

    headers = {"X-Client-Id": "hacker"}
    resp = client.post("/api/v1/evaluate_derived_shadow", headers=headers, json={
        "evaluation_mode": True,
        "include_derived_facts": True
    })
    assert resp.status_code == 403
    assert "client_not_authorized" in json.loads(resp.data)["error"]

@patch('service.app.get_config')
def test_eval_route_kill_switch(mock_get_config, client):
    """kill switch returns 503"""
    mock_conf = MagicMock()
    mock_conf.token = ""
    mock_conf.derived_enabled = False
    mock_get_config.return_value = mock_conf

    headers = {"X-Client-Id": "eval_dashboard"}
    resp = client.post("/api/v1/evaluate_derived_shadow", headers=headers, json={})
    assert resp.status_code == 503
    assert "derived_lane_disabled" in json.loads(resp.data)["error"]

def test_production_prompt_builder_guard():
    """production prompt builder SevStop guard triggers on derived facts"""
    from mnemos.echoframe.prompt_builder import PromptBuilder, SevStop
    pb = PromptBuilder()
    
    # Test safe
    pb.build_prompt({"primary_results": [{"content": "hello"}]}, evaluation_mode=False)

    # Test SEV-STOP
    with pytest.raises(SevStop):
        pb.build_prompt({"derived_results": [{"content": "leak"}]}, evaluation_mode=False)

def test_evaluation_renderer():
    """evaluation renderer outputs [MNEMOS-DERIVED] block"""
    from mnemos.evaluation.derived_evaluation_renderer import render_derived_evaluation_context
    shadow_packet = {
        "shadow_only": True,
        "derived_evaluation_payload": [
            {
                "content": "A derived fact.",
                "authority_matrix": {"evidence_gaps": ["Missing logs"]}
            }
        ]
    }
    block = render_derived_evaluation_context(shadow_packet)
    assert "=== [MNEMOS-DERIVED EVALUATION CONTEXT] ===" in block
    assert "[AUTHORITY: MNEMOS_DERIVED_FACT] [MNEMOS-DERIVED]" in block
    assert "A derived fact." in block
    assert "Missing logs" in block

def test_evaluation_renderer_fails_on_non_shadow():
    from mnemos.evaluation.derived_evaluation_renderer import render_derived_evaluation_context
    with pytest.raises(ValueError, match="SEV-STOP"):
        render_derived_evaluation_context({"shadow_only": False})

@patch('mnemos.evaluation.derived_shadow_packet.DerivedShadowPacketSerializer.serialize')
@patch('service.app.get_config')
def test_evaluate_derived_shadow_success(mock_get_config, mock_serialize, client):
    from service.app import _runtime
    mock_conf = MagicMock()
    mock_conf.token = ""
    mock_conf.derived_enabled = True
    mock_conf.derived_whitelist = ["eval_dashboard"]
    mock_get_config.return_value = mock_conf

    mock_serialize.return_value = {
        "shadow_only": True,
        "derived_fact_count": 1,
        "derived_evaluation_payload": [{"content": "test shadow"}]
    }

    headers = {"X-Client-Id": "eval_dashboard"}
    resp = client.post("/api/v1/evaluate_derived_shadow", headers=headers, json={
        "evaluation_mode": True,
        "include_derived_facts": True,
        "query": "hello"
    })
    
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "shadow_evaluation" in data
    assert "=== [MNEMOS-DERIVED EVALUATION CONTEXT] ===" in data["shadow_evaluation"]["rendered_block"]
    
    assert _runtime._mom_stats["evaluate_derived_shadow.request_count"] > 0
    assert _runtime._mom_stats["evaluate_derived_shadow.rendered_derived_fact_count"] == 1
    assert _runtime._mom_stats["query.default_retrieval.derived_fact_count"] == 0
