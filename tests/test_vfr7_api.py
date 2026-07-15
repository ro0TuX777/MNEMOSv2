import pytest
from unittest.mock import MagicMock
from mnemos.retrieval.api import EvaluationConsoleAPI
from mnemos.retrieval.auditor import EvaluationAuditor
from mnemos.retrieval.sidecar import FactAwareEvaluationSidecar, ShadowModeDisabledError

@pytest.fixture
def env():
    auditor = EvaluationAuditor()
    sidecar = MagicMock(spec=FactAwareEvaluationSidecar)
    api = EvaluationConsoleAPI(sidecar, auditor)
    return api, sidecar, auditor

def test_gate_4_access_restrictions(env):
    api, _, auditor = env
    
    # Missing roles
    session = {"operator_id": "op_1", "roles": ["ROLE_ANALYST"]}
    resp = api.invoke(session, {})
    assert resp["status"] == 403
    
    # Check audit log
    assert len(auditor.events) == 1
    ev = auditor.events[0]
    assert ev["event_type"] == "SIDECAR_INVOKE_BLOCKED"
    assert ev["payload"]["reason_code"] == "RBAC_DENIED"
    assert ev["payload"]["rbac_passed"] is False

def test_explicit_double_opt_in(env):
    api, _, auditor = env
    session = {"operator_id": "op_1", "roles": ["ROLE_MEMORY_EVALUATOR"]}
    
    # Missing acknowledge
    payload = {"query": "test", "operator_override": True, "enable_fact_awareness": True}
    resp = api.invoke(session, payload)
    
    assert resp["status"] == 400
    ev = auditor.events[0]
    assert ev["event_type"] == "SIDECAR_INVOKE_BLOCKED"
    assert ev["payload"]["reason_code"] == "DOUBLE_OPT_IN_MISSING"
    assert ev["payload"]["double_opt_in_state"] is False

def test_gate_5_audit_log_integrity(env):
    api, sidecar, auditor = env
    session = {"operator_id": "op_1", "roles": ["ROLE_MEMORY_EVALUATOR"]}
    payload = {
        "query": "test", 
        "operator_override": True, 
        "enable_fact_awareness": True,
        "evaluation_mode_acknowledged": True,
        "dataset_id": "dataset_alpha",
        "workload_type": "retrospective_capability_audit",
        "approval_ticket_id": "VFR7-TEST",
        "case_id": "gate-5-audit-integrity",
    }
    
    sidecar.execute_fact_aware_query.return_value = ({"context": []}, {"derived_fact_count": 2})
    
    # Invoke
    resp = api.invoke(session, payload)
    assert resp["status"] == 200
    run_id = resp["sidecar_run_id"]
    
    assert auditor.events[0]["event_type"] == "SIDECAR_INVOKED"
    assert auditor.events[0]["payload"]["sidecar_run_id"] == run_id
    
    # Export
    export_payload = {
        "sidecar_run_id": run_id,
        "data": {},
        "dataset_id": "dataset_alpha",
        "workload_type": "retrospective_capability_audit",
        "approval_ticket_id": "VFR7-TEST",
        "export_purpose": "audit_integrity_test",
    }
    api.export(session, export_payload)
    
    assert auditor.events[1]["event_type"] == "SIDECAR_OUTPUT_RELIED_UPON"
    assert auditor.events[1]["payload"]["sidecar_run_id"] == run_id

def test_gate_3_flag_fact_auditing(env):
    api, _, auditor = env
    session = {"operator_id": "op_1", "roles": ["ROLE_MEMORY_EVALUATOR"]}
    
    payload = {
        "fact_id": "f_1",
        "promotion_receipt_id": "pr_1",
        "sidecar_run_id": "run_123"
    }
    
    api.flag(session, payload)
    
    assert auditor.events[0]["event_type"] == "DERIVED_FACT_FLAGGED_FOR_REVIEW"
    assert auditor.events[0]["payload"]["promotion_receipt_id"] == "pr_1"
    assert auditor.events[0]["payload"]["sidecar_run_id"] == "run_123"
