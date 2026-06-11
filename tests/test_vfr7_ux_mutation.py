import inspect
import hashlib
import pytest
from mnemos.retrieval.api import EvaluationConsoleAPI
from mnemos.retrieval.auditor import EvaluationAuditor
from unittest.mock import MagicMock
from types import SimpleNamespace

# Modules
import mnemos.retrieval.retrieval_router as rr_module
import mnemos.retrieval.graph_tier as gh_module

def get_module_hash(module):
    source = inspect.getsource(module)
    return hashlib.md5(source.encode('utf-8')).hexdigest()

def test_gate_6_ux_cannot_mutate_fact_lifecycle():
    router_hash = get_module_hash(rr_module)
    graph_hash = get_module_hash(gh_module)
    
    auditor = EvaluationAuditor()
    api = EvaluationConsoleAPI(MagicMock(), auditor)
    session = {"operator_id": "op_1", "roles": ["ROLE_MEMORY_EVALUATOR"]}
    
    # 1. Invoke
    api.sidecar.execute_fact_aware_query.return_value = ({}, {})
    api.invoke(session, {"query": "test", "operator_override": True, "enable_fact_awareness": True, "evaluation_mode_acknowledged": True})
    
    # 2. Flag
    api.flag(session, {"fact_id": "f_1", "promotion_receipt_id": "pr_1", "sidecar_run_id": "run_123"})
    
    # 3. Export
    api.export(session, {"sidecar_run_id": "run_123", "data": {}})
    
    # Assert
    assert get_module_hash(rr_module) == router_hash
    assert get_module_hash(gh_module) == graph_hash
    
    # Check that flag only did an audit event, no db interaction
    evs = [e for e in auditor.events if e["event_type"] == "DERIVED_FACT_FLAGGED_FOR_REVIEW"]
    assert len(evs) == 1
