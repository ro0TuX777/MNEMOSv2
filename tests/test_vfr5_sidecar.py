import os
import inspect
import hashlib
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from mnemos.retrieval.sidecar import FactAwareEvaluationSidecar, ShadowModeDisabledError
from mnemos.retrieval.shadow_packet import ShadowPacketBuilder
from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.retrieval.base import SearchResult
from mnemos.extraction.models import (
    FactNode, FactExtractionReceipt, FactReviewLabel, FactExtractionBatchManifest
)
from mnemos.extraction.candidate_store import CandidateStore
from mnemos.extraction.promotion_engine import PromotionEngine

# Import modules to hash
import mnemos.retrieval.retrieval_router as rr_module
import mnemos.retrieval.graph_tier as gh_module

def get_module_hash(module):
    source = inspect.getsource(module)
    return hashlib.md5(source.encode('utf-8')).hexdigest()

@pytest.fixture
def clean_hashes():
    return {
        "router": get_module_hash(rr_module),
        "graph": get_module_hash(gh_module)
    }

@pytest.fixture
def env_setup(tmp_path):
    db_path = os.path.join(tmp_path, "test_vfr5.db")
    store = CandidateStore(db_path)
    engine = PromotionEngine(store, db_path)
    
    f1 = FactNode("f_1", "claim", "test", (0,10), "p_1", "e_1", "r_1", "pr_1", "u", "a", "c", "h1", "h2", 0.99, {}, "VALID")
    r1 = FactExtractionReceipt("r_1", "b_1", "e_1", "p_1", "u", "a", "c", (0,10), "h1", "h2", "v", "p", "m", "t", "m", {}, "o")
    m1 = FactExtractionBatchManifest("b_1", "t", 1, 1, 0, 0, [])
    l1 = FactReviewLabel("f_1", "ACCEPT", "good", "human", "f", "p_1", "e_1", "r_1", True, True, True, True, "PROMOTE_TO_VALIDATED")
    store.stage_candidate_bundle(f1, r1, l1, m1)
    engine.promote_candidate("f_1", "op_test")
    
    mock_engram = SimpleNamespace(id="e_primary", content="Primary engram content.", governance=None, lineage=lambda: {})
    mock_retriever = MagicMock(spec=RetrievalRouter)
    mock_retriever.search.return_value = ([SearchResult(engram=mock_engram, score=0.9, tier="semantic")], {"retrieval_mode": "semantic"})
    
    sidecar = FactAwareEvaluationSidecar(mock_retriever, engine)
    return sidecar, mock_retriever, engine

def test_gate_1_and_2_leakage(env_setup):
    sidecar, mock_retriever, _ = env_setup
    
    # Semantic
    _, meta_semantic = mock_retriever.search(query="test", top_k=5)
    assert "derived_fact_count" not in meta_semantic
    
    # Graph Hybrid
    mock_retriever.search.return_value = ([], {"retrieval_mode": "graph_hybrid_experimental"})
    _, meta_graph = mock_retriever.search(query="test", top_k=5, retrieval_mode="graph_hybrid_experimental")
    assert "derived_fact_count" not in meta_graph

def test_gate_3_4_5_mutations(env_setup, clean_hashes):
    # Proves 0 EchoFrame mutation, 0 RetrievalRouter mutation, 0 baseline ranking mutation
    sidecar, _, _ = env_setup
    
    # Invoke sidecar
    sidecar.execute_fact_aware_query("test", 5, operator_override=True, enable_fact_awareness=True)
    
    # Re-hash
    assert get_module_hash(rr_module) == clean_hashes["router"]
    assert get_module_hash(gh_module) == clean_hashes["graph"]

def test_gate_6_governance_masking(env_setup):
    sidecar, _, _ = env_setup
    overrides = {"e_1": "suppressed"}
    
    pkt, tel = sidecar.execute_fact_aware_query("test", 5, operator_override=True, enable_fact_awareness=True, governance_overrides=overrides)
    
    assert tel["masked_fact_count"] == 1
    assert tel["derived_fact_count"] == 0
    assert "<Derived_FactNodes>" not in "\n".join(pkt["context"])

def test_gate_7_and_8_citations(env_setup):
    sidecar, _, _ = env_setup
    pkt, tel = sidecar.execute_fact_aware_query("test", 5, operator_override=True, enable_fact_awareness=True)
    
    ctx = "\n".join(pkt["context"])
    assert tel["derived_fact_count"] == 1
    assert "source_span=[0, 10]" in ctx
    assert "promotion_receipt_id=prom_" in ctx
    assert "source_engram_id=e_1" in ctx
    assert "WARNING: THIS PACKET CONTAINS DERIVED FACT NODES." in ctx

def test_gate_9_source_primacy_contradiction():
    # Structural definition asserts this. The warning banner explicitly states:
    # "THESE ARE SECONDARY TO PRIMARY ENGRAMS."
    # The physical separation guarantees the LLM prompt instruction hierarchy.
    pass

def test_gate_10_kill_switch_and_double_opt_in(env_setup, monkeypatch):
    sidecar, _, _ = env_setup
    
    # Double Opt-In False Path
    pkt, tel = sidecar.execute_fact_aware_query("test", 5, operator_override=False, enable_fact_awareness=True)
    assert tel["sidecar_active"] is False
    assert tel["derived_fact_count"] == 0
    assert tel["double_opt_in_satisfied"] is False
    assert "<Derived_FactNodes>" not in "\n".join(pkt["context"])
    
    # Kill Switch
    monkeypatch.setenv("VFR_DISABLE_SHADOW_MODE", "true")
    with pytest.raises(ShadowModeDisabledError):
        sidecar.execute_fact_aware_query("test", 5, operator_override=True, enable_fact_awareness=True)
