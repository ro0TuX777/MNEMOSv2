import os
import json
import pytest
from types import SimpleNamespace

from mnemos.extraction.models import (
    FactNode,
    FactExtractionReceipt,
    FactReviewLabel,
    FactExtractionBatchManifest
)
from mnemos.extraction.candidate_store import CandidateStore
from mnemos.extraction.promotion_engine import PromotionEngine
import tools.smc_6_validated_audit_cli as cli

@pytest.fixture
def db_path(tmp_path):
    path = os.path.join(tmp_path, "test_smc6.db")
    store = CandidateStore(path)
    engine = PromotionEngine(store, path)
    
    # Stage a valid candidate
    fact = FactNode(
        fact_id="f_v_1",
        statement="Validated test fact",
        evidence_text="test evidence",
        passage_span=(0, 10),
        passage_node_id="p_1",
        source_engram_id="e_1",
        fact_receipt_id="r_1",
        parent_passage_receipt_id="pr_1",
        source_uri="file://test",
        artifact_id="a_1",
        chunk_id="c_1",
        evidence_hash="hash1",
        passage_text_hash="hash2",
        confidence_score=0.99,
        inherited_governance={"policy_flags": []},
        validation_status="VALID"
    )
    receipt = FactExtractionReceipt(
        receipt_id="r_1",
        batch_id="b_1",
        source_engram_id="e_1",
        passage_node_id="p_1",
        source_uri="file://test",
        artifact_id="a_1",
        chunk_id="c_1",
        passage_span=(0,10),
        evidence_text_hash="hash1",
        parent_passage_text_hash="hash2",
        extractor_version="v1",
        prompt_hash="ph1",
        model_name_version="m1",
        timestamp="time",
        extraction_mode="test",
        inherited_governance_snapshot={},
        output_hash="out1"
    )
    label = FactReviewLabel(
        fact_id="f_v_1",
        review_label="ACCEPT_AS_CANDIDATE",
        review_reason="Looks good",
        reviewer_type="human",
        source_file="test.json",
        passage_node_id="p_1",
        source_engram_id="e_1",
        receipt_id="r_1",
        traceability_verified=True,
        governance_verified=True,
        atomicity_verified=True,
        faithfulness_verified=True,
        recommended_action="PROMOTE_TO_VALIDATED"
    )
    manifest = FactExtractionBatchManifest(
        batch_id="b_1",
        timestamp="time",
        input_passage_count=1,
        generated_facts_count=1,
        unsupported_facts_count=0,
        rejected_facts_count=0,
        failures=[],
        extractor_version="v1"
    )
    store.stage_candidate_bundle(fact, receipt, label, manifest)
    engine.promote_candidate("f_v_1", "op_001")
    return path

def test_validated_status_inferred_from_receipts(db_path):
    engine = cli.get_engine(db_path)
    res = engine.fetch_validated_facts()
    assert len(res) == 1
    assert res[0]["candidate_fact"]["fact_id"] == "f_v_1"
    assert res[0]["conflict_metadata"]["terminal_lifecycle_state"] == "PROMOTION_APPROVED"

def test_candidate_payload_unchanged(db_path):
    store = CandidateStore(db_path)
    facts = store.fetch_candidates(include_candidate_facts=True)
    f = facts[0]["fact_node"]
    assert f["status"] == "CANDIDATE"

def test_downgraded_facts_are_masked(db_path):
    engine = cli.get_engine(db_path)
    # Manually append a DOWNGRADED event
    engine._log_lifecycle_event("f_v_1", "DOWNGRADED", "op_001", "Testing downgrade mask")
    res = engine.fetch_validated_facts()
    assert len(res) == 0

def test_source_governance_masking(db_path):
    engine = cli.get_engine(db_path)
    # Set source to suppressed
    engine.store.set_mock_source_state("e_1", "suppressed")
    res = engine.fetch_validated_facts()
    assert len(res) == 0

def test_full_export_chain(capsys, db_path):
    args = SimpleNamespace(db_path=db_path, json=True, fact_id="f_v_1")
    cli.cmd_export_chain(args)
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["status"] == "success"
    
    with open(res["file"], "r") as f:
        chain = json.load(f)
    
    assert "source_engram_id" in chain
    assert "passage_node" in chain
    assert "candidate_fact" in chain
    assert "human_review_label" in chain
    assert "promotion_receipt" in chain
    assert "lifecycle_events" in chain
    assert "conflict_metadata" in chain

def test_default_retrieval_returns_zero(db_path):
    store = CandidateStore(db_path)
    assert len(store.fetch_candidates(include_candidate_facts=False)) == 0

def test_list_validated_cli(capsys, db_path):
    args = SimpleNamespace(db_path=db_path, json=True)
    cli.cmd_list_validated(args)
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert len(res) == 1
    assert res[0]["fact_id"] == "f_v_1"
