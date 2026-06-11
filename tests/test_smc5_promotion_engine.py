import os
import json
import pytest

from mnemos.extraction.models import (
    FactNode,
    FactExtractionReceipt,
    FactReviewLabel,
    FactExtractionBatchManifest
)
from mnemos.extraction.candidate_store import CandidateStore
from mnemos.extraction.promotion_engine import PromotionEngine, PromotionError

@pytest.fixture
def store(tmp_path):
    path = os.path.join(tmp_path, "test_promotion.db")
    return CandidateStore(path)

@pytest.fixture
def engine(store):
    return PromotionEngine(store, store.conn.execute("PRAGMA database_list").fetchall()[0][2] if not ":memory:" in store.conn.execute("PRAGMA database_list").fetchall()[0][2] else ":memory:") # just use the store path
    
@pytest.fixture
def valid_bundle(store):
    fact = FactNode(
        fact_id="f_test_1",
        statement="Promotion test fact",
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
        fact_id="f_test_1",
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
    return "f_test_1"

def test_promotion_writes_disjoint_receipt(engine, store, valid_bundle):
    # Promote
    receipt = engine.promote_candidate(valid_bundle, "op_001")
    assert receipt.promotion_status == "VALIDATED"
    
    # Assert written separately
    receipts = engine.fetch_promotion_receipts(valid_bundle)
    assert len(receipts) == 1
    assert receipts[0]["receipt_id"] == receipt.receipt_id

def test_candidate_payload_remains_unchanged(engine, store, valid_bundle):
    # Fetch before
    before = store.fetch_candidates(include_candidate_facts=True)[0]["fact_node"]
    
    # Promote
    engine.promote_candidate(valid_bundle, "op_001")
    
    # Fetch after
    after = store.fetch_candidates(include_candidate_facts=True)[0]["fact_node"]
    
    # Original fact node status should NOT be VALIDATED, and dict should be exactly equal
    assert after["status"] == "CANDIDATE"
    assert before == after

def test_lifecycle_event_append_only(engine, store, valid_bundle):
    engine.promote_candidate(valid_bundle, "op_001")
    cursor = engine.conn.cursor()
    cursor.execute("SELECT * FROM mnemos_fact_lifecycle_events WHERE fact_id=?", (valid_bundle,))
    events = cursor.fetchall()
    assert len(events) == 1
    assert json.loads(events[0]["payload"])["event_type"] == "PROMOTION_APPROVED"

def test_promotion_fails_non_human_reviewer(engine, store):
    fact = FactNode(fact_id="f_bad", statement="S", evidence_text="E", passage_span=(0,1), passage_node_id="p", source_engram_id="e", fact_receipt_id="r", parent_passage_receipt_id="pr", source_uri="u", artifact_id="a", chunk_id="c", evidence_hash="h1", passage_text_hash="h2", confidence_score=0.9, inherited_governance={}, validation_status="V")
    r = FactExtractionReceipt("r", "b", "e", "p", "u", "a", "c", (0,1), "h1", "h2", "v1", "p1", "m1", "t", "m", {}, "o")
    m = FactExtractionBatchManifest("b", "t", 1, 1, 0, 0, [])
    
    # automated reviewer
    l = FactReviewLabel("f_bad", "ACCEPT", "good", "llm-as-judge", "f", "p", "e", "r", True, True, True, True, "PROMOTE_TO_VALIDATED")
    store.stage_candidate_bundle(fact, r, l, m)
    
    with pytest.raises(PromotionError, match="Human reviewer required"):
        engine.promote_candidate("f_bad", "op_001")

def test_promotion_fails_wrong_recommended_action(engine, store):
    fact = FactNode(fact_id="f_bad2", statement="S", evidence_text="E", passage_span=(0,1), passage_node_id="p", source_engram_id="e", fact_receipt_id="r", parent_passage_receipt_id="pr", source_uri="u", artifact_id="a", chunk_id="c", evidence_hash="h1", passage_text_hash="h2", confidence_score=0.9, inherited_governance={}, validation_status="V")
    r = FactExtractionReceipt("r", "b", "e", "p", "u", "a", "c", (0,1), "h1", "h2", "v1", "p1", "m1", "t", "m", {}, "o")
    m = FactExtractionBatchManifest("b", "t", 1, 1, 0, 0, [])
    l = FactReviewLabel("f_bad2", "ACCEPT", "good", "human", "f", "p", "e", "r", True, True, True, True, "KEEP_AS_CANDIDATE")
    store.stage_candidate_bundle(fact, r, l, m)
    
    with pytest.raises(PromotionError, match="does not explicitly authorize"):
        engine.promote_candidate("f_bad2", "op_001")

def test_promotion_fails_ineligible_source(engine, store, valid_bundle):
    store.set_mock_source_state("e_1", "suppressed")
    with pytest.raises(PromotionError, match="Candidate not found or ineligible."):
        engine.promote_candidate(valid_bundle, "op_001")

def test_promotion_fails_on_conflict(engine, store):
    fact = FactNode(fact_id="f_conf", statement="CONFLICT_TEST in string", evidence_text="E", passage_span=(0,1), passage_node_id="p", source_engram_id="e", fact_receipt_id="r", parent_passage_receipt_id="pr", source_uri="u", artifact_id="a", chunk_id="c", evidence_hash="h1", passage_text_hash="h2", confidence_score=0.9, inherited_governance={}, validation_status="V")
    r = FactExtractionReceipt("r", "b", "e", "p", "u", "a", "c", (0,1), "h1", "h2", "v1", "p1", "m1", "t", "m", {}, "o")
    m = FactExtractionBatchManifest("b", "t", 1, 1, 0, 0, [])
    l = FactReviewLabel("f_conf", "ACCEPT", "good", "human", "f", "p", "e", "r", True, True, True, True, "PROMOTE_TO_VALIDATED")
    store.stage_candidate_bundle(fact, r, l, m)
    
    with pytest.raises(PromotionError, match="Logical contradiction"):
        engine.promote_candidate("f_conf", "op_001")

def test_validated_facts_do_not_enter_default_retrieval(engine, store, valid_bundle):
    engine.promote_candidate(valid_bundle, "op_001")
    
    # We test default retrieval which strictly sets include_candidate_facts=False
    # Because there is no logic promoting to Qdrant, we just assert CandidateStore leakage is 0
    res = store.fetch_candidates(include_candidate_facts=False)
    assert len(res) == 0
    assert store.telemetry.default_retrieval_leakage_count == 0
