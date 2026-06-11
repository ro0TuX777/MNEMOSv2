import pytest
from mnemos.extraction.models import (
    FactNode,
    FactExtractionReceipt,
    FactReviewLabel,
    FactExtractionBatchManifest
)
from mnemos.extraction.candidate_store import CandidateStore, CandidatePersistenceError

@pytest.fixture
def store():
    return CandidateStore(":memory:")

@pytest.fixture
def valid_bundle():
    fact = FactNode(
        fact_id="f_1",
        statement="A test fact",
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
        fact_id="f_1",
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
        recommended_action="KEEP_AS_CANDIDATE"
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
    return fact, receipt, label, manifest

def test_missing_receipt_blocks_staging(store, valid_bundle):
    f, r, l, m = valid_bundle
    with pytest.raises(CandidatePersistenceError, match="Atomic bundle incomplete"):
        store.stage_candidate_bundle(f, None, l, m)
    assert store.telemetry.candidate_fact_persistence_failures == 1

def test_missing_review_label_blocks_staging(store, valid_bundle):
    f, r, l, m = valid_bundle
    with pytest.raises(CandidatePersistenceError):
        store.stage_candidate_bundle(f, r, None, m)

def test_missing_manifest_blocks_staging(store, valid_bundle):
    f, r, l, m = valid_bundle
    with pytest.raises(CandidatePersistenceError):
        store.stage_candidate_bundle(f, r, l, None)

def test_validated_status_blocks_staging(store, valid_bundle):
    f, r, l, m = valid_bundle
    f.status = "VALIDATED"
    with pytest.raises(CandidatePersistenceError, match="Cannot persist VALIDATED status"):
        store.stage_candidate_bundle(f, r, l, m)

def test_missing_mandatory_field_blocks_staging(store, valid_bundle):
    f, r, l, m = valid_bundle
    f.fact_id = ""
    with pytest.raises(CandidatePersistenceError, match="Missing mandatory FactNode fields"):
        store.stage_candidate_bundle(f, r, l, m)

def test_rollback_removes_matching_candidates(store, valid_bundle):
    f, r, l, m = valid_bundle
    store.stage_candidate_bundle(f, r, l, m)
    assert store.telemetry.candidate_facts_staged_count == 1
    
    deleted = store.rollback("batch_id", "b_1")
    assert deleted == 1
    assert store.telemetry.rollback_count == 1
    assert len(store.fetch_candidates(include_candidate_facts=True)) == 0

def test_suppressed_source_masks_candidate(store, valid_bundle):
    f, r, l, m = valid_bundle
    store.stage_candidate_bundle(f, r, l, m)
    
    # Active state
    assert len(store.fetch_candidates(include_candidate_facts=True)) == 1
    
    # Suppressed state
    store.set_mock_source_state("e_1", "suppressed")
    assert len(store.fetch_candidates(include_candidate_facts=True)) == 0
    assert store.telemetry.masked_due_to_source_governance_count == 1

def test_default_retrieval_leakage_is_zero(store, valid_bundle):
    f, r, l, m = valid_bundle
    store.stage_candidate_bundle(f, r, l, m)
    
    # default retrieval simulation
    results = store.fetch_candidates(include_candidate_facts=False)
    assert len(results) == 0
    assert store.telemetry.default_retrieval_leakage_count == 0

def test_no_source_mutation(store, valid_bundle):
    f, r, l, m = valid_bundle
    import copy
    f_copy = copy.deepcopy(f)
    store.stage_candidate_bundle(f, r, l, m)
    
    assert f == f_copy
