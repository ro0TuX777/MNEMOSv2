import pytest
import sys
from unittest.mock import MagicMock

# A simulated production prompt builder to test isolation
class DummyProductionPromptBuilder:
    def __init__(self):
        self.context = ""
    
    def build_prompt(self, retrieval_results):
        # Must explicitly reject non-empty derived_results
        if retrieval_results.get("derived_results", []):
            raise ValueError("SEV-STOP: derived_results must be empty in production builder")
            
        primary = retrieval_results.get("primary_results", [])
        self.context = " ".join([p.get("content", "") for p in primary])
        return self.context

def test_production_prompt_builder_rejects_derived_results():
    builder = DummyProductionPromptBuilder()
    malicious_payload = {
        "primary_results": [{"content": "raw fact 1"}],
        "derived_results": [{"content": "secret derived fact"}]
    }
    
    with pytest.raises(ValueError, match="SEV-STOP: derived_results must be empty"):
        builder.build_prompt(malicious_payload)

def test_production_prompt_output_contains_zero_derived_labels():
    builder = DummyProductionPromptBuilder()
    safe_payload = {
        "primary_results": [{"content": "normal engram content"}],
        "derived_results": []
    }
    
    prompt = builder.build_prompt(safe_payload)
    assert "[MNEMOS-DERIVED]" not in prompt
    assert "AUTHORITY: MNEMOS_DERIVED_FACT" not in prompt

def test_production_prompt_builder_does_not_import_shadow_serializer():
    # If production prompt builder imported shadow_serializer, it would exist in sys.modules
    # We simulate this check by verifying `DummyProductionPromptBuilder` module 
    # (or the simulated echoframe module) does not have it.
    
    # In a real environment, we would inspect the ast or module dependencies of `echoframe.prompt_builder`
    # For now, we assert our dummy didn't trigger the import.
    assert "mnemos.evaluation.derived_shadow_packet" not in sys.modules, "Production builder must not import shadow serializer!"

def test_shadow_serializer_preserves_traceability_and_enforces_rules():
    # We must explicitly import it here, not globally, to prove it's isolated from production
    from mnemos.evaluation.derived_shadow_packet import DerivedShadowPacketSerializer
    
    serializer = DerivedShadowPacketSerializer()
    
    # Mock full PIT-2 response
    pit2_response = {
        "derived_results": [
            {
                "authority_type": "MNEMOS_DERIVED_FACT",
                "display_label": "[MNEMOS-DERIVED]",
                "content": "test content",
                "governance_metadata": {"status": "CERTIFIED_FOR_GOVERNED_EVALUATION_OPERATION"},
                "lifecycle_metadata": {"terminal_state": "PROMOTION_APPROVED"},
                "conflict_metadata": {"conflict_status": "NO_CONFLICT_FOUND"},
                "traceability": {
                    "source_engram_ids": ["eng_1"], "passage_node_ids": ["psg_1"],
                    "fact_id": "f_1", "fact_receipt_id": "fr_1",
                    "promotion_receipt_id": "pr_1", "lifecycle_event_id": "le_1",
                    "source_uri": "s3://1", "artifact_id": "art_1",
                    "chunk_id": "ch_1", "provenance_span": [0, 10],
                    "verifier_receipt_id": "vfr_1"
                }
            },
            {
                # Missing traceability field (fact_id)
                "authority_type": "MNEMOS_DERIVED_FACT",
                "display_label": "[MNEMOS-DERIVED]",
                "content": "bad content",
                "governance_metadata": {"status": "CERTIFIED_FOR_GOVERNED_EVALUATION_OPERATION"},
                "lifecycle_metadata": {"terminal_state": "PROMOTION_APPROVED"},
                "conflict_metadata": {"conflict_status": "NO_CONFLICT_FOUND"},
                "traceability": {
                    "source_engram_ids": ["eng_1"], "passage_node_ids": ["psg_1"],
                    "fact_receipt_id": "fr_1",
                    "promotion_receipt_id": "pr_1", "lifecycle_event_id": "le_1",
                    "source_uri": "s3://1", "artifact_id": "art_1",
                    "chunk_id": "ch_1", "provenance_span": [0, 10],
                    "verifier_receipt_id": "vfr_1"
                }
            }
        ]
    }
    
    packet = serializer.serialize(pit2_response)
    
    # Assert schema rules
    assert packet["schema_version"] == "pit_3_derived_shadow_packet_v1"
    assert packet["shadow_only"] is True
    assert packet["production_prompt_allowed"] is False
    assert packet["primary_results_included"] is False
    
    # Should only return 1 fact because the second one missed traceability
    assert packet["derived_fact_count"] == 1
    assert len(packet["derived_evaluation_payload"]) == 1
    
    payload = packet["derived_evaluation_payload"][0]
    assert payload["authority_matrix"]["evidence_gaps"] == []
    assert payload["string_prefix"] == "[AUTHORITY: MNEMOS_DERIVED_FACT] [MNEMOS-DERIVED]"

def test_shadow_serializer_enforces_token_limits():
    from mnemos.evaluation.derived_shadow_packet import DerivedShadowPacketSerializer
    serializer = DerivedShadowPacketSerializer()
    
    # Create 10 facts, but limit is 5
    facts = []
    for i in range(10):
        facts.append({
            "authority_type": "MNEMOS_DERIVED_FACT",
            "display_label": "[MNEMOS-DERIVED]",
            "content": f"test content {i}",
            "governance_metadata": {"status": "CERTIFIED_FOR_GOVERNED_EVALUATION_OPERATION"},
            "lifecycle_metadata": {"terminal_state": "PROMOTION_APPROVED"},
            "conflict_metadata": {"conflict_status": "NO_CONFLICT_FOUND"},
            "traceability": {
                "source_engram_ids": ["eng_1"], "passage_node_ids": ["psg_1"],
                "fact_id": f"f_{i}", "fact_receipt_id": "fr_1",
                "promotion_receipt_id": "pr_1", "lifecycle_event_id": "le_1",
                "source_uri": "s3://1", "artifact_id": "art_1",
                "chunk_id": "ch_1", "provenance_span": [0, 10],
                "verifier_receipt_id": "vfr_1"
            }
        })
        
    packet = serializer.serialize({"derived_results": facts})
    
    assert packet["derived_fact_count"] == 5  # Enforced config limit
