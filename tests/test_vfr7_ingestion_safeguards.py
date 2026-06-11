import pytest
from mnemos.engram.exceptions import ArtifactPolicyRejectedError
from mnemos.retrieval.auditor import EvaluationAuditor

class DummyIngestionPipeline:
    def __init__(self, auditor: EvaluationAuditor):
        self.auditor = auditor
        
    def ingest(self, artifact: dict):
        if artifact.get("mnemos_artifact_type") == "sidecar_evaluation_export" or \
           artifact.get("production_ingestion_allowed") is False:
            
            self.auditor.log_ingestion_rejection(
                artifact.get("mnemos_artifact_type", "unknown"),
                "2026-06-06T12:00:00Z",
                "Sidecar exports are strictly prohibited from production memory.",
                {"vfr_phase": artifact.get("vfr_phase")}
            )
            raise ArtifactPolicyRejectedError("Sidecar exports are strictly prohibited from production memory.")
        return True

def test_gate_2_ingestion_rejection():
    auditor = EvaluationAuditor()
    pipeline = DummyIngestionPipeline(auditor)
    
    sidecar_payload = {
        "mnemos_artifact_type": "sidecar_evaluation_export",
        "production_ingestion_allowed": False,
        "derived_fact_payload_present": True,
        "vfr_phase": "VFR-7",
        "data": {}
    }
    
    with pytest.raises(ArtifactPolicyRejectedError):
        pipeline.ingest(sidecar_payload)
        
    assert len(auditor.events) == 1
    ev = auditor.events[0]
    assert ev["event_type"] == "SIDECAR_EXPORT_INGESTION_REJECTED"
    assert ev["payload"]["reason"] == "Sidecar exports are strictly prohibited from production memory."

def test_normal_ingestion():
    auditor = EvaluationAuditor()
    pipeline = DummyIngestionPipeline(auditor)
    
    normal_payload = {
        "mnemos_artifact_type": "standard_engram",
        "content": "Real data"
    }
    
    assert pipeline.ingest(normal_payload) is True
    assert len(auditor.events) == 0
