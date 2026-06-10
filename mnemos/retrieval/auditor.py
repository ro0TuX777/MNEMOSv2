from typing import Dict, Any, Optional

class EvaluationAuditor:
    """Handles audit telemetry for the Fact-Aware Evaluation Sidecar."""
    
    def __init__(self):
        self.events = []
        
    def log_event(self, event_type: str, payload: Dict[str, Any]):
        self.events.append({
            "event_type": event_type,
            "payload": payload
        })
        
    def log_sidecar_invoked(
        self, operator_id: str, timestamp: str, sidecar_run_id: str, 
        query_hash: str, double_opt_in_state: bool, baseline_retrieval_mode: str, 
        derived_fact_count: int, masked_fact_count: int,
        dataset_id: str, workload_type: str, approval_ticket_id: str, case_id: str
    ):
        self.log_event("SIDECAR_INVOKED", {
            "operator_id": operator_id,
            "timestamp": timestamp,
            "sidecar_run_id": sidecar_run_id,
            "query_hash": query_hash,
            "double_opt_in_state": double_opt_in_state,
            "baseline_retrieval_mode": baseline_retrieval_mode,
            "derived_fact_count": derived_fact_count,
            "masked_fact_count": masked_fact_count,
            "dataset_id": dataset_id,
            "workload_type": workload_type,
            "approval_ticket_id": approval_ticket_id,
            "case_id": case_id
        })

    def log_sidecar_invoke_blocked(
        self, operator_id: str, timestamp: str, reason_code: str, 
        rbac_passed: bool, double_opt_in_state: bool, kill_switch_state: bool, 
        sidecar_run_id_optional: Optional[str] = None,
        dataset_id: Optional[str] = None, workload_type: Optional[str] = None, 
        approval_ticket_id: Optional[str] = None, case_id: Optional[str] = None
    ):
        self.log_event("SIDECAR_INVOKE_BLOCKED", {
            "operator_id": operator_id,
            "timestamp": timestamp,
            "sidecar_run_id_optional": sidecar_run_id_optional,
            "reason_code": reason_code,
            "rbac_passed": rbac_passed,
            "double_opt_in_state": double_opt_in_state,
            "kill_switch_state": kill_switch_state,
            "dataset_id": dataset_id,
            "workload_type": workload_type,
            "approval_ticket_id": approval_ticket_id,
            "case_id": case_id
        })

    def log_sidecar_output_relied_upon(
        self, operator_id: str, timestamp: str, sidecar_run_id: str, 
        derived_facts_used: list, contradiction_warning_present: bool, 
        export_format: str, export_purpose: str, dataset_id: str,
        workload_type: str, approval_ticket_id: str,
        export_artifact_id_optional: Optional[str] = None
    ):
        self.log_event("SIDECAR_OUTPUT_RELIED_UPON", {
            "operator_id": operator_id,
            "timestamp": timestamp,
            "sidecar_run_id": sidecar_run_id,
            "derived_facts_used": derived_facts_used,
            "contradiction_warning_present": contradiction_warning_present,
            "export_format": export_format,
            "export_purpose": export_purpose,
            "dataset_id": dataset_id,
            "workload_type": workload_type,
            "approval_ticket_id": approval_ticket_id,
            "export_artifact_id_optional": export_artifact_id_optional
        })

    def log_derived_fact_flagged_for_review(
        self, operator_id: str, fact_id: str, promotion_receipt_id: str, 
        source_engram_id: str, reason_code: str, timestamp: str, 
        sidecar_run_id: str, free_text_note_optional: Optional[str] = None
    ):
        self.log_event("DERIVED_FACT_FLAGGED_FOR_REVIEW", {
            "operator_id": operator_id,
            "fact_id": fact_id,
            "promotion_receipt_id": promotion_receipt_id,
            "source_engram_id": source_engram_id,
            "reason_code": reason_code,
            "timestamp": timestamp,
            "sidecar_run_id": sidecar_run_id,
            "free_text_note_optional": free_text_note_optional
        })

    def log_ingestion_rejection(
        self, artifact_type: str, timestamp: str, reason: str, metadata: Dict[str, Any]
    ):
        self.log_event("SIDECAR_EXPORT_INGESTION_REJECTED", {
            "artifact_type": artifact_type,
            "timestamp": timestamp,
            "reason": reason,
            "metadata": metadata
        })
