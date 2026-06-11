import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from mnemos.retrieval.sidecar import FactAwareEvaluationSidecar, ShadowModeDisabledError
from mnemos.retrieval.auditor import EvaluationAuditor

def search_derived_trial(router: Any, query: str, top_k: int, client_id: str, auditor: Any = None) -> Dict[str, Any]:
    """
    DFE-20 Extended Operator Trial execution function.
    Safely executes the derived fact shadow path, filters candidates against strict
    governance schema bounds, and returns them strictly isolated within derived_lane_meta.
    """
    pit2_response = router.search_derived(
        query=query, 
        top_k=top_k,
        client_id=client_id,
        include_derived_facts=True
    )
    
    raw_derived_facts = pit2_response.get("derived_results", [])
    candidate_telemetry = pit2_response.get("derived_lane_meta", {}).get("candidate_telemetry", [])
    
    validated_derived_facts = []
    for fact in raw_derived_facts:
        if fact.get("authority_label") != "MNEMOS_DERIVED_FACT":
            continue
        if not fact.get("rendered_support_excerpt"):
            continue
        if not fact.get("source_document"):
            continue
        if not fact.get("source_engram_id"):
            continue
        if not fact.get("selection_path") in ("STANDARD", "RENDERED_SUPPORT_RESCUE"):
            continue
        validated_derived_facts.append(fact)
        
    if auditor:
        ts = datetime.utcnow().isoformat() + "Z"
        run_id = str(uuid.uuid4())
        auditor.log_sidecar_invoked(
            operator_id=client_id,
            timestamp=ts,
            sidecar_run_id=run_id,
            query_hash=hashlib.md5(query.encode('utf-8')).hexdigest(),
            double_opt_in_state=True,
            baseline_retrieval_mode="derived_trial",
            derived_fact_count=len(validated_derived_facts),
            masked_fact_count=len(raw_derived_facts) - len(validated_derived_facts),
            dataset_id="dfe_20_trial",
            workload_type="extended_operator_trial",
            approval_ticket_id="DFE-20",
            case_id="live_api_trial"
        )
        
    return {
        "derived_lane_meta": {
            "derived_results": validated_derived_facts,
            "candidate_telemetry": candidate_telemetry,
            "trial_status": "DFE_20_EXTENDED_OPERATOR_TRIAL"
        }
    }

ALLOWED_DATASETS = {
    "dataset_alpha": {
        "dataset_name": "Historical Outages Q1",
        "owner": "SRE_Team",
        "classification_sensitivity_label": "internal",
        "approved_workload_types": ["retrospective_capability_audit", "offline_analytic_drafting"]
    },
    "dataset_bravo": {
        "dataset_name": "Support Ticket Archive",
        "owner": "CX_Team",
        "classification_sensitivity_label": "confidential",
        "approved_workload_types": ["knowledge_base_curation"]
    }
}

class EvaluationConsoleAPI:
    def __init__(self, sidecar: FactAwareEvaluationSidecar, auditor: EvaluationAuditor):
        self.sidecar = sidecar
        self.auditor = auditor

    def _check_rbac(self, session: Dict[str, Any]) -> bool:
        roles = session.get("roles", [])
        return "ROLE_MEMORY_EVALUATOR" in roles or "ROLE_SYSTEM_ADMIN" in roles

    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def invoke(self, session: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        op_id = session.get("operator_id", "unknown")
        ts = self._get_timestamp()
        
        dataset_id = payload.get("dataset_id")
        workload_type = payload.get("workload_type")
        approval_ticket_id = payload.get("approval_ticket_id")
        case_id = payload.get("case_id")
        
        # 1. RBAC
        if not self._check_rbac(session):
            self.auditor.log_sidecar_invoke_blocked(
                op_id, ts, "RBAC_DENIED", False, False, False,
                dataset_id=dataset_id, workload_type=workload_type, 
                approval_ticket_id=approval_ticket_id, case_id=case_id
            )
            return {"status": 403, "error": "Forbidden"}

        # 2. Extract explicit metadata
        query = payload.get("query", "")
        op_override = payload.get("operator_override", False)
        fact_aware = payload.get("enable_fact_awareness", False)
        acknowledged = payload.get("evaluation_mode_acknowledged", False)
        
        # 3. Double Opt-In
        opt_in_satisfied = op_override and fact_aware and acknowledged
        if not opt_in_satisfied:
            self.auditor.log_sidecar_invoke_blocked(
                op_id, ts, "DOUBLE_OPT_IN_MISSING", True, False, False,
                dataset_id=dataset_id, workload_type=workload_type, 
                approval_ticket_id=approval_ticket_id, case_id=case_id
            )
            return {"status": 400, "error": "Double opt-in missing"}

        # 4. Allowlist checks
        if not dataset_id or dataset_id not in ALLOWED_DATASETS:
            self.auditor.log_sidecar_invoke_blocked(
                op_id, ts, "DATASET_NOT_APPROVED", True, True, False,
                dataset_id=dataset_id, workload_type=workload_type, 
                approval_ticket_id=approval_ticket_id, case_id=case_id
            )
            return {"status": 403, "error": "Dataset not approved"}
            
        allowed_workloads = ALLOWED_DATASETS[dataset_id]["approved_workload_types"]
        if workload_type not in allowed_workloads:
            self.auditor.log_sidecar_invoke_blocked(
                op_id, ts, "WORKLOAD_NOT_APPROVED", True, True, False,
                dataset_id=dataset_id, workload_type=workload_type, 
                approval_ticket_id=approval_ticket_id, case_id=case_id
            )
            return {"status": 403, "error": "Workload not approved"}

        if not approval_ticket_id:
            self.auditor.log_sidecar_invoke_blocked(
                op_id, ts, "APPROVAL_TICKET_MISSING", True, True, False,
                dataset_id=dataset_id, workload_type=workload_type, 
                approval_ticket_id=approval_ticket_id, case_id=case_id
            )
            return {"status": 403, "error": "Approval ticket missing"}

        # 5. Invoke Sidecar
        run_id = str(uuid.uuid4())
        try:
            packet, tel = self.sidecar.execute_fact_aware_query(
                query, top_k=5, operator_override=True, enable_fact_awareness=True
            )
        except ShadowModeDisabledError:
            self.auditor.log_sidecar_invoke_blocked(
                op_id, ts, "KILL_SWITCH_ACTIVE", True, True, True, run_id,
                dataset_id=dataset_id, workload_type=workload_type, 
                approval_ticket_id=approval_ticket_id, case_id=case_id
            )
            return {"status": 409, "error": "Kill switch active"}

        # 6. Emit SIDECAR_INVOKED
        self.auditor.log_sidecar_invoked(
            op_id, ts, run_id, hashlib.md5(query.encode('utf-8')).hexdigest(),
            True, tel.get("baseline_retrieval_mode", "unknown"),
            tel.get("derived_fact_count", 0), tel.get("masked_fact_count", 0),
            dataset_id=dataset_id, workload_type=workload_type,
            approval_ticket_id=approval_ticket_id, case_id=case_id
        )
        
        return {"status": 200, "data": packet, "sidecar_run_id": run_id}

    def export(self, session: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        op_id = session.get("operator_id", "unknown")
        ts = self._get_timestamp()
        
        if not self._check_rbac(session):
            return {"status": 403, "error": "Forbidden"}
            
        run_id = payload.get("sidecar_run_id", "")
        data = payload.get("data", {})
        used_facts = payload.get("derived_facts_used", [])
        dataset_id = payload.get("dataset_id", "")
        workload_type = payload.get("workload_type", "")
        approval_ticket_id = payload.get("approval_ticket_id", "")
        export_purpose = payload.get("export_purpose", "")
        
        if not dataset_id or not workload_type or not approval_ticket_id or not export_purpose:
            return {"status": 400, "error": "Missing export tracking metadata"}
            
        if dataset_id not in ALLOWED_DATASETS:
            return {"status": 403, "error": "Dataset not approved for export"}
            
        art_id = "export_" + str(uuid.uuid4())
        
        # Emit Reliance
        self.auditor.log_sidecar_output_relied_upon(
            op_id, ts, run_id, used_facts, True, "json", export_purpose,
            dataset_id, workload_type, approval_ticket_id, art_id
        )
        
        raw_payload = {
            "mnemos_artifact_type": "sidecar_evaluation_export",
            "production_ingestion_allowed": False,
            "derived_fact_payload_present": True,
            "vfr_phase": "VFR-9",
            "dataset_id": dataset_id,
            "workload_type": workload_type,
            "approval_ticket_id": approval_ticket_id,
            "export_purpose": export_purpose,
            "derived_facts_used": used_facts,
            "data": data
        }
        
        raw_payload["watermark_hash"] = hashlib.md5(str(raw_payload).encode('utf-8')).hexdigest()
        export_path = f"/tmp/mnemos_evaluation_exports/{art_id}.json"
        
        return {"status": 200, "export_path": export_path, "payload": raw_payload}

    def flag(self, session: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        op_id = session.get("operator_id", "unknown")
        ts = self._get_timestamp()
        
        if not self._check_rbac(session):
            return {"status": 403, "error": "Forbidden"}
            
        self.auditor.log_derived_fact_flagged_for_review(
            op_id,
            payload.get("fact_id", ""),
            payload.get("promotion_receipt_id", ""),
            payload.get("source_engram_id", ""),
            payload.get("reason_code", ""),
            ts,
            payload.get("sidecar_run_id", ""),
            payload.get("free_text_note", "")
        )
        return {"status": 200, "message": "Flagged successfully. No state mutated."}

    def reliance(self, session: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        op_id = session.get("operator_id", "unknown")
        ts = self._get_timestamp()
        
        if not self._check_rbac(session):
            return {"status": 403, "error": "Forbidden"}
            
        self.auditor.log_sidecar_output_relied_upon(
            op_id, ts,
            payload.get("sidecar_run_id", ""),
            payload.get("derived_facts_used", []),
            payload.get("contradiction_warning_present", False),
            payload.get("export_format", "ui_view"),
            payload.get("export_purpose", "view_only"),
            payload.get("dataset_id", "unknown"),
            payload.get("workload_type", "unknown"),
            payload.get("approval_ticket_id", "unknown")
        )
        return {"status": 200, "message": "Reliance logged"}
