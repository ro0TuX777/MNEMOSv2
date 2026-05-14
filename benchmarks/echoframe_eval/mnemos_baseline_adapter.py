import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from service.app import MnemosRuntime
except ImportError:
    MnemosRuntime = None

class MnemosBaselineAdapter:
    def __init__(self):
        if MnemosRuntime is None:
            raise RuntimeError("MNEMOS service package not found. Are you running this from the repository root?")
        self.runtime = MnemosRuntime()
        self.runtime.initialize()

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Executes the real MNEMOS read path via the MnemosRuntime instance.
        This captures the full retrieval, evidence selection, and governance pipeline.
        """
        t0 = time.time()
        
        try:
            # We use enforced governance to ensure policies and contradictions are active
            res = self.runtime.search_documents(
                query=query,
                top_k=top_k,
                tiers=None,
                filters=None,
                retrieval_mode=None, 
                fusion_policy=None,
                explain=True,
                governance="enforced",
                explain_governance=True
            )
            
            latency_ms = (time.time() - t0) * 1000.0
            results_list = res.get("results", [])
            
            selected_evidence_ids = [r["engram"]["id"] for r in results_list]
            selected_source_ids = [r["engram"].get("source", "") for r in results_list]
            retrieval_scores = [r.get("score", 0.0) for r in results_list]
            
            # Simple context rendering as MNEMOS returns raw engram content
            rendered_context = json.dumps([r["engram"]["content"] for r in results_list], indent=2)
            context_token_count = int(len(rendered_context) / 4) # rough estimate
            
            gov_summary = res.get("meta", {}).get("governance_summary", {})
            governance_flags = []
            if gov_summary.get("vetoed", 0) > 0:
                governance_flags.append("VETOED")
            if gov_summary.get("suppressed", 0) > 0:
                governance_flags.append("SUPPRESSED")
            if gov_summary.get("contradictions_detected", 0) > 0:
                governance_flags.append("CONTRADICTION_DETECTED")
                
            contradiction_flags = [flag for flag in governance_flags if "CONTRADICTION" in flag]
            
            return {
                "selected_evidence_ids": selected_evidence_ids,
                "selected_source_ids": selected_source_ids,
                "retrieval_scores": retrieval_scores,
                "rendered_context": rendered_context,
                "context_token_count": context_token_count,
                "answer_text": None,
                "answer_token_count": 0,
                "latency_ms": round(latency_ms, 2),
                "provenance_present": len(selected_source_ids) > 0,
                "evidence_gaps": [],
                "contradiction_flags": contradiction_flags,
                "governance_flags": governance_flags,
                "approval_required": "APPROVAL_REQUIRED" in governance_flags,
                "unknown_preserved": False,
                "errors": None,
                "notes": "Executed real MNEMOS baseline via MnemosRuntime."
            }
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            return {
                "selected_evidence_ids": [],
                "selected_source_ids": [],
                "retrieval_scores": [],
                "rendered_context": "",
                "context_token_count": 0,
                "answer_text": None,
                "answer_token_count": 0,
                "latency_ms": round(latency_ms, 2),
                "provenance_present": False,
                "evidence_gaps": [],
                "contradiction_flags": [],
                "governance_flags": [],
                "approval_required": False,
                "unknown_preserved": False,
                "errors": str(e),
                "notes": "Error during real baseline execution."
            }
