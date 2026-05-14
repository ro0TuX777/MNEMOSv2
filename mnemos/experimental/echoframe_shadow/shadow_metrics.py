import hashlib
from datetime import datetime
from typing import Dict, Any, List

class ShadowMetrics:
    @staticmethod
    def construct_telemetry_payload(
        query: str,
        mode: str,
        baseline_tokens: int,
        echoframe_tokens: int,
        evidence_count: int,
        source_count: int,
        hysteresis_state: Dict[str, Any],
        safety_failures: List[str],
        fallback_used: bool,
        fallback_reason: str,
        session_id: str = "S-000",
        turn_id: str = "T-0"
    ) -> Dict[str, Any]:
        
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        ratio = echoframe_tokens / max(1, baseline_tokens)
        
        # Analyze hysteresis state
        decisions = hysteresis_state.get("hysteresis_decisions", [])
        levels = hysteresis_state.get("evidence_render_levels", {})
        
        e0_count = sum(1 for l in levels.values() if l == "E0")
        e2_count = sum(1 for l in levels.values() if l == "E2")
        e3_count = sum(1 for l in levels.values() if l == "E3" or l == "E4")
        
        unjustified_churn = 0
        for d in decisions:
            if d["decision"] == "drop" and d["reason"] != "irrelevant":
                unjustified_churn += 1
            if d["decision"] in ["expand", "compress"] and "risk" not in d["reason"] and "relevance" not in d["reason"]:
                unjustified_churn += 1
                
        total_sources = len(levels)
        stability_score = 1.0 - (unjustified_churn / max(1, total_sources))
        
        return {
            "event_type": "echoframe.ab_shadow_event",
            "phase": "phase5c_100pct_shadow",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "turn_id": turn_id,
            "query_hash": query_hash,
            "sampled": True,
            "renderer_mode": mode,
            "baseline_token_count": baseline_tokens,
            "echoframe_token_count": echoframe_tokens,
            "token_ratio": round(ratio, 4),
            "selected_evidence_count": evidence_count,
            "source_count": source_count,
            "e0_pointer_only_count": e0_count,
            "e2_window_count": e2_count,
            "e3_full_excerpt_count": e3_count,
            "stability_score": stability_score,
            "unjustified_churn": unjustified_churn,
            "safety_gate_failures": safety_failures,
            "validator_failures": [],
            "non_promotable": False,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "llm_context_modified": False,
            "shadow_error": None
        }
