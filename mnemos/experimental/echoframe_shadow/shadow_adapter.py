import os
import json
import traceback
import sys

# We add the benchmark dir to path to use the existing tested renderer
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "benchmarks", "echoframe_eval"))
try:
    from echoframe_candidate_adapter import EchoFrameCandidateAdapter
except ImportError:
    EchoFrameCandidateAdapter = None

from .shadow_config import ShadowConfig
from .shadow_receipts import ShadowReceipts
from .shadow_metrics import ShadowMetrics

# Global state to maintain hysteresis across turns in shadow mode
_SESSION_STATES = {}
_GLOBAL_ADAPTER = None

class EchoFrameShadowAdapter:
    @staticmethod
    def observe_search(query: str, baseline_payload: dict, session_id: str = "default_session", turn_id: str = "unknown_turn"):
        """
        Observes the baseline MNEMOS retrieval result and asynchronously or synchronously 
        generates the EchoFrame candidate packet, running validation and emitting telemetry.
        """
        global _GLOBAL_ADAPTER
        if not ShadowConfig.is_enabled():
            return
            
        if EchoFrameCandidateAdapter is None:
            print("EchoFrameShadowAdapter: Could not load EchoFrameCandidateAdapter. Shadow mode aborted.")
            return

        try:
            if _GLOBAL_ADAPTER is None:
                _GLOBAL_ADAPTER = EchoFrameCandidateAdapter()
            adapter = _GLOBAL_ADAPTER

            mode = ShadowConfig.get_mode()
            output_dir = ShadowConfig.get_output_dir()
            sample_rate = ShadowConfig.get_sample_rate()
            
            import random
            if random.random() > sample_rate:
                # We skip processing, but we could log that it was bypassed
                return
            
            # Extract baseline data
            results = baseline_payload.get("results", [])
            rendered_context_list = [r.get("engram", {}).get("content", "") for r in results]
            rendered_context_str = json.dumps(rendered_context_list)
            
            selected_evidence_ids = [r.get("engram", {}).get("id", "") for r in results]
            selected_source_ids = [r.get("engram", {}).get("source", "") for r in results]
            
            # Approximate baseline tokens
            baseline_tokens = int(len(rendered_context_str) / 4)
            
            # Reconstruct governance flags
            meta = baseline_payload.get("meta", {})
            gov_summary = meta.get("governance_summary", {})
            governance_flags = []
            if gov_summary.get("vetoed", 0) > 0: governance_flags.append("VETOED")
            if gov_summary.get("suppressed", 0) > 0: governance_flags.append("SUPPRESSED")
            if gov_summary.get("contradictions_detected", 0) > 0: governance_flags.append("CONTRADICTION_DETECTED")
            
            baseline_res = {
                "rendered_context": rendered_context_str,
                "selected_evidence_ids": selected_evidence_ids,
                "selected_source_ids": selected_source_ids,
                "evidence_gaps": [], # MNEMOS real runtime handles gaps differently
                "contradiction_flags": [f for f in governance_flags if "CONTRADICTION" in f],
                "approval_required": "APPROVAL_REQUIRED" in governance_flags,
                "unknown_preserved": False
            }
            
            # Category classification
            q_lower = query.lower()
            category = "general"
            if "delet" in q_lower or "bypas" in q_lower or "contradict" in q_lower: category = "high-risk"
            elif "rule" in q_lower or "except" in q_lower: category = "policy"
            elif "date" in q_lower or "timeline" in q_lower: category = "insufficient"
            elif "api" in q_lower or "key" in q_lower: category = "api"
            elif "threshold" in q_lower or "exact" in q_lower: category = "numeric"
            
            decode_level = adapter.determine_decode_level(category)
            
            # Hysteresis state isolation per session
            if session_id not in _SESSION_STATES:
                _SESSION_STATES[session_id] = {}
            prev_state = _SESSION_STATES[session_id]
            
            fallback_used = False
            fallback_reason = None
            packet_str = ""
            new_state = {}
            
            if mode == "compact_semantic_minEvidence_hysteresis_v0":
                packet_str, new_state = adapter.render_compact_semantic_minEvidence_hysteresis_packet(
                    baseline_results=baseline_res,
                    decode_level=decode_level,
                    category=category,
                    query=query,
                    state=prev_state
                )
            else:
                # Basic minEvidence logic for testing
                packet_str = adapter.render_compact_semantic_minEvidence_packet(
                    baseline_results=baseline_res,
                    decode_level=decode_level,
                    category=category,
                    query=query
                )
                
            _SESSION_STATES[session_id] = new_state
            
            # Check for fallback
            if "fallback=" in packet_str:
                fallback_used = True
                fallback_reason = packet_str.split("fallback=")[1].split()[0]
                
            candidate_tokens = int(len(packet_str) / 4)
            
            # Basic validation
            safety_failures = []
            if len(selected_evidence_ids) > 0 and "NO_EVIDENCE_FOUND" in packet_str:
                safety_failures.append("dropped_provenance")
            
            telemetry = ShadowMetrics.construct_telemetry_payload(
                query=query,
                mode=mode,
                baseline_tokens=baseline_tokens,
                echoframe_tokens=candidate_tokens,
                evidence_count=len(selected_evidence_ids),
                source_count=len(set(selected_source_ids)),
                hysteresis_state=new_state,
                safety_failures=safety_failures,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                session_id=session_id,
                turn_id=turn_id
            )
            
            from .shadow_config import PilotConfig, DefaultOnConfig
            
            is_pilot = PilotConfig.is_enabled()
            is_default_on = DefaultOnConfig.is_enabled()
            is_kill_switch = DefaultOnConfig.is_kill_switch_active()
            
            if is_kill_switch or is_default_on or is_pilot:
                # Precedence:
                # 1. Kill switch
                # 2. Default on
                # 3. Pilot
                
                output_dir = DefaultOnConfig.get_output_dir() if is_default_on or is_kill_switch else PilotConfig.get_output_dir()
                mode = DefaultOnConfig.get_mode() if is_default_on or is_kill_switch else PilotConfig.get_mode()
                sample_rate = 1.0 if is_default_on or is_kill_switch else PilotConfig.get_sample_rate()
                allow_high_risk = DefaultOnConfig.get_allow_high_risk() if is_default_on or is_kill_switch else PilotConfig.get_allow_high_risk()
                
                import random
                pilot_selected = True if is_default_on or is_kill_switch else (random.random() <= sample_rate)
                
                # Check gates
                token_ratio = telemetry.get("token_ratio", 1.0)
                stability_score = telemetry.get("stability_score", 1.0)
                non_promotable = telemetry.get("non_promotable", False)
                
                gate_failures = []
                if is_kill_switch: gate_failures.append("kill_switch_active")
                if safety_failures: gate_failures.append("safety_failures")
                if non_promotable: gate_failures.append("non_promotable")
                if token_ratio > 1.00: gate_failures.append("token_ratio_too_high")
                if stability_score < 0.90: gate_failures.append("unstable_context")
                if category == "high-risk" and not allow_high_risk: gate_failures.append("high_risk_excluded")
                if baseline_res["approval_required"]: gate_failures.append("approval_required_excluded")
                if fallback_used: gate_failures.append("fallback_used")
                
                eligible = len(gate_failures) == 0 or (len(gate_failures) == 1 and gate_failures[0] == "kill_switch_active") # eligible is independent of kill switch
                
                # Recompute eligible to accurately reflect if it WOULD have been eligible without kill switch
                real_gate_failures = [g for g in gate_failures if g != "kill_switch_active"]
                eligible = len(real_gate_failures) == 0
                
                fallback_to_baseline = True
                pilot_packet = None
                
                if not is_kill_switch and pilot_selected and eligible:
                    fallback_to_baseline = False
                    pilot_packet = packet_str
                    
                import hashlib
                from datetime import datetime
                
                event_type = "echoframe.default_on_event" if is_kill_switch or is_default_on else "echoframe.llm_facing_pilot_event"
                phase = "phase7_default_on_eligible" if is_kill_switch or is_default_on else "phase6_controlled_llm_facing_pilot"
                
                pilot_event = {
                  "event_type": event_type,
                  "phase": phase,
                  "session_id": session_id,
                  "turn_id": turn_id,
                  "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
                  "eligible_for_pilot": eligible,  # legacy name
                  "eligible_for_echoframe": eligible, # new name
                  "pilot_selected": pilot_selected,
                  "default_on_enabled": is_default_on,
                  "kill_switch_active": is_kill_switch,
                  "llm_context_source": "baseline" if fallback_to_baseline else "echoframe",
                  "fallback_to_baseline": fallback_to_baseline,
                  "fallback_reason": ", ".join(gate_failures) if gate_failures else None,
                  "renderer_mode": mode,
                  "baseline_token_count": baseline_tokens,
                  "echoframe_token_count": candidate_tokens,
                  "token_ratio": token_ratio,
                  "stability_score": stability_score,
                  "safety_gate_failures": safety_failures,
                  "validator_failures": [],
                  "non_promotable": non_promotable,
                  "risk_level": category,
                  "approval_required": baseline_res["approval_required"],
                  "answer_quality_review_required": not fallback_to_baseline,
                  "source_pointer_count": len(set(selected_source_ids)),
                  "governance_signal_count": 1 if non_promotable else 0 # Simplified
                }
                
                ShadowReceipts.emit_telemetry(pilot_event, output_dir)
                if output_dir != ShadowConfig.get_output_dir():
                    # We still emit the base telemetry for backward compat, but to shadow dir
                    ShadowReceipts.emit_telemetry(telemetry, ShadowConfig.get_output_dir())
                return pilot_packet
                
            ShadowReceipts.emit_telemetry(telemetry, PilotConfig.get_output_dir())
            return None
            
        except Exception as e:
            if ShadowConfig.get_fail_closed():
                raise e
            print(f"Shadow observation failed: {e}")
            traceback.print_exc()
            raise e # re-raise to catch in app.py

    @staticmethod
    def emit_failure(query: str, exc: Exception):
        try:
            output_dir = ShadowConfig.get_output_dir()
            payload = {
                "event_type": "echoframe.shadow_packet_failed",
                "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
                "query_hash": __import__('hashlib').sha256(query.encode()).hexdigest()[:16],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "llm_context_modified": False,
                "baseline_runtime_unaffected": True
            }
            ShadowReceipts.emit_telemetry(payload, output_dir)
        except Exception:
            pass # Silent fail if telemetry itself fails

