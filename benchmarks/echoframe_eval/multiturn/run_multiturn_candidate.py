import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from echoframe_candidate_adapter import EchoFrameCandidateAdapter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BASELINE_FILE = os.path.join(RESULTS_DIR, "multiturn_baseline_results.real.clean.json")

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_candidates():
    baseline_data = load_json(BASELINE_FILE)
    adapter = EchoFrameCandidateAdapter()

    for mode in ["compact_semantic_minEvidence_v0", "compact_semantic_minEvidence_hysteresis_v0"]:
        output_scenarios = []
        for scen in baseline_data["scenarios"]:
            out_turns = []
            
            # Hysteresis state initialization for this scenario
            hysteresis_state = {} if mode == "compact_semantic_minEvidence_hysteresis_v0" else None
            
            for b_res in scen["turns"]:
                cat = b_res["category"]
                query = b_res["query"]
                decode_level = adapter.determine_decode_level(cat)
                
                if mode == "compact_semantic_minEvidence_v0":
                    packet_str = adapter.render_compact_semantic_minEvidence_packet(b_res, decode_level, cat, query)
                    new_state = None
                else:
                    packet_str, new_state = adapter.render_compact_semantic_minEvidence_hysteresis_packet(b_res, decode_level, cat, query, hysteresis_state)
                    
                candidate_tokens = int(len(packet_str) / 4) # approximation
                ratio = candidate_tokens / max(1, b_res.get("context_token_count", 1))
                
                res = {
                    "query_id": b_res.get("query_id", "Q-???"),
                    "scenario_id": scen["scenario_id"],
                    "turn_id": b_res["turn_id"],
                    "candidate_context_token_count": candidate_tokens,
                    "token_reduction_ratio": round(ratio, 4),
                    "decode_level": decode_level,
                    "rendered_echoframe_packet": packet_str,
                    "provenance_present": b_res.get("provenance_present", False),
                    "evidence_gaps": b_res.get("evidence_gaps", []),
                    "contradiction_flags": b_res.get("contradiction_flags", []),
                    "approval_required": b_res.get("approval_required", False),
                    "unknown_preserved": b_res.get("unknown_preserved", False)
                }
                
                if new_state is not None:
                    res["hysteresis_state"] = new_state
                    hysteresis_state = new_state
                    
                out_turns.append(res)
                
            output_scenarios.append({
                "scenario_id": scen["scenario_id"],
                "turns": out_turns
            })

        out_data = {
            "run_type": f"multiturn_candidate_{mode}",
            "scenario_count": len(output_scenarios),
            "scenarios": output_scenarios
        }
        
        out_file = os.path.join(RESULTS_DIR, f"multiturn_candidate_results.echoframe.{mode}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
            
        print(f"Generated multi-turn candidate output at {out_file}")

if __name__ == "__main__":
    run_candidates()
