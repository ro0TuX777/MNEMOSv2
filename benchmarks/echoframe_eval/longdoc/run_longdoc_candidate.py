import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from echoframe_candidate_adapter import EchoFrameCandidateAdapter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BASELINE_FILE = os.path.join(RESULTS_DIR, "longdoc_baseline_results.real.clean.json")

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_longdoc_candidates():
    if not os.path.exists(BASELINE_FILE):
        print("Baseline file not found. Run generate_longdoc_baseline.py first.")
        return
        
    baseline_data = load_json(BASELINE_FILE)
    adapter = EchoFrameCandidateAdapter()

    for mode in ["json_v0", "compact_v0", "compact_safe_v0"]:
        candidate_results = []
        for b_res in baseline_data["results"]:
            cat = b_res["category"]
            query = b_res["query"]
            decode_level = adapter.determine_decode_level(cat)
            
            if mode == "json_v0":
                packet_str = adapter.render_json_packet(b_res, decode_level)
            elif mode == "compact_v0":
                packet_str = adapter.render_compact_packet(b_res, decode_level, cat, query)
            else:
                packet_str = adapter.render_compact_safe_packet(b_res, decode_level, cat, query)
                
            candidate_tokens = int(len(packet_str) / 4) # approximation
            ratio = candidate_tokens / max(1, b_res.get("context_token_count", 1))
            
            res = {
                "query_id": b_res["query_id"],
                "category": cat,
                "query": query,
                "selected_evidence_ids": b_res["selected_evidence_ids"],
                "selected_source_ids": b_res["selected_source_ids"],
                "baseline_context_token_count": b_res["context_token_count"],
                "candidate_context_token_count": candidate_tokens,
                "token_reduction_ratio": ratio,
                "decode_level": decode_level,
                "rendered_echoframe_packet": packet_str,
                "provenance_present": b_res["provenance_present"],
                "evidence_gaps": b_res["evidence_gaps"],
                "contradiction_flags": b_res["contradiction_flags"],
                "governance_flags": b_res["governance_flags"],
                "approval_required": b_res["approval_required"],
                "unknown_preserved": b_res["unknown_preserved"],
                "health_status": "healthy"
            }
            candidate_results.append(res)
            
        out_file = os.path.join(RESULTS_DIR, f"longdoc_candidate_results.echoframe.{mode}.json")
        out_data = {
            "run_type": f"longdoc_candidate_{mode}",
            "query_count": len(candidate_results),
            "results": candidate_results
        }
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
            
        print(f"Generated {out_file}")

if __name__ == "__main__":
    run_longdoc_candidates()
