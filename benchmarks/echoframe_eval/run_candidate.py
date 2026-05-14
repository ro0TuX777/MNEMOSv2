import os
import json
import time
import uuid

# Explicitly load .env.mnemos before importing MNEMOS modules to ensure
# the correct embedding model (all-MiniLM-L6-v2) is used instead of the 768-dim default.
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env.mnemos')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k == "MNEMOS_QDRANT_URL" and "qdrant" in v:
                    v = v.replace("qdrant", "localhost")
                if k == "MNEMOS_POSTGRES_DSN":
                    continue
                os.environ[k] = v

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "eval_queries.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "candidate_results.echoframe.v0.json")

def load_queries(filepath):
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

import argparse

def run_candidate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="json_v0", choices=["json_v0", "compact_v0", "compact_safe_v0"])
    args = parser.parse_args()

    results_file = os.path.join(RESULTS_DIR, f"candidate_results.echoframe.{args.mode}.hardened.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    queries = load_queries(QUERIES_FILE)
    results = []
    
    run_id = f"echoframe_candidate_run_{uuid.uuid4().hex[:8]}"
    print(f"Starting EchoFrame Candidate Evaluation Run: {run_id} (Mode: {args.mode})")
    
    from echoframe_candidate_adapter import EchoFrameCandidateAdapter
    adapter = EchoFrameCandidateAdapter()

    for q in queries:
        print(f"Processing Query [{q['query_id']}]: {q['category']}")
        
        candidate_output = adapter.search(q["query"], q["category"], mode=args.mode)
        
        query_health = "healthy"
        query_warnings = []
        error_type = None

        if candidate_output.get("errors"):
            query_health = "failed"
            error_type = "retrieval_error"
            query_warnings.append(candidate_output["errors"])
        elif not candidate_output.get("selected_evidence_ids"):
            query_health = "degraded"
            query_warnings.append("No evidence retrieved")
            
        result_record = {
            "query_id": q["query_id"],
            "category": q["category"],
            "query": q["query"],
            "selected_evidence_ids": candidate_output.get("selected_evidence_ids", []),
            "selected_source_ids": candidate_output.get("selected_source_ids", []),
            "baseline_context_token_count": candidate_output.get("baseline_context_token_count", 0),
            "candidate_context_token_count": candidate_output.get("candidate_context_token_count", 0),
            "token_reduction_ratio": candidate_output.get("token_reduction_ratio", 0.0),
            "decode_level": candidate_output.get("decode_level", "D3"),
            "rendered_echoframe_packet": candidate_output.get("rendered_echoframe_packet", ""),
            "provenance_present": candidate_output.get("provenance_present", False),
            "evidence_gaps": candidate_output.get("evidence_gaps", []),
            "contradiction_flags": candidate_output.get("contradiction_flags", []),
            "governance_flags": candidate_output.get("governance_flags", []),
            "approval_required": candidate_output.get("approval_required", False),
            "unknown_preserved": candidate_output.get("unknown_preserved", False),
            "health_status": query_health,
            "warnings": query_warnings,
            "errors": error_type
        }
        results.append(result_record)
        
    output_data = {
        "run_type": f"echoframe_candidate_{args.mode}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_count": len(queries),
        "run_id": run_id,
        "mode": args.mode,
        "results": results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nCandidate evaluation complete. Results written to {results_file}")

if __name__ == "__main__":
    run_candidate()
