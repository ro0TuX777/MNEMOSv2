import os
import json
import time
import uuid
import argparse

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

# Configuration
QUERIES_FILE = os.path.join(os.path.dirname(__file__), "eval_queries.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def load_queries(filepath):
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

def stub_mnemos_read_path(query_text):
    """
    STUB: Simulated read path for test-mode validation.
    """
    time.sleep(0.05)
    
    return {
        "selected_evidence_ids": ["doc_1", "doc_2"],
        "selected_source_ids": ["source_doc_1.md", "source_doc_2.md"],
        "retrieval_scores": [0.95, 0.88],
        "rendered_context": f"Simulated raw context for query: {query_text}",
        "context_token_count": len(query_text.split()) * 15,
        "answer_text": "Simulated generated answer.",
        "answer_token_count": 10,
        "latency_ms": 50.0,
        "provenance_present": True,
        "evidence_gaps": [],
        "contradiction_flags": [],
        "governance_flags": ["STANDARD_RETRIEVAL"],
        "approval_required": False,
        "unknown_preserved": False,
        "errors": None,
        "notes": "Stubbed baseline execution."
    }

def get_git_commit():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def run_baseline():
    parser = argparse.ArgumentParser(description="MNEMOS Baseline Runner")
    parser.add_argument("--use-stub", action="store_true", help="Run with simulated stub mode instead of real MNEMOS adapter.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    queries = load_queries(QUERIES_FILE)
    results = []
    
    run_id = f"baseline_run_{uuid.uuid4().hex[:8]}"
    print(f"Starting MNEMOS Baseline Evaluation Run: {run_id}")
    print(f"Mode: {'STUB (Instrumentation test only)' if args.use_stub else 'REAL MNEMOS BASELINE'}")
    
    adapter = None
    baseline_health = "healthy"
    retrieval_health = {
        "status": "healthy",
        "vector_store_available": True,
        "vector_dimension_match": True,
        "lexical_retrieval_available": False,
        "governance_trace_available": True,
        "fallback_used": False,
        "warnings": []
    }
    
    if not args.use_stub:
        from mnemos_baseline_adapter import MnemosBaselineAdapter
        adapter = MnemosBaselineAdapter()
        # Verify basic health
        if adapter.runtime._error:
            baseline_health = "failed"
            retrieval_health["status"] = "failed"
            retrieval_health["warnings"].append(adapter.runtime._error)
        
        # Check tiers
        tiers = adapter.runtime._config.tiers if adapter.runtime._config else []
        if "qdrant" not in tiers:
            retrieval_health["warnings"].append("Qdrant tier not enabled")
        
        lexical = adapter.runtime._lexical_tier
        if not lexical:
            retrieval_health["lexical_retrieval_available"] = False
            retrieval_health["warnings"].append("Lexical retrieval tier unavailable")

    for q in queries:
        print(f"Processing Query [{q['query_id']}]: {q['category']}")
        
        query_health = "healthy"
        query_warnings = []
        fallback_used = False
        error_type = None

        if args.use_stub:
            read_path_output = stub_mnemos_read_path(q["query"])
        else:
            read_path_output = adapter.search(q["query"])
            
            if read_path_output.get("errors"):
                query_health = "failed"
                baseline_health = "degraded"
                retrieval_health["status"] = "degraded"
                error_type = "retrieval_error"
                query_warnings.append(read_path_output["errors"])
                if "dimension" in read_path_output["errors"].lower():
                    retrieval_health["vector_dimension_match"] = False
                    
            elif not read_path_output.get("selected_evidence_ids"):
                query_health = "degraded"
                query_warnings.append("No evidence retrieved")
        
        result_record = {
            "query_id": q["query_id"],
            "category": q["category"],
            "query": q["query"],
            "health_status": query_health,
            "warnings": query_warnings,
            "fallback_used": fallback_used,
            "error_type": error_type,
            "selected_evidence_ids": read_path_output.get("selected_evidence_ids", []),
            "selected_source_ids": read_path_output.get("selected_source_ids", []),
            "retrieval_scores": read_path_output.get("retrieval_scores", []),
            "rendered_context": read_path_output.get("rendered_context", ""),
            "context_token_count": read_path_output.get("context_token_count", 0),
            "answer_text": read_path_output.get("answer_text"),
            "answer_token_count": read_path_output.get("answer_token_count", 0),
            "latency_ms": read_path_output.get("latency_ms", 0.0),
            "provenance_present": read_path_output.get("provenance_present", False),
            "evidence_gaps": read_path_output.get("evidence_gaps", []),
            "contradiction_flags": read_path_output.get("contradiction_flags", []),
            "governance_flags": read_path_output.get("governance_flags", []),
            "approval_required": read_path_output.get("approval_required", False),
            "unknown_preserved": read_path_output.get("unknown_preserved", False),
            "errors": read_path_output.get("errors"),
            "notes": read_path_output.get("notes")
        }
        results.append(result_record)
        
    output_data = {
        "run_type": "mnemos_baseline",
        "adapter": "stub" if args.use_stub else "real_mnemos_read_path",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mnemos_commit": get_git_commit(),
        "query_count": len(queries),
        "stub_mode": args.use_stub,
        "run_id": run_id,
        "baseline_health": baseline_health,
        "retrieval_health": retrieval_health,
        "results": results
    }
    
    if args.use_stub:
        suffix = "stub"
    else:
        suffix = f"real.{'clean' if baseline_health == 'healthy' else 'degraded'}"
        
    results_file = os.path.join(RESULTS_DIR, f"baseline_results.{suffix}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nBaseline evaluation complete. Results written to {results_file}")

if __name__ == "__main__":
    run_baseline()
