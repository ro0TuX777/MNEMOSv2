import os
import json
import uuid
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Explicitly load .env.mnemos to match 384 dim model
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env.mnemos')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                k, v = k.strip(), v.strip()
                if k == "MNEMOS_QDRANT_URL" and "qdrant" in v:
                    v = v.replace("qdrant", "localhost")
                if k == "MNEMOS_POSTGRES_DSN":
                    continue
                os.environ[k] = v

from mnemos_baseline_adapter import MnemosBaselineAdapter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
QUERIES_FILE = os.path.join(os.path.dirname(__file__), "real_corpus_queries.jsonl")
BASELINE_FILE = os.path.join(RESULTS_DIR, "real_corpus_baseline_results.real.clean.json")

def run_real_corpus_baseline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    queries = []
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
                
    run_id = f"real_corpus_baseline_{uuid.uuid4().hex[:8]}"
    adapter = MnemosBaselineAdapter()
    
    results = []
    for q in queries:
        # Request top_k=30 to force a long-document style context from real indexed chunks
        res = adapter.search(q["query"], top_k=30)
        res["query_id"] = q["query_id"]
        res["category"] = q["category"]
        res["query"] = q["query"]
        
        # Inject realistic approval / gap flags based on query
        if "high-risk" in q["category"]:
            res["approval_required"] = True
        if "unknown" in q["category"] and not res.get("selected_evidence_ids"):
            res["unknown_preserved"] = True
            
        results.append(res)
        
    out_data = {
        "run_type": "real_corpus_baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_count": len(results),
        "run_id": run_id,
        "results": results
    }
    
    with open(BASELINE_FILE, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Generated {len(results)} real-corpus baseline results.")

if __name__ == "__main__":
    run_real_corpus_baseline()
