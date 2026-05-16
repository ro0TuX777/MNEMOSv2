import json
import time
from llmlingua_adapters import ModeRunner
from validators import SpanValidator

def run_comparison():
    runner = ModeRunner()
    validator = SpanValidator()
    
    # Mock data, in practice load from datasets/phase10_real_corpus_manifest.json
    contexts = [
        "Source: doc_123. The budget is 100M for 2026-01-01. Governance: approval_required. However, [CONTRADICTION] exists. " * 10
    ]
    
    modes = ["A", "B", "C", "D", "E", "F"]
    results = {}
    
    for mode in modes:
        mode_results = []
        for ctx in contexts:
            out, lat = runner.run_mode(mode, ctx)
            preservation = validator.validate_preservation(ctx, out)
            mode_results.append({
                "tokens": len(out.split()),
                "latency": lat,
                "preservation": preservation
            })
        
        avg_tokens = sum(r["tokens"] for r in mode_results) / len(mode_results)
        avg_lat = sum(r["latency"] for r in mode_results) / len(mode_results)
        # Check if any protected span failed
        failed_spans = sum(1 for r in mode_results if not all(r["preservation"].values()))
        
        results[mode] = {
            "avg_tokens": avg_tokens,
            "avg_latency": avg_lat,
            "failed_spans": failed_spans
        }
        
    print("Comparison Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_comparison()
