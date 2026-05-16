import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../echoframe_phase12a'))
from placement_layouts import EchoFramePacket
from format_renderers import render_format, assert_format_preserves_packet
from format_cases import get_format_cases
from format_metrics import score_format_answer

def mock_llm_call(prompt: str, model: str, case_family: str) -> str:
    lower_prompt = prompt.lower()
    answer = []
    
    if "manager approval" in lower_prompt and "$500" in lower_prompt:
        answer.append("Yes, it requires manager approval for $500. Source S1.")
    elif "october 15" in lower_prompt:
        answer.append("It must be submitted by October 15. Source S2.")
    elif "shall not access" in lower_prompt:
        answer.append("No, contractors shall not access it. Source S3.")
    elif "unless they are encrypted" in lower_prompt:
        answer.append("Only if encrypted and IT-approved. Source S4.")
    elif "contradiction" in lower_prompt or "conflict" in lower_prompt:
        answer.append("There is a contradiction: $50 vs $75. Sources S5, S6.")
    elif "gap" in lower_prompt:
        answer.append("Missing evidence. Gap.")
    else:
        answer.append("I don't know.")
        
    if "warning" in lower_prompt or "approval required" in lower_prompt or "high risk" in lower_prompt or "appr:1" in lower_prompt or "risk:high" in lower_prompt:
        answer.append("Warning: high risk or manager approval required.")
        
    return " ".join(answer)

def ollama_llm_call(prompt: str, model_id: str, base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-provider", default="mock")
    parser.add_argument("--model-id", default="qwen3-coder-next")
    parser.add_argument("--base-url", default="http://localhost:7777")
    parser.add_argument("--formats", default="1_baseline,2_layout_d,3_ultra_compact,4_yaml_lite,5_minified_json,6_markdown_table,7_toon_rows,8_source_table_facts")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--out", default="benchmarks/outputs/raw")
    parser.add_argument("--write-summary", action="store_true", default=True)
    parser.add_argument("--write-decision", action="store_true", default=True)
    args = parser.parse_args()

    formats = args.formats.split(',')
    cases = get_format_cases()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    # We need to compute Baseline token counts and latencies for comparisons
    baseline_stats = {'tokens': [], 'latencies': []}

    for case in cases:
        packet = EchoFramePacket.from_dict(case['packet'])
        for fmt in formats:
            rendered = render_format(packet, fmt)
            # Invariant check
            try:
                assert_format_preserves_packet(packet, rendered)
            except AssertionError as e:
                print(f"Safety invariant failed for format {fmt}: {e}")
                
            prompt = rendered + f"\n\nQuestion: {case['question']}"
            token_count = len(prompt.split())
            
            start_time = time.time()
            if args.model_provider == "ollama":
                answer = ollama_llm_call(prompt, args.model_id, args.base_url)
            else:
                answer = mock_llm_call(prompt, args.model_provider, case['family'])
                
            latency_ms = (time.time() - start_time) * 1000
            
            if fmt == '2_layout_d': # Use Layout D as baseline for comparison
                baseline_stats['tokens'].append(token_count)
                baseline_stats['latencies'].append(latency_ms)

            scores = score_format_answer(answer, case)
            scores['token_count'] = token_count
            scores['latency_ms'] = latency_ms
            scores['format_id'] = fmt
            
            results.append({
                "case_id": case['case_id'],
                "format": fmt,
                "scores": scores,
                "answer": answer
            })

    # compute summary
    format_scores = {f: [] for f in formats}
    format_tokens = {f: [] for f in formats}
    format_latencies = {f: [] for f in formats}
    
    for r in results:
        fmt = r['format']
        format_scores[fmt].append(r['scores']['format_quality_score'])
        format_tokens[fmt].append(r['scores']['token_count'])
        format_latencies[fmt].append(r['scores']['latency_ms'])
        
    avg_scores = {f: sum(s)/len(s) if s else 0 for f, s in format_scores.items()}
    avg_tokens = {f: sum(s)/len(s) if s else 0 for f, s in format_tokens.items()}
    avg_latencies = {f: sum(s)/len(s) if s else 0 for f, s in format_latencies.items()}
    
    baseline_avg_tokens = avg_tokens.get('2_layout_d', 1)
    baseline_avg_latencies = avg_latencies.get('2_layout_d', 1)
    baseline_avg_score = avg_scores.get('2_layout_d', 0)

    best_format = max(avg_scores, key=avg_scores.get) if avg_scores else None
    
    # Decision logic
    decision = "INSUFFICIENT_EVIDENCE"
    
    if len(results) > 0:
        # Check safety regressions across all formats
        # We find format safety passes if for a given format, across all cases, we don't have 0 scores in critical metrics
        # Actually, let's determine the decision based on best_format.
        
        # Verify best_format safety
        safety_passed = True
        best_format_results = [r for r in results if r['format'] == best_format]
        for r in best_format_results:
            sc = r['scores']
            if sc['source_attribution_correct'] == 0 or \
               sc['numeric_span_preserved'] == 0 or \
               sc['date_span_preserved'] == 0 or \
               sc['negation_preserved'] == 0 or \
               sc['exception_preserved'] == 0 or \
               sc['contradiction_acknowledged'] == 0 or \
               sc['evidence_gap_acknowledged'] == 0 or \
               sc['governance_warning_preserved'] == 0:
                safety_passed = False
                break
                
        if not safety_passed:
            decision = "FAIL_FORMAT_SAFETY_REGRESSION"
        else:
            best_format_tokens = avg_tokens.get(best_format, 0)
            best_format_latency = avg_latencies.get(best_format, 0)
            token_ratio = best_format_tokens / baseline_avg_tokens if baseline_avg_tokens else 1.0
            latency_ratio = best_format_latency / baseline_avg_latencies if baseline_avg_latencies else 1.0
            
            if best_format == '2_layout_d':
                decision = "KEEP_CURRENT_TAGGED_D"
            elif avg_scores[best_format] >= baseline_avg_score and token_ratio <= 1.10 and latency_ratio <= 1.20:
                decision = f"SHADOW_FORMAT_VARIANT_{best_format.upper()}"
            else:
                decision = "NO_CHANGE"

    report = {
        "phase": "12B",
        "benchmark": "echoframe_format",
        "timestamp": timestamp,
        "model_provider": args.model_provider,
        "formats": formats,
        "cases": [c['case_id'] for c in cases],
        "results": results,
        "summary": {
            "format_scores": avg_scores,
            "format_tokens": avg_tokens,
            "format_latencies": avg_latencies,
            "winner": best_format,
            "decision": decision
        }
    }

    raw_path = out_dir / f"echoframe_phase12b_{timestamp}_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Benchmark complete. Raw results written to {raw_path}")
    
    if args.write_summary:
        sum_dir = Path("benchmarks/outputs/summaries")
        sum_dir.mkdir(parents=True, exist_ok=True)
        summary_path = sum_dir / f"echoframe_phase12b_{timestamp}_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Phase 12B Format Benchmark Summary\n\nWinner: {best_format}\nScores: {avg_scores}\nDecision: {decision}\n")
            
    if args.write_decision:
        sum_dir = Path("benchmarks/outputs/summaries")
        sum_dir.mkdir(parents=True, exist_ok=True)
        decision_path = sum_dir / f"echoframe_phase12b_{timestamp}_decision.md"
        with open(decision_path, 'w') as f:
            f.write(f"# Phase 12B Decision Report\n\nDecision: {decision}\n")

if __name__ == "__main__":
    main()
