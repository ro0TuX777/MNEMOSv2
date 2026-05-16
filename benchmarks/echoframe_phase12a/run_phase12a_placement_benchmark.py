import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from placement_layouts import EchoFramePacket, render_layout
from placement_cases import get_benchmark_cases
from placement_metrics import score_answer

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
    elif "no evidence" in lower_prompt or "gap" in lower_prompt:
        answer.append("Missing evidence. Gap.")
    else:
        answer.append("I don't know.")
        
    if "warning" in lower_prompt or "approval required" in lower_prompt or "high risk" in lower_prompt:
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
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--model-provider", default="mock")
    parser.add_argument("--model-id", default="qwen3-coder-next")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--layouts", default="A,B,C,D,E")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--out", default="benchmarks/outputs/raw")
    args = parser.parse_args()

    layouts = args.layouts.split(',')
    cases = get_benchmark_cases()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    for case in cases:
        packet = EchoFramePacket.from_dict(case['packet'])
        for layout in layouts:
            start_time = time.time()
            prompt = render_layout(packet, layout)
            prompt += f"\n\nQuestion: {case['question']}"
            
            token_count = len(prompt.split())
            
            if args.model_provider == "ollama":
                answer = ollama_llm_call(prompt, args.model_id, args.base_url)
            else:
                answer = mock_llm_call(prompt, args.model, case['family'])
                
            latency_ms = (time.time() - start_time) * 1000
            
            scores = score_answer(answer, case)
            scores['token_count'] = token_count
            scores['latency_ms'] = latency_ms
            scores['layout_id'] = layout
            scores['model_id'] = args.model
            
            results.append({
                "case_id": case['case_id'],
                "layout": layout,
                "scores": scores,
                "answer": answer
            })

    layout_scores = {l: [] for l in layouts}
    for r in results:
        layout_scores[r['layout']].append(r['scores']['placement_quality_score'])
        
    avg_scores = {l: sum(s)/len(s) if s else 0 for l, s in layout_scores.items()}
    best_layout = max(avg_scores, key=avg_scores.get) if avg_scores else None
    
    recommendation = "NO_CHANGE"
    if best_layout == 'D' and avg_scores[best_layout] > avg_scores.get('A', 0):
        recommendation = "SHADOW_LAYOUT_D"
    elif best_layout == 'E' and avg_scores[best_layout] > avg_scores.get('A', 0):
        recommendation = "SHADOW_LAYOUT_E"

    report = {
        "phase": "12A",
        "benchmark": "echoframe_evidence_placement",
        "timestamp": timestamp,
        "model": args.model,
        "layouts": layouts,
        "cases": [c['case_id'] for c in cases],
        "results": results,
        "summary": {
            "layout_scores": avg_scores,
            "winner": best_layout,
            "promotion_recommendation": recommendation
        }
    }

    raw_path = out_dir / f"echoframe_phase12a_{timestamp}_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Benchmark complete. Raw results written to {raw_path}")
    
    sum_dir = Path("benchmarks/outputs/summaries")
    sum_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = sum_dir / f"echoframe_phase12a_{timestamp}_summary.md"
    decision_path = sum_dir / f"echoframe_phase12a_{timestamp}_decision.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# Benchmark Summary\n\nWinner: {best_layout}\nScores: {avg_scores}\n")
        
    with open(decision_path, 'w') as f:
        f.write(f"# Decision Report\n\nDecision: {recommendation}\n")

if __name__ == "__main__":
    main()
