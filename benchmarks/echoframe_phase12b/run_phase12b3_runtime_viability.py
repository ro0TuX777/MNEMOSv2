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

def get_wrappers():
    return {
        "wrapper_a_explicit": (
            "You must cite source IDs exactly as S1, S2, etc.\n"
            "If no evidence supports the answer, write EVIDENCE_GAP.\n"
            "Do not infer beyond the packet."
        ),
        "wrapper_b_structured": (
            "Return:\n"
            "ANSWER:\n"
            "SOURCES:\n"
            "GOVERNANCE:\n"
            "GAPS:\n\n"
            "SOURCES must include exact S# identifiers.\n"
            "If evidence is missing, GAPS must say EVIDENCE_GAP."
        )
    }

CANDIDATES = [
    ("2_layout_d", "wrapper_a_explicit"),
    ("8_source_table_facts", "wrapper_b_structured")
]

def ollama_llm_call(prompt: str, model_id: str, base_url: str, timeout: int) -> str:
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
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="hf.co/cloudbjorn/Qwen3.6-35B-A3B_Opus-4.6-Reasoning-3300x-GGUF:Q4_K_M")
    parser.add_argument("--base-url", default="http://localhost:7777")
    parser.add_argument("--out", default="benchmarks/outputs/raw")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    cases = get_format_cases()
    wrappers = get_wrappers()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    print(f"Running Phase 12B.3 Runtime Viability Gate.")

    for case in cases:
        packet = EchoFramePacket.from_dict(case['packet'])
        for fmt, wrapper_name in CANDIDATES:
            rendered = render_format(packet, fmt)
            wrapper_text = wrappers[wrapper_name]
            prompt = rendered + f"\n\nQuestion: {case['question']}\n\nInstructions:\n{wrapper_text}"
            
            start_time = time.time()
            answer = ollama_llm_call(prompt, args.model_id, args.base_url, args.timeout)
            latency_ms = (time.time() - start_time) * 1000
            
            is_timeout = answer == ""
            if not is_timeout:
                scores = score_format_answer(answer, case)
                logic_failure = any(v == 0 for k,v in scores.items())
            else:
                scores = {}
                logic_failure = False

            results.append({
                "case_id": case['case_id'],
                "format": fmt,
                "wrapper": wrapper_name,
                "scores": scores,
                "answer": answer,
                "latency_ms": latency_ms,
                "is_timeout": is_timeout,
                "logic_failure": logic_failure
            })

    # Group results
    summary = {}
    for fmt, wrap in CANDIDATES:
        key = f"{fmt} + {wrap}"
        summary[key] = {"total": 0, "timeouts": 0, "logic_failures": 0}

    for r in results:
        key = f"{r['format']} + {r['wrapper']}"
        summary[key]["total"] += 1
        if r["is_timeout"]:
            summary[key]["timeouts"] += 1
        elif r["logic_failure"]:
            summary[key]["logic_failures"] += 1

    decisions = {}
    for key, stats in summary.items():
        if stats["logic_failures"] > 0:
            decisions[key] = "SHADOW_BLOCK_LOGIC_FAILURES"
        elif stats["timeouts"] > 0:
            # Assuming ANY timeout blocks shadow admission
            decisions[key] = "SHADOW_BLOCK_TIMEOUTS"
        else:
            decisions[key] = "SHADOW_ADMIT"

    report = {
        "phase": "12B.3",
        "benchmark": "echoframe_runtime_viability_gate",
        "timestamp": timestamp,
        "results": results,
        "summary": summary,
        "decisions": decisions
    }

    raw_path = out_dir / f"echoframe_phase12b3_{timestamp}_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Benchmark complete. Raw results written to {raw_path}")
    print(json.dumps(summary, indent=2))
    print(json.dumps(decisions, indent=2))

    # Write Decision Report
    sum_dir = Path("benchmarks/outputs/summaries")
    sum_dir.mkdir(parents=True, exist_ok=True)
    decision_path = sum_dir / f"echoframe_phase12b3_{timestamp}_decision.md"
    with open(decision_path, 'w') as f:
        f.write("# Phase 12B.3 Decision Report\n\n")
        for key, dec in decisions.items():
            f.write(f"**{key}**: {dec}\n")
        f.write("\n## Summary\n```json\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n```\n")

if __name__ == "__main__":
    main()
