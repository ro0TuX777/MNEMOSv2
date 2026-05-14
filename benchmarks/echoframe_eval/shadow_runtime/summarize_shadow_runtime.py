import os
import json
import glob

def summarize():
    output_dir = os.environ.get("MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR", "runtime/echoframe_shadow/")
    
    # We should run from MNEMOS root ideally
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    target_dir = os.path.join(base_dir, output_dir)
    
    report_dir = os.path.join(target_dir, "reports")
    res_dir = os.path.join(target_dir, "results")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(target_dir, "shadow_event_*.json"))
    
    total_events = len(files)
    successful = 0
    failed = 0
    
    total_base = 0
    total_cand = 0
    total_stab = 0.0
    
    failures = []
    fallbacks = 0
    non_promotable = 0
    
    cases = []
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            successful += 1
            b_tok = data.get("baseline_token_count", 0)
            c_tok = data.get("echoframe_token_count", 0)
            s_fail = data.get("safety_gate_failures", [])
            fb = data.get("fallback_used", False)
            stab = data.get("stability_score", 1.0)
            
            total_base += b_tok
            total_cand += c_tok
            total_stab += stab
            
            if s_fail:
                failures.extend(s_fail)
                non_promotable += 1
            if fb: fallbacks += 1
            
            cases.append({
                "hash": data.get("query_hash", "???"),
                "base": b_tok,
                "cand": c_tok,
                "diff": b_tok - c_tok
            })
            
        except Exception:
            failed += 1
            
    avg_base = total_base / max(1, successful)
    avg_cand = total_cand / max(1, successful)
    avg_ratio = avg_cand / max(1, avg_base)
    avg_stab = total_stab / max(1, successful)
    
    cases.sort(key=lambda x: x["diff"], reverse=True)
    top_saves = cases[:3]
    worst_reg = cases[-3:] if len(cases) >= 3 else []
    
    metrics = {
        "total_shadow_events": total_events,
        "successful_shadow_events": successful,
        "failed_shadow_events": failed,
        "avg_baseline_tokens": avg_base,
        "avg_echoframe_tokens": avg_cand,
        "avg_token_ratio": avg_ratio,
        "avg_stability_score": avg_stab,
        "safety_gate_failures": list(set(failures)),
        "fallback_count": fallbacks,
        "non_promotable_count": non_promotable
    }
    
    md = [
        "# EchoFrame Shadow Runtime Summary",
        f"- **Total Events**: {total_events}",
        f"- **Successful**: {successful}",
        f"- **Failed**: {failed}",
        f"- **Non-Promotable**: {non_promotable}",
        "",
        "## Metrics",
        f"- **Avg Baseline Tokens**: {avg_base:.2f}",
        f"- **Avg EchoFrame Tokens**: {avg_cand:.2f}",
        f"- **Avg Token Ratio**: {avg_ratio:.4f}",
        f"- **Avg Stability Score**: {avg_stab:.4f}",
        "",
        "## Fallbacks and Safety",
        f"- **Fallbacks Used**: {fallbacks}",
        f"- **Unique Safety Failures**: {', '.join(set(failures)) if failures else 'None'}",
        "",
        "## Extremes",
        "**Top Token-Saving Cases:**"
    ]
    
    for c in top_saves: md.append(f"- {c['hash']}: {c['base']} -> {c['cand']} ({c['diff']} saved)")
    
    md.append("\n**Worst Token-Regression Cases:**")
    for c in reversed(worst_reg): md.append(f"- {c['hash']}: {c['base']} -> {c['cand']} ({-c['diff']} overhead)")

    with open(os.path.join(res_dir, "shadow_runtime_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    with open(os.path.join(report_dir, "shadow_runtime_summary.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
        
    print("Shadow runtime summary generated successfully.")

if __name__ == "__main__":
    summarize()
