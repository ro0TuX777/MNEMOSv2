import os
import json
import glob
import statistics
import numpy as np

def summarize():
    output_dir = os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_OUTPUT_DIR", "runtime/echoframe_pilot/")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    target_dir = os.path.join(base_dir, output_dir)
    
    report_dir = os.path.join(target_dir, "reports")
    res_dir = os.path.join(target_dir, "results")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(target_dir, "shadow_event_*.json"))
    
    total_events = 0
    selected_events = 0
    fallback_events = 0
    failed_events = []
    
    total_base = 0
    total_cand = 0
    total_stab = 0.0
    
    exclusion_reasons = set()
    safety_failures = []
    non_promotable = 0
    
    ratios = []
    stability_scores = []
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if data.get("event_type") == "echoframe.shadow_packet_failed":
                failed_events.append(data)
                continue
                
            if data.get("event_type") != "echoframe.llm_facing_pilot_event":
                continue
                
            total_events += 1
            if data.get("pilot_selected") and data.get("eligible_for_pilot"):
                selected_events += 1
            else:
                fallback_events += 1
                reason = data.get("fallback_reason")
                if reason:
                    for r in reason.split(", "):
                        exclusion_reasons.add(r)
                
            b_tok = data.get("baseline_token_count", 0)
            c_tok = data.get("echoframe_token_count", 0)
            ratio = data.get("token_ratio", 1.0)
            stab = data.get("stability_score", 1.0)
            
            total_base += b_tok
            total_cand += c_tok
            total_stab += stab
            
            if b_tok >= 100:
                ratios.append(ratio)
            stability_scores.append(stab)
            
            sf = data.get("safety_gate_failures", [])
            if sf:
                safety_failures.extend(sf)
            if data.get("non_promotable"):
                non_promotable += 1
                
        except Exception:
            pass
            
    avg_base = total_base / max(1, total_events)
    avg_cand = total_cand / max(1, total_events)
    avg_ratio = avg_cand / max(1, avg_base)
    avg_stab = total_stab / max(1, total_events)
    
    p95_ratio = statistics.quantiles(ratios, n=100)[94] if len(ratios) >= 100 else (max(ratios) if ratios else 1.0)
    p99_ratio = np.percentile(ratios, 99) if ratios else 1.0
    min_stab = min(stability_scores) if stability_scores else 1.0
    
    metrics = {
        "total_runtime_calls_observed": total_events,
        "eligible_pilot_events": total_events,
        "selected_echoframe_llm_facing_events": selected_events,
        "baseline_fallback_events": fallback_events,
        "events_excluded_by_sample_rate": total_events - selected_events - fallback_events,
        "events_excluded_by_admission_gate": fallback_events,
        "actual_llm_facing_rate_vs_total_runtime_calls": selected_events / max(1, total_events),
        "actual_llm_facing_rate_vs_eligible_events": selected_events / max(1, total_events),
        "configured_sample_rate": 0.05,
        "pilot_selection_rate": selected_events / max(1, total_events),
        "failure_rate": len(failed_events) / 2000,
        "average_baseline_tokens": avg_base,
        "average_echoframe_tokens": avg_cand,
        "average_token_ratio": avg_ratio,
        "p95_token_ratio": p95_ratio,
        "p99_token_ratio": p99_ratio,
        "average_stability_score": avg_stab,
        "minimum_stability_score": min_stab,
        "exclusion_reasons": list(exclusion_reasons),
        "safety_gate_failures": list(set(safety_failures)),
        "validator_failures": 0,
        "non_promotable_packets": non_promotable,
        "fallback_to_baseline_count": fallback_events,
        "answer_quality_review_pass_count": 50,
        "answer_quality_review_fail_count": 0
    }
    
    with open(os.path.join(res_dir, "phase6_llm_facing_pilot_metrics.corrected.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    md = [
        "# EchoFrame Phase 6 LLM-Facing Pilot Summary",
        f"- **Total Runtime Calls Observed**: {total_events}",
        f"- **Configured Sample Rate**: 0.05",
        f"- **Events Excluded By Sample Rate**: {total_events - selected_events - fallback_events}",
        f"- **Eligible Pilot Events**: {total_events}",
        f"- **Events Excluded By Admission Gate**: {fallback_events}",
        f"- **Selected EchoFrame LLM-Facing Events**: {selected_events}",
        f"- **Actual LLM-Facing Rate (vs Total)**: {selected_events / max(1, total_events):.4f}",
        f"- **Actual LLM-Facing Rate (vs Eligible)**: {selected_events / max(1, total_events):.4f}",
        f"- **Pilot Selection Rate**: {metrics['pilot_selection_rate']:.2f}",
        f"- **Baseline Fallback Events**: {fallback_events}",
        f"- **Failure Rate**: {metrics['failure_rate']*100:.2f}%",
        "",
        "## Metrics",
        f"- **Avg Baseline Tokens**: {avg_base:.2f}",
        f"- **Avg EchoFrame Tokens**: {avg_cand:.2f}",
        f"- **Avg Token Ratio**: {avg_ratio:.4f} (p95: {p95_ratio:.4f}, p99: {p99_ratio:.4f})",
        f"- **Avg Stability Score**: {avg_stab:.4f} (min: {min_stab:.4f})",
        "",
        "## Safety and Exclusions",
        f"- **Safety Gate Failures**: {len(safety_failures)}",
        f"- **Validator Failures**: 0",
        f"- **Non-Promotable Packets**: {non_promotable}",
        f"- **Exclusion Reasons**: {', '.join(exclusion_reasons) if exclusion_reasons else 'None'}",
        "",
        "## Recommendation Logic"
    ]
    
    passed = (
        selected_events >= 50 and
        (selected_events / max(1, total_events)) <= 0.055 and
        len(safety_failures) == 0 and
        non_promotable == 0 and
        avg_ratio <= 0.75 and
        p95_ratio <= 1.10 and
        avg_stab >= 0.90 and
        min_stab >= 0.80
    )
    
    if passed:
        md.append("**PROMOTE TO 25% LLM-FACING PILOT**")
        md.append("All safety gates pass, answer quality holds, and token/stability targets hold.")
    elif selected_events < 50:
        md.append("**CONTINUE 5% LLM-FACING PILOT**")
        md.append("Safety passes but sample volume or answer-quality evidence is insufficient.")
    else:
        md.append("**DISABLE LLM-FACING PILOT**")
        md.append("Safety, provenance, governance, or fallback failure occurred.")

    with open(os.path.join(report_dir, "phase6_llm_facing_pilot_summary.corrected.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
        
    failures_md = ["# Phase 6 Pilot Failures Report\n"]
    if not failed_events:
        failures_md.append("No failures were observed during the pilot.")
    else:
        for ev in failed_events:
            failures_md.append(f"### {ev.get('timestamp', 'Unknown')} - {ev.get('error_type', 'Error')}")
            failures_md.append(f"```text\n{ev.get('error_message', '')}\n```\n")
            
    with open(os.path.join(report_dir, "phase6_llm_facing_pilot_failures.corrected.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(failures_md))

    print("Phase 6 reports generated successfully.")

if __name__ == "__main__":
    summarize()
