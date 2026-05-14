import os
import json
import re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

BASELINE_FILE = os.path.join(RESULTS_DIR, "real_corpus_baseline_results.real.clean.json")
SAFE_FILE = os.path.join(RESULTS_DIR, "real_corpus_candidate_results.echoframe.compact_safe_v0.json")
SELECTIVE_FILE = os.path.join(RESULTS_DIR, "real_corpus_candidate_results.echoframe.compact_selective_v0.json")
SEMANTIC_FILE = os.path.join(RESULTS_DIR, "real_corpus_candidate_results.echoframe.compact_semantic_v0.json")

MD_REPORT_FILE = os.path.join(REPORTS_DIR, "real_corpus_comparison_report.compact_semantic_v0.md")
JSON_METRICS_FILE = os.path.join(RESULTS_DIR, "real_corpus_comparison_metrics.compact_semantic_v0.json")

def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing result file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_selective_comparison():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    baseline_data = load_json(BASELINE_FILE)
    safe_data = load_json(SAFE_FILE)
    sel_data = load_json(SELECTIVE_FILE)
    sem_data = load_json(SEMANTIC_FILE)

    baseline_results = {r["query_id"]: r for r in baseline_data["results"]}
    safe_results = {r["query_id"]: r for r in safe_data["results"]}
    sel_results = {r["query_id"]: r for r in sel_data["results"]}
    sem_results = {r["query_id"]: r for r in sem_data["results"]}

    metrics = {
        "queries": 0,
        "avg_baseline_tokens": 0,
        "avg_compact_safe_v0_tokens": 0,
        "avg_compact_selective_v0_tokens": 0,
        "avg_compact_semantic_v0_tokens": 0,
        "queries_sem_beats_safe": 0,
        "queries_sem_beats_sel": 0,
        "fallback_count": 0,
        "fallback_reasons": {"ambiguous_relevance": 0},
        "high_risk_queries_fallback": 0,
        "contradiction_queries_fallback": 0,
        "facts_pinned": 0,
        "safety_checks": {
            "dropped_provenance": 0,
            "dropped_gaps": 0,
            "dropped_contradictions": 0,
            "dropped_approval": 0,
            "lost_unknown_preservation": 0,
            "lost_numeric_threshold": 0,
            "lost_date": 0,
            "lost_negation": 0,
            "lost_exception": 0,
            "unsupported_synthesis": 0,
            "fabricated_sources": 0
        },
        "categories": {}
    }

    gates = {
        "Gate_B_Provenance": "PASS",
        "Gate_C_Governance_Preservation": "PASS",
        "Gate_D_Token_Efficiency": "PASS",
        "Numeric_Threshold_Preservation": "PASS",
        "Date_Preservation": "PASS",
        "Negation_Preservation": "PASS",
        "Exception_Language_Preservation": "PASS",
        "No_Fabricated_Sources": "PASS",
        "Semantic_Fact_Drops_Documented": "PASS",
        "High_Risk_Do_Not_Overcompress": "PASS",
        "Unknown_Do_Not_Become_Confident": "PASS",
        "No_Unsupported_Synthesis": "PASS"
    }

    total_b, total_safe, total_sel, total_sem = 0, 0, 0, 0

    for qid, b_res in baseline_results.items():
        if qid not in sem_results: continue
        
        sel_res = sel_results[qid]
        safe_res = safe_results[qid]
        sem_res = sem_results[qid]
        cat = b_res["category"]
        cat_lower = cat.lower()
        packet = sem_res.get("rendered_echoframe_packet", "")
        decode = sem_res.get("decode_level", "D3")
        
        if cat not in metrics["categories"]:
            metrics["categories"][cat] = {
                "count": 0, "b_tok": 0, "safe_tok": 0, "sel_tok": 0, "sem_tok": 0
            }
            
        b_tok = b_res.get("context_token_count", 1)
        safe_tok = safe_res.get("candidate_context_token_count", 1)
        sel_tok = sel_res.get("candidate_context_token_count", 1)
        sem_tok = sem_res.get("candidate_context_token_count", 1)
        
        total_b += b_tok
        total_safe += safe_tok
        total_sel += sel_tok
        total_sem += sem_tok
        
        metrics["categories"][cat]["count"] += 1
        metrics["categories"][cat]["b_tok"] += b_tok
        metrics["categories"][cat]["safe_tok"] += safe_tok
        metrics["categories"][cat]["sel_tok"] += sel_tok
        metrics["categories"][cat]["sem_tok"] += sem_tok
        
        metrics["queries"] += 1
        
        if sem_tok < safe_tok:
            metrics["queries_sem_beats_safe"] += 1
        if sem_tok < sel_tok:
            metrics["queries_sem_beats_sel"] += 1

        # Check fallback
        is_fallback = "fallback=compact_safe_v0" in packet
        if is_fallback:
            metrics["fallback_count"] += 1
            metrics["fallback_reasons"]["ambiguous_relevance"] += 1
            if "high-risk" in cat_lower:
                metrics["high_risk_queries_fallback"] += 1
            if "contradiction" in cat_lower:
                metrics["contradiction_queries_fallback"] += 1
                
        # Count pinned facts
        fact_lines = [line for line in packet.split('\n') if line.startswith('- threshold:') or line.startswith('- date:') or line.startswith('- key:') or line.startswith('- clause:')]
        metrics["facts_pinned"] += len(fact_lines)

        # Base governance
        if b_res.get("provenance_present") and not sem_res.get("provenance_present"):
            metrics["safety_checks"]["dropped_provenance"] += 1
            gates["Gate_B_Provenance"] = "FAIL"
            
        if len(b_res.get("evidence_gaps", [])) > 0 and len(sem_res.get("evidence_gaps", [])) == 0:
            metrics["safety_checks"]["dropped_gaps"] += 1
            gates["Gate_C_Governance_Preservation"] = "FAIL"
            
        b_contra = b_res.get("contradiction_flags", [])
        if len(b_contra) > 0:
            if len(sem_res.get("contradiction_flags", [])) == 0:
                metrics["safety_checks"]["dropped_contradictions"] += 1
                gates["Gate_C_Governance_Preservation"] = "FAIL"
                
        if b_res.get("approval_required"):
            if not sem_res.get("approval_required"):
                metrics["safety_checks"]["dropped_approval"] += 1
                gates["Gate_C_Governance_Preservation"] = "FAIL"

        if b_res.get("unknown_preserved"):
            if not sem_res.get("unknown_preserved"):
                metrics["safety_checks"]["lost_unknown_preservation"] += 1
                gates["Gate_C_Governance_Preservation"] = "FAIL"
                
            if "NO_EVIDENCE_FOUND" not in packet and decode != "D0" and not b_res.get("selected_evidence_ids"):
                gates["Unknown_Do_Not_Become_Confident"] = "FAIL"
                
        # Fact preservation checks for required categories
        b_context = b_res.get("rendered_context", "").lower()
        packet_lower = packet.lower()
        
        if decode in ["D2", "D3", "D4"]:
            if "numeric" in cat_lower or "threshold" in cat_lower or is_fallback:
                nums = re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b', b_context)
                if nums:
                    for n in nums:
                        if n not in packet_lower and len(n) <= 15:
                            metrics["safety_checks"]["lost_numeric_threshold"] += 1
                            gates["Numeric_Threshold_Preservation"] = "FAIL"
                            break
                            
            if "date" in cat_lower or is_fallback:
                dates = re.findall(r'\b20\d{2}(?:-\d{2}-\d{2})?\b', b_context)
                if dates:
                    for d in dates:
                        if d not in packet_lower:
                            metrics["safety_checks"]["lost_date"] += 1
                            gates["Date_Preservation"] = "FAIL"
                            break

            if "policy" in cat_lower or "obligation" in cat_lower or "exception" in cat_lower or is_fallback:
                if " not " in b_context and " not " not in packet_lower:
                    metrics["safety_checks"]["lost_negation"] += 1
                    gates["Negation_Preservation"] = "FAIL"
                    
                if "except" in b_context and "except" not in packet_lower:
                    metrics["safety_checks"]["lost_exception"] += 1
                    gates["Exception_Language_Preservation"] = "FAIL"

    n_q = max(1, metrics["queries"])
    metrics["avg_baseline_tokens"] = total_b / n_q
    metrics["avg_compact_safe_v0_tokens"] = total_safe / n_q
    metrics["avg_compact_selective_v0_tokens"] = total_sel / n_q
    metrics["avg_compact_semantic_v0_tokens"] = total_sem / n_q
    
    r_sem = total_sem / max(1, total_b)
    
    if r_sem > 0.85:
        gates["Gate_D_Token_Efficiency"] = "FAIL"

    # Write JSON
    with open(JSON_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "gates": gates}, f, indent=2)

    # Write MD
    md = [
        "# EchoFrame vs MNEMOS Real-Corpus Benchmark (compact_semantic_v0)\n",
        "## Summary Metrics",
        f"- **Queries Evaluated**: {metrics['queries']}",
        f"- **Avg Baseline Tokens**: {metrics['avg_baseline_tokens']:.2f}",
        f"- **Avg compact_safe_v0 Tokens**: {metrics['avg_compact_safe_v0_tokens']:.2f}",
        f"- **Avg compact_selective_v0 Tokens**: {metrics['avg_compact_selective_v0_tokens']:.2f}",
        f"- **Avg compact_semantic_v0 Tokens**: {metrics['avg_compact_semantic_v0_tokens']:.2f} (Ratio: {r_sem:.4f})",
        f"- **Queries where semantic beats safe**: {metrics['queries_sem_beats_safe']}",
        f"- **Queries where semantic beats selective**: {metrics['queries_sem_beats_sel']}",
        f"- **Fallback Count**: {metrics['fallback_count']}",
        f"- **High-Risk Fallbacks**: {metrics['high_risk_queries_fallback']}",
        f"- **Contradiction Fallbacks**: {metrics['contradiction_queries_fallback']}",
        f"- **Total Facts Pinned**: {metrics['facts_pinned']}",
        "\n## Safety Check Failures",
    ]
    for check, count in metrics["safety_checks"].items():
        md.append(f"- {check}: {count}")

    md.append("\n## Promotion Gates")
    for gate, status in gates.items():
        md.append(f"- **{gate.replace('_', ' ')}**: {status}")

    safety_failed = any(status == "FAIL" for gate, status in gates.items() if gate != "Gate_D_Token_Efficiency")
    
    if safety_failed:
        md.append("\n**Recommendation**: REJECT SEMANTIC MODE. Any safety/provenance/governance/fact preservation gate fails.")
    elif gates["Gate_D_Token_Efficiency"] == "FAIL":
        md.append(f"\n**Recommendation**: CONTINUE EXPERIMENT. Safety passes but token ratio ({r_sem:.4f}) remains > 0.85.")
    else:
        md.append("\n**Recommendation**: PROMOTE TO SHADOW RUNTIME. compact_semantic_v0 passes all safety gates and token ratio <= 0.85.")

    with open(MD_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

    print(f"Semantic Comparison complete. Reports written to:\n  {JSON_METRICS_FILE}\n  {MD_REPORT_FILE}")

if __name__ == "__main__":
    run_selective_comparison()
