import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline_results.real.clean.json")
CANDIDATE_JSON_FILE = os.path.join(RESULTS_DIR, "candidate_results.echoframe.json_v0.json")
CANDIDATE_COMPACT_FILE = os.path.join(RESULTS_DIR, "candidate_results.echoframe.compact_v0.json")

MD_REPORT_FILE = os.path.join(REPORTS_DIR, "echoframe_comparison_report.compact_v0.md")
JSON_METRICS_FILE = os.path.join(RESULTS_DIR, "echoframe_comparison_metrics.compact_v0.json")

def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing result file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    baseline_data = load_json(BASELINE_FILE)
    json_data = load_json(CANDIDATE_JSON_FILE)
    compact_data = load_json(CANDIDATE_COMPACT_FILE)

    baseline_results = {r["query_id"]: r for r in baseline_data["results"]}
    json_results = {r["query_id"]: r for r in json_data["results"]}
    compact_results = {r["query_id"]: r for r in compact_data["results"]}

    metrics = {
        "total_queries": len(baseline_results),
        "total_baseline_tokens": 0,
        "total_json_tokens": 0,
        "total_compact_tokens": 0,
        "queries_compact_smaller_than_baseline": 0,
        "queries_compact_larger_than_baseline": 0,
        "queries_compact_smaller_than_json": 0,
        "queries_with_dropped_provenance": 0,
        "queries_with_dropped_gaps": 0,
        "queries_with_dropped_contradictions": 0,
        "queries_with_dropped_approval": 0,
        "queries_with_lost_unknown_preservation": 0,
        "queries_with_invalid_packet": 0,
        "categories": {}
    }

    gates = {
        "Gate_A_Fidelity": "N/A",
        "Gate_B_Provenance": "PASS",
        "Gate_C_Governance_Preservation": "PASS",
        "Gate_D_Token_Efficiency": "PASS",
        "Gate_E_Context_Stability": "N/A",
        "Gate_F_Explainability": "PASS"
    }

    for qid, b_res in baseline_results.items():
        if qid not in compact_results:
            continue
        c_res = compact_results[qid]
        j_res = json_results.get(qid, {})
        cat = b_res["category"]
        
        if cat not in metrics["categories"]:
            metrics["categories"][cat] = {
                "count": 0, "baseline_tokens": 0, "json_tokens": 0, "compact_tokens": 0,
                "dropped_signals": 0
            }

        metrics["categories"][cat]["count"] += 1
        
        b_tok = b_res.get("context_token_count", 0)
        j_tok = j_res.get("candidate_context_token_count", 0)
        c_tok = c_res.get("candidate_context_token_count", 0)
        
        metrics["total_baseline_tokens"] += b_tok
        metrics["total_json_tokens"] += j_tok
        metrics["total_compact_tokens"] += c_tok
        metrics["categories"][cat]["baseline_tokens"] += b_tok
        metrics["categories"][cat]["json_tokens"] += j_tok
        metrics["categories"][cat]["compact_tokens"] += c_tok

        if c_tok < b_tok:
            metrics["queries_compact_smaller_than_baseline"] += 1
        elif c_tok > b_tok:
            metrics["queries_compact_larger_than_baseline"] += 1
            
        if c_tok < j_tok:
            metrics["queries_compact_smaller_than_json"] += 1

        # Provenance check (must check compact mode to satisfy gate)
        if b_res.get("provenance_present") and not c_res.get("provenance_present"):
            metrics["queries_with_dropped_provenance"] += 1
            metrics["categories"][cat]["dropped_signals"] += 1
            gates["Gate_B_Provenance"] = "FAIL"

        # Evidence Gaps check
        b_gaps = b_res.get("evidence_gaps", [])
        c_gaps = c_res.get("evidence_gaps", [])
        if len(b_gaps) > 0 and len(c_gaps) < len(b_gaps):
            metrics["queries_with_dropped_gaps"] += 1
            metrics["categories"][cat]["dropped_signals"] += 1
            gates["Gate_C_Governance_Preservation"] = "FAIL"

        # Contradictions check
        b_contra = b_res.get("contradiction_flags", [])
        c_contra = c_res.get("contradiction_flags", [])
        if len(b_contra) > 0 and len(c_contra) < len(b_contra):
            metrics["queries_with_dropped_contradictions"] += 1
            metrics["categories"][cat]["dropped_signals"] += 1
            gates["Gate_C_Governance_Preservation"] = "FAIL"

        # Approval Required check
        if b_res.get("approval_required") and not c_res.get("approval_required"):
            metrics["queries_with_dropped_approval"] += 1
            metrics["categories"][cat]["dropped_signals"] += 1
            gates["Gate_C_Governance_Preservation"] = "FAIL"

        # Unknown Preservation check
        if b_res.get("unknown_preserved") and not c_res.get("unknown_preserved"):
            metrics["queries_with_lost_unknown_preservation"] += 1
            metrics["categories"][cat]["dropped_signals"] += 1
            gates["Gate_C_Governance_Preservation"] = "FAIL"

        # Packet validation
        packet = c_res.get("rendered_echoframe_packet", "")
        if not packet or len(packet.strip()) == 0:
            metrics["queries_with_invalid_packet"] += 1
            gates["Gate_F_Explainability"] = "FAIL"
            
        # Additional check: mapping claims back to source
        if "SRC:" not in packet and b_res.get("selected_evidence_ids"):
            metrics["queries_with_invalid_packet"] += 1
            gates["Gate_F_Explainability"] = "FAIL"

    # Averages
    t_queries = max(1, metrics["total_queries"])
    metrics["average_baseline_token_count"] = metrics["total_baseline_tokens"] / t_queries
    metrics["average_json_token_count"] = metrics["total_json_tokens"] / t_queries
    metrics["average_compact_token_count"] = metrics["total_compact_tokens"] / t_queries
    
    metrics["json_token_reduction_ratio"] = metrics["total_json_tokens"] / max(1, metrics["total_baseline_tokens"])
    metrics["compact_token_reduction_ratio"] = metrics["total_compact_tokens"] / max(1, metrics["total_baseline_tokens"])
    
    if metrics["compact_token_reduction_ratio"] >= 1.0:
        gates["Gate_D_Token_Efficiency"] = "FAIL"

    for cat, cdata in metrics["categories"].items():
        cdata["avg_baseline_tokens"] = cdata["baseline_tokens"] / max(1, cdata["count"])
        cdata["avg_compact_tokens"] = cdata["compact_tokens"] / max(1, cdata["count"])
        cdata["reduction_ratio"] = cdata["compact_tokens"] / max(1, cdata["baseline_tokens"])

    # Write JSON
    with open(JSON_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "gates": gates}, f, indent=2)

    # Write Markdown
    md_lines = [
        "# EchoFrame vs MNEMOS Baseline Comparison Report (compact_v0)\n",
        "## Summary Metrics",
        f"- **Total Queries Evaluated**: {metrics['total_queries']}",
        f"- **Avg Baseline Tokens**: {metrics['average_baseline_token_count']:.2f}",
        f"- **Avg JSON_v0 Tokens**: {metrics['average_json_token_count']:.2f}",
        f"- **Avg Compact_v0 Tokens**: {metrics['average_compact_token_count']:.2f}",
        f"- **JSON_v0 Token Reduction Ratio**: {metrics['json_token_reduction_ratio']:.4f}",
        f"- **Compact_v0 Token Reduction Ratio**: {metrics['compact_token_reduction_ratio']:.4f}",
        f"- **Queries where compact < baseline**: {metrics['queries_compact_smaller_than_baseline']}",
        f"- **Queries where compact > baseline**: {metrics['queries_compact_larger_than_baseline']}",
        f"- **Queries where compact < json_v0**: {metrics['queries_compact_smaller_than_json']}",
        "\n## Governance & Provenance Violations (compact_v0)",
        f"- Dropped Provenance: {metrics['queries_with_dropped_provenance']}",
        f"- Dropped Evidence Gaps: {metrics['queries_with_dropped_gaps']}",
        f"- Dropped Contradictions: {metrics['queries_with_dropped_contradictions']}",
        f"- Dropped Approval Requirements: {metrics['queries_with_dropped_approval']}",
        f"- Lost Unknown Preservation: {metrics['queries_with_lost_unknown_preservation']}",
        f"- Invalid/Empty Packets: {metrics['queries_with_invalid_packet']}",
        "\n## Promotion Gates",
    ]
    
    for gate, status in gates.items():
        md_lines.append(f"- **{gate.replace('_', ' ')}**: {status}")

    if gates["Gate_D_Token_Efficiency"] == "FAIL" and gates["Gate_C_Governance_Preservation"] == "PASS":
        md_lines.append("\n**Conclusion:** EchoFrame is safe under this prototype but not beneficial for short-context retrieval workloads.")

    md_lines.append("\n## Category Breakdown (compact_v0 vs baseline)")
    for cat, cdata in metrics["categories"].items():
        md_lines.append(f"### {cat}")
        md_lines.append(f"- Count: {cdata['count']}")
        md_lines.append(f"- Reduction Ratio: {cdata['reduction_ratio']:.4f}")
        md_lines.append(f"- Dropped Signals: {cdata['dropped_signals']}")

    with open(MD_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print(f"Comparison complete. Reports written to:\n  {JSON_METRICS_FILE}\n  {MD_REPORT_FILE}")

if __name__ == "__main__":
    compare()
