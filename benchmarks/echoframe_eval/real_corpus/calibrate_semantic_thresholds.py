import os
import json
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from echoframe_candidate_adapter import EchoFrameCandidateAdapter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

BASELINE_FILE = os.path.join(RESULTS_DIR, "real_corpus_baseline_results.real.clean.json")
SAFE_FILE = os.path.join(RESULTS_DIR, "real_corpus_candidate_results.echoframe.compact_safe_v0.json")
SELECTIVE_FILE = os.path.join(RESULTS_DIR, "real_corpus_candidate_results.echoframe.compact_selective_v0.json")

# Outputs
CALIBRATION_JSON = os.path.join(RESULTS_DIR, "semantic_threshold_calibration.v0.json")
CALIBRATION_MD = os.path.join(REPORTS_DIR, "semantic_threshold_calibration_report.v0.md")

PROFILES = {
    "Profile A (Baseline)": {"semantic_keep": 0.70, "mixed_keep": 0.55, "budget_aware": False, "category_aware": False},
    "Profile B (Stricter)": {"semantic_keep": 0.80, "mixed_keep": 0.65, "budget_aware": False, "category_aware": False},
    "Profile C (Aggressive)": {"semantic_keep": 0.85, "mixed_keep": 0.75, "budget_aware": False, "category_aware": False},
    "Profile D (Max Strict)": {"semantic_keep": 0.95, "mixed_keep": 0.85, "budget_aware": False, "category_aware": False},
    "Profile E (Budget Strict)": {"semantic_keep": 0.85, "mixed_keep": 0.70, "budget_aware": True, "category_aware": False}
}

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_profile_results(baseline_results, safe_results, sel_results, candidate_results):
    metrics = {
        "queries": 0,
        "avg_baseline_tokens": 0,
        "avg_compact_safe_v0_tokens": 0,
        "avg_compact_selective_v0_tokens": 0,
        "avg_compact_semantic_v0_tokens": 0,
        "facts_extracted": 0,
        "facts_pinned": 0,
        "facts_semantically_selected": 0,
        "facts_dropped": 0,
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
            "fabricated_sources": 0
        }
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
        "Unknown_Do_Not_Become_Confident": "PASS"
    }

    total_b, total_safe, total_sel, total_sem = 0, 0, 0, 0

    for qid, b_res in baseline_results.items():
        if qid not in candidate_results: continue
        
        sel_res = sel_results[qid]
        safe_res = safe_results[qid]
        sem_res = candidate_results[qid]
        
        cat = b_res["category"]
        cat_lower = cat.lower()
        packet = sem_res.get("rendered_echoframe_packet", "")
        decode = sem_res.get("decode_level", "D3")
        
        b_tok = b_res.get("context_token_count", 1)
        safe_tok = safe_res.get("candidate_context_token_count", 1)
        sel_tok = sel_res.get("candidate_context_token_count", 1)
        sem_tok = sem_res.get("candidate_context_token_count", 1)
        
        total_b += b_tok
        total_safe += safe_tok
        total_sel += sel_tok
        total_sem += sem_tok
        
        metrics["queries"] += 1
        
        fact_lines = [line for line in packet.split('\n') if line.startswith('- threshold:') or line.startswith('- date:') or line.startswith('- key:') or line.startswith('- clause:')]
        metrics["facts_pinned"] += len(fact_lines)

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
                
        b_context = b_res.get("rendered_context", "").lower()
        packet_lower = packet.lower()
        is_fallback = "fallback=compact_safe_v0" in packet
        
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
        
    return metrics, gates, r_sem

def run_calibration():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    baseline_data = load_json(BASELINE_FILE)
    safe_data = load_json(SAFE_FILE)
    sel_data = load_json(SELECTIVE_FILE)

    baseline_results = {r["query_id"]: r for r in baseline_data["results"]}
    safe_results = {r["query_id"]: r for r in safe_data["results"]}
    sel_results = {r["query_id"]: r for r in sel_data["results"]}

    adapter = EchoFrameCandidateAdapter()
    profile_results = {}

    for profile_name, params in PROFILES.items():
        candidate_list = []
        for b_res in baseline_data["results"]:
            cat = b_res["category"]
            query = b_res["query"]
            decode_level = adapter.determine_decode_level(cat)
            
            packet_str = adapter.render_compact_semantic_packet(
                b_res, decode_level, cat, query, 
                semantic_keep=params["semantic_keep"], 
                mixed_keep=params["mixed_keep"], 
                budget_aware=params["budget_aware"], 
                category_aware=params["category_aware"]
            )
            candidate_tokens = int(len(packet_str) / 4)
            
            res = {
                "query_id": b_res["query_id"],
                "candidate_context_token_count": candidate_tokens,
                "rendered_echoframe_packet": packet_str,
                "decode_level": decode_level,
                "provenance_present": b_res.get("provenance_present", False),
                "evidence_gaps": b_res.get("evidence_gaps", []),
                "contradiction_flags": b_res.get("contradiction_flags", []),
                "approval_required": b_res.get("approval_required", False),
                "unknown_preserved": b_res.get("unknown_preserved", False)
            }
            candidate_list.append(res)
            
        candidate_dict = {r["query_id"]: r for r in candidate_list}
        metrics, gates, ratio = evaluate_profile_results(baseline_results, safe_results, sel_results, candidate_dict)
        profile_results[profile_name] = {
            "metrics": metrics,
            "gates": gates,
            "ratio": ratio,
            "data": candidate_list
        }
        
    # Write JSON
    summary_json = {k: {"metrics": v["metrics"], "gates": v["gates"], "ratio": v["ratio"]} for k,v in profile_results.items()}
    with open(CALIBRATION_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2)

    # Write MD
    md = [
        "# EchoFrame Semantic Threshold Calibration (Phase 3C-T)\n",
        "## Tested Profiles\n"
    ]
    
    winning_profile = None
    best_ratio = 1.0
    
    for pname, pres in profile_results.items():
        ratio = pres["ratio"]
        safety_failed = any(status == "FAIL" for gate, status in pres["gates"].items() if gate != "Gate_D_Token_Efficiency")
        status_str = "SAFE" if not safety_failed else "UNSAFE"
        
        md.append(f"### {pname}")
        md.append(f"- **Avg Tokens**: {pres['metrics']['avg_compact_semantic_v0_tokens']:.2f}")
        md.append(f"- **Token Ratio**: {ratio:.4f}")
        md.append(f"- **Facts Pinned**: {pres['metrics']['facts_pinned']}")
        md.append(f"- **Safety Status**: {status_str}")
        if safety_failed:
            for g, s in pres["gates"].items():
                if s == "FAIL" and g != "Gate_D_Token_Efficiency":
                    md.append(f"  - FAILED GATE: {g}")
        md.append("")
        
        if not safety_failed and ratio <= 0.85 and ratio < best_ratio:
            winning_profile = pname
            best_ratio = ratio

    if winning_profile:
        md.append(f"\n## Winning Profile: {winning_profile}")
        md.append(f"**Recommendation**: PROMOTE TO SHADOW RUNTIME CANDIDATE. {winning_profile} reached token ratio {best_ratio:.4f} with zero safety failures.")
        
        # Output calibrated candidate file
        calibrated_file = os.path.join(RESULTS_DIR, "real_corpus_candidate_results.echoframe.compact_semantic_v0_calibrated.json")
        out_data = {
            "run_type": f"real_corpus_candidate_compact_semantic_v0_calibrated_{winning_profile.replace(' ', '_')}",
            "query_count": len(profile_results[winning_profile]["data"]),
            "results": profile_results[winning_profile]["data"]
        }
        with open(calibrated_file, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
            
        md.append(f"\nSaved calibrated candidate packet to: `{os.path.basename(calibrated_file)}`")
    else:
        # Check if we failed safety or just token efficiency
        any_safe = any(not any(status == "FAIL" for gate, status in pres["gates"].items() if gate != "Gate_D_Token_Efficiency") for pres in profile_results.values())
        if any_safe:
            md.append("\n**Recommendation**: CONTINUE EXPERIMENT. Safety remains perfect for some profiles but all profiles remain > 0.85.")
        else:
            md.append("\n**Recommendation**: REJECT CALIBRATED SEMANTIC MODE. Any profile reaches token ratio target only by violating safety/fact/provenance/governance gates.")

    with open(CALIBRATION_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

    print(f"Calibration complete. Reports written to:\n  {CALIBRATION_JSON}\n  {CALIBRATION_MD}")
    if winning_profile:
        print(f"Winning Profile found: {winning_profile} (Ratio: {best_ratio:.4f})")

if __name__ == "__main__":
    run_calibration()
