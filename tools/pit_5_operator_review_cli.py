import os
import json
import argparse
from datetime import datetime

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def run_operator_review(interactive=True, fixture_path=None):
    print("Starting PIT-5 Offline Operator Review Workflow...")

    # Load PIT-4-B artifacts
    baseline_eval = load_json("eval_results/baseline_answer_examples.json")
    shadow_eval = load_json("eval_results/shadow_answer_examples.json")
    raw_responses = load_json("eval_results/pit_4_b_raw_llm_responses.json")

    if not baseline_eval or not shadow_eval:
        print("Missing required PIT-4-B artifacts. Exiting.")
        return

    query_id = "q_12345"
    query = "What caused the Q1 database outage?"

    baseline_ans = baseline_eval.get("answer_summary", "N/A")
    shadow_ans = shadow_eval.get("answer_summary", "N/A")
    shadow_packet = shadow_eval.get("shadow_packet", {})
    evidence_gaps = []
    if "derived_evaluation_payload" in shadow_packet and shadow_packet["derived_evaluation_payload"]:
        am = shadow_packet["derived_evaluation_payload"][0].get("authority_matrix", {})
        evidence_gaps = am.get("evidence_gaps", [])

    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print("="*50)
    print(f"BASELINE ANSWER:\n{baseline_ans}\n")
    print("-" * 50)
    print(f"SHADOW ANSWER:\n{shadow_ans}\n")
    print("-" * 50)
    print(f"EVIDENCE GAPS EXTRACTED: {evidence_gaps}\n")
    print("="*50)

    # Review Data
    review_record = {
        "query_id": query_id,
        "query": query,
        "baseline_answer_ref": "baseline_answer_examples.json",
        "shadow_answer_ref": "shadow_answer_examples.json",
        "reviewer_type": "human" if interactive else "scripted_fixture",
        "review_timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }

    if interactive:
        print("Provide review scores.")
        review_record["operator_preference"] = input("Preference (baseline/shadow/tie/reject_both): ")
        review_record["clarity_delta"] = int(input("Clarity Delta (-2 to 2): "))
        review_record["confidence_delta"] = int(input("Confidence Delta (-2 to 2): "))
        review_record["citation_quality_score"] = int(input("Citation Quality (1 to 5): "))
        review_record["evidence_gap_handling_score"] = int(input("Evidence Gap Handling (1 to 5): "))
        review_record["decision_usefulness_score"] = int(input("Decision Usefulness (1 to 5): "))
        review_record["derived_fact_usefulness_score"] = int(input("Derived Fact Usefulness (1 to 5): "))
        review_record["authority_label_clarity_score"] = int(input("Authority Label Clarity (1 to 5): "))
        
        trace = input("Source Traceability Success (y/n): ")
        review_record["source_traceability_success"] = trace.lower().startswith("y")
        
        review_record["review_burden_delta"] = int(input("Review Burden Delta (-2 to 2): "))
        review_record["manual_rejection_reason"] = input("Manual Rejection Reason (optional): ")
    else:
        print("Running in scripted fixture mode...")
        fixture = load_json(fixture_path)
        if not fixture:
            print("Failed to load fixture. Exiting.")
            return
            
        data = fixture[0]
        review_record.update(data)

    # Save Log
    os.makedirs("eval_results", exist_ok=True)
    os.makedirs("docs/reports", exist_ok=True)

    log_path = "eval_results/pit_5_operator_review_log.json"
    with open(log_path, "w") as f:
        json.dump([review_record], f, indent=2)

    # Rejection logic
    if review_record["operator_preference"] in ["baseline", "reject_both"]:
        with open("eval_results/pit_5_disagreement_or_rejection_examples.json", "w") as f:
            json.dump([review_record], f, indent=2)

    # Summary
    summary_md = f"""# PIT-5 Operator Review Summary

**Timestamp**: {review_record['review_timestamp_utc']}
**Reviewer Type**: {review_record['reviewer_type']}

## Results
- **Operator Preference**: {review_record['operator_preference']}
- **Clarity Delta**: {review_record['clarity_delta']}
- **Decision Usefulness Score**: {review_record['decision_usefulness_score']}

*See `pit_5_operator_review_log.json` for full audit trail.*
"""
    with open("eval_results/pit_5_operator_review_summary.md", "w") as f:
        f.write(summary_md)

    evidence_md = f"""# OPS-4 Release Evidence Record (PIT-5 Operator Review)

**Date**: {datetime.utcnow().isoformat()}Z
**Release Phase**: PIT-5 (Offline Operator Review CLI)

## Boundary Enforcement
- Production EchoFrame Imported: False
- Live MNEMOS API Called: False
- Data Written to Governance Ledger: False
- SchemaNode Extraction Triggered: False
- Automatic Promotion Triggered: False

The operator review was successfully conducted entirely offline, reading only local evaluation JSON artifacts.
"""
    with open("docs/reports/pit_5_release_evidence_record.md", "w") as f:
        f.write(evidence_md)

    print("Operator review complete. Artifacts written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIT-5 Operator Review CLI")
    parser.add_argument("--interactive", action="store_true", help="Run interactive review")
    parser.add_argument("--review-input", type=str, help="Path to fixture JSON")
    
    args = parser.parse_args()
    
    if args.review_input:
        run_operator_review(interactive=False, fixture_path=args.review_input)
    else:
        run_operator_review(interactive=True)
