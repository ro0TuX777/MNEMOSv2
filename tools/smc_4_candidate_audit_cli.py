import os
import sys
import json
import argparse
from typing import Dict, Any

from mnemos.extraction.candidate_store import CandidateStore

def get_store(db_path: str) -> CandidateStore:
    return CandidateStore(db_path=db_path)

def print_output(data: Any, as_json: bool):
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        if isinstance(data, list):
            for i, item in enumerate(data):
                print(f"[{i+1}]")
                for k, v in item.items():
                    print(f"  {k}: {v}")
                print()
        elif isinstance(data, dict):
            for k, v in data.items():
                print(f"{k}: {v}")
        else:
            print(data)

def cmd_list(args):
    store = get_store(args.db_path)
    # The read-path explicitly requires include_candidate_facts=True to bypass the default 0-leakage
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    
    summary = []
    for payload in raw_facts:
        f = payload["fact_node"]
        summary.append({
            "fact_id": f["fact_id"],
            "batch_id": payload["manifest"]["batch_id"],
            "status": f["status"],
            "statement": f["statement"],
            "is_eligible": store._is_source_eligible(f["source_engram_id"])
        })
    print_output(summary, args.json)

def cmd_inspect(args):
    store = get_store(args.db_path)
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    for p in raw_facts:
        if p["fact_node"]["fact_id"] == args.fact_id:
            print_output(p, args.json)
            return
    print(f"Fact '{args.fact_id}' not found.", file=sys.stderr)
    sys.exit(1)

def cmd_receipt(args):
    store = get_store(args.db_path)
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    for p in raw_facts:
        if p["fact_node"]["fact_id"] == args.fact_id:
            print_output(p["receipt"], args.json)
            return
    print(f"Fact '{args.fact_id}' not found.", file=sys.stderr)
    sys.exit(1)

def cmd_review(args):
    store = get_store(args.db_path)
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    for p in raw_facts:
        if p["fact_node"]["fact_id"] == args.fact_id:
            print_output(p["review_label"], args.json)
            return
    print(f"Fact '{args.fact_id}' not found.", file=sys.stderr)
    sys.exit(1)

def cmd_manifest(args):
    store = get_store(args.db_path)
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    for p in raw_facts:
        if p["manifest"]["batch_id"] == args.batch_id:
            print_output(p["manifest"], args.json)
            return
    print(f"Manifest for batch '{args.batch_id}' not found.", file=sys.stderr)
    sys.exit(1)

def cmd_lineage(args):
    store = get_store(args.db_path)
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    for p in raw_facts:
        if p["fact_node"]["fact_id"] == args.fact_id:
            lineage = {
                "fact_id": p["fact_node"]["fact_id"],
                "source_engram_id": p["fact_node"]["source_engram_id"],
                "passage_node_id": p["fact_node"]["passage_node_id"],
                "parent_passage_receipt_id": p["fact_node"]["parent_passage_receipt_id"]
            }
            print_output(lineage, args.json)
            return
    print(f"Fact '{args.fact_id}' not found.", file=sys.stderr)
    sys.exit(1)

def cmd_masked(args):
    store = get_store(args.db_path)
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    for p in raw_facts:
        if p["fact_node"]["fact_id"] == args.fact_id:
            src_id = p["fact_node"]["source_engram_id"]
            is_eligible = store._is_source_eligible(src_id)
            state = {
                "fact_id": args.fact_id,
                "source_engram_id": src_id,
                "inherited_governance": p["fact_node"]["inherited_governance"],
                "is_masked": not is_eligible
            }
            print_output(state, args.json)
            return
    print(f"Fact '{args.fact_id}' not found.", file=sys.stderr)
    sys.exit(1)

def cmd_export(args):
    store = get_store(args.db_path)
    raw_facts = store.fetch_candidates(include_candidate_facts=True)
    for p in raw_facts:
        if p["fact_node"]["fact_id"] == args.fact_id:
            export_dir = os.path.join("data", "audit_exports")
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, f"{args.fact_id}_audit_bundle.json")
            with open(export_path, "w") as f:
                json.dump(p, f, indent=2)
            
            out = {"status": "success", "file": export_path}
            print_output(out, args.json)
            return
    print(f"Fact '{args.fact_id}' not found.", file=sys.stderr)
    sys.exit(1)

def cmd_rollback(args):
    store = get_store(args.db_path)
    
    # Validation
    dimensions = [
        ("batch_id", args.by_batch_id),
        ("extractor_version", args.by_extractor_version),
        ("source_engram_id", args.by_source_engram_id),
        ("review_batch_id", args.by_review_batch_id)
    ]
    
    selected = [d for d in dimensions if d[1] is not None]
    if len(selected) != 1:
        print("Error: Must specify exactly one rollback dimension.", file=sys.stderr)
        sys.exit(1)
        
    dim_name, dim_val = selected[0]
    
    deleted = store.rollback(dim_name, dim_val)
    out = {
        "status": "success",
        "dimension": dim_name,
        "value": dim_val,
        "records_removed": deleted
    }
    print_output(out, args.json)

def main():
    parser = argparse.ArgumentParser(description="SMC-4 Candidate Fact Audit CLI")
    parser.add_argument("--db-path", default="data/mnemos_candidate_facts.db", help="Path to sqlite store")
    parser.add_argument("--json", action="store_true", help="Output machine readable JSON")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("list")
    
    p_inspect = subparsers.add_parser("inspect")
    p_inspect.add_argument("fact_id")
    
    p_receipt = subparsers.add_parser("receipt")
    p_receipt.add_argument("fact_id")
    
    p_review = subparsers.add_parser("review")
    p_review.add_argument("fact_id")
    
    p_manifest = subparsers.add_parser("manifest")
    p_manifest.add_argument("batch_id")
    
    p_lineage = subparsers.add_parser("lineage")
    p_lineage.add_argument("fact_id")
    
    p_masked = subparsers.add_parser("masked")
    p_masked.add_argument("fact_id")
    
    p_export = subparsers.add_parser("export")
    p_export.add_argument("fact_id")
    
    p_rollback = subparsers.add_parser("rollback")
    p_rollback.add_argument("--by-batch-id")
    p_rollback.add_argument("--by-extractor-version")
    p_rollback.add_argument("--by-source-engram-id")
    p_rollback.add_argument("--by-review-batch-id")
    
    args = parser.parse_args()
    
    if args.command == "list":
        cmd_list(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "receipt":
        cmd_receipt(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "manifest":
        cmd_manifest(args)
    elif args.command == "lineage":
        cmd_lineage(args)
    elif args.command == "masked":
        cmd_masked(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "rollback":
        cmd_rollback(args)

if __name__ == "__main__":
    main()
