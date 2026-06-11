import os
import sys
import json
import argparse
from typing import Dict, Any

from mnemos.extraction.candidate_store import CandidateStore
from mnemos.extraction.promotion_engine import PromotionEngine

def get_engine(db_path: str) -> PromotionEngine:
    store = CandidateStore(db_path=db_path)
    return PromotionEngine(store, db_path=db_path)

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

def cmd_list_validated(args):
    engine = get_engine(args.db_path)
    validated_chains = engine.fetch_validated_facts()
    
    summary = []
    for chain in validated_chains:
        summary.append({
            "fact_id": chain["candidate_fact"]["fact_id"],
            "terminal_state": chain["conflict_metadata"]["terminal_lifecycle_state"],
            "statement": chain["candidate_fact"]["statement"],
            "source_engram_id": chain["source_engram_id"]
        })
    print_output(summary, args.json)

def cmd_export_chain(args):
    engine = get_engine(args.db_path)
    validated_chains = engine.fetch_validated_facts()
    
    for chain in validated_chains:
        if chain["candidate_fact"]["fact_id"] == args.fact_id:
            export_dir = os.path.join("data", "validated_audit_exports")
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, f"{args.fact_id}_validated_chain.json")
            with open(export_path, "w") as f:
                json.dump(chain, f, indent=2)
            
            out = {"status": "success", "file": export_path}
            print_output(out, args.json)
            return
            
    print(f"Validated fact '{args.fact_id}' not found. It may be masked or unpromoted.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SMC-6 Validated Fact Audit CLI")
    parser.add_argument("--db-path", default="data/mnemos_candidate_facts.db", help="Path to sqlite store")
    parser.add_argument("--json", action="store_true", help="Output machine readable JSON")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("list-validated")
    
    p_export = subparsers.add_parser("export-chain")
    p_export.add_argument("fact_id")
    
    args = parser.parse_args()
    
    if args.command == "list-validated":
        cmd_list_validated(args)
    elif args.command == "export-chain":
        cmd_export_chain(args)

if __name__ == "__main__":
    main()
