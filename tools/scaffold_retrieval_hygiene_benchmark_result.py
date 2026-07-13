"""Scaffold a retrieval hygiene benchmark result file from the frozen pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mnemos_seed_manifest import DEFAULT_MANIFEST_PATH, load_seed_manifest

BENCHMARK_PATH = ROOT / "docs" / "experiments" / "retrieval_hygiene_r0_frozen_alias_benchmark.json"
TEMPLATE_PATH = ROOT / "benchmarks" / "results" / "retrieval_hygiene_r0_benchmark_result_template.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection-snapshot", default="replace_with_collection_snapshot")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    manifest = load_seed_manifest(Path(args.manifest_path))

    template["run_id"] = args.run_id
    template["benchmark_id"] = frozen["benchmark_id"]
    template["seed_snapshot_id"] = manifest.get("seed_snapshot_id", "unknown")
    template["collection_snapshot"] = args.collection_snapshot
    template["per_query_results"] = [
        {
            "query_id": item["query_id"],
            "query": item["query"],
            "expected_neighborhood": item["expected_neighborhood"],
            "evaluation_mode": item.get("evaluation_mode", "single_neighborhood"),
            "accepted_neighborhoods": item.get("accepted_neighborhoods", []),
            "runs": [],
        }
        for item in frozen.get("queries", [])
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
