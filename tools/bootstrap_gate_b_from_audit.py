"""
Bootstrap Gate B sanity files from a Track 2 raw benchmark artifact.

Usage:
  python tools/bootstrap_gate_b_from_audit.py --raw benchmarks/outputs/raw/20260329_170925_profile_benchmarks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_union(trace: Dict[str, Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for key in ("baseline_top10", "colbert_top10", "overlap_top10", "mean_cosine_top10"):
        for did in trace.get(key, []):
            if isinstance(did, str) and did not in seen:
                seen.add(did)
                out.append(did)
    return out


def _extract_traces(raw: Dict[str, Any], tier: str) -> List[Dict[str, Any]]:
    try:
        return raw["results"]["rerank"]["tiers"][tier]["audit"]["traces"]
    except Exception:
        return []


def _build_gate_b_files(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    pools: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []

    for i, t in enumerate(traces, start=1):
        qid = t.get("query_id")
        qtext = t.get("query_text", "")
        if not isinstance(qid, str) or not qid.strip():
            qid = f"gateb_auto_{i:03d}"

        candidate_ids = _candidate_union(t)
        gold = set(t.get("gold_ids", []))

        queries.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "regime": "semantic",
            }
        )
        pools.append(
            {
                "query_id": qid,
                "candidate_ids": candidate_ids,
            }
        )
        graded = []
        for did in candidate_ids:
            graded.append(
                {
                    "doc_id": did,
                    "relevance": 2 if did in gold else 0,
                }
            )
        labels.append(
            {
                "query_id": qid,
                "graded_labels": graded,
            }
        )

    return {
        "queries": {
            "version": "gate_b_v1_bootstrapped",
            "description": "Bootstrapped from Track 2 audit traces; review and refine before final Gate B runs.",
            "queries": queries,
        },
        "pool": {
            "version": "gate_b_v1_bootstrapped",
            "description": "Union of baseline/colbert/overlap/mean-cosine top-10 from audit traces.",
            "pools": pools,
        },
        "labels": {
            "version": "gate_b_v1_bootstrapped",
            "description": "Bootstrapped binary labels from audit gold_ids (2 for gold, 0 otherwise).",
            "label_scale": {
                "0": "not relevant",
                "1": "partially relevant",
                "2": "highly relevant",
            },
            "labels": labels,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Gate B truthset files from Track 2 audit raw output.")
    parser.add_argument(
        "--raw",
        type=str,
        required=True,
        help="Path to raw profile benchmark JSON containing Track 2 audit traces.",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="qdrant",
        choices=["qdrant", "pgvector"],
        help="Tier to bootstrap from (default: qdrant).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of audit queries to export (default: 10).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmarks/truthsets",
        help="Output directory (default: benchmarks/truthsets).",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"[ERROR] Raw file not found: {raw_path}")
        return 2

    raw = _load(raw_path)
    traces = _extract_traces(raw, args.tier)
    if not traces:
        print(f"[ERROR] No audit traces found for tier '{args.tier}' in {raw_path}")
        return 2

    traces = traces[: max(1, args.limit)]
    files = _build_gate_b_files(traces)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_path = out_dir / "gate_b_sanity_queries.json"
    p_path = out_dir / "gate_b_candidate_pool.json"
    l_path = out_dir / "gate_b_labels.json"

    q_path.write_text(json.dumps(files["queries"], indent=2) + "\n", encoding="utf-8")
    p_path.write_text(json.dumps(files["pool"], indent=2) + "\n", encoding="utf-8")
    l_path.write_text(json.dumps(files["labels"], indent=2) + "\n", encoding="utf-8")

    print(f"[OK] Bootstrapped Gate B files from {raw_path.name} ({args.tier}, {len(traces)} queries)")
    print(f"  - {q_path}")
    print(f"  - {p_path}")
    print(f"  - {l_path}")
    print("  Review labels and candidate pools before final Gate B reference-fidelity run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

