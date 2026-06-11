"""Run the Phase 9 hierarchy retrieval gate."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.retrieval.qdrant_tier import QdrantTier

DEFAULT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "query_complexity_v1.json"
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def _load_class_c(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["queries"] if row.get("label") == "CLASS_C"]
    missing = [row["id"] for row in rows if not row.get("golden_summary")]
    if missing:
        raise ValueError(f"CLASS_C rows missing golden_summary: {missing}")
    return rows


def evaluate(
    *,
    truthset: Path,
    qdrant_url: str,
    collection: str,
    embedding_model: str,
    gpu_device: str,
    top_k: int,
) -> Dict[str, Any]:
    tier = QdrantTier(
        url=qdrant_url,
        collection_name=collection,
        embedding_model=embedding_model,
        embedding_dim=768,
        gpu_device=gpu_device,
    )
    rows = []
    summary_latencies = []
    flat_latencies = []
    class_c = _load_class_c(truthset.resolve())

    for item in class_c:
        query = item["query"]
        t0 = time.perf_counter()
        summary_hits = tier.search(
            query,
            top_k=top_k,
            filters={"metadata.is_summary_engram": True},
        )
        summary_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        flat_hits = tier.search(query, top_k=max(10, top_k * 4), filters=None)
        flat_ms = (time.perf_counter() - t1) * 1000.0

        summary_latencies.append(summary_ms)
        flat_latencies.append(flat_ms)
        top_summary = summary_hits[0] if summary_hits else None
        if top_summary:
            summary_vec = np.asarray(tier._embed_query([top_summary.engram.content])[0], dtype=np.float32)
            golden_vec = np.asarray(tier._embed_query([item["golden_summary"]])[0], dtype=np.float32)
            similarity = _cosine(summary_vec, golden_vec)
            has_summary = bool(top_summary.engram.metadata.get("is_summary_engram"))
            top_id = top_summary.engram.id
        else:
            similarity = 0.0
            has_summary = False
            top_id = None
        rows.append(
            {
                "id": item["id"],
                "query": query,
                "top_result": top_id,
                "has_summary_engram": has_summary,
                "semantic_similarity": round(similarity, 4),
                "summary_latency_ms": round(summary_ms, 4),
                "flat_latency_ms": round(flat_ms, 4),
            }
        )

    coherence_pass = all(row["has_summary_engram"] for row in rows)
    similarity_pass = all(float(row["semantic_similarity"]) > 0.7 for row in rows)
    summary_p95 = _percentile(summary_latencies, 95)
    flat_p95 = _percentile(flat_latencies, 95)
    latency_pass = bool(flat_p95 > 0 and summary_p95 <= flat_p95 * 0.5)
    metrics = {
        "query_count": len(rows),
        "summary_hit_rate": round(sum(1 for row in rows if row["has_summary_engram"]) / len(rows), 4)
        if rows
        else 0.0,
        "mean_semantic_similarity": round(statistics.mean([float(row["semantic_similarity"]) for row in rows]), 4)
        if rows
        else 0.0,
        "summary_p95_ms": round(summary_p95, 4),
        "flat_p95_ms": round(flat_p95, 4),
        "gates": {
            "coherence": {"threshold": "all CLASS_C results include a summary engram", "pass": coherence_pass},
            "semantic_similarity": {"threshold": "each cosine similarity > 0.7", "pass": similarity_pass},
            "latency": {"threshold": "summary p95 <= 50% of flat p95", "pass": latency_pass},
        },
    }
    metrics["overall_gate_pass"] = all(gate["pass"] for gate in metrics["gates"].values())
    return {
        "truthset": str(truthset.resolve().relative_to(PROJECT_ROOT)),
        "collection": collection,
        "embedding_model": embedding_model,
        "rows": rows,
        "metrics": metrics,
    }


def write_artifacts(result: Dict[str, Any]) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"phase9_hierarchy_gate_{timestamp}_raw.json"
    summary_path = SUMMARY_DIR / f"phase9_hierarchy_gate_{timestamp}_summary.md"
    raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = result["metrics"]
    gate_lines = "\n".join(
        f"- {name}: {'PASS' if gate['pass'] else 'FAIL'} - {gate['threshold']}"
        for name, gate in metrics["gates"].items()
    )
    summary_path.write_text(
        "# Phase 9 Hierarchy Gate Summary\n\n"
        f"- Collection: `{result['collection']}`\n"
        f"- Truthset: `{result['truthset']}`\n"
        f"- Query count: `{metrics['query_count']}`\n"
        f"- Summary hit rate: `{metrics['summary_hit_rate']:.4f}`\n"
        f"- Mean semantic similarity: `{metrics['mean_semantic_similarity']:.4f}`\n"
        f"- Summary p95: `{metrics['summary_p95_ms']:.4f}ms`\n"
        f"- Flat p95: `{metrics['flat_p95_ms']:.4f}ms`\n"
        f"- Gate: **{'PASS' if metrics['overall_gate_pass'] else 'FAIL'}**\n\n"
        "## Gates\n\n"
        f"{gate_lines}\n\n"
        f"- Raw: `{raw_path.name}`\n",
        encoding="utf-8",
    )
    return raw_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 9 hierarchy retrieval gate")
    parser.add_argument("--truthset", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="mnemos_engrams_nomic_mrl")
    parser.add_argument("--embedding-model", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    result = evaluate(
        truthset=args.truthset,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        embedding_model=args.embedding_model,
        gpu_device=args.gpu_device,
        top_k=args.top_k,
    )
    metrics = result["metrics"]
    print(f"queries: {metrics['query_count']}")
    print(f"summary hit rate: {metrics['summary_hit_rate']:.4f}")
    print(f"mean similarity: {metrics['mean_semantic_similarity']:.4f}")
    print(f"summary p95: {metrics['summary_p95_ms']:.4f}ms")
    print(f"flat p95: {metrics['flat_p95_ms']:.4f}ms")
    print(f"gate: {'PASS' if metrics['overall_gate_pass'] else 'FAIL'}")
    if not args.no_artifacts:
        raw, summary = write_artifacts(result)
        print(f"raw: {raw}")
        print(f"summary: {summary}")
    return 0 if metrics["overall_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
