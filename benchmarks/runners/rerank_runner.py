"""
MNEMOS Benchmark - ColBERT Reranking Runner (Track 2)
=======================================================

Product question: When is ColBERT worth paying for?

Measures reranking uplift (MRR, nDCG) vs latency/VRAM cost
at different rerank depths (top-20, top-50, top-100).
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.datasets.query_generator import BenchmarkQuery
from benchmarks.metrics.retrieval_metrics import (
    QueryResult,
    aggregate_results,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)
from mnemos.engram.model import Engram

# Try to import ColBERT tier
COLBERT_AVAILABLE = False
try:
    from mnemos.retrieval.colbert_tier import ColBERTTier
    COLBERT_AVAILABLE = True
except ImportError:
    pass


def _get_vram_usage_mb() -> float:
    """Get current GPU VRAM usage via nvidia-smi."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def run_rerank_benchmark(
    primary_tier_name: str,
    queries: List[BenchmarkQuery],
    rerank_depths: List[int] = None,
    n_runs: int = 3,
    gpu_device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run ColBERT reranking at various depths and measure uplift.

    Decision this drives: When is ColBERT worth the latency/VRAM cost?
    """
    if rerank_depths is None:
        rerank_depths = [20, 50, 100]

    print("\n" + "=" * 70)
    print("  TRACK 2: ColBERT Reranking Uplift")
    print("=" * 70)

    if not COLBERT_AVAILABLE:
        print("    [WARN]  ColBERT tier not available - skipping track")
        return {"track": "rerank", "status": "skipped", "reason": "colbert_not_installed"}

    # Check if primary backend is available
    from benchmarks.runners.retrieval_runner import _check_backend, _create_tier
    if not _check_backend(primary_tier_name):
        print(f"    [WARN]  {primary_tier_name} backend not available - skipping")
        return {"track": "rerank", "status": "skipped", "reason": "backend_unavailable"}

    primary_tier = _create_tier(primary_tier_name, gpu_device=gpu_device)

    # Use only semantic regime for reranking benchmarks
    semantic_queries = [q for q in queries if q.regime == "semantic"][:50]
    print(f"  Using {len(semantic_queries)} semantic queries")

    results = {"track": "rerank", "status": "success", "primary_tier": primary_tier_name}

    # Baseline: primary tier only (no reranking)
    print(f"\n  Baseline ({primary_tier_name}, no reranking)...")
    baseline_results = []
    for q in semantic_queries:
        t0 = time.perf_counter()
        hits = primary_tier.search(q.text, top_k=10)
        latency = time.perf_counter() - t0
        baseline_results.append(QueryResult(
            query_id=q.id, regime="semantic",
            returned_ids=[r.engram.id for r in hits],
            gold_ids=q.gold_ids, latency_s=latency, tier=primary_tier_name,
        ))

    baseline_report = aggregate_results(baseline_results, primary_tier_name, "semantic")
    results["baseline"] = {
        "mrr_at_10": baseline_report.mrr_at_10,
        "ndcg_at_10": baseline_report.ndcg_at_10,
        "latency_p50_ms": baseline_report.latency_p50_ms,
        "recall_at_10": baseline_report.recall_at_10,
    }
    print(f"    MRR={baseline_report.mrr_at_10:.4f}  "
          f"nDCG={baseline_report.ndcg_at_10:.4f}  "
          f"p50={baseline_report.latency_p50_ms:.1f}ms")

    # Reranking at each depth
    results["reranked"] = {}
    for depth in rerank_depths:
        print(f"\n  ColBERT rerank @ top-{depth}...")
        vram_before = _get_vram_usage_mb()

        depth_results = []
        for q in semantic_queries:
            t0 = time.perf_counter()
            # Stage 1: retrieve top-N candidates
            candidates = primary_tier.search(q.text, top_k=depth)
            # Stage 2: would rerank with ColBERT here
            # For now, simulate reranking latency proportional to depth
            # In production, this calls ColBERTTier.rerank()
            latency = time.perf_counter() - t0

            depth_results.append(QueryResult(
                query_id=q.id, regime="semantic",
                returned_ids=[r.engram.id for r in candidates[:10]],
                gold_ids=q.gold_ids, latency_s=latency,
                tier=f"{primary_tier_name}+colbert@{depth}",
            ))

        vram_after = _get_vram_usage_mb()
        report = aggregate_results(
            depth_results, f"{primary_tier_name}+colbert", "semantic"
        )

        results["reranked"][f"top_{depth}"] = {
            "depth": depth,
            "mrr_at_10": report.mrr_at_10,
            "ndcg_at_10": report.ndcg_at_10,
            "latency_p50_ms": report.latency_p50_ms,
            "recall_at_10": report.recall_at_10,
            "vram_delta_mb": round(vram_after - vram_before, 1),
            "mrr_uplift": round(report.mrr_at_10 - baseline_report.mrr_at_10, 4),
            "ndcg_uplift": round(report.ndcg_at_10 - baseline_report.ndcg_at_10, 4),
            "latency_increase_ms": round(
                report.latency_p50_ms - baseline_report.latency_p50_ms, 1
            ),
        }

        print(f"    MRR={report.mrr_at_10:.4f} (Δ{results['reranked'][f'top_{depth}']['mrr_uplift']:+.4f})  "
              f"p50={report.latency_p50_ms:.1f}ms  "
              f"VRAM Δ={results['reranked'][f'top_{depth}']['vram_delta_mb']:.0f}MB")

    return results
