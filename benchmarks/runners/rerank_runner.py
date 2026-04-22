"""
MNEMOS Benchmark - Cross-Encoder Reranking Runner (Track 2)
======================================================

Product question: When is the Cross-Encoder worth the latency and VRAM cost?

Compares baseline retrieval vs Cross-Encoder candidate reranking at
depths 20/50/100 for both primary tiers (Qdrant and pgvector).
"""

import time
from typing import Any, Dict, List

import numpy as np

from benchmarks.datasets.query_generator import BenchmarkQuery
from benchmarks.metrics.retrieval_metrics import QueryResult, aggregate_results
from mnemos.engram.model import Engram


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


def _run_single_tier_rerank(
    primary_tier_name: str,
    queries: List[BenchmarkQuery],
    corpus: List[Engram],
    rerank_depths: List[int],
    n_runs: int,
    gpu_device: str,
    enable_audit: bool = False,
    audit_queries: int = 10,
    audit_depth: int = 50,
) -> Dict[str, Any]:
    from benchmarks.runners.retrieval_runner import _check_backend, _create_tier

    if not _check_backend(primary_tier_name):
        print(f"    [WARN]  {primary_tier_name} backend unavailable - skipping")
        return {"status": "skipped", "reason": "backend_unavailable"}

    # Fresh per-track namespace for deterministic rerank comparisons.
    bench_namespace = f"mnemos_rerank_{int(time.time())}_{primary_tier_name}"
    try:
        primary_tier = _create_tier(
            primary_tier_name, gpu_device=gpu_device, bench_namespace=bench_namespace
        )
    except Exception as e:
        print(f"    [WARN]  {primary_tier_name} tier init failed - skipping ({e})")
        return {"status": "skipped", "reason": "tier_init_failed"}

    # Ensure primary tier has benchmark corpus loaded for this track.
    for i in range(0, len(corpus), 500):
        primary_tier.index(corpus[i:i + 500])

    semantic_queries = [q for q in queries if q.regime == "semantic"][:50]
    print(f"  [{primary_tier_name}] Using {len(semantic_queries)} semantic queries")

    # Try to initialize CrossEncoderReranker
    try:
        from mnemos.retrieval.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    except Exception as e:
        print(f"    [WARN]  Cross-Encoder unavailable for {primary_tier_name}: {e}")
        return {"status": "skipped", "reason": "reranker_not_available"}

    results: Dict[str, Any] = {
        "status": "success",
        "primary_tier": primary_tier_name,
        "n_queries": len(semantic_queries),
    }

    # Baseline runs
    baseline_reports = []
    for _ in range(n_runs):
        run_results = []
        for q in semantic_queries:
            t0 = time.perf_counter()
            hits = primary_tier.search(q.text, top_k=10)
            latency = time.perf_counter() - t0
            run_results.append(QueryResult(
                query_id=q.id,
                regime="semantic",
                returned_ids=[r.engram.id for r in hits],
                gold_ids=q.gold_ids,
                latency_s=latency,
                tier=primary_tier_name,
            ))
        baseline_reports.append(aggregate_results(run_results, primary_tier_name, "semantic"))

    baseline = {
        "mrr_at_10": round(float(np.median([r.mrr_at_10 for r in baseline_reports])), 4),
        "ndcg_at_10": round(float(np.median([r.ndcg_at_10 for r in baseline_reports])), 4),
        "recall_at_10": round(float(np.median([r.recall_at_10 for r in baseline_reports])), 4),
        "latency_p50_ms": round(float(np.median([r.latency_p50_ms for r in baseline_reports])), 2),
        "latency_p95_ms": round(float(np.median([r.latency_p95_ms for r in baseline_reports])), 2),
    }
    results["baseline"] = baseline
    print(f"    Baseline MRR={baseline['mrr_at_10']:.4f} p50={baseline['latency_p50_ms']:.1f}ms")

    # Depth sweeps
    depth_results: Dict[str, Any] = {}

    for depth in rerank_depths:
        print(f"    Rerank @ top-{depth}")
        vram_before = _get_vram_usage_mb()
        reports = []
        for _ in range(n_runs):
            run_results = []
            for q in semantic_queries:
                t0 = time.perf_counter()
                candidates = primary_tier.search(q.text, top_k=depth)
                reranked = reranker.rerank(q.text, candidates, top_k=10)
                latency = time.perf_counter() - t0
                run_results.append(QueryResult(
                    query_id=q.id,
                    regime="semantic",
                    returned_ids=[r.engram.id for r in reranked],
                    gold_ids=q.gold_ids,
                    latency_s=latency,
                    tier=f"{primary_tier_name}+cross_encoder@{depth}",
                ))
            reports.append(aggregate_results(run_results, f"{primary_tier_name}+cross_encoder", "semantic"))
        vram_after = _get_vram_usage_mb()

        med_mrr = float(np.median([r.mrr_at_10 for r in reports]))
        med_ndcg = float(np.median([r.ndcg_at_10 for r in reports]))
        med_latency_p50 = float(np.median([r.latency_p50_ms for r in reports]))
        med_latency_p95 = float(np.median([r.latency_p95_ms for r in reports]))

        latency_increase = med_latency_p50 - baseline["latency_p50_ms"]
        mrr_uplift = med_mrr - baseline["mrr_at_10"]
        ndcg_uplift = med_ndcg - baseline["ndcg_at_10"]

        depth_results[f"top_{depth}"] = {
            "depth": depth,
            "mrr_at_10": round(med_mrr, 4),
            "ndcg_at_10": round(med_ndcg, 4),
            "latency_p50_ms": round(med_latency_p50, 2),
            "latency_p95_ms": round(med_latency_p95, 2),
            "vram_delta_mb": round(vram_after - vram_before, 1),
            "mrr_uplift": round(mrr_uplift, 4),
            "ndcg_uplift": round(ndcg_uplift, 4),
            "latency_increase_ms": round(latency_increase, 2),
            "mrr_uplift_per_ms": round(mrr_uplift / latency_increase, 6) if latency_increase > 0 else 0.0,
            "ndcg_uplift_per_ms": round(ndcg_uplift / latency_increase, 6) if latency_increase > 0 else 0.0,
        }

    results["reranked"] = depth_results

    # Recommend depth by best MRR uplift per added millisecond (positive-only).
    viable = [
        d for d in depth_results.values()
        if d["mrr_uplift"] > 0 and d["latency_increase_ms"] > 0
    ]
    if viable:
        viable.sort(key=lambda x: x["mrr_uplift_per_ms"], reverse=True)
        results["recommended_depth"] = viable[0]["depth"]
    else:
        results["recommended_depth"] = None

    if enable_audit:
        print(f"    [WARN] Audit trace disabled. Reranker natively wraps output, omitting diagnostics.")

    return results


def run_rerank_benchmark(
    queries: List[BenchmarkQuery],
    corpus: List[Engram],
    primary_tiers: List[str] = None, # type: ignore
    rerank_depths: List[int] = None, # type: ignore
    n_runs: int = 3,
    gpu_device: str = "cuda",
    enable_audit: bool = False,
    audit_queries: int = 10,
    audit_depth: int = 50,
) -> Dict[str, Any]:
    """
    Run Track 2 across primary tiers and rerank depths.

    Returns:
      {
        "track": "rerank",
        "status": "success|skipped",
        "tiers": { "qdrant": {...}, "pgvector": {...} }
      }
    """
    if primary_tiers is None:
        primary_tiers = ["qdrant", "pgvector"]
    if rerank_depths is None:
        rerank_depths = [20, 50, 100]

    print("\n" + "=" * 70)
    print("  TRACK 2: Cross-Encoder Reranking Uplift")
    print("=" * 70)

    out: Dict[str, Any] = {"track": "rerank", "status": "success", "tiers": {}}
    any_success = False

    for tier in primary_tiers:
        out["tiers"][tier] = _run_single_tier_rerank(
            tier,
            queries,
            corpus,
            rerank_depths,
            n_runs=n_runs,
            gpu_device=gpu_device,
            enable_audit=enable_audit,
            audit_queries=audit_queries,
            audit_depth=audit_depth,
        )
        if out["tiers"][tier].get("status") == "success":
            any_success = True

    if not any_success:
        out["status"] = "skipped"
        out["reason"] = "no_rerank_results"

    return out
