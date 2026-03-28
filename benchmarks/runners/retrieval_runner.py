"""
MNEMOS Benchmark - Retrieval Runner (Track 1)
================================================

Product question: When does Core (Qdrant) win vs Governance (pgvector)?

Runs search benchmarks against live backends across three query regimes.
Requires Docker services to be running.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.datasets.query_generator import BenchmarkQuery
from benchmarks.metrics.retrieval_metrics import (
    QueryResult,
    RetrievalReport,
    aggregate_results,
    latency_percentiles,
)

# Try to import retrieval tiers - graceful skip if deps missing
QDRANT_AVAILABLE = False
PGVECTOR_AVAILABLE = False

try:
    from mnemos.retrieval.qdrant_tier import QdrantTier
    QDRANT_AVAILABLE = True
except ImportError:
    pass

try:
    from mnemos.retrieval.pgvector_tier import PgvectorTier
    PGVECTOR_AVAILABLE = True
except ImportError:
    pass

from mnemos.engram.model import Engram


def _check_backend(tier_name: str) -> bool:
    """Check if a backend is available."""
    if tier_name == "qdrant":
        if not QDRANT_AVAILABLE:
            return False
        try:
            import requests
            r = requests.get("http://localhost:6333/healthz", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
    elif tier_name == "pgvector":
        if not PGVECTOR_AVAILABLE:
            return False
        try:
            import psycopg
            conn = psycopg.connect(
                "postgresql://mnemos:mnemos@localhost:5432/mnemos",
                connect_timeout=3,
            )
            conn.close()
            return True
        except Exception:
            return False
    return False


def _create_tier(tier_name: str, embedding_model: str = "all-MiniLM-L6-v2",
                 gpu_device: str = "cuda"):
    """Create a retrieval tier instance."""
    if tier_name == "qdrant":
        return QdrantTier(
            url="http://localhost:6333",
            collection_name="mnemos_bench",
            embedding_model=embedding_model,
            gpu_device=gpu_device,
        )
    elif tier_name == "pgvector":
        return PgvectorTier(
            dsn="postgresql://mnemos:mnemos@localhost:5432/mnemos",
            table_name="mnemos_bench_vectors",
            embedding_model=embedding_model,
            gpu_device=gpu_device,
        )
    raise ValueError(f"Unknown tier: {tier_name}")


def run_ingest_benchmark(
    tier_name: str,
    corpus: List[Engram],
    batch_size: int = 500,
    gpu_device: str = "cuda",
) -> Dict[str, Any]:
    """
    Benchmark ingestion: index the full corpus and measure time.

    Decision this drives: How fast can each profile ingest?
    """
    print(f"\n  [{tier_name}] Ingest benchmark: {len(corpus)} engrams")

    if not _check_backend(tier_name):
        print(f"    [WARN]  {tier_name} backend not available - skipping")
        return {"tier": tier_name, "status": "skipped", "reason": "backend_unavailable"}

    tier = _create_tier(tier_name, gpu_device=gpu_device)

    # Ingest in batches
    t0 = time.perf_counter()
    total_indexed = 0
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        count = tier.index(batch)
        total_indexed += count
    elapsed = time.perf_counter() - t0

    stats = tier.stats()

    result = {
        "tier": tier_name,
        "status": "success",
        "corpus_size": len(corpus),
        "indexed": total_indexed,
        "elapsed_s": round(elapsed, 3),
        "docs_per_sec": round(len(corpus) / elapsed, 1) if elapsed > 0 else 0,
        "index_stats": stats,
    }

    print(f"    [OK] {total_indexed} indexed in {elapsed:.2f}s "
          f"({result['docs_per_sec']} docs/s)")
    return result


def run_search_benchmark(
    tier_name: str,
    queries: List[BenchmarkQuery],
    regime: str,
    n_runs: int = 5,
    top_k: int = 10,
    gpu_device: str = "cuda",
) -> Dict[str, Any]:
    """
    Benchmark search: run queries against a backend and measure metrics.

    Decision this drives: Which profile wins for which query regime?
    """
    regime_queries = [q for q in queries if q.regime == regime]
    print(f"\n  [{tier_name}] Search benchmark: {len(regime_queries)} queries, "
          f"regime={regime}, runs={n_runs}")

    if not _check_backend(tier_name):
        print(f"    [WARN]  {tier_name} backend not available - skipping")
        return {"tier": tier_name, "regime": regime, "status": "skipped"}

    tier = _create_tier(tier_name, gpu_device=gpu_device)

    all_reports = []

    for run_idx in range(n_runs):
        results = []
        for q in regime_queries:
            t0 = time.perf_counter()
            search_results = tier.search(q.text, top_k=top_k, filters=q.filters or None)
            latency = time.perf_counter() - t0

            returned_ids = [r.engram.id for r in search_results]
            results.append(QueryResult(
                query_id=q.id,
                regime=q.regime,
                returned_ids=returned_ids,
                gold_ids=q.gold_ids,
                latency_s=latency,
                tier=tier_name,
            ))

        report = aggregate_results(results, tier_name, regime)
        all_reports.append(report)

        if run_idx == 0:
            print(f"    Run {run_idx + 1}: Recall@10={report.recall_at_10:.4f}  "
                  f"MRR={report.mrr_at_10:.4f}  p50={report.latency_p50_ms:.1f}ms  "
                  f"p99={report.latency_p99_ms:.1f}ms")

    # Aggregate across runs (use median for stability)
    import numpy as np
    final = {
        "tier": tier_name,
        "regime": regime,
        "status": "success",
        "n_queries": len(regime_queries),
        "n_runs": n_runs,
        "recall_at_10": round(float(np.median([r.recall_at_10 for r in all_reports])), 4),
        "mrr_at_10": round(float(np.median([r.mrr_at_10 for r in all_reports])), 4),
        "ndcg_at_10": round(float(np.median([r.ndcg_at_10 for r in all_reports])), 4),
        "latency_p50_ms": round(float(np.median([r.latency_p50_ms for r in all_reports])), 2),
        "latency_p95_ms": round(float(np.median([r.latency_p95_ms for r in all_reports])), 2),
        "latency_p99_ms": round(float(np.median([r.latency_p99_ms for r in all_reports])), 2),
        "throughput_qps": round(float(np.median([r.throughput_qps for r in all_reports])), 1),
    }

    print(f"    📊 Median: Recall@10={final['recall_at_10']:.4f}  "
          f"MRR={final['mrr_at_10']:.4f}  p50={final['latency_p50_ms']:.1f}ms")
    return final


def run_retrieval_track(
    corpus: List[Engram],
    queries: List[BenchmarkQuery],
    tiers: List[str] = None,
    n_runs: int = 5,
    gpu_device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full Track 1: Profile Retrieval Benchmarks.

    Returns structured results for all tier × regime combinations.
    """
    if tiers is None:
        tiers = ["qdrant", "pgvector"]

    regimes = ["semantic", "light_filter", "heavy_filter"]

    print("\n" + "=" * 70)
    print("  TRACK 1: Profile Retrieval Benchmarks")
    print("=" * 70)

    results = {
        "track": "retrieval",
        "ingest": {},
        "search": {},
    }

    for tier_name in tiers:
        # Ingest
        results["ingest"][tier_name] = run_ingest_benchmark(
            tier_name, corpus, gpu_device=gpu_device
        )

        # Search per regime
        results["search"][tier_name] = {}
        for regime in regimes:
            results["search"][tier_name][regime] = run_search_benchmark(
                tier_name, queries, regime, n_runs=n_runs, gpu_device=gpu_device
            )

    return results
