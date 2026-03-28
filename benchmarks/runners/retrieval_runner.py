"""
MNEMOS Benchmark - Retrieval Runner (Track 1)
================================================

Product question: When does Core (Qdrant) win vs Governance (pgvector)?

Runs search benchmarks against live backends across three query regimes.
Requires Docker services to be running.
"""

import json
import os
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
QDRANT_IMPORT_ERROR = ""
PGVECTOR_IMPORT_ERROR = ""

try:
    from mnemos.retrieval.qdrant_tier import QdrantTier
    QDRANT_AVAILABLE = True
except ImportError as e:
    QDRANT_IMPORT_ERROR = str(e)

try:
    from mnemos.retrieval.pgvector_tier import PgvectorTier
    PGVECTOR_AVAILABLE = True
except ImportError as e:
    PGVECTOR_IMPORT_ERROR = str(e)

from mnemos.engram.model import Engram

POSTGRES_DSN = os.getenv(
    "MNEMOS_BENCH_POSTGRES_DSN",
    "postgresql://mnemos:mnemos@localhost:5432/mnemos",
)
QDRANT_URL = os.getenv("MNEMOS_BENCH_QDRANT_URL", "http://localhost:6333")


def _matches_required_filters(engram: Engram, filters: Dict[str, Any]) -> bool:
    """Return True if an engram satisfies all required filters."""
    if not filters:
        return True

    for key, value in filters.items():
        if key == "source":
            if engram.source != str(value):
                return False
        elif key == "confidence_min":
            if float(engram.confidence) < float(value):
                return False
        elif key == "neuro_tag":
            if str(value) not in (engram.neuro_tags or []):
                return False
        elif key.startswith("metadata."):
            meta_key = key.split(".", 1)[1]
            if meta_key == "timestamp_epoch_min":
                e_ts = engram.metadata.get("timestamp_epoch")
                if e_ts is None or int(e_ts) < int(value):
                    return False
            elif meta_key == "timestamp_epoch_max":
                e_ts = engram.metadata.get("timestamp_epoch")
                if e_ts is None or int(e_ts) > int(value):
                    return False
            elif str(engram.metadata.get(meta_key, "")) != str(value):
                return False
        else:
            # Generic fallback: metadata key match
            if str(engram.metadata.get(key, "")) != str(value):
                return False

    return True


def _check_backend(tier_name: str) -> bool:
    """Check if a backend is available."""
    if tier_name == "qdrant":
        if not QDRANT_AVAILABLE:
            if QDRANT_IMPORT_ERROR:
                print(f"    [WARN]  qdrant import unavailable: {QDRANT_IMPORT_ERROR}")
            return False
        try:
            import requests
            r = requests.get(f"{QDRANT_URL.rstrip('/')}/healthz", timeout=3)
            return r.status_code == 200
        except Exception as e:
            print(f"    [WARN]  qdrant health check failed: {e}")
            return False
    elif tier_name == "pgvector":
        if not PGVECTOR_AVAILABLE:
            if PGVECTOR_IMPORT_ERROR:
                print(f"    [WARN]  pgvector import unavailable: {PGVECTOR_IMPORT_ERROR}")
            return False
        try:
            import psycopg
            conn = psycopg.connect(POSTGRES_DSN, connect_timeout=3)
            conn.close()
            return True
        except Exception as e:
            print(f"    [WARN]  pgvector health check failed: {e}")
            return False
    return False


def _create_tier(tier_name: str, embedding_model: str = "all-MiniLM-L6-v2",
                 gpu_device: str = "cuda",
                 bench_namespace: str = "mnemos_bench"):
    """Create a retrieval tier instance."""
    if tier_name == "qdrant":
        return QdrantTier(
            url=QDRANT_URL,
            collection_name=bench_namespace,
            embedding_model=embedding_model,
            gpu_device=gpu_device,
        )
    elif tier_name == "pgvector":
        return PgvectorTier(
            dsn=POSTGRES_DSN,
            table_name=f"{bench_namespace}_vectors",
            embedding_model=embedding_model,
            gpu_device=gpu_device,
        )
    raise ValueError(f"Unknown tier: {tier_name}")


def run_ingest_benchmark(
    tier_name: str,
    corpus: List[Engram],
    batch_size: int = 500,
    gpu_device: str = "cuda",
    bench_namespace: str = "mnemos_bench",
) -> Dict[str, Any]:
    """
    Benchmark ingestion: index the full corpus and measure time.

    Decision this drives: How fast can each profile ingest?
    """
    print(f"\n  [{tier_name}] Ingest benchmark: {len(corpus)} engrams")

    if not _check_backend(tier_name):
        print(f"    [WARN]  {tier_name} backend not available - skipping")
        return {"tier": tier_name, "status": "skipped", "reason": "backend_unavailable"}

    tier = _create_tier(tier_name, gpu_device=gpu_device, bench_namespace=bench_namespace)

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
        "docs_per_sec": round(total_indexed / elapsed, 1) if elapsed > 0 else 0,
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
    bench_namespace: str = "mnemos_bench",
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

    tier = _create_tier(tier_name, gpu_device=gpu_device, bench_namespace=bench_namespace)

    all_reports = []
    compliance_scores = []
    violation_rates = []

    for run_idx in range(n_runs):
        results = []
        run_total_hits = 0
        run_compliant_hits = 0
        for q in regime_queries:
            t0 = time.perf_counter()
            search_filters = getattr(q, "retrieval_filters", None) or q.filters or None
            required_filters = q.filters or {}
            search_results = tier.search(q.text, top_k=top_k, filters=search_filters)
            latency = time.perf_counter() - t0

            if required_filters:
                for hit in search_results:
                    run_total_hits += 1
                    if _matches_required_filters(hit.engram, required_filters):
                        run_compliant_hits += 1

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
        if run_total_hits > 0:
            compliance = run_compliant_hits / run_total_hits
            compliance_scores.append(compliance)
            violation_rates.append(1.0 - compliance)

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
        "filter_compliance_at_10": (
            round(float(np.median(compliance_scores)), 4) if compliance_scores else None
        ),
        "constraint_violation_rate_at_10": (
            round(float(np.median(violation_rates)), 4) if violation_rates else None
        ),
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
    # Use a fresh namespace per run to avoid stale data affecting metrics.
    bench_namespace = f"mnemos_bench_{int(time.time())}"

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
            tier_name, corpus, gpu_device=gpu_device, bench_namespace=bench_namespace
        )

        # Search per regime
        results["search"][tier_name] = {}
        for regime in regimes:
            results["search"][tier_name][regime] = run_search_benchmark(
                tier_name,
                queries,
                regime,
                n_runs=n_runs,
                gpu_device=gpu_device,
                bench_namespace=bench_namespace,
            )

    return results
