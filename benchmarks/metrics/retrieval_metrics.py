"""
MNEMOS Benchmark - Retrieval Metrics
======================================

Standard IR metrics: Recall@K, MRR, nDCG, latency percentiles.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class QueryResult:
    """Result of a single benchmark query execution."""
    query_id: str
    regime: str
    returned_ids: List[str]
    gold_ids: List[str]
    latency_s: float
    tier: str = ""


@dataclass
class RetrievalReport:
    """Aggregated retrieval metrics for a benchmark run."""
    tier: str
    regime: str
    n_queries: int = 0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_qps: float = 0.0


def recall_at_k(returned: List[str], gold: List[str], k: int = 10) -> float:
    """Fraction of gold-relevant documents found in top-k returned."""
    if not gold:
        return 0.0
    top_k = set(returned[:k])
    gold_set = set(gold)
    return len(top_k & gold_set) / len(gold_set)


def reciprocal_rank(returned: List[str], gold: List[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    gold_set = set(gold)
    for i, doc_id in enumerate(returned):
        if doc_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(returned: List[str], gold: List[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not gold:
        return 0.0

    gold_set = set(gold)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(returned[:k]):
        rel = 1.0 if doc_id in gold_set else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted([1.0 if gid in gold_set else 0.0 for gid in gold[:k]], reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def latency_percentiles(latencies_s: List[float]) -> Dict[str, float]:
    """Compute latency percentiles in milliseconds."""
    if not latencies_s:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    arr = np.array(latencies_s) * 1000  # convert to ms
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }


def aggregate_results(results: List[QueryResult], tier: str, regime: str) -> RetrievalReport:
    """Aggregate individual query results into a retrieval report."""
    if not results:
        return RetrievalReport(tier=tier, regime=regime)

    recalls = [recall_at_k(r.returned_ids, r.gold_ids, k=10) for r in results]
    mrrs = [reciprocal_rank(r.returned_ids, r.gold_ids) for r in results]
    ndcgs = [ndcg_at_k(r.returned_ids, r.gold_ids, k=10) for r in results]
    latencies = [r.latency_s for r in results]

    pcts = latency_percentiles(latencies)
    total_time = sum(latencies)

    return RetrievalReport(
        tier=tier,
        regime=regime,
        n_queries=len(results),
        recall_at_10=round(float(np.mean(recalls)), 4),
        mrr_at_10=round(float(np.mean(mrrs)), 4),
        ndcg_at_10=round(float(np.mean(ndcgs)), 4),
        latency_p50_ms=pcts["p50_ms"],
        latency_p95_ms=pcts["p95_ms"],
        latency_p99_ms=pcts["p99_ms"],
        throughput_qps=round(len(results) / total_time, 1) if total_time > 0 else 0.0,
    )
