"""
MNEMOS Benchmark - Hybrid Retrieval Runner (Track 5)
====================================================

Product question: when does lexical + semantic fusion beat semantic-only?
"""

import json
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.datasets.query_generator import BenchmarkQuery, _compute_gold_labels
from benchmarks.metrics.retrieval_metrics import (
    QueryResult,
    aggregate_results,
    reciprocal_rank,
)
from benchmarks.runners.retrieval_runner import (
    POSTGRES_DSN,
    _check_backend,
    _create_tier,
)
from mnemos.engram.model import Engram
from mnemos.retrieval.lexical_tier import LexicalTier
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.retrieval_router import RetrievalRouter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERYSET_PATH = PROJECT_ROOT / "benchmarks" / "truthsets" / "gate_c_hybrid_queries.json"


def _load_gate_c_queries(corpus: List[Engram]) -> List[Dict[str, Any]]:
    with open(QUERYSET_PATH, "r", encoding="utf-8") as f:
        specs = json.load(f)

    queries: List[Dict[str, Any]] = []
    for spec in specs:
        q = BenchmarkQuery(
            id=spec["id"],
            text=spec["text"],
            regime="semantic",
            filters=spec.get("filters", {}),
            retrieval_filters=spec.get("filters", {}),
            domain=spec.get("domain", ""),
        )
        q.gold_ids = _compute_gold_labels(q, corpus, top_k=10)
        queries.append({"spec": spec, "query": q})

    return queries


def _summarize_runs(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not run_metrics:
        return {}

    keys = [
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "latency_p50_ms",
        "latency_p95_ms",
        "throughput_qps",
        "lexical_only_contribution_rate",
        "semantic_only_contribution_rate",
        "overlap_rate",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        values = [m[key] for m in run_metrics if m.get(key) is not None]
        out[key] = round(float(median(values)), 4) if values else None

    out["n_queries"] = run_metrics[0].get("n_queries", 0)
    return out


def _evaluate_mode(
    mode_name: str,
    queries: List[Dict[str, Any]],
    search_fn,
) -> Dict[str, Any]:
    query_results: List[QueryResult] = []
    by_class_results: Dict[str, List[QueryResult]] = {}

    lexical_only_hits = 0
    semantic_only_hits = 0
    overlap_hits = 0
    total_hits = 0
    class_contrib: Dict[str, Dict[str, int]] = {}

    rr_by_query: Dict[str, float] = {}

    for row in queries:
        spec = row["spec"]
        q: BenchmarkQuery = row["query"]

        t0 = time.perf_counter()
        hits = search_fn(q)
        latency = time.perf_counter() - t0

        returned_ids = [h.engram.id for h in hits]

        qr = QueryResult(
            query_id=q.id,
            regime=spec["query_class"],
            returned_ids=returned_ids,
            gold_ids=q.gold_ids,
            latency_s=latency,
            tier=mode_name,
        )
        query_results.append(qr)
        by_class_results.setdefault(spec["query_class"], []).append(qr)

        rr_by_query[q.id] = reciprocal_rank(returned_ids, q.gold_ids)

        if mode_name.startswith("hybrid"):
            cls = spec["query_class"]
            class_contrib.setdefault(
                cls,
                {"lexical_only": 0, "semantic_only": 0, "both": 0, "total": 0},
            )
            for hit in hits[:10]:
                src = hit.metadata.get("retrieval_sources", [])
                src_set = set(src)
                if src_set == {"lexical"}:
                    lexical_only_hits += 1
                    class_contrib[cls]["lexical_only"] += 1
                elif src_set == {"semantic"}:
                    semantic_only_hits += 1
                    class_contrib[cls]["semantic_only"] += 1
                elif src_set == {"lexical", "semantic"}:
                    overlap_hits += 1
                    class_contrib[cls]["both"] += 1
                total_hits += 1
                class_contrib[cls]["total"] += 1

    report = aggregate_results(query_results, tier=mode_name, regime="overall")

    by_class: Dict[str, Any] = {}
    for query_class, rows in by_class_results.items():
        r = aggregate_results(rows, tier=mode_name, regime=query_class)
        by_class[query_class] = {
            "n_queries": r.n_queries,
            "recall_at_10": r.recall_at_10,
            "mrr_at_10": r.mrr_at_10,
            "ndcg_at_10": r.ndcg_at_10,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
        }

    metrics: Dict[str, Any] = {
        "n_queries": report.n_queries,
        "recall_at_10": report.recall_at_10,
        "mrr_at_10": report.mrr_at_10,
        "ndcg_at_10": report.ndcg_at_10,
        "latency_p50_ms": report.latency_p50_ms,
        "latency_p95_ms": report.latency_p95_ms,
        "throughput_qps": report.throughput_qps,
        "by_query_class": by_class,
        "rr_by_query": rr_by_query,
        "lexical_only_contribution_rate": None,
        "semantic_only_contribution_rate": None,
        "overlap_rate": None,
    }

    if mode_name.startswith("hybrid") and total_hits > 0:
        metrics["lexical_only_contribution_rate"] = round(lexical_only_hits / total_hits, 4)
        metrics["semantic_only_contribution_rate"] = round(semantic_only_hits / total_hits, 4)
        metrics["overlap_rate"] = round(overlap_hits / total_hits, 4)
        metrics["contribution_breakdown"] = {
            "overall": {
                "lexical_only": lexical_only_hits,
                "semantic_only": semantic_only_hits,
                "both": overlap_hits,
                "total": total_hits,
            },
            "by_query_class": {
                cls: {
                    **row,
                    "lexical_only_rate": round(row["lexical_only"] / row["total"], 4) if row["total"] else 0.0,
                    "semantic_only_rate": round(row["semantic_only"] / row["total"], 4) if row["total"] else 0.0,
                    "both_rate": round(row["both"] / row["total"], 4) if row["total"] else 0.0,
                }
                for cls, row in class_contrib.items()
            },
        }

    return metrics


def _compute_hybrid_win_rates(
    semantic_rr: Dict[str, float],
    hybrid_rr: Dict[str, float],
    queries: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    by_class: Dict[str, List[Tuple[float, float]]] = {}
    for row in queries:
        qid = row["query"].id
        query_class = row["spec"]["query_class"]
        by_class.setdefault(query_class, []).append((semantic_rr.get(qid, 0.0), hybrid_rr.get(qid, 0.0)))

    out: Dict[str, Dict[str, float]] = {}
    all_pairs: List[Tuple[float, float]] = []
    for cls, pairs in by_class.items():
        all_pairs.extend(pairs)
        wins = sum(1 for sem, hyb in pairs if hyb > sem)
        out[cls] = {
            "count": float(wins),
            "total": float(len(pairs)),
            "rate": round(wins / len(pairs), 4) if pairs else 0.0,
        }

    total_wins = sum(1 for sem, hyb in all_pairs if hyb > sem)
    out["overall"] = {
        "count": float(total_wins),
        "total": float(len(all_pairs)),
        "rate": round(total_wins / len(all_pairs), 4) if all_pairs else 0.0,
    }
    return out


def _pick_winning_policy_by_class(mode_results: Dict[str, Any], query_classes: List[str]) -> Dict[str, Any]:
    """Pick best hybrid fusion policy per class by MRR@10."""
    policy_modes = {
        "semantic_dominant": "hybrid_semantic_dominant",
        "balanced": "hybrid_balanced",
        "lexical_dominant": "hybrid_lexical_dominant",
    }
    out: Dict[str, Any] = {}
    for query_class in query_classes:
        best_policy = None
        best_mrr = -1.0
        all_scores: Dict[str, float] = {}
        for policy, mode_name in policy_modes.items():
            mode = mode_results.get(mode_name, {})
            class_row = mode.get("by_query_class", {}).get(query_class, {})
            mrr = float(class_row.get("mrr_at_10", 0.0))
            all_scores[policy] = round(mrr, 4)
            if mrr > best_mrr:
                best_mrr = mrr
                best_policy = policy
        out[query_class] = {
            "winning_policy": best_policy,
            "scores": all_scores,
        }
    return out


def run_hybrid_track(
    corpus: List[Engram],
    n_runs: int = 3,
    gpu_device: str = "cuda",
    semantic_backend: str = "qdrant",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  TRACK 5: Hybrid Retrieval (Gate C)")
    print("=" * 70)

    if not _check_backend(semantic_backend):
        return {
            "track": "hybrid",
            "status": "skipped",
            "reason": f"{semantic_backend}_backend_unavailable",
        }

    if not _check_backend("pgvector"):
        return {
            "track": "hybrid",
            "status": "skipped",
            "reason": "postgres_backend_unavailable_for_lexical",
        }

    semantic_tier = _create_tier(semantic_backend, gpu_device=gpu_device, bench_namespace=f"mnemos_hybrid_{int(time.time())}")
    lexical_tier = LexicalTier(
        dsn=POSTGRES_DSN,
        table_name=f"mnemos_hybrid_lexical_{int(time.time())}",
    )

    print(f"  [hybrid] indexing corpus into semantic={semantic_backend} and lexical lanes")
    semantic_tier.index(corpus)
    lexical_tier.index(corpus)

    fusion = TierFusion([semantic_tier])
    router = RetrievalRouter(semantic_fusion=fusion, lexical_tier=lexical_tier)
    queries = _load_gate_c_queries(corpus)

    modes = [
        "semantic_only",
        "lexical_only",
        "hybrid_semantic_dominant",
        "hybrid_balanced",
        "hybrid_lexical_dominant",
    ]

    per_mode_runs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}

    for _ in range(n_runs):
        for mode in modes:
            if mode == "semantic_only":
                run_metrics = _evaluate_mode(
                    mode,
                    queries,
                    lambda q: fusion.search(q.text, top_k=10, filters=q.filters),
                )
            elif mode == "lexical_only":
                run_metrics = _evaluate_mode(
                    mode,
                    queries,
                    lambda q: lexical_tier.search(q.text, top_k=10, filters=q.filters),
                )
            elif mode == "hybrid_semantic_dominant":
                run_metrics = _evaluate_mode(
                    mode,
                    queries,
                    lambda q: router.search(
                        query=q.text,
                        top_k=10,
                        filters=q.filters,
                        retrieval_mode="hybrid",
                        fusion_policy="semantic_dominant",
                        explain=True,
                    )[0],
                )
            elif mode == "hybrid_balanced":
                run_metrics = _evaluate_mode(
                    mode,
                    queries,
                    lambda q: router.search(
                        query=q.text,
                        top_k=10,
                        filters=q.filters,
                        retrieval_mode="hybrid",
                        fusion_policy="balanced",
                        explain=True,
                    )[0],
                )
            else:
                run_metrics = _evaluate_mode(
                    mode,
                    queries,
                    lambda q: router.search(
                        query=q.text,
                        top_k=10,
                        filters=q.filters,
                        retrieval_mode="hybrid",
                        fusion_policy="lexical_dominant",
                        explain=True,
                    )[0],
                )

            per_mode_runs[mode].append(run_metrics)

    mode_results: Dict[str, Any] = {}
    semantic_rr = per_mode_runs["semantic_only"][0]["rr_by_query"] if per_mode_runs["semantic_only"] else {}

    for mode, run_rows in per_mode_runs.items():
        summary = _summarize_runs(run_rows)
        # Use first run for class breakdown shape and rr maps (deterministic query set).
        class_breakdown = run_rows[0].get("by_query_class", {}) if run_rows else {}
        contribution_breakdown = run_rows[0].get("contribution_breakdown") if run_rows else None
        mode_results[mode] = {
            **summary,
            "by_query_class": class_breakdown,
        }
        if contribution_breakdown:
            mode_results[mode]["contribution_breakdown"] = contribution_breakdown

        if mode.startswith("hybrid") and run_rows:
            wins = _compute_hybrid_win_rates(semantic_rr, run_rows[0]["rr_by_query"], queries)
            mode_results[mode]["hybrid_win_rate_over_semantic_only"] = wins

    # remove internal rr fields from run rows in final payload
    for mode in list(mode_results.keys()):
        mode_results[mode].pop("rr_by_query", None)

    query_classes = sorted({q["spec"]["query_class"] for q in queries})
    winning_policy_by_class = _pick_winning_policy_by_class(mode_results, query_classes)

    return {
        "track": "hybrid",
        "status": "success",
        "semantic_backend": semantic_backend,
        "n_runs": n_runs,
        "queryset": str(QUERYSET_PATH.relative_to(PROJECT_ROOT)),
        "n_queries": len(queries),
        "query_classes": query_classes,
        "winning_policy_by_query_class": winning_policy_by_class,
        "modes": mode_results,
    }
