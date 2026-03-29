"""
MNEMOS Benchmark - ColBERT Reranking Runner (Track 2)
======================================================

Product question: When is ColBERT worth the latency and VRAM cost?

Compares baseline retrieval vs ColBERT candidate reranking at
depths 20/50/100 for both primary tiers (Qdrant and pgvector).
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.datasets.query_generator import BenchmarkQuery
from benchmarks.metrics.retrieval_metrics import QueryResult, aggregate_results
from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult


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


def _maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """ColBERT late-interaction MaxSim score."""
    sim_matrix = query_emb @ doc_emb.T
    max_sims = sim_matrix.max(axis=1)
    return float(max_sims.mean())


def _maxsim_details(query_emb: np.ndarray, doc_emb: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """ColBERT MaxSim score with compact token-level diagnostics."""
    sim_matrix = query_emb @ doc_emb.T
    max_sims = sim_matrix.max(axis=1)
    score = float(max_sims.mean())
    details = {
        "query_tokens": int(query_emb.shape[0]),
        "doc_tokens": int(doc_emb.shape[0]),
        "maxsim_mean": score,
        "maxsim_p50": float(np.percentile(max_sims, 50)),
        "maxsim_p90": float(np.percentile(max_sims, 90)),
        "maxsim_min": float(np.min(max_sims)),
        "maxsim_max": float(np.max(max_sims)),
    }
    return score, details


def _tokenize_for_overlap(text: str) -> List[str]:
    """Simple alnum tokenization for lexical overlap baseline."""
    tokens = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def _overlap_score(query_text: str, doc_text: str) -> float:
    """Simple lexical overlap baseline score."""
    q_tokens = set(_tokenize_for_overlap(query_text))
    if not q_tokens:
        return 0.0
    d_tokens = set(_tokenize_for_overlap(doc_text))
    return len(q_tokens & d_tokens) / len(q_tokens)


def _mean_cosine(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Cosine over mean-pooled token embeddings as trivial baseline."""
    q = query_emb.mean(axis=0)
    d = doc_emb.mean(axis=0)
    qn = np.linalg.norm(q)
    dn = np.linalg.norm(d)
    if qn == 0 or dn == 0:
        return 0.0
    return float(np.dot(q, d) / (qn * dn))


def _rank_of(doc_id: str, ranked_ids: List[str]) -> Optional[int]:
    """1-based rank of doc_id in ranked_ids."""
    try:
        return ranked_ids.index(doc_id) + 1
    except ValueError:
        return None


def _rerank_candidates(
    query_text: str,
    candidates: List[SearchResult],
    colbert_tier: Any,
    query_cache: Dict[str, np.ndarray],
    doc_cache: Dict[str, np.ndarray],
    top_k: int = 10,
) -> List[SearchResult]:
    """Rerank primary candidates with ColBERT embeddings."""
    if not candidates:
        return []

    if query_text not in query_cache:
        query_cache[query_text] = colbert_tier._encode_multi_vector(query_text)
    q_emb = query_cache[query_text]

    rescored: List[Tuple[SearchResult, float]] = []
    for hit in candidates:
        did = hit.engram.id
        if did not in doc_cache:
            doc_cache[did] = colbert_tier._encode_multi_vector(hit.engram.content)
        score = _maxsim(q_emb, doc_cache[did])
        rescored.append((hit, score))

    rescored.sort(key=lambda x: x[1], reverse=True)
    out: List[SearchResult] = []
    for hit, score in rescored[:top_k]:
        out.append(SearchResult(engram=hit.engram, score=score, tier=f"{hit.tier}+colbert"))
    return out


def _score_candidates_for_audit(
    query_text: str,
    candidates: List[SearchResult],
    colbert_tier: Any,
    query_cache: Dict[str, np.ndarray],
    doc_cache: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Return per-candidate score breakdown for audit traces."""
    if not candidates:
        return []

    if query_text not in query_cache:
        query_cache[query_text] = colbert_tier._encode_multi_vector(query_text)
    q_emb = query_cache[query_text]

    out: List[Dict[str, Any]] = []
    for i, hit in enumerate(candidates):
        did = hit.engram.id
        if did not in doc_cache:
            doc_cache[did] = colbert_tier._encode_multi_vector(hit.engram.content)
        d_emb = doc_cache[did]

        colbert_score, detail = _maxsim_details(q_emb, d_emb)
        overlap = _overlap_score(query_text, hit.engram.content)
        mean_cos = _mean_cosine(q_emb, d_emb)
        out.append(
            {
                "doc_id": did,
                "baseline_rank": i + 1,
                "baseline_score": float(hit.score),
                "colbert_score": colbert_score,
                "overlap_score": overlap,
                "mean_cosine_score": mean_cos,
                "maxsim_detail": detail,
            }
        )
    return out


def _top_ids_by(score_rows: List[Dict[str, Any]], key: str, k: int = 10) -> List[str]:
    rows = sorted(score_rows, key=lambda x: x[key], reverse=True)
    return [r["doc_id"] for r in rows[:k]]


def _build_rerank_audit(
    primary_tier: Any,
    colbert_tier: Any,
    semantic_queries: List[BenchmarkQuery],
    query_cache: Dict[str, np.ndarray],
    doc_cache: Dict[str, np.ndarray],
    depth: int = 50,
    n_queries: int = 10,
) -> Dict[str, Any]:
    """Create query-level trace diagnostics for rerank correctness audit."""
    traces: List[Dict[str, Any]] = []
    audit_queries = semantic_queries[:n_queries]
    baseline_qr: List[QueryResult] = []
    colbert_qr: List[QueryResult] = []
    overlap_qr: List[QueryResult] = []
    mean_cos_qr: List[QueryResult] = []

    for q in audit_queries:
        t0 = time.perf_counter()
        candidates = primary_tier.search(q.text, top_k=depth)
        baseline_latency = time.perf_counter() - t0
        score_rows = _score_candidates_for_audit(
            q.text, candidates, colbert_tier, query_cache, doc_cache
        )
        baseline_top10 = [c.engram.id for c in candidates[:10]]
        colbert_top10 = _top_ids_by(score_rows, "colbert_score", k=10)
        overlap_top10 = _top_ids_by(score_rows, "overlap_score", k=10)
        mean_cos_top10 = _top_ids_by(score_rows, "mean_cosine_score", k=10)

        baseline_qr.append(
            QueryResult(
                query_id=q.id,
                regime="semantic",
                returned_ids=baseline_top10,
                gold_ids=q.gold_ids,
                latency_s=baseline_latency,
                tier="baseline",
            )
        )
        colbert_qr.append(
            QueryResult(
                query_id=q.id,
                regime="semantic",
                returned_ids=colbert_top10,
                gold_ids=q.gold_ids,
                latency_s=0.0,
                tier="colbert",
            )
        )
        overlap_qr.append(
            QueryResult(
                query_id=q.id,
                regime="semantic",
                returned_ids=overlap_top10,
                gold_ids=q.gold_ids,
                latency_s=0.0,
                tier="overlap",
            )
        )
        mean_cos_qr.append(
            QueryResult(
                query_id=q.id,
                regime="semantic",
                returned_ids=mean_cos_top10,
                gold_ids=q.gold_ids,
                latency_s=0.0,
                tier="mean_cosine",
            )
        )

        gold_moves: List[Dict[str, Any]] = []
        for gid in q.gold_ids:
            before = _rank_of(gid, baseline_top10)
            after = _rank_of(gid, colbert_top10)
            if before is not None or after is not None:
                if before is None:
                    delta = None
                elif after is None:
                    delta = None
                else:
                    delta = before - after
                gold_moves.append(
                    {
                        "doc_id": gid,
                        "baseline_rank_top10": before,
                        "rerank_rank_top10": after,
                        "rank_delta": delta,
                    }
                )

        # Store top-3 diagnostics to keep raw output compact.
        top3_colbert = sorted(score_rows, key=lambda x: x["colbert_score"], reverse=True)[:3]
        top3_details = [
            {
                "doc_id": r["doc_id"],
                "colbert_score": round(r["colbert_score"], 6),
                "baseline_rank": r["baseline_rank"],
                "maxsim_detail": {
                    "query_tokens": r["maxsim_detail"]["query_tokens"],
                    "doc_tokens": r["maxsim_detail"]["doc_tokens"],
                    "maxsim_mean": round(r["maxsim_detail"]["maxsim_mean"], 6),
                    "maxsim_p50": round(r["maxsim_detail"]["maxsim_p50"], 6),
                    "maxsim_p90": round(r["maxsim_detail"]["maxsim_p90"], 6),
                },
            }
            for r in top3_colbert
        ]

        traces.append(
            {
                "query_id": q.id,
                "query_text": q.text,
                "gold_ids": q.gold_ids,
                "baseline_top10": baseline_top10,
                "colbert_top10": colbert_top10,
                "overlap_top10": overlap_top10,
                "mean_cosine_top10": mean_cos_top10,
                "gold_rank_moves": gold_moves,
                "top3_colbert_token_diagnostics": top3_details,
            }
        )

    summary_reports = {
        "baseline": aggregate_results(baseline_qr, "baseline", "semantic"),
        "colbert": aggregate_results(colbert_qr, "colbert", "semantic"),
        "lexical_overlap": aggregate_results(overlap_qr, "lexical_overlap", "semantic"),
        "mean_cosine": aggregate_results(mean_cos_qr, "mean_cosine", "semantic"),
    }
    summary = {
        key: {
            "mrr_at_10": rep.mrr_at_10,
            "ndcg_at_10": rep.ndcg_at_10,
            "recall_at_10": rep.recall_at_10,
        }
        for key, rep in summary_reports.items()
    }
    colbert_mrr = summary["colbert"]["mrr_at_10"]
    overlap_mrr = summary["lexical_overlap"]["mrr_at_10"]
    mean_cos_mrr = summary["mean_cosine"]["mrr_at_10"]
    gate_a1_pass = colbert_mrr >= max(overlap_mrr, mean_cos_mrr)

    return {
        "status": "success",
        "depth": depth,
        "n_queries": len(audit_queries),
        "summary": summary,
        "gate_a1_pass": gate_a1_pass,
        "traces": traces,
    }


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

    # Try to initialize ColBERT tier for reranking scorer.
    try:
        from mnemos.retrieval.colbert_tier import ColBERTTier, ColBERTConfig
        colbert_tier = ColBERTTier(
            ColBERTConfig(index_dir=f"benchmarks/outputs/tmp_colbert_{primary_tier_name}")
        )
    except Exception as e:
        print(f"    [WARN]  ColBERT unavailable for {primary_tier_name}: {e}")
        return {"status": "skipped", "reason": "colbert_not_available"}

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
    query_cache: Dict[str, np.ndarray] = {}
    doc_cache: Dict[str, np.ndarray] = {}
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
                reranked = _rerank_candidates(
                    q.text, candidates, colbert_tier, query_cache, doc_cache, top_k=10
                )
                latency = time.perf_counter() - t0
                run_results.append(QueryResult(
                    query_id=q.id,
                    regime="semantic",
                    returned_ids=[r.engram.id for r in reranked],
                    gold_ids=q.gold_ids,
                    latency_s=latency,
                    tier=f"{primary_tier_name}+colbert@{depth}",
                ))
            reports.append(aggregate_results(run_results, f"{primary_tier_name}+colbert", "semantic"))
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
        print(f"    Audit trace: {audit_queries} queries @ top-{audit_depth}")
        results["audit"] = _build_rerank_audit(
            primary_tier=primary_tier,
            colbert_tier=colbert_tier,
            semantic_queries=semantic_queries,
            query_cache=query_cache,
            doc_cache=doc_cache,
            depth=audit_depth,
            n_queries=audit_queries,
        )
    return results


def run_rerank_benchmark(
    queries: List[BenchmarkQuery],
    corpus: List[Engram],
    primary_tiers: Optional[List[str]] = None,
    rerank_depths: Optional[List[int]] = None,
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
    print("  TRACK 2: ColBERT Reranking Uplift")
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
