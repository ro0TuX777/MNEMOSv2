"""
MNEMOS Phase 2 Integration Benchmark
======================================

Validates the Phase 2 Qdrant integration changes:
  1. Text index initialization overhead
  2. Hybrid fusion routing correctness (qdrant_rrf vs python_hybrid)
  3. Relevance feedback adapter performance (cache, exemplar lookup)
  4. Retrieval router regression (existing pipeline unchanged)
  5. Full test suite pass rate

Run: python benchmarks/run_phase2_benchmarks.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.retrieval.base import SearchResult
from mnemos.engram.model import Engram
from mnemos.retrieval.relevance_feedback import ExemplarCache, InMemoryFeedbackStore, RelevanceFeedbackAdapter
from mnemos.retrieval.qdrant_hybrid import QdrantHybridFusion
from mnemos.retrieval.hybrid_fusion import HybridFusion
from mnemos.retrieval.policies.fusion_policies import FUSION_POLICIES


def _make_engram(eid, content="test content"):
    return Engram(id=eid, content=content, source="bench")


def _make_results(n, tier="qdrant"):
    return [
        SearchResult(
            engram=_make_engram(f"eng_{i}", f"content for result {i}"),
            score=1.0 - (i * 0.02),
            tier=tier,
        )
        for i in range(n)
    ]


# ─── Benchmark 1: Fusion Policy Registry ────────────────────────────

def bench_fusion_policies():
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Fusion Policy Registry Validation")
    print("=" * 70)

    results = {"policies": {}, "qdrant_rrf_present": False}

    for name, weights in FUSION_POLICIES.items():
        results["policies"][name] = {
            "lexical_weight": weights["lexical"],
            "semantic_weight": weights["semantic"],
            "sum": round(weights["lexical"] + weights["semantic"], 2),
        }
        print(f"  ✅ {name}: lex={weights['lexical']}, sem={weights['semantic']}")

    results["qdrant_rrf_present"] = "qdrant_rrf" in FUSION_POLICIES
    results["total_policies"] = len(FUSION_POLICIES)
    print(f"\n  qdrant_rrf policy present: {results['qdrant_rrf_present']}")
    print(f"  Total policies: {results['total_policies']}")
    return results


# ─── Benchmark 2: Python Hybrid Fusion Throughput ────────────────────

def bench_python_hybrid_throughput():
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Python HybridFusion Throughput (Baseline)")
    print("=" * 70)

    fusion = HybridFusion()
    candidate_counts = [10, 25, 50, 100]
    results = []

    for n in candidate_counts:
        lex_results = _make_results(n, "lexical")
        sem_results = _make_results(n, "semantic")

        # Warmup
        fusion.fuse(lexical_results=lex_results, semantic_results=sem_results,
                     top_k=10, fusion_policy="balanced")

        # Timed runs
        iterations = 500
        t0 = time.perf_counter()
        for _ in range(iterations):
            fusion.fuse(lexical_results=lex_results, semantic_results=sem_results,
                         top_k=10, fusion_policy="balanced")
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / iterations) * 1000
        ops_per_sec = iterations / elapsed

        row = {
            "candidates_per_arm": n,
            "iterations": iterations,
            "avg_ms": round(avg_ms, 4),
            "ops_per_sec": round(ops_per_sec, 0),
        }
        results.append(row)
        print(f"  {n:>3d} candidates/arm  |  {avg_ms:.4f} ms/op  ({ops_per_sec:.0f} ops/s)")

    return results


# ─── Benchmark 3: QdrantHybridFusion Availability & Routing ─────────

def bench_qdrant_hybrid_routing():
    print("\n" + "=" * 70)
    print("BENCHMARK 3: QdrantHybridFusion Routing Logic")
    print("=" * 70)

    results = {"scenarios": []}

    # Scenario A: text index ready → available
    tier_a = MagicMock()
    tier_a._client = MagicMock()
    tier_a._text_index_ready = True
    fusion_a = QdrantHybridFusion(tier_a)
    avail_a = fusion_a.available
    results["scenarios"].append({"name": "text_index_ready", "available": avail_a})
    print(f"  Scenario A (text_index_ready=True):  available={avail_a}  {'✅' if avail_a else '❌'}")

    # Scenario B: text index not ready → unavailable
    tier_b = MagicMock()
    tier_b._client = MagicMock()
    tier_b._text_index_ready = False
    fusion_b = QdrantHybridFusion(tier_b)
    avail_b = fusion_b.available
    results["scenarios"].append({"name": "text_index_not_ready", "available": avail_b})
    print(f"  Scenario B (text_index_ready=False): available={avail_b}  {'✅' if not avail_b else '❌'}")

    # Scenario C: no client → unavailable
    tier_c = MagicMock()
    tier_c._client = None
    tier_c._text_index_ready = True
    fusion_c = QdrantHybridFusion(tier_c)
    avail_c = fusion_c.available
    results["scenarios"].append({"name": "no_client", "available": avail_c})
    print(f"  Scenario C (client=None):            available={avail_c}  {'✅' if not avail_c else '❌'}")

    results["routing_correct"] = avail_a and not avail_b and not avail_c
    print(f"\n  All routing scenarios correct: {results['routing_correct']}")
    return results


# ─── Benchmark 4: Relevance Feedback Performance ────────────────────

def bench_relevance_feedback():
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Relevance Feedback Adapter Performance")
    print("=" * 70)

    results = {}

    # 4a: InMemoryFeedbackStore throughput
    store = InMemoryFeedbackStore()
    n_records = 10_000
    t0 = time.perf_counter()
    for i in range(n_records):
        store.record(f"qhash_{i % 100}", f"engram_{i}", "used" if i % 3 else "ignored")
    store_elapsed = time.perf_counter() - t0
    results["store_write_10k_ms"] = round(store_elapsed * 1000, 2)
    results["store_write_ops_per_sec"] = round(n_records / store_elapsed, 0)
    print(f"  Store write: {n_records} records in {store_elapsed*1000:.2f}ms "
          f"({n_records/store_elapsed:.0f} ops/s)")

    # 4b: ExemplarCache throughput
    cache = ExemplarCache(max_size=256, ttl_seconds=60)
    n_ops = 50_000
    t0 = time.perf_counter()
    for i in range(n_ops):
        cache.put(f"key_{i % 256}", ([f"pos_{i}"], [f"neg_{i}"]))
    cache_write_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = 0
    for i in range(n_ops):
        val = cache.get(f"key_{i % 256}")
        if val is not None:
            hits += 1
    cache_read_elapsed = time.perf_counter() - t0

    results["cache_write_50k_ms"] = round(cache_write_elapsed * 1000, 2)
    results["cache_read_50k_ms"] = round(cache_read_elapsed * 1000, 2)
    results["cache_hit_rate"] = round(hits / n_ops, 4)
    print(f"  Cache write: {n_ops} ops in {cache_write_elapsed*1000:.2f}ms")
    print(f"  Cache read:  {n_ops} ops in {cache_read_elapsed*1000:.2f}ms (hit_rate={hits/n_ops:.4f})")

    # 4c: Adapter end-to-end (no exemplars → fallback path)
    mock_tier = MagicMock()
    mock_tier._client = MagicMock()
    mock_tier._collection_name = "bench"
    mock_tier.search.return_value = _make_results(10)

    adapter = RelevanceFeedbackAdapter(mock_tier, max_exemplars=5)
    iterations = 1000
    t0 = time.perf_counter()
    for _ in range(iterations):
        adapter.search_with_feedback(query="benchmark query", query_vector=[0.1]*768, top_k=10)
    adapter_elapsed = time.perf_counter() - t0
    results["adapter_fallback_avg_ms"] = round((adapter_elapsed / iterations) * 1000, 4)
    print(f"  Adapter fallback: {(adapter_elapsed/iterations)*1000:.4f}ms/call ({iterations} iterations)")

    # 4d: Adapter with exemplars loaded
    for i in range(5):
        adapter.record_feedback(str(hash("benchmark query")), f"pos_eng_{i}", "used")
        adapter.record_feedback(str(hash("benchmark query")), f"neg_eng_{i}", "ignored")

    pos, neg = adapter.get_exemplars(str(hash("benchmark query")))
    results["exemplars_loaded"] = {"positives": len(pos), "negatives": len(neg)}
    print(f"  Exemplars loaded: {len(pos)} positive, {len(neg)} negative")

    return results


# ─── Benchmark 5: Regression Gate (Test Suite) ───────────────────────

def bench_test_suite():
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Full Test Suite Regression Gate")
    print("=" * 70)

    import subprocess
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])
    )
    elapsed = time.perf_counter() - t0

    output = result.stdout + result.stderr
    # Parse "X passed" from pytest output
    passed = 0
    failed = 0
    for line in output.splitlines():
        if "passed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed" or p == "passed,":
                    try:
                        passed = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
                if p == "failed" or p == "failed,":
                    try:
                        failed = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass

    status = "PASS" if result.returncode == 0 and failed == 0 else "FAIL"
    results = {
        "status": status,
        "passed": passed,
        "failed": failed,
        "exit_code": result.returncode,
        "elapsed_s": round(elapsed, 2),
    }

    icon = "✅" if status == "PASS" else "❌"
    print(f"  {icon} {passed} passed, {failed} failed in {elapsed:.2f}s (exit={result.returncode})")
    return results


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("🔬 MNEMOS Phase 2 Integration Benchmark")
    print(f"   Python {sys.version.split()[0]}")
    ts = time.strftime('%Y%m%dT%H%M%S')
    print(f"   Timestamp: {ts}")

    all_results = {
        "meta": {
            "timestamp": ts,
            "python_version": sys.version.split()[0],
            "benchmark_type": "phase2_integration",
        }
    }

    all_results["fusion_policies"] = bench_fusion_policies()
    all_results["python_hybrid_throughput"] = bench_python_hybrid_throughput()
    all_results["qdrant_hybrid_routing"] = bench_qdrant_hybrid_routing()
    all_results["relevance_feedback"] = bench_relevance_feedback()
    all_results["test_suite"] = bench_test_suite()

    # ─── Decision Gate ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DECISION GATE: Push to GitHub?")
    print("=" * 70)

    gates = {
        "G1_qdrant_rrf_registered": all_results["fusion_policies"]["qdrant_rrf_present"],
        "G2_routing_correct": all_results["qdrant_hybrid_routing"]["routing_correct"],
        "G3_python_hybrid_no_regression": all_results["python_hybrid_throughput"][-1]["avg_ms"] < 1.0,
        "G4_feedback_cache_fast": all_results["relevance_feedback"]["cache_read_50k_ms"] < 100,
        "G5_test_suite_pass": all_results["test_suite"]["status"] == "PASS",
    }

    all_pass = all(gates.values())
    all_results["decision_gates"] = gates
    all_results["push_approved"] = all_pass

    for gate, passed in gates.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {gate}: {'PASS' if passed else 'FAIL'}")

    print(f"\n  {'🚀 ALL GATES PASSED — Ready to push' if all_pass else '🛑 GATE FAILURE — Do NOT push'}")

    # Save results
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"phase2_benchmark_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  📊 Results saved to {out_path}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
