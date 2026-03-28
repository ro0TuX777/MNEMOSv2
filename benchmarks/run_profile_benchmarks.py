"""
MNEMOS Profile Benchmark Suite - Entry Point
================================================

Runs the 4-track profile benchmark suite:
  Track 1: Profile Retrieval (Core vs Governance)
  Track 2: ColBERT Reranking Uplift
  Track 3: Installer Overhead
  Track 4: Migration & Recovery

Usage:
    python benchmarks/run_profile_benchmarks.py                  # All tracks
    python benchmarks/run_profile_benchmarks.py --track retrieval # Single track
    python benchmarks/run_profile_benchmarks.py --track installer # No Docker needed
    python benchmarks/run_profile_benchmarks.py --corpus-size 1000 # Small corpus
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.datasets.corpus_generator import generate_corpus, save_corpus
from benchmarks.datasets.query_generator import generate_queries, save_queries
from benchmarks.metrics.system_metrics import capture_environment
from benchmarks.metrics.reporting import save_raw_results, generate_markdown_report


TRACKS = ["retrieval", "rerank", "installer", "migration"]


def _print_header():
    print()
    print("=" * 50)
    print("  MNEMOS Profile Benchmark Suite")
    print("  Proving when each profile wins")
    print("=" * 50)
    print()


def run_suite(
    tracks: list = None,
    corpus_size: int = 10_000,
    n_runs: int = 5,
    gpu_device: str = "cuda",
    seed: int = 42,
):
    """Run the benchmark suite."""
    if tracks is None:
        tracks = TRACKS

    _print_header()

    env = capture_environment()
    print(f"  Environment: {env.gpu_name or 'No GPU'}, "
          f"{env.ram_gb}GB RAM, {env.cpu_cores} cores")
    print(f"  Python {env.python_version}, NumPy {env.numpy_version}")
    print(f"  Timestamp: {env.timestamp}")
    print()

    # Generate corpus
    print(f"  Generating corpus ({corpus_size} engrams, 4 domains)...")
    corpus = generate_corpus(n_docs=corpus_size, seed=seed)
    print(f"    [OK] {len(corpus)} engrams generated")

    # Save corpus for reproducibility
    datasets_dir = PROJECT_ROOT / "benchmarks" / "outputs" / "datasets"
    corpus_path = datasets_dir / "corpus.json"
    save_corpus(corpus, corpus_path)

    # Generate queries
    print(f"  Generating queries (3 regimes × 100 queries)...")
    queries = generate_queries(corpus, n_per_regime=100, seed=99)
    queries_path = datasets_dir / "queries.json"
    save_queries(queries, queries_path)
    print(f"    [OK] {len(queries)} queries generated "
          f"(semantic: {sum(1 for q in queries if q.regime == 'semantic')}, "
          f"light: {sum(1 for q in queries if q.regime == 'light_filter')}, "
          f"heavy: {sum(1 for q in queries if q.regime == 'heavy_filter')})")

    results = {}

    # ─────────── Track 1: Retrieval ───────────
    if "retrieval" in tracks:
        from benchmarks.runners.retrieval_runner import run_retrieval_track
        results["retrieval"] = run_retrieval_track(
            corpus, queries, n_runs=n_runs, gpu_device=gpu_device,
        )

    # ─────────── Track 2: Reranking ───────────
    if "rerank" in tracks:
        from benchmarks.runners.rerank_runner import run_rerank_benchmark
        results["rerank"] = run_rerank_benchmark(
            "qdrant", queries, gpu_device=gpu_device,
        )

    # ─────────── Track 3: Installer ───────────
    if "installer" in tracks:
        from benchmarks.runners.install_runner import run_installer_track
        results["installer"] = run_installer_track()

    # ─────────── Track 4: Migration ───────────
    if "migration" in tracks:
        from benchmarks.runners.migration_runner import run_migration_track
        results["migration"] = run_migration_track(
            corpus, gpu_device=gpu_device,
        )

    # ─────────── Save results ───────────
    output_dir = PROJECT_ROOT / "benchmarks" / "outputs"
    raw_path = save_raw_results(results, output_dir / "raw")
    report_path = generate_markdown_report(results, output_dir / "summaries")

    print("\n" + "=" * 70)
    print("  [OK] Benchmark suite complete!")
    print(f"  Raw:     {raw_path}")
    print(f"  Report:  {report_path}")
    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MNEMOS Profile Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tracks:
  retrieval   Core (Qdrant) vs Governance (pgvector) - requires Docker
  rerank      ColBERT reranking uplift - requires Docker + colbert-ir
  installer   Installer vs manual deployment - NO Docker required
  migration   Profile migration behavior - requires Docker
""",
    )
    parser.add_argument(
        "--track", type=str, choices=TRACKS,
        help="Run only a specific track (default: all)",
    )
    parser.add_argument(
        "--corpus-size", type=int, default=10_000,
        help="Number of synthetic engrams to generate (default: 10000)",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of repeated search runs for median calculation (default: 5)",
    )
    parser.add_argument(
        "--gpu", type=str, default="cuda",
        help="GPU device (default: cuda, use 'cpu' for CPU-only)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for corpus generation (default: 42)",
    )
    args = parser.parse_args()

    tracks = [args.track] if args.track else TRACKS

    run_suite(
        tracks=tracks,
        corpus_size=args.corpus_size,
        n_runs=args.runs,
        gpu_device=args.gpu,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
