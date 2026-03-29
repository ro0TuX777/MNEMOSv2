"""
MNEMOS Profile Benchmark Suite - Entry Point
================================================

Runs the 5-track profile benchmark suite:
  Track 1: Profile Retrieval (Core vs Governance)
  Track 2: ColBERT Reranking Uplift
  Track 3: Installer Overhead
  Track 4: Migration & Recovery
  Track 5: Hybrid Retrieval (lexical + semantic fusion)

Usage:
    python benchmarks/run_profile_benchmarks.py                   # All tracks, synthetic
    python benchmarks/run_profile_benchmarks.py --track retrieval # Single track
    python benchmarks/run_profile_benchmarks.py --track hybrid    # Gate C hybrid track
    python benchmarks/run_profile_benchmarks.py --track installer # No Docker needed
    python benchmarks/run_profile_benchmarks.py --corpus-size 1000 # Small synthetic
    python benchmarks/run_profile_benchmarks.py --corpus-type real --pdf-dir /path/to/pdfs
"""

import argparse
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.datasets.corpus_generator import generate_corpus, save_corpus
from benchmarks.datasets.query_generator import generate_queries, save_queries
from benchmarks.metrics.system_metrics import capture_environment
from benchmarks.metrics.reporting import save_raw_results, generate_markdown_report


TRACKS = ["retrieval", "rerank", "installer", "migration", "hybrid"]


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
    corpus_type: str = "synthetic",
    pdf_dir: str = None,
    n_runs: int = 5,
    gpu_device: str = "cuda",
    seed: int = 42,
    rerank_audit: bool = False,
    rerank_audit_queries: int = 10,
    rerank_audit_depth: int = 50,
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

    # Build corpus
    datasets_dir = PROJECT_ROOT / "benchmarks" / "outputs" / "datasets"

    if corpus_type == "real":
        if not pdf_dir:
            print("  [ERROR] --pdf-dir required when --corpus-type is 'real'")
            sys.exit(1)
        from benchmarks.datasets.pdf_loader import load_pdf_corpus
        print(f"  Loading real-world corpus from {pdf_dir}...")
        corpus = load_pdf_corpus(pdf_dir)
        corpus_path = datasets_dir / "corpus_real.json"
        save_corpus(corpus, corpus_path)
        print(f"    Corpus type: real-world PDFs ({len(corpus)} chunks)")
    else:
        print(f"  Generating synthetic corpus ({corpus_size} engrams, 4 domains)...")
        corpus = generate_corpus(n_docs=corpus_size, seed=seed)
        corpus_path = datasets_dir / "corpus.json"
        save_corpus(corpus, corpus_path)
        print(f"    [OK] {len(corpus)} engrams generated")

    if len(corpus) == 0:
        print("  [ERROR] Corpus is empty. Cannot generate benchmark queries.")
        if corpus_type == "real":
            print("  [HINT] Check --pdf-dir points to a directory with readable PDF files.")
        else:
            print("  [HINT] Increase --corpus-size to a positive value.")
        sys.exit(1)

    # Generate baseline retrieval queries from corpus
    print("  Generating queries (3 regimes x 100 queries)...")
    queries = generate_queries(corpus, n_per_regime=100, seed=99)
    queries_path = datasets_dir / "queries.json"
    save_queries(queries, queries_path)
    print(f"    [OK] {len(queries)} queries generated "
          f"(semantic: {sum(1 for q in queries if q.regime == 'semantic')}, "
          f"light: {sum(1 for q in queries if q.regime == 'light_filter')}, "
          f"heavy: {sum(1 for q in queries if q.regime == 'heavy_filter')})")

    results = {}

    # Track 1: Retrieval
    if "retrieval" in tracks:
        from benchmarks.runners.retrieval_runner import run_retrieval_track
        results["retrieval"] = run_retrieval_track(
            corpus, queries, n_runs=n_runs, gpu_device=gpu_device,
        )

    # Track 2: Reranking
    if "rerank" in tracks:
        from benchmarks.runners.rerank_runner import run_rerank_benchmark
        results["rerank"] = run_rerank_benchmark(
            queries=queries,
            corpus=corpus,
            gpu_device=gpu_device,
            enable_audit=rerank_audit,
            audit_queries=rerank_audit_queries,
            audit_depth=rerank_audit_depth,
        )

    # Track 3: Installer
    if "installer" in tracks:
        from benchmarks.runners.install_runner import run_installer_track
        results["installer"] = run_installer_track()

    # Track 4: Migration
    if "migration" in tracks:
        from benchmarks.runners.migration_runner import run_migration_track
        results["migration"] = run_migration_track(
            corpus, gpu_device=gpu_device,
        )

    # Track 5: Hybrid retrieval
    if "hybrid" in tracks:
        from benchmarks.runners.hybrid_runner import run_hybrid_track
        results["hybrid"] = run_hybrid_track(
            corpus=corpus,
            n_runs=n_runs,
            gpu_device=gpu_device,
        )

    # Save results
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
  hybrid      Semantic vs lexical vs hybrid fusion policies - requires Docker
""",
    )
    parser.add_argument(
        "--track", type=str, choices=TRACKS,
        help="Run only a specific track (default: all)",
    )
    parser.add_argument(
        "--corpus-type", type=str, choices=["synthetic", "real"], default="synthetic",
        help="Corpus type: 'synthetic' (generated) or 'real' (from PDFs)",
    )
    parser.add_argument(
        "--pdf-dir", type=str, default=None,
        help="Path to directory of PDF files (required when --corpus-type is 'real')",
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
    parser.add_argument(
        "--rerank-audit", action="store_true",
        help="Enable reranker correctness audit traces (Track 2 only)",
    )
    parser.add_argument(
        "--rerank-audit-queries", type=int, default=10,
        help="Number of semantic queries for Track 2 audit traces (default: 10)",
    )
    parser.add_argument(
        "--rerank-audit-depth", type=int, default=50,
        help="Candidate depth for Track 2 audit traces (default: 50)",
    )
    args = parser.parse_args()

    tracks = [args.track] if args.track else TRACKS

    run_suite(
        tracks=tracks,
        corpus_size=args.corpus_size,
        corpus_type=args.corpus_type,
        pdf_dir=args.pdf_dir,
        n_runs=args.runs,
        gpu_device=args.gpu,
        seed=args.seed,
        rerank_audit=args.rerank_audit,
        rerank_audit_queries=args.rerank_audit_queries,
        rerank_audit_depth=args.rerank_audit_depth,
    )


if __name__ == "__main__":
    main()
