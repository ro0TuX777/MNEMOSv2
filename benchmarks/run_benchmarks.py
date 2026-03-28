"""
MNEMOS Benchmark Suite
=======================

Generates empirical benchmark data for the whitepaper.
Run: python benchmarks/run_benchmarks.py

Benchmarks:
  1. TurboQuant compression ratio & MSE (all bit-widths, multiple dimensions)
  2. TurboQuant Recall@10 (nearest-neighbour fidelity)
  3. TurboQuant encoding throughput
  4. File size comparison (float32 vs compressed .npz)
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.compression.turbo_quant import TurboQuant


def generate_corpus(n_docs: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate a synthetic corpus of random unit-norm vectors."""
    rng = np.random.RandomState(seed)
    vectors = rng.randn(n_docs, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)


# ────────────────────── Benchmark 1: Compression Ratio & MSE ──────────────────────

def bench_compression_mse():
    """Measure actual compression ratio and MSE across bit-widths and dimensions."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: TurboQuant Compression Ratio & MSE")
    print("=" * 70)

    dims = [128, 384]
    n_docs = 10_000
    results = []

    for dim in dims:
        vectors = generate_corpus(n_docs, dim, seed=42)
        original_bytes = vectors.nbytes

        for bits in [1, 2, 3, 4]:
            tq = TurboQuant(bits=bits, mode="mse", seed=42)
            qt = tq.quantize(vectors)
            recon = tq.dequantize(qt)

            # MSE (per-coordinate, on unit-norm vectors)
            mse = np.mean((vectors - recon) ** 2)

            # Cosine similarity (practical fidelity)
            dots = np.sum(vectors * recon, axis=1)
            v_norms = np.linalg.norm(vectors, axis=1)
            r_norms = np.linalg.norm(recon, axis=1)
            cosine_sims = dots / (v_norms * r_norms + 1e-10)
            avg_cosine = float(np.mean(cosine_sims))

            # File size comparison
            tmp_path = Path(tempfile.mktemp(suffix=".npz"))
            try:
                tq.save(qt, tmp_path)
                compressed_bytes = tmp_path.stat().st_size
            finally:
                tmp_path.unlink(missing_ok=True)

            raw_ratio = original_bytes / qt.codes.nbytes
            file_ratio = original_bytes / compressed_bytes

            theoretical_mse = TurboQuant._theoretical_mse(bits)
            mse_pass = "✅" if mse <= theoretical_mse * 1.5 else "⚠️"

            row = {
                "dim": dim, "bits": bits, "n_docs": n_docs,
                "mse": round(mse, 6),
                "theoretical_mse": theoretical_mse,
                "mse_pass": mse_pass,
                "avg_cosine_similarity": round(avg_cosine, 4),
                "raw_compression_ratio": round(raw_ratio, 1),
                "file_compression_ratio": round(file_ratio, 1),
                "original_mb": round(original_bytes / 1e6, 2),
                "compressed_mb": round(compressed_bytes / 1e6, 2),
            }
            results.append(row)

            print(f"  dim={dim:4d}  bits={bits}  |  MSE={mse:.6f} {mse_pass} (bound={theoretical_mse})"
                  f"  |  cosine={avg_cosine:.4f}"
                  f"  |  ratio={raw_ratio:.1f}× raw, {file_ratio:.1f}× file"
                  f"  |  {original_bytes/1e6:.1f}MB → {compressed_bytes/1e6:.2f}MB")

    return results


# ────────────────────── Benchmark 2: Recall@10 ──────────────────────

def bench_recall_at_k(k: int = 10):
    """Measure search fidelity: how many true top-k neighbours survive quantisation."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK 2: Recall@{k} (Nearest-Neighbour Fidelity)")
    print("=" * 70)

    dims = [128, 384]
    n_corpus = 10_000
    n_queries = 100
    results = []

    for dim in dims:
        corpus = generate_corpus(n_corpus, dim, seed=42)
        queries = generate_corpus(n_queries, dim, seed=99)

        # Ground truth: exact top-k by cosine similarity
        exact_scores = queries @ corpus.T  # (n_queries, n_corpus)
        exact_topk = np.argsort(-exact_scores, axis=1)[:, :k]

        for bits in [1, 2, 3, 4]:
            tq = TurboQuant(bits=bits, mode="mse", seed=42)
            qt_corpus = tq.quantize(corpus)
            recon_corpus = tq.dequantize(qt_corpus)

            # Approximate top-k using reconstructed vectors
            approx_scores = queries @ recon_corpus.T
            approx_topk = np.argsort(-approx_scores, axis=1)[:, :k]

            # Recall@k: fraction of true top-k found in approximate top-k
            recalls = []
            for q in range(n_queries):
                true_set = set(exact_topk[q])
                approx_set = set(approx_topk[q])
                recalls.append(len(true_set & approx_set) / k)

            avg_recall = float(np.mean(recalls))
            std_recall = float(np.std(recalls))

            row = {
                "dim": dim, "bits": bits, "k": k,
                "n_corpus": n_corpus, "n_queries": n_queries,
                "recall_at_k": round(avg_recall, 4),
                "recall_std": round(std_recall, 4),
            }
            results.append(row)

            print(f"  dim={dim:4d}  bits={bits}  |  Recall@{k} = {avg_recall:.4f} ± {std_recall:.4f}")

    return results


# ────────────────────── Benchmark 3: Encoding Throughput ──────────────────────

def bench_throughput():
    """Measure quantization throughput (documents/second)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: TurboQuant Encoding Throughput")
    print("=" * 70)

    dims = [128, 384]
    batch_sizes = [100, 1_000, 10_000]
    results = []

    for dim in dims:
        for n in batch_sizes:
            vectors = generate_corpus(n, dim, seed=42)
            tq = TurboQuant(bits=4, mode="mse", seed=42)

            # Warm up (codebook computation)
            _ = tq.quantize(vectors[:10])

            # Timed run
            t0 = time.perf_counter()
            qt = tq.quantize(vectors)
            elapsed = time.perf_counter() - t0

            docs_per_sec = n / elapsed
            ms_per_doc = (elapsed / n) * 1000

            row = {
                "dim": dim, "n_docs": n, "bits": 4,
                "elapsed_s": round(elapsed, 4),
                "docs_per_sec": round(docs_per_sec, 0),
                "ms_per_doc": round(ms_per_doc, 3),
            }
            results.append(row)

            print(f"  dim={dim:4d}  n={n:>6d}  |  {elapsed:.4f}s  "
                  f"({docs_per_sec:.0f} docs/s, {ms_per_doc:.3f} ms/doc)")

    return results


# ────────────────────── Benchmark 4: Scale Projection ──────────────────────

def bench_scale_projection():
    """Project storage requirements at various corpus sizes."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Storage Projection at Scale")
    print("=" * 70)

    dim = 128
    corpus_sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    results = []

    for n in corpus_sizes:
        float32_bytes = n * dim * 4
        for bits in [4]:
            quant_bytes = (n * dim * bits + 7) // 8
            # Overhead: norms (4 bytes each), codebook/boundaries negligible
            total_quant = quant_bytes + n * 4
            ratio = float32_bytes / total_quant
            row = {
                "n_docs": n,
                "dim": dim,
                "bits": bits,
                "float32_mb": round(float32_bytes / 1e6, 1),
                "quantized_mb": round(total_quant / 1e6, 1),
                "ratio": round(ratio, 1),
            }
            results.append(row)

            print(f"  {n:>10,} docs × {dim}d  |  float32: {float32_bytes/1e6:>8.1f} MB  "
                  f"→  4-bit: {total_quant/1e6:>8.1f} MB  ({ratio:.1f}×)")

    return results


# ────────────────────── Main ──────────────────────

def main():
    print("🔬 MNEMOS Benchmark Suite")
    print(f"   numpy {np.__version__}, Python {sys.version.split()[0]}")
    print(f"   Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    all_results = {}

    all_results["compression_mse"] = bench_compression_mse()
    all_results["recall_at_k"] = bench_recall_at_k(k=10)
    all_results["throughput"] = bench_throughput()
    all_results["scale_projection"] = bench_scale_projection()

    # Save results as JSON
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n📊 Results saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
