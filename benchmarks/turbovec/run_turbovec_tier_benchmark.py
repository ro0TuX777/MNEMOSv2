"""Run benchmarks on the TurbovecTier."""
import argparse
import os
import sys
import time
import json
import datetime
import numpy as np

from mnemos.retrieval.turbovec_config import TurbovecConfig
from mnemos.retrieval.turbovec_tier import TurbovecTier
from mnemos.retrieval.turbovec_fusion import TurbovecFusion

def load_or_generate_corpus(size, dim):
    np.random.seed(42)
    families = [
        "policy_threshold",
        "technical_reference",
        "governance_warning",
        "source_attribution",
        "lexical_exact_match",
        "semantic_near_match",
        "metadata_filter_match",
        "deleted_record_exclusion"
    ]
    corpus = []
    for i in range(size):
        family = families[i % len(families)]
        vec = np.random.randn(dim).astype(np.float32)
        vec_norm = np.linalg.norm(vec)
        vec = (vec / vec_norm).tolist()
        content = f"This is a {family} document. Some standard content for index {i}."
        if family == "lexical_exact_match":
            content += " exact_magic_word_123"
        record = {
            "uuid": f"uuid-{i}",
            "embedding": vec,
            "content": content,
            "source_uri": f"doc://{family}/{i}",
            "metadata": {"family": family, "tag": "A" if i % 2 == 0 else "B"},
            "governance": {"clearance": "high" if i % 10 == 0 else "standard"},
            "content_hash": "hash"
        }
        corpus.append(record)
    return corpus

def main():
    parser = argparse.ArgumentParser(description="Turbovec Tier Benchmark")
    parser.add_argument("--adapter", choices=["mock", "real"], default="mock", help="Adapter type to use")
    parser.add_argument("--corpus-size", type=int, default=10000, help="Number of engrams to generate")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--bit-width", type=int, default=4, help="Quantization bit width")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results to retrieve")
    parser.add_argument("--filters", choices=["on", "off"], default="on", help="Enable metadata filters")
    parser.add_argument("--hybrid", choices=["on", "off"], default="on", help="Enable Hybrid RRF")
    parser.add_argument("--out", type=str, default="benchmarks/outputs/raw", help="Output directory for raw metrics")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    summary_dir = "benchmarks/outputs/summaries"
    os.makedirs(summary_dir, exist_ok=True)
    
    print(f"Starting Turbovec Benchmark")
    print(f"Adapter: {args.adapter}")
    
    if args.adapter == "real":
        try:
            import turbovec
            print("Successfully imported turbovec package.")
        except ImportError:
            print("ERROR: turbovec package is not installed, but --adapter real was requested.")
            sys.exit(1)
            
    print("Generating corpus...")
    corpus = load_or_generate_corpus(args.corpus_size, args.embedding_dim)
    
    profile_dir = os.path.join(args.out, "run_db")
    config = TurbovecConfig(embedding_dim=args.embedding_dim, bit_width=args.bit_width, storage_path=profile_dir)
    tier = TurbovecTier(config, use_mock=(args.adapter == "mock"))
    
    if args.hybrid == "on":
        fusion = TurbovecFusion(tier)
        
    metrics = {
        "adapter": args.adapter,
        "corpus_size": args.corpus_size,
        "embedding_dim": args.embedding_dim,
        "hybrid": args.hybrid,
        "filters": args.filters
    }
    
    # 1. Ingestion
    print("Indexing corpus...")
    t0 = time.time()
    tier.index(corpus)
    t1 = time.time()
    metrics["ingest_time_sec"] = t1 - t0
    metrics["ingest_rate"] = args.corpus_size / (t1 - t0)
    
    # 2. Dense Search
    dense_latencies = []
    dense_hits = []
    for _ in range(20):
        query_emb = corpus[np.random.randint(0, args.corpus_size)]["embedding"]
        t0 = time.time()
        dense_hits = tier.search(query_emb, args.top_k, filters={"tag": "A"} if args.filters == "on" else None)
        t1 = time.time()
        dense_latencies.append((t1 - t0) * 1000)
        
    metrics["dense_latency"] = {
        "p50": float(np.percentile(dense_latencies, 50)),
        "p95": float(np.percentile(dense_latencies, 95)),
        "p99": float(np.percentile(dense_latencies, 99))
    }
    metrics["dense_search_matches"] = len(dense_hits)
    
    # 3. Hybrid Search
    if args.hybrid == "on":
        hybrid_latencies = []
        hybrid_hits = []
        for _ in range(20):
            query_emb = corpus[np.random.randint(0, args.corpus_size)]["embedding"]
            t0 = time.time()
            hybrid_hits = fusion.search("exact_magic_word_123", query_emb, top_k=args.top_k, 
                                        filters={"tag": "A"} if args.filters == "on" else None)
            t1 = time.time()
            hybrid_latencies.append((t1 - t0) * 1000)
            
        metrics["hybrid_latency"] = {
            "p50": float(np.percentile(hybrid_latencies, 50)),
            "p95": float(np.percentile(hybrid_latencies, 95)),
            "p99": float(np.percentile(hybrid_latencies, 99))
        }
        metrics["hybrid_search_matches"] = len(hybrid_hits)
        
    # 4. Soft delete exclusion
    tier.delete("uuid-0")
    t0 = time.time()
    query_emb = corpus[0]["embedding"]
    dense_hits_after = tier.search(query_emb, args.top_k)
    t1 = time.time()
    metrics["deleted_exclusion_latency_ms"] = (t1 - t0) * 1000
    metrics["deleted_successfully_excluded"] = all(h.engram_uuid != "uuid-0" for h in dense_hits_after)
    
    # 5. Persistence
    t0 = time.time()
    tier.save(profile_dir)
    t1 = time.time()
    metrics["save_time_sec"] = t1 - t0
    
    t0 = time.time()
    loaded_tier = TurbovecTier.load(profile_dir, use_mock=(args.adapter == "mock"))
    t1 = time.time()
    metrics["load_time_sec"] = t1 - t0
    
    metrics["persistence_ok"] = (loaded_tier.health()["engram_count"] == args.corpus_size - 1)
    metrics["index_size_mb"] = os.path.getsize(os.path.join(profile_dir, "index.tvim")) / (1024 * 1024)
    metrics["metadata_size_mb"] = os.path.getsize(os.path.join(profile_dir, "metadata.sqlite")) / (1024 * 1024)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_json_path = os.path.join(args.out, f"metrics_{timestamp}.json")
    with open(raw_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Write Summary Markdown
    summary_path = os.path.join(summary_dir, f"turbovec_tq1_{args.adapter}_benchmark_{timestamp}.md")
    with open(summary_path, "w") as f:
        f.write(f"# Turbovec TQ-1 {args.adapter.capitalize()} Benchmark\n\n")
        if args.adapter == "mock":
            f.write("> **NOTICE:** This run uses MockDenseIndexAdapter.\n")
            f.write("> It validates benchmark mechanics and MNEMOS sidecar/fusion correctness.\n")
            f.write("> It does not measure real turbovec speed, compression, SIMD behavior, or disk footprint.\n\n")
        f.write("## Configuration\n")
        f.write(f"- Corpus Size: {args.corpus_size}\n")
        f.write(f"- Embedding Dim: {args.embedding_dim}\n")
        f.write(f"- Hybrid: {args.hybrid}\n")
        f.write(f"- Filters: {args.filters}\n\n")
        f.write("## Metrics\n")
        f.write(f"- Ingestion Rate: {metrics.get('ingest_rate', 0):.2f} docs/sec\n")
        f.write(f"- Dense Latency: {metrics.get('dense_search_latency_ms', 0):.2f} ms\n")
        if "hybrid_search_latency_ms" in metrics:
            f.write(f"- Hybrid Latency: {metrics['hybrid_search_latency_ms']:.2f} ms\n")
        f.write(f"- Delete Exclusion Works: {metrics.get('deleted_successfully_excluded', False)}\n")
        f.write(f"- Persistence Works: {metrics.get('persistence_ok', False)}\n")
        
    print(f"Benchmark completed successfully. Summary written to {summary_path}")

if __name__ == "__main__":
    main()
