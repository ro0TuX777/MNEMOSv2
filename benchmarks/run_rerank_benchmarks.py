import json
import time
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from itertools import islice

from sentence_transformers import SentenceTransformer, CrossEncoder

def calculate_metrics(relevant_ids, retrieved_ids, hard_negative_ids, k):
    """Calculates info retrieval metrics at K. Returns MRR, nDCG, Recall, Hits, HN_reject"""
    retrieved = retrieved_ids[:k]
    
    # Hits
    hit = 0
    mrr = 0.0
    dcg = 0.0
    idcg = 0.0
    
    # For nDCG we assume relevance = 1 for rel_ids, 0 otherwise
    # IDCG with single relevant item is 1.0 if we only have 1 rel item.
    # We might have multiple rel items? Yes, len(relevant_ids)
    
    for i, r_id in enumerate(retrieved):
        if r_id in relevant_ids:
            hit = 1
            if mrr == 0.0:
                mrr = 1.0 / (i + 1)
            dcg += 1.0 / np.log2(i + 2)
            
    for i in range(min(k, len(relevant_ids))):
         idcg += 1.0 / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    recall = len(set(retrieved) & set(relevant_ids)) / float(len(relevant_ids)) if relevant_ids else 0.0
    hit_1 = 1 if retrieved and retrieved[0] in relevant_ids else 0
    hit_3 = 1 if len(set(retrieved[:3]) & set(relevant_ids)) > 0 else 0
    
    # Hard negative rejection rate = percentage of target hard negatives NOT in top K
    # Wait, the definition: "improved hard-negative rejection rate". 
    # Usually, rejection rate is 1.0 - (hard negatives found / total hard negatives)
    hn_found = len(set(retrieved) & set(hard_negative_ids))
    hn_total = len(hard_negative_ids)
    hn_reject = 1.0 - (hn_found / float(hn_total)) if hn_total > 0 else None
    
    return {
        f"mrr_{k}": mrr,
        f"ndcg_{k}": ndcg,
        f"recall_{k}": recall,
        f"hit_1": hit_1,
        f"hit_3": hit_3,
        f"hn_reject_{k}": hn_reject
    }

def run_benchmark_for_length(length_mode, dense_model, rerank_model, base_dir):
    data_path = base_dir / f"datasets/rerank_dataset_{length_mode}.jsonl"
    
    queries = []
    corpus_dict = {}
    
    # Load dataset
    with open(data_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            queries.append(rec)
            for doc in rec["documents"]:
                corpus_dict[doc["chunk_id"]] = doc["text"]

    corpus_ids = list(corpus_dict.keys())
    corpus_texts = [corpus_dict[cid] for cid in corpus_ids]
    
    print(f"[{length_mode}] Encoding {len(corpus_texts)} corpus chunks...")
    corpus_embeddings = dense_model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
    
    aggregated_metrics = defaultdict(lambda: defaultdict(list))
    latencies = defaultdict(list)
    
    candidate_recalls = {"20": [], "50": [], "100": []}
    
    print(f"[{length_mode}] Running queries...")
    for q_idx, rec in enumerate(queries):
        if q_idx % 50 == 0:
            print(f"[{length_mode}] Processed {q_idx}/{len(queries)} queries")
                
        q_text = rec["query_text"]
        rel_ids = rec["relevant_chunk_ids"]
        hn_ids = rec["hard_negative_chunk_ids"]
        family = rec["query_family"]
        track = rec["track"]
        
        # Dense Retrieval (Stage 1)
        t0 = time.perf_counter()
        q_emb = dense_model.encode(q_text, convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(corpus_embeddings, q_emb)
        top_100_idx = np.argsort(scores)[::-1][:100]
        t1 = time.perf_counter()
        
        dense_lat = (t1 - t0) * 1000
        latencies["dense"].append(dense_lat)
        
        top_100_ids = [corpus_ids[idx] for idx in top_100_idx]
        
        # Candidate Recall
        for k in [20, 50, 100]:
            found = 1 if len(set(top_100_ids[:k]) & set(rel_ids)) > 0 else 0
            candidate_recalls[str(k)].append(found)
            
        # Dense baseline metrics (k=10)
        dm = calculate_metrics(rel_ids, top_100_ids, hn_ids, 10)
        dm["recall_50"] = calculate_metrics(rel_ids, top_100_ids, hn_ids, 50)["recall_50"]
        for metric, val in dm.items():
            if val is not None:
                aggregated_metrics["dense_only"][metric].append((track, family, val))

        # Cross-Encoder (Stage 2)
        for depth in [20, 50, 100]:
            candidate_ids = top_100_ids[:depth]
            candidate_texts = [corpus_dict[cid] for cid in candidate_ids]
            pairs = [[q_text, text] for text in candidate_texts]
            
            t2 = time.perf_counter()
            ce_scores = rerank_model.predict(pairs)
            # sort by ce_score (highest first)
            reranked_indices = np.argsort(ce_scores)[::-1]
            t3 = time.perf_counter()
            
            ce_lat = (t3 - t2) * 1000
            # Total latency = dense + ce
            latencies[f"rerank_{depth}"].append(dense_lat + ce_lat)
            
            reranked_ids = [candidate_ids[idx] for idx in reranked_indices]
            
            rm = calculate_metrics(rel_ids, reranked_ids, hn_ids, 10)
            rm["recall_50"] = calculate_metrics(rel_ids, reranked_ids, hn_ids, 50)["recall_50"]
            for metric, val in rm.items():
                if val is not None:
                    aggregated_metrics[f"rerank_{depth}"][metric].append((track, family, val))

    summary_stats = {
         "latencies": {},
         "candidate_recall": {
              "recall@20": np.mean(candidate_recalls["20"]),
              "recall@50": np.mean(candidate_recalls["50"]),
              "recall@100": np.mean(candidate_recalls["100"]),
         },
         "configurations": {}
    }
    
    for mode, lats in latencies.items():
        summary_stats["latencies"][mode] = {
            "p50": np.percentile(lats, 50),
            "p95": np.percentile(lats, 95),
            "p99": np.percentile(lats, 99)
        }
        
    for config, metrics in aggregated_metrics.items():
        config_summary = {}
        for metric, values in metrics.items():
            # Overall mean
            config_summary[metric] = {"overall": np.mean([v[2] for v in values])}
            
            # By track
            by_track = defaultdict(list)
            by_family = defaultdict(list)
            for track, fam, val in values:
                by_track[track].append(val)
                by_family[fam].append(val)
                
            config_summary[metric]["by_track"] = {t: np.mean(vs) for t, vs in by_track.items()}
            config_summary[metric]["by_family"] = {f: np.mean(vs) for f, vs in by_family.items()}
            
        summary_stats["configurations"][config] = config_summary
        
    return summary_stats

def main():
    base_dir = Path(__file__).resolve().parent

    print("Loading Dense Model (BAAI/bge-base-en-v1.5) ...")
    dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    print("Loading Reranker Model (BAAI/bge-reranker-base) ...")
    rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    
    all_results = {}
    
    for length_mode in ["short", "medium", "long"]:
        print(f"\n=== Starting Benchmark for {length_mode.upper()} chunks ===")
        stats = run_benchmark_for_length(length_mode, dense_model, rerank_model, base_dir)
        all_results[length_mode] = stats
        
    out_dir = base_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"benchmark_results_{ts}.json"
    
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nSuccessfully wrote results to {out_path}")

if __name__ == "__main__":
    main()
