import json
import random
import uuid
import sys
from pathlib import Path
from datetime import datetime

random.seed(42)

TRACKS = ["technical_longform", "dense_evidence", "hard_negative", "code_technical", "factoid_control"]
FAMILIES = ["factoid", "constraint_heavy", "multi_clause", "why_how", "hard_negative", "code_behavior"]
DIFFICULTIES = ["easy", "medium", "hard"]
DOMAINS = ["governance", "forensics", "hybrid_retrieval", "cross_encoder", "tokenization", "memory_over_maps"]

def expand_text(base_text: str, length_mode: str) -> str:
    """Expands text to approximate 'short', 'medium', or 'long' chunk lengths."""
    # A multiplier to artificially extend text logically
    pad_sentences = [
        " Furthermore, the system ensures strong consistency across partition boundaries.",
        " This approach mitigates latency spikes during high-throughput ingestion phases.",
        " Concurrently, the memory allocator reserves bounded pages for thread safety.",
        " In contrast to legacy patterns, this modern overlay reduces cache misses.",
        " Security protocols mandate encryption at rest via AES-256 for all stored artifacts.",
        " Operational metrics are heavily monitored, targeting a p99 latency below 100ms.",
        " During failover, the redundant controller takes precedence seamlessly.",
        " Detailed telemetery is exported to the central observability layer."
    ]
    
    if length_mode == "short":
        n_pad = 2
    elif length_mode == "medium":
        n_pad = 6
    else:  # long
        n_pad = 14
        
    text = base_text
    for _ in range(n_pad):
        text += random.choice(pad_sentences)
    return text

def generate_query_record(q_id, length_mode):
    track = random.choices(
        TRACKS, 
        weights=[0.30, 0.25, 0.20, 0.15, 0.10],
        k=1
    )[0]
    
    domain = random.choice(DOMAINS)
    
    if track == "technical_longform":
        family = random.choice(["why_how", "multi_clause"])
        diff = "hard"
        query_text = f"Why did we migrate {domain} to a multi-threaded asynchronous architecture, and how does it interact with the memory pool?"
        rel_text = f"The migration of {domain} to async was necessitated by thread starvation. By adopting multi-threading, we eliminated blocking I/O. The memory pool interaction is handled by direct pointer sharing, which avoids copying."
        hn_text = f"Prior to migration, {domain} used a single-threaded blocking architecture. Multi-threading is not supported backward. The memory pool was decoupled from {domain} entirely to prevent OOM errors."
    
    elif track == "dense_evidence":
        family = "constraint_heavy"
        diff = "medium"
        query_text = f"Which configuration of {domain} is best when metadata filtering matters more than raw throughput?"
        rel_text = f"When metadata filtering is the primary constraint over throughput, configuring {domain} with dense HNSW indexes and pre-filtering guarantees the highest precision."
        hn_text = f"If raw throughput is your goal regardless of metadata filtering, optimizing {domain} with product quantization and post-filtering is the best configuration."
    
    elif track == "hard_negative":
        family = "hard_negative"
        diff = "hard"
        query_text = f"What happens to {domain} losers in enforced resolution mode?"
        rel_text = f"In enforced resolution mode, {domain} losers are completely suppressed from the final candidate envelope and logged as rejected."
        hn_text = f"In soft resolution mode, {domain} losers rank lower but remain in the envelope. Enforced mode is rarely used."
        
    elif track == "code_technical":
        family = "code_behavior"
        diff = "medium"
        query_text = f"Which python module owns reranking and candidate bounding for {domain}?"
        rel_text = f"For {domain}, the reranking logic is wrapped inside `mnemos.retrieval.cross_encoder`, while candidate bounding is managed by `candidate_envelope.py`."
        hn_text = f"The `mnemos.retrieval.dense` module owns first stage retrieval for {domain}. Reranking is turned off by default."
        
    else: # factoid_control
        family = "factoid"
        diff = "easy"
        query_text = f"What database backs the {domain} ledger?"
        rel_text = f"The {domain} ledger is backed entirely by PostgreSQL using pgvector."
        hn_text = f"The {domain} cache is backed by Redis."

    # Expand texts
    rel_text = expand_text(rel_text, length_mode)
    if hn_text:
        hn_text = expand_text(hn_text, length_mode)
        
    rel_id = f"D{q_id}_R1"
    hn_id = f"D{q_id}_HN1"

    docs = [
        {"doc_id": f"D{q_id}", "chunk_id": rel_id, "source_type": "whitepaper", "title": f"{domain} Notes", "text": rel_text}
    ]
    
    hn_ids = []
    if track == "hard_negative" or random.random() < 0.5:
        docs.append({"doc_id": f"D{q_id}_H", "chunk_id": hn_id, "source_type": "whitepaper", "title": f"Old {domain}", "text": hn_text})
        hn_ids.append(hn_id)
        
    # generate random distractors for the global corpus
    pad_docs = []
    for p in range(10):
        pad_id = f"D{q_id}_P{p}"
        pad_text = expand_text(f"Topic {random.choice(DOMAINS)} has an unrelated distractor chunk. It shouldn't match anything perfectly.", length_mode)
        pad_docs.append({"doc_id": f"D{q_id}_PAD_{p}", "chunk_id": pad_id, "source_type": "wiki", "title": "Distractor", "text": pad_text})
        
    record = {
        "query_id": f"Q{q_id:04d}",
        "track": track,
        "query_family": family,
        "query_text": query_text,
        "difficulty": diff,
        "documents": docs + pad_docs,  # They will all merge into a single corpus
        "relevant_chunk_ids": [rel_id],
        "acceptable_chunk_ids": [],
        "hard_negative_chunk_ids": hn_ids,
        "metadata_filters": {},
        "notes": f"Synthetic {track} query"
    }
    return record

def main():
    base_dir = Path(__file__).resolve().parent
    base_dir.mkdir(parents=True, exist_ok=True)
    
    n_queries = 300 # 300 queries * ~11.5 docs = ~3450 docs per variant.
    
    for length_mode in ["short", "medium", "long"]:
        out_path = base_dir / f"rerank_dataset_{length_mode}.jsonl"
        with open(out_path, "w") as f:
            for i in range(1, n_queries + 1):
                rec = generate_query_record(i, length_mode)
                f.write(json.dumps(rec) + "\n")
        print(f"Generated {out_path} ({n_queries} tasks).")

if __name__ == "__main__":
    main()
