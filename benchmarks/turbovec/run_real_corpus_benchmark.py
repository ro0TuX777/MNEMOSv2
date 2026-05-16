import os
import glob
import time
import uuid
import hashlib
import json
import argparse
import datetime
from typing import List, Dict

import pypdf
from sentence_transformers import SentenceTransformer

from mnemos.retrieval.turbovec_config import TurbovecConfig
from mnemos.retrieval.turbovec_tier import TurbovecTier
from mnemos.retrieval.turbovec_fusion import TurbovecFusion

def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 10:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Real MNEMOS PDF Benchmark")
    parser.add_argument("--folders", nargs='+', required=True, help="Folders containing PDFs")
    parser.add_argument("--out", type=str, default="benchmarks/outputs/real_corpus", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    summary_dir = "benchmarks/outputs/summaries"
    os.makedirs(summary_dir, exist_ok=True)
    
    print("Loading embedding model (BAAI/bge-base-en-v1.5)...")
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    corpus_records = []
    
    for folder in args.folders:
        category = os.path.basename(os.path.normpath(folder))
        print(f"Processing folder: {folder} (Category: {category})")
        pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
        
        for filepath in pdf_files:
            print(f"  Extracting {os.path.basename(filepath)}...")
            text = extract_text_from_pdf(filepath)
            chunks = chunk_text(text)
            
            if not chunks:
                continue
                
            print(f"  Encoding {len(chunks)} chunks...")
            embeddings = model.encode(chunks, normalize_embeddings=True)
            
            for chunk_text_str, emb in zip(chunks, embeddings):
                record = {
                    "uuid": str(uuid.uuid4()),
                    "embedding": emb.tolist(),
                    "content": chunk_text_str,
                    "source_uri": f"file://{filepath}",
                    "metadata": {"category": category, "filename": os.path.basename(filepath)},
                    "governance": {"clearance": "standard"},
                    "content_hash": hashlib.sha256(chunk_text_str.encode()).hexdigest()
                }
                corpus_records.append(record)
                
    total_records = len(corpus_records)
    print(f"Total extracted engrams: {total_records}")
    if total_records == 0:
        print("No engrams generated. Exiting.")
        return
        
    profile_dir = os.path.join(args.out, "run_db")
    config = TurbovecConfig(embedding_dim=768, bit_width=4, storage_path=profile_dir)
    tier = TurbovecTier(config, use_mock=False)
    fusion = TurbovecFusion(tier)
    
    print("Indexing into TurbovecTier...")
    t0 = time.time()
    tier.index(corpus_records)
    t1 = time.time()
    ingest_time = t1 - t0
    ingest_rate = total_records / ingest_time
    print(f"Indexed in {ingest_time:.2f}s ({ingest_rate:.2f} docs/sec)")
    
    # Run a test query
    test_query = "What is signal intelligence?"
    query_emb = model.encode([test_query], normalize_embeddings=True)[0].tolist()
    
    print(f"Running Hybrid Search for: '{test_query}'")
    t0 = time.time()
    hybrid_hits = fusion.search(test_query, query_emb, top_k=5)
    t1 = time.time()
    hybrid_latency = (t1 - t0) * 1000
    print(f"Hybrid search took {hybrid_latency:.2f} ms. Found {len(hybrid_hits)} hits.")
    
    tier.save(profile_dir)
    
    index_size = os.path.getsize(os.path.join(profile_dir, "index.tvim")) / (1024 * 1024)
    db_size = os.path.getsize(os.path.join(profile_dir, "metadata.sqlite")) / (1024 * 1024)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(summary_dir, f"turbovec_tq1_real_corpus_{timestamp}.md")
    
    with open(summary_path, "w") as f:
        f.write("# Turbovec TQ-1 Real Corpus Benchmark\n\n")
        f.write("## Dataset\n")
        f.write(f"- Folders: {', '.join(args.folders)}\n")
        f.write(f"- Total Engrams: {total_records}\n\n")
        f.write("## Ingestion Metrics\n")
        f.write(f"- Ingestion Time: {ingest_time:.2f} seconds\n")
        f.write(f"- Ingestion Rate: {ingest_rate:.2f} docs/sec\n")
        f.write(f"- Index Size (`index.tvim`): {index_size:.2f} MB\n")
        f.write(f"- Metadata Size (`metadata.sqlite`): {db_size:.2f} MB\n\n")
        f.write("## Retrieval Performance\n")
        f.write(f"- Query: '{test_query}'\n")
        f.write(f"- Hybrid Latency: {hybrid_latency:.2f} ms\n")
        f.write("### Top Hits:\n")
        for i, hit in enumerate(hybrid_hits):
            f.write(f"{i+1}. Source: {hit.metadata.get('filename')} (Score: {hit.score:.4f})\n")
            f.write(f"   Preview: {hit.content[:150]}...\n")
            
    print(f"Benchmark summary written to {summary_path}")

if __name__ == "__main__":
    main()
