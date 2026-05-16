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
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(filepath: str, max_pages: int = 10) -> List[Dict]:
    pages = []
    try:
        with open(filepath, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for i in range(min(len(reader.pages), max_pages)):
                page_text = reader.pages[i].extract_text()
                if page_text and page_text.strip():
                    pages.append({"page": i + 1, "text": page_text})
    except Exception as e:
        raise RuntimeError(f"PDF Parse Error: {e}")
    return pages

def chunk_pages(pages: List[Dict], chunk_size: int = 250, overlap: int = 50) -> List[Dict]:
    chunks = []
    full_text = []
    page_map = [] # map word index to page number
    
    for p in pages:
        words = p["text"].split()
        full_text.extend(words)
        page_map.extend([p["page"]] * len(words))
        
    i = 0
    chunk_index = 0
    while i < len(full_text):
        chunk_words = full_text[i:i + chunk_size]
        chunk_text_str = " ".join(chunk_words)
        if len(chunk_text_str.strip()) > 20:
            start_page = page_map[i]
            end_page = page_map[min(i + len(chunk_words) - 1, len(page_map) - 1)]
            chunks.append({
                "content": chunk_text_str,
                "page_start": start_page,
                "page_end": end_page,
                "chunk_index": chunk_index
            })
            chunk_index += 1
        i += chunk_size - overlap
    return chunks

def process_dataset(name: str, folders: List[str], max_pdfs: int, max_pages: int, model: SentenceTransformer, out_dir: str):
    print(f"\nProcessing Dataset: {name}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    chunks_out = os.path.join(out_dir, f"tq13_real_pdf_chunks_{name}_{timestamp}.jsonl")
    emb_out = os.path.join(out_dir, f"tq13_real_pdf_embeddings_{name}_{timestamp}.npz")
    failures_out = os.path.join(out_dir, f"tq13_failed_parses_{name}_{timestamp}.json")
    
    failed_parses = []
    all_chunks = []
    all_embeddings = []
    all_uuids = []
    all_hashes = []
    all_uris = []
    
    for folder in folders:
        pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
        
        for filepath in pdf_files[:max_pdfs]:
            doc_name = os.path.basename(filepath)
            print(f"  Extracting {doc_name}...")
            try:
                pages = extract_text_from_pdf(filepath, max_pages=max_pages)
                if not pages:
                    failed_parses.append({"file": filepath, "error": "No text extracted"})
                    continue
                    
                doc_chunks = chunk_pages(pages)
                
                texts = [c["content"] for c in doc_chunks]
                embeddings = model.encode(texts, normalize_embeddings=True)
                
                for i, c in enumerate(doc_chunks):
                    engram_uuid = str(uuid.uuid4())
                    content_hash = hashlib.sha256(c["content"].encode()).hexdigest()
                    source_uri = f"file://{filepath}"
                    
                    chunk_record = {
                        "engram_uuid": engram_uuid,
                        "dataset": name,
                        "document_path": filepath,
                        "document_name": doc_name,
                        "source_uri": source_uri,
                        "page_start": c["page_start"],
                        "page_end": c["page_end"],
                        "chunk_index": c["chunk_index"],
                        "content": c["content"],
                        "content_hash": content_hash,
                        "metadata": {
                            "dataset": name,
                            "document_name": doc_name,
                            "page_start": c["page_start"],
                            "page_end": c["page_end"],
                            "chunk_index": c["chunk_index"],
                            "chunk_family": doc_name
                        },
                        "governance": {
                            "clearance": "standard",
                            "source_type": "pdf",
                            "parse_status": "parsed"
                        }
                    }
                    all_chunks.append(chunk_record)
                    all_embeddings.append(embeddings[i])
                    all_uuids.append(engram_uuid)
                    all_hashes.append(content_hash)
                    all_uris.append(source_uri)
                    
            except Exception as e:
                failed_parses.append({"file": filepath, "error": str(e)})
                
    # Save artifacts
    print(f"  Writing {len(all_chunks)} chunks to {chunks_out}")
    with open(chunks_out, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
            
    print(f"  Writing embeddings to {emb_out}")
    np.savez_compressed(
        emb_out,
        engram_uuids=np.array(all_uuids),
        embeddings=np.array(all_embeddings, dtype=np.float32),
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
        content_hashes=np.array(all_hashes),
        source_uris=np.array(all_uris)
    )
    
    if failed_parses:
        with open(failures_out, "w") as f:
            json.dump(failed_parses, f, indent=2)
            
    # Validation
    print(f"  Validating artifact integrity...")
    assert len(all_chunks) == len(all_uuids), "Mismatch chunk/uuid count"
    assert len(all_embeddings) == len(all_chunks), "Mismatch embedding/chunk count"
    if len(all_embeddings) > 0:
        assert all_embeddings[0].shape[0] == 768, f"Embedding dim mismatch: {all_embeddings[0].shape[0]}"
    assert len(set(all_uuids)) == len(all_uuids), "Duplicate engram_uuids found"
    
    for c in all_chunks:
        assert c.get("engram_uuid"), "Missing engram_uuid"
        assert c.get("source_uri"), "Missing source_uri"
        assert c.get("content_hash"), "Missing content_hash"
        
    print(f"  [PASS] {name} artifact integrity validated.")
    return {
        "chunks": chunks_out,
        "embeddings": emb_out
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to-learn", required=True)
    parser.add_argument("--sigint", required=True)
    parser.add_argument("--out", default="benchmarks/outputs/raw")
    parser.add_argument("--max-pdfs", type=int, default=10, help="Max PDFs per folder (for benchmark feasibility)")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages per PDF")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print("Loading model...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # Process
    process_dataset("ToLearn", [args.to_learn], args.max_pdfs, args.max_pages, model, args.out)
    process_dataset("SIGINT", [args.sigint], args.max_pdfs, args.max_pages, model, args.out)
    combined = process_dataset("Combined", [args.to_learn, args.sigint], args.max_pdfs, args.max_pages, model, args.out)
    
    print(f"\nArtifact extraction complete.")
    print(f"Combined Chunks: {combined['chunks']}")
    print(f"Combined Embeddings: {combined['embeddings']}")
    
if __name__ == "__main__":
    main()
