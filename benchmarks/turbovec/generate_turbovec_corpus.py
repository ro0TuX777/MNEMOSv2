import argparse
import json
import uuid
import numpy as np
import hashlib
import os

def generate_corpus(size: int, dim: int, seed: int, out_path: str):
    np.random.seed(seed)
    
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
        engram_uuid = str(uuid.uuid4())
        
        vec = np.random.randn(dim).astype(np.float32)
        vec_norm = np.linalg.norm(vec)
        vec = (vec / vec_norm).tolist()
        
        content = f"This is a {family} document. Some standard content for index {i}."
        if family == "lexical_exact_match":
            content += " exact_magic_word_123"
            
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        metadata = {
            "family": family,
            "index": i,
            "tag": "A" if i % 2 == 0 else "B"
        }
        
        governance = {
            "clearance": "high" if i % 10 == 0 else "standard"
        }
        
        record = {
            "uuid": engram_uuid,
            "embedding": vec,
            "content": content,
            "source_uri": f"doc://{family}/{i}",
            "metadata": metadata,
            "governance": governance,
            "content_hash": content_hash
        }
        corpus.append(record)
        
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(corpus, f)
        
    print(f"Generated {size} records to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=10000)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="benchmarks/turbovec/corpus.json")
    args = parser.parse_args()
    
    generate_corpus(args.corpus_size, args.embedding_dim, args.seed, args.out)

if __name__ == "__main__":
    main()
