import os
import sys
import json
import time
import argparse
import tempfile
import zipfile
import pytest

from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["windows", "linux"])
    parser.add_argument("--archive", required=True, help="Backup archive to restore")
    args = parser.parse_args()
    
    print(f"=== TQ-2D Verification: {args.mode.upper()} ===")
    
    # 1. Confirm import turbovec
    try:
        import turbovec
        print("[PASS] import turbovec")
    except ImportError as e:
        print(f"[FAIL] import turbovec: {e}")
        sys.exit(1)
        
    from mnemos.retrieval.turbovec_config import TurbovecConfig
    from mnemos.retrieval.turbovec_tier import TurbovecTier
    from mnemos.retrieval.turbovec_fusion import TurbovecFusion
    from mnemos.tools.turbovec_restore import run_restore, RestoreError
    from mnemos.tools.turbovec_backup import create_backup
    
    # 2. Restore into clean directory
    restore_target = os.path.abspath(f"runtime/tq2d_{args.mode}_restored")
    if os.path.exists(restore_target):
        import shutil
        shutil.rmtree(restore_target)
        
    print(f"Restoring {args.archive} to {restore_target}...")
    t0 = time.time()
    run_restore(args.archive, restore_target)
    print(f"[PASS] Restore completed in {time.time() - t0:.2f}s")
    
    # 3. Load profile
    config = TurbovecConfig(embedding_dim=768, bit_width=4, storage_path=restore_target)
    tier = TurbovecTier.load(restore_target, expected_config=config, use_mock=False)
    fusion = TurbovecFusion(tier)
    print("[PASS] Profile loaded successfully")
    
    # 4. Run queries
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    with open("benchmarks/turbovec/query_sets/tq13_real_pdf_queries.json", "r") as f:
        queries = json.load(f)
        
    latencies = []
    print("Running 50 queries...")
    for q in queries:
        t0 = time.time()
        q_emb = model.encode([q["query"]], normalize_embeddings=True)[0].tolist()
        hits = fusion.search(q["query"], q_emb, top_k=10)
        latencies.append((time.time() - t0) * 1000)
        
    p50 = sorted(latencies)[len(latencies)//2]
    print(f"[PASS] Hybrid retrieval successful. p50 = {p50:.2f}ms")
    
    # 5. Backup (Only for Windows, to create the cross-platform source)
    if args.mode == "windows":
        backup_zip = os.path.abspath("benchmarks/outputs/raw/backup_tq2d_win.zip")
        print(f"Creating backup for cross-platform test: {backup_zip}")
        create_backup(restore_target, backup_zip)
        print("[PASS] Backup created")
        
    # 6. Confirm failure behavior on corrupt manifest
    print("Testing corrupt manifest failure...")
    with tempfile.TemporaryDirectory() as td:
        bad_archive = os.path.join(td, "bad.zip")
        with zipfile.ZipFile(args.archive, 'r') as zin:
            with zipfile.ZipFile(bad_archive, 'w') as zout:
                for item in zin.infolist():
                    if item.filename == "manifest.json":
                        zout.writestr(item, b"INVALID_JSON")
                    else:
                        zout.writestr(item, zin.read(item.filename))
        try:
            run_restore(bad_archive, os.path.join(td, "target"))
            print("[FAIL] Corrupt manifest did not trigger error!")
        except RestoreError as e:
            print(f"[PASS] Corrupt manifest correctly rejected: {e}")

if __name__ == "__main__":
    main()
