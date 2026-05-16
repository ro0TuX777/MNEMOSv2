import argparse
import os
import json
import zipfile
import datetime
import shutil
import tempfile

from mnemos.tools.turbovec_archive import (
    hash_file, check_sqlite_integrity, check_sqlite_tables_exist,
    write_receipt, RestoreError
)

def run_restore(archive_path: str, target_dir: str, replace: bool = False, allow_extra_files: bool = False):
    if not os.path.exists(archive_path):
        raise RestoreError(f"Archive missing: {archive_path}")
        
    if os.path.exists(target_dir) and os.listdir(target_dir):
        if not replace:
            raise RestoreError(f"Target directory {target_dir} is not empty. Use --replace to overwrite.")
            
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Extract
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(temp_dir)
        except zipfile.BadZipFile:
            raise RestoreError("Archive is not a valid zip file")
            
        # 2. Verify required files
        req_files = ["index.tvim", "metadata.sqlite", "manifest.json", "archive_manifest.json", "checksums.sha256"]
        for rf in req_files:
            if not os.path.exists(os.path.join(temp_dir, rf)):
                raise RestoreError(f"Missing required file in archive: {rf}")
                
        # Unknown files
        if not allow_extra_files:
            for f in os.listdir(temp_dir):
                if f not in req_files:
                    raise RestoreError(f"Unknown extra file in archive: {f}")
                    
        # 3. Verify checksums
        expected_hashes = {}
        with open(os.path.join(temp_dir, "checksums.sha256"), "r") as f:
            for line in f:
                if line.strip():
                    parts = line.split("  ")
                    if len(parts) == 2:
                        expected_hashes[parts[1].strip()] = parts[0].strip()
                        
        actual_hashes = {}
        for core_file in ["index.tvim", "metadata.sqlite", "manifest.json"]:
            actual_hash = hash_file(os.path.join(temp_dir, core_file))
            actual_hashes[core_file] = actual_hash
            if actual_hash != expected_hashes.get(core_file):
                raise RestoreError(f"Checksum mismatch for {core_file}")
                
        # 4. Validate manifest & archive_manifest
        try:
            with open(os.path.join(temp_dir, "manifest.json"), "r") as f:
                manifest = json.load(f)
            with open(os.path.join(temp_dir, "archive_manifest.json"), "r") as f:
                arc_manifest = json.load(f)
        except Exception as e:
            raise RestoreError(f"Malformed JSON manifest: {e}")
            
        if manifest.get("embedding_dim") != 768 or manifest.get("bit_width") != 4:
            raise RestoreError("Incompatible manifest configuration")
            
        # 5. Validate metadata.sqlite
        meta_path = os.path.join(temp_dir, "metadata.sqlite")
        if not check_sqlite_integrity(meta_path) or not check_sqlite_tables_exist(meta_path):
            raise RestoreError("metadata.sqlite corrupt or invalid")
            
        # 6. Load Validation (Atomic test load)
        try:
            from mnemos.retrieval.turbovec_config import TurbovecConfig
            from mnemos.retrieval.turbovec_tier import TurbovecTier
            config = TurbovecConfig(embedding_dim=768, bit_width=4, storage_path=temp_dir)
            tier = TurbovecTier.load(temp_dir, expected_config=config, use_mock=False)
            del tier
            import gc; gc.collect()
        except Exception as e:
            raise RestoreError(f"TurbovecTier.load() failed: {e}")
            
        # 7. Atomic Move
        if os.path.exists(target_dir) and os.listdir(target_dir):
            rollback_dir = target_dir + f".rollback_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.rename(target_dir, rollback_dir)
            
        os.makedirs(target_dir, exist_ok=True)
        for core_file in ["index.tvim", "metadata.sqlite", "manifest.json"]:
            shutil.copy2(os.path.join(temp_dir, core_file), os.path.join(target_dir, core_file))
            
        # Receipt
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        receipt_path = f"runtime/receipts/turbovec_restore_{timestamp}.json"
        write_receipt(
            receipt_path=receipt_path,
            operation="restore",
            status="success",
            profile_dir=target_dir,
            archive_path=archive_path,
            file_hashes=actual_hashes,
            validation={
                "manifest_valid": True,
                "sqlite_integrity": "ok",
                "checksums_valid": True,
                "load_validation": True
            }
        )
        return receipt_path

def main():
    parser = argparse.ArgumentParser(description="Turbovec Profile Restore Tool")
    parser.add_argument("--archive", required=True, help="Path to backup zip archive")
    parser.add_argument("--target-dir", required=True, help="Directory to restore into")
    parser.add_argument("--replace", action="store_true", help="Replace existing profile directory")
    parser.add_argument("--allow-extra-files", action="store_true", help="Allow unknown files in archive")
    args = parser.parse_args()
    
    try:
        receipt = run_restore(args.archive, args.target_dir, args.replace, args.allow_extra_files)
        print(f"Restore successful. Receipt: {receipt}")
    except RestoreError as e:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        receipt_path = f"runtime/receipts/turbovec_restore_{timestamp}.json"
        write_receipt(
            receipt_path=receipt_path,
            operation="restore",
            status="failure",
            profile_dir=args.target_dir,
            archive_path=args.archive,
            file_hashes={},
            validation={},
            error=str(e)
        )
        print(f"Restore failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
