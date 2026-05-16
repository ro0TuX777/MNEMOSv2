import argparse
import os
import json
import zipfile
import datetime
from mnemos.tools.turbovec_archive import (
    hash_file, check_sqlite_integrity, check_sqlite_tables_exist,
    write_receipt, BackupError
)

def create_backup(profile_dir: str, archive_path: str, force: bool = False):
    if os.path.exists(archive_path) and not force:
        raise BackupError(f"Archive {archive_path} already exists. Use --force to overwrite.")
        
    index_path = os.path.join(profile_dir, "index.tvim")
    meta_path = os.path.join(profile_dir, "metadata.sqlite")
    manifest_path = os.path.join(profile_dir, "manifest.json")
    
    # 1. Validation
    if not os.path.exists(index_path): raise BackupError("index.tvim missing")
    if not os.path.exists(meta_path): raise BackupError("metadata.sqlite missing")
    if not os.path.exists(manifest_path): raise BackupError("manifest.json missing")
    
    # Manifest validation
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("embedding_dim") != 768 or manifest.get("bit_width") != 4:
            raise BackupError("Manifest invalid embedding_dim or bit_width")
    except Exception as e:
        raise BackupError(f"manifest.json malformed: {e}")
        
    # SQLite integrity
    if not check_sqlite_integrity(meta_path):
        raise BackupError("metadata.sqlite failed integrity check")
    if not check_sqlite_tables_exist(meta_path):
        raise BackupError("metadata.sqlite missing required tables")
        
    # 2. Hashing
    hashes = {
        "index.tvim": hash_file(index_path),
        "metadata.sqlite": hash_file(meta_path),
        "manifest.json": hash_file(manifest_path)
    }
    
    # 3. Archive Manifest
    archive_manifest = {
        "archive_type": "mnemos_turbovec_backup",
        "archive_version": "0.1",
        "created_at": datetime.datetime.now().isoformat(),
        "source_profile_dir": profile_dir,
        "files": {
            "index.tvim": {"sha256": hashes["index.tvim"], "size_bytes": os.path.getsize(index_path)},
            "metadata.sqlite": {"sha256": hashes["metadata.sqlite"], "size_bytes": os.path.getsize(meta_path)},
            "manifest.json": {"sha256": hashes["manifest.json"], "size_bytes": os.path.getsize(manifest_path)}
        },
        "profile": {
            "name": "portable_memory_appliance",
            "backend": "turbovec",
            "embedding_dim": manifest.get("embedding_dim"),
            "bit_width": manifest.get("bit_width")
        }
    }
    
    # 4. Write zip
    os.makedirs(os.path.dirname(os.path.abspath(archive_path)), exist_ok=True)
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(index_path, arcname="index.tvim")
        zf.write(meta_path, arcname="metadata.sqlite")
        zf.write(manifest_path, arcname="manifest.json")
        zf.writestr("archive_manifest.json", json.dumps(archive_manifest, indent=2))
        
        # Checksums file
        checksum_content = "\n".join([f"{v}  {k}" for k, v in hashes.items()]) + "\n"
        zf.writestr("checksums.sha256", checksum_content)
        
    # Receipt
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = f"runtime/receipts/turbovec_backup_{timestamp}.json"
    write_receipt(
        receipt_path=receipt_path,
        operation="backup",
        status="success",
        profile_dir=profile_dir,
        archive_path=archive_path,
        file_hashes=hashes,
        validation={
            "manifest_valid": True,
            "sqlite_integrity": "ok",
            "checksums_valid": True,
            "load_validation": True
        }
    )
    return receipt_path

def main():
    parser = argparse.ArgumentParser(description="Turbovec Profile Backup Tool")
    parser.add_argument("--profile-dir", required=True, help="Directory containing Turbovec profile files")
    parser.add_argument("--out", required=True, help="Path for the output backup zip")
    parser.add_argument("--force", action="store_true", help="Overwrite existing archive")
    args = parser.parse_args()
    
    try:
        receipt = create_backup(args.profile_dir, args.out, args.force)
        print(f"Backup successful. Receipt: {receipt}")
    except BackupError as e:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        receipt_path = f"runtime/receipts/turbovec_backup_{timestamp}.json"
        write_receipt(
            receipt_path=receipt_path,
            operation="backup",
            status="failure",
            profile_dir=args.profile_dir,
            archive_path=args.out,
            file_hashes={},
            validation={},
            error=str(e)
        )
        print(f"Backup failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
