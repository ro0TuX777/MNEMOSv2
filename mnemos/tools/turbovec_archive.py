import hashlib
import sqlite3
import json
import os
from typing import Dict, Any

class BackupError(Exception):
    pass

class RestoreError(Exception):
    pass

def hash_file(filepath: str) -> str:
    """Return SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_sqlite_integrity(filepath: str) -> bool:
    """Run PRAGMA integrity_check on the sqlite database."""
    conn = None
    try:
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        return result and result[0].lower() == "ok"
    except sqlite3.Error:
        return False
    finally:
        if conn:
            conn.close()

def check_sqlite_tables_exist(filepath: str) -> bool:
    """Ensure basic tables exist in the sidecar."""
    conn = None
    try:
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='engram_metadata';")
        meta = cursor.fetchone()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='engram_fts';")
        fts = cursor.fetchone()
        return bool(meta and fts)
    except sqlite3.Error:
        return False
    finally:
        if conn:
            conn.close()

def write_receipt(receipt_path: str, operation: str, status: str, profile_dir: str, archive_path: str, file_hashes: Dict[str, str], validation: Dict[str, Any], error: str = None):
    """Write an operation receipt."""
    os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
    import datetime
    receipt = {
        "operation": operation,
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
        "profile_dir": profile_dir,
        "archive_path": archive_path,
        "file_hashes": file_hashes,
        "validation": validation,
        "error": error
    }
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
