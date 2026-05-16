import os
import tempfile
import pytest
import sqlite3
import json
import zipfile

from mnemos.tools.turbovec_backup import create_backup, BackupError
from mnemos.tools.turbovec_restore import run_restore, RestoreError
from mnemos.retrieval.turbovec_config import TurbovecConfig
from mnemos.retrieval.turbovec_tier import TurbovecTier

@pytest.fixture
def mock_profile():
    with tempfile.TemporaryDirectory() as td:
        # Create a valid profile footprint
        config = TurbovecConfig(embedding_dim=768, bit_width=4, storage_path=td)
        tier = TurbovecTier(config, use_mock=False)
        tier.save(td)
        del tier
        import gc; gc.collect()
        yield td

@pytest.fixture
def mock_archive(mock_profile):
    with tempfile.TemporaryDirectory() as td:
        archive_path = os.path.join(td, "test_backup.zip")
        receipt = create_backup(mock_profile, archive_path)
        yield archive_path, mock_profile

def test_backup_creates_archive_with_required_files(mock_archive):
    archive_path, _ = mock_archive
    assert os.path.exists(archive_path)
    with zipfile.ZipFile(archive_path, 'r') as zf:
        files = zf.namelist()
        assert "index.tvim" in files
        assert "metadata.sqlite" in files
        assert "manifest.json" in files
        assert "archive_manifest.json" in files
        assert "checksums.sha256" in files

def test_backup_fails_missing_index(mock_profile):
    os.remove(os.path.join(mock_profile, "index.tvim"))
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(BackupError, match="index.tvim missing"):
            create_backup(mock_profile, os.path.join(td, "b.zip"))

def test_backup_fails_corrupt_metadata(mock_profile):
    # Corrupt sqlite
    with open(os.path.join(mock_profile, "metadata.sqlite"), "wb") as f:
        f.write(b"NOT_A_SQLITE_DB")
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(BackupError, match="integrity"):
            create_backup(mock_profile, os.path.join(td, "b.zip"))

def test_restore_succeeds_empty_target(mock_archive):
    archive_path, _ = mock_archive
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "target")
        receipt = run_restore(archive_path, target)
        
        assert os.path.exists(target)
        assert os.path.exists(os.path.join(target, "index.tvim"))
        assert os.path.exists(receipt)

def test_restore_fails_on_tampered_index(mock_archive):
    archive_path, _ = mock_archive
    
    # Tamper with archive
    with tempfile.TemporaryDirectory() as td:
        tampered_archive = os.path.join(td, "tampered.zip")
        
        with zipfile.ZipFile(archive_path, 'r') as zin:
            with zipfile.ZipFile(tampered_archive, 'w') as zout:
                for item in zin.infolist():
                    if item.filename == "index.tvim":
                        zout.writestr(item, b"TAMPERED_DATA_TAMPERED_DATA")
                    else:
                        zout.writestr(item, zin.read(item.filename))
                        
        with pytest.raises(RestoreError, match="Checksum mismatch for index.tvim"):
            run_restore(tampered_archive, os.path.join(td, "target"))

def test_restore_is_atomic(mock_archive):
    archive_path, mock_profile = mock_archive
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "target")
        os.makedirs(target)
        with open(os.path.join(target, "existing_file.txt"), "w") as f:
            f.write("test")
            
        # Tamper archive
        tampered_archive = os.path.join(td, "tampered.zip")
        with zipfile.ZipFile(archive_path, 'r') as zin:
            with zipfile.ZipFile(tampered_archive, 'w') as zout:
                for item in zin.infolist():
                    if item.filename == "manifest.json":
                        zout.writestr(item, b"INVALID_JSON__{")
                    else:
                        zout.writestr(item, zin.read(item.filename))
                        
        # Attempt restore with replace
        with pytest.raises(RestoreError, match="Checksum mismatch for manifest.json"):
            run_restore(tampered_archive, target, replace=True)
            
        # Assert atomic failure didn't destroy target completely (or it didn't even start)
        # Because validation happens in temp dir BEFORE atomic move
        assert os.path.exists(os.path.join(target, "existing_file.txt"))

def test_restore_fails_target_exists_without_replace(mock_archive):
    archive_path, _ = mock_archive
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "target")
        os.makedirs(target)
        with open(os.path.join(target, "f"), "w") as f:
            f.write("f")
            
        with pytest.raises(RestoreError, match="Target directory .* is not empty"):
            run_restore(archive_path, target, replace=False)
