import pytest
import os
import json
from mnemos.retrieval.turbovec_tier import TurbovecTier
from mnemos.retrieval.turbovec_config import TurbovecConfig

@pytest.fixture
def base_config(tmp_path):
    return TurbovecConfig(embedding_dim=4, bit_width=4, storage_path=str(tmp_path / "run_db"))

def test_save_creates_files(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0]}])
    
    profile_dir = str(tmp_path / "profile")
    tier.save(profile_dir)
    
    assert os.path.exists(os.path.join(profile_dir, "index.tvim.npz")) or os.path.exists(os.path.join(profile_dir, "index.tvim"))
    assert os.path.exists(os.path.join(profile_dir, "metadata.sqlite"))
    assert os.path.exists(os.path.join(profile_dir, "manifest.json"))
    
def test_load_returns_usable_tier(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0]}])
    
    profile_dir = str(tmp_path / "profile")
    tier.save(profile_dir)
    
    loaded_tier = TurbovecTier.load(profile_dir)
    assert loaded_tier.health()["engram_count"] == 1
    
def test_save_load_search_returns_same(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    tier.index([{
        "uuid": "u2", 
        "embedding": [0.0, 1.0, 0.0, 0.0],
        "metadata": {"source": "manual"},
        "governance": {"clearance": "high"}
    }])
    
    profile_dir = str(tmp_path / "profile2")
    tier.save(profile_dir)
    
    loaded = TurbovecTier.load(profile_dir)
    hits = loaded.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    
    assert len(hits) == 1
    assert hits[0].engram_uuid == "u2"
    assert hits[0].metadata["source"] == "manual"
    assert hits[0].governance["clearance"] == "high"
    
def test_deleted_engram_remains_excluded(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    tier.index([{"uuid": "u3", "embedding": [0.0, 0.0, 1.0, 0.0]}])
    tier.delete("u3")
    
    profile_dir = str(tmp_path / "profile3")
    tier.save(profile_dir)
    
    loaded = TurbovecTier.load(profile_dir)
    hits = loaded.search([0.0, 0.0, 1.0, 0.0], top_k=1)
    assert len(hits) == 0
    assert loaded.health()["deleted_count"] == 1
    
def test_duplicate_uuid_mapping_remains_stable(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    tier.index([{"uuid": "u4", "embedding": [1.0, 0.0, 0.0, 0.0]}])
    t_id1 = tier.sidecar.get_engram("u4")["turbovec_id"]
    
    profile_dir = str(tmp_path / "profile4")
    tier.save(profile_dir)
    
    loaded = TurbovecTier.load(profile_dir)
    loaded.index([{"uuid": "u4", "embedding": [0.0, 1.0, 0.0, 0.0]}])
    t_id2 = loaded.sidecar.get_engram("u4")["turbovec_id"]
    
    assert t_id1 == t_id2

def test_missing_manifest_fails_closed(base_config, tmp_path):
    profile_dir = str(tmp_path / "profile_broken1")
    os.makedirs(profile_dir)
    with pytest.raises(FileNotFoundError):
        TurbovecTier.load(profile_dir)
        
def test_missing_index_fails_closed(base_config, tmp_path):
    profile_dir = str(tmp_path / "profile_broken2")
    os.makedirs(profile_dir)
    with open(os.path.join(profile_dir, "manifest.json"), "w") as f:
        json.dump({"embedding_dim": 4, "bit_width": 4, "index_file": "index.tvim", "metadata_file": "metadata.sqlite"}, f)
    with pytest.raises(FileNotFoundError):
        TurbovecTier.load(profile_dir)
        
def test_embedding_dim_mismatch_fails_closed(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    profile_dir = str(tmp_path / "profile5")
    tier.save(profile_dir)
    
    expected_config = TurbovecConfig(embedding_dim=8, bit_width=4)
    with pytest.raises(ValueError, match="embedding_dim mismatch"):
        TurbovecTier.load(profile_dir, expected_config=expected_config)
        
def test_bit_width_mismatch_fails_closed(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    profile_dir = str(tmp_path / "profile6")
    tier.save(profile_dir)
    
    expected_config = TurbovecConfig(embedding_dim=4, bit_width=2)
    with pytest.raises(ValueError, match="bit_width mismatch"):
        TurbovecTier.load(profile_dir, expected_config=expected_config)
        
def test_health_reports_loaded_persisted_tier(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    profile_dir = str(tmp_path / "profile7")
    tier.save(profile_dir)
    
    loaded = TurbovecTier.load(profile_dir)
    health = loaded.health()
    assert health["status"] == "ok"
    assert health["index_loaded"] is True
    
def test_validate_persistence_integrity(base_config, tmp_path):
    tier = TurbovecTier(base_config, use_mock=True)
    tier.index([{"uuid": "u5", "embedding": [1.0, 0.0, 0.0, 0.0]}])
    integrity = tier.validate_persistence_integrity()
    assert integrity["is_consistent"] is True
    assert integrity["dense_count"] == 1
    assert integrity["active_metadata_count"] == 1
