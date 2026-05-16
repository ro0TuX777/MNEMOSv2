import pytest
from mnemos.retrieval.turbovec_tier import TurbovecTier
from mnemos.retrieval.turbovec_config import TurbovecConfig

@pytest.fixture
def tier(tmp_path):
    config = TurbovecConfig(embedding_dim=4, storage_path=str(tmp_path))
    return TurbovecTier(config, use_mock=True)

def test_index_and_search(tier):
    engram = {
        "uuid": "u1",
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "content": "test 1",
        "source_uri": "doc1"
    }
    tier.index([engram])
    hits = tier.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].engram_uuid == "u1"
    assert hits[0].source_uri == "doc1"

def test_nearest_vector(tier):
    e1 = {"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0]}
    e2 = {"uuid": "u2", "embedding": [0.0, 1.0, 0.0, 0.0]}
    tier.index([e1, e2])
    
    hits = tier.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].engram_uuid == "u2"

def test_metadata_filter(tier):
    e1 = {"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0], "metadata": {"tag": "A"}}
    e2 = {"uuid": "u2", "embedding": [0.9, 0.1, 0.0, 0.0], "metadata": {"tag": "B"}}
    tier.index([e1, e2])
    
    # Query leans towards u1, but filter for B
    hits = tier.search([1.0, 0.0, 0.0, 0.0], top_k=2, filters={"tag": "B"})
    assert len(hits) == 1
    assert hits[0].engram_uuid == "u2"

def test_soft_deleted_not_returned(tier):
    tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0]}])
    tier.delete("u1")
    hits = tier.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 0

def test_duplicate_uuid_reuses_id(tier):
    tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0]}])
    row1 = tier.sidecar.get_engram("u1")
    t_id_1 = row1["turbovec_id"]
    
    tier.index([{"uuid": "u1", "embedding": [0.0, 1.0, 0.0, 0.0], "content": "updated"}])
    row2 = tier.sidecar.get_engram("u1")
    t_id_2 = row2["turbovec_id"]
    
    assert t_id_1 == t_id_2
    assert row2["content"] == "updated"

def test_embedding_dim_mismatch(tier):
    with pytest.raises(ValueError):
        tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0]}]) # dim 3 instead of 4

def test_empty_search_returns_empty(tier):
    hits = tier.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert hits == []

def test_health_reports_state(tier):
    tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0]}])
    tier.index([{"uuid": "u2", "embedding": [0.0, 1.0, 0.0, 0.0]}])
    tier.delete("u1")
    
    health = tier.health()
    assert health["status"] == "ok"
    assert health["engram_count"] == 1
    assert health["deleted_count"] == 1
    assert health["embedding_dim"] == 4
