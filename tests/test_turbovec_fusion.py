import pytest
from mnemos.retrieval.turbovec_tier import TurbovecTier
from mnemos.retrieval.turbovec_config import TurbovecConfig
from mnemos.retrieval.turbovec_fusion import TurbovecFusion

@pytest.fixture
def tier(tmp_path):
    config = TurbovecConfig(embedding_dim=4, storage_path=str(tmp_path))
    return TurbovecTier(config, use_mock=True)

@pytest.fixture
def fusion(tier):
    return TurbovecFusion(tier)

def setup_test_data(tier):
    # Lexical only match (u1)
    tier.index([{"uuid": "u1", "embedding": [0.0, 0.0, 0.0, 1.0], "content": "unique lexical word fox"}])
    # Dense only match (u2)
    tier.index([{"uuid": "u2", "embedding": [1.0, 0.0, 0.0, 0.0], "content": "random semantic text"}])
    # Shared match (u3)
    tier.index([{"uuid": "u3", "embedding": [0.9, 0.1, 0.0, 0.0], "content": "the semantic fox jumps"}])
    # Add dummy vectors so u1 doesn't accidentally get a high dense rank (e.g. rank 3 out of 3)
    for i in range(60):
        tier.index([{"uuid": f"dummy{i}", "embedding": [0.8, 0.2, 0.0, 0.0], "content": "nothing"}])
    
def test_dense_only_hit(fusion, tier):
    setup_test_data(tier)
    hits = fusion.search("zebra", [1.0, 0.0, 0.0, 0.0])
    
    # "zebra" isn't in any doc, so only dense matching works
    assert len(hits) > 0
    # u2 and u3 have high dense similarity to [1,0,0,0], u2 is exact
    assert hits[0].engram_uuid == "u2"
    assert hits[0].lexical_rank is None
    
def test_lexical_only_hit(fusion, tier):
    setup_test_data(tier)
    hits = fusion.search("unique", [0.0, 0.0, 1.0, 0.0])
    
    # Dense query doesn't match much, but lexical matches u1
    assert any(h.engram_uuid == "u1" for h in hits)
    # Check that u1 has a lexical rank
    u1_hit = next(h for h in hits if h.engram_uuid == "u1")
    assert u1_hit.lexical_rank is not None

def test_shared_ranks_above_single_lane(fusion, tier):
    setup_test_data(tier)
    # Query matches u3 semantically and lexically ("fox")
    hits = fusion.search("fox", [1.0, 0.0, 0.0, 0.0], policy="balanced")
    
    # u3 matches BOTH. u2 matches dense. u1 matches lexical.
    # Due to RRF, u3 should bubble to the top.
    assert hits[0].engram_uuid == "u3"

def test_semantic_dominant_favors_dense(fusion, tier):
    setup_test_data(tier)
    # Query: "fox" -> lexical finds u1, u3. Dense finds u2, u3. 
    # With semantic_dominant, u2 (which is perfectly matched dense) should rank higher or similarly to u3
    hits = fusion.search("fox", [1.0, 0.0, 0.0, 0.0], policy="semantic_dominant")
    assert any(h.engram_uuid == "u2" for h in hits[:2]) # u2 gets boosted because dense is heavily weighted
    
def test_lexical_dominant_favors_fts(fusion, tier):
    setup_test_data(tier)
    hits = fusion.search("unique", [1.0, 0.0, 0.0, 0.0], policy="lexical_dominant")
    # lexical matches u1, dense matches u2. lexical dominant favors u1.
    assert hits[0].engram_uuid == "u1"
    
def test_metadata_filters_apply(fusion, tier):
    tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0], "content": "dog", "metadata": {"tag": "A"}}])
    tier.index([{"uuid": "u2", "embedding": [1.0, 0.0, 0.0, 0.0], "content": "dog", "metadata": {"tag": "B"}}])
    
    hits = fusion.search("dog", [1.0, 0.0, 0.0, 0.0], filters={"tag": "B"})
    assert len(hits) == 1
    assert hits[0].engram_uuid == "u2"
    
def test_soft_deleted_never_appear(fusion, tier):
    tier.index([{"uuid": "u1", "embedding": [1.0, 0.0, 0.0, 0.0], "content": "dog"}])
    tier.delete("u1")
    
    hits = fusion.search("dog", [1.0, 0.0, 0.0, 0.0])
    assert len(hits) == 0

def test_explain_includes_details(fusion, tier):
    setup_test_data(tier)
    hits = fusion.search("fox", [1.0, 0.0, 0.0, 0.0], explain=True)
    
    assert len(hits) > 0
    expl = hits[0].explanation
    assert "dense_rank" in expl
    assert "lexical_rank" in expl
    assert expl["policy"] == "balanced"
    
def test_unknown_policy_fails(fusion, tier):
    with pytest.raises(ValueError):
        fusion.search("dog", [1.0, 0.0, 0.0, 0.0], policy="magic")
        
def test_empty_results(fusion, tier):
    hits = fusion.search("nonexistent", [0.0, 0.0, 0.0, 0.0])
    assert hits == []
