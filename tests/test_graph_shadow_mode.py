import pytest
from typing import List

from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.retrieval.graph_tier import GraphTier, InMemoryEngramResolver
from mnemos.retrieval.fusion import TierFusion

class DummyRetriever(BaseRetriever):
    def __init__(self, name: str, engrams: List[Engram]):
        self.name = name
        self.engrams = engrams

    def search(self, query: str, top_k: int = 10, **kwargs):
        return [SearchResult(engram=e, score=0.9) for e in self.engrams[:top_k]]
        
    def get_by_ids(self, ids: List[str]):
        return []
        
    def delete(self, ids: List[str]):
        pass
        
    def index(self, documents: List[dict]):
        pass
        
    def stats(self):
        return {}
        
    @property
    def tier_name(self):
        return self.name

def make_test_engrams():
    # Seed 1: valid, links to 2 (valid) and 3 (vetoed)
    e1 = Engram(id="seed_1", content="Seed 1", edges=["neighbor_2", "neighbor_3"])
    e1.metadata = {"artifact_id": "art_1", "chunk_id": "c1", "source_uri": "uri_1"}
    
    # Neighbor 2: valid, complete lineage
    e2 = Engram(id="neighbor_2", content="Neighbor 2", edges=[])
    e2.metadata = {"artifact_id": "art_2", "chunk_id": "c2", "source_uri": "uri_2"}
    
    # Neighbor 3: vetoed
    e3 = Engram(id="neighbor_3", content="Neighbor 3", edges=[])
    e3.metadata = {"artifact_id": "art_3", "chunk_id": "c3", "source_uri": "uri_3"}
    e3.governance = GovernanceMeta(conflict_status="vetoed")
    
    # Seed 4: vetoed, links to 5
    e4 = Engram(id="seed_4", content="Seed 4", edges=["neighbor_5"])
    e4.metadata = {"artifact_id": "art_4", "chunk_id": "c4", "source_uri": "uri_4"}
    e4.governance = GovernanceMeta(conflict_status="vetoed")
    
    # Neighbor 5: valid, complete lineage
    e5 = Engram(id="neighbor_5", content="Neighbor 5", edges=[])
    e5.metadata = {"artifact_id": "art_5", "chunk_id": "c5", "source_uri": "uri_5"}
    
    # Neighbor 6: incomplete lineage
    e6 = Engram(id="neighbor_6", content="Neighbor 6", edges=[])
    e6.metadata = {} # Missing source_uri and artifact_id
    
    # Seed 7: links to 6
    e7 = Engram(id="seed_7", content="Seed 7", edges=["neighbor_6"])
    e7.metadata = {"artifact_id": "art_7", "chunk_id": "c7", "source_uri": "uri_7"}
    
    return [e1, e2, e3, e4, e5, e6, e7]

@pytest.fixture
def test_data():
    engrams = make_test_engrams()
    engram_dict = {e.id: e for e in engrams}
    
    semantic = DummyRetriever("semantic", [engrams[0], engrams[3], engrams[6]]) # seeds 1, 4, 7
    fusion = TierFusion([semantic])
    
    graph_tier = GraphTier(engram_resolver=InMemoryEngramResolver(engram_dict))
    
    return {
        "fusion": fusion,
        "graph_tier": graph_tier,
        "seeds": [engrams[0], engrams[3], engrams[6]]
    }

def test_shadow_graph_disabled_has_zero_behavior_change(test_data):
    router_disabled = RetrievalRouter(
        semantic_fusion=test_data["fusion"],
        graph_tier=test_data["graph_tier"],
        graph_shadow_enabled=False
    )
    
    hits, meta = router_disabled.search(query="test", top_k=10)
    
    assert len(hits) == 3
    assert "graph_shadow_telemetry" not in meta

def test_shadow_graph_enabled_returns_exact_same_hits(test_data):
    router_disabled = RetrievalRouter(
        semantic_fusion=test_data["fusion"],
        graph_tier=test_data["graph_tier"],
        graph_shadow_enabled=False
    )
    hits_disabled, _ = router_disabled.search(query="test", top_k=10)
    
    router_enabled = RetrievalRouter(
        semantic_fusion=test_data["fusion"],
        graph_tier=test_data["graph_tier"],
        graph_shadow_enabled=True
    )
    hits_enabled, meta = router_enabled.search(query="test", top_k=10)
    
    assert "graph_shadow_telemetry" in meta
    
    assert [h.engram.id for h in hits_disabled] == [h.engram.id for h in hits_enabled]

def test_shadow_graph_does_not_mutate_results(test_data):
    router = RetrievalRouter(
        semantic_fusion=test_data["fusion"],
        graph_tier=test_data["graph_tier"],
        graph_shadow_enabled=True
    )
    hits, meta = router.search(query="test", top_k=10)
    
    telemetry = meta["graph_shadow_telemetry"]
    assert telemetry["mutated_results"] is False
    
    for hit in hits:
        assert hit.tier != "graph_shadow"

def test_graph_respects_parent_governance_veto(test_data):
    # Seed 4 is vetoed, it links to Neighbor 5. Neighbor 5 should not be traversed.
    router = RetrievalRouter(
        semantic_fusion=test_data["fusion"],
        graph_tier=test_data["graph_tier"],
        graph_shadow_enabled=True
    )
    hits, meta = router.search(query="test", top_k=10)
    telemetry = meta["graph_shadow_telemetry"]
    
    # Expect neighbor_2 and neighbor_6, wait, neighbor_6 has incomplete lineage so lineage_complete=False.
    # But neighbor_5 should NOT be in candidates.
    c_ids = [c["candidate_id"] for c in telemetry["candidates"]]
    assert "neighbor_5" not in c_ids

def test_graph_respects_candidate_governance_veto(test_data):
    # Seed 1 links to Neighbor 3 (vetoed). Neighbor 3 should not be traversed.
    router = RetrievalRouter(
        semantic_fusion=test_data["fusion"],
        graph_tier=test_data["graph_tier"],
        graph_shadow_enabled=True
    )
    hits, meta = router.search(query="test", top_k=10)
    telemetry = meta["graph_shadow_telemetry"]
    
    c_ids = [c["candidate_id"] for c in telemetry["candidates"]]
    assert "neighbor_3" not in c_ids

def test_graph_lineage_complete_ratio(test_data):
    router = RetrievalRouter(
        semantic_fusion=test_data["fusion"],
        graph_tier=test_data["graph_tier"],
        graph_shadow_enabled=True
    )
    hits, meta = router.search(query="test", top_k=10)
    telemetry = meta["graph_shadow_telemetry"]
    
    # We expect neighbor_2 (lineage complete) and neighbor_6 (lineage incomplete)
    # neighbor_6 should be filtered out to ineligible
    assert telemetry["graph_candidate_count"] == 1
    assert telemetry["graph_candidate_lineage_complete_count"] == 1
    assert telemetry["graph_candidate_lineage_complete_ratio"] == 1.0
    assert telemetry["graph_lineage_filtered_count"] == 1
    assert len(telemetry["ineligible_candidates"]) == 1

def test_graph_depth_greater_than_one_blocked_in_mg_test_1(test_data):
    # Call expand_candidates directly
    seeds = [SearchResult(engram=test_data["seeds"][0], score=1.0)]
    candidates, t_meta = test_data["graph_tier"].expand_candidates(
        seed_candidates=seeds,
        max_depth=5 # Request depth 5
    )
    
    for c in candidates:
        assert c.edge_depth == 1 # Still 1

def test_graph_caps_max_neighbors_and_total_candidates():
    # Construct a highly connected hub
    e_hub = Engram(id="hub", content="hub", edges=[f"neighbor_{i}" for i in range(100)])
    e_hub.metadata = {"artifact_id": "art_1", "chunk_id": "c1", "source_uri": "uri_1"}
    
    engrams = [e_hub]
    for i in range(100):
        n = Engram(id=f"neighbor_{i}")
        n.metadata = {"artifact_id": f"art_{i}", "chunk_id": f"c_{i}", "source_uri": f"uri_{i}"}
        engrams.append(n)
        
    engram_dict = {e.id: e for e in engrams}
    
    graph_tier = GraphTier(engram_resolver=InMemoryEngramResolver(engram_dict))
    
    seeds = [SearchResult(engram=e_hub, score=1.0)]
    
    candidates, _ = graph_tier.expand_candidates(
        seed_candidates=seeds,
        max_depth=1,
        max_neighbors_per_seed=5,
        max_total_graph_candidates=20
    )
    
    # Limited to 5 by max_neighbors_per_seed
    assert len(candidates) == 5
    
    # Now try multiple seeds to hit max_total
    seeds_multi = []
    for i in range(10):
        e_seed = Engram(id=f"seed_{i}", content=f"seed_{i}", edges=[f"n_{i}_{j}" for j in range(10)])
        e_seed.metadata = {"artifact_id": "a", "chunk_id": "c", "source_uri": "u"}
        engram_dict[e_seed.id] = e_seed
        for j in range(10):
            n = Engram(id=f"n_{i}_{j}")
            n.metadata = {"artifact_id": "a", "chunk_id": "c", "source_uri": "u"}
            engram_dict[n.id] = n
        seeds_multi.append(SearchResult(engram=e_seed, score=1.0))
        
    graph_tier2 = GraphTier(engram_resolver=InMemoryEngramResolver(engram_dict))
    
    candidates2, _ = graph_tier2.expand_candidates(
        seed_candidates=seeds_multi,
        max_depth=1,
        max_neighbors_per_seed=5,
        max_total_graph_candidates=15
    )
    
    assert len(candidates2) == 15
