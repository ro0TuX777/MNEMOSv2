import time
import json
from unittest.mock import MagicMock
from mnemos.engram.model import Engram
from mnemos.retrieval.resolvers.qdrant_resolver import QdrantEngramResolver
from mnemos.retrieval.graph_tier import InMemoryEngramResolver
from mnemos.retrieval.qdrant_tier import QdrantTier

def run_benchmark():
    num_seeds = 10
    neighbors_per_seed = 5
    total_unique_neighbors = num_seeds * neighbors_per_seed
    
    ids_to_fetch = [f"neighbor_{i}" for i in range(total_unique_neighbors)]
    
    # Setup InMemoryEngramResolver
    engrams_dict = {eid: Engram(id=eid, content="t", embedding=[], edges=[], neuro_tags=[]) for eid in ids_to_fetch}
    in_memory = InMemoryEngramResolver(engrams_dict)
    
    # Setup Mock QdrantTier to simulate network latency
    tier_mock = MagicMock(spec=QdrantTier)
    tier_mock._client = MagicMock()
    tier_mock._to_point_id.side_effect = lambda x: x
    tier_mock._collection_name = "test"
    
    # Simulate network latency of 3.5ms for a batched retrieve
    def mock_retrieve(*args, **kwargs):
        time.sleep(0.0035)
        # Mock hits to be reconstructed
        hits = []
        for eid in kwargs.get("ids", []):
            hit = MagicMock()
            hit.id = eid
            hit.score = 1.0
            hit.payload = {"_mnemos_id": eid, "content": "t", "confidence": 1.0, "edges": []}
            hits.append(hit)
        return hits
        
    tier_mock._client.retrieve.side_effect = mock_retrieve
    
    def mock_hit_to_result(hit):
        eng = Engram(id=hit.id, content=hit.payload["content"], embedding=[], edges=[], neuro_tags=[], confidence=hit.payload["confidence"])
        from mnemos.retrieval.base import SearchResult
        return SearchResult(engram=eng, score=hit.score, tier="qdrant")
        
    tier_mock._hit_to_result.side_effect = mock_hit_to_result
    
    qdrant_resolver = QdrantEngramResolver(tier_mock)
    
    # Benchmark InMemory
    start = time.perf_counter()
    in_memory.prefetch(ids_to_fetch)
    for eid in ids_to_fetch:
        in_memory.get_by_id(eid)
    in_memory_time = (time.perf_counter() - start) * 1000
    
    # Benchmark QdrantEngramResolver
    start = time.perf_counter()
    qdrant_resolver.prefetch(ids_to_fetch)
    for eid in ids_to_fetch:
        qdrant_resolver.get_by_id(eid)
    qdrant_time = (time.perf_counter() - start) * 1000
    
    results = {
        "benchmark": "QdrantEngramResolver vs InMemoryEngramResolver",
        "seeds": num_seeds,
        "unique_edges": total_unique_neighbors,
        "in_memory_latency_ms": round(in_memory_time, 2),
        "qdrant_batched_latency_ms": round(qdrant_time, 2),
        "qdrant_retrieve_calls": tier_mock._client.retrieve.call_count,
        "budget_ms": 10.0,
        "meets_budget": qdrant_time < 10.0
    }
    
    print(json.dumps(results, indent=2))
    
if __name__ == "__main__":
    run_benchmark()
