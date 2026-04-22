import pytest
from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.retrieval.base import SearchResult
import json
from pathlib import Path

# Mocks
class MockFusion:
    @property
    def tier_names(self): return ["qdrant"]
    def search(self, query, top_k, filters, tiers):
        return [
            SearchResult(engram=type("Engram", (), {"id": "1", "content": "mock content"})(), score=0.9, tier="qdrant", metadata={}),
            SearchResult(engram=type("Engram", (), {"id": "2", "content": "mock content"})(), score=0.8, tier="qdrant", metadata={})
        ]

class MockCrossEncoder:
    model_name = "BAAI/bge-reranker-base"
    def health(self):
        return {"healthy": True, "model_loaded": True, "last_error": None, "warm": True, "latency_ms": 0.0}
    def rerank(self, query, results):
        if query == "trigger timeout":
            raise TimeoutError("timeout")
        # Reverse order to simulate top k change
        results[0].score, results[1].score = results[1].score, results[0].score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

def test_telemetry_emission_and_health(tmp_path):
    log_file = tmp_path / "telemetry.jsonl"
    fusion = MockFusion()
    reranker = MockCrossEncoder()
    
    router = RetrievalRouter(semantic_fusion=fusion, reranker=reranker)
    # Patch config to use tmp path
    router._telemetry_sink.logger.handlers[0].baseFilename = str(log_file)
    router._telemetry_sink.logger.handlers[0].stream = open(log_file, "a", encoding="utf-8")
    
    # Send a code query
    hits, meta = router.search(query="how to use python api", top_k=2)
    
    assert meta["retrieval_mode"] == "semantic"
    assert "rerank_telemetry" in meta
    tt = meta["rerank_telemetry"]
    
    assert tt["query_family"] == "code_behavior"
    assert tt["service_healthy"] is True
    assert tt["rerank_applied"] is False
    assert tt["shadow_evaluated"] is True
    assert tt["rerank_skip_reason"] == "insufficient_candidates"
    assert tt["top3_changed"] is False
    assert not tt["timeout_occurred"]
    
    # Test timeout triggering circuit breaker
    hits2, meta2 = router.search(query="trigger timeout", top_k=2)
    tt2 = meta2["rerank_telemetry"]
    assert tt2["timeout_occurred"] is True
    assert tt2["rerank_applied"] is False
    assert tt2["rerank_skip_reason"] == "timeout"
    
    # Read the JSONL
    router._telemetry_sink.logger.handlers[0].stream.flush()
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2
        evt1 = json.loads(lines[0])
        assert evt1["query_family"] == "code_behavior"
        evt2 = json.loads(lines[1])
        assert evt2["timeout_occurred"] is True
