import json
from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.engram.model import Engram

class DummyTier(BaseRetriever):
    def __init__(self, name):
        self.name = name
    @property
    def tier_name(self): return self.name
    def search(self, query, top_k, filters=None):
        e = Engram(id="1", content="test", embedding=[0.1]*128, edges=[], neuro_tags=[])
        return [SearchResult(engram=e, score=0.9, tier=self.name)]
    def index(self, engrams): pass
    def delete(self, ids): pass
    def stats(self): return {}

class DummyGraphTier:
    def expand_candidates(self, *args, **kwargs):
        e = Engram(id="g1", content="graph", embedding=[0.1]*128, edges=[], neuro_tags=[])
        s = SearchResult(engram=e, score=0.8, tier="graph")
        s.graph_score = 0.8
        s.lineage_complete = True
        return [s], {}

def test_gates():
    sem_fusion = TierFusion([DummyTier("semantic")])
    graph_tier = DummyGraphTier()
    
    # Gate 1: default config (enable=False), request graph_hybrid -> fallback
    router = RetrievalRouter(semantic_fusion=sem_fusion, graph_tier=graph_tier, enable_experimental_graph_hybrid=False)
    hits, meta = router.search(query="test", top_k=5, retrieval_mode="graph_hybrid_experimental")
    print("Gate 1 (Fallback):")
    print(json.dumps(meta.get("experimental_graph_telemetry"), indent=2))
    assert meta["retrieval_mode"] == "semantic"
    assert "graph_experiment_telemetry" not in meta
    
    # Gate 2: double opt-in
    router_opt = RetrievalRouter(semantic_fusion=sem_fusion, graph_tier=graph_tier, enable_experimental_graph_hybrid=True)
    hits_opt, meta_opt = router_opt.search(query="test", top_k=5, retrieval_mode="graph_hybrid_experimental")
    print("\nGate 2 (Executed):")
    print(json.dumps(meta_opt.get("experimental_graph_telemetry"), indent=2))
    assert meta_opt["retrieval_mode"] == "graph_hybrid_experimental"
    assert "graph_experiment_telemetry" in meta_opt
    
    # Gate 3: request semantic only (even if opt-in is True)
    hits_sem, meta_sem = router_opt.search(query="test", top_k=5, retrieval_mode="semantic")
    print("\nGate 3 (Semantic Only):")
    print("Experimental telemetry present:", "experimental_graph_telemetry" in meta_sem)
    assert meta_sem["retrieval_mode"] == "semantic"
    assert "experimental_graph_telemetry" not in meta_sem

if __name__ == "__main__":
    test_gates()
