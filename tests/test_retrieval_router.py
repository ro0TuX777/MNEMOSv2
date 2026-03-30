"""Tests for retrieval mode router."""

from typing import Any, Dict, List, Optional

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.retrieval_router import RetrievalRouter


class DummyRetriever(BaseRetriever):
    def __init__(self, name: str, results: List[str]):
        self._name = name
        self._results = results

    @property
    def tier_name(self) -> str:
        return self._name

    def index(self, engrams):
        return len(engrams)

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
        out = []
        for i, doc_id in enumerate(self._results[:top_k]):
            out.append(
                SearchResult(
                    engram=Engram(id=doc_id, content=f"{self._name}-{doc_id}"),
                    score=float(top_k - i),
                    tier=self._name,
                )
            )
        return out

    def delete(self, engram_ids: List[str]) -> int:
        return len(engram_ids)

    def stats(self):
        return {"tier": self._name, "document_count": len(self._results)}


def test_router_semantic_mode_default():
    semantic = DummyRetriever("qdrant", ["a", "b", "c"])
    router = RetrievalRouter(semantic_fusion=TierFusion([semantic]), lexical_tier=None)

    results, meta = router.search(query="q", top_k=2)
    assert len(results) == 2
    assert meta["retrieval_mode"] == "semantic"
    assert meta["fusion_policy"] is None
    stats = router.stats()
    assert stats["retrieval_mode_counters"]["semantic"] == 1
    assert stats["retrieval_mode_counters"]["hybrid"] == 0


def test_router_hybrid_mode_with_policy_and_explain():
    semantic = DummyRetriever("qdrant", ["a", "c", "d"])
    lexical = DummyRetriever("lexical", ["a", "b", "e"])
    router = RetrievalRouter(semantic_fusion=TierFusion([semantic]), lexical_tier=lexical)

    results, meta = router.search(
        query="controls SOC2",
        top_k=3,
        retrieval_mode="hybrid",
        fusion_policy="balanced",
        explain=True,
        lexical_top_k=3,
        semantic_top_k=3,
    )

    assert len(results) == 3
    assert meta["retrieval_mode"] == "hybrid"
    assert meta["fusion_policy"] == "balanced"
    assert results[0].tier == "hybrid"
    assert "component_scores" in results[0].metadata

    stats = router.stats()
    assert stats["hybrid_query_count"] == 1
    assert stats["hybrid_available"] is True
    assert "hybrid_latency_p50_ms" in stats
    assert stats["retrieval_mode_counters"]["hybrid"] == 1
    assert stats["fusion_policy_counters"]["balanced"] == 1
    assert stats["hybrid_last_telemetry"]["union_candidates"] >= 1.0


def test_router_hybrid_preserves_filters_and_avoids_leakage():
    class FilteredRetriever(DummyRetriever):
        def __init__(self, name: str, docs):
            super().__init__(name, [])
            self.docs = docs
            self.last_filters = None

        def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
            self.last_filters = filters or {}
            out = []
            for doc_id, source in self.docs:
                if filters and filters.get("source") and source != filters["source"]:
                    continue
                out.append(
                    SearchResult(
                        engram=Engram(
                            id=doc_id,
                            content=f"{self._name}-{doc_id}",
                            source=source,
                        ),
                        score=1.0,
                        tier=self._name,
                    )
                )
            return out[:top_k]

    semantic = FilteredRetriever("qdrant", [("a", "dept-legal"), ("b", "dept-finance")])
    lexical = FilteredRetriever("lexical", [("a", "dept-legal"), ("c", "dept-finance")])
    router = RetrievalRouter(semantic_fusion=TierFusion([semantic]), lexical_tier=lexical)

    results, _ = router.search(
        query="controls",
        top_k=10,
        filters={"source": "dept-legal"},
        retrieval_mode="hybrid",
        fusion_policy="balanced",
        explain=True,
    )

    assert semantic.last_filters == {"source": "dept-legal"}
    assert lexical.last_filters == {"source": "dept-legal"}
    assert all(r.engram.source == "dept-legal" for r in results)
    assert all(r.metadata.get("filters_applied") == {"source": "dept-legal"} for r in results)


def test_router_phase2_bounded_envelope_applies_caps():
    semantic = DummyRetriever("qdrant", ["a1", "a2", "a3", "b1", "b2", "c1"])

    class ArtifactAwareRetriever(DummyRetriever):
        def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
            out = []
            artifact_map = {
                "a1": "art-a",
                "a2": "art-a",
                "a3": "art-a",
                "b1": "art-b",
                "b2": "art-b",
                "c1": "art-c",
            }
            for i, doc_id in enumerate(self._results[:top_k]):
                out.append(
                    SearchResult(
                        engram=Engram(
                            id=doc_id,
                            content=f"content-{doc_id}",
                            metadata={"artifact_id": artifact_map[doc_id]},
                        ),
                        score=float(top_k - i),
                        tier=self._name,
                    )
                )
            return out

    semantic = ArtifactAwareRetriever("qdrant", ["a1", "a2", "a3", "b1", "b2", "c1"])
    router = RetrievalRouter(semantic_fusion=TierFusion([semantic]), lexical_tier=None)
    results, meta = router.search(
        query="q",
        top_k=10,
        bounded_envelope={
            "enabled": True,
            "candidate_pool_limit": 3,
            "dedupe_similarity_threshold": 1.0,
            "max_per_source_artifact": 1,
            "diversity_policy": "off",
        },
    )
    assert len(results) == 3
    env = meta["candidate_envelope"]
    assert env["enabled"] is True
    assert env["final_candidate_count"] == 3
    assert env["suppression_summary"]["source_cap_exceeded"] >= 1
    stats = router.stats()
    assert "candidate_envelope_avg_compression_ratio" in stats
    assert stats["candidate_envelope_avg_compression_ratio"] > 0.0
    assert "candidate_envelope_total_reduction" in stats
