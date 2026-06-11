"""Wiring tests: latency_budget_ms drives stage plans through RetrievalRouter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.retrieval.complexity import ComplexityResult


class RecordingRetriever(BaseRetriever):
    """Returns canned hits; records the filters and top_k it received."""

    def __init__(self, name: str, doc_ids: List[str]):
        self._name = name
        self._doc_ids = doc_ids
        self.last_filters: Optional[Dict[str, Any]] = None
        self.last_top_k: Optional[int] = None
        self.last_query_vector = None

    @property
    def tier_name(self) -> str:
        return self._name

    def index(self, engrams):
        return len(engrams)

    def _embed_query(self, texts: List[str]):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query_vector=None,
    ):
        self.last_filters = dict(filters or {})
        self.last_top_k = top_k
        self.last_query_vector = query_vector
        return [
            SearchResult(
                engram=Engram(id=d, content=f"{self._name}-{d}"),
                score=float(len(self._doc_ids) - i),
                tier=self._name,
            )
            for i, d in enumerate(self._doc_ids[:top_k])
        ]

    def delete(self, engram_ids: List[str]) -> int:
        return len(engram_ids)

    def stats(self):
        return {"tier": self._name}


def _router():
    tier = RecordingRetriever("qdrant", ["a", "b", "c", "d", "e"])
    router = RetrievalRouter(semantic_fusion=TierFusion([tier]), lexical_tier=None)
    return router, tier


class FakeComplexityClassifier:
    def __init__(self, label: str = "CLASS_B"):
        self._label = label
        self.used_vector = False

    def classify(self, query: str) -> ComplexityResult:
        self.used_vector = False
        return self._result()

    def classify_vector(self, query_vector) -> ComplexityResult:
        self.used_vector = True
        return self._result()

    def _result(self) -> ComplexityResult:
        scores = {"CLASS_A": 0.02, "CLASS_B": 0.07, "CLASS_C": 0.02}
        scores[self._label] = 0.91
        return ComplexityResult(
            label=self._label,
            confidence=0.91,
            scores=scores,
            route_posture=_route_posture(self._label),
            latency_ms=1.5,
            model_name="fake",
        )


def _route_posture(label: str) -> Dict[str, Any]:
    if label == "CLASS_A":
        return {
            "retrieval_posture": "semantic_dominant",
            "fusion_policy": "semantic_dominant",
            "graph": "skip",
            "hierarchical": "skip",
        }
    if label == "CLASS_C":
        return {
            "retrieval_posture": "global_hierarchical",
            "fusion_policy": "lexical_dominant",
            "graph": "optional",
            "hierarchical": "trigger_future_raptor",
        }
    return {
        "retrieval_posture": "balanced",
        "fusion_policy": "balanced",
        "graph": "trigger_memgraph_rag",
        "hierarchical": "skip",
    }


class RecordingReranker:
    model_name = "fake-reranker"

    def __init__(self):
        self.calls = 0

    def health(self):
        return {"healthy": True}

    def rerank(self, query: str, results: List[SearchResult]):
        self.calls += 1
        return list(reversed(results))


def test_no_budget_is_byte_identical_to_pre_budget_path():
    router, tier = _router()
    results, meta = router.search(query="q", top_k=3)

    assert "budget_plan" not in meta
    assert tier.last_filters is not None
    assert "__mrl_oversample__" not in tier.last_filters
    assert "__hnsw_ef__" not in tier.last_filters
    assert len(results) == 3


def test_generous_budget_runs_full_plan_with_sentinels():
    router, tier = _router()
    _, meta = router.search(query="q", top_k=3, latency_budget_ms=5000.0)

    plan = meta["budget_plan"]
    assert plan["degraded"] is False
    assert plan["rescore"] is True
    assert tier.last_filters is not None
    assert tier.last_filters["__mrl_oversample__"] == 3.0
    assert tier.last_filters["__hnsw_ef__"] == 128
    assert "__prefetch_only__" not in tier.last_filters


def test_caller_reserved_filter_keys_are_stripped_without_budget():
    router, tier = _router()
    router.search(
        query="q",
        top_k=3,
        filters={
            "tenant_id": "tenant-a",
            "__mrl_oversample__": 99,
            "__hnsw_ef__": 1,
            "__prefetch_only__": True,
        },
    )

    assert tier.last_filters is not None
    assert tier.last_filters["tenant_id"] == "tenant-a"
    assert "__mrl_oversample__" not in tier.last_filters
    assert "__hnsw_ef__" not in tier.last_filters
    assert "__prefetch_only__" not in tier.last_filters


def test_budget_plan_overrides_caller_reserved_filter_keys():
    router, tier = _router()
    router.search(
        query="q",
        top_k=3,
        filters={
            "tenant_id": "tenant-a",
            "__mrl_oversample__": 99,
            "__hnsw_ef__": 1,
            "__prefetch_only__": True,
        },
        latency_budget_ms=5000.0,
    )

    assert tier.last_filters is not None
    assert tier.last_filters["tenant_id"] == "tenant-a"
    assert tier.last_filters["__mrl_oversample__"] == 3.0
    assert tier.last_filters["__hnsw_ef__"] == 128
    assert "__prefetch_only__" not in tier.last_filters


def test_tight_budget_drops_rerank_and_reports_plan():
    router, _ = _router()
    # priors: embed 10 + prefetch 8 + rescore 12 + rerank 60 = 90; 40ms drops rerank
    _, meta = router.search(query="q", top_k=3, latency_budget_ms=40.0)

    plan = meta["budget_plan"]
    assert plan["degraded"] is True
    assert plan["degradation_steps"] == ["drop_rerank"]
    assert meta["rerank_telemetry"]["rerank_skip_reason"] == "latency_budget"


def test_floor_budget_goes_prefetch_only():
    router, tier = _router()
    _, meta = router.search(query="q", top_k=3, latency_budget_ms=15.0)

    plan = meta["budget_plan"]
    assert plan["rescore"] is False
    assert tier.last_filters is not None
    assert tier.last_filters["__prefetch_only__"] is True
    assert plan["degradation_steps"][-1] == "drop_rescore"


def test_infeasible_budget_still_returns_results():
    router, _ = _router()
    results, meta = router.search(query="q", top_k=3, latency_budget_ms=1.0)

    assert meta["budget_plan"]["budget_infeasible"] is True
    assert len(results) == 3  # never fails, always answers


def test_dense_latency_feeds_cost_model():
    router, _ = _router()
    before = router._budget_router.stats()["observation_counts"]
    router.search(query="q", top_k=3, latency_budget_ms=5000.0)
    after = router._budget_router.stats()["observation_counts"]
    assert after.get("prefetch", 0) > before.get("prefetch", 0)


def test_complexity_shadow_records_route_posture():
    tier = RecordingRetriever("qdrant", ["a", "b", "c"])
    router = RetrievalRouter(
        semantic_fusion=TierFusion([tier]),
        lexical_tier=None,
        complexity_classifier=FakeComplexityClassifier(),
    )

    _, meta = router.search(query="q", top_k=3, complexity_shadow=True)

    shadow = meta["complexity_shadow"]
    assert shadow["status"] == "ok"
    assert shadow["label"] == "CLASS_B"
    assert shadow["route_posture"]["graph"] == "trigger_memgraph_rag"


def test_complexity_shadow_disabled_by_default():
    tier = RecordingRetriever("qdrant", ["a", "b", "c"])
    router = RetrievalRouter(
        semantic_fusion=TierFusion([tier]),
        lexical_tier=None,
        complexity_classifier=FakeComplexityClassifier(),
    )

    _, meta = router.search(query="q", top_k=3)

    assert "complexity_shadow" not in meta


def test_active_complexity_class_a_uses_aggressive_semantic_plan():
    tier = RecordingRetriever("qdrant", ["a", "b", "c"])
    reranker = RecordingReranker()
    classifier = FakeComplexityClassifier("CLASS_A")
    router = RetrievalRouter(
        semantic_fusion=TierFusion([tier]),
        lexical_tier=None,
        reranker=reranker,
        complexity_classifier=classifier,
        adaptive_routing_enabled=True,
    )

    _, meta = router.search(query="direct lookup", top_k=3)

    assert meta["complexity_classification"]["label"] == "CLASS_A"
    assert meta["routing_posture"]["budget_strategy"] == "aggressive"
    assert meta["retrieval_mode"] == "semantic"
    assert meta["budget_plan"]["rerank"] is False
    assert meta["rerank_telemetry"]["rerank_skip_reason"] == "latency_budget"
    assert meta["complexity_classification"]["query_vector_reused"] is True
    assert classifier.used_vector is True
    assert tier.last_query_vector is not None
    assert reranker.calls == 0


def test_active_complexity_class_b_forces_rerank():
    tier = RecordingRetriever("qdrant", ["a", "b", "c"])
    reranker = RecordingReranker()
    router = RetrievalRouter(
        semantic_fusion=TierFusion([tier]),
        lexical_tier=None,
        reranker=reranker,
        complexity_classifier=FakeComplexityClassifier("CLASS_B"),
        adaptive_routing_enabled=True,
    )

    _, meta = router.search(query="which policy overlaps", top_k=3)

    assert meta["complexity_classification"]["label"] == "CLASS_B"
    assert meta["routing_posture"]["budget_strategy"] == "conservative"
    assert meta["budget_plan"]["force_rerank"] is True
    assert meta["rerank_telemetry"]["rerank_applied"] is True
    assert reranker.calls == 1


def test_active_complexity_class_c_uses_lexical_dominant_hybrid():
    semantic = RecordingRetriever("qdrant", ["a", "b", "c"])
    lexical = RecordingRetriever("lexical", ["b", "d", "e"])
    router = RetrievalRouter(
        semantic_fusion=TierFusion([semantic]),
        lexical_tier=lexical,
        complexity_classifier=FakeComplexityClassifier("CLASS_C"),
        adaptive_routing_enabled=True,
    )

    _, meta = router.search(query="summarize all policy themes", top_k=3)

    assert meta["complexity_classification"]["label"] == "CLASS_C"
    assert meta["routing_posture"]["budget_strategy"] == "balanced"
    assert meta["routing_posture"]["graph"] == "skip"
    assert meta["retrieval_mode"] == "hybrid"
    assert meta["fusion_policy"] == "lexical_dominant"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
