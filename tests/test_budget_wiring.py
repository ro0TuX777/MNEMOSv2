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


class RecordingRetriever(BaseRetriever):
    """Returns canned hits; records the filters and top_k it received."""

    def __init__(self, name: str, doc_ids: List[str]):
        self._name = name
        self._doc_ids = doc_ids
        self.last_filters: Optional[Dict[str, Any]] = None
        self.last_top_k: Optional[int] = None

    @property
    def tier_name(self) -> str:
        return self._name

    def index(self, engrams):
        return len(engrams)

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
        self.last_filters = dict(filters or {})
        self.last_top_k = top_k
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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
