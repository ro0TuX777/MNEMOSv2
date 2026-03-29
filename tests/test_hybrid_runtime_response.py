"""Runtime-level hybrid response tests."""

from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from service.app import MnemosRuntime


class StubRouter:
    def __init__(self, hits):
        self._hits = hits
        self.last_kwargs = None

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        return self._hits, {
            "retrieval_mode": kwargs.get("retrieval_mode", "semantic"),
            "fusion_policy": kwargs.get("fusion_policy"),
            "lexical_available": True,
            "telemetry": {
                "lexical_candidates": 5.0,
                "semantic_candidates": 5.0,
                "union_candidates": 7.0,
                "overlap_candidates": 3.0,
            },
        }

    def stats(self):
        return {}


def _mk_runtime_with_router(router):
    rt = MnemosRuntime()
    rt._config = SimpleNamespace(
        retrieval_mode="semantic",
        fusion_policy="balanced",
        explain_default=False,
        lexical_top_k=25,
        semantic_top_k=25,
        has_compression=False,
        quant_bits=4,
    )
    rt._router = router
    rt._semantic_fusion = None
    rt._lexical_tier = None
    rt._ledger = None
    rt._status = "healthy"
    rt._error = None
    return rt


def test_runtime_hybrid_explain_fields_are_typed_and_present():
    hit = SearchResult(
        engram=Engram(id="doc1", content="SOC2 control objective"),
        score=0.88,
        tier="hybrid",
        metadata={
            "component_scores": {"lexical": 0.7, "semantic": 0.9, "fused": 0.8},
            "retrieval_sources": ["lexical", "semantic"],
            "fusion_policy": "balanced",
            "filters_applied": {"source": "dept-legal"},
        },
    )
    router = StubRouter([hit])
    rt = _mk_runtime_with_router(router)

    out = rt.search_documents(
        query="SOC2 control objective",
        top_k=5,
        tiers=None,
        filters={"source": "dept-legal"},
        retrieval_mode="hybrid",
        fusion_policy="balanced",
        explain=True,
    )

    row = out["results"][0]
    assert isinstance(row["component_scores"]["lexical"], float)
    assert isinstance(row["component_scores"]["semantic"], float)
    assert isinstance(row["component_scores"]["fused"], float)
    assert set(row["retrieval_sources"]).issubset({"lexical", "semantic"})
    assert row["fusion_policy"] == "balanced"
    assert row["filters_applied"] == {"source": "dept-legal"}


def test_runtime_hybrid_explain_false_omits_explain_fields():
    hit = SearchResult(
        engram=Engram(id="doc2", content="Policy mapping"),
        score=0.77,
        tier="hybrid",
        metadata={
            "component_scores": {"lexical": 0.4, "semantic": 0.8, "fused": 0.6},
            "retrieval_sources": ["lexical"],
            "fusion_policy": "lexical_dominant",
        },
    )
    router = StubRouter([hit])
    rt = _mk_runtime_with_router(router)

    out = rt.search_documents(
        query="policy mapping",
        top_k=5,
        tiers=None,
        filters=None,
        retrieval_mode="hybrid",
        fusion_policy="lexical_dominant",
        explain=False,
    )

    row = out["results"][0]
    assert "component_scores" not in row
    assert "retrieval_sources" not in row
    assert "fusion_policy" not in row
