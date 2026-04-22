"""Runtime-level hybrid response tests."""

from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from mnemos.engram.model import Engram
from mnemos.memory_over_maps.view_cache import DerivedViewCache
from mnemos.retrieval.base import SearchResult
from service.app import MnemosRuntime


class StubRouter:
    def __init__(self, hits):
        self._hits = hits
        self.last_kwargs = None

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        meta = {
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
        if kwargs.get("bounded_envelope"):
            meta["candidate_envelope"] = {
                "enabled": True,
                "initial_candidate_count": 7,
                "final_candidate_count": 5,
                "suppression_summary": {},
            }
        return self._hits, meta

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
        memory_over_maps_phase2=False,
        memory_over_maps_phase3=False,
        memory_over_maps_phase4=False,
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


def test_runtime_phase2_candidate_envelope_meta_exposed_when_enabled():
    hit = SearchResult(
        engram=Engram(id="doc3", content="Envelope"),
        score=0.66,
        tier="hybrid",
        metadata={"retrieval_sources": ["semantic"]},
    )
    router = StubRouter([hit])
    rt = _mk_runtime_with_router(router)
    rt._config.memory_over_maps_phase2 = True

    out = rt.search_documents(
        query="envelope",
        top_k=5,
        tiers=None,
        filters=None,
        retrieval_mode="hybrid",
        fusion_policy="balanced",
        explain=True,
        bounded_envelope={"enabled": True, "candidate_pool_limit": 5},
    )

    assert router.last_kwargs["bounded_envelope"] == {"enabled": True, "candidate_pool_limit": 5}
    assert "candidate_envelope" in out["meta"]
    assert out["meta"]["economics"]["candidate_envelope_initial"] == 7
    assert out["meta"]["economics"]["candidate_envelope_final"] == 5
    assert out["meta"]["economics"]["candidate_envelope_compression_ratio"] == 0.7143


def test_runtime_phase3_derived_views_generated_when_enabled():
    hit = SearchResult(
        engram=Engram(
            id="doc4",
            content="Derived view test",
            metadata={"artifact_id": "art-1", "chunk_id": "chunk-1"},
        ),
        score=0.8,
        tier="hybrid",
    )
    router = StubRouter([hit])
    rt = _mk_runtime_with_router(router)
    rt._config.memory_over_maps_phase3 = True

    out = rt.search_documents(
        query="show evidence",
        top_k=5,
        tiers=None,
        filters={"subject_id": "user-1"},
        retrieval_mode="hybrid",
        fusion_policy="balanced",
        explain=True,
        derive_views=["evidence_bundle", "preference_snapshot"],
    )

    assert "derived_views" in out
    assert {v["view_type"] for v in out["derived_views"]} == {
        "evidence_bundle",
        "preference_snapshot",
    }


def test_runtime_phase4_derived_view_cache_hit_and_invalidate():
    hit = SearchResult(
        engram=Engram(
            id="doc5",
            content="Cache test",
            metadata={"artifact_id": "art-5", "chunk_id": "chunk-5"},
        ),
        score=0.8,
        tier="hybrid",
    )
    router = StubRouter([hit])
    rt = _mk_runtime_with_router(router)
    rt._config.memory_over_maps_phase3 = True
    rt._config.memory_over_maps_phase4 = True
    rt._view_cache = DerivedViewCache(ttl_seconds=3600)

    out1 = rt.search_documents(
        query="show evidence",
        top_k=5,
        tiers=None,
        filters=None,
        retrieval_mode="hybrid",
        fusion_policy="balanced",
        explain=True,
        derive_views=["evidence_bundle"],
    )
    assert out1["derived_views"][0]["_cache"]["hit"] is False

    out2 = rt.search_documents(
        query="show evidence",
        top_k=5,
        tiers=None,
        filters=None,
        retrieval_mode="hybrid",
        fusion_policy="balanced",
        explain=True,
        derive_views=["evidence_bundle"],
    )
    assert out2["derived_views"][0]["_cache"]["hit"] is True

    trace = rt.invalidate_derived_view_cache(
        event_type="source_artifact_updated",
        refs={"artifact_id": "art-5"},
        dry_run=False,
    )
    assert trace["impacted_keys"]
    stats = rt.get_stats()["stats"]["economics"]
    assert stats["invalidation_event_count"] >= 1
    assert stats["invalidation_fanout_total"] >= 1
