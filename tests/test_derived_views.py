"""Tests for Phase 3 on-demand derived views."""

from mnemos.engram.model import Engram
from mnemos.memory_over_maps.view_builder import (
    SUPPORTED_DERIVED_VIEWS,
    build_requested_views,
)
from mnemos.retrieval.base import SearchResult


def _hit(doc_id: str, artifact_id: str, created_at: str) -> SearchResult:
    return SearchResult(
        engram=Engram(
            id=doc_id,
            content=f"content-{doc_id}",
            created_at=created_at,
            metadata={"artifact_id": artifact_id, "chunk_id": f"chunk-{doc_id}"},
        ),
        score=0.9,
        tier="qdrant",
    )


def test_supported_derived_views_declared():
    assert SUPPORTED_DERIVED_VIEWS == {
        "evidence_bundle",
        "contradiction_bundle",
        "preference_snapshot",
        "timeline_summary",
    }


def test_build_requested_views_reproducible():
    results = [
        _hit("a1", "art-a", "2026-03-29T10:00:00Z"),
        _hit("b1", "art-b", "2026-03-29T11:00:00Z"),
    ]
    req = ["evidence_bundle", "preference_snapshot", "timeline_summary"]
    v1 = build_requested_views(
        requested=req,
        query="what is latest preference",
        results=results,
        decisions=[],
        contradiction_records=[],
        subject_id="user-1",
    )
    v2 = build_requested_views(
        requested=req,
        query="what is latest preference",
        results=results,
        decisions=[],
        contradiction_records=[],
        subject_id="user-1",
    )
    def _canonical(v):
        out = dict(v)
        out.pop("view_id", None)
        out.pop("created_at", None)
        return out

    assert [_canonical(v) for v in v1] == [_canonical(v) for v in v2]
    assert all(view.get("inputs", {}).get("artifact_ids") for view in v1)
    assert all(view.get("inputs", {}).get("chunk_ids") for view in v1)
