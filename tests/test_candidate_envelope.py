"""Tests for Phase 2 bounded candidate envelope."""

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.candidate_envelope import (
    CandidateEnvelopeConfig,
    apply_candidate_envelope,
)


def _hit(doc_id: str, content: str, artifact_id: str, score: float) -> SearchResult:
    return SearchResult(
        engram=Engram(
            id=doc_id,
            content=content,
            metadata={"artifact_id": artifact_id, "chunk_id": f"chunk-{doc_id}"},
        ),
        score=score,
        tier="qdrant",
    )


def test_candidate_envelope_dedupe_source_cap_and_limit():
    candidates = [
        _hit("a1", "alpha policy", "art-a", 0.99),
        _hit("a2", "alpha policy", "art-a", 0.95),  # duplicate by text
        _hit("a3", "alpha policy update", "art-a", 0.93),
        _hit("b1", "beta control", "art-b", 0.91),
        _hit("b2", "beta control details", "art-b", 0.90),
        _hit("c1", "gamma evidence", "art-c", 0.89),
    ]
    cfg = CandidateEnvelopeConfig(
        enabled=True,
        candidate_pool_limit=3,
        dedupe_similarity_threshold=0.95,
        max_per_source_artifact=1,
        diversity_policy="off",
        bounded_adjudication_enabled=True,
    )
    narrowed, meta = apply_candidate_envelope(candidates, cfg)
    assert len(narrowed) == 3
    assert meta["initial_candidate_count"] == 6
    assert meta["final_candidate_count"] == 3
    assert meta["suppression_summary"]["duplicate_similarity"] >= 1
    assert meta["suppression_summary"]["source_cap_exceeded"] >= 1


def test_candidate_envelope_disabled_is_noop():
    candidates = [_hit("a1", "alpha", "art-a", 0.99), _hit("b1", "beta", "art-b", 0.88)]
    cfg = CandidateEnvelopeConfig(enabled=False)
    narrowed, meta = apply_candidate_envelope(candidates, cfg)
    assert [h.engram.id for h in narrowed] == ["a1", "b1"]
    assert meta["enabled"] is False

