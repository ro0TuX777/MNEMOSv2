"""Tests for hybrid fusion logic."""

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.hybrid_fusion import HybridFusion


def _hit(doc_id: str, score: float, tier: str) -> SearchResult:
    return SearchResult(
        engram=Engram(id=doc_id, content=f"content-{doc_id}"),
        score=score,
        tier=tier,
    )


def _ranked_ids(policy: str):
    fusion = HybridFusion()
    lexical = [_hit("a", 0.9, "lexical"), _hit("b", 0.8, "lexical"), _hit("d", 0.7, "lexical")]
    semantic = [_hit("a", 0.85, "qdrant"), _hit("c", 0.75, "qdrant"), _hit("e", 0.65, "qdrant")]

    ranked, telemetry = fusion.fuse(
        lexical_results=lexical,
        semantic_results=semantic,
        top_k=5,
        fusion_policy=policy,
        explain=True,
    )
    return ranked, telemetry


def test_hybrid_fusion_balanced_explain_payload():
    ranked, telemetry = _ranked_ids("balanced")

    assert len(ranked) == 5
    assert ranked[0].engram.id == "a"
    assert ranked[0].metadata["component_scores"]["fused"] > 0
    assert set(ranked[0].metadata["retrieval_sources"]) == {"lexical", "semantic"}
    assert telemetry["overlap_candidates"] == 1.0


def test_hybrid_fusion_non_explain_minimal_payload():
    fusion = HybridFusion()
    lexical = [_hit("x", 1.0, "lexical")]
    semantic = [_hit("y", 1.0, "qdrant")]

    ranked, _ = fusion.fuse(
        lexical_results=lexical,
        semantic_results=semantic,
        top_k=2,
        fusion_policy="lexical_dominant",
        explain=False,
    )

    assert len(ranked) == 2
    for hit in ranked:
        assert "component_scores" not in hit.metadata
        assert "fusion_policy" not in hit.metadata
        assert "retrieval_sources" in hit.metadata


def test_hybrid_fusion_all_policies_are_deterministic_and_deduped():
    for policy in ["semantic_dominant", "balanced", "lexical_dominant"]:
        ranked1, telemetry1 = _ranked_ids(policy)
        ranked2, telemetry2 = _ranked_ids(policy)

        ids1 = [h.engram.id for h in ranked1]
        ids2 = [h.engram.id for h in ranked2]
        assert ids1 == ids2
        assert len(ids1) == len(set(ids1))  # dedupe check

        assert telemetry1 == telemetry2
        assert telemetry1["union_candidates"] == 5.0
        assert telemetry1["overlap_candidates"] == 1.0


def test_hybrid_fusion_stable_tie_ordering():
    fusion = HybridFusion()
    lexical = [_hit("z2", 0.9, "lexical"), _hit("z1", 0.8, "lexical")]
    semantic = []

    ranked, _ = fusion.fuse(
        lexical_results=lexical,
        semantic_results=semantic,
        top_k=2,
        fusion_policy="lexical_dominant",
        explain=True,
    )

    # With lexical-only and deterministic rank normalization,
    # first lexical hit should remain first across tie-adjacent candidates.
    assert [r.engram.id for r in ranked] == ["z2", "z1"]
