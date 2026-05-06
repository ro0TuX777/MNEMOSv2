"""
Tests for QdrantHybridFusion (server-side RRF)
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from mnemos.retrieval.qdrant_hybrid import QdrantHybridFusion
from mnemos.retrieval.base import SearchResult
from mnemos.engram.model import Engram


@pytest.fixture
def mock_qdrant_tier():
    """Create a mock QdrantTier with text index ready."""
    tier = MagicMock()
    tier._client = MagicMock()
    tier._collection_name = "test_collection"
    tier._text_index_ready = True
    tier._build_filter.return_value = None

    # _hit_to_result returns a SearchResult
    def fake_hit_to_result(hit):
        return SearchResult(
            engram=Engram(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                source="test",
            ),
            score=hit.score,
            tier="qdrant",
        )

    tier._hit_to_result.side_effect = fake_hit_to_result
    return tier


@pytest.fixture
def hybrid_fusion(mock_qdrant_tier):
    return QdrantHybridFusion(mock_qdrant_tier)


def _make_scored_point(point_id, score, content="test content"):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={"content": content, "_mnemos_id": str(point_id)},
    )


class TestQdrantHybridFusion:
    def test_available_when_ready(self, hybrid_fusion):
        assert hybrid_fusion.available is True

    def test_not_available_without_client(self, mock_qdrant_tier):
        mock_qdrant_tier._client = None
        fusion = QdrantHybridFusion(mock_qdrant_tier)
        assert fusion.available is False

    def test_not_available_without_text_index(self, mock_qdrant_tier):
        mock_qdrant_tier._text_index_ready = False
        fusion = QdrantHybridFusion(mock_qdrant_tier)
        assert fusion.available is False

    def test_fuse_raises_when_unavailable(self, mock_qdrant_tier):
        mock_qdrant_tier._client = None
        fusion = QdrantHybridFusion(mock_qdrant_tier)
        with pytest.raises(RuntimeError, match="not available"):
            fusion.fuse(query="test", query_vector=[0.1, 0.2], top_k=5)

    def test_fuse_returns_results_and_telemetry(self, hybrid_fusion, mock_qdrant_tier):
        """Test successful fusion returns properly structured results."""
        with patch.dict("sys.modules", {
            "qdrant_client": MagicMock(),
            "qdrant_client.models": MagicMock(),
        }):
            # Mock query_points response
            mock_response = SimpleNamespace(
                points=[
                    _make_scored_point("id1", 0.95, "hello world"),
                    _make_scored_point("id2", 0.85, "foo bar"),
                ]
            )
            mock_qdrant_tier._client.query_points.return_value = mock_response

            results, telemetry = hybrid_fusion.fuse(
                query="test query",
                query_vector=[0.1, 0.2, 0.3],
                top_k=10,
            )

            assert len(results) == 2
            assert all(r.tier == "hybrid" for r in results)
            assert all(r.metadata.get("fusion_engine") == "qdrant_rrf" for r in results)
            assert telemetry["fusion_engine"] == "qdrant_rrf"
            assert telemetry["fused_result_count"] == 2.0
            assert "trace_id" in telemetry
            assert "elapsed_ms" in telemetry

    def test_fuse_with_filters(self, hybrid_fusion, mock_qdrant_tier):
        """Test that filters are passed through to the tier."""
        with patch.dict("sys.modules", {
            "qdrant_client": MagicMock(),
            "qdrant_client.models": MagicMock(),
        }):
            mock_response = SimpleNamespace(points=[])
            mock_qdrant_tier._client.query_points.return_value = mock_response

            results, telemetry = hybrid_fusion.fuse(
                query="test",
                query_vector=[0.1],
                top_k=5,
                filters={"source": "test_source"},
            )

            mock_qdrant_tier._build_filter.assert_called_once_with({"source": "test_source"})

    def test_fuse_error_raises_runtime(self, hybrid_fusion, mock_qdrant_tier):
        """Test that Qdrant errors are wrapped in RuntimeError."""
        with patch.dict("sys.modules", {
            "qdrant_client": MagicMock(),
            "qdrant_client.models": MagicMock(),
        }):
            mock_qdrant_tier._client.query_points.side_effect = Exception("connection lost")

            with pytest.raises(RuntimeError, match="Qdrant hybrid fusion failed"):
                hybrid_fusion.fuse(query="test", query_vector=[0.1], top_k=5)

    def test_trace_id_unique(self):
        id1 = QdrantHybridFusion._make_trace_id()
        id2 = QdrantHybridFusion._make_trace_id()
        assert id1 != id2
