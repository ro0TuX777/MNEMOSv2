"""Tests for the Qdrant retrieval tier.

Updated for Qdrant v1.17+ / query_points() API.
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.retrieval.base import SearchResult


class MockQdrantHit:
    """Mock for qdrant_client search/query results."""
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class MockQueryResponse:
    """Mock for qdrant_client query_points response."""
    def __init__(self, points):
        self.points = points


class MockCollectionInfo:
    """Mock for qdrant_client collection info."""
    def __init__(self, name, points_count=0):
        self.name = name
        self.points_count = points_count


class MockCollections:
    def __init__(self, collections):
        self.collections = collections


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    # QdrantClient is imported locally inside _initialize(), so we need
    # create=True to make the patch target accessible at module scope.
    with patch("mnemos.retrieval.qdrant_tier.QdrantClient", create=True) as MockClient:
        client = MockClient.return_value
        client.get_collections.return_value = MockCollections([])
        yield client


@pytest.fixture
def qdrant_tier(mock_qdrant_client):
    """Create a QdrantTier with mocked dependencies."""
    with patch.dict("sys.modules", {"qdrant_client": MagicMock(), "qdrant_client.models": MagicMock()}):
        from mnemos.retrieval.qdrant_tier import QdrantTier
        tier = QdrantTier.__new__(QdrantTier)
        tier._url = "http://localhost:6333"
        tier._collection_name = "test_engrams"
        tier._embedding_model_name = "BAAI/bge-base-en-v1.5"
        tier._embedding_dim = 768
        tier._gpu_device = "cpu"
        tier._client = mock_qdrant_client
        tier._model = None
        return tier


class TestQdrantTier:
    def test_tier_name(self, qdrant_tier):
        assert qdrant_tier.tier_name == "qdrant"

    def test_qdrant_min_version(self, qdrant_tier):
        """Ensure the tier documents the minimum Qdrant version."""
        assert qdrant_tier.QDRANT_MIN_VERSION == "1.17.0"

    def test_index_empty(self, qdrant_tier):
        assert qdrant_tier.index([]) == 0

    def test_index_with_engrams(self, qdrant_tier):
        """Test indexing engrams with mocked embeddings."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)
        qdrant_tier._model = mock_model

        engrams = [
            Engram(id="e1", content="hello world", source="test"),
            Engram(id="e2", content="foo bar", source="test"),
        ]
        count = qdrant_tier.index(engrams)
        assert count == 2
        qdrant_tier._client.upsert.assert_called_once()

    def test_index_persists_governance_payload(self, qdrant_tier):
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        qdrant_tier._model = mock_model

        engram = Engram(id="e1", content="hello", source="test")
        engram.governance = GovernanceMeta(
            entity_key="project:x",
            attribute_key="status",
            normalized_value="active",
        )

        assert qdrant_tier.index([engram]) == 1
        point = qdrant_tier._client.upsert.call_args.kwargs["points"][0]
        assert point.payload["gov_entity_key"] == "project:x"
        assert point.payload["gov_attribute_key"] == "status"
        assert point.payload["gov_normalized_value"] == "active"

    def test_index_with_update_mode(self, qdrant_tier):
        """Test that index() accepts update_mode parameter."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        qdrant_tier._model = mock_model

        engrams = [Engram(id="e1", content="hello", source="test")]
        count = qdrant_tier.index(engrams, update_mode="insert")
        assert count == 1
        qdrant_tier._client.upsert.assert_called_once()

    def test_search_empty_client(self, qdrant_tier):
        qdrant_tier._client = None
        results = qdrant_tier.search("test query")
        assert results == []

    def test_search_returns_results(self, qdrant_tier):
        """Test search with mocked Qdrant query_points response."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        qdrant_tier._model = mock_model

        # query_points() returns a response object with a .points attribute
        qdrant_tier._client.query_points.return_value = MockQueryResponse([
            MockQdrantHit(
                id="e1", score=0.95,
                payload={"content": "hello world", "source": "test",
                         "confidence": 0.9, "neuro_tags": ["greeting"],
                         "created_at": "2026-01-01T00:00:00Z", "edges": []},
            ),
        ])

        results = qdrant_tier.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].engram.content == "hello world"
        assert results[0].tier == "qdrant"

        # Verify query_points was called (not deprecated search)
        qdrant_tier._client.query_points.assert_called_once()

    def test_search_with_filters(self, qdrant_tier):
        """Test that filters are passed to query_points."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        qdrant_tier._model = mock_model

        qdrant_tier._client.query_points.return_value = MockQueryResponse([])

        results = qdrant_tier.search(
            "test", top_k=5,
            filters={"metadata.department": "finance"}
        )
        assert results == []
        # Verify filter was passed (query_filter kwarg is not None)
        call_kwargs = qdrant_tier._client.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is not None

    def test_nomic_search_uses_prefix_and_named_vectors(self, qdrant_tier):
        """Nomic v1.5 cutover searches dense_64 prefetch + dense_768 rescore."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        qdrant_tier._model = mock_model
        qdrant_tier._embedding_model_name = "nomic-ai/nomic-embed-text-v1.5"
        qdrant_tier._embedding_dim = 768
        qdrant_tier._client.query_points.return_value = MockQueryResponse([])

        results = qdrant_tier.search("hello", top_k=5)

        assert results == []
        mock_model.encode.assert_called_once()
        assert mock_model.encode.call_args.args[0] == ["search_query: hello"]
        call_kwargs = qdrant_tier._client.query_points.call_args.kwargs
        assert call_kwargs["using"] == "dense_768"
        assert "prefetch" in call_kwargs

    def test_nomic_index_uses_document_prefix_and_named_vectors(self, qdrant_tier):
        """Nomic v1.5 indexing stores both MRL and full named vectors."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        qdrant_tier._model = mock_model
        qdrant_tier._embedding_model_name = "nomic-ai/nomic-embed-text-v1.5"
        qdrant_tier._embedding_dim = 768

        count = qdrant_tier.index([Engram(id="e1", content="hello", source="test")])

        assert count == 1
        mock_model.encode.assert_called_once()
        assert mock_model.encode.call_args.args[0] == ["search_document: hello"]
        point = qdrant_tier._client.upsert.call_args.kwargs["points"][0]
        assert set(point.vector) == {"dense_64", "dense_768"}
        assert len(point.vector["dense_64"]) == 64
        assert len(point.vector["dense_768"]) == 768

    def test_delete_empty(self, qdrant_tier):
        assert qdrant_tier.delete([]) == 0

    def test_delete_calls_client(self, qdrant_tier):
        count = qdrant_tier.delete(["e1", "e2"])
        assert count == 2
        qdrant_tier._client.delete.assert_called_once()

    def test_get_found(self, qdrant_tier):
        """Test direct ID lookup."""
        mock_point = MagicMock()
        mock_point.id = "e1"
        mock_point.payload = {
            "content": "test content", "source": "src",
            "confidence": 0.8, "neuro_tags": ["tag1"],
            "created_at": "2026-01-01T00:00:00Z", "edges": [],
        }
        qdrant_tier._client.retrieve.return_value = [mock_point]

        engram = qdrant_tier.get("e1")
        assert engram is not None
        assert engram.id == "e1"
        assert engram.content == "test content"

    def test_get_not_found(self, qdrant_tier):
        qdrant_tier._client.retrieve.return_value = []
        assert qdrant_tier.get("nonexistent") is None

    def test_stats(self, qdrant_tier):
        info = MockCollectionInfo("test_engrams", points_count=42)
        qdrant_tier._client.get_collection.return_value = info

        s = qdrant_tier.stats()
        assert s["tier"] == "qdrant"
        assert s["document_count"] == 42
        assert s["gpu_device"] == "cpu"

    def test_trace_id_generation(self, qdrant_tier):
        """Verify trace IDs are generated as valid UUIDs."""
        import uuid
        trace_id = qdrant_tier._make_trace_id()
        parsed = uuid.UUID(trace_id)  # will raise if invalid
        assert str(parsed) == trace_id

    def test_hit_to_result_static_method(self, qdrant_tier):
        """Test the static _hit_to_result converter."""
        hit = MockQdrantHit(
            id="abc",
            score=0.88,
            payload={
                "_mnemos_id": "real_id",
                "content": "some content",
                "source": "src",
                "confidence": 0.7,
                "neuro_tags": ["a", "b"],
                "created_at": "2026-01-01",
                "edges": ["x"],
                "app_department": "hr",
                "gov_entity_key": "project:x",
                "gov_attribute_key": "status",
                "gov_normalized_value": "active",
            }
        )
        result = qdrant_tier._hit_to_result(hit)
        assert result.engram.id == "real_id"
        assert result.score == 0.88
        assert result.tier == "qdrant"
        assert result.engram.metadata == {"department": "hr"}
        assert result.engram.governance is not None
        assert result.engram.governance.entity_key == "project:x"
        assert result.engram.governance.attribute_key == "status"

    def test_build_filter_none(self, qdrant_tier):
        """Verify None filters produce None output."""
        assert qdrant_tier._build_filter(None) is None
        assert qdrant_tier._build_filter({}) is None
