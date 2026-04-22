"""Tests for the Qdrant retrieval tier."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult


class MockQdrantHit:
    """Mock for qdrant_client search results."""
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


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
    with patch("mnemos.retrieval.qdrant_tier.QdrantClient") as MockClient:
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

    def test_search_empty_client(self, qdrant_tier):
        qdrant_tier._client = None
        results = qdrant_tier.search("test query")
        assert results == []

    def test_search_returns_results(self, qdrant_tier):
        """Test search with mocked Qdrant response."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        qdrant_tier._model = mock_model

        qdrant_tier._client.search.return_value = [
            MockQdrantHit(
                id="e1", score=0.95,
                payload={"content": "hello world", "source": "test",
                         "confidence": 0.9, "neuro_tags": ["greeting"],
                         "created_at": "2026-01-01T00:00:00Z", "edges": []},
            ),
        ]

        results = qdrant_tier.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].engram.content == "hello world"
        assert results[0].tier == "qdrant"

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
