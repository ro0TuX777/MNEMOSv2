"""Tests for the pgvector retrieval tier."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult


@pytest.fixture
def mock_pool():
    """Create a mock psycopg connection pool."""
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()

    # Context manager setup
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    return pool, conn, cursor


@pytest.fixture
def pgvector_tier(mock_pool):
    """Create a PgvectorTier with mocked dependencies."""
    pool, conn, cursor = mock_pool

    with patch.dict("sys.modules", {
        "psycopg": MagicMock(),
        "psycopg_pool": MagicMock(),
    }):
        from mnemos.retrieval.pgvector_tier import PgvectorTier
        tier = PgvectorTier.__new__(PgvectorTier)
        tier._dsn = "postgresql://test:test@localhost:5432/test"
        tier._table_name = "test_vectors"
        tier._embedding_model_name = "BAAI/bge-base-en-v1.5"
        tier._embedding_dim = 768
        tier._gpu_device = "cpu"
        tier._pool = pool
        tier._model = None
        return tier


class TestPgvectorTier:
    def test_tier_name(self, pgvector_tier):
        assert pgvector_tier.tier_name == "pgvector"

    def test_index_empty(self, pgvector_tier):
        assert pgvector_tier.index([]) == 0

    def test_index_with_engrams(self, pgvector_tier, mock_pool):
        """Test indexing engrams with mocked embeddings."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)
        pgvector_tier._model = mock_model

        engrams = [
            Engram(id="e1", content="hello world", source="test"),
            Engram(id="e2", content="foo bar", source="test"),
        ]
        count = pgvector_tier.index(engrams)
        assert count == 2

    def test_search_no_pool(self, pgvector_tier):
        pgvector_tier._pool = None
        results = pgvector_tier.search("test query")
        assert results == []

    def test_search_returns_results(self, pgvector_tier, mock_pool):
        """Test search with mocked Postgres response."""
        pool, conn, cursor = mock_pool

        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        pgvector_tier._model = mock_model

        # Mock cursor.fetchall() to return a row
        cursor.fetchall.return_value = [
            (
                "e1",                  # id
                "hello world",         # content
                "test",                # source
                0.9,                   # confidence
                ["greeting"],          # neuro_tags
                "2026-01-01",          # created_at
                [],                    # edges
                {},                    # metadata
                0.95,                  # score
            ),
        ]

        results = pgvector_tier.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].engram.content == "hello world"
        assert results[0].tier == "pgvector"

    def test_search_with_filters(self, pgvector_tier, mock_pool):
        """Test search with metadata filters — the Governance Native value prop."""
        pool, conn, cursor = mock_pool

        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        pgvector_tier._model = mock_model

        cursor.fetchall.return_value = [
            ("e3", "classified doc", "dept-a", 0.95, ["compliance"], "", [], {"department": "finance"}, 0.88),
        ]

        results = pgvector_tier.search(
            "financial report",
            top_k=10,
            filters={
                "source": "dept-a",
                "confidence_min": 0.8,
                "metadata.department": "finance",
            },
        )
        assert len(results) == 1
        assert results[0].engram.source == "dept-a"

    def test_delete_empty(self, pgvector_tier):
        assert pgvector_tier.delete([]) == 0

    def test_delete_calls_pool(self, pgvector_tier, mock_pool):
        pool, conn, cursor = mock_pool
        cursor.rowcount = 2

        count = pgvector_tier.delete(["e1", "e2"])
        assert count == 2

    def test_get_found(self, pgvector_tier, mock_pool):
        """Test direct ID lookup."""
        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = (
            "e1", "test content", "src", 0.8, ["tag1"], "2026-01-01", [], {},
        )

        engram = pgvector_tier.get("e1")
        assert engram is not None
        assert engram.id == "e1"
        assert engram.content == "test content"

    def test_get_not_found(self, pgvector_tier, mock_pool):
        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = None

        assert pgvector_tier.get("nonexistent") is None

    def test_stats(self, pgvector_tier, mock_pool):
        pool, conn, cursor = mock_pool
        cursor.fetchone.side_effect = [
            (42,),       # COUNT(*)
            (1048576,),  # pg_total_relation_size
            (524288,),   # pg_indexes_size
        ]

        s = pgvector_tier.stats()
        assert s["tier"] == "pgvector"
        assert s["document_count"] == 42
        assert s["index_type"] == "hnsw"
        assert s["gpu_device"] == "cpu"
        assert s["table_size_mb"] == 1.0
        assert s["index_size_mb"] == 0.5

    def test_stats_no_pool(self, pgvector_tier):
        pgvector_tier._pool = None
        s = pgvector_tier.stats()
        assert s["tier"] == "pgvector"
        assert s["document_count"] == 0
