"""
Tests for RelevanceFeedbackAdapter
"""
import pytest
import time
from unittest.mock import MagicMock

from mnemos.retrieval.relevance_feedback import (
    ExemplarCache,
    InMemoryFeedbackStore,
    RelevanceFeedbackAdapter,
)
from mnemos.retrieval.base import SearchResult
from mnemos.engram.model import Engram


class TestExemplarCache:
    def test_put_and_get(self):
        cache = ExemplarCache(max_size=10, ttl_seconds=60)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        cache = ExemplarCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = ExemplarCache(ttl_seconds=0.01)
        cache.put("key1", "value1")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        cache = ExemplarCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_clear(self):
        cache = ExemplarCache()
        cache.put("key", "val")
        cache.clear()
        assert cache.get("key") is None


class TestInMemoryFeedbackStore:
    def test_record_and_get(self):
        store = InMemoryFeedbackStore()
        store.record("q1", "engram_a", "used")
        store.record("q1", "engram_b", "ignored")

        labels = store.get_labels("q1")
        assert "engram_a" in labels["used"]
        assert "engram_b" in labels["ignored"]

    def test_get_empty(self):
        store = InMemoryFeedbackStore()
        labels = store.get_labels("nonexistent")
        assert labels == {"used": [], "ignored": []}

    def test_no_duplicates(self):
        store = InMemoryFeedbackStore()
        store.record("q1", "engram_a", "used")
        store.record("q1", "engram_a", "used")
        assert store.get_labels("q1")["used"].count("engram_a") == 1


class TestRelevanceFeedbackAdapter:
    @pytest.fixture
    def mock_tier(self):
        tier = MagicMock()
        tier._client = MagicMock()
        tier._collection_name = "test"
        tier._to_point_id.side_effect = lambda x: x
        tier._build_filter.return_value = None
        tier.search.return_value = [
            SearchResult(
                engram=Engram(id="fallback1", content="test", source="test"),
                score=0.5,
                tier="qdrant",
            )
        ]
        return tier

    @pytest.fixture
    def adapter(self, mock_tier):
        return RelevanceFeedbackAdapter(mock_tier, max_exemplars=3)

    def test_no_exemplars_uses_standard_search(self, adapter, mock_tier):
        results, telemetry = adapter.search_with_feedback(
            query="test query",
            query_vector=[0.1, 0.2],
            top_k=5,
        )
        assert not telemetry["feedback_applied"]
        assert telemetry["feedback_positive_count"] == 0
        mock_tier.search.assert_called_once()

    def test_record_and_get_exemplars(self, adapter):
        adapter.record_feedback("qhash1", "eng_a", "used")
        adapter.record_feedback("qhash1", "eng_b", "ignored")

        positives, negatives = adapter.get_exemplars("qhash1")
        assert "eng_a" in positives
        assert "eng_b" in negatives

    def test_exemplar_cache_hit(self, adapter):
        adapter.record_feedback("qhash1", "eng_a", "used")
        # First call populates cache
        adapter.get_exemplars("qhash1")
        # Second call hits cache
        pos, neg = adapter.get_exemplars("qhash1")
        assert "eng_a" in pos

    def test_max_exemplars_capped(self, mock_tier):
        adapter = RelevanceFeedbackAdapter(mock_tier, max_exemplars=2)
        for i in range(5):
            adapter.record_feedback("q1", f"pos_{i}", "used")
        positives, _ = adapter.get_exemplars("q1")
        assert len(positives) <= 2

    def test_trace_id_in_telemetry(self, adapter):
        results, telemetry = adapter.search_with_feedback(
            query="test", query_vector=[0.1], top_k=5,
        )
        assert "trace_id" in telemetry
