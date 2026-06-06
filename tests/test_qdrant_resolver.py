import pytest
from unittest.mock import MagicMock
from mnemos.engram.model import Engram
from mnemos.retrieval.resolvers.qdrant_resolver import QdrantEngramResolver
from mnemos.retrieval.qdrant_tier import QdrantTier
from mnemos.retrieval.graph_tier import GraphTier, SearchResult

def test_mutating_methods_raise():
    tier_mock = MagicMock(spec=QdrantTier)
    resolver = QdrantEngramResolver(tier_mock)
    
    with pytest.raises(NotImplementedError):
        resolver.add_edge("1", "2")
        
    with pytest.raises(NotImplementedError):
        resolver.update_engram(Engram(id="1", content="t", embedding=[], edges=[], neuro_tags=[]))
        
    with pytest.raises(NotImplementedError):
        resolver.delete_edge("1", "2")
        
    with pytest.raises(NotImplementedError):
        resolver.save()

def test_qdrant_batch_retrieval_avoids_n_plus_one():
    tier_mock = MagicMock(spec=QdrantTier)
    tier_mock._client = MagicMock()
    # Mock to_point_id to just return the id
    tier_mock._to_point_id.side_effect = lambda x: x
    tier_mock._collection_name = "test_collection"
    
    resolver = QdrantEngramResolver(tier_mock)
    
    # Pre-fetch 50 unique ids
    ids_to_fetch = [str(i) for i in range(50)]
    
    # Mock retrieve to return empty list
    tier_mock._client.retrieve.return_value = []
    
    resolver.prefetch(ids_to_fetch)
    
    # Assert retrieve called exactly once
    tier_mock._client.retrieve.assert_called_once()
    
def test_missing_neighbor_ids_do_not_crash():
    tier_mock = MagicMock(spec=QdrantTier)
    tier_mock._client = MagicMock()
    tier_mock._to_point_id.side_effect = lambda x: x
    tier_mock._client.retrieve.return_value = [] # Missing results
    
    resolver = QdrantEngramResolver(tier_mock)
    # Should not throw exception
    resolver.prefetch(["missing1", "missing2"])
    
    # get_by_id should return None
    assert resolver.get_by_id("missing1") is None

def test_qdrant_failure_falls_back_safely():
    tier_mock = MagicMock(spec=QdrantTier)
    tier_mock._client = MagicMock()
    tier_mock._to_point_id.side_effect = lambda x: x
    
    # Mock retrieve to throw an exception
    tier_mock._client.retrieve.side_effect = Exception("Connection Refused")
    
    resolver = QdrantEngramResolver(tier_mock)
    # Should catch exception and emit telemetry
    resolver.prefetch(["1"])
    
    # Cache should be empty, get_by_id returns None
    assert resolver.get_by_id("1") is None

def test_downstream_rejection_logic():
    from mnemos.engram.model import Engram
    from mnemos.governance.models.memory_state import GovernanceMeta
    tier_mock = MagicMock(spec=QdrantTier)
    resolver = QdrantEngramResolver(tier_mock)
    
    graph_tier = GraphTier(resolver)
    
    e1 = Engram(id="1", content="t", embedding=[], edges=[], neuro_tags=[])
    # Lineage incomplete
    assert not graph_tier._is_lineage_complete(e1)
    
    e2 = Engram(id="2", content="t", embedding=[], edges=[], neuro_tags=[])
    e2.metadata = {"artifact_id": "a", "source_uri": "s", "chunk_id": "c"}
    # Lineage complete
    assert graph_tier._is_lineage_complete(e2)
    
    e3 = Engram(id="3", content="t", embedding=[], edges=[], neuro_tags=[])
    e3.governance = GovernanceMeta(lifecycle_state="archived", conflict_status="none", deletion_state="active")
    # Governance blocked
    assert graph_tier._is_blocked(e3)
