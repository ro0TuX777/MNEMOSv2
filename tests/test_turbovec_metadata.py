import pytest
import os
import json
from mnemos.retrieval.turbovec_metadata import TurbovecMetadata

@pytest.fixture
def metadata_db(tmp_path):
    db_path = tmp_path / "test_metadata.sqlite"
    return TurbovecMetadata(str(db_path))

def test_upsert_and_retrieve(metadata_db):
    t_id1 = metadata_db.upsert_engram(
        engram_uuid="uuid-1",
        source_uri="doc://doc1",
        content="This is the first document.",
        metadata={"author": "alice"},
        governance={"level": "public"},
        content_hash="hash1",
        created_at="2026-05-16",
        updated_at="2026-05-16"
    )
    assert t_id1 == 1
    
    engram = metadata_db.get_engram("uuid-1")
    assert engram["engram_uuid"] == "uuid-1"
    assert engram["turbovec_id"] == 1
    assert engram["metadata_json"]["author"] == "alice"
    assert engram["governance_json"]["level"] == "public"
    
def test_soft_delete(metadata_db):
    metadata_db.upsert_engram("uuid-2", "doc2", "content2", {}, {}, "hash2", "date", "date")
    assert metadata_db.get_engram("uuid-2") is not None
    
    metadata_db.delete_engram("uuid-2")
    assert metadata_db.get_engram("uuid-2") is None
    
def test_lexical_search(metadata_db):
    metadata_db.upsert_engram("uuid-3", "doc3", "Quick brown fox jumps over the lazy dog.", {}, {}, "hash3", "date", "date")
    metadata_db.upsert_engram("uuid-4", "doc4", "The fast rabbit hops.", {}, {}, "hash4", "date", "date")
    
    results = metadata_db.lexical_search("fox")
    assert len(results) == 1
    assert results[0]["engram_uuid"] == "uuid-3"

def test_candidate_filtering(metadata_db):
    metadata_db.upsert_engram("uuid-5", "doc5", "Content 5", {"type": "A"}, {}, "hash5", "date", "date")
    metadata_db.upsert_engram("uuid-6", "doc6", "Content 6", {"type": "B"}, {}, "hash6", "date", "date")
    
    results = metadata_db.filter_candidates({"type": "A"})
    assert len(results) == 1
    assert results[0]["engram_uuid"] == "uuid-5"
    
    results = metadata_db.filter_candidates({"type": "B"}, candidate_uuids=["uuid-5", "uuid-6"])
    assert len(results) == 1
    assert results[0]["engram_uuid"] == "uuid-6"
    
def test_uuid_to_uint64_allocation(metadata_db):
    id1 = metadata_db.upsert_engram("uuid-7", "doc", "text", {}, {}, "h", "d", "d")
    id2 = metadata_db.upsert_engram("uuid-8", "doc", "text", {}, {}, "h", "d", "d")
    id3 = metadata_db.upsert_engram("uuid-9", "doc", "text", {}, {}, "h", "d", "d")
    assert id1 == 1
    assert id2 == 2
    assert id3 == 3
    
    # Update shouldn't change ID
    id3_updated = metadata_db.upsert_engram("uuid-9", "doc", "text updated", {}, {}, "h2", "d", "d2")
    assert id3_updated == 3
