from types import SimpleNamespace

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos_sdk.client import MnemosClient, MnemosResponse
from mnemos_sdk.config import MnemosConfig
from service.app import MnemosRuntime


def test_evidence_packet_preserves_document_provenance():
    engram = Engram(
        id="engram-1",
        content="paper excerpt",
        source="file:///example-corpus/example_research_note.pdf",
        metadata={
            "filename": "example_research_note.pdf",
            "title": "Example Research Note",
            "file_id": "file-1",
            "file_hash": "hash-1",
            "chunk_index": 1,
            "chunk_count": 17,
            "page_start": 2,
            "page_end": 3,
            "char_start": 100,
            "char_end": 250,
            "ingestion_source": "local-folder",
            "canonical_store": "mnemos",
        },
    )
    result = SearchResult(engram=engram, score=0.88091, tier="qdrant")

    evidence = MnemosRuntime._build_evidence_packet(result, rank=1)

    assert evidence["engram_id"] == "engram-1"
    assert evidence["rank"] == 1
    assert evidence["score"] == 0.8809
    assert evidence["filename"] == "example_research_note.pdf"
    assert evidence["document_title"] == "Example Research Note"
    assert evidence["chunk_index"] == 1
    assert evidence["chunk_count"] == 17
    assert evidence["label"] == "example_research_note.pdf chunk 1/17"
    assert evidence["page_start"] == 2
    assert evidence["char_end"] == 250
    assert evidence["canonical_store"] == "mnemos"


def test_evidence_summary_groups_sources_and_counts_chunks():
    entries = [
        {
            "evidence": {
                "filename": "example_research_note.pdf",
                "document_title": "Paper",
                "source_uri": "file:///paper.pdf",
                "score": 0.8,
                "rank": 1,
                "chunk_index": 1,
            }
        },
        {
            "evidence": {
                "filename": "example_research_note.pdf",
                "document_title": "Paper",
                "source_uri": "file:///paper.pdf",
                "score": 0.9,
                "rank": 2,
                "chunk_index": 2,
            }
        },
    ]

    summary = MnemosRuntime._build_evidence_summary(entries)

    assert summary["source_count"] == 1
    source = summary["sources"][0]
    assert source["filename"] == "example_research_note.pdf"
    assert source["title"] == "Paper"
    assert source["top_score"] == 0.9
    assert source["chunks_returned"] == 2
    assert source["chunk_indices"] == [1, 2]
    assert source["ranks"] == [1, 2]


def test_sdk_search_hit_exposes_rank_and_evidence(monkeypatch):
    client = MnemosClient(MnemosConfig(enabled=True, base_url="http://mnemos.local"))

    def fake_request(method, path, *, payload=None):
        return MnemosResponse(
            status="healthy",
            source="test",
            error=None,
            data={
                "results": [
                    {
                        "engram": {"id": "engram-1", "content": "paper excerpt"},
                        "score": 0.8809,
                        "rank": 1,
                        "tier": "qdrant",
                        "tiers": ["qdrant"],
                        "evidence": {
                            "filename": "example_research_note.pdf",
                            "chunk_index": 1,
                            "chunk_count": 17,
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(client, "_request", fake_request)

    hits = client.search("paper", top_k=1)

    assert hits[0].rank == 1
    assert hits[0].evidence["filename"] == "example_research_note.pdf"
    assert hits[0].evidence["chunk_count"] == 17
