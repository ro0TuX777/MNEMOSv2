from __future__ import annotations

import json

from mnemos_sdk.client import SearchHit
from tools import mnemos_ollama_chat as adapter


def _hit(
    content: str,
    *,
    source: str = "docs/example.md",
    score: float = 0.91,
    rank: int = 1,
) -> SearchHit:
    return SearchHit(
        engram={
            "id": f"hit-{rank}",
            "content": content,
            "source": source,
            "metadata": {
                "source_path": source,
                "family": "integration_docs",
                "role": "current_state_record",
            },
        },
        score=score,
        tier="qdrant",
        tiers=["qdrant"],
        rank=rank,
    )


def test_format_evidence_block_numbers_and_bounds_hits():
    block, citations = adapter.format_evidence_block(
        [_hit("A" * 60, rank=1), _hit("B" * 60, source="docs/other.md", rank=2)],
        max_chars_per_hit=12,
    )

    assert "[1] source=docs/example.md score=0.9100" in block
    assert "AAAAAAAAAAAA..." in block
    assert "[2] source=docs/other.md score=0.9100" in block
    assert citations == [
        {"index": 1, "engram_id": "hit-1", "source": "docs/example.md", "score": 0.91},
        {"index": 2, "engram_id": "hit-2", "source": "docs/other.md", "score": 0.91},
    ]


def test_build_chat_payload_includes_mfs_boundary_instruction_and_context():
    payload = adapter.build_ollama_chat_payload(
        model="llama3.1",
        query="What supports this?",
        evidence_block="[1] source=docs/example.md score=0.9\nEvidence text",
        temperature=0.2,
        num_predict=300,
    )

    assert payload["model"] == "llama3.1"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.2, "num_predict": 300}
    system = payload["messages"][0]["content"]
    user = payload["messages"][1]["content"]
    assert "MFS-governed MNEMOS memory adapter" in system
    assert "MNEMOS is the evidence source" in system
    assert "Ollama is only the local model runtime" in system
    assert "Do not claim unsupported facts" in system
    assert "MNEMOS_EVIDENCE" in user
    assert "What supports this?" in user


def test_run_query_returns_no_evidence_without_calling_ollama():
    class EmptyClient:
        def search(self, *args, **kwargs):
            return []

    class FailingOllama:
        def chat(self, payload):
            raise AssertionError("Ollama should not be called without evidence")

    result = adapter.run_query(
        query="missing thing",
        model="llama3.1",
        mnemos_client=EmptyClient(),
        ollama_client=FailingOllama(),
    )

    assert result["status"] == "no_evidence"
    assert result["answer"] == ""
    assert result["citations"] == []
    assert result["claim_boundary"] == adapter.CLAIM_BOUNDARY


def test_run_query_calls_ollama_with_mnemos_evidence_and_returns_answer():
    class OneHitClient:
        def search(self, query, *, top_k, retrieval_mode, fusion_policy):
            assert query == "current R1 decision"
            assert top_k == 3
            assert retrieval_mode == "semantic"
            assert fusion_policy == "balanced"
            return [_hit("R1 enforcement is not retained.", rank=1)]

    class RecordingOllama:
        def __init__(self):
            self.payload = None

        def chat(self, payload):
            self.payload = payload
            return {"message": {"content": "R1 is not retained. [1]"}}

    ollama = RecordingOllama()
    result = adapter.run_query(
        query="current R1 decision",
        model="llama3.1",
        top_k=3,
        mnemos_client=OneHitClient(),
        ollama_client=ollama,
    )

    assert result["status"] == "ok"
    assert result["answer"] == "R1 is not retained. [1]"
    assert result["citations"][0]["source"] == "docs/example.md"
    assert "R1 enforcement is not retained." in json.dumps(ollama.payload)
