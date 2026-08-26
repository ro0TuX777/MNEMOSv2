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
    assert result["evidence_block"] == ""
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
    assert "R1 enforcement is not retained." in result["evidence_block"]
    assert "R1 enforcement is not retained." in json.dumps(ollama.payload)


def test_run_query_uses_search_raw_metadata_when_available():
    class RawClient:
        def search_raw(self, query, *, top_k, retrieval_mode, fusion_policy):
            assert query == "workflow"
            return type(
                "Response",
                (),
                {
                    "ok": True,
                    "status": "healthy",
                    "error": None,
                    "data": {
                        "results": [
                            {
                                "engram": {
                                    "id": "raw-hit",
                                    "content": "Workflow evidence.",
                                    "metadata": {"source_path": "workflow.pdf"},
                                },
                                "score": 0.77,
                                "tier": "qdrant",
                                "tiers": ["qdrant"],
                                "rank": 1,
                            }
                        ],
                        # Real service shape: mode/fingerprint nest under meta.
                        "meta": {
                            "retrieval_mode": "hybrid",
                            "fusion_policy": "balanced",
                            "retrieval_fingerprint": {"mode": "hybrid", "profile": "test"},
                        },
                    },
                },
            )()

    class RecordingOllama:
        def chat(self, payload):
            return {"message": {"content": "Answer. [1]"}}

    result = adapter.run_query(
        query="workflow",
        model="llama3.1",
        mnemos_client=RawClient(),
        ollama_client=RecordingOllama(),
    )

    assert result["status"] == "ok"
    assert result["retrieval_metadata"]["retrieval_mode"] == "hybrid"
    assert result["retrieval_metadata"]["fusion_policy"] == "balanced"
    assert result["retrieval_metadata"]["retrieval_fingerprint"] == {
        "mode": "hybrid",
        "profile": "test",
    }


def test_run_query_filters_retrieval_when_query_mentions_filename():
    class RawClient:
        def __init__(self):
            self.filters = None

        def search_raw(self, query, *, top_k, retrieval_mode, fusion_policy, filters=None):
            self.filters = filters
            return type(
                "Response",
                (),
                {
                    "ok": True,
                    "status": "healthy",
                    "error": None,
                    "data": {
                        "results": [
                            {
                                "engram": {
                                    "id": "title5-hit",
                                    "content": "Title 5 evidence.",
                                    "source": "file:///app/data/research_uploads/USCODE-2024-title5.pdf",
                                    "metadata": {"filename": "USCODE-2024-title5.pdf"},
                                },
                                "score": 0.64,
                                "tier": "qdrant",
                                "tiers": ["qdrant"],
                                "rank": 1,
                            }
                        ],
                        "meta": {"retrieval_mode": "semantic"},
                    },
                },
            )()

    class RecordingOllama:
        def chat(self, payload):
            return {"message": {"content": "Title 5 answer. [1]"}}

    mnemos = RawClient()
    result = adapter.run_query(
        query="Using MNEMOS evidence from **USCODE-2024-title5.pdf**, summarize it.",
        model="llama3.1",
        mnemos_client=mnemos,
        ollama_client=RecordingOllama(),
    )

    assert result["status"] == "ok"
    assert mnemos.filters == {"metadata.filename": "USCODE-2024-title5.pdf"}
    assert result["retrieval_metadata"]["artifact_filter"] == {
        "metadata.filename": "USCODE-2024-title5.pdf"
    }


def test_filename_filter_ignores_surrounding_prompt_text_and_punctuation():
    assert adapter.filename_filter_from_query(
        "Using MNEMOS evidence from USCODE-2024-title5.pdf, answer in one sentence."
    ) == {"metadata.filename": "USCODE-2024-title5.pdf"}


def test_filename_filter_uses_manifest_stored_path_for_reuploaded_file(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "research_uploads"
    manifest_dir.mkdir()
    (manifest_dir / ".manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "identity_key": "mnemos::uscode-2024-title34.pdf",
                        "stored_path": "USCODE-2024-title34-1.pdf",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "RESEARCH_UPLOAD_DIR", manifest_dir)

    assert adapter.filename_filter_from_query(
        "Can you summarize the key provisions of USCODE-2024-title34.pdf?"
    ) == {"metadata.filename": "USCODE-2024-title34-1.pdf"}


class _ScriptedOllama:
    """Returns queued chat responses; records every payload it receives."""

    def __init__(self, responses):
        self.payloads = []
        self._responses = list(responses)

    def chat(self, payload):
        self.payloads.append(payload)
        return self._responses.pop(0)


class _StreamingOllama:
    def __init__(self, chunks):
        self.payload = None
        self._chunks = list(chunks)

    def chat_stream(self, payload):
        self.payload = payload
        yield from self._chunks


def test_build_chat_payload_forwards_history_between_system_and_query():
    payload = adapter.build_ollama_chat_payload(
        model="llama3.1",
        query="What about its abstention gap?",
        evidence_block="[1] source=docs/example.md score=0.9\nEvidence text",
        history=[
            {"role": "user", "content": "What is R1?"},
            {"role": "assistant", "content": "R1 was closed. [1]"},
            {"role": "tool", "content": "should be dropped"},
        ],
    )

    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert payload["messages"][1]["content"] == "What is R1?"
    assert payload["messages"][2]["content"] == "R1 was closed. [1]"
    assert "resolve references" in payload["messages"][0]["content"]
    assert "What about its abstention gap?" in payload["messages"][-1]["content"]


def test_build_chat_payload_stream_flag_controls_ollama_streaming():
    payload = adapter.build_ollama_chat_payload(
        model="llama3.1",
        query="q",
        evidence_block="e",
        stream=True,
    )
    assert payload["stream"] is True


def test_condense_query_with_history_rewrites_and_records_metadata():
    ollama = _ScriptedOllama(
        [{"message": {"content": "What is the R1 abstention gap?"}}]
    )
    query, metadata = adapter.condense_query_with_history(
        query="What about its abstention gap?",
        history=[
            {"role": "user", "content": "What is R1?"},
            {"role": "assistant", "content": "R1 was closed."},
        ],
        model="llama3.1",
        ollama_client=ollama,
    )

    assert query == "What is the R1 abstention gap?"
    assert metadata["query_condensed"] is True
    assert metadata["original_query"] == "What about its abstention gap?"
    assert metadata["retrieval_query"] == "What is the R1 abstention gap?"
    assert "Standalone rewrite:" in ollama.payloads[0]["messages"][1]["content"]


def test_condense_query_with_history_falls_back_on_failure():
    class ExplodingOllama:
        def chat(self, payload):
            raise RuntimeError("ollama unavailable")

    query, metadata = adapter.condense_query_with_history(
        query="What about it?",
        history=[{"role": "user", "content": "Tell me about R1"}],
        model="llama3.1",
        ollama_client=ExplodingOllama(),
    )

    assert query == "What about it?"
    assert metadata["query_condensed"] is False
    assert "ollama unavailable" in metadata["condense_error"]


def test_run_query_condenses_follow_up_before_retrieval():
    searches = []

    class RecordingMnemos:
        def search(self, query, *, top_k, retrieval_mode, fusion_policy):
            searches.append(query)
            return [_hit("R1 abstention details.", rank=1)]

    ollama = _ScriptedOllama(
        [
            {"message": {"content": "What is the R1 abstention gap?"}},
            {"message": {"content": "The gap is documented. [1]"}},
        ]
    )
    result = adapter.run_query(
        query="What about its abstention gap?",
        model="llama3.1",
        history=[
            {"role": "user", "content": "What is R1?"},
            {"role": "assistant", "content": "R1 was closed."},
        ],
        mnemos_client=RecordingMnemos(),
        ollama_client=ollama,
    )

    assert searches == ["What is the R1 abstention gap?"]
    assert result["status"] == "ok"
    assert result["retrieval_metadata"]["query_condensed"] is True
    assert result["retrieval_metadata"]["history_turns"] == 2
    # The generation prompt keeps the user's original wording and the history.
    final_messages = ollama.payloads[1]["messages"]
    assert final_messages[1]["content"] == "What is R1?"
    assert "What about its abstention gap?" in final_messages[-1]["content"]


def test_run_query_stream_yields_retrieval_deltas_and_done():
    class OneHitClient:
        def search(self, query, *, top_k, retrieval_mode, fusion_policy):
            return [_hit("Workflow evidence.", rank=1)]

    ollama = _StreamingOllama(
        [
            {"message": {"content": "Hello "}, "done": False},
            {"message": {"content": "world. [1]"}, "done": False},
            {"message": {"content": ""}, "done": True, "done_reason": "stop"},
        ]
    )
    events = list(
        adapter.run_query_stream(
            query="workflow",
            model="llama3.1",
            mnemos_client=OneHitClient(),
            ollama_client=ollama,
        )
    )

    assert [event["event"] for event in events] == ["retrieval", "delta", "delta", "done"]
    assert events[0]["result"]["citations"][0]["source"] == "docs/example.md"
    assert events[1]["content"] == "Hello "
    done = events[-1]["result"]
    assert done["status"] == "ok"
    assert done["answer"] == "Hello world. [1]"
    assert done["ollama_response"]["done_reason"] == "stop"
    assert ollama.payload["stream"] is True


def test_run_query_stream_terminal_no_evidence_yields_single_done():
    class EmptyClient:
        def search(self, *args, **kwargs):
            return []

    class NeverOllama:
        def chat_stream(self, payload):
            raise AssertionError("Ollama must not be called without evidence")

    events = list(
        adapter.run_query_stream(
            query="missing",
            model="llama3.1",
            mnemos_client=EmptyClient(),
            ollama_client=NeverOllama(),
        )
    )

    assert [event["event"] for event in events] == ["done"]
    assert events[0]["result"]["status"] == "no_evidence"


def test_run_query_stream_reports_ollama_error_with_partial_answer():
    class OneHitClient:
        def search(self, query, *, top_k, retrieval_mode, fusion_policy):
            return [_hit("Workflow evidence.", rank=1)]

    class ExplodingOllama:
        def chat_stream(self, payload):
            yield {"message": {"content": "partial"}, "done": False}
            raise RuntimeError("boom")

    events = list(
        adapter.run_query_stream(
            query="workflow",
            model="llama3.1",
            mnemos_client=OneHitClient(),
            ollama_client=ExplodingOllama(),
        )
    )

    assert [event["event"] for event in events] == ["retrieval", "delta", "done"]
    done = events[-1]["result"]
    assert done["status"] == "ollama_error"
    assert done["answer"] == "partial"
    assert "boom" in done["warning"]


def test_run_query_reports_mnemos_error_when_search_raw_not_ok():
    class FailingClient:
        def search_raw(self, query, *, top_k, retrieval_mode, fusion_policy):
            return type(
                "Response",
                (),
                {
                    "ok": False,
                    "status": "unavailable",
                    "error": "connection refused",
                    "data": {},
                },
            )()

    class NeverOllama:
        def chat(self, payload):
            raise AssertionError("Ollama must not be called when MNEMOS errors")

    result = adapter.run_query(
        query="workflow",
        model="llama3.1",
        mnemos_client=FailingClient(),
        ollama_client=NeverOllama(),
    )

    assert result["status"] == "mnemos_error"
    assert "MNEMOS search failed" in result["warning"]
    assert result["citations"] == []
    assert result["retrieval_metadata"]["response_status"] == "unavailable"
    assert result["retrieval_metadata"]["response_error"] == "connection refused"
