from __future__ import annotations

from tools.mnemos_ollama_openwebui_proxy import (
    append_evidence_footer,
    create_app,
    extract_latest_user_text,
    normalize_openwebui_model_id,
    should_append_footer,
)


def test_extract_latest_user_text_prefers_last_user_message():
    assert (
        extract_latest_user_text(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "middle"},
                {"role": "user", "content": [{"type": "text", "text": "second"}]},
            ]
        )
        == "second"
    )


def test_v1_models_maps_ollama_tags_to_openai_shape():
    class TagsClient:
        def tags(self):
            return {
                "models": [
                    {"name": "qwen3-coder-next:latest"},
                    {"model": "llama3.2-vision:latest"},
                ]
            }

    app = create_app(ollama_tags_client=TagsClient())
    client = app.test_client()

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.get_json() == {
        "object": "list",
        "data": [
            {
                "id": "qwen3-coder-next:latest",
                "object": "model",
                "owned_by": "ollama",
            },
            {
                "id": "llama3.2-vision:latest",
                "object": "model",
                "owned_by": "ollama",
            },
        ],
    }


def test_v1_api_tags_alias_supports_openwebui_ollama_probe():
    class TagsClient:
        def tags(self):
            return {"models": [{"name": "qwen3-coder-next:latest"}]}

    app = create_app(ollama_tags_client=TagsClient())
    client = app.test_client()

    response = client.get("/v1/api/tags")

    assert response.status_code == 200
    assert response.get_json() == {"models": [{"name": "qwen3-coder-next:latest"}]}


def test_v1_api_ps_alias_supports_openwebui_ollama_probe():
    app = create_app(ollama_tags_client=type("TagsClient", (), {"tags": lambda self: {"models": []}})())
    client = app.test_client()

    response = client.get("/v1/api/ps")

    assert response.status_code == 200
    assert response.get_json() == {"models": []}


def test_v1_api_version_alias_supports_openwebui_ollama_probe():
    app = create_app(ollama_tags_client=type("TagsClient", (), {"tags": lambda self: {"models": []}})())
    client = app.test_client()

    response = client.get("/v1/api/version")

    assert response.status_code == 200
    assert response.get_json()["version"] == "mnemos-proxy"


def test_normalize_openwebui_model_id_strips_known_prefix():
    assert (
        normalize_openwebui_model_id(
            "mnemos.hf.co/cloudbjorn/Qwen3.6-35B-A3B_Opus-4.6-Reasoning-3300x-GGUF:Q4_K_M"
        )
        == "hf.co/cloudbjorn/Qwen3.6-35B-A3B_Opus-4.6-Reasoning-3300x-GGUF:Q4_K_M"
    )
    assert normalize_openwebui_model_id("mnemos/qwen3-coder-next:latest") == "qwen3-coder-next:latest"
    assert normalize_openwebui_model_id("qwen3-coder-next:latest") == "qwen3-coder-next:latest"


def test_append_evidence_footer_lists_citations_and_receipt_link():
    answer = append_evidence_footer(
        "Cycle convexity differs from manifold convexity. [1]",
        {
            "status": "ok",
            "citations": [
                {
                    "index": 1,
                    "source": "paper.pdf",
                    "score": 0.8123,
                    "engram_id": "engram-1",
                }
            ],
            "claim_boundary": "boundary",
        },
        receipt_url="http://127.0.0.1:8790/evidence/chatcmpl-1",
    )

    assert "MNEMOS Evidence Used" in answer
    assert "[1] source=paper.pdf" in answer
    assert "score=0.8123" in answer
    assert "engram_id=engram-1" in answer
    assert "MNEMOS Evidence Receipt:" in answer
    assert "http://127.0.0.1:8790/evidence/chatcmpl-1" in answer
    assert "boundary" in answer


def test_append_evidence_footer_handles_no_evidence():
    answer = append_evidence_footer(
        "",
        {
            "status": "no_evidence",
            "citations": [],
            "claim_boundary": "boundary",
            "warning": "MNEMOS returned no evidence; Ollama was not called.",
        },
        receipt_url=None,
    )

    assert "No MNEMOS evidence retrieved" in answer
    assert "Boundary:" in answer


def test_should_append_footer_suppresses_openwebui_task_prompts(monkeypatch):
    monkeypatch.delenv("MNEMOS_PROXY_FOOTER", raising=False)
    assert should_append_footer({"messages": [{"role": "user", "content": "### Task:\nGenerate a title"}]}) is False
    assert should_append_footer({"messages": [{"role": "user", "content": "What did I index?"}]}) is True


def test_v1_chat_completions_retrieves_mnemos_returns_footer_and_receipt(tmp_path):
    calls = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {
            "status": "ok",
            "answer": "Use the workflow checkpoints. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
            "evidence_block": "[1] source=workflow.pdf score=0.9000\nEvidence text",
        }

    app = create_app(query_runner=fake_runner, receipt_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "messages": [{"role": "user", "content": "How should I run the workflow?"}],
            "temperature": 0.1,
            "max_tokens": 400,
        },
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["model"] == "qwen3-coder-next:latest"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": body["choices"][0]["message"]["content"],
    }
    content = body["choices"][0]["message"]["content"]
    assert "Use the workflow checkpoints. [1]" in content
    assert "MNEMOS Evidence Used" in content
    assert "MNEMOS Evidence Receipt:" in content
    assert body["mnemos"]["citations"][0]["source"] == "workflow.pdf"
    receipt_files = list(tmp_path.glob("*.json"))
    assert len(receipt_files) == 1
    assert "Evidence text" in receipt_files[0].read_text(encoding="utf-8")
    assert calls["query"] == "How should I run the workflow?"
    assert calls["temperature"] == 0.1
    assert calls["num_predict"] == 400


def test_v1_chat_completions_passes_unprefixed_model_to_ollama_runner():
    calls = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {
            "status": "ok",
            "answer": "Prefix normalized. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mnemos.hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "How should I run the workflow?"}],
        },
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["model"] == "mnemos.hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M"
    assert calls["model"] == "hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M"


def test_v1_chat_completions_no_evidence_footer_and_receipt_page(tmp_path):
    def fake_runner(**kwargs):
        return {
            "status": "no_evidence",
            "answer": "",
            "query": kwargs["query"],
            "citations": [],
            "evidence_block": "",
            "warning": "MNEMOS returned no evidence; Ollama was not called.",
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner, receipt_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "messages": [{"role": "user", "content": "Unknown thing"}],
        },
    )

    body = response.get_json()
    content = body["choices"][0]["message"]["content"]
    assert response.status_code == 200
    assert "No MNEMOS evidence retrieved" in content
    receipt_id = next(tmp_path.glob("*.json")).stem
    receipt_page = client.get(f"/evidence/{receipt_id}")
    assert receipt_page.status_code == 200
    assert "MNEMOS Evidence Receipt" in receipt_page.get_data(as_text=True)


def test_v1_chat_completions_suppresses_footer_for_openwebui_task_prompts(tmp_path):
    def fake_runner(**kwargs):
        return {
            "status": "ok",
            "answer": "Short title",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "evidence_block": "[1] source=workflow.pdf score=0.9000\nEvidence text",
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner, receipt_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "messages": [{"role": "user", "content": "### Task:\nGenerate a chat title"}],
        },
    )

    content = response.get_json()["choices"][0]["message"]["content"]
    assert content == "Short title"
    assert list(tmp_path.glob("*.json")) == []


def test_v1_chat_completions_streams_openai_compatible_events():
    def fake_runner(**kwargs):
        return {
            "status": "ok",
            "answer": "Streamed MNEMOS answer. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert '"object": "chat.completion.chunk"' in text
    assert "Streamed MNEMOS answer. [1]" in text
    assert "MNEMOS Evidence Used" in text
    assert '"finish_reason": "stop"' in text
    assert "data: [DONE]" in text


def test_api_chat_accepts_ollama_shape_and_returns_ollama_shape():
    def fake_runner(**kwargs):
        return {
            "status": "ok",
            "answer": "MNEMOS found the workflow evidence. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner)
    client = app.test_client()

    response = client.post(
        "/api/chat",
        json={
            "model": "qwen3-coder-next:latest",
            "messages": [{"role": "user", "content": "Find workflow evidence"}],
            "options": {"temperature": 0.2, "num_predict": 256},
        },
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["model"] == "qwen3-coder-next:latest"
    assert body["message"]["role"] == "assistant"
    assert "MNEMOS found the workflow evidence. [1]" in body["message"]["content"]
    assert "MNEMOS Evidence Used" in body["message"]["content"]
    assert body["done"] is True
    assert body["mnemos"]["claim_boundary"] == "boundary"


def test_api_chat_streams_ollama_compatible_json_lines():
    def fake_runner(**kwargs):
        return {
            "status": "ok",
            "answer": "Ollama stream answer. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner)
    client = app.test_client()

    response = client.post(
        "/api/chat",
        json={
            "model": "qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "Find workflow evidence"}],
        },
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert response.mimetype == "application/x-ndjson"
    assert "Ollama stream answer. [1]" in text
    assert "MNEMOS Evidence Used" in text
    assert '"done": true' in text


def test_v1_api_chat_alias_accepts_openwebui_ollama_shape():
    calls = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {
            "status": "ok",
            "answer": "Nested Ollama route answer. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner)
    client = app.test_client()

    response = client.post(
        "/v1/api/chat",
        json={
            "model": "mnemos.qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "Find workflow evidence"}],
        },
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert response.mimetype == "application/x-ndjson"
    assert "Nested Ollama route answer. [1]" in text
    assert "MNEMOS Evidence Used" in text
    assert calls["model"] == "qwen3-coder-next:latest"
