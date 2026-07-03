from __future__ import annotations

from tools.mnemos_ollama_openwebui_proxy import create_app, extract_latest_user_text


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


def test_v1_chat_completions_retrieves_mnemos_and_returns_openai_shape():
    calls = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {
            "status": "ok",
            "answer": "Use the workflow checkpoints. [1]",
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
        "content": "Use the workflow checkpoints. [1]",
    }
    assert body["mnemos"]["citations"][0]["source"] == "workflow.pdf"
    assert calls["query"] == "How should I run the workflow?"
    assert calls["temperature"] == 0.1
    assert calls["num_predict"] == 400


def test_v1_chat_completions_rejects_streaming_until_supported():
    app = create_app(query_runner=lambda **kwargs: {})
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert "streaming is not supported" in response.get_json()["error"]["message"]


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
    assert body["message"] == {
        "role": "assistant",
        "content": "MNEMOS found the workflow evidence. [1]",
    }
    assert body["done"] is True
    assert body["mnemos"]["claim_boundary"] == "boundary"
