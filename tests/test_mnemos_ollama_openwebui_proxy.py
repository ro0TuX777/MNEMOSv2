from __future__ import annotations

import json

from tools.mnemos_ollama_openwebui_proxy import (
    append_evidence_footer,
    create_app,
    extract_latest_user_text,
    generation_info,
    normalize_openwebui_model_id,
    receipt_content_hash,
    render_evidence_receipt_html,
    should_append_footer,
    split_query_and_history,
    strip_evidence_footer,
    verify_citations,
    write_evidence_receipt,
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


def test_append_evidence_footer_keeps_markdown_structure():
    answer = append_evidence_footer(
        "Answer text. [1]",
        {
            "status": "ok",
            "citations": [
                {"index": 1, "source": "paper.pdf", "score": 0.5, "engram_id": "e-1"}
            ],
            "claim_boundary": "boundary",
        },
        receipt_url=None,
    )

    # Blank line before the rule so "---" cannot become a setext heading, and
    # citations inside a fenced block so their line breaks survive rendering.
    assert "Answer text. [1]\n\n---\n" in answer
    assert "```text\n[1] source=paper.pdf" in answer
    assert answer.count("```") == 2


def test_append_evidence_footer_distinguishes_mnemos_error():
    answer = append_evidence_footer(
        "",
        {
            "status": "mnemos_error",
            "citations": [],
            "claim_boundary": "boundary",
            "warning": "MNEMOS search failed (status=unavailable, error=connection refused); Ollama was not called.",
        },
        receipt_url=None,
    )

    assert "MNEMOS retrieval failed" in answer
    assert "No MNEMOS evidence retrieved" not in answer


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
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_v1_chat_completions_writes_receipt_even_when_footer_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOS_PROXY_FOOTER", "off")

    def fake_runner(**kwargs):
        return {
            "status": "ok",
            "answer": "Receipt should still be written.",
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
            "messages": [{"role": "user", "content": "Show me the evidence trail"}],
        },
    )

    assert response.status_code == 200
    receipt_files = list(tmp_path.glob("*.json"))
    assert len(receipt_files) == 1
    assert "Receipt should still be written." in receipt_files[0].read_text(encoding="utf-8")


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


def test_verify_citations_all_evidence_cited():
    check = verify_citations(
        "First point. [1] Second point. [2]",
        [{"index": 1, "source": "a.pdf"}, {"index": 2, "source": "b.pdf"}],
    )
    assert check["verdict"] == "all_evidence_cited"
    assert check["cited_indices"] == [1, 2]
    assert check["invalid_indices"] == []
    assert check["uncited_evidence_indices"] == []
    assert check["coverage"] == 1.0


def test_verify_citations_flags_missing_and_uncited_evidence():
    check = verify_citations(
        "Claim. [1] Bogus claim. [7]",
        [{"index": 1}, {"index": 2}, {"index": 3}],
    )
    assert check["verdict"] == "cites_missing_evidence"
    assert check["invalid_indices"] == [7]
    assert check["uncited_evidence_indices"] == [2, 3]
    assert check["coverage"] == round(1 / 3, 4)


def test_verify_citations_uncited_answer_and_no_evidence():
    uncited = verify_citations("No brackets here.", [{"index": 1}])
    assert uncited["verdict"] == "no_citations_in_answer"
    assert uncited["coverage"] == 0.0

    empty = verify_citations("Anything. [1]", [])
    assert empty["verdict"] == "no_evidence_available"
    assert empty["coverage"] is None


def test_generation_info_flags_length_truncation():
    info = generation_info(
        {"ollama_response": {"done_reason": "length", "eval_count": 700, "prompt_eval_count": 2100}}
    )
    assert info["truncated"] is True
    assert info["eval_count"] == 700

    assert generation_info({})["truncated"] is False
    assert generation_info({"ollama_response": {"done_reason": "stop"}})["truncated"] is False


def test_footer_warns_when_answer_truncated_at_token_limit():
    answer = append_evidence_footer(
        "Cut off mid-sentence [1",
        {
            "status": "ok",
            "citations": [{"index": 1, "source": "a.pdf", "score": 0.9, "engram_id": "e-1"}],
            "claim_boundary": "boundary",
            "ollama_response": {"done_reason": "length"},
        },
        receipt_url=None,
    )
    assert "done_reason=length" in answer
    assert "citations may be incomplete" in answer


def test_render_evidence_receipt_html_embeds_receipt_and_provenance_ui():
    receipt = {
        "receipt_id": "chatcmpl-mnemos-render1",
        "created": 123,
        "query": "What did they add?",
        "requested_model": "m",
        "actual_model": "m",
        "answer": "They added the Bill of Rights [1].",
        "status": "ok",
        "citations": [
            {"index": 1, "source": "a.pdf", "score": 0.9, "engram_id": "e-1"},
            {"index": 2, "source": "b.pdf", "score": 0.5, "engram_id": "e-2"},
        ],
        "evidence_block": "[1] source=a.pdf score=0.9\n\ntext one\n\n[2] source=b.pdf score=0.5\n\ntext two",
        "citation_check": verify_citations("They added the Bill of Rights [1].", []),
        "claim_boundary": "boundary text",
        "content_hash": "sha256:abc",
    }
    page = render_evidence_receipt_html(receipt)
    assert "MNEMOS Evidence Receipt" in page
    # The full receipt is embedded as JSON for the client-side provenance UI.
    assert '<script type="application/json" id="receipt-data">' in page
    embedded = page.split('id="receipt-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded) == receipt
    assert "provenance graph" in page


def test_render_evidence_receipt_html_escapes_script_close_tag():
    receipt = {"receipt_id": "r1", "answer": "bad </script><script>alert(1)</script>"}
    page = render_evidence_receipt_html(receipt)
    embedded = page.split('id="receipt-data">', 1)[1].split("</script>", 1)[0]
    assert "</script>" not in embedded
    assert json.loads(embedded)["answer"] == "bad </script><script>alert(1)</script>"


def test_write_evidence_receipt_records_verification_and_hash(tmp_path):
    result = {
        "status": "ok",
        "citations": [
            {"index": 1, "source": "a.pdf", "score": 0.9},
            {"index": 2, "source": "b.pdf", "score": 0.5},
        ],
        "evidence_block": "[1] a\n\n[2] b",
        "ollama_response": {"done_reason": "stop", "prompt_eval_count": 100, "eval_count": 50},
    }
    path = write_evidence_receipt(
        tmp_path,
        receipt_id="chatcmpl-mnemos-test1",
        created=123,
        query="q",
        requested_model="m",
        actual_model="m",
        answer="Uses evidence. [1]",
        result=result,
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))

    assert receipt["citation_check"]["verdict"] == "partial_evidence_cited"
    assert receipt["citation_check"]["uncited_evidence_indices"] == [2]
    assert receipt["generation"] == {
        "done_reason": "stop",
        "truncated": False,
        "prompt_eval_count": 100,
        "eval_count": 50,
    }
    assert receipt["score_stats"] == {"count": 2, "max": 0.9, "min": 0.5, "mean": 0.7}
    # The stored hash must be recomputable from the receipt's factual core.
    assert receipt["content_hash"].startswith("sha256:")
    assert receipt_content_hash(receipt) == receipt["content_hash"]


def test_write_evidence_receipt_archives_overflow_instead_of_deleting(tmp_path, monkeypatch):
    import os
    import time as time_module

    monkeypatch.setenv("MNEMOS_EVIDENCE_RECEIPT_MAX_FILES", "2")
    result = {"status": "ok", "citations": [], "evidence_block": ""}
    now = time_module.time()
    for offset, receipt_id in enumerate(["chatcmpl-old", "chatcmpl-mid", "chatcmpl-new"]):
        path = write_evidence_receipt(
            tmp_path,
            receipt_id=receipt_id,
            created=123 + offset,
            query="q",
            requested_model="m",
            actual_model="m",
            answer="",
            result=result,
        )
        os.utime(path, (now + offset, now + offset))

    live = sorted(item.name for item in tmp_path.glob("*.json"))
    archived = sorted(item.name for item in (tmp_path / "archive").glob("*.json"))
    assert live == ["chatcmpl-mid.json", "chatcmpl-new.json"]
    assert archived == ["chatcmpl-old.json"]


def test_v1_chat_completions_reports_real_token_usage():
    def fake_runner(**kwargs):
        return {
            "status": "ok",
            "answer": "Counted. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "a.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
            "ollama_response": {"done_reason": "stop", "prompt_eval_count": 321, "eval_count": 45},
        }

    app = create_app(query_runner=fake_runner)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "messages": [{"role": "user", "content": "count my tokens"}],
        },
    )

    body = response.get_json()
    assert body["usage"] == {"prompt_tokens": 321, "completion_tokens": 45, "total_tokens": 366}


def test_strip_evidence_footer_removes_footer_and_keeps_answer():
    footered = append_evidence_footer(
        "R1 was closed. [1]",
        {
            "status": "ok",
            "citations": [{"index": 1, "source": "paper.pdf", "score": 0.9, "engram_id": "e-1"}],
            "claim_boundary": "boundary",
        },
        receipt_url="http://127.0.0.1:8790/evidence/chatcmpl-1",
    )

    assert strip_evidence_footer(footered) == "R1 was closed. [1]"


def test_strip_evidence_footer_handles_footer_only_and_plain_text():
    footer_only = append_evidence_footer(
        "",
        {"status": "no_evidence", "citations": [], "claim_boundary": "boundary"},
        receipt_url=None,
    )
    assert strip_evidence_footer(footer_only) == ""
    # A withheld answer whose body is the warning keeps that body.
    withheld = append_evidence_footer(
        "",
        {"status": "no_evidence", "citations": [], "claim_boundary": "boundary", "warning": "w"},
        receipt_url=None,
    )
    assert strip_evidence_footer(withheld) == "w"
    assert strip_evidence_footer("plain answer") == "plain answer"


def test_split_query_and_history_strips_footers_and_returns_prior_turns(monkeypatch):
    monkeypatch.delenv("MNEMOS_PROXY_HISTORY", raising=False)
    monkeypatch.delenv("MNEMOS_PROXY_HISTORY_MAX_TURNS", raising=False)
    assistant_turn = append_evidence_footer(
        "R1 was closed. [1]",
        {"status": "ok", "citations": [], "claim_boundary": "boundary"},
        receipt_url=None,
    )

    query, history = split_query_and_history(
        [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "What is R1?"},
            {"role": "assistant", "content": assistant_turn},
            {"role": "user", "content": "What about its abstention gap?"},
        ]
    )

    assert query == "What about its abstention gap?"
    assert history == [
        {"role": "user", "content": "What is R1?"},
        {"role": "assistant", "content": "R1 was closed. [1]"},
    ]


def test_split_query_and_history_respects_env_gates(monkeypatch):
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "latest"},
    ]

    monkeypatch.setenv("MNEMOS_PROXY_HISTORY_MAX_TURNS", "2")
    query, history = split_query_and_history(messages)
    assert query == "latest"
    assert [m["content"] for m in history] == ["three", "four"]

    monkeypatch.setenv("MNEMOS_PROXY_HISTORY", "off")
    query, history = split_query_and_history(messages)
    assert query == "latest"
    assert history == []


def test_v1_chat_completions_forwards_history_and_condense_flag(monkeypatch):
    monkeypatch.delenv("MNEMOS_PROXY_HISTORY", raising=False)
    monkeypatch.delenv("MNEMOS_PROXY_QUERY_CONDENSE", raising=False)
    calls = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {
            "status": "ok",
            "answer": "Follow-up answered. [1]",
            "model": kwargs["model"],
            "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
            "claim_boundary": "boundary",
        }

    app = create_app(query_runner=fake_runner)
    client = app.test_client()

    assistant_turn = append_evidence_footer(
        "R1 was closed. [1]",
        {"status": "ok", "citations": [], "claim_boundary": "boundary"},
        receipt_url=None,
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "messages": [
                {"role": "user", "content": "What is R1?"},
                {"role": "assistant", "content": assistant_turn},
                {"role": "user", "content": "What about its abstention gap?"},
            ],
        },
    )

    assert response.status_code == 200
    assert calls["query"] == "What about its abstention gap?"
    assert calls["history"] == [
        {"role": "user", "content": "What is R1?"},
        {"role": "assistant", "content": "R1 was closed. [1]"},
    ]
    assert calls["condense_queries"] is True


def _stream_events(retrieval_result, deltas, done_result):
    def stream_runner(**kwargs):
        yield {"event": "retrieval", "result": retrieval_result}
        for delta in deltas:
            yield {"event": "delta", "content": delta}
        yield {"event": "done", "result": done_result}

    return stream_runner


def test_v1_chat_completions_live_streams_deltas_then_footer_and_receipt(tmp_path):
    retrieval = {
        "status": "ok",
        "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
        "evidence_block": "[1] source=workflow.pdf score=0.9000\nEvidence text",
        "claim_boundary": "boundary",
    }
    done = dict(retrieval)
    done["answer"] = "First second. [1]"
    runner = _stream_events(retrieval, ["First ", "second. [1]"], done)

    app = create_app(stream_query_runner=runner, receipt_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "Find workflow evidence"}],
        },
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    # Each delta arrives as its own SSE chunk, in order, before the footer.
    assert text.index('"content": "First "') < text.index('"content": "second. [1]"')
    assert text.index('"content": "second. [1]"') < text.index("MNEMOS Evidence Used")
    assert "MNEMOS Evidence Receipt:" in text
    assert '"finish_reason": "stop"' in text
    assert text.rstrip().endswith("data: [DONE]")

    receipt_files = list(tmp_path.glob("*.json"))
    assert len(receipt_files) == 1
    receipt = json.loads(receipt_files[0].read_text(encoding="utf-8"))
    assert receipt["answer"] == "First second. [1]"
    assert receipt["status"] == "ok"


def test_v1_chat_completions_streams_terminal_no_evidence_fallback(tmp_path):
    def runner(**kwargs):
        yield {
            "event": "done",
            "result": {
                "status": "no_evidence",
                "answer": "",
                "citations": [],
                "evidence_block": "",
                "warning": "MNEMOS returned no evidence; Ollama was not called.",
                "claim_boundary": "boundary",
            },
        }

    app = create_app(stream_query_runner=runner, receipt_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "Unknown thing"}],
        },
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "No MNEMOS evidence retrieved" in text
    assert "data: [DONE]" in text
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_v1_chat_completions_stream_reports_mid_generation_failure(tmp_path):
    retrieval = {
        "status": "ok",
        "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
        "evidence_block": "[1] source=workflow.pdf score=0.9000\nEvidence text",
        "claim_boundary": "boundary",
    }
    done = dict(retrieval)
    done.update(
        {
            "status": "ollama_error",
            "answer": "partial",
            "warning": "Ollama streaming failed after retrieval: boom",
        }
    )
    runner = _stream_events(retrieval, ["partial"], done)

    app = create_app(stream_query_runner=runner, receipt_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "Find workflow evidence"}],
        },
    )

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "[MNEMOS proxy] Ollama streaming failed after retrieval: boom" in text
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert receipt["status"] == "ollama_error"
    assert receipt["answer"] == "partial"


def test_api_chat_live_streams_ndjson_deltas_then_footer(tmp_path):
    retrieval = {
        "status": "ok",
        "citations": [{"index": 1, "source": "workflow.pdf", "score": 0.9}],
        "evidence_block": "[1] source=workflow.pdf score=0.9000\nEvidence text",
        "claim_boundary": "boundary",
    }
    done = dict(retrieval)
    done["answer"] = "Alpha beta. [1]"
    runner = _stream_events(retrieval, ["Alpha ", "beta. [1]"], done)

    app = create_app(stream_query_runner=runner, receipt_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/chat",
        json={
            "model": "qwen3-coder-next:latest",
            "stream": True,
            "messages": [{"role": "user", "content": "Find workflow evidence"}],
        },
    )

    assert response.status_code == 200
    assert response.mimetype == "application/x-ndjson"
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line]
    contents = [line["message"]["content"] for line in lines]
    assert contents[0] == "Alpha "
    assert contents[1] == "beta. [1]"
    assert "MNEMOS Evidence Used" in contents[2]
    assert lines[-1]["done"] is True
    assert all(line["done"] is False for line in lines[:-1])
    assert len(list(tmp_path.glob("*.json"))) == 1


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
