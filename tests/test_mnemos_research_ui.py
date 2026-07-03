from __future__ import annotations

import io
from pathlib import Path

from tools.mnemos_research_ui import create_app, default_ollama_base_url


def test_default_ollama_base_url_prefers_ollama_base_url_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:7777")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9999")

    assert default_ollama_base_url() == "http://127.0.0.1:9999"


def test_default_ollama_base_url_uses_ollama_host_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:7777")

    assert default_ollama_base_url() == "http://127.0.0.1:7777"


def test_index_page_contains_upload_fields_and_controls(tmp_path):
    app = create_app(upload_dir=tmp_path)
    client = app.test_client()

    response = client.get("/")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "MNEMOS Research Intake" in text
    assert 'type="file"' in text
    assert "Test Connection" in text
    assert "Run Intake" in text
    assert "ollamaModel" in text


def test_index_page_uses_detected_ollama_default(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:7777")
    app = create_app(upload_dir=tmp_path)
    client = app.test_client()

    text = client.get("/").get_data(as_text=True)

    assert 'id="ollamaBaseUrl" value="http://127.0.0.1:7777"' in text


def test_models_endpoint_returns_available_ollama_models(tmp_path):
    app = create_app(
        upload_dir=tmp_path,
        ollama_models_fn=lambda base_url: [
            {"name": "llama3.1", "size": 123},
            {"name": "qwen3-coder", "size": 456},
        ],
    )
    client = app.test_client()

    response = client.get("/api/ollama-models?ollama_base_url=http://127.0.0.1:7777")

    assert response.status_code == 200
    assert response.get_json() == {
        "ok": True,
        "models": [
            {"name": "llama3.1", "size": 123},
            {"name": "qwen3-coder", "size": 456},
        ],
    }


def test_connection_endpoint_reports_mnemos_and_ollama_status(tmp_path):
    app = create_app(
        upload_dir=tmp_path,
        mnemos_health_fn=lambda base_url: {"ok": True, "status": "ok"},
        ollama_models_fn=lambda base_url: [{"name": "llama3.1"}],
    )
    client = app.test_client()

    response = client.post(
        "/api/test-connection",
        json={
            "mnemos_base_url": "http://127.0.0.1:8700",
            "ollama_base_url": "http://127.0.0.1:7777",
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "ok": True,
        "mnemos": {"ok": True, "status": "ok"},
        "ollama": {"ok": True, "model_count": 1},
    }


def test_intake_endpoint_saves_uploaded_files_and_calls_runner(tmp_path):
    calls = {}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return {"status": "ok", "indexed": 2, "claim_boundary": "boundary"}

    app = create_app(upload_dir=tmp_path, intake_runner=fake_runner)
    client = app.test_client()

    response = client.post(
        "/api/intake",
        data={
            "files": (io.BytesIO(b"# Research\nUseful idea"), "notes.md"),
            "mnemos_base_url": "http://127.0.0.1:8700",
            "ollama_base_url": "http://127.0.0.1:7777",
            "project": "MNEMOS",
            "capability": "local research memory",
            "status": "reviewed",
            "tags": "workflow, pdf",
            "summarize": "true",
            "ollama_model": "llama3.1",
            "output": "docs/research/packet.md",
        },
        content_type="multipart/form-data",
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["ok"] is True
    assert body["result"]["indexed"] == 2
    assert len(calls["files"]) == 1
    assert Path(calls["files"][0]).name == "notes.md"
    assert Path(calls["files"][0]).read_text(encoding="utf-8") == "# Research\nUseful idea"
    assert calls["project"] == "MNEMOS"
    assert calls["capability"] == "local research memory"
    assert calls["status"] == "reviewed"
    assert calls["tags"] == ["workflow", "pdf"]
    assert calls["summarize"] is True
    assert calls["ollama_model"] == "llama3.1"
    assert calls["output_path"] == Path("docs/research/packet.md")
