from __future__ import annotations

import io
import json
import os
from pathlib import Path

from tools.mnemos_research_ui import (
    create_app,
    default_ollama_base_url,
    load_evidence_receipt,
    list_evidence_receipts,
    safe_receipt_id,
)


def _write_receipt(path: Path, *, receipt_id: str = "chatcmpl-mnemos-1") -> Path:
    data = {
        "receipt_id": receipt_id,
        "created": 1783149000,
        "query": "How does cycle convexity differ?",
        "requested_model": "mnemos.qwen3-coder-next:latest",
        "actual_model": "qwen3-coder-next:latest",
        "answer": "Cycle convexity answer. [1]",
        "status": "ok",
        "warning": None,
        "citations": [
            {
                "index": 1,
                "source": "C:\\research\\cycle-convexity.pdf",
                "score": 0.8123,
                "engram_id": "research::abc123",
            }
        ],
        "evidence_block": "[1] source=C:\\research\\cycle-convexity.pdf score=0.8123\nEvidence text",
        "retrieval_metadata": {
            "requested_retrieval_mode": "semantic",
            "retrieval_mode": "semantic",
            "retrieval_fingerprint": {
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "lexical_lane_available": False,
            },
        },
        "claim_boundary": "MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY",
    }
    target = path / f"{receipt_id}.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


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
    assert "MNEMOS timeout seconds" in text
    assert "Index batch size" in text
    assert "Evidence Receipts" in text


def test_safe_receipt_id_allows_expected_ids_and_rejects_empty():
    assert safe_receipt_id("chatcmpl-mnemos-abc_123") == "chatcmpl-mnemos-abc_123"
    assert safe_receipt_id("../../etc/passwd") == "etcpasswd"
    assert safe_receipt_id("!!!") is None


def test_list_evidence_receipts_returns_recent_summary(tmp_path):
    _write_receipt(tmp_path, receipt_id="chatcmpl-mnemos-1")

    receipts = list_evidence_receipts(tmp_path)

    assert len(receipts) == 1
    assert receipts[0]["receipt_id"] == "chatcmpl-mnemos-1"
    assert receipts[0]["query_preview"] == "How does cycle convexity differ?"
    assert receipts[0]["source_count"] == 1
    assert receipts[0]["retrieval_mode"] == "semantic"


def test_load_evidence_receipt_rejects_missing_and_bad_ids(tmp_path):
    _write_receipt(tmp_path, receipt_id="chatcmpl-mnemos-1")

    assert load_evidence_receipt(tmp_path, "chatcmpl-mnemos-1")["receipt_id"] == "chatcmpl-mnemos-1"
    assert load_evidence_receipt(tmp_path, "!!!") is None
    assert load_evidence_receipt(tmp_path, "missing") is None


def test_evidence_list_page_renders_recent_receipts(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    _write_receipt(receipt_dir, receipt_id="chatcmpl-mnemos-1")
    app = create_app(upload_dir=tmp_path / "uploads", receipt_dir=receipt_dir)
    client = app.test_client()

    response = client.get("/evidence")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "MNEMOS Evidence Receipts" in text
    assert "How does cycle convexity differ?" in text
    assert "chatcmpl-mnemos-1" in text
    assert "semantic" in text


def test_evidence_detail_page_renders_graph_and_receipt(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    _write_receipt(receipt_dir, receipt_id="chatcmpl-mnemos-1")
    app = create_app(upload_dir=tmp_path / "uploads", receipt_dir=receipt_dir)
    client = app.test_client()

    response = client.get("/evidence/chatcmpl-mnemos-1")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Evidence Graph" in text
    assert "User Query" in text
    assert "Retrieved Evidence Chunks" in text
    assert "Source Files / Metadata" in text
    assert "Model Answer With Citations" in text
    assert "cycle-convexity.pdf" in text
    assert "BAAI/bge-base-en-v1.5" in text


def test_evidence_detail_missing_receipt_returns_404(tmp_path):
    app = create_app(upload_dir=tmp_path / "uploads", receipt_dir=tmp_path / "receipts")
    client = app.test_client()

    response = client.get("/evidence/missing")

    assert response.status_code == 404


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
            "mnemos_timeout_s": "180",
            "batch_size": "12",
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
    assert calls["batch_size"] == 12
    assert calls["output_path"] == Path("docs/research/packet.md")
    assert os.environ["MNEMOS_TIMEOUT_S"] == "180"
