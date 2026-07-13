from __future__ import annotations

from tools import start_mnemos_openwebui_stack as stack


def test_normalize_base_url_adds_scheme_and_rewrites_wildcard_host():
    assert stack.normalize_base_url("0.0.0.0:7777") == "http://127.0.0.1:7777"
    assert stack.normalize_base_url("http://127.0.0.1:8790/") == "http://127.0.0.1:8790"


def test_default_ollama_base_url_prefers_explicit_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:7777")
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")

    assert stack.default_ollama_base_url() == "http://127.0.0.1:7777"


def test_default_ollama_base_url_uses_ollama_host(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:7777")

    assert stack.default_ollama_base_url() == "http://127.0.0.1:7777"
