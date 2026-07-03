"""Open WebUI/Ollama-compatible MNEMOS evidence proxy.

This local-only bridge lets chat front ends point at a familiar Ollama or
OpenAI-compatible endpoint while keeping MNEMOS as the evidence source.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Protocol

import requests
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnemos_ollama_chat import (  # noqa: E402
    CLAIM_BOUNDARY,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    OllamaChatClient,
    normalize_base_url,
    run_query,
)


class OllamaTagsClient(Protocol):
    def tags(self) -> dict[str, Any]:
        ...


class RequestsOllamaTagsClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE_URL, *, timeout_s: float = 10.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self.timeout_s = timeout_s

    def tags(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=max(0.1, self.timeout_s))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Ollama tags response must be a JSON object")
        return data


QueryRunner = Callable[..., dict[str, Any]]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part).strip()
    return str(content or "").strip()


def extract_latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _content_to_text(message.get("content"))
            if text:
                return text
    return ""


def _model_ids_from_tags(tags: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for model in tags.get("models") or []:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
            if name:
                ids.append(str(name))
    return ids


def _openai_error(message: str, *, status: int = 400):
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            }
        ),
        status,
    )


def _run_query(
    runner: QueryRunner,
    *,
    model: str,
    query: str,
    temperature: float,
    num_predict: int,
    top_k: int,
    retrieval_mode: str,
    fusion_policy: str,
    max_chars_per_hit: int,
) -> dict[str, Any]:
    return runner(
        query=query,
        model=model,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        fusion_policy=fusion_policy,
        max_chars_per_hit=max_chars_per_hit,
        temperature=temperature,
        num_predict=num_predict,
    )


def _mnemos_meta(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "citations": result.get("citations") or [],
        "claim_boundary": result.get("claim_boundary") or CLAIM_BOUNDARY,
        "warning": result.get("warning"),
    }


def create_app(
    *,
    query_runner: QueryRunner = run_query,
    ollama_tags_client: OllamaTagsClient | None = None,
    ollama_base_url: str | None = None,
) -> Flask:
    app = Flask(__name__)
    resolved_ollama_url = normalize_base_url(
        ollama_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    )
    tags_client = ollama_tags_client or RequestsOllamaTagsClient(resolved_ollama_url)

    app.config["CLAIM_BOUNDARY"] = CLAIM_BOUNDARY
    app.config["OLLAMA_BASE_URL"] = resolved_ollama_url

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "service": "mnemos-ollama-openwebui-proxy",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    @app.get("/api/tags")
    def api_tags():
        return jsonify(tags_client.tags())

    @app.get("/v1/models")
    def v1_models():
        models = [
            {"id": model_id, "object": "model", "owned_by": "ollama"}
            for model_id in _model_ids_from_tags(tags_client.tags())
        ]
        return jsonify({"object": "list", "data": models})

    @app.post("/v1/chat/completions")
    def v1_chat_completions():
        payload = request.get_json(silent=True) or {}
        if payload.get("stream") is True:
            return _openai_error("streaming is not supported by the MNEMOS proxy yet")

        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return _openai_error("messages must be a list")

        query = extract_latest_user_text(messages)
        if not query:
            return _openai_error("at least one user message with text content is required")

        model = str(payload.get("model") or DEFAULT_MODEL)
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        num_predict = int(payload.get("max_tokens") or payload.get("num_predict") or 700)
        top_k = int(payload.get("mnemos_top_k") or os.getenv("MNEMOS_PROXY_TOP_K", "5"))
        retrieval_mode = str(payload.get("mnemos_retrieval_mode") or "semantic")
        fusion_policy = str(payload.get("mnemos_fusion_policy") or "balanced")
        max_chars = int(payload.get("mnemos_max_chars_per_hit") or 1200)

        result = _run_query(
            query_runner,
            model=model,
            query=query,
            temperature=temperature,
            num_predict=num_predict,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            fusion_policy=fusion_policy,
            max_chars_per_hit=max_chars,
        )
        answer = str(result.get("answer") or result.get("warning") or "")
        created = int(time.time())
        return jsonify(
            {
                "id": f"chatcmpl-mnemos-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "mnemos": _mnemos_meta(result),
            }
        )

    @app.post("/api/chat")
    def api_chat():
        payload = request.get_json(silent=True) or {}
        if payload.get("stream") is True:
            return jsonify({"error": "streaming is not supported by the MNEMOS proxy yet"}), 400

        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return jsonify({"error": "messages must be a list"}), 400

        query = extract_latest_user_text(messages)
        if not query:
            return jsonify({"error": "at least one user message with text content is required"}), 400

        options = payload.get("options") or {}
        if not isinstance(options, dict):
            options = {}

        model = str(payload.get("model") or DEFAULT_MODEL)
        result = _run_query(
            query_runner,
            model=model,
            query=query,
            temperature=float(options.get("temperature", 0.0) or 0.0),
            num_predict=int(options.get("num_predict") or 700),
            top_k=int(payload.get("mnemos_top_k") or os.getenv("MNEMOS_PROXY_TOP_K", "5")),
            retrieval_mode=str(payload.get("mnemos_retrieval_mode") or "semantic"),
            fusion_policy=str(payload.get("mnemos_fusion_policy") or "balanced"),
            max_chars_per_hit=int(payload.get("mnemos_max_chars_per_hit") or 1200),
        )
        answer = str(result.get("answer") or result.get("warning") or "")
        return jsonify(
            {
                "model": model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": answer},
                "done": True,
                "mnemos": _mnemos_meta(result),
            }
        )

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(
        ollama_base_url=args.ollama_base_url,
        query_runner=lambda **kwargs: run_query(
            **kwargs,
            ollama_client=OllamaChatClient(args.ollama_base_url),
        ),
    )
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
