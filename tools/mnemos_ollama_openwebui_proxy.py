"""Open WebUI/Ollama-compatible MNEMOS evidence proxy.

This local-only bridge lets chat front ends point at a familiar Ollama or
OpenAI-compatible endpoint while keeping MNEMOS as the evidence source.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Protocol

import requests
from flask import Flask, Response, jsonify, request

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
DEFAULT_RECEIPT_DIR = ROOT / "logs" / "evidence_receipts"


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


def normalize_openwebui_model_id(model_id: str) -> str:
    text = str(model_id or "").strip()
    prefixes = [
        item.strip()
        for item in os.getenv("MNEMOS_PROXY_MODEL_PREFIXES", "mnemos").split(",")
        if item.strip()
    ]
    for prefix in prefixes:
        for separator in (".", "/"):
            marker = f"{prefix}{separator}"
            if text.startswith(marker):
                return text[len(marker) :]
    return text


def should_append_footer(payload: dict[str, Any]) -> bool:
    if os.getenv("MNEMOS_PROXY_FOOTER", "on").strip().lower() in {"0", "false", "no", "off"}:
        return False
    query = extract_latest_user_text(payload.get("messages") or [])
    # Open WebUI sends hidden title/tag/task prompts through the same provider.
    # Keep those responses clean. If multi-turn forwarding is added later,
    # strip this deterministic footer from prior assistant messages first.
    if query.lstrip().startswith("### Task:"):
        return False
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("task"):
        return False
    return True


def append_evidence_footer(answer: str, result: dict[str, Any], *, receipt_url: str | None) -> str:
    base = str(answer or result.get("warning") or "").strip()
    citations = result.get("citations") or []
    lines = ["", "---", "MNEMOS Evidence Used"]
    if citations:
        for citation in citations:
            index = citation.get("index", "?")
            source = citation.get("source", "unknown")
            score = citation.get("score", 0.0)
            engram_id = citation.get("engram_id") or "unknown"
            try:
                score_text = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_text = str(score)
            lines.append(f"[{index}] source={source}")
            lines.append(f"    score={score_text}")
            lines.append(f"    engram_id={engram_id}")
    else:
        lines.append("No MNEMOS evidence retrieved - answer withheld by the MNEMOS proxy.")
        warning = result.get("warning")
        if warning:
            lines.append(f"Warning: {warning}")

    if receipt_url:
        lines.extend(["", f"MNEMOS Evidence Receipt: {receipt_url}"])
    lines.extend(["", f"Boundary: {result.get('claim_boundary') or CLAIM_BOUNDARY}"])
    footer = "\n".join(lines)
    return f"{base}{footer}" if base else footer.lstrip()


def _safe_receipt_id(receipt_id: str) -> str:
    return "".join(ch for ch in receipt_id if ch.isalnum() or ch in {"-", "_"})


def _receipt_path(receipt_dir: Path, receipt_id: str) -> Path:
    safe_id = _safe_receipt_id(receipt_id)
    if not safe_id:
        raise ValueError("empty receipt id")
    return receipt_dir / f"{safe_id}.json"


def write_evidence_receipt(
    receipt_dir: Path,
    *,
    receipt_id: str,
    created: int,
    query: str,
    requested_model: str,
    actual_model: str,
    answer: str,
    result: dict[str, Any],
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "receipt_id": receipt_id,
        "created": created,
        "query": query,
        "requested_model": requested_model,
        "actual_model": actual_model,
        "answer": answer,
        "status": result.get("status"),
        "citations": result.get("citations") or [],
        "evidence_block": result.get("evidence_block") or "",
        "retrieval_metadata": result.get("retrieval_metadata") or {},
        "claim_boundary": result.get("claim_boundary") or CLAIM_BOUNDARY,
        "warning": result.get("warning"),
    }
    path = _receipt_path(receipt_dir, receipt_id)
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")

    max_receipts = int(os.getenv("MNEMOS_EVIDENCE_RECEIPT_MAX_FILES", "500"))
    receipt_files = sorted(receipt_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
    for stale in receipt_files[: max(0, len(receipt_files) - max_receipts)]:
        stale.unlink(missing_ok=True)
    return path


def render_evidence_receipt_html(receipt: dict[str, Any]) -> str:
    citations = receipt.get("citations") or []
    citation_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('index', '')))}</td>"
        f"<td>{html.escape(str(item.get('source', 'unknown')))}</td>"
        f"<td>{html.escape(str(item.get('score', '')))}</td>"
        f"<td>{html.escape(str(item.get('engram_id', '')))}</td>"
        "</tr>"
        for item in citations
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MNEMOS Evidence Receipt</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; max-width: 1120px; }}
    pre {{ white-space: pre-wrap; background: #f5f5f5; padding: 12px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>MNEMOS Evidence Receipt</h1>
  <p><strong>Receipt:</strong> {html.escape(str(receipt.get("receipt_id", "")))}</p>
  <p><strong>Status:</strong> {html.escape(str(receipt.get("status", "")))}</p>
  <p><strong>Model:</strong> {html.escape(str(receipt.get("requested_model", "")))}</p>
  <h2>Query</h2>
  <pre>{html.escape(str(receipt.get("query", "")))}</pre>
  <h2>Citations</h2>
  <table>
    <thead><tr><th>#</th><th>Source</th><th>Score</th><th>Engram ID</th></tr></thead>
    <tbody>{citation_rows}</tbody>
  </table>
  <h2>Evidence Block Sent To Ollama</h2>
  <pre>{html.escape(str(receipt.get("evidence_block", "")))}</pre>
  <h2>Boundary</h2>
  <pre>{html.escape(str(receipt.get("claim_boundary", "")))}</pre>
</body>
</html>"""


def _public_receipt_base_url() -> str:
    configured = os.getenv("MNEMOS_EVIDENCE_PUBLIC_BASE_URL")
    if configured:
        return configured.rstrip("/")
    host = request.host.split(":", 1)
    port = f":{host[1]}" if len(host) == 2 else ""
    return f"http://127.0.0.1{port}"


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


def _openai_chat_completion_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": completion_id,
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


def _openai_stream_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> Response:
    metadata = _mnemos_meta(result)

    def events():
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            "mnemos": metadata,
        }
        yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"
        if answer:
            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
                "mnemos": metadata,
            }
            yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
        stop_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "mnemos": metadata,
        }
        yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(events(), mimetype="text/event-stream")


def _ollama_chat_response(
    *,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": answer},
        "done": True,
        "mnemos": _mnemos_meta(result),
    }


def _ollama_stream_response(
    *,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> Response:
    metadata = _mnemos_meta(result)

    def lines():
        if answer:
            yield json.dumps(
                {
                    "model": model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": answer},
                    "done": False,
                    "mnemos": metadata,
                },
                ensure_ascii=False,
            ) + "\n"
        yield json.dumps(
            {
                "model": model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "mnemos": metadata,
            },
            ensure_ascii=False,
        ) + "\n"

    return Response(lines(), mimetype="application/x-ndjson")


def create_app(
    *,
    query_runner: QueryRunner = run_query,
    ollama_tags_client: OllamaTagsClient | None = None,
    ollama_base_url: str | None = None,
    receipt_dir: str | Path | None = None,
) -> Flask:
    app = Flask(__name__)
    resolved_ollama_url = normalize_base_url(
        ollama_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    )
    tags_client = ollama_tags_client or RequestsOllamaTagsClient(resolved_ollama_url)

    app.config["CLAIM_BOUNDARY"] = CLAIM_BOUNDARY
    app.config["OLLAMA_BASE_URL"] = resolved_ollama_url
    resolved_receipt_dir = Path(
        receipt_dir or os.getenv("MNEMOS_EVIDENCE_RECEIPT_DIR", str(DEFAULT_RECEIPT_DIR))
    )
    app.config["MNEMOS_EVIDENCE_RECEIPT_DIR"] = str(resolved_receipt_dir)

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

    @app.get("/v1/api/tags")
    def v1_api_tags():
        return jsonify(tags_client.tags())

    @app.get("/api/ps")
    @app.get("/v1/api/ps")
    def api_ps():
        return jsonify({"models": []})

    @app.get("/api/version")
    @app.get("/v1/api/version")
    def api_version():
        return jsonify({"version": "mnemos-proxy"})

    @app.get("/evidence/<receipt_id>.json")
    def evidence_receipt_json(receipt_id: str):
        path = _receipt_path(resolved_receipt_dir, receipt_id)
        if not path.exists():
            return jsonify({"error": "receipt_not_found"}), 404
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/evidence/<receipt_id>")
    def evidence_receipt(receipt_id: str):
        path = _receipt_path(resolved_receipt_dir, receipt_id)
        if not path.exists():
            return "MNEMOS evidence receipt not found.", 404
        receipt = json.loads(path.read_text(encoding="utf-8"))
        return Response(render_evidence_receipt_html(receipt), mimetype="text/html")

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
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return _openai_error("messages must be a list")

        query = extract_latest_user_text(messages)
        if not query:
            return _openai_error("at least one user message with text content is required")

        requested_model = str(payload.get("model") or DEFAULT_MODEL)
        model = normalize_openwebui_model_id(requested_model)
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
        created = int(time.time())
        completion_id = f"chatcmpl-mnemos-{uuid.uuid4().hex}"
        raw_answer = str(result.get("answer") or result.get("warning") or "")
        footer_enabled = should_append_footer(payload)
        receipt_url = None
        if footer_enabled:
            write_evidence_receipt(
                resolved_receipt_dir,
                receipt_id=completion_id,
                created=created,
                query=query,
                requested_model=requested_model,
                actual_model=model,
                answer=raw_answer,
                result=result,
            )
            receipt_url = f"{_public_receipt_base_url()}/evidence/{completion_id}"
        answer = (
            append_evidence_footer(raw_answer, result, receipt_url=receipt_url)
            if footer_enabled
            else raw_answer
        )
        if payload.get("stream") is True:
            return _openai_stream_response(
                completion_id=completion_id,
                created=created,
                model=requested_model,
                answer=answer,
                result=result,
            )
        return jsonify(
            _openai_chat_completion_response(
                completion_id=completion_id,
                created=created,
                model=requested_model,
                answer=answer,
                result=result,
            )
        )

    def _handle_ollama_chat():
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return jsonify({"error": "messages must be a list"}), 400

        query = extract_latest_user_text(messages)
        if not query:
            return jsonify({"error": "at least one user message with text content is required"}), 400

        options = payload.get("options") or {}
        if not isinstance(options, dict):
            options = {}

        requested_model = str(payload.get("model") or DEFAULT_MODEL)
        model = normalize_openwebui_model_id(requested_model)
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
        created = int(time.time())
        receipt_id = f"chatcmpl-mnemos-{uuid.uuid4().hex}"
        raw_answer = str(result.get("answer") or result.get("warning") or "")
        footer_enabled = should_append_footer(payload)
        receipt_url = None
        if footer_enabled:
            write_evidence_receipt(
                resolved_receipt_dir,
                receipt_id=receipt_id,
                created=created,
                query=query,
                requested_model=requested_model,
                actual_model=model,
                answer=raw_answer,
                result=result,
            )
            receipt_url = f"{_public_receipt_base_url()}/evidence/{receipt_id}"
        answer = (
            append_evidence_footer(raw_answer, result, receipt_url=receipt_url)
            if footer_enabled
            else raw_answer
        )
        if payload.get("stream") is True:
            return _ollama_stream_response(model=requested_model, answer=answer, result=result)
        return jsonify(_ollama_chat_response(model=requested_model, answer=answer, result=result))

    @app.post("/api/chat")
    def api_chat():
        return _handle_ollama_chat()

    @app.post("/v1/api/chat")
    def v1_api_chat():
        return _handle_ollama_chat()

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
