"""MNEMOS-backed Ollama chat adapter.

This is an MFS-style boundary adapter for hosts that use Ollama as the local
model runtime but do not natively mount MCP tools. It retrieves bounded evidence
through the MNEMOS SDK, calls Ollama's local chat API, and returns the answer
with explicit MNEMOS citations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Protocol

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnemos_sdk import MnemosClient, MnemosConfig  # noqa: E402
from mnemos_sdk.client import MnemosResponse, SearchHit  # noqa: E402

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
RESEARCH_UPLOAD_DIR = Path(os.getenv("MNEMOS_RESEARCH_UPLOAD_DIR", str(ROOT / "data" / "research_uploads")))
CLAIM_BOUNDARY = (
    "MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY: retrieves MNEMOS evidence and asks "
    "Ollama to answer from that evidence; it does not alter MNEMOS retrieval, "
    "write memory, or enforce R1/R2 admission policy."
)


class MnemosSearchClient(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_mode: str,
        fusion_policy: str,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        ...

    def search_raw(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_mode: str,
        fusion_policy: str,
        filters: dict[str, Any] | None = None,
    ) -> MnemosResponse:
        ...


FILENAME_PATTERN = re.compile(
    r"(?i)\b([^\s\\/:\*\?\"<>\|]+\.(?:pdf|docx|md|markdown|txt|rst|py|js|ts|tsx|jsx|json|ya?ml|toml|csv))\b"
)


def filename_filter_from_query(query: str) -> dict[str, str] | None:
    match = FILENAME_PATTERN.search(str(query or ""))
    if not match:
        return None
    filename = Path(match.group(1).strip()).name
    filename = resolve_research_upload_filename(filename)
    return {"metadata.filename": filename} if filename else None


def resolve_research_upload_filename(filename: str) -> str:
    """Map a user-visible upload filename to the stored filename MNEMOS indexed."""
    requested = Path(str(filename or "").strip()).name
    if not requested:
        return requested
    manifest_path = RESEARCH_UPLOAD_DIR / ".manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return requested
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        return requested
    identity = f"mnemos::{requested}".lower()
    for record in records:
        if not isinstance(record, dict):
            continue
        stored_path = Path(str(record.get("stored_path") or "")).name
        if not stored_path:
            continue
        if str(record.get("identity_key") or "").lower() == identity:
            return stored_path
    return requested


def _hits_from_raw_response(response: MnemosResponse) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for row in response.data.get("results", []):
        hits.append(
            SearchHit(
                engram=row.get("engram", {}),
                score=row.get("score", 0.0),
                tier=row.get("tier", ""),
                tiers=row.get("tiers", []),
                rank=row.get("rank"),
                evidence=row.get("evidence"),
                component_scores=row.get("component_scores"),
                retrieval_sources=row.get("retrieval_sources"),
                fusion_policy=row.get("fusion_policy"),
            )
        )
    return hits


def _retrieval_metadata_from_raw_response(response: MnemosResponse) -> dict[str, Any]:
    # The service reports mode/fingerprint under data["meta"], not top-level data.
    meta = (response.data or {}).get("meta") or {}
    metadata: dict[str, Any] = {
        "response_status": response.status,
        "response_error": response.error,
    }
    for key in (
        "retrieval_mode",
        "fusion_policy",
        "retrieval_fingerprint",
        "lexical_lane_available",
        "result_count",
        "latency_s",
        "low_relevance_abstention",
    ):
        if key in meta:
            metadata[key] = meta[key]
    return metadata


class OllamaChatClient:
    """Tiny client for Ollama's native ``/api/chat`` endpoint."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE_URL, *, timeout_s: float = 180.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self.timeout_s = timeout_s

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=max(0.1, self.timeout_s),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Ollama response must be a JSON object")
        return data

    def chat_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield Ollama's NDJSON chat chunks as they arrive."""
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=max(0.1, self.timeout_s),
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                if isinstance(chunk, dict):
                    yield chunk


def normalize_base_url(value: str) -> str:
    text = (value or "").strip() or DEFAULT_OLLAMA_BASE_URL
    if "://" not in text:
        text = "http://" + text
    return text.rstrip("/").replace("http://0.0.0.0", "http://127.0.0.1", 1)


def _hit_source(hit: SearchHit) -> str:
    engram = hit.engram or {}
    metadata = engram.get("metadata") or {}
    return str(
        metadata.get("source_path")
        or metadata.get("source_uri")
        or engram.get("source")
        or "unknown"
    )


def _truncate(text: str, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max(0, max_chars)].rstrip() + "..."


def format_evidence_block(
    hits: list[SearchHit],
    *,
    max_chars_per_hit: int = 1200,
) -> tuple[str, list[dict[str, Any]]]:
    """Return a numbered evidence block and structured citation metadata."""
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        engram = hit.engram or {}
        source = _hit_source(hit)
        score = float(hit.score or 0.0)
        content = _truncate(str(engram.get("content") or ""), max_chars_per_hit)
        lines.append(f"[{idx}] source={source} score={score:.4f}")
        lines.append(content)
        citations.append(
            {
                "index": idx,
                "engram_id": engram.get("id"),
                "source": source,
                "score": score,
            }
        )
    return "\n\n".join(lines), citations


def build_ollama_chat_payload(
    *,
    model: str,
    query: str,
    evidence_block: str,
    temperature: float = 0.0,
    num_predict: int = 700,
    history: list[dict[str, Any]] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    system = (
        "You are answering through an MFS-governed MNEMOS memory adapter. "
        "MNEMOS is the evidence source. Ollama is only the local model runtime. "
        "Use only the supplied MNEMOS_EVIDENCE when making factual claims. "
        "Cite sources with bracket numbers like [1]. "
        "If the evidence is insufficient, say what is missing. "
        "Do not claim unsupported facts."
    )
    if history:
        system += (
            " Earlier conversation turns are provided only to resolve references; "
            "factual claims must still come from the supplied MNEMOS_EVIDENCE."
        )
    user = (
        "MNEMOS_EVIDENCE:\n"
        f"{evidence_block}\n\n"
        "USER_QUERY:\n"
        f"{query}"
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in history or []:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }


CONDENSE_MAX_HISTORY_CHARS = 700
CONDENSE_MAX_QUERY_CHARS = 400


def build_condense_payload(
    *,
    model: str,
    query: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    transcript_lines: list[str] = []
    for message in history:
        role = str(message.get("role") or "user").upper()
        content = _truncate(str(message.get("content") or ""), CONDENSE_MAX_HISTORY_CHARS)
        if content:
            transcript_lines.append(f"{role}: {content}")
    system = (
        "You rewrite the latest user message as one self-contained search query. "
        "Use the conversation only to resolve pronouns and references. "
        "Output only the rewritten query with no commentary."
    )
    transcript = "\n".join(transcript_lines)
    user = (
        "CONVERSATION:\n"
        f"{transcript}\n\n"
        "LATEST_USER_MESSAGE:\n"
        f"{query}\n\n"
        "Standalone rewrite:"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 160},
    }


def condense_query_with_history(
    *,
    query: str,
    history: list[dict[str, Any]],
    model: str,
    ollama_client: OllamaChatClient,
) -> tuple[str, dict[str, Any]]:
    """Rewrite a follow-up question into a standalone retrieval query.

    Any failure falls back to the original query so retrieval never blocks on
    the condensation call; the metadata records what actually happened.
    """
    metadata: dict[str, Any] = {"query_condensed": False, "retrieval_query": query}
    if not history:
        return query, metadata
    try:
        response = ollama_client.chat(
            build_condense_payload(model=model, query=query, history=history)
        )
    except Exception as exc:  # noqa: BLE001 - condensation must never block retrieval
        metadata["condense_error"] = str(exc)
        return query, metadata
    message = response.get("message") if isinstance(response, dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        metadata["condense_error"] = "condense response had no message content"
        return query, metadata
    # Some local templates leak reasoning tags into content; keep the tail only.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    clean = " ".join(text.split()).strip().strip('"').strip()
    if not clean or len(clean) > CONDENSE_MAX_QUERY_CHARS:
        metadata["condense_error"] = "condense output empty or too long"
        return query, metadata
    if clean != query:
        metadata.update(
            {
                "query_condensed": True,
                "original_query": query,
                "retrieval_query": clean,
            }
        )
    return clean, metadata


def _answer_from_ollama_response(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(response.get("response"), str):
        return response["response"].strip()
    return json.dumps(response, ensure_ascii=False, sort_keys=True)


def _terminal_result(
    status: str,
    *,
    query: str,
    retrieval_metadata: dict[str, Any],
    warning: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "answer": "",
        "query": query,
        "citations": [],
        "evidence_block": "",
        "retrieval_metadata": retrieval_metadata,
        "claim_boundary": CLAIM_BOUNDARY,
        "warning": warning,
    }


def _prepare_evidence(
    *,
    query: str,
    model: str,
    top_k: int,
    retrieval_mode: str,
    fusion_policy: str,
    max_chars_per_hit: int,
    history: list[dict[str, Any]] | None,
    condense_queries: bool,
    mnemos: MnemosSearchClient,
    ollama: OllamaChatClient,
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]], dict[str, Any]]:
    """Condense the query if needed, retrieve MNEMOS evidence, format it.

    Returns ``(terminal_result, evidence_block, citations, retrieval_metadata)``
    where ``terminal_result`` is non-None when Ollama must not be called.
    """
    retrieval_metadata: dict[str, Any] = {
        "requested_retrieval_mode": retrieval_mode,
        "requested_fusion_policy": fusion_policy,
    }
    retrieval_query = query
    artifact_filter = filename_filter_from_query(query)
    if artifact_filter:
        retrieval_metadata["artifact_filter"] = artifact_filter
    if history:
        retrieval_metadata["history_turns"] = len(history)
        if condense_queries:
            retrieval_query, condense_metadata = condense_query_with_history(
                query=query,
                history=history,
                model=model,
                ollama_client=ollama,
            )
            retrieval_metadata.update(condense_metadata)
    if hasattr(mnemos, "search_raw"):
        raw_kwargs: dict[str, Any] = {
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "fusion_policy": fusion_policy,
        }
        if artifact_filter:
            raw_kwargs["filters"] = artifact_filter
        raw_response = mnemos.search_raw(retrieval_query, **raw_kwargs)
        retrieval_metadata.update(_retrieval_metadata_from_raw_response(raw_response))
        if not raw_response.ok:
            terminal = _terminal_result(
                "mnemos_error",
                query=query,
                retrieval_metadata=retrieval_metadata,
                warning=(
                    f"MNEMOS search failed (status={raw_response.status}, "
                    f"error={raw_response.error}); Ollama was not called."
                ),
            )
            return terminal, "", [], retrieval_metadata
        hits = _hits_from_raw_response(raw_response)
    else:
        search_kwargs: dict[str, Any] = {
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "fusion_policy": fusion_policy,
        }
        if artifact_filter:
            search_kwargs["filters"] = artifact_filter
        hits = mnemos.search(retrieval_query, **search_kwargs)
    evidence_block, citations = format_evidence_block(
        hits,
        max_chars_per_hit=max_chars_per_hit,
    )
    if not hits:
        terminal = _terminal_result(
            "no_evidence",
            query=query,
            retrieval_metadata=retrieval_metadata,
            warning="MNEMOS returned no evidence; Ollama was not called.",
        )
        return terminal, "", [], retrieval_metadata
    return None, evidence_block, citations, retrieval_metadata


def run_query(
    *,
    query: str,
    model: str,
    top_k: int = 5,
    retrieval_mode: str = "semantic",
    fusion_policy: str = "balanced",
    max_chars_per_hit: int = 1200,
    temperature: float = 0.0,
    num_predict: int = 700,
    history: list[dict[str, Any]] | None = None,
    condense_queries: bool = True,
    mnemos_client: MnemosSearchClient | None = None,
    ollama_client: OllamaChatClient | None = None,
) -> dict[str, Any]:
    mnemos = mnemos_client or MnemosClient(MnemosConfig.from_env())
    ollama = ollama_client or OllamaChatClient(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))

    terminal, evidence_block, citations, retrieval_metadata = _prepare_evidence(
        query=query,
        model=model,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        fusion_policy=fusion_policy,
        max_chars_per_hit=max_chars_per_hit,
        history=history,
        condense_queries=condense_queries,
        mnemos=mnemos,
        ollama=ollama,
    )
    if terminal is not None:
        return terminal

    payload = build_ollama_chat_payload(
        model=model,
        query=query,
        evidence_block=evidence_block,
        temperature=temperature,
        num_predict=num_predict,
        history=history,
    )
    response = ollama.chat(payload)
    return {
        "status": "ok",
        "answer": _answer_from_ollama_response(response),
        "query": query,
        "model": model,
        "citations": citations,
        "evidence_block": evidence_block,
        "retrieval_metadata": retrieval_metadata,
        "claim_boundary": CLAIM_BOUNDARY,
        "ollama_response": response,
    }


def run_query_stream(
    *,
    query: str,
    model: str,
    top_k: int = 5,
    retrieval_mode: str = "semantic",
    fusion_policy: str = "balanced",
    max_chars_per_hit: int = 1200,
    temperature: float = 0.0,
    num_predict: int = 700,
    history: list[dict[str, Any]] | None = None,
    condense_queries: bool = True,
    mnemos_client: MnemosSearchClient | None = None,
    ollama_client: OllamaChatClient | None = None,
) -> Iterator[dict[str, Any]]:
    """Streaming variant of :func:`run_query`.

    Yields ``{"event": "retrieval", "result": ...}`` once evidence is admitted,
    ``{"event": "delta", "content": ...}`` per generated chunk, and finally
    ``{"event": "done", "result": ...}`` with the complete result. Terminal
    retrieval outcomes (no evidence, MNEMOS error) yield only a done event.
    """
    mnemos = mnemos_client or MnemosClient(MnemosConfig.from_env())
    ollama = ollama_client or OllamaChatClient(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))

    terminal, evidence_block, citations, retrieval_metadata = _prepare_evidence(
        query=query,
        model=model,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        fusion_policy=fusion_policy,
        max_chars_per_hit=max_chars_per_hit,
        history=history,
        condense_queries=condense_queries,
        mnemos=mnemos,
        ollama=ollama,
    )
    if terminal is not None:
        yield {"event": "done", "result": terminal}
        return

    partial: dict[str, Any] = {
        "status": "ok",
        "query": query,
        "model": model,
        "citations": citations,
        "evidence_block": evidence_block,
        "retrieval_metadata": retrieval_metadata,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    yield {"event": "retrieval", "result": dict(partial)}

    payload = build_ollama_chat_payload(
        model=model,
        query=query,
        evidence_block=evidence_block,
        temperature=temperature,
        num_predict=num_predict,
        history=history,
        stream=True,
    )
    parts: list[str] = []
    final_chunk: dict[str, Any] | None = None
    try:
        for chunk in ollama.chat_stream(payload):
            message = chunk.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content:
                parts.append(content)
                yield {"event": "delta", "content": content}
            if chunk.get("done"):
                final_chunk = chunk
    except Exception as exc:  # noqa: BLE001 - surface the failure inside the stream
        result = dict(partial)
        result.update(
            {
                "status": "ollama_error",
                "answer": "".join(parts).strip(),
                "warning": f"Ollama streaming failed after retrieval: {exc}",
            }
        )
        yield {"event": "done", "result": result}
        return

    result = dict(partial)
    result.update(
        {
            "answer": "".join(parts).strip(),
            "ollama_response": final_chunk or {},
        }
    )
    yield {"event": "done", "result": result}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question to answer from MNEMOS evidence.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", default="semantic")
    parser.add_argument("--fusion-policy", default="balanced")
    parser.add_argument("--max-chars-per-hit", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=700)
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of a readable answer.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_query(
        query=args.query,
        model=args.model,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        fusion_policy=args.fusion_policy,
        max_chars_per_hit=args.max_chars_per_hit,
        temperature=args.temperature,
        num_predict=args.num_predict,
        ollama_client=OllamaChatClient(args.ollama_base_url),
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ok" else 2

    if result["status"] != "ok":
        print(result.get("warning") or result["status"])
        print(result["claim_boundary"])
        return 2

    print(result["answer"])
    print("\nSources:")
    for citation in result["citations"]:
        print(f"[{citation['index']}] {citation['source']} score={citation['score']:.4f}")
    print(f"\n{result['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
