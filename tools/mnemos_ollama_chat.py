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
import sys
from pathlib import Path
from typing import Any, Protocol

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnemos_sdk import MnemosClient, MnemosConfig  # noqa: E402
from mnemos_sdk.client import MnemosResponse, SearchHit  # noqa: E402

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
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
    ) -> list[SearchHit]:
        ...


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
) -> dict[str, Any]:
    system = (
        "You are answering through an MFS-governed MNEMOS memory adapter. "
        "MNEMOS is the evidence source. Ollama is only the local model runtime. "
        "Use only the supplied MNEMOS_EVIDENCE when making factual claims. "
        "Cite sources with bracket numbers like [1]. "
        "If the evidence is insufficient, say what is missing. "
        "Do not claim unsupported facts."
    )
    user = (
        "MNEMOS_EVIDENCE:\n"
        f"{evidence_block}\n\n"
        "USER_QUERY:\n"
        f"{query}"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }


def _answer_from_ollama_response(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(response.get("response"), str):
        return response["response"].strip()
    return json.dumps(response, ensure_ascii=False, sort_keys=True)


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
    mnemos_client: MnemosSearchClient | None = None,
    ollama_client: OllamaChatClient | None = None,
) -> dict[str, Any]:
    mnemos = mnemos_client or MnemosClient(MnemosConfig.from_env())
    ollama = ollama_client or OllamaChatClient(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))

    retrieval_metadata: dict[str, Any] = {
        "requested_retrieval_mode": retrieval_mode,
        "requested_fusion_policy": fusion_policy,
    }
    if hasattr(mnemos, "search_raw"):
        raw_response = mnemos.search_raw(
            query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            fusion_policy=fusion_policy,
        )
        retrieval_metadata.update(_retrieval_metadata_from_raw_response(raw_response))
        if not raw_response.ok:
            return {
                "status": "mnemos_error",
                "answer": "",
                "query": query,
                "citations": [],
                "evidence_block": "",
                "retrieval_metadata": retrieval_metadata,
                "claim_boundary": CLAIM_BOUNDARY,
                "warning": (
                    f"MNEMOS search failed (status={raw_response.status}, "
                    f"error={raw_response.error}); Ollama was not called."
                ),
            }
        hits = _hits_from_raw_response(raw_response)
    else:
        hits = mnemos.search(
            query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            fusion_policy=fusion_policy,
        )
    evidence_block, citations = format_evidence_block(
        hits,
        max_chars_per_hit=max_chars_per_hit,
    )
    if not hits:
        return {
            "status": "no_evidence",
            "answer": "",
            "query": query,
            "citations": [],
            "evidence_block": "",
            "retrieval_metadata": retrieval_metadata,
            "claim_boundary": CLAIM_BOUNDARY,
            "warning": "MNEMOS returned no evidence; Ollama was not called.",
        }

    payload = build_ollama_chat_payload(
        model=model,
        query=query,
        evidence_block=evidence_block,
        temperature=temperature,
        num_predict=num_predict,
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
