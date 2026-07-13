"""Capture retrieval hygiene and reproducibility evidence for MNEMOS."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / "mcp_servers" / "mnemos" / "server.py"


def _load_mcp_module() -> Any:
    spec = importlib.util.spec_from_file_location("mnemos_mcp_server", MCP_SERVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MCP server module from {MCP_SERVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_json(url: str, timeout_s: float) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return data


def collect_service_snapshot(base_url: str, timeout_s: float) -> dict[str, Any]:
    health = _get_json(f"{base_url}/health", timeout_s)
    capabilities = _get_json(f"{base_url}/v1/mnemos/capabilities", timeout_s)
    stats = _get_json(f"{base_url}/v1/mnemos/stats", timeout_s)

    retrieval = (((stats.get("stats") or {}).get("retrieval")) or {})
    qdrant = ((retrieval.get("tiers") or {}).get("qdrant")) or {}
    collection = str(qdrant.get("collection") or "unknown")
    qdrant_url = str(qdrant.get("url") or "").rstrip("/")
    if "://qdrant:" in qdrant_url:
        qdrant_url = qdrant_url.replace("://qdrant:", "://localhost:")
    qdrant_collection = {}
    if qdrant_url and collection and collection != "unknown":
        try:
            qdrant_collection = _get_json(
                f"{qdrant_url}/collections/{collection}",
                timeout_s,
            )
        except Exception as exc:
            qdrant_collection = {"error": str(exc)}

    return {
        "base_url": base_url,
        "health": health,
        "capabilities": capabilities,
        "stats": stats,
        "collection": collection,
        "collection_snapshot": f"{collection}:{qdrant.get('document_count', 'unknown')}",
        "vector_config": ((((qdrant_collection.get("result") or {}).get("config") or {}).get("params") or {}).get("vectors")),
        "embedding_model": qdrant.get("embedding_model"),
        "embedding_dim": qdrant.get("embedding_dim"),
        "retrieval_mode_default": capabilities.get("retrieval_mode_default"),
        "fusion_policy_default": capabilities.get("fusion_policy_default"),
        "service_revision": {
            "source": stats.get("source"),
            "contract_version": stats.get("contract_version"),
            "generated_at": stats.get("generated_at"),
        },
        "cache": {
            "enabled": True,
            "cache_schema_version": "r0",
            "query_normalization_version": "v1",
            "ttl_seconds": ((((stats.get("stats") or {}).get("memory_over_maps") or {}).get("derived_view_cache") or {}).get("ttl_seconds")),
            "seed_snapshot": os.getenv("MNEMOS_SEED_SNAPSHOT", "unknown"),
        },
    }


def collect_mcp_snapshot(base_url: str) -> dict[str, Any]:
    module = _load_mcp_module()
    search_signature = module.search_memory.__defaults__ or ()
    return {
        "server_path": str(MCP_SERVER),
        "base_url": base_url,
        "timeout_s": float(os.getenv("MNEMOS_TIMEOUT_S", getattr(module, "DEFAULT_TIMEOUT_S", 10.0))),
        "tool_defaults": {
            "search_memory": {
                "top_k": search_signature[0] if len(search_signature) >= 1 else 5,
                "filters_json": search_signature[1] if len(search_signature) >= 2 else "{}",
                "retrieval_mode": search_signature[2] if len(search_signature) >= 3 else "hybrid",
                "explain": search_signature[3] if len(search_signature) >= 4 else True,
            }
        },
        "service_revision": {
            "module_default_base_url": getattr(module, "DEFAULT_BASE_URL", None),
            "module_default_timeout_s": getattr(module, "DEFAULT_TIMEOUT_S", None),
        },
    }


def compare_configuration(service: dict[str, Any], mcp: dict[str, Any]) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    intentional_differences: list[dict[str, Any]] = []

    if service.get("base_url") != mcp.get("base_url"):
        differences.append(
            {"field": "base_url", "service": service.get("base_url"), "mcp": mcp.get("base_url")}
        )

    service_mode = service.get("retrieval_mode_default")
    mcp_mode = (((mcp.get("tool_defaults") or {}).get("search_memory")) or {}).get("retrieval_mode")
    if service_mode != mcp_mode:
        intentional_differences.append(
            {
                "field": "retrieval_mode_default",
                "service": service_mode,
                "mcp": mcp_mode,
                "reason": "MCP bridge intentionally defaults search_memory to hybrid unless caller overrides it.",
            }
        )

    return {
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
        "intentional_differences": intentional_differences,
        "pass": not differences,
    }


def _simplify_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    simplified = []
    for row in payload.get("results", []):
        engram = row.get("engram", {})
        simplified.append(
            {
                "source": engram.get("source"),
                "source_uri": (engram.get("metadata") or {}).get("source_uri"),
                "score": row.get("score"),
                "rank": row.get("rank"),
                "duplicate_suppression": row.get("duplicate_suppression"),
                "retrieval_fingerprint": (payload.get("meta") or {}).get("retrieval_fingerprint"),
            }
        )
    return simplified


async def _mcp_search(base_url: str, query: str, top_k: int, retrieval_mode: str, explain: bool) -> dict[str, Any]:
    env = dict(os.environ)
    env["MNEMOS_BASE_URL"] = base_url
    env["MNEMOS_TIMEOUT_S"] = str(max(float(env.get("MNEMOS_TIMEOUT_S", "10")), 120.0))
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.call_tool(
                "search_memory",
                {
                    "query": query,
                    "top_k": top_k,
                    "retrieval_mode": retrieval_mode,
                    "explain": explain,
                },
            )
            text = getattr(response.content[0], "text", "{}")
            return json.loads(text)


def compare_probe_result(query: str, direct: dict[str, Any], mcp: dict[str, Any]) -> dict[str, Any]:
    direct_results = _simplify_results(direct)
    mcp_results = _simplify_results(mcp)
    return {
        "query": query,
        "direct_sources": [row.get("source_uri") or row.get("source") for row in direct_results[:3]],
        "mcp_sources": [row.get("source_uri") or row.get("source") for row in mcp_results[:3]],
        "direct_duplicate_suppression": (direct.get("meta") or {}).get("duplicate_suppression"),
        "mcp_duplicate_suppression": (mcp.get("meta") or {}).get("duplicate_suppression"),
        "direct_retrieval_fingerprint": (direct.get("meta") or {}).get("retrieval_fingerprint"),
        "mcp_retrieval_fingerprint": (mcp.get("meta") or {}).get("retrieval_fingerprint"),
        "agreement_top1": (
            bool(direct_results and mcp_results)
            and (direct_results[0].get("source_uri") or direct_results[0].get("source"))
            == (mcp_results[0].get("source_uri") or mcp_results[0].get("source"))
        ),
    }


async def collect_observed_retrieval_parity(
    base_url: str,
    *,
    queries: list[str],
    top_k: int,
    retrieval_mode: str,
    explain: bool,
    timeout_s: float,
) -> dict[str, Any]:
    if not queries:
        return {
            "status": "METHOD_DEFINED_NOT_RUN",
            "method": {
                "direct_request": {
                    "retrieval_mode": retrieval_mode,
                    "top_k": top_k,
                    "explain": explain,
                },
                "mcp_tool": "search_memory",
                "requirements": [
                    "same frozen query set",
                    "same seed snapshot",
                    "same collection snapshot",
                    "cold-cache and warm-cache runs recorded separately",
                ],
            },
        }

    probes = []
    for query in queries:
        direct = _post_json(
            f"{base_url}/v1/mnemos/search",
            {
                "query": query,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "explain": explain,
            },
            timeout_s,
        )
        mcp = await _mcp_search(base_url, query, top_k, retrieval_mode, explain)
        probes.append(compare_probe_result(query, direct, mcp))

    return {
        "status": "OBSERVED",
        "probe_count": len(probes),
        "probes": probes,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("MNEMOS_BASE_URL", "http://localhost:8700"))
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", default="hybrid")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    service_snapshot = collect_service_snapshot(base_url, args.timeout_s)
    mcp_snapshot = collect_mcp_snapshot(base_url)
    payload = {
        "CONFIGURATION_PARITY": compare_configuration(service_snapshot, mcp_snapshot),
        "OBSERVED_RETRIEVAL_PARITY": await collect_observed_retrieval_parity(
            base_url,
            queries=args.queries or [],
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            explain=True,
            timeout_s=args.timeout_s,
        ),
        "service_snapshot": service_snapshot,
        "mcp_snapshot": mcp_snapshot,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
