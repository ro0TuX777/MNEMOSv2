"""Live smoke-test MNEMOS MCP tools against a running MNEMOS service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp_servers" / "mnemos" / "server.py"


def _parse_content(content: list[Any]) -> list[Any]:
    parsed = []
    for item in content:
        text = getattr(item, "text", None)
        if text is None:
            parsed.append(getattr(item, "model_dump", lambda: item)())
            continue
        try:
            parsed.append(json.loads(text))
        except json.JSONDecodeError:
            parsed.append(text)
    return parsed


async def live_smoke(base_url: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["MNEMOS_BASE_URL"] = base_url
    env.setdefault("MNEMOS_TIMEOUT_S", "90")
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            capabilities = await session.call_tool("get_capabilities", {})
            write = await session.call_tool(
                "write_observation",
                {
                    "content": (
                        "MNEMOS MCP live smoke memory: GateMem G4 is frozen for "
                        "regression testing only, and agent memory must preserve "
                        "primary-evidence grounding."
                    ),
                    "source": "tools/smoke_mnemos_mcp_live.py",
                    "confidence": 0.9,
                    "metadata_json": json.dumps(
                        {
                            "smoke_test_id": "mnemos_mcp_live",
                            "topic": "agent_memory_navigation",
                        }
                    ),
                },
            )
            search = await session.call_tool(
                "search_memory",
                {
                    "query": "GateMem G4 frozen regression testing agent memory primary evidence",
                    "top_k": 3,
                    "retrieval_mode": "semantic",
                    "explain": True,
                },
            )

    cap_payload = _parse_content(capabilities.content)
    write_payload = _parse_content(write.content)
    search_payload = _parse_content(search.content)
    return {
        "base_url": base_url,
        "capabilities_is_error": bool(getattr(capabilities, "isError", False)),
        "write_is_error": bool(getattr(write, "isError", False)),
        "search_is_error": bool(getattr(search, "isError", False)),
        "capabilities": cap_payload,
        "write": write_payload,
        "search": search_payload,
    }


def _first_dict(payload: list[Any]) -> dict[str, Any]:
    for item in payload:
        if isinstance(item, dict):
            return item
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("MNEMOS_BASE_URL", "http://localhost:8700"))
    args = parser.parse_args()
    result = asyncio.run(live_smoke(args.base_url))
    capabilities = _first_dict(result["capabilities"])
    write = _first_dict(result["write"])
    search = _first_dict(result["search"])
    search_results = search.get("results", [])

    print(f"Base URL: {result['base_url']}")
    print(f"Capabilities status: {capabilities.get('status')}")
    print(f"Capabilities feature: {capabilities.get('feature')}")
    print(f"Write status: {write.get('status')}")
    print(f"Search status: {search.get('status')}")
    print(f"Search result count: {len(search_results)}")
    if search_results:
        top = search_results[0]
        engram = top.get("engram", {})
        print(f"Top score: {top.get('score')}")
        print(f"Top content: {str(engram.get('content', ''))[:160]}")

    if capabilities.get("status") != "healthy":
        raise SystemExit("capabilities did not report healthy")
    if not search_results:
        raise SystemExit("search_memory returned no results")


if __name__ == "__main__":
    main()
