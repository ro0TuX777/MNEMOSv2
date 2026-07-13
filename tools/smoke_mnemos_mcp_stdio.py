"""Smoke-test the MNEMOS MCP bridge over stdio.

This test exercises the same basic path a Claude Desktop/Codex-style MCP client
uses: start the server as a subprocess, initialize an MCP session, list tools,
and call health_check. A live MNEMOS REST service is optional; when it is down,
health_check should return a structured unavailable response.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp_servers" / "mnemos" / "server.py"
EXPECTED_TOOLS = {
    "health_check",
    "get_capabilities",
    "search_memory",
    "write_observation",
    "record_decision",
    "find_related_context",
    "detect_contradictions",
    "summarize_session_handoff",
    "explain_memory_provenance",
}


def _content_to_value(content: list[Any]) -> Any:
    values = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            values.append(text)
            continue
        values.append(getattr(item, "model_dump", lambda: item)())
    return values


async def smoke(base_url: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["MNEMOS_BASE_URL"] = base_url
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tool_names = {tool.name for tool in listed.tools}
            health = await session.call_tool("health_check", {})
            missing = sorted(EXPECTED_TOOLS - tool_names)
            return {
                "server": str(SERVER),
                "base_url": base_url,
                "tool_count": len(tool_names),
                "missing_tools": missing,
                "health_is_error": bool(getattr(health, "isError", False)),
                "health_content": _content_to_value(health.content),
                "all_expected_tools_present": not missing,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("MNEMOS_BASE_URL", "http://localhost:8700"))
    args = parser.parse_args()
    result = asyncio.run(smoke(args.base_url))
    print(f"Server: {result['server']}")
    print(f"Base URL: {result['base_url']}")
    print(f"Tools listed: {result['tool_count']}")
    print(f"Expected tools present: {result['all_expected_tools_present']}")
    print(f"Health tool MCP error: {result['health_is_error']}")
    if result["missing_tools"]:
        print(f"Missing tools: {', '.join(result['missing_tools'])}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
