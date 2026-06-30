"""Verify the MNEMOS MSF/MCP agent-memory bridge scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registry" / "services.json"
CONTRACT = ROOT / "service" / "contract.json"
MCP_DIR = ROOT / "mcp_servers" / "mnemos"
SERVER = MCP_DIR / "server.py"
OPENAPI = MCP_DIR / "openapi.json"
README = MCP_DIR / "README.md"
REQUIREMENTS = MCP_DIR / "requirements.txt"
LOCKFILE = MCP_DIR / "requirements.lock.txt"
DOCKERFILE = MCP_DIR / "Dockerfile"

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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_mnemos_msf_mcp() -> dict[str, Any]:
    registry = _load_json(REGISTRY)
    contract = _load_json(CONTRACT)
    openapi = _load_json(OPENAPI)
    server_text = SERVER.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    requirements_text = REQUIREMENTS.read_text(encoding="utf-8")
    lockfile_text = LOCKFILE.read_text(encoding="utf-8")
    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")

    services = {service["name"]: service for service in registry.get("services", [])}
    mnemos = services.get("mnemos-service", {})
    contract_tools = {tool["name"] for tool in contract.get("agent_memory_tools", [])}
    operation_ids = {
        operation.get("operationId")
        for path_item in openapi.get("paths", {}).values()
        for operation in path_item.values()
        if isinstance(operation, dict)
    }
    server_tools = {
        name
        for name in EXPECTED_TOOLS
        if f"def {name}" in server_text and "@mcp.tool()" in server_text
    }

    checks = {
        "registry_points_to_contract": mnemos.get("contract_file") == "service/contract.json",
        "registry_points_to_mcp_bridge": mnemos.get("mcp_bridge", {}).get("server") == "mcp_servers/mnemos/server.py",
        "contract_declares_expected_tools": EXPECTED_TOOLS.issubset(contract_tools),
        "contract_declares_mcp_bridge": contract.get("mcp_bridge", {}).get("package") == "mcp_servers/mnemos",
        "server_file_present": SERVER.is_file(),
        "server_declares_expected_tools": EXPECTED_TOOLS.issubset(server_tools),
        "openapi_file_present": OPENAPI.is_file(),
        "openapi_declares_core_operations": {
            "health_check",
            "get_capabilities",
            "search_memory",
            "write_observation",
            "explain_memory_provenance",
        }.issubset(operation_ids),
        "readme_mentions_claude_config": "claude_desktop_config" in readme_text,
        "requirements_include_mcp": "mcp" in requirements_text,
        "lockfile_pins_mcp": "mcp==" in lockfile_text,
        "lockfile_pins_requests": "requests==" in lockfile_text,
        "dockerfile_runs_server": "server.py" in dockerfile_text,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"MNEMOS MSF/MCP bridge verification failed: {failed}")

    return {
        "status": "MNEMOS_MSF_MCP_AGENT_MEMORY_ALPHA",
        "service": "mnemos-service",
        "tool_count": len(EXPECTED_TOOLS),
        "checks": checks,
        "all_checks_passed": True,
    }


def main() -> None:
    result = verify_mnemos_msf_mcp()
    print(result["status"])
    print(f"Service: {result['service']}")
    print(f"Tools: {result['tool_count']}")
    print(f"Checks passed: {len(result['checks'])}/{len(result['checks'])}")


if __name__ == "__main__":
    main()
