import json
import subprocess
import sys
from pathlib import Path

from tools.verify_mnemos_msf_mcp import EXPECTED_TOOLS, OPENAPI, SERVER, verify_mnemos_msf_mcp


ROOT = Path(__file__).resolve().parents[1]


def test_mnemos_msf_mcp_bridge_verifies():
    result = verify_mnemos_msf_mcp()
    assert result["all_checks_passed"] is True
    assert result["status"] == "MNEMOS_MSF_MCP_AGENT_MEMORY_ALPHA"
    assert result["tool_count"] == len(EXPECTED_TOOLS)


def test_mcp_server_exposes_expected_agent_memory_tools():
    text = SERVER.read_text(encoding="utf-8")
    for tool in EXPECTED_TOOLS:
        assert f"def {tool}" in text
    assert "MNEMOS_BASE_URL" in text
    assert "MNEMOS_TOKEN" in text
    assert '"status": "unavailable"' in text


def test_openapi_companion_spec_contains_core_operations():
    spec = json.loads(OPENAPI.read_text(encoding="utf-8"))
    operation_ids = {
        operation.get("operationId")
        for path_item in spec["paths"].values()
        for operation in path_item.values()
    }
    assert {
        "health_check",
        "get_capabilities",
        "search_memory",
        "write_observation",
        "explain_memory_provenance",
    }.issubset(operation_ids)


def test_msf_mcp_verifier_command_runs_from_repository_root():
    completed = subprocess.run(
        [sys.executable, "tools/verify_mnemos_msf_mcp.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "MNEMOS_MSF_MCP_AGENT_MEMORY_ALPHA" in completed.stdout
    assert "Checks passed" in completed.stdout


def test_mcp_stdio_smoke_lists_tools_from_repository_root():
    completed = subprocess.run(
        [sys.executable, "tools/smoke_mnemos_mcp_stdio.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Tools listed: 9" in completed.stdout
    assert "Expected tools present: True" in completed.stdout
    assert "Health tool MCP error: False" in completed.stdout
