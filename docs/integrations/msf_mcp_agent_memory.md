# MNEMOS MSF MCP Agent-Memory Integration

Status: `MNEMOS_MSF_MCP_AGENT_MEMORY_ALPHA`

```text
LOCAL_MFS_COMPATIBLE_SERVICE_BRIDGE
AGENT_MEMORY_TOOL_SURFACE
NO_RUNTIME_BEHAVIOR_CHANGE
NO_CONNECTOR_DIRECTORY_CLAIM
```

MNEMOS already follows the core MFS service pattern:

- `/health`
- `/v1/mnemos/capabilities`
- `service/contract.json`
- `registry/services.json`
- a typed Python boundary SDK in `mnemos_sdk/`

The MCP integration adds a local bridge package at:

```text
mcp_servers/mnemos
```

This package exposes MNEMOS as agent-callable tools for Claude Desktop,
Codex-style clients, VSCode/Antigravity MCP clients, and custom orchestrators.

## Agent Tools

The initial tool surface is deliberately small:

- `health_check`
- `get_capabilities`
- `search_memory`
- `write_observation`
- `record_decision`
- `find_related_context`
- `detect_contradictions`
- `summarize_session_handoff`
- `explain_memory_provenance`

These tools map to existing MNEMOS REST routes. They do not add a new memory
policy, delete behavior, hosted service, or production connector claim.

## Local Run

```bash
pip install -r mcp_servers/mnemos/requirements.txt
python mcp_servers/mnemos/server.py
```

Claude Desktop-style config:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "python",
      "args": ["mcp_servers/mnemos/server.py"],
      "env": {
        "MNEMOS_BASE_URL": "http://localhost:8700"
      }
    }
  }
}
```

For a complete Claude Desktop setup guide, see:

```text
docs/integrations/claude_desktop_mnemos_mcp.md
```

## Verification

```bash
python tools/verify_mnemos_msf_mcp.py
python tools/smoke_mnemos_mcp_stdio.py
python tools/smoke_mnemos_mcp_live.py
```

The verifier checks the registry, contract metadata, MCP server file, OpenAPI
operation IDs, README, requirements, and Dockerfile without requiring a live
MNEMOS service or MCP SDK installation.

The stdio smoke test starts the MCP server as a subprocess, initializes an MCP
client session, lists the available tools, and calls `health_check`. A live
MNEMOS REST service is optional for this transport smoke test; when MNEMOS is
not running, `health_check` returns a structured `unavailable` result rather
than failing the MCP session.

The live smoke test expects MNEMOS to be running at `MNEMOS_BASE_URL`, calls
`get_capabilities`, writes a small smoke memory through `write_observation`,
and retrieves it through `search_memory`. It defaults `MNEMOS_TIMEOUT_S` to 90
seconds because the first local embedding/model load can be slower than a normal
warm request.
