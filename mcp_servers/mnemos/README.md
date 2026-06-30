# MNEMOS MCP Bridge

This package exposes MNEMOS as an agent-callable MCP server using the MFS
service contract in `service/contract.json` and registry entry in
`registry/services.json`.

It is a local agent-memory bridge for Claude Desktop, Codex-style clients,
VSCode/Antigravity MCP clients, and custom orchestrators.

## Tools

- `health_check`
- `get_capabilities`
- `search_memory`
- `write_observation`
- `record_decision`
- `find_related_context`
- `detect_contradictions`
- `summarize_session_handoff`
- `explain_memory_provenance`

## Run

Install optional MCP bridge dependencies:

```bash
python tools/setup_mnemos_mcp_env.py
```

Start MNEMOS, then run the MCP server:

```bash
mcp_servers/mnemos/.venv/Scripts/python.exe mcp_servers/mnemos/server.py
```

For Claude Desktop-style stdio config in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "G:\\MNEMOS\\mcp_servers\\mnemos\\.venv\\Scripts\\python.exe",
      "args": ["G:\\MNEMOS\\mcp_servers\\mnemos\\server.py"],
      "env": {
        "MNEMOS_BASE_URL": "http://localhost:8700"
      }
    }
  }
}
```

The Windows-ready example lives at:

```text
mcp_servers/mnemos/claude_desktop_config.example.json
```

Full setup guide:

```text
docs/integrations/claude_desktop_mnemos_mcp.md
```

For SSE or streamable HTTP:

```bash
mcp_servers/mnemos/.venv/Scripts/python.exe mcp_servers/mnemos/server.py --transport sse --port 9700
mcp_servers/mnemos/.venv/Scripts/python.exe mcp_servers/mnemos/server.py --transport streamable-http --port 9700
```

## Boundary

This bridge uses its own virtual environment and pinned dependency lockfile in
`mcp_servers/mnemos`. It does not change MNEMOS runtime behavior. It wraps
existing REST routes as MCP tools and keeps provenance-bearing memory operations
explicit.
