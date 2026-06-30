# Claude Desktop: MNEMOS MCP Setup

Status: `MNEMOS_CLAUDE_DESKTOP_MCP_LOCAL_SETUP`

This guide makes MNEMOS available to Claude Desktop as a local MCP tool server.

## 1. Start MNEMOS

From the MNEMOS repo:

```powershell
cd G:\MNEMOS
python tools/setup_mnemos_mcp_env.py
python tools/run_mnemos_mcp_isolation_check.py
```

For a quick bridge-only smoke:

```powershell
python tools/verify_mnemos_msf_mcp.py
.\mcp_servers\mnemos\.venv\Scripts\python.exe tools/smoke_mnemos_mcp_stdio.py
.\mcp_servers\mnemos\.venv\Scripts\python.exe tools/smoke_mnemos_mcp_live.py
```

Expected signals:

```text
Checks passed: 11/11
Tools listed: 9
Capabilities status: healthy
Search status: healthy
```

If the live smoke fails, start the MNEMOS REST service and backing store first,
then rerun the smoke tests.

## 2. Find Python

```powershell
.\mcp_servers\mnemos\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

Use the isolated MCP venv Python path as the `command` value in Claude Desktop
config. Do not point Claude Desktop at MNEMOS's normal runtime or Anaconda
environment for this bridge.

## 3. Configure Claude Desktop

Open:

```text
%APPDATA%\Claude\claude_desktop_config.json
```

Add or merge:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "G:\\MNEMOS\\mcp_servers\\mnemos\\.venv\\Scripts\\python.exe",
      "args": [
        "G:\\MNEMOS\\mcp_servers\\mnemos\\server.py"
      ],
      "env": {
        "MNEMOS_BASE_URL": "http://localhost:8700",
        "MNEMOS_TIMEOUT_S": "90"
      }
    }
  }
}
```

An example file is available at:

```text
mcp_servers/mnemos/claude_desktop_config.example.json
```

## 4. Restart Claude Desktop

Fully quit and reopen Claude Desktop after changing the config.

## 5. Test In Claude

Ask Claude:

```text
Use the mnemos MCP server. Call health_check and get_capabilities, then search
MNEMOS for "GateMem G4 frozen regression baseline".
```

Expected tool names:

```text
mnemos.health_check
mnemos.get_capabilities
mnemos.search_memory
mnemos.write_observation
mnemos.record_decision
mnemos.find_related_context
mnemos.detect_contradictions
mnemos.summarize_session_handoff
mnemos.explain_memory_provenance
```

If memory writes work but repo-context search is weak or noisy, seed a focused
set of repo summaries and canonical documents first:

```powershell
python tools/seed_mnemos_repo_summaries.py
python tools/seed_mnemos_repo_context.py
```

## Troubleshooting

- If Claude does not list MNEMOS tools, verify the config JSON is valid and
  restart Claude Desktop.
- If tools list but calls fail, run
  `.\mcp_servers\mnemos\.venv\Scripts\python.exe tools/smoke_mnemos_mcp_live.py`.
- If `health_check` returns `unavailable`, MNEMOS REST is not reachable at
  `MNEMOS_BASE_URL`.
- If first write/search calls time out, keep `MNEMOS_TIMEOUT_S=90` for cold
  local embedding/model load.
- If repo searches return unrelated governance/compliance content, the active
  collection is noisy. Seed summary cards with
  `python tools/seed_mnemos_repo_summaries.py` or switch to a clean collection.
