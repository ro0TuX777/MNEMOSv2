# MNEMOS Claude Desktop Install Reference

## Purpose

Use this skill to make MNEMOS available to Claude Desktop as a local MCP
server with minimal manual editing.

## Required checks

From the MNEMOS repo root:

```powershell
python tools/verify_mnemos_msf_mcp.py
python tools/smoke_mnemos_mcp_stdio.py
python tools/smoke_mnemos_mcp_live.py
```

Expected signals:

```text
Checks passed: 11/11
Tools listed: 9
Capabilities status: healthy
Search status: healthy
```

## Claude config target

Claude Desktop config file:

```text
%APPDATA%\Claude\claude_desktop_config.json
```

MNEMOS MCP entry shape:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "C:\\Users\\vin\\anaconda3\\python.exe",
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

## Preferred automation

Use the helper script in this skill:

```powershell
python skills/mnemos-claude-desktop-install/scripts/merge_claude_desktop_config.py --repo-root . --python-command "<absolute-python-path>"
```

Use `--dry-run` to preview the merge.

## Post-install validation

After the config is written:

1. Fully quit and reopen Claude Desktop.
2. Ask Claude to call:

```text
mnemos.health_check
mnemos.get_capabilities
```

## Repo context seeding

If MCP works but repo-context search is weak:

```powershell
python tools/seed_mnemos_repo_summaries.py
python tools/seed_mnemos_repo_context.py
```

## Troubleshooting

- If Claude does not list `mnemos` tools, the JSON is probably invalid or
  Claude Desktop was not fully restarted.
- If `health_check` fails, the MNEMOS REST service is not reachable at
  `MNEMOS_BASE_URL`.
- If first calls are slow, keep `MNEMOS_TIMEOUT_S=90` for local cold starts.
- Preserve unrelated `mcpServers` entries and non-MCP preferences when merging
  config.
