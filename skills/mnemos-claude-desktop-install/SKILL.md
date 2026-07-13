---
name: mnemos-claude-desktop-install
description: Install or update MNEMOS as a Claude Desktop MCP server on a local machine. Use when Codex needs to verify the MNEMOS MCP bridge, merge or repair `%APPDATA%\\Claude\\claude_desktop_config.json`, generate the correct `mnemos` server entry, confirm local smoke tests, or walk a user through Claude Desktop setup and troubleshooting for the MNEMOS MCP integration.
---

# Mnemos Claude Desktop Install

## Overview

Set up MNEMOS as a local Claude Desktop MCP tool with a valid Claude config,
working Python command, verified bridge, and post-install smoke checks.

## Workflow

1. Confirm the MNEMOS repo root.
2. Run the bridge verification and smoke checks.
3. Detect the Python interpreter path that should launch `mcp_servers/mnemos/server.py`.
4. Merge or repair the Claude Desktop config with the helper script.
5. Tell the user to fully restart Claude Desktop.
6. Verify `mnemos.health_check` and `mnemos.get_capabilities`.
7. If retrieval is noisy, seed focused repo context.

Read [references/claude-desktop-install.md](references/claude-desktop-install.md)
before acting.

## Steps

### 1. Verify the local bridge

From the repo root, run:

```powershell
python tools/verify_mnemos_msf_mcp.py
python tools/smoke_mnemos_mcp_stdio.py
python tools/smoke_mnemos_mcp_live.py
```

If the live smoke fails, stop and report that the backing MNEMOS service is not
ready yet.

### 2. Detect the interpreter path

Use:

```powershell
python -c "import sys; print(sys.executable)"
```

Use that absolute path as the Claude Desktop `command` value unless the user
explicitly wants a different interpreter.

### 3. Merge the Claude Desktop config

Prefer the helper script:

```powershell
python skills/mnemos-claude-desktop-install/scripts/merge_claude_desktop_config.py --repo-root . --python-command "<absolute-python-path>"
```

The script backs up the current Claude config before writing.

If the user wants review before writing, use `--dry-run`.

### 4. Restart and test

After the config change, tell the user to fully quit and reopen Claude Desktop.
Then test:

```text
mnemos.health_check
mnemos.get_capabilities
```

### 5. Seed repo context only when needed

If health/capabilities work but repo search is weak or noisy, run:

```powershell
python tools/seed_mnemos_repo_summaries.py
python tools/seed_mnemos_repo_context.py
```

## Notes

- This skill is optimized for Windows and Claude Desktop local MCP setup.
- Do not overwrite unrelated `mcpServers` entries.
- Preserve existing Claude preferences and non-MCP settings when editing the
  config file.
- If the repo root is unclear, ask for it instead of guessing a server path.
