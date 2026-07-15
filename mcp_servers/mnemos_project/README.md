# MNEMOS Project Memory MCP R0

This separate MCP server exposes one immutable local project-memory packet. It
does not contact the MNEMOS REST service or any collection, and it has no write,
lint, shell, decision, observation, or mutation tool.

## 1. Build an explicit packet

There is deliberately no current-directory or whole-repository default. Supply
`--project-root`, `--repo-id`, `--output`, and at least one `--scope-root` or
`--scope-file`. The output must be a new path outside the selected project.

```powershell
$packet = Join-Path $env:TEMP 'mnemos-project-memory.md'
python tools/build_local_project_memory_packet.py `
  --project-root G:\MNEMOS `
  --repo-id mnemos `
  --scope-root mnemos `
  --scope-root service `
  --scope-root mcp_servers/mnemos `
  --scope-file README.md `
  --output $packet
```

Only Git-tracked Python and Markdown inside that scope are admitted. Inspect the
packet's scope, exclusions, snapshot ID, and approval checkpoints before using
it.

## 2. Configure the VS Code MCP client

Build the packet outside the target project, then configure the server:

```json
{
  "mcpServers": {
    "mnemos-project": {
      "command": "G:\\MNEMOS\\mcp_servers\\mnemos\\.venv\\Scripts\\python.exe",
      "args": ["G:\\MNEMOS\\mcp_servers\\mnemos_project\\server.py"],
      "env": {
        "MNEMOS_PROJECT_PACKET": "C:\\Users\\operator\\AppData\\Local\\Temp\\mnemos-project-memory.md",
        "MNEMOS_PROJECT_ROOT": "G:\\MNEMOS",
        "MNEMOS_PROJECT_REPO_ID": "mnemos"
      }
    }
  }
}
```

Restart or reload the MCP client after changing this configuration. Confirm
`project_memory_health` reports the expected repository, snapshot, and
`fresh=true` before asking the agent about project logic.

## 3. Give the frontier agent its boundary

Recommended frontier-agent instruction:

> Search MNEMOS project memory before making claims about project logic. Treat
> returned Python and Markdown spans as evidence, verify them against the live
> file while editing, display the exact read-only lint command and obtain human
> approval before lint, obtain separate approval before edits, and rebuild the
> packet after any admitted-file change.

The server abstains from evidence retrieval when a packet is tampered,
incomplete, configured for the wrong repository, or stale against the live
project root.

## 4. Verify an installed MNEMOS stack separately

Project-memory construction does not depend on Docker, Research Intake, the
MNEMOS REST runtime, Qdrant, PostgreSQL, or any MNEMOS collection. If an operator
also wants to diagnose an installed MNEMOS stack, run the separate read-only
preflight against the Compose file that owns the running containers:

```powershell
python tools/verify_mnemos_local_stack.py `
  --compose-file G:\MNEMOS\docker-compose.yml `
  --require-research-ui `
  --require-openwebui-proxy `
  --output-json (Join-Path $env:TEMP 'mnemos-local-stack-receipt.json')
```

The verifier does not start or change containers. A Compose copy in a Git
worktree can have a different derived project identity; select the installed
checkout's Compose file or use the explicit service-name and URL overrides.
