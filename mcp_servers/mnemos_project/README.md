# MNEMOS Project Memory MCP R0

This separate MCP server exposes one immutable local project-memory packet. It
does not contact the MNEMOS REST service or any collection, and it has no write,
lint, shell, decision, observation, or mutation tool.

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

Recommended frontier-agent instruction:

> Search MNEMOS project memory before making claims about project logic. Treat
> returned Python and Markdown spans as evidence, verify them against the live
> file while editing, display the exact read-only lint command and obtain human
> approval before lint, obtain separate approval before edits, and rebuild the
> packet after any admitted-file change.

The server abstains from evidence retrieval when a packet is tampered,
incomplete, configured for the wrong repository, or stale against the live
project root.
