# Claude Desktop MNEMOS Startup Contract

To make sure Claude uses MNEMOS in a new Claude Desktop project, treat MNEMOS
as a startup contract, not just an installed tool.

Add this instruction to the start of the new chat or project:

```text
Before doing implementation work, use MNEMOS as the project memory layer.

1. Call mnemos.health_check.
2. Call mnemos.get_capabilities.
3. Search MNEMOS for relevant project/task context before reading broadly.
4. Report whether useful memory was found, ignored, or rejected.
5. If memory is used, cite the source IDs or source labels and verify them
   against local project files before relying on them.
6. Record one decision with mnemos.record_decision.
7. Record one observation with mnemos.write_observation.
8. At the end, call mnemos.summarize_session_handoff.

Do not treat MNEMOS memory as authoritative by itself. Use it for orientation
and continuity, then verify against current repo files.
```

## Before Opening Claude Desktop

Make sure the MNEMOS service is running from your local MNEMOS repository root.

```powershell
cd <MNEMOS_REPO>
docker compose up -d
```

For the E1 task collection specifically, run this from the same repo root:

```powershell
cd <MNEMOS_REPO>
docker compose -f docker-compose.yml -f docker-compose.ai_dev_task_01.override.yml up -d mnemos
```

Replace `<MNEMOS_REPO>` with the path where you cloned MNEMOS, for example
`C:\src\MNEMOS`, `D:\projects\MNEMOS`, or `/home/alex/MNEMOS`.

## Example Startup Contract File

Suggested file name:

```text
MNEMOS_STARTUP_CONTRACT.md
```

If your agent or project tooling already reads a conventional instruction file,
you can also put the same contract in `CLAUDE.md`, `AGENTS.md`, or the
project's existing instructions Markdown file.

Example contents:

```markdown
# MNEMOS Startup Contract

Before doing implementation work, use MNEMOS as the project memory layer.

1. Call `mnemos.health_check`.
2. Call `mnemos.get_capabilities`.
3. Search MNEMOS for relevant project/task context before reading broadly.
4. Report whether useful memory was found, ignored, or rejected.
5. If memory is used, cite the source IDs or source labels and verify them
   against local project files before relying on them.
6. Record one decision with `mnemos.record_decision`.
7. Record one observation with `mnemos.write_observation`.
8. At the end, call `mnemos.summarize_session_handoff`.

Do not treat MNEMOS memory as authoritative by itself. Use it for orientation
and continuity, then verify against current repo files.
```

## First Claude Desktop Prompt

After Claude Desktop opens, start with:

```text
Use the MNEMOS startup contract in MNEMOS_STARTUP_CONTRACT.md before doing
implementation work. Call mnemos.health_check and mnemos.get_capabilities,
search for relevant project context, and tell me what memory was found,
ignored, or rejected before you read broadly.
```

Practical success signals:

- Claude reports that it used `mnemos.health_check`.
- Claude reports that `mnemos.get_capabilities` is healthy.
- A memory search returns relevant source labels.
- The retrieval fingerprint shows the expected collection, such as
  `mnemos_ai_dev_e1_task_01`.
- The agent logs or reports whether retrieved context influenced the next
  action.

The important boundary: Claude may have the MNEMOS tool available but still not
use it unless you explicitly require it in the project instructions. For future
projects, put the startup contract in the project's first Markdown instruction
file and in your first prompt.
