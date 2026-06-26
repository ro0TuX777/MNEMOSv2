# AI Developer Trial: MNEMOS-Enabled Run

Purpose: test whether MNEMOS helps an AI developer build, fix, enhance, and
refine a small app with less orientation cost, fewer wrong turns, better
continuity, and lower user tax.

This file is meant to be copied into a fresh app project and read by the AI
developer agent before it begins work.

## Run Label

```text
AI_DEV_MEMORY_TRIAL_MNEMOS_ENABLED
MNEMOS_MCP_AGENT_MEMORY_ALPHA
LOCAL_DEVELOPMENT_EVIDENCE_ONLY
NO_GENERAL_MEMORY_CLAIM
```

## Required Setup

The agent should assume MNEMOS is available through the MCP server named
`mnemos`.

Expected local services:

```text
MNEMOS REST: http://localhost:8700
MNEMOS MCP: mcp_servers/mnemos/server.py
```

Expected MCP tools:

```text
health_check
get_capabilities
search_memory
write_observation
record_decision
find_related_context
detect_contradictions
summarize_session_handoff
explain_memory_provenance
```

If the MCP tools are unavailable, record that in
`trial_results/mnemos_enabled/blockers.md` and continue only if the user asks
you to proceed without MNEMOS.

## App Task

Build a small but nontrivial local app. Unless the user gives a different app
idea, build:

```text
Local Issue Tracker
```

Required features:

- create, edit, delete issues;
- issue status: `todo`, `in_progress`, `done`;
- priority: `low`, `medium`, `high`;
- text search and status/priority filtering;
- persisted local state;
- useful empty states;
- basic responsive UI;
- focused tests for core behavior;
- final polish pass after tests pass.

Do not make a marketing landing page. Build the usable app as the first screen.

## Trial Folder Structure

Create this folder structure in the new project:

```text
trial_results/
  mnemos_enabled/
    README.md
    run_manifest.json
    agent_route_log.jsonl
    memory_calls.jsonl
    repo_activity.jsonl
    user_interventions.jsonl
    wrong_turns.jsonl
    test_runs.jsonl
    final_report.md
    blockers.md
```

All files should be append-only during the run where practical.

## What To Track

Track more than tokens. Tokens matter, but the goal is usefulness.

Required metrics:

- total estimated input tokens;
- total estimated output tokens;
- elapsed time;
- files opened/read;
- files created;
- files edited;
- searches run;
- commands run;
- tests run;
- test failures;
- wrong turns;
- user interventions;
- memory tool calls;
- memory results used;
- memory results rejected;
- stale or conflicting memory detected;
- primary evidence checks after memory retrieval;
- repeated context requests to the user;
- final app completion status.

If exact token counts are unavailable, estimate them and mark them as
`estimated`.

## Required Logging

### `run_manifest.json`

Write at the start:

```json
{
  "run_label": "AI_DEV_MEMORY_TRIAL_MNEMOS_ENABLED",
  "memory_condition": "mnemos_enabled",
  "agent": "unknown",
  "model": "unknown",
  "started_at": "ISO-8601 timestamp",
  "app_task": "Local Issue Tracker",
  "mnemos_base_url": "http://localhost:8700",
  "mcp_server_name": "mnemos",
  "token_counts_available": false,
  "claim_boundary": {
    "local_development_evidence_only": true,
    "general_memory_claim": false
  }
}
```

Update it at the end with:

```json
{
  "completed_at": "ISO-8601 timestamp",
  "final_status": "completed|partial|blocked",
  "estimated_input_tokens": 0,
  "estimated_output_tokens": 0
}
```

### `memory_calls.jsonl`

Append one JSON object per MNEMOS call:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "tool": "search_memory",
  "purpose": "find related project context",
  "query_or_summary": "short string",
  "result_count": 0,
  "used_result_ids": [],
  "rejected_result_ids": [],
  "verified_against_primary_evidence": true,
  "helpfulness": "useful|neutral|harmful",
  "notes": "short note"
}
```

### `agent_route_log.jsonl`

Append major reasoning/navigation steps:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": "orientation|implementation|debugging|testing|polish|handoff",
  "action": "short description",
  "why": "short rationale",
  "inputs_used": ["file path", "memory id", "command"],
  "outcome": "short result"
}
```

### `repo_activity.jsonl`

Append file/search/command activity:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "activity": "read_file|edit_file|create_file|search|command",
  "target": "path or command",
  "purpose": "short purpose",
  "result": "short result"
}
```

### `user_interventions.jsonl`

Append every time the user had to clarify, correct, redirect, or restate:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "intervention_type": "clarification|correction|preference|blocker|redirect",
  "user_message_summary": "short summary",
  "avoidable": true,
  "notes": "short note"
}
```

### `wrong_turns.jsonl`

Append mistakes or inefficient paths:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "wrong_turn_type": "bad_assumption|irrelevant_file|failed_approach|stale_memory|test_failure",
  "description": "short description",
  "detected_by": "agent|user|test|memory|tool",
  "recovery": "short recovery action"
}
```

### `test_runs.jsonl`

Append each test/check run:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "command": "npm test",
  "passed": true,
  "summary": "short result",
  "failure_count": 0
}
```

## Required MNEMOS Behavior

Use MNEMOS in these moments:

1. At the start, call `health_check` and `get_capabilities`.
2. During orientation, call `find_related_context` or `search_memory`.
3. After making a meaningful design decision, call `record_decision`.
4. After fixing a bug or failed test, call `write_observation`.
5. Before final handoff, call `summarize_session_handoff`.
6. If memory suggests something important, verify against project files before
   relying on it.
7. If memory appears stale or contradictory, record it as rejected rather than
   silently using it.

Memory should reduce wandering, not replace evidence.

## Final Report

At the end, write `trial_results/mnemos_enabled/final_report.md` with:

```text
# MNEMOS-Enabled Trial Final Report

## Outcome
- final status:
- app summary:
- tests:

## Metrics
- estimated input tokens:
- estimated output tokens:
- elapsed time:
- files opened/read:
- files created:
- files edited:
- searches run:
- commands run:
- tests run:
- test failures:
- wrong turns:
- user interventions:
- memory tool calls:
- memory results used:
- memory results rejected:
- stale/conflicting memories detected:
- primary evidence checks after memory:

## Memory Usefulness
- useful memory moments:
- neutral memory moments:
- harmful memory moments:
- did MNEMOS reduce orientation cost:
- did MNEMOS increase user tax:

## Route Summary
- orientation path:
- implementation path:
- debugging path:
- final verification path:

## Caveats
- missing measurements:
- known confounds:
- unresolved issues:
```

## Operating Instruction To The Agent

Proceed autonomously where reasonable. Do not ask the user to manage memory.
Use MNEMOS as your project-memory substrate, but keep primary project evidence
authoritative. Keep the trial logs current as you work.
