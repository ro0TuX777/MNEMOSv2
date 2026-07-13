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

Operational note for local verification:

- Do not treat MNEMOS as unavailable just because the MCP tool interface is not
  exposed in the current session.
- First verify the live REST service by checking `http://localhost:8700/health`
  and by issuing a real retrieval request against the running container.
- If the REST service is healthy and returns real retrieval output, record that
  as live evidence and note the MCP-tool fallback in the trial artifacts.
- Do not write placeholder "service unavailable" results when a live REST check
  has already succeeded.

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
you to proceed without MNEMOS. If a live REST health check and a real retrieval
request succeed, treat that as a verified local MNEMOS service state and record
that evidence explicitly rather than treating the run as blocked by the missing
MCP interface.

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

## Evidence-Grade Instrumentation Requirement

This next lane is `AI_DEV_MEMORY_QUALITY_E1`. The trial must record retrieval
state at the point of use, not reconstruct it later.

Before implementation begins, freeze `run_manifest.json` with the run identity,
task identity, repository state, client/tooling state, and configured MNEMOS
retrieval state. During execution, append one `memory_calls.jsonl` row per
memory call with the actual retrieval fingerprint and whether the returned
material affected the next action.

Timestamp and token-accounting rules:

- `started_at` must record the real run start, not a placeholder midnight or
  copied value.
- `completed_at` must record the real completion time for the same run.
- Every JSONL `timestamp` must fall within the manifest start/end window.
- If exact token counts are unavailable, do not write fake zeros. Use `null`
  or omit the exact count and provide an explicit accounting method.
- Record whether the run used native MCP tools or a verified REST fallback.

### `run_manifest.json`

Write at the start:

```json
{
  "trial_id": "string",
  "run_label": "AI_DEV_MEMORY_TRIAL_MNEMOS_ENABLED",
  "memory_condition": "mnemos_enabled",
  "task_id": "local_issue_tracker_v1",
  "task_spec_hash": "sha256-or-other-frozen-hash",
  "agent": "unknown",
  "model": "unknown",
  "client_version": "unknown",
  "started_at": "ISO-8601 timestamp",
  "app_task": "Local Issue Tracker",
  "repo_root": "absolute or project-relative path",
  "initial_repo_commit_or_hash": "git commit, tree hash, or not_available",
  "tool_configuration": {
    "shell": "powershell|bash|other",
    "test_command": "npm test",
    "build_command": "npm run build"
  },
  "mnemos_base_url": "http://localhost:8700",
  "mcp_server_name": "mnemos",
  "mnemos_service_revision": "unknown",
  "mcp_revision": "unknown",
  "collection_name": "string",
  "seed_snapshot": "string",
  "configured_retrieval_profile": "string",
  "execution_path": "mcp|rest_fallback",
  "cache_state_at_start": "cold|warm|unknown",
  "cache_policy_version": "string|unknown",
  "token_counts_available": false,
  "token_accounting_method": "exact|estimated|unavailable",
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
  "estimated_input_tokens": null,
  "estimated_output_tokens": null,
  "acceptance_test_command": "npm test",
  "acceptance_test_result": "pass|fail|partial",
  "cache_state_at_end": "cold|warm|mixed|unknown"
}
```

Do not overwrite the frozen starting values. Only append completion fields or
previously unknown values discovered during the run.

### `memory_calls.jsonl`

Append one JSON object per MNEMOS call:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "tool": "search_memory",
  "purpose": "find related project context",
  "query_or_summary": "short string",
  "query_text": "actual query or request payload summary",
  "retrieval_fingerprint": "actual executed-route fingerprint or not_available",
  "execution_path": "mcp|rest_fallback",
  "cache_state_observed": "cold|warm|cache_hit|cache_miss|unknown",
  "cache_hit": false,
  "configured_retrieval_profile": "string",
  "duplicate_suppression_count": 0,
  "result_count": 0,
  "returned_source_ids": [],
  "returned_source_labels": [],
  "returned_source_types": [],
  "returned_scores": [],
  "used_result_ids": [],
  "rejected_result_ids": [],
  "abstention_reason": null,
  "verified_against_primary_evidence": true,
  "used_in_next_action": true,
  "next_action_summary": "short string",
  "helpfulness": "useful|neutral|harmful",
  "notes": "short note"
}
```

If scores, cache details, or duplicate-suppression counts are not surfaced by
the tool, write `null`, `unknown`, or `not_available` explicitly rather than
silently omitting the field.

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
8. Record whether each memory result changed the next concrete action.
9. If MCP tools are not exposed but the live REST service is used instead,
   record `execution_path: rest_fallback` in both the manifest and each memory
   call row.

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
