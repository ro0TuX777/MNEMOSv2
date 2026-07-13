# AI Developer Trial: No-Memory Control Run

Purpose: test the same AI developer app-building task without MNEMOS or any
external/project memory tool, so results can be compared against the
MNEMOS-enabled run.

This file is meant to be copied into a fresh app project and read by the AI
developer agent before it begins work.

## Run Label

```text
AI_DEV_MEMORY_TRIAL_NO_MEMORY_CONTROL
NO_MNEMOS_MCP
LOCAL_DEVELOPMENT_EVIDENCE_ONLY
NO_GENERAL_MEMORY_CLAIM
```

## Required Constraint

Do not use MNEMOS, MCP memory tools, vector memory, external project memory, or
any prior trial results from the MNEMOS-enabled run.

Prohibited tools/surfaces:

```text
mnemos
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

Use only the project files in this new folder, ordinary repo/file inspection,
normal shell commands, and the user’s messages in this run.

If a MNEMOS/MCP memory tool is available in the environment, ignore it and
record that it was intentionally not used in
`trial_results/no_memory/blockers.md`.

## App Task

Build the same small but nontrivial local app used by the MNEMOS-enabled run.
Unless the user gives a different app idea, build:

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
  no_memory/
    README.md
    run_manifest.json
    agent_route_log.jsonl
    repo_activity.jsonl
    user_interventions.jsonl
    wrong_turns.jsonl
    test_runs.jsonl
    final_report.md
    blockers.md
```

All files should be append-only during the run where practical.

Do not create `memory_calls.jsonl` for this run. If you do, explain why in
`blockers.md`; the expected memory-call count is zero.

## What To Track

Track the same metrics as the MNEMOS-enabled run except memory-call details.

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
- repeated context requests to the user;
- final app completion status.

If exact token counts are unavailable, estimate them and mark them as
`estimated`.

## Required Logging

## Evidence-Grade Instrumentation Requirement

This next lane is `AI_DEV_MEMORY_QUALITY_E1`. The control run must emit the
same run-manifest structure as the MNEMOS-enabled run, but MNEMOS-only fields
should be written as `null`, `unknown`, or `not_applicable` rather than
invented.

Timestamp and token-accounting rules:

- `started_at` and `completed_at` must reflect the real run window.
- Every JSONL `timestamp` must fall within that same window.
- If exact token counts are unavailable, do not write fake zeros. Use `null`
  plus an explicit accounting method.

### `run_manifest.json`

Write at the start:

```json
{
  "trial_id": "string",
  "run_label": "AI_DEV_MEMORY_TRIAL_NO_MEMORY_CONTROL",
  "memory_condition": "no_memory",
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
  "mnemos_or_memory_tools_allowed": false,
  "mnemos_base_url": null,
  "mcp_server_name": null,
  "mnemos_service_revision": null,
  "mcp_revision": null,
  "collection_name": null,
  "seed_snapshot": null,
  "configured_retrieval_profile": "not_applicable",
  "execution_path": "no_memory_control",
  "cache_state_at_start": "not_applicable",
  "cache_policy_version": "not_applicable",
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
  "cache_state_at_end": "not_applicable"
}
```

Do not backfill MNEMOS state from the paired run. Symmetry matters, but false
symmetry is worse than explicit nulls.

### `agent_route_log.jsonl`

Append major reasoning/navigation steps:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": "orientation|implementation|debugging|testing|polish|handoff",
  "action": "short description",
  "why": "short rationale",
  "inputs_used": ["file path", "command"],
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
  "wrong_turn_type": "bad_assumption|irrelevant_file|failed_approach|test_failure",
  "description": "short description",
  "detected_by": "agent|user|test|tool",
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

## Required No-Memory Behavior

1. Do not call MNEMOS or any memory MCP tools.
2. Do not inspect the MNEMOS-enabled run folder or reuse its artifacts.
3. Use normal project inspection only.
4. If you need context, inspect local files or ask the user; log any user
   intervention.
5. Keep the trial logs current as you work.

This is the control condition. The point is not to avoid all context; the point
is to avoid external/project memory assistance.

## Final Report

At the end, write `trial_results/no_memory/final_report.md` with:

```text
# No-Memory Control Trial Final Report

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
- repeated context requests:

## Control Condition
- MNEMOS/MCP memory used: false
- memory tools observed but ignored:
- any accidental memory use:

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

Proceed autonomously where reasonable. Do not use MNEMOS or any memory tool.
Use only local project evidence and normal commands. Keep the trial logs current
as you work.
