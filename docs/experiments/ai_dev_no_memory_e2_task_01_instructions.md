# AI Developer Trial: No-Memory E2 Task 01

Purpose: run the same durable-context/stale-guidance task without MNEMOS or any
external/project memory tool.

```text
AI_DEV_MEMORY_QUALITY_E2
DURABLE_CONTEXT_AND_STALE_GUIDANCE_REJECTION
NO_MEMORY_CONTROL
LOCAL_DEVELOPMENT_EVIDENCE_ONLY
NO_GENERAL_MEMORY_CLAIM
```

## Workspace

Work only in the assigned project folder. Ignore archived prior-run folders
except to note that they exist and must not be used as task context.

## Required Constraint

Do not use MNEMOS, MCP memory tools, vector memory, external project memory, or
prior trial results.

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

Use only local project files and normal commands.

## Required Task

Read and follow:

- `TASK_BRIEF.md`
- `ACCEPTANCE_CRITERIA.md`
- `task_control_manifest.json`
- current files under `docs/`
- source under `src/`
- frozen tests under `acceptance/`

Do not modify:

- `TASK_BRIEF.md`
- `ACCEPTANCE_CRITERIA.md`
- `task_control_manifest.json`
- `acceptance/`
- `docs/`

Complete the Local Release Review Queue so it passes:

```powershell
npm run test:acceptance
npm run build
```

## Trial Folder

Write artifacts under:

```text
trial_results/e2/task_01/no_memory/
```

Required files:

```text
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

Do not create `memory_calls.jsonl`.

## Manifest

Use the E1 manifest schema, with these task-specific values:

```json
{
  "run_label": "AI_DEV_MEMORY_QUALITY_E2_NO_MEMORY",
  "memory_condition": "no_memory",
  "task_id": "release_review_queue_v1",
  "app_task": "Local Release Review Queue",
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
  "cache_policy_version": "not_applicable"
}
```

## Final Report

At the end, include:

- outcome and tests;
- estimated tokens;
- elapsed time;
- files read/edited;
- wrong turns;
- user interventions;
- stale archived guidance encountered and rejected;
- caveats and missing measurements.

Then run:

```powershell
python G:\MNEMOS\tools\verify_ai_dev_memory_trial.py <trial-folder> --condition no_memory --require-e1
```

Report the verifier output.

