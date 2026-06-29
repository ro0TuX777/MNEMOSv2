# AI Developer Trial: MNEMOS-Enabled E2 Task 01

Purpose: test whether MNEMOS helps an AI developer preserve durable project
constraints, retrieve current source-linked context, and reject superseded
guidance while repairing a seeded regression.

```text
AI_DEV_MEMORY_QUALITY_E2
DURABLE_CONTEXT_AND_STALE_GUIDANCE_REJECTION
MNEMOS_ENABLED
LOCAL_DEVELOPMENT_EVIDENCE_ONLY
NO_GENERAL_MEMORY_CLAIM
```

## Workspace

Work only in the assigned project folder. Ignore archived prior-run folders
except to note that they exist and must not be used as task context.

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

## MNEMOS Requirement

Use MNEMOS before implementation:

1. Call `health_check`.
2. Call `get_capabilities`.
3. Call `find_related_context` or `search_memory` for the E2 task.
4. Identify current-authority memory versus superseded/stale memory.
5. Verify any retrieved memory against local project files before relying on it.
6. Record one implementation decision with `record_decision`.
7. Record one fix/learning observation with `write_observation`.
8. Record final continuity with `summarize_session_handoff`.

Expected collection:

```text
mnemos_ai_dev_e2_task_01
```

Expected seed snapshot should match `task_control_manifest.json` after setup.

## Trial Folder

Write artifacts under:

```text
trial_results/e2/task_01/mnemos_enabled/
```

Required files:

```text
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

## Manifest

Use the E1 manifest schema, with these task-specific values:

```json
{
  "run_label": "AI_DEV_MEMORY_QUALITY_E2_MNEMOS_ENABLED",
  "memory_condition": "mnemos_enabled",
  "task_id": "release_review_queue_v1",
  "app_task": "Local Release Review Queue",
  "collection_name": "mnemos_ai_dev_e2_task_01",
  "execution_path": "mcp"
}
```

If the MCP tools are not exposed but the REST service is healthy and retrieval
works, use `execution_path: "rest_fallback"` and log that honestly.

## Memory Call Logging

Every memory call must append the full E1 `memory_calls.jsonl` schema. For
retrieval rows, include:

- actual query text;
- retrieval fingerprint;
- returned source labels;
- returned source types or authority states where available;
- IDs/sources used;
- IDs/sources rejected;
- whether each result was verified against local files;
- whether the result changed the next concrete action.

Superseded archive results are not automatically harmful. They are useful only
if they help identify what must be rejected. Do not implement from them.

## Final Report

At the end, include:

- outcome and tests;
- estimated tokens;
- elapsed time;
- files read/edited;
- wrong turns;
- user interventions;
- memory calls;
- current-authority results used;
- superseded/stale results rejected;
- whether retrieved context changed implementation choices;
- caveats and missing measurements.

Then run:

```powershell
python G:\MNEMOS\tools\verify_ai_dev_memory_trial.py <trial-folder> --condition mnemos_enabled --require-e1
```

Report the verifier output.

