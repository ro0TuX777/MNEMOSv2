# E1 Task 01 — Issue Tracker: Context Recovery, Feature Completion, and Safe Refactor

## Purpose

This is a controlled paired AI-developer task for evaluating whether MNEMOS-backed, source-grounded retrieval changes task execution under equivalent conditions.

The task is intentionally **not** a blank-app build. Both conditions must begin from the same frozen starter repository containing an incomplete Issue Tracker application and scattered project-context records.

The MNEMOS-enabled condition may retrieve the source-grounded context through MNEMOS. The baseline condition must recover the same information through ordinary inspection of the supplied repository only.

## Evaluation Boundary

```text
AI_DEV_MEMORY_QUALITY_E1
TASK_01_CONTEXT_RECOVERY_AND_FEATURE_COMPLETION
LOCAL_DEVELOPMENT_EVIDENCE_ONLY
NO_GENERAL_PERFORMANCE_CLAIM
NO_RETRIEVAL_TUNING_DURING_RUNS
```

## Evaluator Preparation — Required Before Either Agent Starts

Create two fresh, byte-equivalent copies of the same frozen starter repository.

The starter repository must contain:

```text
TASK_BRIEF.md
ACCEPTANCE_CRITERIA.md
acceptance/
docs/
src/
package.json
```

The repository must also contain these project-context records. They are intentionally distributed so ordinary repository inspection is possible, while MNEMOS retrieval has a legitimate opportunity to reduce rediscovery:

```text
docs/product_scope.md
docs/architecture_decisions.md
docs/data_contract.md
docs/known_issues.md
docs/release_constraints.md
```

Before the MNEMOS-enabled condition begins:

1. Seed the exact frozen copies of the five `docs/` context records into MNEMOS.
2. Record the collection name, seed snapshot, service revision, and retrieval configuration in the task-control manifest.
3. Do not modify the seeded records, summary cards, retrieval profile, cache policy, routing configuration, or task files during either paired leg.

Before the baseline condition begins:

1. Confirm it has the same five context files in its local repository.
2. Confirm the agent has no MNEMOS, MCP memory, vector-memory, or prior-trial access.
3. Do not provide manually copied retrieval results, summaries, or handoff notes.

The two conditions must use the same model, client version, allowed time budget, shell/tool configuration, starter-repository hash, task files, and shared acceptance suite.

## Product Context

The existing application is a local Issue Tracker. It already supports basic issue creation, editing, deletion, status, priority, text search, filtering, and browser-local persistence.

The task is to complete a constrained enhancement without breaking existing behavior.

The authoritative product, architecture, data, issue, and release details are in the supplied `docs/` records. Treat those source files as authoritative. Do not assume a document is current merely because it is easy to find; resolve conflicts using the current-state and supersession information in the repository.

## Required Work

Implement the following feature set in the existing starter repository.

### 1. Saved Views

Add saved issue-list views.

A saved view must contain:

- a user-provided name;
- selected status filters;
- selected priority filters;
- the current text-search term;
- sort mode;
- a stable local identifier.

Users must be able to:

- create a saved view from the current active filters;
- apply a saved view;
- rename a saved view;
- delete a saved view;
- see an understandable empty state when no saved views exist.

Saved views must persist locally across browser reloads.

### 2. Deterministic Sorting

Add a visible sort control with these modes:

```text
updated_desc
priority_desc
title_asc
```

Sorting must be deterministic. The tie-breaking rules and persistence/migration behavior are defined by the supplied project-context records.

### 3. Data Migration and Compatibility

The starter application may contain pre-existing persisted issue data from an earlier schema.

Implement the required backward-compatible migration behavior described in `docs/data_contract.md`.

The migration must:

- preserve existing valid issues;
- avoid data loss;
- create valid defaults only where permitted by the documented contract;
- not silently reinterpret invalid data as valid user intent;
- remain safe to run more than once.

### 4. Known-Issue Repair

Resolve the seeded issue described in `docs/known_issues.md`.

Do not bypass the defect by weakening a test, removing validation, or changing the documented product behavior.

### 5. UI and Accessibility

Keep the application usable on narrow and desktop layouts.

The updated UI must provide:

- keyboard-operable saved-view controls;
- accessible names for interactive controls;
- visible focus behavior consistent with the starter application;
- useful empty states;
- no unrequested marketing or landing-page work.

### 6. Verification and Polish

Use the shared acceptance suite exactly as supplied.

You may add focused implementation tests, but they do not replace the shared acceptance suite.

Only after the shared acceptance suite passes, perform a final polish pass. Do not introduce unrelated features.

## Explicit Constraints

```text
Do not replace the application framework.
Do not replace the persistence mechanism.
Do not add a backend, cloud service, authentication system, analytics system,
or external dependency unless the existing repository already requires it.
Do not change the task files or acceptance suite.
Do not use hidden or external project context.
Do not add MNEMOS to the product unless explicitly asked by the evaluator.
Do not weaken, delete, skip, or rewrite acceptance tests.
```

## Required End State

A completed run must leave the repository with:

```text
all shared acceptance checks passing
all required features implemented
starter behavior preserved unless the task explicitly changes it
no known seeded defect remaining
trial logs current and complete
a final report written in the required trial-results directory
```

## Agent Operating Instruction

Read `ACCEPTANCE_CRITERIA.md` before editing code.

Proceed autonomously where reasonable. Inspect source evidence before acting. Keep the trial logs current during the run. Do not backfill operational events after completion.
