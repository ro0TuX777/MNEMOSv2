# E1 Task 01 — Shared Acceptance Criteria

## Purpose

This is the sole scored acceptance suite for both conditions in the E1 paired trial.

Both the MNEMOS-enabled condition and the baseline condition must run this suite unchanged against their completed application. Agent-authored tests may supplement local development but do not substitute for these checks.

## Acceptance Command

The frozen starter repository must expose one command:

```text
npm run test:acceptance
```

The command must run the complete shared acceptance suite and return a non-zero exit code on any failure.

The evaluator must record:

```text
command
timestamp
pass/fail result
failure count
time to first passing run
time to final passing run
```

## Functional Acceptance Criteria

### A. Existing Issue Tracker Behavior

The completed application must preserve the existing core behavior:

1. Create an issue with title, status, and priority.
2. Edit an existing issue.
3. Delete an existing issue.
4. Filter issues by status.
5. Filter issues by priority.
6. Search issues by text.
7. Persist issue data across a reload.
8. Show a useful empty state when no issues match the active filters.

### B. Saved Views

The application must:

1. Save the current combination of:
   - status filters;
   - priority filters;
   - search term;
   - selected sort mode.
2. Require a non-empty saved-view name after trimming leading/trailing whitespace.
3. Apply a saved view so the visible issue list and controls reflect the saved state.
4. Allow renaming a saved view.
5. Allow deleting a saved view.
6. Persist saved views across a reload.
7. Show an understandable empty state when no views exist.
8. Prevent duplicate saved-view identifiers even when view names match.
9. Handle a saved view that references no matching issues without error.

### C. Sorting

The application must support:

```text
updated_desc
priority_desc
title_asc
```

The test suite will verify that:

1. the selected sort mode changes list order;
2. sorting is deterministic;
3. tie-breaking follows the project-context contract;
4. the selected sort mode persists as required by the project-context contract;
5. applying a saved view restores its saved sort mode.

### D. Migration and Persistence

The application must correctly handle supplied legacy persisted data fixtures.

The test suite will verify that:

1. valid legacy issues remain present after migration;
2. required defaults are applied only where the documented contract permits;
3. migration is idempotent;
4. malformed persisted data does not crash the application;
5. malformed records are handled according to the project-context contract;
6. saved views remain valid after a reload and migration pass.

### E. Seeded Defect Repair

The application must pass the regression test derived from `docs/known_issues.md`.

The test suite will confirm that the documented defect is repaired without weakening filtering, sorting, persistence, or validation behavior.

### F. Accessibility and Usability

The acceptance suite will verify:

1. saved-view controls have accessible names;
2. the primary saved-view actions are keyboard operable;
3. focus remains usable after create, rename, apply, and delete actions;
4. empty states remain understandable;
5. narrow-layout rendering does not hide required controls.

### G. Build and Scope Integrity

The completed repository must:

1. pass the production build command:
   ```text
   npm run build
   ```
2. retain the starter application framework and local persistence mechanism;
3. contain no required network service or backend;
4. contain no modifications to `TASK_BRIEF.md`, `ACCEPTANCE_CRITERIA.md`, or `acceptance/`;
5. not skip, delete, weaken, or conditionally bypass acceptance tests.

## Required Acceptance Artifacts

The acceptance suite must emit or make available:

```text
acceptance-result summary
pass/fail status
failed-check names, if any
test count
test duration
build result
```

## Scoring Rule

A task outcome is:

```text
PASS
```

only when:

```text
npm run test:acceptance passes
AND npm run build passes
AND task-control integrity checks pass
```

Otherwise the outcome is:

```text
FAIL
```

or:

```text
PARTIAL
```

only when the evaluator explicitly records which acceptance criteria were not met.

## Trial-Integrity Rule

The acceptance suite is frozen before either agent starts.

Any modification to:

```text
TASK_BRIEF.md
ACCEPTANCE_CRITERIA.md
acceptance/
```

invalidates the paired comparison unless the evaluator restarts both legs from the same newly frozen task package.

## Reporting Boundary

A passing acceptance suite means only that the condition completed this defined task under these defined criteria. It does not establish general coding ability, general memory value, or broad MNEMOS performance.
