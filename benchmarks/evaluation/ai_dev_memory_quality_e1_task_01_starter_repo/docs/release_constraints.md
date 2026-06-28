# Release Constraints

Status: authoritative for this frozen starter package.

## Scope Constraints

- No backend.
- No cloud sync.
- No authentication.
- No analytics.
- No framework replacement.
- No acceptance-suite edits.

## Dependency Constraint

Do not add external dependencies unless the existing repository already
requires them. This starter package does not require any external dependency.

## Trial Integrity

The following are frozen during the paired run:

- `TASK_BRIEF.md`
- `ACCEPTANCE_CRITERIA.md`
- `acceptance/`
- the five context records under `docs/`

If the evaluator changes any frozen task-package file, both paired legs must be
restarted from the same newly frozen package.
