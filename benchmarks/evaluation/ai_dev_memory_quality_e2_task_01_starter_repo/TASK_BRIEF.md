# E2 Task 01 - Durable Context / Stale Guidance Rejection

## Mission

Complete the Local Release Review Queue app.

This benchmark is designed to test whether an AI developer can preserve
non-obvious historical constraints while repairing a seeded regression. Current
project docs, ADRs, known-issue notes, and archived superseded guidance all
exist in the repo. The correct implementation must follow current docs and
reject stale archived guidance.

## Required Product Behavior

Implement the release-review logic and UI so that:

- review items migrate from older saved-state shapes into the current v2 schema;
- missing `severity` and `impact` default to `1`;
- missing `blocker` defaults to `false`;
- legacy status `waiting` migrates to `in_review`;
- legacy status `accepted` migrates to `approved`;
- status `deferred` remains `deferred` and is not promoted to `approved`;
- the app remains local-only with no account sync, backend, sharing, or cloud export;
- `risk_desc` sorting uses current risk score:
  - `severity * impact`
  - plus `10` when `blocker` is true
  - tie-break by `updatedAt` descending
  - then `title` ascending
  - then `id` ascending;
- the default review queue includes only `open` and `in_review` items;
- closed statuses (`approved`, `rejected`, `deferred`) appear only when explicitly requested;
- applying a review decision accepts only current statuses:
  - `in_review`
  - `approved`
  - `rejected`
  - `deferred`.

## Stale Guidance Trap

The archived file under `docs/archive/` describes an older policy. It is
intentionally plausible but superseded. Do not implement from it.

In particular, do **not**:

- enable sync or cloud export;
- treat `deferred` as `approved`;
- use old severity-only ordering;
- use old status `accepted` as an active status.

## Constraints

- Do not modify `TASK_BRIEF.md`, `ACCEPTANCE_CRITERIA.md`, `task_control_manifest.json`, `acceptance/`, or `docs/`.
- Keep the implementation local and dependency-light.
- Build the usable app as the first screen, not a landing page.
- Preserve vanilla JavaScript architecture unless a change is required by the tests.

## Verification

Run:

```powershell
npm run test:acceptance
npm run build
```

The acceptance suite is the scored contract for both conditions.

