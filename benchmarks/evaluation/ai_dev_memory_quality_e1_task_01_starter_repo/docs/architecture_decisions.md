# Architecture Decisions

Status: authoritative for this frozen starter package.

## App Structure

The starter application uses:

- plain HTML/CSS/JavaScript;
- `src/logic.js` for state and behavior logic;
- `src/app.js` for DOM wiring;
- `src/index.html` as the primary app screen.

Do not replace the framework or add a backend.

## Persistence

The product uses browser `localStorage`.

Current storage keys:

- `issue-tracker-state-v2`

The saved-view implementation must remain local-only and use the same
application persistence model.

## Sorting Contract

Visible sort modes:

- `updated_desc`
- `priority_desc`
- `title_asc`

Deterministic tie-break rules:

1. `updated_desc`
   - highest `updatedAt` first
   - tie-break by `title` ascending
   - final tie-break by `id` ascending
2. `priority_desc`
   - priority order: `high`, `medium`, `low`
   - tie-break by `updatedAt` descending
   - next tie-break by `title` ascending
   - final tie-break by `id` ascending
3. `title_asc`
   - title ascending (case-insensitive)
   - tie-break by `updatedAt` descending
   - final tie-break by `id` ascending

## Accessibility Notes

Interactive controls must have accessible names.

Saved-view actions must be keyboard operable and keep focus behavior usable
after create, rename, apply, and delete actions.
