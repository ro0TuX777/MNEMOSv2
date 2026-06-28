# Product Scope

Status: authoritative for this frozen starter package.

The product is a local Issue Tracker that runs entirely in the browser and
stores state in `localStorage`.

## Existing Core Behavior

The starter application already supports:

- creating an issue with title, status, and priority;
- editing an existing issue;
- deleting an issue;
- filtering by status;
- filtering by priority;
- searching by text;
- persisting issue data across reloads; and
- showing an empty state when no issues match.

## E1 Additions Required

This task adds:

- saved views for filter/search/sort state;
- deterministic visible sorting with named modes;
- backward-compatible persisted-data migration;
- repair of the seeded known defect;
- accessibility-safe interaction updates.

## Saved View Scope

A saved view must persist:

- a user-provided display name;
- selected status filters;
- selected priority filters;
- the active text-search term;
- the active sort mode; and
- a stable local identifier.

Saved views are a local-only productivity feature. No account sync, backend,
sharing, or cloud export is in scope.
