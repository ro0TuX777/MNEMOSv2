# ADR 0005: Context Atlas P0 Is Spec-Only

Date: 2026-06-20

Status: Accepted

## Context

Context Atlas describes exploratory context navigation over existing memory,
cycle, and audit records. It is useful as a design lane, but it is not yet a
production runtime surface.

## Decision

Context Atlas P0 remains research / specification-only until its prerequisite
gates are complete and a future ADR promotes a bounded runtime implementation.

## Alternatives Considered

- Expose Context Atlas as a production API immediately.
- Fold Context Atlas behavior into default search responses.
- Remove the spec until implementation begins.

## Invariants

- Context Atlas is not answer evidence by itself.
- Context Atlas must not mutate memory or promote derived claims.
- Any future runtime must preserve redaction and audit boundaries.
- Default search behavior must remain unchanged by the spec.

## Rollback

If experimental Context Atlas behavior appears in production routes, remove the
route exposure and return the lane to spec-only status.

## Evidence

- `docs/context_atlas_spec.md`
- `docs/support_matrix.md`

