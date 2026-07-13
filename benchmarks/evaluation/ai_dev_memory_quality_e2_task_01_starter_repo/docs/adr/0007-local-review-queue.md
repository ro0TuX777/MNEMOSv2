# ADR 0007 - Local Review Queue and Risk Ordering

Status: accepted

Date: 2026-06-28

## Decision

The Release Review Queue remains local-only. Current behavior is intentionally
not synced to any account, backend, team workspace, cloud export, or shared
review service.

Risk ordering is defined by:

```text
riskScore = severity * impact + (blocker ? 10 : 0)
```

The `risk_desc` comparator sorts by:

1. risk score descending;
2. `updatedAt` descending;
3. `title` ascending;
4. `id` ascending.

## Superseded Guidance

The archived 2025 queue policy is superseded. It used severity-only ordering,
treated some deferred decisions as accepted, and assumed future sync. None of
those rules are active.

