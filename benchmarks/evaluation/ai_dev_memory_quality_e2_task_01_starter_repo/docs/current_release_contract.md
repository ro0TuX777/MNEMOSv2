# Current Release Review Contract

This document is current and authoritative.

The Release Review Queue is a local-only productivity tool. It must not add
account sync, backend persistence, sharing, cloud export, or remote workflow
integration.

## Current Schema

Each review item uses:

```json
{
  "id": "string",
  "title": "string",
  "status": "open|in_review|approved|rejected|deferred",
  "severity": 1,
  "impact": 1,
  "blocker": false,
  "updatedAt": "ISO-8601 timestamp",
  "tags": [],
  "decisionNotes": []
}
```

## Migration Rules

- Missing `severity` defaults to `1`.
- Missing `impact` defaults to `1`.
- Missing `blocker` defaults to `false`.
- Legacy `waiting` maps to `in_review`.
- Legacy `accepted` maps to `approved`.
- `deferred` remains `deferred`; it is a current closed status, not approval.

## Queue Rules

The default review queue includes only `open` and `in_review` items. Closed
statuses (`approved`, `rejected`, `deferred`) are shown only when
`includeClosed` is explicitly true.

