# Data Contract

Status: authoritative for this frozen starter package.

## Current Application State Shape

The persisted state is versioned.

### Schema v2

```json
{
  "schemaVersion": 2,
  "issues": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "status": "todo|in_progress|done",
      "priority": "low|medium|high",
      "createdAt": "ISO-8601 timestamp",
      "updatedAt": "ISO-8601 timestamp"
    }
  ],
  "savedViews": [
    {
      "id": "string",
      "name": "string",
      "statusFilters": ["todo", "in_progress", "done"],
      "priorityFilters": ["low", "medium", "high"],
      "searchTerm": "string",
      "sortMode": "updated_desc|priority_desc|title_asc"
    }
  ],
  "uiState": {
    "sortMode": "updated_desc|priority_desc|title_asc"
  }
}
```

## Legacy Schema v1

The starter repository may encounter older persisted state with:

- missing `schemaVersion`;
- missing `savedViews`;
- missing `uiState.sortMode`;
- issues missing `description`;
- issues missing `priority`.

## Migration Rules

Migration must be idempotent.

Rules:

1. Missing `description` becomes `""`.
2. Missing `priority` defaults to `low`.
3. Missing `uiState.sortMode` defaults to `updated_desc`.
4. Missing `savedViews` defaults to `[]`.
5. Invalid `status` records are discarded.
6. Invalid `priority` values become `low`.
7. Invalid or missing `title` records are discarded.
8. Migration must not crash on malformed top-level state.

The migration may drop invalid records, but it must not invent user intent
beyond the documented defaults above.
