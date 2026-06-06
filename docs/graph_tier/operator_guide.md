# GraphHybrid Experimental Mode: Operator Guide

**Status:** `EXPERIMENTAL / IN-MEMORY VALIDATED ONLY`
**Do not enable in production environments handling live user queries without explicit shadow-traffic validation.**

## Overview
The `graph_hybrid_experimental` retrieval mode augments standard semantic RAG queries with contextual relationships sourced from the internal `GraphTier`. It operates by injecting highly curated knowledge graph neighbors directly into the retrieval envelope.

This mode has passed offline benchmarking (MG-Test-1 through MG-Test-4B) to ensure it does not compromise metadata integrity, lineage enforcement, or LLM token caps. However, it has **not** been stress-tested against a live distributed graph resolver.

## Isolation and Opt-In Mechanics
To prevent accidental exposure, this mode requires a strict **Double Opt-In** to execute.

### 1. Global Config Flag
The core retrieval system defaults to blocking the mode entirely. It must be turned on at the service/router level:
```python
router = RetrievalRouter(
    # ... other tiers ...
    enable_experimental_graph_hybrid=True
)
```

### 2. Request-Level Mode
Each individual query must explicitly request the mode:
```python
hits, meta = router.search(
    query="Example query",
    top_k=10,
    retrieval_mode="graph_hybrid_experimental"
)
```

## Fallback & Rollback Behavior
If a query requests `graph_hybrid_experimental` but the global `enable_experimental_graph_hybrid` flag is `False`, the system will **gracefully fall back** to the standard `"semantic"` mode. 

**To Rollback:**
Simply revert the global config flag to `False` or stop sending `"graph_hybrid_experimental"` in the request payload. Both actions will fully terminate the GraphTier evaluation paths.

## Telemetry Visibility
You can monitor whether the experimental mode is actively running by inspecting the `experimental_graph_telemetry` dictionary in the returned metadata object.

**When properly enabled and executing:**
```json
{
  "experimental_graph_hybrid_requested": true,
  "experimental_graph_hybrid_enabled": true,
  "experimental_graph_hybrid_action": "executed",
  "warning": "EXPERIMENTAL_IN_MEMORY_VALIDATED_ONLY"
}
```

**When falling back to semantic (due to global config being False):**
```json
{
  "experimental_graph_hybrid_requested": true,
  "experimental_graph_hybrid_enabled": false,
  "experimental_graph_hybrid_action": "fallback_to_semantic",
  "warning": "GRAPH_HYBRID_EXPERIMENTAL_NOT_ENABLED"
}
```
