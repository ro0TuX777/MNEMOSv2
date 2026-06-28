# MNEMOS Agent Navigation Pair 001 Comparison

```text
MNEMOS_AGENT_NAVIGATION_DEVELOPMENT_STUDY_READY
LOCAL_REPO_AGENT_ORIENTATION_AND_BOUNDARY_RECALL_DEVELOPMENT_STUDY
NO_GATEMEM_REOPENING
NO_GENERAL_MEMORY_CLAIM
```

Task: `nav-stale-memory-rejection`

Question: decide whether the memory claim that internal G4 policy work can
continue is current.

| Mode | Files opened | Searches | Memory cards retrieved | Rejected memories | Boundary decision | Forbidden claims |
|---|---:|---:|---:|---:|---|---|
| `mnemos_memory_assisted` | 2 | 0 | 3 | 1 | `reject_stale_memory_and_preserve_pause` | 0 |
| `baseline_repo_search` | 2 | 1 | 0 | 1 | `reject_stale_memory_and_preserve_pause` | 0 |

## Observation

Both modes reached the same boundary-safe result: the internal-continuation
memory is stale, G4 remains frozen for regression testing only, and further
GateMem work is paused pending independent sealed-evaluation inputs.

The memory-assisted run saved one explicit repository search and still rejected
the stale memory against primary evidence. The baseline run also succeeded, but
needed an orienting search before opening the same two evidence files.

This is local development evidence only. It does not support a general memory
claim, a GateMem reopening, or a fresh evaluation claim.
