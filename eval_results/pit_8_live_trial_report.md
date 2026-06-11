# PIT-8 Live Trial Report

**Status**: EXPERIMENT COMPLETED
**Kill Switch State**: manual_restart_verified

> **NOTICE**: This harness performed read-only queries and did NOT mutate the server state. No automatic promotion, no Candidate Envelope mixing, and no schema extraction was executed.

## Boundary Tests
- `api_v1_query_rejects_eval_mode`: **PASS**
- `api_v1_query_zero_derived`: **PASS**
- `v1_mnemos_search_zero_derived`: **PASS**
- `evaluate_derived_shadow_success_path`: **SKIPPED_DUE_TO_KILL_SWITCH**

## Telemetry Deltas
```json
{
  "derived_lane.denied_count": 1,
  "derived_lane.execution_count": 0,
  "derived_lane.kill_switch_count": 3,
  "echoframe.production_prompt.derived_count": 0,
  "evaluate_derived_shadow.denied_count": 0,
  "evaluate_derived_shadow.rendered_derived_fact_count": 0,
  "evaluate_derived_shadow.request_count": 3,
  "query.default_retrieval.derived_fact_count": 0
}
```

## Latency
- `api_v1_query_eval_mode`: 1.69 ms
- `api_v1_query_default`: 7716.75 ms
- `v1_mnemos_search_default`: 20.36 ms
- `evaluate_derived_shadow_missing_flags`: 12.16 ms
- `evaluate_derived_shadow_unauthorized`: 15.70 ms
- `evaluate_derived_shadow_success`: 15.64 ms
