# PIT-9B Latency Investigation Report

## 1. Diagnostics
- Warm-up Request Latency: 8251.69 ms
- Suspected Source: **default_search_ms (cold-start initialization)**
- Cold Start Excluded from Burst: True

## 2. Server-Side Stage Latencies (Average)
- `auth_whitelist_check_ms`: 0.00 ms
- `default_search_ms`: 2.30 ms
- `search_derived_ms`: 0.02 ms
- `governance_ledger_check_ms`: 0.00 ms
- `shadow_serializer_ms`: 0.06 ms
- `evaluation_renderer_ms`: 0.02 ms
- `telemetry_stats_update_ms`: 0.00 ms
- `response_serialization_ms`: 0.00 ms
- `total_request_ms`: 2.45 ms

## 3. Burst Latency Percentiles
- p50: 2.34 ms
- p90: 2.62 ms
- p95: 4.04 ms
- p99: 4.04 ms

## 4. Boundary Enforcement
- `default_retrieval_leakage_count_zero`: **PASS**
- `production_prompt_leakage_count_zero`: **PASS**
