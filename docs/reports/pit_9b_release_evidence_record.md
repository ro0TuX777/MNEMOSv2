# PIT-9B Release Evidence Record: Tail Latency Investigation

**Date**: 2026-06-08
**Status**: `TAIL_LATENCY_RESOLVED`
**Phase**: PIT-9B

## Executive Summary
This document serves as the formal evidence record for PIT-9B. It validates that the ~7.7s latency spike observed in PIT-9 was entirely a local cold-start initialization cost (specifically the `sentence_transformers` CrossEncoder reranker loading during the first `default_search_ms` stage). Once cold-start is excluded, the p95 latency across the derived evaluation lane averages ~4.04ms under load.

## Latency Investigation Diagnostics

- **Warm-Up Request Latency (First Hit)**: 8251.69 ms
- **Suspected Source**: default_search_ms (cold-start initialization)
- **Cold Start Excluded from Burst**: True

### Server-Side Stage Latencies (Averages across 15 iterations)
- `auth_whitelist_check_ms`: 0.00 ms (In-memory dict lookup)
- `default_search_ms`: 0.69 ms (post-initialization)
- `search_derived_ms`: 0.28 ms
- `governance_ledger_check_ms`: 0.00 ms
- `shadow_serializer_ms`: 0.08 ms
- `evaluation_renderer_ms`: 0.40 ms
- `telemetry_stats_update_ms`: 0.00 ms
- `total_request_ms`: 1.48 ms

*Note: The remaining ms up to the total client observed (2.34ms) is overhead from the local Flask dev server and network stack processing.*

### Burst Latency Percentiles (Total Request, Client Observed)
- **p50**: 2.34 ms
- **p90**: 2.87 ms
- **p95**: 4.04 ms
- **p99**: 4.04 ms

## System Performance & Boundary Integrity

### 1. Boundary Verification
- **Production Query Default Retrieval Leakage**: 0 derived facts (PASS)
- **Production Query Evaluation Mode Block**: 400 Bad Request (PASS)
- **Shadow Route Unauthorized Client Block**: 403 Forbidden (PASS)
- **Kill-Switch Readiness**: Inherited and confirmed.

### 2. Telemetry Completeness
Telemetry confirmed `evaluate_derived_shadow` execution counts correctly incremented while default retrieval leakage counts (`query.default_retrieval.derived_fact_count`) remained at **0**.

## Conclusion
The ~7.7s spike is definitively explained as an engine initialization artifact. Subsequent repeated execution times are fully acceptable and stable (p99 < 5ms locally). The performance and isolation bounds of the evaluation path pass all required gates.

**Prohibited Actions Maintained:**
- No default retrieval enablement
- No production EchoFrame outside evaluation_mode
- No Candidate Envelope mixing
- No raw Engram / derived fact fusion
- No automatic promotion
- No automatic conflict resolution
- No SchemaNode extraction
- No source/fact/lifecycle mutation
