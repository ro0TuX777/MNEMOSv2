# PIT-10: Production-Adjacent Evaluation Lane Closeout and Limited Pilot Readiness

**Date**: 2026-06-08
**Status**: `PIT_10_PRODUCTION_ADJACENT_EVALUATION_LANE_READY_FOR_LIMITED_PILOT`

## Executive Summary
This document serves as the formal closeout report for the Memory Over Maps (MOM) "Derived Facts" capability development track (PIT-0 through PIT-9B). The system has successfully proven the viability of offline shadow evaluation and the safety of the production-adjacent `/api/v1/evaluate_derived_shadow` endpoint. 

MNEMOS is now approved for a **Limited Controlled Operator Pilot** using this endpoint, while default production use remains strictly blocked.

---

## 1. Completed PIT Phases and Pass Gates

| Phase | Description | Status |
|---|---|---|
| **PIT-0** | Initial Scaffold & Architecture Sync | `PASS` |
| **PIT-1** | Candidate Envelope & Derived Fact Schema | `PASS` |
| **PIT-2** | Shadow Data Lane & Routing Integration | `PASS` |
| **PIT-3** | Shadow Evaluation Serializer | `PASS` |
| **PIT-4** | Local Model Evaluator | `PASS` |
| **PIT-4-B** | Baseline vs Shadow Output Validation | `PASS` |
| **PIT-5** | Operator Review CLI Implementation | `PASS` |
| **PIT-5-B** | Actual Operator Review Run (Offline) | `PASS` |
| **PIT-6** | Production-Adjacent Design Gate | `PASS` |
| **PIT-7** | Production-Adjacent Implementation | `PASS` |
| **PIT-8** | Controlled Live Route Trial | `PASS` |
| **PIT-9** | Controlled Operator Evaluation Route Trial | `PASS` |
| **PIT-9B** | Tail Latency Investigation & Stabilization | `PASS` |

---

## 2. What Is Proven

1. **Safety and Isolation**: Derived facts do not leak into default retrieval (`/api/v1/query`). Telemetry metrics (`query.default_retrieval.derived_fact_count`) remained at 0 throughout all live trials.
2. **Endpoint Security**: The shadow endpoint requires whitelist client headers, double opt-in JSON flags, and kill-switch enablement. Hard denials (400, 403, 503) successfully trigger when violated.
3. **Operator Utility**: In controlled trials, operators preferred shadow packets with derived facts 100% of the time, rating usefulness, trust, label clarity, and traceability highly (5/5). Review burden is manageable.
4. **Latency Stability**: After isolating a single ~7.7s cold-start initialization artifact, p50 and p95 latencies stabilize at `~2-4ms` for shadow packet generation.

---

## 3. What Remains Unproven

1. Long-term human trust at production scale.
2. The performance overhead of the derived lane at extremely high production concurrency.
3. The impact of mixed confidence facts during multi-document conflicts.
4. Schema node extraction accuracy.

---

## 4. Approved Limited-Use Boundary

MNEMOS is approved to enter a Limited Controlled Operator Pilot under the following conditions:

**Allowed:**
- Execution of the `/api/v1/evaluate_derived_shadow` route.
- Whitelisted clients only (`X-Client-Id: eval_dashboard`).
- Double opt-in payloads: `evaluation_mode=true`, `include_derived_facts=true`.
- Environment variable `MNEMOS_DERIVED_ENABLED=true`.
- Rendering of `[MNEMOS-DERIVED]` context tags.
- Display of Authority Matrices, source traceability, and evidence gaps.
- Operator-facing offline and live evaluation workflows.

---

## 5. Prohibited Uses

The following actions and paths remain strictly **BLOCKED** and are subject to SEV-STOP protection:

- **NO** default retrieval use (must return 0 derived facts).
- **NO** derived fact access via `/api/v1/query`.
- **NO** production EchoFrame usage outside `evaluation_mode`.
- **NO** Candidate Envelope mixing.
- **NO** raw Engram / derived fact fusion.
- **NO** automatic promotion to governance ledgers.
- **NO** automatic conflict resolution.
- **NO** SchemaNode extraction.
- **NO** production default enablement.
- **NO** source, fact, or lifecycle mutation.

---

## 6. Required Runtime Configuration & Telemetry

### Required Flags
- `MNEMOS_DERIVED_ENABLED=true` (must be explicitly set, default is false).
- A valid whitelist mapped via config (e.g., `["eval_dashboard"]`).

### Required Telemetry Gates
The system must actively track the following metrics without zero-leakage variance on default retrieval:
- `query.default_retrieval.derived_fact_count`
- `echoframe.production_prompt.derived_count`
- `evaluate_derived_shadow.request_count`
- `evaluate_derived_shadow.denied_count`
- `derived_lane.kill_switch_count`

---

## 7. Kill-Switch, Rollback, and Warm-Up Behavior

### Kill-Switch
If `MNEMOS_DERIVED_ENABLED=false` or is omitted, all requests to the shadow lane return HTTP 503 (`derived_lane_disabled`). This is instantaneous and does not require code rollback.

### Rollback Procedure
1. Set `MNEMOS_DERIVED_ENABLED=false`.
2. Delete the whitelist client header from authorized consumers.
3. Monitor `derived_lane.kill_switch_count` to ensure traffic hits the barrier.

### Latency Warm-Up Requirement
Due to the cross-encoder initialization artifact, operators must be informed that the *first* search request to the MNEMOS cluster (warm-up) will incur an ~8s penalty. Automated pilots should invoke a dummy `/v1/mnemos/search` warm-up query during bootstrap to shield operators from cold-start stalls.

---

## 8. Next Possible Phase After Limited Pilot

Following a successful limited pilot, the next logical milestone is **Limited Controlled Operator Pilot Execution** to collect real-world feedback on usefulness and system stability. Do not begin Candidate Envelope mixing or conflict-resolution design until pilot feedback is formally collected, reviewed, and approved.

**Current Recommended Status:**
`PIT_10_PRODUCTION_ADJACENT_EVALUATION_LANE_READY_FOR_LIMITED_PILOT`

PIT-10 is accepted. The Derived Fact production-adjacent evaluation lane is ready for Limited Controlled Operator Pilot Execution. No broader production integration, Candidate Envelope mixing, conflict-resolution automation, or SchemaNode extraction is authorized by this closeout.
