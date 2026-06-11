# VFR-8 Controlled Operator Pilot Runbook

## Overview
This runbook defines the exact execution constraints and timeline for the Fact-Aware Evaluation Mode (Operator Review Console) Pilot. It governs a closed cohort of 3 named operators over an 8-day testing window.

## Target Cohort
1. **Op-Alpha** (`ROLE_MEMORY_EVALUATOR`)
2. **Op-Bravo** (`ROLE_MEMORY_EVALUATOR`)
3. **Op-Charlie** (`ROLE_MEMORY_EVALUATOR`)

## Authorized Pilot Window
*   **Duration:** 8 Calendar Days.
*   **Day 0:** Preflight & Security Validation (RBAC Assault Testing).
*   **Day 1–7:** Active Operator Usage (Gap Analysis & Stale Data Traversal).
*   **Day 8:** Retention Verification and Audit Closeout.

## Emergency Rollback Triggers
The pilot halts immediately and `VFR_DISABLE_SHADOW_MODE` is activated if:
1.  **Leakage Incident:** Any `.json`, `.md`, or `.pdf` marked as `sidecar_evaluation_export` enters a production store.
2.  **State Mutation:** Operator activity triggers a write event against `RetrievalRouter` logic, `FactNode` candidate payloads, or `PromotionEngine` database indices.
3.  **Source Confusion:** Any operator flags that they cannot discern between a Primary Engram and a Derived Fact, or cites a Derived Fact without its watermark.

## Execution Sequence

### Day 0: Preflight RBAC Assurance
*   Execute simulated brute-force access attempts using an account with `ROLE_ANALYST` against `/invoke` and `/export`.
*   *Expected Result:* 100% rejection, trapped by `SIDECAR_INVOKE_BLOCKED`.

### Days 1–7: Controlled Operator Workload
*   Operators query the console specifically targeting known gap recovery scenarios and stale-data resolution proofs.
*   Every request must explicitly carry `operator_override=True` and `enable_fact_awareness=True`.
*   Engineers perform daily syncs on `EvaluationAuditor` telemetry.

### Day 8: Retention and Audit Assurance
*   Suspend operator access (disable pilot cohort).
*   Execute `tools/purge_sidecar_evaluations.py`.
*   Verify that 100% of generated payload artifacts from Day 1 are permanently deleted from `/tmp/`.
*   Verify that 100% of the associated audit events (`SIDECAR_INVOKED`) remain permanently archived.
