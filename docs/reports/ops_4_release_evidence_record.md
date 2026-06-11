# OPS-4 Release Evidence Record: Production-Adjacent Derived Fact Integration

**Date**: 2026-06-07
**Status**: `GO_FOR_OPS_4`
**Phase**: PIT-7 (Production-Adjacent Evaluation Mode Implementation)

## Executive Summary
This document serves as the formal evidence record for the OPS-4 release gate. It validates that the Memory Over Maps (phase 4) derived fact lane has been securely integrated into the `MNEMOS` service boundary without exposing unverified derived facts to production retrieval flows. The implementation relies on a strictly separated shadow evaluation lane, double opt-in gates, and robust kill-switch mechanisms.

## Boundary Validations

### 1. Production API Integrity
- **Invariant:** `POST /api/v1/query` and `POST /v1/mnemos/search` must **never** return derived facts or accept evaluation mode flags.
- **Evidence:** 
  - Test `test_production_route_ignores_and_returns_zero_derived` passes: `derived_results` is empty.
  - Test `test_production_route_rejects_eval_mode` passes: Returns HTTP 400 when `evaluation_mode=true` is requested on production routes.
  - Assertions are embedded directly into `app.py` returning `SEV-STOP` if the internal default retrieval leaks derived facts.

### 2. Double Opt-In Evaluation Lane
- **Invariant:** The shadow lane is only accessible via `POST /api/v1/evaluate_derived_shadow` when both `evaluation_mode=true` and `include_derived_facts=true` are explicitly passed by whitelisted clients.
- **Evidence:**
  - Test `test_eval_route_missing_flags` passes: Returns HTTP 400 when flags are missing.
  - Test `test_eval_route_not_whitelisted` passes: Returns HTTP 403 when `X-Client-Id` is not in `MNEMOS_DERIVED_WHITELIST`.
  - Test `test_evaluate_derived_shadow_success` passes: Correctly generates the shadow packet when fully authorized.

### 3. Kill-Switch Enforcement
- **Invariant:** If `MNEMOS_DERIVED_ENABLED` is `false`, the shadow lane must immediately reject all requests.
- **Evidence:** 
  - Test `test_eval_route_kill_switch` passes: Returns HTTP 503 (`derived_lane_disabled`) under kill-switch.

### 4. Prompt Builder Isolation
- **Invariant:** The production `EchoFrame` prompt builder must remain physically separate from derived fact rendering and fail hard if exposed to derived payloads.
- **Evidence:** 
  - Test `test_production_prompt_builder_guard` passes: `PromptBuilder.build_prompt` raises `SevStop` when `derived_results` are detected.
  - Rendering is entirely localized to `mnemos/evaluation/derived_evaluation_renderer.py` for the shadow block.

### 5. Telemetry and Observability
- **Invariant:** Execution counts, leak counts, and shadow rendering metrics must be tracked via the `_mom_stats` runtime telemetry.
- **Evidence:** 
  - `service/app.py` exposes `/v1/mnemos/stats` with `derived_lane` sub-dictionary tracking:
    - `query.default_retrieval.derived_fact_count`
    - `echoframe.production_prompt.derived_count`
    - `derived_lane.execution_count`
    - `derived_lane.denied_count`
    - `derived_lane.kill_switch_count`
    - `evaluate_derived_shadow.request_count`
    - `evaluate_derived_shadow.denied_count`
    - `evaluate_derived_shadow.rendered_derived_fact_count`

## Conclusion
The PIT-7 implementation meets all constraints outlined in the design gate. Production isolation is guaranteed by strict API routing, separate renderers, fail-closed access controls, and hard assertions on derived context counts. The system is ready for the OPS-4 release.
