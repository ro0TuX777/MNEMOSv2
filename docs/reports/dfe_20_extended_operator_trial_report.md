# DFE-20 Extended Operator Trial Execution Report

## Overview
Phase DFE-20 executed the Extended Operator Trial using the "Option B" Feature-Flagged API approach. The objective was to integrate Derived Facts directly into the primary `/api/v1/query` endpoint for live operator evaluation without compromising the isolation, integrity, or structure of the default production retrieval pipeline.

## Implementation Rules Enforced
1. **Delegated Runtime Logic**: The retrieval orchestration was moved from `app.py` directly into a specialized, tightly controlled `search_derived_trial` method within `api.py`.
2. **Feature Flag Prerequisite**: The endpoint strictly requires the `enable_derived_facts=true` parameter. The legacy `evaluation_mode=true` parameter remains explicitly blocked to prevent backward-compatibility leakage.
3. **Strict Whitelisting**: `enable_derived_facts` is completely blocked unless the `X-Client-Id` header explicitly matches an operator in the `config.derived_whitelist`.
4. **Kill-Switch Guard**: `config.derived_enabled` must be set to True. If toggled off, the feature-flag path returns a `503 Service Unavailable` immediately.
5. **Absolute Quarantine**: Retrieved Derived Facts are serialized exclusively within a dedicated `derived_lane_meta` block. They are entirely blocked from merging into the primary `documents` array, Candidate Envelopes, or EchoFrame production evidence payloads.
6. **Hard Schema Filtering**: The specialized runtime path actively drops any fact candidate that lacks the `MNEMOS_DERIVED_FACT` authority label, a rendered support excerpt, source document metadata, or a valid selection/rescue decision code. 

## Trial Harness Validation Results
The automated validation harness (`tools/dfe_20_extended_trial_harness.py`) ran successfully, verifying:

* **[PASS] Default Retrieval Integrity**: Standard `/api/v1/query` calls without the feature flag generated exactly 0 Derived Facts. The invariant SEV-STOP remained untriggered.
* **[PASS] Legacy Parameter Blocking**: Attempting to query with `evaluation_mode=true` was successfully rejected (`400 Bad Request`).
* **[PASS] Allowlist Enforcement**: Passing the feature flag without being on the `derived_whitelist` resulted in a `403 Forbidden` error.
* **[PASS] Kill-Switch Activation**: Toggling the `derived_enabled` kill switch to False resulted in a rapid `503 Service Unavailable` block.
* **[PASS] Clean Execution & Schema Filtering**: A successful execution for an authorized operator returned an isolated `derived_lane_meta` block containing perfectly structured Derived Facts, automatically dropping any invalid/malformed candidate elements.

## Conclusion
The DFE-20 Extended Operator Trial architecture is successfully deployed and validated. The integration perfectly adheres to the Option B strict-isolation requirements, ensuring that no experimental Derived Facts can leak into the standard retrieval pathways, and all test telemetry is cleanly separated. 
