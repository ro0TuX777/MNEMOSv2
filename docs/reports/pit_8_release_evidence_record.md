# PIT-8 Release Evidence Record: Controlled Evaluation Route Live Trial

**Date**: 2026-06-08
**Status**: `GO_FOR_OPERATOR_REVIEW`
**Phase**: PIT-8 (Controlled Evaluation Route Live Trial)

## Executive Summary
This document serves as the formal evidence record for PIT-8. It validates that the Memory Over Maps derived fact lane integration boundary, finalized in PIT-7, holds under live HTTP request conditions against an active MNEMOS instance. The trial ensures that production queries never leak derived facts, evaluation boundaries are strictly enforced, and the kill switch fully disables the evaluation lane when toggled.

## Live Trial Validations

### 1. Production Request Hardening
- **POST `/api/v1/query` with `evaluation_mode=true`**
  - **Result:** HTTP 400 Bad Request.
  - **Status:** PASS
- **POST `/api/v1/query` (Default Retrieval)**
  - **Result:** `derived_results` count is exactly 0.
  - **Status:** PASS
- **POST `/v1/mnemos/search` (Internal Default Retrieval)**
  - **Result:** `derived_results` count is exactly 0.
  - **Status:** PASS

### 2. Shadow Evaluation Lane Access
- **POST `/api/v1/evaluate_derived_shadow` missing `include_derived_facts`**
  - **Result:** HTTP 400 Bad Request (`missing_required_eval_flags`).
  - **Status:** PASS
- **POST `/api/v1/evaluate_derived_shadow` with Unauthorized Client ID**
  - **Result:** HTTP 403 Forbidden (`derived_fact_client_not_authorized`).
  - **Status:** PASS

### 3. Kill-Switch Verification
- **MNEMOS_DERIVED_ENABLED=false**
  - **Result:** All requests to `/api/v1/evaluate_derived_shadow` immediately return HTTP 503 (`derived_lane_disabled`).
  - **Status:** PASS
  - **Kill Switch Live Mode**: `manual_restart_verified`

### 4. Telemetry Verification
Telemetry deltas captured via `/v1/mnemos/stats` successfully incremented the required shadow request metrics while proving that `query.default_retrieval.derived_fact_count` remained exactly zero throughout the live trial execution.

### 5. Latency & Operator Usefulness
The harness generated `pit_8_live_trial_metrics.json` tracking endpoint latency. Qualitative operator usefulness is confirmed through the static `pit_8_live_trial_report.md` output containing the raw `[MNEMOS-DERIVED]` context blocks. No mutation occurred during testing.

## Conclusion
The live trial confirms that the MNEMOS service safely isolates derived facts and correctly processes the double opt-in shadow lane under live HTTP traffic. The system is structurally sound for controlled operator review.

**Prohibited Actions Maintained:**
- No default retrieval enablement
- No production EchoFrame outside evaluation_mode
- No Candidate Envelope mixing
- No raw Engram / derived fact fusion
- No automatic promotion
- No automatic conflict resolution
- No SchemaNode extraction
- No source/fact/lifecycle mutation
