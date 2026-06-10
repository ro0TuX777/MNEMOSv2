# PIT-0 Production Integration Threat Model and Boundary Review Report

**Status**: PRODUCTION THREAT MODEL CERTIFIED
**Date**: 2026-06-07

## 1. Executive Summary
This report formalizes the absolute, safe architectural boundary for any future integration of `VALIDATED` derived facts into production retrieval paths. No integration code is authorized by this report. It serves strictly as the governance framework defining the "rules of engagement" for production exposure.

## 2. Integration Architecture & Isolation Rules
- **Lane Isolation**: Derived facts must be stored in a dedicated `DerivedFactNode` graph. They are mathematically prohibited from merging with the primary raw `Engram` index.
- **Traversal Limits**: Automatic `graph_hybrid` traversal merging raw and derived contexts is explicitly banned.
- **Mechanically Enforced Lane Isolation**: Isolation is enforced at the query and data-model levels, preventing implicit bleeding of derived material into standard search results.

## 3. Double Opt-In Access Requirements
Derived facts are banned from default retrieval (`GET /query`). Access requires a strict double opt-in:
1. **Application Context**: The calling application route must be explicitly whitelisted via configuration.
2. **User Context**: The specific API request must contain explicit intent (e.g., `include_derived_facts=true`).

## 4. Query-Time Ledger Verification and Default-Deny State
The retrieval engine must perform a real-time cryptographic lookup against the Governance Ledger before returning any `DerivedFactNode`.
- **Allowed State**: The *only* allowed positive state is `CERTIFIED_FOR_GOVERNED_EVALUATION_OPERATION`.
- **Denied States**: The query engine enforces a hard database-level filter dropping all other states by default, specifically including: `UNKNOWN`, `MISSING`, `DOWNGRADED`, `REJECTED`, `REVOKED`, `SUPERSEDED`, `CONFLICTED`, `STALE`, `EXPIRED`, and `UNVERIFIED`.

## 5. UI Presentation and EchoFrame Rendering
To prevent user confusion, any presentation layer (including EchoFrame) rendering derived facts must enforce:
- **Explicit Markers**: Clear prefixing (e.g., `[MNEMOS-DERIVED]`) and distinct visual styling.
- **Authority Matrix**: An explicitly rendered widget detailing Confidence Level, Source Diversity, and explicitly highlighting any Evidence Gaps where context is redacted or missing.
- **Provenance Linking**: Mandatory citations linking to the raw source Engrams and the Verifier receipt.

## 6. The Kill Switch
A P0 global configuration flag (`MNEMOS_DERIVED_ENABLED`) serves as a **no-redeployment kill switch effective no later than the next request evaluation**. When toggled to `false`, it instantly severs the `DerivedFactNode` lane across all APIs, reverting the system to raw-only mode.

## 7. Telemetry and Leakage Guards
The core retrieval engine must emit a `query.default_retrieval.derived_fact_count` metric. This must strictly equal `0`. Any value `>0` instantly fires a P0 `SEV-STOP` alert and automatically engages the Kill Switch.

## 8. Release Gate Enforcement
Any code proposing integration must be classified as Tier 1 and successfully pass the OPS-3 framework:
- Blocked-capability scans (import graph & negative integration tests).
- EchoFrame prompt absence tests.
- Evidence binder hash checks.
- Unanimous CODEOWNERS approval from Security, Governance, and Operations Leads.

## 9. Final Recommendation
**PIT_0_PRODUCTION_INTEGRATION_THREAT_MODEL_PASS**
