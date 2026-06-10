# OPS-3 Release Governance and Certification Drift Guard Report

**Status**: RELEASE GOVERNANCE AND DRIFT GUARD CERTIFIED
**Date**: 2026-06-07

## 1. Release Classification Matrix
All changes to the MNEMOS repository are mechanically classified into three tiers. Any unclassified operational or configuration drift defaults to Tier 1.

## 2. Tier 1 / Tier 2 / Tier 3 Examples
- **Tier 1 (Certification Impact)**: Code touching sidecar logic, IAM roles, sideband metadata, ledger schema, dependencies, container images, CI/CD workflows, secrets, WORM configuration, alert routing, or redaction logic.
- **Tier 2 (Operational/Config)**: Changes to non-red-line runbooks or generic dashboard layout elements (non-role-scoped).
- **Tier 3 (Minor/Non-Operational)**: Typo fixes, internal comments, non-normative documentation.

## 3. CI/CD Gate Inventory
Every PR must pass the following mechanical gates before merge:
- `blocked_capability_scan`
- `certification_impact_assessment_checklist`
- `package_boundary_diff` (protects `02_System_Boundaries_and_Red_Lines.md`)
- `evidence_binder_hash_check` (protects `cert_binder` hashes vs manifest)
- `CODEOWNERS_enforcement`

## 4. Blocked-Capability Scan Inventory
The `blocked_capability_scan` executes:
- Static grep/AST scan.
- Import graph scan (detecting invalid bridges).
- Route/API surface scan.
- Integration negative tests.
- EchoFrame prompt absence test.
- Canonical retrieval immutability test.
- Zero ranking delta test.
- Ledger forbidden-field schema test.
- Sidecar export ingestion rejection test.

*Trigger*: STOP if any blocked path is reachable.

## 5. Required Reviewer Matrix
- **Tier 1**: Security Auditor, Governance Lead, Operations Lead.
- **Tier 2**: Operations Lead, Affected Control Owner.
- **Tier 3**: Standard Reviewer (assuming CI confirms no T1/T2 touchpoints).

## 6. CODEOWNERS Enforcement Summary
- Mechanical enforcement blocks self-approval.
- Merges are blocked until all required matrix reviewers approve and all CI/CD scans are perfectly green.

## 7. Release Evidence Record Schema
Every release emits an auditable record containing:
- `release_id`, `commit_hash`, `branch`, `release_tier`
- `changed_files`, `affected_controls`
- `certification_impact_assessment`
- `ci_results`, `blocked_capability_scan_result`
- `package_boundary_diff_result`, `binder_hash_check_result`
- `approvers`, `approval_timestamps`
- `decision` (approve / reject / stop)
- `linked_ledger_event_ids`, `rollback_plan`

## 8. IAM/Role Change Review Workflow
- Modifications require dual-control approval (IAM Team Lead + Security Auditor).
- Must emit an `OPERATOR_RECERTIFICATION_REQUIRED` state if permissions expand.

## 9. OPERATOR_RECERTIFICATION_REQUIRED Workflow
- Triggered automatically on role permission expansion.
- Newly expanded permissions remain suspended for all holding users.
- Role-specific retraining is required if the expansion touches redaction, ledger, sideband, release, or IAM controls.
- Uncertified expanded access is revoked within 24 hours if recertification does not pass.

## 10. Ledger Schema Change Workflow
- Schema for WORM and events (`VERIFIER_HEALTH_FAILURE`, `OPERATOR_CERTIFIED`, etc.) is strictly locked.
- Modifications require a full `CERT-1` level re-baseline to prove cryptographic non-repudiation is preserved.

## 11. Metadata Registry Change Workflow
- Additions to the sideband registry require DPO / Privacy Auditor review to prevent inadvertent raw payload or PII leakage via inference.

## 12. Emergency Rollback Workflow
- **Before Rollback**: Emit `RELEASE_CERTIFICATION_VIOLATION`. Capture affected commit, deployment ID, and failing alert IDs. Create a WORM forensic checkpoint if ledger paths are affected.
- **Rollback**: Revert to the last known CERT-4 compliant commit.
- **Post-Rollback**: Freeze affected subsystem. Run verifier, blocked-capability scan, prompt absence, and canonical immutability checks. Unfreeze requires Security Auditor + Operations Lead approval.

## 13. Certification Invalidation Release Triggers
A release immediately invalidates certification if it:
- Bypasses the Blocked-Capability Scan.
- Edits the Package Boundaries (`02_System_Boundaries_and_Red_Lines.md`) without Executive Sponsor sign-off.
- Modifies Evidence Binder files without updating the manifest.
- Rolls back a deployment in a way that deletes/hides evidence of the violation.

## 14. Final Recommendation
**OPS_3_RELEASE_GOVERNANCE_DRIFT_GUARD_PASS**
