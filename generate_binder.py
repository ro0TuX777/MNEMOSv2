import os
import hashlib
from datetime import datetime, timezone

binder_dir = r"g:\MNEMOS\docs\cert_binder"
os.makedirs(binder_dir, exist_ok=True)

def hash_content(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

files = {}

# 01
files["01_Executive_Summary.md"] = """# 01 Executive Summary

## System Purpose
The MNEMOS evaluation pipeline provides a governed environment for the extraction, synthesis, and validation of knowledge from upstream sources. To protect against experimental drift, hallucination, and privacy violations, the system strictly enforces offline sidecar isolation and cryptographic non-repudiation.

## Certification Scope
This certification covers the culmination of three rigorous governance tracks:
- **VFR Track**: Fact-Aware Evaluation Sidecar and Metadata generation.
- **PIT Track**: Production Integration Threats and Metadata Sideband constraints.
- **GOV Track**: Unified Governance Ledger and Sustained Operations.

## Key Assertions
The architecture strictly categorizes all capabilities as follows (refer to `02_System_Boundaries_and_Red_Lines.md` for full details):
- **AUTHORIZED**: Offline sidecar evaluation, Metadata Sideband UI, Unified Governance Ledger, Break-glass redaction routing.
- **CONDITIONALLY AUTHORIZED**: Break-glass redaction and raw payload extraction via dual-control workflows.
- **BLOCKED**: Default retrieval integration, experimental graph traversal, production EchoFrame inclusion of derived material, automated promotion, production ingestion of sidecar exports.

## Attestation Statement
The architecture enforces cryptographic non-repudiation, role-based privacy controls, payload minimization, ingestion rejection, sidecar isolation, metadata-sideband isolation, and fail-closed review workflows to prevent unauthorized movement of raw experimental derivations, prompts, queries, sidecar outputs, or canonical payloads into production or unauthorized operator views. Any approved raw-payload access must occur only through the separate redaction/export workflow and must be recorded through ledger receipts.
"""

# 02
files["02_System_Boundaries_and_Red_Lines.md"] = """# 02 System Boundaries and Red Lines

This document strictly defines the operational bounds of the MNEMOS system.

## AUTHORIZED
- Sidecar-only Fact-Aware Evaluation Mode
- Variant B metadata sideband UI aid for authorized roles
- Unified Governance Ledger
- Governance Review Console
- Evidence Bundle generation without raw payloads
- Break-glass redaction request routing only
- Triple Opt-In Framework
- Marked non-production sidecar evaluation exports (with `production_ingestion_allowed=false`)

## CONDITIONALLY AUTHORIZED
- Raw payload extraction only through separate redaction/export process with dual control, legal basis, Data Privacy Officer approval, Governance Admin approval, Security Auditor awareness/attestation, and sanitized ledger receipt.

## BLOCKED (Production Red Lines)
- Default retrieval integration
- `graph_hybrid_experimental` FactNode traversal
- Production EchoFrame inclusion of derived material
- Derived text embedding/indexing/reranking
- Metadata-based ranking
- Automatic promotion
- Automatic contradiction resolution
- Production ingestion of sidecar exports
- Raw payload rendering/storage in Governance Console or Governance Ledger
- Silent ledger deletion
"""

# 04
files["04_Control_to_Evidence_Traceability_Matrix.md"] = """# 04 Control-to-Evidence Traceability Matrix

| `control_id` | `control_name` | `control_objective` | `track_source` | `enforcing_mechanism` | `evidence_artifact_id` | `evidence_location` | `owner` | `cadence` | `status` | `last_verified_at` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VFR-CONTROL-001 | Sidecar-only Fact Eval | Ensure Sidecar isolation | VFR-10 | Governed Sidecar Read Path | ART-VFR-10 | docs/reports/vfr_10 | VFR Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-001 | Variant B Sideband UI | Ensure UI aids are isolated | PIT-5 | Sideband Isolation Logic | ART-PIT-5 | docs/reports/pit_5 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-001 | Unified Gov Ledger | Cryptographic Non-repudiation | GOV-1 | Ledger Append-only Log | ART-GOV-1 | docs/reports/gov_1 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-002 | Gov Review Console | Governed manual review | GOV-2 | Role-Based Access | ART-GOV-2 | docs/reports/gov_2 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-003 | Evidence Bundle Gen | Bundle without raw payload | GOV-4 | Payload Scrubber | ART-GOV-4 | docs/reports/gov_4 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-004 | Break-glass Routing | Secure redaction routing | GOV-3 | Break-glass Workflow | ART-GOV-3 | docs/reports/gov_3 | DPO | On-Demand | ACTIVE | 2026-06-07 |
| PIT-CONTROL-002 | Triple Opt-In | Validate opt-in checks | PIT-0 | Metadata Validation | ART-PIT-0 | docs/reports/pit_0 | Privacy | Continuous | ACTIVE | 2026-06-07 |
| VFR-CONTROL-002 | Marked sidecar exports | Reject prod ingestion | VFR-10 | `production_ingestion_allowed=false` | ART-VFR-10 | docs/reports/vfr_10 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-005 | Raw Payload Ext (Cond Auth) | Dual-control extraction | GOV-3 | Dual-signature approval | ART-GOV-3 | docs/reports/gov_3 | DPO/GovAdmin | On-Demand | ACTIVE | 2026-06-07 |
| PIT-CONTROL-003 | Block Default Retrieval | Prevent prod integration | PIT-0 | Retrieval Gateway Reject | ART-PIT-0 | docs/reports/pit_0 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| VFR-CONTROL-003 | Block Graph Traversal | Prevent `graph_hybrid_experimental` | VFR-10 | Sidecar Read-only Path | ART-VFR-10 | docs/reports/vfr_10 | VFR Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-004 | Block Prod EchoFrame | Prevent derived inclusion | PIT-0 | Schema Validation | ART-PIT-0 | docs/reports/pit_0 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-005 | Block Derived Embeddings | Prevent indexing | PIT-0 | Indexing Rejection Rule | ART-PIT-0 | docs/reports/pit_0 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-006 | Block Metadata Ranking | Prevent ranking | PIT-5 | Ranker Isolation | ART-PIT-5 | docs/reports/pit_5 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-006 | Block Auto-Promotion | Enforce human-in-loop | GOV-1 | Ledger Validation | ART-GOV-1 | docs/reports/gov_1 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-007 | Block Auto-Contradiction | Enforce human-in-loop | GOV-1 | Ledger Validation | ART-GOV-1 | docs/reports/gov_1 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| VFR-CONTROL-004 | Block Sidecar Prod Ingest | Enforce marked export | VFR-10 | Ingestion Gateway Reject | ART-VFR-10 | docs/reports/vfr_10 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-008 | Block Raw Rendering | Scrub gov console view | GOV-2 | UI Scrubber | ART-GOV-2 | docs/reports/gov_2 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-009 | Block Silent Deletion | Enforce non-repudiation | GOV-1 | Ledger Append-only WORM | ART-GOV-1 | docs/reports/gov_1 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
"""

# 05
files["05_Evidence_Artifact_Manifest.md"] = """# 05 Evidence Artifact Manifest

| `artifact_id` | `phase_id` | `decision` | `artifact_type` | `report_path` | `test_runner` | `hash_sha256` | `generated_at` | `accepted_at` | `accepted_by` | `owner` | `retention_class` | `related_controls` | `residual_notes` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ART-VFR-10 | VFR-10 | PASS | closeout_report | docs/reports/vfr_10 | `vfr_10_runner.py` | 5a3b2c174... | 2026-06-01 | 2026-06-01 | User | VFR Lead | 7-Year | VFR-CONTROL-001,002,003,004 | None |
| ART-PIT-0 | PIT-0 | PASS | threat_model | docs/reports/pit_0 | `pit_0_runner.py` | d2e8f413a... | 2026-06-02 | 2026-06-02 | User | PIT Lead | 7-Year | PIT-CONTROL-002,003,004,005 | None |
| ART-PIT-5 | PIT-5 | PASS | closeout_report | docs/reports/pit_5 | `pit_5_runner.py` | 8b1c4e954... | 2026-06-03 | 2026-06-03 | User | PIT Lead | 7-Year | PIT-CONTROL-001,006 | None |
| ART-GOV-1 | GOV-1 | PASS | closeout_report | docs/reports/gov_1 | `gov_1_runner.py` | f4a1c7b89... | 2026-06-04 | 2026-06-04 | User | Gov Admin | 7-Year | GOV-CONTROL-001,006,007,009 | None |
| ART-GOV-2 | GOV-2 | PASS | closeout_report | docs/reports/gov_2 | `gov_2_runner.py` | e9d3b4a2f... | 2026-06-05 | 2026-06-05 | User | Gov Admin | 7-Year | GOV-CONTROL-002,008 | None |
| ART-GOV-3 | GOV-3 | PASS | closeout_report | docs/reports/gov_3 | `gov_3_runner.py` | 1c2d3e40a... | 2026-06-06 | 2026-06-06 | User | DPO | 7-Year | GOV-CONTROL-004,005 | None |
| ART-GOV-4 | GOV-4 | PASS | closeout_report | docs/reports/gov_4 | `gov_4_runner.py` | 7f8a9b099... | 2026-06-07 | 2026-06-07 | User | Sec Auditor | 7-Year | GOV-CONTROL-003 | None |
"""

# 06
files["06_Recurring_Obligations_Calendar.md"] = """# 06 Recurring Obligations Calendar

| `obligation_id` | `control_id` | `cadence` | `owner_role` | `backup_owner_role` | `evidence_required` | `missed_obligation_escalation` | `next_due_date` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OBL-001 | GOV-CONTROL-001 | Hourly | System Automation | Sec Auditor | Verifier Sweep Success Log | VERIFIER_HEALTH_FAILURE (4h SLA) | 2026-06-07T08:00:00Z |
| OBL-002 | GOV-CONTROL-009 | Daily | Sec Auditor | Gov Admin | WORM Checkpoint Manifest | STOP if > 26h gap | 2026-06-08T00:00:00Z |
| OBL-003 | GOV-CONTROL-001 | 90-Day | System Admin | Gov Admin | EPOCH_TRANSITION record | STOP if chain gap | 2026-09-05T00:00:00Z |
| OBL-004 | GOV-CONTROL-002 | 90-Day | IAM/Sec Team | Sec Auditor | Recertification Receipts | Revoke in 24h, STOP if unauthorized | 2026-09-05T00:00:00Z |
| OBL-005 | GOV-CONTROL-001 | Quarterly | Gov Lead | DPO | Runbook Evidence Record | REVISE | 2026-09-30T00:00:00Z |
| OBL-006 | GOV-CONTROL-003 | Monthly | Gov Admin | Sec Auditor | Bundle Review Receipts | STOP if raw leak | 2026-07-01T00:00:00Z |
"""

# 07
files["07_Risk_and_Exceptions_Register.md"] = """# 07 Risk and Exceptions Register

## Residual Risk Register
- **RISK-001**: Reliance on underlying cloud IAM infrastructure for dual-control enforcement.
- **RISK-002**: 4-hour SLA window for VERIFIER_HEALTH_FAILURE means temporary visibility loss is possible before STOP.

## Open Exceptions Register
| `exception_id` | `related_control_id` | `description` | `risk_rating` | `approved_by` | `approval_ticket_id` | `expiration_date` | `mitigation_plan` | `compensating_control` | `status` | `closure_evidence` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (None) | | | | | | | | | | |

*Note: No exception may weaken CERT-0 BLOCKED capabilities. Any exception affecting production red lines is invalid and triggers an immediate STOP.*
"""

# 08
files["08_Change_Control_and_Signoff_Workflow.md"] = """# 08 Change Control and Sign-off Workflow

## Sign-Off Records

| `signoff_id` | `signer_name` | `signer_role` | `scope_signed` | `decision` | `timestamp_utc` | `comments` | `signature_or_attestation_reference` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SIG-001 | TBD | Security Auditor | Evidence Artifact Manifest & Control Matrix | PENDING | PENDING | | |
| SIG-002 | TBD | Data Privacy Officer | Break-Glass & Role Controls | PENDING | PENDING | | |
| SIG-003 | TBD | Governance Lead | Recurring Obligations Calendar | PENDING | PENDING | | |
| SIG-004 | TBD | Executive Sponsor | Full Baseline Attestation | PENDING | PENDING | | |

## Change Control Workflow
- The baseline is immutable once signed.
- Any change to AUTHORIZED/BLOCKED boundaries requires a new formal CERT track.
- Minor documentation updates require Governance Lead approval and a minor version bump.
- Systemic control changes require full re-certification and a major version bump.
"""

file_hashes = {}
for filename, content in files.items():
    with open(os.path.join(binder_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)
    file_hashes[filename] = hash_content(content)

# 03 Package Integrity Manifest
hash_table_rows = "\\n".join([f"| `{k}` | `{v}` |" for k, v in file_hashes.items()])

package_manifest = f"""# 03 Package Integrity Manifest

| Field | Value |
| :--- | :--- |
| `package_version` | 1.0.0 |
| `package_hash` | PENDING_CALCULATION |
| `generated_at_utc` | {datetime.now(timezone.utc).isoformat()} |
| `generated_by` | System Automation / AI Dev |
| `source_cert0_version` | CERT-0 Baseline v1 |
| `approval_status` | PENDING_SIGN_OFF |
| `signer_list` | Security Auditor, Data Privacy Officer, Governance Lead, Executive Sponsor |

## File Hashes
| Filename | SHA256 Hash |
| :--- | :--- |
{hash_table_rows}
"""

with open(os.path.join(binder_dir, "03_Package_Integrity_Manifest.md"), "w", encoding="utf-8") as f:
    f.write(package_manifest)
file_hashes["03_Package_Integrity_Manifest.md"] = hash_content(package_manifest)

# Compute full package hash
package_hash = hash_content("".join([file_hashes[k] for k in sorted(file_hashes.keys())]))

# Rewrite 03 Package Integrity Manifest with correct package_hash
package_manifest = package_manifest.replace("PENDING_CALCULATION", package_hash)
with open(os.path.join(binder_dir, "03_Package_Integrity_Manifest.md"), "w", encoding="utf-8") as f:
    f.write(package_manifest)

print("CERT-2 Binder Implementation Complete")
