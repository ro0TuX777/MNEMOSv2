# 06 Recurring Obligations Calendar

| `obligation_id` | `control_id` | `cadence` | `owner_role` | `backup_owner_role` | `evidence_required` | `missed_obligation_escalation` | `next_due_date` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OBL-001 | GOV-CONTROL-001 | Hourly | System Automation | Sec Auditor | Verifier Sweep Success Log | VERIFIER_HEALTH_FAILURE (4h SLA) | 2026-06-07T08:00:00Z |
| OBL-002 | GOV-CONTROL-009 | Daily | Sec Auditor | Gov Admin | WORM Checkpoint Manifest | STOP if > 26h gap | 2026-06-08T00:00:00Z |
| OBL-003 | GOV-CONTROL-001 | 90-Day | System Admin | Gov Admin | EPOCH_TRANSITION record | STOP if chain gap | 2026-09-05T00:00:00Z |
| OBL-004 | GOV-CONTROL-002 | 90-Day | IAM/Sec Team | Sec Auditor | Recertification Receipts | Revoke in 24h, STOP if unauthorized | 2026-09-05T00:00:00Z |
| OBL-005 | GOV-CONTROL-001 | Quarterly | Gov Lead | DPO | Runbook Evidence Record | REVISE | 2026-09-30T00:00:00Z |
| OBL-006 | GOV-CONTROL-003 | Monthly | Gov Admin | Sec Auditor | Bundle Review Receipts | STOP if raw leak | 2026-07-01T00:00:00Z |
