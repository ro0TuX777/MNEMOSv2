# OPS-1 Maintenance Dashboard and Alert Ownership Report

**Status**: DASHBOARD AND ALERT OWNERSHIP CERTIFIED
**Date**: 2026-06-07

## 1. Dashboard Widget Inventory
The "Governance Ledger Health & Continuity Dashboard" is scoped strictly to:
- **Ledger Health**: Hourly verifier pass/fail rates, time since last successful WORM checkpoint, and current WORM sequence gap.
- **Role Continuity**: Privileged role count, days until next scheduled recertification, and uncertified roles past SLA.
- **Privacy Workflows**: Break-glass redaction requests, dual-control approval times, and payload extraction counts.
- **Operational Burdens**: Count of active/overdue maintenance tasks and false-positive rates.

*Freshness Rule*: Every widget displays `last_updated_utc` and `data_source_status`. A stale feed with a "green" state is an immediate STOP condition.

## 2. Alert Class Inventory & Severity Mapping
- **LEDGER_INTEGRITY_FAILURE**: P0 (SEV-STOP)
- **VERIFIER_HEALTH_FAILURE**: P1 (SEV-MAJOR)
- **WORM_CHECKPOINT_GAP**: P0 (SEV-STOP)
- **ROLE_RECERT_OVERDUE**: P1 (SEV-MAJOR)
- **UNAUTHORIZED_REDACTION**: P0 (SEV-STOP)
- **BLOCKED_CAPABILITY_SCAN**: P0 (SEV-STOP)
- **RAW_PAYLOAD_IN_BUNDLE**: P0 (SEV-STOP)
- **ROUTINE_OBLIGATION_DUE**: P2 (SEV-MINOR)

## 3. Owner and Backup Owner Mapping
| Alert Class | Primary Owner | Backup Owner |
| :--- | :--- | :--- |
| Cryptographic Integrity (Ledger/WORM) | Security Auditor | Operations Lead |
| Release Gates & Blocked Scans | Operations Lead | Governance Lead |
| Privacy & Redaction | DPO | Privacy Lead |
| Role & Identity SLA | IAM Team | Operations Lead |
| Verifier Health | Security Auditor | System Administrator |

## 4. Escalation Routing Map
- **P2 (Minor)**: Work queue. Escalates to P1 if unacknowledged >48h.
- **P1 (Major)**: PagerDuty/Slack to Primary & Backup. Escalates to P0 if unresolved >4h.
- **P0 (Stop)**: Broadcast to Operations Lead, Security Auditor, DPO, and Exec Sponsor.

## 5. SLA / SLO Table
| Metric | Threshold | Consequence of Breach |
| :--- | :--- | :--- |
| Verifier Sweep | 4 hours | P1 Alert -> P0 if unmitigated |
| WORM Checkpoint | 26 hours | P0 Alert -> Certification Invalidated |
| Role Revocation | 24 hours post-expiry | P0 Alert -> Certification Invalidated |
| Alert Ack (P0) | < 15 minutes | Immediate Backup Escalation |
| Alert Ack (P1) | < 1 hour | Immediate Backup Escalation |

## 6. False-Positive Tuning Workflow
- False-positive classifications are **append-only** and emit an Operational Evidence Record.
- The original alert is never deleted, hidden, pre-muted, or suppressed.
- Rule tuning requires explicit dual-control approval (Operations Lead + Governance Lead).

## 7. Dashboard Access-Control Matrix
| Role Profile | Access Scope |
| :--- | :--- |
| `read_only_governance_viewer` | Aggregate health only; no actor identity. |
| `security_auditor` | Verifier, WORM, integrity, release-gate alerts. |
| `privacy_auditor` / `DPO` | Redaction, break-glass, payload-minimization alerts. |
| `operations_lead` | Full operational alert routing and ownership view. |
| `executive_sponsor` | Summary posture and P0/P1 status only. |
| `standard_analyst` | No Access. |

## 8. Data Minimization Validation
**VALIDATED**: Raw query text, prompt text, derived fact text, sidecar output text, and canonical retrieval payloads are mathematically prohibited from dashboard rendering. Widgets use aggregate counts, hashes, and receipt IDs only.

## 9. Alert Evidence Record Schema
All dashboard interactions must emit an append-only event (`ALERT_TRIGGERED`, `ALERT_ACKNOWLEDGED`, `ALERT_RESOLVED`, `ALERT_FALSE_POSITIVE_CLASSIFIED`, etc.) containing:
- `alert_id`
- `severity`
- `trigger_time_utc`
- `acknowledged_by`
- `resolution_time_utc`
- `root_cause_analysis`
- `mitigation_steps`
- `false_positive_flag` (True/False)

## 10. Certification Invalidation Alert List (P0 Direct)
These alerts trigger immediate system freeze and certification invalidation:
1. `LEDGER_INTEGRITY_FAILURE` (Verified hash mismatch uncorrected > 24h)
2. `WORM_CHECKPOINT_GAP` (>26h)
3. `BLOCKED_CAPABILITY_ENABLED` (e.g., Default retrieval or graph traversal attempt)
4. `RAW_PAYLOAD_IN_BUNDLE`
5. `ROLE_RECERT_OVERDUE` (>24h unmitigated)

## 11. Final Recommendation
**OPS_1_MAINTENANCE_DASHBOARD_ALERT_OWNERSHIP_PASS**
