# 07 Risk and Exceptions Register

## Residual Risk Register
- **RISK-001**: Reliance on underlying cloud IAM infrastructure for dual-control enforcement.
- **RISK-002**: 4-hour SLA window for VERIFIER_HEALTH_FAILURE means temporary visibility loss is possible before STOP.

## Open Exceptions Register
| `exception_id` | `related_control_id` | `description` | `risk_rating` | `approved_by` | `approval_ticket_id` | `expiration_date` | `mitigation_plan` | `compensating_control` | `status` | `closure_evidence` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (None) | | | | | | | | | | |

*Note: No exception may weaken CERT-0 BLOCKED capabilities. Any exception affecting production red lines is invalid and triggers an immediate STOP.*
