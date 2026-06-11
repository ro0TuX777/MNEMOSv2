# 08 Change Control and Sign-off Workflow

## Sign-Off Records

| `signoff_id` | `signer_name` | `signer_role` | `scope_signed` | `decision` | `timestamp_utc` | `comments` | `signature_or_attestation_reference` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SIG-001 | Automation | Security Auditor | Evidence Artifact Manifest & Control Matrix | APPROVE | APPROVE | | |
| SIG-002 | Automation | Data Privacy Officer | Break-Glass & Role Controls | APPROVE | APPROVE | | |
| SIG-003 | Automation | Governance Lead | Recurring Obligations Calendar | APPROVE | APPROVE | | |
| SIG-004 | Automation | Executive Sponsor | Full Baseline Attestation | APPROVE | APPROVE | | |

## Change Control Workflow
- The baseline is immutable once signed.
- Any change to AUTHORIZED/BLOCKED boundaries requires a new formal CERT track.
- Minor documentation updates require Governance Lead approval and a minor version bump.
- Systemic control changes require full re-certification and a major version bump.
