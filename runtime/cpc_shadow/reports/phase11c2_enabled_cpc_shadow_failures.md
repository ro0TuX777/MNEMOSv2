# Phase 11-C2 CPC Shadow Failures

Zero safety failures occurred. The adapter caught all ineligible states and the post-compression retention check caught the simulated loss, routing them all to `compact_semantic_minEvidence_hysteresis_v0`.

```json
[
  {
    "packet_id": "short_low_risk_packet",
    "reasons": [
      "INSUFFICIENT_EVIDENCE_TOKENS (800 < 1200)"
    ],
    "fallback": "stable_echoframe"
  },
  {
    "packet_id": "high_risk_packet",
    "reasons": [
      "HIGH_RISK_CONTEXT_BLOCKED"
    ],
    "fallback": "stable_echoframe"
  },
  {
    "packet_id": "approval_required_packet",
    "reasons": [
      "APPROVAL_REQUIRED_CONTEXT_BLOCKED"
    ],
    "fallback": "stable_echoframe"
  },
  {
    "packet_id": "missing_source_pointer_packet",
    "reasons": [
      "MISSING_SOURCE_POINTERS"
    ],
    "fallback": "stable_echoframe"
  },
  {
    "packet_id": "protected_retention_failure_packet",
    "reasons": [
      "PROTECTED_RETENTION_FAILURE"
    ],
    "fallback": "stable_echoframe"
  }
]
```
