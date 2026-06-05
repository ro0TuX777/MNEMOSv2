# Phase 11-C CPC Shadow Failures

Zero safety failures occurred. All packets that failed admission gates were safely routed to `compact_semantic_minEvidence_hysteresis_v0`.

```json
[
  {
    "packet_id": "tc1",
    "reasons": [
      "CPC_SHADOW_DISABLED_BY_CONFIG",
      "INSUFFICIENT_EVIDENCE_TOKENS (800 < 1200)"
    ],
    "fallback": "stable_echoframe"
  },
  {
    "packet_id": "tc2",
    "reasons": [
      "CPC_SHADOW_DISABLED_BY_CONFIG",
      "HIGH_RISK_CONTEXT_BLOCKED"
    ],
    "fallback": "stable_echoframe"
  },
  {
    "packet_id": "tc3",
    "reasons": [
      "CPC_SHADOW_DISABLED_BY_CONFIG",
      "APPROVAL_REQUIRED_CONTEXT_BLOCKED"
    ],
    "fallback": "stable_echoframe"
  },
  {
    "packet_id": "tc4",
    "reasons": [
      "CPC_SHADOW_DISABLED_BY_CONFIG"
    ],
    "fallback": "stable_echoframe"
  }
]
```
