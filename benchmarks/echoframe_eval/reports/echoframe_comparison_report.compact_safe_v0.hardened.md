# EchoFrame vs MNEMOS Hardened Comparison Report (compact_safe_v0)

## Summary Metrics
- **Original 20-Query Failures**: 0 / 20
- **Expanded Hardened Failures**: 0 / 20
- **Overall Failures**: 0 / 40
- **Overall Token Reduction Ratio**: 0.8929
- **Best Case Token Reduction**: 0.3819
- **Worst Case Token Regression**: 7.1600
- **Packets < Baseline**: 16
- **Packets > Baseline**: 24

## Packet Characteristics
- Multiple Sources: 40
- Contradiction Flags: 0
- Approval Required: 0
- Unknown Preservation: 0

## Safety Check Failures
- dropped_provenance: 0
- dropped_gaps: 0
- dropped_contradictions: 0
- dropped_approval: 0
- lost_unknown_preservation: 0
- invalid_empty_packet: 0
- confident_claim_when_unknown: 0
- lost_numeric_threshold: 0
- lost_date: 0
- lost_negation: 0
- lost_exception: 0

## Promotion Gates
- **Gate B Provenance**: PASS
- **Gate C Governance Preservation**: PASS
- **Gate D Token Efficiency**: PASS
- **Gate F Explainability**: PASS
- **Numeric Threshold Preservation**: PASS
- **Date Preservation**: PASS
- **Negation Preservation**: PASS
- **Exception Language Preservation**: PASS

**Conclusion**: compact_v0 passed all hardened adversarial cases and maintained strong token efficiency.