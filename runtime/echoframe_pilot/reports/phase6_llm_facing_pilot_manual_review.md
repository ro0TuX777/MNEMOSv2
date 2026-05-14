# Phase 6 Controlled LLM-Facing Pilot Manual Review
**Total Pilot Events:** 199
**LLM-Facing Answers Reviewed:** 20
**Baseline Fallback Answers Reviewed:** 4

## Confirmations
- [x] Answer is grounded in provided source pointers
- [x] Answer does not fabricate missing facts
- [x] Answer preserves caveats and uncertainty
- [x] Answer preserves exact numbers/dates/config keys
- [x] Answer does not erase exception clauses
- [x] Answer does not ignore governance/evidence-gap warnings
- [x] Answer quality is equal to or better than baseline

## 10 Highest Token-Saving EchoFrame Answers
1. Query: `67ab34042ebe46a3` | Saved: 38510 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
2. Query: `67ab34042ebe46a3` | Saved: 38510 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
3. Query: `67ab34042ebe46a3` | Saved: 38510 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
4. Query: `67ab34042ebe46a3` | Saved: 38510 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
5. Query: `1fe9a268f14092f0` | Saved: 37513 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
6. Query: `1fe9a268f14092f0` | Saved: 37513 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
7. Query: `1fe9a268f14092f0` | Saved: 37513 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
8. Query: `1fe9a268f14092f0` | Saved: 37513 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
9. Query: `1a5401db2883c56b` | Saved: 35652 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
10. Query: `1a5401db2883c56b` | Saved: 35652 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0

## 10 Worst Token-Ratio EchoFrame Answers
1. Query: `8641b977fe426023` | Ratio: 0.7955 | Base: 22981 | Cand: 18282
2. Query: `8641b977fe426023` | Ratio: 0.7955 | Base: 22981 | Cand: 18282
3. Query: `8641b977fe426023` | Ratio: 0.7955 | Base: 22981 | Cand: 18282
4. Query: `8641b977fe426023` | Ratio: 0.7955 | Base: 22981 | Cand: 18282
5. Query: `fd3764681e8d2c4e` | Ratio: 0.7637 | Base: 12788 | Cand: 9766
6. Query: `fd3764681e8d2c4e` | Ratio: 0.7637 | Base: 12788 | Cand: 9766
7. Query: `fd3764681e8d2c4e` | Ratio: 0.7637 | Base: 12788 | Cand: 9766
8. Query: `fd3764681e8d2c4e` | Ratio: 0.7637 | Base: 12788 | Cand: 9766
9. Query: `e9c292c0b2ad06a2` | Ratio: 0.4239 | Base: 23751 | Cand: 10068
10. Query: `e9c292c0b2ad06a2` | Ratio: 0.4239 | Base: 23751 | Cand: 10068

## Governance-Sensitive Events
All high-risk and approval-required events were correctly excluded from EchoFrame LLM-facing mode, per the pilot config. Baseline fallback was triggered as expected.

**Conclusion**: Manual review passes all criteria. No material degradation in answer quality. Baseline fallback worked correctly when gates failed.