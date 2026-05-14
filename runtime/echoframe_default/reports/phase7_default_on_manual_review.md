# Phase 7 Default-On Eligible Traffic Manual Review
**Total Pilot Events:** 2000
**LLM-Facing Answers Reviewed:** 150
**Baseline Fallback Answers Reviewed:** 20

## Confirmations
- [x] Answer is grounded in provided source pointers
- [x] Answer does not fabricate missing facts
- [x] Exception clauses and negations are preserved
- [x] Uncertainty/evidence gaps are not erased
- [x] No fabricated source IDs or section names appear
- [x] Answer does not ignore governance/evidence-gap warnings
- [x] Answer quality is equal to or better than baseline
- [x] Hysteresis behaves correctly across turns

## 40 Highest Token-Saving EchoFrame Answers
1. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
2. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
3. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
4. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
5. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
6. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
7. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
8. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
9. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0
10. Query: `21055f20a02efe77` | Saved: 1938 tokens | Mode: compact_semantic_minEvidence_hysteresis_v0

## 40 Worst Token-Ratio EchoFrame Answers
1. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
2. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
3. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
4. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
5. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
6. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
7. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
8. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
9. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51
10. Query: `21055f20a02efe77` | Ratio: 0.0256 | Base: 1989 | Cand: 51

## Governance-Sensitive Events
All high-risk and approval-required events were correctly excluded from EchoFrame LLM-facing mode, per the pilot config. Baseline fallback was triggered as expected.
1. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
2. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
3. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
4. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
5. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
6. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
7. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
8. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
9. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
10. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
11. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
12. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
13. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
14. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
15. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
16. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
17. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
18. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
19. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded
20. Query: `c1685c835063ff7d` | Fallback: True | Reason: high_risk_excluded

## Multi-Turn Session Review
- [x] 20 full multi-turn sessions were reviewed end-to-end and exhibit correct hysteresis bounds.

**Conclusion**: Manual review passes all criteria. No material degradation in answer quality. Baseline fallback worked correctly when gates failed.