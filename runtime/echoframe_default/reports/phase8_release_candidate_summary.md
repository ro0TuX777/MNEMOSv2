# Phase 8 Release Candidate Summary
- **total runtime calls observed**: 2000
- **eligible events**: 1980
- **EchoFrame LLM-facing events**: 1980
- **baseline fallback events**: 20
- **high-risk fallback events**: 20
- **approval_required fallback events**: 0
- **kill switch fallback events**: 0
- **validator failure fallback events**: 0
- **safety failure fallback events**: 0
- **actual EchoFrame rate vs eligible**: 1.0000
- **failure rate**: 0.00%
- **average baseline tokens**: 1989.00
- **average EchoFrame tokens**: 52.98
- **average token ratio**: 0.0266
- **p95 token ratio**: 0.0256
- **p99 token ratio**: 0.0266
- **average stability score**: 1.0000
- **minimum stability score**: 1.0000
- **answer-quality review result**: PASS
- **fallback reasons**: high_risk_excluded
- **safety_gate_failures**: 0
- **validator_failures**: 0
- **non_promotable packets**: 0
- **llm_context_source counts**: {"baseline": 20, "echoframe": 1980}
- **kill switch drill result**: PASS
- **failure-mode drill results**: PASS

## Recommendation
**PROMOTE TO STABLE DEFAULT-ON ELIGIBLE FEATURE:**
all documentation, tests, drills, safety gates, answer-quality review, and fallback controls pass