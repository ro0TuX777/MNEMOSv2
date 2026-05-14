# EchoFrame Phase 6D 75% LLM-Facing Pilot Summary
- **Total Runtime Calls Observed**: 2000
- **Configured Sample Rate**: 0.75
- **Events Excluded By Sample Rate**: 509
- **Eligible Pilot Events**: 1980
- **Events Excluded By Admission Gate**: 20
- **Selected EchoFrame LLM-Facing Events**: 1476
- **Actual LLM-Facing Rate (vs Total)**: 0.7380
- **Actual LLM-Facing Rate (vs Eligible)**: 0.7455
- **Pilot Selection Rate**: 0.74
- **Baseline Fallback Events**: 524
- **Failure Rate**: 0.00%

## Metrics
- **Avg Baseline Tokens**: 1989.00
- **Avg EchoFrame Tokens**: 52.98
- **Avg Token Ratio**: 0.0266 (p95: 0.0256, p99: 0.0266)
- **Avg Stability Score**: 1.0000 (min: 1.0000)

## Safety and Exclusions
- **Safety Gate Failures**: 0
- **Validator Failures**: 0
- **Non-Promotable Packets**: 0
- **Exclusion Reasons**: high_risk_excluded

## Recommendation Logic
**PROMOTE TO 100% ELIGIBLE LLM-FACING PILOT**
All safety gates pass, answer quality holds, sample volume is sufficient, and token/stability targets hold.