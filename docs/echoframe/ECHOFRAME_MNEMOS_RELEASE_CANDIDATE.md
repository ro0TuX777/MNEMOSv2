# EchoFrame MNEMOS Release Candidate

## 1. Executive Summary
EchoFrame Phase 8 represents the Release Candidate for integrating the EchoFrame Context Optimizer into the MNEMOS native runtime. This candidate enforces a Default-On Eligible Traffic operational mode while strictly maintaining full baseline fallbacks for all high-risk, governance-restricted, and unstable contexts.

## 2. Feature Scope
The release candidate includes:
- `compact_semantic_minEvidence_hysteresis_v0` rendering mode
- Default-on eligible traffic routing
- Baseline fallback for all exclusions
- High-risk and approval-required exclusions
- Global kill switch override
- Comprehensive telemetry and answer-quality review infrastructure

## 3. Architecture Overview
EchoFrame operates as a shadow adapter immediately following MNEMOS baseline document retrieval. It transforms large document chunks into dense, token-efficient evidence structures.

## 4. Runtime Decision Flow
1. Fetch `MNEMOS_ECHOFRAME_KILL_SWITCH`. If true, bypass completely.
2. Fetch `MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE`. If false, bypass completely (unless legacy pilot is active).
3. Evaluate packet against admission gates.
4. If eligible, replace LLM context with EchoFrame packet.
5. Emit telemetry to `runtime/echoframe_default/`.

## 5. Eligibility Rules
A packet is eligible ONLY if it passes all of the following:
- No safety/validation failures (source pointers must exist).
- Packet must be promotable (no `NO_EVIDENCE_FOUND`).
- Token ratio must be <= 1.0.
- Stability score must be >= 0.90.
- Query category must NOT be `high-risk`.
- Governance meta must NOT flag `approval_required`.

## 6. Fallback Rules
If any eligibility rule fails, the system immediately falls back to the baseline MNEMOS context, preventing any potential answer degradation or leakage.

## 7. Kill Switch Behavior
The kill switch (`MNEMOS_ECHOFRAME_KILL_SWITCH=true`) operates as a hard override at the top of the decision flow, forcing 100% baseline routing regardless of pilot or default-on state.

## 8. Telemetry Schema
Event type: `echoframe.default_on_event`
Includes: `llm_context_source`, `fallback_reason`, `fallback_to_baseline`, `token_ratio`, `stability_score`, and governance signals.

## 9. Validation History Summary
- **Phase 6**: Controlled pilot scaling from 5% to 100%. Verified stable token compression and state hysteresis.
- **Phase 7**: Default-on eligible architecture with kill switch. Validated against 2000 mock queries with 0% failure rate.
- **Phase 8**: Hardening, documentation, and regression suite completion.

## 10. Known Non-Goals
This release candidate strictly does NOT include:
- Upstream EchoFrame code integration.
- Routing of high-risk or approval_required traffic through EchoFrame.

## 11. Release Candidate Acceptance Criteria
- 0 safety/validator failures.
- Baseline fallback succeeds for all exclusions.
- Manual review proves no answer-quality degradation.
- Token ratios remain highly efficient.

## 12. Open Risks
See `ECHOFRAME_RISK_REGISTER.md` for tracked operational risks.

## 13. Recommendation
PROMOTE TO STABLE DEFAULT-ON ELIGIBLE FEATURE.
