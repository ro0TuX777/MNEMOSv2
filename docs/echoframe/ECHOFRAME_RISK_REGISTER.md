# EchoFrame Risk Register

| Risk | Severity | Likelihood | Mitigation | Owner | Status | Evidence Path |
|------|----------|------------|------------|-------|--------|---------------|
| source pointer omission | HIGH | LOW | Fallback to baseline if validator fails | MNEMOS Team | mitigated | tests/test_echoframe_release_candidate.py |
| governance signal omission | HIGH | LOW | Strict token checking and telemetry validation | MNEMOS Team | mitigated | phase7_default_on_summary.md |
| answer-quality degradation | MEDIUM | LOW | Manual QA review of generated answers | MNEMOS Team | monitored | reports/phase7_default_on_manual_review.md |
| high-risk traffic leakage | CRITICAL | LOW | Strict regex/classification fallback rules | MNEMOS Team | mitigated | tests/test_echoframe_release_candidate.py |
| approval_required leakage | CRITICAL | LOW | Payload inspection for approval flags | MNEMOS Team | mitigated | tests/test_echoframe_release_candidate.py |
| kill switch failure | CRITICAL | LOW | Hard-coded override at the top of adapter flow | MNEMOS Team | mitigated | tests/test_echoframe_release_candidate.py |
| telemetry write failure | LOW | LOW | Silent fails wrapped in try/except blocks | MNEMOS Team | accepted | shadow_adapter.py source |
| cross-session hysteresis contamination | HIGH | LOW | Session ID isolating dicts in adapter state | MNEMOS Team | mitigated | Phase 6/7 Review |
| token-ratio regression | MEDIUM | LOW | Ratio monitoring in telemetry; >1.00 ratio triggers fallback | MNEMOS Team | monitored | phase7_default_on_summary.md |
| baseline fallback failure | HIGH | LOW | Fallback defaults to original payload | MNEMOS Team | mitigated | tests/test_echoframe_release_candidate.py |
| configuration precedence error | HIGH | LOW | Explicit if-else hierarchy implemented and tested | MNEMOS Team | mitigated | shadow_adapter.py source |
| operator misunderstanding | MEDIUM | LOW | Operational Runbook created; explicit naming used | MNEMOS Team | mitigated | docs/echoframe/ECHOFRAME_OPERATIONAL_RUNBOOK.md |
