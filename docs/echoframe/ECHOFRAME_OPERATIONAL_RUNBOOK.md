# EchoFrame Operational Runbook

## How to enable default-on eligible mode
Set the environment variable:
`MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=true`

## How to disable EchoFrame
Set both the default-on and legacy pilot flags to false:
`MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=false`
`MNEMOS_ECHOFRAME_LLM_FACING_ENABLED=false`

## How to activate kill switch
Set the kill switch to true:
`MNEMOS_ECHOFRAME_KILL_SWITCH=true`
This overrides all other settings.

## How to inspect telemetry
Telemetry is logged as JSON files in:
`runtime/echoframe_default/shadow_event_*.json`

## How to summarize reports
Run the summarization script:
`python benchmarks/echoframe_eval/shadow_runtime/summarize_phase7_default_on.py`
This generates a markdown summary in the reports directory.

## How to detect fallback spikes
Monitor the generated summary report. If `Actual LLM-Facing Rate (vs Eligible)` drops significantly below 1.0, or if `Baseline Fallback Events` spikes, investigate the reported `fallback_reason`.

## How to detect token-ratio regressions
Review the metrics section of the summary report. Ensure the average token ratio remains < 0.10 and the p99 ratio remains <= 1.25.

## How to detect answer-quality regressions
Generate the manual review report:
`python benchmarks/echoframe_eval/shadow_runtime/generate_manual_review_report_7.py`
Perform manual human review on the sampled events.

## Production Operations
- **how to confirm EchoFrame is active**: Ensure `MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=true` and check telemetry in `runtime/echoframe_default/` for `llm_context_source: echoframe`.
- **how to confirm baseline fallback is working**: Look for `fallback_to_baseline: true` and `llm_context_source: baseline` in telemetry events.
- **how to activate kill switch**: Set `MNEMOS_ECHOFRAME_KILL_SWITCH=true`.
- **how to verify no high-risk leakage**: Generate the summary report and confirm `high-risk fallback events` matches expectations, with zero high-risk events proceeding to `echoframe`.
- **how to inspect token-ratio regressions**: Review the generated summary for `average token ratio` and `p99 token ratio`. Ratios > 1.0 will trigger fallback.
- **how to inspect answer-quality concerns**: Use the manual review scripts to extract sampled payload pairs and perform qualitative human review.
- **how to rollback**: To rollback fully, activate the kill switch and then set `MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=false`.

## How to rollback to baseline-only
Activate the kill switch (`MNEMOS_ECHOFRAME_KILL_SWITCH=true`). If issues persist, disable the default-on mode (`MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=false`).
