# EchoFrame Kill Switch Procedure

## Overview
The kill switch provides an emergency global override to force all read traffic to use the baseline MNEMOS retrieval context. It disables the EchoFrame Context Optimizer completely.

## Activation
To activate, set:
`MNEMOS_ECHOFRAME_KILL_SWITCH=true`

## Verification
1. Activate the switch.
2. Check telemetry logs in `runtime/echoframe_default/`.
3. Verify that `kill_switch_active` is `True` and `fallback_to_baseline` is `True` for all subsequent events.
4. Verify that `llm_context_source` is `baseline`.

## Deactivation
To return to normal operation, set:
`MNEMOS_ECHOFRAME_KILL_SWITCH=false`
Ensure that `MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE` remains properly configured.
