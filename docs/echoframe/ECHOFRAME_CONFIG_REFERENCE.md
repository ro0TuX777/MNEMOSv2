# EchoFrame Configuration Reference

## Active Operational Settings

- `MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE`: (bool) Enables default-on routing for eligible low/medium-risk traffic.
- `MNEMOS_ECHOFRAME_KILL_SWITCH`: (bool) Hard override to force all traffic to baseline context.
- `MNEMOS_ECHOFRAME_MODE`: (string) The renderer mode. Recommended: `compact_semantic_minEvidence_hysteresis_v0`.
- `MNEMOS_ECHOFRAME_FAIL_CLOSED`: (bool) If true, exceptions during EchoFrame processing bubble up. If false, they fallback silently to baseline.
- `MNEMOS_ECHOFRAME_REQUIRE_VALIDATION`: (bool) Enforces strict validation on generated packets.
- `MNEMOS_ECHOFRAME_ALLOW_HIGH_RISK`: (bool) MUST REMAIN FALSE. Prevents high-risk traffic from using EchoFrame.
- `MNEMOS_ECHOFRAME_OUTPUT_DIR`: (string) Destination for telemetry. Default: `runtime/echoframe_default/`.

## Precedence Hierarchy

1. `MNEMOS_ECHOFRAME_KILL_SWITCH=true` → Baseline only
2. `MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=true` → EchoFrame for eligible traffic
3. `MNEMOS_ECHOFRAME_LLM_FACING_ENABLED=true` → Legacy pilot behavior
4. otherwise → Baseline only
