# GateMem G2 Frozen Offline Baseline

```text
GATEMEM_G2_BASELINE_COMPLETE
GATEMEM_G2_OFFLINE_BASELINE_FROZEN
MEDICAL_DOMAIN_ONLY
OFFLINE_ADAPTER_MECHANICS_VERIFIED
PROVENANCE_INTEGRITY_VERIFIED
DISCLOSURE_POLICY_INADEQUATE_FOR_PRODUCTION
ACTIVE_FORGETTING_NOT_EVALUATED
NO_POLICY_TUNING_ON_MEDICAL_RESULTS
NO_PRODUCTION_ROLE_ENFORCEMENT_CLAIM
NO_DELETION_CAPABILITY_CLAIM
```

Frozen G1/G2 core composite SHA-256:

```text
4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209
```

GateMem upstream commit:

```text
603f9f4b4ba4b77f043c20f85687fa016fd720b0
```

The medical result is an internal benchmark baseline, not a successful
governance-performance result. G2A cross-domain replay must use this exact core
and configuration without tuning retrieval weights, disclosure rules, deletion
detection, thresholds, or normalizer behavior.

See `gatemem_g2_baseline_manifest.json` for the per-file hashes and frozen
configuration.

