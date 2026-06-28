# GateMem G2 Baseline Closeout

Date: 2026-06-24

```text
GATEMEM_G2_BASELINE_COMPLETE
OFFLINE_ADAPTER_MECHANICS_VERIFIED
PROVENANCE_INTEGRITY_VERIFIED
DISCLOSURE_POLICY_INADEQUATE_FOR_PRODUCTION
ACTIVE_FORGETTING_NOT_EVALUATED

GATEMEM_G2_OFFLINE_BASELINE_FROZEN
MEDICAL_DOMAIN_ONLY
NO_POLICY_TUNING_ON_MEDICAL_RESULTS
NO_PRODUCTION_ROLE_ENFORCEMENT_CLAIM
NO_DELETION_CAPABILITY_CLAIM
```

G2 is closed as a successful benchmark-baseline implementation and an
unsuccessful governance-performance result.

The baseline proves that MNEMOS has an honest external measurement path with:

- clean-input isolation;
- predictions frozen before evaluator-only joins;
- exact used-record provenance;
- explicit refusal for detected deletion cases; and
- no runtime, GateMem, network, provider, or hosted-model import path.

The medical result is a do-not-integrate signal:

| Dimension | Result |
|---|---:|
| Auxiliary utility | 27.62% |
| Privacy leakage | 44.27% |
| Over-refusal | 52.86% |
| Provenance integrity | 100% |
| Deletion refusal consistency | 177/177 |
| Active forgetting | Not scored |

Blocking 91.65% of cross-principal candidates while leaking on 44.27% of
privacy checkpoints demonstrates that candidate filtering is materially
different from content-safe disclosure. Visible-text heuristics are not a
substitute for principal-bound authorization and entitlement semantics.

The frozen core and configuration are recorded in
`benchmarks/results/gatemem_g2_baseline_manifest.json` under composite SHA-256:

```text
4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209
```

G2A replay may characterize this exact baseline on other domains. It may not
tune retrieval weights, disclosure rules, deletion detection, thresholds, or
normalizer behavior against the medical result and then present that tuned run
as a fresh benchmark outcome.

