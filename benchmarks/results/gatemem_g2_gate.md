# GateMem G2 Offline Adapter Gate

`GATEMEM_G2_BASELINE_COMPLETE` `DISCLOSURE_POLICY_INADEQUATE_FOR_PRODUCTION`

This is a successful benchmark-baseline closeout, not a successful governance-performance result.

| Gate | Result |
|---|---|
| offline_only | PASS |
| clean_g1_projections_only | PASS |
| prediction_count_complete | PASS |
| prediction_schema_compatible | PASS |
| no_scoring_fields_in_predictions | PASS |
| provenance_integrity_1_0 | PASS |
| deletion_refusal_consistency_1_0 | PASS |
| deletion_false_negatives_0 | PASS |
| deletion_not_scored | PASS |
| no_deletion_capability_claim | PASS |
| utility_measured | PASS |
| disclosure_measured | PASS |
| over_refusal_measured | PASS |
| limitations_retained | PASS |
| no_runtime_or_gatemem_imports | PASS |
| no_network_imports | PASS |
| no_hosted_provider_imports | PASS |
| no_hosted_judge | PASS |
| no_leaderboard_submission | PASS |

**Overall: PASS**

Measured limitations remain gate output, not hidden failures:

- auxiliary utility: `0.2762`
- privacy end-to-end leakage: `0.4427`
- over-refusal: `0.5286`

This pass validates offline adapter mechanics and measurement only. It authorizes no runtime policy claim or deletion lifecycle.
