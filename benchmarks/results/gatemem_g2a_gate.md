# GateMem G2A Cross-Domain Replay Gate

`FROZEN_ADAPTER` `NO_POLICY_TUNING` `ACTIVE_FORGETTING_NOT_SCORED`

| Gate | Result |
|---|---|
| frozen_core_hash_valid | PASS |
| all_four_domains_present | PASS |
| all_2218_checkpoints_processed | PASS |
| projection_prediction_counts_match | PASS |
| external_evidence_hashes_valid | PASS |
| predictions_contain_no_scoring_fields | PASS |
| provenance_integrity_1_0_all_domains | PASS |
| deletion_refusal_measured_all_domains | PASS |
| deletion_false_positives_measured_all_domains | PASS |
| active_forgetting_not_scored | PASS |
| no_hosted_judge_artifacts | PASS |
| no_policy_tuning | PASS |
| aggregate_is_weighted_from_all_domains | PASS |
| production_claims_remain_blocked | PASS |

**Overall: PASS**

Aggregate characterization:

- checkpoints: `2218`
- auxiliary utility: `0.2775`
- privacy leakage: `0.1788`
- over-refusal: `0.4052`
- provenance integrity: `1.0000`
- deletion refusal consistency: `0.9738`
- deletion detector false positives / negatives: `136 / 20`
- active forgetting: **NOT SCORED**

G2A characterizes the frozen offline baseline only. It authorizes no policy tuning, runtime integration, role-enforcement claim, or deletion claim.
