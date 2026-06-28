# GateMem G2 Offline Retrieval/Disclosure Report

`GATEMEM_G2_BASELINE_COMPLETE`

`OFFLINE_ONLY` `CLEAN_G1_PROJECTIONS_ONLY` `NO_HOSTED_JUDGE` `NO_DELETION_CAPABILITY_CLAIM`

| Dimension | Result |
|---|---:|
| Authorized retrieval utility (auxiliary) | 0.2762 (58/210) |
| Privacy action correctness | 0.2500 |
| Expected-refusal correctness | 0.2743 |
| Privacy answer leakage | 0.4427 |
| Privacy context leakage | 0.4427 |
| Expected-redaction correctness | 0.0000 |
| Cross-principal candidates blocked | 0.9165 (1701/1856) |
| Over-refusal | 0.5286 |
| Used-record provenance integrity | 1.0000 (579/579) |
| Deletion refusal consistency | 1.0000 (177/177) |
| Active-forgetting score | **NOT SCORED** |

## Interpretation

The adapter proves clean offline wiring and exact used-record provenance, but its metadata-poor visible policy is not production-grade disclosure enforcement. Privacy end-to-end leakage is 0.4427, and over-refusal is 0.5286.

Deletion checkpoints are explicitly refused. The official deletion leakage value is excluded because refusal does not demonstrate removal, non-recoverability, or non-confirmation. The visible detector produced 32 conservative false positives and 0 false negatives.

## Persistent limitations

- Production role enforcement remains unsupported.
- G1 medical projections contain no explicit entitlement metadata.
- The disclosure policy is a conservative visible-input heuristic.
- The utility metric is auxiliary rule-based coverage, not a hosted or human judge.
- Deletion rows are refusal-consistency checks only and do not measure active forgetting.

No runtime route, authorization change, hosted judge, leaderboard submission, or deletion capability claim is authorized by this report.
