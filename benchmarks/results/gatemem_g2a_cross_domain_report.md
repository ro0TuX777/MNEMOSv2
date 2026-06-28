# GateMem G2A Frozen Cross-Domain Baseline Replay

`GATEMEM_G2A_CROSS_DOMAIN_BASELINE_REPLAY_COMPLETE`

Frozen core: `4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209`

| Domain | Checkpoints | Utility | Privacy leakage | Over-refusal | Provenance | Deletion refusal | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 0.2762 | 0.4427 | 0.5286 | 1.0000 | 1.0000 | 32 | 0 |
| Office | 547 | 0.4416 | 0.0702 | 0.2987 | 1.0000 | 0.9685 | 34 | 7 |
| Education | 540 | 0.1667 | 0.0944 | 0.4278 | 1.0000 | 0.9556 | 15 | 8 |
| Household | 552 | 0.2500 | 0.0870 | 0.3315 | 1.0000 | 0.9728 | 55 | 5 |
| **Weighted aggregate** | **2218** | **0.2775** | **0.1788** | **0.4052** | **1.0000** | **0.9738** | **136** | **20** |

## Interpretation

Medical is an outlier for privacy leakage, but the limitation is structural: every domain has low auxiliary utility, material over-refusal, nonzero privacy leakage, and incomplete deletion-language coverage outside medical.

Candidate blocking is not content-safe disclosure. Provenance integrity remains perfect while governance performance remains inadequate.

Deletion metrics are refusal consistency only. Active forgetting is not scored.

No retrieval weights, disclosure rules, deletion detection, thresholds, or normalizer behavior were changed for G2A.
