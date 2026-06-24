# GateMem G2A Frozen Cross-Domain Baseline Replay

Date: 2026-06-24

Status: `GATEMEM_G2A_CROSS_DOMAIN_BASELINE_REPLAY_COMPLETE`

```text
SAME_FROZEN_ADAPTER
ALL_FOUR_GATEMEM_DOMAINS
NO_POLICY_TUNING
NO_RUNTIME_INTEGRATION
NO_HOSTED_JUDGE
NO_LEADERBOARD_SUBMISSION
ACTIVE_FORGETTING_NOT_SCORED
```

## Objective and freeze

G2A characterizes the exact G2 medical baseline across office, education, and
household. It does not improve or tune the policy.

Frozen G1/G2 core:

```text
4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209
```

Frozen defaults:

```text
top_k = 8
minimum_score = 0.08
max_disclosed_records = 4
max_answer_characters = 2200
deletion_mode = unsupported
```

The replay changed no retrieval weight, disclosure rule, deletion signal,
threshold, or normalizer behavior.

## Results

| Domain | Checkpoints | Utility | Privacy leakage | Over-refusal | Provenance | Deletion refusal | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 0.2762 | 0.4427 | 0.5286 | 1.0000 | 1.0000 | 32 | 0 |
| Office | 547 | 0.4416 | 0.0702 | 0.2987 | 1.0000 | 0.9685 | 34 | 7 |
| Education | 540 | 0.1667 | 0.0944 | 0.4278 | 1.0000 | 0.9556 | 15 | 8 |
| Household | 552 | 0.2500 | 0.0870 | 0.3315 | 1.0000 | 0.9728 | 55 | 5 |
| **Weighted aggregate** | **2,218** | **0.2775** | **0.1788** | **0.4052** | **1.0000** | **0.9738** | **136** | **20** |

Aggregate counts:

- utility: 202/728 auxiliary rule matches;
- privacy: 130/727 end-to-end leak matches;
- utility refused or redacted: 295/728;
- provenance-valid predictions: 2,218/2,218;
- cross-principal candidates blocked: 6,277/6,564 (`0.9563`); and
- true deletion checkpoints explicitly refused: 743/763.

The education evaluator emitted one upstream dataset warning: one deletion
checkpoint's leak target was not found before its checkpoint boundary. G2A did
not alter or suppress the warning.

## Interpretation

Medical is unusually difficult for the visible policy's privacy behavior, but
the limitation is not medical-only:

- every domain has nonzero content leakage;
- every domain trades low utility against substantial over-refusal;
- perfect provenance does not imply safe disclosure; and
- blocking 95.63% of cross-principal candidates still leaves 17.88% aggregate
  privacy leakage.

The remaining domains also expose a deletion-language generalization gap. The
unchanged detector misses 20 true deletion checkpoints and conservatively
refuses 136 non-deletion checkpoints. Those values are characterization, not a
prompt to tune against the now-observed benchmark.

## Evaluation boundary

For each domain:

1. G1 wrote clean projections externally.
2. The hash-pinned G2 adapter wrote predictions and content-free diagnostics.
3. Predictions were frozen.
4. GateMem's rule scorer joined evaluator annotations without a hosted judge.
5. MNEMOS received only aggregate reports and hashes.

GateMem's stub was not used. No scoring annotation entered projection,
retrieval, disclosure, normalization, or prediction construction.

All row-level artifacts remain under:

```text
G:\MNEMOS-research\gatemem_g2a_artifacts\
```

The 21 replay files total 83,733,506 bytes. They are not copied into MNEMOS or
the GateMem clone. Per-artifact hashes are recorded in each domain JSON report.

## Evidence

- `benchmarks/results/gatemem_g2_baseline_manifest.json`
- `benchmarks/results/gatemem_g2a_medical_report.{json,md}`
- `benchmarks/results/gatemem_g2a_office_report.{json,md}`
- `benchmarks/results/gatemem_g2a_education_report.{json,md}`
- `benchmarks/results/gatemem_g2a_household_report.{json,md}`
- `benchmarks/results/gatemem_g2a_cross_domain_report.{json,md}`
- `benchmarks/results/gatemem_g2a_gate.{json,md}`

## Advancement boundary

G2/G2A is closed as a frozen internal baseline. Any future disclosure-policy
work must declare separate development and untouched evaluation data. No G2A
result authorizes production role enforcement or a deletion capability claim.

