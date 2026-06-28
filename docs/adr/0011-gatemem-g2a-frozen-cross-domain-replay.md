# ADR 0011: GateMem G2A Replays the Frozen Baseline Without Tuning

Date: 2026-06-24

Status: Accepted — completed offline characterization

## Context

G2 closed with a reproducible medical-domain baseline whose adapter mechanics
and provenance were correct but whose disclosure performance was inadequate.
Tuning against the medical results would convert that domain into a development
set and invalidate a fresh benchmark claim.

The next honest question was whether the medical failure was exceptional or a
cross-domain property of the metadata-poor policy.

## Decision

Replay the exact frozen G1/G2 core and configuration against GateMem's office,
education, and household domains. The replay is pinned to composite SHA-256:

```text
4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209
```

No retrieval weights, disclosure rules, deletion detection, thresholds,
normalizer behavior, runtime behavior, or GateMem source may change.

Predictions must be frozen externally before evaluator-only scoring. Only
aggregate per-domain and weighted reports may enter MNEMOS.

## Result

The medical privacy-leakage rate is an outlier, but the baseline limitation is
structural across all four domains:

- auxiliary utility remains low (`0.1667`–`0.4416`);
- privacy leakage remains nonzero (`0.0702`–`0.4427`);
- over-refusal remains material (`0.2987`–`0.5286`);
- provenance integrity is `1.0` in every domain; and
- visible deletion detection has 20 aggregate false negatives outside the
  medical result and 136 aggregate false positives.

Active forgetting remains unscored. Deletion refusal consistency is a behavior
measurement, not a deletion capability.

## Consequences

The G2 baseline is closed and must not be tuned then re-presented as a fresh
result on these four domains. Future policy development requires a separately
declared development corpus and untouched evaluation set.

Principal-bound authorization and entitlement semantics remain the substantive
gap. A governed deletion lifecycle remains a separate architectural decision.

## Evidence

- `docs/benchmarks/gatemem_g2a_cross_domain_replay.md`
- `benchmarks/results/gatemem_g2_baseline_manifest.json`
- `benchmarks/results/gatemem_g2a_cross_domain_report.json`
- `benchmarks/results/gatemem_g2a_cross_domain_report.md`
- `benchmarks/results/gatemem_g2a_{domain}_report.json`
- `benchmarks/results/gatemem_g2a_{domain}_report.md`
- `benchmarks/results/gatemem_g2a_gate.json`
- `benchmarks/results/gatemem_g2a_gate.md`
- `tools/compile_gatemem_g2a_cross_domain.py`
- `tools/run_gatemem_g2a_gate.py`
- `tests/test_gatemem_g2a.py`

