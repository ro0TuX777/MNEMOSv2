# GateMem G5 Independent Evaluation Handoff Checklist

Status: not started. Checkboxes require named-party evidence.

## A. MNEMOS policy group

- [x] G4 candidate nominated by immutable composite.
- [x] G4 frozen manifest and read-only verifier available.
- [x] G3 clean-input and authorization/disclosure contracts documented.
- [x] Development and historical GateMem limitations disclosed.
- [ ] Exact candidate archive/container delivered to custodian.
- [ ] Dependency/environment artifact delivered and hashed.
- [ ] Policy group signs no-access/no-post-unseal-change acknowledgement.

## B. Independent custodian

- [ ] Named custodian appointed.
- [ ] Independence/conflict attestation signed.
- [ ] Separate credentials, storage, logs, and evaluator identity established.
- [ ] Newly sealed or independent corpus accepted into custody.
- [ ] Corpus commitment published without prohibited disclosure.
- [ ] Novelty/non-overlap and license/reporting authority verified.
- [ ] Candidate nomination accepted or rejected before unsealing.
- [ ] Row-level retention and aggregate release policy recorded.

## C. Preregistration and evaluator

- [ ] All preregistration fields and numeric thresholds complete.
- [ ] Candidate, corpus, environment, scorer, aggregation, and protocol hashes
  frozen.
- [ ] Input/output schemas and annotation isolation tested with non-sealed smoke
  fixtures.
- [ ] One-shot/invalidation and infrastructure-failure rule adopted.
- [ ] Network/output allowlists and append-only audit verified.
- [ ] All required signatures present.
- [ ] Custodian explicitly authorizes unsealing.

## D. Execution

- [ ] Preflight passes before sealed access.
- [ ] One claim-eligible run begins under custodian control.
- [ ] Predictions freeze before annotation join.
- [ ] Scoring and aggregation match preregistered hashes.
- [ ] Exceptions/exposures classified.
- [ ] Custodian signs final run classification.
- [ ] Only approved aggregate/report artifacts released.

## Current stop condition

```text
SEALED_EVALUATION_BLOCKED_EXTERNAL_INPUTS_REQUIRED
```

Sections B–D cannot be completed by the existing G0–G4 policy-development
group acting alone.
