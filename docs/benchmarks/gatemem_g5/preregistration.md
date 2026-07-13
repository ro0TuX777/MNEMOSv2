# GateMem G5 Sealed-Evaluation Preregistration

Status: `DRAFT_EXTERNAL_FIELDS_REQUIRED`

This document must be completed, reviewed, hashed, and signed before the
custodian unseals any case. It specializes the G3 preregistration template for
the frozen G4 candidate.

## Registration and roles

```yaml
registration_id:
registered_at:
custodian:
evaluation_operator:
corpus_author_or_provider:
policy_development_group: MNEMOS GateMem G0-G4 contributors
release_reviewer:
independence_attestation_digest:
```

## Candidate nomination

```yaml
candidate_name: MNEMOS GateMem G4 offline authorization/disclosure reference
candidate_composite_sha256: ed3b5c7672e591b039183eaa2d8c7c7630a655575bfc7866558d53f2eb874c52
candidate_manifest_sha256: 4924b06e5d77c7a5cc1825a77a03d5e1405fdd3d3872dafb758e18418b32f15d
candidate_manifest_path: benchmarks/results/gatemem_g4_frozen_reference_manifest.json
candidate_nomination_path: benchmarks/evaluation/gatemem_g5_candidate_nomination.json
candidate_accepted_by_custodian: false
candidate_acceptance_time:
candidate_acceptance_signature:
```

No G4 source, policy, threshold, schema, or fixture change is allowed after
candidate acceptance. A change creates a new development candidate.

## Sealed corpus

```yaml
corpus_name_or_opaque_id:
corpus_provenance:
corpus_custodian:
commitment_sha256:
commitment_published_at:
case_count: sealed_or_declared
domain_metadata: sealed_or_declared
license_and_reporting_authority:
newly_sealed_or_independent: false
overlap_check_method:
overlap_check_result:
policy_developer_access: false
prior_candidate_results_exist: false
```

The corpus must not be any split, transformation, or relabeling of the released
medical, office, education, or household GateMem domains already observed in
G2/G2A.

## Frozen execution configuration

The custodian must resolve every blank before freeze:

```yaml
evaluation_package_version:
clean_projection_contract:
candidate_entry_point:
dependency_environment_sha256:
container_or_environment_digest:
clock_policy:
concurrency:
timeouts:
retry_policy: governed_by_one_shot_rules
random_seeds:
network_policy:
output_allowlist:
prediction_schema:
scorer_version_and_sha256:
aggregation_version_and_sha256:
```

## Permitted policy inputs

Only G3/G4 contract inputs may cross into the candidate: validated synthetic or
evaluation principal context, identity-derived tenant/session scope, operation,
registered purpose, role assignment, entitlement snapshot, artifact/source
descriptor, time, obligations, content eligible for disclosure/redaction, and
replay state.

The candidate must never receive evaluator fields such as:

```text
expected_action
query_type
attack_type
judge_spec
leak_targets
scoring_labels
```

## Measurements and thresholds

Formulas, denominators, confidence intervals, missing-data treatment, and
numeric thresholds must be supplied by the custodian/release reviewer before
unsealing.

| Primary measurement | Formula/version | Pass threshold |
|---|---|---|
| Authorized retrieval utility | | |
| End-to-end privacy leakage | | |
| Over-refusal | | |
| Disclosure-denial correctness | | |
| Redaction correctness | | |
| Provenance/used-ID integrity | | |

Secondary measurements may include cross-principal exposure, cross-tenant
denial, replay-drift rejection, content-free audit compliance, and invalid-case
handling. Active forgetting and deletion remain excluded and unscored.

## Claims

Predeclare exact wording for:

- all-primary-pass outcome;
- mixed/partial outcome;
- invalid run;
- public versus internal reporting; and
- offline held-out performance versus production enforcement.

Regardless of result, prohibit claims of production authorization security,
legal compliance, deletion, active forgetting, or general performance beyond
the sealed corpus's defined population.

## Freeze and signatures

```yaml
preregistration_sha256:
frozen_at:
custodian_signature:
evaluation_operator_signature:
release_reviewer_signature:
policy_group_acknowledgement:
all_required_fields_complete: false
unsealing_authorized: false
```
