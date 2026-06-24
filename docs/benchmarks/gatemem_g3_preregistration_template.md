# GateMem Authorization/Disclosure Preregistration Template

Status: Template only — complete and freeze before any sealed evaluation corpus
is opened.

## 1. Registration

```yaml
registration_id:
registered_at:
policy_workstream:
repository_commit:
dependency_environment_hash:
g1_contract_version:
prediction_contract_version:
scorer_version:
evaluation_custodian:
policy_developer_group:
release_reviewers:
```

## 2. Corpus separation

```yaml
development_corpus:
  name:
  purpose:
  manifest_path:
  manifest_sha256:
  case_count:
  domains:
  developer_access_authorized: true
  tuning_authorized: true

sealed_evaluation_corpus:
  name:
  provenance:
  custodian:
  commitment_or_manifest_sha256:
  case_count: sealed | declared
  domains: sealed | declared
  developer_access_authorized: false
  used_in_g0_through_g2a: false
  opened_at: null
```

Confirm explicitly:

- [ ] The evaluation corpus is newly sealed or independent.
- [ ] No policy developer has inspected its cases, labels, leak targets, or
  prior method outputs.
- [ ] It is not a relabeling or retrospective split of the already-observed
  G2/G2A four-domain corpus.
- [ ] The development and evaluation manifests cannot overlap.
- [ ] Licensing and attribution permit the planned evaluation and reporting.

If any item is false, label the run development/retrospective and prohibit a
fresh held-out claim.

## 3. Frozen authorization artifact

```yaml
identity_contract_version:
tenant_session_resolver_version:
role_assignment_schema_version:
entitlement_policy_version:
artifact_class_registry_version:
source_class_registry_version:
redaction_policy_version:
audit_schema_version:
policy_source_sha256:
policy_data_sha256:
configuration_sha256:
```

Record all fixed thresholds, budgets, timeouts, seeds, concurrency, caching,
replay, and error-handling behavior. “Default” is not a sufficient value;
resolve defaults before freeze.

## 4. Inputs and prohibited fields

List every permitted policy input field. Confirm that the method cannot read:

```text
query_type
attack_type
expected_action
judge_spec
leak_targets
```

Document how schema enforcement is tested and how evaluator annotations remain
one-way after prediction freeze.

## 5. Primary measurements

Define formulas, denominators, missing-data behavior, and confidence intervals
before the run.

```yaml
primary:
  authorized_utility:
  privacy_end_to_end_leakage:
  over_refusal:
  disclosure_denial_correctness:
  redaction_correctness:
  provenance_integrity:
secondary:
  cross_principal_candidate_exposure:
  cross_tenant_denial:
  replay_policy_mismatch_detection:
  content_free_audit_compliance:
excluded:
  active_forgetting: true
  deletion_capability: true
```

Set advancement thresholds here before unsealing. G3 does not prescribe or
tune their numeric values.

## 6. Claims

Predeclare:

- claims permitted if every primary threshold passes;
- claims permitted for partial or mixed results;
- claims prohibited regardless of result;
- whether results are internal, public, or submission-eligible; and
- exact language distinguishing offline semantics from production enforcement.

Production role enforcement, legal compliance, and deletion claims remain
prohibited without their own evidence and authorization.

## 7. Run procedure

```text
verify environment and hashes
verify development/evaluation non-overlap
verify policy freeze
unseal through evaluator custodian
clean projection
offline authorization/disclosure evaluation
freeze predictions
evaluator-only scoring join
aggregate report
seal row-level artifacts
record exceptions
```

No policy developer may inspect intermediate sealed-corpus failures during the
run.

## 8. Invalid-run and stop rules

Predeclare treatment of:

- manifest/hash mismatch;
- schema or hidden-field violation;
- missing or duplicate predictions;
- policy/service exception;
- redaction verification failure;
- evaluator warning or malformed case;
- retry and infrastructure failure; and
- accidental developer exposure to sealed content.

An invalid run is not silently rerun after a policy change. Record the defect,
re-freeze, and obtain a fresh evaluation corpus or downgrade the claim to
retrospective.

## 9. Post-run change control

```yaml
predictions_frozen_at:
prediction_manifest_sha256:
scoring_started_at:
row_level_artifact_custodian:
aggregate_report_sha256:
exceptions:
policy_changes_after_unseal:
claim_classification: held_out | retrospective | development | invalid
```

Any policy or threshold change after unsealing ends the registered held-out
claim for that corpus.

