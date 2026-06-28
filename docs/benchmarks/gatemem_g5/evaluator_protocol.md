# GateMem G5 Custodian-Controlled Evaluator Protocol

Status: interface specification; no sealed corpus or run exists.

## Trust boundary

```text
custodian-only sealed corpus + annotations
                  |
                  | clean projection only
                  v
         frozen G4 candidate process
                  |
                  | predictions + content-free audit
                  v
          immutable prediction freeze
                  |
                  | evaluator-only annotation join
                  v
       row scores -> preregistered aggregates
                  |
                  v
        custodian-approved release report
```

Policy developers must not operate the sealed evaluator or read its row-level
inputs, outputs, exceptions, or labels during the claim-eligible run.

## Required package supplied by MNEMOS

- candidate nomination JSON and frozen G4 manifest;
- exact source archive/container matching the candidate composite;
- offline entry-point instructions;
- strict input and prediction schemas;
- content-free audit schema;
- dependency/environment commitment;
- this protocol and frozen preregistration;
- no development corpus, tuning notebook, or known GateMem labels unless the
  custodian separately needs them for overlap detection outside candidate view.

## Custodian-supplied interfaces

### Clean case stream

Each candidate-visible case must contain only preregistered G3 inputs and an
opaque `evaluation_case_id`. Evaluator annotations must be physically or
cryptographically unavailable to the candidate process.

### Candidate output

```yaml
evaluation_case_id: opaque identifier
outcome: DISCLOSED | DISCLOSED_WITH_REDACTION | DENIED
disclosed_content_or_digest: schema-defined
used_artifact_ids: authorized opaque IDs
used_source_ids: authorized opaque IDs
redaction_receipts: content-free references
external_reason_code: non-sensitive code | null
policy_fingerprint: sha256
```

The final schema and whether content or only a digest crosses the package
boundary must be frozen in preregistration.

### Prediction freeze

Before annotation join, the evaluator writes an append-only prediction manifest
containing row count, ordered/commutative row hashes as preregistered, candidate
hash, environment hash, start/end times, and an aggregate prediction digest.
The custodian signs or timestamps this manifest.

### Scoring join

Only after prediction freeze may the evaluator join expected actions, attack
types, leak targets, redaction expectations, and other scoring annotations.
The scorer writes row-level scores to custodian-only storage and the
preregistered aggregate report separately.

## Preflight

The evaluator must fail before unsealing if:

- candidate, environment, preregistration, scorer, or corpus commitment differs;
- required signatures or thresholds are incomplete;
- policy code can access annotation storage or unrestricted network/filesystem;
- output paths are not empty and uniquely assigned to the registration;
- the G4 frozen verifier fails; or
- an exposure/conflict declaration is unresolved.

## Report states

Exactly one final state is required:

```text
HELD_OUT_EVALUATION_COMPLETE
RETROSPECTIVE_OR_DEVELOPMENT_RESULT
INVALID_RUN_NO_PERFORMANCE_CLAIM
```

The report must preserve limitations, denominators, confidence intervals,
exceptions, and the prohibited-claim boundary even when all thresholds pass.
