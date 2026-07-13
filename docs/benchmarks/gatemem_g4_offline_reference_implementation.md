# GateMem G4 Offline Authorization/Disclosure Reference Implementation

Date: 2026-06-24

Status: `GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_COMPLETE`

Freeze status: `GATEMEM_G4_FROZEN_REFERENCE_CONTRACT_BASELINE`

Classification:
`REFERENCE_CONTRACT_CONFORMANCE_ON_INSPECTABLE_SYNTHETIC_DEVELOPMENT_CASES`

## Authorization

```text
ADR_0013_ACCEPTED
GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_AUTHORIZED
LOCAL_OFFLINE_ONLY
MNEMOS_OWNED_SYNTHETIC_DEVELOPMENT_CASES_ONLY
NO_NETWORK_OR_RUNTIME_ROUTE
NO_PRODUCTION_AUTHORITY_CLAIM
NO_GATEMEM_SCORE_OR_HELD_OUT_CLAIM
NO_DELETION_ENGINEERING
```

## Implementation

The standalone `prototype/gatemem_g4` package implements:

- a harness-owned test HMAC identity authority;
- identity-derived tenant/session validation;
- scoped role plus entitlement evaluation with explicit-deny precedence;
- artifact/source class, relationship, classification, time, and lineage checks;
- deterministic structural/literal redaction with residue verification;
- purpose-, identity-, entitlement-, policy-, descriptor-, and
  redaction-version-bound replay checks;
- authorized package provenance fingerprints;
- strict content-free audit events; and
- a deterministic MNEMOS-owned synthetic development generator.

No MNEMOS runtime module imports the package. The package has no network,
hosted-model, GateMem, service, SDK, durable-memory, shared-cache, or deletion
dependency.

Every audit event carries the `g4_case_audit` retention class and a 30-day
retention value. A rollback rehearsal generated isolated temporary corpus/run
artifacts and removed only the known G4 filenames beneath an explicitly allowed
parent; an unknown-file mutation was refused and preserved.

## HMAC key isolation

Key generation or loading occurs only inside the fixture harness. The identity
authority constructor requires a private harness capability and is not exported
by the package API. Case files contain unsigned synthetic claims and mutation
instructions; the harness signs immediately before validation and never writes
an envelope.

A dedicated mutation run injects a known 32-byte sentinel key, runs the complete
generator/harness path, then scans every corpus and output artifact for raw,
hex, standard Base64, and URL-safe Base64 forms. The mutation passed. Static
inspection also confirms that authority construction and `secrets.token_bytes`
occur only in the harness module.

## Development corpus

The deterministic seed-404 corpus contains 36 inspectable cases covering:

- valid disclosure and required redaction;
- valid scoped delegation plus expired and operation-widening delegation denial;
- forged, unknown-issuer, tampered, and expired identity envelopes;
- caller scope widening and query claims of authority;
- inactive membership/session/role and missing entitlement;
- purpose, operation, relationship, artifact/source class, and classification denial;
- explicit-denial precedence and incomplete lineage;
- redaction residue and unknown transform denial;
- package record and character budget denial;
- replay drift across identity, entitlement, policy, descriptor, and redaction versions;
- evaluator-field injection; and
- HMAC-key artifact isolation.

Decision-path cases and expectations are stored separately. The corpus manifest
sets `contains_gatemem_data: false`, `contains_production_data: false`, and
`held_out_eligible: false`.

External artifacts:

```text
G:\MNEMOS-research\gatemem_g4_development_corpus
G:\MNEMOS-research\gatemem_g4_reference_run
```

## Results

| Measure | Result |
|---|---:|
| Synthetic development cases | 36 |
| Exact expected outcomes | 36/36 |
| Disclosed | 3 |
| Disclosed with redaction | 1 |
| Denied | 32 |
| Content-free audit events | 36/36 |
| Persisted denied/redaction canaries | 0 |
| Provenance-valid packages/denials | 36/36 |
| Reference gates | 33/33 passed |

Pinned evidence:

| Artifact | SHA-256 |
|---|---|
| Synthetic development corpus composite | `b0cb2522f2d21fd90705cac921a5c533d4cb569e1f5ad267f6e35b8b52cd01e3` |
| G4 implementation/corpus composite | `ed3b5c7672e591b039183eaa2d8c7c7630a655575bfc7866558d53f2eb874c52` |

The evidence is recorded in `benchmarks/results/gatemem_g4_gate.json` and
`benchmarks/results/gatemem_g4_gate.md`.

The frozen regression manifest is
`benchmarks/results/gatemem_g4_frozen_reference_manifest.json`. The read-only
verifier is `tools/verify_gatemem_g4_frozen.py`. Any mismatch in a pinned source
hash, composite, case count, expected outcome count, gate count, corpus hash, or
claim classification fails verification.

## Claim boundary

This result demonstrates reference-contract conformance on inspectable synthetic
development cases. It is not authorization security, production readiness,
held-out evaluation, GateMem benchmark performance, legal compliance, or a
deletion capability.

All four existing GateMem domains remain historical characterization data only.
Sealed evaluation remains blocked until an independent custodian accepts a new
sealed or independent corpus under the G3 preregistration protocol.

Deletion remains a separate later ADR and was not changed by G4.

Further GateMem implementation is paused until an independent sealed-evaluation
custodian and a newly sealed or independent corpus exist. Until then, G4 may be
used for regression testing only.
