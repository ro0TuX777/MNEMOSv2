# GateMem G4 Offline Authorization/Disclosure Reference Implementation Proposal

Date: 2026-06-24

Status: accepted and implemented within the local offline reference boundary.

Classification: **Historical proposal plus accepted local reference boundary.
No runtime or benchmark claim.**

## 1. Objective

G4 proposes the smallest executable evidence path for the G3 decision model:

```text
synthetic signed principal envelope
        |
        v
fixture identity validation and identity-derived scope
        |
        v
immutable entitlement + session + artifact descriptors
        |
        v
deterministic deny-by-default policy evaluation
        |
        v
isolated obligation/redaction transform and verification
        |
        v
disclosed package or content-free denial
        |
        v
content-free local audit evidence
```

The lane exists to test contract correctness, fail-closed behavior, provenance,
and non-disclosure properties. It does not test production authentication,
distributed consistency, legal policy, or GateMem performance.

## 2. Proposed authorization boundary

This document requested, and ADR 0013 subsequently activated:

```text
GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_AUTHORIZED
LOCAL_OFFLINE_ONLY
MNEMOS_OWNED_SYNTHETIC_DEVELOPMENT_CASES_ONLY
NO_NETWORK_OR_RUNTIME_ROUTE
NO_PRODUCTION_AUTHORITY_CLAIM
NO_GATEMEM_SCORE_OR_HELD_OUT_CLAIM
NO_DELETION_ENGINEERING
```

The authorization is exhausted by the completed bounded implementation. Any new
implementation scope requires separate review.

## 3. Operational choices

| Concern | G4 reference choice | Boundary / deferred production decision |
|---|---|---|
| Identity authority | Versioned fixture authority issues test-only HMAC-signed principal envelopes; validator constructs the G3 validated-principal context | Not a production IdP, authentication protocol, or proof of user identity; production issuer and assurance requirements remain unselected |
| Policy authority / evaluator | Original deterministic MNEMOS evaluator over a versioned declarative policy bundle; explicit-deny precedence and default deny | No OPA, cloud policy service, model call, or production policy catalogue; a later proposal must decide build-versus-integrate |
| Entitlement-store source | Immutable JSONL snapshots generated with each synthetic corpus, schema-validated and hash-pinned | No database, live grants, revocation feed, or production source of truth |
| Trusted session registry | Immutable JSONL mapping of session to tenant, permitted principals/delegations, state, validity, and version | No runtime session service or caller-provided scope authority |
| Artifact-descriptor source | Sidecar descriptors emitted by the synthetic generator and checked against corpus content/lineage manifests | No retroactive classification of production memory and no runtime metadata migration |
| Redaction-transform boundary | Isolated deterministic structural and literal-span transforms after authorization and before package assembly, followed by independent verification | No LLM redaction, best-effort regex-only safety claim, or production DLP selection |
| Audit store / retention | Append-only local JSONL using the G3 allowlist; raw per-case audit retained 30 days by default, CI copies at most 14 days, aggregate manifests/reports retained with the research record | No protected content; production audit platform, access control, jurisdiction, legal hold, and retention remain unselected |
| Deployment / rollback | Standalone local library and CLI, no listener, network, runtime import, durable memory write, or shared cache; rollback disables the entry point and discards versioned outputs | No canary, service deployment, or runtime failover claim |
| Development corpus generator | Deterministic seeded MNEMOS-owned generator with authored base cases plus bounded adversarial mutations | Development evidence only; generated cases and labels are inspectable and therefore never held out |
| Sealed-evaluation custodian | None in G4; the role is deliberately unassigned | A fresh claim is blocked until an independent custodian is named and accepts a newly sealed/independent corpus |

The test HMAC key belongs only to the offline fixture harness, is never accepted
from a case under test, and must not be described as production-grade asymmetric
identity. Tampered, unknown-issuer, expired, replay-conflicting, or malformed
envelopes fail closed.

## 4. Components and trust boundaries

### 4.1 Proposed package boundary

The future implementation should live in a standalone research package and
tools/tests dedicated to G4. It must not be imported by existing runtime,
retrieval, session assembly, SDK, service, or consumer modules.

Proposed components:

1. `fixture_identity` — verifies synthetic envelopes and emits validated context;
2. `scope_resolver` — derives exactly one tenant/session scope from trusted fixtures;
3. `fixture_stores` — loads immutable sessions, entitlements, descriptors, and policy;
4. `policy_evaluator` — returns G3 decisions without reading protected content;
5. `redaction_boundary` — applies and verifies permitted transforms;
6. `package_assembler` — emits authorized content/provenance or uniform denial;
7. `audit_sink` — writes only allowlisted content-free events;
8. `corpus_generator` — produces development cases and immutable manifests; and
9. `offline_gate` — validates behavior, isolation, determinism, and evidence.

### 4.2 Data-flow rules

- Caller request fields may narrow a derived scope but cannot widen it.
- Only fixture authorities can create validated principal, session, entitlement,
  and descriptor state.
- Policy evaluation reads descriptors, not protected artifact content.
- Denied candidate content and identifiers never enter the redactor, package,
  consumer error, general log, or audit record.
- The redactor receives only content already permitted subject to obligations.
- Package assembly verifies decision, policy, identity, entitlement, descriptor,
  transform, purpose, and expiry fingerprints.
- Evaluator labels are unavailable to all decision-path components.

## 5. Deterministic reference policy

The proposed evaluator implements G3 as an intersection:

```text
valid synthetic identity envelope
AND one active tenant membership
AND trusted session ownership/membership
AND scoped role assignment
AND matching operation and purpose entitlement
AND resource/subject relationship
AND artifact/source class permission
AND classification ceiling
AND temporal validity
AND satisfied obligations
AND no overriding denial
```

Any missing, unknown, ambiguous, stale, malformed, or conflicting input returns
`DENY`. Explicit denial overrides every grant. A role only selects candidate
grants and cannot independently permit disclosure.

Policy bundles, fixture snapshots, transforms, and schemas are versioned and
included in a composite run hash. The evaluator must be pure for fixed inputs,
time, and versions.

## 6. Redaction proposal

The first implementation should support a deliberately small transform set:

- remove a named structured field by schema path;
- replace a generator-labelled literal span with a fixed marker;
- minimize a permitted record set to a policy maximum;
- attach required citation/non-authoritative labels; and
- enforce package character/record budgets.

Every transform produces input/output digests and a content-free receipt.
Verification compares the output against generator-owned forbidden-field/span
labels and package obligations. Unknown transforms, overlapping/invalid spans,
verification failure, lineage mismatch, or budget failure become denial. Raw
pre-redaction content is never written to audit or diagnostics.

These transforms are reference semantics, not a production redaction or DLP
claim.

## 7. Audit and retention proposal

The local audit sink uses the G3 allowlist. It rejects unknown fields and scans
serialized output for synthetic secrets, protected text, raw query text,
credentials, artifact/source IDs, and scoring labels.

Retention classes:

| Class | Contents | Proposed retention |
|---|---|---|
| `g4_case_audit` | Content-free per-case decision events | 30 days locally; no longer than 14 days in CI artifacts |
| `g4_run_manifest` | Versions, hashes, counts, gate results | Retained with research record |
| `g4_failure_diagnostic` | Codes and digests only, no protected content | 30 days locally; no longer than 14 days in CI artifacts |

Cleanup is an offline artifact operation, not memory deletion evidence. Any
production audit retention, legal hold, deletion, or access model requires a
separate authority and review.

## 8. Deployment and rollback

The proposed deployment mode is a process-local CLI/library invoked only from
an explicit development command or focused tests. It has:

- no network access or hosted model dependency;
- no service route, listener, background worker, or SDK export;
- no import from a production entry point;
- no write to MNEMOS durable memory or shared cache;
- explicit input/output directories outside GateMem upstream; and
- deterministic offline execution.

Rollback disables/removes the standalone invocation and deletes its generated
working outputs after verifying paths remain within the declared output root.
Source/evidence removal is governed by normal repository change control, not an
automated destructive rollback. Because there is no runtime integration,
rollback cannot affect live requests or durable memory.

## 9. Development evidence plan

The synthetic corpus specification is
`docs/benchmarks/gatemem_g4_synthetic_development_corpus.md`.

Minimum evidence categories:

- authorized same-principal and properly delegated disclosure;
- cross-principal, cross-session, and cross-tenant denial;
- role-present but entitlement-absent denial;
- purpose, operation, class, relationship, and time-bound denial;
- explicit-deny precedence;
- incomplete/changed lineage denial;
- required redaction and redaction-verification failure;
- uniform denial/error non-disclosure;
- replay rejection after identity, entitlement, policy, descriptor, or transform drift;
- content-free audit/schema enforcement;
- evaluator-label isolation; and
- deterministic rerun equivalence.

All case-level results are development results. There is no generalization,
held-out, production-security, or GateMem performance claim.

## 10. Proposed implementation gates

A future G4 implementation is complete only if all of these pass:

1. package/import isolation from MNEMOS runtime and GateMem upstream;
2. no network, hosted model, durable memory, or shared-cache access;
3. all G3 contracts represented with strict schemas;
4. caller widening, role-only permits, and query-derived authority impossible;
5. explicit denial and fail-closed behavior verified;
6. denied content/identifiers absent from prompts, outputs, errors, logs, and audit;
7. redaction failures always deny;
8. replay drift matrix rejects stale packages;
9. provenance and decision fingerprints validate for every disclosure;
10. synthetic generator determinism and manifest hashes verify;
11. audit allowlist and retention metadata validate;
12. no GateMem inputs, predictions, scores, or claims produced;
13. no deletion state or deletion capability added; and
14. rollback/isolation procedure rehearsed without affecting runtime state.

Passing these gates would establish reference-contract mechanics only.

## 11. Sealed evaluation and future runtime blockers

G4 has no sealed evaluation corpus or custodian. Before any fresh performance
claim, a separate proposal must name an independent custodian, use the G3
preregistration template, freeze the policy artifact, and run once on a newly
sealed or independent corpus.

Before any runtime proposal, owners must separately choose and review the real
identity issuer, policy authority, entitlement source of truth, session
registry, artifact-classification migration, redaction service, audit platform,
retention/legal-hold policy, deployment topology, availability model, incident
response, and rollback semantics. Reference fixtures cannot silently become
production authorities.

## 12. Deletion separation

G4 does not authorize a deletion operation, tombstone, cascade, cache purge,
re-ingestion defense, negative verification, backup claim, or answer-layer
non-confirmation. Deletion remains blocked pending a separate durable-memory
ADR after authorization/disclosure has an accepted evidence path.

## 13. Advancement boundary

This proposal contains no implementation. Explicit acceptance may authorize the
bounded offline reference lane described here. Completion of that lane may
support review of a separate offline shadow-evaluation proposal only; it cannot
activate a runtime path.
