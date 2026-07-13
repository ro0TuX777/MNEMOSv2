# GateMem G3 Principal-Bound Authorization and Disclosure Semantics

Date: 2026-06-24

Status: `GATEMEM_G3_AUTHORIZATION_DISCLOSURE_SEMANTICS_DESIGN_COMPLETE`

Classification: **Research design only. No implementation or performance
claim.**

## 1. Authorization boundary

```text
authenticated channel and credential
        |
        v
validated principal context
        |
        v
identity-derived tenant and session scope
        |
        v
scoped role assignment + entitlement snapshot
        |
        v
artifact/source-class authorization
        |
        v
disclosure obligations and redaction
        |
        v
bounded disclosed package or content-free denial
        |
        v
content-free audit correlation
```

The authorization service is conceptually upstream of content delivery. It is
not a ranking feature, query classifier, or post-answer safety check.

## 2. Design goals

G3 must support a future system that can:

- bind every request to a cryptographically authenticated principal;
- derive tenant and session membership from trusted identity state;
- evaluate roles only within explicit scope and validity intervals;
- require an applicable entitlement, relationship grant, or resource ownership
  rule in addition to role;
- restrict artifact and source classes before content leaves a trusted boundary;
- express field-, span-, record-, and package-level disclosure obligations;
- deny or redact deterministically and fail closed;
- preserve provenance for every disclosed artifact;
- prevent denied content, identifiers, and policy internals from leaking through
  prompts, responses, errors, caches, metrics, or logs; and
- produce sufficient content-free audit evidence to reconstruct why a decision
  was made without reconstructing the protected content itself.

## 3. Non-goals

G3 does not select or implement:

- an identity provider or authentication protocol;
- a policy language, engine, database, or network API;
- a production role matrix or entitlement catalogue;
- retrieval weights, redaction patterns, policy thresholds, or benchmark logic;
- MNEMOS runtime, service, SDK, or consumer changes;
- memory deletion, tombstones, cascade, purge, or non-confirmation;
- a new GateMem run or performance target; or
- legal or regulatory compliance guarantees.

## 4. Threat model

The future decision model must fail safely under:

| Threat | Required response |
|---|---|
| Caller supplies another tenant/session in filters | Ignore as authority; derive scope from trusted identity and deny mismatch |
| Role inflation or stale role token | Validate assignment, issuer, scope, version, issue/expiry, and revocation state |
| Query claims urgency, consent, relationship, or professional status | Treat as untrusted text; never grant authority from wording |
| Confused-deputy consumer uses its own broad token for another principal | Require end-principal delegation and purpose binding |
| Cross-principal candidate enters a shared retrieval pool | Apply authorized prefilter and mandatory candidate-level post-check before disclosure |
| Unknown artifact/source class | Deny by default |
| Derived or synthetic artifact loses parent restrictions | Compute effective restriction from lineage; deny incomplete lineage |
| Redaction transform fails or policy is ambiguous | Deny; never return the original content |
| Denial reveals that a protected record exists | Return a content-free response indistinguishable from other denials where policy requires |
| Cached response is replayed after entitlement drift | Revalidate and compare policy/entitlement fingerprints; reject mismatch |
| Telemetry or error captures raw content | Enforce an allowlisted content-free schema |
| Evaluator annotation reaches policy logic | Reject schema; preserve G1 clean-input isolation |

## 5. Normative contracts

The following schemas are language-neutral design contracts, not executable
models.

### 5.1 Validated principal context

Produced only after credential and channel validation:

```yaml
validated_principal_context:
  principal_id: stable opaque identifier
  identity_issuer: trusted issuer identifier
  credential_fingerprint: non-secret digest
  authentication_time: timestamp
  authentication_strength: named assurance class
  channel_binding: non-secret channel/session digest
  tenant_memberships:
    - tenant_id: trusted tenant identifier
      membership_id: opaque reference
      status: active | suspended | revoked
      valid_from: timestamp
      valid_until: timestamp | null
  scoped_role_assignments:
    - role_id: policy-registry identifier
      tenant_id: trusted tenant identifier
      resource_scope: structured scope reference
      assignment_reference: opaque authority reference
      valid_from: timestamp
      valid_until: timestamp | null
  delegation:
    delegating_principal_id: opaque identifier | null
    delegation_reference: opaque authority reference | null
    permitted_operations: list
    permitted_purposes: list
    expires_at: timestamp | null
  identity_snapshot_version: version
  expires_at: timestamp
```

Consumer-provided `principal_id`, `tenant_id`, roles, or delegation fields do
not satisfy this contract.

### 5.2 Derived request scope

```yaml
derived_request_scope:
  request_id: unique identifier
  principal_id: from validated context
  tenant_id: resolved from active membership
  session_id: resolved through trusted session registry
  operation: read_context | retrieve_evidence | inspect_provenance
  declared_purpose: policy-registry identifier
  consumer_id: authenticated consumer identifier
  adapter_id: reviewed adapter identifier
  requested_artifact_classes: list
  requested_source_classes: list
  requested_budget: bounded integer
  issued_at: timestamp
  expires_at: timestamp
```

Scope derivation rules:

1. resolve one active tenant membership applicable to the consumer and request;
2. reject ambiguous, absent, suspended, expired, or revoked membership;
3. resolve the session from a trusted registry and verify tenant ownership;
4. verify that the principal or a valid delegation may access the session;
5. intersect requested operation, purpose, classes, and budget with entitlement
   maxima; and
6. reject any request field that attempts to broaden the derived scope.

Cross-tenant union scope is prohibited. A request spanning multiple tenants
must be split into independently authorized requests.

### 5.3 Entitlement snapshot

```yaml
entitlement_snapshot:
  snapshot_id: opaque identifier
  principal_id: opaque identifier
  tenant_id: trusted tenant identifier
  policy_version: immutable version
  grants:
    - grant_id: opaque reference
      operations: list
      purposes: list
      resource_scope: structured scope
      subject_relationships: list
      artifact_classes: list
      source_classes: list
      classification_ceiling: named level
      obligations: list of obligation identifiers
      valid_from: timestamp
      valid_until: timestamp | null
  denials:
    - denial_id: opaque reference
      precedence: integer
      matching_scope: structured scope
      reason_code: stable internal code
  snapshot_fingerprint: sha256
  issued_at: timestamp
  expires_at: timestamp
```

Explicit denial overrides grant. Absence of a matching grant is denial. A role
may select candidate grants but cannot independently permit disclosure.

### 5.4 Content-free artifact descriptor

Authorization evaluates descriptors before protected content is released:

```yaml
artifact_descriptor:
  artifact_id: opaque identifier
  tenant_id: trusted tenant identifier
  session_id: trusted session identifier | null
  artifact_class: raw_evidence | session_turn | summary | resolution | derived | synthetic_context
  source_class: user_input | document | api | system | derived
  subject_principal_ids: opaque identifiers
  owner_principal_id: opaque identifier | null
  classification: named level
  purpose_tags: policy identifiers
  parent_artifact_ids: opaque identifiers
  parent_source_ids: opaque identifiers
  lineage_complete: boolean
  synthetic: boolean
  authoritative: boolean
  policy_labels: list
  descriptor_version: version
```

Unknown classes or incomplete lineage fail closed. Derived, summary,
resolution, and synthetic artifacts inherit the most restrictive applicable
parent scope unless an independently authorized source re-grounding process
establishes otherwise.

### 5.5 Authorization decision

```yaml
authorization_decision:
  decision_id: unique identifier
  request_id: request identifier
  artifact_id: opaque identifier
  outcome: PERMIT | PERMIT_WITH_REDACTION | DENY
  permitted_operation: identifier | null
  effective_purpose: identifier | null
  obligations:
    - obligation_id: policy identifier
      type: redact_fields | redact_spans | minimize_records | citation_required | no_cache | expiry | non_authoritative_label
      parameters_digest: sha256
  internal_reason_codes: list
  policy_version: immutable version
  identity_snapshot_version: version
  entitlement_snapshot_id: opaque reference
  decision_fingerprint: sha256
  issued_at: timestamp
  expires_at: timestamp
```

Only stable, non-sensitive external outcome codes cross to a consumer. Internal
reason codes and policy structure remain inside the authorization/audit trust
boundary.

### 5.6 Disclosed package or denial

```yaml
disclosure_result:
  request_id: request identifier
  outcome: DISCLOSED | DISCLOSED_WITH_REDACTION | DENIED
  disclosed_artifacts:
    - artifact_id: opaque identifier
      content: authorized or transformed content
      parent_source_ids: authorized provenance
      authorization_decision_id: reference
      redaction_receipt_id: reference | null
  omitted_artifact_count: non-negative integer
  package_digest: sha256
  policy_version: immutable version
  issued_at: timestamp
  expires_at: timestamp
  external_reason_code: stable non-sensitive code | null
```

A denied result contains no protected content, candidate IDs, source IDs,
subject IDs, policy internals, or counts when those counts would reveal
existence. Policy may require a uniform response across “not found,” “outside
scope,” and “denied.”

## 6. Decision semantics

### 6.1 Evaluation order

1. validate the authenticated principal context and channel binding;
2. derive tenant/session scope from trusted identity state;
3. validate operation, purpose, consumer, adapter, time, and budget;
4. load an immutable entitlement snapshot;
5. construct authorized retrieval prefilters from scope and class ceilings;
6. retrieve candidate descriptors inside the trusted boundary;
7. evaluate every candidate descriptor independently;
8. remove denied candidates before content reaches any answer model;
9. apply all obligations in a reviewed redaction/minimization boundary;
10. verify transformed content, provenance, lineage, labels, budget, and expiry;
11. assemble and digest the package; and
12. emit content-free decision and audit events.

Prefiltering limits exposure and cost. Candidate-level post-checking is still
mandatory defense in depth. Post-answer filtering is too late.

### 6.2 Role and entitlement

The decision is an intersection, never a union:

```text
authenticated membership
AND scoped role assignment
AND operation/purpose entitlement
AND resource or subject relationship
AND artifact/source class permission
AND classification ceiling
AND temporal validity
AND satisfied disclosure obligations
AND no overriding denial
```

Failure or indeterminacy in any term yields `DENY`.

### 6.3 Purpose binding

Purpose is a registry identifier authorized by entitlement. Free-form purpose
text is informational only and cannot grant access. A package authorized for
one purpose cannot be replayed for another.

### 6.4 Redaction

Redaction is a policy obligation, not a best-effort string replacement.

The future redaction boundary must:

- use versioned structural and span transforms;
- process only content already eligible for redacted disclosure;
- retain an internal transformation receipt and input/output digests;
- verify that prohibited fields/spans are absent after transformation;
- prevent redacted source material from reappearing in summaries, citations,
  metadata, answer structure, or prompt context;
- preserve authorized provenance without revealing denied sources; and
- convert any transform, verification, or lineage failure into `DENY`.

### 6.5 Batch and inference safety

Authorization is per artifact. Package assembly may disclose permitted items
and report a policy-approved generic omission state, but it must not reveal the
identity or protected attributes of denied candidates.

Repeated queries, yes/no confirmation, counts, ordering, latency, cache hits,
and error differences are disclosure channels. The future implementation must
define response normalization, timing/caching boundaries, and rate controls.

## 7. Error model

External errors are stable and non-sensitive:

```text
AUTHENTICATION_REQUIRED
AUTHORIZATION_DENIED
REQUEST_SCOPE_INVALID
POLICY_VERSION_UNSUPPORTED
REDACTION_UNAVAILABLE
PACKAGE_EXPIRED
REPLAY_CONFLICT
SERVICE_UNAVAILABLE
```

Internal audit reason codes may be more specific:

```text
TENANT_MEMBERSHIP_MISSING
TENANT_MEMBERSHIP_REVOKED
SESSION_TENANT_MISMATCH
SESSION_ACCESS_DENIED
ROLE_ASSIGNMENT_MISSING
ROLE_ASSIGNMENT_EXPIRED
ENTITLEMENT_MISSING
PURPOSE_DENIED
OPERATION_DENIED
SUBJECT_RELATIONSHIP_DENIED
ARTIFACT_CLASS_DENIED
SOURCE_CLASS_DENIED
CLASSIFICATION_EXCEEDED
LINEAGE_INCOMPLETE
OVERRIDING_DENIAL
REDACTION_REQUIRED
REDACTION_VERIFICATION_FAILED
POLICY_INDETERMINATE
```

External messages do not reveal which internal reason occurred unless a policy
explicitly authorizes that disclosure.

## 8. Replay and cache rules

A response may be replayed only when all of the following still match:

- canonical request digest;
- principal, tenant, session, consumer, adapter, operation, and purpose;
- identity and entitlement snapshot fingerprints;
- policy and redaction versions;
- artifact descriptor and eligible-content digests;
- obligations and requested budget;
- authorization and package expiry; and
- current revocation state.

Policy, entitlement, role, tenant membership, session membership, artifact
classification, lineage, or redaction drift invalidates replay. Consumers may
not extend expiry or remove obligations.

## 9. Content-free audit model

The audit event uses a strict allowlist:

```yaml
authorization_audit_event:
  event_id: unique identifier
  request_id: request identifier
  decision_id: decision identifier | null
  principal_id_digest: keyed digest
  tenant_id_digest: keyed digest
  session_id_digest: keyed digest | null
  consumer_id: reviewed identifier
  adapter_id: reviewed identifier
  operation: identifier
  purpose: identifier
  outcome: PERMIT | PERMIT_WITH_REDACTION | DENY | ERROR
  artifact_class_counts: content-free aggregate
  source_class_counts: content-free aggregate
  disclosed_count: integer
  redacted_count: integer
  denied_count: integer or policy-redacted
  internal_reason_codes: list
  policy_version: immutable version
  identity_snapshot_version: version
  entitlement_snapshot_id: opaque reference
  request_digest: sha256
  decision_fingerprint: sha256 | null
  package_digest: sha256 | null
  event_time: timestamp
  retention_class: policy identifier
```

Prohibited audit content includes raw queries, raw memory, answers, redacted
spans, candidate/source IDs, credentials, tokens, grant contents, relationship
details, and GateMem scoring annotations.

Audit access is itself entitlement-controlled. Digests must be keyed where
plain hashes could enable dictionary attacks against predictable identifiers.

## 10. Development/evaluation separation

### 10.1 Current corpus status

The released medical, office, education, and household data were all processed
and scored during G2/G2A. Their aggregate and per-domain outcomes are known.
They remain useful for:

- frozen historical comparison;
- regression detection;
- tooling and schema tests; and
- explicitly labeled retrospective analysis.

They are not eligible for a future “untouched held-out” claim after policy
work. Creating a deterministic split now does not erase prior observation.

### 10.2 Development corpus

Policy design and tuning must use a declared development corpus with:

- original MNEMOS-authored synthetic authorization cases or an explicitly
  designated development release;
- principal, tenant, session, relationship, role, entitlement, artifact/source
  class, and redaction-obligation annotations;
- adversarial confused-deputy, role-inflation, cross-tenant, replay-drift,
  existence-inference, and redaction-bypass cases;
- no GateMem scorer-only fields in policy inputs;
- an immutable manifest, generator version, license, and hashes; and
- explicit permission to inspect failures and tune policy.

### 10.3 Held-out evaluation corpus

A fresh claim requires a newly sealed or independent GateMem-compatible corpus:

- created or held by an evaluation custodian independent of policy tuning;
- inaccessible to policy developers before policy/code freeze;
- stored separately from the development workspace;
- pinned by a coordinator-visible commitment/hash before the run;
- revealed only to the evaluator process;
- projected through the clean G1 boundary;
- joined to scoring annotations only after predictions are frozen; and
- evaluated once under a preregistered protocol.

If a newly sealed corpus cannot be obtained, future work may report development
or retrospective regression results only—never a fresh held-out benchmark
claim.

### 10.4 Role separation

| Role | Permitted access |
|---|---|
| Policy developer | Development cases, aggregate historical baselines, no sealed evaluation content/labels |
| Evaluation custodian | Sealed corpus, manifests, scorer, frozen policy artifact |
| Release reviewer | Preregistration, hashes, aggregate report, exception log |
| Runtime operator | Out of scope until a separate implementation authorization |

### 10.5 Freeze and one-shot rule

Before unsealing, freeze:

- repository commit and dependency environment;
- policy code, data, versions, role/entitlement schema, and redaction rules;
- model/provider configuration if any future evaluation uses one;
- thresholds, budgets, seeds, and retry behavior;
- projection, prediction, and scorer contract versions;
- primary, secondary, and excluded metrics;
- statistical analysis and missing/error handling;
- allowed claims and stop/invalid-run criteria; and
- all artifact paths and hashes.

After unsealing, any policy change creates a new development iteration and
requires a new untouched evaluation corpus for a fresh claim. Rerunning the
same sealed corpus after tuning is retrospective, not held-out.

The normative template is
`docs/benchmarks/gatemem_g3_preregistration_template.md`.

## 11. Deletion separation

Authorization may eventually determine whether a principal is permitted to
request deletion. That decision is not deletion itself.

G3 defines no mutation of `deletion_state`, no tombstone, and no proof of
non-recoverability. The later deletion ADR must define:

- authenticated deletion-request authority and target resolution;
- durable tombstone and lifecycle transitions;
- lineage cascade across raw, summary, resolution, derived, graph, cache, and
  extracted artifacts;
- cache invalidation and replay rejection;
- re-ingestion and rederivation defenses;
- cross-tier, replica, backup, and external-copy boundaries;
- negative verification and recurring sweep behavior; and
- answer-layer reconstruction and non-confirmation semantics.

Authorization and deletion may integrate later through reviewed interfaces,
but their state machines, evidence, rollback, and claims remain separate.

## 12. Design acceptance gates

G3 design is complete only when:

- principal, tenant/session, role, entitlement, descriptor, decision,
  disclosure, replay, error, and audit contracts are explicit;
- scope is identity-derived and caller widening is prohibited;
- roles alone cannot permit disclosure;
- unknown or indeterminate policy state fails closed;
- redaction failure becomes denial;
- denied content and existence signals are bounded;
- content-free audit fields and prohibited fields are enumerated;
- development and newly sealed evaluation corpora are separated;
- the already-observed G2A corpus is excluded from fresh held-out claims;
- preregistration and one-shot rules are explicit;
- deletion is assigned to a separate future ADR; and
- no runtime, policy, benchmark, or deletion implementation is added.

## 13. Advancement boundary

This specification authorizes no code. A future implementation proposal must
select concrete authorities and trust boundaries, define rollback and
retention, include mutation/adversarial tests, and obtain separate approval.

