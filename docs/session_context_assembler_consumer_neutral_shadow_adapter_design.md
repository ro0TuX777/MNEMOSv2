# Consumer-Neutral Read-Only Shadow Adapter Design

Status: `CONTRACT_ACCEPTED / ISOLATED_SHADOW_IMPLEMENTATION_AUTHORIZED /
NO_RUNTIME_OR_CONSUMER_CONNECTION`.

Implementation status: `ISOLATED_SHADOW_IMPLEMENTATION_GATE_PASS` (local
in-process boundary only).

This contract is implemented only by the isolated local prototype under
`prototype/session_context_assembler/shadow_adapter/`. No route, SDK, network
listener, consumer connection, or production path exists.

## Architectural position

```text
MNEMOS
-> governed durable memory
-> session-context assembler
-> read-only context package
-> authorized consumer adapter
-> external application, agent, workflow, or operator interface
```

The session-context assembler remains a MNEMOS capability. An adapter only
transports a bounded, governed package to an authorized consumer; it neither
owns the assembler nor becomes a memory or authority layer. SAM is one
possible future consumer for testing only. It is not part of MNEMOS's core
architecture or product boundary.

## Generic request contract

```yaml
request:
  request_id: string
  current_task: string
  consumer_session_reference: opaque string
  eligible_context_scope:
    tenant_scope: opaque policy-bound identifier
    session_scope: opaque policy-bound identifier
    allowed_artifact_classes: list[string]
    eligibility_policy_id: string
  requested_budget:
    token_limit: integer
    budget_policy_id: string
  consumer_identity:
    consumer_id: string
    adapter_id: string
    purpose: string
  authorization_context:
    authorization_reference: opaque string
    permitted_operation: read_context_package
    expiry: timestamp or policy-defined lifetime
```

`consumer_session_reference` is correlation metadata, not an Engram ID and
not authority to enumerate a session. `eligible_context_scope` must already be
policy-bounded before selection. A requested budget is a ceiling request; the
effective budget may be lower under MNEMOS policy.

## Generic response contract

```yaml
response:
  request_id: string
  package_id: ephemeral string
  consumer_id: string
  adapter_contract_version: string
  issued_at: timestamp
  expires_at: timestamp
  package_digest:
    algorithm: sha256
    canonicalization: canonical-json-v1
    value: lowercase hex string
  context_package:
    selected_session_artifacts:
      - artifact_id: ephemeral string
        artifact_type: selected_session_segment | source_evidence | decision_artifact
        content: policy-filtered payload
        synthetic_context: true
        non_authoritative: true
        non_promotable: true
        parent_engram_ids: list[string]
        parent_source_ids: list[string]
        lineage_complete: boolean
    selection_metadata:
      selected_episode_ids: list[string]
      selection_rationale_codes: list[string]
  synthetic_context_labels:
    - artifact_id: string
      label: synthetic_context
  provenance_metadata:
    corpus_or_snapshot_reference: opaque string
    package_lineage_complete: boolean
    eligibility_policy_id: string
    disclosure_policy_id: string
    redaction_policy_id: string
  abstention_state:
    context_budget_insufficient: boolean
    omitted_required_artifact_types: list[string]
    selection_abstention_reason: string | null
  token_estimate: integer
  policy_identifiers:
    assembler_policy_version: string
    adapter_contract_version: string
    budget_policy_id: string
```

Provenance is artifact-local. Package-level provenance metadata describes the
common policy and snapshot boundary but never substitutes for each artifact's
parent IDs and `lineage_complete` flag. An artifact with incomplete lineage is
not delivered as a normal artifact; the request fails closed or produces an
explicit package-level abstention.

The response-level `synthetic_context_labels` list is an artifact-ID index for
consumer convenience. It does not carry or imply lineage; the artifact-local
parent fields remain authoritative for provenance inspection.

The response is ephemeral and read-only. `selection_rationale_codes` must not
expose scoring-only benchmark fields, hidden expected answers, or unrestricted
internal policy data.

### Package digest

`package_digest` covers the complete response payload except the
`package_digest` field itself. `canonical-json-v1` means UTF-8 JSON with object
keys sorted lexicographically, no insignificant whitespace, integers rendered
in base ten, and no NaN or infinity values. The digest is SHA-256 over those
canonical bytes. It detects alteration and supports audit correlation; it does
not make the package authoritative or durable memory.

## Authorization, scope, and disclosure boundary

Before assembly, the adapter must validate consumer identity, purpose,
authorization reference, tenant/session scope, permitted operation, and
expiry. Authorization is deny-by-default and limited to
`read_context_package`. The adapter must not broaden the eligible scope or
request retrieval outside it.

Before content is assembled or disclosed, policy evaluation must also
validate:

- tenant boundary and cross-tenant prohibition;
- session boundary and cross-session restrictions;
- artifact classification and permitted artifact classes;
- consumer clearance, entitlement, and declared purpose;
- mandatory redaction or field suppression;
- disclosure policy and data-residency constraints; and
- whether source or Engram identifiers themselves may be disclosed.

Redaction occurs before digest calculation and delivery. Redaction must not
remove provenance fields in a way that makes lineage appear complete. If
required provenance cannot be safely disclosed, the adapter fails closed with
`LINEAGE_INCOMPLETE` or a stricter policy error.

The consumer receives no authority to:

- write, update, or delete Engrams;
- alter governance, trust, authority, or contradiction state;
- create or modify Resolution Engrams;
- promote or persist synthetic context;
- alter retrieval ranking; or
- treat context-package content as source truth.

Consumers are contractually prohibited from treating packages as durable
memory. MNEMOS enforces short expiry, policy-bound delivery, audit correlation,
and read-only package semantics, and may require consumer retention
attestation. MNEMOS cannot guarantee deletion, non-retention, or non-copying
after content leaves the MNEMOS trust boundary. That external-system risk must
be explicit in any implementation threat model and consumer authorization.

## Request replay and idempotency

`request_id` is unique within `(consumer_id, adapter_contract_version)` for at
least the package lifetime plus the configured replay window.

- After revalidating current authorization, entitlement, and disclosure
  policy, an identical replay while the original package is unexpired returns
  the byte-equivalent cached response, including the same `package_id`,
  `issued_at`, `expires_at`, and digest. Revoked or expired authorization fails
  with `AUTHORIZATION_DENIED` even if a replay-cache entry exists.
- Reuse of a request ID with a different canonical request returns
  `REQUEST_REPLAY_CONFLICT` and no package content.
- Replay after expiry returns `PACKAGE_EXPIRED`; it does not silently assemble
  a new package under the old request ID.
- A consumer seeking fresh context must submit a new request ID.

The replay cache is an ephemeral delivery-control mechanism, not durable
memory. Its retention is bounded by the retention policy below.

## Shadow evaluation behavior

A future consumer-neutral technical shadow path, if separately authorized,
must remain off the live decision path. It may assemble and record technical
telemetry for comparison, but must not replace consumer context, affect an
answer, write memory, or mutate policy. Required telemetry is limited to
request/package correlation, latency, token estimate, budget/abstention state,
lineage completeness, policy versions, and delivery outcome. Context content
must not be copied into an ungoverned telemetry store.

## Failure and abstention contract

The adapter returns an explicit non-success or abstention when authorization
fails, scope is empty, disclosure is denied, required lineage is incomplete,
mandatory artifacts cannot fit, a package is expired, replay conflicts, or
policy/version compatibility fails. It must never silently degrade into an
unlabeled or apparently complete package.

Failure responses contain only a request correlation reference, a
non-sensitive error code, adapter-contract version, and optional safe retry
guidance. They must not expose raw policy rules, internal retrieval details,
undisclosed source identifiers, Engram identifiers, candidate counts, or
authorization internals.

```yaml
error_response:
  request_id: string
  adapter_contract_version: string
  error_code: enum
  retryable: boolean
  safe_retry_after: timestamp | null
```

Defined fail-closed codes:

```text
AUTHORIZATION_DENIED
SCOPE_EMPTY
DISCLOSURE_DENIED
REDACTION_REQUIRED
POLICY_VERSION_INCOMPATIBLE
LINEAGE_INCOMPLETE
BUDGET_INSUFFICIENT
PACKAGE_EXPIRED
REQUEST_REPLAY_CONFLICT
REPLAY_POLICY_MISMATCH
CONTRACT_VERSION_UNSUPPORTED
KILL_SWITCH_ACTIVE
```

`BUDGET_INSUFFICIENT` may accompany a safely constructed abstention package
only when policy permits disclosure of that partial package. Otherwise it is a
content-free failure. Error text must not reveal omitted IDs or hidden policy
criteria.

## Versioning and compatibility

The request and response carry an adapter-contract version and assembler-policy
version. Unknown major versions fail closed. A consumer cannot request removal
of artifact-local provenance, synthetic labels, non-authoritative status,
integrity fields, or abstention state. Consumer-specific convenience mappings
belong outside this core contract and require separate review.

## Rollback and data retention

Any future shadow implementation must have a single control that disables new
assembly and delivery without changing MNEMOS retrieval, memory, governance,
or consumer active-context behavior. Rollback must:

1. stop accepting new adapter requests;
2. stop shadow delivery and comparison telemetry;
3. invalidate unexpired replay-cache entries where technically possible;
4. retain only the minimum audit metadata required by approved policy;
5. leave Engrams, source evidence, governance state, ranking, and consumer
   active context untouched; and
6. record adapter/policy versions and the rollback reason without copying
   package content into the audit record.

Retention classes are explicit:

- package content and replay cache: no longer than `expires_at` plus a narrowly
  approved delivery-retry grace period;
- comparison telemetry: content-free and retained under a named telemetry
  retention policy;
- audit correlation: request ID, package ID, consumer ID, digest, timestamps,
  policy/version identifiers, outcome, and error code only;
- raw selected content: prohibited from general telemetry and logs.

An implementation cannot recall or delete copies already made outside the
MNEMOS trust boundary. Consumer contracts and attestations must address that
residual risk; expiry is a usage-policy boundary, not proof of remote erasure.

## Isolated implementation boundary

ADR 0008 authorizes only the local isolated adapter mechanics and tests. The
implementation adds no consumer-specific code, SDK behavior, runtime route,
network endpoint, consumer connection, production wiring, memory write,
retrieval change, or governance/promotion behavior. Any expansion requires a
new authorization or ADR amendment.

## Advancement boundary

A Phase 5A PASS authorizes a separate proposal for a read-only,
consumer-neutral technical shadow adapter.

It does not authorize live routing, production use, memory writes, governance
mutation, promotion behavior, or a human-value claim.
