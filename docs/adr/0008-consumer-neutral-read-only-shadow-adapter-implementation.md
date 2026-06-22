# ADR 0008: Consumer-Neutral Read-Only Shadow Adapter Implementation

Date: 2026-06-22

Status: Accepted — isolated shadow implementation only

## Context

The session-context assembler is a MNEMOS capability that produces bounded,
provenance-labeled, non-authoritative context packages. Phase 5A technical
verification passed on the held-out R2 corpus, and the consumer-neutral adapter
contract now defines artifact-local lineage, package integrity, replay
semantics, policy-bound disclosure, structured failure, retention, and
rollback requirements.

The next engineering question is whether those requirements can be implemented
in an isolated technical shadow without creating a consumer-specific product,
a second memory authority, or a path that affects a consumer's active context.
This ADR is the proposed authorization boundary for that implementation. Its
presence in the repository does not authorize work until the ADR is explicitly
accepted.

## Decision

If accepted, this ADR authorizes only a local or otherwise isolated,
consumer-neutral, read-only shadow adapter implementation with:

- authenticated request validation and deny-by-default authorization;
- tenant, session, artifact-class, entitlement, disclosure, and redaction
  enforcement before package assembly;
- invocation of the existing session-context assembler without changing
  retrieval or selection authority;
- artifact-local provenance and synthetic-context labeling;
- canonical response construction and SHA-256 package digesting;
- policy-pinned replay-cache handling;
- structured, non-sensitive fail-closed responses;
- content-free technical telemetry and audit correlation; and
- a kill switch that disables assembly and delivery without disturbing MNEMOS
  state or a consumer's active context path.

The implementation remains a technical shadow. It may compare what would have
been delivered, but it must not supply or replace live consumer context, affect
an answer or workflow decision, or become a dependency of a consumer request.

SAM may be used later as one example test consumer under separate approval. It
is not the architectural owner, default runtime, or product identity of the
assembler or adapter.

## Architectural Boundary

```text
MNEMOS governed durable memory
-> session-context assembler
-> read-only context package
-> consumer-neutral shadow adapter
-> isolated shadow sink / comparison telemetry

separate and unchanged:
consumer active context path
```

The shadow adapter receives no write-capable MNEMOS handle. Its permitted
operation is exactly `read_context_package` over an already authorized scope.

## Authorized Components

An accepted implementation may contain only the following bounded components:

1. **Request validator** — validates contract version, request shape,
   consumer identity, declared purpose, authorization context, expiry, and
   requested budget.
2. **Scope/disclosure policy adapter** — resolves the effective tenant/session
   scope, allowed artifact classes, clearance, redaction, and disclosure
   decision without exposing raw policy internals.
3. **Assembler invocation boundary** — passes only approved runtime inputs to
   the existing assembler and receives an ephemeral package.
4. **Response builder** — attaches artifact-local lineage, policy/version
   identifiers, abstention state, issuance/expiry metadata, and the canonical
   digest.
5. **Replay controller** — provides bounded idempotency under the policy
   fingerprint rules below.
6. **Shadow sink** — receives delivery outcome and content-free comparison
   telemetry only. It is not a live consumer route.
7. **Kill switch** — atomically prevents new shadow assembly and delivery.

No public SDK, network API, consumer-specific adapter, or production deployment
is authorized by this ADR draft.

## Replay Policy Pinning

A cached response may be replayed only after current authorization has been
revalidated and every issuance fingerprint still matches.

The issuance fingerprint covers canonical, non-secret identifiers or digests
for:

```text
consumer identity and adapter identity
authorization grant and declared purpose
tenant/session scope and allowed artifact classes
entitlement or clearance decision
disclosure policy and decision
redaction policy and applied redaction profile
eligibility policy
budget policy and effective budget
assembler policy/version
adapter contract version
canonical request payload
```

The fingerprint must not store bearer tokens, credentials, raw policy content,
or undisclosed artifact identifiers.

Replay behavior is fail-closed:

- exact request and exact current fingerprint, while unexpired: return the
  byte-equivalent cached response;
- revoked or expired authorization: `AUTHORIZATION_DENIED`;
- any fingerprint mismatch after policy, entitlement, disclosure, redaction,
  assembler-policy, or contract-version change: `REPLAY_POLICY_MISMATCH`;
- same request ID with a different canonical request: `REQUEST_REPLAY_CONFLICT`;
- replay after package expiry: `PACKAGE_EXPIRED`;
- cache miss: do not reconstruct an old package under the reused request ID;
  require a fresh request ID.

The cache key includes at least consumer ID, adapter ID, request ID, and
contract major version. Cache entries expire no later than the package plus an
explicitly approved delivery-retry grace period.

## Transport Authenticity and Delivery Binding

The package digest detects payload alteration but does not authenticate the
sender or delivery channel.

Any future networked consumer path requires a separately reviewed authenticated
transport consistent with MNEMOS's approved authentication posture. The exact
mechanism may be selected during implementation review, but it must provide:

- authenticated peer identity;
- confidentiality and integrity in transit;
- replay-resistant channel/session semantics;
- verification that the authenticated peer matches `consumer_id`; and
- an auditable binding between consumer ID, adapter ID, request ID, package ID,
  package digest, authenticated channel/session identifier, and delivery time.

Credentials and raw channel secrets must never enter package telemetry. A
local isolated implementation must use an explicit authenticated test identity
and transport abstraction; it may not bypass identity validation merely because
the process is local.

No network listener or external consumer connection is authorized by this ADR
draft. Enabling either requires an amendment or separate ADR after the local
shadow implementation passes its gates.

## Package and Error Contract

The implementation must conform to
`docs/session_context_assembler_consumer_neutral_shadow_adapter_design.md`.
Every selected artifact independently carries:

```text
artifact_id
artifact_type
synthetic_context
non_authoritative
non_promotable
parent_engram_ids
parent_source_ids
lineage_complete
```

The complete canonical response, excluding the digest field itself, is hashed
as specified by `canonical-json-v1`. Digest verification is mandatory at the
shadow sink.

Errors and abstentions use the contract's non-sensitive codes, including
`AUTHORIZATION_DENIED`, `SCOPE_EMPTY`, `DISCLOSURE_DENIED`,
`LINEAGE_INCOMPLETE`, `BUDGET_INSUFFICIENT`, `PACKAGE_EXPIRED`,
`REPLAY_POLICY_MISMATCH`, `REQUEST_REPLAY_CONFLICT`, and
`CONTRACT_VERSION_UNSUPPORTED`. `KILL_SWITCH_ACTIVE` is returned when the local
shadow boundary is disabled. Errors must not reveal raw policy, candidate,
retrieval, source, Engram, or authorization details.

## Telemetry and Data Retention

Permitted telemetry is content-free:

```text
request/package/consumer correlation identifiers
package digest
adapter, assembler, policy, and contract versions
latency and token estimate
budget and abstention state
lineage-complete boolean
error/outcome code
kill-switch state
authenticated delivery-channel correlation identifier
```

Raw task text, selected content, source text, Engram content, authorization
credentials, and raw policy decisions are prohibited from general telemetry.
Package content and replay-cache entries expire under the bounded retention
rules in the contract. Audit correlation may outlive package content only under
an approved named retention policy.

MNEMOS cannot guarantee deletion after content leaves its trust boundary.
Consumer retention obligations and attestations remain required even in a
future authenticated path.

## Kill Switch and Rollback

The implementation must default to disabled. The kill switch must prevent new
assembly, cache insertion, shadow delivery, and comparison telemetry while
leaving MNEMOS retrieval, Engrams, governance, contradiction state, ranking,
and the consumer active path unchanged.

Rollback must:

1. activate the kill switch;
2. invalidate unexpired replay-cache entries where technically possible;
3. stop shadow sinks and background comparison work;
4. preserve only approved content-free audit correlation;
5. prove no consumer active-path dependency exists; and
6. require explicit reauthorization before re-enablement.

## Explicitly Prohibited

This ADR draft does not authorize:

- live consumer routing or modification of consumer context;
- public or internal SDK release;
- a network endpoint or external consumer connection;
- production, staging-live, or canary deployment;
- Engram or Resolution Engram writes, updates, or deletion;
- governance, trust, authority, contradiction, or promotion mutation;
- retrieval-query expansion or retrieval-ranking change;
- persistence or promotion of synthetic context;
- content-bearing telemetry or logging;
- treating a package as source truth or durable memory;
- consumer-specific behavior in the core adapter; or
- a human-value, productivity, usability, or production-readiness claim.

## Implementation Acceptance Gates

Before any later proposal may request an authorized consumer-neutral shadow
evaluation,
the isolated implementation must demonstrate:

```text
authorization/scope/disclosure/redaction denial tests pass
artifact-local lineage preservation = 1.0
package digest reproducibility and tamper detection = 1.0
policy-fingerprint replay mismatch rejection = 1.0
request replay conflict and expiry rejection = 1.0
structured error leakage count = 0
budget compliance = 1.0
silent required-artifact omission count = 0
content-bearing telemetry events = 0
unauthorized memory/governance/retrieval mutations = 0
kill-switch and rollback tests pass
consumer active-path effects = 0
```

Mutation tests must prove the gates fail if lineage, digest verification,
policy pinning, authorization, redaction, abstention, telemetry filtering, or
the kill switch is bypassed.

## Alternatives Considered

- **Consumer-specific implementation first.** Rejected because it would make
  one consumer appear to own a MNEMOS capability and would weaken portability.
- **Digest-only delivery security.** Rejected because integrity correlation is
  not sender or channel authenticity.
- **Replay by request ID alone.** Rejected because changed authorization or
  policy could make a formerly valid package inappropriate.
- **Direct live shadow injection into consumer context.** Rejected because it
  would affect the active path before technical isolation is proven.
- **Persist packages for later analysis.** Rejected because it would create an
  ungoverned secondary memory store.

## Consequences

The proposed implementation is intentionally narrow and operationally
conservative. It adds policy and audit complexity, but preserves MNEMOS as the
only memory authority and allows technical behavior to be tested without
consumer-path impact.

## Approval and Advancement Boundary

This ADR is **Accepted** for the isolated implementation defined above only.
It does not authorize integration or connection. Passing the isolated
implementation's gates would authorize review of a separate
authorized consumer-neutral shadow-evaluation proposal—not live routing, SDK release,
production deployment, writes, retrieval changes, authority/governance
mutation, or human-value claims.

## Evidence

- `docs/adr/0007-session-context-assembler-shadow-only.md`
- `docs/session_context_assembler_spec.md`
- `docs/session_context_assembler_consumer_neutral_shadow_adapter_design.md`
- `docs/session_context_assembler_phase_5a_protocol.md`
- `docs/session_context_assembler_phase_5a_notes.md`
- `benchmarks/results/session_context_assembler_r2_verification.md`
- `tests/test_session_context_assembler_phase_5a.py`
- `prototype/session_context_assembler/shadow_adapter/`
- `tests/test_session_context_assembler_shadow_adapter.py`
- `tools/run_session_context_assembler_shadow_adapter_gate.py`
- `benchmarks/results/session_context_assembler_shadow_adapter_gate.json`
- `benchmarks/results/session_context_assembler_shadow_adapter_gate.md`
- `docs/session_context_assembler_shadow_adapter_implementation_notes.md`
