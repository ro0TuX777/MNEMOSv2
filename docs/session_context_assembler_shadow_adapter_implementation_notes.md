# Consumer-Neutral Shadow Adapter — Isolated Implementation Notes

Status: `ISOLATED_SHADOW_IMPLEMENTATION_GATE_PASS`.

ADR 0008's bounded local implementation exists under:

```text
prototype/session_context_assembler/shadow_adapter/
```

It is an in-process technical shadow only. There is no listener, external
consumer connection, live routing, SDK surface, deployment configuration,
durable-memory write, retrieval change, or governance/authority mutation.

## Implemented components

- strict request and authenticated local-transport validation;
- policy/scope/disclosure/redaction boundary;
- whitelisted S1 assembler invocation boundary;
- artifact-local response and canonical SHA-256 digest construction;
- request and policy-fingerprint-pinned in-memory replay controller;
- content-free local shadow sink;
- atomic kill switch and cache-invalidating rollback; and
- structured, non-sensitive fail-closed errors.

The policy fingerprint covers the canonical request, authorization grant
fingerprint, consumer/adapter identity, tenant/session scope, allowed artifact
and provenance IDs, denial/redaction state, filtered eligible-content digest,
snapshot reference, budget, disclosure/redaction/eligibility policy IDs,
assembler policy version, and adapter contract version.

## Acceptance result

Ten frozen R2 cases pass the isolated adapter gate:

```text
package assembly                              10/10
canonical digest verification                 10/10
artifact-local lineage/labels                  10/10
budget compliance                              10/10
content-free telemetry                         10/10
shadow-only result marking                     10/10
fixed-seed determinism                         PASS
network/runtime authority import paths         0
R1 and R2 manifest integrity                   PASS
```

Mutation checks detect digest tampering, parent-source removal, authorization
and redaction bypass, abstention suppression, telemetry allowlist escape,
kill-switch bypass, and replay-policy drift. The first
acceptance test proves an active kill switch prevents assembly, cache writes,
sink events, and delivery accounting while leaving external core-state
sentinels unchanged. Mid-assembly activation discards the package before any
cache, sink, or delivery side effect.

## Security and operational limitations

The transport context is an explicit local authenticated-transport
abstraction, not a network authentication mechanism. No networked transport is
implemented or authorized. A future proposal must select and review the actual
authentication/channel-binding mechanism before any connection exists.

Policy decisions are injected as immutable local snapshots for testing. This
prototype does not connect to live authorization, disclosure, redaction,
retrieval, or governance services. It proves adapter mechanics and fail-closed
boundaries only.

The replay cache and sink are process-local and ephemeral. No persistence is
implemented. MNEMOS still cannot guarantee deletion after content leaves its
trust boundary; no content leaves that boundary in this implementation.

## Advancement boundary

This PASS authorizes review of a separate authorized consumer-neutral
shadow-evaluation proposal only. It does not authorize a network listener,
external consumer connection, live routing, SDK release, deployment, memory
writes, retrieval changes, governance mutation, or a human-value claim.
