# ADR 0013: GateMem G4 Proposes an Offline Authorization Reference Implementation

Date: 2026-06-24

Status: Accepted — local offline reference implementation complete

## Context

G3 is accepted as the normative authorization/disclosure design. It establishes
that disclosure depends on authenticated identity, identity-derived scope,
scoped role assignment, entitlement, artifact/source classification, purpose,
time validity, and satisfied obligations. G3 intentionally does not choose
operational authorities or authorize code.

The next safe step is to propose a local reference implementation that can make
those contracts executable against MNEMOS-owned synthetic development cases.
It must produce implementation evidence without changing a live MNEMOS path or
using the already-observed GateMem domains as a tuning or fresh-evaluation set.

## Proposal status

This ADR was proposed under:

```text
GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_PROPOSAL_COMPLETE
PROPOSAL_ONLY
NO_IMPLEMENTATION_AUTHORIZED
NO_RUNTIME_RETROFIT
NO_GATEMEM_RUN
NO_DELETION_ENGINEERING
```

The detailed proposal is
`docs/benchmarks/gatemem_g4_offline_reference_implementation_proposal.md`.

## Accepted authorization

The proposal was accepted under:

```text
GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_AUTHORIZED
LOCAL_OFFLINE_ONLY
MNEMOS_OWNED_SYNTHETIC_DEVELOPMENT_CASES_ONLY
NO_NETWORK_OR_RUNTIME_ROUTE
NO_PRODUCTION_AUTHORITY_CLAIM
NO_GATEMEM_SCORE_OR_HELD_OUT_CLAIM
NO_DELETION_ENGINEERING
```

This block is now active only for the completed local reference lane described
here. It does not authorize runtime integration or a subsequent evaluation.

## Proposed decisions

The reference lane would use:

- a fixture-backed synthetic identity authority with test-only signed envelopes;
- a deterministic, deny-by-default MNEMOS policy evaluator;
- immutable, hash-pinned entitlement and session-registry fixtures;
- synthetic-corpus-owned artifact descriptors with validated lineage;
- an isolated deterministic structural/span redaction boundary;
- an append-only, content-free local audit sink with bounded raw retention;
- a local CLI/library deployment with no listener or runtime imports;
- deletion of isolated outputs and disabling the entry point as rollback; and
- a deterministic seeded development-corpus generator and adversarial mutation
  matrix.

No production identity provider, policy service, entitlement system, session
registry, artifact catalogue, audit platform, or retention authority is selected.
Those choices require a later runtime proposal.

## Evaluation boundary

G4 development evidence may use only newly generated MNEMOS-owned synthetic
cases. The four GateMem domains are historical characterization data and are
excluded from policy development and performance claims in this lane.

There is no sealed evaluation in G4. An independent evaluation custodian is
intentionally unassigned until an organization or person outside policy
development accepts custody. Without that named custodian and a newly sealed or
independent corpus, no fresh held-out claim is possible.

## Rollback boundary

The proposed implementation has no runtime route and writes no durable MNEMOS
memory. Rollback consists of disabling/removing the standalone entry point and
discarding its versioned generated outputs. It cannot alter or restore live
authorization state because it owns none.

## Explicit exclusions

This proposal does not authorize:

- creation of implementation, policy, fixture, or generator code;
- import by a MNEMOS service, SDK, consumer, retrieval path, or answer path;
- a network listener, identity-provider connection, or policy-service call;
- production role, entitlement, redaction, retention, or compliance claims;
- policy tuning on GateMem medical, office, education, or household data;
- a GateMem prediction, scoring, hosted-judge, or leaderboard run;
- a fresh held-out benchmark claim; or
- deletion authorization or durable deletion behavior.

## Advancement

The implementation passed its local reference gate and is now frozen for
regression testing only. Further internal GateMem prototyping is paused.
Advancement requires an independent sealed-evaluation custodian and a newly
sealed or independent corpus under the G3 preregistration protocol. It cannot
authorize runtime integration, production authority selection, GateMem scoring,
held-out claims, or deletion engineering.

## Implementation closeout

```text
GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_COMPLETE
REFERENCE_CONTRACT_CONFORMANCE_ON_INSPECTABLE_SYNTHETIC_DEVELOPMENT_CASES
SYNTHETIC_DEVELOPMENT_ONLY
SEALED_EVALUATION_STILL_BLOCKED
NO_RUNTIME_OR_DELETION_CHANGE
```

The implementation evaluated 36 inspectable synthetic development cases with
36 exact expected outcomes. All reference gates passed, including the dedicated
mutation proving that the harness-owned HMAC key was absent from case files,
corpus artifacts, audit output, and run output.

## Evidence

- `docs/benchmarks/gatemem_g4_offline_reference_implementation_proposal.md`
- `docs/benchmarks/gatemem_g4_synthetic_development_corpus.md`
- `benchmarks/results/gatemem_g4_proposal_review.json`
- `benchmarks/results/gatemem_g4_proposal_review.md`
- `docs/benchmarks/gatemem_g4_offline_reference_implementation.md`
- `benchmarks/results/gatemem_g4_gate.json`
- `benchmarks/results/gatemem_g4_gate.md`
- `benchmarks/results/gatemem_g4_frozen_reference_manifest.json`
- `benchmarks/results/gatemem_g4_frozen_reference_manifest.md`
