# ADR 0012: GateMem G3 Defines Authorization and Disclosure Semantics Only

Date: 2026-06-24

Status: Accepted — design only

## Context

GateMem G0 through G2A established a clean external evaluation path and a
frozen cross-domain baseline. The baseline preserved provenance perfectly but
showed that visible-text heuristics and candidate filtering do not provide
content-safe disclosure:

- weighted auxiliary utility: `0.2775`;
- weighted privacy leakage: `0.1788`;
- weighted over-refusal: `0.4052`; and
- used-record provenance integrity: `1.0`.

The missing capability is not another retrieval heuristic. It is a governed
decision model that binds an authenticated principal to tenant/session scope,
roles, entitlements, artifact/source classes, disclosure obligations, and
auditable outcomes.

## Decision

Authorize G3 as a specification-only lane under:

```text
GATEMEM_G3_AUTHORIZATION_DISCLOSURE_SEMANTICS_DESIGN_ONLY
NO_RUNTIME_IMPLEMENTATION
NO_POLICY_TUNING
NO_NEW_GATEMEM_SCORE_CLAIM
NO_DELETION_ENGINEERING
```

The normative design is
`docs/benchmarks/gatemem_g3_authorization_disclosure_semantics.md`.

G3 defines:

- validated authenticated-principal context;
- identity-derived tenant and session scope;
- scoped role assignments and explicit entitlements;
- content-free artifact/source descriptors;
- deny-by-default authorization decisions;
- disclosure and redaction obligations;
- non-sensitive consumer errors and content-free audit correlation;
- replay and policy-version pinning; and
- a development-versus-sealed-evaluation governance protocol.

## Core invariants

- Caller-supplied tenant or session filters are never authorization evidence.
- Query wording is never evidence of identity, role, relationship, entitlement,
  purpose, or consent.
- A role is not sufficient authority. A scoped role assignment and applicable
  entitlement or relationship grant are required.
- Unknown tenant, session, role, entitlement, artifact class, source class,
  policy version, or redaction obligation fails closed.
- Denied content and denied identifiers do not cross into retrieval prompts,
  answer generation, consumer errors, telemetry, or general logs.
- Redaction failure becomes denial; it never degrades to unredacted disclosure.
- Every permit is bound to a policy fingerprint, authorization snapshot,
  purpose, operation, scope, issuance time, and expiry.
- G3 defines no deletion state transition and makes no active-forgetting claim.

## Evaluation integrity decision

All four released GateMem domains were observed during G2/G2A. They remain a
valid frozen historical baseline but cannot become a fresh untouched held-out
evaluation after policy development.

Before implementation or tuning, a future workstream must register:

1. a development corpus that policy developers may inspect and tune against;
2. a newly sealed or independent GateMem-compatible evaluation corpus that
   policy developers cannot inspect before freeze;
3. immutable manifests and hashes for both;
4. policy, code, thresholds, seeds, metrics, and claims frozen before unsealing;
5. evaluator custody and one-way annotation flow; and
6. one-shot evaluation and post-run change rules.

A retrospective split of the already-observed G2A corpus may be used for
engineering regression only. It cannot support a fresh held-out claim.

## Explicit exclusions

This ADR does not authorize:

- an identity provider, policy engine, entitlement store, or runtime route;
- changes to MNEMOS search filters, ranking, governance, SDK, service, or
  consumer adapters;
- policy thresholds, role matrices, entitlement grants, or redaction rules;
- a GateMem adapter change, prediction run, scorer run, hosted judge, or public
  submission;
- deletion authorization, tombstones, lineage cascade, cache purge,
  re-ingestion defense, negative verification, or answer-layer
  non-confirmation; or
- production access-control, compliance, or deletion claims.

## Deletion separation

Deletion remains a distinct future ADR because it changes durable-memory
semantics. Authorization may later decide who is allowed to request deletion,
but it cannot itself prove that deletion occurred or remained effective.

The deletion ADR must separately address durable tombstones, target resolution,
lineage cascade, cache invalidation, re-ingestion defenses, cross-tier negative
verification, retention/backup boundaries, and answer-layer non-confirmation.

## Advancement

Completion of G3 authorizes review of a separate implementation proposal only.
That proposal must name the identity authority, policy authority, entitlement
store, trust boundaries, deployment mode, rollback, tests, and sealed
evaluation plan. G3 does not authorize implementation by itself.

## Evidence

- `docs/benchmarks/gatemem_program_status.md`
- `docs/benchmarks/gatemem_g3_authorization_disclosure_semantics.md`
- `docs/benchmarks/gatemem_g3_preregistration_template.md`
- `benchmarks/results/gatemem_g3_design_review.json`
- `benchmarks/results/gatemem_g3_design_review.md`

