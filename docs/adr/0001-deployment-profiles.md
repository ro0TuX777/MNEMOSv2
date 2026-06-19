# ADR 0001: Deployment Profiles As Public Support Boundaries

Date: 2026-06-20

Status: Accepted

## Context

MNEMOS has multiple runtime capabilities: Qdrant retrieval, PostgreSQL/pgvector,
audit storage, governance controls, predictive sidecars, graph lanes, hybrid
retrieval, and offline evaluation harnesses. Without explicit support
boundaries, readers may treat every capability as equally production-ready.

## Decision

MNEMOS will present deployment profiles as public support boundaries:

- Core Memory Appliance is the default supported semantic retrieval profile.
- Governance Native is the supported provenance-heavy PostgreSQL/pgvector
  profile.
- Custom Manual is available for advanced operators, but requires explicit
  operator ownership.
- Optional lanes such as TimesFM, Graph Tier, EBIR, hybrid retrieval, and
  rerankers must be classified separately in the support matrix.

## Alternatives Considered

- Present all capabilities as a single platform surface.
- Hide experimental capabilities from public documentation.
- Split every capability into a separate product profile.

## Rationale

Profiles let users start with a small, reproducible stack and add complexity
only when there is a clear reason. They also prevent experimental or shadow-only
work from being mistaken for supported default behavior.

## Invariants

- Supported profiles must have documented start and validation commands.
- Experimental and shadow-only lanes must not be described as production
  defaults.
- EBIR paths remain offline and shadow-only unless a future ADR and gate
  evidence explicitly promote them.
- Synthetic benchmark results must not be used as standalone product-quality
  claims.

## Rollback

If profile boundaries become inaccurate, update `docs/support_matrix.md`,
`docs/deployment_profiles.md`, and this ADR before changing README claims.

## Evidence

- `README.md`
- `INSTALL.md`
- `docs/benchmark.md`
- `docs/ebir_r1_acceptance.md`
- `docs/ebir_r2_trial_protocol.md`
- `docs/chat_integration_evidence_contract.md`

