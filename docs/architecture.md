# MNEMOS Architecture

MNEMOS is a containerised memory and retrieval service that separates source evidence, retrieval, governance, audit, and consumer context assembly.

The whitepaper remains the detailed technical reference: [docs/whitepaper.md](whitepaper.md).

## System Flow

```text
Sources and project artifacts
            |
            v
   Engrams + source lineage
            |
            v
 Semantic / hybrid retrieval
            |
            v
 Governance and evidence checks
            |
            v
 Bounded context or retrieval result
            |
            v
 AI application, operator, or agent
```

![MNEMOS architecture overview](assets/mnemos-architecture-overview.svg)

## Primary Components

| Component | Role |
| --- | --- |
| MNEMOS service | Flask REST API exposing health, capabilities, index, search, audit, stats, and warmup endpoints. |
| Engram model | Stores content, metadata, source, tags, confidence, and lineage-ready fields. |
| Retrieval tiers | Qdrant and pgvector-backed retrieval profiles, with semantic and configurable hybrid modes. |
| Governance layer | Candidate evaluation, contradiction handling, lifecycle controls, and optional advisory/enforced read-path behavior. |
| Forensic ledger | Auditable records for index, search, mutation, and operational events. |
| Boundary SDK | Python client for readiness, retry, timeout handling, and typed index/search calls. |

## Deployment Shape

The default checked-in local stack is Docker Compose:

- `mnemos` service on port `8700`
- Qdrant on port `6333`
- PostgreSQL on port `5432`

Named deployment profiles and optional components are documented in [deployment profiles](deployment_profiles.md). Operational rollout, rollback, and incident handling are documented in the [operator playbook](mnemos_operator_playbook.md).

## Boundary Rules

- Source evidence, retrieval ranking, governance decisions, and context assembly are separate concerns.
- Research lanes do not change default retrieval, governance, authority, disclosure, promotion, or deletion behavior unless explicitly enabled and independently evaluated.
- Experimental or shadow results are not production-readiness, security, or broad performance claims unless the linked evidence artifact says so.

For the public status boundary, see the [support matrix](support_matrix.md). For decision history, see the [ADR index](adr/README.md).
