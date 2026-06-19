# MNEMOS Deployment Profiles

MNEMOS is intentionally modular, but external maturity depends on making the
smallest safe deployment obvious. Start small, validate the contract, then add
capabilities deliberately.

## Deployment Tiers

| Tier | Purpose | Components | When to use |
|---|---|---|---|
| Starter | Local developer evaluation | MNEMOS API, one vector backend, local or default embedding model, local audit fallback or PostgreSQL | First install, SDK checks, integration exploration. |
| Core | Production semantic retrieval | MNEMOS API, Qdrant, PostgreSQL audit store, health audit tooling | Latency-sensitive semantic memory, agent memory, general RAG. |
| Governed | Provenance-heavy retrieval | MNEMOS API, PostgreSQL/pgvector, audit ledger, policy/governance controls, tenant or metadata filters | Compliance-aware knowledge stores, strict provenance, metadata filtering. |
| Extended | Controlled experiments | Core or Governed plus optional TimesFM, Graph Tier, derived-fact lane, EBIR tooling, hybrid retrieval pilots | Evaluation and research lanes that need clear separation from supported defaults. |

## Supported Profiles

### Core Memory Appliance

Core Memory Appliance is the default production retrieval profile.

Required components:

- MNEMOS service
- Qdrant
- PostgreSQL for audit and shared metadata paths
- Embedding model configured by `MNEMOS_EMBEDDING_MODEL`

Best for:

- agent memory
- general semantic search
- RAG over mixed documents
- latency-sensitive retrieval

Primary validation:

```bash
python -m installer --profile core_memory_appliance
docker compose -f docker-compose.generated.yml up -d --build
python tools/mnemos_health_audit.py
```

### Governance Native

Governance Native uses PostgreSQL/pgvector as the primary retrieval profile
when metadata filtering, provenance, and operational simplicity around one
database are more important than Qdrant's raw ANN throughput.

Required components:

- MNEMOS service
- PostgreSQL with pgvector support
- audit and provenance configuration

Best for:

- provenance-heavy corpora
- strict metadata and tenant filters
- compliance-oriented deployments

Primary validation:

```bash
python -m installer --profile governance_native
docker compose -f docker-compose.generated.yml up -d --build
python tools/mnemos_health_audit.py
```

### Custom Manual

Custom Manual is for advanced operators who need to assemble a non-default
stack. It is not the recommended first path.

Use it when:

- profile defaults do not match the target environment
- ports, stores, model paths, or sidecars need custom wiring
- an experimental lane must be isolated from the normal stack

Document every manual deviation in the generated `mnemos_profile.yaml` or an
operator change record.

## Optional Components

| Component | Default posture | Notes |
|---|---|---|
| TimesFM sidecar | Off / advisory | Enabled with `docker-compose.timesfm.yml` and `MNEMOS_TIMESFM_ENABLED=true`. Keep `MNEMOS_PULSE_ACTIONS=advisory` unless a deployment-specific gate promotes actions. |
| Graph Tier | Experimental | Keep read-only and outside the public retrieval-mode default. |
| EBIR tooling | Shadow-only | Offline benchmark and reviewer-trial tooling only. No live memory writes, ranking changes, or authority promotion. |
| Hybrid retrieval | Targeted pilot | Semantic remains default. Enable hybrid only for evaluation classes where exact-term failures are suspected. |
| Reranker / ColBERT | Blocked for production | Requires reference-fidelity gate evidence before any production claim. |

## Minimal Safe Start

Use the guided installer unless you are intentionally testing manual deployment:

```bash
python -m installer
docker compose -f docker-compose.generated.yml up -d --build
python tools/mnemos_health_audit.py
```

For quick local evaluation where generated files are not needed:

```bash
docker compose up -d --build
python tools/mnemos_health_audit.py
```

## Promotion Rule

Move from Starter to Core or Governed only after:

- `/health` passes
- `/v1/mnemos/capabilities` reports the expected profile and backends
- `python tools/mnemos_health_audit.py` passes
- the selected profile is listed as supported in [support matrix](support_matrix.md)
- any enabled optional component has an explicit owner and rollback plan

