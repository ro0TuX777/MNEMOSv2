# MNEMOS Dependency Map

This document makes MNEMOS runtime dependencies, optional dependencies, and fork
posture explicit.

## Fork And Source Policy

Current repository scan:

- No `.gitmodules` file is present.
- Root `requirements.txt` uses package names from public Python package indexes,
  not Git URLs.
- Runtime Compose files reference public upstream images for Qdrant, Postgres,
  pgvector, NVIDIA CUDA, and Python.
- No private package index is declared in the scanned dependency files.

Policy:

- No hidden forks.
- Any fork, patched image, private package source, or Git-pinned dependency must
  be declared here before release.
- A declared fork must include upstream URL, reason, divergence summary,
  maintenance owner, and exit or rebase plan.

## Runtime Components

| Component | Required for | Source | Current reference | Notes |
|---|---|---|---|---|
| MNEMOS service | All profiles | Local repository | `Dockerfile` | Flask REST API and core memory service. |
| Qdrant | Core Memory Appliance | Public container image | `qdrant/qdrant:v1.17.1` | Default vector backend for Core. |
| PostgreSQL | Core and Governed profiles | Public container image | `postgres:16-alpine` | Audit, metadata, and pgvector storage paths. |
| pgvector | Governance Native and benchmark profile | Public container image / extension | `pgvector/pgvector:pg16` in benchmark stack | Governance Native requires pgvector availability. |
| NVIDIA CUDA runtime | GPU container runtime | Public container image | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | Root service image assumes NVIDIA runtime for GPU deployment. |
| TimesFM sidecar | Optional predictive pulse | Local sidecar Dockerfile plus local model path | `sidecars/timesfm/Dockerfile` | Disabled by default; model is mounted read-only. |

## Python Dependencies

| Area | Packages | Required posture |
|---|---|---|
| API | `flask`, `gunicorn`, `requests` | Required. |
| Vector stores | `qdrant-client`, `pgvector` | Required by supported profiles. |
| Embeddings and inference | `sentence-transformers`, `torch`, `einops` | Required for normal embedding-backed retrieval. |
| PostgreSQL | `psycopg[binary]`, `psycopg_pool` | Required for PostgreSQL-backed audit and pgvector paths. |
| Numerical utilities | `numpy`, `scipy` | Required by compression and benchmark paths. |
| Configuration | `python-dotenv` | Required for local env loading. |

## Network Requirements

| Path | Required when | Notes |
|---|---|---|
| MNEMOS API on port `8700` | Always | External clients call REST and SDK operations here. |
| Qdrant on port `6333` | Core profile | Internal Compose networking is preferred in production. |
| PostgreSQL on port `5432` | Core or Governed profile | Expose only when host access is required. |
| TimesFM on port `8711` | TimesFM sidecar enabled | Keep internal to Compose unless explicitly needed. |
| Model/package downloads | First install or image build | Pin and cache models for controlled environments. |

## Storage Requirements

| Path / volume | Purpose |
|---|---|
| `qdrant_data` | Qdrant vector collections. |
| `postgres_data` | PostgreSQL audit, metadata, and pgvector state. |
| `./data:/app/data` | Local MNEMOS data, including SQLite audit fallback when configured. |
| `./models/timesfm-2.5-200m-pytorch` | Optional read-only TimesFM model mount. |

## Version And Upgrade Policy

- Container image tags must be explicit; avoid floating `latest`.
- Python dependency ranges should be tightened before external release
  candidates when reproducibility matters more than installation flexibility.
- Upgrades to Qdrant, PostgreSQL, pgvector, CUDA, embedding models, or TimesFM
  must be recorded with:
  - previous version
  - new version
  - migration impact
  - benchmark or smoke evidence
  - rollback plan

## SBOM Posture

The source/Python SPDX 2.3 artifact is
[mnemos-python.spdx.json](sbom/mnemos-python.spdx.json), with current hygiene
findings in [dependency-hygiene.json](sbom/dependency-hygiene.json). The release
workflow regenerates and uploads both, then blocks while dependencies remain
non-exact or unresolved.

This does not cover container base-image or OS packages. The current dependency
set uses ranges rather than a hash-pinned lockfile, so the generated hygiene
report correctly marks `release_ready: false`.
