# MNEMOS — Installation Guide

## Prerequisites

- **Python** 3.10+
- **Docker** with **NVIDIA Container Toolkit** (for GPU deployment)
- **GPU**: NVIDIA GPU with CUDA 12.x support
- **pip** (Python package manager)

---

## Option 1: Guided Installer *(recommended)*

The installer asks 5 questions, probes your system, recommends a deployment profile, and generates all configuration files.

### 1. Run the Installer

```bash
python -m installer
```

**Interactive flow:**
1. Primary use case? (agent_memory / rag_knowledge_base / compliance_governed / other)
2. Main priority? (semantic_speed / metadata_governance / simplest_deployment)
3. Expected scale? (under_100k / 100k_to_1m / over_1m)
4. Need strict metadata / provenance / tenant filtering? (yes / no)
5. Prefer guided install or manual control? (guided / manual)

The installer then probes for GPU, RAM, Docker, NVIDIA runtime, and existing services, then recommends a profile.

### 2. Review Generated Files

| File | Purpose |
|---|---|
| `docker-compose.generated.yml` | Profile-specific Docker stack |
| `.env.mnemos` | Environment variables |
| `mnemos_profile.yaml` | Source of truth: profile, host facts, user answers |

### 3. Start

```bash
docker compose -f docker-compose.generated.yml up -d --build
```

### 4. Validate

```bash
python tools/mnemos_health_audit.py
```

### Non-Interactive Mode

```bash
python -m installer --profile core_memory_appliance
python -m installer --profile governance_native
python -m installer --profile custom_manual
```

---

## Deployment Profiles

| Profile | Stack | Containers | Best for |
|---|---|---|---|
| **Core Memory Appliance** | Qdrant + PostgreSQL + MNEMOS | 3 | Semantic memory, agent systems, RAG |
| **Governance Native** | PostgreSQL/pgvector + MNEMOS | 2 | Compliance-aware, metadata-filtered retrieval |
| **Custom Manual** | Operator-defined | — | Advanced setups, experimentation |

---

## Option 2: Manual Docker Setup

### Core Memory Appliance

```bash
docker compose up -d --build
```

Uses the default `docker-compose.yml` (Qdrant + Postgres + MNEMOS).

### Governance Native

```bash
# Use the governance-native compose template
docker compose -f installer/templates/governance_native.yml up -d --build
```

---

## Option 3: Local Development

### 1. Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `flask` | REST API framework |
| `qdrant-client` | Qdrant vector store (Core Memory Appliance) |
| `pgvector` | pgvector support (Governance Native) |
| `psycopg[binary]`, `psycopg_pool` | PostgreSQL driver + connection pooling |
| `sentence-transformers` | Embedding models (MiniLM, ColBERT) |
| `torch` | GPU-accelerated inference (CUDA) |
| `numpy`, `scipy` | TurboQuant compression |
| `gunicorn` | Production WSGI server |

### 3. Configure

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `MNEMOS_PROFILE` | `core_memory_appliance` | Deployment profile |
| `MNEMOS_TIERS` | `qdrant` | Active backends: `qdrant`, `pgvector`, `colbert` |
| `MNEMOS_GPU_DEVICE` | `cuda` | GPU device (`cuda`, `cuda:0`, `cpu`) |
| `MNEMOS_QDRANT_URL` | `http://localhost:6333` | Qdrant URL (Core Memory Appliance) |
| `MNEMOS_POSTGRES_DSN` | *(empty)* | PostgreSQL DSN (required for audit + pgvector) |
| `MNEMOS_PORT` | `8700` | API port |
| `MNEMOS_TOKEN` | *(empty)* | Optional bearer auth token |
| `MNEMOS_QUANT_BITS` | `4` | TurboQuant bit-width (0 = disabled) |

### 4. Run

```bash
# Development
python -m service.app

# Production
gunicorn -w 2 -b 0.0.0.0:8700 service.app:app
```

### 5. Verify

```bash
curl http://localhost:8700/health
curl http://localhost:8700/v1/mnemos/capabilities
python tools/mnemos_health_audit.py
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Operational Tools

| Tool | Usage |
|---|---|
| Installer | `python -m installer` |
| Health audit | `python tools/mnemos_health_audit.py` |
| Contract diff | `python tools/contract_diff.py --old v1.json --new v2.json` |
| CI gates | `python tools/mnemos_ci_gates.py --run-health-audit` |
| Cutover scaffold | `python tools/mnemos_cutover_scaffold.py --app my-app` |
| Onboard consumer | `python tools/mnemos_onboard.py --target /path/to/app` |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: qdrant_client` | Run `pip install -r requirements.txt` |
| Qdrant connection refused | Ensure Qdrant container is running: `docker compose up qdrant -d` |
| PostgreSQL connection refused | Ensure Postgres container is running and healthy |
| pgvector extension not found | Run `CREATE EXTENSION IF NOT EXISTS vector` in Postgres |
| CUDA not available | Check `nvidia-smi` and ensure NVIDIA Container Toolkit is installed |
| Embedding model slow on first run | Model download — `all-MiniLM-L6-v2` (~80 MB, one-time) |
| Port 8700 in use | Change `MNEMOS_PORT` in `.env.mnemos` |
