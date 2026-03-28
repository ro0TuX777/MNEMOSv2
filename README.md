# MNEMOS

**Multi-tier Neuro-tagged Engram Memory with Optimal Near-lossless Index Compression**

A containerised, contract-governed memory and retrieval service for AI-native applications.

---

## The Problem

Every AI application that persists and retrieves knowledge re-implements the same infrastructure from scratch — embedding pipelines, vector databases, ad-hoc search logic, compression hacks, and no audit trail. The result is fragile, inconsistent, and impossible to reuse across projects.

## The Solution

MNEMOS is a **drop-in memory service** that deploys as a GPU-accelerated Docker stack. You choose a deployment profile, run the guided installer, and get production-grade memory infrastructure without building it yourself.

| Capability | What you get |
|---|---|
| **Deployment profiles** | Core Memory Appliance (Qdrant), Governance Native (pgvector), or Custom Manual — named profiles with distinct retrieval architectures |
| **GPU-native retrieval** | CUDA-accelerated embeddings via Qdrant or pgvector, optional ColBERT reranking |
| **TurboQuant compression** | 4-bit near-lossless quantisation — 8× storage reduction, 84% Recall@10 (arXiv:2504.19874) |
| **Engram enrichment** | Every document becomes a tagged, scored, provenanced knowledge unit with relationship edges |
| **Forensic audit** | Every operation logged to PostgreSQL — compliance, debugging, and analytics built in |
| **Contract-governed API** | MFS-compatible versioned contract — every response carries `status`, `contract_version`, `error` |
| **Boundary SDK** | Python client with readiness polling, retry, timeout, and graceful degradation |
| **Guided installer** | Q/A + host probes → profile recommendation → compose + env generation |

---

## Deployment Profiles

| Profile | Stack | Best for |
|---|---|---|
| **Core Memory Appliance** | Qdrant + PostgreSQL + MNEMOS (3 containers) | Semantic memory, agent systems, general-purpose RAG |
| **Governance Native** | PostgreSQL/pgvector + MNEMOS (2 containers) | Provenance-heavy, metadata-filtered, compliance-aware retrieval |
| **Custom Manual** | Operator-defined | Advanced multi-tier setups, experimentation |

---

## Quick Start

```bash
# Install — guided profile selection
python -m installer

# Start — uses generated compose
docker compose -f docker-compose.generated.yml up -d --build

# Validate
python tools/mnemos_health_audit.py
```

```python
from mnemos_sdk import MnemosClient, MnemosConfig

client = MnemosClient(MnemosConfig.from_env())
client.wait_until_ready()

# Store knowledge
client.index([{
    "content": "Gravity waves detected by LIGO in 2015",
    "source": "arxiv:1602.03837",
    "neuro_tags": ["physics", "gravitational-waves"],
}])

# Recall knowledge
for hit in client.search("gravitational wave detection", top_k=5):
    print(f"  [{hit.score:.3f}] {hit.engram['content'][:80]}")
```

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              REST API (:8700)                 │
│  /index  /search  /engrams  /audit  /stats   │
├──────────────────────────────────────────────┤
│           Engram Enrichment Layer            │
├──────────────────────────────────────────────┤
│  Core Memory     │   Governance Native       │
│  Qdrant (CUDA)   │   pgvector (Postgres)     │
│  + opt. ColBERT  │   + opt. ColBERT          │
├──────────────────────────────────────────────┤
│        TurboQuant Compression (4-bit)        │
├──────────────────────────────────────────────┤
│       Forensic Ledger (PostgreSQL)           │
└──────────────────────────────────────────────┘
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/mnemos/capabilities` | GET | Feature discovery + active profile |
| `/v1/mnemos/index` | POST | Ingest documents |
| `/v1/mnemos/search` | POST | Query across active backends |
| `/v1/mnemos/engrams/{id}` | GET | Retrieve engram |
| `/v1/mnemos/engrams/{id}` | DELETE | Remove engram |
| `/v1/mnemos/audit` | GET | Query audit log |
| `/v1/mnemos/stats` | GET | Index statistics + profile info |

## Operational Tooling

| Tool | Command | Purpose |
|---|---|---|
| Installer | `python -m installer` | Guided profile selection + config generation |
| Health audit | `python tools/mnemos_health_audit.py` | Validate health, contract fields, version drift |
| Contract diff | `python tools/contract_diff.py --old v1.json --new v2.json` | Backward/forward compatibility checks |
| CI gates | `python tools/mnemos_ci_gates.py --run-health-audit` | Pipeline gate runner (+ GitHub Actions template) |
| Cutover | `python tools/mnemos_cutover_scaffold.py --app my-app` | Staged rollout for backend migration |

## Repository Layout

```
mnemos/              Core library: retrieval, engram, compression, audit
mnemos_sdk/          Boundary SDK (client library) for consumer apps
service/             Flask REST API + MFS contract
installer/           Guided installer (Q/A, probes, profiles, renderer)
tools/               Operational tooling
benchmarks/          Reproducible benchmark suite
tests/               Unit tests
docs/                Whitepaper + AI dev hand-off
```

## Documentation

- **[Whitepaper](docs/whitepaper.md)** — Architecture deep-dive, deployment profiles, and benchmarks
- **[Installation Guide](INSTALL.md)** — Installer usage, profiles, and manual setup
- **[Use Cases](Use%20Cases.md)** — Example integration scenarios mapped to profiles
- **[AI Dev Hand-off](docs/FORTHEAIDEV.md)** — Context doc for AI developer assistants

## License

Proprietary.
