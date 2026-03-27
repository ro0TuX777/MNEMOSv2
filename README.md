# MNEMOS

**Multi-tier Neuro-tagged Engram Memory with Optimal Near-lossless Index Compression**

A containerised, contract-governed memory and retrieval service for AI-native applications.

## Quick Start

```bash
# Local development
pip install -r requirements.txt
python -m service.app

# Docker
docker compose up -d --build

# Validate
python tools/mnemos_health_audit.py
```

## Architecture

```
┌──────────────────────────────────────────────┐
│              REST API (:8700)                 │
│  /index  /search  /engrams  /audit  /stats   │
├──────────────────────────────────────────────┤
│           Engram Enrichment Layer            │
├──────────────────────────────────────────────┤
│  ChromaDB  │  LanceDB  │  ColBERT MV        │
├──────────────────────────────────────────────┤
│        TurboQuant Compression (4-bit)        │
├──────────────────────────────────────────────┤
│           Forensic Ledger (audit)            │
└──────────────────────────────────────────────┘
```

## Configuration

All settings via environment variables — see `.env.example`.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/mnemos/capabilities` | GET | Feature discovery |
| `/v1/mnemos/index` | POST | Ingest documents |
| `/v1/mnemos/search` | POST | Query across tiers |
| `/v1/mnemos/engrams/{id}` | GET | Retrieve engram |
| `/v1/mnemos/engrams/{id}` | DELETE | Remove engram |
| `/v1/mnemos/audit` | GET | Query audit log |
| `/v1/mnemos/stats` | GET | Index statistics |

## Boundary SDK (Client Library)

Ship-ready Python client for any app to call MNEMOS with readiness, retry, and graceful degradation:

```python
from mnemos_sdk import MnemosClient, MnemosConfig

config = MnemosConfig.from_env()
client = MnemosClient(config)
client.wait_until_ready()

# Search
hits = client.search("quantum entanglement", top_k=5)
for hit in hits:
    print(f"  [{hit.score:.3f}] {hit.engram['content'][:80]}")

# Index
client.index([{"content": "New knowledge", "source": "app", "neuro_tags": ["physics"]}])
```

## Onboarding a Consumer App

Bootstrap MNEMOS integration in any app with one command:

```bash
python tools/mnemos_onboard.py --target /path/to/your-app
```

This generates a pre-wired boundary adapter, env template, smoke test spec, and quickstart doc.

## Operational Tooling

### Health & Contract Audit
```bash
python tools/mnemos_health_audit.py
```

### Contract Evolution
```bash
python tools/contract_diff.py --old service/contract.json --new contracts/mnemos_v2.json --mode both
```

### CI/CD Gates
```bash
python tools/mnemos_ci_gates.py --run-health-audit --run-container-build
```

See `.github/workflows/mnemos-gates.yml` for GitHub Actions template.

### Cutover Scaffold (Migration from Other Backends)
```bash
python tools/mnemos_cutover_scaffold.py --app my-app
```

Generates a staged rollout manifest (shadow → canary → full) with health gates and rollback paths.

## Repository Layout

```
mnemos/              Core library: retrieval, engram, compression, audit
mnemos_sdk/          Client boundary SDK for consumer apps
service/             REST API + MFS contract
registry/            Service registry (MFS-compatible)
tools/               Operational tooling (audit, diff, onboard, CI, cutover)
tests/               Unit tests
.github/workflows/   CI gate template
```

## License

Proprietary — extracted from SAM (SmallAgentModel).
