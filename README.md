# MNEMOS

**Multi-tier Neuro-tagged Engram Memory with Optimal Near-lossless Index Compression**

A containerised, contract-governed memory and retrieval service for AI-native applications.

---

## The Problem

Every AI application that persists and retrieves knowledge re-implements the same infrastructure from scratch — embedding pipelines, vector databases, ad-hoc search logic, compression hacks, and no audit trail. The result is fragile, inconsistent, and impossible to reuse across projects.

## The Solution

MNEMOS is a **drop-in memory service** that ships as a single Docker container. You configure it with environment variables, call it through a typed Python SDK, and get production-grade memory infrastructure without building it yourself.

| Capability | What you get |
|---|---|
| **Multi-tier retrieval** | ChromaDB (semantic), LanceDB (hybrid), ColBERT (token-level precision) — use one or all three |
| **TurboQuant compression** | 4-bit near-lossless quantisation — 5.6× storage reduction, 88% Recall@10 (arXiv:2504.19874) |
| **Engram enrichment** | Every document becomes a tagged, scored, provenanced knowledge unit with relationship edges |
| **Forensic audit** | Every operation immutably logged — compliance, debugging, and analytics built in |
| **Contract-governed API** | MFS-compatible versioned contract — every response carries `status`, `contract_version`, `error` |
| **Boundary SDK** | Python client with readiness polling, retry, timeout, and graceful degradation |
| **Operational tooling** | Health audit, contract diff, consumer onboarding, CI gates, staged cutover — all included |

---

## Quick Start

```bash
# Start
docker compose up -d --build

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
│  ChromaDB  │  LanceDB  │  ColBERT MV        │
├──────────────────────────────────────────────┤
│        TurboQuant Compression (4-bit)        │
├──────────────────────────────────────────────┤
│           Forensic Ledger (audit)            │
└──────────────────────────────────────────────┘
```

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

## Integrate in One Command

```bash
python tools/mnemos_onboard.py --target /path/to/your-app
```

Generates a pre-wired boundary adapter, `.env.mnemos` template, smoke test spec, and quickstart doc in your consumer app.

## Operational Tooling

| Tool | Command | Purpose |
|---|---|---|
| Health audit | `python tools/mnemos_health_audit.py` | Validate health, contract fields, version drift |
| Contract diff | `python tools/contract_diff.py --old v1.json --new v2.json --mode both` | Backward/forward compatibility checks |
| CI gates | `python tools/mnemos_ci_gates.py --run-health-audit` | Pipeline gate runner (+ GitHub Actions template) |
| Cutover | `python tools/mnemos_cutover_scaffold.py --app my-app` | Staged rollout for backend migration |

## Repository Layout

```
mnemos/              Core library: retrieval, engram, compression, audit
mnemos_sdk/          Boundary SDK (client library) for consumer apps
service/             Flask REST API + MFS contract
registry/            Service registry (MFS-compatible)
tools/               Operational tooling
tests/               Unit tests
docs/                Whitepaper + AI dev hand-off
```

## Documentation

- **[Whitepaper](docs/whitepaper.md)** — Architecture deep-dive, design principles, and provenance
- **[Installation Guide](INSTALL.md)** — Local dev, Docker, and MFS stack integration
- **[Use Cases](Use%20Cases.md)** — Example integration scenarios
- **[AI Dev Hand-off](docs/FORTHEAIDEV.md)** — Context doc for AI developer assistants

## License

Proprietary — extracted from SAM (SmallAgentModel).
