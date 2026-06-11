MNEMOS
Adaptive, Governed Memory Appliance with Hierarchical Synthesis and Matryoshka Economics
A containerised, contract-governed memory service that understands query complexity, resolves factual contradictions, and delivers 65% faster retrieval via Matryoshka embeddings.

The Solution
MNEMOSv2 is a production-hardened memory service that deploys as a GPU-accelerated Docker stack. It moves beyond simple vector search to provide a governed "Source of Truth" for AI-native applications.

Capability: What you get
- Adaptive Routing	Embedded-Reflex Classifier detects query complexity (Simple/Multi-hop/Global) in <0.1ms and orchestrates the optimal retrieval path.
- Matryoshka Economics	64-dim prefetch / 768-dim rescore via Nomic v1.5. Delivers 65% reduction in p95 latency while maintaining 1.0 rank stability.
- Hierarchical Synthesis	RAPTOR-lite thematic summaries (Depth-1 & 2) provide "Executive Overviews" for global queries, isolated from raw factoid search.
- Knowledge Consensus	Reconciliation Engine synthesizes "Resolution Engrams" from conflicting memories, providing a structured "Single Source of Truth."
- NLI-Grounded Precision	Bidirectional NLI Critic validates memory usage via semantic entailment. USED precision improved from 0.57 to 1.00.
- Counterfactual Audit	Deterministic explainability traces: "Why won/lost" and Age Inversion (decay modifiers explained in human-readable days).
- SLO-Driven Orchestration	Budget-Aware Router uses an EWMA cost model to shed load (skip rerank/rescore) to hit your latency_budget_ms.
- TurboQuant Compression	4-bit near-lossless quantisation — 8× storage reduction, 0.995 cosine fidelity (arXiv:2504.19874).
- Forensic Ledger	Every operation, classification, and reconciliation is logged to PostgreSQL for immutable auditability.

What's New in v2.x (Phase 7-10)
- Feature	Impact	Benchmark
- Matryoshka MRL	64-dim prefetch + 768-dim rescore	45ms p95 (vs 129ms baseline)
- Adaptive-RAG	Automatic routing between Flat, Graph, and Hierarchical paths	100% Hold-out Accuracy
- Consensus Engine	Automated resolution of factual contradictions	5/5 Collisions Resolved
- NLI Reflect	Entailment-based feedback loop (No false-positives)	1.0 Precision
- Warmup Readiness	Zero user-perceived cold-starts for GPU models	25s load hit managed in preflight

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
┌──────────────────────────────────────────────────────┐
│ REST API / MFS Contract                              │
│ /index /search /engrams /audit /stats /warmup        │
├──────────────────────────────────────────────────────┤
│ Adaptive Router                                      │
│ Complexity classifier + latency budget policy        │
├──────────────────────────────────────────────────────┤
│ Governed Retrieval Path                              │
│ Flat evidence │ Hierarchical summaries │ Resolutions │
├──────────────────────────────────────────────────────┤
│ Governance + Consensus                               │
│ Contradiction policy, explainability, reflect/NLI    │
├──────────────────────────────────────────────────────┤
│ Vector Backends                                      │
│ Qdrant MRL prefetch/rescore │ pgvector profile       │
├──────────────────────────────────────────────────────┤
│ Compression + Persistence                            │
│ TurboQuant │ PostgreSQL audit/metadata/lineage       │
└──────────────────────────────────────────────────────┘

```

## API

Every response includes `contract_version`, `status`, `profile`, `tiers`, and `degraded_components`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Container health check |
| `/v1/mnemos/capabilities` | GET | Active profile, backends, degraded components, compression config |
| `/v1/mnemos/index` | POST | Ingest documents → engrams |
| `/v1/mnemos/search` | POST | Query across active backends (with optional metadata filters) |
| `/v1/mnemos/engrams/{id}` | GET | Retrieve engram by ID |
| `/v1/mnemos/engrams/{id}` | DELETE | Remove from all backends |
| `/v1/mnemos/audit` | GET | Query the forensic ledger |
| `/v1/mnemos/stats` | GET | Backend sizes, profile info, compression ratios |

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

- **[Whitepaper](docs/whitepaper.md)** — Architecture deep-dive, profile benchmarks, deployment manifest, and migration rules
- **[Installation Guide](INSTALL.md)** — Installer usage, profiles, and manual setup
- **[AI Dev Hand-off](docs/FORTHEAIDEV.md)** — Context doc for AI developer assistants

## License

Proprietary.
