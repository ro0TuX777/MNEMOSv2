# MNEMOS — White Paper

**A containerised, contract-governed memory and retrieval service for AI-native applications.**

*Version 2.0 · March 2026*

---

## 1. Problem Statement

Every AI application that persists and retrieves knowledge must solve the same set of problems:

- **Storage**: Where do embeddings live, and how do they scale?
- **Retrieval**: How do you find the right information across thousands of documents with high precision?
- **Compression**: How do you keep memory footprint manageable as the index grows?
- **Enrichment**: How do you go beyond raw vectors to attach semantic meaning, provenance, and relationships?
- **Auditability**: How do you trace what was stored, retrieved, modified, and when?
- **Integration**: How do you wire memory into an application without tight coupling or fragile glue code?

Today, each project re-implements these capabilities from scratch — writing custom embedding pipelines, bolting on vector databases, and building ad-hoc search logic. The result is fragile, inconsistent, and impossible to reuse across projects.

## 2. Solution: MNEMOS

**MNEMOS** (Multi-tier Neuro-tagged Engram Memory with Optimal Near-lossless Index Compression) is a self-contained Docker service that provides production-grade memory infrastructure for any application. It ships as a single image, configurable via environment variables, and exposes a versioned REST API governed by an MFS contract.

It is **application-agnostic** — it knows nothing about the domain of the consuming application. It stores, enriches, compresses, retrieves, and audits knowledge. That's it.

MNEMOS also ships with a **Boundary SDK** (Python client library) and a suite of **operational tools** (health audit, contract evolution, onboarding, CI gates, and staged cutover) — making it a complete prefab that can be plugged into any application with a single command.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       REST API (:8700)                      │
│   /index   /search   /engrams   /audit   /stats            │
├────────────────────────────────────────────────────────────┤
│                  Engram Enrichment Layer                    │
│   neuro-tags · provenance · confidence · relationship      │
│   edges · extensible metadata                              │
├────────────────────────────────────────────────────────────┤
│              Retrieval Tiers (configurable)                 │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐    │
│  │  Qdrant  │  │ LanceDB  │  │ ColBERT Multi-Vector  │    │
│  │ Semantic │  │ Hybrid   │  │ Token-level matching   │    │
│  │ (384-dim)│  │ Struct+  │  │ (128-dim, MaxSim)     │    │
│  │  (CUDA)  │  │ Vector   │  │   (GPU-accelerated)   │    │
│  └──────────┘  └──────────┘  └───────────────────────┘    │
├────────────────────────────────────────────────────────────┤
│             TurboQuant Compression Layer                    │
│   4-bit quantised storage · 5.6× file compression          │
│   Near-optimal distortion (arXiv:2504.19874)               │
├────────────────────────────────────────────────────────────┤
│     Embedding Engine (GPU-accelerated, swappable)          │
│   all-MiniLM-L6-v2 │ ColBERTv2.0 │ custom model           │
├────────────────────────────────────────────────────────────┤
│          Forensic Ledger (PostgreSQL audit trail)           │
│   Immutable · every operation logged · replayable          │
│   tsvector FTS · connection pooling · SQLite fallback      │
└────────────────────────────────────────────────────────────┘
```

The architecture is layered by concern: the API layer handles routing and auth; the engram layer enriches raw content; the retrieval layer manages tier-specific backends; the compression layer reduces storage footprint; and the audit layer logs every mutation.

---

## 4. Core Components

### 4.1 Engram: The Universal Memory Unit

An **Engram** is the atomic unit of knowledge in MNEMOS. It wraps a raw document chunk with machine-generated enrichments that make retrieval smarter and operations auditable.

| Field | Type | Purpose |
|---|---|---|
| `id` | `string` | Unique identifier (UUID) |
| `content` | `string` | Raw text content |
| `embedding` | `vector` | Dense vector (quantised on disk) |
| `neuro_tags` | `string[]` | Auto-generated semantic labels |
| `source` | `string` | Provenance URI (where this data came from) |
| `confidence` | `float` | Quality signal (0.0–1.0) used for ranking |
| `created_at` | `datetime` | Ingestion timestamp |
| `metadata` | `object` | Extensible application-specific data |
| `edges` | `string[]` | IDs of related engrams (knowledge graph) |

The Engram is **domain-agnostic** — the consuming application decides what `neuro_tags` mean, what `source` URIs look like, and what goes in `metadata`. MNEMOS provides the schema, storage, indexing, and retrieval.

### 4.2 Multi-Tier Retrieval

MNEMOS supports three retrieval tiers, each serving a different fidelity level. Applications can enable one, two, or all three via the `MNEMOS_TIERS` environment variable:

| Tier | Backend | Embedding Model | Strength |
|---|---|---|---|
| **Tier 1** | Qdrant | all-MiniLM-L6-v2 (384-dim, CUDA) | Fast semantic vector retrieval, payload filtering |
| **Tier 2** | LanceDB | Configurable | Hybrid keyword + vector queries, structured filtering |
| **Tier 3** | ColBERT Multi-Vector | colbert-ir/colbertv2.0 (128-dim) | Token-level late-interaction matching — highest precision |

**Tier fusion**: When multiple tiers are active, the API returns merged results with per-tier scores and a fused ranking. The consuming application can weight tiers differently or target specific tiers per query.

### 4.3 TurboQuant Compression

All stored embeddings are compressed via **TurboQuant** (arXiv:2504.19874), a near-optimal vector quantisation algorithm.

**How it works:**
1. Input vectors are randomly rotated so each coordinate follows a Beta distribution
2. Optimal Lloyd-Max scalar quantisers compress each coordinate to 1–4 bits
3. For inner-product operations, a 1-bit QJL residual preserves unbiased estimation

**Performance at 4-bit (default):**

| Metric | Value |
|---|---|
| Storage compression | 5.6× (compressed file) to 8× (raw bytes) |
| Recall@10 | 88% of float32 baseline |
| MSE bound | ≤ 0.009 (practically lossless) |
| Encode overhead | ~55ms per document |

**Why it matters**: Without compression, a 1M-document index at 128 dimensions consumes ~488 MB in float32. With 4-bit TurboQuant, that drops to ~61 MB — enabling deployment on memory-constrained edge devices, smaller cloud instances, and faster cold starts.

### 4.4 Forensic Ledger

Every operation that touches stored memory is immutably logged:

| Field | Purpose |
|---|---|
| `timestamp` | When the operation occurred |
| `component` | Which service component performed it |
| `action` | What happened (index, search, delete, update) |
| `content` | Human-readable description |
| `status` | `success`, `failure`, `warning` |
| `latency` | Operation duration in seconds |
| `metadata` | Structured details (IDs affected, query text, result count) |

**Use cases:**
- **Compliance** — demonstrate when data was ingested, accessed, or deleted
- **Debugging** — replay the sequence of operations that led to a retrieval failure
- **Analytics** — track query patterns, ingestion rates, and error trends

---

## 5. API Contract

MNEMOS follows the **MFS Contract Pattern**: every response includes `contract_version`, `status`, `source`, and `error` fields, ensuring the consuming application can always determine the health and trustworthiness of the data it receives.

### Contract (service/contract.json)

```json
{
  "service_name": "mnemos-service",
  "contract_version": "v1",
  "endpoint": "/v1/mnemos/capabilities",
  "required_fields": {
    "contract_version": "str",
    "status": "str",
    "source": "str",
    "generated_at": "str",
    "feature": "str",
    "supports": "list",
    "error": "nullable_str"
  },
  "allowed_status": ["healthy", "degraded", "unavailable"]
}
```

### Core Endpoints

```
GET    /health                      — Container health check
GET    /v1/mnemos/capabilities      — Feature discovery + contract version
POST   /v1/mnemos/index             — Ingest documents → engrams
POST   /v1/mnemos/search            — Query across active tiers
GET    /v1/mnemos/engrams/{id}      — Retrieve a specific engram
DELETE /v1/mnemos/engrams/{id}      — Remove from all tiers
GET    /v1/mnemos/audit             — Query the forensic ledger
GET    /v1/mnemos/stats             — Compression ratios, index sizes, tier health
```

### Example: Indexing a Document

```json
POST /v1/mnemos/index
{
  "documents": [
    {
      "content": "The quarterly results exceeded expectations...",
      "source": "s3://reports/Q1-2026.pdf",
      "neuro_tags": ["finance", "quarterly"],
      "confidence": 0.95,
      "metadata": { "department": "finance", "fiscal_year": 2026 }
    }
  ],
  "options": {
    "tiers": ["qdrant", "colbert"]
  }
}
```

### Example: Searching

```json
POST /v1/mnemos/search
{
  "query": "What were the Q1 revenue figures?",
  "top_k": 10,
  "tiers": ["qdrant", "colbert"],
  "filters": { "metadata.department": "finance" }
}
```

---

## 6. Boundary SDK

MNEMOS ships with a Python client library (`mnemos_sdk/`) that provides the standard way for consumer applications to call the service. The SDK handles concerns that raw HTTP calls don't:

| Capability | Description |
|---|---|
| **Readiness polling** | `wait_until_ready()` polls `/health` until the service is available or timeout |
| **Retry with backoff** | Configurable retry count and delay for transient failures |
| **Timeout management** | Per-request timeout with graceful fallback |
| **Auto-start** | Optional: start MNEMOS container on first call (for local dev) |
| **Typed responses** | `MnemosResponse`, `SearchHit`, `IndexResult` data classes |
| **Graceful degradation** | Returns structured `degraded`/`unavailable` envelopes instead of exceptions |
| **Env-var config** | `MnemosConfig.from_env()` wires everything from `MNEMOS_*` env vars |

### Usage

```python
from mnemos_sdk import MnemosClient, MnemosConfig

# Configure from environment (MNEMOS_BASE_URL, MNEMOS_TOKEN, etc.)
config = MnemosConfig.from_env()
client = MnemosClient(config)

# Wait for service startup
client.wait_until_ready()

# Store knowledge
client.index([{
    "content": "Gravity waves detected by LIGO in 2015",
    "source": "arxiv:1602.03837",
    "neuro_tags": ["physics", "gravitational-waves"],
}])

# Recall knowledge
hits = client.search("gravitational wave detection", top_k=5)
for hit in hits:
    print(f"  [{hit.score:.3f}] {hit.engram['content'][:80]}")
```

**Rule**: Consumer apps should always use the SDK. Direct HTTP calls bypass readiness, retry, and degradation handling.

---

## 7. Operational Tooling

MNEMOS includes a suite of tools adapted from the MFS framework for production operation:

### 7.1 Health & Contract Audit

```bash
python tools/mnemos_health_audit.py
```

Validates: `/health` returns 200, contract endpoint fields match `contract.json` types, status is in allowed values, contract version has not drifted.

### 7.2 Contract Evolution

```bash
python tools/contract_diff.py --old service/contract.json --new contracts/mnemos_v2.json --mode both
```

Checks backward and forward compatibility when evolving the contract: field additions/removals, type changes, enum narrowing, default value transitions, and major version bump advisories.

### 7.3 Consumer Onboarding

```bash
python tools/mnemos_onboard.py --target /path/to/consumer-app
```

Generates in the consumer app: a pre-wired boundary adapter, `.env.mnemos` template, smoke test spec, and integration quickstart doc.

### 7.4 CI/CD Gates

```bash
python tools/mnemos_ci_gates.py --run-health-audit --run-container-build
```

Runs contract validation, health audit, and container build as CI pipeline steps. Includes a GitHub Actions workflow template (`.github/workflows/mnemos-gates.yml`).

### 7.5 Cutover Scaffold

```bash
python tools/mnemos_cutover_scaffold.py --app my-app
```

Generates a staged rollout manifest (shadow → canary 5/25/50% → full) for apps migrating from another memory backend to MNEMOS, with health gates and rollback paths.

---

## 8. Deployment

MNEMOS v2 deploys as a **3-container GPU-accelerated stack**: Qdrant (vector store), PostgreSQL (audit ledger), and the MNEMOS service (CUDA-enabled).

### Docker Compose

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=mnemos_audit
      - POSTGRES_USER=mnemos
      - POSTGRES_PASSWORD=mnemos
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mnemos:
    build: .
    runtime: nvidia
    ports:
      - "8700:8700"
    environment:
      - MNEMOS_TIERS=qdrant               # qdrant,lancedb,colbert
      - MNEMOS_EMBEDDING_MODEL=all-MiniLM-L6-v2
      - MNEMOS_QUANT_BITS=4
      - MNEMOS_GPU_DEVICE=cuda
      - MNEMOS_QDRANT_URL=http://qdrant:6333
      - MNEMOS_POSTGRES_DSN=postgresql://mnemos:mnemos@postgres:5432/mnemos_audit
      - MNEMOS_PORT=8700
    depends_on: [qdrant, postgres]

volumes:
  qdrant_data:
  postgres_data:
```

### Resource Requirements

| Configuration | RAM | Disk | GPU |
|---|---|---|---|
| Qdrant only | ~2 GB | ~200 MB base | Required (CUDA) |
| Qdrant + ColBERT, 4-bit TurboQuant | ~4 GB | ~400 MB base | Required (CUDA) |
| Full stack (all tiers) | ~6 GB | ~600 MB base | Required (CUDA, recommended ≥8 GB VRAM) |

---

## 9. Integration Workflow

A consumer application adopts MNEMOS in five steps:

```
1. Onboard       →  python tools/mnemos_onboard.py --target <app>
2. Configure     →  Set MNEMOS_BASE_URL, MNEMOS_TOKEN in app env
3. Start         →  docker compose up -d --build
4. Validate      →  python tools/mnemos_health_audit.py
5. Wire          →  Import mnemos_sdk, use client.index() / client.search()
```

For apps migrating from another memory backend (Redis, Elasticsearch, FAISS):

```
6. Scaffold cutover   →  python tools/mnemos_cutover_scaffold.py --app <app>
7. Shadow writes      →  Write to both old + MNEMOS, read from old
8. Canary promotion   →  Route 5% → 25% → 50% → 100% of reads to MNEMOS
9. Decommission       →  Remove old backend
```

---

## 10. Use Cases

The following are the highest-value scenarios where MNEMOS provides immediate benefit as a drop-in memory layer.

### 10.1 AI Agent / Copilot Platforms

The most natural fit. Any system that has an LLM doing multi-step work needs persistent, searchable memory.

- **What MNEMOS provides**: Engram-enriched conversation history, tool output recall, semantic search over past interactions.
- **Why not just a raw vector DB**: Neuro-tags give semantic labels for retrieval boosting. The forensic ledger tracks what the agent remembered and when — critical for debugging hallucinations.
- **Example**: A coding assistant that remembers past codebases it has worked on, retrieves relevant patterns, and audits what context influenced each generation.

### 10.2 RAG-Powered Knowledge Bases

Enterprise document search where accuracy and audit trails matter — legal, medical, compliance.

- **What MNEMOS provides**: Multi-tier retrieval (Qdrant for fast semantic, ColBERT for precision when it counts), TurboQuant for scaling to millions of chunks without blowing up storage costs.
- **Why it wins**: The forensic ledger gives compliance-ready logging of every query and retrieval — *"show me exactly what documents were retrieved for this answer and when."*
- **Example**: Internal knowledge base for a law firm — lawyers query it, and each retrieval is logged for audit.

### 10.3 IoT / Edge Deployments

Devices with limited memory and storage that still need intelligent retrieval.

- **What MNEMOS provides**: TurboQuant 4-bit compression means a 1M-document index fits in ~61 MB instead of ~488 MB. Single-tier mode (Qdrant only) keeps the footprint tiny.
- **Why it wins**: Most vector DBs assume cloud-scale resources. MNEMOS can run on a Raspberry Pi-class device.
- **Example**: A smart home hub that remembers user preferences, schedules, and sensor patterns — compressed on-device, searchable locally without cloud dependency.

### 10.4 Multi-Agent Orchestration Systems

Systems where multiple specialised agents need shared memory without stepping on each other.

- **What MNEMOS provides**: A centralised, contract-governed memory service that any agent can index to and search from via REST. The MFS contract pattern means agents can trust the response schema.
- **Why it wins**: Without shared memory, each agent re-discovers context. With MNEMOS, Agent A's research becomes Agent B's retrieval — and the audit trail shows who stored what.
- **Example**: A research pipeline where a "Scout" agent gathers papers, an "Analyst" agent extracts insights, and a "Writer" agent drafts reports — all sharing one MNEMOS instance.

### 10.5 Content / Creative Platforms

Story generators, game engines, or creative tools that need long-term world memory.

- **What MNEMOS provides**: Engram edges create a knowledge graph of relationships (characters → events → locations). Neuro-tags categorise memory by theme. ColBERT tier finds nuanced, token-level matches for continuity.
- **Why it wins**: Creative tools need precise recall (*"what did character X say about Y in chapter 3?"*) — multi-vector retrieval is dramatically better than single-vector for this.
- **Example**: An interactive fiction engine where the story adapts based on retrieving and referencing past plot points from a compressed engram store.

### Common Thread

Any application that stores, enriches, retrieves, and audits knowledge — and wants to deploy it as a single container without building the plumbing from scratch. MNEMOS gives you the full stack in one `docker compose up`.

---

## 11. Design Principles

1. **Application-agnostic** — The service has zero knowledge of what domain it serves. It stores vectors, enriches engrams, and answers queries. Period.
2. **Contract-governed** — Every API response follows a strict MFS contract schema, enabling reliable integration without tight coupling.
3. **Tier-composable** — Enable one backend or all three. The service adapts its retrieval strategy to what's configured.
4. **Compression by default** — TurboQuant is on at 4-bit out of the box. Storage scales sublinearly with document count.
5. **Audit by default** — Every mutation is logged immutably. Compliance is a feature, not an afterthought.
6. **Graceful degradation** — If a tier goes unhealthy, remaining tiers continue serving. Status is always reported honestly via the contract.
7. **SDK-first integration** — Consumer apps use the boundary SDK, never raw HTTP. This ensures readiness, retry, and degradation are handled consistently.
8. **Tooling-complete** — Health audit, contract evolution, onboarding, CI gates, and cutover are included — not left as an exercise for the adopter.

---

## 12. Repository Structure

```
MNEMOS/
├── mnemos/                    Core library
│   ├── compression/           TurboQuant (arXiv:2504.19874)
│   ├── engram/                Engram model and enrichment
│   ├── retrieval/             Multi-tier backends + fusion
│   └── audit/                 Forensic ledger
├── mnemos_sdk/                Boundary adapter SDK (client library)
│   ├── client.py              MnemosClient with typed methods
│   └── config.py              MnemosConfig.from_env()
├── service/                   Flask REST API + MFS contract
├── registry/                  Service registry (MFS-compatible)
├── tools/                     Operational tooling
│   ├── mnemos_health_audit.py
│   ├── contract_diff.py
│   ├── mnemos_onboard.py
│   ├── mnemos_ci_gates.py
│   └── mnemos_cutover_scaffold.py
├── tests/                     Unit tests
├── .github/workflows/         CI gate template
├── docs/                      AI dev hand-off doc
├── Dockerfile                 Production container
└── docker-compose.yml         Standalone stack
```

---

## 13. Provenance

MNEMOS was designed from the ground up as a reusable memory service. Its architecture draws on production experience operating multi-tier vector retrieval, near-lossless compression, and forensic audit logging under continuous autonomous workloads.

| Capability | MNEMOS Component |
|---|---|
| Multi-vector retrieval | Multi-Tier Retrieval Engine |
| Semantic tagging | Engram Enrichment Layer |
| Immutable operation logging | Audit Trail |
| Near-optimal quantisation (arXiv:2504.19874) | Compression Layer |
| Telemetry & health reporting | Stats & Health API |
| Versioned contract schema | API Contract + Contract Diff |
| Client library with degradation | MNEMOS SDK (mnemos_sdk/) |
| Service validation | Health & Contract Audit |
| Consumer scaffolding | Consumer Onboarding |
| Pipeline integration | CI/CD Gates |
| Staged rollout | Cutover Scaffold |

What remains is a **pure infrastructure service** — a reusable, tooling-complete foundation for any application that needs intelligent, compressed, auditable memory.
